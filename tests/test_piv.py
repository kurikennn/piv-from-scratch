import numpy as np
import pytest
from conftest import SHAPE, core_rms, interior_errors

from pivfs.flows import LambOseen, Uniform
from pivfs.metrics import rms
from pivfs.piv import PIVConfig, run_piv, window_grid
from pivfs.synth import SynthConfig, generate_pair


def test_integer_shift_exact():
    flow = Uniform(3.0, -2.0)
    img_a, img_b, _ = generate_pair(SynthConfig(seed=3), flow)
    vf = run_piv(img_a, img_b)
    assert interior_errors(vf, flow).max() < 1e-3


def test_uniform_subpixel_accuracy():
    flow = Uniform(0.3, 0.0)
    img_a, img_b, _ = generate_pair(SynthConfig(d_tau=2.5, seed=5), flow)
    vf = run_piv(img_a, img_b, PIVConfig.single_pass(32))
    assert rms(interior_errors(vf, flow)) < 0.05


def test_deformation_beats_single_pass():
    flow = LambOseen()
    img_a, img_b, _ = generate_pair(SynthConfig(seed=7), flow)
    single = run_piv(img_a, img_b, PIVConfig.single_pass(32))
    multi = run_piv(img_a, img_b, PIVConfig())
    assert core_rms(multi, flow) < 0.5 * core_rms(single, flow)


def test_deterministic():
    cfg = SynthConfig(seed=11, noise_sigma=0.02, out_of_plane_loss=0.1)
    a1, b1, _ = generate_pair(cfg, LambOseen())
    a2, b2, _ = generate_pair(cfg, LambOseen())
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    v1, v2 = run_piv(a1, b1), run_piv(a2, b2)
    np.testing.assert_array_equal(v1.u, v2.u)
    np.testing.assert_array_equal(v1.v, v2.v)
    a3, _, _ = generate_pair(SynthConfig(seed=12), LambOseen())
    assert not np.array_equal(a1, a3)


def test_grid_drops_border_windows_and_is_centred():
    rows, cols = window_grid((100, 90), 32, 0.5)
    assert rows[-1] + 32 <= 100 and cols[-1] + 32 <= 90
    assert rows[0] == (100 - 32 - (rows[-1] - rows[0])) // 2
    np.testing.assert_array_equal(np.diff(cols), 16)


def test_single_pass_metadata():
    img_a, img_b, _ = generate_pair(SynthConfig(seed=1, shape=(128, 128)), Uniform(1.0, 0.0))
    vf = run_piv(img_a, img_b, PIVConfig.single_pass(32, estimator="centroid"))
    assert vf.meta["windows"] == [32] and vf.meta["estimator"] == "centroid"
    assert vf.u.shape == vf.x.shape == vf.peak_ratio.shape == vf.outlier.shape == (7, 7)
    assert np.all(vf.peak_ratio > 1.0)


def test_against_openpiv():
    """Optional cross-check against an independent implementation."""
    pyprocess = pytest.importorskip("openpiv.pyprocess")
    flow = Uniform(0.3, 0.0)
    img_a, img_b, _ = generate_pair(SynthConfig(seed=2), flow)
    ours = run_piv(img_a, img_b, PIVConfig.single_pass(32, validate=False))
    u, v, _ = pyprocess.extended_search_area_piv(
        img_a.astype(np.int32),
        img_b.astype(np.int32),
        window_size=32,
        overlap=16,
        dt=1.0,
        search_area_size=32,
        sig2noise_method="peak2peak",
    )
    interior = (slice(2, -2), slice(2, -2))
    assert abs(np.nanmean(u[interior]) - ours.u[interior].mean()) < 0.05
    assert abs(np.nanmean(np.abs(v[interior])) - np.abs(ours.v[interior]).mean()) < 0.05


def test_shape_is_respected():
    assert SHAPE == SynthConfig().shape


def test_search_radius_matches_full_plane_for_small_displacement():
    flow = LambOseen(v_max=4.0)
    img_a, img_b, _ = generate_pair(SynthConfig(seed=9), flow)
    full = run_piv(img_a, img_b, PIVConfig.single_pass(32, validate=False))
    bounded = run_piv(img_a, img_b, PIVConfig.single_pass(32, validate=False, search_radius=8))
    same = (full.u == bounded.u) & (full.v == bounded.v)
    assert same.mean() > 0.99


def test_search_mask_caps_radius():
    from pivfs.piv import search_mask

    assert search_mask(32, 8).sum() == 17**2
    assert search_mask(16, 8).sum() == 15**2
