import numpy as np
import pytest

from pivfs.flows import (
    LAMB_OSEEN_PEAK_RADIUS,
    LambOseen,
    LinearShear,
    SolidBodyRotation,
    Uniform,
)
from pivfs.synth import SynthConfig, generate_pair, render_particles, rk4_advect, save_pair


def _fitted_diameter(d_tau: float, rng: np.random.Generator) -> np.ndarray:
    """Render isolated particles and recover d_τ from second moments.

    A pixel-integrated Gaussian is the Gaussian (σ = d_τ / 4) convolved with a
    1-px box, whose variance adds 1/12 px², so d_τ = 4 √(var − 1/12).
    """
    spacing, n = 24, 15
    centres = spacing * (np.arange(n) + 1.0)
    cy, cx = [c.ravel() for c in np.meshgrid(centres, centres, indexing="ij")]
    cy = cy + rng.uniform(-0.5, 0.5, cy.size)
    cx = cx + rng.uniform(-0.5, 0.5, cx.size)
    shape = (spacing * (n + 1), spacing * (n + 1))
    img = render_particles(cx, cy, np.ones(cx.size), shape, d_tau)
    out = []
    for x0, y0 in zip(cx, cy, strict=True):
        i, j = round(y0), round(x0)
        patch = img[i - 10 : i + 11, j - 10 : j + 11]
        yy, xx = np.mgrid[i - 10 : i + 11, j - 10 : j + 11]
        m = patch.sum()
        var_x = (patch * (xx - x0) ** 2).sum() / m
        var_y = (patch * (yy - y0) ** 2).sum() / m
        out.append(4.0 * np.sqrt(0.5 * (var_x + var_y) - 1.0 / 12.0))
    return np.array(out)


@pytest.mark.parametrize("d_tau", [2.0, 2.5, 4.0])
def test_generator_density_and_diameter(d_tau, rng):
    cfg = SynthConfig(ppp=0.05, d_tau=d_tau, seed=4)
    _, _, truth = generate_pair(cfg, Uniform(0.0, 0.0))
    rows, cols = cfg.shape
    inside = (truth.xa >= -0.5) & (truth.xa < cols - 0.5) & (truth.ya >= -0.5)
    inside &= truth.ya < rows - 0.5
    assert inside.sum() / (rows * cols) == pytest.approx(cfg.ppp, rel=0.05)

    fitted = _fitted_diameter(d_tau, rng)
    assert fitted.mean() == pytest.approx(d_tau, rel=0.05)


def test_render_conserves_energy():
    img = render_particles(np.array([20.3]), np.array([19.8]), np.array([0.8]), (40, 40), 0.7)
    # ∫∫ I₀ exp(−8 r² / d²) dA = I₀ π d² / 8
    assert img.sum() == pytest.approx(0.8 * np.pi * 0.7**2 / 8, rel=1e-6)


def test_camera_quantisation_and_noise():
    cfg = SynthConfig(shape=(64, 64), bit_depth=12, noise_sigma=0.02, seed=1)
    img_a, _, _ = generate_pair(cfg, Uniform())
    assert img_a.min() >= 0 and img_a.max() <= 4095
    np.testing.assert_array_equal(img_a, np.rint(img_a))


def test_out_of_plane_loss_keeps_density():
    cfg = SynthConfig(out_of_plane_loss=0.3, seed=2)
    _, _, t = generate_pair(cfg, Uniform(1.0, 0.0))
    assert t.xb.size == t.xa.size
    kept = np.isin(np.round(t.xb - 1.0, 9), np.round(t.xa, 9))
    assert kept.mean() == pytest.approx(0.7, abs=0.02)


def test_rk4_matches_rotation():
    flow = SolidBodyRotation(omega=0.1, x0=0.0, y0=0.0)
    p = rk4_advect(flow, np.array([10.0]), np.array([0.0]), 1.0, 4)
    np.testing.assert_allclose(p[:, 0], [10 * np.cos(0.1), 10 * np.sin(0.1)], atol=1e-6)


def test_lamb_oseen_peak_velocity():
    flow = LambOseen()
    r = LAMB_OSEEN_PEAK_RADIUS * flow.r_c
    u, v = flow(flow.x0 + r, flow.y0)
    assert float(np.hypot(u, v)) == pytest.approx(flow.v_max, rel=1e-4)
    u0, v0 = flow(flow.x0, flow.y0)
    assert float(u0) == 0.0 and float(v0) == 0.0


def test_linear_shear():
    u, v = LinearShear(du_dy=0.02, y0=0.0)(np.array([5.0]), np.array([100.0]))
    assert u[0] == pytest.approx(2.0) and v[0] == 0.0


def test_save_pair(tmp_path):
    from PIL import Image

    img_a, img_b, _ = generate_pair(SynthConfig(shape=(32, 32)), Uniform(1.0, 0.0))
    save_pair(tmp_path / "pair", img_a, img_b)
    loaded = np.asarray(Image.open(tmp_path / "pair_a.png"))
    assert loaded.shape == (32, 32) and loaded.max() == img_a.max() * 257
