import numpy as np
import pytest

from pivfs.piv import cross_correlate, overlap_weight
from pivfs.subpixel import ESTIMATORS, refine_peaks


def _peak_plane(y0: float, x0: float, n: int = 9, s: float = 1.1) -> np.ndarray:
    k = np.arange(n) - n // 2
    return np.exp(-((k[:, None] - y0) ** 2 + (k[None, :] - x0) ** 2) / (2 * s**2))[None]


@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.3, -0.27), (-0.49, 0.45), (0.11, 0.07)])
def test_subpixel_gaussian_on_gaussian_peak(offset):
    corr = _peak_plane(*offset)
    iy, ix = np.array([4]), np.array([4])
    fit = refine_peaks(corr, iy, ix, "gaussian")
    assert abs(fit.dy[0] - offset[0]) < 1e-6
    assert abs(fit.dx[0] - offset[1]) < 1e-6
    assert not fit.fallback[0]


def test_parabolic_exact_on_parabola():
    k = np.arange(-4, 5, dtype=float)
    plane = 10.0 - (k[:, None] - 0.2) ** 2 - (k[None, :] + 0.35) ** 2
    fit = refine_peaks(plane[None], np.array([4]), np.array([4]), "parabolic")
    np.testing.assert_allclose([fit.dy[0], fit.dx[0]], [0.2, -0.35], atol=1e-12)


def test_fallback_to_parabolic_when_non_positive():
    plane = _peak_plane(0.2, 0.1)
    plane[0, 4, 3] = -0.01  # left neighbour of the peak in x
    for method in ("gaussian", "centroid"):
        fit = refine_peaks(plane, np.array([4]), np.array([4]), method)
        assert fit.fallback[0]
    para = refine_peaks(plane, np.array([4]), np.array([4]), "parabolic")
    gauss = refine_peaks(plane, np.array([4]), np.array([4]), "gaussian")
    assert gauss.dx[0] == para.dx[0]


def test_unknown_estimator_raises():
    with pytest.raises(ValueError):
        refine_peaks(_peak_plane(0, 0), np.array([4]), np.array([4]), "spline")


def test_estimator_list():
    assert ESTIMATORS == ("gaussian", "parabolic", "centroid")


def test_cross_correlation_integer_shift(rng):
    a = rng.random((1, 32, 32))
    b = np.roll(a, (2, -3), axis=(1, 2))
    corr = cross_correlate(a, b)
    iy, ix = np.unravel_index(corr[0].argmax(), corr[0].shape)
    assert (iy - 16, ix - 16) == (2, -3)
    assert corr[0].max() == pytest.approx(1.0)


def test_overlap_weight():
    w = overlap_weight(32)
    assert w[16, 16] == 1.0
    assert w[16, 17] == pytest.approx(1 - 1 / 32)
    assert w[0, 0] == pytest.approx(0.25)
