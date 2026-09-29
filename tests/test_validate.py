import numpy as np

from pivfs.flows import LambOseen
from pivfs.validate import median_test, replace_outliers


def test_median_test_flags_injected_outliers(rng):
    y, x = np.mgrid[0:60, 0:60] * 8.0 + 16.0
    u, v = LambOseen(x0=250, y0=250, r_c=60, v_max=5)(x, y)
    u = u + rng.normal(0, 0.03, u.shape)
    v = v + rng.normal(0, 0.03, v.shape)
    bad = rng.random(u.shape) < 0.05
    u[bad] = rng.uniform(-8, 8, bad.sum())
    v[bad] = rng.uniform(-8, 8, bad.sum())

    flagged = median_test(u, v, eps=0.1, threshold=2.0)
    assert flagged[bad].mean() >= 0.95
    assert flagged[~bad].mean() <= 0.01


def test_replacement_uses_valid_neighbours():
    u = np.ones((5, 5))
    v = np.zeros((5, 5))
    u[2, 2], v[2, 2] = 50.0, -40.0
    mask = median_test(u, v)
    assert mask[2, 2] and mask.sum() == 1
    ur, vr = replace_outliers(u, v, mask)
    assert ur[2, 2] == 1.0 and vr[2, 2] == 0.0
    np.testing.assert_array_equal(ur[~mask], u[~mask])
