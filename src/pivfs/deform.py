"""Predictor interpolation and symmetric image deformation (Scarano 2002)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve1d, map_coordinates

FloatArray = NDArray[np.float64]


def smooth_field(a: FloatArray) -> FloatArray:
    """3×3 binomial low-pass of a vector component on its grid.

    Applies the kernel [1, 2, 1] / 4 along each axis (edges padded with the
    nearest value). Its response cos²(kΔ/2) is zero at the grid Nyquist
    wavenumber k = π/Δ. With 50 % overlap a top-hat window of width W = 2Δ also
    has zero response there (sinc(π) = 0), so predictor errors at that
    wavenumber are invisible to the residual correlation and would never be
    corrected; filtering the predictor removes them and keeps the iteration
    stable (Scarano 2002; Schrijer & Scarano 2008, Exp. Fluids 45).
    """
    k = np.array([0.25, 0.5, 0.25])
    return convolve1d(convolve1d(a, k, axis=0, mode="nearest"), k, axis=1, mode="nearest")


def interpolate_field(
    grid_x: FloatArray,
    grid_y: FloatArray,
    u: FloatArray,
    v: FloatArray,
    x: FloatArray,
    y: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Interpolate a vector field given on a regular grid to arbitrary points.

    The vector grid has centres x_k = x₀ + k Δ, y_l = y₀ + l Δ. A point (x, y)
    maps to fractional grid indices ((y − y₀)/Δ, (x − x₀)/Δ), where u and v are
    evaluated by cubic B-spline interpolation (``map_coordinates``, order 3,
    with the spline prefilter so the interpolant passes through the vectors).
    Outside the grid the field is extended with the nearest edge value.
    """
    x0, y0 = grid_x[0, 0], grid_y[0, 0]
    step_x = grid_x[0, 1] - x0 if grid_x.shape[1] > 1 else 1.0
    step_y = grid_y[1, 0] - y0 if grid_y.shape[0] > 1 else 1.0
    coords = np.stack([(np.asarray(y) - y0) / step_y, (np.asarray(x) - x0) / step_x])
    return (
        map_coordinates(u, coords, order=3, mode="nearest"),
        map_coordinates(v, coords, order=3, mode="nearest"),
    )


def dense_predictor(
    grid_x: FloatArray,
    grid_y: FloatArray,
    u: FloatArray,
    v: FloatArray,
    shape: tuple[int, int],
) -> tuple[FloatArray, FloatArray]:
    """Predictor displacement at every pixel centre of an image of ``shape``."""
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return interpolate_field(grid_x, grid_y, u, v, x, y)


def warp(img: FloatArray, dx: FloatArray, dy: FloatArray) -> FloatArray:
    """Resample an image at displaced positions: I'(x, y) = I(x + dx, y + dy).

    Uses cubic B-spline interpolation (``map_coordinates``, order 3); samples
    outside the image take the nearest edge value.
    """
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    return map_coordinates(img, np.stack([y + dy, x + dx]), order=3, mode="nearest")


def deform_pair(
    img_a: FloatArray, img_b: FloatArray, u: FloatArray, v: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Symmetric (central-difference) window deformation (Scarano 2002).

    With a dense predictor u(x), both frames are warped half-way towards the
    mid-time:

        A'(x) = A(x − ½ u(x)),   B'(x) = B(x + ½ u(x)).

    A particle at mid-time position x_m sits at x_m − ½u in A and x_m + ½u in B,
    so it appears at x_m in both deformed frames when the predictor is exact.
    Correlating A' and B' therefore measures only the residual Δu = u_true − u,
    and the corrected estimate is u + Δu. Centring the deformation in time gives
    second-order accuracy, versus first order for a forward-only warp of B.
    """
    return warp(img_a, -0.5 * u, -0.5 * v), warp(img_b, 0.5 * u, 0.5 * v)
