"""Outlier detection and replacement for vector fields on a regular grid."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Offsets of the eight neighbours in a 3×3 neighbourhood.
_NEIGHBOURS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def neighbours(a: FloatArray) -> FloatArray:
    """Stack the eight 3×3 neighbours of every grid point: shape (rows, cols, 8).

    Points outside the grid are NaN, so border vectors use only the neighbours
    that exist.
    """
    rows, cols = a.shape
    padded = np.full((rows + 2, cols + 2), np.nan)
    padded[1:-1, 1:-1] = a
    return np.stack(
        [padded[1 + di : 1 + di + rows, 1 + dj : 1 + dj + cols] for di, dj in _NEIGHBOURS],
        axis=-1,
    )


def _nanmedian(a: FloatArray) -> FloatArray:
    """Median over the last axis ignoring NaN (NaN if every entry is NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=-1)


def normalized_residual(a: FloatArray, eps: float = 0.1) -> FloatArray:
    """Normalised median residual of one velocity component (Westerweel & Scarano 2005).

    With U_m the median of the eight neighbours U_i and r_m the median of the
    neighbour residuals r_i = |U_i − U_m|:

        r₀* = |U₀ − U_m| / (r_m + ε),

    where ε (px) is the acceptable fluctuation level of the measurement.
    """
    nb = neighbours(a)
    um = _nanmedian(nb)
    rm = _nanmedian(np.abs(nb - um[..., None]))
    return np.abs(a - um) / (rm + eps)


def median_test(
    u: FloatArray, v: FloatArray, eps: float = 0.1, threshold: float = 2.0
) -> BoolArray:
    """Flag vectors whose combined normalised residual exceeds ``threshold``.

    The two components are combined as r* = √(r_u*² + r_v*²) and a vector is an
    outlier when r* > threshold (Westerweel & Scarano 2005 recommend ε = 0.1 px
    and a threshold of 2).
    """
    r = np.hypot(normalized_residual(u, eps), normalized_residual(v, eps))
    return r > threshold


def replace_outliers(
    u: FloatArray, v: FloatArray, outlier: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """Replace flagged vectors by the median of their valid 3×3 neighbours.

    Flagged neighbours are excluded from the median. If every neighbour is
    flagged, the median of all neighbours is used; a vector with no usable
    neighbour at all is left unchanged.
    """
    out = []
    for a in (u, v):
        valid = np.where(outlier, np.nan, a)
        med = _nanmedian(neighbours(valid))
        med = np.where(np.isnan(med), _nanmedian(neighbours(a)), med)
        med = np.where(np.isnan(med), a, med)
        out.append(np.where(outlier, med, a))
    return out[0], out[1]
