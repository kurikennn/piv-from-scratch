"""Sub-pixel correlation-peak estimators.

Given the integer peak location of a correlation plane and its neighbours
c₋ = R(k − 1), c₀ = R(k), c₊ = R(k + 1), each estimator returns the fractional
offset δ ∈ (−½, ½) so that the peak lies at k + δ. The x and y directions are
treated separately (Raffel et al. 2018, §5.4.6; Willert & Gharib 1991).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Estimator = Literal["gaussian", "parabolic", "centroid"]
ESTIMATORS: tuple[Estimator, ...] = ("gaussian", "parabolic", "centroid")


def parabolic3(cm: FloatArray, c0: FloatArray, cp: FloatArray) -> FloatArray:
    """Three-point parabolic fit.

    Fitting R(k + t) = a t² + b t + c through the three samples gives the vertex

        δ = (c₋ − c₊) / (2 c₋ − 4 c₀ + 2 c₊).

    Returns 0 where the denominator is not negative (no maximum).
    """
    den = 2.0 * cm - 4.0 * c0 + 2.0 * cp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den < 0.0, (cm - cp) / den, 0.0)


def gaussian3(cm: FloatArray, c0: FloatArray, cp: FloatArray) -> FloatArray:
    """Three-point Gaussian fit (Willert & Gharib 1991).

    Assumes R(k + t) = A exp(−(t − δ)² / (2s²)); ln R is then a parabola in t,
    and applying the parabolic vertex formula to ln R gives

        δ = (ln c₋ − ln c₊) / (2 ln c₋ − 4 ln c₀ + 2 ln c₊).

    This is exact for a sampled Gaussian peak. Requires c₋, c₀, c₊ > 0; the
    caller must fall back to :func:`parabolic3` otherwise.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return parabolic3(np.log(cm), np.log(c0), np.log(cp))


def centroid3x3(patch: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Centre of mass of the 3×3 neighbourhood of the peak.

    For weights c_ij with i, j ∈ {−1, 0, 1}:
    δy = Σ i c_ij / Σ c_ij and δx = Σ j c_ij / Σ c_ij.
    ``patch`` has shape (..., 3, 3). Requires all c_ij > 0.
    """
    off = np.array([-1.0, 0.0, 1.0])
    total = patch.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        dy = (patch.sum(axis=-1) @ off) / total
        dx = (patch.sum(axis=-2) @ off) / total
    return dy, dx


@dataclass(frozen=True)
class PeakFit:
    """Sub-pixel peak offsets for a stack of correlation planes.

    ``fallback`` marks windows where the requested estimator was replaced by the
    parabolic fit because a required correlation value was ≤ 0.
    """

    dy: FloatArray
    dx: FloatArray
    fallback: NDArray[np.bool_]


def refine_peaks(corr: FloatArray, iy: IntArray, ix: IntArray, method: Estimator) -> PeakFit:
    """Sub-pixel offsets of the integer peaks (iy, ix) of each plane in ``corr``.

    ``corr`` has shape (n, H, W). Neighbours are taken with periodic wrap-around,
    which is exact for the circular (FFT) correlation used in :mod:`pivfs.piv`.
    """
    n, h, w = corr.shape
    k = np.arange(n)[:, None, None]
    rows = (iy[:, None, None] + np.array([-1, 0, 1])[None, :, None]) % h
    cols = (ix[:, None, None] + np.array([-1, 0, 1])[None, None, :]) % w
    patch = corr[k, rows, cols]  # (n, 3, 3), patch[:, 1, 1] is the peak

    cy = patch[:, :, 1]  # column through the peak: c₋, c₀, c₊ in y
    cx = patch[:, 1, :]  # row through the peak: c₋, c₀, c₊ in x
    para_y = parabolic3(cy[:, 0], cy[:, 1], cy[:, 2])
    para_x = parabolic3(cx[:, 0], cx[:, 1], cx[:, 2])

    if method == "parabolic":
        return PeakFit(para_y, para_x, np.zeros(n, bool))
    if method == "gaussian":
        bad_y = (cy <= 0.0).any(axis=1)
        bad_x = (cx <= 0.0).any(axis=1)
        dy = np.where(bad_y, para_y, gaussian3(cy[:, 0], cy[:, 1], cy[:, 2]))
        dx = np.where(bad_x, para_x, gaussian3(cx[:, 0], cx[:, 1], cx[:, 2]))
        return PeakFit(dy, dx, bad_y | bad_x)
    if method == "centroid":
        bad = (patch <= 0.0).any(axis=(1, 2))
        cdy, cdx = centroid3x3(patch)
        return PeakFit(np.where(bad, para_y, cdy), np.where(bad, para_x, cdx), bad)
    raise ValueError(f"unknown sub-pixel estimator {method!r}; choose from {ESTIMATORS}")
