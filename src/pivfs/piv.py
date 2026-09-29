"""FFT cross-correlation PIV with multi-pass symmetric window deformation.

Pipeline (Raffel et al. 2018, ch. 5; Scarano 2002):

1. Divide both frames into interrogation windows (IW) of W × W px with a given
   overlap; vectors sit at the window centres.
2. Cross-correlate every window pair with the FFT (all windows at once).
3. Locate the correlation peak, remove the finite-window bias, and refine it
   with a sub-pixel estimator.
4. Validate with the normalised median test and replace outliers.
5. (Multi-pass) filter the field, interpolate it to every pixel, deform both frames
   symmetrically, correlate again at the next (smaller) W and add the
   residual to the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from pivfs.deform import deform_pair, dense_predictor, interpolate_field, smooth_field
from pivfs.subpixel import Estimator, refine_peaks
from pivfs.validate import median_test, replace_outliers

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PIVConfig:
    """Processing parameters.

    Attributes
    ----------
    windows: IW sizes W (px) of the successive passes, e.g. (64, 32, 16).
    overlap: fractional overlap of neighbouring windows (0.5 = 50 %).
    estimator: sub-pixel estimator: "gaussian", "parabolic" or "centroid".
    deform: if False, run a single pass at ``windows[-1]`` with no window offset
        or deformation (the classical baseline).
    final_iterations: extra deformation iterations at the final W.
    smooth_predictor: low-pass the predictor with a 3×3 binomial filter before
        deforming (see :func:`pivfs.deform.smooth_field`).
    validate: apply the normalised median test after every pass.
    validate_final: validate the last pass too (set False to inspect raw vectors).
    median_eps, median_threshold: ε (px) and threshold of the median test.
    """

    windows: tuple[int, ...] = (64, 32, 16)
    overlap: float = 0.5
    estimator: Estimator = "gaussian"
    deform: bool = True
    final_iterations: int = 2
    smooth_predictor: bool = True
    validate: bool = True
    validate_final: bool = True
    median_eps: float = 0.1
    median_threshold: float = 2.0

    @classmethod
    def single_pass(cls, window: int = 32, **kwargs: Any) -> PIVConfig:
        """Single pass at IW size ``window`` with no offset or deformation."""
        return cls(windows=(window,), deform=False, final_iterations=0, **kwargs)

    def pass_windows(self) -> list[int]:
        """IW size of every pass actually run."""
        if not self.deform:
            return [self.windows[-1]]
        return [*self.windows, *[self.windows[-1]] * self.final_iterations]


@dataclass(frozen=True)
class VectorField:
    """A PIV result on a regular grid of shape (rows, cols).

    x, y: vector locations (window centres) in px; u, v: displacement in px/frame;
    peak_ratio: primary / secondary correlation peak height of the last pass;
    outlier: vectors flagged (and replaced) by the median test in the last pass;
    meta: processing settings and per-pass statistics.
    """

    x: FloatArray
    y: FloatArray
    u: FloatArray
    v: FloatArray
    peak_ratio: FloatArray
    outlier: NDArray[np.bool_]
    meta: dict[str, Any] = field(default_factory=dict)


def window_grid(shape: tuple[int, int], window: int, overlap: float) -> tuple[IntArray, IntArray]:
    """Top-left row and column indices of all IWs that fit inside the image.

    The step is Δ = round(W (1 − overlap)); windows that would cross the border
    are dropped, and the grid is centred so the unused border is split evenly.
    The vector of a window starting at index s sits at its centre s + (W − 1)/2.
    """
    step = max(1, round(window * (1.0 - overlap)))
    starts = []
    for n in shape:
        s = np.arange(0, n - window + 1, step)
        starts.append(s + (n - window - s[-1]) // 2)
    return starts[0], starts[1]


def extract_windows(img: FloatArray, rows: IntArray, cols: IntArray, window: int) -> FloatArray:
    """Stack of IWs with shape (len(rows) · len(cols), W, W), in row-major order."""
    view = sliding_window_view(img, (window, window))
    return view[rows[:, None], cols[None, :]].reshape(-1, window, window)


def cross_correlate(wa: FloatArray, wb: FloatArray) -> FloatArray:
    """Normalised circular cross-correlation of stacked windows via the FFT.

    With zero-mean windows a' = a − ā and b' = b − b̄,

        R(s) = Σ_x a'(x) b'(x + s) / (W² σ_a σ_b)
             = F⁻¹{ conj(F{a'}) · F{b'} }(s) / (W² σ_a σ_b)

    (Willert & Gharib 1991). If b(x) = a(x − d) the peak is at s = d. The
    output is fft-shifted so that zero displacement sits at index W // 2.
    Normalising by the window standard deviations makes R a correlation
    coefficient; it does not move the peak.
    """
    w = wa.shape[-1]
    a = wa - wa.mean(axis=(1, 2), keepdims=True)
    b = wb - wb.mean(axis=(1, 2), keepdims=True)
    spec = np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b)
    corr = np.fft.irfft2(spec, s=(w, w))
    norm = np.sqrt((a**2).sum(axis=(1, 2)) * (b**2).sum(axis=(1, 2)))
    norm = np.where(norm > 0, norm, 1.0)
    return np.fft.fftshift(corr, axes=(1, 2)) / norm[:, None, None]


def overlap_weight(window: int) -> FloatArray:
    """Fraction of pixel pairs that contribute to R at each shift s.

    Only W − |s| of the W samples along each axis overlap at shift s (the rest
    pair with unrelated, wrapped-around content), so the expected correlation
    peak is weighted by

        w(s) = (1 − |s_x| / W)(1 − |s_y| / W).

    This slope skews the peak towards zero and biases the sub-pixel estimate by
    roughly −σ_R² sign(d) / W (Raffel et al. 2018, §5.4.5; Westerweel 1997).
    Dividing R by w before the sub-pixel fit removes that bias. The array is
    fft-shifted like the correlation planes.
    """
    s = np.arange(window) - window // 2
    w1 = 1.0 - np.abs(s) / window
    return w1[:, None] * w1[None, :]


def find_peaks(corr: FloatArray, exclusion: int = 2) -> tuple[IntArray, IntArray, FloatArray]:
    """Integer location of the highest peak of each plane, and the peak ratio.

    The peak ratio is R₁ / R₂, where R₂ is the highest value outside a
    (2·exclusion + 1)² neighbourhood of the primary peak R₁ (periodic, as the
    correlation is circular). It is +inf when R₂ ≤ 0.
    """
    n, h, w = corr.shape
    flat = corr.reshape(n, -1).argmax(axis=1)
    iy, ix = np.divmod(flat, w)
    k = np.arange(n)
    r1 = corr[k, iy, ix]

    offs = np.arange(-exclusion, exclusion + 1)
    rows = (iy[:, None, None] + offs[None, :, None]) % h
    cols = (ix[:, None, None] + offs[None, None, :]) % w
    masked = corr.copy()
    masked[k[:, None, None], rows, cols] = -np.inf
    r2 = masked.reshape(n, -1).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r2 > 0, r1 / r2, np.inf)
    return iy, ix, ratio


def correlate_pass(
    img_a: FloatArray,
    img_b: FloatArray,
    window: int,
    overlap: float,
    estimator: Estimator,
) -> dict[str, Any]:
    """One correlation pass: grid, FFT correlation, peak search, sub-pixel fit.

    Returns a dict with grid coordinates ``x, y`` and displacements ``u, v``
    (all of shape (rows, cols)), the ``peak_ratio``, and ``n_fallback``, the
    number of windows where the estimator fell back to the parabolic fit.
    """
    rows, cols = window_grid(img_a.shape, window, overlap)
    shape = (rows.size, cols.size)
    corr = cross_correlate(
        extract_windows(img_a, rows, cols, window), extract_windows(img_b, rows, cols, window)
    )
    iy, ix, ratio = find_peaks(corr)
    fit = refine_peaks(corr / overlap_weight(window), iy, ix, estimator)
    centre = window // 2
    gy, gx = np.meshgrid(rows + (window - 1) / 2, cols + (window - 1) / 2, indexing="ij")
    return {
        "x": gx,
        "y": gy,
        "u": (ix - centre + fit.dx).reshape(shape),
        "v": (iy - centre + fit.dy).reshape(shape),
        "peak_ratio": ratio.reshape(shape),
        "n_fallback": int(fit.fallback.sum()),
    }


def run_piv(img_a: FloatArray, img_b: FloatArray, cfg: PIVConfig | None = None) -> VectorField:
    """Process an image pair; returns the final :class:`VectorField`.

    Each pass correlates the (deformed) frames on its own grid. From the second
    pass on, the previous field is low-pass filtered (optional, default on) and
    interpolated to every pixel (bicubic), both frames are deformed
    symmetrically by ±½ the predictor, and the measured residual is added to
    the predictor interpolated at the new vector locations. Every pass is
    followed by the median test and replacement.
    """
    cfg = cfg or PIVConfig()
    img_a = np.asarray(img_a, float)
    img_b = np.asarray(img_b, float)
    windows = cfg.pass_windows()
    prev: dict[str, Any] | None = None
    stats = []

    for i, window in enumerate(windows):
        if prev is None:
            a, b = img_a, img_b
        else:
            gu, gv = prev["u"], prev["v"]
            if cfg.smooth_predictor:
                gu, gv = smooth_field(gu), smooth_field(gv)
            ud, vd = dense_predictor(prev["x"], prev["y"], gu, gv, img_a.shape)
            a, b = deform_pair(img_a, img_b, ud, vd)
        res = correlate_pass(a, b, window, cfg.overlap, cfg.estimator)
        if prev is not None:
            pu, pv = interpolate_field(prev["x"], prev["y"], gu, gv, res["x"], res["y"])
            res["u"] = res["u"] + pu
            res["v"] = res["v"] + pv

        last = i == len(windows) - 1
        outlier = np.zeros(res["u"].shape, bool)
        if cfg.validate and (cfg.validate_final or not last):
            outlier = median_test(res["u"], res["v"], cfg.median_eps, cfg.median_threshold)
            res["u"], res["v"] = replace_outliers(res["u"], res["v"], outlier)
        res["outlier"] = outlier
        stats.append(
            {
                "window": window,
                "n_vectors": outlier.size,
                "outlier_fraction": float(outlier.mean()),
                "n_fallback": res["n_fallback"],
            }
        )
        prev = res

    assert prev is not None
    meta = {
        "windows": windows,
        "overlap": cfg.overlap,
        "passes": len(windows),
        "estimator": cfg.estimator,
        "deform": cfg.deform,
        "smooth_predictor": cfg.smooth_predictor,
        "outlier_fraction": stats[-1]["outlier_fraction"],
        "n_fallback": stats[-1]["n_fallback"],
        "pass_stats": stats,
    }
    return VectorField(
        x=prev["x"],
        y=prev["y"],
        u=prev["u"],
        v=prev["v"],
        peak_ratio=prev["peak_ratio"],
        outlier=prev["outlier"],
        meta=meta,
    )
