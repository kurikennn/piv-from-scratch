"""Synthetic particle-image pairs with a known displacement field.

The generator follows the standard recipe for PIV Monte-Carlo studies
(Raffel et al. 2018, §6): particles are seeded uniformly at random, rendered as
pixel-integrated Gaussians, displaced by an analytic flow, and digitised by a
camera with finite bit depth and additive noise.

Modelling note: each particle is placed at a *mid-time* position x_m and
advected ½ frame backward (frame A) and ½ frame forward (frame B) along its
pathline. The ground truth at a vector location x is the Eulerian velocity
u(x). In curved or accelerating flow the chord x_B − x_A of a pathline differs
from u(x_m) by O(|u|² |∇u| / 24) (e.g. ~6·10⁻³ px for the default Lamb–Oseen
vortex at its peak), so this is a small, known modelling error that is
included in every error figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.special import erf

from pivfs.flows import Flow

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SynthConfig:
    """Parameters of the synthetic camera and seeding.

    Attributes
    ----------
    shape: image size (rows, cols) in px.
    ppp: particle density N_ppp in particles per pixel.
    d_tau: particle-image diameter d_τ in px, defined at the e⁻² intensity level.
    peak_min, peak_max: peak intensities are drawn from U(peak_min, peak_max),
        as a fraction of the camera's full scale.
    bit_depth: camera bit depth; images are quantised to 2^bit_depth − 1 levels.
    noise_sigma: std of additive Gaussian noise as a fraction of full scale.
    out_of_plane_loss: fraction of frame-A particles that leave the light sheet
        before frame B; the same number of new particles enter in frame B.
    margin: particles are seeded this many px beyond every image border so that
        particles can flow into the field of view.
    rk4_steps: RK4 sub-steps for each ½-frame advection.
    seed: seed for ``numpy.random.default_rng``.
    """

    shape: tuple[int, int] = (512, 512)
    ppp: float = 0.05
    d_tau: float = 2.5
    peak_min: float = 0.5
    peak_max: float = 1.0
    bit_depth: int = 8
    noise_sigma: float = 0.0
    out_of_plane_loss: float = 0.0
    margin: float = 16.0
    rk4_steps: int = 4
    seed: int = 0


@dataclass(frozen=True)
class TruthField:
    """Ground truth for one image pair.

    ``truth(x, y)`` returns the exact displacement u(x) in px/frame. The particle
    positions in each frame are kept for diagnostics and tests.
    """

    flow: Flow
    shape: tuple[int, int]
    xa: FloatArray = field(repr=False)
    ya: FloatArray = field(repr=False)
    xb: FloatArray = field(repr=False)
    yb: FloatArray = field(repr=False)

    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        return self.flow(x, y)

    def dense(self) -> tuple[FloatArray, FloatArray]:
        """Truth at every pixel centre, as two (rows, cols) arrays."""
        y, x = np.mgrid[0 : self.shape[0], 0 : self.shape[1]].astype(float)
        return self.flow(x, y)


def rk4_advect(flow: Flow, x: FloatArray, y: FloatArray, dt: float, steps: int) -> FloatArray:
    """Integrate dx/dt = u(x) over time ``dt`` with ``steps`` classical RK4 steps.

    For step h: k1 = u(x), k2 = u(x + h/2 k1), k3 = u(x + h/2 k2), k4 = u(x + h k3),
    x ← x + h/6 (k1 + 2k2 + 2k3 + k4). Returns an array of shape (2, N) = (x, y).
    """
    h = dt / steps
    p = np.stack([np.asarray(x, float), np.asarray(y, float)])
    for _ in range(steps):
        k1 = np.stack(flow(p[0], p[1]))
        q = p + 0.5 * h * k1
        k2 = np.stack(flow(q[0], q[1]))
        q = p + 0.5 * h * k2
        k3 = np.stack(flow(q[0], q[1]))
        q = p + h * k3
        k4 = np.stack(flow(q[0], q[1]))
        p = p + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return p


def render_particles(
    x: FloatArray,
    y: FloatArray,
    peak: FloatArray,
    shape: tuple[int, int],
    d_tau: float,
) -> FloatArray:
    """Render particles as 2D Gaussians integrated over each pixel's area.

    A particle at (x_p, y_p) with peak intensity I₀ has the continuous image
    I(x, y) = I₀ exp(−8 [(x − x_p)² + (y − y_p)²] / d_τ²), so d_τ is the e⁻²
    diameter. The value recorded by pixel (i, j), whose area is
    [j − ½, j + ½] × [i − ½, i + ½], is the integral of I over that area
    (Raffel et al. 2018, eq. 6.2):

        I_ij = I₀ (π d_τ² / 32)
               · [erf(√8 (j + ½ − x_p) / d_τ) − erf(√8 (j − ½ − x_p) / d_τ)]
               · [erf(√8 (i + ½ − y_p) / d_τ) − erf(√8 (i − ½ − y_p) / d_τ)].

    For d_τ ≫ 1 px this tends to I₀ at the centre pixel; for d_τ ≲ 1 px it
    conserves the particle's energy, which point sampling at pixel centres
    would not. Each particle is evaluated on a (2R + 1)² patch with R = ⌈d_τ⌉ + 1
    (beyond 4σ = d_τ the intensity is below e⁻⁸ ≈ 3·10⁻⁴ I₀).
    """
    rows, cols = shape
    radius = int(np.ceil(d_tau)) + 1
    offs = np.arange(-radius, radius + 1)
    a = np.sqrt(8.0) / d_tau

    jc = np.rint(x).astype(np.int64)[:, None] + offs  # (N, K) pixel columns
    ic = np.rint(y).astype(np.int64)[:, None] + offs  # (N, K) pixel rows
    ex = erf(a * (jc + 0.5 - x[:, None])) - erf(a * (jc - 0.5 - x[:, None]))
    ey = erf(a * (ic + 0.5 - y[:, None])) - erf(a * (ic - 0.5 - y[:, None]))
    vals = (peak * np.pi * d_tau**2 / 32.0)[:, None, None] * ey[:, :, None] * ex[:, None, :]

    ii = np.broadcast_to(ic[:, :, None], vals.shape)
    jj = np.broadcast_to(jc[:, None, :], vals.shape)
    inside = (ii >= 0) & (ii < rows) & (jj >= 0) & (jj < cols)
    flat = ii[inside] * cols + jj[inside]
    img = np.bincount(flat, weights=vals[inside], minlength=rows * cols)
    return img.reshape(rows, cols)


def digitise(img: FloatArray, cfg: SynthConfig, rng: np.random.Generator) -> FloatArray:
    """Camera model: add Gaussian noise, clip to [0, 1] full scale, quantise.

    Returns integer grey levels round(clip(I + n) · (2^b − 1)) as float64,
    with n ~ N(0, σ²) and σ = ``cfg.noise_sigma``.
    """
    if cfg.noise_sigma > 0:
        img = img + rng.normal(0.0, cfg.noise_sigma, img.shape)
    levels = 2**cfg.bit_depth - 1
    return np.rint(np.clip(img, 0.0, 1.0) * levels)


def _seeding_box(cfg: SynthConfig) -> tuple[float, float, float, float]:
    """Extent (x_lo, x_hi, y_lo, y_hi) of the image grown by ``cfg.margin``."""
    rows, cols = cfg.shape
    m = cfg.margin
    return -0.5 - m, cols - 0.5 + m, -0.5 - m, rows - 0.5 + m


def _uniform_particles(
    cfg: SynthConfig, rng: np.random.Generator, n: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``n`` particles uniform in the seeding box with U(peak_min, peak_max) peaks."""
    x_lo, x_hi, y_lo, y_hi = _seeding_box(cfg)
    x = rng.uniform(x_lo, x_hi, n)
    y = rng.uniform(y_lo, y_hi, n)
    peak = rng.uniform(cfg.peak_min, cfg.peak_max, n)
    return x, y, peak


def seed_particles(
    cfg: SynthConfig, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Draw mid-time positions and peak intensities of the particles.

    Particles are uniform in the image extended by ``cfg.margin`` on every side.
    Their number is Poisson with mean N_ppp × (extended area), so the density
    inside the image is N_ppp in expectation.
    """
    x_lo, x_hi, y_lo, y_hi = _seeding_box(cfg)
    n = rng.poisson(cfg.ppp * (x_hi - x_lo) * (y_hi - y_lo))
    return _uniform_particles(cfg, rng, n)


def generate_pair(cfg: SynthConfig, flow: Flow) -> tuple[FloatArray, FloatArray, TruthField]:
    """Generate one image pair displaced by ``flow``.

    Returns ``(img_a, img_b, truth_field)``; images are float64 arrays of grey
    levels in [0, 2^bit_depth − 1].
    """
    rng = np.random.default_rng(cfg.seed)
    xm, ym, peak = seed_particles(cfg, rng)
    xa, ya = rk4_advect(flow, xm, ym, -0.5, cfg.rk4_steps)
    xb, yb = rk4_advect(flow, xm, ym, +0.5, cfg.rk4_steps)
    peak_b = peak

    if cfg.out_of_plane_loss > 0:
        # Lost particles appear only in A; an equal number of fresh ones only in B.
        lost = rng.random(xm.size) < cfg.out_of_plane_loss
        xn, yn, pn = _uniform_particles(cfg, rng, int(lost.sum()))
        xb = np.concatenate([xb[~lost], xn])
        yb = np.concatenate([yb[~lost], yn])
        peak_b = np.concatenate([peak[~lost], pn])

    img_a = digitise(render_particles(xa, ya, peak, cfg.shape, cfg.d_tau), cfg, rng)
    img_b = digitise(render_particles(xb, yb, peak_b, cfg.shape, cfg.d_tau), cfg, rng)
    truth = TruthField(flow=flow, shape=cfg.shape, xa=xa, ya=ya, xb=xb, yb=yb)
    return img_a, img_b, truth


def save_pair(prefix: str | Path, img_a: FloatArray, img_b: FloatArray, bit_depth: int = 8) -> None:
    """Write ``<prefix>_a.png`` and ``<prefix>_b.png`` as 16-bit greyscale PNGs.

    Grey levels are rescaled from [0, 2^bit_depth − 1] to [0, 65535].
    """
    scale = 65535.0 / (2**bit_depth - 1)
    for tag, img in (("a", img_a), ("b", img_b)):
        arr = np.clip(np.rint(img * scale), 0, 65535).astype(np.uint16)
        Image.fromarray(arr).save(f"{prefix}_{tag}.png")
