"""Analytic, steady 2D flow fields used as ground truth.

Every field is a callable ``flow(x, y) -> (dx, dy)`` that returns the velocity
in px/frame at pixel coordinates ``(x, y)``. Coordinates follow the image
convention used throughout the package: ``x`` is the column index, ``y`` the
row index, and pixel ``(i, j)`` covers ``[j - 1/2, j + 1/2] × [i - 1/2, i + 1/2]``
so its centre sits at integer coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# Lamb–Oseen peak location and value.  With s = r / r_c the profile is
# u_θ = Γ / (2π r_c) · (1 − exp(−s²)) / s, whose maximum satisfies
# exp(s²) = 1 + 2 s², i.e. s* = 1.12091 and (1 − exp(−s*²)) / s* = 0.638161.
LAMB_OSEEN_PEAK_RADIUS = 1.1209
LAMB_OSEEN_PEAK_FACTOR = 0.6382


class Flow(Protocol):
    """A steady velocity field ``u(x, y) -> (dx, dy)`` in px/frame."""

    def __call__(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True)
class Uniform:
    """Uniform translation: u = dx, v = dy everywhere."""

    dx: float = 0.0
    dy: float = 0.0

    def __call__(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return np.full(x.shape, self.dx), np.full(y.shape, self.dy)


@dataclass(frozen=True)
class LambOseen:
    """Lamb–Oseen vortex centred at (x0, y0).

    Azimuthal velocity u_θ(r) = Γ / (2π r) · (1 − exp(−r² / r_c²)).

    The circulation Γ is derived from the requested peak velocity: the peak lies
    at r ≈ 1.1209 r_c with value 0.6382 · Γ / (2π r_c), hence
    Γ = 2π r_c v_max / 0.6382. The vortex rotates counter-clockwise in (x, y)
    for v_max > 0 (clockwise on screen, where y points down).

    Cartesian components: u = −u_θ (y − y0) / r, v = u_θ (x − x0) / r. At r → 0
    u_θ / r → Γ / (2π r_c²) (solid-body rotation), which is used to avoid 0/0.
    """

    x0: float = 255.5
    y0: float = 255.5
    r_c: float = 40.0
    v_max: float = 6.0

    @property
    def gamma(self) -> float:
        """Circulation Γ = 2π r_c v_max / 0.6382 in px²/frame."""
        return 2.0 * np.pi * self.r_c * self.v_max / LAMB_OSEEN_PEAK_FACTOR

    def __call__(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        rx = np.asarray(x, float) - self.x0
        ry = np.asarray(y, float) - self.y0
        r2 = rx**2 + ry**2
        # u_θ / r = Γ / (2π) · (1 − exp(−r²/r_c²)) / r²; −expm1 keeps precision at small r.
        with np.errstate(invalid="ignore", divide="ignore"):
            omega = np.where(
                r2 > 0.0,
                -np.expm1(-r2 / self.r_c**2) / r2,
                1.0 / self.r_c**2,
            ) * (self.gamma / (2.0 * np.pi))
        return -omega * ry, omega * rx


@dataclass(frozen=True)
class LinearShear:
    """Simple shear: u = (du/dy) · (y − y0), v = 0."""

    du_dy: float = 0.01
    y0: float = 255.5

    def __call__(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return self.du_dy * (y - self.y0), np.zeros(x.shape)


@dataclass(frozen=True)
class SolidBodyRotation:
    """Solid-body rotation with angular velocity ω (rad/frame) about (x0, y0).

    u = −ω (y − y0), v = ω (x − x0).
    """

    omega: float = 0.01
    x0: float = 255.5
    y0: float = 255.5

    def __call__(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return -self.omega * (y - self.y0), self.omega * (x - self.x0)
