"""Error measures against the synthetic ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pivfs.flows import Flow
from pivfs.piv import VectorField

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Errors:
    """Per-vector error e = u_measured − u_true, in px/frame, at every vector."""

    eu: FloatArray
    ev: FloatArray

    @property
    def magnitude(self) -> FloatArray:
        """|e| = √(e_u² + e_v²)."""
        return np.hypot(self.eu, self.ev)


def vector_errors(vf: VectorField, flow: Flow) -> Errors:
    """Error of every vector against the exact displacement u(x) at its location."""
    ut, vt = flow(vf.x, vf.y)
    return Errors(vf.u - ut, vf.v - vt)


def interior_mask(vf: VectorField, shape: tuple[int, int], margin: float = 32.0) -> FloatArray:
    """Vectors whose location is at least ``margin`` px from every image border."""
    rows, cols = shape
    return (
        (vf.x >= margin)
        & (vf.x <= cols - 1 - margin)
        & (vf.y >= margin)
        & (vf.y <= rows - 1 - margin)
    )


def rms(e: FloatArray) -> float:
    """Root mean square √(mean e²) of the given errors (a total error: bias included)."""
    return float(np.sqrt(np.mean(np.square(e))))
