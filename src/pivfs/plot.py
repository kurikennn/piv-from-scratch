"""Shared figure style so all experiment figures read as one set."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Okabe–Ito colour-blind-safe palette, fixed per estimator / method in every figure.
ESTIMATOR_COLORS = {"gaussian": "#0072B2", "parabolic": "#E69F00", "centroid": "#009E73"}
ESTIMATOR_LABELS = {
    "gaussian": "Gaussian 3-pt",
    "parabolic": "Parabolic 3-pt",
    "centroid": "Centroid 3×3",
}
METHOD_COLORS = {"single": "#D55E00", "multi": "#0072B2"}
MAGNITUDE_CMAP = "viridis"  # sequential, velocity magnitude
ERROR_CMAP = "magma"  # sequential, shared by all error panels


def apply_style() -> None:
    """Set matplotlib defaults used by every figure."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
        }
    )


def save(fig: Figure, path: str | Path) -> None:
    """Write a figure (creating the directory) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def field_panel(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    value: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
) -> mpl.collections.QuadMesh:
    """Colour map of a scalar on a vector grid, in image orientation and equal aspect."""
    mesh = ax.pcolormesh(x, y, value, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.grid(False)
    return mesh


def quiver_overlay(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    every: int = 4,
) -> None:
    """Light, subsampled quiver in image coordinates (y down)."""
    s = (slice(every // 2, None, every), slice(every // 2, None, every))
    ax.quiver(
        x[s],
        y[s],
        u[s],
        v[s],
        color="white",
        alpha=0.75,
        angles="xy",
        scale_units="xy",
        scale=0.25,
        width=0.004,
    )
