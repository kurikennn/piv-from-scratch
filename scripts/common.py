"""Paths and small helpers shared by the experiment scripts."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
IMAGE_SHAPE = (512, 512)


def write_csv(name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows (dicts with identical keys) to ``results/<name>``; floats get 6 decimals."""
    rows = list(rows)
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    return path
