"""Regenerate every CSV in results/ and figure in figures/ (E1–E4)."""

from __future__ import annotations

import json
import time

import exp_diameter
import exp_peak_locking
import exp_robustness
import exp_vortex
from common import RESULTS


def main() -> None:
    summary: dict[str, float] = {}
    for module in (exp_peak_locking, exp_diameter, exp_vortex, exp_robustness):
        t0 = time.perf_counter()
        summary.update(module.run())
        print(f"{module.__name__}: done in {time.perf_counter() - t0:.1f} s")
    path = RESULTS / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Headline numbers written to {path.relative_to(RESULTS.parent)}:")
    for key, value in summary.items():
        print(f"  {key:45s} {value:.4f}")


if __name__ == "__main__":
    main()
