"""E2 — RMS error vs particle-image diameter d_τ.

Uniform flow with a random sub-pixel dx ~ U(0, 1) px per seed (dy = 0), single
pass at W = 32 px without validation. The Gaussian estimator is the subject;
the other two are shown for reference.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, IMAGE_SHAPE, write_csv
from matplotlib.ticker import NullFormatter, ScalarFormatter

from pivfs.flows import Uniform
from pivfs.metrics import interior_mask, rms, vector_errors
from pivfs.piv import PIVConfig, run_piv
from pivfs.plot import ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style, save
from pivfs.subpixel import ESTIMATORS
from pivfs.synth import SynthConfig, generate_pair

DIAMETERS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SEEDS = range(12)
WINDOW = 32


def run() -> dict[str, float]:
    rows = []
    for d_tau in DIAMETERS:
        errors = {m: [] for m in ESTIMATORS}
        fallback = dict.fromkeys(ESTIMATORS, 0)
        n_vectors = 0
        for seed in SEEDS:
            dx = float(np.random.default_rng(10_000 + seed).uniform(0.0, 1.0))
            flow = Uniform(dx, 0.0)
            img_a, img_b, _ = generate_pair(
                SynthConfig(shape=IMAGE_SHAPE, d_tau=d_tau, seed=seed), flow
            )
            for method in ESTIMATORS:
                vf = run_piv(
                    img_a, img_b, PIVConfig.single_pass(WINDOW, estimator=method, validate=False)
                )
                errors[method].append(
                    vector_errors(vf, flow).magnitude[interior_mask(vf, IMAGE_SHAPE)]
                )
                fallback[method] += vf.meta["n_fallback"]
            n_vectors += vf.u.size
        for method in ESTIMATORS:
            rows.append(
                {
                    "estimator": method,
                    "d_tau_px": d_tau,
                    "rms_px": rms(np.concatenate(errors[method])),
                    "fallback_fraction": fallback[method] / n_vectors,
                }
            )
    write_csv("error_vs_diameter.csv", rows)
    plot(rows)

    gauss = [r for r in rows if r["estimator"] == "gaussian"]
    best = min(gauss, key=lambda r: r["rms_px"])
    return {
        "e2_gaussian_best_d_tau_px": best["d_tau_px"],
        "e2_gaussian_min_rms_px": best["rms_px"],
        "e2_gaussian_rms_at_0.5px": gauss[0]["rms_px"],
        "e2_gaussian_rms_at_5px": gauss[-1]["rms_px"],
    }


def plot(rows: list[dict]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for method in ESTIMATORS:
        r = [row for row in rows if row["estimator"] == method]
        ax.plot(
            [x["d_tau_px"] for x in r],
            [x["rms_px"] for x in r],
            marker="o",
            color=ESTIMATOR_COLORS[method],
            label=ESTIMATOR_LABELS[method],
            lw=2.2 if method == "gaussian" else 1.2,
            alpha=1.0 if method == "gaussian" else 0.8,
        )
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("particle-image diameter d_τ (px)")
    ax.set_ylabel("RMS error |Δu| (px)")
    ax.set_title(f"E2 · Error vs d_τ (W = {WINDOW} px, single pass)")
    ax.legend()
    save(fig, FIGURES / "error_vs_diameter.png")


if __name__ == "__main__":
    print(run())
