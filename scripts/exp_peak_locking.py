"""E1 — peak locking: bias and RMS error vs sub-pixel displacement per estimator.

Uniform dx from 0 to 1 px in 0.05 px steps (dy = 0), d_τ = 2.5 px, single pass
at W = 32 px without validation, 5 seeds. Errors of the x component at interior
vectors; RMS includes the bias.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, IMAGE_SHAPE, write_csv

from pivfs.flows import Uniform
from pivfs.metrics import interior_mask, rms, vector_errors
from pivfs.piv import PIVConfig, run_piv
from pivfs.plot import ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style, save
from pivfs.subpixel import ESTIMATORS
from pivfs.synth import SynthConfig, generate_pair

DISPLACEMENTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SEEDS = range(5)
WINDOW = 32
D_TAU = 2.5


def run() -> dict[str, float]:
    rows = []
    for dx in DISPLACEMENTS:
        eu = {m: [] for m in ESTIMATORS}
        fallback = dict.fromkeys(ESTIMATORS, 0)
        n_vectors = 0
        for seed in SEEDS:
            img_a, img_b, _ = generate_pair(
                SynthConfig(shape=IMAGE_SHAPE, d_tau=D_TAU, seed=seed), Uniform(dx, 0.0)
            )
            for method in ESTIMATORS:
                cfg = PIVConfig.single_pass(WINDOW, estimator=method, validate=False)
                vf = run_piv(img_a, img_b, cfg)
                eu[method].append(
                    vector_errors(vf, Uniform(dx, 0.0)).eu[interior_mask(vf, IMAGE_SHAPE)]
                )
                fallback[method] += vf.meta["n_fallback"]
                n_vectors += vf.u.size if method == ESTIMATORS[0] else 0
        for method in ESTIMATORS:
            e = np.concatenate(eu[method])
            rows.append(
                {
                    "estimator": method,
                    "dx_px": float(dx),
                    "bias_px": float(e.mean()),
                    "rms_px": rms(e),
                    "std_px": float(e.std()),
                    "n_interior_vectors": e.size,
                    "fallback_fraction": fallback[method] / n_vectors,
                }
            )
    write_csv("peak_locking.csv", rows)
    plot(rows)

    summary = {}
    for method in ESTIMATORS:
        r = [row for row in rows if row["estimator"] == method]
        summary[f"e1_{method}_max_abs_bias_px"] = max(abs(x["bias_px"]) for x in r)
        summary[f"e1_{method}_max_rms_px"] = max(x["rms_px"] for x in r)
        summary[f"e1_{method}_mean_rms_px"] = float(np.mean([x["rms_px"] for x in r]))
        summary[f"e1_{method}_fallback_fraction"] = float(
            np.mean([x["fallback_fraction"] for x in r])
        )
    return summary


def plot(rows: list[dict]) -> None:
    apply_style()
    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True, layout="constrained")
    for method in ESTIMATORS:
        r = [row for row in rows if row["estimator"] == method]
        dx = [x["dx_px"] for x in r]
        kw = {"color": ESTIMATOR_COLORS[method], "label": ESTIMATOR_LABELS[method], "marker": "o"}
        ax_b.plot(dx, [x["bias_px"] for x in r], **kw)
        ax_r.plot(dx, [x["rms_px"] for x in r], **kw)
    ax_b.axhline(0, color="0.3", lw=0.8)
    ax_b.set_ylabel("bias error ⟨Δu⟩ (px)")
    ax_r.set_ylabel("RMS error of u (px)")
    ax_r.set_ylim(bottom=0)
    for ax in (ax_b, ax_r):
        ax.set_xlabel("imposed displacement dx (px)")
    ax_b.set_title("Bias (peak locking)")
    ax_r.set_title("Total RMS error")
    ax_r.legend()
    fig.suptitle(
        f"E1 · Peak locking: uniform flow, d_τ = {D_TAU} px, W = {WINDOW} px, single pass, "
        f"{len(SEEDS)} seeds",
        fontsize=10,
    )
    save(fig, FIGURES / "peak_locking.png")


if __name__ == "__main__":
    print(run())
