"""E4 — robustness to image noise and seeding density.

Lamb–Oseen vortex processed with the default multi-pass pipeline. The final
pass is kept raw, then validated with the median test, so outliers can be
counted before and after validation. A vector is a true outlier when its error
exceeds 0.5 px. Interior vectors, 3 seeds per point.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, IMAGE_SHAPE, write_csv

from pivfs.flows import LambOseen
from pivfs.metrics import interior_mask, rms, vector_errors
from pivfs.piv import PIVConfig, VectorField, run_piv
from pivfs.plot import apply_style, save
from pivfs.synth import SynthConfig, generate_pair
from pivfs.validate import median_test, replace_outliers

NOISE = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DENSITY = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SEEDS = range(3)
OUTLIER_PX = 0.5
C_BEFORE, C_AFTER, C_FLAG = "#D55E00", "#0072B2", "#999999"


def evaluate(cfg: SynthConfig, flow: LambOseen) -> dict[str, float]:
    before, after, flagged = [], [], []
    for seed in SEEDS:
        img_a, img_b, _ = generate_pair(SynthConfig(**{**cfg.__dict__, "seed": seed}), flow)
        raw = run_piv(img_a, img_b, PIVConfig(validate_final=False))
        mask = median_test(raw.u, raw.v)
        u, v = replace_outliers(raw.u, raw.v, mask)
        fixed = VectorField(raw.x, raw.y, u, v, raw.peak_ratio, mask, raw.meta)
        m = interior_mask(raw, IMAGE_SHAPE)
        before.append(vector_errors(raw, flow).magnitude[m])
        after.append(vector_errors(fixed, flow).magnitude[m])
        flagged.append(mask[m])
    b, a = np.concatenate(before), np.concatenate(after)
    return {
        "outliers_before_pct": 100.0 * float((b > OUTLIER_PX).mean()),
        "flagged_pct": 100.0 * float(np.concatenate(flagged).mean()),
        "outliers_after_pct": 100.0 * float((a > OUTLIER_PX).mean()),
        "rms_after_px": rms(a),
    }


def run() -> dict[str, float]:
    flow = LambOseen(x0=(IMAGE_SHAPE[1] - 1) / 2, y0=(IMAGE_SHAPE[0] - 1) / 2)
    rows = []
    for sigma in NOISE:
        cfg = SynthConfig(shape=IMAGE_SHAPE, noise_sigma=sigma)
        rows.append({"sweep": "noise", "noise_sigma": sigma, "ppp": cfg.ppp, **evaluate(cfg, flow)})
    for ppp in DENSITY:
        cfg = SynthConfig(shape=IMAGE_SHAPE, ppp=ppp)
        rows.append({"sweep": "density", "noise_sigma": 0.0, "ppp": ppp, **evaluate(cfg, flow)})
    write_csv("robustness.csv", rows)
    plot(rows)

    noise = [r for r in rows if r["sweep"] == "noise"]
    return {
        "e4_rms_after_px_noise_0": noise[0]["rms_after_px"],
        "e4_rms_after_px_noise_5pct": noise[-1]["rms_after_px"],
        "e4_outliers_before_pct_noise_5pct": noise[-1]["outliers_before_pct"],
        "e4_outliers_after_pct_noise_5pct": noise[-1]["outliers_after_pct"],
    }


def plot(rows: list[dict]) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex="col")
    axes[0, 0].sharey(axes[0, 1])
    for col, (sweep, key, xlabel, scale) in enumerate(
        [
            ("noise", "noise_sigma", "noise σ (% of full scale)", 100.0),
            ("density", "ppp", "seeding density N_ppp (particles/px)", 1.0),
        ]
    ):
        r = [row for row in rows if row["sweep"] == sweep]
        x = np.array([row[key] for row in r]) * scale
        top, bottom = axes[0, col], axes[1, col]
        top.plot(
            x,
            [row["outliers_before_pct"] for row in r],
            "o-",
            color=C_BEFORE,
            label=f"error > {OUTLIER_PX} px, before validation",
        )
        top.plot(
            x,
            [row["outliers_after_pct"] for row in r],
            "o-",
            color=C_AFTER,
            label=f"error > {OUTLIER_PX} px, after validation",
        )
        top.plot(
            x,
            [row["flagged_pct"] for row in r],
            "s--",
            color=C_FLAG,
            label="flagged by median test",
        )
        top.set_ylabel("vectors (%)")
        bottom.plot(x, [row["rms_after_px"] for row in r], "o-", color=C_AFTER)
        bottom.set_ylabel("RMS error after validation (px)")
        bottom.set_xlabel(xlabel)
        if sweep == "density":
            top.set_xscale("log")
            bottom.set_xscale("log")
            bottom.set_yscale("log")
        else:
            bottom.set_ylim(bottom=0)
    axes[0, 0].set_title("Noise sweep (N_ppp = 0.05)")
    axes[0, 1].set_title("Density sweep (no noise)")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("E4 · Robustness on the Lamb–Oseen vortex, multi-pass 64→32→16 px", fontsize=10)
    fig.tight_layout()
    save(fig, FIGURES / "robustness.png")


if __name__ == "__main__":
    print(run())
