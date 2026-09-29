"""E3 — Lamb–Oseen vortex: single pass vs multi-pass symmetric deformation.

Default vortex (r_c = 40 px, v_max = 6 px/frame, centred) with default seeding.
RMS error magnitude over interior vectors and inside the core r < 2 r_c,
averaged over 5 seeds. The figure shows seed 0.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, IMAGE_SHAPE, write_csv

from pivfs.flows import LambOseen
from pivfs.metrics import interior_mask, rms, vector_errors
from pivfs.piv import PIVConfig, VectorField, run_piv
from pivfs.plot import ERROR_CMAP, MAGNITUDE_CMAP, apply_style, field_panel, quiver_overlay, save
from pivfs.synth import SynthConfig, generate_pair

SEEDS = range(5)
METHODS = {
    "single pass W=32": PIVConfig.single_pass(32),
    "single pass W=16": PIVConfig.single_pass(16),
    "multi-pass 64→32 (+2 it.)": PIVConfig(windows=(64, 32)),
    "multi-pass 64→32→16 (+2 it.)": PIVConfig(),
}
SINGLE, MULTI = "single pass W=32", "multi-pass 64→32→16 (+2 it.)"


def core_mask(vf: VectorField, flow: LambOseen) -> np.ndarray:
    """Interior vectors inside r < 2 r_c."""
    r = np.hypot(vf.x - flow.x0, vf.y - flow.y0)
    return interior_mask(vf, IMAGE_SHAPE) & (r < 2.0 * flow.r_c)


def run() -> dict[str, float]:
    flow = LambOseen(x0=(IMAGE_SHAPE[1] - 1) / 2, y0=(IMAGE_SHAPE[0] - 1) / 2)
    acc = {name: {"all": [], "core": [], "outlier": []} for name in METHODS}
    shown: dict[str, VectorField] = {}
    for seed in SEEDS:
        img_a, img_b, _ = generate_pair(SynthConfig(shape=IMAGE_SHAPE, seed=seed), flow)
        for name, cfg in METHODS.items():
            vf = run_piv(img_a, img_b, cfg)
            err = vector_errors(vf, flow).magnitude
            acc[name]["all"].append(err[interior_mask(vf, IMAGE_SHAPE)])
            acc[name]["core"].append(err[core_mask(vf, flow)])
            acc[name]["outlier"].append(vf.meta["outlier_fraction"])
            if seed == 0:
                shown[name] = vf

    rows = [
        {
            "method": name,
            "final_window_px": cfg.windows[-1],
            "passes": len(cfg.pass_windows()),
            "rms_interior_px": rms(np.concatenate(acc[name]["all"])),
            "rms_core_px": rms(np.concatenate(acc[name]["core"])),
            "flagged_fraction": float(np.mean(acc[name]["outlier"])),
        }
        for name, cfg in METHODS.items()
    ]
    write_csv("vortex.csv", rows)
    plot(shown[SINGLE], shown[MULTI], flow)

    by = {r["method"]: r for r in rows}
    return {
        "e3_single32_rms_interior_px": by[SINGLE]["rms_interior_px"],
        "e3_single32_rms_core_px": by[SINGLE]["rms_core_px"],
        "e3_single16_rms_core_px": by["single pass W=16"]["rms_core_px"],
        "e3_multi32_rms_core_px": by["multi-pass 64→32 (+2 it.)"]["rms_core_px"],
        "e3_multi16_rms_interior_px": by[MULTI]["rms_interior_px"],
        "e3_multi16_rms_core_px": by[MULTI]["rms_core_px"],
        "e3_core_improvement_factor": by[SINGLE]["rms_core_px"] / by[MULTI]["rms_core_px"],
        "e3_core_improvement_factor_same_W32": by[SINGLE]["rms_core_px"]
        / by["multi-pass 64→32 (+2 it.)"]["rms_core_px"],
    }


def plot(single: VectorField, multi: VectorField, flow: LambOseen) -> None:
    apply_style()
    ut, vt = flow(multi.x, multi.y)
    mag = np.hypot(ut, vt)
    e_single = vector_errors(single, flow).magnitude
    e_multi = vector_errors(multi, flow).magnitude
    emax = float(np.percentile(e_single[interior_mask(single, IMAGE_SHAPE)], 99))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    m0 = field_panel(axes[0], multi.x, multi.y, mag, MAGNITUDE_CMAP, 0.0, float(mag.max()))
    quiver_overlay(axes[0], multi.x, multi.y, ut, vt, every=4)
    axes[0].set_title("Ground truth |u|\n(quiver: every 4th vector)")
    fig.colorbar(m0, ax=axes[0], shrink=0.85, label="|u| (px/frame)")

    field_panel(axes[1], single.x, single.y, e_single, ERROR_CMAP, 0.0, emax)
    core_single = rms(e_single[core_mask(single, flow)])
    axes[1].set_title(f"Single pass, W = 32 px\ncore RMS {core_single:.3f} px")
    m2 = field_panel(axes[2], multi.x, multi.y, e_multi, ERROR_CMAP, 0.0, emax)
    core_multi = rms(e_multi[core_mask(multi, flow)])
    axes[2].set_title(f"Symmetric deformation, 64→32→16 px\ncore RMS {core_multi:.3f} px")
    fig.colorbar(m2, ax=axes[1:], shrink=0.85, label="error |Δu| (px/frame)", extend="max")

    theta = np.linspace(0, 2 * np.pi, 200)
    for ax in axes:
        ax.plot(
            flow.x0 + 2 * flow.r_c * np.cos(theta),
            flow.y0 + 2 * flow.r_c * np.sin(theta),
            ls="--",
            lw=0.9,
            color="white",
            alpha=0.8,
        )
        ax.set_xlim(0, IMAGE_SHAPE[1] - 1)
        ax.set_ylim(IMAGE_SHAPE[0] - 1, 0)
    fig.suptitle(
        f"E3 · Lamb–Oseen vortex (r_c = {flow.r_c:g} px, v_max = {flow.v_max:g} px/frame); "
        "dashed circle: r = 2 r_c",
        fontsize=10,
    )
    save(fig, FIGURES / "vortex.png")


if __name__ == "__main__":
    print(run())
