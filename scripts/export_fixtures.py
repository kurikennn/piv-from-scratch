"""Export image pairs and Python PIV results for the WebGPU parity test.

Writes ``web/fixtures/parity.json``. Each case holds an 8-bit image pair
(base64), the processing settings, and the vectors from
``pivfs.piv.run_piv(..., PIVConfig.single_pass(W, search_radius=8, validate=False))``,
which is the algorithm the WGSL correlation shader implements.
"""

from __future__ import annotations

import base64
import json

import numpy as np
from common import ROOT

from pivfs.flows import LambOseen, Uniform
from pivfs.piv import PIVConfig, run_piv
from pivfs.synth import SynthConfig, generate_pair

SHAPE = (256, 256)
SEARCH_RADIUS = 8
CASES = [
    ("uniform (2.3, -1.7) px, W=32, Gaussian", Uniform(2.3, -1.7), {}, 32, "gaussian"),
    ("vortex, W=32, Gaussian", LambOseen(127.5, 127.5, 20.0, 5.0), {}, 32, "gaussian"),
    ("vortex, W=16, Gaussian", LambOseen(127.5, 127.5, 20.0, 3.0), {}, 16, "gaussian"),
    (
        "vortex, noise 5 %, d=1 px, W=32, parabolic",
        LambOseen(127.5, 127.5, 20.0, 4.0),
        {"noise_sigma": 0.05, "d_tau": 1.0},
        32,
        "parabolic",
    ),
    (
        "vortex, d=0.75 px, W=32, centroid",
        LambOseen(127.5, 127.5, 20.0, 4.0),
        {"d_tau": 0.75},
        32,
        "centroid",
    ),
]


def encode(img: np.ndarray) -> str:
    """Base64 of the image as row-major uint8 grey levels."""
    return base64.b64encode(img.astype(np.uint8).tobytes()).decode("ascii")


def main() -> None:
    cases = []
    for i, (name, flow, synth_kw, window, estimator) in enumerate(CASES):
        img_a, img_b, _ = generate_pair(SynthConfig(shape=SHAPE, seed=100 + i, **synth_kw), flow)
        cfg = PIVConfig.single_pass(
            window, estimator=estimator, validate=False, search_radius=SEARCH_RADIUS
        )
        vf = run_piv(img_a, img_b, cfg)
        cases.append(
            {
                "name": name,
                "width": SHAPE[1],
                "height": SHAPE[0],
                "window": window,
                "overlap": cfg.overlap,
                "estimator": estimator,
                "search_radius": SEARCH_RADIUS,
                "img_a": encode(img_a),
                "img_b": encode(img_b),
                "x": vf.x.ravel().tolist(),
                "y": vf.y.ravel().tolist(),
                "u": np.round(vf.u.ravel(), 9).tolist(),
                "v": np.round(vf.v.ravel(), 9).tolist(),
            }
        )
    path = ROOT / "web" / "fixtures" / "parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tolerance_px": 1e-3, "cases": cases}) + "\n")
    print(f"wrote {len(cases)} cases to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
