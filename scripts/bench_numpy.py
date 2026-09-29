"""NumPy throughput of a single correlation pass at 1024² px, in vectors/s.

Times ``correlate_pass`` (FFT correlation, peak search, sub-pixel fit) for the
window sizes the WebGPU demo supports, with the full correlation plane and with
the ±8 px search range used on the GPU. Writes ``results/throughput_numpy.json``,
and a copy the web page loads to compare against the GPU.
"""

from __future__ import annotations

import json
import platform
import time

import numpy as np
from common import RESULTS, ROOT

from pivfs.flows import LambOseen
from pivfs.piv import correlate_pass
from pivfs.synth import SynthConfig, generate_pair

SHAPE = (1024, 1024)
REPEAT = 5


def main() -> None:
    img_a, img_b, _ = generate_pair(SynthConfig(shape=SHAPE, seed=0), LambOseen(511.5, 511.5))
    rows = []
    for window in (16, 32):
        for radius in (None, 8):
            correlate_pass(img_a, img_b, window, 0.5, "gaussian", radius)  # warm-up
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                res = correlate_pass(img_a, img_b, window, 0.5, "gaussian", radius)
            seconds = (time.perf_counter() - t0) / REPEAT
            n = res["u"].size
            rows.append(
                {
                    "window": window,
                    "search_radius": radius,
                    "vectors": n,
                    "ms_per_pass": 1000 * seconds,
                    "vectors_per_second": n / seconds,
                }
            )
            print(f"W={window} R={radius}: {n / seconds:,.0f} vectors/s, {1000 * seconds:.1f} ms")
    out = {
        "image": list(SHAPE),
        "overlap": 0.5,
        "cpu": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "runs": rows,
    }
    text = json.dumps(out, indent=2) + "\n"
    (RESULTS / "throughput_numpy.json").write_text(text)
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "web" / "data" / "throughput_numpy.json").write_text(text)


if __name__ == "__main__":
    main()
