"""Headless WebGPU parity test: run web/parity.html in Chromium via Playwright.

Serves ``web/`` on a local port, opens the parity page, waits for
``window.parityResult`` and writes it to ``results/webgpu_parity.json``.
Exits non-zero if any fixture differs from Python by more than the tolerance.
Without a GPU, Chromium's SwiftShader software adapter is used.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from common import RESULTS, ROOT
from playwright.sync_api import sync_playwright

CHROMIUM_FLAGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--disable-vulkan-surface",
    "--ignore-gpu-blocklist",
]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds to wait")
    args = parser.parse_args()

    handler = functools.partial(QuietHandler, directory=str(ROOT / "web"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The throughput benchmark is skipped: on a software adapter it is slow and meaningless.
    url = f"http://127.0.0.1:{server.server_address[1]}/parity.html?bench=0"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=CHROMIUM_FLAGS)
        page = browser.new_page()
        page.on("console", lambda msg: print(f"[browser] {msg.text}"))
        page.goto(url)
        page.wait_for_function("window.parityResult !== undefined", timeout=args.timeout * 1000)
        result = page.evaluate("window.parityResult")
        browser.close()
    server.shutdown()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "webgpu_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"adapter: {result.get('adapter')}")
    for case in result.get("cases", []):
        status = "pass" if case["pass"] else "FAIL"
        print(f"  {status}  max|Δ| = {case['maxDiff']:.2e} px  {case['name']}")
    if "error" in result:
        print(f"error: {result['error']}")
    print("PARITY OK" if result.get("pass") else "PARITY FAILED")
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
