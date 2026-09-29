// Runs the Python-exported fixtures through the WGSL correlation and compares.
// The outcome is shown on the page and exposed as window.parityResult for CI.

import { PivGpu, requestDevice } from "./gpu.js";

function decode(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return Float32Array.from(bytes);
}

async function run() {
  const log = document.querySelector("#log");
  const table = document.querySelector("#cases tbody");
  const { device, adapterName } = await requestDevice();
  document.querySelector("#adapter").textContent = adapterName;
  const gpu = new PivGpu(device);
  const fixtures = await (await fetch("fixtures/parity.json")).json();
  const tol = fixtures.tolerance_px;
  const cases = [];
  for (const c of fixtures.cases) {
    gpu.uploadImages(decode(c.img_a), decode(c.img_b), c.width, c.height);
    const r = await gpu.correlate({ win: c.window, overlap: c.overlap, estimator: c.estimator });
    if (r.n !== c.u.length) throw new Error(`${c.name}: ${r.n} vectors, Python has ${c.u.length}`);
    let maxDiff = 0;
    let over = 0;
    let gridDiff = 0;
    for (let k = 0; k < r.n; k++) {
      const d = Math.max(Math.abs(r.u[k] - c.u[k]), Math.abs(r.v[k] - c.v[k]));
      maxDiff = Math.max(maxDiff, d);
      if (d > tol) over++;
      gridDiff = Math.max(gridDiff, Math.abs(r.x[k] - c.x[k]), Math.abs(r.y[k] - c.y[k]));
    }
    const pass = over === 0 && gridDiff === 0;
    cases.push({ name: c.name, vectors: r.n, maxDiff, over, gridDiff, pass });
    const row = table.insertRow();
    for (const v of [c.name, r.n, maxDiff.toExponential(2), over, pass ? "pass" : "FAIL"]) {
      row.insertCell().textContent = v;
    }
    row.className = pass ? "pass" : "fail";
  }

  // ?bench=0 skips the throughput benchmark (CI runs on a software GPU, where it is slow and meaningless).
  const bench = {};
  if (new URLSearchParams(location.search).get("bench") !== "0") {
    gpu.resize(1024, 1024);
    const rand = new Float32Array(1024 * 1024).map(() => Math.floor(Math.random() * 256));
    gpu.uploadImages(rand, rand, 1024, 1024);
    for (const win of [16, 32]) bench[win] = await gpu.benchmark({ win, repeat: 20 });
  }

  const pass = cases.every((c) => c.pass);
  window.parityResult = { adapter: adapterName, tolerance_px: tol, pass, cases, benchmark: bench };
  log.textContent = pass
    ? `All ${cases.length} fixtures match Python within ${tol} px.`
    : "Parity FAILED.";
  document.querySelector("#bench").textContent = bench[32]
    ? [16, 32]
        .map((w) => `W=${w}: ${Math.round(bench[w].vectorsPerSecond).toLocaleString()} vectors/s`)
        .join(" · ")
    : "skipped";
}

run().catch((err) => {
  document.querySelector("#log").textContent = `Error: ${err.message}`;
  window.parityResult = { pass: false, error: String(err) };
});
