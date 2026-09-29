// Live demo: GPU particle rendering + GPU correlation, error against the exact vortex.

import { PivGpu, requestDevice } from "./gpu.js";
import { lambOseen, particlePair } from "./synth.js";
import { medianTest } from "./validate.js";

const $ = (id) => document.getElementById(id);
const INTERIOR_MARGIN = 32; // px, as in pivfs.metrics.interior_mask
const OUTLIER_PX = 0.5;
const ERROR_MAX = 0.5; // top of the error colour ramp, px

// Viridis from 0.2 upward so that the lowest errors stay visible on the dark image.
const VIRIDIS = [
  [65, 68, 135], [42, 120, 142], [34, 168, 132], [122, 209, 81], [253, 231, 37],
];
const OUTLIER_COLOR = "#ff5a36";
const DISPLAY_GAIN = 0.75; // dimmed so the vectors stand out
const OVERLAY_SCALE = 2; // vector overlay is drawn at 2× the image resolution

/** Grey-level image → ImageData (with a display gain; particles peak at 50–100 % of full scale). */
function toImageData(img, size, levels) {
  const out = new ImageData(size, size);
  const k = (255 * DISPLAY_GAIN) / levels;
  for (let i = 0; i < img.length; i++) {
    const g = Math.min(255, img[i] * k);
    out.data[4 * i] = out.data[4 * i + 1] = out.data[4 * i + 2] = g;
    out.data[4 * i + 3] = 255;
  }
  return out;
}

function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const c = VIRIDIS[i].map((v, k) => Math.round(v + f * (VIRIDIS[i + 1][k] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const controls = {
  dTau: { fmt: (v) => `${v.toFixed(2)} px` },
  ppp: { fmt: (v) => v.toFixed(3) },
  noise: { fmt: (v) => `${v.toFixed(1)} %` },
  vmax: { fmt: (v) => `${v.toFixed(2)} px/frame` },
  arrow: { fmt: (v) => `×${v}` },
};

const state = { seed: 1, dirty: true, busy: false, animate: false, last: null, flip: 0 };

function params() {
  return {
    dTau: +$("dTau").value,
    ppp: +$("ppp").value,
    noise: +$("noise").value / 100,
    vmax: +$("vmax").value,
    win: +$("win").value,
    size: +$("size").value,
    estimator: $("estimator").value,
    validate: $("validate").checked,
    arrow: +$("arrow").value,
  };
}

function vortexFor(size, vmax) {
  return { x0: (size - 1) / 2, y0: (size - 1) / 2, rc: (40 * size) / 512, vmax };
}

/** Generate a pair on the GPU, correlate it, and score it against the exact flow. */
async function compute(gpu, p, seed) {
  const vortex = vortexFor(p.size, p.vmax);
  const flow = lambOseen(vortex);
  const t0 = performance.now();
  const frames = particlePair({ width: p.size, height: p.size, ppp: p.ppp, seed, flow });
  const t1 = performance.now();
  gpu.resize(p.size, p.size);
  const levels = gpu.render(frames, { dTau: p.dTau, noise: p.noise, bitDepth: 8, seed });
  const r = await gpu.correlate({ win: p.win, overlap: 0.5, estimator: p.estimator });
  const t2 = performance.now();
  const images = (await gpu.readImages()).map((img) => toImageData(img, p.size, levels));

  let u = r.u;
  let v = r.v;
  let flag = new Uint8Array(r.n);
  if (p.validate) ({ flag, u, v } = medianTest(r.u, r.v, r.nx, r.ny));

  const err = new Float32Array(r.n);
  let sum = 0, n = 0, coreSum = 0, coreN = 0, bad = 0, flagged = 0, fallback = 0;
  for (let k = 0; k < r.n; k++) {
    const [ut, vt] = flow(r.x[k], r.y[k]);
    err[k] = Math.hypot(u[k] - ut, v[k] - vt);
    fallback += r.fallback[k];
    const interior =
      r.x[k] >= INTERIOR_MARGIN && r.x[k] <= p.size - 1 - INTERIOR_MARGIN &&
      r.y[k] >= INTERIOR_MARGIN && r.y[k] <= p.size - 1 - INTERIOR_MARGIN;
    if (!interior) continue;
    n++;
    sum += err[k] ** 2;
    bad += err[k] > OUTLIER_PX;
    flagged += flag[k];
    if (Math.hypot(r.x[k] - vortex.x0, r.y[k] - vortex.y0) < 2 * vortex.rc) {
      coreN++;
      coreSum += err[k] ** 2;
    }
  }
  return {
    ...r, u, v, flag, err, images, particles: frames[0].x.length,
    stats: {
      rms: Math.sqrt(sum / n),
      core: Math.sqrt(coreSum / coreN),
      bad: (100 * bad) / n,
      flagged: (100 * flagged) / n,
      fallback: (100 * fallback) / r.n,
      cpuMs: t1 - t0,
      gpuMs: t2 - t1,
    },
  };
}

function drawVectors(ctx, res, scale) {
  const { width, height } = ctx.canvas;
  const px = width / res.size; // overlay pixels per image pixel
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1.4 * px * Math.max(1, res.size / 512);
  ctx.lineCap = "round";
  const head = 3 * ctx.lineWidth;
  for (let k = 0; k < res.n; k++) {
    const x = (res.x[k] + 0.5) * px;
    const y = (res.y[k] + 0.5) * px;
    const dx = res.u[k] * scale * px;
    const dy = res.v[k] * scale * px;
    const len = Math.hypot(dx, dy);
    ctx.strokeStyle = ctx.fillStyle = res.flag[k] ? OUTLIER_COLOR : viridis(res.err[k] / ERROR_MAX);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx, y + dy);
    ctx.stroke();
    if (len > head) {
      const ux = dx / len, uy = dy / len;
      ctx.beginPath();
      ctx.moveTo(x + dx, y + dy);
      ctx.lineTo(x + dx - head * (ux - 0.5 * uy), y + dy - head * (uy + 0.5 * ux));
      ctx.lineTo(x + dx - head * (ux + 0.5 * uy), y + dy - head * (uy - 0.5 * ux));
      ctx.fill();
    }
  }
}

function showStats(res) {
  const s = res.stats;
  $("s-rms").textContent = `${s.rms.toFixed(4)} px`;
  $("s-core").textContent = `${s.core.toFixed(4)} px`;
  $("s-out").textContent = `${s.bad.toFixed(2)} %`;
  $("s-flag").textContent = `${s.flagged.toFixed(2)} %`;
  $("s-fb").textContent = `${s.fallback.toFixed(1)} %`;
  $("s-n").textContent = `${res.nx} × ${res.ny} = ${res.n.toLocaleString()}`;
  $("s-cpu").textContent = `${res.particles.toLocaleString()} in ${s.cpuMs.toFixed(1)} ms`;
  $("s-gpu").textContent = `${s.gpuMs.toFixed(1)} ms (incl. readback)`;
}

async function benchmark(gpu) {
  const out = $("bench-out");
  const button = $("bench");
  button.disabled = true;
  out.textContent = "Running…";
  state.busy = true;
  try {
    const p = { ...params(), size: 1024 };
    const flow = lambOseen(vortexFor(1024, p.vmax));
    gpu.resize(1024, 1024);
    gpu.render(particlePair({ width: 1024, height: 1024, ppp: p.ppp, seed: 7, flow }),
      { dTau: p.dTau, noise: p.noise, bitDepth: 8, seed: 7 });
    let numpy = null;
    try {
      numpy = await (await fetch("data/throughput_numpy.json")).json();
    } catch { /* optional */ }
    const lines = [];
    for (const win of [16, 32]) {
      const g = await gpu.benchmark({ win, estimator: p.estimator, repeat: 30 });
      const np = numpy?.runs.find((r) => r.window === win && r.search_radius === null);
      let line = `W = ${win}: GPU ${Math.round(g.vectorsPerSecond).toLocaleString()} vectors/s (${g.msPerPass.toFixed(2)} ms/pass)`;
      if (np) {
        const ratio = g.vectorsPerSecond / np.vectors_per_second;
        line += `, NumPy ${Math.round(np.vectors_per_second).toLocaleString()} vectors/s → ×${ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(2)}`;
      }
      lines.push(line);
    }
    out.innerHTML = lines.join("<br>") +
      (numpy ? `<br><span class="small">NumPy measured on ${numpy.cpu}, numpy ${numpy.numpy}.</span>` : "");
  } catch (err) {
    out.textContent = `Benchmark failed: ${err.message}`;
  } finally {
    button.disabled = false;
    state.busy = false;
    state.dirty = true;
  }
}

async function main() {
  for (const [id, c] of Object.entries(controls)) {
    const update = () => ($(`${id}-out`).textContent = c.fmt(+$(id).value));
    $(id).addEventListener("input", () => { update(); state.dirty = true; });
    update();
  }
  for (const id of ["win", "size", "estimator", "validate"]) {
    $(id).addEventListener("change", () => (state.dirty = true));
  }

  let gpuInfo;
  try {
    gpuInfo = await requestDevice();
  } catch (err) {
    const box = $("unsupported");
    box.hidden = false;
    box.textContent = `${err.message} Try a recent Chrome or Edge (desktop), or Safari 26+. ` +
      "The static figures and numbers are in the repository README.";
    return;
  }
  const { device, adapterName } = gpuInfo;
  $("adapter").textContent = `Adapter: ${adapterName}`;
  device.lost.then((info) => {
    $("unsupported").hidden = false;
    $("unsupported").textContent = `GPU device lost: ${info.message}`;
  });

  const gpu = new PivGpu(device);
  const imageCanvas = $("image");
  const overlay = $("overlay");
  const view = imageCanvas.getContext("2d");
  const ctx = overlay.getContext("2d");

  $("reseed").addEventListener("click", () => { state.seed++; state.dirty = true; });
  $("animate").addEventListener("click", (e) => {
    state.animate = !state.animate;
    e.target.setAttribute("aria-pressed", String(state.animate));
    e.target.textContent = state.animate ? "Stop" : "Animate";
  });
  $("arrow").addEventListener("input", () => state.last && drawVectors(ctx, state.last, +$("arrow").value));
  $("bench").addEventListener("click", () => benchmark(gpu));

  const draw = () => {
    const res = state.last;
    if (!res || state.busy) return;
    const mode = $("view").value;
    const frame = mode === "flicker" ? state.flip : +mode;
    view.putImageData(res.images[frame], 0, 0);
  };
  setInterval(() => { state.flip ^= 1; if ($("view").value === "flicker") draw(); }, 400);
  $("view").addEventListener("change", draw);

  const tick = async () => {
    if (!state.busy && (state.dirty || state.animate)) {
      state.busy = true;
      state.dirty = false;
      if (state.animate) state.seed++;
      const p = params();
      try {
        const res = await compute(gpu, p, state.seed);
        res.size = p.size;
        if (imageCanvas.width !== p.size) {
          imageCanvas.width = imageCanvas.height = p.size;
          overlay.width = overlay.height = p.size * OVERLAY_SCALE;
        }
        state.last = res;
        state.busy = false;
        draw();
        drawVectors(ctx, res, p.arrow);
        showStats(res);
      } catch (err) {
        console.error(err);
        state.busy = false;
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

main();
