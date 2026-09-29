// WebGPU pipelines: particle rendering, direct correlation, and image display.

import { correlateWGSL, MAX_RADIUS, MAX_WINDOW, renderWGSL } from "./shaders.js";

export const ESTIMATORS = ["gaussian", "parabolic", "centroid"];
const CELL = 8; // px; must exceed the largest particle support radius (ceil(5) + 1 = 6)

export async function requestDevice() {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found.");
  const device = await adapter.requestDevice();
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .join(" ");
  return { device, adapterName: name || "unknown adapter" };
}

/** Interrogation-window grid, identical to `pivfs.piv.window_grid`. */
export function windowGrid(width, height, win, overlap) {
  const step = Math.max(1, Math.round(win * (1 - overlap)));
  const axis = (n) => {
    const count = Math.floor((n - win) / step) + 1;
    const last = (count - 1) * step;
    return { count, offset: Math.floor((n - win - last) / 2) };
  };
  const gx = axis(width);
  const gy = axis(height);
  return { step, nx: gx.count, ny: gy.count, x0: gx.offset, y0: gy.offset };
}

/** Effective search radius, identical to `pivfs.piv.search_mask`. */
export function effectiveRadius(win, radius = MAX_RADIUS) {
  return Math.min(radius, win / 2 - 1);
}

export class PivGpu {
  constructor(device) {
    this.device = device;
    const mk = (code, entryPoint = "main") =>
      device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint },
      });
    this.correlatePipeline = mk(correlateWGSL);
    this.renderPipeline = mk(renderWGSL);
    this.corrParams = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.renderParams = [0, 1].map(() =>
      device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );
    this.width = 0;
    this.height = 0;
    this.particles = [null, null];
    this.cells = [null, null];
    this.resultSize = 0;
  }

  /** (Re)allocate the two image buffers for a new image size. */
  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    for (const b of this.images ?? []) b.destroy();
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.images = [0, 1].map(() => this.device.createBuffer({ size: width * height * 4, usage }));
  }

  /** Upload two images (Float32Array of grey levels) directly, e.g. parity fixtures. */
  uploadImages(a, b, width, height) {
    this.resize(width, height);
    this.device.queue.writeBuffer(this.images[0], 0, a);
    this.device.queue.writeBuffer(this.images[1], 0, b);
  }

  _ensure(list, i, bytes, usage) {
    if (!list[i] || list[i].size < bytes) {
      list[i]?.destroy();
      list[i] = this.device.createBuffer({ size: Math.max(bytes, 16), usage });
    }
    return list[i];
  }

  /**
   * Render both frames on the GPU.
   * frames: [{x, y, peak}, {x, y, peak}] particle arrays (Float32Array) for frames A and B.
   * cam: {dTau, noise, bitDepth, seed}
   */
  render(frames, cam) {
    const { device, width, height } = this;
    const cellsX = Math.ceil(width / CELL);
    const cellsY = Math.ceil(height / CELL);
    const radius = Math.ceil(cam.dTau) + 1;
    const levels = 2 ** cam.bitDepth - 1;
    const encoder = device.createCommandEncoder();
    frames.forEach((frame, f) => {
      const { packed, starts } = binParticles(frame, cellsX, cellsY);
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const pb = this._ensure(this.particles, f, packed.byteLength, usage);
      const cb = this._ensure(this.cells, f, starts.byteLength, usage);
      device.queue.writeBuffer(pb, 0, packed);
      device.queue.writeBuffer(cb, 0, starts);
      const params = new ArrayBuffer(48);
      new Uint32Array(params, 0, 5).set([width, height, cellsX, cellsY, CELL]);
      new Int32Array(params, 20, 1)[0] = radius;
      new Uint32Array(params, 24, 1)[0] = (cam.seed * 2 + f) >>> 0;
      new Float32Array(params, 32, 4).set([
        Math.sqrt(8) / cam.dTau,
        (Math.PI * cam.dTau ** 2) / 32,
        cam.noise,
        levels,
      ]);
      const ub = this.renderParams[f];
      device.queue.writeBuffer(ub, 0, params);
      const bind = device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pb } },
          { binding: 1, resource: { buffer: cb } },
          { binding: 2, resource: { buffer: this.images[f] } },
          { binding: 3, resource: { buffer: ub } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
      pass.end();
    });
    device.queue.submit([encoder.finish()]);
    return levels;
  }

  /** Encode `repeat` correlation passes over the current images; returns the grid. */
  _encodeCorrelation(encoder, win, overlap, estimator, repeat = 1) {
    if (win > MAX_WINDOW || win % 2) throw new Error(`window must be even and <= ${MAX_WINDOW}`);
    const grid = windowGrid(this.width, this.height, win, overlap);
    const radius = effectiveRadius(win);
    const n = grid.nx * grid.ny;
    if (this.resultSize < n) {
      this.result?.destroy();
      this.readback?.destroy();
      this.result = this.device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.readback = this.device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.resultSize = n;
    }
    const params = new ArrayBuffer(48);
    new Uint32Array(params, 0, 8).set([
      this.width, this.height, win, grid.step, grid.nx, grid.ny, grid.x0, grid.y0,
    ]);
    new Int32Array(params, 32, 1)[0] = radius;
    new Uint32Array(params, 36, 1)[0] = ESTIMATORS.indexOf(estimator);
    this.device.queue.writeBuffer(this.corrParams, 0, params);
    const bind = this.device.createBindGroup({
      layout: this.correlatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.images[0] } },
        { binding: 1, resource: { buffer: this.images[1] } },
        { binding: 2, resource: { buffer: this.result } },
        { binding: 3, resource: { buffer: this.corrParams } },
      ],
    });
    for (let r = 0; r < repeat; r++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.correlatePipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(grid.nx, grid.ny);
      pass.end();
    }
    return { ...grid, win, radius, n };
  }

  /**
   * Single-pass PIV on the current images. Returns grid info and Float32Arrays
   * x, y (vector centres), u, v, peakRatio (Infinity where the secondary peak <= 0)
   * and fallback (1 where the estimator fell back to the parabolic fit).
   */
  async correlate({ win = 32, overlap = 0.5, estimator = "gaussian" } = {}) {
    const encoder = this.device.createCommandEncoder();
    const grid = this._encodeCorrelation(encoder, win, overlap, estimator);
    encoder.copyBufferToBuffer(this.result, 0, this.readback, 0, grid.n * 16);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ, 0, grid.n * 16);
    const raw = new Float32Array(this.readback.getMappedRange(0, grid.n * 16).slice(0));
    this.readback.unmap();
    const out = {
      ...grid,
      x: new Float32Array(grid.n),
      y: new Float32Array(grid.n),
      u: new Float32Array(grid.n),
      v: new Float32Array(grid.n),
      peakRatio: new Float32Array(grid.n),
      fallback: new Uint8Array(grid.n),
    };
    for (let j = 0; j < grid.ny; j++) {
      for (let i = 0; i < grid.nx; i++) {
        const k = j * grid.nx + i;
        out.x[k] = grid.x0 + i * grid.step + (win - 1) / 2;
        out.y[k] = grid.y0 + j * grid.step + (win - 1) / 2;
        out.u[k] = raw[4 * k];
        out.v[k] = raw[4 * k + 1];
        out.peakRatio[k] = raw[4 * k + 2] < 0 ? Infinity : raw[4 * k + 2];
        out.fallback[k] = raw[4 * k + 3];
      }
    }
    return out;
  }

  /** Read both rendered frames back as Float32Arrays of grey levels. */
  async readImages() {
    const bytes = this.width * this.height * 4;
    if (!this.imageReadback || this.imageReadback.size < 2 * bytes) {
      this.imageReadback?.destroy();
      this.imageReadback = this.device.createBuffer({
        size: 2 * bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.images[0], 0, this.imageReadback, 0, bytes);
    encoder.copyBufferToBuffer(this.images[1], 0, this.imageReadback, bytes, bytes);
    this.device.queue.submit([encoder.finish()]);
    await this.imageReadback.mapAsync(GPUMapMode.READ, 0, 2 * bytes);
    const data = new Float32Array(this.imageReadback.getMappedRange(0, 2 * bytes).slice(0));
    this.imageReadback.unmap();
    return [data.subarray(0, bytes / 4), data.subarray(bytes / 4)];
  }

  /** Time `repeat` correlation passes on the current images; returns vectors/s. */
  async benchmark({ win = 32, overlap = 0.5, estimator = "gaussian", repeat = 20 } = {}) {
    await this.correlate({ win, overlap, estimator }); // warm-up and pipeline compile
    const encoder = this.device.createCommandEncoder();
    const grid = this._encodeCorrelation(encoder, win, overlap, estimator, repeat);
    const t0 = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const seconds = (performance.now() - t0) / 1000;
    return { vectorsPerSecond: (grid.n * repeat) / seconds, msPerPass: (seconds * 1000) / repeat, n: grid.n };
  }
}

/** Counting sort of particles into CELL×CELL bins (edge cells also hold particles outside the image). */
function binParticles({ x, y, peak }, cellsX, cellsY) {
  const n = x.length;
  const cellOf = new Uint32Array(n);
  const counts = new Uint32Array(cellsX * cellsY + 1);
  for (let k = 0; k < n; k++) {
    const cx = Math.min(cellsX - 1, Math.max(0, Math.floor(x[k] / CELL)));
    const cy = Math.min(cellsY - 1, Math.max(0, Math.floor(y[k] / CELL)));
    cellOf[k] = cy * cellsX + cx;
    counts[cellOf[k] + 1]++;
  }
  for (let c = 1; c < counts.length; c++) counts[c] += counts[c - 1];
  const starts = counts.slice();
  const cursor = counts.slice(0, -1);
  const packed = new Float32Array(Math.max(n, 1) * 4);
  for (let k = 0; k < n; k++) {
    const dst = cursor[cellOf[k]]++;
    packed.set([x[k], y[k], peak[k], 0], dst * 4);
  }
  return { packed, starts };
}
