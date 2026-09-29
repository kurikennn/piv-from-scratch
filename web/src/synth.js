// Particle seeding and advection for the live demo (rendering happens on the GPU).
// Mirrors pivfs.synth / pivfs.flows: uniform seeding in the image grown by a margin,
// Lamb–Oseen vortex, RK4 pathlines ½ frame backward (A) and forward (B) from mid-time.

const LAMB_OSEEN_PEAK_FACTOR = 0.6382;
const MARGIN = 16;

/** Small seeded PRNG (mulberry32). */
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(next() + 1e-12)) * Math.cos(2 * Math.PI * next());
  return { next, normal };
}

/** Lamb–Oseen vortex u_θ(r) = Γ/(2πr)(1 − exp(−r²/r_c²)), Γ = 2π r_c v_max / 0.6382. */
export function lambOseen({ x0, y0, rc, vmax }) {
  const k = (2 * Math.PI * rc * vmax) / LAMB_OSEEN_PEAK_FACTOR / (2 * Math.PI);
  return (x, y) => {
    const rx = x - x0;
    const ry = y - y0;
    const r2 = rx * rx + ry * ry;
    const omega = r2 > 0 ? (-Math.expm1(-r2 / (rc * rc)) / r2) * k : k / (rc * rc);
    return [-omega * ry, omega * rx];
  };
}

function rk4(flow, x, y, dt, steps = 4) {
  const h = dt / steps;
  for (let s = 0; s < steps; s++) {
    const [k1x, k1y] = flow(x, y);
    const [k2x, k2y] = flow(x + 0.5 * h * k1x, y + 0.5 * h * k1y);
    const [k3x, k3y] = flow(x + 0.5 * h * k2x, y + 0.5 * h * k2y);
    const [k4x, k4y] = flow(x + h * k3x, y + h * k3y);
    x += (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    y += (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y);
  }
  return [x, y];
}

/** Particle positions in both frames: [{x, y, peak}, {x, y, peak}] as Float32Arrays. */
export function particlePair({ width, height, ppp, seed, flow }) {
  const r = rng(seed);
  const w = width + 2 * MARGIN;
  const h = height + 2 * MARGIN;
  const mean = ppp * w * h;
  const n = Math.max(0, Math.round(mean + Math.sqrt(mean) * r.normal())); // Poisson ≈ normal
  const frames = [0, 1].map(() => ({
    x: new Float32Array(n),
    y: new Float32Array(n),
    peak: new Float32Array(n),
  }));
  for (let k = 0; k < n; k++) {
    const xm = -0.5 - MARGIN + r.next() * w;
    const ym = -0.5 - MARGIN + r.next() * h;
    const peak = 0.5 + 0.5 * r.next();
    const [xa, ya] = rk4(flow, xm, ym, -0.5);
    const [xb, yb] = rk4(flow, xm, ym, +0.5);
    frames[0].x[k] = xa;
    frames[0].y[k] = ya;
    frames[1].x[k] = xb;
    frames[1].y[k] = yb;
    frames[0].peak[k] = frames[1].peak[k] = peak;
  }
  return frames;
}
