// WGSL compute and render shaders.
//
// The correlation shader implements exactly the algorithm of
// `pivfs.piv.correlate_pass(..., search_radius=R)` in the Python package:
// mean-subtracted circular cross-correlation normalised by the window energies,
// peak search over |s| ≤ R, finite-window weighting divided out, then a
// three-point sub-pixel fit. It is evaluated directly in the spatial domain
// (no FFT), one workgroup per interrogation window.

export const MAX_WINDOW = 32;
export const MAX_RADIUS = 8;

export const correlateWGSL = /* wgsl */ `
struct Params {
  width: u32, height: u32, win: u32, step: u32,
  nx: u32, ny: u32, x0: u32, y0: u32,
  radius: i32, estimator: u32, pad0: u32, pad1: u32,
};

@group(0) @binding(0) var<storage, read> imgA: array<f32>;
@group(0) @binding(1) var<storage, read> imgB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;  // u, v, peak ratio, fallback
@group(0) @binding(3) var<uniform> P: Params;

const THREADS: u32 = 256u;
const MAX_PLANE: u32 = ${(2 * MAX_RADIUS + 3) ** 2}u;  // shifts in [-R-1, R+1]²

var<workgroup> wa: array<f32, ${MAX_WINDOW * MAX_WINDOW}>;
var<workgroup> wb: array<f32, ${MAX_WINDOW * MAX_WINDOW}>;
var<workgroup> plane: array<f32, MAX_PLANE>;
var<workgroup> redA: array<f32, THREADS>;
var<workgroup> redB: array<f32, THREADS>;

fn reduce(li: u32) {
  // Tree reduction of redA and redB; the totals end up in element 0.
  for (var s = THREADS / 2u; s > 0u; s = s >> 1u) {
    if (li < s) {
      redA[li] += redA[li + s];
      redB[li] += redB[li + s];
    }
    workgroupBarrier();
  }
}

// Circular distance between two shifts on a period-W lattice.
fn circ(d: i32, w: i32) -> i32 {
  let m = ((d % w) + w) % w;
  return min(m, w - m);
}

// Finite-window weighting w(s) = (1 - |sx|/W)(1 - |sy|/W).
fn weight(sx: i32, sy: i32, w: f32) -> f32 {
  return (1.0 - f32(abs(sx)) / w) * (1.0 - f32(abs(sy)) / w);
}

fn parabolic3(cm: f32, c0: f32, cp: f32) -> f32 {
  let den = 2.0 * cm - 4.0 * c0 + 2.0 * cp;
  if (den < 0.0) { return (cm - cp) / den; }
  return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let W = P.win;
  let n = W * W;
  let ox = P.x0 + wg.x * P.step;
  let oy = P.y0 + wg.y * P.step;

  // 1. Load both windows and sum them.
  var sa = 0.0;
  var sb = 0.0;
  for (var k = li; k < n; k += THREADS) {
    let g = (oy + k / W) * P.width + ox + k % W;
    let a = imgA[g];
    let b = imgB[g];
    wa[k] = a;
    wb[k] = b;
    sa += a;
    sb += b;
  }
  redA[li] = sa;
  redB[li] = sb;
  workgroupBarrier();
  reduce(li);
  let meanA = redA[0] / f32(n);
  let meanB = redB[0] / f32(n);
  workgroupBarrier();

  // 2. Subtract the means; sum of squares for the normalisation.
  var qa = 0.0;
  var qb = 0.0;
  for (var k = li; k < n; k += THREADS) {
    let a = wa[k] - meanA;
    let b = wb[k] - meanB;
    wa[k] = a;
    wb[k] = b;
    qa += a * a;
    qb += b * b;
  }
  redA[li] = qa;
  redB[li] = qb;
  workgroupBarrier();
  reduce(li);
  var norm = sqrt(redA[0] * redB[0]);
  if (norm <= 0.0) { norm = 1.0; }

  // 3. Circular correlation R(s) = sum_x a'(x) b'((x + s) mod W) / norm for |s| <= R + 1.
  let R = P.radius;
  let side = 2 * R + 3;
  let m = u32(side * side);
  let wi = i32(W);
  for (var e = li; e < m; e += THREADS) {
    let sy = i32(e) / side - (R + 1);
    let sx = i32(e) % side - (R + 1);
    var acc = 0.0;
    for (var y = 0; y < wi; y++) {
      let yb = ((y + sy + wi) % wi) * wi;
      let ya = y * wi;
      for (var x = 0; x < wi; x++) {
        acc += wa[ya + x] * wb[yb + (x + sx + wi) % wi];
      }
    }
    plane[e] = acc / norm;
  }
  workgroupBarrier();

  // 4. Peak search, peak ratio and sub-pixel fit (one thread).
  if (li != 0u) { return; }
  var best = -3.4e38;
  var py = 0;
  var px = 0;
  for (var sy = -R; sy <= R; sy++) {
    for (var sx = -R; sx <= R; sx++) {
      let c = plane[(sy + R + 1) * side + sx + R + 1];
      if (c > best) { best = c; py = sy; px = sx; }
    }
  }
  var second = -3.4e38;
  for (var sy = -R; sy <= R; sy++) {
    for (var sx = -R; sx <= R; sx++) {
      if (circ(sy - py, wi) <= 2 && circ(sx - px, wi) <= 2) { continue; }
      second = max(second, plane[(sy + R + 1) * side + sx + R + 1]);
    }
  }
  var ratio = -1.0;  // -1 encodes +inf (secondary peak <= 0)
  if (second > 0.0) { ratio = best / second; }

  // 3x3 neighbourhood with the finite-window weighting divided out.
  var c: array<f32, 9>;
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      let sy = py + i - 1;
      let sx = px + j - 1;
      c[i * 3 + j] = plane[(sy + R + 1) * side + sx + R + 1] / weight(sx, sy, f32(W));
    }
  }
  let dyPara = parabolic3(c[1], c[4], c[7]);
  let dxPara = parabolic3(c[3], c[4], c[5]);
  var dy = dyPara;
  var dx = dxPara;
  var fallback = 0.0;
  if (P.estimator == 0u) {  // Gaussian, per axis, falls back if a value <= 0
    if (c[1] > 0.0 && c[4] > 0.0 && c[7] > 0.0) {
      dy = parabolic3(log(c[1]), log(c[4]), log(c[7]));
    } else { fallback = 1.0; }
    if (c[3] > 0.0 && c[4] > 0.0 && c[5] > 0.0) {
      dx = parabolic3(log(c[3]), log(c[4]), log(c[5]));
    } else { fallback = 1.0; }
  } else if (P.estimator == 2u) {  // centroid over 3x3, falls back if any value <= 0
    var tot = 0.0;
    var my = 0.0;
    var mx = 0.0;
    var ok = true;
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        let v = c[i * 3 + j];
        ok = ok && v > 0.0;
        tot += v;
        my += f32(i - 1) * v;
        mx += f32(j - 1) * v;
      }
    }
    if (ok) { dy = my / tot; dx = mx / tot; } else { fallback = 1.0; }
  }
  result[wg.y * P.nx + wg.x] = vec4<f32>(f32(px) + dx, f32(py) + dy, ratio, fallback);
}
`;

export const renderWGSL = /* wgsl */ `
struct Params {
  width: u32, height: u32, cellsX: u32, cellsY: u32,
  cell: u32, radius: i32, seed: u32, pad0: u32,
  a: f32, scale: f32, noise: f32, levels: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<vec4<f32>>;  // x, y, peak, -
@group(0) @binding(1) var<storage, read> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> img: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

// erf by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                 + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}

fn hash(v: u32) -> u32 {  // PCG hash
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn uniform01(v: u32) -> f32 {
  return (f32(hash(v) >> 8u) + 0.5) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let px = i32(gid.x);
  let py = i32(gid.y);
  let cx = px / i32(P.cell);
  let cy = py / i32(P.cell);
  var sum = 0.0;
  for (var j = max(cy - 1, 0); j <= min(cy + 1, i32(P.cellsY) - 1); j++) {
    for (var i = max(cx - 1, 0); i <= min(cx + 1, i32(P.cellsX) - 1); i++) {
      let cell = u32(j) * P.cellsX + u32(i);
      for (var k = cellStart[cell]; k < cellStart[cell + 1u]; k++) {
        let p = particles[k];
        // Same support as the Python renderer: |pixel - round(particle)| <= radius.
        if (abs(px - i32(round(p.x))) > P.radius || abs(py - i32(round(p.y))) > P.radius) {
          continue;
        }
        let ex = erf(P.a * (f32(px) + 0.5 - p.x)) - erf(P.a * (f32(px) - 0.5 - p.x));
        let ey = erf(P.a * (f32(py) + 0.5 - p.y)) - erf(P.a * (f32(py) - 0.5 - p.y));
        sum += p.z * P.scale * ex * ey;
      }
    }
  }
  let idx = gid.y * P.width + gid.x;
  if (P.noise > 0.0) {  // Box-Muller
    let u1 = uniform01(idx * 2u + P.seed * 0x9E3779B9u);
    let u2 = uniform01(idx * 2u + 1u + P.seed * 0x9E3779B9u);
    sum += P.noise * sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
  }
  img[idx] = round(clamp(sum, 0.0, 1.0) * P.levels);
}
`;
