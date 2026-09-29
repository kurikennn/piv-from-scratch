// Normalised median test (Westerweel & Scarano 2005), a port of pivfs.validate.

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

function neighbours(a, nx, ny, i, j, skip) {
  const out = [];
  for (let dj = -1; dj <= 1; dj++) {
    for (let di = -1; di <= 1; di++) {
      if (!di && !dj) continue;
      const ii = i + di;
      const jj = j + dj;
      if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
      const k = jj * nx + ii;
      if (skip && skip[k]) continue;
      out.push(a[k]);
    }
  }
  return out;
}

function residual(a, nx, ny, i, j, eps) {
  const nb = neighbours(a, nx, ny, i, j);
  const um = median(nb);
  const rm = median(nb.map((v) => Math.abs(v - um)));
  return Math.abs(a[j * nx + i] - um) / (rm + eps);
}

/** Flags (Uint8Array) and replaced copies of u, v. */
export function medianTest(u, v, nx, ny, eps = 0.1, threshold = 2) {
  const flag = new Uint8Array(u.length);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const r = Math.hypot(residual(u, nx, ny, i, j, eps), residual(v, nx, ny, i, j, eps));
      flag[j * nx + i] = r > threshold ? 1 : 0;
    }
  }
  const uu = Float32Array.from(u);
  const vv = Float32Array.from(v);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!flag[k]) continue;
      for (const [src, dst] of [[u, uu], [v, vv]]) {
        let nb = neighbours(src, nx, ny, i, j, flag);
        if (!nb.length) nb = neighbours(src, nx, ny, i, j);
        if (nb.length) dst[k] = median(nb);
      }
    }
  }
  return { flag, u: uu, v: vv };
}
