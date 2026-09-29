# Method

This note documents every step of the processor and generator, with the equation each
function implements. Code references are to `src/pivfs/`.

**Conventions.** `x` is the column index and `y` the row index (y points down). Pixel
`(i, j)` covers `[j − ½, j + ½] × [i − ½, i + ½]`, so its centre sits at integer
coordinates. Displacements are in px/frame.

## 1. Synthetic images (`synth.py`, `flows.py`)

**Particle image.** A particle at `(x_p, y_p)` with peak intensity `I₀` has the continuous
image `I(x, y) = I₀ exp(−8 r² / d_τ²)`, so `d_τ` is the e⁻² diameter (σ = d_τ / 4). Each
pixel records the integral of `I` over its area (Raffel et al. 2018, eq. 6.2):

```
I_ij = I₀ (π d_τ² / 32) · [erf(√8 (j + ½ − x_p)/d_τ) − erf(√8 (j − ½ − x_p)/d_τ)]
                        · [erf(√8 (i + ½ − y_p)/d_τ) − erf(√8 (i − ½ − y_p)/d_τ)]
```

Sampling at pixel centres instead would lose energy and distort the shape for d_τ ≲ 1 px,
exactly where peak locking is studied. The test suite checks energy conservation
(`Σ I_ij = I₀ π d_τ² / 8`) and recovers d_τ from second moments:
`d_τ = 4 √(var − 1/12)`, where 1/12 px² is the variance added by the 1-px box.

**Seeding and camera.** The particle count is Poisson with mean `N_ppp × area`, drawn in the
image grown by a 16 px margin. Peaks are drawn from U(0.5, 1.0) of full scale. Frames get
Gaussian noise `N(0, σ²)`, are clipped to [0, 1] and quantised to `2^b − 1` levels. With
out-of-plane loss `f`, a fraction `f` of particles appears only in A, and as many new ones
appear only in B.

**Displacement.** Each particle has a mid-time position `x_m`. Frame A is `x_m` advected
½ frame backward along the pathline, and frame B ½ frame forward, both with RK4
(4 sub-steps). The ground truth at a vector location `x` is the Eulerian `u(x)`. In curved
flow the pathline chord differs from `u(x_m)` by `O(|u|² |∇u| / 24)`, about 6·10⁻³ px at the
default vortex's peak. This small, known modelling error is included in every figure.

**Lamb–Oseen vortex.** `u_θ(r) = Γ/(2πr) · (1 − exp(−r²/r_c²))`. The maximum satisfies
`exp(s²) = 1 + 2s²` with `s = r/r_c`, giving `s* = 1.1209` and `u_θ,max = 0.6382 Γ/(2π r_c)`.
Hence `Γ = 2π r_c v_max / 0.6382`. At `r → 0`, `u_θ / r → Γ / (2π r_c²)`.

## 2. Correlation (`piv.py`)

**Grid.** Windows of `W × W` px step by `Δ = W (1 − overlap)`. Windows that cross the
border are dropped, and the grid is centred. A window starting at `s` has its vector at
`s + (W − 1)/2`.

**FFT cross-correlation** (Willert & Gharib 1991). For zero-mean windows `a' = a − ā` and
`b' = b − b̄`, all windows are stacked and transformed at once with `numpy.fft.rfft2`:

```
R(s) = F⁻¹{ conj(F{a'}) · F{b'} }(s) / (W² σ_a σ_b)
```

If `b(x) = a(x − d)`, the peak is at `s = d`. The plane is fft-shifted so `s = 0` sits at
index `W // 2`.

**Finite-window bias.** The correlation is circular, but at shift `s` only `W − |s|` of the
`W` samples per axis overlap with related content; the rest pair with wrapped, unrelated
pixels. The expected peak is therefore weighted by

```
w(s) = (1 − |s_x|/W)(1 − |s_y|/W)
```

and the sub-pixel fit is biased towards zero by roughly `σ_R² / W` (Raffel et al. 2018
§5.4.5; Westerweel 1997). With d_τ = 2.5 px and W = 32 px this was −0.029 px, the largest
error term in E1 before correction. `R / w` is used for the sub-pixel fit only. Peak search
and the peak ratio use the raw `R`, so the ×4 amplification at the plane's edges cannot
promote noise peaks.

**Peak ratio.** `R₁ / R₂`, where `R₂` is the highest value outside the 5 × 5 neighbourhood
of the primary peak `R₁`.

## 3. Sub-pixel estimators (`subpixel.py`)

With `c₋, c₀, c₊` the correlation at the integer peak `k` and its neighbours, the peak is
at `k + δ`. The estimators are applied separately in x and y.

- **Parabolic.** Fit `R(k + t) = a t² + b t + c` through the three points; the vertex is
  `δ = (c₋ − c₊) / (2c₋ − 4c₀ + 2c₊)`.
- **Gaussian (default).** Assume `R(k + t) = A exp(−(t − δ)² / 2s²)`. Then `ln R` is the
  parabola `ln A − (t − δ)²/2s²`, so the parabolic vertex applied to `ln c` is exact:

  ```
  δ = (ln c₋ − ln c₊) / (2 ln c₋ − 4 ln c₀ + 2 ln c₊)
  ```

  Derivation: write `L_t = ln c_t = α − (t − δ)²/(2s²)` for `t ∈ {−1, 0, 1}`. Then
  `L₋ − L₊ = [(1 + δ)² − (1 − δ)²]/(2s²) = 2δ/s²` and
  `L₋ − 2L₀ + L₊ = −[(1 + δ)² − 2δ² + (1 − δ)²]/(2s²) = −1/s²`. Dividing,
  `δ = (L₋ − L₊) / (−2 (L₋ − 2L₀ + L₊))`, which is the formula above.
- **Centroid 3 × 3.** `δ_x = Σ j c_ij / Σ c_ij` and `δ_y = Σ i c_ij / Σ c_ij` over
  `i, j ∈ {−1, 0, 1}`. It is biased towards the integer peak whenever the peak is wider than
  the 3 × 3 support, which is the textbook peak-locking estimator.

**Fallback.** The Gaussian needs `c₋, c₀, c₊ > 0` (per axis) and the centroid needs all nine
values > 0. Otherwise the parabolic fit is used, and every fallback is counted
(`meta["n_fallback"]`, `fallback_fraction` in the CSVs). For small particles the correlation
peak is narrower than a pixel, and mean subtraction makes its neighbours negative. At
d_τ = 0.5 px, 49 % of Gaussian fits fell back; at d_τ ≥ 2 px, none did.

## 4. Outlier detection (`validate.py`)

Normalised median test (Westerweel & Scarano 2005), applied to each component over the
eight 3 × 3 neighbours `U_i`:

```
U_m = median(U_i),  r_i = |U_i − U_m|,  r_m = median(r_i),  r₀* = |U₀ − U_m| / (r_m + ε)
```

with `ε = 0.1 px`. A vector is flagged when `√(r_u*² + r_v*²) > 2`. Border vectors use the
neighbours that exist. A flagged vector is replaced by the median of its unflagged
neighbours.

## 5. Multi-pass symmetric window deformation (`deform.py`, `piv.py`)

Default passes are `W = 64 → 32 → 16`, then 2 further iterations at `W = 16`, all at 50 %
overlap. Every pass is followed by the median test. From the second pass on:

1. **Filter the predictor.** Each component is smoothed on its grid with the binomial kernel
   `[1, 2, 1]/4` per axis. At 50 % overlap a top-hat window `W = 2Δ` has zero response at the
   grid Nyquist wavenumber (`sinc(π) = 0`). A predictor error in that checkerboard mode is
   therefore invisible to the residual correlation and is never corrected. Iterating then
   accumulates noise: for a (3, −2) px uniform shift, the maximum error grew from 5·10⁻³ px to
   4·10⁻² px over 8 iterations. The binomial filter also has zero response at Nyquist
   (`cos²(kΔ/2)`), which removes exactly that mode and makes the iteration converge
   (Scarano 2002; Schrijer & Scarano 2008). With the filter, the same case converges to
   < 1·10⁻³ px, and the vortex core error halves.
2. **Interpolate** the filtered field to every pixel with a cubic B-spline
   (`scipy.ndimage.map_coordinates`, order 3).
3. **Deform symmetrically:** `A'(x) = A(x − ½u(x))` and `B'(x) = B(x + ½u(x))`, using cubic
   B-spline image interpolation. A particle at mid-time position `x_m` then appears at `x_m`
   in both frames when the predictor is exact. Centring in time is second-order accurate;
   warping only B is first order.
4. **Correlate** `A'` and `B'` on the new grid, which yields the residual `Δu`. The new
   estimate is `u_pred(x) + Δu`.

The single-pass mode (`PIVConfig.single_pass(W)`) uses no offset and no deformation; it is
the baseline in the experiments.

## 6. Error definitions (`metrics.py`)

- Per-vector error: `e = u_measured − u(x)` at the vector location.
- Only **interior** vectors are used: those at least 32 px from every image border.
- **RMS** is `√⟨e²⟩` and includes the bias. E1 reports the x-component (`bias = ⟨e_u⟩`,
  `RMS = √⟨e_u²⟩`); E2–E4 use the magnitude `|e| = √(e_u² + e_v²)`.
- **Vortex core:** interior vectors with `r < 2 r_c`.
- **True outlier (E4):** `|e| > 0.5 px`.

## References

- Raffel, Willert, Scarano, Kähler, Wereley & Kompenhans, *Particle Image Velocimetry: A
  Practical Guide*, 3rd ed., Springer, 2018.
- Willert & Gharib (1991), "Digital particle image velocimetry", *Exp. Fluids* 10.
- Scarano (2002), "Iterative image deformation methods in PIV", *Meas. Sci. Technol.* 13.
- Westerweel & Scarano (2005), "Universal outlier detection for PIV data", *Exp. Fluids* 39.
- Westerweel (1997), "Fundamentals of digital particle image velocimetry", *Meas. Sci.
  Technol.* 8.
- Schrijer & Scarano (2008), "Effect of predictor–corrector filtering on the stability and
  spatial resolution of iterative PIV interrogation", *Exp. Fluids* 45.
