# CLAUDE.md — HI first-order harmonic decomposition pipeline

## 0. What this project is

Measure the first-order harmonic radial term `s1 = V_rad(R)` from a MeerKAT HI
velocity field, ring by ring, holding the rotation curve fixed at the values from
a 3D-Barolo tilted-ring model. This replaces an earlier single-file script
(`harmonic_fit.py`, superseded) that had a sign-convention bug in its
approaching/receding split and several hardcoded values.

The scientific point of the rewrite is not a better number — it is a defensible
error budget. `V_rad` at the level we are measuring it is degenerate with the
disc position angle, and the pipeline must quantify that degeneracy rather than
ignore it.

### Deliverables (exactly three files, in the project root)

| File | Contains |
|---|---|
| `harmonic_fit.py` | All computation. Writes results to `results/`. No matplotlib import. |
| `harmonic_plots.py` | All figures. Reads only `results/`. No FITS reading, no fitting. |
| `harmonic_pipeline.ipynb` | Runs both, displays every figure inline with commentary. |

**Hard rule:** if a function both computes a number and draws something, it is in
the wrong place. The separation exists so that figures can be re-iterated in the
notebook without re-running the fit.

---

## 1. Directory layout

```
<project_root>/
  maps/
    *_0mom.fits              # data moment 0   (integrated intensity, unused)
    *_1mom.fits              # data moment 1   (km/s; BUNIT=KM/S, verify not m/s)
    *_2mom.fits              # data moment 2   (km/s; BUNIT=KM/S, verify not m/s)
    *_local_0mom.fits        # Barolo MODEL moment 0 (unused)
    *_local_1mom.fits        # Barolo MODEL moment 1
    *_local_2mom.fits        # Barolo MODEL moment 2
  stage_1_opt_parameters.txt
  harmonic_fit.py
  harmonic_plots.py
  harmonic_pipeline.ipynb
  results/
    ring_results.ecsv       # scalar per (ring, side, weighting)
    maps.npz                # 2D arrays: theta, R, weights, residuals
    scans.npz               # chi2 cubes from the PA / VSYS scans
  figures/
    *.pdf
```

Note the moment index is a *suffix*, not `mom1`/`mom2` — actual filenames look like
`SoFiA_<name>_1mom.fits` (data) and `SoFiA_<name>_local_1mom.fits` (model). Glob on
`*_1mom.fits` / `*_2mom.fits`, never on `*mom1.fits` / `*mom2.fits`, which will match
nothing. A `_0mom` pair (moment 0 / integrated intensity) also exists in `maps/` and
is not used by this pipeline — do not glob it in by accident with an unanchored
`*mom*` pattern. Glob for the map files; do not hardcode filenames. Distinguish model
from data by the presence of `_local_` in the basename. Fail loudly with a clear
message if either the data or model pair (`_1mom`/`_2mom`) is missing.

---

## 2. Conventions — read this section before writing any code

Every sign error in the previous script came from this block being implicit.
Make it explicit and assert it at runtime.

### 2.1 Position angle

`stage_1_opt_parameters.txt` gives `P.A. = 53.445`, measured north through east,
pointing along the **receding** half of the major axis (Barolo's convention).

Array coordinates use `arctan2`, which measures counter-clockwise from `+x`.
For a standard image (north up, east left, i.e. `CDELT1 < 0`, displayed
`origin='lower'`):

```
PA_math = PA_barolo + 90.0
```

This is only valid for east-left. **Read `CDELT1` from the header and assert its
sign.** If `CDELT1 > 0`, the conversion is `PA_math = 90.0 - PA_barolo` and the
handedness of `theta` flips. Do not silently support both — raise unless the
orientation is confirmed.

### 2.2 Deprojection

With `dx = x - XPOS`, `dy = y - YPOS` (pixels, `np.indices` order is `(y, x)`):

```
dx_rot   =  dx*cos(PA_math) + dy*sin(PA_math)     # along major axis
dy_rot   = -dx*sin(PA_math) + dy*cos(PA_math)     # along minor axis
dy_deproj = dy_rot / cos(inc)
R      = hypot(dx_rot, dy_deproj)                 # pixels
theta  = arctan2(dy_deproj, dx_rot)               # radians, from receding major axis
```

So `theta = 0` on the receding major axis, `theta = 180°` on the approaching
side, and `theta = ±90°` on the minor axis where `V_rad` has maximum leverage.

### 2.3 Runtime assertion (do not skip)

Immediately after building the first geometry, verify the receding side:

```
Within the outermost ring mask, the weighted mean of
(model_mom1 - VSYS) over pixels with cos(theta) > 0 must be POSITIVE,
and negative over pixels with cos(theta) < 0.
```

Use the Barolo **model** mom1, not the data, so the test is not confused by
real non-circular motion. Raise `RuntimeError` with both mean values printed if
it fails. This single check permanently closes the `+90` / east-left question,
and would have caught the bug in the old script immediately.

### 2.4 Approaching / receding split

Derive it from the same `theta` used in the fit — never from a separately
hardcoded sky angle. That decoupling is what produced the original bug.

```
mask_receding   = cos(theta) > 0
mask_approaching = ~mask_receding
```

### 2.5 The harmonic model

```
dv(R, theta) = (V_los - VSYS) / sin(inc)
             = c0 + c1*cos(theta) + s1*sin(theta) + c2*cos(2 theta) + s2*sin(2 theta)
```

- `c1` is **fixed** to `VROT` from the ringlog. Implement as a subtraction from
  the data vector, not as a design-matrix column.
- `s1` is the quantity of interest, `V_rad`.
- `c0` is a per-ring offset. It is **not** simply a `VSYS` error: it absorbs both
  `dVSYS / sin(inc)` and any vertical motion `V_z / tan(inc)`. Fit it as a free
  nuisance parameter and report it; a radially *varying* `c0` cannot be a `VSYS`
  error and would be scientifically interesting.
- `c2, s2` are optional (config flag), used to test whether the residual has
  structure a first-order model cannot capture.

### 2.6 Sign of V_rad — do not resolve this in code

`s1 > 0` corresponds to outward motion **only if the near side of the disc is
known**. Inclination alone is degenerate under `i -> -i`. The code must:

- keep the sign of `s1` throughout (**never** apply `np.abs`, which the old
  script did and which discards the physics);
- write a field `near_side_assumed = "UNRESOLVED"` into the results file;
- have every plot axis labelled `s1` with a footnote that the inflow/outflow
  mapping requires external information (dust lanes, trailing-arm assumption).

---

## 3. Ring definitions — resolved

`stage_1_opt_parameters.txt` lists `RAD(arcs) = 43.0, 52.7, 62.4, 72.1`,
uniformly spaced by 9.7″. **`RAD` is the ring *center*, confirmed by the project
owner.** Ring *i* spans `RAD - 4.85` → `RAD + 4.85`:
38.15–47.85, 47.85–57.55, 57.55–67.25, 67.25–76.95.

This is not configurable — there is no `rad_convention` toggle and no
`"inner_edge"` reading. It reproduces the ring bounds of the previous script
exactly, each row carries its own `VROT`/`E_VROT` (per-ring quantities), and it
is BBarolo's documented ringlog convention. Print the resulting ring bounds at
the top of every run, and stamp `"RAD = ring center"` into every figure caption
and the results file header, so the convention is still visible downstream even
though it is no longer a choice.

Derive ring width from `np.diff(RAD)` rather than hardcoding 9.7, and assert
uniform spacing.

Also derive `kpc_per_arcsec` from `RAD(Kpc) / RAD(arcs)` (≈ 0.33888) instead of
the stale 0.329 in the old script. Assert consistency across rows.

---

## 4. Values that must come from the ringlog, never hardcoded

`VROT`, `DISP`, `INC`, `P.A.`, `XPOS`, `YPOS`, `VSYS`, `VRAD`, `E_VROT1/2`,
ring radii, and `kpc_per_arcsec`.

The old script's hardcoded values disagree with this file (`VSYS` 4676 vs 4677,
`INC` 49 vs 49.345, `PA` 53 vs 53.445, and all four `VROT` values differ), which
means they came from a different Barolo run. Do not carry any of them forward.

### Known properties of this particular ringlog

- `VRAD = 0.000` in every ring — Barolo was run with the radial term fixed off,
  deliberately, because of the degeneracy this pipeline exists to characterise.
  **There is no stage-2 run and no external `V_rad` to compare against.** Do not
  add a "TRM fit" series to any plot. The old script had a commented-out block
  with values of exactly 60.000 from an unrelated run; discard it.
- Consequence: `VROT` was fitted under the assumption `V_rad = 0`, so holding it
  fixed here is formally circular. Section 6.4 measures how much that matters
  instead of assuming it away.
- `PA` and `INC` are identical across all four rings — this model has no warp,
  so the per-ring geometry machinery will be a no-op on this file. Build it
  anyway; it is required for the PA scan regardless.
- `DISP` = 1.000 and 0.263 km/s in rings 1 and 4 have almost certainly hit
  Barolo's lower search bound. Do not use model `DISP` for weighting.

---

## 5. `harmonic_fit.py`

### 5.1 Config

A single `@dataclass` at the top, instantiated in `main()`. Fields:

```
project_dir, maps_dir, ringlog_path, results_dir
weighting_schemes       : ("uniform", "sin2", "invvar")
primary_weighting       : "sin2"
model_terms             : ("c0", "s1")                 # c1 always fixed
sigma_floor_kms         : 5.0                          # weight clipping only
sigma_artifact_floor_kms: 0.5                          # data-quality mask, see 5.3
pa_scan_halfwidth_deg   : 15.0
pa_scan_step_deg        : 0.25
vsys_scan_halfwidth_kms : 20.0
vsys_scan_step_kms      : 0.5
s1_grid_halfwidth_kms   : 80.0
s1_grid_step_kms        : 0.5
n_bootstrap             : 2000
random_seed             : 42
```

No science constant may appear anywhere else in either file.

### 5.2 I/O

- `read_ringlog(path) -> astropy.table.Table` — whitespace-delimited, `#`-comment
  header. Attach ring edges as `RAD ± width/2` (`RAD` is the ring center).
- `load_maps(maps_dir) -> MapSet` — data mom1/mom2 and model mom1/mom2, squeezed
  to 2D. **Read `BUNIT` from each header and assert it is `KM/S` or `M/S` — raise
  on anything else.** For this dataset all four velocity maps (`_1mom`, `_2mom`,
  `_local_1mom`, `_local_2mom`) are already `KM/S` (confirmed: `_1mom` values run
  4436–4987, matching `VSYS≈4677`), so no conversion is applied. If `BUNIT` says
  `M/S`, divide by 1000 at this boundary and nowhere else. **Do not silently
  divide by 1000 the way the old script did** — that assumed m/s unconditionally
  and would corrupt this dataset by a factor of 1000. Carry the WCS, `CDELT1/2`,
  and beam (`BMAJ`, `BMIN`, `BPA`) — present as header keywords on every map
  here. If beam keywords are ever absent, look for a `HISTORY`/`AIPS CLEAN BMAJ`
  card; if still absent, raise — the bootstrap depends on it.

### 5.3 Geometry

```
make_geometry(shape, xc, yc, pa_barolo_deg, inc_deg, cdelt1_sign) -> (R_pix, theta_rad)
```

Called once per ring (parameters may vary per ring even though they don't here)
and once per grid point in the scans. Pure function, no globals, no I/O.

`ring_mask(R_pix, r_in, r_out, mom1, mom2)` — finite `mom1`, finite `mom2`,
inside the annulus.

**Do not reproduce the old `v_err_map >= 5.0` pixel cut.** That removed real
low-dispersion pixels from the fit. The dispersion floor belongs only in the
`invvar` weighting, applied as `sigma = np.maximum(mom2, sigma_floor_kms)`.

**Separately, apply a data-quality mask for near-zero data `mom2`.** On this
dataset, `mom2` is empirically bimodal: values are either `< 4e-4 km/s` or
`>= 2.75 km/s` — a four-order-of-magnitude gap, nothing in between (~20% of
finite pixels fall in the near-zero cluster, and they correlate almost
exactly with low-`mom0` / low-S/N pixels near the edge of the detection).
This is a numerical collapse of SoFiA's linewidth estimate at marginal S/N,
not real narrow-but-detected HI linewidth, and it is a *different* failure
mode from the prohibited `>= 5.0` cut above: that one discarded genuine
low-but-nonzero dispersion measurements; this one discards pixels where the
measurement itself is pathological. Section 5.8 requires **raw, unfloored**
`mom2` as the chi2/covariance sigma so `delta chi2` stays meaningful — with
these pixels included unmasked, a handful of near-`1e-5`-km/s sigmas inflate
chi2 by ~10 orders of magnitude and make every chi2-based diagnostic (PA/VSYS
contour figures, the `chi2_min/n_eff` calibration) meaningless.

Add `sigma_artifact_floor_kms` to the config (Section 5.1) and exclude
`mom2 < sigma_artifact_floor_kms` from `ring_mask` for **data** mom2 only
(never applied to the model). A value anywhere in the empirical gap works;
use `0.5 km/s` — comfortably inside the gap, nowhere near the `5.0 km/s`
threshold of the prohibited cut, so there is no ambiguity between the two.
Report the number of pixels this removes per ring in the results table
metadata so the cut stays visible.

### 5.4 Weighting

```
weights(theta, sigma, scheme) -> w
  "uniform" : ones
  "sin2"    : sin(theta)**2
  "invvar"  : 1 / sigma**2
```

Normalise each to `sum(w) = N` so chi-squared values remain comparable.

`sin2` is the primary scheme. Its justification for the paper is **robustness,
not optimality**: it is formally less precise than uniform weighting under
homoscedastic noise, but it downweights the major axis, where residuals are
dominated by errors in the fixed `VROT`, by beam smearing across the steep
velocity gradient, and by any warp. State this in a module docstring so it is
not later mis-described as optimal weighting.

### 5.5 The fit

Build it as a weighted linear least-squares problem. **Do not use
`scipy.optimize.curve_fit`.** The model is linear in every free parameter; the
optimiser converges in one step to the analytic solution, ~100× slower and with
the estimator hidden inside a keyword argument.

```
d = (mom1 - VSYS)/sin(inc) - VROT*cos(theta)      # fixed c1 removed
A = columns for the free terms in model_terms      # [1, sin(theta), ...]
W = diag(w)
M = A.T @ W @ A
p = solve(M, A.T @ W @ d)
```

**Covariance must use the sandwich form.** Because `w != 1/sigma**2` in the
`uniform` and `sin2` schemes, the usual `inv(M)` is wrong:

```
Cov = inv(M) @ (A.T @ W @ diag(sigma**2) @ W @ A) @ inv(M)
```

Return `p`, `Cov`, the parameter correlation matrix, and the residual vector.
Label these formal errors clearly as **reference only** — they assume
independent pixels, which is false at 4″ pixels under a MeerKAT beam. The
bootstrap is the quoted statistical error.

### 5.6 Leakage diagnostics — compute these per ring

For the estimator `s1_hat = sum(w*d*sin) / sum(w*sin^2)`, the bias from a
constant offset `eps` and from a rotation-curve error `dVROT` are:

```
L0 = sum(w*sin(theta))            / sum(w*sin(theta)**2)    # multiplies eps
L1 = sum(w*sin(theta)*cos(theta)) / sum(w*sin(theta)**2)    # multiplies dVROT
```

Analytically both vanish over a full ring with uniform azimuthal coverage, for
all three weighting schemes — and also over a half-ring split at the **minor**
axis, since `sin^3(theta)` is odd about `theta = 0`. So `VSYS` and `VROT` errors
do **not** bias `s1` when coverage is symmetric. Blanked pixels break that
symmetry, and `L0`, `L1` measure by how much. Write both to the results table
per ring and per side; they are the quantitative justification for holding `VSYS`
and `VROT` fixed.

Note the asymmetry that motivates the whole error budget: a PA error contributes
a term in `sin(theta)`, which is *the same harmonic as the signal*. No
weighting scheme can suppress it. PA is the only first-order leak.

**The naive coefficient `-VROT * dPA` is a flat-sky approximation and is not
accurate at this galaxy's inclination.** A PA error does not shift `theta` by
a uniform `dPA` once you deproject through `cos(inc)` — differentiating
`make_geometry` gives, to first order in a small PA offset `dPA` (radians):

```
dtheta/d(dPA) = -(cos(theta)^2/cos(inc) + cos(inc)*sin(theta)^2)   ≡ g(theta)
```

which ranges from `-1/cos(inc)` on the major axis to `-cos(inc)` on the minor
axis — not a constant `-1`. Propagating this through the weighted `s1`
estimator (projecting the induced residual `VROT*dPA*g(theta)*sin(theta)` onto
the `sin(theta)` basis under weight `w(theta)`) gives the leakage slope:

```
K(scheme) = sum(w * g(theta) * sin(theta)**2) / sum(w * sin(theta)**2)
s1_leak   = VROT * dPA[rad] * K(scheme)
```

evaluated over the same fixed pixel set used by the PA scan. Closed forms for
constant-`sigma` weighting (uniform and, when `sigma` is ~constant over the
ring, invvar too), with `ci = cos(inc)`:

```
K_uniform = -(1/4) * (1/ci + 3*ci)
K_sin2    = -(1/6) * (1/ci + 5*ci)
```

At this galaxy's inclination (`INC = 49.345 deg`, `ci = 0.6515`):
`K_uniform ≈ -0.872`, `K_sin2 ≈ -0.799` — i.e. the true slope is **80–87% of
the naive `-VROT` value**, about `-4.2` to `-4.6 km/s/deg` at `VROT ≈ 300`
km/s, not `-5.24 km/s/deg`. This was verified two independent ways: numerical
finite-differencing of `make_geometry`, and the full weighted-LS fit recovering
`K*VROT*dPA` to within 1%. **Use `K(scheme)`, not the flat-sky `-VROT`, for
both the PA-degeneracy self-test (Section 8, test 2) and the analytic overlay
line in `fig_pa_degeneracy` (Section 6.4).** For `invvar` weighting with
spatially-varying `sigma` (i.e. on real, not synthetic, data), compute `K`
numerically from the actual per-pixel `theta`, `w`, and `sigma` of the fixed
scan mask rather than using the closed form, since `w` is then not a pure
function of `theta` alone.

### 5.7 Bootstrap

Block bootstrap over beam-sized cells, because adjacent pixels are not
independent.

```
pixels_per_beam = 1.1331 * BMAJ * BMIN / |CDELT1 * CDELT2|
cell_side_pix   = ceil(sqrt(pixels_per_beam))
cell_id         = (y // cell_side_pix) * n_cells_x + (x // cell_side_pix)
```

Per draw: resample cell IDs with replacement, concatenate their pixels, refit.
Report the 16th / 50th / 84th percentiles of `s1`. Also record
`n_cells` and `n_eff = n_pixels / pixels_per_beam` per ring — with 4″ pixels
this may be a small number, and it should be visible in the paper.

### 5.8 Scans

**PA scan.** For each ring, for each PA on the grid: rebuild `theta` with that
PA, refit `s1`, record `s1(PA)` and `chi2(PA)`. Also evaluate `chi2` on a 2D
grid of `(PA, s1)` for the contour figure.

Keep the **pixel set fixed** at the fiducial ring mask throughout the scan.
Recomputing the mask per PA changes `R` and therefore the degrees of freedom,
making chi-squared discontinuous across the grid. The scan is isolating the
angular effect, so freeze the sample.

**VSYS scan.** Same structure, gridding `VSYS`. Geometry is unaffected — only
the data vector shifts — so this is cheap. Expect the `(VSYS, s1)` contours to
come out **untilted**; that is the predicted result from Section 5.6, not a
failed figure. Any tilt is a direct measurement of coverage asymmetry.

**Chi-squared definition for contours.** Always use `sum((residual/sigma)**2)`
with `sigma` from mom2, independent of the fitting weights, so `delta chi2`
levels mean something. Then rescale so `chi2_min / n_eff = 1` per ring, and say
plainly in the docstring that this is a calibration to the effective sample
size, not a claim that the noise model is correct.

### 5.9 `main()`

Loop over rings × sides {both, approaching, receding} × weighting schemes.
Write:

- `results/ring_results.ecsv` — one row per combination: `ring_index`,
  `r_in_arcsec`, `r_out_arcsec`, `r_center_kpc`, `side`, `weighting`, `s1`,
  `s1_formal_err`, `s1_boot_lo/med/hi`, `c0`, `c0_err`, `c2`, `s2` (if fitted),
  `chi2`, `n_pix`, `n_eff`, `L0`, `L1`, `rms_residual`, `n_removed_quality_mask`
  (pixels dropped by the `sigma_artifact_floor_kms` cut, Section 5.3), plus the
  config and `"RAD = ring center"` in the table metadata.
- `results/maps.npz` — `theta`, `R_arcsec`, `weights` (primary scheme),
  `dv_prefit`, `dv_postfit`, `ring_mask_stack`.
- `results/scans.npz` — the chi-squared cubes, grids, and `s1(PA)` curves.

Use `rms = sqrt(mean(residual**2))`, not `np.std`, which subtracts the residual
mean and is not the RMS about zero.

Print a compact summary table to stdout. Do **not** call
`warnings.filterwarnings("ignore", category=RuntimeWarning)` globally — the old
script did, which would also have hidden genuine all-NaN slices. Scope any
suppression to the specific statement that needs it.

---

## 6. `harmonic_plots.py`

One function per figure, each taking the loaded results object and returning a
`Figure`. `main()` writes all of them to `figures/`. Keep the existing
`custom_rcparams` block from the old script — it matches the rotation-curve
figures in the paper — and put it in this file only.

Every figure caption/title must carry `"RAD = ring center"` and the primary
weighting scheme.

### 6.1 Convention checks — look at these before trusting anything

1. **`fig_theta_map`** — `theta` on a cyclic colormap (`twilight`), contours at
   0/90/180/270°, major and minor axes drawn, an arrow and label at `theta = 0`
   annotated "receding", ring ellipses overlaid. Beside it, a two-colour panel of
   `sign(cos theta)` labelled approaching/receding. This is the figure that makes
   the old bug impossible to miss; regenerate it every run.
2. **`fig_coverage`** — per ring, a polar histogram of `sum(w)` against `theta`,
   with `L0` and `L1` printed in each panel. Tells you whether the orthogonality
   protection actually holds for this data.
3. **`fig_weight_map`** — `w(theta)` inside the mask. Makes visible that `sin2`
   concentrates the constraint near the minor axis and that most of the disc
   contributes almost nothing.

### 6.2 Pre-fit

4. **`fig_prefit_residual`** — 2D map of
   `dv = (mom1 - VSYS)/sin(inc) - VROT*cos(theta)`, diverging colormap centred
   on zero, rings overlaid. **This is the most important diagnostic in the
   project.** Genuine axisymmetric radial motion appears as a clean dipole
   aligned with the minor axis. A localised blob, a one-sided feature, or
   something tracking tidal structure means a single `s1` per ring is the wrong
   model — and given the group environment, that possibility has to be
   ruled out before the fit is meaningful, not after.

### 6.3 Data vs model, azimuthal

5. **`fig_azimuthal_vlos`** — one panel per ring, `V_los` in **km/s, not
   deprojected**, against `theta`:
   - binned data points from the data mom1 (wedges of `360/n_wedge` degrees,
     `n_wedge = 24`), error bars = weighted std / sqrt(n_beams in wedge);
   - the Barolo model mom1 sampled through the *same* wedges and the *same*
     mask — using the model moment map rather than an analytic curve is the
     right comparison because it carries the same beam smearing as the data;
   - the harmonic model `VSYS + sin(inc)*(VROT*cos + s1*sin)` overlaid.

   Where data and Barolo diverge but data and harmonic agree, that gap *is* the
   `s1` detection, shown directly. Bin the points with the **same weights as the
   fit**, so the curve actually tracks the markers — in the old script the points
   were unweighted means while the fit was inverse-variance weighted, so
   disagreement was expected and uninformative.

   Keep the old script's `theta` axis remapped to 90°–450° so the approaching
   region is contiguous, with the shaded bands and labels.

### 6.4 Post-fit and systematics

6. **`fig_postfit_residual`** — residual map after subtracting `s1*sin(theta)`.
   Should be structureless; any coherent pattern is the argument for adding
   `c2, s2`.
7. **`fig_pa_degeneracy`** — per ring, filled `chi2(PA, s1)` contours at
   `delta chi2 = 1, 4, 9`, with the inclination-corrected analytic degeneracy
   line `s1 = VROT * K(scheme) * (PA - PA_0)[rad]` overplotted (see Section
   5.6 for `K`). At `VROT ≈ 300` km/s and this galaxy's inclination the slope
   is ≈ −4.4 km/s per degree (`sin2`), not the flat-sky −5.24. **This doubles
   as the acceptance test for the geometry code**: if the numerical valley
   does not follow the inclination-corrected analytic line, something in the
   deprojection is wrong — the flat-sky line will *not* match even for
   correct code, so do not use it as the check.
8. **`fig_vsys_degeneracy`** — same layout for `(VSYS, s1)`, presented
   deliberately as the contrast case: expected to be axis-aligned.
9. **`fig_s1_vs_pa`** — `s1` against PA offset, one line per ring, on one axis.
   **This is the decisive plot for the paper.** If all four rings cross zero at
   the same PA offset, the signal is consistent with a single PA error and the
   detection is not robust. If they cross at different offsets, no single PA can
   null the signal and the radial motion is real. Mark Barolo's PA uncertainty as
   a shaded band if one is available.
10. **`fig_bootstrap`** — histograms of the bootstrap `s1` per ring, with the
    point estimate and the 16/84 percentiles marked.
11. **`fig_weighting_comparison`** — `s1` per ring under all three schemes, side
    by side. If the schemes disagree by more than the bootstrap width, that
    disagreement *is* a result and belongs in the paper.
12. **`fig_vrad_profile`** — the headline figure. **Signed** `s1(R)` against
    radius, approaching / receding / both, with statistical (bootstrap) and
    systematic (PA-scan half-width) errors drawn as distinct bars. Twin top axis
    in kpc using the ringlog-derived scale. Keep the shaded optical/HI-extent
    band from the old script. No `np.abs`, and no Barolo `VRAD` series.

---

## 7. `harmonic_pipeline.ipynb`

Generate it programmatically with `nbformat`. Structure:

1. Markdown: purpose, the ring-center convention (`RAD` is the ring center, not
   the inner edge) and its consequence, the sign-convention caveat.
2. Config cell — the one place a user edits anything.
3. Load ringlog and maps; display the ring table and the header-derived beam.
4. Geometry + the Section 2.3 assertion, then figures 1–3.
5. Figure 4 (pre-fit residual), with a markdown cell prompting explicit
   inspection before continuing.
6. Run the fit; display `ring_results.ecsv` as a table.
7. Figures 5–6.
8. Scans (slowest cells; note approximate runtime), then figures 7–9.
9. Bootstrap, then figures 10–11.
10. Figure 12 and a closing markdown summary of the error budget.

Every figure appears inline. The notebook should run top to bottom on a clean
checkout with no manual edits beyond the config cell.

---

## 8. Acceptance tests

Put these in a `if __name__ == "__main__"` self-test block in `harmonic_fit.py`,
runnable as `python harmonic_fit.py --selftest`. They must pass before any
result from real data is trusted.

1. **Synthetic recovery.** Build a mock `V_los` map from known `VROT(R)`,
   `V_rad = +25` km/s, `VSYS`, `PA`, `INC` on the real pixel grid. Run the
   pipeline. Recover `s1 = 25 ± 0.5` km/s in every ring, all three weighting
   schemes. This is the single most valuable test in the project.
2. **PA degeneracy.** Inject `V_rad = 0`, then fit with PA offset by `+2°`.
   Recover `s1 ≈ VROT * K(scheme) * 2° in radians` (the inclination-corrected
   slope from Section 5.6, ≈ −8.4 to −9.2 km/s at `VROT = 300`, `INC = 49.345°`
   — *not* the flat-sky ≈ −10.5) to within 5%. Confirms both the degeneracy and
   the analytic slope in figure 7.
3. **Receding-side assertion** fires correctly: flip `PA_math` by 180° in a mock
   and confirm the assertion raises.
4. **Leakage.** On an unmasked synthetic ring, `|L0| < 1e-10` and `|L1| < 1e-10`
   for all three schemes; on a ring with an azimuthal wedge blanked, both become
   measurably nonzero.
5. **Analytic agreement.** For a single free parameter and `invvar` weighting,
   the linear solve reproduces the old `curve_fit` answer to machine precision —
   confirms the rewrite changed the method, not the arithmetic.
6. **Round trip.** `V_z` injected as a constant offset appears in `c0`, not in
   `s1`.

---

## 9. Explicitly do not

- Do not apply `np.abs` to `s1` anywhere.
- Do not use `curve_fit` or any optimiser.
- Do not hardcode `VSYS`, `INC`, `PA`, `VROT`, ring radii, or `kpc_per_arcsec`.
- Do not reintroduce the `mom2 >= 5.0` pixel cut.
- Do not call `warnings.filterwarnings` at module scope.
- Do not plot a Barolo `VRAD` comparison series — it is fixed at zero.
- Do not import matplotlib in `harmonic_fit.py`, or `astropy.io.fits` in
  `harmonic_plots.py`.
- Do not use `np.std` where RMS about zero is meant.
- Do not reintroduce a `rad_convention` toggle — `RAD` is the ring center, fixed.
- Do not describe `sin2` weighting as optimal — it is a robustness trade.
