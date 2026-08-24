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

### Deliverables (two shared modules in the project root, one notebook per TRM model)

| File | Contains |
|---|---|
| `harmonic_fit.py` | All computation. Writes results to `<trm_dir>/results/`. No matplotlib import. |
| `harmonic_plots.py` | All figures. Reads only `<trm_dir>/results/`. No FITS reading, no fitting. |
| `harmonic_pipeline_<model>.ipynb` | Runs both for one TRM model, displays every figure inline with commentary. |

**Hard rule:** if a function both computes a number and draws something, it is in
the wrong place. The separation exists so that figures can be re-iterated in the
notebook without re-running the fit.

`harmonic_fit.py` and `harmonic_plots.py` are single, shared modules — there is
exactly one of each, in the project root, regardless of how many TRM models exist.
What varies per TRM model is only the *input* (a directory under the project root,
Section 1) and the generated notebook that runs the shared modules against it
(Section 7). Do not fork or copy either module per model; a new TRM model should
never require touching `harmonic_fit.py` or `harmonic_plots.py` at all.

---

## 1. Directory layout

Each **TRM model** (a 3D-Barolo tilted-ring run) lives in its own subdirectory of
the project root — `TRM_paper/` (the run used in the paper) and `fixed_PA42/` (a
PA-fixed-to-42° comparison run) as of this writing, with more addable at any time
(Section 1.1). Every TRM model directory is self-contained: its own `maps/`, its
own ringlog file, and its own `results/`/`figures/` written by a run against it.
`harmonic_fit.py` and `harmonic_plots.py` are shared, single modules at the
project root that operate on whichever TRM model directory they're pointed at —
they do not live inside a TRM model directory and are never copied per model.

```
<project_root>/
  harmonic_fit.py
  harmonic_plots.py
  harmonic_pipeline_<model>.ipynb   # one per TRM model, e.g. harmonic_pipeline_TRM_paper.ipynb
  vrm_pipeline_<model>.ipynb        # Section 9's supplementary cross-check, also one per model
  scripts/
    build_notebook.py               # generates harmonic_pipeline_<model>.ipynb for every TRM model
    build_vrm_notebook.py           # generates vrm_pipeline_<model>.ipynb for every TRM model
  <model_a>/                        # a TRM model directory, e.g. TRM_paper/
    maps/
      *_0mom.fits              # data moment 0   (integrated intensity, unused)
      *_1mom.fits              # data moment 1   (km/s; BUNIT=KM/S, verify not m/s)
      *_2mom.fits              # data moment 2   (km/s; BUNIT=KM/S, verify not m/s)
      *_local_0mom.fits        # Barolo MODEL moment 0 (unused)
      *_local_1mom.fits        # Barolo MODEL moment 1
      *_local_2mom.fits        # Barolo MODEL moment 2
    <ringlog>.txt               # filename varies per model (Section 1.1) -- auto-discovered
    results/
      ring_results.ecsv       # scalar per (ring, side, weighting)
      maps.npz                # 2D arrays: theta, R, weights, residuals
      scans.npz               # chi2 cubes from the PA / VSYS scans
    figures/
      *.pdf
  <model_b>/                        # a second TRM model directory, e.g. fixed_PA42/
    maps/
    <ringlog>.txt
    results/
    figures/
```

Note the moment index is a *suffix*, not `mom1`/`mom2` — actual filenames look like
`SoFiA_<name>_1mom.fits` (data) and `SoFiA_<name>_local_1mom.fits` (model). Glob on
`*_1mom.fits` / `*_2mom.fits`, never on `*mom1.fits` / `*mom2.fits`, which will match
nothing. A `_0mom` pair (moment 0 / integrated intensity) also exists in `maps/` and
is not used by this pipeline — do not glob it in by accident with an unanchored
`*mom*` pattern. Glob for the map files; do not hardcode filenames. Distinguish model
from data by the presence of `_local_` in the basename. Fail loudly with a clear
message if either the data or model pair (`_1mom`/`_2mom`) is missing.

### 1.1 Adding a new TRM model

To add a new TRM model, create a new subdirectory of the project root containing:

- a `maps/` folder with the six FITS files above (glob rules are per-directory, so
  this works exactly like the existing models);
- a ringlog `.txt` file: whitespace-delimited, `#`-commented header, with columns
  including `RAD(Kpc)`, `RAD(arcs)`, `VROT(km/s)`, `E_VROT1`, `E_VROT2`, and the
  rest of Section 4's required columns. The **filename is not fixed** — Barolo
  ringlogs are named differently across runs (`stage_1_opt_parameters.txt`,
  `PA_fixed_to_42.txt` are both seen in this project) — `harmonic_fit.find_ringlog`
  identifies it by its columns, not its name, and deliberately rejects
  `*_initial.txt`-style files (initial guesses only, missing `RAD(Kpc)`/`E_VROT1/2`)
  and any other stray `.txt` sitting alongside it (stats dumps, etc.). If a
  directory has zero or multiple files matching the required columns,
  `find_ringlog` raises rather than guessing — pass `--ringlog` explicitly to
  `harmonic_fit.py` in that case.

That's it — no code changes are required. `harmonic_fit.discover_trm_models(repo_root)`
finds any subdirectory with a `maps/` folder and a resolvable ringlog; run
`python harmonic_fit.py --list-trm-models` to confirm a new model is picked up,
then `python harmonic_fit.py --trm-dir <new_model>` to fit it and
`python scripts/build_notebook.py` (no args rebuilds notebooks for *every*
discovered model, including the new one) to get it a pipeline notebook.

---

## 2. Conventions — read this section before writing any code

Every sign error in the previous script came from this block being implicit.
Make it explicit and assert it at runtime.

### 2.1 Position angle

The ringlog gives `P.A.`, measured north through east, pointing along the
**receding** half of the major axis (Barolo's convention) — e.g. `TRM_paper`'s
`stage_1_opt_parameters.txt` gives `P.A. = 53.445`; other TRM models (Section
1.1) have their own `P.A.` in their own ringlog, by construction in the case of
`fixed_PA42`.

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

### 2.6.1 Resolving the sign, given external near-side information

The code must not resolve this (2.6) — but once a human supplies the near
side from external information (dust lanes, trailing-arm morphology, or a
direct statement), the sign is resolved by a short, checkable derivation,
recorded here rather than in code, so it doesn't have to be re-derived from
scratch (and re-risk a sign error) every time it comes up.

`theta` increases **counter-clockwise on the sky image**. This follows from
`make_geometry`'s construction: `(dx_rot, dy_rot)` is a *proper* rotation of
the sky-pixel frame by `PA_math` (determinant +1, no mirroring), followed by
dividing `dy_rot` by `cos(i) > 0` (a positive stretch, also no mirroring) —
neither step flips handedness, so `theta` inherits its sense directly from
the pixel frame. For this project's confirmed East-left/North-up
orientation (`CDELT1 < 0`), `dx = West`, `dy = North`, and the standard
astrometric identity `E-hat x N-hat = r-hat` (pointing *away* from the
observer) makes `(West, North, toward-observer)` right-handed — so `theta`
increasing is counter-clockwise as displayed. `theta = 0` is receding by
construction (Section 2.3's assertion, an empirical fact, not an
assumption).

Evaluate the model exactly at `theta = 90 deg` (the minor axis): `cos(theta)
= 0`, so pure rotation contributes nothing there, and the entire residual is
the radial term: `v_los - VSYS = sin(i) * s1`. Whichever side of the minor
axis is the *near* side (tilted toward the observer) is, at that location,
made *closer* to the observer by outward motion and *farther* by inward
motion — so blueshift there means outward motion, redshift means inward.

Combined with `sin(i) > 0`, this gives two cases (the `i -> -i` mirror pair
from Section 2.6):

| Near side at | Rotation on sky | True outward `V_rad` |
|---|---|---|
| `theta = +90 deg` | clockwise | `-s1` |
| `theta = -90 deg` (`270 deg`) | counter-clockwise | `+s1` |

The two columns in each row are not independent evidence: given `theta = 0`
is receding (always true) and `theta` increases CCW on the sky (always true
for this project's orientation), *either one* of "near side" or "rotation
sense" determines the other. Agreement between an independently-stated near
side and rotation sense is a consistency check on that determination, not
two separate pieces of evidence for it.

**For this galaxy**, the project owner states the near side is at
`theta = +90 deg` (consistent with the observed clockwise rotation on the
sky, per the table above) — so `true outward V_rad = -s1`. Since the
primary fit (`sin2` weighting, side=`both`) gives `s1` negative in every
ring (approximately -7 to -48 km/s), this resolves to **outward motion —
expansion, not infall** — contingent entirely on that near-side
determination. `near_side_assumed` in the results file stays `"UNRESOLVED"`;
this section documents the interpretation for the paper text, it does not
change what the code writes.

---

## 3. Ring definitions — resolved

`TRM_paper`'s `stage_1_opt_parameters.txt` lists `RAD(arcs) = 43.0, 52.7, 62.4,
72.1`, uniformly spaced by 9.7″. **`RAD` is the ring *center*, confirmed by the
project owner.** Ring *i* spans `RAD - 4.85` → `RAD + 4.85`:
38.15–47.85, 47.85–57.55, 57.55–67.25, 67.25–76.95. This is the `TRM_paper`
example; other TRM models (Section 1.1) have their own `RAD` grid in their own
ringlog — `fixed_PA42`'s, for instance, is `42.0, 51.8, 61.6, 71.4`, spaced by
9.8″ — but the *convention* below (RAD = ring center, width from `np.diff`) is
the same for every model, never per-model configuration.

This convention is not configurable — there is no `rad_convention` toggle and no
`"inner_edge"` reading, for any TRM model. It reproduces the ring bounds of the
previous script exactly for `TRM_paper`, each row carries its own `VROT`/`E_VROT`
(per-ring quantities), and it is BBarolo's documented ringlog convention. Print
the resulting ring bounds at the top of every run, and stamp `"RAD = ring
center"` into every figure caption and the results file header, so the
convention is still visible downstream even though it is no longer a choice.

Derive ring width from `np.diff(RAD)` rather than hardcoding 9.7, and assert
uniform spacing.

**`kpc_per_arcsec` is an adopted physical constant, not derived from the
ringlog.** An earlier version of this section said to derive it from
`RAD(Kpc) / RAD(arcs)` (≈ 0.33888) instead of the old script's hardcoded
0.329 — that guidance is now reversed. Barolo's `RAD(Kpc)` column bakes in
whatever cosmology/distance Barolo itself assumed internally, which is
unknown and need not match the project's adopted cosmology. `harmonic_fit.
KPC_PER_ARCSEC` instead **computes** the scale from Planck18
(`astropy.cosmology.Planck18`) at the source's redshift
(`z = VSYS_FOR_DISTANCE_KMS / c`, the low-z/optical convention, anchored to
`TRM_paper`'s own Barolo VSYS — 4677.0 km/s — so the scale doesn't drift by
run-to-run VSYS fit noise between TRM models): `kpc_per_arcsec =
D_A(z) * (1 arcsec in radians)`, which evaluates to ≈ 0.3288 kpc/arcsec —
matching the ≈ 0.329 the old script hardcoded, but now for a documented,
citable reason rather than as an unexplained literal. `read_ringlog` still
reads `RAD(Kpc)` and checks Barolo's own implied `RAD(Kpc)/RAD(arcs)` for
internal row-to-row consistency (a QA check on Barolo's file, independent
of which absolute scale is adopted) and warns if it diverges from the
adopted value by more than 1% (it does, by ~3%, for both `TRM_paper` and
`fixed_PA42`) — but Barolo's value is never used for any kpc conversion.
`r_center_kpc` (used everywhere downstream, including all of
`timescales.py`) is `RAD(arcs) * KPC_PER_ARCSEC`, not `RAD(Kpc)`.

---

## 4. Values that must come from the ringlog, never hardcoded

`VROT`, `DISP`, `INC`, `P.A.`, `XPOS`, `YPOS`, `VSYS`, `VRAD`, `E_VROT1/2`,
and ring radii (angular: `RAD(arcs)`, `r_in_arcsec`, `r_out_arcsec`).
`kpc_per_arcsec` is the one exception — see Section 3: it is an adopted
Planck18-cosmology constant (`harmonic_fit.KPC_PER_ARCSEC`), not read from
the ringlog.

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
    # project_dir is a TRM model directory (Section 1), not the repo root --
    # maps_dir/ringlog_path/results_dir default to <project_dir>/maps,
    # find_ringlog(project_dir), and <project_dir>/results respectively.
kpc_per_arcsec          : 0.329 (computed, Planck18 -- see Section 3, not read from the ringlog)
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

- `read_ringlog(path, kpc_per_arcsec=KPC_PER_ARCSEC) -> astropy.table.Table` —
  whitespace-delimited, `#`-comment header. Attach ring edges as `RAD ± width/2`
  (`RAD` is the ring center). `r_center_kpc = RAD(arcs) * kpc_per_arcsec`
  (Section 3), not Barolo's own `RAD(Kpc)`.
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
- `results/maps.npz` — per ring: `theta`, `R_arcsec`, `weights` (primary
  scheme), `dv_prefit`, `dv_postfit`, `ring_mask_stack`. Also, once (not
  per-ring, since they don't vary by ring): `data_mom1`, `data_mom2`,
  `model_mom1`, `model_mom2`, `pixscale_arcsec` — `harmonic_plots.py` cannot
  read FITS itself, so `fig_azimuthal_vlos` (raw, non-deprojected `V_los` vs.
  the Barolo model) needs the maps carried through here instead. Likewise
  `ring_results.ecsv` carries `vrot_kms`, `inc_deg`, `pa0_deg`, `vsys_kms`,
  `xpos_pix`, `ypos_pix` per ring (repeated across its side/weighting rows,
  like `r_in_arcsec`) so plots needing the fiducial geometry or the harmonic
  model curve don't need the ringlog either.
- `results/scans.npz` — the chi-squared cubes, grids, `s1(PA)` curves, and (per
  ring, side, weighting) the raw bootstrap `s1` draws as `boot_s1_{side}_{scheme}`
  -- `fig_bootstrap` needs the actual draws to histogram, not just the
  16/50/84 percentiles already in `ring_results.ecsv`.

Use `rms = sqrt(mean(residual**2))`, not `np.std`, which subtracts the residual
mean and is not the RMS about zero.

Print a compact summary table to stdout. Do **not** call
`warnings.filterwarnings("ignore", category=RuntimeWarning)` globally — the old
script did, which would also have hidden genuine all-NaN slices. Scope any
suppression to the specific statement that needs it.

### 5.10 The `c0` toggle

`c0` absorbs both `dVSYS / sin(i)` and `V_z / tan(i)` without separating them
(Section 2.5). There is no separate boolean flag for this — `model_terms`
(Section 5.1) is already the single config knob that controls which terms are
free (`s1` is always required; `c0`, and optionally `c2`/`s2`, are toggled by
including or omitting them from the tuple), so fitting without `c0` means
calling `main()` a second time with `model_terms=("s1",)` into a second
`results_dir` (the notebook uses `cfg.results_dir / "no_c0"`), rather than
reusing the first one — the two result trees must not collide.

`design_matrix` and the scan/profile helpers in `harmonic_fit.py` must handle
an **empty** nuisance-term tuple (i.e. `model_terms=("s1",)`, so the PA/VSYS
scans profile over zero nuisance parameters) without raising — this is what
makes the toggle actually usable, not just representable in `model_terms`.

---

## 6. `harmonic_plots.py`

One function per figure, each taking the loaded results object and returning a
`Figure`. `main()` writes all of them to `figures/`. Keep the existing
`custom_rcparams` block from the old script — it matches the rotation-curve
figures in the paper — and put it in this file only.

Every figure caption/title must carry `"RAD = ring center"` and the primary
weighting scheme.

### 6.1 Convention checks — look at these before trusting anything

1. **`fig_theta_map`** — three panels, sharing one figure. (a) `theta` on a
   cyclic colormap (`twilight`), contours at 0/90/180/270°, major and minor
   axes drawn, an arrow and label at `theta = 0` annotated "receding", ring
   ellipses overlaid. (b) a two-colour panel of `sign(cos theta)` labelled
   approaching/receding. (c) `w(theta)` inside the mask, primary weighting,
   ring ellipses overlaid — makes visible that `sin2` concentrates the
   constraint near the minor axis and that most of the disc contributes
   almost nothing. All three are full-frame maps (every ring merged into one
   image, not one panel per ring — the rings are disjoint annuli on the same
   pixel grid, so merging is exact) with the ring boundaries as ellipse
   overlays. This is the figure that makes the old bug impossible to miss;
   regenerate it every run.
2. **`fig_coverage`** — per ring, a polar histogram of `sum(w)` against `theta`,
   with `L0` and `L1` printed in each panel. Tells you whether the orthogonality
   protection actually holds for this data. (This one stays one panel per ring,
   unlike `fig_theta_map`'s panels — a polar histogram from a single merged ring
   would erase exactly the per-ring `L0`/`L1` comparison the figure exists for.)

### 6.2 Pre-fit and post-fit residual

3. **`fig_residual_maps`** — two panels in one figure, pre-fit and post-fit,
   each a full-frame map (all rings merged, not split ring by ring) with ring
   ellipses overlaid, sharing one diverging colormap centred on zero so the
   two panels are visually comparable.
   - **Pre-fit** (left): `dv = (mom1 - VSYS)/sin(inc) - VROT*cos(theta)`.
     **This is the most important diagnostic in the project.** Genuine
     axisymmetric radial motion appears as a clean dipole aligned with the
     minor axis. A localised blob, a one-sided feature, or something tracking
     tidal structure means a single `s1` per ring is the wrong model — and
     given the group environment, that possibility has to be ruled out before
     the fit is meaningful, not after.
   - **Post-fit** (right): residual after subtracting `s1*sin(theta)`. Should
     be structureless; any coherent pattern remaining is the argument for
     adding `c2, s2`.

3b. **`fig_c0_toggle_residuals`** and **`fig_c0_toggle_s1`** — the `c0`-toggle
    comparison (Section 5.10). Unlike every other figure function, these two
    take **two** `Results` objects (one fit with `c0` free, one with `c0`
    fixed at 0) rather than one, since the comparison is the whole point; they
    are not part of `ALL_FIGURES` / `harmonic_plots.main()` and are only
    called from the notebook, which is the one place both result trees exist
    at once.
    - `fig_c0_toggle_residuals`: post-fit residual maps, `c0` free vs. fixed
      at 0, same layout and shared colormap/scale as `fig_residual_maps`, with
      each ring's fitted `c0` ± its formal error in the title. Under symmetric
      coverage this should look like the same structure shifted by a
      near-uniform per-ring offset, not a change in shape.
    - `fig_c0_toggle_s1`: `s1(R)` and post-fit RMS residual, `c0` free vs.
      fixed at 0, side by side. An `s1` shift larger than the bootstrap width
      between the two is a direct, on-this-data measurement of `L0` leakage
      (Section 5.6) from this ring's masked coverage — not expected under
      symmetric coverage, where `c0` and `s1` are orthogonal.

### 6.3 Data vs model, azimuthal

4. **`fig_azimuthal_vlos`** — one panel per ring, `V_los` in **km/s, not
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

### 6.4 Systematics

5. **`fig_pa_degeneracy`** — per ring, filled `chi2(PA, s1)` contours at
   `delta chi2 = 1, 4, 9`, with the inclination-corrected analytic degeneracy
   line `s1 = VROT * K(scheme) * (PA - PA_0)[rad]` overplotted (see Section
   5.6 for `K`). At `VROT ≈ 300` km/s and this galaxy's inclination the slope
   is ≈ −4.4 km/s per degree (`sin2`), not the flat-sky −5.24. **This doubles
   as the acceptance test for the geometry code**: if the numerical valley
   does not follow the inclination-corrected analytic line, something in the
   deprojection is wrong — the flat-sky line will *not* match even for
   correct code, so do not use it as the check.
6. **`fig_vsys_degeneracy`** — same layout for `(VSYS, s1)`, presented
   deliberately as the contrast case: expected to be axis-aligned.
7. **`fig_s1_vs_pa`** — `s1` against PA offset, one line per ring, on one axis.
   **This is the decisive plot for the paper.** If all four rings cross zero at
   the same PA offset, the signal is consistent with a single PA error and the
   detection is not robust. If they cross at different offsets, no single PA can
   null the signal and the radial motion is real. Mark Barolo's PA uncertainty as
   a shaded band if one is available.
8. **`fig_bootstrap`** — histograms of the bootstrap `s1` per ring, with the
   point estimate and the 16/84 percentiles marked.
9. **`fig_weighting_comparison`** — `s1` per ring under all three schemes, side
   by side. If the schemes disagree by more than the bootstrap width, that
   disagreement *is* a result and belongs in the paper.
10. **`fig_vrad_profile`** — the headline figure. **Signed** `s1(R)` against
    radius, approaching / receding / both, with statistical (bootstrap) and
    systematic (PA-scan half-width) errors drawn as distinct bars. Twin top axis
    in kpc using the ringlog-derived scale. Keep the shaded optical/HI-extent
    band from the old script. No `np.abs`, and no Barolo `VRAD` series.

---

## 7. `harmonic_pipeline_<model>.ipynb`

Generated programmatically with `nbformat` by `scripts/build_notebook.py`, one
notebook per TRM model directory (Section 1): `harmonic_pipeline_TRM_paper.ipynb`,
`harmonic_pipeline_fixed_PA42.ipynb`, and so on. `build_notebook.py` discovers
TRM models via `harmonic_fit.discover_trm_models(repo_root)` and, with no
arguments, (re)builds every discovered model's notebook in one run — a new TRM
model (Section 1.1) gets a notebook automatically, with no template edits.
`--trm-dir <name>` rebuilds just one. Structure (identical for every model,
parameterized only by `TRM_DIR` in the config cell):

1. Markdown: purpose, the ring-center convention (`RAD` is the ring center, not
   the inner edge) and its consequence, the sign-convention caveat, and which
   TRM model this particular notebook targets.
2. Config cell — the one place a user edits anything, normally just `TRM_DIR`
   (e.g. `TRM_DIR = "TRM_paper"`); `maps_dir`, `ringlog_path` (via
   `hf.find_ringlog`), and `results_dir` are all derived from it.
3. Load ringlog and maps; display the ring table and the header-derived beam.
4. Geometry + the Section 2.3 assertion.
5. Run the fit (this one call also produces the PA/VSYS scans and the
   bootstrap — they are bundled per ring since they share masks/weights, see
   Section 5.9; note approximate runtime).
6. Figures 1–2 (`fig_theta_map`, `fig_coverage`).
7. Display `ring_results.ecsv` as a table.
8. Figure 3 (`fig_azimuthal_vlos`).
9. Figure 4 (`fig_residual_maps`), with a markdown cell prompting explicit
   inspection of the pre-fit panel before trusting anything below it.
9b. The `c0` toggle (Section 5.10): rerun `main()` with a second `Config`
    (`dataclasses.replace(cfg, results_dir=cfg.results_dir / "no_c0",
    model_terms=("s1",))`), load both result trees, and display
    `fig_c0_toggle_residuals` / `fig_c0_toggle_s1`.
10. Figures 5–7 (`fig_pa_degeneracy`, `fig_vsys_degeneracy`, `fig_s1_vs_pa`) —
    reads the scan grids already computed in step 5, no extra computation.
11. Figures 8–9 (`fig_bootstrap`, `fig_weighting_comparison`).
12. Figure 10 (`fig_vrad_profile`) and a closing markdown summary of the
    error budget.

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

## 9. Supplementary: VRM/VRMA cross-check

`vrm_pipeline_<model>.ipynb` (generated by `scripts/build_vrm_notebook.py`, one
per TRM model directory, same discovery/CLI pattern as
`scripts/build_notebook.py` in Section 7) is an independent cross-check of
`s1(R)` against the Velocity Ring Model (VRM) and VRM-with-arcs (VRMA) method —
Sylos Labini, Straccamore, De Marzo & Comerón, MNRAS 524, 1560 (2023),
arXiv:2306.12902 — plus a reproduction of the rank-correlation / toy-model warp
diagnostic (their Figure 8) from the companion paper Sylos Labini, De Marzo &
Straccamore, ApJ 988, 122 (2025), doi:10.3847/1538-4357/adc71c,
arXiv:2503.22306. It is **not** part of the deliverables in Section 0 and does
not change any of its rules; it is a separate, additive analysis that reads
its own TRM model's `<trm_dir>/results/ring_results.ecsv` plus its maps, and
otherwise stands alone.

- `vrm/VRM_VRMA.py` is vendored **unmodified** from
  https://github.com/MatteoStraccamore/VRM_VRMA. See `vrm/README.md` for
  attribution, the license caveat (no upstream LICENSE file found), how VRM
  differs methodologically from `harmonic_fit.py` (fits `v_t` and `v_r`
  together per ring, nothing fixed but geometry; rescaled-radius ring bins
  over the full detected extent, not Barolo's edges; no bootstrap or
  degeneracy scan — point estimates only), and a real upstream footgun
  (`phi0input` is accepted but never used internally).
- `vrm/bridge.py` connects this project's geometry and data-quality
  conventions to the vendored class without duplicating `make_geometry`'s
  PA/handedness logic — it derives VRM's expected `(xp, yp)` by inverting
  `make_geometry`'s `(R_pix, theta_rad)` output algebraically, and
  reconstructs the physical radius/angle and point count of each VRM/VRMA
  bin (which the vendored code does not return) from the same pixel set
  actually fed to it. **Note the theta range mismatch if extending this**:
  `make_geometry`'s `theta_rad` (from `arctan2`) ranges `(-pi, pi]`, while
  VRM's internal binning theta ranges `[0, 2*pi)` — `bridge.py` converts via
  `% (2*np.pi)` before reconstructing bin membership; skipping this makes
  any bin containing negative-`theta_rad` pixels look emptier than it is.

On this data, VRM's independently-fit `v_r(R)` (no shared code path with
`harmonic_fit.py` beyond geometry and the data-quality mask) lands within a
few km/s of `s1(R)` at every ring with enough points to trust — read as
independent support for the detection, not a from-scratch validation, since
both methods still share the same input maps and geometry.

`run_figure8_analysis` in `vrm/bridge.py` reproduces the second paper's
Figure 8 (`v_t`, `v_r`, and their rank-correlation map, for the data and for
a toy/null model; the velocity-dispersion map and its correlation with
`v_t`/`v_r`), plus an added LOS-residual panel. Two things that paper leaves
underspecified get a documented, explicit choice rather than a silent guess
— see `vrm/README.md`: the toy model reuses Barolo's own (warp-free,
`VRAD=0`) `_local_1mom.fits` model map directly rather than resynthesizing a
duplicate, and the "correlation map" is read as the per-cell term of the
paper's own correlation formula (Eq. 3), since that is the only quantity in
the paper's equations that both varies spatially and reduces to its stated
scalar (Eq. 2) when summed. Also fixed there: `VRM._matrix()` returns `0.0`,
not `NaN`, for an empty cell, which was silently corrupting the correlation
maps before cells below `min_points` were explicitly excluded.

---

## 10. Explicitly do not

- Do not apply `np.abs` to `s1` anywhere.
- Do not use `curve_fit` or any optimiser.
- Do not hardcode `VSYS`, `INC`, `PA`, `VROT`, or ring radii — read them from
  the ringlog. (`kpc_per_arcsec` is deliberately the exception: an adopted
  Planck18-cosmology constant, Section 3, not read from the ringlog.)
- Do not reintroduce the `mom2 >= 5.0` pixel cut.
- Do not call `warnings.filterwarnings` at module scope.
- Do not plot a Barolo `VRAD` comparison series — it is fixed at zero.
- Do not import matplotlib in `harmonic_fit.py`, or `astropy.io.fits` in
  `harmonic_plots.py`.
- Do not use `np.std` where RMS about zero is meant.
- Do not reintroduce a `rad_convention` toggle — `RAD` is the ring center, fixed.
- Do not describe `sin2` weighting as optimal — it is a robustness trade.
