"""Generates harmonic_pipeline.ipynb programmatically. Run this to rebuild
the notebook from scratch; do not hand-edit the .ipynb directly except in
the config cell while actually using it."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# HI first-order harmonic decomposition

Measures the first-order harmonic radial term `s1 = V_rad(R)` from a MeerKAT
HI velocity field, ring by ring, holding the rotation curve fixed at the
3D-Barolo tilted-ring values. See `CLAUDE.md` in the project root for the
full specification this notebook and its two modules (`harmonic_fit.py`,
`harmonic_plots.py`) implement.

**Ring-radius convention.** `stage_1_opt_parameters.txt` lists `RAD(arcs)`
uniformly spaced by 9.7". `RAD` is the ring *center* (confirmed by the
project owner), so ring *i* spans `RAD - 4.85" -> RAD + 4.85"`. This
reproduces the ring bounds of the earlier single-file script exactly. There
is no `rad_convention` toggle -- this is fixed, not configurable.

**Sign-convention caveat.** `s1 > 0` means outward motion *only if the near
side of the disc is known*; inclination alone is degenerate under `i -> -i`.
Every table and figure below carries `near_side_assumed = "UNRESOLVED"`.
Treat the sign of `s1` as a number to be interpreted with external
information (dust lanes, trailing-arm assumption), not a ready-made
inflow/outflow claim.

**On the structure below.** `harmonic_fit.process_ring` bundles the fit, the
PA/VSYS scans, and the bootstrap together per ring, because they share the
same per-ring masks and weights -- splitting them into separate notebook
cells would mean recomputing that geometry three times for no benefit. So
there is one cell below ("Run the pipeline") that does all of the
computation and writes `results/`; every figure after that point is read
back from disk via `harmonic_plots.py`, which never re-fits anything.
""")

md(r"""
## Config

The one cell in this notebook you are expected to edit.
""")

code(r"""
%matplotlib inline
from pathlib import Path
import harmonic_fit as hf
import harmonic_plots as hp

project_dir = Path.cwd()

cfg = hf.Config(
    project_dir=project_dir,
    maps_dir=project_dir / "maps",
    ringlog_path=project_dir / "stage_1_opt_parameters.txt",
    results_dir=project_dir / "results",
)
cfg
""")

md(r"""
## Ring log and maps

Everything physical (`VROT`, `INC`, `P.A.`, `XPOS`, `YPOS`, `VSYS`, ring
radii, `kpc_per_arcsec`) comes from `stage_1_opt_parameters.txt` -- nothing
here is hardcoded. The moment maps are matched by globbing `maps/` for
`*_1mom.fits` / `*_2mom.fits` (data) and `*_local_1mom.fits` /
`*_local_2mom.fits` (Barolo model).
""")

code(r"""
ringlog = hf.read_ringlog(cfg.ringlog_path)
mapset = hf.load_maps(cfg.maps_dir)

print(f"kpc_per_arcsec = {ringlog.meta['kpc_per_arcsec']:.5f}")
print(f"ring width = {ringlog.meta['ring_width_arcsec']:.2f} arcsec  (rad_convention: {ringlog.meta['rad_convention']})")
print(f"beam: BMAJ={mapset.bmaj_deg*3600:.2f}\" BMIN={mapset.bmin_deg*3600:.2f}\" BPA={mapset.bpa_deg:.1f} deg")
print(f"pixel scale: {mapset.pixscale_arcsec:.3f} arcsec/pix, map shape {mapset.shape}")

ringlog["RAD(arcs)", "VROT(km/s)", "INC(deg)", "P.A.(deg)", "VSYS(km/s)", "r_in_arcsec", "r_out_arcsec"]
""")

md(r"""
## Geometry and the Section 2.3 receding-side assertion

`PA_math = PA_barolo + 90 deg` is only valid for an east-left image
(`CDELT1 < 0`, `origin='lower'`); this is asserted from the header, not
assumed. Before trusting anything downstream, we check the one thing that
would have caught the old script's sign bug immediately: within the
outermost ring, the *Barolo model* mom1 (not the data, so real non-circular
motion can't confuse the test) must be redshifted of `VSYS` on the
`cos(theta) > 0` (receding) side and blueshifted on the other.
""")

code(r"""
cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)
print(f"CDELT1 = {mapset.cdelt1_deg} deg/pix -> sign={cdelt1_sign} ({'east-left' if cdelt1_sign < 0 else 'UNCONFIRMED orientation'})")

outer_row = ringlog[-1]
xc0, yc0 = float(outer_row["XPOS(pix)"]), float(outer_row["YPOS(pix)"])
pa0, inc0, vsys0 = float(outer_row["P.A.(deg)"]), float(outer_row["INC(deg)"]), float(outer_row["VSYS(km/s)"])

R_pix0, theta0 = hf.make_geometry(mapset.shape, xc0, yc0, pa0, inc0, cdelt1_sign)
outer_mask0 = hf.ring_mask(R_pix0 * mapset.pixscale_arcsec, 0.0, float(outer_row["r_out_arcsec"]),
                            mapset.model_mom1, mapset.model_mom1)
mean_rec, mean_app = hf.assert_receding_side(theta0, mapset.model_mom1, vsys0, outer_mask0)
print(f"PASSED: mean(model-VSYS) receding={mean_rec:.2f} km/s, approaching={mean_app:.2f} km/s")
""")

md(r"""
## Run the pipeline

This single call performs, per ring: the weighted-LS fit (all three
weighting schemes, all three sides), the leakage diagnostics, the block
bootstrap over beam-sized cells, and the PA/VSYS scans with their chi2
contour grids -- then writes `results/ring_results.ecsv`, `results/maps.npz`,
and `results/scans.npz`. On this dataset (4 rings, ~50-100 pixels/ring) it
takes a few seconds; it would dominate the runtime on a larger map or a much
finer PA/VSYS grid.
""")

code(r"""
results_table = hf.main(cfg)
""")

md(r"""
## Figures 1-3: convention checks

Look at these before trusting anything else. `fig_theta_map` is the figure
that makes the old sign bug impossible to miss -- regenerate it every run.
`fig_coverage` shows whether the `L0`/`L1` orthogonality protection actually
holds for this data's (patchy) azimuthal coverage. `fig_weight_map` makes it
visible that `sin2` weighting concentrates the constraint near the minor
axis.
""")

code(r"""
res = hp.load_results(cfg.results_dir)

hp.fig_theta_map(res)
hp.fig_coverage(res)
hp.fig_weight_map(res)
None
""")

md(r"""
## Figure 4: pre-fit residual -- STOP and look at this before continuing

**This is the most important diagnostic in the project.** Genuine
axisymmetric radial motion appears as a clean dipole aligned with the minor
axis. A localised blob, a one-sided feature, or something tracking tidal
structure means a single `s1` per ring is the wrong model -- and given the
group environment this target sits in, that possibility has to be ruled out
before anything below is taken at face value.
""")

code(r"""
hp.fig_prefit_residual(res)
None
""")

md(r"""
## `ring_results.ecsv`
""")

code(r"""
res.table
""")

md(r"""
## Figures 5-6: azimuthal comparison and post-fit residual

`fig_azimuthal_vlos` compares the data, the Barolo model (through the same
wedges and mask, so it carries the same beam smearing), and the harmonic
fit. Where data and Barolo diverge but data and the harmonic fit agree, that
gap *is* the `s1` detection, shown directly. `fig_postfit_residual` should
be structureless -- any coherent pattern is the argument for enabling
`c2, s2`.
""")

code(r"""
hp.fig_azimuthal_vlos(res)
hp.fig_postfit_residual(res)
None
""")

md(r"""
## Figures 7-9: PA/VSYS degeneracy and s1 vs. PA offset

These read the scan grids already computed in the pipeline cell above (no
extra computation here). `fig_pa_degeneracy` doubles as an acceptance test
for the geometry code: the numerical chi2 valley should track the
inclination-corrected analytic line closely. `fig_vsys_degeneracy` is the
deliberate contrast case -- expected to come out axis-aligned (untilted),
confirming `VSYS` errors don't leak into `s1` under symmetric coverage.
`fig_s1_vs_pa` is the decisive plot: if every ring nulls at the same PA
offset, the signal is consistent with a single PA error and the detection is
not robust; if they null at different offsets, no single PA error explains
away the signal.
""")

code(r"""
hp.fig_pa_degeneracy(res)
hp.fig_vsys_degeneracy(res)
hp.fig_s1_vs_pa(res)
None
""")

md(r"""
## Figures 10-11: bootstrap and weighting-scheme comparison

The bootstrap (block-resampled over beam-sized cells, since adjacent 4"
pixels are not independent under the MeerKAT beam) is the quoted statistical
error throughout this project -- the formal covariance from the fit assumes
independent pixels, which is false here. `fig_weighting_comparison` shows
whether `s1` is stable across `uniform`/`sin2`/`invvar`; if the schemes
disagree by more than the bootstrap width, that disagreement is itself a
result.
""")

code(r"""
hp.fig_bootstrap(res)
hp.fig_weighting_comparison(res)
None
""")

md(r"""
## Figure 12: V_rad(R) -- the headline result
""")

code(r"""
hp.fig_vrad_profile(res)
None
""")

md(r"""
## Closing summary: the error budget

- **Formal (covariance) errors** are reference only -- they assume
  independent pixels, false at 4" pixels under this beam. The **bootstrap**
  interval (figure 10) is the quoted statistical error.
- **`L0`/`L1`** (figure 2) measure leakage from `VSYS`/`VROT` errors into
  `s1` due to asymmetric azimuthal coverage; they are the quantitative
  justification for holding `VSYS` and `VROT` fixed rather than refitting
  them per ring.
- **PA is the one leak no weighting scheme can suppress** (figure 7): a PA
  error contributes a `sin(theta)` term, the same harmonic as the signal
  itself. Figure 9 (`s1` vs. PA offset) is the direct test of whether the
  detected `s1` survives a plausible PA error, ring by ring.
- **`VSYS` does not leak** under symmetric coverage (figure 8, by contrast
  with figure 7) -- any tilt seen there is a direct, quantitative measurement
  of this data's coverage asymmetry, not a generic property of the method.
- **The weighting-scheme comparison** (figure 11) is a robustness check, not
  a precision one: `sin2` is not claimed to be optimal, only less sensitive
  to the systematics (fixed-`VROT` error, beam smearing, warp) concentrated
  on the major axis.
- **The sign of `s1` is not a physical inflow/outflow claim on its own** --
  `near_side_assumed = "UNRESOLVED"` throughout. That mapping needs external
  information this pipeline does not have (dust lanes, a trailing-arm
  assumption).
- Finally: `VROT` in this ringlog was fitted by 3D-Barolo with `VRAD` fixed
  at zero, so holding it fixed here is formally circular. Figure 7's PA scan
  is the quantitative handle on how much that circularity actually matters
  for the conclusion.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out_path = "harmonic_pipeline.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")
