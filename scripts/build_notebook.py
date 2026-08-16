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
project owner), so ring *i* spans `RAD - 4.85" -> RAD + 4.85"`.

**Sign-convention caveat.** `s1 > 0` means outward motion *only if the near
side of the disc is known*; inclination alone is degenerate under `i -> -i`.
Every table and figure below carries `near_side_assumed = "UNRESOLVED"`.
Treat the sign of `s1` as a number to be interpreted with external
information (dust lanes, trailing-arm assumption), not a ready-made
inflow/outflow claim.
""")

md(r"""
## Config

The one cell in this notebook you are expected to edit.
""")

code(r"""
%matplotlib inline
from pathlib import Path
import numpy as np
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
radii, `kpc_per_arcsec`) comes from `stage_1_opt_parameters.txt`. The moment
maps are matched by globbing `maps/` for `*_1mom.fits` / `*_2mom.fits`
(data) and `*_local_1mom.fits` / `*_local_2mom.fits` (Barolo model).
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
## Geometry and the receding-side check

`PA_math = PA_barolo + 90 deg` is only valid for an east-left image
(`CDELT1 < 0`, `origin='lower'`); this is asserted from the header, not
assumed. As a sanity check before fitting anything: within the outermost
ring, the *Barolo model* mom1 (not the data, so real non-circular motion
can't confuse the test) must be redshifted of `VSYS` on the
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

Fits `s1` per ring (all three weighting schemes, all three sides), runs the
PA/VSYS scans, and bootstraps the statistical error -- then writes
`results/ring_results.ecsv`, `results/maps.npz`, and `results/scans.npz`.
Takes a few seconds on this dataset.
""")

code(r"""
results_table = hf.main(cfg)
""")

md(r"""
## Figure 1-2: convention checks

`fig_theta_map` shows `theta`, the approaching/receding split, and the
`w(theta)` weight map -- confirms the geometry and weighting look right
before trusting anything downstream. `fig_coverage` shows the azimuthal
coverage per ring with `L0`/`L1` printed on each panel: how much a `VSYS`
or `VROT` error would leak into `s1` for this ring's specific coverage gaps.
""")

code(r"""
res = hp.load_results(cfg.results_dir)

hp.fig_theta_map(res)
hp.fig_coverage(res)
None
""")

md(r"""
## `ring_results.ecsv`
""")

code(r"""
res.table
""")

md(r"""
## Figure 3: azimuthal comparison

Data, the Barolo model (through the same wedges and mask, so it carries the
same beam smearing), and the harmonic fit, all against azimuth. Where data
and Barolo diverge but data and the harmonic fit agree, that gap *is* the
`s1` detection, shown directly.

Error bars are the per-ring fit residual RMS divided by `sqrt(beams per
wedge)`, not the raw within-wedge pixel scatter -- at this dataset's beam
size (~20.7 pixels/beam) a wedge holds well under one independent beam even
at the `AZIMUTHAL_N_WEDGE = 12` used here, so the within-wedge scatter would
mostly measure the model's own gradient across the wedge (largest near the
minor axis, exactly where the `s1` signal lives) rather than measurement
noise. Markers sit at each wedge's weighted circular centroid, not its
geometric center.
""")

code(r"""
hp.fig_azimuthal_vlos(res, n_wedge=hp.AZIMUTHAL_N_WEDGE)
None
""")

md(r"""
**Beams per wedge** (this belongs in the paper regardless of how the figure
ends up looking, since it is what justifies the error-bar model above):

| ring | n_eff (beams in ring) | median beams/wedge (N=12) | old median error bar | new median error bar | old max/min ratio | new max/min ratio |
|---|---|---|---|---|---|---|
| 0 | 2.85 | 1.00 | 14.8 km/s | 19.0 km/s | inf (exact-zero wedges) | 1.00 |
| 1 | 5.51 | 1.00 | 20.4 km/s | 15.6 km/s | 41.4 | 1.00 |
| 2 | 6.72 | 1.00 | 15.0 km/s | 17.6 km/s | inf (exact-zero wedges) | 1.00 |
| 3 | 4.74 | 1.00 | 13.7 km/s | 18.1 km/s | inf (exact-zero wedges) | 1.00 |

Every wedge in every ring, even at the coarser 12-wedge resolution, holds
fewer pixels than one beam (`pixels_per_beam = 20.69`) -- so the beam floor
in the new error model binds everywhere, and the error bars come out exactly
flat within each ring (ratio = 1.00), not merely flatter than before. The
old (pre-fix) formula's ratio is not just "a few times too large": three of
the four rings have at least one wedge with exactly one pixel, where the
within-wedge weighted variance of a single point about its own mean is
exactly zero, making that wedge's old error bar exactly zero and the
max/min ratio formally infinite -- a sharper failure than the systematic
`sin(theta)`-tracking bias the fix targets, and a direct consequence of the
same root cause (estimating scatter from far fewer independent samples than
pixels).
""")

md(r"""
## Figure 4: pre-fit and post-fit residual maps

**The pre-fit panel (left) is the most important diagnostic in the
project.** Genuine axisymmetric radial motion appears as a clean dipole
aligned with the minor axis. A localised blob, a one-sided feature, or
something tracking tidal structure means a single `s1` per ring is the
wrong model -- worth ruling out given the group environment this target
sits in. The post-fit panel (right, after subtracting `s1*sin(theta)`)
should be structureless; any coherent pattern remaining is the argument for
enabling `c2, s2`.
""")

code(r"""
hp.fig_residual_maps(res)
None
""")

md(r"""
## The V_rad-PA degeneracy: analytic derivation

A PA error `dPA` contributes a `sin(theta)` term to the residual -- the same
harmonic as the `s1` signal itself, so no weighting scheme can suppress it
(Section 5.6 of `CLAUDE.md`). The flat-sky approximation for this leak,
`s1_leak = -VROT * dPA[rad]`, is **not accurate at this galaxy's
inclination**: differentiating `make_geometry` shows a PA error does not
shift `theta` by a uniform amount once you deproject through `cos(inc)`:

```
dtheta/d(dPA) = -(cos(theta)^2/cos(inc) + cos(inc)*sin(theta)^2)  =  g(theta)
```

which ranges from `-1/cos(inc)` on the major axis to `-cos(inc)` on the
minor axis. Propagating this through the weighted `s1` estimator gives the
actual leakage slope:

```
K(scheme) = sum(w * g(theta) * sin(theta)**2) / sum(w * sin(theta)**2)
s1_leak   = VROT * dPA[rad] * K(scheme)
```

with closed forms for constant-`sigma` weighting (`ci = cos(inc)`):
`K_uniform = -(1/ci + 3*ci)/4`, `K_sin2 = -(1/ci + 5*ci)/6`. This is the
line overplotted on `fig_pa_degeneracy` below (via
`harmonic_fit.pa_degeneracy_slope`, computed numerically from each ring's
actual masked `theta`/`w`, not the closed form) -- and it is also the
inclination-corrected acceptance test in `harmonic_fit.py --selftest`
(test 2): if the numerical chi2 valley did not follow this line, the
deprojection would be wrong.
""")

code(r"""
ci = np.cos(np.radians(res.table["inc_deg"][0]))
print(f"INC = {res.table['inc_deg'][0]:.3f} deg, cos(inc) = {ci:.4f}")
print(f"K_uniform (closed form) = {-(1/ci + 3*ci)/4:.3f}")
print(f"K_sin2 (closed form)    = {-(1/ci + 5*ci)/6:.3f}")
print()
print(f"{'ring':<6}{'VROT [km/s]':>14}{'K(sin2)':>10}{'slope [km/s/deg]':>20}")
for i in range(res.n_rings):
    theta = res.maps[f"ring{i}_theta"]
    mask = res.maps[f"ring{i}_ring_mask_both"]
    w = res.maps[f"ring{i}_weights_primary"]
    row = res.row(i)
    K = hf.pa_degeneracy_slope(theta[mask], w[mask], 1.0, row["inc_deg"])  # VROT=1 -> K itself
    slope_per_deg = row["vrot_kms"] * K * np.pi / 180.0
    print(f"{i:<6}{row['vrot_kms']:14.1f}{K:10.3f}{slope_per_deg:20.2f}")
""")

md(r"""
## Figures 5-7: PA/VSYS degeneracy and s1 vs. PA offset

`fig_pa_degeneracy` doubles as an acceptance test for the geometry code: the
numerical chi2 valley should track the inclination-corrected analytic line
(just derived above) closely. `fig_vsys_degeneracy` is the contrast case --
expected to come out axis-aligned (untilted), confirming `VSYS` errors don't
leak into `s1` under symmetric coverage. `fig_s1_vs_pa` is the decisive
plot: if every ring nulls at the same PA offset, the signal is consistent
with a single PA error and the detection is not robust; if they null at
different offsets, no single PA error explains away the signal.
""")

code(r"""
hp.fig_pa_degeneracy(res)
hp.fig_vsys_degeneracy(res)
hp.fig_s1_vs_pa(res)
None
""")

md(r"""
## Figures 8-9: bootstrap and weighting-scheme comparison

The bootstrap interval is the quoted statistical error throughout this
project (the formal covariance from the fit assumes independent pixels,
which is false at 4" pixels under this beam). `fig_weighting_comparison`
shows whether `s1` is stable across `uniform`/`sin2`/`invvar`; if the
schemes disagree by more than the bootstrap width, that disagreement is
itself a result.
""")

code(r"""
hp.fig_bootstrap(res)
hp.fig_weighting_comparison(res)
None
""")

md(r"""
## Figure 10: V_rad(R) -- the headline result
""")

code(r"""
hp.fig_vrad_profile(res)
None
""")

md(r"""
## Interpreting the sign of s1

`s1`'s sign only maps onto physical inflow/outflow given external
information about which side of the disc is nearer the observer -- Section
2.6 of `CLAUDE.md`. Given that information, the mapping is a short,
checkable derivation (recorded in full in `CLAUDE.md` Section 2.6.1):

`theta` increases counter-clockwise on the sky (a direct consequence of this
project's confirmed East-left, `CDELT1 < 0` orientation -- `make_geometry`
only ever rotates and positively stretches the pixel frame, never mirrors
it), and `theta = 0` is receding by construction. At `theta = 90 deg` (the
minor axis), pure rotation drops out (`cos(90 deg) = 0`), so the entire
residual there is the radial term: `v_los - VSYS = sin(i) * s1`. The near
side is, by definition, tilted toward the observer -- so outward motion
there blueshifts, inward motion redshifts.

**Given the near side is at `theta = +90 deg`** (consistent with the
observed clockwise rotation on the sky -- the two aren't independent
evidence, see `CLAUDE.md` Section 2.6.1), true outward `V_rad = -s1`. Since
`s1` came out negative in every ring above, that resolves to **outward
motion -- expansion, not infall** -- contingent entirely on that near-side
determination. `near_side_assumed` in `results/ring_results.ecsv` stays
`"UNRESOLVED"`; this is a human interpretation for the paper text, not
something the pipeline itself asserts.
""")

md(r"""
## Closing summary: the error budget

- **Formal (covariance) errors** are reference only -- they assume
  independent pixels, false at 4" pixels under this beam. The **bootstrap**
  interval (figure 8) is the quoted statistical error.
- **`L0`/`L1`** (figure 2) measure leakage from `VSYS`/`VROT` errors into
  `s1` due to asymmetric azimuthal coverage; they are the quantitative
  justification for holding `VSYS` and `VROT` fixed rather than refitting
  them per ring.
- **PA is the one leak no weighting scheme can suppress** (figure 5): a PA
  error contributes a `sin(theta)` term, the same harmonic as the signal
  itself. Figure 7 (`s1` vs. PA offset) is the direct test of whether the
  detected `s1` survives a plausible PA error, ring by ring.
- **`VSYS` does not leak** under symmetric coverage (figure 6, by contrast
  with figure 5) -- any tilt seen there is a direct, quantitative measurement
  of this data's coverage asymmetry, not a generic property of the method.
- **The weighting-scheme comparison** (figure 9) is a robustness check, not
  a precision one: `sin2` is not claimed to be optimal, only less sensitive
  to the systematics (fixed-`VROT` error, beam smearing, warp) concentrated
  on the major axis.
- **The sign of `s1` is not a physical inflow/outflow claim from the
  pipeline alone** -- `near_side_assumed = "UNRESOLVED"` throughout, and the
  code never resolves it. Given the near side stated above
  (`theta = +90 deg`), it reads as outward motion (expansion); flip the near
  side and the reading flips too.
- Finally: `VROT` in this ringlog was fitted by 3D-Barolo with `VRAD` fixed
  at zero, so holding it fixed here is formally circular. Figure 5's PA scan
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
