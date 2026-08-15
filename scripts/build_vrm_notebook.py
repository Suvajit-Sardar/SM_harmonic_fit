"""Generates vrm_pipeline.ipynb programmatically. Run this to rebuild the
notebook from scratch; do not hand-edit the .ipynb directly."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# VRM / VRMA cross-check

Applies the Velocity Ring Model (VRM) and its azimuthally-resolved extension
(VRMA) to this galaxy's velocity field, using the vendored reference
implementation in `vrm/VRM_VRMA.py`
(https://github.com/MatteoStraccamore/VRM_VRMA), and compares the result
against this project's own `s1(R)` measurement from `harmonic_fit.py`.

Method: Sylos Labini, Straccamore, De Marzo & Comerón, "Mapping
non-axisymmetric velocity fields of external galaxies", ApJ 988, 122 (2025),
doi:10.3847/1538-4357/adc71c. See `vrm/README.md` for how VRM/VRMA differs
methodologically from this project's own pipeline, and for a real upstream
gotcha (`phi0input` is accepted but never used) that `vrm/bridge.py` works
around explicitly.

**The short version of the difference:** `harmonic_fit.py` fixes the
rotation curve at Barolo's `VROT` and fits only `c0` and `s1` (=`v_r`). VRM
fits `v_t` *and* `v_r` together, per ring, with nothing held fixed but the
global geometry -- and its rings are equal-width bins in *rescaled* radius
over the full detected extent, not the Barolo tilted-ring edges this
project's other notebook uses. VRM also has **no built-in error
quantification** -- every number below is a point estimate, unlike the
bootstrap + PA/VSYS-scan error budget in `harmonic_pipeline.ipynb`.
""")

code(r"""
%matplotlib inline
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table

import harmonic_fit as hf
from harmonic_plots import custom_rcparams
from vrm.bridge import run_vrm

mpl.rcParams.update(custom_rcparams)
project_dir = Path.cwd()
""")

md(r"""
## Geometry and data-quality cut

Reuses this project's own ringlog, maps, and data-quality mask
(`sigma_artifact_floor_kms`) rather than re-deriving them -- see
`CLAUDE.md` Section 5.3 for why the near-zero-`mom2` cut exists. `P.A.` and
`INC` are identical across all four rings in this ringlog (no warp), so one
global geometry is used for VRM, matching the tilted-ring model's own
assumption in the inner disk.
""")

code(r"""
ringlog = hf.read_ringlog(project_dir / "stage_1_opt_parameters.txt")
mapset = hf.load_maps(project_dir / "maps")
cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)

row0 = ringlog[0]
xc, yc = float(row0["XPOS(pix)"]), float(row0["YPOS(pix)"])
pa_deg, inc_deg = float(row0["P.A.(deg)"]), float(row0["INC(deg)"])
vsys_barolo = float(row0["VSYS(km/s)"])
sigma_artifact_floor_kms = 0.5  # same cut as harmonic_fit.Config's default

print(f"PA = {pa_deg} deg, INC = {inc_deg} deg, Barolo VSYS = {vsys_barolo} km/s")
""")

md(r"""
## VRM: one (v_t, v_r) per ring

`number_rings=4` to sit alongside this project's own 4-ring result -- but
note the x-axis below is **not** guaranteed to line up with the Barolo ring
edges, since VRM bins by rescaled radius over the full extent of the good
pixels, whatever that extent turns out to be.
""")

code(r"""
result_vrm = run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                      number_rings=4, number_arch=1)

print(f"VRM-fit VSYS = {result_vrm['vsys_fit']:.2f} km/s  "
      f"(Barolo ringlog VSYS = {vsys_barolo:.2f} km/s, "
      f"difference = {result_vrm['vsys_fit'] - vsys_barolo:+.2f} km/s)")
print(f"{result_vrm['n_good_pixels']} good pixels used")

result_vrm["table"]
""")

md(r"""
## v_t(R) vs. Barolo's VROT, and v_r(R) vs. this project's s1(R)

Marker size scales with `sqrt(n_points)` in each VRM bin -- a small marker
is a bin fit from very few pixels and should not be trusted. The outermost
VRM ring in particular can extend past where Barolo's rings (and this
project's own analysis) stop, into much sparser territory.
""")

code(r"""
harm_table = Table.read(project_dir / "results" / "ring_results.ecsv", format="ascii.ecsv")
primary_weighting = harm_table.meta["primary_weighting"]
harm_both = harm_table[(harm_table["side"] == "both") & (harm_table["weighting"] == primary_weighting)]
harm_both.sort("ring_index")
r_arcsec_ours = (harm_both["r_in_arcsec"] + harm_both["r_out_arcsec"]) / 2.0

t = result_vrm["table"]
sizes = 15 + 10 * np.sqrt(t["n_points"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.scatter(t["r_arcsec_mean"], t["v_t"], s=sizes, c="tab:blue", edgecolor="black", label="VRM $v_t$", zorder=3)
ax1.plot(r_arcsec_ours, harm_both["vrot_kms"], "s--", color="black", label="Barolo VROT (fixed input)")
ax1.set_xlabel("Radius [arcsec]")
ax1.set_ylabel(r"$v_t$ [km s$^{-1}$]")
ax1.legend()
ax1.set_title("Tangential velocity")

ax2.scatter(t["r_arcsec_mean"], t["v_r"], s=sizes, c="tab:red", edgecolor="black", label="VRM $v_r$", zorder=3)
ax2.errorbar(r_arcsec_ours, harm_both["s1"],
             yerr=[harm_both["s1"] - harm_both["s1_boot_lo"], harm_both["s1_boot_hi"] - harm_both["s1"]],
             fmt="o", color="black", capsize=4, label=f"harmonic_fit $s_1$ ({primary_weighting}, bootstrap err)", zorder=4)
ax2.axhline(0, color="gray", linestyle=":")
ax2.set_xlabel("Radius [arcsec]")
ax2.set_ylabel(r"$v_r$ [km s$^{-1}$]")
ax2.legend()
ax2.set_title("Radial velocity")

fig.suptitle("VRM vs. this project's harmonic fit (VRM marker size ~ sqrt(n_points))")
fig.tight_layout()
None
""")

md(r"""
## VRMA: azimuthally-resolved v_r(R, theta)

Splitting each ring into arcs trades points-per-bin for angular resolution.
Given how patchy this data's azimuthal coverage already is at the per-ring
level (see `fig_coverage` in `harmonic_pipeline.ipynb`), most bins at even
modest resolution end up under-constrained -- shown explicitly via
`n_points` rather than smoothed over. Bins with fewer than 3 points (an
under- or exactly-determined 2-parameter fit) are greyed out.
""")

code(r"""
result_vrma = run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                       number_rings=4, number_arch=4)
t2 = result_vrma["table"]

n_total = len(t2)
n_unreliable = int(np.sum(t2["n_points"] < 3))
print(f"{n_unreliable} / {n_total} (ring, arc) bins have fewer than 3 points")

t2["ring_index", "arc_index", "r_arcsec_mean", "n_points", "v_t", "v_r"]
""")

code(r"""
def _map_panel(ax, result, field, cmap, vlim, title, cbar_label, min_points=3):
    # A *map*: every good pixel plotted at its own (x, y) in the deprojected
    # plane of the galaxy (x = R*cos(theta), y = R*sin(theta)), colored by
    # its VRMA cell's fitted value -- not one point per cell at the cell's
    # mean position. Pixels whose cell has fewer than `min_points` points
    # are shown in gray rather than colored by an unreliable estimate.
    t = result["table"]
    x, y = result["x_arcsec"], result["y_arcsec"]
    cell_index = result["cell_index"]
    n_points = np.asarray(t["n_points"])
    vals_per_cell = np.asarray(t[field], dtype=float)

    in_cell = cell_index >= 0
    idx = np.clip(cell_index, 0, None)
    pixel_vals = np.where(in_cell, vals_per_cell[idx], np.nan)
    reliable = in_cell & (n_points[idx] >= min_points) & np.isfinite(pixel_vals)

    ax.scatter(x[in_cell & ~reliable], y[in_cell & ~reliable],
               s=10, c="lightgray", edgecolor="none", zorder=2)
    sc = ax.scatter(x[reliable], y[reliable], c=pixel_vals[reliable],
                     cmap=cmap, vmin=vlim[0], vmax=vlim[1], s=18, edgecolor="none", zorder=3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.06, label=cbar_label)
    return sc


fig, ax = plt.subplots(figsize=(7.5, 7.5))
_map_panel(ax, result_vrma, "v_r", "RdBu_r", (-100, 100),
           "VRMA $v_r(R,\\theta)$, $N_r=4$, $N_a=4$", r"$v_r$ [km s$^{-1}$]")
fig.tight_layout()
None
""")

md(r"""
## Figure 8 reproduction: v_t, v_r, and their rank-correlation map, vs. a toy null model

Reproduces Figure 8 of Sylos Labini, De Marzo & Straccamore (2025), ApJ 988,
122, doi:10.3847/1538-4357/adc71c -- their warp/radial-flow diagnostic, built
from: the VRMA-reconstructed `v_t(R,theta)` and `v_r(R,theta)` for this
galaxy; the same for a toy/null model; the rank (Spearman) correlation
coefficient map between `v_t` and `v_r`, for both; the observed velocity
dispersion `sigma(R,theta)`; and its rank-correlation with `v_t` and `v_r`.
A tenth panel (not in the paper) adds the LOS residual, at your request.

**Two interpretive choices, documented in `vrm/README.md`:**

1. **The toy model is Barolo's own model map.** The paper's toy model is
   pure circular rotation built from the TRM's `i(R)`, `phi0(R)`, `v_c(R)`,
   with `v_r=0`. This ringlog has **no warp** (`P.A.`, `INC` constant across
   all four rings) and `VRAD` fixed at 0 -- which means `_local_1mom.fits`
   (Barolo's own model map, already loaded as `mapset.model_mom1`) *is*
   exactly that toy model, generated independently by BBarolo. Rather than
   resynthesize a duplicate, this reuses it directly: same mask, same
   geometry, run through the identical VRMA pipeline.
2. **The correlation "map."** The paper's Eq. 2 sums a per-cell term (Eq. 3)
   over *all* cells to give a single scalar; the text and the Figure 8
   caption both call the plotted quantity a spatial "map" `r_fg(R,theta)`
   without giving a separate formula for it. The map used below is that
   per-cell term itself (Spearman version: standardized ranks, not raw
   values) -- the only quantity in the paper's own equations that varies by
   position *and* whose sum, divided by `N_cells - 1`, reproduces Eq. 2's
   scalar exactly.

**Resolution:** the paper's fiducial grid is `Nr=50, Na=32` on THINGS data
(thousands of pixels per galaxy). This dataset has ~440 good pixels *total*
-- `Nr=4, Na=4` (matching the VRMA demo above) is already only ~20-30
points/cell in the best rings. Cells with fewer than 3 points (the minimum
for a determined 2-parameter fit) are excluded from every quantity below,
not just greyed out, since `VRM._matrix()` returns exactly `0.0` for an
empty cell -- a real number that would otherwise silently corrupt the
correlation maps and the residual.
""")

code(r"""
from vrm.bridge import run_figure8_analysis

fig8 = run_figure8_analysis(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                             number_rings=4, number_arch=4)

n_usable = int(np.sum(np.asarray(fig8["table"]["n_points"]) >= 3))
n_cells = len(fig8["table"])

print(f"VRM-fit VSYS: data = {fig8['vsys_fit']:.2f} km/s, toy model = {fig8['vsys_fit_toy']:.2f} km/s "
      f"(Barolo ringlog VSYS = {vsys_barolo:.2f} km/s)")
print(f"{n_usable} / {n_cells} cells usable (>= 3 points)")
print(f"Overall Spearman C(r_vtvr data, r_vtvr toy) = {fig8['C_warp']:.2f}  "
      f"(paper's threshold: |C| > 0.2 suggests a warp signature -- but with only "
      f"{n_usable} usable cells here, versus the paper's Nr=50 x Na=32 = 1600, "
      f"treat this as exploratory, not a firm detection)")
print(f"C(sigma, v_t) = {fig8['C_sigma_vt']:.2f}, C(sigma, v_r) = {fig8['C_sigma_vr']:.2f}")

fig8["table"]["ring_index", "arc_index", "n_points", "v_t", "v_r", "v_t_tm", "v_r_tm", "sigma_mean", "residual"]
""")

code(r"""
t = fig8["table"]
v_t_all = np.concatenate([np.asarray(t["v_t"]), np.asarray(t["v_t_tm"])])
v_t_lim = (np.nanmin(v_t_all), np.nanmax(v_t_all))
v_r_lim_abs = np.nanmax(np.abs(np.concatenate([np.asarray(t["v_r"]), np.asarray(t["v_r_tm"])])))
r_lim_abs = np.nanmax(np.abs(np.concatenate([
    np.asarray(t["r_vtvr"]), np.asarray(t["r_vtvr_tm"]), np.asarray(t["r_sigma_vt"]), np.asarray(t["r_sigma_vr"]),
])))
sigma_lim = (0, np.nanmax(np.asarray(t["sigma_mean"])))

fig, axes = plt.subplots(3, 3, figsize=(15, 16))

_map_panel(axes[0, 0], fig8, "v_t", "viridis", v_t_lim, "(a) $v_t(R,\\theta)$ -- data", "km/s")
_map_panel(axes[0, 1], fig8, "v_r", "RdBu_r", (-v_r_lim_abs, v_r_lim_abs), "(b) $v_r(R,\\theta)$ -- data", "km/s")
_map_panel(axes[0, 2], fig8, "r_vtvr", "RdBu_r", (-r_lim_abs, r_lim_abs), "(c) $r_{v_rv_t}(R,\\theta)$ -- data", "")

_map_panel(axes[1, 0], fig8, "v_t_tm", "viridis", v_t_lim, "(d) $v_t^{tm}(R,\\theta)$ -- toy model", "km/s")
_map_panel(axes[1, 1], fig8, "v_r_tm", "RdBu_r", (-v_r_lim_abs, v_r_lim_abs), "(e) $v_r^{tm}(R,\\theta)$ -- toy model", "km/s")
_map_panel(axes[1, 2], fig8, "r_vtvr_tm", "RdBu_r", (-r_lim_abs, r_lim_abs), "(f) $r^{tm}_{v_rv_t}(R,\\theta)$ -- toy model", "")

_map_panel(axes[2, 0], fig8, "sigma_mean", "magma", sigma_lim, "(g) $\\sigma(R,\\theta)$ -- data", "km/s")
_map_panel(axes[2, 1], fig8, "r_sigma_vt", "RdBu_r", (-r_lim_abs, r_lim_abs), "(h) $r_{\\sigma v_t}(R,\\theta)$", "")
_map_panel(axes[2, 2], fig8, "r_sigma_vr", "RdBu_r", (-r_lim_abs, r_lim_abs), "(i) $r_{\\sigma v_r}(R,\\theta)$", "")

fig.suptitle("Figure 8 reproduction (Sylos Labini, De Marzo & Straccamore 2025)", fontsize=16, y=0.995)
fig.tight_layout()
None
""")

md(r"""
### Added panel: LOS residual (not in the paper)

Observed `v_los` minus the VRMA-reconstructed `v_los` (each cell's own
fitted `v_t`, `v_r` run back through Eq. 1), binned the same way as every
panel above -- the same diagnostic role as `fig_residual_maps`'s post-fit
panel elsewhere in this project, here for the VRM fit rather than the
harmonic fit.
""")

code(r"""
resid_lim = np.nanmax(np.abs(np.asarray(t["residual"])))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
_map_panel(ax, fig8, "residual", "RdBu_r", (-resid_lim, resid_lim), "LOS residual (data - VRMA reconstruction)", "km/s")
fig.tight_layout()
None
""")

md(r"""
## Summary

- VRM's independently-fit systemic velocity and this project's Barolo-ringlog
  `VSYS` should agree closely if the two methods' geometry assumptions are
  compatible; a large discrepancy is itself informative.
- The `v_t(R)` comparison is a sanity check on VRM's geometry, not a new
  result -- Barolo's `VROT` was held fixed as an *input* to `harmonic_fit.py`,
  so close agreement there is closer to "necessary" than "confirmatory";
  VRM fits it independently, and its rings extend past this project's own
  fitted radius range, into territory Barolo's tilted-ring model was never
  asked to cover either.
- The `v_r(R)` comparison is the actual cross-check: VRM makes no assumption
  about `VROT`, uses a different radial binning, and has entirely different
  systematics (no PA-degeneracy handling, no bootstrap) from this project's
  `s1(R)`. Agreement between the two, within VRM's un-quantified point-to-point
  scatter, is independent support for the radial-motion detection; disagreement
  points at systematics specific to one method rather than the other.
- Take every VRM/VRMA point estimate with its `n_points` in hand -- the
  method returns a number even from an empty or near-empty bin (see
  `vrm/README.md`), and this dataset's coverage is patchy enough that this
  happens often once arcs are introduced.
- The Figure 8 reproduction's `C_warp` is a much noisier version of the same
  idea as Table 1 in Sylos Labini, De Marzo & Straccamore (2025): a large
  `|C|` between the data's and the toy model's `r_vtvr(R,theta)` maps points
  at a warp rather than intrinsic radial motion. With only a handful of
  usable cells at this dataset's resolution, read it as a rough, exploratory
  signal, not a resolved warp/no-warp verdict.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out_path = "vrm_pipeline.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")
