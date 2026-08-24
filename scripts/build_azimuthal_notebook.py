"""Generates one azimuthal_modulation_<model>.ipynb per TRM model directory
that has already been fit by harmonic_fit.py (i.e. has a results/
ring_results.ecsv). Mirrors scripts/build_notebook.py / build_timescales_
notebook.py: run this to rebuild from scratch; do not hand-edit the .ipynb
files except in the config cell while actually using one.

This is the azimuthal-modulation analysis described in azimuthal_
modulation.py's module docstring -- not one of the CLAUDE.md Section 0
deliverables, additive only. The main pipeline notebook
(harmonic_pipeline_<model>.ipynb) ends at the V_rad result and the PA
degeneracy; this notebook picks up from there.

Usage:
    python scripts/build_azimuthal_notebook.py                # all eligible models
    python scripts/build_azimuthal_notebook.py --trm-dir X     # just model X
"""

import argparse
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import harmonic_fit as hf  # noqa: E402


def eligible_models(root_dir: Path) -> dict:
    models = hf.discover_trm_models(root_dir)
    return {name: path for name, path in models.items() if (path / "results" / "ring_results.ecsv").is_file()}


def build_notebook(trm_dir_name: str, available_models: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    md(rf"""
# Azimuthal-modulation analysis -- TRM model: `{trm_dir_name}`

Companion, additive analysis alongside the harmonic-fit pipeline (see
`harmonic_pipeline_{trm_dir_name}.ipynb`, which ends at the `V_rad` result
and the PA degeneracy) -- not one of its Section 0 deliverables. All
computation lives in `azimuthal_modulation.py`; this notebook only imports
it and displays figures (`azimuthal_plots.py`), mirroring the split already
used by `harmonic_fit.py` / `harmonic_plots.py`.

Other TRM models with a results/ring_results.ecsv already fit: {", ".join(m for m in available_models if m != trm_dir_name) or "(none)"}.

**Motivation.** The harmonic decomposition returns `V_rad` (`s1`) consistent
with zero in every ring. Two explanations are compatible with that:
**(A)** there is genuinely no coherent radial motion, or **(B)** there is
radial motion whose amplitude and sign vary with azimuth, averaged toward
zero by a fit with only one `sin(theta)` term. Wallin & Struck-Marcell
(1994) predict (B) for an off-centre collision like the one proposed here.

**Three statistics used by an earlier version of this analysis were
invalid** and are fixed in `azimuthal_modulation.py` before anything else is
added:

1. **`delta_chi2` was not a likelihood ratio.** It mixed the fit's own
   objective (`sum(w*resid^2)`, `w=sin^2(theta)`) with a differently-weighted
   chi2 (`sum((resid/sigma)^2)`), so it could -- and did (ring 3:
   `delta_chi2 = -0.04`) -- go negative for a nested model, which is
   impossible. Replaced with a **parametric bootstrap likelihood-ratio
   test** (`parametric_null_bootstrap`): one consistent objective `S`
   throughout, a null distribution built by resampling the best-fit 2-term
   model's own residuals in beam-sized blocks, and `p_bootstrap` calibrated
   empirically rather than assumed to follow chi2(2).
2. **`V_r1` is positive-definite** (a Rayleigh amplitude, like a
   polarization fraction) and a bootstrap interval on it can never contain
   zero -- "boot_lo excludes zero" is not evidence of anything. Replaced
   with: `c2`, `s2` reported separately with their joint (2 dof)
   significance `chi2_2`; the **null distribution of `V_r1`** from the same
   parametric bootstrap; and the **debiased amplitude**
   `V_r1_debiased = 2*sqrt(max(c2^2+s2^2-sigma_c^2, 0))`.
3. **`L_corr` was compared against the beam FWHM directly**, which cannot
   happen for beam-smoothed data (both values came out *below* the beam).
   A Gaussian beam's own autocorrelation has HWHM `beam_FWHM/sqrt(2)`, not
   `beam_FWHM`. Replaced with an **empirical reference**: beam-convolved
   white noise on the same grid and mask, measured with the same estimator,
   500 times.

See `azimuthal_modulation.py`'s module and function docstrings for the full
derivations, and its self-test (`python azimuthal_modulation.py --selftest`)
for the six acceptance tests, including a Rayleigh-bias calibration check
and a regression check against `results/ring_results.ecsv`.
""")

    md(r"""
## Config

The one cell in this notebook you are expected to edit -- normally just
`TRM_DIR`. `azimuthal_modulation.Config`'s other fields are documented in
the module: the model ladder always uses the primary weighting recorded in
`ring_results.ecsv`; `n_bootstrap`/`n_null`/`n_noise_trials` control the
three bootstrap-based analyses above; `n_sector` controls the azimuthal
segment analysis (Section 2.4); the rest configure the three geometric
scans (Section 3).
""")

    code(rf"""
%matplotlib inline
from pathlib import Path
import numpy as np
import harmonic_fit as hf
import azimuthal_modulation as az
import azimuthal_plots as azp

repo_root = Path.cwd()
TRM_DIR = "{trm_dir_name}"  # which (already-fit) TRM model directory to run

trm_dir = repo_root / TRM_DIR
cfg = az.Config(
    trm_dir=trm_dir,
    maps_dir=trm_dir / "maps",
    ringlog_path=hf.find_ringlog(trm_dir),
    results_dir=trm_dir / "results",
)
cfg
""")

    md(r"""
## Run the analysis

Fits the model ladder (`M_c0`, `M_s1`, `M_c0s1`, `M_m2`, `M_m3`) per ring,
per side, with bootstrap parameter intervals; the parametric null-bootstrap
model comparison and `V_r1` null distribution (side="both"); the empirical
`L_corr` reference; the azimuthal-sector and `(R, theta)` analyses; and the
three geometric scans (inclination, centre, PA). Writes
`results/azimuthal_modulation.ecsv`, `results/azimuthal_maps.npz`, and
`results/azimuthal_diagnostics.json`. A few tens of seconds on this dataset
(dominated by the `n_bootstrap` x 5-model ladder and the `n_null`-draw
parametric bootstrap).
""")

    code(r"""
result = az.main(cfg)
""")

    md(r"""
## Model ladder table

`over_parameterised = True` means `n_params >= n_eff` -- flagged explicitly,
not left implicit. `n_eff` on this dataset is a handful of independent
beams per ring; `M_m2` (4 params) and `M_m3` (6 params) are the ones to
watch. `AIC`/`BIC` use `n_eff`, not `n_pix`, and are indicative only given
how small `n_eff` is here.

`M_c0` and `M_s1` are included specifically to show how much of the fit
quality comes from the offset alone versus the radial term alone -- the
comparison behind the manuscript's claim that the residual improvement is
attributed largely to `c0`.
""")

    code(r"""
res = azp.load_results(cfg.results_dir)
sub = res.rows(side="both")
sub["ring_index", "model", "n_params", "n_eff", "n_eff_minus_k", "over_parameterised",
    "S", "rms", "AIC", "BIC", "c0", "s1", "c2", "s2"]
""")

    md(r"""
## Section 1.1-1.2: the bootstrap-calibrated model comparison

Printed above by `az.main`, repeated here as a table: `dS_obs`, `p_bootstrap`
(the calibrated significance of the 2-term -> 4-term improvement), `c2`,
`s2` and their joint significance `chi2_2` (2 dof; **note** this uses the
*formal* sandwich covariance, which assumes independent pixels and is
therefore optimistic -- `p_bootstrap` is the properly beam-correlated
number), `V_r1`, the debiased amplitude, and the null-distribution
percentiles it should be read against.
""")

    code(r"""
rows = []
for i in range(res.n_rings):
    d = result["diagnostics"]["per_ring"][f"ring{i}"]
    nb, vs = d["null_bootstrap"], d["vr1_summary"]
    rows.append(dict(ring=i, dS_obs=nb["dS_obs"], p_bootstrap=nb["p_bootstrap"],
                      c2=vs["c2"], s2=vs["s2"], chi2_2=vs["chi2_2"],
                      V_r1=vs["V_r1"], V_r1_debiased=vs["V_r1_debiased"],
                      V_r1_null_p50=vs["V_r1_null_p50"], V_r1_null_p84=vs["V_r1_null_p84"],
                      theta_1=vs["theta_1"], mean_to_halfwidth=vs["V_r1_mean_to_halfwidth"]))
from astropy.table import Table
Table(rows=rows)
""")

    md(r"""
## Figures: residual diagnostics (Section 2.2-2.3)

`fig_residual_ladder_maps`: one row per model, raw residual and residual
smoothed to 2x the beam, shared colour scale -- coherent structure should
be visible by eye in the smoothed column and shrink as the ladder grows.
`fig_m2_component_map`: `resid(M_c0s1) - resid(M_m2)`, exactly the fitted
`m=2` component, with each ring's `theta_1` direction marked -- a clean
two-lobed pattern aligned with `theta_1` supports a real modulation; a blob
or single bright region would mean the fit is responding to a local
feature. `fig_residual_histograms`: per-model residual distributions, per
ring and pooled -- pair any narrowing with the bootstrap `p_bootstrap`
above, not on its own.
""")

    code(r"""
azp.fig_residual_ladder_maps(res)
None
""")

    code(r"""
azp.fig_m2_component_map(res)
None
""")

    code(r"""
azp.fig_residual_histograms(res)
None
""")

    md(r"""
## Figures: azimuthal segment analysis (Section 2.4)

The most direct display of the hypothesis being tested. `fig_sector_analysis`
bins `M_c0s1`'s residuals into `n_sector` azimuthal sectors and overlays the
fitted `m=2` curve -- if the modulation is real, the sector means should
trace it; if they scatter randomly, `c2`/`s2` are fitting noise.
`fig_rtheta_grid` puts every ring's sectors in one `(R, theta)` image:
coherent vertical banding means a phase that doesn't change with radius,
diagonal banding means it drifts.
""")

    code(r"""
azp.fig_sector_analysis(res)
None
""")

    code(r"""
azp.fig_rtheta_grid(res)
None
""")

    md(r"""
## Figure: approaching/receding split (Section 2.5)

`s1` disagrees in sign between the two halves (already visible in
`harmonic_pipeline_{trm}.ipynb`'s figures); the question here is whether
`c2`, `s2` (and therefore `V_r1`, `theta_1`) agree between halves. A genuine
azimuthal modulation should appear consistently in both; a fitting artifact
need not.
""".replace("{trm}", trm_dir_name))

    code(r"""
azp.fig_side_comparison(res)
None
""")

    md(r"""
## Figures: geometric alternatives (Section 3)

`c2`/`s2` are produced by an inclination error, an oval distortion, or a
warp, not only by a modulated radial flow (Schoenmakers, Franx & de Zeeuw
1997). Three scans, all holding the pixel mask fixed at the fiducial
geometry:

- **Inclination** (`fig_inclination_scan`): if `V_r1` can be driven to zero
  within a plausible inclination error, the detection is not safe.
- **Kinematic centre** (`fig_centre_scan`): the offset minimising
  `|s1_approaching - s1_receding|` -- directly testing the centring
  explanation already offered for that disagreement. Watch for "AT SCAN
  EDGE" in the title: it means the scan range was too narrow to bracket the
  true minimum for that ring.
- **Position angle** (`fig_pa_scan_vr1`): `V_r1(PA)` should be flat -- a PA
  error only produces `m=2` at second order.
""")

    code(r"""
azp.fig_inclination_scan(res)
None
""")

    code(r"""
azp.fig_centre_scan(res)
None
""")

    code(r"""
azp.fig_pa_scan_vr1(res)
None
""")

    md(r"""
## Figures: residual autocorrelation and theta_1 compass
""")

    code(r"""
azp.fig_residual_autocorr(res)
None
""")

    code(r"""
void_azimuth_deg = result["diagnostics"]["void_azimuth_deg"]
azp.fig_theta1_compass(res, void_azimuth_deg=void_azimuth_deg)
None
""")

    md(rf"""
## Caveats

Each caveat below cites the number that quantifies it -- read from the run
above, not restated from a prior run.
""")

    code(r"""
diag = result["diagnostics"]
n_rings = res.n_rings
beam = diag["beam_fwhm_arcsec"]
ring_w = diag["ring_width_arcsec"]
dilution = diag["beam_dilution_ratio"]
rayleigh = diag["rayleigh_theta1"]
void_az = diag["void_azimuth_deg"]
theta1s = diag["theta1_both_per_ring_deg"]

n_eff_list = [res.row(i, model="M_m2")["n_eff"] for i in range(n_rings)]
n_overparam_m2 = sum(1 for i in range(n_rings) if bool(res.row(i, model="M_m2")["over_parameterised"]))
n_overparam_m3 = sum(1 for i in range(n_rings) if bool(res.row(i, model="M_m3")["over_parameterised"]))

print("1. Parameter count.")
print(f"   n_eff per ring = {[f'{v:.1f}' for v in n_eff_list]} against 4 params in M_m2, 6 in M_m3.")
print(f"   {n_overparam_m2}/{n_rings} rings over-parameterised at M_m2; {n_overparam_m3}/{n_rings} at M_m3.\n")

print("2. Positive-definite amplitude.")
print("   V_r1 cannot be consistent with zero by construction. Per-ring V_r1 / V_r1_debiased / "
      "V_r1_null_p50 / joint chi2_2 (formal cov, optimistic) / p_bootstrap (beam-correlated, the number to trust):")
for i in range(n_rings):
    d = diag["per_ring"][f"ring{i}"]
    vs, nb = d["vr1_summary"], d["null_bootstrap"]
    print(f"     ring{i}: V_r1={vs['V_r1']:.1f}  debiased={vs['V_r1_debiased']:.1f}  "
          f"null_p50={vs['V_r1_null_p50']:.1f}  chi2_2={vs['chi2_2']:.1f}  p_bootstrap={nb['p_bootstrap']:.3f}")
print()

print("3. Beam dilution.")
print(f"   beam FWHM = {beam:.2f}\" vs. ring width = {ring_w:.2f}\" -> ratio = {dilution:.2f}. "
      "Coherent radial motion is averaged across radii within a resolution element.\n")

print("4. Inclination degeneracy (Section 3.1).")
for i in range(n_rings):
    d = diag["per_ring"][f"ring{i}"]["inclination_scan"]
    print(f"     ring{i}: V_r1=0 reachable within scan = {d['vr1_zero_in_range']}, "
          f"min V_r1={d['vr1_min_kms']:.1f} km/s at INC offset {d['inc_offset_at_vr1_min_deg']:+.1f} deg")
print()

print("5. Warp.")
print("   A radially varying position angle reproduces ring-dependent behaviour as naturally as a "
      "modulated flow and cannot be separated from it with a line-of-sight velocity field alone "
      "(Sylos Labini et al. 2025).\n")

print("6. Phase coherence, and what it is worth.")
print(f"   theta_1 = {[f'{v:.1f}' for v in theta1s]} deg, scatter = {np.std(theta1s):.1f} deg.")
print(f"   Rayleigh test: Rbar={rayleigh['Rbar']:.3f}, z={rayleigh['z']:.3f}, p={rayleigh['p']:.4f} (n={rayleigh['n']}).")
print(f"   Beam ({beam:.1f}\") exceeds the ring width ({ring_w:.1f}\"), so the {n_rings} rings are not "
      "independent -- read p as suggestive, not a calibrated small-n result.\n")

print("7. The morphological cross-check failed.")
if void_az is not None:
    mean_theta1 = np.degrees(np.arctan2(np.mean(np.sin(np.radians(theta1s))), np.mean(np.cos(np.radians(theta1s))))) % 360
    print(f"   Void azimuth (disk frame) = {void_az:.1f} deg vs. theta_1 ~ {mean_theta1:.1f} deg: "
          f"neither aligned nor anti-aligned (difference = {min(abs(void_az-mean_theta1), 360-abs(void_az-mean_theta1)):.1f} deg). "
          "The modulation phase does not point at the void, so the off-centre-impact interpretation "
          "has no independent morphological support.\n")
else:
    print("   Void azimuth cross-check unavailable for this TRM model.\n")

print("8. Gas cannot multi-stream.")
print("   Wallin & Struck-Marcell's Fig. 6 shows infalling and outflowing particles coexisting at one "
      "position, which HI cannot do. Azimuthal modulation (different theta, same R) remains available "
      "as an explanation; multi-streaming at a point does not.\n")

print("9. Coincidence with the alternative TRM.")
print(f"   V_r1 ~ {min(res.rows(side='both', model='M_m2')['c2']**2 + res.rows(side='both', model='M_m2')['s2']**2)**0.5*2:.0f}"
      f"-{max((res.rows(side='both', model='M_m2')['c2']**2 + res.rows(side='both', model='M_m2')['s2']**2)**0.5*2):.0f} km/s "
      "is close to the s1 ~ 47 km/s obtained at PA = 53.445 (the other TRM model). A PA error generates m=2 "
      "only at second order (~5 km/s at dphi=11 deg), so this is not a direct artifact, but the numerical "
      "closeness should be stated.")
""")

    md(r"""
## Verdict
""")

    code(r"""
v = diag["verdict"]
print(f"VERDICT: {v['verdict']}")
print(f"  p_bootstrap < 0.05: {v['n_rings_p_bootstrap_lt_0.05']}/{n_rings} rings -- {v['p_bootstrap_per_ring']}")
print(f"  joint (c2,s2) chi2_2 > 5.99 (95%, 2 dof, formal cov): {v['n_rings_chi2_2_gt_5.99']}/{n_rings} rings -- "
      f"{[f'{c:.1f}' for c in v['chi2_2_per_ring']]}")
print(f"  inclination-safe (V_r1=0 unreachable within scan): {v['n_rings_inclination_safe']}/{n_rings} rings")
print()
print("Read alongside caveats 1-9 above -- in particular, chi2_2 uses the formal (pixel-independence-"
      "assuming) covariance and is optimistic; p_bootstrap is the number that accounts for beam "
      "correlation and is also, per azimuthal_modulation.py's own calibration note, itself somewhat "
      "conservative -- so a p_bootstrap in the 0.3-0.5 range is not strong evidence for the null either.")
""")

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trm-dir", default=None,
                         help="build only this TRM model's notebook (default: all eligible TRM model directories)")
    args = parser.parse_args()

    models = eligible_models(REPO_ROOT)
    if not models:
        raise SystemExit(f"No TRM model directories with results/ring_results.ecsv found under {REPO_ROOT}.")

    targets = [args.trm_dir] if args.trm_dir else sorted(models)
    for name in targets:
        if name not in models:
            raise SystemExit(f"--trm-dir {name!r} not found or not yet fit; available: {sorted(models)}")
        nb = build_notebook(name, sorted(models))
        out_path = REPO_ROOT / f"azimuthal_modulation_{name}.ipynb"
        with open(out_path, "w") as f:
            nbf.write(nb, f)
        print(f"Wrote {out_path}")
