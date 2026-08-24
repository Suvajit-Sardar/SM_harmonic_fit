"""Figures for the azimuthal-modulation analysis (azimuthal_modulation.py).

Reads only results/ (azimuthal_modulation.ecsv, azimuthal_maps.npz,
azimuthal_diagnostics.json) -- no FITS reading, no fitting. One function per
figure, each returning a matplotlib Figure. main() writes all of them to
figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter

try:
    from harmonic_plots import custom_rcparams

    mpl.rcParams.update(custom_rcparams)
except ImportError:
    pass

MODEL_ORDER = ("M_c0", "M_s1", "M_c0s1", "M_m2", "M_m3")
MODEL_LABELS = {"M_c0": "c0", "M_s1": "s1", "M_c0s1": "c0,s1", "M_m2": "c0,s1,c2,s2", "M_m3": "c0,s1,c2,s2,c3,s3"}
RING_COLORS = ["darkblue", "magenta", "green", "red"]
SIDE_COLORS = {"both": "green", "approaching": "blue", "receding": "red"}


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


class Results:
    def __init__(self, results_dir: Path):
        results_dir = Path(results_dir)
        self.table = Table.read(str(results_dir / "azimuthal_modulation.ecsv"), format="ascii.ecsv")
        self.maps = dict(np.load(results_dir / "azimuthal_maps.npz"))
        self.diagnostics = json.loads((results_dir / "azimuthal_diagnostics.json").read_text())
        self.primary_weighting = self.table.meta["primary_weighting"]
        self.n_rings = int(np.max(self.table["ring_index"])) + 1
        self.n_sector = len(self.diagnostics["per_ring"]["ring0"]["sectors"]["centers_deg"])

    def rows(self, ring_index=None, side=None, model=None):
        t = self.table
        mask = np.ones(len(t), dtype=bool)
        if ring_index is not None:
            mask &= t["ring_index"] == ring_index
        if side is not None:
            mask &= t["side"] == side
        if model is not None:
            mask &= t["model"] == model
        return t[mask]

    def row(self, ring_index, side="both", model="M_c0s1"):
        sub = self.rows(ring_index, side, model)
        if len(sub) != 1:
            raise RuntimeError(f"Expected 1 row for ring={ring_index}, side={side}, model={model}, got {len(sub)}")
        return sub[0]

    def caption_tag(self):
        return f"[primary weighting = {self.primary_weighting}]"


def load_results(results_dir) -> Results:
    return Results(results_dir)


def _ring_edges_arcsec(res: Results):
    sub = res.rows(side="both", model="M_c0s1")
    edges = sorted(set(sub["r_in_arcsec"]).union(sub["r_out_arcsec"]))
    return np.array(edges)


def _overlay_ring_ellipses(res: Results, ax, colors="black"):
    edges = _ring_edges_arcsec(res)
    for i in range(res.n_rings):
        R = res.maps[f"ring{i}_R_arcsec"]
        row = res.row(i)
        ax.contour(R, levels=[row["r_in_arcsec"], row["r_out_arcsec"]], colors=colors, linewidths=0.8, linestyles=":")


def _combined_map(res: Results, key: str):
    shape = res.maps["ring0_theta"].shape
    out = np.full(shape, np.nan)
    for i in range(res.n_rings):
        arr = res.maps[f"ring{i}_{key}"]
        m = np.isfinite(arr)
        out[m] = arr[m]
    return out


# --------------------------------------------------------------------------
# Section 2.2: residual maps grid
# --------------------------------------------------------------------------


def fig_residual_ladder_maps(res: Results, smooth_beams: float = 2.0):
    """A grid: rows are models in the ladder, columns are (residual,
    residual smoothed to smooth_beams x the beam) so coherent structure is
    separable from pixel noise by eye. Shared diverging colour scale,
    masked to the annuli, ring ellipses overlaid."""
    pixscale = float(res.maps.get("pixscale_arcsec", 4.0))
    combined = {label: _combined_map(res, f"resid_{label}") for label in MODEL_ORDER}
    all_vals = np.concatenate([v[np.isfinite(v)] for v in combined.values()])
    vmax = np.nanpercentile(np.abs(all_vals), 98) if len(all_vals) else 1.0

    fig, axes = plt.subplots(len(MODEL_ORDER), 2, figsize=(11, 4.2 * len(MODEL_ORDER)))
    for i, label in enumerate(MODEL_ORDER):
        raw = combined[label]
        im0 = axes[i, 0].imshow(raw, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        _overlay_ring_ellipses(res, axes[i, 0])
        axes[i, 0].set_title(f"{label} ({MODEL_LABELS[label]}): residual")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, label="km/s")

        sigma_pix = smooth_beams * 4.27 / 2.3548  # ~beam FWHM in pixels for this dataset, converted to a Gaussian sigma
        smoothed = np.where(np.isfinite(raw), raw, 0.0)
        weight = np.isfinite(raw).astype(float)
        num = gaussian_filter(smoothed, sigma=sigma_pix)
        den = gaussian_filter(weight, sigma=sigma_pix)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed_map = np.where(den > 0.1, num / den, np.nan)
        im1 = axes[i, 1].imshow(smoothed_map, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        _overlay_ring_ellipses(res, axes[i, 1])
        axes[i, 1].set_title(f"{label}: smoothed x{smooth_beams} beam")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, label="km/s")

    fig.suptitle(f"Residual maps per model in the ladder {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_m2_component_map(res: Results):
    """The difference map resid(M_c0s1) - resid(M_m2), exactly the fitted
    m=2 component. A clean two-lobed pattern aligned with theta_1 supports
    a real modulation; a blob or single bright region means the fit is
    responding to a local feature, not an azimuthal one."""
    c0s1 = _combined_map(res, "resid_M_c0s1")
    m2 = _combined_map(res, "resid_M_m2")
    diff = c0s1 - m2
    vmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98) if np.any(np.isfinite(diff)) else 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _overlay_ring_ellipses(res, ax)
    fig.colorbar(im, ax=ax, fraction=0.046, label="km/s")

    for i in range(res.n_rings):
        row = res.row(i, model="M_m2")
        th1 = np.radians(res.diagnostics["per_ring"][f"ring{i}"]["vr1_summary"]["theta_1"])
        r_mid = (row["r_in_arcsec"] + row["r_out_arcsec"]) / 2.0 / float(res.maps.get("pixscale_arcsec", 4.0))
        # Approximated from the frame center, not the galaxy's own XPOS/YPOS
        # (not carried in azimuthal_modulation.ecsv) -- close enough for an
        # illustrative direction arrow, not a precision geometric claim.
        xc = res.maps["ring0_theta"].shape[1] / 2.0
        yc = res.maps["ring0_theta"].shape[0] / 2.0
        ax.plot([xc, xc + r_mid * np.cos(th1)], [yc, yc + r_mid * np.sin(th1)],
                color=RING_COLORS[i % len(RING_COLORS)], linewidth=1.5, alpha=0.8,
                label=f"ring{i} theta_1={np.degrees(th1)%360:.0f} deg")

    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"resid(M_c0s1) - resid(M_m2): the fitted m=2 component {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Section 2.3: 1D residual histograms
# --------------------------------------------------------------------------


def fig_residual_histograms(res: Results):
    fig, axes = plt.subplots(1, res.n_rings + 1, figsize=(5.5 * (res.n_rings + 1), 5), sharey=False)

    ref_L = res.diagnostics["autocorrelation"]["L_corr_reference_mean_arcsec"]
    all_resid = {label: [] for label in MODEL_ORDER}

    for i, ax in enumerate(axes[:-1]):
        for j, label in enumerate(MODEL_ORDER):
            arr = res.maps[f"ring{i}_resid_{label}"]
            vals = arr[np.isfinite(arr)]
            all_resid[label].append(vals)
            ax.hist(vals, bins=15, histtype="step", linewidth=1.6, label=f"{label} (rms={np.sqrt(np.mean(vals**2)):.1f})")
        ax.set_title(f"ring {i}")
        ax.set_xlabel("residual [km/s]")
        if i == 0:
            ax.set_ylabel("N pixels")
        ax.legend(fontsize=7)

    ax_pool = axes[-1]
    for label in MODEL_ORDER:
        vals = np.concatenate(all_resid[label])
        rms = np.sqrt(np.mean(vals**2))
        med = np.median(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        frac2 = np.mean(np.abs(vals) > 2 * ref_L) if ref_L else np.nan
        frac3 = np.mean(np.abs(vals) > 3 * ref_L) if ref_L else np.nan
        ax_pool.hist(vals, bins=25, histtype="step", linewidth=1.6,
                     label=f"{label}: rms={rms:.1f}, med={med:.1f}, IQR={iqr:.1f}")
        print(f"[fig_residual_histograms] pooled {label}: rms={rms:.2f} med={med:.2f} IQR={iqr:.2f} "
              f"frac>2*Lref={frac2:.3f} frac>3*Lref={frac3:.3f} (not a significance claim -- pair with the bootstrap)")
    ax_pool.set_title("all rings pooled")
    ax_pool.set_xlabel("residual [km/s]")
    ax_pool.legend(fontsize=7)

    fig.suptitle(f"Residual distributions per model (pair narrowing with the Section 1.1 bootstrap) {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Section 2.4: azimuthal segment analysis
# --------------------------------------------------------------------------


def fig_sector_analysis(res: Results):
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5.5 * res.n_rings, 5), sharey=False)
    if res.n_rings == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sec = res.diagnostics["per_ring"][f"ring{i}"]["sectors"]
        centers = np.array(sec["centers_deg"])
        means = np.array(sec["means"])
        errs = np.array(sec["errs"])
        model_curve = np.array(sec["model_curve"])

        ax.errorbar(centers, means, yerr=errs, fmt="o", color="black", capsize=3, label="sector mean residual")
        theta_smooth = np.linspace(0, 360, 200)
        row = res.row(i, model="M_m2")
        model_smooth = row["c2"] * np.cos(2 * np.radians(theta_smooth)) + row["s2"] * np.sin(2 * np.radians(theta_smooth))
        ax.plot(theta_smooth, model_smooth, "-", color=RING_COLORS[i % len(RING_COLORS)], label="fitted m=2")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)

        ax.set_xlabel("Azimuth [deg]")
        if i == 0:
            ax.set_ylabel("Weighted mean residual [km/s]")
        ax.set_title(f"ring {i}: chi2(flat)={sec['chi2_flat']:.1f}, chi2(m=2)={sec['chi2_model']:.1f}")
        ax.legend(fontsize=9)

    fig.suptitle(f"Azimuthal segment analysis of M_c0s1 residuals, n_sector={res.n_sector} {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_rtheta_grid(res: Results):
    """A single (R, theta) image: rows are rings, columns are azimuthal
    sectors. Coherent vertical or diagonal banding is the signature of a
    modulation whose phase is fixed (vertical) or drifts (diagonal) with
    radius."""
    grid = res.maps["rtheta_grid"]
    vmax = np.nanpercentile(np.abs(grid[np.isfinite(grid)]), 98) if np.any(np.isfinite(grid)) else 1.0

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    extent=[0, 360, -0.5, res.n_rings - 0.5])
    fig.colorbar(im, ax=ax, label="mean M_c0s1 residual [km/s]")
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("ring index")
    ax.set_yticks(range(res.n_rings))
    ax.set_title(f"(R, theta) residual grid -- radius-dependence of the phase in one panel {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Section 2.5: approaching/receding split
# --------------------------------------------------------------------------


def fig_side_comparison(res: Results):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    quantities = ["s1", "c2", "s2", "V_r1"]
    for ax, q in zip(axes, quantities):
        for side in ("approaching", "both", "receding"):
            vals = []
            for i in range(res.n_rings):
                ss = res.diagnostics["per_ring"][f"ring{i}"]["side_summary"].get(side)
                vals.append(ss[q] if ss else np.nan)
            ax.plot(range(res.n_rings), vals, "o-", color=SIDE_COLORS[side], label=side, markeredgecolor="black")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(range(res.n_rings))
        ax.set_xlabel("ring index")
        ax.set_ylabel(q)
        ax.set_title(q)
        if q == "s1":
            ax.legend(fontsize=9)
    fig.suptitle(f"Approaching/receding split: does the m=2 signal agree between halves? {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Carried over from the earlier version, adapted to this module's Results
# --------------------------------------------------------------------------


def fig_theta1_compass(res: Results, void_azimuth_deg=None, companion_azimuth_deg=None):
    """Polar compass of theta_1 (M_m2, side=both) per ring."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    vr1s = [res.diagnostics["per_ring"][f"ring{i}"]["vr1_summary"]["V_r1"] for i in range(res.n_rings)]
    rmax = max(vr1s) * 1.15 if vr1s else 1.0
    ax.set_rmax(rmax)

    for i in range(res.n_rings):
        vs = res.diagnostics["per_ring"][f"ring{i}"]["vr1_summary"]
        th = np.radians(vs["theta_1"])
        err = np.radians(vs["theta_1_err"]) if np.isfinite(vs["theta_1_err"]) else 0.0
        r = vs["V_r1"]
        color = RING_COLORS[i % len(RING_COLORS)]
        ax.plot([th, th], [0, r], color=color, linewidth=2,
                label=f"ring {i}: theta_1={vs['theta_1']:.0f} deg, V_r1={r:.0f} km/s")
        ax.plot(th, r, "o", color=color, markeredgecolor="black", zorder=5)
        if err > 0:
            ax.fill_betweenx([0, r], th - err, th + err, color=color, alpha=0.15)

    if void_azimuth_deg is not None:
        th_v = np.radians(void_azimuth_deg)
        ax.plot([th_v, th_v], [0, rmax], color="black", linestyle="--", linewidth=2, label="void azimuth")
    if companion_azimuth_deg is not None:
        th_c = np.radians(companion_azimuth_deg)
        ax.plot([th_c, th_c], [0, rmax], color="purple", linestyle=":", linewidth=2, label="companion direction")

    rayleigh = res.diagnostics["rayleigh_theta1"]
    ax.set_title(f"theta_1 compass -- Rayleigh p={rayleigh['p']:.3f} (n={rayleigh['n']}) {res.caption_tag()}", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=10)
    fig.tight_layout()
    return fig


def fig_residual_autocorr(res: Results):
    rs = res.diagnostics["autocorrelation"]
    L_ref = rs["L_corr_reference_mean_arcsec"]
    L_ref_std = rs["L_corr_reference_std_arcsec"]

    fig, ax = plt.subplots(figsize=(9, 7))
    for key, label, color in (("prefit", "pre-fit", "crimson"), ("postfit", "post-fit (M_c0s1)", "navy")):
        d = rs[key]
        lag = np.asarray(d["lag_arcsec"])
        acf = np.asarray(d["radial_acf"])
        acf_norm = acf / d["acf0"]
        ratio = rs[f"{key}_ratio_to_reference"]
        ratio_err = rs[f"{key}_ratio_err"]
        ax.plot(lag, acf_norm, "-o", color=color, markersize=3,
                label=f"{label} (L_corr={d['L_corr_arcsec']:.2f}\", ratio={ratio:.2f}+/-{ratio_err:.2f})")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvspan(L_ref - L_ref_std, L_ref + L_ref_std, color="gold", alpha=0.25,
               label=f"empirical pure-noise reference ({L_ref:.2f}+/-{L_ref_std:.2f}\")")
    ax.axvline(L_ref, color="goldenrod", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, L_ref * 5)
    ax.set_xlabel("Lag [arcsec]")
    ax.set_ylabel("Normalised azimuthally-averaged ACF")
    ax.legend(fontsize=10)
    ax.set_title(f"Residual autocorrelation vs. the empirical (not beam-FWHM) reference {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_inclination_scan(res: Results):
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5.5 * res.n_rings, 5))
    if res.n_rings == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        inc_grid = res.maps[f"ring{i}_inc_inc_grid_deg"]
        vr1_grid = res.maps[f"ring{i}_inc_vr1_grid_kms"]
        chi2_grid = res.maps[f"ring{i}_inc_chi2_grid"]
        chi2_min = np.min(chi2_grid)
        delta = chi2_grid - chi2_min

        X, Y = np.meshgrid(inc_grid, vr1_grid, indexing="ij")
        cs = ax.contourf(X, Y, delta, levels=[0, 1, 4, 9, delta.max() if delta.max() > 9 else 10],
                          cmap="viridis_r", alpha=0.85)
        ax.contour(X, Y, delta, levels=[1, 4, 9], colors="white", linewidths=0.8)

        inc0 = float(res.maps[f"ring{i}_inc0_deg"])
        ax.axvline(inc0, color="red", linestyle=":", linewidth=1.5, label="fiducial INC")
        d = res.diagnostics["per_ring"][f"ring{i}"]["inclination_scan"]
        ax.set_xlabel("Inclination [deg]")
        if i == 0:
            ax.set_ylabel("V_r1 [km/s]")
        ax.set_title(f"ring {i}: min V_r1={d['vr1_min_kms']:.1f} at INC{d['inc_offset_at_vr1_min_deg']:+.1f} deg")
        ax.legend(fontsize=9, loc="upper right")

    fig.colorbar(cs, ax=axes, label=r"$\Delta\chi^2$", fraction=0.02, pad=0.02)
    fig.suptitle(f"Inclination-V_r1 degeneracy (Section 3.1) {res.caption_tag()}")
    return fig


def fig_pa_scan_vr1(res: Results):
    """Section 3.3: V_r1(PA) should be flat if the m=2 signal is real."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(res.n_rings):
        pa_grid = res.maps[f"ring{i}_pa_grid_deg"]
        vr1 = res.maps[f"ring{i}_pa_vr1_kms"]
        ax.plot(pa_grid, vr1, color=RING_COLORS[i % len(RING_COLORS)], label=f"ring {i}")
    ax.set_xlabel("PA [deg]")
    ax.set_ylabel("V_r1 [km/s]")
    ax.legend(fontsize=10)
    ax.set_title(f"V_r1(PA) -- should be flat if the m=2 signal is real, not a PA artifact {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_centre_scan(res: Results):
    """Section 3.2: |s1_approaching - s1_receding| over the centre scan."""
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5.5 * res.n_rings, 5))
    if res.n_rings == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        offsets = res.maps[f"ring{i}_centre_offsets_pix"]
        diff_grid = res.maps[f"ring{i}_centre_diff_grid"]
        im = ax.imshow(diff_grid.T, origin="lower", cmap="viridis",
                        extent=[offsets[0], offsets[-1], offsets[0], offsets[-1]], aspect="equal")
        fig.colorbar(im, ax=ax, label="|s1_approaching - s1_receding| [km/s]", fraction=0.046)
        ax.axhline(0, color="white", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="white", linestyle=":", linewidth=0.8)
        d = res.diagnostics["per_ring"][f"ring{i}"]["centre_scan"]
        edge = " (AT EDGE)" if d["at_scan_edge"] else ""
        ax.set_title(f"ring {i}: min at ({d['best_dx_beams']:+.2f},{d['best_dy_beams']:+.2f}) beams{edge}")
        ax.set_xlabel("dx [pix]")
        if i == 0:
            ax.set_ylabel("dy [pix]")
    fig.suptitle(f"Centre scan (Section 3.2) {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

ALL_FIGURES = [
    ("fig_residual_ladder_maps", fig_residual_ladder_maps),
    ("fig_m2_component_map", fig_m2_component_map),
    ("fig_residual_histograms", fig_residual_histograms),
    ("fig_sector_analysis", fig_sector_analysis),
    ("fig_rtheta_grid", fig_rtheta_grid),
    ("fig_side_comparison", fig_side_comparison),
    ("fig_theta1_compass", fig_theta1_compass),
    ("fig_residual_autocorr", fig_residual_autocorr),
    ("fig_inclination_scan", fig_inclination_scan),
    ("fig_pa_scan_vr1", fig_pa_scan_vr1),
    ("fig_centre_scan", fig_centre_scan),
]


def main(results_dir="results", figures_dir="figures"):
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    res = load_results(results_dir)
    for name, fn in ALL_FIGURES:
        fig = fn(res)
        out = figures_dir / f"{name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[azimuthal_plots] Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render all figures from one TRM model's azimuthal-modulation results/."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir.parent / "figures"
    main(results_dir=results_dir, figures_dir=figures_dir)
