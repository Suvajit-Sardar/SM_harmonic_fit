"""Figures for the first-order harmonic decomposition pipeline.

Reads only ``results/`` (written by ``harmonic_fit.py``) -- no FITS reading,
no fitting. One function per figure, each returning a ``matplotlib.Figure``.
``main()`` writes all of them to ``figures/``.

Every figure caption/title carries "RAD = ring center" and the primary
weighting scheme, per CLAUDE.md.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator

from harmonic_fit import pa_degeneracy_slope

# --------------------------------------------------------------------------
# Style (kept from the old script -- matches the rotation-curve figures)
# --------------------------------------------------------------------------

custom_rcparams = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.pad": 10,
    "ytick.major.pad": 2,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "figure.figsize": (10, 8),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.loc": "best",
    "axes.edgecolor": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "text.usetex": False,
}
mpl.rcParams.update(custom_rcparams)

# Kept from the old script: shaded optical/HI-extent band on fig_vrad_profile.
# These are external literature values for this target, not derivable from
# the ringlog -- not a "science constant" of the fit itself.
_EXTENT_BAND_ARCSEC = (73.2228 / 2.0, 148.0 / 2.0)

N_WEDGE = 24
# fig_azimuthal_vlos's actual resolution: at this dataset's beam (~20.7
# pixels/beam), 24 wedges draws several markers per independent beam within
# a ring (ring 0 has only 3.87 beams total). 12 roughly halves that
# oversampling without dropping below the angular resolution sin2 weighting
# needs near the minor axis to show the fit tracking the data.
AZIMUTHAL_N_WEDGE = 12
RING_COLORS = ["darkblue", "magenta", "green", "red"]
RING_MARKERS = ["o", "s", "^", "D"]
SIDE_COLORS = {"both": "green", "approaching": "blue", "receding": "red"}
SIDE_MARKERS = {"both": "o", "approaching": "s", "receding": "^"}


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


class Results:
    def __init__(self, results_dir: Path):
        results_dir = Path(results_dir)
        self.table = Table.read(results_dir / "ring_results.ecsv", format="ascii.ecsv")
        self.maps = dict(np.load(results_dir / "maps.npz"))
        self.scans = dict(np.load(results_dir / "scans.npz"))
        self.rad_convention = self.table.meta.get("rad_convention", "RAD = ring center")
        self.primary_weighting = self.table.meta["primary_weighting"]
        self.kpc_per_arcsec = self.table.meta["kpc_per_arcsec"]
        self.n_rings = int(np.max(self.table["ring_index"])) + 1

    def rows(self, ring_index=None, side=None, weighting=None):
        t = self.table
        mask = np.ones(len(t), dtype=bool)
        if ring_index is not None:
            mask &= t["ring_index"] == ring_index
        if side is not None:
            mask &= t["side"] == side
        if weighting is not None:
            mask &= t["weighting"] == weighting
        return t[mask]

    def row(self, ring_index, side="both", weighting=None):
        weighting = weighting or self.primary_weighting
        sub = self.rows(ring_index, side, weighting)
        if len(sub) != 1:
            raise RuntimeError(f"Expected 1 row for ring={ring_index}, side={side}, weighting={weighting}, got {len(sub)}")
        return sub[0]

    def caption_tag(self):
        return f"[{self.rad_convention}, primary weighting = {self.primary_weighting}]"


def load_results(results_dir) -> Results:
    return Results(results_dir)


def _ring_edges_arcsec(res: Results):
    sub = res.rows(side="both", weighting=res.primary_weighting)
    edges = sorted(set(sub["r_in_arcsec"]).union(sub["r_out_arcsec"]))
    return np.array(edges)


# --------------------------------------------------------------------------
# 6.1 Convention checks
# --------------------------------------------------------------------------


def _combined_ring_map(res: Results, key: str, combine="nan_fill"):
    """Merges a per-ring map into one full-frame array. Rings are disjoint
    annuli, so for `key`s that are zero outside their mask (e.g. weights,
    combine="sum") summing is exact; for `key`s that are NaN outside their
    mask (e.g. dv_prefit/dv_postfit, combine="nan_fill") later rings simply
    fill in the initial NaN wherever they have finite data."""
    shape = res.maps["ring0_theta"].shape
    if combine == "sum":
        out = np.zeros(shape)
        for i in range(res.n_rings):
            out += res.maps[f"ring{i}_{key}"]
        return out
    out = np.full(shape, np.nan)
    for i in range(res.n_rings):
        arr = res.maps[f"ring{i}_{key}"]
        m = np.isfinite(arr)
        out[m] = arr[m]
    return out


def _overlay_ring_ellipses(res: Results, ax, colors="black"):
    for i in range(res.n_rings):
        R = res.maps[f"ring{i}_R_arcsec"]
        row = res.row(i)
        ax.contour(R, levels=[row["r_in_arcsec"], row["r_out_arcsec"]], colors=colors,
                   linewidths=0.8, linestyles=":")


def fig_theta_map(res: Results):
    theta = res.maps["ring0_theta"]
    R_arcsec = res.maps["ring0_R_arcsec"]
    edges = _ring_edges_arcsec(res)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5))

    im = ax1.imshow(np.degrees(theta) % 360, origin="lower", cmap="twilight", vmin=0, vmax=360)
    ax1.contour(np.degrees(theta) % 360, levels=[0, 90, 180, 270], colors="black", linewidths=0.8, alpha=0.6)
    ax1.contour(R_arcsec, levels=edges, colors="gray", linewidths=0.8, linestyles=":")
    fig.colorbar(im, ax=ax1, label=r"$\theta$ [deg]")

    ny, nx = theta.shape
    yc, xc = ny / 2.0, nx / 2.0  # geometric center of the plotted frame, for axis arrows only
    # Find the pixel of theta closest to 0 near the outer ring to anchor the "receding" label.
    outer_mask = (R_arcsec > edges[-2]) & (R_arcsec < edges[-1])
    if np.any(outer_mask):
        idx = np.argmin(np.abs(theta[outer_mask]))
        yy, xx = np.nonzero(outer_mask)
        y0, x0 = yy[idx], xx[idx]
        ax1.annotate("receding", xy=(x0, y0), xytext=(x0 + 5, y0 + 5), color="white",
                     fontsize=13, arrowprops=dict(arrowstyle="->", color="white"))
    ax1.set_title(r"$\theta$ map")

    sign_map = np.where(np.isfinite(theta), np.sign(np.cos(theta)), np.nan)
    ax2.imshow(sign_map, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    ax2.contour(R_arcsec, levels=edges, colors="black", linewidths=0.8, linestyles=":")
    ax2.set_title("sign(cos theta): approaching (blue) / receding (red)")

    w_combined = _combined_ring_map(res, "weights_primary", combine="sum")
    w_plot = np.where(w_combined > 0, w_combined, np.nan)
    im3 = ax3.imshow(w_plot, origin="lower", cmap="viridis")
    _overlay_ring_ellipses(res, ax3, colors="white")
    fig.colorbar(im3, ax=ax3, fraction=0.046, label="w")
    ax3.set_title(f"w(theta) inside the mask, {res.primary_weighting} weighting")

    fig.suptitle(f"Convention check: theta, side, and weight definitions {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_coverage(res: Results):
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5 * res.n_rings, 5), subplot_kw=dict(projection="polar"))
    if res.n_rings == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        theta = res.maps[f"ring{i}_theta"]
        mask = res.maps[f"ring{i}_ring_mask_both"]
        w = res.maps[f"ring{i}_weights_primary"]
        th = theta[mask]
        ww = w[mask]

        n_bins = 24
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        sums, _ = np.histogram(th, bins=edges, weights=ww)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = 2 * np.pi / n_bins
        ax.bar(centers, sums, width=width, color=RING_COLORS[i % len(RING_COLORS)], alpha=0.7, edgecolor="black")

        row = res.row(i)
        ax.set_title(f"ring {i}\nL0={row['L0']:.3f}  L1={row['L1']:.3f}", fontsize=13)
    fig.suptitle(f"Azimuthal coverage of sum(w) {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 6.2 Pre-fit
# --------------------------------------------------------------------------


def fig_residual_maps(res: Results):
    """Pre-fit and post-fit residual, each as one full-frame map (all rings
    merged, disjoint annuli) rather than split ring by ring, with the ring
    boundaries overlaid as ellipses."""
    pre = _combined_ring_map(res, "dv_prefit", combine="nan_fill")
    post = _combined_ring_map(res, "dv_postfit", combine="nan_fill")

    all_vals = np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]])
    vmax = np.nanpercentile(np.abs(all_vals), 98) if len(all_vals) else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    im1 = ax1.imshow(pre, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _overlay_ring_ellipses(res, ax1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, label="km/s")
    ax1.set_title("Pre-fit: dv = (mom1-VSYS)/sin(i) - VROT*cos(theta)")

    im2 = ax2.imshow(post, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _overlay_ring_ellipses(res, ax2)
    fig.colorbar(im2, ax=ax2, fraction=0.046, label="km/s")
    ax2.set_title("Post-fit: dv - s1*sin(theta)")

    fig.suptitle(f"Residual maps, before and after the fit {res.caption_tag()}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 6.3 Data vs model, azimuthal
# --------------------------------------------------------------------------


def fig_azimuthal_vlos(res: Results, n_wedge: int = N_WEDGE):
    """Error bars are the ring-level residual RMS (from the fit, which has
    the cos(theta)/radial gradients already removed) divided by
    sqrt(beams-per-wedge), not the within-wedge pixel scatter -- at this
    dataset's beam size every wedge holds under one independent beam, so the
    within-wedge scatter measures the model gradient across the wedge, not
    measurement noise (see CLAUDE.md / the fig_azimuthal_vlos task notes)."""
    data_mom1 = res.maps["data_mom1"]
    model_mom1 = res.maps["model_mom1"]

    fig, axes = plt.subplots(1, res.n_rings, figsize=(6 * res.n_rings, 5.5), sharey=False)
    if res.n_rings == 1:
        axes = [axes]

    theta_edges = np.linspace(0, 360, n_wedge + 1)

    for i, ax in enumerate(axes):
        theta = res.maps[f"ring{i}_theta"]
        mask = res.maps[f"ring{i}_ring_mask_both"]
        w = res.maps[f"ring{i}_weights_primary"]
        row = res.row(i)
        ppb = row["pixels_per_beam"]
        sin_inc = np.sin(np.radians(row["inc_deg"]))
        rms_residual = row["rms_residual"]

        theta_deg = np.degrees(theta) % 360.0
        data_pts = np.full(n_wedge, np.nan)
        data_err = np.full(n_wedge, np.nan)
        model_pts = np.full(n_wedge, np.nan)
        theta_plot = np.full(n_wedge, np.nan)
        n_beams_wedge = np.full(n_wedge, np.nan)

        for j in range(n_wedge):
            wedge = mask & (theta_deg >= theta_edges[j]) & (theta_deg < theta_edges[j + 1])
            n_wedge_pix = int(np.sum(wedge))
            if n_wedge_pix == 0:
                continue
            wv = w[wedge]
            dv = data_mom1[wedge]
            mv = model_mom1[wedge]
            wsum = np.sum(wv)
            if wsum <= 0:
                continue
            wmean = np.sum(wv * dv) / wsum
            n_beams = max(n_wedge_pix / ppb, 1.0) if np.isfinite(ppb) else max(float(n_wedge_pix), 1.0)
            theta_w = np.degrees(np.arctan2(np.sum(wv * np.sin(theta[wedge])),
                                             np.sum(wv * np.cos(theta[wedge])))) % 360.0
            data_pts[j] = wmean
            data_err[j] = (rms_residual * sin_inc) / np.sqrt(n_beams)
            model_pts[j] = np.sum(wv * mv) / wsum
            theta_plot[j] = theta_w
            n_beams_wedge[j] = n_beams

        valid = np.isfinite(theta_plot)
        if np.sum(valid) >= 2:
            ratio = np.nanmax(data_err[valid]) / np.nanmin(data_err[valid])
        else:
            ratio = np.nan
        print(f"[fig_azimuthal_vlos] ring {i}: max/min error bar ratio = {ratio:.2f}")

        theta_c_shift = np.where(theta_plot < 90, theta_plot + 360, theta_plot)
        order = np.argsort(theta_c_shift)

        ax.errorbar(theta_c_shift[order], data_pts[order], yerr=data_err[order], fmt="o", color="black",
                    mfc="none", capsize=3, label="data (binned)", zorder=5)
        ax.plot(theta_c_shift[order], model_pts[order], "s--", color="gray", alpha=0.8, label="Barolo model (binned)", zorder=4)

        theta_smooth_deg = np.linspace(0, 360, 400)
        theta_smooth_shift = np.where(theta_smooth_deg < 90, theta_smooth_deg + 360, theta_smooth_deg)
        order_s = np.argsort(theta_smooth_shift)
        harmonic = row["vsys_kms"] + np.sin(np.radians(row["inc_deg"])) * (
            row["vrot_kms"] * np.cos(np.radians(theta_smooth_deg)) + row["s1"] * np.sin(np.radians(theta_smooth_deg))
        )
        ax.plot(theta_smooth_shift[order_s], harmonic[order_s], "-", color=RING_COLORS[i % len(RING_COLORS)],
                 linewidth=2, label="harmonic fit", zorder=3)

        ax.axvspan(90, 270, facecolor="#F7F8FF", alpha=0.6, zorder=0)
        ax.axvspan(270, 450, facecolor="#FFF7F7", alpha=0.6, zorder=0)
        ax.set_xlim(90, 450)
        ticks = np.arange(90, 451, 90)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t if t <= 360 else t - 360}" for t in ticks])
        ax.set_xlabel(r"Azimuth [$^{\circ}$]")
        if i == 0:
            ax.set_ylabel(r"$V_{\rm LOS}$ [km s$^{-1}$]")
        ax.set_title(f"ring {i}: {row['r_in_arcsec']:.1f}-{row['r_out_arcsec']:.1f}\"")
        ax.legend(fontsize=10, loc="lower right")

        median_beams = np.nanmedian(n_beams_wedge[np.isfinite(n_beams_wedge)]) if np.any(np.isfinite(n_beams_wedge)) else np.nan
        info_text = (f"n_beams(ring) = {row['n_eff']:.1f}\n"
                     f"n_wedges = {n_wedge}\n"
                     f"median n_beams/wedge = {median_beams:.2f}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

        ax.xaxis.set_minor_locator(AutoMinorLocator(3))

    fig.suptitle(f"Azimuthal V_LOS: data vs. Barolo model vs. harmonic fit {res.caption_tag()}\n"
                 f"error bars = ring residual RMS / sqrt(beams per wedge); adjacent points are correlated on the beam scale")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 6.4 Post-fit and systematics
# --------------------------------------------------------------------------


def _degeneracy_contour_figure(res: Results, kind: str):
    assert kind in ("pa", "vsys")
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5.5 * res.n_rings, 5))
    if res.n_rings == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        x_grid = res.scans[f"ring{i}_{kind}_{'pa_grid_deg' if kind=='pa' else 'vsys_grid_kms'}"]
        s1_grid = res.scans[f"ring{i}_{kind}_s1_grid_kms"]
        chi2_grid = res.scans[f"ring{i}_{kind}_chi2_grid"]
        chi2_min = np.min(chi2_grid)
        delta = chi2_grid - chi2_min

        X, Y = np.meshgrid(x_grid, s1_grid, indexing="ij")
        cs = ax.contourf(X, Y, delta, levels=[0, 1, 4, 9, delta.max() if delta.max() > 9 else 10],
                          cmap="viridis_r", alpha=0.85)
        ax.contour(X, Y, delta, levels=[1, 4, 9], colors="white", linewidths=0.8)

        if kind == "pa":
            row = res.row(i)
            theta = res.maps[f"ring{i}_theta"]
            mask = res.maps[f"ring{i}_ring_mask_both"]
            w = res.maps[f"ring{i}_weights_primary"]
            slope = pa_degeneracy_slope(theta[mask], w[mask], row["vrot_kms"], row["inc_deg"])
            pa0 = row["pa0_deg"]
            line_s1 = slope * np.radians(x_grid - pa0)
            ax.plot(x_grid, line_s1, "r--", linewidth=1.5, label="inclination-corrected analytic line")
            ax.set_xlabel("PA [deg]")
            ax.legend(fontsize=9, loc="upper right")
        else:
            ax.set_xlabel(r"$V_{\rm sys}$ [km s$^{-1}$]")

        if i == 0:
            ax.set_ylabel(r"$s_1$ [km s$^{-1}$]")
        ax.set_title(f"ring {i}")

    fig.colorbar(cs, ax=axes, label=r"$\Delta\chi^2$", fraction=0.02, pad=0.02)
    label = "PA" if kind == "pa" else "VSYS"
    fig.suptitle(f"{label}-s1 degeneracy, delta chi2 = 1,4,9 (rescaled to chi2_min/n_eff=1) {res.caption_tag()}")
    return fig


def fig_pa_degeneracy(res: Results):
    return _degeneracy_contour_figure(res, "pa")


def fig_vsys_degeneracy(res: Results):
    return _degeneracy_contour_figure(res, "vsys")


def fig_s1_vs_pa(res: Results):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(res.n_rings):
        pa_grid = res.scans[f"ring{i}_pa_pa_grid_deg"]
        s1_best = res.scans[f"ring{i}_pa_s1_best"]
        pa0 = res.scans[f"ring{i}_pa_pa0_deg"]
        row = res.row(i)
        ax.plot(pa_grid - pa0, s1_best, color=RING_COLORS[i % len(RING_COLORS)],
                 label=f"ring {i}: {row['r_in_arcsec']:.1f}-{row['r_out_arcsec']:.1f}\"")
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("PA offset from fiducial [deg]")
    ax.set_ylabel(r"$s_1$ [km s$^{-1}$]")
    ax.legend(fontsize=11)
    ax.set_title(f"s1 vs. PA offset -- do all rings null at the same PA? {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_bootstrap(res: Results):
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5 * res.n_rings, 4.5), sharey=False)
    if res.n_rings == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        row = res.row(i, side="both", weighting=res.primary_weighting)
        draws = res.scans[f"ring{i}_boot_s1_both_{res.primary_weighting}"]
        ax.hist(draws, bins=40, color=RING_COLORS[i % len(RING_COLORS)], alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.axvline(row["s1"], color="black", linewidth=2, label="point estimate")
        ax.axvline(row["s1_boot_lo"], color="black", linestyle="--", linewidth=1, label="16th/84th pct")
        ax.axvline(row["s1_boot_hi"], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(r"$s_1$ [km s$^{-1}$]")
        if i == 0:
            ax.set_ylabel("bootstrap draws")
            ax.legend(fontsize=9)
        ax.set_title(f"ring {i}")
    fig.suptitle(f"Bootstrap s1 distribution per ring, side=both, {res.primary_weighting} weighting {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_weighting_comparison(res: Results):
    fig, ax = plt.subplots(figsize=(9, 6))
    schemes = sorted(set(res.table["weighting"]))
    offsets = np.linspace(-0.2, 0.2, len(schemes))
    for off, scheme in zip(offsets, schemes):
        sub = res.rows(side="both", weighting=scheme)
        sub = sub[np.argsort(sub["ring_index"])]
        yerr = np.vstack([sub["s1"] - sub["s1_boot_lo"], sub["s1_boot_hi"] - sub["s1"]])
        ax.errorbar(sub["ring_index"] + off, sub["s1"], yerr=yerr, fmt="o", capsize=4, label=scheme)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(range(res.n_rings))
    ax.set_xlabel("ring index")
    ax.set_ylabel(r"$s_1$ [km s$^{-1}$]")
    ax.legend()
    ax.set_title(f"s1 across weighting schemes {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_vrad_profile(res: Results):
    fig, ax = plt.subplots(figsize=(10, 8))

    for side in ("approaching", "both", "receding"):
        sub = res.rows(side=side, weighting=res.primary_weighting)
        sub = sub[np.argsort(sub["ring_index"])]
        r = np.asarray(sub["r_center_kpc"]) / res.kpc_per_arcsec  # back to arcsec for the x-axis
        s1 = np.asarray(sub["s1"])
        stat_err = np.vstack([s1 - sub["s1_boot_lo"], sub["s1_boot_hi"] - s1])

        # Systematic error = half the total spread of s1(PA) over the PA scan
        # (+/- pa_scan_halfwidth_deg around fiducial), i.e. the s1 swing implied
        # by the assumed PA uncertainty. The PA scan itself is only run once per
        # ring, at side="both" (see harmonic_fit.process_ring); reused here for
        # all three side curves rather than tripling the scan cost.
        sys_err = np.empty(len(sub))
        for k, row in enumerate(sub):
            i = int(row["ring_index"])
            pa_grid = res.scans[f"ring{i}_pa_s1_best"]
            sys_err[k] = (np.max(pa_grid) - np.min(pa_grid)) / 2.0

        ax.errorbar(r, s1, yerr=stat_err, fmt=SIDE_MARKERS[side], color=SIDE_COLORS[side],
                    markeredgecolor="black", capsize=4, elinewidth=1.5, label=f"{side} (statistical)", zorder=5)
        ax.errorbar(r, s1, yerr=sys_err, fmt="none", ecolor=SIDE_COLORS[side], alpha=0.4,
                    elinewidth=6, capsize=0, zorder=2)

    ax.axvspan(*_EXTENT_BAND_ARCSEC, color="lightgray", alpha=0.5, zorder=0)
    for v in _EXTENT_BAND_ARCSEC:
        ax.axvline(v, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)

    ax.set_xlabel("Radius [arcsec]")
    ax.set_ylabel(r"$V_{\rm rad}$ ($s_1$) [km s$^{-1}$]")
    ax.legend(loc="lower left", fontsize=11)

    ax_top = ax.twiny()
    xlim_as = ax.get_xlim()
    ax_top.set_xlim(xlim_as[0] * res.kpc_per_arcsec, xlim_as[1] * res.kpc_per_arcsec)
    ax_top.set_xlabel("Radius [kpc]", labelpad=10)

    ax.minorticks_on()
    ax.set_title(
        f"V_rad(R), signed -- statistical (bootstrap) + systematic (PA-scan half-width) errors\n"
        f"near_side_assumed = UNRESOLVED (sign is not a physical inflow/outflow claim) {res.caption_tag()}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

ALL_FIGURES = [
    ("fig_theta_map", fig_theta_map),
    ("fig_coverage", fig_coverage),
    ("fig_residual_maps", fig_residual_maps),
    ("fig_azimuthal_vlos", functools.partial(fig_azimuthal_vlos, n_wedge=AZIMUTHAL_N_WEDGE)),
    ("fig_pa_degeneracy", fig_pa_degeneracy),
    ("fig_vsys_degeneracy", fig_vsys_degeneracy),
    ("fig_s1_vs_pa", fig_s1_vs_pa),
    ("fig_bootstrap", fig_bootstrap),
    ("fig_weighting_comparison", fig_weighting_comparison),
    ("fig_vrad_profile", fig_vrad_profile),
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
        print(f"[harmonic_plots] Wrote {out}")


if __name__ == "__main__":
    main()
