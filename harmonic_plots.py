"""Figures for the first-order harmonic decomposition pipeline.

Reads only ``results/`` (written by ``harmonic_fit.py``) -- no FITS reading,
no fitting. One function per figure, each returning a ``matplotlib.Figure``.
``main()`` writes all of them to ``figures/``.

Every figure caption/title carries "RAD = ring center" and the primary
weighting scheme, per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import functools
import json
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
        self.table = Table.read(str(results_dir / "ring_results.ecsv"), format="ascii.ecsv")
        self.maps = dict(np.load(results_dir / "maps.npz"))
        self.scans = dict(np.load(results_dir / "scans.npz"))
        self.rad_convention = self.table.meta.get("rad_convention", "RAD = ring center")
        self.primary_weighting = self.table.meta["primary_weighting"]
        self.kpc_per_arcsec = self.table.meta["kpc_per_arcsec"]
        self.n_rings = int(np.max(self.table["ring_index"])) + 1

        residual_structure_path = results_dir / "residual_structure.json"
        self.residual_structure = json.loads(residual_structure_path.read_text()) if residual_structure_path.is_file() else None

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
# c0 toggle comparison -- fits with vs. without the c0 nuisance term
# --------------------------------------------------------------------------
#
# c0 absorbs both dVSYS/sin(i) and V_z/tan(i) without separating them
# (CLAUDE.md 2.5). "Toggling c0 off" means running harmonic_fit.main() a
# second time with "c0" dropped from Config.model_terms (e.g.
# model_terms=("s1",)) into a second results_dir -- this module never fits,
# it only compares the two already-written Results.


def fig_c0_toggle_residuals(res_with_c0: Results, res_without_c0: Results):
    """Post-fit residual maps (side=both, primary weighting), c0 free vs. c0
    fixed at 0. Under symmetric azimuthal coverage c0 and s1 are orthogonal,
    so fixing c0=0 should reappear almost entirely as a near-uniform,
    per-ring offset in the residual map (rather than biasing s1) -- this
    figure shows whether that holds ring by ring on the real, masked data."""
    post_with = _combined_ring_map(res_with_c0, "dv_postfit", combine="nan_fill")
    post_without = _combined_ring_map(res_without_c0, "dv_postfit", combine="nan_fill")

    all_vals = np.concatenate([post_with[np.isfinite(post_with)], post_without[np.isfinite(post_without)]])
    vmax = np.nanpercentile(np.abs(all_vals), 98) if len(all_vals) else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    im1 = ax1.imshow(post_with, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _overlay_ring_ellipses(res_with_c0, ax1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, label="km/s")
    ax1.set_title("Post-fit residual, c0 free")

    im2 = ax2.imshow(post_without, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _overlay_ring_ellipses(res_without_c0, ax2)
    fig.colorbar(im2, ax=ax2, fraction=0.046, label="km/s")
    ax2.set_title("Post-fit residual, c0 fixed at 0")

    c0_text = "   ".join(
        f"ring{i}: c0={res_with_c0.row(i)['c0']:.2f}±{res_with_c0.row(i)['c0_err']:.2f} km/s"
        for i in range(res_with_c0.n_rings)
    )
    fig.suptitle(
        f"c0 toggle: post-fit residual, c0 free vs. fixed at 0 {res_with_c0.caption_tag()}\n{c0_text}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def fig_c0_toggle_s1(res_with_c0: Results, res_without_c0: Results):
    """s1(R) and post-fit RMS residual, side=both, primary weighting, c0 free
    vs. fixed at 0. A shift in s1 beyond the bootstrap width is a direct
    measurement of L0 leakage (CLAUDE.md 5.6) from this ring's specific
    coverage asymmetry -- not expected under symmetric coverage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for res, off, color, label in (
        (res_with_c0, -0.1, "black", "c0 free"),
        (res_without_c0, 0.1, "crimson", "c0 = 0"),
    ):
        sub = res.rows(side="both", weighting=res.primary_weighting)
        sub = sub[np.argsort(sub["ring_index"])]
        yerr = np.vstack([sub["s1"] - sub["s1_boot_lo"], sub["s1_boot_hi"] - sub["s1"]])
        ax1.errorbar(sub["ring_index"] + off, sub["s1"], yerr=yerr, fmt="o", color=color,
                     markeredgecolor="black", capsize=4, label=label)

    ax1.axhline(0, color="gray", linestyle=":")
    ax1.set_xticks(range(res_with_c0.n_rings))
    ax1.set_xlabel("ring index")
    ax1.set_ylabel(r"$s_1$ [km s$^{-1}$]")
    ax1.legend()
    ax1.set_title("s1: c0 free vs. c0 = 0")

    sub_with = res_with_c0.rows(side="both", weighting=res_with_c0.primary_weighting)
    sub_with = sub_with[np.argsort(sub_with["ring_index"])]
    sub_without = res_without_c0.rows(side="both", weighting=res_without_c0.primary_weighting)
    sub_without = sub_without[np.argsort(sub_without["ring_index"])]
    ring_idx = np.asarray(sub_with["ring_index"], dtype=float)
    width = 0.35
    ax2.bar(ring_idx - width / 2, sub_with["rms_residual"], width=width, color="black", label="c0 free")
    ax2.bar(ring_idx + width / 2, sub_without["rms_residual"], width=width, color="crimson", label="c0 = 0")
    ax2.set_xticks(ring_idx)
    ax2.set_xlabel("ring index")
    ax2.set_ylabel("post-fit RMS residual [km s$^{-1}$]")
    ax2.legend()
    ax2.set_title("Post-fit RMS: c0 free vs. c0 = 0")

    fig.suptitle(f"c0 toggle comparison {res_with_c0.caption_tag()}")
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
# Azimuthal-modulation test figures (does the axisymmetric sin(theta) fit
# suppress a genuinely modulated V_rad? -- harmonic_fit.py's process_ring /
# vr1_theta1_from_c2s2 / inclination_scan_vr1 / residual_autocorrelation)
# --------------------------------------------------------------------------


def fig_m1_modulation(res: Results, n_wedge: int = AZIMUTHAL_N_WEDGE, min_abs_sin_theta: float = 0.15):
    """Test 1's headline figure: V_rad(theta) = s1 + V_r1*cos(theta-theta_1)
    (the modulation test's own fit, side="both", primary weighting) drawn
    as a curve over azimuth, with the binned per-sector empirical estimate
    overlaid: (weighted-mean dv_prefit in the wedge - c0) / sin(theta_wedge),
    using this ring's 4-term c0. A clean modulated flow should show a curve
    swinging well away from zero whose mean (s1) is near zero -- the
    suppression, shown directly.

    Wedges within min_abs_sin_theta of the major axis are dropped rather
    than shown with inflated error bars: V_rad has essentially no leverage
    there (same reasoning as this project's "do not fit V_rad in narrow
    sectors" rule -- this is binning for display, not a refit, but the
    same 1/sin(theta) blow-up applies to a per-wedge empirical estimate).
    """
    fig, axes = plt.subplots(1, res.n_rings, figsize=(6 * res.n_rings, 5.5), sharey=False)
    if res.n_rings == 1:
        axes = [axes]

    theta_edges = np.linspace(0, 360, n_wedge + 1)

    for i, ax in enumerate(axes):
        theta = res.maps[f"ring{i}_theta"]
        mask = res.maps[f"ring{i}_ring_mask_both"]
        w = res.maps[f"ring{i}_weights_primary"]
        dv_pre = res.maps[f"ring{i}_dv_prefit"]
        row = res.row(i)
        c0, s1, V_r1, theta_1_deg = row["c0"], row["s1"], row["V_r1"], row["theta_1"]
        rms_residual = row["rms_residual"]

        theta_deg = np.degrees(theta) % 360.0
        vrad_pts = np.full(n_wedge, np.nan)
        vrad_err = np.full(n_wedge, np.nan)
        theta_plot = np.full(n_wedge, np.nan)

        for j in range(n_wedge):
            wedge = mask & (theta_deg >= theta_edges[j]) & (theta_deg < theta_edges[j + 1])
            if not np.any(wedge):
                continue
            wv = w[wedge]
            wsum = np.sum(wv)
            if wsum <= 0:
                continue
            theta_w = np.degrees(np.arctan2(np.sum(wv * np.sin(theta[wedge])),
                                             np.sum(wv * np.cos(theta[wedge])))) % 360.0
            sin_tw = np.sin(np.radians(theta_w))
            if abs(sin_tw) < min_abs_sin_theta:
                continue
            dv_mean = np.sum(wv * dv_pre[wedge]) / wsum
            n_beams = max(np.sum(wedge) / row["pixels_per_beam"], 1.0)
            vrad_pts[j] = (dv_mean - c0) / sin_tw
            vrad_err[j] = (rms_residual / abs(sin_tw)) / np.sqrt(n_beams)
            theta_plot[j] = theta_w

        valid = np.isfinite(theta_plot)
        theta_c_shift = np.where(theta_plot < 90, theta_plot + 360, theta_plot)
        order = np.argsort(np.where(valid, theta_c_shift, np.inf))

        ax.errorbar(theta_c_shift[order], vrad_pts[order], yerr=vrad_err[order], fmt="o", color="black",
                    mfc="none", capsize=3, label=f"binned data ($|\\sin\\theta|$ > {min_abs_sin_theta})", zorder=5)

        theta_smooth_deg = np.linspace(0, 360, 400)
        theta_smooth_shift = np.where(theta_smooth_deg < 90, theta_smooth_deg + 360, theta_smooth_deg)
        order_s = np.argsort(theta_smooth_shift)
        vrad_model = s1 + V_r1 * np.cos(np.radians(theta_smooth_deg - theta_1_deg))
        ax.plot(theta_smooth_shift[order_s], vrad_model[order_s], "-", color=RING_COLORS[i % len(RING_COLORS)],
                 linewidth=2, label=r"$s_1+V_{r1}\cos(\theta-\theta_1)$", zorder=3)

        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(s1, color=RING_COLORS[i % len(RING_COLORS)], linestyle="--", linewidth=1, alpha=0.6,
                   label=f"$s_1$={s1:.1f} km/s")
        ax.axvspan(90, 270, facecolor="#F7F8FF", alpha=0.6, zorder=0)
        ax.axvspan(270, 450, facecolor="#FFF7F7", alpha=0.6, zorder=0)
        ax.set_xlim(90, 450)
        ticks = np.arange(90, 451, 90)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t if t <= 360 else t - 360}" for t in ticks])
        ax.set_xlabel(r"Azimuth [$^{\circ}$]")
        if i == 0:
            ax.set_ylabel(r"$V_{\rm rad}(\theta)$ [km s$^{-1}$]")
        ax.set_title(f"ring {i}: $V_{{r1}}$={V_r1:.1f} km/s, $\\theta_1$={theta_1_deg:.0f}$^\\circ$")
        ax.legend(fontsize=9, loc="upper right")
        ax.xaxis.set_minor_locator(AutoMinorLocator(3))

    fig.suptitle(f"Test 1: recovered V_rad(theta) modulation from c2, s2 {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_theta1_compass(res: Results, void_azimuth_deg=None, companion_azimuth_deg=None):
    """Polar compass of theta_1 (the m=2 signal's phase) per ring. Takes
    void_azimuth_deg / companion_azimuth_deg as plain floats rather than
    reading them from results/: computing the void's azimuth needs FITS/WCS
    access (timescales.py's void geometry), which this module deliberately
    never touches (CLAUDE.md: no astropy.io.fits in harmonic_plots.py) --
    the notebook computes them and passes them in, same pattern as the c0
    toggle figures taking two Results objects instead of reading a second
    tree itself."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    sub = res.rows(side="both", weighting=res.primary_weighting)
    sub = sub[np.argsort(sub["ring_index"])]
    rmax = float(np.max(sub["V_r1"])) * 1.15 if len(sub) else 1.0
    ax.set_rmax(rmax)

    for row in sub:
        th = np.radians(row["theta_1"])
        err = np.radians(row["theta_1_err"]) if np.isfinite(row["theta_1_err"]) else 0.0
        r = row["V_r1"]
        color = RING_COLORS[int(row["ring_index"]) % len(RING_COLORS)]
        ax.plot([th, th], [0, r], color=color, linewidth=2,
                label=f"ring {int(row['ring_index'])}: $\\theta_1$={row['theta_1']:.0f}$^\\circ$, "
                      f"$V_{{r1}}$={r:.0f} km/s")
        ax.plot(th, r, "o", color=color, markeredgecolor="black", zorder=5)
        if err > 0:
            ax.fill_betweenx([0, r], th - err, th + err, color=color, alpha=0.15)

    if void_azimuth_deg is not None:
        th_v = np.radians(void_azimuth_deg)
        ax.plot([th_v, th_v], [0, rmax], color="black", linestyle="--", linewidth=2, label="void azimuth")
    if companion_azimuth_deg is not None:
        th_c = np.radians(companion_azimuth_deg)
        ax.plot([th_c, th_c], [0, rmax], color="purple", linestyle=":", linewidth=2, label="companion direction")

    ax.set_title(f"theta_1 compass -- phase of the m=2 (V_r1) signal per ring {res.caption_tag()}", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=10)
    fig.tight_layout()
    return fig


def fig_residual_autocorr(res: Results):
    """Test 2.1: azimuthally averaged autocorrelation for the pre-fit and
    post-fit residual maps, with the beam FWHM marked. Pure (beam-
    correlated) noise gives L_corr/beam ~= 0.75, not 1:1 -- see
    harmonic_fit.residual_autocorrelation's docstring and selftest Test 11
    for why (a Gaussian beam's autocorrelation is itself Gaussian with
    std sigma*sqrt(2), so its HWHM is sqrt(2)/2 of the beam's FWHM, not
    equal to it). L_corr well above that reproducible baseline means
    spatially coherent structure the harmonic model hasn't captured."""
    if res.residual_structure is None:
        raise RuntimeError("fig_residual_autocorr: results/residual_structure.json not found -- "
                            "re-run harmonic_fit.main() (it writes this file) first.")
    rs = res.residual_structure
    beam_fwhm = rs["beam_fwhm_geomean_arcsec"]

    fig, ax = plt.subplots(figsize=(9, 7))
    for key, label, color in (("prefit", "pre-fit", "crimson"), ("postfit", "post-fit", "navy")):
        d = rs["autocorrelation"][key]
        lag = np.asarray(d["lag_arcsec"])
        acf = np.asarray(d["radial_acf"])
        acf_norm = acf / d["acf0"]
        ax.plot(lag, acf_norm, "-o", color=color, markersize=3,
                label=f"{label} (L_corr={d['L_corr_arcsec']:.2f}\", ratio to beam={d['L_corr_arcsec']/beam_fwhm:.2f})")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(beam_fwhm, color="black", linestyle="--", linewidth=1.5, label=f"beam FWHM = {beam_fwhm:.2f}\"")
    ax.set_xlim(0, beam_fwhm * 4)
    ax.set_xlabel("Lag [arcsec]")
    ax.set_ylabel("Normalised azimuthally-averaged ACF")
    ax.legend(fontsize=10)
    ax.set_title(f"Test 2.1: residual autocorrelation, pre-fit vs. post-fit {res.caption_tag()}")
    fig.tight_layout()
    return fig


def fig_inclination_scan(res: Results):
    """Test 1.3 discriminant: chi2(inc, V_r1) contours, same style as
    fig_pa_degeneracy. If V_r1 can be driven to zero within a plausible
    inclination error, the m=2 signal could be an inclination-error
    artifact (Schoenmakers, Franx & de Zeeuw 1997) rather than a modulated
    radial flow; if it cannot, it isn't."""
    fig, axes = plt.subplots(1, res.n_rings, figsize=(5.5 * res.n_rings, 5))
    if res.n_rings == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        inc_grid = res.scans[f"ring{i}_inc_inc_grid_deg"]
        vr1_grid = res.scans[f"ring{i}_inc_vr1_grid_kms"]
        chi2_grid = res.scans[f"ring{i}_inc_chi2_grid"]
        chi2_min = np.min(chi2_grid)
        delta = chi2_grid - chi2_min

        X, Y = np.meshgrid(inc_grid, vr1_grid, indexing="ij")
        cs = ax.contourf(X, Y, delta, levels=[0, 1, 4, 9, delta.max() if delta.max() > 9 else 10],
                          cmap="viridis_r", alpha=0.85)
        ax.contour(X, Y, delta, levels=[1, 4, 9], colors="white", linewidths=0.8)

        inc0 = float(res.scans[f"ring{i}_inc_inc0_deg"])
        ax.axvline(inc0, color="red", linestyle=":", linewidth=1.5, label="fiducial INC")
        ax.set_xlabel("Inclination [deg]")
        if i == 0:
            ax.set_ylabel(r"$V_{r1}$ [km s$^{-1}$]")
        ax.set_title(f"ring {i}")
        ax.legend(fontsize=9, loc="upper right")

    fig.colorbar(cs, ax=axes, label=r"$\Delta\chi^2$", fraction=0.02, pad=0.02)
    fig.suptitle(f"Inclination-$V_{{r1}}$ degeneracy, delta chi2 = 1,4,9 (rescaled to chi2_min/n_eff=1) {res.caption_tag()}")
    return fig


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

ALL_FIGURES = [
    ("fig_theta_map", fig_theta_map),
    ("fig_coverage", fig_coverage),
    ("fig_residual_maps", fig_residual_maps),
    ("fig_azimuthal_vlos", functools.partial(fig_azimuthal_vlos, n_wedge=AZIMUTHAL_N_WEDGE)),
    ("fig_m1_modulation", fig_m1_modulation),
    ("fig_residual_autocorr", fig_residual_autocorr),
    ("fig_inclination_scan", fig_inclination_scan),
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
    parser = argparse.ArgumentParser(
        description="Render all figures from one TRM model's results/ (see harmonic_fit.py --trm-dir)."
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="directory written by harmonic_fit.py, e.g. TRM_paper/results or fixed_PA42/results. Default: results",
    )
    parser.add_argument(
        "--figures-dir", default=None,
        help="where to write the .pdf figures (default: <results-dir>/../figures)",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir.parent / "figures"
    main(results_dir=results_dir, figures_dir=figures_dir)
