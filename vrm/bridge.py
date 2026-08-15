"""Bridges this project's geometry and data-quality conventions to the
vendored VRM/VRMA class in vrm/VRM_VRMA.py. Pure array construction and
result bookkeeping -- no FITS reading, no plotting -- so a notebook can call
it directly and stay thin.

VRM._matrix() returns only the fitted v_t, v_r values, in (ring, arc)
enumeration order, with no per-bin radius/angle and no per-bin point count.
This module reconstructs both, by algebraic identity with VRM's own
internals (see vrm/README.md), from the same pixel set actually fed to VRM
-- rather than re-deriving them independently and hoping they match.

Also implements the Figure-8-style analysis (VRMA maps, toy-model null test,
rank-correlation maps, velocity-dispersion correlation) from Sylos Labini,
De Marzo & Straccamore 2025 (ApJ 988, 122; arXiv:2503.22306). See
vrm/README.md for the interpretive choices made where that paper's Eq. 2-3
describe a single scalar correlation but the text and Figure 8 caption both
call the plotted quantity a spatial "map" -- the map used here is the
per-cell standardized-rank-product term whose sum (divided by N_cells-1)
recovers that scalar, which is the only reading consistent with both.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from scipy.stats import rankdata

import harmonic_fit as hf
from vrm.VRM_VRMA import VRM


def sky_aligned_offsets_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign):
    """PA-rotated but NOT-yet-inclination-deprojected sky-plane offsets, in
    arcsec -- exactly what VRM expects as (xp, yp), since its `phi0input` is
    never actually used (see vrm/README.md). Derived by inverting
    harmonic_fit.make_geometry's (R_pix, theta_rad) output rather than
    duplicating the PA/handedness logic, so this can never drift out of sync
    with the main pipeline's geometry: dx_rot = R*cos(theta),
    dy_rot = R*sin(theta)*cos(inc) (the inverse of make_geometry's
    dy_deproj = dy_rot/cos(inc), R = hypot(dx_rot, dy_deproj)).
    """
    R_pix, theta_rad = hf.make_geometry(mapset.shape, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    inc_rad = np.radians(inc_deg)
    dx_rot_pix = R_pix * np.cos(theta_rad)
    dy_rot_pix = R_pix * np.sin(theta_rad) * np.cos(inc_rad)
    return dx_rot_pix * mapset.pixscale_arcsec, dy_rot_pix * mapset.pixscale_arcsec


def good_pixel_mask(mapset, sigma_artifact_floor_kms):
    """Same data-quality reasoning as harmonic_fit's ring_mask, minus the
    ring-annulus restriction -- VRM bins radius itself, over whatever points
    it is given."""
    finite = np.isfinite(mapset.data_mom1) & np.isfinite(mapset.data_mom2)
    quality = hf.data_quality_mask(mapset.data_mom2, sigma_artifact_floor_kms)
    return finite & quality


def physical_R_theta_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign):
    """R (arcsec, unrescaled) and theta for the full frame -- algebraically
    identical to what VRM computes internally as R, theta (before VRM's own
    min/max rescaling), since it's the same make_geometry call used to build
    the (xp, yp) fed to VRM in the first place. theta is returned already
    wrapped to [0, 2*pi), matching VRM's internal convention (see
    vrm/README.md) -- make_geometry's atan2-based theta is (-pi, pi]."""
    R_pix, theta_rad = hf.make_geometry(mapset.shape, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    return R_pix * mapset.pixscale_arcsec, theta_rad % (2.0 * np.pi)


def _cell_indices(R_arcsec, theta_rad, number_rings, number_arch):
    """(ring_idx, arc_idx, boolean_mask) for every VRMA cell, in VRM's own
    enumeration order (ring outer loop, arc inner loop), replicating its
    rescaled-radius binning -- including its boundary convention -- exactly,
    so point counts and any field binned this way describe the same points
    VRM actually fit."""
    R_min, R_max = R_arcsec.min(), R_arcsec.max()
    R_rescaled = (R_arcsec - R_min) / (R_max - R_min)
    passo = 1.0 / number_rings
    passo2 = 2.0 * np.pi / number_arch

    cells = []
    for ring_idx, i in enumerate(np.arange(0, 1.0, passo)):
        for arc_idx, ang in enumerate(np.arange(0, 2.0 * np.pi, passo2)):
            in_bin = (R_rescaled <= i + passo) & (R_rescaled > i) & (theta_rad <= ang + passo2) & (theta_rad > ang)
            cells.append((ring_idx, arc_idx, in_bin))
    return cells


def _cell_index_per_point(cells, n_points_total):
    """Inverse of _cell_indices: for each point (in the same order as the
    masked arrays used to build `cells`), the flat cell index k it falls
    in (matching the row order of run_vrm's table), or -1 if it falls in
    none (possible at a bin boundary -- see VRM_VRMA.py's own edge
    convention in vrm/README.md)."""
    idx = np.full(n_points_total, -1, dtype=int)
    for k, (_, _, in_bin) in enumerate(cells):
        idx[in_bin] = k
    return idx


def build_vreal(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms, mom1_source=None):
    """Builds the (xp, yp, Vlos) array VRM expects. Vlos defaults to the RAW
    (not VSYS-subtracted) data mom1 -- VRM determines its own v_sys
    internally via _OFF(), used as an independent cross-check against the
    ringlog VSYS rather than assumed equal to it. Pass `mom1_source` (e.g.
    mapset.model_mom1) to run VRM on a different velocity field -- e.g. the
    toy/null model -- through the exact same mask and geometry."""
    dx_rot, dy_rot = sky_aligned_offsets_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    mask = good_pixel_mask(mapset, sigma_artifact_floor_kms)
    mom1 = mapset.data_mom1 if mom1_source is None else mom1_source
    Vreal = np.column_stack([dx_rot[mask], dy_rot[mask], mom1[mask]])
    return Vreal, mask


def run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
            number_rings, number_arch, mom1_source=None):
    """Runs the vendored VRM/VRMA fit and returns a tidy astropy Table (one
    row per (ring, arc) bin: r_arcsec_mean, theta_mean, n_points, v_t, v_r)
    plus VRM's independently-fit v_sys and the number of good pixels used.
    """
    Vreal, mask = build_vreal(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign,
                               sigma_artifact_floor_kms, mom1_source=mom1_source)
    vrm = VRM(Vreal, Iinput=inc_deg, phi0input=0.0,
              number_rings=number_rings, number_arch=number_arch, plot=False, save=False)

    vsys_fit = float(vrm._OFF())  # no side effects; _matrix() below repeats this internally before subtracting it
    predictions = vrm._matrix()

    n_bins = number_rings * number_arch
    v_t = predictions[0, 0:n_bins]
    v_r = predictions[0, n_bins:2 * n_bins]

    R_arcsec_full, theta_rad_full = physical_R_theta_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    R_arcsec, theta_rad = R_arcsec_full[mask], theta_rad_full[mask]
    cells = _cell_indices(R_arcsec, theta_rad, number_rings, number_arch)
    cell_index = _cell_index_per_point(cells, len(R_arcsec))

    rows = []
    for k, (ring_idx, arc_idx, in_bin) in enumerate(cells):
        n = int(np.sum(in_bin))
        rows.append(dict(
            ring_index=ring_idx,
            arc_index=arc_idx,
            r_arcsec_mean=float(np.mean(R_arcsec[in_bin])) if n else np.nan,
            theta_mean=float(np.mean(theta_rad[in_bin])) if n else np.nan,
            n_points=n,
            v_t=float(v_t[k]),
            v_r=float(v_r[k]),
        ))

    # Per-pixel (not per-cell-mean) plane-of-the-galaxy coordinates, and
    # which table row (cell) each pixel belongs to -- for plotting an actual
    # map (every pixel shown at its own position, colored by its cell's
    # fitted value) instead of one point per cell at the cell's mean position.
    x_arcsec = R_arcsec * np.cos(theta_rad)
    y_arcsec = R_arcsec * np.sin(theta_rad)

    return dict(table=Table(rows=rows), vsys_fit=vsys_fit, n_good_pixels=int(mask.sum()), mask=mask,
                x_arcsec=x_arcsec, y_arcsec=y_arcsec, cell_index=cell_index)


def bin_mean_field(values_full, mask, cells):
    """Mean of `values_full` (full-frame array) within each VRMA cell, using
    the same `mask` and `cells` (from _cell_indices) as a run_vrm call on the
    same geometry -- for binning an independent field (e.g. mom2) onto the
    identical grid, or for computing a fit residual cell-by-cell."""
    values_masked = values_full[mask]
    out = np.full(len(cells), np.nan)
    for k, (_, _, in_bin) in enumerate(cells):
        if np.any(in_bin):
            out[k] = np.mean(values_masked[in_bin])
    return out


def spearman_correlation_map(f_cells, g_cells):
    """Per-cell standardized-rank-product map (Eq. 3 of arXiv:2503.22306,
    Spearman version: ranks instead of raw values) between two per-cell
    fields, plus the overall Spearman correlation scalar (Eq. 2: the mean of
    the map over valid cells). NaN cells (no points) propagate as NaN and are
    excluded from the ranking and the overall scalar.

    This is a documented interpretive choice, not a verbatim formula from the
    paper: Eq. 2 as written gives a single scalar (a global mean/std over all
    cells), but the text and Figure 8 caption call the plotted quantity a
    spatial "map" r_fg(R,theta). The per-cell term is the only quantity in
    the paper's own equations whose sum, divided by N_cells-1, reproduces
    the scalar of Eq. 2 exactly while also varying by position -- see
    vrm/README.md.
    """
    f_cells = np.asarray(f_cells, dtype=float)
    g_cells = np.asarray(g_cells, dtype=float)
    valid = np.isfinite(f_cells) & np.isfinite(g_cells)

    rank_map = np.full(f_cells.shape, np.nan)
    if np.sum(valid) < 2:
        return rank_map, np.nan

    rf = rankdata(f_cells[valid])
    rg = rankdata(g_cells[valid])
    zf = (rf - rf.mean()) / rf.std(ddof=1)
    zg = (rg - rg.mean()) / rg.std(ddof=1)

    per_cell = zf * zg
    rank_map[valid] = per_cell
    overall = float(np.sum(per_cell) / (np.sum(valid) - 1))
    return rank_map, overall


def reconstructed_los_residual(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, mask, cells,
                                v_t_cells, v_r_cells, vsys_fit):
    """Observed v_los minus the VRMA-reconstructed v_los (Eq. 1, using each
    cell's own fitted v_t, v_r), binned to a per-cell mean residual map --
    the same diagnostic role as harmonic_plots.fig_residual_maps' post-fit
    panel, for the VRM fit instead of the harmonic fit."""
    R_arcsec_full, theta_rad_full = physical_R_theta_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    theta_rad = theta_rad_full[mask]
    v_los_obs = mapset.data_mom1[mask]

    inc_rad = np.radians(inc_deg)
    resid_full = np.full(theta_rad.shape, np.nan)
    for k, (_, _, in_bin) in enumerate(cells):
        model = vsys_fit + (v_t_cells[k] * np.cos(theta_rad[in_bin]) + v_r_cells[k] * np.sin(theta_rad[in_bin])) * np.sin(inc_rad)
        resid_full[in_bin] = v_los_obs[in_bin] - model

    out = np.full(len(cells), np.nan)
    for k, (_, _, in_bin) in enumerate(cells):
        if np.any(in_bin):
            out[k] = np.mean(resid_full[in_bin])
    return out


def run_figure8_analysis(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                          number_rings, number_arch, min_points=3):
    """Reproduces the ingredients of Figure 8 of Sylos Labini, De Marzo &
    Straccamore (2025): VRMA v_t, v_r, and their rank-correlation map for the
    observed galaxy and for a toy/null model, plus the velocity-dispersion
    map and its rank-correlation with v_t and v_r, plus a LOS residual map.

    The toy model here is Barolo's own `_local_1mom.fits` model map, not a
    resynthesized mock: this ringlog has no warp (PA, INC constant across
    all rings) and VRAD fixed at 0, which is exactly the paper's toy-model
    recipe (pure circular rotation from the TRM's i(R), phi0(R), v_c(R)) --
    see vrm/README.md.

    `min_points` (default 3, the minimum for a determined 2-parameter fit):
    cells with fewer points than this have their v_t, v_r set to NaN before
    anything downstream is computed. This matters because VRM._matrix()
    returns exactly 0.0 (not NaN) for an empty cell -- a real value that
    would otherwise silently enter the rank-correlation maps and the
    residual as if it were a genuine (and, worse, perfectly-fit) estimate.

    Returns a dict with a merged astropy Table (one row per cell) and the
    two overall Spearman scalars for r_vtvr (data) and r_vtvr (toy model).
    """
    data_result = run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                           number_rings, number_arch)
    toy_result = run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
                          number_rings, number_arch, mom1_source=mapset.model_mom1)

    mask = data_result["mask"]
    R_arcsec_full, theta_rad_full = physical_R_theta_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    cells = _cell_indices(R_arcsec_full[mask], theta_rad_full[mask], number_rings, number_arch)

    t = data_result["table"]
    n_points = np.asarray(t["n_points"])
    underdetermined = n_points < min_points

    v_t, v_r = np.asarray(t["v_t"], dtype=float), np.asarray(t["v_r"], dtype=float)
    v_t_tm = np.asarray(toy_result["table"]["v_t"], dtype=float)
    v_r_tm = np.asarray(toy_result["table"]["v_r"], dtype=float)
    v_t[underdetermined] = np.nan
    v_r[underdetermined] = np.nan
    v_t_tm[underdetermined] = np.nan
    v_r_tm[underdetermined] = np.nan
    t["v_t"], t["v_r"] = v_t, v_r

    sigma_cells = bin_mean_field(mapset.data_mom2, mask, cells)
    residual_cells = reconstructed_los_residual(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, mask, cells,
                                                 v_t, v_r, data_result["vsys_fit"])

    r_vtvr_map, C_data = spearman_correlation_map(v_t, v_r)
    r_vtvr_tm_map, C_toy = spearman_correlation_map(v_t_tm, v_r_tm)
    r_sigma_vt_map, C_sigma_vt = spearman_correlation_map(sigma_cells, v_t)
    r_sigma_vr_map, C_sigma_vr = spearman_correlation_map(sigma_cells, v_r)
    # Table 1 of arXiv:2503.22306: Spearman correlation between the data's
    # r_vtvr(R,theta) map and the toy model's r_tm_vtvr(R,theta) map, over
    # all cells -- C > 0.2 signals a warp signature.
    _, C_warp = spearman_correlation_map(r_vtvr_map, r_vtvr_tm_map)

    t["v_t_tm"] = v_t_tm
    t["v_r_tm"] = v_r_tm
    t["sigma_mean"] = sigma_cells
    t["residual"] = residual_cells
    t["r_vtvr"] = r_vtvr_map
    t["r_vtvr_tm"] = r_vtvr_tm_map
    t["r_sigma_vt"] = r_sigma_vt_map
    t["r_sigma_vr"] = r_sigma_vr_map

    return dict(
        table=t,
        vsys_fit=data_result["vsys_fit"],
        vsys_fit_toy=toy_result["vsys_fit"],
        n_good_pixels=data_result["n_good_pixels"],
        C_vtvr_data=C_data,
        C_vtvr_toy=C_toy,
        C_sigma_vt=C_sigma_vt,
        C_sigma_vr=C_sigma_vr,
        C_warp=C_warp,
        # Per-pixel plane-of-the-galaxy coordinates and each pixel's cell
        # index (same for data and toy branches: same mask, same geometry,
        # same cells -- only the fitted values differ) -- for plotting an
        # actual map instead of one point per cell at its mean position.
        x_arcsec=data_result["x_arcsec"],
        y_arcsec=data_result["y_arcsec"],
        cell_index=data_result["cell_index"],
    )
