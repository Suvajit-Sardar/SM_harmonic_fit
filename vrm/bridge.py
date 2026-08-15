"""Bridges this project's geometry and data-quality conventions to the
vendored VRM/VRMA class in vrm/VRM_VRMA.py. Pure array construction and
result bookkeeping -- no FITS reading, no plotting -- so a notebook can call
it directly and stay thin.

VRM._matrix() returns only the fitted v_t, v_r values, in (ring, arc)
enumeration order, with no per-bin radius/angle and no per-bin point count.
This module reconstructs both, by algebraic identity with VRM's own
internals (see vrm/README.md), from the same pixel set actually fed to VRM
-- rather than re-deriving them independently and hoping they match.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

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
    the (xp, yp) fed to VRM in the first place."""
    R_pix, theta_rad = hf.make_geometry(mapset.shape, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    return R_pix * mapset.pixscale_arcsec, theta_rad


def build_vreal(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms):
    """Builds the (xp, yp, Vlos) array VRM expects. Vlos is the RAW (not
    VSYS-subtracted) data mom1 -- VRM determines its own v_sys internally via
    _OFF(), used here as an independent cross-check against the ringlog VSYS
    rather than assumed equal to it."""
    dx_rot, dy_rot = sky_aligned_offsets_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    mask = good_pixel_mask(mapset, sigma_artifact_floor_kms)
    Vreal = np.column_stack([dx_rot[mask], dy_rot[mask], mapset.data_mom1[mask]])
    return Vreal, mask


def run_vrm(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms,
            number_rings, number_arch):
    """Runs the vendored VRM/VRMA fit and returns a tidy astropy Table (one
    row per (ring, arc) bin: r_arcsec_mean, theta_mean, n_points, v_t, v_r)
    plus VRM's independently-fit v_sys and the number of good pixels used.
    """
    Vreal, mask = build_vreal(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign, sigma_artifact_floor_kms)
    vrm = VRM(Vreal, Iinput=inc_deg, phi0input=0.0,
              number_rings=number_rings, number_arch=number_arch, plot=False, save=False)

    vsys_fit = float(vrm._OFF())  # no side effects; _matrix() below repeats this internally before subtracting it
    predictions = vrm._matrix()

    n_bins = number_rings * number_arch
    v_t = predictions[0, 0:n_bins]
    v_r = predictions[0, n_bins:2 * n_bins]

    R_arcsec_full, theta_rad_full = physical_R_theta_arcsec(mapset, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    R_arcsec, theta_rad = R_arcsec_full[mask], theta_rad_full[mask]
    # make_geometry's theta is atan2-based, range (-pi, pi]; VRM's internal
    # theta (via arccos + quadrant correction) ranges [0, 2*pi) to match its
    # own binning loop below -- convert or every bin with negative atan2
    # theta silently reconstructs as empty even though VRM's own (correctly
    # ranged) theta put real points in it.
    theta_rad = theta_rad % (2.0 * np.pi)

    R_min, R_max = R_arcsec.min(), R_arcsec.max()
    R_rescaled = (R_arcsec - R_min) / (R_max - R_min)
    passo = 1.0 / number_rings
    passo2 = 2.0 * np.pi / number_arch

    rows = []
    k = 0
    for ring_idx, i in enumerate(np.arange(0, 1.0, passo)):
        for arc_idx, ang in enumerate(np.arange(0, 2.0 * np.pi, passo2)):
            in_bin = (R_rescaled <= i + passo) & (R_rescaled > i) & (theta_rad <= ang + passo2) & (theta_rad > ang)
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
            k += 1

    return dict(table=Table(rows=rows), vsys_fit=vsys_fit, n_good_pixels=int(mask.sum()), mask=mask)
