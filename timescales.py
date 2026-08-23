"""Epicyclic timescales for the observed HI void (and, if configured, the
tidal bridge) in the adopted TRM model.

Companion, additive analysis alongside harmonic_fit.py / harmonic_plots.py --
not one of the CLAUDE.md Section 0 deliverables and does not change any of
its rules. Reads:

  - the adopted TRM ringlog (rotation curve: RAD(Kpc), VROT, E_VROT1/2,
    INC(deg), P.A.(deg), XPOS(pix), YPOS(pix));
  - <trm_dir>/results/ring_results.ecsv, written by harmonic_fit.py, for the
    measured radial velocity s1(R) (side="both", primary weighting) and its
    bootstrap 16/84 percentiles;
  - the data moment-1 FITS header, for the WCS needed to place the void's
    RA/Dec corner coordinates in the same pixel frame as the ringlog's XPOS/
    YPOS.

All computation lives here -- no matplotlib import. See the generated
notebook for figures.

Method: Wallin & Struck-Marcell (1994), ApJ, 430, 121. Sec 3.1 gives the
perturbation-amplitude scaling used for the "kick" model's alpha (not a free
parameter -- see EpicyclicConfig.alpha). Sec 3.3.3 notes the perturbed hole
continues to expand after formation, which is why the void-derived
interaction timescale here is reported as an upper limit (see
void_interaction_timescale).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_dist

from harmonic_fit import deproject_pixel_offsets, find_ringlog, load_maps, read_ringlog

# 1 km/s/kpc = 1 / CONVERSION_MYR Myr^-1 (unit conversion, not a free science
# constant -- kept centralised so it never needs to be re-derived by hand).
CONVERSION_MYR = 977.8


# --------------------------------------------------------------------------
# Config -- every science/model constant lives here; nothing hardcoded
# elsewhere in this module.
# --------------------------------------------------------------------------


@dataclass
class Config:
    trm_dir: Path
    ringlog_path: Path
    results_dir: Path  # <trm_dir>/results -- source of ring_results.ecsv
    output_dir: Optional[Path] = None  # where timescales.json is written; default results_dir

    side: str = "both"

    # Epicyclic "kick" model (Section 3.2). alpha is fixed, not fitted --
    # Wallin & Struck-Marcell (1994) Sec 3.1 derive the perturbation scaling
    # (Delta v ~ q' omega' ~ (q/eps)^(-1/2)) rather than leaving it free; four
    # data points cannot constrain a third free parameter (see module
    # docstring / CLAUDE.md-style acceptance tests).
    alpha: float = 0.5
    r_ref_kpc: float = 15.0  # reference radius for the kick amplitude V0

    v0_grid_max_kms: float = 200.0
    v0_grid_step_kms: float = 1.0
    t_grid_max_myr: float = 400.0
    t_grid_step_myr: float = 1.0

    # Epoch range indicated by the other two clocks (T_phi, T_kappa/2), used
    # only to quote a V0 upper limit -- not fitted.
    t_estimate_lo_myr: float = 84.0
    t_estimate_hi_myr: float = 100.0
    upper_limit_cl: float = 0.68

    # Void geometry: RA/Dec corner coordinates are external inputs (read off
    # the moment maps / continuum image), not derivable from the ringlog --
    # same status as harmonic_plots.py's _EXTENT_BAND_ARCSEC.
    void_center_hms: tuple = ((17, 29, 9.588425), (-62, 26, 44.518200))
    void_top_hms: tuple = ((17, 29, 12.171038), (-62, 26, 27.349434))
    void_bottom_hms: tuple = ((17, 29, 11.630122), (-62, 27, 20.094748))

    # Legacy (superseded) void result, from the original notebook run with
    # PA=48, incl=51.76, scale=0.329 and a non-project-standard deprojection
    # -- kept only as a literal snapshot to print alongside the corrected
    # value (Section 2.1), never recomputed with a second deprojection.
    legacy_void_r_avg_kpc: float = 13.88
    legacy_void_dtheta_deg: float = 105.75

    # Bridge (Section 4). Not yet supplied: a real measurement of the
    # bridge's sky-plane extent / LOS velocity difference and its position
    # angle are required and are not present anywhere in this repo. Leave
    # None to skip Section 4's reported result (the machinery is still
    # exercised in the self-test against the worked example in the spec).
    bridge_dR_over_dV_kpc_per_kms: Optional[float] = None
    bridge_pa_deg: Optional[float] = None
    bridge_theta_sensitivity_step_deg: float = 5.0

    def __post_init__(self):
        self.trm_dir = Path(self.trm_dir)
        self.ringlog_path = Path(self.ringlog_path)
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else self.results_dir


# --------------------------------------------------------------------------
# Rotation curve spline (one definition, used everywhere)
# --------------------------------------------------------------------------


def generate_kinematic_spline(radii_kpc: np.ndarray, v_kms: np.ndarray) -> CubicSpline:
    """Natural cubic spline of the rotation curve. 'natural' boundary
    conditions enforce zero second derivative at the edges, preventing
    artificial oscillation (Runge's phenomenon)."""
    radii_kpc = np.asarray(radii_kpc, dtype=float)
    v_kms = np.asarray(v_kms, dtype=float)
    order = np.argsort(radii_kpc)
    return CubicSpline(radii_kpc[order], v_kms[order], bc_type="natural")


def warn_if_extrapolating(name: str, R_kpc, r_min: float, r_max: float):
    """CLAUDE.md-style guard (Section 2.2): CubicSpline extrapolates
    silently, and kappa depends on dv/dR, which is least trustworthy in the
    extrapolated region. Warn explicitly rather than letting it pass
    unnoticed."""
    R_arr = np.atleast_1d(np.asarray(R_kpc, dtype=float))
    below = R_arr[R_arr < r_min]
    above = R_arr[R_arr > r_max]
    if below.size:
        warnings.warn(
            f"{name}: evaluating spline {r_min - below.min():.3f} kpc below the "
            f"innermost knot ({r_min:.3f} kpc) -- dv/dR (and kappa) is extrapolated, not measured."
        )
    if above.size:
        warnings.warn(
            f"{name}: evaluating spline {above.max() - r_max:.3f} kpc beyond the "
            f"outermost knot ({r_max:.3f} kpc) -- dv/dR (and kappa) is extrapolated, not measured."
        )


def calculate_frequencies(R_kpc, v_spline: CubicSpline):
    """Omega (angular) and kappa (epicyclic) frequencies, in km/s/kpc.
    kappa^2 = 2*Omega^2 + 2*Omega*(dv/dR)."""
    R_kpc = np.asarray(R_kpc, dtype=float)
    dv_spline = v_spline.derivative()
    v_c = v_spline(R_kpc)
    dv_dR = dv_spline(R_kpc)
    Omega = v_c / R_kpc
    kappa = np.sqrt(2.0 * Omega**2 + 2.0 * Omega * dv_dR)
    return Omega, kappa


def calculate_timescales(Omega, kappa, conversion_myr: float = CONVERSION_MYR):
    """Converts frequencies (km/s/kpc) to periods in Myr."""
    T_phi = (2.0 * np.pi / Omega) * conversion_myr
    T_kappa = (2.0 * np.pi / kappa) * conversion_myr
    return T_phi, T_kappa


# --------------------------------------------------------------------------
# 1. Single source of truth for inputs
# --------------------------------------------------------------------------


@dataclass
class RotationCurve:
    radii_kpc: np.ndarray
    v_kms: np.ndarray
    v_err_lo: np.ndarray  # E_VROT1, absolute lower error (positive)
    v_err_hi: np.ndarray  # E_VROT2, absolute upper error (positive)
    inc_deg: float
    pa_deg: float
    xpos_pix: float
    ypos_pix: float
    kpc_per_arcsec: float
    spline: CubicSpline


def load_rotation_curve(ringlog_path: Path) -> RotationCurve:
    """Section 1.1: every ring parameter traces to the adopted ringlog --
    no numeric literal for radius, velocity, PA, inclination, or scale
    anywhere else in this module."""
    t = read_ringlog(ringlog_path)
    radii_kpc = np.asarray(t["r_center_kpc"], dtype=float)
    v_kms = np.asarray(t["VROT(km/s)"], dtype=float)
    # E_VROT1/2 are signed offsets from Barolo (can be negative for the
    # lower bound) -- Section 1.1 of this task calls them "absolute lower/
    # upper errors" for errorbar plotting, so take abs() here, once.
    v_err_lo = np.abs(np.asarray(t["E_VROT1"], dtype=float))
    v_err_hi = np.abs(np.asarray(t["E_VROT2"], dtype=float))

    inc_vals = np.asarray(t["INC(deg)"], dtype=float)
    pa_vals = np.asarray(t["P.A.(deg)"], dtype=float)
    xpos_vals = np.asarray(t["XPOS(pix)"], dtype=float)
    ypos_vals = np.asarray(t["YPOS(pix)"], dtype=float)
    if not (np.allclose(inc_vals, inc_vals[0]) and np.allclose(pa_vals, pa_vals[0])
            and np.allclose(xpos_vals, xpos_vals[0]) and np.allclose(ypos_vals, ypos_vals[0])):
        raise RuntimeError(
            "load_rotation_curve: INC/P.A./XPOS/YPOS vary across rings in this ringlog -- "
            "this module assumes a single fiducial geometry (true for TRM_paper and "
            "fixed_PA42, both warp-free); a warped model needs per-ring geometry threaded "
            "through the void deprojection instead of one fiducial row."
        )

    kpc_per_arcsec = float(t.meta["kpc_per_arcsec"])
    spline = generate_kinematic_spline(radii_kpc, v_kms)

    return RotationCurve(
        radii_kpc=radii_kpc,
        v_kms=v_kms,
        v_err_lo=v_err_lo,
        v_err_hi=v_err_hi,
        inc_deg=float(inc_vals[0]),
        pa_deg=float(pa_vals[0]),
        xpos_pix=float(xpos_vals[0]),
        ypos_pix=float(ypos_vals[0]),
        kpc_per_arcsec=kpc_per_arcsec,
        spline=spline,
    )


@dataclass
class RadialVelocities:
    ring_index: np.ndarray
    radii_kpc: np.ndarray
    v_R_outward_kms: np.ndarray  # sign-flipped s1, see load_radial_velocities
    sigma_kms: np.ndarray  # half the bootstrap 16-84 interval width
    s1_raw_kms: np.ndarray  # unflipped, as fit by harmonic_fit.py, for reference


def load_radial_velocities(results_dir: Path, side: str = "both") -> RadialVelocities:
    """Reads s1(R) and its bootstrap percentiles from ring_results.ecsv, for
    the given side and the primary weighting recorded in the table meta.

    Sign convention (Section 1.2, CLAUDE.md 2.6.1): harmonic_fit.py's s1
    keeps the raw fit sign; CLAUDE.md 2.6.1 resolves, for this galaxy (near
    side at theta=+90 deg), true outward V_rad = -s1. The epicyclic "kick"
    model here is written with V_R > 0 meaning outward (Wallin &
    Struck-Marcell 1994), so the sign is flipped explicitly on read.
    """
    t = Table.read(str(Path(results_dir) / "ring_results.ecsv"), format="ascii.ecsv")
    primary_weighting = t.meta["primary_weighting"]
    mask = (t["side"] == side) & (t["weighting"] == primary_weighting)
    sub = t[mask]
    if len(sub) == 0:
        raise RuntimeError(f"load_radial_velocities: no rows for side={side!r}, weighting={primary_weighting!r}")
    order = np.argsort(sub["ring_index"])
    sub = sub[order]

    s1 = np.asarray(sub["s1"], dtype=float)
    v_R_outward = -s1  # explicit sign flip -- see docstring
    sigma = (np.asarray(sub["s1_boot_hi"], dtype=float) - np.asarray(sub["s1_boot_lo"], dtype=float)) / 2.0

    return RadialVelocities(
        ring_index=np.asarray(sub["ring_index"], dtype=int),
        radii_kpc=np.asarray(sub["r_center_kpc"], dtype=float),
        v_R_outward_kms=v_R_outward,
        sigma_kms=sigma,
        s1_raw_kms=s1,
    )


# --------------------------------------------------------------------------
# 2. Void geometry
# --------------------------------------------------------------------------


def hms_to_deg(h, m, s):
    return (h + m / 60.0 + s / 3600.0) * 15.0


def dms_to_deg(d, m, s):
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def _radec_deg(hms_dms):
    (h, m, s), (d, dm, ds) = hms_dms
    return hms_to_deg(h, m, s), dms_to_deg(d, dm, ds)


def radec_to_pixel(header, ra_deg: float, dec_deg: float):
    """RA/Dec -> pixel position, in the same 0-based array-index convention
    as make_geometry's np.indices(shape) (origin=0 in astropy.wcs terms)."""
    wcs = WCS(header).celestial
    x_pix, y_pix = wcs.wcs_world2pix(ra_deg, dec_deg, 0)
    return float(x_pix), float(y_pix)


@dataclass
class VoidGeometry:
    r_avg_kpc: float
    dtheta_deg: float
    fraction_of_orbit: float
    v_c_void_kms: float
    T_phi_void_myr: float
    t_interaction_myr: float  # upper limit -- see Section 2.3 caveat
    legacy_r_avg_kpc: float
    legacy_dtheta_deg: float


def void_geometry(cfg: Config, rc: RotationCurve, header, cdelt1_sign: int) -> VoidGeometry:
    """Section 2.1: deprojects the void's top/bottom sky corners about the
    void's own reference point (its center, e.g. the AGN -- the same point
    the original notebook used) using harmonic_fit.deproject_pixel_offsets,
    the same rotation/handedness convention make_geometry uses internally,
    rather than a second, independently-defined deprojection. Only the two
    bugs CLAUDE.md-style Section 2.1 identifies are fixed here: the rotation
    handedness, and dx's sky direction (the notebook's own dx = (ra-ra_c)*
    cos(dec)*3600 increases east; the FITS grid's x increases west, since
    CDELT1 < 0) -- both now handled correctly by going through the WCS
    pixel position instead of a hand-rolled sky-plane formula. The void's
    reference point (not the disc's XPOS/YPOS) is kept unchanged: R and
    theta are being computed relative to the AGN as originally intended, and
    deproject_pixel_offsets does not care what point dx, dy are measured
    from (see acceptance test 3 in selftest()), only that the PA/inc
    rotation is applied correctly.

    Section 2.2: guards the resulting radius against silent spline
    extrapolation. Section 2.3: the reported interaction timescale is an
    upper limit (Wallin & Struck-Marcell 1994, Sec 3.3.3 -- the void
    continues to expand after formation, so a constant-Delta-theta
    assumption underestimates the true elapsed time)."""
    ra_c, dec_c = _radec_deg(cfg.void_center_hms)
    ra_t, dec_t = _radec_deg(cfg.void_top_hms)
    ra_b, dec_b = _radec_deg(cfg.void_bottom_hms)

    xc_void, yc_void = radec_to_pixel(header, ra_c, dec_c)
    x_t, y_t = radec_to_pixel(header, ra_t, dec_t)
    x_b, y_b = radec_to_pixel(header, ra_b, dec_b)
    dx_t, dy_t = x_t - xc_void, y_t - yc_void
    dx_b, dy_b = x_b - xc_void, y_b - yc_void

    R_pix_t, theta_t = deproject_pixel_offsets(dx_t, dy_t, rc.pa_deg, rc.inc_deg, cdelt1_sign)
    R_pix_b, theta_b = deproject_pixel_offsets(dx_b, dy_b, rc.pa_deg, rc.inc_deg, cdelt1_sign)

    th_diff = abs(float(theta_t) - float(theta_b))
    if th_diff > np.pi:
        th_diff = 2 * np.pi - th_diff
    dtheta_deg = np.degrees(th_diff)

    pixscale_arcsec = abs(header["CDELT1"]) * 3600.0
    R_avg_arcsec = ((float(R_pix_t) + float(R_pix_b)) / 2.0) * pixscale_arcsec
    R_avg_kpc = R_avg_arcsec * rc.kpc_per_arcsec

    warn_if_extrapolating("void R_avg", R_avg_kpc, rc.radii_kpc.min(), rc.radii_kpc.max())
    v_c_void = float(rc.spline(R_avg_kpc))
    Omega, _ = calculate_frequencies(np.array([R_avg_kpc]), rc.spline)
    T_phi_void = float((2.0 * np.pi / Omega[0]) * CONVERSION_MYR)

    fraction = dtheta_deg / 360.0
    t_interaction = T_phi_void * fraction

    return VoidGeometry(
        r_avg_kpc=R_avg_kpc,
        dtheta_deg=dtheta_deg,
        fraction_of_orbit=fraction,
        v_c_void_kms=v_c_void,
        T_phi_void_myr=T_phi_void,
        t_interaction_myr=t_interaction,
        legacy_r_avg_kpc=cfg.legacy_void_r_avg_kpc,
        legacy_dtheta_deg=cfg.legacy_void_dtheta_deg,
    )


# --------------------------------------------------------------------------
# 3. Epicyclic ("kick") grid search -- no optimiser
# --------------------------------------------------------------------------


def kick_model_v_R(R_kpc, t_myr, V0_kms, alpha, v_spline, r_ref_kpc, conversion_myr=CONVERSION_MYR):
    """Wallin & Struck-Marcell (1994)-style radial kick, damped/oscillating
    through the epicyclic frequency: V_R(R, t) = V0*(R/r_ref)^-alpha *
    sin(kappa(R)*t). V0 > 0, R_kpc scalar or array; t_myr scalar."""
    R_kpc = np.asarray(R_kpc, dtype=float)
    _, kappa = calculate_frequencies(R_kpc, v_spline)
    kappa_myr = kappa / conversion_myr
    v_kick = V0_kms * (R_kpc / r_ref_kpc) ** (-alpha)
    return v_kick * np.sin(kappa_myr * t_myr)


@dataclass
class EpicyclicGrid:
    V0_grid_kms: np.ndarray
    t_grid_myr: np.ndarray
    chi2: np.ndarray  # shape (n_V0, n_t)
    chi2_min: float
    V0_at_min: float
    t_at_min: float


def epicyclic_chi2_grid(cfg: Config, rv: RadialVelocities, rc: RotationCurve) -> EpicyclicGrid:
    """Section 3.2: grid, don't optimise. chi2(V0, t) = sum_i [(V_R_obs_i -
    model_i)^2 / sigma_i^2], evaluated on a 2D grid. Fully vectorised --
    no scipy.optimize call anywhere in this path."""
    warn_if_extrapolating("epicyclic grid (ring radii)", rv.radii_kpc, rc.radii_kpc.min(), rc.radii_kpc.max())

    V0_grid = np.arange(0.0, cfg.v0_grid_max_kms + 1e-9, cfg.v0_grid_step_kms)
    t_grid = np.arange(0.0, cfg.t_grid_max_myr + 1e-9, cfg.t_grid_step_myr)

    _, kappa = calculate_frequencies(rv.radii_kpc, rc.spline)
    kappa_myr = kappa / CONVERSION_MYR  # (n_rings,)
    R_ratio = (rv.radii_kpc / cfg.r_ref_kpc) ** (-cfg.alpha)  # (n_rings,)

    V0 = V0_grid[:, None, None]
    t = t_grid[None, :, None]
    model = V0 * R_ratio[None, None, :] * np.sin(kappa_myr[None, None, :] * t)  # (nV0, nt, n_rings)
    resid = (rv.v_R_outward_kms[None, None, :] - model) / rv.sigma_kms[None, None, :]
    chi2 = np.sum(resid**2, axis=2)  # (nV0, nt)

    i_min = np.unravel_index(np.argmin(chi2), chi2.shape)
    return EpicyclicGrid(
        V0_grid_kms=V0_grid,
        t_grid_myr=t_grid,
        chi2=chi2,
        chi2_min=float(chi2[i_min]),
        V0_at_min=float(V0_grid[i_min[0]]),
        t_at_min=float(t_grid[i_min[1]]),
    )


def v0_upper_limit(grid: EpicyclicGrid, t_lo_myr: float, t_hi_myr: float, cl: float = 0.68):
    """Section 3.2: 'quote an upper limit on V0'. At each t in
    [t_lo_myr, t_hi_myr], profile chi2 over V0 relative to chi2(V0=0, t) --
    the best fit sits at the V0=0 boundary (Section 3.1), so a one-parameter
    (1 dof) likelihood-ratio upper limit is well posed there even though a
    two-sided error bar is not. Returns the tightest (most conservative)
    upper limit over the t range, plus the full per-t array."""
    delta_chi2_crit = chi2_dist.ppf(cl, df=1)
    it_lo = np.searchsorted(grid.t_grid_myr, t_lo_myr)
    it_hi = np.searchsorted(grid.t_grid_myr, t_hi_myr, side="right")
    if it_hi <= it_lo:
        raise ValueError(f"v0_upper_limit: [{t_lo_myr}, {t_hi_myr}] Myr does not overlap the t grid")

    V0_ul = np.empty(it_hi - it_lo)
    t_used = grid.t_grid_myr[it_lo:it_hi]
    for j, it in enumerate(range(it_lo, it_hi)):
        profile = grid.chi2[:, it]
        delta = profile - profile[0]  # profile[0] is V0=0
        below = np.nonzero(delta <= delta_chi2_crit)[0]
        V0_ul[j] = grid.V0_grid_kms[below.max()] if below.size else 0.0

    j_tightest = np.argmin(V0_ul)
    return {
        "t_grid_myr": t_used,
        "V0_upper_limit_kms": V0_ul,
        "V0_upper_limit_tightest_kms": float(V0_ul[j_tightest]),
        "t_at_tightest_myr": float(t_used[j_tightest]),
        "cl": cl,
        "delta_chi2_crit": float(delta_chi2_crit),
    }


def t_kappa_half_per_ring(rc: RotationCurve):
    """Section 3.2: 'report T_kappa/2 per ring, not globally' -- a single t
    cannot place every ring at turnaround simultaneously."""
    Omega, kappa = calculate_frequencies(rc.radii_kpc, rc.spline)
    T_phi, T_kappa = calculate_timescales(Omega, kappa)
    return T_phi, T_kappa, T_kappa / 2.0


# --------------------------------------------------------------------------
# 4. Bridge timescale deprojection
# --------------------------------------------------------------------------


def bridge_timescale_myr(dR_over_dV_kpc_per_kms: float, theta_deg: float, conversion_myr: float = CONVERSION_MYR) -> float:
    """t_bridge = (dR_sky/dV_los) * cot(theta) * conversion_myr. A naive
    t = dR/dV*conversion_myr (no cot(theta) factor) implicitly assumes
    theta=45 deg, since cot(45 deg)=1; this makes the assumed line-of-sight
    angle explicit and correctable."""
    theta_rad = np.radians(theta_deg)
    return (dR_over_dV_kpc_per_kms / np.tan(theta_rad)) * conversion_myr


def bridge_timescale_sensitivity(theta_deg: float, dtheta_deg: float) -> float:
    """delta_t/t = delta_theta / (sin(theta)*cos(theta)) (delta_theta in
    radians for the ratio to be dimensionless)."""
    theta_rad = np.radians(theta_deg)
    return np.radians(dtheta_deg) / (np.sin(theta_rad) * np.cos(theta_rad))


def validate_bridge_orientation(bridge_pa_deg: Optional[float], disc_pa_deg: float):
    """A bridge along the disk normal (theta=i, Wallin & Struck-Marcell
    Table 1: orbital inclination 90 deg) must project onto the sky as the
    disc's projected minor axis, i.e. at disc_pa_deg + 90. Prints both
    angles so the theta=i assumption is testable, not just assumed; skips
    (rather than guesses) if no measured bridge PA is available."""
    expected_pa = (disc_pa_deg + 90.0) % 180.0
    if bridge_pa_deg is None:
        print(
            "[timescales] Bridge orientation validation SKIPPED: no measured bridge "
            f"position angle supplied (expected, if theta=i: {expected_pa:.2f} deg)."
        )
        return None
    observed_pa = bridge_pa_deg % 180.0
    print(
        f"[timescales] Bridge orientation validation: measured PA={observed_pa:.2f} deg "
        f"vs. expected (disc minor axis, theta=i) PA={expected_pa:.2f} deg "
        f"(delta={abs(observed_pa - expected_pa):.2f} deg)"
    )
    return observed_pa, expected_pa


# --------------------------------------------------------------------------
# Results assembly / I/O
# --------------------------------------------------------------------------


def run(cfg: Config) -> dict:
    ringlog = read_ringlog(cfg.ringlog_path)
    rc = load_rotation_curve(cfg.ringlog_path)
    rv = load_radial_velocities(cfg.results_dir, side=cfg.side)

    mapset = load_maps(cfg.trm_dir / "maps")
    cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)
    data_1mom = sorted((cfg.trm_dir / "maps").glob("*_1mom.fits"))
    data_1mom = [f for f in data_1mom if "_local_" not in f.name][0]
    header = fits.getheader(data_1mom)

    T_phi_ring, T_kappa_ring, T_kappa_half_ring = t_kappa_half_per_ring(rc)

    void = void_geometry(cfg, rc, header, cdelt1_sign)
    grid = epicyclic_chi2_grid(cfg, rv, rc)
    ul = v0_upper_limit(grid, cfg.t_estimate_lo_myr, cfg.t_estimate_hi_myr, cfg.upper_limit_cl)

    bridge = None
    if cfg.bridge_dR_over_dV_kpc_per_kms is not None:
        theta_deg = rc.inc_deg  # Section 4: theta = i (bridge along the disc normal)
        t_bridge = bridge_timescale_myr(cfg.bridge_dR_over_dV_kpc_per_kms, theta_deg)
        sensitivity = bridge_timescale_sensitivity(theta_deg, cfg.bridge_theta_sensitivity_step_deg)
        validation = validate_bridge_orientation(cfg.bridge_pa_deg, rc.pa_deg)
        bridge = {
            "theta_deg": theta_deg,
            "t_bridge_myr": t_bridge,
            "sensitivity_dlnt_per_step": sensitivity,
            "sensitivity_step_deg": cfg.bridge_theta_sensitivity_step_deg,
            "validation": validation,
        }
    else:
        print("[timescales] Section 4 (bridge) SKIPPED: bridge_dR_over_dV_kpc_per_kms not supplied in Config.")

    results = {
        "rotation_curve": {
            "radii_kpc": rc.radii_kpc.tolist(),
            "v_kms": rc.v_kms.tolist(),
            "v_err_lo_kms": rc.v_err_lo.tolist(),
            "v_err_hi_kms": rc.v_err_hi.tolist(),
            "inc_deg": rc.inc_deg,
            "pa_deg": rc.pa_deg,
            "kpc_per_arcsec": rc.kpc_per_arcsec,
        },
        "radial_velocities": {
            "ring_index": rv.ring_index.tolist(),
            "radii_kpc": rv.radii_kpc.tolist(),
            "s1_raw_kms": rv.s1_raw_kms.tolist(),
            "v_R_outward_kms": rv.v_R_outward_kms.tolist(),
            "sigma_kms": rv.sigma_kms.tolist(),
        },
        "per_ring_clocks": {
            "radii_kpc": rc.radii_kpc.tolist(),
            "T_phi_myr": T_phi_ring.tolist(),
            "T_kappa_myr": T_kappa_ring.tolist(),
            "T_kappa_half_myr": T_kappa_half_ring.tolist(),
        },
        "void": {
            "r_avg_kpc": void.r_avg_kpc,
            "dtheta_deg": void.dtheta_deg,
            "fraction_of_orbit": void.fraction_of_orbit,
            "v_c_void_kms": void.v_c_void_kms,
            "T_phi_void_myr": void.T_phi_void_myr,
            "t_interaction_myr_UPPER_LIMIT": void.t_interaction_myr,
            "caveat": "Wallin & Struck-Marcell (1994) Sec 3.3.3: the hole continues to "
                      "expand after formation; assuming constant dtheta makes this an "
                      "upper limit on the true interaction time, not a measurement.",
            "legacy_r_avg_kpc": void.legacy_r_avg_kpc,
            "legacy_dtheta_deg": void.legacy_dtheta_deg,
        },
        "epicyclic_grid": {
            "alpha_fixed": cfg.alpha,
            "alpha_citation": "Wallin & Struck-Marcell (1994) Sec 3.1",
            "chi2_min": grid.chi2_min,
            "V0_at_chi2_min_kms": grid.V0_at_min,
            "t_at_chi2_min_myr": grid.t_at_min,
            "note": "chi2 minimum sits at V0->0 with t unconstrained (Section 3.1); "
                    "not a measurement -- see V0 upper limit instead.",
        },
        "V0_upper_limit": {
            "t_range_myr": [cfg.t_estimate_lo_myr, cfg.t_estimate_hi_myr],
            "cl": ul["cl"],
            "V0_upper_limit_tightest_kms": ul["V0_upper_limit_tightest_kms"],
            "t_at_tightest_myr": ul["t_at_tightest_myr"],
        },
        "bridge": bridge,
    }
    return results


def write_results(cfg: Config, results: dict) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "timescales.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[timescales] Wrote {out_path}")
    return out_path


def main(cfg: Config) -> dict:
    results = run(cfg)

    print(f"[timescales] Rotation curve from {cfg.ringlog_path}")
    print(f"[timescales] kpc_per_arcsec = {results['rotation_curve']['kpc_per_arcsec']:.5f}")

    print("\n[timescales] Void geometry:")
    print(f"    R_avg   = {results['void']['r_avg_kpc']:.2f} kpc   (legacy: {results['void']['legacy_r_avg_kpc']:.2f} kpc)")
    print(f"    dtheta  = {results['void']['dtheta_deg']:.2f} deg  (legacy: {results['void']['legacy_dtheta_deg']:.2f} deg)")
    print(f"    t_interaction (UPPER LIMIT) = {results['void']['t_interaction_myr_UPPER_LIMIT']:.1f} Myr")

    print("\n[timescales] Per-ring clocks (T_phi, T_kappa, T_kappa/2) [Myr]:")
    for R, tphi, tk, tkh in zip(
        results["per_ring_clocks"]["radii_kpc"],
        results["per_ring_clocks"]["T_phi_myr"],
        results["per_ring_clocks"]["T_kappa_myr"],
        results["per_ring_clocks"]["T_kappa_half_myr"],
    ):
        print(f"    R={R:6.2f} kpc: T_phi={tphi:7.1f}  T_kappa={tk:7.1f}  T_kappa/2={tkh:7.1f}")

    print(f"\n[timescales] Epicyclic grid: chi2_min={results['epicyclic_grid']['chi2_min']:.2f} "
          f"at V0={results['epicyclic_grid']['V0_at_chi2_min_kms']:.1f} km/s, "
          f"t={results['epicyclic_grid']['t_at_chi2_min_myr']:.1f} Myr (boundary/degenerate -- not a measurement)")
    print(f"[timescales] V0 upper limit ({results['V0_upper_limit']['cl']:.0%} CL, "
          f"t in {results['V0_upper_limit']['t_range_myr']} Myr): "
          f"V0 <~ {results['V0_upper_limit']['V0_upper_limit_tightest_kms']:.1f} km/s "
          f"at t={results['V0_upper_limit']['t_at_tightest_myr']:.1f} Myr")

    write_results(cfg, results)
    return results


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------


def selftest():
    n_pass = 0
    n_fail = 0

    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        status = "PASS" if cond else "FAIL"
        print(f"[selftest] {name}: {status} {detail}")
        if cond:
            n_pass += 1
        else:
            n_fail += 1

    repo_root = Path(__file__).parent
    trm_dir = repo_root / "fixed_PA42"
    ringlog_path = find_ringlog(trm_dir)
    results_dir = trm_dir / "results"
    cfg = Config(trm_dir=trm_dir, ringlog_path=ringlog_path, results_dir=results_dir)

    rc = load_rotation_curve(ringlog_path)
    rv = load_radial_velocities(results_dir, side=cfg.side)

    # ---------------- Test: inputs trace to the ringlog, kpc_per_arcsec ----------------
    check("kpc_per_arcsec derived from ringlog == 0.33890 +/- 1e-4",
          abs(rc.kpc_per_arcsec - 0.33890) < 1e-4, f"(got {rc.kpc_per_arcsec:.5f})")
    check("Rotation curve radii/velocities match PA_fixed_to_42.txt (first ring)",
          np.isclose(rc.radii_kpc[0], 14.234, atol=1e-3) and np.isclose(rc.v_kms[0], 299.404, atol=1e-3),
          f"(R0={rc.radii_kpc[0]}, V0={rc.v_kms[0]})")

    # ---------------- Test: sign flip on read ----------------
    check("v_R_outward is the sign-flipped s1 (v_R_outward == -s1)",
          np.allclose(rv.v_R_outward_kms, -rv.s1_raw_kms))

    # ---------------- Test: void deprojection matches harmonic_fit.make_geometry ----------------
    # void_geometry() calls deproject_pixel_offsets directly (the same core
    # algebra make_geometry uses internally, see harmonic_fit.py). Confirm
    # the two agree to 1e-9 by evaluating both at the same arbitrary pixel.
    from harmonic_fit import make_geometry
    shape = (48, 48)
    px, py = 31, 18  # an arbitrary integer pixel, away from the center
    R_grid, th_grid = make_geometry(shape, rc.xpos_pix, rc.ypos_pix, rc.pa_deg, rc.inc_deg, -1)
    R_point, th_point = deproject_pixel_offsets(px - rc.xpos_pix, py - rc.ypos_pix, rc.pa_deg, rc.inc_deg, -1)
    check("void deprojection (deproject_pixel_offsets) agrees with make_geometry to 1e-9",
          abs(float(R_point) - R_grid[py, px]) < 1e-9 and abs(float(th_point) - th_grid[py, px]) < 1e-9)

    # ---------------- Test: extrapolation warning fires ----------------
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_extrapolating("test", np.array([1.0]), rc.radii_kpc.min(), rc.radii_kpc.max())
        fired = any("evaluating spline" in str(wi.message) for wi in w)
    check("extrapolation warning fires below the innermost knot", fired)

    # ---------------- Test: no optimiser -- grid chi2 matches independent scalar computation ----------------
    grid = epicyclic_chi2_grid(cfg, rv, rc)
    iv, it = 7, 33
    V0_test, t_test = grid.V0_grid_kms[iv], grid.t_grid_myr[it]
    manual = 0.0
    for R, obs, sig in zip(rv.radii_kpc, rv.v_R_outward_kms, rv.sigma_kms):
        model = kick_model_v_R(R, t_test, V0_test, cfg.alpha, rc.spline, cfg.r_ref_kpc)
        manual += ((obs - model) / sig) ** 2
    check("vectorised grid chi2 matches an independent scalar computation (1e-6)",
          abs(grid.chi2[iv, it] - manual) < 1e-6,
          f"(grid={grid.chi2[iv, it]:.8f}, manual={manual:.8f})")

    # ---------------- Test: reproduces the old (pegged) fit's RSS on the old fixture ----------------
    # Historical fixture only -- NOT used anywhere in production Config. This
    # is the exact data from the superseded notebook (hardcoded ring_radii/
    # ring_v_c, and the placeholder ring_v_R this task replaces), used solely
    # to confirm the new deterministic model function reproduces the old
    # (curve_fit-derived, pegged-at-bound) unweighted RSS -- i.e. that
    # removing the optimiser did not change the underlying model formula.
    legacy_radii = np.array([14.9, 18.2, 21.5, 24.8])
    legacy_v = np.array([297.223, 295.973, 304.49, 307.541])
    legacy_v_R = np.array([9.73993963, -0.81866769, -1.4475939, 13.23452213])
    legacy_spline = generate_kinematic_spline(legacy_radii, legacy_v)
    legacy_t, legacy_V0, legacy_alpha = 94.5, 10.0, 1.31
    legacy_model = kick_model_v_R(legacy_radii, legacy_t, legacy_V0, legacy_alpha, legacy_spline, cfg.r_ref_kpc)
    legacy_rss = float(np.sum((legacy_v_R - legacy_model) ** 2))
    check("grid model formula reproduces the old pegged-fit RSS (179.47, notebook printout)",
          abs(legacy_rss - 179.47) < 0.05, f"(got {legacy_rss:.2f})")

    # ---------------- Test: bridge worked example (91.7 -> 78.8 Myr) ----------------
    ratio = 91.7 / CONVERSION_MYR  # implied by the naive (theta=45 deg) t=91.7 Myr result
    t_naive = bridge_timescale_myr(ratio, 45.0)
    t_corrected = bridge_timescale_myr(ratio, 49.345)
    check("bridge worked example: naive (theta=45) reproduces 91.7 Myr",
          abs(t_naive - 91.7) < 0.05, f"(got {t_naive:.2f})")
    check("bridge worked example: corrected (theta=i=49.345) gives 78.8 Myr",
          abs(t_corrected - 78.8) < 0.1, f"(got {t_corrected:.2f})")
    sens = bridge_timescale_sensitivity(49.345, 5.0)
    check("bridge sensitivity: ~18% per 5 deg near i=49 deg",
          abs(sens - 0.18) < 0.01, f"(got {sens:.3f})")

    print(f"\n[selftest] {n_pass} passed, {n_fail} failed")
    return n_fail == 0


if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Epicyclic timescales for the observed HI void.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--trm-dir", default="fixed_PA42")
    args = parser.parse_args()

    if args.selftest:
        import sys
        ok = selftest()
        sys.exit(0 if ok else 1)

    trm_dir = repo_root / args.trm_dir
    cfg = Config(
        trm_dir=trm_dir,
        ringlog_path=find_ringlog(trm_dir),
        results_dir=trm_dir / "results",
    )
    main(cfg)
