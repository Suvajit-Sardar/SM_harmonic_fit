"""Every dynamical/interaction timescale quoted in the paper, consolidated
into one module. Companion, additive analysis alongside harmonic_fit.py /
harmonic_plots.py -- not one of CLAUDE.md's Section 0 deliverables and does
not change any of its rules. Reads:

  - the adopted TRM ringlog (rotation curve: RAD(Kpc), VROT, E_VROT1/2,
    INC(deg), P.A.(deg), XPOS(pix), YPOS(pix));
  - <trm_dir>/results/ring_results.ecsv, written by harmonic_fit.py, for the
    measured radial velocity s1(R) (side="both", primary weighting), its
    bootstrap 16/84 percentiles, and (Section 2.8) its PA-scan chi2 curve
    from <trm_dir>/results/scans.npz;
  - the data moment-0/moment-1 FITS headers/data, for the WCS needed to
    place the void's RA/Dec corner coordinates (Section 2.3) and to measure
    the HI surface-density profile (Section 1).

All computation lives here -- no matplotlib import. See
scripts/build_timescales_notebook.py for figures; nothing may be quoted in
the manuscript that is not computed in this module.

Nothing here duplicates a deprojection or an NFW potential: geometry always
goes through harmonic_fit.make_geometry / deproject_pixel_offsets, and every
potential/enclosed-mass/escape-velocity evaluation goes through the ONE
galpy.potential.NFWPotential built by build_nfw_potential().

Method citations:
  - Wallin & Struck-Marcell (1994), ApJ, 430, 121. Sec 3.1: the epicyclic
    "kick" model's amplitude scaling (Section 3 below, used as the
    two-parameter alternative to the null in Section 2.8's exclusion test).
    Sec 3.3.3: the perturbed hole continues to expand after formation, so
    the void-derived interaction timescale (Section 2.3) is an upper limit.
    Sec 4.2 and Table 1: the eps=R_ring identification (Section 2.9) and the
    r_peri=12 kpc "half a disk radius" impact (Section 2.6).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_dist

from galpy.orbit import Orbit
from galpy.potential import NFWPotential
from galpy.potential import evaluatePotentials
from galpy.potential import vesc as galpy_vesc

from harmonic_fit import (
    KPC_PER_ARCSEC, deproject_pixel_offsets, find_ringlog, load_maps, make_geometry, pa_degeneracy_slope, read_ringlog,
)

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
    results_dir: Path  # <trm_dir>/results -- source of ring_results.ecsv/maps.npz/scans.npz
    output_dir: Optional[Path] = None  # where timescales.json is written; default results_dir

    side: str = "both"

    # Adopted physical scale (Planck18 cosmology at the source distance,
    # harmonic_fit.KPC_PER_ARCSEC by default) -- used for every kpc
    # conversion in this module (R_ring, void, ring turnaround, disk shear,
    # ...), deliberately NOT Barolo's own RAD(Kpc) column. See
    # harmonic_fit.KPC_PER_ARCSEC's docstring and CLAUDE.md Section 3.
    kpc_per_arcsec: float = KPC_PER_ARCSEC

    # Rotation-curve model: weighted least-squares polynomial (order=1,
    # linear, by default), NOT an interpolating spline -- see
    # generate_kinematic_curve's docstring.
    rotation_curve_poly_order: int = 1

    # Epicyclic "kick" model (Section 2.8's two-parameter alternative to the
    # V_rad=0 null). alpha is fixed, not fitted -- Wallin & Struck-Marcell
    # (1994) Sec 3.1 derive the perturbation scaling rather than leaving it
    # free; four rings cannot constrain a third free parameter.
    alpha: float = 0.5
    r_ref_kpc: float = 15.0

    v0_grid_max_kms: float = 200.0
    v0_grid_step_kms: float = 1.0
    t_grid_max_myr: float = 400.0
    t_grid_step_myr: float = 1.0

    # ------------------------------------------------------------------
    # Section 0: NFW halo (galpy.potential.NFWPotential)
    # ------------------------------------------------------------------
    M200_Msun: float = 1.9e12
    halo_concentration: float = 10.0
    halo_H0_kms_Mpc: float = 70.0
    halo_Om0: float = 0.3
    galpy_ro_kpc: float = 8.0
    galpy_vo_kms: float = 220.0

    # ------------------------------------------------------------------
    # Void geometry (Section 2.3): RA/Dec corner coordinates are external
    # inputs (read off the moment maps / continuum image), not derivable
    # from the ringlog -- same status as harmonic_plots.py's
    # _EXTENT_BAND_ARCSEC.
    # ------------------------------------------------------------------
    void_center_hms: tuple = ((17, 29, 9.588425), (-62, 26, 44.518200))
    void_top_hms: tuple = ((17, 29, 12.171038), (-62, 26, 27.349434))
    void_bottom_hms: tuple = ((17, 29, 11.630122), (-62, 27, 20.094748))
    legacy_void_r_avg_kpc: float = 13.88
    legacy_void_dtheta_deg: float = 105.75

    # ------------------------------------------------------------------
    # Section 2.4: disk shear. None -> default to the ringlog's own
    # innermost/outermost ring radius (never a free-floating literal).
    # ------------------------------------------------------------------
    shear_r_in_kpc: Optional[float] = None
    shear_r_out_kpc: Optional[float] = None

    # ------------------------------------------------------------------
    # Sections 2.5/2.6/2.7: the pair (companion). External measurements --
    # none of these are derivable from this repo's own data products (the
    # ringlog / moment maps cover only this galaxy, not its companion), so
    # every section below skips cleanly (printed, not guessed) while they
    # are None, rather than fabricating a placeholder.
    # ------------------------------------------------------------------
    vsys_table_kms: Optional[dict] = None    # {label: (value_kms, source_str)}
    vsys_pair_label_a: Optional[str] = None  # dV_sys = table[label_b] - table[label_a]
    vsys_pair_label_b: Optional[str] = None
    R_sep_kpc: Optional[float] = None
    bridge_pa_deg: Optional[float] = None
    bridge_theta_sensitivity_step_deg: float = 5.0

    r_peri_kpc: float = 12.0  # Wallin & Struck-Marcell (1994) Table 1: impact at half a disk radius
    r_peri_scan_kpc: tuple = (0.0, 6.0, 12.0)
    theta_scan_lo_deg: float = 40.0
    theta_scan_hi_deg: float = 85.0
    theta_scan_step_deg: float = 0.5

    debris_dR_kpc: Optional[float] = None
    debris_dV_kms: Optional[float] = None

    # ------------------------------------------------------------------
    # Section 2.9: Wallin & Struck-Marcell epoch conversion
    # ------------------------------------------------------------------
    wsm_eps_grid_kpc: tuple = (10.0, 15.0, 20.0, 25.0)
    wsm_epochs: tuple = (0.45, 0.56)

    def __post_init__(self):
        self.trm_dir = Path(self.trm_dir)
        self.ringlog_path = Path(self.ringlog_path)
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else self.results_dir


def _q(value, unit: str, type_: str, assumption: str) -> dict:
    """Section 3 (Output): the {value, unit, type, assumption} wrapper every
    quantity in timescales.json (and every row of the Table 2 summary) must
    use. type_ is one of "estimate" | "upper_limit" | "lower_limit" |
    "rejected" | "exclusion". "lower_limit" is for a constraint that bounds
    t from below (e.g. the pair-orbit boundedness limit, Section 2.6) --
    distinct from "upper_limit", which bounds t from above (e.g. the void
    interaction timescale, Section 2.3)."""
    assert type_ in ("estimate", "upper_limit", "lower_limit", "rejected", "exclusion"), type_
    return {"value": value, "unit": unit, "type": type_, "assumption": assumption}


# --------------------------------------------------------------------------
# Rotation curve fit (one definition, used everywhere)
# --------------------------------------------------------------------------


class RotationCurveFit:
    """Callable rotation-curve model wrapping a numpy Polynomial, exposing
    the same __call__/.derivative() interface scipy's CubicSpline has (so
    calculate_frequencies and everything downstream don't care which fit
    backs the curve)."""

    def __init__(self, poly: Polynomial):
        self._poly = poly

    def __call__(self, R_kpc):
        return self._poly(np.asarray(R_kpc, dtype=float))

    def derivative(self) -> "RotationCurveFit":
        return RotationCurveFit(self._poly.deriv())


def generate_kinematic_curve(radii_kpc, v_kms, v_err_lo=None, v_err_hi=None, order: int = 1) -> RotationCurveFit:
    """Weighted least-squares polynomial fit to the rotation curve --
    order=1 (linear) by default, NOT an interpolating spline. With only a
    handful of rings, forcing the curve exactly through every point turns
    ring-to-ring scatter that is well within each ring's own E_VROT
    uncertainty into spurious curvature. Weights are 1/sigma^2 with
    sigma = (v_err_lo + v_err_hi)/2 per ring when errors are supplied, else
    unweighted. order is clipped to len(radii)-1 so this never raises for a
    TRM model with fewer rings than the requested order.
    """
    radii_kpc = np.asarray(radii_kpc, dtype=float)
    v_kms = np.asarray(v_kms, dtype=float)
    order_idx = np.argsort(radii_kpc)
    radii_kpc, v_kms = radii_kpc[order_idx], v_kms[order_idx]

    weights = None
    if v_err_lo is not None and v_err_hi is not None:
        sigma = (np.asarray(v_err_lo, dtype=float)[order_idx] + np.asarray(v_err_hi, dtype=float)[order_idx]) / 2.0
        weights = 1.0 / sigma**2

    order_eff = min(order, len(radii_kpc) - 1)
    poly = Polynomial.fit(radii_kpc, v_kms, deg=order_eff, w=weights)
    return RotationCurveFit(poly)


def warn_if_extrapolating(name: str, R_kpc, r_min: float, r_max: float):
    """The rotation-curve fit extrapolates silently outside [r_min, r_max],
    and kappa depends on dv/dR, which is least trustworthy in the
    extrapolated region. Warn explicitly rather than letting it pass
    unnoticed."""
    R_arr = np.atleast_1d(np.asarray(R_kpc, dtype=float))
    below = R_arr[R_arr < r_min]
    above = R_arr[R_arr > r_max]
    if below.size:
        warnings.warn(
            f"{name}: evaluating the rotation-curve fit {r_min - below.min():.3f} kpc below the "
            f"innermost ring ({r_min:.3f} kpc) -- dv/dR (and kappa) is extrapolated, not measured."
        )
    if above.size:
        warnings.warn(
            f"{name}: evaluating the rotation-curve fit {above.max() - r_max:.3f} kpc beyond the "
            f"outermost ring ({r_max:.3f} kpc) -- dv/dR (and kappa) is extrapolated, not measured."
        )


def calculate_frequencies(R_kpc, v_curve: RotationCurveFit):
    """Omega (angular) and kappa (epicyclic) frequencies, in km/s/kpc.
    kappa^2 = 2*Omega^2 + 2*Omega*(dv/dR)."""
    R_kpc = np.asarray(R_kpc, dtype=float)
    dv_curve = v_curve.derivative()
    v_c = v_curve(R_kpc)
    dv_dR = dv_curve(R_kpc)
    Omega = v_c / R_kpc
    kappa = np.sqrt(2.0 * Omega**2 + 2.0 * Omega * dv_dR)
    return Omega, kappa


def calculate_timescales(Omega, kappa, conversion_myr: float = CONVERSION_MYR):
    """Converts frequencies (km/s/kpc) to periods in Myr. T_phi = 2*pi/Omega,
    T_kappa = 2*pi/kappa (Section 2.1)."""
    T_phi = (2.0 * np.pi / Omega) * conversion_myr
    T_kappa = (2.0 * np.pi / kappa) * conversion_myr
    return T_phi, T_kappa


# --------------------------------------------------------------------------
# Single source of truth for inputs: rotation curve, radial velocities
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
    curve: RotationCurveFit


def load_rotation_curve(ringlog_path: Path, order: int = 1, kpc_per_arcsec: float = KPC_PER_ARCSEC) -> RotationCurve:
    """Every ring parameter traces to the adopted ringlog -- no numeric
    literal for radius, velocity, PA, inclination, or scale anywhere else in
    this module. kpc_per_arcsec is the adopted Planck18-cosmology scale
    (harmonic_fit.KPC_PER_ARCSEC by default) -- see read_ringlog."""
    t = read_ringlog(ringlog_path, kpc_per_arcsec)
    radii_kpc = np.asarray(t["r_center_kpc"], dtype=float)
    v_kms = np.asarray(t["VROT(km/s)"], dtype=float)
    # E_VROT1/2 are signed offsets from Barolo (can be negative for the
    # lower bound) -- treat as absolute lower/upper errors for weighting and
    # errorbar plotting.
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
            "through the void/R_ring deprojection instead of one fiducial row."
        )

    kpc_per_arcsec = float(t.meta["kpc_per_arcsec"])
    curve = generate_kinematic_curve(radii_kpc, v_kms, v_err_lo, v_err_hi, order=order)

    return RotationCurve(
        radii_kpc=radii_kpc, v_kms=v_kms, v_err_lo=v_err_lo, v_err_hi=v_err_hi,
        inc_deg=float(inc_vals[0]), pa_deg=float(pa_vals[0]),
        xpos_pix=float(xpos_vals[0]), ypos_pix=float(ypos_vals[0]),
        kpc_per_arcsec=kpc_per_arcsec, curve=curve,
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

    Sign convention (CLAUDE.md 2.6.1): harmonic_fit.py's s1 keeps the raw
    fit sign; for this galaxy (near side at theta=+90 deg), true outward
    V_rad = -s1. The epicyclic model here is written with V_R > 0 meaning
    outward (Wallin & Struck-Marcell 1994), so the sign is flipped
    explicitly on read.
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
        v_R_outward_kms=v_R_outward, sigma_kms=sigma, s1_raw_kms=s1,
    )


# --------------------------------------------------------------------------
# 0. NFW halo potential (galpy) -- the ONE Phi/M/v_esc implementation
# --------------------------------------------------------------------------
#
# Never write a second phi(r)/M(r)/v_esc(r) NFW implementation anywhere in
# this module -- every potential evaluation downstream (Sections 2.6, 2.9's
# escape-velocity check) goes through build_nfw_potential()'s
# galpy.potential.NFWPotential.


def _halo_scale_quantities(cfg: Config):
    """Exact-precision M_s (mass scale) and a (scale radius) from M_200 and
    the concentration -- Section 0.1. Uses full float64 precision
    throughout; the spec's rounded printed values (delta_c=1.4888,
    M_s=1.276e12 Msun, a=25.55 kpc) are for the human reader only -- plugging
    rounded intermediates into the actual verification is exactly what would
    break the 1e-6 relative tolerance on Pot.mass(r_200) == M_200 (Section
    4, acceptance test 2): with full precision that identity is exact by
    construction (a = r_200/c, M_s = M_200/delta_c => M(r_200) = M_200
    algebraically), so a 1e-6 failure there would mean a real bug, not a
    rounding artifact.
    """
    c = cfg.halo_concentration
    M200 = cfg.M200_Msun * u.Msun
    delta_c = np.log(1.0 + c) - c / (1.0 + c)
    Ms = M200 / delta_c
    H0 = cfg.halo_H0_kms_Mpc * u.km / u.s / u.Mpc
    rho_crit = (3.0 * H0**2 / (8.0 * np.pi * G)).to(u.Msun / u.kpc**3)
    r200 = ((M200 / ((4.0 / 3.0) * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)).to(u.kpc)
    a = r200 / c
    return Ms, a, r200, delta_c


def build_nfw_potential(cfg: Config):
    """Section 0: the ONE NFW potential used everywhere in this module.
    Instantiated with astropy.units quantities and ro/vo set (Section 0.2),
    so every evaluatePotentials/.mass/vesc call downstream returns physical
    units directly -- no manual natural-unit conversion anywhere else."""
    Ms, a, r200, _ = _halo_scale_quantities(cfg)
    pot = NFWPotential(amp=Ms, a=a, ro=cfg.galpy_ro_kpc * u.kpc, vo=cfg.galpy_vo_kms * u.km / u.s)
    return pot, Ms, a, r200


def potential_kms2(pot, r_kpc) -> np.ndarray:
    """Phi(r) in (km/s)^2, physical units in and out (Section 0.2)."""
    r = np.atleast_1d(np.asarray(r_kpc, dtype=float)) * u.kpc
    phi = np.asarray(evaluatePotentials(pot, r, 0.0 * u.kpc)) * (u.km**2 / u.s**2)
    return phi.to(u.km**2 / u.s**2).value


def escape_velocity_kms(pot, r_kpc) -> np.ndarray:
    r = np.atleast_1d(np.asarray(r_kpc, dtype=float)) * u.kpc
    ve = np.asarray(galpy_vesc(pot, r)) * (u.km / u.s)
    return ve.to(u.km / u.s).value


def verify_nfw_potential(cfg: Config, r_grid_kpc=None) -> dict:
    """Section 0.1: the four assertions galpy's NFWPotential must satisfy
    before anything downstream trusts it. Raises RuntimeError with both
    values printed on any mismatch -- never proceeds silently."""
    if r_grid_kpc is None:
        r_grid_kpc = np.linspace(1.0, 200.0, 80)
    pot, Ms, a, r200 = build_nfw_potential(cfg)
    r = np.asarray(r_grid_kpc, dtype=float) * u.kpc

    # 1. Potential: Phi(r) = -(G*Ms/r)*ln(1+r/a)
    phi_g = potential_kms2(pot, r_grid_kpc) * (u.km**2 / u.s**2)
    phi_a = (-(G * Ms / r) * np.log(1.0 + r / a)).to(u.km**2 / u.s**2)
    rel1 = float(np.max(np.abs((phi_g - phi_a) / phi_a)).decompose())
    if rel1 > 1e-10:
        raise RuntimeError(f"NFW potential verification FAILED: galpy={phi_g}, analytic={phi_a}, max rel diff={rel1:.3e}")

    # 2. Enclosed mass: M(r) = Ms*(ln(1+r/a) - (r/a)/(1+r/a)); M(r_200) == M_200
    mass_g = np.asarray(pot.mass(r)) * u.Msun
    mass_a = Ms * (np.log(1.0 + r / a) - (r / a) / (1.0 + r / a))
    rel2 = float(np.max(np.abs((mass_g - mass_a) / mass_a)).decompose())
    if rel2 > 1e-10:
        raise RuntimeError(f"NFW enclosed-mass verification FAILED: galpy={mass_g}, analytic={mass_a}, max rel diff={rel2:.3e}")
    m200_check = float(pot.mass(r200)) * u.Msun
    M200_q = cfg.M200_Msun * u.Msun
    rel_m200 = float(abs((m200_check - M200_q) / M200_q))
    if rel_m200 > 1e-6:
        raise RuntimeError(f"NFW mass(r_200) != M_200: got {m200_check}, expected {M200_q}, rel diff={rel_m200:.3e}")

    # 3. Concentration constructor -- wrtcrit=True is essential (galpy
    # defaults to wrtcrit=False: a matter-density-referenced r_200 that gives
    # a substantially different answer; checked explicitly below).
    pot_c = NFWPotential(
        mvir=cfg.M200_Msun / 1e12, conc=cfg.halo_concentration, wrtcrit=True, overdens=200.0,
        H=cfg.halo_H0_kms_Mpc, Om=cfg.halo_Om0, ro=cfg.galpy_ro_kpc * u.kpc, vo=cfg.galpy_vo_kms * u.km / u.s,
    )
    # .a is stored in ro-scaled natural units regardless of the use_physical
    # setting (only evaluate*/mass/vesc function OUTPUTS respect it) --
    # multiply by ro explicitly to get physical kpc.
    a_c_kpc = float(pot_c.a) * cfg.galpy_ro_kpc
    r200_c_kpc = a_c_kpc * cfg.halo_concentration
    a_expected_kpc = a.to(u.kpc).value
    r200_expected_kpc = r200.to(u.kpc).value
    rel_a = abs(a_c_kpc - a_expected_kpc) / a_expected_kpc
    rel_r200 = abs(r200_c_kpc - r200_expected_kpc) / r200_expected_kpc
    if rel_a > 0.01 or rel_r200 > 0.01:
        raise RuntimeError(
            f"NFW concentration-constructor verification FAILED: a={a_c_kpc:.3f} kpc "
            f"(expected {a_expected_kpc:.3f}), r_200={r200_c_kpc:.3f} kpc (expected {r200_expected_kpc:.3f})"
        )
    pot_wrong = NFWPotential(
        mvir=cfg.M200_Msun / 1e12, conc=cfg.halo_concentration, wrtcrit=False, overdens=200.0,
        H=cfg.halo_H0_kms_Mpc, Om=cfg.halo_Om0, ro=cfg.galpy_ro_kpc * u.kpc, vo=cfg.galpy_vo_kms * u.km / u.s,
    )
    a_wrong_kpc = float(pot_wrong.a) * cfg.galpy_ro_kpc
    if abs(a_wrong_kpc - a_c_kpc) / a_c_kpc < 0.05:
        raise RuntimeError(
            f"NFW wrtcrit sanity check FAILED: wrtcrit=False gave a={a_wrong_kpc:.3f} kpc, barely "
            f"different from wrtcrit=True's {a_c_kpc:.3f} kpc -- expected a substantial difference "
            "(matter- vs critical-density-referenced r_200)."
        )

    # 4. Escape velocity: vesc(r) == sqrt(-2*Phi(r))
    vesc_g = escape_velocity_kms(pot, r_grid_kpc) * (u.km / u.s)
    vesc_a = np.sqrt(-2.0 * phi_a)
    rel4 = float(np.max(np.abs((vesc_g - vesc_a) / vesc_a)).decompose())
    if rel4 > 1e-6:
        raise RuntimeError(f"NFW escape-velocity verification FAILED: galpy vesc={vesc_g}, sqrt(-2*Phi)={vesc_a}, max rel diff={rel4:.3e}")

    result = {
        "Ms_Msun": Ms.to(u.Msun).value, "a_kpc": a.to(u.kpc).value, "r200_kpc": r200.to(u.kpc).value,
        "potential_max_rel_diff": rel1, "mass_max_rel_diff": rel2, "mass_r200_rel_diff": rel_m200,
        "concentration_a_rel_diff": rel_a, "concentration_r200_rel_diff": rel_r200,
        "wrtcrit_false_a_kpc": a_wrong_kpc, "vesc_max_rel_diff": rel4,
    }
    print(
        f"[timescales] NFW verification PASSED (Section 0.1, all 4 assertions): "
        f"M_s={result['Ms_Msun']:.6e} Msun, a={result['a_kpc']:.4f} kpc, r_200={result['r200_kpc']:.4f} kpc"
    )
    print(
        f"    potential rel diff={rel1:.2e}  mass rel diff={rel2:.2e}  mass(r200) rel diff={rel_m200:.2e}\n"
        f"    concentration-constructor: a rel diff={rel_a:.2e}, r200 rel diff={rel_r200:.2e} "
        f"(wrtcrit=False gives a={a_wrong_kpc:.2f} kpc instead -- confirms wrtcrit=True matters)\n"
        f"    vesc rel diff={rel4:.2e}"
    )
    return result


# --------------------------------------------------------------------------
# 1. R_ring: the surface-density peak, from the moment-0 map
# --------------------------------------------------------------------------


def load_data_mom0(maps_dir: Path):
    """The data moment-0 map (integrated intensity) -- unused by
    harmonic_fit.py (CLAUDE.md Section 1: "the 0mom pair ... is not used by
    this pipeline"), so loaded here directly rather than extending
    harmonic_fit.MapSet, which would touch that module's own frozen
    deliverable rules for no benefit to it."""
    maps_dir = Path(maps_dir)
    files = sorted(maps_dir.glob("*_0mom.fits"))
    files = [f for f in files if "_local_" not in f.name]
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one data _0mom.fits in {maps_dir}, found {len(files)}: {files}")
    data = np.squeeze(fits.getdata(files[0])).astype(float)
    header = fits.getheader(files[0])
    return data, header


@dataclass
class SigmaHIProfile:
    R_kpc: np.ndarray
    R_edges_kpc: np.ndarray
    sigma: np.ndarray        # inclination-corrected, azimuthally-averaged intensity per annulus
    sigma_err: np.ndarray    # standard error of the mean within each annulus
    n_pix: np.ndarray
    R_ring_kpc: float
    R_ring_arcsec: float
    R_ring_err_kpc: float    # half the annulus width, in kpc
    ring_index: int
    annulus_width_arcsec: float


def measure_R_ring(rc: RotationCurve, mom0_data: np.ndarray, header0, cdelt1_sign: int,
                    annulus_width_arcsec: float) -> SigmaHIProfile:
    """Section 1: the inclination-corrected, azimuthally-averaged Sigma_HI(R)
    profile in concentric deprojected annuli, using harmonic_fit.make_geometry
    -- the SAME deprojection harmonic_fit.py's ring fits use, not a second,
    independently-defined one -- evaluated on the moment-0 pixel grid with
    the disc's own fiducial geometry (rc.xpos_pix/ypos_pix/pa_deg/inc_deg).

    Sigma is inclination-corrected as Sigma_face_on = Sigma_observed *
    cos(i): a geometrically-thin, optically-thin disc's line-of-sight path
    length through the disc scales as 1/cos(i), so its observed
    (line-of-sight-integrated) column density is enhanced by that same
    factor relative to the face-on value; dividing by that factor removes it.

    Annulus width is the adopted TRM's own ring width (the same width
    read_ringlog derives for the harmonic fit's rings), so the radial
    binning resolution matches the rest of this project rather than
    introducing a second, arbitrary free parameter. R_ring's uncertainty is
    half that width."""
    R_pix, _theta = make_geometry(mom0_data.shape, rc.xpos_pix, rc.ypos_pix, rc.pa_deg, rc.inc_deg, cdelt1_sign)
    pixscale_arcsec = abs(header0["CDELT1"]) * 3600.0
    R_arcsec = R_pix * pixscale_arcsec

    finite = np.isfinite(mom0_data)
    if not np.any(finite):
        raise RuntimeError("measure_R_ring: moment-0 map has no finite pixels.")
    R_max_arcsec = float(np.max(R_arcsec[finite]))

    n_annuli = max(int(np.ceil(R_max_arcsec / annulus_width_arcsec)), 1)
    edges_arcsec = np.arange(n_annuli + 1) * annulus_width_arcsec
    centers_arcsec = 0.5 * (edges_arcsec[:-1] + edges_arcsec[1:])

    cos_i = np.cos(np.radians(rc.inc_deg))
    sigma = np.full(n_annuli, np.nan)
    sigma_err = np.full(n_annuli, np.nan)
    n_pix = np.zeros(n_annuli, dtype=int)
    for i in range(n_annuli):
        m = finite & (R_arcsec >= edges_arcsec[i]) & (R_arcsec < edges_arcsec[i + 1])
        n_pix[i] = int(np.sum(m))
        if n_pix[i] == 0:
            continue
        vals = mom0_data[m] * cos_i
        sigma[i] = float(np.mean(vals))
        if n_pix[i] > 1:
            sigma_err[i] = float(np.std(vals, ddof=1) / np.sqrt(n_pix[i]))

    valid = np.isfinite(sigma)
    if not np.any(valid):
        raise RuntimeError("measure_R_ring: no annulus has finite Sigma_HI -- check the moment-0 map/geometry.")
    i_peak = int(np.argmax(np.where(valid, sigma, -np.inf)))

    R_ring_arcsec = float(centers_arcsec[i_peak])
    R_ring_kpc = R_ring_arcsec * rc.kpc_per_arcsec
    R_ring_err_kpc = (annulus_width_arcsec / 2.0) * rc.kpc_per_arcsec

    return SigmaHIProfile(
        R_kpc=centers_arcsec * rc.kpc_per_arcsec, R_edges_kpc=edges_arcsec * rc.kpc_per_arcsec,
        sigma=sigma, sigma_err=sigma_err, n_pix=n_pix,
        R_ring_kpc=R_ring_kpc, R_ring_arcsec=R_ring_arcsec, R_ring_err_kpc=R_ring_err_kpc,
        ring_index=i_peak, annulus_width_arcsec=annulus_width_arcsec,
    )


# --------------------------------------------------------------------------
# 2.1 Epicyclic and orbital periods, per ring (already implemented)
# --------------------------------------------------------------------------


def t_kappa_half_per_ring(rc: RotationCurve):
    """T_phi, T_kappa, T_kappa/2 per ring -- NOT a single global T_kappa/2:
    it varies by up to 50% ring to ring, so a single number for the whole
    ring is never reported."""
    Omega, kappa = calculate_frequencies(rc.radii_kpc, rc.curve)
    T_phi, T_kappa = calculate_timescales(Omega, kappa)
    dv_dR = rc.curve.derivative()(rc.radii_kpc)
    return T_phi, T_kappa, T_kappa / 2.0, dv_dR


# --------------------------------------------------------------------------
# 2.2 Ring turnaround (new): t = T_kappa(R_ring)/2
# --------------------------------------------------------------------------


def ring_turnaround_timescale(rc: RotationCurve, R_ring_kpc: float, R_ring_err_kpc: float):
    """t = T_kappa(R_ring)/2. Uncertainty propagated from R_ring_err_kpc by
    evaluating T_kappa at R_ring +/- its error and taking half the resulting
    spread. A dissipative gas ring lags the collisionless (stellar) caustic
    that T_kappa/2 formally predicts, so this is a LOWER LIMIT relative to
    the stellar-population turnaround time, not an exact match to it."""
    warn_if_extrapolating("ring turnaround (R_ring)", np.array([R_ring_kpc]), rc.radii_kpc.min(), rc.radii_kpc.max())
    Omega0, kappa0 = calculate_frequencies(np.array([R_ring_kpc]), rc.curve)
    _, T_kappa0 = calculate_timescales(Omega0, kappa0)
    t0 = float(T_kappa0[0]) / 2.0

    R_lo = max(R_ring_kpc - R_ring_err_kpc, 1e-6)
    R_hi = R_ring_kpc + R_ring_err_kpc
    _, kappa_lo = calculate_frequencies(np.array([R_lo]), rc.curve)
    _, kappa_hi = calculate_frequencies(np.array([R_hi]), rc.curve)
    Omega_lo, _ = calculate_frequencies(np.array([R_lo]), rc.curve)
    Omega_hi, _ = calculate_frequencies(np.array([R_hi]), rc.curve)
    _, T_kappa_lo = calculate_timescales(Omega_lo, kappa_lo)
    _, T_kappa_hi = calculate_timescales(Omega_hi, kappa_hi)
    t_err = abs(float(T_kappa_hi[0]) - float(T_kappa_lo[0])) / 4.0

    return t0, t_err


# --------------------------------------------------------------------------
# 2.3 Crescent void (already implemented)
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
    """Deprojects the void's top/bottom sky corners about the void's own
    reference point (its center) using harmonic_fit.deproject_pixel_offsets
    -- the same rotation/handedness convention make_geometry uses
    internally, not a second, independently-defined deprojection.

    T_phi = 2*pi*R_void/V_c, t = T_phi*(dtheta/360). Wallin & Struck-Marcell
    (1994) Sec 3.3.3: the void continues to expand after formation, so a
    constant-dtheta assumption makes this an UPPER LIMIT, not a
    measurement."""
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
    v_c_void = float(rc.curve(R_avg_kpc))
    Omega, _ = calculate_frequencies(np.array([R_avg_kpc]), rc.curve)
    T_phi_void = float((2.0 * np.pi / Omega[0]) * CONVERSION_MYR)

    fraction = dtheta_deg / 360.0
    t_interaction = T_phi_void * fraction

    return VoidGeometry(
        r_avg_kpc=R_avg_kpc, dtheta_deg=dtheta_deg, fraction_of_orbit=fraction,
        v_c_void_kms=v_c_void, T_phi_void_myr=T_phi_void, t_interaction_myr=t_interaction,
        legacy_r_avg_kpc=cfg.legacy_void_r_avg_kpc, legacy_dtheta_deg=cfg.legacy_void_dtheta_deg,
    )


def void_interaction_nearest_vc(rc: RotationCurve, void: VoidGeometry) -> dict:
    """Section 2.3 systematic: the interaction timescale obtained using the
    nearest MEASURED ring's V_c (not the fitted/extrapolated rotation-curve
    polynomial) at the void's radius -- reported alongside the fitted-curve
    value (VoidGeometry.t_interaction_myr) as a systematic, not a second
    central estimate."""
    i_nearest = int(np.argmin(np.abs(rc.radii_kpc - void.r_avg_kpc)))
    v_nearest = float(rc.v_kms[i_nearest])
    R_nearest = float(rc.radii_kpc[i_nearest])
    T_phi_nearest = (2.0 * np.pi * void.r_avg_kpc / v_nearest) * CONVERSION_MYR
    t_nearest = T_phi_nearest * void.fraction_of_orbit
    return {
        "ring_index": i_nearest, "R_ring_kpc": R_nearest, "v_c_kms": v_nearest,
        "T_phi_myr": T_phi_nearest, "t_interaction_myr": t_nearest,
    }


def void_azimuth_in_disk_frame(cfg: Config, rc: RotationCurve, header, cdelt1_sign: int) -> float:
    """harmonic_fit.py-style azimuthal-modulation cross-check: the void's
    mean azimuth in the SAME theta frame harmonic_fit.py's ring fits use --
    relative to the disc's dynamical center (rc.xpos_pix, rc.ypos_pix), NOT
    the void-centered frame void_geometry() uses for R_avg/dtheta. Returns
    degrees in [0, 360): the circular mean of the void's top/bottom corner
    azimuths."""
    ra_t, dec_t = _radec_deg(cfg.void_top_hms)
    ra_b, dec_b = _radec_deg(cfg.void_bottom_hms)
    x_t, y_t = radec_to_pixel(header, ra_t, dec_t)
    x_b, y_b = radec_to_pixel(header, ra_b, dec_b)
    _, theta_t = deproject_pixel_offsets(x_t - rc.xpos_pix, y_t - rc.ypos_pix, rc.pa_deg, rc.inc_deg, cdelt1_sign)
    _, theta_b = deproject_pixel_offsets(x_b - rc.xpos_pix, y_b - rc.ypos_pix, rc.pa_deg, rc.inc_deg, cdelt1_sign)
    mean_theta = np.arctan2(np.sin(theta_t) + np.sin(theta_b), np.cos(theta_t) + np.cos(theta_b))
    return float(np.degrees(mean_theta) % 360.0)


# --------------------------------------------------------------------------
# 2.4 Disk shear (new)
# --------------------------------------------------------------------------


def disk_shear_timescale(cfg: Config, rc: RotationCurve, adopted_age_myr: Optional[float] = None) -> dict:
    """Delta_Omega = Omega(R_in) - Omega(R_out), t_shear = 1/Delta_Omega (one
    radian of shear). An UPPER LIMIT on the age of any COHERENT MATERIAL
    feature spanning [r_in, r_out] -- it does NOT apply to the ring itself,
    which this project treats as a wave (epicyclic) pattern, not advected
    material (see Section 2.2's T_kappa/2 for the ring's own clock).

    r_in/r_out default to the ringlog's own innermost/outermost ring radius
    (never a free-floating literal); Config.shear_r_in_kpc/shear_r_out_kpc
    override for a specific feature."""
    r_in = cfg.shear_r_in_kpc if cfg.shear_r_in_kpc is not None else float(rc.radii_kpc.min())
    r_out = cfg.shear_r_out_kpc if cfg.shear_r_out_kpc is not None else float(rc.radii_kpc.max())
    if r_out <= r_in:
        raise ValueError(f"disk_shear_timescale: r_out ({r_out}) must exceed r_in ({r_in})")

    Omega, _ = calculate_frequencies(np.array([r_in, r_out]), rc.curve)
    delta_Omega = float(Omega[0] - Omega[1])
    if delta_Omega <= 0:
        raise RuntimeError(
            f"disk_shear_timescale: Omega(r_in)={Omega[0]:.4f} <= Omega(r_out)={Omega[1]:.4f} km/s/kpc -- "
            "the rotation curve is not differentially rotating (Omega decreasing outward) over this "
            "radial range; t_shear is undefined."
        )
    t_shear_myr = (1.0 / delta_Omega) * CONVERSION_MYR

    shear_angle_deg = None
    if adopted_age_myr is not None:
        shear_angle_deg = np.degrees(adopted_age_myr / t_shear_myr)

    return {
        "r_in_kpc": r_in, "r_out_kpc": r_out, "delta_Omega_km_s_kpc": delta_Omega,
        "t_shear_myr": t_shear_myr, "adopted_age_myr": adopted_age_myr,
        "shear_angle_at_adopted_age_deg": shear_angle_deg,
    }


# --------------------------------------------------------------------------
# 2.5 Pair separation, linear (new)
# --------------------------------------------------------------------------


def compute_dV_sys(vsys_table_kms: dict, label_a: str, label_b: str) -> float:
    """dV_sys = table[label_b] - table[label_a], read from a named table of
    systemic-velocity measurements rather than hand-typed: the manuscript
    previously carried inconsistent values (207 km/s in one place, 209 km/s
    in another, with 240 km/s implied by the source table) precisely
    because this difference was never computed from one single source.
    Prints which pair was differenced so the provenance is always visible."""
    if label_a not in vsys_table_kms or label_b not in vsys_table_kms:
        raise KeyError(
            f"compute_dV_sys: label {label_a!r} or {label_b!r} not in vsys_table_kms "
            f"(available labels: {list(vsys_table_kms)})"
        )
    v_a, src_a = vsys_table_kms[label_a]
    v_b, src_b = vsys_table_kms[label_b]
    dV = float(v_b) - float(v_a)
    print(
        f"[timescales] dV_sys = VSYS[{label_b!r}] - VSYS[{label_a!r}] "
        f"= ({v_b:.1f} km/s, {src_b}) - ({v_a:.1f} km/s, {src_a}) = {dV:.1f} km/s"
    )
    return dV


def bridge_timescale_myr(R_over_dV_kpc_per_kms: float, theta_deg: float, conversion_myr: float = CONVERSION_MYR) -> float:
    """t_lin = (R_sep/dV_sys)*cot(theta)*977.8 [Myr] (Section 2.5). A naive
    t = R_sep/dV_sys*conversion_myr (no cot(theta) factor) implicitly
    assumes theta=45 deg, since cot(45 deg)=1; this makes the assumed
    line-of-sight angle explicit and correctable."""
    theta_rad = np.radians(theta_deg)
    return (R_over_dV_kpc_per_kms / np.tan(theta_rad)) * conversion_myr


def bridge_timescale_sensitivity(theta_deg: float, dtheta_deg: float) -> float:
    """dt/t = dtheta / (sin(theta)*cos(theta)) (dtheta in radians for the
    ratio to be dimensionless)."""
    theta_rad = np.radians(theta_deg)
    return np.radians(dtheta_deg) / (np.sin(theta_rad) * np.cos(theta_rad))


def validate_bridge_orientation(bridge_pa_deg: Optional[float], disc_pa_deg: float):
    """The bridge lying along the disc's projected minor axis (PA=phi+90) is
    what licenses theta=i (the disc-normal case) -- checked here, not
    assumed. |bridge_pa - (phi+90)| < 5 deg is the pass condition; prints
    both angles either way, and skips (rather than guesses) if no measured
    bridge PA is supplied."""
    expected_pa = (disc_pa_deg + 90.0) % 180.0
    if bridge_pa_deg is None:
        print(
            "[timescales] Bridge orientation check SKIPPED: bridge_pa_deg not supplied "
            f"(expected, if theta=i: {expected_pa:.2f} deg)."
        )
        return None
    observed_pa = bridge_pa_deg % 180.0
    diff = abs(observed_pa - expected_pa)
    diff = min(diff, 180.0 - diff)
    passed = diff < 5.0
    print(
        f"[timescales] Bridge orientation check: measured PA={observed_pa:.2f} deg vs. expected "
        f"(disc minor axis, phi+90={expected_pa:.2f} deg) -- delta={diff:.2f} deg -- "
        f"{'PASSED' if passed else 'FAILED'} (theta=i requires delta < 5 deg)"
    )
    return {"observed_pa_deg": observed_pa, "expected_pa_deg": expected_pa, "delta_deg": diff, "passed": passed}


def pair_separation_linear(cfg: Config, rc: RotationCurve) -> Optional[dict]:
    """Reports the naive theta=45 deg case explicitly (the value the
    manuscript previously quoted) alongside the disk-normal theta=i case,
    plus the sensitivity dt/t and the bridge-orientation check that licenses
    theta=i. Skips cleanly (printed, not guessed) if the pair separation /
    systemic-velocity table are not supplied."""
    if cfg.R_sep_kpc is None or cfg.vsys_table_kms is None:
        print("[timescales] Section 2.5 (pair separation, linear) SKIPPED: "
              "R_sep_kpc and/or vsys_table_kms not supplied in Config.")
        return None

    dV_sys = compute_dV_sys(cfg.vsys_table_kms, cfg.vsys_pair_label_a, cfg.vsys_pair_label_b)
    ratio = cfg.R_sep_kpc / dV_sys

    t_naive = bridge_timescale_myr(ratio, 45.0)
    t_disk_normal = bridge_timescale_myr(ratio, rc.inc_deg)
    sens_naive = bridge_timescale_sensitivity(45.0, cfg.bridge_theta_sensitivity_step_deg)
    sens_disk_normal = bridge_timescale_sensitivity(rc.inc_deg, cfg.bridge_theta_sensitivity_step_deg)
    orientation = validate_bridge_orientation(cfg.bridge_pa_deg, rc.pa_deg)

    return {
        "R_sep_kpc": cfg.R_sep_kpc, "dV_sys_kms": dV_sys, "R_sep_over_dV_sys": ratio,
        "t_naive_theta45_myr": t_naive, "sensitivity_naive_per_step": sens_naive,
        "t_disk_normal_theta_i_myr": t_disk_normal, "sensitivity_disk_normal_per_step": sens_disk_normal,
        "sensitivity_step_deg": cfg.bridge_theta_sensitivity_step_deg,
        "theta_i_deg": rc.inc_deg, "orientation_check": orientation,
    }


# --------------------------------------------------------------------------
# 2.6 Pair orbit integration (new, galpy)
# --------------------------------------------------------------------------


def orbit_time_quadrature_myr(pot, r_peri_kpc: float, r_now_kpc: float, v_now_kms: float) -> float:
    """t = integral_{r_peri}^{r_now} dr/v(r), v(r) = sqrt(2*(E - Phi(r))),
    E = v_now^2/2 + Phi(r_now). Phi from the ONE galpy NFWPotential built by
    build_nfw_potential -- no second NFW implementation. r_peri >= r_now
    returns 0 (already at/past the reference radius)."""
    if r_peri_kpc >= r_now_kpc:
        return 0.0
    phi_now = potential_kms2(pot, r_now_kpc)[0]
    E = 0.5 * v_now_kms**2 + phi_now

    def integrand(r):
        phi_r = potential_kms2(pot, r)[0]
        v2 = 2.0 * (E - phi_r)
        if v2 <= 0:
            raise RuntimeError(
                f"orbit_time_quadrature_myr: orbit not energetically allowed at r={r:.3f} kpc "
                f"(E={E:.2f}, Phi(r)={phi_r:.2f} (km/s)^2)."
            )
        return 1.0 / np.sqrt(v2)

    t_kpc_per_kms, _ = quad(integrand, r_peri_kpc, r_now_kpc, limit=200)
    return t_kpc_per_kms * CONVERSION_MYR


def orbit_time_via_galpy_orbit_myr(pot, r_peri_kpc: float, r_now_kpc: float, v_now_kms: float,
                                    ro_kpc: float, vo_kms: float, n_steps: int = 20000) -> float:
    """Cross-check for orbit_time_quadrature_myr (Section 4, acceptance test
    3): integrates the SAME purely-radial orbit forward in time with
    galpy.orbit.Orbit under the same potential, starting at r_now moving
    inward at v_now, and finds when R(t) first reaches r_peri. Verification
    only -- scipy's quadrature is far cheaper and is what the main results
    path (pair_orbit_report) actually uses."""
    if r_peri_kpc >= r_now_kpc:
        return 0.0
    ro, vo = ro_kpc * u.kpc, vo_kms * u.km / u.s
    t_guess_myr = orbit_time_quadrature_myr(pot, r_peri_kpc, r_now_kpc, v_now_kms)
    o = Orbit([r_now_kpc * u.kpc, -v_now_kms * u.km / u.s, 0.0 * u.km / u.s,
               0.0 * u.kpc, 0.0 * u.km / u.s, 0.0 * u.rad], ro=ro, vo=vo)
    ts = np.linspace(0.0, max(2.0 * t_guess_myr, 1.0), n_steps) * u.Myr
    o.integrate(ts, pot)
    Rs = np.asarray(o.R(ts))
    below = np.nonzero(Rs <= r_peri_kpc)[0]
    if below.size == 0:
        raise RuntimeError(
            f"orbit_time_via_galpy_orbit_myr: R(t) never reached r_peri={r_peri_kpc} kpc within the "
            f"integrated time window (0 to {ts[-1]}); widen n_steps/time window."
        )
    i0 = int(below[0])
    t_arr = ts.to(u.Myr).value
    if i0 == 0:
        return float(t_arr[0])
    R0, R1 = Rs[i0 - 1], Rs[i0]
    t0, t1 = t_arr[i0 - 1], t_arr[i0]
    frac = (r_peri_kpc - R0) / (R1 - R0)
    return float(t0 + frac * (t1 - t0))


def pair_orbit_r_now_v_now(R_sep_kpc: float, dV_sys_kms: float, theta_deg: float):
    """r_now = R_sep/sin(theta), v_now = dV_sys/cos(theta)."""
    theta_rad = np.radians(theta_deg)
    return R_sep_kpc / np.sin(theta_rad), dV_sys_kms / np.cos(theta_rad)


def theta_for_target_t(theta_grid_deg, t_grid_myr, target_t_myr: float) -> Optional[float]:
    """The theta at which t(theta) matches a target time (e.g. the
    ring-turnaround estimate) -- linear interpolation at the first sign
    change of (t - target) on the grid; None if the target is outside the
    grid's range."""
    t = np.asarray(t_grid_myr, dtype=float)
    theta = np.asarray(theta_grid_deg, dtype=float)
    diff = t - target_t_myr
    cross = np.nonzero(np.diff(np.sign(diff)) != 0)[0]
    if cross.size == 0:
        return None
    j = cross[0]
    frac = -diff[j] / (diff[j + 1] - diff[j])
    return float(theta[j] + frac * (theta[j + 1] - theta[j]))


def pair_orbit_report(cfg: Config, rc: RotationCurve, target_t_myr: Optional[float] = None) -> Optional[dict]:
    """t for r_peri in cfg.r_peri_scan_kpc (sensitivity), a theta scan from
    cfg.theta_scan_lo_deg to cfg.theta_scan_hi_deg tabulating
    r_now/v_now/t/v_over_vesc, and the headline numbers: t at theta=i
    (disk-normal), the theta at which v/v_esc=1 (a hard lower bound on t if
    the pair is bound), and the theta needed to match target_t_myr (e.g. the
    ring-turnaround estimate). Skips cleanly if the pair separation /
    systemic-velocity table are not supplied."""
    if cfg.R_sep_kpc is None or cfg.vsys_table_kms is None:
        print("[timescales] Section 2.6 (pair orbit integration) SKIPPED: "
              "R_sep_kpc and/or vsys_table_kms not supplied in Config.")
        return None

    pot, Ms, a, r200 = build_nfw_potential(cfg)
    dV_sys = compute_dV_sys(cfg.vsys_table_kms, cfg.vsys_pair_label_a, cfg.vsys_pair_label_b)

    r_now_i, v_now_i = pair_orbit_r_now_v_now(cfg.R_sep_kpc, dV_sys, rc.inc_deg)
    r_peri_report = []
    for r_peri in cfg.r_peri_scan_kpc:
        t = orbit_time_quadrature_myr(pot, r_peri, r_now_i, v_now_i)
        r_peri_report.append({"r_peri_kpc": r_peri, "t_myr": t})

    theta_grid = np.arange(cfg.theta_scan_lo_deg, cfg.theta_scan_hi_deg + 1e-9, cfg.theta_scan_step_deg)
    r_now_grid = np.empty_like(theta_grid)
    v_now_grid = np.empty_like(theta_grid)
    t_grid = np.empty_like(theta_grid)
    vesc_grid = np.empty_like(theta_grid)
    for i, theta in enumerate(theta_grid):
        r_now, v_now = pair_orbit_r_now_v_now(cfg.R_sep_kpc, dV_sys, theta)
        r_now_grid[i] = r_now
        v_now_grid[i] = v_now
        vesc_grid[i] = escape_velocity_kms(pot, r_now)[0]
        t_grid[i] = orbit_time_quadrature_myr(pot, cfg.r_peri_kpc, r_now, v_now)
    v_over_vesc_grid = v_now_grid / vesc_grid

    i_theta_i = int(np.argmin(np.abs(theta_grid - rc.inc_deg)))
    t_at_theta_i = float(t_grid[i_theta_i])

    unbound = v_over_vesc_grid >= 1.0
    theta_v_eq_vesc = None
    if np.any(unbound) and np.any(~unbound):
        cross = np.nonzero(np.diff(unbound.astype(int)) != 0)[0]
        if cross.size:
            j = cross[0]
            frac = (1.0 - v_over_vesc_grid[j]) / (v_over_vesc_grid[j + 1] - v_over_vesc_grid[j])
            theta_v_eq_vesc = float(theta_grid[j] + frac * (theta_grid[j + 1] - theta_grid[j]))

    theta_match_target = theta_for_target_t(theta_grid, t_grid, target_t_myr) if target_t_myr is not None else None

    return {
        "R_sep_kpc": cfg.R_sep_kpc, "dV_sys_kms": dV_sys, "r_peri_fiducial_kpc": cfg.r_peri_kpc,
        "r_peri_scan": r_peri_report,
        "theta_grid_deg": theta_grid, "r_now_grid_kpc": r_now_grid, "v_now_grid_kms": v_now_grid,
        "t_grid_myr": t_grid, "v_over_vesc_grid": v_over_vesc_grid,
        "theta_i_deg": rc.inc_deg, "t_at_theta_i_myr": t_at_theta_i,
        "theta_v_eq_vesc_deg": theta_v_eq_vesc,
        "target_t_myr": target_t_myr, "theta_matching_target_deg": theta_match_target,
        "Ms_Msun": Ms.to(u.Msun).value, "a_kpc": a.to(u.kpc).value, "r200_kpc": r200.to(u.kpc).value,
    }


# --------------------------------------------------------------------------
# 2.7 Debris expansion and the free-expansion test (new)
# --------------------------------------------------------------------------


def debris_free_expansion_test(cfg: Config, rc: RotationCurve) -> Optional[dict]:
    """Free expansion from a common origin implies v=r/t, so the velocity
    gradient must be equal throughout the system: grad_debris =
    dV_debris/dR_debris vs. grad_pair = dV_sys/R_sep. If they differ, free
    expansion is REJECTED, and t_debris is reported for completeness ONLY --
    flagged as rejected, never returned/used as a timescale estimate. Skips
    cleanly if the debris measurements or the pair separation/velocity table
    are not supplied."""
    if cfg.debris_dR_kpc is None or cfg.debris_dV_kms is None:
        print("[timescales] Section 2.7 (debris expansion) SKIPPED: "
              "debris_dR_kpc and/or debris_dV_kms not supplied in Config.")
        return None
    if cfg.R_sep_kpc is None or cfg.vsys_table_kms is None:
        print("[timescales] Section 2.7 (debris expansion) SKIPPED: "
              "R_sep_kpc and/or vsys_table_kms not supplied in Config (needed for grad_pair).")
        return None

    dV_sys = compute_dV_sys(cfg.vsys_table_kms, cfg.vsys_pair_label_a, cfg.vsys_pair_label_b)
    grad_debris = cfg.debris_dV_kms / cfg.debris_dR_kpc
    grad_pair = dV_sys / cfg.R_sep_kpc
    ratio = grad_debris / grad_pair
    t_debris_myr = bridge_timescale_myr(cfg.debris_dR_kpc / cfg.debris_dV_kms, rc.inc_deg)

    print(
        f"[timescales] Section 2.7: grad_debris={grad_debris:.4f} km/s/kpc, grad_pair={grad_pair:.4f} km/s/kpc, "
        f"ratio={ratio:.3f} -- free expansion is REJECTED if this ratio is far from 1. "
        f"t_debris={t_debris_myr:.1f} Myr is reported for completeness ONLY -- NOT a clock."
    )
    return {"grad_debris_km_s_kpc": grad_debris, "grad_pair_km_s_kpc": grad_pair, "ratio": ratio,
            "t_debris_myr_REJECTED": t_debris_myr}


# --------------------------------------------------------------------------
# 3. Epicyclic ("kick") chi2 grid -- the two-parameter model in Section 2.8
# --------------------------------------------------------------------------


def kick_model_v_R(R_kpc, t_myr, V0_kms, alpha, v_curve, r_ref_kpc, conversion_myr=CONVERSION_MYR):
    """Wallin & Struck-Marcell (1994)-style radial kick, damped/oscillating
    through the epicyclic frequency: V_R(R, t) = V0*(R/r_ref)^-alpha *
    sin(kappa(R)*t)."""
    R_kpc = np.asarray(R_kpc, dtype=float)
    _, kappa = calculate_frequencies(R_kpc, v_curve)
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
    """chi2(V0, t) = sum_i [(V_R_obs_i - model_i)^2 / sigma_i^2], evaluated
    on a 2D grid -- no scipy.optimize call anywhere in this path. This grid's
    minimum is the "two-parameter epicyclic model" chi2 used in Section
    2.8's exclusion test against the V_rad=0 null."""
    warn_if_extrapolating("epicyclic grid (ring radii)", rv.radii_kpc, rc.radii_kpc.min(), rc.radii_kpc.max())

    V0_grid = np.arange(0.0, cfg.v0_grid_max_kms + 1e-9, cfg.v0_grid_step_kms)
    t_grid = np.arange(0.0, cfg.t_grid_max_myr + 1e-9, cfg.t_grid_step_myr)

    _, kappa = calculate_frequencies(rv.radii_kpc, rc.curve)
    kappa_myr = kappa / CONVERSION_MYR
    R_ratio = (rv.radii_kpc / cfg.r_ref_kpc) ** (-cfg.alpha)

    V0 = V0_grid[:, None, None]
    t = t_grid[None, :, None]
    model = V0 * R_ratio[None, None, :] * np.sin(kappa_myr[None, None, :] * t)
    resid = (rv.v_R_outward_kms[None, None, :] - model) / rv.sigma_kms[None, None, :]
    chi2 = np.sum(resid**2, axis=2)

    i_min = np.unravel_index(np.argmin(chi2), chi2.shape)
    return EpicyclicGrid(
        V0_grid_kms=V0_grid, t_grid_myr=t_grid, chi2=chi2, chi2_min=float(chi2[i_min]),
        V0_at_min=float(V0_grid[i_min[0]]), t_at_min=float(t_grid[i_min[1]]),
    )


# --------------------------------------------------------------------------
# 2.8 Radial oscillation exclusion (already partly implemented)
# --------------------------------------------------------------------------


def pa_uncertainty_from_scan(results_dir: Path, ring_index: int) -> Optional[float]:
    """Formal PA uncertainty for one ring: half the delta_chi2=1 width of the
    PA scan's chi2_best(PA) curve, already computed and written by
    harmonic_fit.py's pa_scan -- not a second, independently fitted PA
    uncertainty. Returns None if scans.npz lacks that ring's PA scan."""
    path = Path(results_dir) / "scans.npz"
    if not path.exists():
        return None
    d = np.load(path)
    grid_key, chi2_key = f"ring{ring_index}_pa_pa_grid_deg", f"ring{ring_index}_pa_chi2_best"
    if grid_key not in d.files or chi2_key not in d.files:
        return None
    pa_grid, chi2_best = d[grid_key], d[chi2_key]
    i_min = int(np.argmin(chi2_best))
    delta = chi2_best - chi2_best[i_min]
    within = np.nonzero(delta <= 1.0)[0]
    if within.size < 2:
        return None
    return float((pa_grid[within.max()] - pa_grid[within.min()]) / 2.0)


def radial_oscillation_exclusion(cfg: Config, rv: RadialVelocities, rc: RotationCurve, grid: EpicyclicGrid) -> dict:
    """chi2/dof for the null V_rad=0, delta_chi2 against the two-parameter
    (V0, t) epicyclic grid (Section 3), and the per-annulus significance
    V_rad_i/sigma_i. Deliberately does NOT quote an amplitude limit in pc --
    the statistical limit on V0 is swamped by the PA systematic
    (harmonic_fit.pa_degeneracy_slope, with phi's own formal uncertainty
    read from the PA scan already computed by harmonic_fit.py, never
    re-derived or hardcoded here) and by beam dilution (the beam is wider
    than the ring); both are printed alongside the statistical error so the
    comparison is explicit."""
    chi2_null = float(np.sum((rv.v_R_outward_kms / rv.sigma_kms) ** 2))
    dof_null = len(rv.v_R_outward_kms)
    chi2_null_reduced = chi2_null / dof_null
    chi2_2param = grid.chi2_min
    delta_chi2 = chi2_null - chi2_2param
    significance = rv.v_R_outward_kms / rv.sigma_kms

    n = len(rv.ring_index)
    pa_systematic_kms_per_deg = np.full(n, np.nan)
    pa_uncertainty_deg = np.full(n, np.nan)
    maps_path = Path(cfg.results_dir) / "maps.npz"
    if maps_path.exists():
        maps = np.load(maps_path)
        for k, ring_idx in enumerate(rv.ring_index):
            theta_key = f"ring{ring_idx}_theta"
            mask_key = f"ring{ring_idx}_ring_mask_both"
            w_key = f"ring{ring_idx}_weights_primary"
            if theta_key not in maps.files:
                continue
            mask = maps[mask_key]
            theta_m = maps[theta_key][mask]
            w_m = maps[w_key][mask]
            slope_per_rad = pa_degeneracy_slope(theta_m, w_m, float(rc.v_kms[ring_idx]), rc.inc_deg)
            pa_systematic_kms_per_deg[k] = slope_per_rad * np.pi / 180.0
            pa_unc = pa_uncertainty_from_scan(cfg.results_dir, int(ring_idx))
            if pa_unc is not None:
                pa_uncertainty_deg[k] = pa_unc

    pa_systematic_kms = np.abs(pa_systematic_kms_per_deg) * pa_uncertainty_deg

    print(
        f"[timescales] Section 2.8: chi2/dof (null V_rad=0) = {chi2_null:.2f}/{dof_null} = "
        f"{chi2_null_reduced:.3f}; delta_chi2 (vs. 2-param epicyclic grid min) = {delta_chi2:.2f}"
    )
    for k, ring_idx in enumerate(rv.ring_index):
        print(
            f"    ring {ring_idx}: V_rad/sigma = {significance[k]:+.2f}   "
            f"PA systematic = {pa_systematic_kms_per_deg[k]:+.3f} km/s/deg * "
            f"{pa_uncertainty_deg[k]:.2f} deg (TRM PA scan) = {pa_systematic_kms[k]:.2f} km/s   "
            f"vs. statistical sigma = {rv.sigma_kms[k]:.2f} km/s"
        )
    print(
        "[timescales] No amplitude limit in pc is quoted: the statistical limit on V0 is swamped by the "
        "PA systematic and by beam dilution (beam wider than the ring), shown above."
    )

    return {
        "chi2_null": chi2_null, "dof_null": dof_null, "chi2_null_reduced": chi2_null_reduced,
        "chi2_2param_epicyclic_min": chi2_2param, "delta_chi2": delta_chi2,
        "ring_index": rv.ring_index.tolist(), "significance_V_rad_over_sigma": significance.tolist(),
        "pa_systematic_kms_per_deg": pa_systematic_kms_per_deg.tolist(),
        "pa_uncertainty_deg": pa_uncertainty_deg.tolist(),
        "pa_systematic_kms": pa_systematic_kms.tolist(),
        "statistical_sigma_kms": rv.sigma_kms.tolist(),
    }


# --------------------------------------------------------------------------
# 2.9 Wallin & Struck-Marcell (1994) epoch conversion (new)
# --------------------------------------------------------------------------


def wsm_epoch_conversion(cfg: Config, rc: RotationCurve, R_ring_kpc: float) -> dict:
    """Their model times are in units of 2*pi/omega(eps), the epicyclic
    period AT THE SOFTENING LENGTH eps -- t_WSM = f*T_kappa(eps). Their
    AM 1724-like epochs are f=0.45 and 0.56 (Figs. 5a, 6). Tabulated over
    cfg.wsm_eps_grid_kpc so the sensitivity to that identification is
    visible; eps=R_ring is marked separately since their Section 4.2
    supports it (best morphological match when the ring has propagated to
    about one softening length, near the rotation-curve turnover)."""
    eps_grid = np.array(cfg.wsm_eps_grid_kpc, dtype=float)
    warn_if_extrapolating("WSM epoch conversion (eps grid)", eps_grid, rc.radii_kpc.min(), rc.radii_kpc.max())
    Omega_eps, kappa_eps = calculate_frequencies(eps_grid, rc.curve)
    _, T_kappa_eps = calculate_timescales(Omega_eps, kappa_eps)

    table = []
    for eps, T_kappa in zip(eps_grid, T_kappa_eps):
        row = {"eps_kpc": float(eps), "T_kappa_eps_myr": float(T_kappa)}
        for f in cfg.wsm_epochs:
            row[f"t_WSM_f{f}_myr"] = float(f) * float(T_kappa)
        table.append(row)

    warn_if_extrapolating("WSM epoch conversion (eps=R_ring)", np.array([R_ring_kpc]), rc.radii_kpc.min(), rc.radii_kpc.max())
    Omega_ring, kappa_ring = calculate_frequencies(np.array([R_ring_kpc]), rc.curve)
    _, T_kappa_ring = calculate_timescales(Omega_ring, kappa_ring)
    ring_row = {"eps_kpc": R_ring_kpc, "T_kappa_eps_myr": float(T_kappa_ring[0])}
    for f in cfg.wsm_epochs:
        ring_row[f"t_WSM_f{f}_myr"] = float(f) * float(T_kappa_ring[0])

    return {"eps_grid_table": table, "eps_equals_R_ring": ring_row, "epochs": list(cfg.wsm_epochs)}


# --------------------------------------------------------------------------
# Results assembly / I/O
# --------------------------------------------------------------------------


def _adopted_age_range(quantities: dict):
    """The "adopted age and range" used for fig_clocks's horizontal band and
    the shear-angle report: whichever of {ring turnaround (2.2), WSM at
    eps=R_ring (2.9, both epochs), pair orbit at theta=i (2.6)} are
    available (the pair-orbit contribution only when the pair data is
    supplied), taking the overall min/max -- never a hand-typed number."""
    candidates = {}
    for name, q in quantities.items():
        if q is not None and q.get("unit") == "Myr" and np.isfinite(q["value"]):
            candidates[name] = q["value"]
    if not candidates:
        return None
    lo_name = min(candidates, key=candidates.get)
    hi_name = max(candidates, key=candidates.get)
    return {
        "lo_myr": candidates[lo_name], "hi_myr": candidates[hi_name],
        "lo_source": lo_name, "hi_source": hi_name, "contributors": candidates,
    }


def run(cfg: Config) -> dict:
    ringlog = read_ringlog(cfg.ringlog_path, cfg.kpc_per_arcsec)
    rc = load_rotation_curve(cfg.ringlog_path, order=cfg.rotation_curve_poly_order, kpc_per_arcsec=cfg.kpc_per_arcsec)
    rv = load_radial_velocities(cfg.results_dir, side=cfg.side)

    mapset = load_maps(cfg.trm_dir / "maps")
    cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)
    data_1mom = sorted((cfg.trm_dir / "maps").glob("*_1mom.fits"))
    data_1mom = [f for f in data_1mom if "_local_" not in f.name][0]
    header1 = fits.getheader(data_1mom)

    # -------- Section 0: NFW verification --------
    nfw_check = verify_nfw_potential(cfg)

    # -------- Section 1: R_ring --------
    mom0_data, header0 = load_data_mom0(cfg.trm_dir / "maps")
    sigma_profile = measure_R_ring(rc, mom0_data, header0, cdelt1_sign, ringlog.meta["ring_width_arcsec"])

    # -------- 2.1 per-ring clocks --------
    T_phi_ring, T_kappa_ring, T_kappa_half_ring, dv_dR_ring = t_kappa_half_per_ring(rc)

    # -------- 2.2 ring turnaround --------
    t_ring_turnaround, t_ring_turnaround_err = ring_turnaround_timescale(rc, sigma_profile.R_ring_kpc, sigma_profile.R_ring_err_kpc)

    # -------- 2.3 crescent void --------
    void = void_geometry(cfg, rc, header1, cdelt1_sign)
    void_nearest_vc = void_interaction_nearest_vc(rc, void)

    # -------- 3 + 2.8: epicyclic grid + radial-oscillation exclusion --------
    grid = epicyclic_chi2_grid(cfg, rv, rc)
    exclusion = radial_oscillation_exclusion(cfg, rv, rc, grid)

    # -------- 2.9 WSM epoch conversion --------
    wsm = wsm_epoch_conversion(cfg, rc, sigma_profile.R_ring_kpc)

    # -------- 2.5/2.6/2.7: the pair (computed before the adopted-age range,
    # so a supplied pair-orbit theta=i estimate is eligible to contribute to
    # it, not just ring turnaround/void/WSM) --------
    pair_linear = pair_separation_linear(cfg, rc)
    pair_orbit = pair_orbit_report(cfg, rc, target_t_myr=t_ring_turnaround)
    debris = debris_free_expansion_test(cfg, rc)

    # -------- headline quantities, then the adopted age range they define --------
    quantities = {
        "ring_turnaround": _q(t_ring_turnaround, "Myr", "estimate",
                               "T_kappa(R_ring)/2; lower limit relative to the stellar (collisionless) prediction"),
        "void_interaction_upper_limit": _q(void.t_interaction_myr, "Myr", "upper_limit",
                                            "constant dtheta (Wallin & Struck-Marcell 1994 Sec 3.3.3: the hole keeps expanding)"),
        "wsm_eps_R_ring_f_lo": _q(wsm["eps_equals_R_ring"][f"t_WSM_f{cfg.wsm_epochs[0]}_myr"], "Myr", "estimate",
                                  f"f={cfg.wsm_epochs[0]}, eps=R_ring (Wallin & Struck-Marcell 1994 Sec 4.2)"),
        "wsm_eps_R_ring_f_hi": _q(wsm["eps_equals_R_ring"][f"t_WSM_f{cfg.wsm_epochs[-1]}_myr"], "Myr", "estimate",
                                  f"f={cfg.wsm_epochs[-1]}, eps=R_ring (Wallin & Struck-Marcell 1994 Sec 4.2)"),
    }
    if pair_orbit is not None:
        quantities["pair_orbit_theta_i"] = _q(
            pair_orbit["t_at_theta_i_myr"], "Myr", "estimate",
            f"radial orbit, theta=i, r_peri={cfg.r_peri_kpc} kpc (Wallin & Struck-Marcell 1994 Table 1)")

    adopted = _adopted_age_range(quantities)
    adopted_mid_myr = None
    if adopted is not None:
        adopted_mid_myr = 0.5 * (adopted["lo_myr"] + adopted["hi_myr"])
        print(f"[timescales] Adopted age range: {adopted['lo_myr']:.1f} ({adopted['lo_source']}) - "
              f"{adopted['hi_myr']:.1f} Myr ({adopted['hi_source']}); midpoint {adopted_mid_myr:.1f} Myr "
              "used for the shear-angle report.")

    # -------- 2.4 disk shear --------
    shear = disk_shear_timescale(cfg, rc, adopted_age_myr=adopted_mid_myr)

    if pair_linear is not None:
        quantities["pair_linear_naive_theta45"] = _q(
            pair_linear["t_naive_theta45_myr"], "Myr", "estimate", "theta=45 deg assumed (naive, cot(45)=1)")
        quantities["pair_linear_disk_normal_theta_i"] = _q(
            pair_linear["t_disk_normal_theta_i_myr"], "Myr", "estimate",
            "theta=i (disk-normal); requires bridge PA ~ phi+90 (orientation check)")
    if pair_orbit is not None:
        # pair_orbit_theta_i was already added above, before the adopted-age
        # range was computed, so it's eligible to contribute to it.
        if pair_orbit["theta_v_eq_vesc_deg"] is not None:
            i_cross = int(np.argmin(np.abs(pair_orbit["theta_grid_deg"] - pair_orbit["theta_v_eq_vesc_deg"])))
            # v/v_esc(theta) increases monotonically with theta (larger theta
            # -> smaller r_now and larger v_now, both pushing v/v_esc up),
            # while t(theta) decreases monotonically with theta -- so the
            # bound region is theta <= theta_v_eq_vesc (smaller theta, LONGER
            # orbit), and the unbound region is theta > theta_v_eq_vesc
            # (larger theta, shorter/unphysical-if-bound orbit). Boundedness
            # therefore excludes every t below the crossing's t, making that
            # t a LOWER bound, not an upper one.
            quantities["pair_orbit_bound_lower_limit"] = _q(
                float(pair_orbit["t_grid_myr"][i_cross]), "Myr", "lower_limit",
                f"v=v_esc at theta={pair_orbit['theta_v_eq_vesc_deg']:.1f} deg -- t is a LOWER bound if the pair is bound "
                "(smaller theta is bound, a longer orbit; larger theta is unbound)")
    if debris is not None:
        quantities["debris_t"] = _q(debris["t_debris_myr_REJECTED"], "Myr", "rejected",
                                     f"free-expansion gradients differ by a factor of {debris['ratio']:.2f} -- not a clock")
    quantities["radial_oscillation_exclusion_delta_chi2"] = _q(
        exclusion["delta_chi2"], "dimensionless", "exclusion",
        "delta_chi2 of the null V_rad=0 vs. the 2-parameter (V0,t) epicyclic grid minimum")

    results = {
        "trm_dir": str(cfg.trm_dir), "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "nfw_verification": nfw_check,
        "rotation_curve": {
            "radii_kpc": rc.radii_kpc.tolist(), "v_kms": rc.v_kms.tolist(),
            "v_err_lo_kms": rc.v_err_lo.tolist(), "v_err_hi_kms": rc.v_err_hi.tolist(),
            "inc_deg": rc.inc_deg, "pa_deg": rc.pa_deg, "kpc_per_arcsec": rc.kpc_per_arcsec,
        },
        "radial_velocities": {
            "ring_index": rv.ring_index.tolist(), "radii_kpc": rv.radii_kpc.tolist(),
            "s1_raw_kms": rv.s1_raw_kms.tolist(), "v_R_outward_kms": rv.v_R_outward_kms.tolist(),
            "sigma_kms": rv.sigma_kms.tolist(),
        },
        "sigma_hi_profile": {
            "R_kpc": sigma_profile.R_kpc.tolist(), "R_edges_kpc": sigma_profile.R_edges_kpc.tolist(),
            "sigma": sigma_profile.sigma.tolist(), "sigma_err": sigma_profile.sigma_err.tolist(),
            "n_pix": sigma_profile.n_pix.tolist(), "R_ring_kpc": sigma_profile.R_ring_kpc,
            "R_ring_arcsec": sigma_profile.R_ring_arcsec, "R_ring_err_kpc": sigma_profile.R_ring_err_kpc,
            "ring_index": sigma_profile.ring_index, "annulus_width_arcsec": sigma_profile.annulus_width_arcsec,
        },
        "per_ring_clocks": {
            "radii_kpc": rc.radii_kpc.tolist(), "T_phi_myr": T_phi_ring.tolist(),
            "T_kappa_myr": T_kappa_ring.tolist(), "T_kappa_half_myr": T_kappa_half_ring.tolist(),
            "dv_dR_km_s_kpc": dv_dR_ring.tolist(),
        },
        "ring_turnaround": {"t_myr": t_ring_turnaround, "t_err_myr": t_ring_turnaround_err,
                             "R_ring_kpc": sigma_profile.R_ring_kpc, "R_ring_err_kpc": sigma_profile.R_ring_err_kpc},
        "void": {
            "r_avg_kpc": void.r_avg_kpc, "dtheta_deg": void.dtheta_deg, "fraction_of_orbit": void.fraction_of_orbit,
            "v_c_void_kms": void.v_c_void_kms, "T_phi_void_myr": void.T_phi_void_myr,
            "t_interaction_myr_UPPER_LIMIT": void.t_interaction_myr,
            "nearest_measured_Vc_systematic": void_nearest_vc,
            "legacy_r_avg_kpc": void.legacy_r_avg_kpc, "legacy_dtheta_deg": void.legacy_dtheta_deg,
        },
        "disk_shear": shear,
        "pair_separation_linear": pair_linear,
        "pair_orbit": pair_orbit,
        "debris": debris,
        "epicyclic_grid": {
            "alpha_fixed": cfg.alpha, "chi2_min": grid.chi2_min,
            "V0_at_chi2_min_kms": grid.V0_at_min, "t_at_chi2_min_myr": grid.t_at_min,
        },
        "radial_oscillation_exclusion": exclusion,
        "wsm_epoch_conversion": wsm,
        "adopted_age_range": adopted,
        "quantities": quantities,
    }
    return results


def build_summary_table(results: dict) -> list:
    """Table 2 (Method, Observable, t [Myr], Type, Limiting assumption) --
    generated directly from results["quantities"] (the same dict written to
    timescales.json), never hand-typed.

    Two kinds of entries in results["quantities"] are deliberately excluded
    from this table (though they remain in results["quantities"] and thus in
    timescales.json -- "keep them available as named variables" -- and the
    notebook prints them itself, e.g. as a linear-vs-orbit-integration
    deceleration illustration):
      - "pair_linear_naive_theta45" / "pair_linear_disk_normal_theta_i": the
        linear R_sep/dV_sys estimate (Section 2.5) is superseded by the
        galpy orbit integration (Section 2.6, "pair_orbit_theta_i") as the
        headline pair-timescale number.
      - "wsm_eps_R_ring_f_lo" / "wsm_eps_R_ring_f_hi": consolidated below
        into a single WSM_EPOCH_RANGE_KEYS row spanning both epochs, rather
        than two separate rows for the same eps=R_ring identification.
    """
    EXCLUDED_FROM_TABLE2 = {"pair_linear_naive_theta45", "pair_linear_disk_normal_theta_i"}
    WSM_EPOCH_RANGE_KEYS = ("wsm_eps_R_ring_f_lo", "wsm_eps_R_ring_f_hi")

    label_method = {
        "ring_turnaround": ("Epicyclic (ring)", "T_kappa(R_ring)/2"),
        "void_interaction_upper_limit": ("Crescent void", "T_phi * dtheta/360"),
        "pair_orbit_theta_i": ("Pair orbit (galpy)", "radial infall, theta=i"),
        "pair_orbit_bound_lower_limit": ("Pair orbit (galpy)", "v=v_esc bound"),
        "debris_t": ("Debris expansion", "(dR/dV)*cot(i) [REJECTED]"),
        "radial_oscillation_exclusion_delta_chi2": ("Radial-oscillation exclusion", "delta_chi2 (null vs. 2-param)"),
    }
    rows = []
    for key, q in results["quantities"].items():
        if key in EXCLUDED_FROM_TABLE2 or key in WSM_EPOCH_RANGE_KEYS:
            continue
        if q["unit"] != "Myr":
            # Non-timescale exclusion diagnostics (e.g. delta_chi2) don't fit
            # a "t [Myr]" column; they are reported by radial_oscillation_
            # exclusion's own print statements and carried in the JSON
            # instead, not forced into this table.
            continue
        method, observable = label_method.get(key, (key, ""))
        rows.append({
            "Method": method, "Observable": observable, "t [Myr]": q["value"],
            "Type": q["type"], "Limiting assumption": q["assumption"],
        })

    if all(k in results["quantities"] for k in WSM_EPOCH_RANGE_KEYS):
        q_lo = results["quantities"]["wsm_eps_R_ring_f_lo"]
        q_hi = results["quantities"]["wsm_eps_R_ring_f_hi"]
        epochs = results["wsm_epoch_conversion"]["epochs"]
        t_lo, t_hi = sorted((q_lo["value"], q_hi["value"]))
        rows.append({
            "Method": "WSM epoch (eps=R_ring)", "Observable": f"f={epochs[0]}-{epochs[-1]}",
            "t [Myr]": f"{t_lo:.1f}-{t_hi:.1f}", "Type": "estimate",
            "Limiting assumption": f"f={epochs[0]}-{epochs[-1]}, eps=R_ring "
                                    "(Wallin & Struck-Marcell 1994 Sec 4.2)",
        })

    disk_shear = results.get("disk_shear")
    if disk_shear is not None:
        rows.append({
            "Method": "Disk shear", "Observable": "1/Delta_Omega (1 radian)",
            "t [Myr]": disk_shear["t_shear_myr"], "Type": "upper_limit",
            "Limiting assumption": "upper limit on coherent-material age over "
                                    f"[{disk_shear['r_in_kpc']:.1f}, {disk_shear['r_out_kpc']:.1f}] kpc; "
                                    "does not apply to the ring (a wave pattern)",
        })
    return rows


def print_summary_table(rows: list):
    if not rows:
        print("[timescales] Summary table is empty.")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n[timescales] Table 2 -- timescale summary:")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(
            (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c]) for c in cols
        )
        print(line)


def _json_default(o):
    """json.dump's default= hook: converts numpy arrays/scalars and Path
    objects (pair_orbit's grids in particular are left as ndarrays in the
    results dict for direct use by the notebook's figures, only converted
    here at the JSON boundary)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable: {o!r}")


def write_results(cfg: Config, results: dict) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "timescales.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[timescales] Wrote {out_path}")
    return out_path


def main(cfg: Config) -> dict:
    results = run(cfg)

    print(f"\n[timescales] Rotation curve from {cfg.ringlog_path}")
    print(f"[timescales] R_ring = {results['sigma_hi_profile']['R_ring_kpc']:.2f} +/- "
          f"{results['sigma_hi_profile']['R_ring_err_kpc']:.2f} kpc "
          f"({results['sigma_hi_profile']['R_ring_arcsec']:.1f} arcsec, annulus {results['sigma_hi_profile']['ring_index']})")

    print("\n[timescales] Per-ring clocks (T_phi, T_kappa, T_kappa/2) [Myr]:")
    for R, tphi, tk, tkh in zip(
        results["per_ring_clocks"]["radii_kpc"], results["per_ring_clocks"]["T_phi_myr"],
        results["per_ring_clocks"]["T_kappa_myr"], results["per_ring_clocks"]["T_kappa_half_myr"],
    ):
        print(f"    R={R:6.2f} kpc: T_phi={tphi:7.1f}  T_kappa={tk:7.1f}  T_kappa/2={tkh:7.1f}")

    rows = build_summary_table(results)
    print_summary_table(rows)

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

    rc = load_rotation_curve(ringlog_path, order=cfg.rotation_curve_poly_order, kpc_per_arcsec=cfg.kpc_per_arcsec)
    rv = load_radial_velocities(results_dir, side=cfg.side)

    # ---------------- Test: kpc_per_arcsec is the adopted Planck18 scale, NOT Barolo's own ----------------
    raw = read_ringlog(ringlog_path, cfg.kpc_per_arcsec)
    check("rc.kpc_per_arcsec equals cfg.kpc_per_arcsec (the adopted scale, threaded through Config)",
          np.isclose(rc.kpc_per_arcsec, cfg.kpc_per_arcsec, rtol=1e-12),
          f"(rc={rc.kpc_per_arcsec:.6f}, cfg={cfg.kpc_per_arcsec:.6f})")
    check("kpc_per_arcsec derived from Planck18 cosmology at VSYS_FOR_DISTANCE_KMS is ~0.329",
          abs(KPC_PER_ARCSEC - 0.329) < 0.002,
          f"(got {KPC_PER_ARCSEC:.5f})")
    check("adopted kpc_per_arcsec is deliberately NOT Barolo's own RAD(Kpc)/RAD(arcs)",
          abs(raw.meta["kpc_per_arcsec"] - raw.meta["kpc_per_arcsec_barolo"]) / raw.meta["kpc_per_arcsec_barolo"] > 0.01,
          f"(adopted={raw.meta['kpc_per_arcsec']:.5f}, Barolo's own={raw.meta['kpc_per_arcsec_barolo']:.5f})")
    check("r_center_kpc is RAD(arcs)*adopted kpc_per_arcsec, not the ringlog's own RAD(Kpc) column",
          np.isclose(float(raw["r_center_kpc"][0]), float(raw["RAD(arcs)"][0]) * cfg.kpc_per_arcsec, rtol=1e-9),
          f"(r_center_kpc[0]={raw['r_center_kpc'][0]:.5f}, RAD(Kpc)[0] (Barolo, unused)={raw['r_center_kpc_barolo'][0]:.5f})")
    check("Rotation curve radii/velocities (first ring) match the raw ringlog table",
          np.isclose(rc.radii_kpc[0], float(raw["r_center_kpc"][0]), atol=1e-9)
          and np.isclose(rc.v_kms[0], float(raw["VROT(km/s)"][0]), atol=1e-9),
          f"(R0={rc.radii_kpc[0]}, V0={rc.v_kms[0]})")
    check("v_R_outward is the sign-flipped s1 (v_R_outward == -s1)",
          np.allclose(rv.v_R_outward_kms, -rv.s1_raw_kms))

    from harmonic_fit import make_geometry as _make_geometry
    shape = (48, 48)
    px, py = 31, 18
    R_grid, th_grid = _make_geometry(shape, rc.xpos_pix, rc.ypos_pix, rc.pa_deg, rc.inc_deg, -1)
    R_point, th_point = deproject_pixel_offsets(px - rc.xpos_pix, py - rc.ypos_pix, rc.pa_deg, rc.inc_deg, -1)
    check("void/R_ring deprojection agrees with make_geometry to 1e-9",
          abs(float(R_point) - R_grid[py, px]) < 1e-9 and abs(float(th_point) - th_grid[py, px]) < 1e-9)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_extrapolating("test", np.array([1.0]), rc.radii_kpc.min(), rc.radii_kpc.max())
        fired = any("evaluating the rotation-curve fit" in str(wi.message) for wi in w)
    check("extrapolation warning fires below the innermost ring", fired)

    grid = epicyclic_chi2_grid(cfg, rv, rc)
    iv, it = 7, 33
    V0_test, t_test = grid.V0_grid_kms[iv], grid.t_grid_myr[it]
    manual = 0.0
    for R, obs, sig in zip(rv.radii_kpc, rv.v_R_outward_kms, rv.sigma_kms):
        model = kick_model_v_R(R, t_test, V0_test, cfg.alpha, rc.curve, cfg.r_ref_kpc)
        manual += ((obs - model) / sig) ** 2
    check("vectorised grid chi2 matches an independent scalar computation (1e-6)",
          abs(grid.chi2[iv, it] - manual) < 1e-6, f"(grid={grid.chi2[iv, it]:.8f}, manual={manual:.8f})")

    ratio = 91.7 / CONVERSION_MYR
    t_naive = bridge_timescale_myr(ratio, 45.0)
    t_corrected = bridge_timescale_myr(ratio, 49.345)
    check("pair-separation worked example: naive (theta=45) reproduces 91.7 Myr", abs(t_naive - 91.7) < 0.05, f"(got {t_naive:.2f})")
    check("pair-separation worked example: corrected (theta=i=49.345) gives 78.8 Myr", abs(t_corrected - 78.8) < 0.1, f"(got {t_corrected:.2f})")
    sens = bridge_timescale_sensitivity(49.345, 5.0)
    check("pair-separation sensitivity: ~18% per 5 deg near i=49 deg", abs(sens - 0.18) < 0.01, f"(got {sens:.3f})")

    # ================= Section 4: new acceptance tests =================

    # ---- 1. All four galpy verification assertions pass ----
    try:
        nfw_result = verify_nfw_potential(cfg)
        check("1. All four galpy NFW verification assertions pass", True,
              f"(a={nfw_result['a_kpc']:.3f} kpc, r200={nfw_result['r200_kpc']:.3f} kpc)")
    except RuntimeError as e:
        check("1. All four galpy NFW verification assertions pass", False, f"({e})")

    # ---- 2. Pot.mass(r200) == M200 to 1e-6 relative ----
    pot, Ms, a, r200 = build_nfw_potential(cfg)
    m200_check = float(pot.mass(r200))
    rel_m200 = abs(m200_check - cfg.M200_Msun) / cfg.M200_Msun
    check("2. Pot.mass(r_200) == M_200 to 1e-6 relative", rel_m200 < 1e-6, f"(rel diff={rel_m200:.2e})")

    # ---- 3. Quadrature vs. galpy Orbit integration agree to 1% ----
    r_now_test, v_now_test, r_peri_test = 80.0, 150.0, 12.0
    t_quad = orbit_time_quadrature_myr(pot, r_peri_test, r_now_test, v_now_test)
    t_orbit = orbit_time_via_galpy_orbit_myr(pot, r_peri_test, r_now_test, v_now_test, cfg.galpy_ro_kpc, cfg.galpy_vo_kms)
    rel_orbit = abs(t_orbit - t_quad) / t_quad
    check("3. Quadrature vs. galpy Orbit integration agree to 1%", rel_orbit < 0.01,
          f"(quad={t_quad:.2f} Myr, orbit={t_orbit:.2f} Myr, rel diff={rel_orbit:.2e})")

    # ---- 4. r_peri -> r_now gives t -> 0 ----
    t_degenerate = orbit_time_quadrature_myr(pot, r_now_test - 1e-6, r_now_test, v_now_test)
    check("4. In the limit r_peri -> r_now, t -> 0", t_degenerate < 1e-3, f"(got {t_degenerate:.6f} Myr)")
    t_equal = orbit_time_quadrature_myr(pot, r_now_test, r_now_test, v_now_test)
    check("4b. r_peri == r_now gives t == 0 exactly", t_equal == 0.0, f"(got {t_equal})")

    # ---- 5. Kepler potential reproduces the analytic radial free-fall time ----
    from galpy.potential import KeplerPotential
    M_kepler = 1e11 * u.Msun
    pot_kepler = KeplerPotential(amp=M_kepler, ro=cfg.galpy_ro_kpc * u.kpc, vo=cfg.galpy_vo_kms * u.km / u.s)
    r_now_k, r_peri_k = 50.0, 5.0
    phi_now_k = potential_kms2(pot_kepler, r_now_k)[0]
    v_now_k = float(np.sqrt(-2.0 * phi_now_k))  # E=0, parabolic infall
    t_quad_k = orbit_time_quadrature_myr(pot_kepler, r_peri_k, r_now_k, v_now_k)
    GM = (G * M_kepler).to(u.kpc**3 / u.Myr**2)
    t_analytic_k = ((2.0 / 3.0) * np.sqrt(1.0 / (2.0 * GM)) * ((r_now_k * u.kpc) ** 1.5 - (r_peri_k * u.kpc) ** 1.5)).to(u.Myr).value
    rel_k = abs(t_quad_k - t_analytic_k) / t_analytic_k
    check("5. Kepler potential reproduces the analytic parabolic free-fall time", rel_k < 1e-4,
          f"(quad={t_quad_k:.3f} Myr, analytic={t_analytic_k:.3f} Myr, rel diff={rel_k:.2e})")

    # ---- 6. theta=45 deg reproduces the naive R/V value to 1e-9 ----
    R_test, V_test = 40.0, 80.0
    t_naive_rv = (R_test / V_test) * CONVERSION_MYR
    t_theta45 = bridge_timescale_myr(R_test / V_test, 45.0)
    check("6. theta=45 deg reproduces the naive R/V value to 1e-9", abs(t_theta45 - t_naive_rv) < 1e-9 * max(t_naive_rv, 1.0),
          f"(naive={t_naive_rv:.6f}, theta45={t_theta45:.6f})")

    # ---- 7. Every ring parameter traces to the ringlog (spot check disk shear defaults) ----
    shear_default = disk_shear_timescale(cfg, rc)
    check("7. Disk shear r_in/r_out default to the ringlog's own min/max radius (no literal)",
          np.isclose(shear_default["r_in_kpc"], rc.radii_kpc.min()) and np.isclose(shear_default["r_out_kpc"], rc.radii_kpc.max()),
          f"(r_in={shear_default['r_in_kpc']:.2f}, r_out={shear_default['r_out_kpc']:.2f}, "
          f"ringlog min/max={rc.radii_kpc.min():.2f}/{rc.radii_kpc.max():.2f})")

    # ---- 8. dV_sys computed from the table matches the value used everywhere ----
    test_table = {"A": (4677.0, "this galaxy, Barolo VSYS"), "B": (4884.0, "synthetic companion, test fixture")}
    dV_direct = compute_dV_sys(test_table, "A", "B")
    cfg_pair_test = Config(
        trm_dir=trm_dir, ringlog_path=ringlog_path, results_dir=results_dir,
        vsys_table_kms=test_table, vsys_pair_label_a="A", vsys_pair_label_b="B",
        R_sep_kpc=50.0, bridge_pa_deg=None,
    )
    pair_lin_test = pair_separation_linear(cfg_pair_test, rc)
    check("8. dV_sys computed from the table matches the value used in pair_separation_linear",
          pair_lin_test is not None and abs(pair_lin_test["dV_sys_kms"] - dV_direct) < 1e-9,
          f"(direct={dV_direct}, used={pair_lin_test['dV_sys_kms'] if pair_lin_test else None})")
    pair_orbit_test = pair_orbit_report(cfg_pair_test, rc)
    check("8b. dV_sys computed from the table matches the value used in pair_orbit_report",
          pair_orbit_test is not None and abs(pair_orbit_test["dV_sys_kms"] - dV_direct) < 1e-9,
          f"(direct={dV_direct}, used={pair_orbit_test['dV_sys_kms'] if pair_orbit_test else None})")

    print(f"\n[selftest] {n_pass} passed, {n_fail} failed")
    return n_fail == 0


if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Every dynamical/interaction timescale quoted in the paper.")
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
