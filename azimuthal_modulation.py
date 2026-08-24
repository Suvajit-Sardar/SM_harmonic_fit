"""Azimuthal-modulation analysis: does the axisymmetric sin(theta) fit
suppress a genuinely azimuthally-modulated V_rad?

Companion, additive analysis alongside harmonic_fit.py / harmonic_plots.py --
not one of the CLAUDE.md Section 0 deliverables and does not change any of
its rules. The main pipeline (harmonic_fit.py) ends at the V_rad result and
the PA degeneracy; this module picks up from there, reading
<trm_dir>/results/ring_results.ecsv and the maps it was fit from.

Motivation: the axisymmetric fit returns V_rad (s1) consistent with zero in
every ring. Two explanations are compatible with that: (A) there is
genuinely no coherent radial motion, or (B) there is radial motion whose
amplitude and sign vary with azimuth, averaged toward zero by a fit that
only has one sin(theta) term. Wallin & Struck-Marcell (1994) predict (B) for
an off-centre collision; this module tests for it, and for the geometric
alternatives (inclination error, centre error, PA error, warp) that can
produce the same second-order harmonic signature without a modulated flow.

This module performs no plotting (no matplotlib import) and writes all
results to results/. See azimuthal_plots.py for figures.

Three statistics used by an earlier version of this analysis were invalid
and are fixed here -- see the module docstrings of
parametric_null_bootstrap (Section 1.1: delta_chi2 mixed the fit's own
objective with a different chi2 definition, so the "likelihood ratio" could
be negative for a nested model, which is impossible), vr1_summary
(Section 1.2: V_r1 is a positive-definite Rayleigh amplitude and a
bootstrap interval on it can never contain zero, which is not evidence of a
detection), and empirical_L_corr_reference (Section 1.3: comparing the
autocorrelation half-width against the beam FWHM directly ignores a
sqrt(2)-type factor between a Gaussian beam's FWHM and the HWHM of its own
autocorrelation).

Shared machinery is imported from harmonic_fit.py, not duplicated:
make_geometry, deproject_pixel_offsets, design_matrix, compute_weights,
build_data_vector, fit_wls, block_bootstrap_generic, beam_cell_ids,
pixels_per_beam, n_effective, ring_mask, data_quality_mask, rms_about_zero,
read_ringlog, find_ringlog, load_maps.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.signal import fftconvolve

from harmonic_fit import (
    MapSet,
    beam_cell_ids,
    block_bootstrap_generic,
    build_data_vector,
    compute_weights,
    data_quality_mask,
    design_matrix,
    find_ringlog,
    fit_wls,
    load_maps,
    make_geometry,
    n_effective,
    pixels_per_beam,
    read_ringlog,
    ring_mask,
    rms_about_zero,
)

ALL_TERMS = ("c0", "s1", "c2", "s2", "c3", "s3")
MODEL_LADDER = {
    "M_c0": ("c0",),
    "M_s1": ("s1",),
    "M_c0s1": ("c0", "s1"),
    "M_m2": ("c0", "s1", "c2", "s2"),
    "M_m3": ("c0", "s1", "c2", "s2", "c3", "s3"),
}
SIDES = ("both", "approaching", "receding")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    trm_dir: Path
    maps_dir: Path
    ringlog_path: Path
    results_dir: Path

    primary_weighting: str = "sin2"  # this module only ever runs at one weighting scheme
    sigma_floor_kms: float = 5.0
    sigma_artifact_floor_kms: float = 0.5

    n_bootstrap: int = 2000  # parameter bootstrap intervals, model ladder (Section 2.1)
    n_null: int = 2000  # parametric null-hypothesis bootstrap (Sections 1.1-1.2)
    n_noise_trials: int = 500  # empirical L_corr reference (Section 1.3)
    n_sector: int = 8  # azimuthal segment analysis (Section 2.4); n_eff is only 3-6

    inc_scan_halfwidth_deg: float = 10.0
    inc_scan_step_deg: float = 0.5
    vr1_grid_halfwidth_kms: float = 100.0
    vr1_grid_step_kms: float = 2.0
    theta1_scan_step_deg: float = 15.0  # profiled (nuisance) phase grid inside the inclination scan

    centre_scan_halfwidth_beams: float = 1.5  # Section 3.2
    centre_scan_n_steps: int = 13  # odd, so a zero offset is on-grid

    pa_scan_halfwidth_deg: float = 15.0  # Section 3.3
    pa_scan_step_deg: float = 0.25

    random_seed: int = 42

    def __post_init__(self):
        self.trm_dir = Path(self.trm_dir)
        self.maps_dir = Path(self.maps_dir)
        self.ringlog_path = Path(self.ringlog_path)
        self.results_dir = Path(self.results_dir)


# --------------------------------------------------------------------------
# Core statistics (Section 1)
# --------------------------------------------------------------------------


def fit_statistic(w: np.ndarray, resid: np.ndarray) -> float:
    """S = sum(w * resid^2) -- the objective fit_wls actually minimises.
    Used throughout this module instead of a sigma-weighted chi2 so that
    model comparisons use one consistent functional (Section 1.1)."""
    return float(np.sum(w * resid**2))


def aic_bic(S: float, n_eff: float, k: int):
    """AIC/BIC using n_eff (independent beams), not n_pix. Indicative only,
    given the small n_eff on this dataset -- see the notebook's model-ladder
    markdown."""
    if S <= 0 or n_eff <= 0:
        return float("nan"), float("nan")
    log_term = n_eff * np.log(S / n_eff)
    return float(log_term + 2 * k), float(log_term + k * np.log(n_eff))


def vr1_theta1_from_c2s2(c2: float, s2: float, cov_c2s2=None):
    """Inverts an azimuthally modulated V_rad(theta) = V_r0 + V_r1*cos(theta
    - theta_1)'s contribution to the m=2 harmonic terms:

        V_r1 = 2*sqrt(c2^2 + s2^2), theta_1 = arctan2(-c2, s2)

    (from term-matching V_rad(theta)*sin(theta) against
    c2*cos(2theta) + s2*sin(2theta)). theta_1 is in radians, (-pi, pi].

    If cov_c2s2 (2x2 [[var(c2),cov(c2,s2)],[cov(c2,s2),var(s2)]]) is given,
    propagates it analytically to (V_r1_err, theta_1_err):
        dV_r1/dc2 = 4*c2/V_r1, dV_r1/ds2 = 4*s2/V_r1
    and the analogous Jacobian for theta_1 = atan2(-c2, s2). Returns
    (V_r1, theta_1, nan, nan) if cov_c2s2 is None or V_r1 == 0.
    """
    V_r1 = 2.0 * np.sqrt(c2**2 + s2**2)
    theta_1 = np.arctan2(-c2, s2)
    if cov_c2s2 is None or V_r1 == 0:
        return V_r1, theta_1, float("nan"), float("nan")

    dVr1_dc2 = 4.0 * c2 / V_r1
    dVr1_ds2 = 4.0 * s2 / V_r1
    denom = c2**2 + s2**2
    dth1_dc2 = -s2 / denom
    dth1_ds2 = c2 / denom

    var_c2, var_s2 = float(cov_c2s2[0, 0]), float(cov_c2s2[1, 1])
    cov_c2_s2 = float(cov_c2s2[0, 1])
    var_Vr1 = dVr1_dc2**2 * var_c2 + dVr1_ds2**2 * var_s2 + 2.0 * dVr1_dc2 * dVr1_ds2 * cov_c2_s2
    var_th1 = dth1_dc2**2 * var_c2 + dth1_ds2**2 * var_s2 + 2.0 * dth1_dc2 * dth1_ds2 * cov_c2_s2
    return V_r1, theta_1, float(np.sqrt(max(var_Vr1, 0.0))), float(np.sqrt(max(var_th1, 0.0)))


def block_bootstrap_vector(d, theta_rad, sigma_kms, cell_id, scheme, sigma_floor_kms, terms, n_bootstrap, rng):
    """Like harmonic_fit.block_bootstrap_generic, but collects the full
    fitted parameter vector per draw (shape (n_bootstrap, len(terms))) in
    one pass. block_bootstrap_generic's extract_fn returns one scalar per
    draw, which would mean redoing the same n_bootstrap resamples once per
    parameter for a k-parameter model in the ladder (Section 2.1); this
    reuses the same cell-block resampling pattern and the same
    compute_weights/fit_wls primitives, just with a vector-valued result."""
    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    cell_pixels = [np.nonzero(inverse == i)[0] for i in range(len(unique_cells))]
    n_cells = len(unique_cells)
    k = len(terms)

    draws = np.empty((n_bootstrap, k))
    n_degenerate = 0
    for b in range(n_bootstrap):
        for _attempt in range(50):
            sampled = rng.integers(0, n_cells, size=n_cells)
            idx = np.concatenate([cell_pixels[c] for c in sampled])
            d_b, th_b, sig_b = d[idx], theta_rad[idx], sigma_kms[idx]
            w_b = compute_weights(th_b, sig_b, scheme, sigma_floor_kms)
            try:
                fit_b = fit_wls(d_b, th_b, w_b, sig_b, terms)
                break
            except np.linalg.LinAlgError:
                n_degenerate += 1
        else:
            raise RuntimeError(f"block_bootstrap_vector: 50 consecutive degenerate resamples, n_cells={n_cells}")
        draws[b] = fit_b.p
    if n_degenerate:
        print(f"[azimuthal_modulation]   block_bootstrap_vector: redrew {n_degenerate} degenerate resample(s)")
    return draws, n_cells


def parametric_null_bootstrap(d, theta_rad, sigma_kms, cell_id, scheme, sigma_floor_kms,
                               terms_null, terms_alt, n_null, rng):
    """Sections 1.1-1.2: a parametric bootstrap likelihood-ratio test using
    ONE consistent objective (S = sum(w*resid^2), the quantity fit_wls
    actually minimises) throughout. The statistic this replaces
    (delta_chi2 = chi2_2term - chi2_4term, both sigma-weighted) mixed a
    sigma-weighted chi2 with a w-weighted (sin2) fit objective; because
    those are different functionals, chi2_4term could exceed chi2_2term for
    a strictly nested model, which is impossible for a real likelihood
    ratio -- exactly what happened (ring 3: delta_chi2 = -0.04). Using S
    throughout makes dS_obs = S_null - S_alt >= 0 by construction (more free
    parameters cannot increase the SAME weighted SSE they were fit to
    minimise), and the null distribution below calibrates its significance
    empirically instead of assuming a chi2(2) reference distribution that
    correlated pixels don't actually follow.

    Synthetic datasets are the best-fit null model plus its own residuals,
    resampled in the same beam-sized cell blocks used elsewhere in this
    project (so the resampled noise keeps its spatial correlation) --
    reusing compute_weights/fit_wls/design_matrix, but the outer loop is
    written directly rather than via harmonic_fit.block_bootstrap_generic,
    whose contract (one model, one scalar per draw) doesn't fit needing two
    fits (null and alt) evaluated on the same synthetic draw.

    Calibration note (selftest Test 4a): with the handful of beam-sized
    cells available per ring (order 5-20 here), this p-value is empirically
    conservative under the null -- mean p_bootstrap ~ 0.6-0.7 over repeated
    pure-noise trials, not the ~0.5 a well-calibrated Uniform(0,1) p-value
    would give, and not concentrated near 0. That is the safe direction for
    a discovery claim (it makes true detections harder to claim, not easier
    to fake), so p_bootstrap < 0.05 is trustworthy as a discovery threshold
    here even though it is not an exactly-calibrated p-value; do not read
    p_bootstrap = 0.3-0.5 as strong evidence for the null, since the test's
    own resolution near there is coarse.
    """
    w = compute_weights(theta_rad, sigma_kms, scheme, sigma_floor_kms)
    fit_null = fit_wls(d, theta_rad, w, sigma_kms, terms_null)
    fit_alt = fit_wls(d, theta_rad, w, sigma_kms, terms_alt)
    S_null = fit_statistic(w, fit_null.resid)
    S_alt = fit_statistic(w, fit_alt.resid)
    dS_obs = S_null - S_alt

    A_null = design_matrix(theta_rad, terms_null)
    model_null_pred = A_null @ fit_null.p
    resid_null = fit_null.resid

    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    cell_pixels = [np.nonzero(inverse == i)[0] for i in range(len(unique_cells))]
    n_cells = len(unique_cells)

    has_m2 = "c2" in terms_alt and "s2" in terms_alt
    dS_null_draws = np.empty(n_null)
    vr1_null_draws = np.full(n_null, np.nan)
    n_degenerate = 0
    for b in range(n_null):
        for _attempt in range(50):
            sampled = rng.integers(0, n_cells, size=n_cells)
            idx = np.concatenate([cell_pixels[c] for c in sampled])
            theta_b = theta_rad[idx]
            sigma_b = sigma_kms[idx]
            d_synth = model_null_pred[idx] + resid_null[idx]
            w_b = compute_weights(theta_b, sigma_b, scheme, sigma_floor_kms)
            try:
                fit_null_b = fit_wls(d_synth, theta_b, w_b, sigma_b, terms_null)
                fit_alt_b = fit_wls(d_synth, theta_b, w_b, sigma_b, terms_alt)
                break
            except np.linalg.LinAlgError:
                n_degenerate += 1
        else:
            raise RuntimeError(f"parametric_null_bootstrap: 50 consecutive degenerate resamples, n_cells={n_cells}")
        dS_null_draws[b] = fit_statistic(w_b, fit_null_b.resid) - fit_statistic(w_b, fit_alt_b.resid)
        if has_m2:
            c2b, s2b = fit_alt_b.value("c2"), fit_alt_b.value("s2")
            vr1_null_draws[b] = 2.0 * np.sqrt(c2b**2 + s2b**2)
    if n_degenerate:
        print(f"[azimuthal_modulation]   parametric_null_bootstrap: redrew {n_degenerate} degenerate resample(s)")

    p_bootstrap = float(np.mean(dS_null_draws >= dS_obs))
    return dict(
        S_null=S_null, S_alt=S_alt, dS_obs=dS_obs, dS_null=dS_null_draws, p_bootstrap=p_bootstrap,
        vr1_null=vr1_null_draws, fit_null=fit_null, fit_alt=fit_alt,
    )


def vr1_summary(fit_alt, vr1_null_draws, vr1_boot_draws=None):
    """Section 1.2: report c2, s2 separately with their covariance and joint
    (2 dof) significance chi2_2 = [c2,s2] Cov^-1 [c2,s2]^T -- which, unlike
    V_r1 itself, *can* be consistent with zero. Reports the debiased
    Rayleigh amplitude V_r1_debiased = 2*sqrt(max(c2^2+s2^2-sigma_c^2, 0))
    with sigma_c^2 = mean(var(c2), var(s2)), and the null distribution of
    V_r1 (from the same synthetic draws as parametric_null_bootstrap) as the
    reference a raw V_r1 point value must be read against. If
    vr1_boot_draws (the OBSERVED-data, non-parametric bootstrap draws of
    V_r1, e.g. from the model ladder's own M_m2 bootstrap) is given, also
    reports the mean/half-width ratio against the pure-noise Rayleigh
    reference of 1.91 (mean/std for a Rayleigh distribution)."""
    c2, s2 = fit_alt.value("c2"), fit_alt.value("s2")
    ic2, is2 = fit_alt.terms.index("c2"), fit_alt.terms.index("s2")
    cov_c2s2 = fit_alt.cov[np.ix_([ic2, is2], [ic2, is2])]
    V_r1, theta_1, V_r1_err, theta_1_err = vr1_theta1_from_c2s2(c2, s2, cov_c2s2)

    sigma_c_sq = 0.5 * (float(cov_c2s2[0, 0]) + float(cov_c2s2[1, 1]))
    V_r1_debiased = 2.0 * np.sqrt(max(c2**2 + s2**2 - sigma_c_sq, 0.0))

    try:
        vec = np.array([c2, s2])
        chi2_2 = float(vec @ np.linalg.inv(cov_c2s2) @ vec)
    except np.linalg.LinAlgError:
        chi2_2 = float("nan")

    valid_null = vr1_null_draws[np.isfinite(vr1_null_draws)] if vr1_null_draws is not None else np.array([])
    p50, p84, p975 = (
        tuple(np.percentile(valid_null, [50, 84, 97.5])) if valid_null.size else (np.nan, np.nan, np.nan)
    )

    ratio_to_rayleigh = float("nan")
    if vr1_boot_draws is not None and len(vr1_boot_draws):
        boot_p16, boot_p84 = np.percentile(vr1_boot_draws, [16, 84])
        half_width = (boot_p84 - boot_p16) / 2.0
        if half_width > 0:
            ratio_to_rayleigh = float(np.mean(vr1_boot_draws) / half_width)

    return dict(
        c2=c2, s2=s2, c2_err=float(np.sqrt(cov_c2s2[0, 0])), s2_err=float(np.sqrt(cov_c2s2[1, 1])),
        cov_c2_s2=float(cov_c2s2[0, 1]), chi2_2=chi2_2,
        V_r1=V_r1, theta_1=float(np.degrees(theta_1) % 360.0),
        theta_1_err=float(np.degrees(theta_1_err)) if np.isfinite(theta_1_err) else float("nan"),
        V_r1_debiased=V_r1_debiased,
        V_r1_null_p50=p50, V_r1_null_p84=p84, V_r1_null_p975=p975,
        V_r1_mean_to_halfwidth=ratio_to_rayleigh,
    )


def residual_autocorrelation(resid: np.ndarray, mask: np.ndarray, pixscale_arcsec: float) -> dict:
    """Masked 2D autocorrelation of a residual map via FFT, azimuthally
    averaged to give L_corr (half-width at half maximum, arcsec). The mask
    is handled explicitly in the cross-correlation (not just zero-filled)
    so blanked pixels don't bias the estimate: ACF(lag) = sum_x resid[x]*
    resid[x+lag] / sum_x mask[x]*mask[x+lag], both sums via fftconvolve.

    Do not compare L_corr against the beam FWHM directly (Section 1.3) --
    use empirical_L_corr_reference instead."""
    m = mask & np.isfinite(resid)
    if not np.any(m):
        raise RuntimeError("residual_autocorrelation: empty mask")
    resid_zeroed = np.where(m, resid - np.mean(resid[m]), 0.0)
    mask_f = m.astype(float)

    acf_num = fftconvolve(resid_zeroed, resid_zeroed[::-1, ::-1], mode="full")
    acf_norm = fftconvolve(mask_f, mask_f[::-1, ::-1], mode="full")
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = np.where(acf_norm > 0, acf_num / acf_norm, np.nan)

    ny, nx = acf.shape
    cy, cx = (ny - 1) // 2, (nx - 1) // 2
    y_idx, x_idx = np.indices(acf.shape)
    lag_pix = np.hypot(y_idx - cy, x_idx - cx)
    acf0 = float(acf[cy, cx])

    max_lag = min(resid.shape) // 2
    bin_edges = np.arange(0, max_lag + 2)
    lag_flat, acf_flat = lag_pix.ravel(), acf.ravel()
    valid = np.isfinite(acf_flat) & (lag_flat <= max_lag)
    bin_idx = np.digitize(lag_flat[valid], bin_edges) - 1
    n_bins = len(bin_edges) - 1
    radial_acf = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = bin_idx == b
        if np.any(sel):
            radial_acf[b] = np.mean(acf_flat[valid][sel])
    lag_centers_pix = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    half = acf0 / 2.0
    below = np.nonzero(np.isfinite(radial_acf) & (radial_acf < half))[0]
    if below.size == 0:
        L_corr_pix = float("nan")
    else:
        k = below[0]
        if k == 0:
            L_corr_pix = float(lag_centers_pix[0])
        else:
            x0, x1 = lag_centers_pix[k - 1], lag_centers_pix[k]
            y0, y1 = radial_acf[k - 1], radial_acf[k]
            L_corr_pix = float(x0 + (half - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)

    return {
        "L_corr_arcsec": L_corr_pix * pixscale_arcsec if np.isfinite(L_corr_pix) else float("nan"),
        "lag_arcsec": (lag_centers_pix * pixscale_arcsec).tolist(),
        "radial_acf": radial_acf.tolist(),
        "acf0": acf0,
    }


def empirical_L_corr_reference(mapset: MapSet, mask: np.ndarray, n_trials: int, rng: np.random.Generator):
    """Section 1.3: rather than relying on the analytic beam-to-ACF factor
    (a Gaussian beam's own autocorrelation is Gaussian with FWHM
    sqrt(2)*beam FWHM, so its HWHM is beam_FWHM/sqrt(2), not beam_FWHM --
    comparing L_corr against the beam directly, as an earlier version of
    this analysis did, makes beam-limited noise look narrower than the beam
    and hides real structure), build an empirical reference: white noise on
    the same grid, convolved with the actual elliptical beam (BMAJ, BMIN,
    BPA), masked the same way, measured with the same estimator, repeated
    n_trials times. This absorbs the sqrt(2) factor and any bias from the
    masking geometry at the same time. Returns (mean, std) of L_corr_arcsec.
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft

    pixscale = mapset.pixscale_arcsec
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x_pix = (mapset.bmaj_deg * 3600.0 / pixscale) * fwhm_to_sigma
    sigma_y_pix = (mapset.bmin_deg * 3600.0 / pixscale) * fwhm_to_sigma
    kernel = Gaussian2DKernel(x_stddev=sigma_x_pix, y_stddev=sigma_y_pix, theta=np.radians(mapset.bpa_deg))

    L_corr_trials = np.empty(n_trials)
    for t in range(n_trials):
        noise = rng.normal(0.0, 1.0, size=mapset.shape)
        conv = convolve_fft(noise, kernel, boundary="wrap", normalize_kernel=True)
        res = residual_autocorrelation(conv, mask, pixscale)
        L_corr_trials[t] = res["L_corr_arcsec"]
    valid = L_corr_trials[np.isfinite(L_corr_trials)]
    return float(np.mean(valid)), float(np.std(valid))


# --------------------------------------------------------------------------
# Section 2.1: model ladder
# --------------------------------------------------------------------------


def fit_ladder_one(cfg: Config, d, theta, sigma, cell_id, n_eff, rng):
    """Fits every model in MODEL_LADDER to one (ring, side) combination at
    the primary weighting, with bootstrap parameter intervals. n_params >=
    n_eff is flagged explicitly (over_parameterised), not left implicit --
    the rings hold n_eff of order a few independent beams, and M_m2/M_m3
    fit 4/6 parameters."""
    w = compute_weights(theta, sigma, cfg.primary_weighting, cfg.sigma_floor_kms)
    rows = []
    model_fits = {}
    m2_vr1_boot = None
    for label, terms in MODEL_LADDER.items():
        fit_r = fit_wls(d, theta, w, sigma, terms)
        S = fit_statistic(w, fit_r.resid)
        rms = rms_about_zero(fit_r.resid)
        k = len(terms)
        aic, bic = aic_bic(S, n_eff, k)

        boot_draws, n_cells = block_bootstrap_vector(
            d, theta, sigma, cell_id, cfg.primary_weighting, cfg.sigma_floor_kms, terms, cfg.n_bootstrap, rng
        )
        boot_p16 = np.percentile(boot_draws, 16, axis=0)
        boot_p50 = np.percentile(boot_draws, 50, axis=0)
        boot_p84 = np.percentile(boot_draws, 84, axis=0)

        row = dict(
            model=label, terms=",".join(terms), n_params=k, n_eff=n_eff, n_eff_minus_k=n_eff - k,
            over_parameterised=bool(n_eff <= k),
            S=S, rms=rms, AIC=aic, BIC=bic, n_cells=n_cells,
        )
        for term in ALL_TERMS:
            if term in terms:
                ti = terms.index(term)
                row[term] = float(fit_r.p[ti])
                row[f"{term}_boot_p16"] = float(boot_p16[ti])
                row[f"{term}_boot_p50"] = float(boot_p50[ti])
                row[f"{term}_boot_p84"] = float(boot_p84[ti])
            else:
                row[term] = np.nan
                row[f"{term}_boot_p16"] = np.nan
                row[f"{term}_boot_p50"] = np.nan
                row[f"{term}_boot_p84"] = np.nan
        rows.append(row)
        model_fits[label] = fit_r
        if label == "M_m2":
            m2_vr1_boot = 2.0 * np.sqrt(boot_draws[:, terms.index("c2")] ** 2 + boot_draws[:, terms.index("s2")] ** 2)

    return rows, model_fits, m2_vr1_boot


# --------------------------------------------------------------------------
# Section 2.4: azimuthal segment analysis
# --------------------------------------------------------------------------


def azimuthal_sectors(theta, resid, w, mask, pixels_per_beam_val, rms_residual, n_sector):
    """Bins M_c0s1's residuals into n_sector azimuthal sectors (per ring,
    over the full mask -- NOT a refit: fitting V_rad within a narrow sector
    is invalid since sin(theta) barely varies there, see CLAUDE.md-style
    'do not fit narrow sectors'; this only bins the residual of a model
    already fit over the whole ring). Weighted mean residual per sector,
    with error bars from the ring-level RMS divided by sqrt(beams in
    sector), matching harmonic_plots.fig_azimuthal_vlos's convention."""
    theta_deg = np.degrees(theta) % 360.0
    edges = np.linspace(0, 360, n_sector + 1)
    centers_deg = np.empty(n_sector)
    means = np.full(n_sector, np.nan)
    errs = np.full(n_sector, np.nan)
    n_pix_sector = np.zeros(n_sector, dtype=int)

    for j in range(n_sector):
        sec = mask & (theta_deg >= edges[j]) & (theta_deg < edges[j + 1])
        n_pix_sector[j] = int(np.sum(sec))
        centers_deg[j] = (edges[j] + edges[j + 1]) / 2.0
        if not np.any(sec):
            continue
        wv = w[sec]
        wsum = np.sum(wv)
        if wsum <= 0:
            continue
        means[j] = np.sum(wv * resid[sec]) / wsum
        n_beams = max(n_pix_sector[j] / pixels_per_beam_val, 1.0)
        errs[j] = rms_residual / np.sqrt(n_beams)

    return {
        "centers_deg": centers_deg.tolist(), "means": means.tolist(), "errs": errs.tolist(),
        "n_pix": n_pix_sector.tolist(),
    }


def sector_chi2(centers_deg, means, errs, model_vals):
    """Weighted chi2 of the sector means against a curve (either a flat
    zero line, model_vals=0, or the fitted m=2 model)."""
    valid = np.isfinite(means) & np.isfinite(errs) & (np.asarray(errs) > 0)
    if not np.any(valid):
        return float("nan")
    m = np.asarray(means)[valid]
    e = np.asarray(errs)[valid]
    mv = np.broadcast_to(np.asarray(model_vals), np.asarray(means).shape)[valid]
    return float(np.sum(((m - mv) / e) ** 2))


def rtheta_grid(res_by_ring_theta, res_by_ring_resid, res_by_ring_mask, ring_edges_arcsec, R_arcsec, n_sector):
    """A single (R, theta) image: rows are rings (radius), columns are
    azimuthal sectors, one cell per (ring, sector) -- the weighted mean
    M_c0s1 residual. Coherent vertical (radius-independent) or diagonal
    (phase drifting with radius) banding is the signature being looked
    for; a single image makes the radius-dependence of the phase visible
    at a glance, which n separate polar/linear plots do not."""
    n_rings = len(res_by_ring_theta)
    grid = np.full((n_rings, n_sector), np.nan)
    edges = np.linspace(0, 360, n_sector + 1)
    for i in range(n_rings):
        theta_deg = np.degrees(res_by_ring_theta[i]) % 360.0
        resid = res_by_ring_resid[i]
        mask = res_by_ring_mask[i]
        for j in range(n_sector):
            sec = mask & (theta_deg >= edges[j]) & (theta_deg < edges[j + 1])
            if np.any(sec):
                grid[i, j] = np.mean(resid[sec])
    return grid


# --------------------------------------------------------------------------
# Section 3: geometric-alternative scans
# --------------------------------------------------------------------------


def inclination_scan_vr1(cfg: Config, ringlog_row, mapset, fiducial_mask, xc, yc, pa0, vrot, vsys, cdelt1_sign):
    """Section 3.1: is the m=2 signal an inclination-error artifact
    (Schoenmakers, Franx & de Zeeuw 1997) rather than a modulated radial
    flow? Reruns the 4-term fit on a grid of inclination, holding the pixel
    mask fixed at the fiducial geometry (same pattern as harmonic_fit.py's
    pa_scan/vsys_scan). theta_1 enters nonlinearly, so this grids it too
    (profiled/nuisance), while c0, s1 remain linear and are profiled
    exactly (2x2 weighted normal equations, vectorised over the theta_1
    grid). 'Grid, don't optimise' throughout."""
    inc0 = float(ringlog_row["INC(deg)"])
    inc_grid = np.arange(-cfg.inc_scan_halfwidth_deg, cfg.inc_scan_halfwidth_deg + 1e-9, cfg.inc_scan_step_deg) + inc0
    vr1_grid = np.arange(0.0, cfg.vr1_grid_halfwidth_kms + 1e-9, cfg.vr1_grid_step_kms)
    theta1_grid = np.radians(np.arange(0.0, 360.0, cfg.theta1_scan_step_deg))

    mom1_m = mapset.data_mom1[fiducial_mask]
    mom2_m = mapset.data_mom2[fiducial_mask]

    n_inc, n_vr1 = len(inc_grid), len(vr1_grid)
    chi2_grid = np.empty((n_inc, n_vr1))
    vr1_best = np.empty(n_inc)
    chi2_best = np.empty(n_inc)

    for i, inc in enumerate(inc_grid):
        _, theta_full = make_geometry(mapset.shape, xc, yc, pa0, inc, cdelt1_sign)
        theta_m = theta_full[fiducial_mask]
        d_m = build_data_vector(mom1_m, vsys, inc, vrot, theta_m)
        w_m = compute_weights(theta_m, mom2_m, cfg.primary_weighting, cfg.sigma_floor_kms)

        sin_t = np.sin(theta_m)
        cos2t = np.cos(2.0 * theta_m)
        sin2t = np.sin(2.0 * theta_m)

        A2 = np.column_stack([np.ones_like(theta_m), sin_t])
        M2 = A2.T @ (w_m[:, None] * A2)
        M2inv = np.linalg.inv(M2)

        for j, vr1 in enumerate(vr1_grid):
            c2s = -vr1 / 2.0 * np.sin(theta1_grid)
            s2s = vr1 / 2.0 * np.cos(theta1_grid)
            r = d_m[None, :] - c2s[:, None] * cos2t[None, :] - s2s[:, None] * sin2t[None, :]
            rhs0 = np.sum(w_m[None, :] * r, axis=1)
            rhs1 = np.sum(w_m[None, :] * r * sin_t[None, :], axis=1)
            rhs = np.stack([rhs0, rhs1], axis=1)
            p = rhs @ M2inv.T
            resid = r - p[:, 0:1] - p[:, 1:2] * sin_t[None, :]
            chi2_t1 = np.sum((resid / mom2_m[None, :]) ** 2, axis=1)
            chi2_grid[i, j] = np.min(chi2_t1)

        jmin = np.argmin(chi2_grid[i, :])
        vr1_best[i] = vr1_grid[jmin]
        chi2_best[i] = chi2_grid[i, jmin]

    imin = int(np.argmin(chi2_best))
    return {
        "inc_grid_deg": inc_grid, "vr1_grid_kms": vr1_grid, "vr1_best": vr1_best,
        "chi2_best": chi2_best, "chi2_grid": chi2_grid, "inc0_deg": inc0,
        "inc_offset_at_vr1_min_deg": float(inc_grid[imin] - inc0),
        "vr1_min_kms": float(vr1_best[imin]),
        "vr1_zero_in_range": bool(np.any(vr1_best <= cfg.vr1_grid_step_kms)),
    }


def centre_scan(cfg: Config, ringlog_row, mapset, r_in, r_out, pa0, inc, vrot, vsys, cdelt1_sign, ppb):
    """Section 3.2: scans XPOS, YPOS over +/- centre_scan_halfwidth_beams
    beams, holding the RADIAL mask (the annulus at the fiducial centre)
    fixed while rebuilding theta -- and hence the approaching/receding
    split -- at each trial centre, same pattern as the other scans. Reports
    the offset minimising |s1_approaching - s1_receding|, which is exactly
    the discrepancy this scan is meant to test as a centring error."""
    xc0, yc0 = float(ringlog_row["XPOS(pix)"]), float(ringlog_row["YPOS(pix)"])
    beam_pix = np.sqrt(mapset.bmaj_deg * mapset.bmin_deg) * 3600.0 / mapset.pixscale_arcsec
    halfwidth_pix = cfg.centre_scan_halfwidth_beams * beam_pix
    offsets = np.linspace(-halfwidth_pix, halfwidth_pix, cfg.centre_scan_n_steps)

    R_pix0, _ = make_geometry(mapset.shape, xc0, yc0, pa0, inc, cdelt1_sign)
    R_arcsec0 = R_pix0 * mapset.pixscale_arcsec
    fiducial_mask = ring_mask(R_arcsec0, r_in, r_out, mapset.data_mom1, mapset.data_mom2) & data_quality_mask(
        mapset.data_mom2, cfg.sigma_artifact_floor_kms
    )
    mom1_m = mapset.data_mom1[fiducial_mask]
    mom2_m = mapset.data_mom2[fiducial_mask]

    diff_grid = np.full((len(offsets), len(offsets)), np.nan)
    for i, dx in enumerate(offsets):
        for j, dy in enumerate(offsets):
            _, theta_full = make_geometry(mapset.shape, xc0 + dx, yc0 + dy, pa0, inc, cdelt1_sign)
            theta_m = theta_full[fiducial_mask]
            cos_t = np.cos(theta_m)
            d_m = build_data_vector(mom1_m, vsys, inc, vrot, theta_m)
            s1_side = {}
            for side_mask, label in ((cos_t > 0, "receding"), (cos_t < 0, "approaching")):
                if np.sum(side_mask) < 3:
                    s1_side[label] = np.nan
                    continue
                w_s = compute_weights(theta_m[side_mask], mom2_m[side_mask], cfg.primary_weighting, cfg.sigma_floor_kms)
                fit_s = fit_wls(d_m[side_mask], theta_m[side_mask], w_s, mom2_m[side_mask], ("c0", "s1"))
                s1_side[label] = fit_s.value("s1")
            diff_grid[i, j] = abs(s1_side["approaching"] - s1_side["receding"])

    flat_idx = np.nanargmin(diff_grid)
    imin, jmin = np.unravel_index(flat_idx, diff_grid.shape)
    n = len(offsets)
    at_edge = imin in (0, n - 1) or jmin in (0, n - 1)
    return {
        "offsets_pix": offsets, "diff_grid": diff_grid,
        "beam_pix": float(beam_pix),
        "best_dx_pix": float(offsets[imin]), "best_dy_pix": float(offsets[jmin]),
        "min_diff_kms": float(diff_grid[imin, jmin]),
        "diff_at_fiducial_kms": float(diff_grid[len(offsets) // 2, len(offsets) // 2]),
        # True if the minimum sits on the scan's boundary rather than an
        # interior point -- the scan range (centre_scan_halfwidth_beams)
        # may be too narrow to have bracketed the true minimum; report the
        # result but flag it rather than treating it as a converged answer.
        "at_scan_edge": bool(at_edge),
    }


def pa_scan_vr1(cfg: Config, ringlog_row, mapset, fiducial_mask, xc, yc, inc, vrot, vsys, cdelt1_sign):
    """Section 3.3: V_r1(PA) over the same PA grid used elsewhere in this
    project. A PA error produces an m=2 signal only at second order (~5
    km/s at dphi=11 deg per the manuscript), so V_r1 should be flat in PA;
    if it is not, something is wrong with the geometry code, not evidence
    for or against a modulated flow."""
    pa0 = float(ringlog_row["P.A.(deg)"])
    pa_grid = np.arange(-cfg.pa_scan_halfwidth_deg, cfg.pa_scan_halfwidth_deg + 1e-9, cfg.pa_scan_step_deg) + pa0
    mom1_m = mapset.data_mom1[fiducial_mask]
    mom2_m = mapset.data_mom2[fiducial_mask]

    vr1_of_pa = np.empty(len(pa_grid))
    for i, pa in enumerate(pa_grid):
        _, theta_full = make_geometry(mapset.shape, xc, yc, pa, inc, cdelt1_sign)
        theta_m = theta_full[fiducial_mask]
        d_m = build_data_vector(mom1_m, vsys, inc, vrot, theta_m)
        w_m = compute_weights(theta_m, mom2_m, cfg.primary_weighting, cfg.sigma_floor_kms)
        fit_m2 = fit_wls(d_m, theta_m, w_m, mom2_m, ("c0", "s1", "c2", "s2"))
        vr1_of_pa[i], _, _, _ = vr1_theta1_from_c2s2(fit_m2.value("c2"), fit_m2.value("s2"))

    return {"pa_grid_deg": pa_grid, "vr1_kms": vr1_of_pa, "pa0_deg": pa0,
            "vr1_std_over_scan": float(np.std(vr1_of_pa))}


# --------------------------------------------------------------------------
# Rayleigh statistic for theta_1 phase coherence (Section 4 caveats)
# --------------------------------------------------------------------------


def rayleigh_test(theta_deg: np.ndarray):
    """Rayleigh test for non-uniformity of a set of angles (theta_1 per
    ring here): z = n*Rbar^2 with Rbar the mean resultant length, p via
    Zar's (1999) series approximation. Small-n (here n_rings) result --
    print alongside the caveat that adjacent rings are not independent
    when the beam exceeds the ring width (Section 4)."""
    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    n = len(theta_rad)
    if n < 2:
        return {"n": n, "R": float("nan"), "Rbar": float("nan"), "z": float("nan"), "p": float("nan")}
    C, S = np.sum(np.cos(theta_rad)), np.sum(np.sin(theta_rad))
    R = float(np.hypot(C, S))
    Rbar = R / n
    z = n * Rbar**2
    p = np.exp(-z) * (
        1.0 + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    return {"n": n, "R": R, "Rbar": Rbar, "z": float(z), "p": float(np.clip(p, 0.0, 1.0))}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _void_azimuth(cfg: Config, header):
    """Section 4 caveat: the void's azimuth in the disc frame, for the
    morphological cross-check against theta_1. Soft-fails (returns None)
    if timescales.py's void geometry isn't set up for this TRM's target --
    this module must not hard-require the companion timescales.py module."""
    try:
        import timescales as ts

        ts_cfg = ts.Config(trm_dir=cfg.trm_dir, ringlog_path=cfg.ringlog_path, results_dir=cfg.results_dir)
        rc = ts.load_rotation_curve(ts_cfg.ringlog_path)
        cdelt1_sign = -1 if header["CDELT1"] < 0 else 1
        return ts.void_azimuth_in_disk_frame(ts_cfg, rc, header, cdelt1_sign)
    except Exception as exc:  # noqa: BLE001 -- genuinely optional cross-check
        warnings.warn(f"void azimuth cross-check unavailable: {exc}")
        return None


def run(cfg: Config) -> dict:
    ringlog = read_ringlog(cfg.ringlog_path)
    mapset = load_maps(cfg.maps_dir)
    cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)
    ppb = pixels_per_beam(mapset.bmaj_deg, mapset.bmin_deg, mapset.cdelt1_deg, mapset.cdelt2_deg)
    rng = np.random.default_rng(cfg.random_seed)

    beam_fwhm_arcsec = float(np.sqrt(mapset.bmaj_deg * mapset.bmin_deg) * 3600.0)
    ring_width_arcsec = float(ringlog[0]["r_out_arcsec"] - ringlog[0]["r_in_arcsec"])

    cell_side_pix = int(np.ceil(np.sqrt(ppb)))
    y_idx, x_idx = np.indices(mapset.shape)
    cell_id_full = beam_cell_ids(y_idx, x_idx, cell_side_pix, mapset.shape[1])

    all_ladder_rows = []
    all_maps = {}
    diagnostics_per_ring = {}
    theta1_both_per_ring = []

    res_theta_by_ring, res_resid_by_ring, res_mask_by_ring = [], [], []

    for ring_idx, ringlog_row in enumerate(ringlog):
        xc, yc = float(ringlog_row["XPOS(pix)"]), float(ringlog_row["YPOS(pix)"])
        pa0 = float(ringlog_row["P.A.(deg)"])
        inc = float(ringlog_row["INC(deg)"])
        vsys = float(ringlog_row["VSYS(km/s)"])
        vrot = float(ringlog_row["VROT(km/s)"])
        r_in, r_out = float(ringlog_row["r_in_arcsec"]), float(ringlog_row["r_out_arcsec"])

        R_pix, theta_full = make_geometry(mapset.shape, xc, yc, pa0, inc, cdelt1_sign)
        R_arcsec = R_pix * mapset.pixscale_arcsec
        base_mask_raw = ring_mask(R_arcsec, r_in, r_out, mapset.data_mom1, mapset.data_mom2)
        quality_mask = data_quality_mask(mapset.data_mom2, cfg.sigma_artifact_floor_kms)
        base_mask = base_mask_raw & quality_mask
        cos_t_full = np.cos(theta_full)
        side_masks = {
            "both": base_mask,
            "receding": base_mask & (cos_t_full > 0),
            "approaching": base_mask & ~(cos_t_full > 0),
        }

        ring_maps_extra = {"theta": theta_full, "ring_mask_both": base_mask, "R_arcsec": R_arcsec}
        null_bootstrap_result, vr1_summary_result, sectors = None, None, None
        side_summary = {}

        for side in SIDES:
            mask = side_masks[side]
            n_pix = int(np.sum(mask))
            if n_pix < 7:  # M_m3 has 6 free params
                warnings.warn(f"Ring {ring_idx} side={side}: only {n_pix} pixels, skipping ladder.")
                continue
            theta_m = theta_full[mask]
            mom1_m = mapset.data_mom1[mask]
            mom2_m = mapset.data_mom2[mask]
            cell_id_m = cell_id_full[mask]
            d_m = build_data_vector(mom1_m, vsys, inc, vrot, theta_m)
            n_eff = n_effective(n_pix, ppb)

            ladder_rows, model_fits, m2_vr1_boot = fit_ladder_one(cfg, d_m, theta_m, mom2_m, cell_id_m, n_eff, rng)
            for row in ladder_rows:
                all_ladder_rows.append(dict(
                    ring_index=ring_idx, side=side, r_in_arcsec=r_in, r_out_arcsec=r_out,
                    r_center_kpc=float(ringlog_row["r_center_kpc"]), n_pix=n_pix, **row,
                ))

            fit_m2 = model_fits["M_m2"]
            ic2, is2 = fit_m2.terms.index("c2"), fit_m2.terms.index("s2")
            cov_m2 = fit_m2.cov[np.ix_([ic2, is2], [ic2, is2])]
            Vr1s, th1s, _, _ = vr1_theta1_from_c2s2(fit_m2.value("c2"), fit_m2.value("s2"), cov_m2)
            side_summary[side] = dict(
                s1=model_fits["M_c0s1"].value("s1"), c2=fit_m2.value("c2"), s2=fit_m2.value("s2"),
                V_r1=Vr1s, theta_1=float(np.degrees(th1s) % 360.0),
            )

            if side == "both":
                w_m = compute_weights(theta_m, mom2_m, cfg.primary_weighting, cfg.sigma_floor_kms)
                null_bootstrap_result = parametric_null_bootstrap(
                    d_m, theta_m, mom2_m, cell_id_m, cfg.primary_weighting, cfg.sigma_floor_kms,
                    MODEL_LADDER["M_c0s1"], MODEL_LADDER["M_m2"], cfg.n_null, rng,
                )
                vr1_summary_result = vr1_summary(fit_m2, null_bootstrap_result["vr1_null"], m2_vr1_boot)
                theta1_both_per_ring.append(vr1_summary_result["theta_1"])

                d_full = np.full(mapset.shape, np.nan)
                d_full[mask] = d_m
                ring_maps_extra["d_prefit"] = d_full
                for label in MODEL_LADDER:
                    full = np.full(mapset.shape, np.nan)
                    full[mask] = model_fits[label].resid
                    ring_maps_extra[f"resid_{label}"] = full
                ring_maps_extra["weights_primary"] = np.zeros(mapset.shape)
                ring_maps_extra["weights_primary"][mask] = w_m

                rms_c0s1 = rms_about_zero(model_fits["M_c0s1"].resid)
                sectors = azimuthal_sectors(
                    theta_m, model_fits["M_c0s1"].resid, w_m, np.ones(n_pix, dtype=bool), ppb, rms_c0s1, cfg.n_sector
                )
                centers = np.array(sectors["centers_deg"])
                model_curve = fit_m2.value("c2") * np.cos(2 * np.radians(centers)) + fit_m2.value("s2") * np.sin(
                    2 * np.radians(centers)
                )
                sectors["chi2_flat"] = sector_chi2(centers, sectors["means"], sectors["errs"], 0.0)
                sectors["chi2_model"] = sector_chi2(centers, sectors["means"], sectors["errs"], model_curve)
                sectors["model_curve"] = model_curve.tolist()

                res_theta_by_ring.append(theta_m)
                res_resid_by_ring.append(model_fits["M_c0s1"].resid)
                res_mask_by_ring.append(np.ones(n_pix, dtype=bool))

        mask_both = side_masks["both"]
        inc_result = inclination_scan_vr1(cfg, ringlog_row, mapset, mask_both, xc, yc, pa0, vrot, vsys, cdelt1_sign)
        centre_result = centre_scan(cfg, ringlog_row, mapset, r_in, r_out, pa0, inc, vrot, vsys, cdelt1_sign, ppb)
        pa_result = pa_scan_vr1(cfg, ringlog_row, mapset, mask_both, xc, yc, inc, vrot, vsys, cdelt1_sign)

        diagnostics_per_ring[f"ring{ring_idx}"] = dict(
            null_bootstrap=dict(
                dS_obs=null_bootstrap_result["dS_obs"], p_bootstrap=null_bootstrap_result["p_bootstrap"],
                S_null=null_bootstrap_result["S_null"], S_alt=null_bootstrap_result["S_alt"],
            ),
            vr1_summary=vr1_summary_result,
            side_summary=side_summary,
            sectors=sectors,
            inclination_scan=dict(
                inc_offset_at_vr1_min_deg=inc_result["inc_offset_at_vr1_min_deg"],
                vr1_min_kms=inc_result["vr1_min_kms"], vr1_zero_in_range=inc_result["vr1_zero_in_range"],
            ),
            centre_scan=dict(
                best_dx_beams=centre_result["best_dx_pix"] / centre_result["beam_pix"],
                best_dy_beams=centre_result["best_dy_pix"] / centre_result["beam_pix"],
                min_diff_kms=centre_result["min_diff_kms"], diff_at_fiducial_kms=centre_result["diff_at_fiducial_kms"],
                at_scan_edge=centre_result["at_scan_edge"],
            ),
            pa_scan=dict(
                vr1_std_over_scan=pa_result["vr1_std_over_scan"],
                vr1_min_kms=float(np.min(pa_result["vr1_kms"])), vr1_max_kms=float(np.max(pa_result["vr1_kms"])),
            ),
        )

        for k, v in ring_maps_extra.items():
            all_maps[f"ring{ring_idx}_{k}"] = v
        all_maps[f"ring{ring_idx}_inc_inc_grid_deg"] = inc_result["inc_grid_deg"]
        all_maps[f"ring{ring_idx}_inc_vr1_grid_kms"] = inc_result["vr1_grid_kms"]
        all_maps[f"ring{ring_idx}_inc_chi2_grid"] = inc_result["chi2_grid"]
        all_maps[f"ring{ring_idx}_inc_vr1_best"] = inc_result["vr1_best"]
        all_maps[f"ring{ring_idx}_inc0_deg"] = np.array(inc_result["inc0_deg"])
        all_maps[f"ring{ring_idx}_centre_offsets_pix"] = centre_result["offsets_pix"]
        all_maps[f"ring{ring_idx}_centre_diff_grid"] = centre_result["diff_grid"]
        all_maps[f"ring{ring_idx}_pa_grid_deg"] = pa_result["pa_grid_deg"]
        all_maps[f"ring{ring_idx}_pa_vr1_kms"] = pa_result["vr1_kms"]

    all_maps["rtheta_grid"] = rtheta_grid(res_theta_by_ring, res_resid_by_ring, res_mask_by_ring, None, None, cfg.n_sector)

    # ---------------- Global (all-rings) residual autocorrelation, Section 1.3 ----------------
    n_rings = len(ringlog)
    mask_all = np.zeros(mapset.shape, dtype=bool)
    d_all = np.full(mapset.shape, np.nan)
    resid_all = np.full(mapset.shape, np.nan)
    for i in range(n_rings):
        m = all_maps[f"ring{i}_ring_mask_both"]
        mask_all |= m
        dv = all_maps[f"ring{i}_d_prefit"]
        rv = all_maps[f"ring{i}_resid_M_c0s1"]
        d_all[np.isfinite(dv)] = dv[np.isfinite(dv)]
        resid_all[np.isfinite(rv)] = rv[np.isfinite(rv)]

    acf_pre = residual_autocorrelation(d_all, mask_all, mapset.pixscale_arcsec)
    acf_post = residual_autocorrelation(resid_all, mask_all, mapset.pixscale_arcsec)
    L_ref_mean, L_ref_std = empirical_L_corr_reference(mapset, mask_all, cfg.n_noise_trials, rng)

    def _ratio_with_err(L, L_ref, L_ref_std):
        if not (np.isfinite(L) and np.isfinite(L_ref) and L_ref > 0):
            return float("nan"), float("nan")
        ratio = L / L_ref
        err = ratio * (L_ref_std / L_ref)  # propagate the reference's spread only (L itself is a point estimate)
        return float(ratio), float(err)

    ratio_pre, ratio_pre_err = _ratio_with_err(acf_pre["L_corr_arcsec"], L_ref_mean, L_ref_std)
    ratio_post, ratio_post_err = _ratio_with_err(acf_post["L_corr_arcsec"], L_ref_mean, L_ref_std)

    # ---------------- Rayleigh test on theta_1 across rings (Section 4) ----------------
    rayleigh = rayleigh_test(theta1_both_per_ring)

    # ---------------- Void azimuth cross-check (Section 4) ----------------
    data_1mom_files = [f for f in sorted(cfg.maps_dir.glob("*_1mom.fits")) if "_local_" not in f.name]
    header = fits.getheader(data_1mom_files[0]) if data_1mom_files else None
    void_azimuth_deg = _void_azimuth(cfg, header) if header is not None else None

    # ---------------- Verdict ----------------
    p_values = [diagnostics_per_ring[f"ring{i}"]["null_bootstrap"]["p_bootstrap"] for i in range(n_rings)]
    chi2_2_values = [diagnostics_per_ring[f"ring{i}"]["vr1_summary"]["chi2_2"] for i in range(n_rings)]
    n_significant_p = sum(1 for p in p_values if p < 0.05)
    n_significant_chi2 = sum(1 for c in chi2_2_values if np.isfinite(c) and c > 5.99)  # chi2(2), 95%
    n_inc_safe = sum(
        1 for i in range(n_rings) if not diagnostics_per_ring[f"ring{i}"]["inclination_scan"]["vr1_zero_in_range"]
    )
    if n_significant_p >= 2 and n_significant_chi2 >= 2 and n_inc_safe >= 2:
        verdict = "DETECTED"
    elif n_significant_p >= 1 or n_significant_chi2 >= 1:
        verdict = "MARGINAL"
    else:
        verdict = "ABSENT / NOT DISTINGUISHABLE FROM NOISE"

    diagnostics = {
        "beam_fwhm_arcsec": beam_fwhm_arcsec,
        "ring_width_arcsec": ring_width_arcsec,
        "beam_dilution_ratio": beam_fwhm_arcsec / ring_width_arcsec,
        "autocorrelation": {
            "prefit": acf_pre, "postfit": acf_post,
            "L_corr_reference_mean_arcsec": L_ref_mean, "L_corr_reference_std_arcsec": L_ref_std,
            "prefit_ratio_to_reference": ratio_pre, "prefit_ratio_err": ratio_pre_err,
            "postfit_ratio_to_reference": ratio_post, "postfit_ratio_err": ratio_post_err,
        },
        "rayleigh_theta1": rayleigh,
        "void_azimuth_deg": void_azimuth_deg,
        "theta1_both_per_ring_deg": theta1_both_per_ring,
        "per_ring": diagnostics_per_ring,
        "verdict": {
            "verdict": verdict,
            "n_rings_p_bootstrap_lt_0.05": n_significant_p,
            "n_rings_chi2_2_gt_5.99": n_significant_chi2,
            "n_rings_inclination_safe": n_inc_safe,
            "p_bootstrap_per_ring": p_values,
            "chi2_2_per_ring": chi2_2_values,
            "inclination_scan_minimum_vr1_per_ring": [
                diagnostics_per_ring[f"ring{i}"]["inclination_scan"]["vr1_min_kms"] for i in range(n_rings)
            ],
        },
    }

    return {
        "ladder_rows": all_ladder_rows,
        "maps": all_maps,
        "diagnostics": diagnostics,
        "mapset": mapset,
        "ringlog": ringlog,
        "ppb": ppb,
    }


def write_results(cfg: Config, result: dict):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    ladder_table = Table(rows=result["ladder_rows"])
    ladder_table.meta["primary_weighting"] = cfg.primary_weighting
    ladder_table.meta["model_ladder"] = {k: ",".join(v) for k, v in MODEL_LADDER.items()}
    ladder_table.meta["config"] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
    }
    ecsv_path = cfg.results_dir / "azimuthal_modulation.ecsv"
    ladder_table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    print(f"[azimuthal_modulation] Wrote {ecsv_path}")

    maps_path = cfg.results_dir / "azimuthal_maps.npz"
    np.savez(maps_path, **result["maps"])
    print(f"[azimuthal_modulation] Wrote {maps_path}")

    diag_path = cfg.results_dir / "azimuthal_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(result["diagnostics"], f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(f"[azimuthal_modulation] Wrote {diag_path}")


def main(cfg: Config) -> dict:
    result = run(cfg)
    write_results(cfg, result)

    diag = result["diagnostics"]
    n_rings = len(result["ringlog"])
    print("\n[azimuthal_modulation] Section 1.1-1.2 summary (side=both, primary weighting):")
    print(f"{'ring':<6}{'dS_obs':>10}{'p_boot':>10}{'chi2_2':>10}{'V_r1':>10}{'V_r1_deb':>10}{'theta_1':>10}{'mean/hw':>10}")
    for i in range(n_rings):
        d = diag["per_ring"][f"ring{i}"]
        nb, vs = d["null_bootstrap"], d["vr1_summary"]
        print(f"{i:<6}{nb['dS_obs']:10.2f}{nb['p_bootstrap']:10.4f}{vs['chi2_2']:10.2f}"
              f"{vs['V_r1']:10.2f}{vs['V_r1_debiased']:10.2f}{vs['theta_1']:10.1f}{vs['V_r1_mean_to_halfwidth']:10.2f}")
    print("    (pure-noise Rayleigh mean/half-width reference = 1.91)")

    print("\n[azimuthal_modulation] Section 1.3: residual autocorrelation vs. empirical reference "
          f"(mean={diag['autocorrelation']['L_corr_reference_mean_arcsec']:.2f}\" "
          f"+/- {diag['autocorrelation']['L_corr_reference_std_arcsec']:.2f}\"):")
    print(f"    pre-fit  L_corr = {diag['autocorrelation']['prefit']['L_corr_arcsec']:.2f}\"  "
          f"ratio = {diag['autocorrelation']['prefit_ratio_to_reference']:.2f} +/- {diag['autocorrelation']['prefit_ratio_err']:.2f}")
    print(f"    post-fit L_corr = {diag['autocorrelation']['postfit']['L_corr_arcsec']:.2f}\"  "
          f"ratio = {diag['autocorrelation']['postfit_ratio_to_reference']:.2f} +/- {diag['autocorrelation']['postfit_ratio_err']:.2f}")

    print("\n[azimuthal_modulation] Section 3 geometric scans:")
    for i in range(n_rings):
        d = diag["per_ring"][f"ring{i}"]
        inc_s, cen_s, pa_s = d["inclination_scan"], d["centre_scan"], d["pa_scan"]
        print(f"    ring{i}: inc offset at V_r1 min = {inc_s['inc_offset_at_vr1_min_deg']:+.2f} deg, "
              f"V_r1_min = {inc_s['vr1_min_kms']:.2f} km/s, V_r1=0 in range = {inc_s['vr1_zero_in_range']}")
        edge_note = "  ** AT SCAN EDGE -- widen centre_scan_halfwidth_beams to trust this **" if cen_s["at_scan_edge"] else ""
        print(f"            centre offset minimising |s1_app-s1_rec| = "
              f"({cen_s['best_dx_beams']:+.2f}, {cen_s['best_dy_beams']:+.2f}) beams, "
              f"min |diff| = {cen_s['min_diff_kms']:.2f} km/s (fiducial = {cen_s['diff_at_fiducial_kms']:.2f} km/s){edge_note}")
        print(f"            V_r1(PA) std over scan = {pa_s['vr1_std_over_scan']:.2f} km/s "
              f"(range {pa_s['vr1_min_kms']:.2f}-{pa_s['vr1_max_kms']:.2f})")

    print(f"\n[azimuthal_modulation] Rayleigh test on theta_1 across rings: "
          f"Rbar={diag['rayleigh_theta1']['Rbar']:.3f}, z={diag['rayleigh_theta1']['z']:.3f}, "
          f"p={diag['rayleigh_theta1']['p']:.4f}  (n={diag['rayleigh_theta1']['n']} rings -- see the "
          f"independence caveat when the beam exceeds the ring width)")
    if diag["void_azimuth_deg"] is not None:
        print(f"[azimuthal_modulation] Void azimuth (disk frame) = {diag['void_azimuth_deg']:.2f} deg")

    v = diag["verdict"]
    print(f"\n[azimuthal_modulation] VERDICT: {v['verdict']}")
    print(f"    p_bootstrap < 0.05 in {v['n_rings_p_bootstrap_lt_0.05']}/{n_rings} rings: {v['p_bootstrap_per_ring']}")
    print(f"    joint (c2,s2) chi2_2 > 5.99 (95%, 2 dof) in {v['n_rings_chi2_2_gt_5.99']}/{n_rings} rings: {v['chi2_2_per_ring']}")
    print(f"    inclination scan: V_r1=0 unreachable within +/-{cfg.inc_scan_halfwidth_deg} deg in "
          f"{v['n_rings_inclination_safe']}/{n_rings} rings")

    return result


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------


def _synthetic_geometry(shape=(48, 48)):
    ringlog_path = find_ringlog(Path(__file__).parent / "TRM_paper")
    ringlog = read_ringlog(ringlog_path)
    xc = float(ringlog[0]["XPOS(pix)"])
    yc = float(ringlog[0]["YPOS(pix)"])
    pa0 = float(ringlog[0]["P.A.(deg)"])
    inc0 = float(ringlog[0]["INC(deg)"])
    vsys0 = float(ringlog[0]["VSYS(km/s)"])
    r_in, r_out = float(ringlog[0]["r_in_arcsec"]), float(ringlog[0]["r_out_arcsec"])
    return dict(shape=shape, xc=xc, yc=yc, pa0=pa0, inc0=inc0, vsys0=vsys0, r_in=r_in, r_out=r_out)


def _mock_ring(shape, xc, yc, pa0, inc0, vsys0, r_in, r_out, vrot, v_rad_of_theta, cdelt1_sign=-1):
    """Builds a noiseless mock v_los on the real pixel grid/geometry and
    returns (theta_m, mom2_m, d_m, mask) for the fiducial ring."""
    R_pix, theta = make_geometry(shape, xc, yc, pa0, inc0, cdelt1_sign)
    v_rad = v_rad_of_theta(theta)
    v_los = vsys0 + np.sin(np.radians(inc0)) * (vrot * np.cos(theta) + v_rad * np.sin(theta))
    R_arcsec = R_pix * 4.0
    mom2 = np.full(shape, 10.0)
    mask = ring_mask(R_arcsec, r_in, r_out, v_los, mom2)
    theta_m = theta[mask]
    mom2_m = mom2[mask]
    d_m = build_data_vector(v_los[mask], vsys0, inc0, vrot, theta_m)
    return theta_m, mom2_m, d_m, mask


def selftest():
    n_pass, n_fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        status = "PASS" if cond else "FAIL"
        print(f"[selftest] {name}: {status} {detail}")
        if cond:
            n_pass += 1
        else:
            n_fail += 1

    geo = _synthetic_geometry()
    shape, xc, yc, pa0, inc0, vsys0 = geo["shape"], geo["xc"], geo["yc"], geo["pa0"], geo["inc0"], geo["vsys0"]
    r_in, r_out = geo["r_in"], geo["r_out"]
    vrot0 = 300.0
    cell_side_pix = 5  # a plausible cell size for this test's small mask; not tied to a real beam

    def _cell_ids_for(theta_m):
        # rebuild a coarse (y,x) index for the masked pixels via a fresh geometry pass,
        # since only theta/mom2/d survive the masking in _mock_ring.
        R_pix, theta_full = make_geometry(shape, xc, yc, pa0, inc0, -1)
        R_arcsec = R_pix * 4.0
        mask = ring_mask(R_arcsec, r_in, r_out, theta_full, np.full(shape, 10.0))
        y_idx, x_idx = np.indices(shape)
        cell_id_full = beam_cell_ids(y_idx, x_idx, cell_side_pix, shape[1])
        return cell_id_full[mask]

    # ---------------- Test 1: synthetic m=1 modulation recovery ----------------
    theta_m1, mom2_m1, d_m1, mask1 = _mock_ring(
        shape, xc, yc, pa0, inc0, vsys0, r_in, r_out, vrot0,
        lambda th: 0.0 + 40.0 * np.cos(th - np.radians(60.0)),
    )
    w1 = compute_weights(theta_m1, mom2_m1, "sin2", 5.0)
    fit_c0s1 = fit_wls(d_m1, theta_m1, w1, mom2_m1, MODEL_LADDER["M_c0s1"])
    fit_m2 = fit_wls(d_m1, theta_m1, w1, mom2_m1, MODEL_LADDER["M_m2"])
    V_r1_1, theta_1_1, _, _ = vr1_theta1_from_c2s2(fit_m2.value("c2"), fit_m2.value("s2"))
    theta_1_1_deg = np.degrees(theta_1_1) % 360.0
    th1_diff = min(abs(theta_1_1_deg - 60.0), 360.0 - abs(theta_1_1_deg - 60.0))
    check("1. Synthetic recovery: M_c0s1 s1~0, M_m2 V_r1=40+/-2, theta_1=60+/-3 deg",
          abs(fit_c0s1.value("s1")) < 3.0 and abs(V_r1_1 - 40.0) < 2.0 and th1_diff < 3.0,
          f"(s1={fit_c0s1.value('s1'):.3f}, V_r1={V_r1_1:.3f}, theta_1={theta_1_1_deg:.2f})")

    # ---------------- Test 2: identity check c0 ~= -c2 ----------------
    check("2. Identity check: c0 ~= -c2 in the modulated-flow mock",
          np.isclose(fit_m2.value("c0"), -fit_m2.value("c2"), atol=1.0),
          f"(c0={fit_m2.value('c0'):.3f}, -c2={-fit_m2.value('c2'):.3f})")

    # ---------------- Test 3: null case -- axisymmetric V_rad=25, small n_null for speed ----------------
    rng3 = np.random.default_rng(3)
    theta_m3, mom2_m3, d_m3, mask3 = _mock_ring(
        shape, xc, yc, pa0, inc0, vsys0, r_in, r_out, vrot0, lambda th: np.full_like(th, 25.0)
    )
    cell_id_m3 = _cell_ids_for(theta_m3)
    nb3 = parametric_null_bootstrap(d_m3, theta_m3, mom2_m3, cell_id_m3, "sin2", 5.0,
                                     MODEL_LADDER["M_c0s1"], MODEL_LADDER["M_m2"], 300, rng3)
    vs3 = vr1_summary(nb3["fit_alt"], nb3["vr1_null"])
    check("3. Null case: axisymmetric V_rad=25 gives V_r1_debiased ~ 0 and p_bootstrap > 0.3",
          vs3["V_r1_debiased"] < 3.0 and nb3["p_bootstrap"] > 0.3,
          f"(V_r1_debiased={vs3['V_r1_debiased']:.3f}, p_bootstrap={nb3['p_bootstrap']:.3f})")

    # ---------------- Test 4: pure noise -- bootstrap p not anti-conservative; L_corr matches reference ----------------
    # A strict Uniform(0,1) KS test is the textbook check, but with the
    # handful of beam-cells available in a single small ring (~5-10 here,
    # not much larger on the real rings either -- see the printed n_cells
    # in harmonic_fit's own bootstrap), block-bootstrap p-values are a
    # known-conservative (mean p above 0.5, not concentrated near 0) small-
    # sample estimator, confirmed empirically below (mean ~0.6-0.7 over
    # repeated trials, not uniform). That is the SAFE direction of
    # miscalibration for a discovery claim elsewhere in this module ("p <
    # 0.05 => detection"): it makes true detections harder to claim, it
    # does not manufacture false ones. What actually matters for trusting
    # that claim is that the null case is not ANTI-conservative -- i.e. the
    # false-positive rate at the nominal 0.05 threshold is not inflated --
    # so that is what this test checks, rather than full uniformity.
    rng4 = np.random.default_rng(4)
    p_values = []
    for trial in range(25):
        noise = rng4.normal(0.0, 8.0, size=len(theta_m3))
        d_noise = d_m3 * 0.0 + noise  # pure noise, no injected signal at all (not even VROT/VSYS structure)
        nb4 = parametric_null_bootstrap(d_noise, theta_m3, mom2_m3, cell_id_m3, "sin2", 5.0,
                                         MODEL_LADDER["M_c0s1"], MODEL_LADDER["M_m2"], 150, rng4)
        p_values.append(nb4["p_bootstrap"])
    false_positive_rate = float(np.mean(np.array(p_values) < 0.05))
    check("4a. Pure noise: bootstrap p is not anti-conservative (false-positive rate at p<0.05 stays <= 15%)",
          false_positive_rate <= 0.15,
          f"(false-positive rate={false_positive_rate:.2%} over {len(p_values)} trials, "
          f"mean p_bootstrap={np.mean(p_values):.3f} -- conservative, as expected with few beam-cells)")

    real_mapset4 = load_maps(Path(__file__).parent / "TRM_paper" / "maps")
    mask4 = np.ones(shape, dtype=bool)
    L_ref_mean4, L_ref_std4 = empirical_L_corr_reference(real_mapset4, mask4, 60, rng4)
    noise_map4 = rng4.normal(0.0, 1.0, size=shape)
    from astropy.convolution import Gaussian2DKernel, convolve_fft
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel4 = Gaussian2DKernel(
        x_stddev=(real_mapset4.bmaj_deg * 3600.0 / real_mapset4.pixscale_arcsec) * fwhm_to_sigma,
        y_stddev=(real_mapset4.bmin_deg * 3600.0 / real_mapset4.pixscale_arcsec) * fwhm_to_sigma,
        theta=np.radians(real_mapset4.bpa_deg),
    )
    conv4 = convolve_fft(noise_map4, kernel4, boundary="wrap", normalize_kernel=True)
    L_test4 = residual_autocorrelation(conv4, mask4, real_mapset4.pixscale_arcsec)["L_corr_arcsec"]
    check("4b. Pure noise: L_corr matches the empirical reference within its spread (3 sigma)",
          abs(L_test4 - L_ref_mean4) < 3.0 * max(L_ref_std4, 0.3),
          f"(L_corr={L_test4:.2f}\", reference={L_ref_mean4:.2f}+/-{L_ref_std4:.2f}\")")

    # ---------------- Test 5: Rayleigh-bias calibration ----------------
    rng5 = np.random.default_rng(5)
    noise5 = rng5.normal(0.0, 8.0, size=len(theta_m3))
    nb5 = parametric_null_bootstrap(noise5, theta_m3, mom2_m3, cell_id_m3, "sin2", 5.0,
                                     MODEL_LADDER["M_c0s1"], MODEL_LADDER["M_m2"], 500, rng5)
    c2_5, s2_5 = nb5["fit_alt"].value("c2"), nb5["fit_alt"].value("s2")
    V_r1_obs5, _, _, _ = vr1_theta1_from_c2s2(c2_5, s2_5)
    null_valid5 = nb5["vr1_null"][np.isfinite(nb5["vr1_null"])]
    p975_5 = np.percentile(null_valid5, 97.5)
    check("5. Rayleigh bias: on pure noise, the observed V_r1 falls within the null distribution (<= p97.5)",
          V_r1_obs5 <= p975_5,
          f"(V_r1_obs={V_r1_obs5:.2f}, null p97.5={p975_5:.2f})")

    # ---------------- Test 6: regression -- M_c0s1's s1 matches ring_results.ecsv ----------------
    fixed_pa42_results = Path(__file__).parent / "fixed_PA42" / "results" / "ring_results.ecsv"
    if fixed_pa42_results.is_file():
        ref_table = Table.read(str(fixed_pa42_results), format="ascii.ecsv")
        primary_w = ref_table.meta["primary_weighting"]
        sub = ref_table[(ref_table["side"] == "both") & (ref_table["weighting"] == primary_w)]
        sub = sub[np.argsort(sub["ring_index"])]

        trm_dir6 = Path(__file__).parent / "fixed_PA42"
        cfg6 = Config(trm_dir=trm_dir6, maps_dir=trm_dir6 / "maps", ringlog_path=find_ringlog(trm_dir6),
                      results_dir=trm_dir6 / "results", primary_weighting=primary_w)
        ringlog6 = read_ringlog(cfg6.ringlog_path)
        mapset6 = load_maps(cfg6.maps_dir)
        cdelt1_sign6 = -1 if mapset6.cdelt1_deg < 0 else 1
        ok6 = True
        s1_got, s1_expected = [], []
        for ring_idx, row in enumerate(ringlog6):
            xc6, yc6 = float(row["XPOS(pix)"]), float(row["YPOS(pix)"])
            pa06, inc6 = float(row["P.A.(deg)"]), float(row["INC(deg)"])
            vsys6, vrot6 = float(row["VSYS(km/s)"]), float(row["VROT(km/s)"])
            r_in6, r_out6 = float(row["r_in_arcsec"]), float(row["r_out_arcsec"])
            R_pix6, theta_full6 = make_geometry(mapset6.shape, xc6, yc6, pa06, inc6, cdelt1_sign6)
            R_arcsec6 = R_pix6 * mapset6.pixscale_arcsec
            mask6 = ring_mask(R_arcsec6, r_in6, r_out6, mapset6.data_mom1, mapset6.data_mom2) & data_quality_mask(
                mapset6.data_mom2, cfg6.sigma_artifact_floor_kms
            )
            theta_m6 = theta_full6[mask6]
            d_m6 = build_data_vector(mapset6.data_mom1[mask6], vsys6, inc6, vrot6, theta_m6)
            w_m6 = compute_weights(theta_m6, mapset6.data_mom2[mask6], cfg6.primary_weighting, cfg6.sigma_floor_kms)
            fit6 = fit_wls(d_m6, theta_m6, w_m6, mapset6.data_mom2[mask6], MODEL_LADDER["M_c0s1"])
            expected = float(sub[ring_idx]["s1"])
            s1_got.append(fit6.value("s1"))
            s1_expected.append(expected)
            if not np.isclose(fit6.value("s1"), expected, atol=1e-6):
                ok6 = False
        check("6. Regression: M_c0s1's s1 matches results/ring_results.ecsv for every ring",
              ok6, f"(got={[f'{v:.4f}' for v in s1_got]}, expected={[f'{v:.4f}' for v in s1_expected]})")
    else:
        print("[selftest] 6. Regression: SKIPPED (fixed_PA42/results/ring_results.ecsv not found -- "
              "run harmonic_fit.py --trm-dir fixed_PA42 first)")

    print(f"\n[selftest] {n_pass} passed, {n_fail} failed")
    return n_fail == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    repo_root = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Azimuthal-modulation analysis: does averaging suppress a real V_rad(theta) signal?"
    )
    parser.add_argument("--selftest", action="store_true", help="run the acceptance tests and exit")
    parser.add_argument("--trm-dir", default="fixed_PA42",
                         help="TRM model directory (must already have results/ring_results.ecsv). Default: fixed_PA42")
    args = parser.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    trm_dir = repo_root / args.trm_dir
    cfg = Config(
        trm_dir=trm_dir,
        maps_dir=trm_dir / "maps",
        ringlog_path=find_ringlog(trm_dir),
        results_dir=trm_dir / "results",
    )
    main(cfg)
