"""First-order harmonic decomposition of a MeerKAT HI velocity field.

Measures ``s1 = V_rad(R)`` ring by ring from the data moment-1 map, holding
the rotation curve fixed at the 3D-Barolo tilted-ring values. All science
constants live in :class:`Config`; nothing is hardcoded elsewhere.

Weighting-scheme note: ``sin2`` is the primary scheme used throughout this
module, and its justification is *robustness, not optimality*. It is formally
less precise than uniform weighting under homoscedastic noise, but it
downweights the major axis -- where residuals are dominated by errors in the
fixed VROT, by beam smearing across the steep velocity gradient, and by any
warp -- more than it costs in raw precision. Do not describe it as optimal.

This module performs no plotting (no matplotlib import) and writes all
results to ``results/``. See ``harmonic_plots.py`` for figures.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

# --------------------------------------------------------------------------
# 5.1 Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    # project_dir is the TRM model directory being processed (e.g.
    # TRM_paper/, fixed_PA42/, or any new model directory the user adds) --
    # not the repo root. Each TRM model is self-contained: maps_dir and
    # ringlog_path live under it, and results_dir defaults to a "results"
    # subdirectory of it, so multiple models' outputs never collide. See
    # find_ringlog() / discover_trm_models() and the --trm-dir CLI flag.
    project_dir: Path
    maps_dir: Path
    ringlog_path: Path
    results_dir: Path

    weighting_schemes: tuple = ("uniform", "sin2", "invvar")
    primary_weighting: str = "sin2"
    model_terms: tuple = ("c0", "s1")  # c1 (VROT) is always fixed, never a column
    sigma_floor_kms: float = 5.0  # weight clipping only -- never applied to chi2/covariance sigma
    sigma_artifact_floor_kms: float = 0.5  # data-quality mask: exclude near-zero data mom2
    # (SoFiA linewidth collapse at marginal S/N, empirically < 4e-4 km/s, vs. real dispersion
    # >= 2.75 km/s -- a 4-order-of-magnitude gap. Distinct from sigma_floor_kms and from the
    # prohibited old ">= 5.0" cut; see CLAUDE.md 5.3.)

    pa_scan_halfwidth_deg: float = 15.0
    pa_scan_step_deg: float = 0.25
    vsys_scan_halfwidth_kms: float = 20.0
    vsys_scan_step_kms: float = 0.5
    s1_grid_halfwidth_kms: float = 80.0
    s1_grid_step_kms: float = 0.5

    n_bootstrap: int = 2000
    random_seed: int = 42

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        self.maps_dir = Path(self.maps_dir)
        self.ringlog_path = Path(self.ringlog_path)
        self.results_dir = Path(self.results_dir)
        if "s1" not in self.model_terms:
            raise ValueError("model_terms must include 's1' -- that is the quantity of interest.")


SIDES = ("both", "approaching", "receding")

_MODEL_TERM_FUNCS = {
    "c0": lambda theta: np.ones_like(theta),
    "s1": lambda theta: np.sin(theta),
    "c2": lambda theta: np.cos(2.0 * theta),
    "s2": lambda theta: np.sin(2.0 * theta),
    "c3": lambda theta: np.cos(3.0 * theta),
    "s3": lambda theta: np.sin(3.0 * theta),
}


def design_matrix(theta: np.ndarray, terms: Sequence[str]) -> np.ndarray:
    if len(terms) == 0:
        # np.column_stack([]) raises -- an empty design matrix is a legitimate
        # case here: it arises when "c0" is toggled out of model_terms (see
        # Config.model_terms) and the scan/profile code is left profiling over
        # zero nuisance terms.
        return np.zeros((len(theta), 0))
    return np.column_stack([_MODEL_TERM_FUNCS[t](theta) for t in terms])


# --------------------------------------------------------------------------
# 5.2 I/O
# --------------------------------------------------------------------------


REQUIRED_RINGLOG_COLUMNS = ("RAD(Kpc)", "RAD(arcs)", "VROT(km/s)", "E_VROT1", "E_VROT2")


def find_ringlog(trm_dir: Path) -> Path:
    """Locate the ringlog file inside a TRM model directory.

    Each TRM model (e.g. ``TRM_paper/``, ``fixed_PA42/``, or any new model
    directory the user adds) is self-contained: a ``maps/`` subdirectory plus
    one whitespace-delimited, '#'-commented-header ``.txt`` ringlog. The
    ringlog's filename is not fixed (``stage_1_opt_parameters.txt``,
    ``PA_fixed_to_42.txt`` are both seen in practice), so it is identified by
    its columns rather than its name. This also rejects BBarolo's
    ``*_initial.txt`` files (initial guesses only -- no ``RAD(Kpc)`` or
    ``E_VROT1/2`` columns, i.e. no fitted values), and any other stray
    ``.txt`` file (stats dumps, etc.) that might sit alongside it.
    """
    trm_dir = Path(trm_dir)
    candidates = []
    for f in sorted(trm_dir.glob("*.txt")):
        try:
            with open(f) as fh:
                header_line = fh.readline()
        except OSError:
            continue
        if not header_line.lstrip().startswith("#"):
            continue
        tokens = header_line.lstrip("#").split()
        if all(col in tokens for col in REQUIRED_RINGLOG_COLUMNS):
            candidates.append(f)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            f"No ringlog file found in {trm_dir} -- expected a whitespace-delimited, "
            f"'#'-commented-header .txt file with columns including {REQUIRED_RINGLOG_COLUMNS}. "
            "Pass --ringlog explicitly if the file uses different column names."
        )
    raise RuntimeError(
        f"Multiple candidate ringlog files found in {trm_dir}: {candidates}. "
        "Pass --ringlog explicitly to disambiguate."
    )


def discover_trm_models(root_dir: Path) -> dict:
    """Find TRM model directories under root_dir: any immediate subdirectory
    containing a maps/ folder and a ringlog findable by find_ringlog. Returns
    {name: path}, sorted by name. Used by --list-trm-models and to let the
    notebook config cell enumerate what's available without hardcoding names."""
    root_dir = Path(root_dir)
    models = {}
    for d in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        if not (d / "maps").is_dir():
            continue
        try:
            find_ringlog(d)
        except RuntimeError:
            continue
        models[d.name] = d
    return models


def read_ringlog(path: Path) -> Table:
    """Whitespace-delimited, '#'-comment header. Ring edges are attached as
    RAD +/- width/2 -- RAD is the ring center (confirmed by the project owner,
    reproduces the ring bounds of the previous script exactly)."""
    t = Table.read(str(path), format="ascii.commented_header")

    rad_arcs = np.asarray(t["RAD(arcs)"], dtype=float)
    rad_kpc = np.asarray(t["RAD(Kpc)"], dtype=float)

    diffs = np.diff(rad_arcs)
    if not np.allclose(diffs, diffs[0], rtol=1e-6):
        raise RuntimeError(f"Ring radii are not uniformly spaced: diffs = {diffs}")
    width = diffs[0]

    kpc_per_arcsec = rad_kpc / rad_arcs
    if not np.allclose(kpc_per_arcsec, kpc_per_arcsec[0], rtol=1e-3):
        raise RuntimeError(f"kpc_per_arcsec is inconsistent across rows: {kpc_per_arcsec}")

    t["r_in_arcsec"] = rad_arcs - width / 2.0
    t["r_out_arcsec"] = rad_arcs + width / 2.0
    t["r_center_kpc"] = rad_kpc
    t["kpc_per_arcsec"] = kpc_per_arcsec
    t.meta["ring_width_arcsec"] = float(width)
    t.meta["kpc_per_arcsec"] = float(np.mean(kpc_per_arcsec))
    t.meta["rad_convention"] = "RAD = ring center"
    return t


def _bunit_to_kms(data: np.ndarray, bunit: str, fname: str) -> np.ndarray:
    b = (bunit or "").replace(" ", "").upper()
    if b in ("KM/S", "KMS-1", "KM.S-1", "KM.S**-1"):
        return data
    if b in ("M/S", "MS-1", "M.S-1", "M.S**-1"):
        return data / 1000.0
    raise RuntimeError(
        f"{fname}: unrecognized BUNIT {bunit!r}; expected KM/S or M/S. "
        "Refusing to guess -- the old script silently divided by 1000 and "
        "would corrupt data already in km/s."
    )


@dataclass
class MapSet:
    shape: tuple
    data_mom1: np.ndarray
    data_mom2: np.ndarray
    model_mom1: np.ndarray
    model_mom2: np.ndarray
    cdelt1_deg: float
    cdelt2_deg: float
    pixscale_arcsec: float
    bmaj_deg: float
    bmin_deg: float
    bpa_deg: float


def _read_beam(header, fname: str):
    bmaj, bmin, bpa = header.get("BMAJ"), header.get("BMIN"), header.get("BPA")
    if bmaj is not None and bmin is not None and bpa is not None:
        return float(bmaj), float(bmin), float(bpa)
    for card in header.get("HISTORY", []):
        s = str(card)
        if "BMAJ" in s.upper():
            raise RuntimeError(
                f"{fname}: BMAJ/BMIN/BPA missing from header keywords but found in "
                f"a HISTORY card ('{s}'); parsing AIPS-style HISTORY beam cards is "
                "not implemented -- add it if this ever fires."
            )
    raise RuntimeError(
        f"{fname}: no BMAJ/BMIN/BPA in header and no HISTORY beam card found. "
        "The bootstrap needs the beam size to define independent cells; cannot proceed."
    )


def load_maps(maps_dir: Path) -> MapSet:
    maps_dir = Path(maps_dir)
    all_1mom = sorted(maps_dir.glob("*_1mom.fits"))
    all_2mom = sorted(maps_dir.glob("*_2mom.fits"))

    data_1mom = [f for f in all_1mom if "_local_" not in f.name]
    model_1mom = [f for f in all_1mom if "_local_" in f.name]
    data_2mom = [f for f in all_2mom if "_local_" not in f.name]
    model_2mom = [f for f in all_2mom if "_local_" in f.name]

    for label, files in (
        ("data _1mom", data_1mom),
        ("model _local_1mom", model_1mom),
        ("data _2mom", data_2mom),
        ("model _local_2mom", model_2mom),
    ):
        if len(files) != 1:
            raise RuntimeError(
                f"Expected exactly one {label}.fits file in {maps_dir}, found {len(files)}: {files}"
            )

    data_1mom, model_1mom, data_2mom, model_2mom = (
        data_1mom[0],
        model_1mom[0],
        data_2mom[0],
        model_2mom[0],
    )

    hdu1 = fits.open(data_1mom)[0]
    header = hdu1.header
    data_mom1 = _bunit_to_kms(np.squeeze(hdu1.data).astype(float), header.get("BUNIT"), data_1mom.name)
    data_mom2 = _bunit_to_kms(
        np.squeeze(fits.getdata(data_2mom)).astype(float),
        fits.getheader(data_2mom).get("BUNIT"),
        data_2mom.name,
    )
    model_mom1 = _bunit_to_kms(
        np.squeeze(fits.getdata(model_1mom)).astype(float),
        fits.getheader(model_1mom).get("BUNIT"),
        model_1mom.name,
    )
    model_mom2 = _bunit_to_kms(
        np.squeeze(fits.getdata(model_2mom)).astype(float),
        fits.getheader(model_2mom).get("BUNIT"),
        model_2mom.name,
    )

    if not (data_mom1.shape == data_mom2.shape == model_mom1.shape == model_mom2.shape):
        raise RuntimeError(
            f"Map shape mismatch: data_mom1={data_mom1.shape}, data_mom2={data_mom2.shape}, "
            f"model_mom1={model_mom1.shape}, model_mom2={model_mom2.shape}"
        )

    cdelt1 = float(header["CDELT1"])
    cdelt2 = float(header["CDELT2"])
    if not np.isclose(abs(cdelt1), abs(cdelt2), rtol=1e-6):
        raise RuntimeError(f"|CDELT1| != |CDELT2| ({cdelt1} vs {cdelt2}); non-square pixels not supported.")
    pixscale_arcsec = abs(cdelt1) * 3600.0

    bmaj, bmin, bpa = _read_beam(header, data_1mom.name)

    return MapSet(
        shape=data_mom1.shape,
        data_mom1=data_mom1,
        data_mom2=data_mom2,
        model_mom1=model_mom1,
        model_mom2=model_mom2,
        cdelt1_deg=cdelt1,
        cdelt2_deg=cdelt2,
        pixscale_arcsec=pixscale_arcsec,
        bmaj_deg=bmaj,
        bmin_deg=bmin,
        bpa_deg=bpa,
    )


# --------------------------------------------------------------------------
# 5.3 Geometry
# --------------------------------------------------------------------------


def deproject_pixel_offsets(dx, dy, pa_barolo_deg, inc_deg, cdelt1_sign):
    """Core deprojection algebra (CLAUDE.md 2.1-2.2), factored out of
    make_geometry so callers with a point (or a handful of points) rather
    than a full image grid -- e.g. timescales.py's void geometry -- can
    reuse the exact same rotation/deprojection instead of maintaining a
    second implementation. dx, dy: pixel offsets from the galaxy center
    (same convention as make_geometry: dx = x - xc, dy = y - yc), scalar or
    array. Returns (R_pix, theta_rad).
    """
    if cdelt1_sign < 0:
        pa_math_deg = pa_barolo_deg + 90.0
    elif cdelt1_sign > 0:
        raise NotImplementedError(
            "CDELT1 > 0 (east-right display orientation) is not a confirmed "
            "convention for this project -- only east-left (CDELT1 < 0) has "
            "been validated against the Barolo PA convention. Refusing to "
            "silently support both handedness conventions; see CLAUDE.md 2.1."
        )
    else:
        raise RuntimeError("CDELT1 == 0; cannot determine image orientation.")

    pa_rad = np.radians(pa_math_deg)
    inc_rad = np.radians(inc_deg)

    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    dx_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    dy_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
    dy_deproj = dy_rot / np.cos(inc_rad)

    R_pix = np.hypot(dx_rot, dy_deproj)
    theta_rad = np.arctan2(dy_deproj, dx_rot)
    return R_pix, theta_rad


def make_geometry(shape, xc, yc, pa_barolo_deg, inc_deg, cdelt1_sign):
    """Pure function, no globals, no I/O. Returns (R_pix, theta_rad).

    theta = 0 on the receding major axis, theta = 180 deg on the approaching
    side, theta = +/-90 deg on the minor axis (max V_rad leverage).
    """
    y_idx, x_idx = np.indices(shape)
    dx = x_idx - xc
    dy = y_idx - yc
    return deproject_pixel_offsets(dx, dy, pa_barolo_deg, inc_deg, cdelt1_sign)


def ring_mask(R_arcsec, r_in, r_out, mom1, mom2):
    return (R_arcsec >= r_in) & (R_arcsec < r_out) & np.isfinite(mom1) & np.isfinite(mom2)


def data_quality_mask(data_mom2, sigma_artifact_floor_kms):
    """Excludes pixels where data mom2 has collapsed to near-zero (SoFiA
    linewidth artifact at marginal S/N), distinct from the prohibited
    mom2 >= 5.0 cut -- see CLAUDE.md 5.3. Never applied to the model."""
    return data_mom2 >= sigma_artifact_floor_kms


def assert_receding_side(theta_rad, model_mom1, vsys, outer_mask):
    """Section 2.3: within the outermost ring, the (unweighted) mean of
    (model_mom1 - VSYS) over cos(theta) > 0 pixels must be positive, and
    negative over cos(theta) < 0. Uses the Barolo MODEL mom1 so the test is
    not confused by real non-circular motion. This single check permanently
    closes the +90 / east-left question."""
    cos_t = np.cos(theta_rad)
    rec = outer_mask & (cos_t > 0)
    app = outer_mask & (cos_t < 0)
    if not np.any(rec) or not np.any(app):
        raise RuntimeError("Receding-side assertion: outer ring has no pixels on one side; cannot test.")
    mean_rec = float(np.nanmean(model_mom1[rec] - vsys))
    mean_app = float(np.nanmean(model_mom1[app] - vsys))
    if not (mean_rec > 0 and mean_app < 0):
        raise RuntimeError(
            "Receding-side assertion FAILED: expected mean(model_mom1-VSYS) > 0 "
            f"for cos(theta)>0 and < 0 for cos(theta)<0. Got mean_receding={mean_rec:.3f}, "
            f"mean_approaching={mean_app:.3f}. The PA/handedness convention (CLAUDE.md 2.1) "
            "is very likely wrong."
        )
    return mean_rec, mean_app


# --------------------------------------------------------------------------
# 5.4 Weighting
# --------------------------------------------------------------------------


def compute_weights(theta_rad, sigma_kms, scheme, sigma_floor_kms):
    if scheme == "uniform":
        w = np.ones_like(theta_rad)
    elif scheme == "sin2":
        w = np.sin(theta_rad) ** 2
    elif scheme == "invvar":
        sigma_w = np.maximum(sigma_kms, sigma_floor_kms)
        w = 1.0 / sigma_w**2
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")

    total = np.sum(w)
    if total <= 0:
        raise RuntimeError(f"Weighting scheme {scheme!r} produced sum(w) <= 0; cannot normalise.")
    w = w * (len(w) / total)
    return w


# --------------------------------------------------------------------------
# 5.5 The fit
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    terms: tuple
    p: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    resid: np.ndarray

    def value(self, term):
        return float(self.p[self.terms.index(term)])

    def err(self, term):
        i = self.terms.index(term)
        return float(np.sqrt(self.cov[i, i]))


def build_data_vector(mom1, vsys, inc_deg, vrot, theta_rad):
    """d = (V_los - VSYS)/sin(inc) - VROT*cos(theta); c1 (VROT) removed as a
    subtraction, never a design-matrix column."""
    return (mom1 - vsys) / np.sin(np.radians(inc_deg)) - vrot * np.cos(theta_rad)


def fit_wls(d, theta_rad, w, sigma_kms, terms):
    """Weighted linear least squares. Do not use curve_fit -- the model is
    linear in every free parameter and this is the analytic solution.

    Covariance uses the sandwich form because w != 1/sigma**2 in general
    (uniform, sin2): Cov = inv(M) (A' W diag(sigma^2) W A) inv(M). These are
    formal errors, reference only -- they assume independent pixels, which is
    false at 4-arcsec pixels under a MeerKAT beam. The bootstrap is the
    quoted statistical error.
    """
    A = design_matrix(theta_rad, terms)
    M = A.T @ (w[:, None] * A)
    rhs = A.T @ (w * d)
    p = np.linalg.solve(M, rhs)
    resid = d - A @ p

    Minv = np.linalg.inv(M)
    sandwich = A.T @ ((w * sigma_kms**2 * w)[:, None] * A)
    cov = Minv @ sandwich @ Minv
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)

    return FitResult(terms=tuple(terms), p=p, cov=cov, corr=corr, resid=resid)


def leakage_diagnostics(theta_rad, w):
    """Bias in s1_hat = sum(w d sin)/sum(w sin^2) from a constant offset eps
    (L0) and a VROT error dVROT (L1). Both vanish analytically for symmetric
    azimuthal coverage; blanked pixels break that symmetry."""
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    denom = np.sum(w * sin_t**2)
    L0 = np.sum(w * sin_t) / denom
    L1 = np.sum(w * sin_t * cos_t) / denom
    return float(L0), float(L1)


def rms_about_zero(resid):
    return float(np.sqrt(np.mean(resid**2)))


def pa_degeneracy_slope(theta_rad, w, vrot, inc_deg):
    """Inclination-corrected PA->s1 leakage slope (km/s per radian of PA
    error), for the analytic overlay in fig_pa_degeneracy and for the PA
    self-test. The flat-sky slope -VROT is only exact at inc=0: deprojecting
    through cos(inc) makes dtheta/d(dPA) range from -1/cos(inc) on the major
    axis to -cos(inc) on the minor axis (see CLAUDE.md 5.6), so the estimator
    picks up a weighted average K of that, not -1. Computed numerically from
    the actual pixel theta/weight arrays so it is correct for invvar
    weighting with spatially-varying sigma too, not just the closed forms
    that hold for constant-sigma schemes."""
    cos_i = np.cos(np.radians(inc_deg))
    g = -(np.cos(theta_rad) ** 2 / cos_i + cos_i * np.sin(theta_rad) ** 2)
    sin2_t = np.sin(theta_rad) ** 2
    K = np.sum(w * g * sin2_t) / np.sum(w * sin2_t)
    return vrot * K  # multiply by dPA in radians to get s1_leak


# --------------------------------------------------------------------------
# 5.7 Bootstrap
# --------------------------------------------------------------------------


def pixels_per_beam(bmaj_deg, bmin_deg, cdelt1_deg, cdelt2_deg):
    return 1.1331 * bmaj_deg * bmin_deg / abs(cdelt1_deg * cdelt2_deg)


def beam_cell_ids(y_idx, x_idx, cell_side_pix, n_cols):
    n_cells_x = int(np.ceil(n_cols / cell_side_pix)) + 1
    return (y_idx // cell_side_pix).astype(np.int64) * n_cells_x + (x_idx // cell_side_pix).astype(np.int64)


def block_bootstrap_generic(
    d, theta_rad, sigma_kms, cell_id, scheme, sigma_floor_kms, terms, n_bootstrap, rng, extract_fn
):
    """Resample beam-sized cells with replacement, concatenate their pixels,
    refit, and extract an arbitrary scalar from each refit via
    extract_fn(FitResult) -> float. Returns (draws, n_cells). Generalizes
    the s1-specific bootstrap below so other modules (e.g.
    azimuthal_modulation.py) can reuse the same resampling scheme for other
    derived quantities without duplicating it.

    A draw can be degenerate (rank-deficient design matrix, e.g. when a
    small-n side/ring resamples the same single-pixel cell for every slot,
    so every resampled row shares one theta) -- observed in practice on
    rings with only a handful of beam-cells. Redraw those rather than
    letting one bad draw crash an otherwise-valid bootstrap; a run of 50
    consecutive degenerate draws instead raises, since that means the
    ring/side/scheme genuinely has too few independent cells to bootstrap.
    """
    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    cell_pixels = [np.nonzero(inverse == i)[0] for i in range(len(unique_cells))]
    n_cells = len(unique_cells)

    draws = np.empty(n_bootstrap)
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
            raise RuntimeError(
                f"block_bootstrap_generic: 50 consecutive degenerate resamples (rank-deficient "
                f"design matrix) with n_cells={n_cells}. This ring/side/weighting combination "
                "has too few independent beam-cells to bootstrap reliably."
            )
        draws[b] = extract_fn(fit_b)
    if n_degenerate:
        print(f"[harmonic_fit]   block_bootstrap_generic: redrew {n_degenerate} degenerate "
              f"resample(s) out of {n_bootstrap} (rank-deficient design, n_cells={n_cells})")
    return draws, n_cells


def block_bootstrap_s1(
    d, theta_rad, sigma_kms, cell_id, scheme, sigma_floor_kms, terms, n_bootstrap, rng
):
    """s1-extraction wrapper around block_bootstrap_generic (kept as its own
    name/signature since it's the estimator quoted throughout the pipeline)."""
    s1_idx = terms.index("s1")
    return block_bootstrap_generic(
        d, theta_rad, sigma_kms, cell_id, scheme, sigma_floor_kms, terms, n_bootstrap, rng,
        extract_fn=lambda fit: fit.p[s1_idx],
    )


# --------------------------------------------------------------------------
# 5.8 Scans
# --------------------------------------------------------------------------


def _profile_chi2_grid_over_s1(d, theta_rad, w, sigma_kms, nuisance_terms, s1_grid):
    """Fix s1 on a grid, profile out the remaining (nuisance) free terms by
    weighted least squares, and evaluate chi2 = sum((resid/sigma)^2) with
    sigma from mom2 -- independent of the fitting weights -- at each point.
    Fast path for a single nuisance term (the default: just c0)."""
    sin_t = np.sin(theta_rad)
    n_grid = len(s1_grid)

    if len(nuisance_terms) == 1 and nuisance_terms[0] == "c0":
        # closed form: c0_hat(s1) = sum(w*(d - s1*sin)) / sum(w)
        r = d[None, :] - s1_grid[:, None] * sin_t[None, :]
        sum_w = np.sum(w)
        c0_hat = np.sum(w[None, :] * r, axis=1) / sum_w
        resid = r - c0_hat[:, None]
        chi2 = np.sum((resid / sigma_kms[None, :]) ** 2, axis=1)
        return chi2, c0_hat[:, None]

    A_nuis = design_matrix(theta_rad, nuisance_terms)
    k = A_nuis.shape[1]
    chi2 = np.empty(n_grid)
    p_nuis = np.empty((n_grid, k))
    WA = w[:, None] * A_nuis
    M = A_nuis.T @ WA
    for i, s1 in enumerate(s1_grid):
        r = d - s1 * sin_t
        rhs = A_nuis.T @ (w * r)
        p = np.linalg.solve(M, rhs)
        resid = r - A_nuis @ p
        chi2[i] = np.sum((resid / sigma_kms) ** 2)
        p_nuis[i] = p
    return chi2, p_nuis


def pa_scan(cfg, ringlog_row, mapset, fiducial_mask, xc, yc, inc_deg, vrot, vsys, cdelt1_sign):
    """Freeze the pixel set at the fiducial ring mask; only theta is rebuilt
    at each scanned PA. See CLAUDE.md 5.8 for why the mask must not change."""
    pa0 = float(ringlog_row["P.A.(deg)"])
    pa_grid = np.arange(-cfg.pa_scan_halfwidth_deg, cfg.pa_scan_halfwidth_deg + 1e-9, cfg.pa_scan_step_deg) + pa0
    s1_grid = np.arange(-cfg.s1_grid_halfwidth_kms, cfg.s1_grid_halfwidth_kms + 1e-9, cfg.s1_grid_step_kms)

    y_idx, x_idx = np.indices(mapset.shape)
    y_m, x_m = y_idx[fiducial_mask], x_idx[fiducial_mask]
    mom1_m = mapset.data_mom1[fiducial_mask]
    mom2_m = mapset.data_mom2[fiducial_mask]

    s1_best = np.empty(len(pa_grid))
    chi2_best = np.empty(len(pa_grid))
    chi2_grid = np.empty((len(pa_grid), len(s1_grid)))
    nuisance_terms = tuple(t for t in cfg.model_terms if t != "s1")

    for i, pa in enumerate(pa_grid):
        _, theta_full = make_geometry(mapset.shape, xc, yc, pa, inc_deg, cdelt1_sign)
        theta_m = theta_full[fiducial_mask]
        d_m = build_data_vector(mom1_m, vsys, inc_deg, vrot, theta_m)
        w_m = compute_weights(theta_m, mom2_m, cfg.primary_weighting, cfg.sigma_floor_kms)

        fit_i = fit_wls(d_m, theta_m, w_m, mom2_m, cfg.model_terms)
        s1_best[i] = fit_i.value("s1")
        chi2_best[i] = np.sum((fit_i.resid / mom2_m) ** 2)

        chi2_grid[i], _ = _profile_chi2_grid_over_s1(d_m, theta_m, w_m, mom2_m, nuisance_terms, s1_grid)

    return {"pa_grid_deg": pa_grid, "s1_grid_kms": s1_grid, "s1_best": s1_best,
            "chi2_best": chi2_best, "chi2_grid": chi2_grid, "pa0_deg": pa0}


def vsys_scan(cfg, ringlog_row, mapset, fiducial_mask, xc, yc, pa0, inc_deg, vrot, vsys0, cdelt1_sign):
    """Geometry is unaffected by VSYS -- only the data vector shifts -- so
    theta and weights are computed once and reused for every grid point."""
    vsys_grid = np.arange(-cfg.vsys_scan_halfwidth_kms, cfg.vsys_scan_halfwidth_kms + 1e-9, cfg.vsys_scan_step_kms) + vsys0
    s1_grid = np.arange(-cfg.s1_grid_halfwidth_kms, cfg.s1_grid_halfwidth_kms + 1e-9, cfg.s1_grid_step_kms)

    _, theta_full = make_geometry(mapset.shape, xc, yc, pa0, inc_deg, cdelt1_sign)
    theta_m = theta_full[fiducial_mask]
    mom1_m = mapset.data_mom1[fiducial_mask]
    mom2_m = mapset.data_mom2[fiducial_mask]
    w_m = compute_weights(theta_m, mom2_m, cfg.primary_weighting, cfg.sigma_floor_kms)
    nuisance_terms = tuple(t for t in cfg.model_terms if t != "s1")

    s1_best = np.empty(len(vsys_grid))
    chi2_best = np.empty(len(vsys_grid))
    chi2_grid = np.empty((len(vsys_grid), len(s1_grid)))

    for i, vsys in enumerate(vsys_grid):
        d_m = build_data_vector(mom1_m, vsys, inc_deg, vrot, theta_m)
        fit_i = fit_wls(d_m, theta_m, w_m, mom2_m, cfg.model_terms)
        s1_best[i] = fit_i.value("s1")
        chi2_best[i] = np.sum((fit_i.resid / mom2_m) ** 2)
        chi2_grid[i], _ = _profile_chi2_grid_over_s1(d_m, theta_m, w_m, mom2_m, nuisance_terms, s1_grid)

    return {"vsys_grid_kms": vsys_grid, "s1_grid_kms": s1_grid, "s1_best": s1_best,
            "chi2_best": chi2_best, "chi2_grid": chi2_grid, "vsys0_kms": vsys0}


# --------------------------------------------------------------------------
# 5.9 main()
# --------------------------------------------------------------------------


def n_effective(n_pix, ppb):
    return n_pix / ppb


def process_ring(cfg, ring_idx, ringlog_row, mapset, cdelt1_sign, rng, ppb):
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
    n_removed_by_quality_mask = int(np.sum(base_mask_raw & ~quality_mask))

    cell_side_pix = int(np.ceil(np.sqrt(ppb)))
    y_idx, x_idx = np.indices(mapset.shape)
    cell_id_full = beam_cell_ids(y_idx, x_idx, cell_side_pix, mapset.shape[1])

    cos_t_full = np.cos(theta_full)
    side_masks = {
        "both": base_mask,
        "receding": base_mask & (cos_t_full > 0),
        "approaching": base_mask & ~(cos_t_full > 0),
    }

    rows = []
    maps_extra = {"R_arcsec": R_arcsec, "theta": theta_full, "ring_mask_both": base_mask}
    boot_extra = {}

    for side in SIDES:
        mask = side_masks[side]
        n_pix = int(np.sum(mask))
        if n_pix < len(cfg.model_terms) + 1:
            warnings.warn(f"Ring {ring_idx} side={side}: only {n_pix} pixels, skipping.")
            continue

        theta_m = theta_full[mask]
        mom1_m = mapset.data_mom1[mask]
        mom2_m = mapset.data_mom2[mask]
        cell_id_m = cell_id_full[mask]
        d_m = build_data_vector(mom1_m, vsys, inc, vrot, theta_m)

        n_eff = n_effective(n_pix, ppb)

        for scheme in cfg.weighting_schemes:
            w_m = compute_weights(theta_m, mom2_m, scheme, cfg.sigma_floor_kms)
            fit_r = fit_wls(d_m, theta_m, w_m, mom2_m, cfg.model_terms)
            L0, L1 = leakage_diagnostics(theta_m, w_m)
            rms = rms_about_zero(fit_r.resid)
            chi2 = float(np.sum((fit_r.resid / mom2_m) ** 2))

            boot_draws, n_cells = block_bootstrap_s1(
                d_m, theta_m, mom2_m, cell_id_m, scheme, cfg.sigma_floor_kms,
                cfg.model_terms, cfg.n_bootstrap, rng,
            )
            lo, med, hi = np.percentile(boot_draws, [16, 50, 84])
            boot_extra[f"boot_s1_{side}_{scheme}"] = boot_draws

            row = dict(
                ring_index=ring_idx,
                r_in_arcsec=r_in,
                r_out_arcsec=r_out,
                r_center_kpc=float(ringlog_row["r_center_kpc"]),
                vrot_kms=vrot,
                inc_deg=inc,
                pa0_deg=pa0,
                vsys_kms=vsys,
                xpos_pix=xc,
                ypos_pix=yc,
                side=side,
                weighting=scheme,
                s1=fit_r.value("s1"),
                s1_formal_err=fit_r.err("s1"),
                s1_boot_lo=float(lo),
                s1_boot_med=float(med),
                s1_boot_hi=float(hi),
                c0=fit_r.value("c0") if "c0" in cfg.model_terms else np.nan,
                c0_err=fit_r.err("c0") if "c0" in cfg.model_terms else np.nan,
                c2=fit_r.value("c2") if "c2" in cfg.model_terms else np.nan,
                s2=fit_r.value("s2") if "s2" in cfg.model_terms else np.nan,
                chi2=chi2,
                n_pix=n_pix,
                n_eff=n_eff,
                n_cells=n_cells,
                pixels_per_beam=ppb,
                L0=L0,
                L1=L1,
                rms_residual=rms,
                near_side_assumed="UNRESOLVED",
                n_removed_quality_mask=n_removed_by_quality_mask,
            )
            rows.append(row)

            if side == "both" and scheme == cfg.primary_weighting:
                maps_extra["weights_primary"] = np.zeros(mapset.shape)
                maps_extra["weights_primary"][mask] = w_m
                dv_pre = np.full(mapset.shape, np.nan)
                dv_pre[mask] = d_m
                dv_post = np.full(mapset.shape, np.nan)
                dv_post[mask] = fit_r.resid
                maps_extra["dv_prefit"] = dv_pre
                maps_extra["dv_postfit"] = dv_post

    pa_result = pa_scan(cfg, ringlog_row, mapset, side_masks["both"], xc, yc, inc, vrot, vsys, cdelt1_sign)
    vsys_result = vsys_scan(cfg, ringlog_row, mapset, side_masks["both"], xc, yc, pa0, inc, vrot, vsys, cdelt1_sign)

    n_eff_both = n_effective(int(np.sum(side_masks["both"])), ppb)
    scale = n_eff_both / max(np.min(pa_result["chi2_grid"]), 1e-30)
    pa_result["chi2_grid"] *= scale
    pa_result["chi2_best"] *= scale
    vsys_result["chi2_grid"] *= scale
    vsys_result["chi2_best"] *= scale
    pa_result["chi2_rescale_note"] = "rescaled so chi2_min/n_eff = 1: a calibration to n_eff, not a noise-model claim"
    vsys_result["chi2_rescale_note"] = pa_result["chi2_rescale_note"]

    return rows, maps_extra, pa_result, vsys_result, boot_extra


def main(cfg: Config):
    print(f"[harmonic_fit] RAD convention: ring center (RAD +/- width/2)")
    ringlog = read_ringlog(cfg.ringlog_path)
    print(f"[harmonic_fit] Ring bounds (arcsec):")
    for row in ringlog:
        print(f"    ring: {row['r_in_arcsec']:.2f} -- {row['r_out_arcsec']:.2f}  "
              f"(VROT={row['VROT(km/s)']:.1f} km/s, INC={row['INC(deg)']:.2f} deg, PA={row['P.A.(deg)']:.2f} deg)")
    print(f"[harmonic_fit] kpc_per_arcsec = {ringlog.meta['kpc_per_arcsec']:.5f}")

    mapset = load_maps(cfg.maps_dir)
    cdelt1_sign = -1 if mapset.cdelt1_deg < 0 else (1 if mapset.cdelt1_deg > 0 else 0)
    print(f"[harmonic_fit] CDELT1 = {mapset.cdelt1_deg} deg/pix (sign={cdelt1_sign}); "
          f"pixel scale = {mapset.pixscale_arcsec:.3f} arcsec/pix")

    ppb = pixels_per_beam(mapset.bmaj_deg, mapset.bmin_deg, mapset.cdelt1_deg, mapset.cdelt2_deg)
    print(f"[harmonic_fit] beam = {mapset.bmaj_deg*3600:.3f}\" x {mapset.bmin_deg*3600:.3f}\"; "
          f"pixels_per_beam = {ppb:.3f}")

    # Section 2.3 assertion -- fire on the outermost ring, once, before trusting anything else.
    outer_row = ringlog[-1]
    xc0, yc0 = float(outer_row["XPOS(pix)"]), float(outer_row["YPOS(pix)"])
    pa00, inc0 = float(outer_row["P.A.(deg)"]), float(outer_row["INC(deg)"])
    vsys0 = float(outer_row["VSYS(km/s)"])
    R_pix0, theta0 = make_geometry(mapset.shape, xc0, yc0, pa00, inc0, cdelt1_sign)
    outer_mask0 = ring_mask(R_pix0 * mapset.pixscale_arcsec, 0.0, float(outer_row["r_out_arcsec"]),
                             mapset.model_mom1, mapset.model_mom1)
    mean_rec, mean_app = assert_receding_side(theta0, mapset.model_mom1, vsys0, outer_mask0)
    print(f"[harmonic_fit] Receding-side assertion PASSED: "
          f"mean(model-VSYS)|receding={mean_rec:.2f}, |approaching={mean_app:.2f}")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.random_seed)

    all_rows = []
    all_maps = {
        "data_mom1": mapset.data_mom1,
        "data_mom2": mapset.data_mom2,
        "model_mom1": mapset.model_mom1,
        "model_mom2": mapset.model_mom2,
        "pixscale_arcsec": np.array(mapset.pixscale_arcsec),
    }
    all_scans = {}
    for ring_idx, ringlog_row in enumerate(ringlog):
        rows, maps_extra, pa_result, vsys_result, boot_extra = process_ring(
            cfg, ring_idx, ringlog_row, mapset, cdelt1_sign, rng, ppb
        )
        all_rows.extend(rows)
        for k, v in maps_extra.items():
            all_maps[f"ring{ring_idx}_{k}"] = v
        for k, v in pa_result.items():
            if k != "chi2_rescale_note":
                all_scans[f"ring{ring_idx}_pa_{k}"] = v
        for k, v in vsys_result.items():
            if k != "chi2_rescale_note":
                all_scans[f"ring{ring_idx}_vsys_{k}"] = v
        for k, v in boot_extra.items():
            all_scans[f"ring{ring_idx}_{k}"] = v

    results_table = Table(rows=all_rows)
    results_table.meta["rad_convention"] = "RAD = ring center"
    results_table.meta["primary_weighting"] = cfg.primary_weighting
    results_table.meta["kpc_per_arcsec"] = ringlog.meta["kpc_per_arcsec"]
    results_table.meta["bmaj_arcsec"] = mapset.bmaj_deg * 3600.0
    results_table.meta["bmin_arcsec"] = mapset.bmin_deg * 3600.0
    results_table.meta["pixels_per_beam"] = ppb
    results_table.meta["config"] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
    }

    ecsv_path = cfg.results_dir / "ring_results.ecsv"
    results_table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    print(f"[harmonic_fit] Wrote {ecsv_path}")

    np.savez(cfg.results_dir / "maps.npz", **all_maps)
    print(f"[harmonic_fit] Wrote {cfg.results_dir / 'maps.npz'}")

    np.savez(cfg.results_dir / "scans.npz", **all_scans)
    print(f"[harmonic_fit] Wrote {cfg.results_dir / 'scans.npz'}")

    print("\n[harmonic_fit] Summary (side=both, primary weighting):")
    print(f"{'ring':<6}{'r_in':>8}{'r_out':>8}{'s1':>10}{'boot_lo':>10}{'boot_hi':>10}{'c0':>10}{'chi2':>10}{'n_eff':>8}")
    for row in all_rows:
        if row["side"] == "both" and row["weighting"] == cfg.primary_weighting:
            print(f"{row['ring_index']:<6}{row['r_in_arcsec']:8.2f}{row['r_out_arcsec']:8.2f}"
                  f"{row['s1']:10.2f}{row['s1_boot_lo']:10.2f}{row['s1_boot_hi']:10.2f}"
                  f"{row['c0']:10.2f}{row['chi2']:10.1f}{row['n_eff']:8.1f}")

    return results_table


# --------------------------------------------------------------------------
# 8. Self-tests
# --------------------------------------------------------------------------


def _synthetic_mapset_and_ringlog(shape=(48, 48)):
    """Build a synthetic MapSet + ringlog on the real pixel grid, geometry
    taken from the TRM_paper model (the canonical paper run) so tests
    exercise the real WCS. Any TRM model directory would do for this --
    TRM_paper is just a fixed, always-present choice."""
    ringlog_path = find_ringlog(Path(__file__).parent / "TRM_paper")
    ringlog = read_ringlog(ringlog_path)

    xc = float(ringlog[0]["XPOS(pix)"])
    yc = float(ringlog[0]["YPOS(pix)"])
    pa0 = float(ringlog[0]["P.A.(deg)"])
    inc0 = float(ringlog[0]["INC(deg)"])
    vsys0 = float(ringlog[0]["VSYS(km/s)"])

    return ringlog, dict(shape=shape, xc=xc, yc=yc, pa0=pa0, inc0=inc0, vsys0=vsys0)


def _build_synthetic_mom1(shape, xc, yc, pa_deg, inc_deg, vsys, vrot_of_R, vrad_kms, cdelt1_sign=-1, extra_offset=0.0):
    R_pix, theta = make_geometry(shape, xc, yc, pa_deg, inc_deg, cdelt1_sign)
    vrot = vrot_of_R(R_pix)
    v_los = vsys + extra_offset + np.sin(np.radians(inc_deg)) * (vrot * np.cos(theta) + vrad_kms * np.sin(theta))
    return v_los, R_pix, theta


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

    ringlog, geo = _synthetic_mapset_and_ringlog()
    shape, xc, yc, pa0, inc0, vsys0 = geo["shape"], geo["xc"], geo["yc"], geo["pa0"], geo["inc0"], geo["vsys0"]

    def vrot_of_R_flat(R_pix):
        return np.full_like(R_pix, 300.0)

    # ---------------- Test 1: synthetic recovery ----------------
    v_los, R_pix, theta = _build_synthetic_mom1(shape, xc, yc, pa0, inc0, vsys0, vrot_of_R_flat, 25.0)
    R_arcsec = R_pix * 4.0  # pixscale from the real header (verified separately in load_maps)
    mom2 = np.full(shape, 10.0)

    ok1 = True
    for ring_idx, row in enumerate(ringlog):
        r_in, r_out = float(row["r_in_arcsec"]), float(row["r_out_arcsec"])
        mask = ring_mask(R_arcsec, r_in, r_out, v_los, mom2)
        theta_m = theta[mask]
        d_m = build_data_vector(v_los[mask], vsys0, inc0, 300.0, theta_m)
        for scheme in ("uniform", "sin2", "invvar"):
            w_m = compute_weights(theta_m, mom2[mask], scheme, 5.0)
            fit_r = fit_wls(d_m, theta_m, w_m, mom2[mask], ("c0", "s1"))
            if abs(fit_r.value("s1") - 25.0) > 0.5:
                ok1 = False
    check("1. Synthetic recovery (s1=25+/-0.5 km/s, all rings/weightings)", ok1)

    # ---------------- Test 2: PA degeneracy ----------------
    v_los2, R_pix2, theta2 = _build_synthetic_mom1(shape, xc, yc, pa0, inc0, vsys0, vrot_of_R_flat, 0.0)
    R_arcsec2 = R_pix2 * 4.0
    pa_offset_deg = 2.0
    _, theta_wrong = make_geometry(shape, xc, yc, pa0 + pa_offset_deg, inc0, -1)
    row0 = ringlog[0]
    mask2 = ring_mask(R_arcsec2, float(row0["r_in_arcsec"]), float(row0["r_out_arcsec"]), v_los2, mom2)
    theta_m2 = theta_wrong[mask2]
    d_m2 = build_data_vector(v_los2[mask2], vsys0, inc0, 300.0, theta_m2)
    w_m2 = compute_weights(theta_m2, mom2[mask2], "sin2", 5.0)
    fit2 = fit_wls(d_m2, theta_m2, w_m2, mom2[mask2], ("c0", "s1"))
    # Inclination-corrected slope (CLAUDE.md 5.6) -- the flat-sky -VROT*dPA is
    # NOT accurate at finite inclination; evaluated at the wrong-PA theta/weights
    # since that is what the fit actually sees.
    slope = pa_degeneracy_slope(theta_m2, w_m2, 300.0, inc0)
    expected = slope * np.radians(pa_offset_deg)
    rel_err = abs((fit2.value("s1") - expected) / expected)
    check("2. PA degeneracy (s1 ~= VROT*K*dPA[rad] within 5%, inclination-corrected)", rel_err < 0.05,
          f"(got s1={fit2.value('s1'):.3f}, expected={expected:.3f}, rel_err={rel_err:.3%})")

    # ---------------- Test 3: receding-side assertion fires ----------------
    _, theta_flipped = make_geometry(shape, xc, yc, pa0 + 180.0, inc0, -1)
    outer_row = ringlog[-1]
    outer_mask = ring_mask(R_pix2 * 4.0, 0.0, float(outer_row["r_out_arcsec"]), v_los2, v_los2)
    raised = False
    try:
        assert_receding_side(theta_flipped, v_los2, vsys0, outer_mask)
    except RuntimeError:
        raised = True
    check("3. Receding-side assertion fires on 180-deg-flipped PA", raised)

    # ---------------- Test 4: leakage ----------------
    # An idealized synthetic ring (continuous, uniformly-sampled theta), not
    # the real pixel grid: pixelization of an annulus on a Cartesian grid is
    # not perfectly theta-symmetric even for a "full" ring, which would fail
    # the 1e-10 tolerance for reasons unrelated to the leakage formulas being
    # tested here.
    theta_ideal = np.linspace(-np.pi, np.pi, 100_000, endpoint=False)
    sigma_ideal = np.ones_like(theta_ideal) * 10.0
    ok4a = True
    for scheme in ("uniform", "sin2", "invvar"):
        w4 = compute_weights(theta_ideal, sigma_ideal, scheme, 5.0)
        L0, L1 = leakage_diagnostics(theta_ideal, w4)
        if abs(L0) > 1e-10 or abs(L1) > 1e-10:
            ok4a = False
    wedge_mask = ~((np.degrees(theta_ideal) % 360 >= 40) & (np.degrees(theta_ideal) % 360 < 80))
    w4w = compute_weights(theta_ideal[wedge_mask], sigma_ideal[wedge_mask], "sin2", 5.0)
    L0w, L1w = leakage_diagnostics(theta_ideal[wedge_mask], w4w)
    ok4b = (abs(L0w) > 1e-6) or (abs(L1w) > 1e-6)
    check("4. Leakage: |L0|,|L1| < 1e-10 unmasked; measurably nonzero with a blanked wedge", ok4a and ok4b,
          f"(unmasked_ok={ok4a}, wedge L0={L0w:.2e} L1={L1w:.2e})")

    # ---------------- Test 5: analytic agreement with curve_fit ----------------
    # curve_fit used ONLY here, as an independent cross-check of the linear
    # solve -- never as the estimator in the pipeline itself (CLAUDE.md 9).
    from scipy.optimize import curve_fit as _curve_fit

    row5 = ringlog[2]
    v_los5, R_pix5, theta5 = _build_synthetic_mom1(shape, xc, yc, pa0, inc0, vsys0, vrot_of_R_flat, 18.0)
    R_arcsec5 = R_pix5 * 4.0
    mom2_5 = np.random.default_rng(0).uniform(5.0, 15.0, size=shape)
    mask5 = ring_mask(R_arcsec5, float(row5["r_in_arcsec"]), float(row5["r_out_arcsec"]), v_los5, mom2_5)
    theta_m5 = theta5[mask5]
    d_m5 = build_data_vector(v_los5[mask5], vsys0, inc0, 300.0, theta_m5)
    sigma_m5 = mom2_5[mask5]
    w_m5 = compute_weights(theta_m5, sigma_m5, "invvar", 5.0)

    fit5 = fit_wls(d_m5, theta_m5, w_m5, sigma_m5, ("s1",))

    def model_s1_only(t, s1):
        return s1 * np.sin(t)

    popt, _ = _curve_fit(model_s1_only, theta_m5, d_m5, p0=[0.0], sigma=sigma_m5, absolute_sigma=True)
    check("5. Analytic solve matches curve_fit to machine precision (invvar, 1 param)",
          np.isclose(fit5.value("s1"), popt[0], atol=1e-8, rtol=1e-8),
          f"(solve={fit5.value('s1'):.10f}, curve_fit={popt[0]:.10f})")

    # ---------------- Test 6: round trip (V_z into c0, not s1) ----------------
    v_los6a, R_pix6, theta6 = _build_synthetic_mom1(shape, xc, yc, pa0, inc0, vsys0, vrot_of_R_flat, 25.0, extra_offset=0.0)
    v_los6b, _, _ = _build_synthetic_mom1(shape, xc, yc, pa0, inc0, vsys0, vrot_of_R_flat, 25.0, extra_offset=12.3)
    R_arcsec6 = R_pix6 * 4.0
    row6 = ringlog[0]
    mask6 = ring_mask(R_arcsec6, float(row6["r_in_arcsec"]), float(row6["r_out_arcsec"]), v_los6a, mom2)
    theta_m6 = theta6[mask6]

    d_a = build_data_vector(v_los6a[mask6], vsys0, inc0, 300.0, theta_m6)
    d_b = build_data_vector(v_los6b[mask6], vsys0, inc0, 300.0, theta_m6)
    w_m6 = compute_weights(theta_m6, mom2[mask6], "sin2", 5.0)
    fit_a = fit_wls(d_a, theta_m6, w_m6, mom2[mask6], ("c0", "s1"))
    fit_b = fit_wls(d_b, theta_m6, w_m6, mom2[mask6], ("c0", "s1"))
    dc0 = fit_b.value("c0") - fit_a.value("c0")
    ds1 = fit_b.value("s1") - fit_a.value("s1")
    # The offset is injected into V_los (mom1), so after build_data_vector's
    # /sin(inc) it lands in c0 as offset/sin(inc), not the raw offset.
    expected_dc0 = 12.3 / np.sin(np.radians(inc0))
    check("6. Round trip: constant V_z-like offset (12.3 km/s in V_los) appears in c0, not s1",
          np.isclose(dc0, expected_dc0, atol=1e-6) and np.isclose(ds1, 0.0, atol=1e-6),
          f"(dc0={dc0:.4f}, expected={expected_dc0:.4f}, ds1={ds1:.6f})")

    # ---------------- Test 7: beam bookkeeping ----------------
    real_mapset = load_maps(Path(__file__).parent / "TRM_paper" / "maps")
    ppb7 = pixels_per_beam(real_mapset.bmaj_deg, real_mapset.bmin_deg,
                            real_mapset.cdelt1_deg, real_mapset.cdelt2_deg)
    ppb7_manual = (1.1331 * real_mapset.bmaj_deg * real_mapset.bmin_deg
                   / abs(real_mapset.cdelt1_deg * real_mapset.cdelt2_deg))
    check("7. Beam bookkeeping: pixels_per_beam == 1.1331*BMAJ*BMIN/|CDELT1*CDELT2| "
          "(header formula) and == 20.69 +/- 0.01 for the shipped map",
          np.isclose(ppb7, ppb7_manual, atol=1e-9) and np.isclose(ppb7, 20.69, atol=0.01),
          f"(pixels_per_beam={ppb7:.4f})")

    print(f"\n[selftest] {n_pass} passed, {n_fail} failed")
    return n_fail == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Fit s1 = V_rad(R) for one TRM model directory (e.g. TRM_paper, fixed_PA42)."
    )
    parser.add_argument("--selftest", action="store_true", help="run the acceptance tests and exit")
    parser.add_argument(
        "--list-trm-models", action="store_true",
        help="list TRM model directories found under the repo root and exit",
    )
    parser.add_argument(
        "--trm-dir", default="TRM_paper",
        help="TRM model directory: either a name relative to the repo root (e.g. 'fixed_PA42') "
             "or an absolute/relative path to any directory containing maps/ and a ringlog. "
             "Default: TRM_paper.",
    )
    parser.add_argument(
        "--ringlog", default=None,
        help="explicit path to the ringlog file, overriding auto-discovery within --trm-dir",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="where to write results/ (default: <trm-dir>/results)",
    )
    args = parser.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if args.list_trm_models:
        models = discover_trm_models(repo_root)
        if not models:
            print(f"[harmonic_fit] No TRM model directories found under {repo_root}")
        else:
            print(f"[harmonic_fit] TRM model directories found under {repo_root}:")
            for name, path in models.items():
                print(f"    {name}  ({find_ringlog(path).name})")
        sys.exit(0)

    trm_dir = Path(args.trm_dir)
    if not trm_dir.is_absolute() and not trm_dir.exists():
        trm_dir = repo_root / args.trm_dir
    if not trm_dir.is_dir():
        raise SystemExit(
            f"--trm-dir {args.trm_dir!r} is not a directory (looked in {trm_dir}). "
            f"Available: {sorted(discover_trm_models(repo_root))}"
        )

    ringlog_path = Path(args.ringlog) if args.ringlog else find_ringlog(trm_dir)
    results_dir = Path(args.results_dir) if args.results_dir else trm_dir / "results"

    cfg = Config(
        project_dir=trm_dir,
        maps_dir=trm_dir / "maps",
        ringlog_path=ringlog_path,
        results_dir=results_dir,
    )
    main(cfg)
