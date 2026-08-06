"""
Matched-filtering benchmark for the GravNet ML pipeline.

Fully self-contained analysis (no TensorFlow needed). It reuses the EXISTING
signal generation (`data_pre_processing.chirp_BW_conv_signal_generation`) and
data loading (`data_pre_processing.tiq_data_loader`) so templates and injected
signals are EXACTLY the waveforms the ML pipeline trains on, with the same
trimming rule (1e-3 of peak) and the same peak-SNR injection convention
(amplitude = target_peak_snr * noise_std_amp).

What it produces (mirroring efficiency_curve.py):
  * per-mass threshold calibration on noise-only data with a tail fit in
    log-FPR, threshold extrapolation to a target FP/year, and a 1-sigma
    THRESHOLD ERROR from the fit covariance;
  * per-mass detection-efficiency curves vs injected PEAK SNR with Wilson-95
    error bars and a weighted sigmoid fit;
  * SNR95 (peak SNR needed for 95% efficiency) per mass with
      - statistical error (sigmoid-fit covariance), and
      - SYSTEMATIC error from propagating the threshold uncertainty
        (efficiency re-evaluated at threshold +/- sigma_threshold);
  * the final sensitivity plot: SNR95 vs PBH mass (1e-13 .. 1e-6 M_sun)
    at the primary FP/year target (default 0.99/yr);
  * an FP/year scan: SNR95 as a function of the allowed FP/year with a
    linear-in-log10(FP) trend line per mass.

The matched-filter statistic uses the STANDARD definition (see
docs/MATCHED_FILTERING.md for the full math):

    rho(t) = | sum_k x[t+k] * conj(s[k]) | / ( sigma_q * sqrt(E_s) )

with per-quadrature noise std sigma_q and template energy E_s = sum|s|^2.
All result axes are the project's PEAK SNR definition
(max|s| / std(|noise|)); the per-mass conversion factor to the optimal
matched-filter SNR (rho_opt = conv * peak_snr) is computed and saved so both
conventions are always available.

Event-based detection mirrors the ML pipeline: the rho time series is sliced
into the SAME sliding windows (window_size / step_size); a window score is the
max of rho inside the window; an event counts as detected if AT LEAST ONE
window overlapping the injection exceeds the threshold; FP/year is
(window FPR) x (windows per year) with the identical windows-per-year formula.

Usage (from anywhere):
    python scripts/evaluation/matched_filter_benchmark.py            # full run
    python scripts/evaluation/matched_filter_benchmark.py --quick    # smoke run
"""

from __future__ import annotations

# --- resolve project-root imports ----------------------------------------- #
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
for _p in (_PROJECT_ROOT,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --------------------------------------------------------------------------- #

import argparse
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# The user's global matplotlib config may enable full LaTeX text rendering
# (e.g. via scienceplots). Force it OFF for this script: labels containing
# ±/σ/% then render via matplotlib's built-in mathtext and never require a
# TeX toolchain or escaping.
plt.rcParams.update({"text.usetex": False})
from scipy.optimize import curve_fit
from scipy.signal import oaconvolve
from numpy.lib.stride_tricks import sliding_window_view

from data_pre_processing.tiq_data_loader import load_tiq_data_segment
from data_pre_processing.chirp_BW_conv_signal_generation import get_trimmed_waveform


# ===================================================================== #
# Configuration
# ===================================================================== #

M_SOLAR_KG = 1.988e30
CACHE_SCHEMA_VERSION = 2

CONFIG = {
    # ---- Data (anchored to the project root) ----
    "filepath_template": os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
    # Noise-only data for threshold calibration (NOT used for injections):
    "calibration_suffixes": ["19.20.36.730"],
    "calibration_samples_per_file": 112_000_000,
    # Data used for injections / efficiency evaluation:
    "test_suffix": "19.23.28.791",
    "test_samples": 80_000_000,
    "offset": 0,

    # ---- Windowing — MUST match the ML pipeline for comparability ----
    "window_size": 1024,
    "step_size": 1024 // 10,
    "sampling_rate_hz": 14e6,

    # ---- Signal family (identical to the ML injections) ----
    "pbh_masses_solar": list(np.logspace(-13, -6, 8)),
    "f0_gw": 5.0e9,
    "Gamma_gw": 100e3,
    "N_gw": 32768,
    "relative_threshold_factor": 1e-3,     # SAME trimming rule as injection
    "response_mode": "real_lorentzian",

    # ---- Injection convention (identical to the ML pipeline) ----
    # amplitude = target_peak_snr * noise_std_amp  (peak of |s| in raw units)
    "noise_std_amp_injection": 2.7052e-5,  # GLOBAL training std of |noise|
    # Fixed grid (used when adaptive_snr_grid=False):
    "target_peak_snrs": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                         5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    # Adaptive grid: center the injected SNRs on the PREDICTED transition
    # x0 ~ threshold / (conv * match). The matched filter is so sensitive at
    # low masses (SNR95 << 1) and needs > 8 at the highest mass, so a fixed
    # 1..8 grid cannot constrain the sigmoid everywhere.
    "adaptive_snr_grid": True,
    "snr_grid_rel_span": (0.35, 2.6),   # geometric span around predicted x0
    "snr_grid_points": 15,

    # ---- Statistics per (mass, SNR) operating point ----
    "chunk_samples": 8_000_000,    # independent noise stretches per MF pass
    "chunks_per_point": 10,        # "runs"
    "events_per_chunk": 40,        # -> 400 events per operating point

    # ---- Template-mismatch study ----
    # For every injected mass, additionally analyze with "close" templates:
    # template mass = injected mass x factor. With a decade-spaced template
    # bank the worst-case nearest template is half a decade away.
    "run_mismatch_study": True,
    "mismatch_factors": [10 ** -0.5, 10 ** 0.5],

    # ---- FP/year targets ----
    "primary_fp_per_year": 0.99,
    "fp_scan_targets": [0.1, 0.3, 0.99, 3.0, 10.0, 30.0, 100.0, 365.0],

    # ---- Threshold calibration tail fit ----
    # Fit ln(FPR) = a * t + c ("linear_t", the ML pipeline's ansatz) on the
    # relative tail: only points with FPR <= tail_fraction_of_max * max(FPR)
    # enter the fit (0.1% of the maximum measured FPR by default).
    # This benchmark deliberately uses the same linear-in-threshold ansatz as
    # the ML pipeline. Do not silently switch extrapolation models.
    "threshold_fit_model": "linear_t",
    "tail_fraction_of_max": 1e-3,
    "min_tail_points": 5,
    "threshold_sweep_points": 250,

    # ---- Efficiency / summary ----
    "target_efficiency": 0.95,

    # ---- I/O ----
    "output_dir": os.path.join(_PROJECT_ROOT, "matched_filter_results"),
    "random_seed": 42,
}

QUICK_OVERRIDES = {
    "pbh_masses_solar": [1e-13, 1e-6],
    "target_peak_snrs": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
    "calibration_samples_per_file": 16_000_000,
    "test_samples": 16_000_000,
    "chunks_per_point": 2,
    "events_per_chunk": 20,
    "mismatch_factors": [10 ** 0.5],
    "output_dir": os.path.join(_PROJECT_ROOT, "matched_filter_results_quick"),
}


# ===================================================================== #
# Small statistics helpers (self-contained on purpose)
# ===================================================================== #

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def sigmoid_func(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fit_sigmoid_and_snr95(snrs, effs, sigmas, target_efficiency):
    """
    Weighted sigmoid fit; returns dict with fit params, SNR at target
    efficiency, and its statistical error (full covariance propagation).
    Mirrors the math in efficiency_curve.py.
    """
    out = {
        "fit_ok": False, "x0": np.nan, "k": np.nan,
        "x0_err": np.nan, "k_err": np.nan, "cov_x0_k": np.nan,
        "chi2_red": np.nan, "dof": np.nan,
        "snr_at_target": np.nan, "snr_at_target_err": np.nan,
        "note": "",
    }
    snrs = np.asarray(snrs, dtype=float)
    effs = np.asarray(effs, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-6)

    order = np.argsort(snrs)
    s_sorted, e_sorted = snrs[order], effs[order]
    # Measured efficiencies fluctuate statistically. np.interp requires an
    # increasing x-axis, so use a monotone envelope only for crossing-based
    # initial guesses and the no-fit fallback; the fit still sees raw data.
    e_monotone = np.maximum.accumulate(e_sorted)

    # Data-driven initial guesses: x0 from the 50% crossing, k from the
    # 12%..88% transition width (a pure logistic has width 4/k there).
    if e_monotone.max() >= 0.5 >= e_monotone.min():
        x0_guess = float(np.interp(0.5, e_monotone, s_sorted))
    else:
        x0_guess = float(s_sorted[np.argmin(np.abs(e_sorted - 0.5))])
    lo_x = float(np.interp(0.12, e_monotone, s_sorted))
    hi_x = float(np.interp(0.88, e_monotone, s_sorted))
    width = max(hi_x - lo_x, (s_sorted[-1] - s_sorted[0]) / max(len(s_sorted) - 1, 1))
    k_guess = float(np.clip(4.0 / max(width, 1e-3), 0.1, 40.0))

    if e_sorted.min() > target_efficiency:
        out["note"] = "all efficiencies above target — SNR_target below grid"
    elif e_sorted.max() < target_efficiency:
        out["note"] = "all efficiencies below target — SNR_target above grid"

    try:
        popt, pcov = curve_fit(
            sigmoid_func, snrs, effs, p0=[x0_guess, k_guess],
            sigma=sigmas, absolute_sigma=True,
            bounds=([0.0, 0.01], [1e4, 200.0]), maxfev=20000,
        )
        x0, k = popt
        resid = effs - sigmoid_func(snrs, *popt)
        chi2 = float(np.sum((resid / sigmas) ** 2))
        dof = max(len(snrs) - 2, 1)

        # SNR at target efficiency:  x = x0 - ln(1/eff - 1) / k
        A = np.log(1.0 / target_efficiency - 1.0)
        snr_t = x0 - A / k
        dS_dx0 = 1.0
        dS_dk = A / k ** 2
        var = (dS_dx0 ** 2 * pcov[0, 0] + dS_dk ** 2 * pcov[1, 1]
               + 2.0 * dS_dx0 * dS_dk * pcov[0, 1])
        out.update({
            "fit_ok": True, "x0": float(x0), "k": float(k),
            "x0_err": float(np.sqrt(max(pcov[0, 0], 0.0))),
            "k_err": float(np.sqrt(max(pcov[1, 1], 0.0))),
            "cov_x0_k": float(pcov[0, 1]),
            "chi2_red": chi2 / dof, "dof": int(dof),
            "snr_at_target": float(snr_t),
            "snr_at_target_err": float(np.sqrt(var)) if var > 0 else np.nan,
        })
    except Exception as exc:
        print(f"      [warn] sigmoid fit failed: {exc!r} — "
              "falling back to monotone interpolation for SNR_target")
        if e_monotone.min() <= target_efficiency <= e_monotone.max():
            out["snr_at_target"] = float(
                np.interp(target_efficiency, e_monotone, s_sorted)
            )
            out["note"] = "interpolation fallback (no fit error available)"
    return out


# ===================================================================== #
# Core matched filtering
# ===================================================================== #

def windows_per_year(window_size: int, step_size: int, fs: float) -> float:
    """Identical convention to the ML pipeline (efficiency_curve.py)."""
    seconds_per_year = 31_536_000
    total = seconds_per_year * fs
    return (total - window_size) / step_size + 1


def matched_filter_rho(x: np.ndarray, template_hat: np.ndarray, sigma_q: float) -> np.ndarray:
    """
    Standard normalized matched-filter statistic on complex baseband data.

        z(t)   = sum_k x[t+k] conj(s[k])              (FFT overlap-add)
        rho(t) = |z(t)| / ( sigma_q * sqrt(E_s) )

    Under white circular Gaussian noise rho is Rayleigh(1) per lag; for a
    matched injection with amplitude A the expected peak is
    rho_opt = A * sqrt(E_s) / sigma_q.
    """
    if sigma_q <= 0 or not np.isfinite(sigma_q):
        raise ValueError(f"sigma_q must be finite and positive, got {sigma_q!r}.")
    if x.size < template_hat.size:
        raise ValueError(
            f"Input length ({x.size}) is shorter than template length "
            f"({template_hat.size})."
        )
    e_s = float(np.sum(np.abs(template_hat) ** 2))
    if e_s <= 0 or not np.isfinite(e_s):
        raise ValueError("Matched-filter template must have finite non-zero energy.")
    z = oaconvolve(x, np.conj(template_hat[::-1]), mode="valid")
    return (np.abs(z) / (sigma_q * np.sqrt(e_s))).astype(np.float32)


def window_max_scores(rho: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Per-window maximum of rho — the MF analogue of the ML window score."""
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must both be positive.")
    if rho.size < window_size:
        return np.empty(0, dtype=np.float32)
    return sliding_window_view(rho, window_size)[::step_size].max(axis=1)


def scored_window_geometry(n_samples: int, filter_length: int,
                           window_size: int, step_size: int) -> dict:
    """
    Geometry of valid matched-filter lags and their sliding score windows.

    `scored_end` is the exclusive raw/lag coordinate covered by the final
    scored window. Samples after it exist in the raw chunk but cannot start a
    valid scored matched-filter lag, so injections are not placed there.
    """
    values = {
        "n_samples": n_samples, "filter_length": filter_length,
        "window_size": window_size, "step_size": step_size,
    }
    if any(int(v) != v or v <= 0 for v in values.values()):
        raise ValueError(f"Matched-filter geometry values must be positive integers: {values}")
    rho_length = int(n_samples - filter_length + 1)
    if rho_length < window_size:
        raise ValueError(
            "Chunk/filter/window geometry yields no scored windows: "
            f"chunk={n_samples}, filter={filter_length}, valid_rho={rho_length}, "
            f"window={window_size}."
        )
    n_windows = 1 + (rho_length - window_size) // step_size
    scored_end = (n_windows - 1) * step_size + window_size
    return {
        "rho_length": rho_length,
        "n_windows": n_windows,
        "scored_end": scored_end,
        "unscored_raw_tail": n_samples - scored_end,
    }


def overlapping_window_bounds(start: int, end: int, n_windows: int,
                              window_size: int, step_size: int):
    """Inclusive indices of score windows overlapping [start, end), or None."""
    if end <= start:
        raise ValueError(f"Invalid half-open interval [{start}, {end}).")
    first = max(0, (start - window_size) // step_size + 1)
    last = min(n_windows - 1, (end - 1) // step_size)
    return None if first > last else (first, last)


def estimate_noise_stats(x: np.ndarray) -> dict:
    """Per-quadrature std, complex std and amplitude std of a noise stretch."""
    xr, xi = x.real, x.imag
    sigma_q = float(np.sqrt(0.5 * (np.var(xr) + np.var(xi))))
    return {
        "sigma_q": sigma_q,
        "sigma_complex": float(np.sqrt(np.var(xr) + np.var(xi))),
        "sigma_amp": float(np.std(np.abs(x))),
        "mean_amp": float(np.mean(np.abs(x))),
    }


def get_template(mass_solar: float, cfg: dict) -> np.ndarray:
    """Peak-normalized trimmed waveform (copy — never mutate the LRU cache)."""
    w = get_trimmed_waveform(
        mass_solar * M_SOLAR_KG, 1.0,
        cfg["f0_gw"], cfg["Gamma_gw"], cfg["N_gw"], M_SOLAR_KG,
        relative_threshold_factor=cfg["relative_threshold_factor"],
        response_mode=cfg["response_mode"],
    )
    w = np.array(w, dtype=np.complex128, copy=True)
    if w.size == 0:
        raise RuntimeError(f"Empty trimmed waveform for mass {mass_solar:.3e} M_sun")
    peak = np.max(np.abs(w))
    if peak <= 0 or not np.isfinite(peak):
        raise RuntimeError(
            f"Invalid trimmed waveform peak ({peak!r}) for mass "
            f"{mass_solar:.3e} M_sun"
        )
    return w / peak


# ===================================================================== #
# Data loading (chunked)
# ===================================================================== #

def iter_chunks(cfg: dict, suffixes, total_samples_per_file: int):
    """
    Yield complex, mean-subtracted data chunks one at a time (constant
    memory: one chunk per read instead of the whole file).
    """
    nc = cfg["chunk_samples"]
    yielded = 0
    for suffix in suffixes:
        path = cfg["filepath_template"].format(suffix)
        n_chunks = max(total_samples_per_file // nc, 1)
        for j in range(n_chunks):
            data, fs = load_tiq_data_segment(path, cfg["offset"] + j * nc, nc)
            if data is None:
                if j == 0:
                    raise FileNotFoundError(f"Could not load tiq data: {path}")
                print(f"  [warn] chunk {j} of {path} could not be read — stopping file.")
                break
            if j == 0 and abs(fs - cfg["sampling_rate_hz"]) > 1e-3 * cfg["sampling_rate_hz"]:
                print(f"  [warn] file fs={fs:.4g} differs from configured "
                      f"{cfg['sampling_rate_hz']:.4g}; using configured value for FP/year.")
            chunk = np.asarray(data, dtype=np.complex128)
            del data
            if chunk.size < nc:
                print(f"  [warn] chunk {j} of {path} is short ({chunk.size}) — skipped.")
                break
            chunk = chunk - np.mean(chunk)   # remove DC / carrier leakage
            yielded += 1
            yield chunk
    if yielded == 0:
        raise RuntimeError("No data chunks could be built — check paths/sizes.")


def load_chunks(cfg: dict, suffixes, total_samples_per_file: int):
    """Materialized list of chunks (used for the injection data only)."""
    return list(iter_chunks(cfg, suffixes, total_samples_per_file))


def load_one_chunk(cfg: dict, suffix: str, j: int):
    """Load a single mean-subtracted chunk by index (lazy, constant memory)."""
    nc = cfg["chunk_samples"]
    path = cfg["filepath_template"].format(suffix)
    data, _fs = load_tiq_data_segment(path, cfg["offset"] + j * nc, nc)
    if data is None:
        raise FileNotFoundError(f"Could not load chunk {j} of {path}")
    chunk = np.asarray(data, dtype=np.complex128)
    if chunk.size < nc:
        raise RuntimeError(f"Chunk {j} of {path} is short ({chunk.size} < {nc}).")
    return chunk - np.mean(chunk)


# --------------------------------------------------------------------- #
# Atomic result cache (makes the run RESUMABLE: any number of restarts
# skip completed work units; a killed in-flight unit is simply redone)
# --------------------------------------------------------------------- #

def atomic_save_npz(path: str, **arrays) -> None:
    # Unique temp files avoid collisions between advertised --mass-indices
    # worker processes that happen to calculate the same cache unit.
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp.npz",
                               dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_save_json(path: str, value) -> None:
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cache_fingerprint(cfg: dict) -> str:
    """
    Fingerprint values that alter cached noise statistics or score arrays.

    Mass/SNR grids are deliberately excluded so `--mass-indices` processes
    can share the same cache; individual unit filenames already encode them.
    """
    keys = (
        "filepath_template", "calibration_suffixes", "test_suffix", "offset",
        "chunk_samples", "window_size", "step_size", "f0_gw", "Gamma_gw",
        "N_gw", "relative_threshold_factor", "response_mode",
        "noise_std_amp_injection", "events_per_chunk", "random_seed",
        "event_placement_end", "event_placement_guard",
    )
    payload = {"cache_schema_version": CACHE_SCHEMA_VERSION}
    payload.update({key: cfg.get(key) for key in keys})
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def validate_config(cfg: dict, templates: dict = None) -> None:
    """Fail before expensive filtering when the benchmark geometry is invalid."""
    positive_ints = (
        "window_size", "step_size", "chunk_samples", "chunks_per_point",
        "events_per_chunk", "threshold_sweep_points", "min_tail_points",
    )
    for key in positive_ints:
        value = cfg.get(key)
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"CONFIG['{key}'] must be a positive integer, got {value!r}.")
    if cfg.get("threshold_fit_model") != "linear_t":
        raise ValueError(
            "CONFIG['threshold_fit_model'] must be 'linear_t'; t^2 "
            "extrapolation is disabled."
        )
    if cfg["events_per_chunk"] > cfg["chunk_samples"]:
        raise ValueError("events_per_chunk cannot exceed chunk_samples.")
    if cfg["sampling_rate_hz"] <= 0 or cfg["noise_std_amp_injection"] <= 0:
        raise ValueError("sampling_rate_hz and noise_std_amp_injection must be positive.")
    if templates is not None:
        placement_end = cfg.get("event_placement_end")
        placement_guard = cfg.get("event_placement_guard")
        if placement_end is None or placement_guard is None:
            raise ValueError(
                "event_placement_end and event_placement_guard must be derived "
                "from all filter templates before template validation."
            )
        for key, template in templates.items():
            geometry = scored_window_geometry(
                cfg["chunk_samples"], template.size,
                cfg["window_size"], cfg["step_size"],
            )
            if placement_end > geometry["scored_end"]:
                raise ValueError(
                    f"Shared event_placement_end={placement_end} exceeds the "
                    f"score-covered domain for template {key} "
                    f"({geometry['scored_end']})."
                )
            slot = placement_end // cfg["events_per_chunk"]
            if slot < template.size + 2 * placement_guard:
                raise ValueError(
                    f"Template {key} (length {template.size}) cannot fit "
                    f"{cfg['events_per_chunk']} separated events inside the "
                    f"shared placement region ({placement_end} samples): "
                    f"slot={slot}, "
                    f"required>={template.size + 2 * placement_guard}."
                )


# ===================================================================== #
# Threshold calibration (per template)
# ===================================================================== #

def _fpr_curve(scores_sorted: np.ndarray, thresholds: np.ndarray):
    """FPR(t) = P(score >= t) with Wilson errors, computed by binary search."""
    n = scores_sorted.size
    counts = n - np.searchsorted(scores_sorted, thresholds, side="left")
    fpr = counts / n
    lo = np.empty_like(fpr)
    hi = np.empty_like(fpr)
    for i, c in enumerate(counts):
        lo[i], hi[i] = wilson_score_interval(int(c), n)
    sigma = np.maximum((hi - lo) / 2.0, 1e-12)
    return counts, fpr, lo, hi, sigma


def calibrate_threshold(noise_window_scores: np.ndarray, cfg: dict) -> dict:
    """
    Sweep thresholds over noise-only window scores, fit the low-FPR tail and
    return everything needed to (a) extrapolate the threshold for ANY FP/year
    target and (b) propagate the fit uncertainty into a threshold error.

    Fit model:
        ln FPR = a * t + c   (the ML pipeline's linear-tail ansatz)
    """
    s = np.sort(noise_window_scores.astype(np.float64))
    if s.size == 0:
        raise ValueError("Threshold calibration received no noise-window scores.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Threshold calibration scores contain NaN or infinity.")
    t_lo = float(np.quantile(s, 0.50))
    t_hi = float(s[-1]) * 1.02
    thresholds = np.linspace(t_lo, t_hi, cfg["threshold_sweep_points"])
    counts, fpr, lo, hi, sigma = _fpr_curve(s, thresholds)

    # Relative tail definition (as in the ML calibration): only thresholds
    # with FPR <= tail_fraction_of_max * max(FPR) enter the fit.
    fpr_max = float(fpr.max())
    mask = (fpr > 0) & (fpr <= cfg["tail_fraction_of_max"] * fpr_max)
    if int(mask.sum()) < cfg["min_tail_points"]:
        # widen the tail until enough points are available
        mask = fpr > 0
        mask &= fpr <= np.quantile(fpr[fpr > 0], 0.25)
    if int(mask.sum()) < cfg["min_tail_points"]:
        raise RuntimeError("Not enough tail points for the threshold fit — "
                           "increase calibration_samples_per_file.")

    ln_f = np.log(fpr[mask])
    ln_sigma = sigma[mask] / fpr[mask]   # d(ln f) = df / f

    if cfg.get("threshold_fit_model") != "linear_t":
        raise ValueError(
            "matched_filter_benchmark.py supports only threshold_fit_model="
            "'linear_t'; the t^2 extrapolation is intentionally disabled."
        )
    u = thresholds[mask]
    popt, pcov = curve_fit(
        lambda uu, a, c: a * uu + c, u, ln_f,
        sigma=ln_sigma, absolute_sigma=True,
    )
    a, c = (float(v) for v in popt)
    resid = ln_f - (a * u + c)
    dof = max(len(u) - 2, 1)
    chi2_red = float(np.sum((resid / ln_sigma) ** 2)) / dof
    if a >= 0:
        raise RuntimeError(
            "Linear threshold-tail fit has non-negative slope; the measured "
            "FPR tail is not falling."
        )
    fit = {"a": a, "c": c, "cov": pcov.tolist(), "chi2_red": chi2_red, "dof": dof}

    return {
        "model": "linear_t", "fits": {"linear_t": fit},
        "a": a, "c": c, "cov": fit["cov"], "chi2_red": chi2_red,
        "dof": dof, "num_tail_points": int(mask.sum()),
        "num_noise_windows": int(s.size),
        "thresholds": thresholds, "fpr": fpr, "fpr_lo": lo, "fpr_hi": hi,
        "tail_mask": mask,
        "score_max": float(s[-1]),
    }


def threshold_for_fp_per_year(calib: dict, fp_per_year: float, wpy: float,
                              model: str = None):
    """
    Invert the tail fit for a target FP/year and propagate the fit covariance
    into a 1-sigma threshold error.

        f* = fp_per_year / windows_per_year      (target FPR fraction)
        t* = (ln f* - c) / a

        du/da = -u/a,  du/dc = -1/a
        var_t = J Cov J^T
    """
    model = model or calib["model"]
    if model != "linear_t":
        raise ValueError("Only the linear_t threshold extrapolation is supported.")
    fit = calib["fits"][model]
    a, c = fit["a"], fit["c"]
    cov = np.asarray(fit["cov"])
    f_star = fp_per_year / wpy
    u = (np.log(f_star) - c) / a
    J = np.array([-u / a, -1.0 / a])
    var_u = float(J @ cov @ J)
    sigma_u = np.sqrt(max(var_u, 0.0))
    return float(u), float(sigma_u)


# ===================================================================== #
# Injection runs (per mass, per SNR)
# ===================================================================== #

def template_match(inject_hat: np.ndarray, filter_hat: np.ndarray) -> float:
    """
    Standard 'match' between two waveforms: the normalized correlation
    maximized over relative time shift (and phase via the modulus):

        M = max_t |<s_inj(t+.), s_f>| / ( sqrt(E_inj) sqrt(E_f) )  in [0, 1].

    M = 1 for the perfectly matched template; the mismatched expected
    matched-filter SNR is rho_opt_mismatch = M * rho_opt_matched.
    """
    z = oaconvolve(inject_hat, np.conj(filter_hat[::-1]), mode="full")
    e_i = float(np.sum(np.abs(inject_hat) ** 2))
    e_f = float(np.sum(np.abs(filter_hat) ** 2))
    if e_i <= 0 or e_f <= 0 or not np.isfinite(e_i + e_f):
        raise ValueError("Templates must have finite non-zero energy.")
    return float(np.max(np.abs(z)) / np.sqrt(e_i * e_f))


def choose_event_starts(placement_end: int, injection_length: int, n_events: int,
                        guard: int, rng) -> np.ndarray:
    """Choose separated event starts using filter-independent placement rules."""
    if n_events <= 0:
        raise ValueError(f"n_events must be positive, got {n_events}.")
    slot = placement_end // n_events
    required_slot = injection_length + 2 * guard
    if slot < required_slot:
        raise RuntimeError(
            f"Placement region is too short for {n_events} separated events "
            f"(covered={placement_end}, L_inj={injection_length}, slot={slot}, "
            f"required_slot={required_slot}) — reduce events_per_chunk or "
            "increase chunk_samples."
        )
    starts = np.empty(n_events, dtype=np.int64)
    for e in range(n_events):
        slot_start = e * slot + guard
        latest_start = (e + 1) * slot - guard - injection_length
        starts[e] = int(rng.integers(slot_start, latest_start + 1))
    return starts


def run_injection_chunk(noise_chunk, inject_hat, filter_hat, sigma_q, amp,
                        n_events, window_size, step_size, rng,
                        placement_end=None, placement_guard=None):
    """
    Inject `n_events` scaled copies of `inject_hat` into one noise chunk (one
    per equal slot, so events never overlap/merge), matched-filter once with
    `filter_hat` (== inject_hat for the perfectly matched analysis), and
    return (per-event max window score, sidelobe-guarded noise-window scores).
    """
    L_i = inject_hat.size
    L_f = filter_hat.size
    nc = noise_chunk.size
    geometry = scored_window_geometry(nc, L_f, window_size, step_size)
    placement_end = geometry["scored_end"] if placement_end is None else placement_end
    placement_guard = max(window_size, L_i, L_f) if placement_guard is None else placement_guard
    if placement_end > geometry["scored_end"]:
        raise ValueError(
            f"placement_end={placement_end} exceeds current filter's scored "
            f"domain ({geometry['scored_end']})."
        )
    # The main benchmark passes one shared domain/guard derived from all
    # variants, so matched and mismatched filters receive identical injections.
    starts = choose_event_starts(
        placement_end, L_i, n_events, placement_guard, rng
    )
    x = noise_chunk.copy()
    for t0 in starts:
        x[t0:t0 + L_i] += amp * inject_hat

    rho = matched_filter_rho(x, filter_hat, sigma_q)
    wscores = window_max_scores(rho, window_size, step_size)
    nw = wscores.size
    if nw != geometry["n_windows"]:
        raise RuntimeError(
            f"Internal scored-window geometry mismatch: expected "
            f"{geometry['n_windows']}, received {nw}."
        )

    event_scores = np.empty(n_events, dtype=np.float64)
    noise_mask = np.ones(nw, dtype=bool)
    for e, t0 in enumerate(starts):
        t1 = t0 + L_i
        # Event windows: same labeling as the ML pipeline — windows whose
        # SAMPLE range overlaps the injected support [t0, t1).
        bounds = overlapping_window_bounds(t0, t1, nw, window_size, step_size)
        if bounds is None:
            raise RuntimeError(
                "Injected event maps to no scored matched-filter window: "
                f"event={e}, support=[{t0}, {t1}), n_windows={nw}, "
                f"scored_end={geometry['scored_end']}."
            )
        w_first, w_last = bounds
        event_scores[e] = float(wscores[w_first:w_last + 1].max())

        # Noise-window mask for the FP cross-check: the matched filter
        # CORRELATES, so rho(t) is elevated for every filter lag t whose
        # support [t, t+L_f) overlaps the injection, i.e. t in (t0-L_f, t1).
        # Windows containing such lags are signal SIDELOBES, not noise.
        guard_bounds = overlapping_window_bounds(
            t0 - L_f + 1, t1, nw, window_size, step_size
        )
        if guard_bounds is not None:
            g_first, g_last = guard_bounds
            noise_mask[g_first:g_last + 1] = False

    return event_scores, wscores[noise_mask]


def build_snr_grid(x0_pred: float, cfg: dict) -> list:
    """
    Injection grid for one (mass, template) variant.

    With adaptive_snr_grid=True the grid is geometric around the PREDICTED
    sigmoid midpoint x0_pred ~ threshold / (conv * match) so the efficiency
    transition is always well sampled (the fixed 1..8 grid saturates at 1.0
    for low masses and never reaches 95% at the highest mass).
    """
    if not cfg.get("adaptive_snr_grid", False) or not np.isfinite(x0_pred) or x0_pred <= 0:
        return [float(s) for s in cfg["target_peak_snrs"]]
    lo, hi = cfg["snr_grid_rel_span"]
    grid = x0_pred * np.geomspace(lo, hi, cfg["snr_grid_points"])
    # round to 3 significant digits for tidy CSVs / reproducibility
    grid = np.unique([float(f"{v:.3g}") for v in grid])
    return [float(v) for v in grid]


def collect_events(mass_solar, label, inject_hat, filter_hat, sigma_q,
                   snr_list, t_main, cfg, cache_dir):
    """
    All event max-scores for one injected mass / one filter template over the
    (SNR x chunk) grid. The rng seed depends ONLY on (mass, snr, chunk), so
    matched and mismatched analyses see IDENTICAL injections (paired design).

    Every (snr, chunk) unit is cached atomically in `cache_dir`, so a killed
    run resumes exactly where it stopped. Test chunks are loaded lazily (one
    in memory at a time) and reused for all missing SNRs of that chunk.
    """
    amp_unit = cfg["noise_std_amp_injection"]
    n_chunks = cfg["chunks_per_point"]
    mass_tag = f"{mass_solar:.4e}"
    threshold_tag = hashlib.sha256(np.float64(t_main).tobytes()).hexdigest()[:10]

    def unit_path(snr, j):
        return os.path.join(cache_dir,
                            f"ev_{mass_tag}_{label}_snr{snr:.6g}_c{j}"
                            f"_thr{threshold_tag}.npz")

    rows = []
    n_noise_total = 0
    n_noise_above = 0
    for j in range(n_chunks):
        missing = [snr for snr in snr_list if not os.path.exists(unit_path(snr, j))]
        if missing:
            chunk = load_one_chunk(cfg, cfg["test_suffix"], j)
            for snr in missing:
                amp = float(snr) * amp_unit
                rng = np.random.default_rng(
                    (cfg["random_seed"], int(1e6 * mass_solar * 1e13) % (2**31),
                     int(snr * 1000), j)
                )
                ev, noise_sc = run_injection_chunk(
                    chunk, inject_hat, filter_hat, sigma_q, amp,
                    cfg["events_per_chunk"], cfg["window_size"],
                    cfg["step_size"], rng,
                    placement_end=cfg["event_placement_end"],
                    placement_guard=cfg["event_placement_guard"],
                )
                atomic_save_npz(
                    unit_path(snr, j),
                    event_scores=ev,
                    n_noise=np.int64(noise_sc.size),
                    n_noise_above=np.int64(int(np.sum(noise_sc >= t_main))),
                )
            del chunk
        for snr in snr_list:
            with np.load(unit_path(snr, j)) as z:
                ev = z["event_scores"]
                n_noise_total += int(z["n_noise"])
                n_noise_above += int(z["n_noise_above"])
            for score in ev:
                rows.append({"mass_solar": mass_solar,
                             "target_peak_snr": float(snr),
                             "chunk": j, "max_window_score": float(score)})
    return pd.DataFrame(rows), n_noise_total, n_noise_above


# ===================================================================== #
# Efficiency evaluation at a threshold
# ===================================================================== #

def efficiency_table(events_df: pd.DataFrame, threshold: float, cfg: dict) -> pd.DataFrame:
    if events_df.empty:
        raise ValueError("Cannot calculate efficiency from an empty event table.")
    rows = []
    for snr, grp in events_df.groupby("target_peak_snr"):
        n = len(grp)
        k = int((grp["max_window_score"] >= threshold).sum())
        eff = k / n if n else 0.0
        lo, hi = wilson_score_interval(k, n)
        rows.append({
            "target_peak_snr": float(snr), "detected": k, "total_events": n,
            "efficiency": eff, "eff_ci_lower": lo, "eff_ci_upper": hi,
            "sigma_val": max(((hi - eff) + (eff - lo)) / 2.0, 1e-6),
        })
    return pd.DataFrame(rows).sort_values("target_peak_snr").reset_index(drop=True)


def snr95_at_threshold(events_df, threshold, cfg):
    table = efficiency_table(events_df, threshold, cfg)
    fit = fit_sigmoid_and_snr95(
        table["target_peak_snr"], table["efficiency"], table["sigma_val"],
        cfg["target_efficiency"],
    )
    return table, fit


# ===================================================================== #
# Plot helpers
# ===================================================================== #

def plot_calibration(calib, t_main, sigma_t, target_fraction, mass, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    m = calib["tail_mask"]
    ax.errorbar(calib["thresholds"], np.maximum(calib["fpr"], 1e-16),
                yerr=[np.maximum(calib["fpr"] - calib["fpr_lo"], 0),
                      np.maximum(calib["fpr_hi"] - calib["fpr"], 0)],
                fmt="o", ms=2.5, alpha=0.5, label="measured FPR (Wilson)")
    tt = np.linspace(calib["thresholds"][0], max(t_main * 1.05, calib["thresholds"][-1]), 400)
    ax.plot(tt, np.exp(calib["a"] * tt + calib["c"]), "k-.",
            label=f"tail fit ({calib['model']}, chi2_red={calib['chi2_red']:.2f})")
    ax.scatter(calib["thresholds"][m], calib["fpr"][m], s=12, color="tab:red",
               zorder=5, label="tail points used")
    ax.axhline(target_fraction, color="green", ls=":", label="target FPR fraction")
    ax.axvline(t_main, color="orange", lw=2,
               label=f"threshold = {t_main:.3f} $\\pm$ {sigma_t:.3f}")
    ax.axvspan(t_main - sigma_t, t_main + sigma_t, color="orange", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("matched-filter window score threshold")
    ax.set_ylabel("window FPR fraction")
    ax.set_title(f"Threshold calibration | mass={mass:.2e} M_sun")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_efficiency_curve(table, fit, tables_sys, mass, threshold, cfg, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    yerr = [np.maximum(table["efficiency"] - table["eff_ci_lower"], 0),
            np.maximum(table["eff_ci_upper"] - table["efficiency"], 0)]
    ax.errorbar(table["target_peak_snr"], table["efficiency"], yerr=yerr,
                fmt="o", color="black", capsize=3, label="MF data (Wilson 95)")
    xx = np.linspace(float(table["target_peak_snr"].min()),
                     float(table["target_peak_snr"].max()), 300)
    if fit["fit_ok"]:
        ax.plot(xx, sigmoid_func(xx, fit["x0"], fit["k"]), "r--",
                label=f"sigmoid fit (chi2_red={fit['chi2_red']:.2f})")
        ax.axhline(cfg["target_efficiency"], color="gray", ls=":", alpha=0.6)
        ax.axvline(fit["snr_at_target"], color="gray", ls=":", alpha=0.6)
    # systematic band from threshold +/- sigma_threshold
    if tables_sys is not None:
        t_lo_tab, t_hi_tab = tables_sys
        ax.fill_between(
            t_lo_tab["target_peak_snr"],
            t_hi_tab["efficiency"],  # higher threshold -> lower efficiency
            t_lo_tab["efficiency"],
            color="tab:orange", alpha=0.25,
            label=r"threshold $\pm 1\sigma$ systematic",
        )
    ax.set_xlabel("injected PEAK SNR", size=13)
    ax.set_ylabel("event detection efficiency", size=13)
    ax.set_title(f"Matched filter | mass={mass:.2e} M_sun | thr={threshold:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===================================================================== #
# Main driver
# ===================================================================== #

def main(cfg):
    validate_config(cfg)
    out_dir = cfg["output_dir"]
    calib_dir = os.path.join(out_dir, "calibration")
    curve_dir = os.path.join(out_dir, "S_curve_fits")
    csv_dir = os.path.join(curve_dir, "S_curve_fits_csv_files")
    fp_dir = os.path.join(out_dir, "fp_scan")
    sys_dir = os.path.join(out_dir, "threshold_systematics")
    for d in (out_dir, calib_dir, curve_dir, csv_dir, fp_dir, sys_dir):
        os.makedirs(d, exist_ok=True)

    wpy = windows_per_year(cfg["window_size"], cfg["step_size"], cfg["sampling_rate_hz"])
    print(f"windows/year = {wpy:.4e}")

    # ---- 1. Enumerate ALL filter templates upfront -------------------- #
    # (matched masses + mismatch variants), so calibration can run in ONE
    # streaming pass over the noise data with constant memory.
    template_masses = []
    for mass in cfg["pbh_masses_solar"]:
        template_masses.append(float(mass))
        if cfg.get("run_mismatch_study", False):
            for f in cfg["mismatch_factors"]:
                template_masses.append(float(mass * f))
    template_masses = sorted(set(template_masses))
    templates = {f"{m:.6e}": get_template(m, cfg) for m in template_masses}

    # Parallel --mass-indices workers must derive the same conservative event
    # placement geometry as the final all-mass assembly run.
    placement_template_masses = []
    for mass in cfg.get("placement_pbh_masses_solar", cfg["pbh_masses_solar"]):
        placement_template_masses.append(float(mass))
        if cfg.get("run_mismatch_study", False):
            for f in cfg["mismatch_factors"]:
                placement_template_masses.append(float(mass * f))
    placement_templates = []
    for mass in sorted(set(placement_template_masses)):
        key = f"{mass:.6e}"
        placement_templates.append(
            templates[key] if key in templates else get_template(mass, cfg)
        )
    geometries = [
        scored_window_geometry(
            cfg["chunk_samples"], template.size,
            cfg["window_size"], cfg["step_size"],
        )
        for template in placement_templates
    ]
    cfg["event_placement_end"] = min(g["scored_end"] for g in geometries)
    cfg["event_placement_guard"] = max(
        cfg["window_size"], max(template.size for template in placement_templates)
    )
    validate_config(cfg, templates)
    atomic_save_json(os.path.join(out_dir, "config.json"), cfg)
    print(f"\n--- {len(templates)} unique filter templates to calibrate ---")
    print(f"    shared event placement: [0, {cfg['event_placement_end']}) "
          f"with guard={cfg['event_placement_guard']} samples")

    # ---- 2. Streaming calibration over noise-only data ---------------- #
    # Every (template, chunk) unit is cached atomically -> fully resumable.
    cache_id = cache_fingerprint(cfg)
    cache_dir = os.path.join(out_dir, "cache", cache_id)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"--- Calibrating thresholds (resumable cache {cache_id}) ---")

    # Noise statistics from the first calibration chunk (cached for
    # consistency across restarts).
    stats_path = os.path.join(cache_dir, "noise_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            noise_stats = json.load(fh)
    else:
        chunk0 = load_one_chunk(cfg, cfg["calibration_suffixes"][0], 0)
        noise_stats = estimate_noise_stats(chunk0)
        del chunk0
        atomic_save_json(stats_path, noise_stats)
    sigma_q = noise_stats["sigma_q"]
    print(f"    noise stats: sigma_q={sigma_q:.4e}, "
          f"sigma_amp={noise_stats['sigma_amp']:.4e} "
          f"(injection uses noise_std_amp={cfg['noise_std_amp_injection']:.4e})")

    nc = cfg["chunk_samples"]
    n_calib_chunks = max(cfg["calibration_samples_per_file"] // nc, 1) \
        * len(cfg["calibration_suffixes"])

    def calib_unit_path(key, j):
        return os.path.join(cache_dir, f"calib_{key}_c{j}.npz")

    for j in range(n_calib_chunks):
        missing = [k for k in templates if not os.path.exists(calib_unit_path(k, j))]
        if not missing:
            continue
        # map global chunk index -> (file, local index)
        per_file = max(cfg["calibration_samples_per_file"] // nc, 1)
        suffix = cfg["calibration_suffixes"][j // per_file]
        chunk = load_one_chunk(cfg, suffix, j % per_file)
        for k in missing:
            scores = window_max_scores(
                matched_filter_rho(chunk, templates[k], sigma_q),
                cfg["window_size"], cfg["step_size"],
            )
            atomic_save_npz(calib_unit_path(k, j), scores=scores)
        del chunk
        print(f"    calibration chunk {j + 1}/{n_calib_chunks} done")
    print(f"    calibration units complete "
          f"({n_calib_chunks} chunks x {len(templates)} templates)")

    calib_cache: dict = {}
    for key, template_hat in templates.items():
        mass_t = float(key)
        parts = []
        for j in range(n_calib_chunks):
            with np.load(calib_unit_path(key, j)) as z:
                parts.append(z["scores"])
        noise_scores = np.concatenate(parts)
        calib = calibrate_threshold(noise_scores, cfg)
        t_main, sigma_t = threshold_for_fp_per_year(
            calib, cfg["primary_fp_per_year"], wpy
        )
        tag = f"{mass_t:.2e}"
        pd.DataFrame({
            "threshold": calib["thresholds"], "fpr_fraction": calib["fpr"],
            "fpr_lo": calib["fpr_lo"], "fpr_hi": calib["fpr_hi"],
            "in_tail_fit": calib["tail_mask"],
        }).to_csv(os.path.join(calib_dir, f"calibration_points_template_{tag}.csv"),
                  index=False)
        plot_calibration(calib, t_main, sigma_t,
                         cfg["primary_fp_per_year"] / wpy, mass_t,
                         os.path.join(calib_dir, f"calibration_template_{tag}.png"))
        calib_cache[key] = {"template_hat": template_hat, "calib": calib,
                            "t_main": t_main, "sigma_t": sigma_t}

    def calibrated_template(mass_t: float) -> dict:
        return calib_cache[f"{mass_t:.6e}"]

    # ---- 3. Injection phase (test chunks are loaded lazily) ----------- #
    n_test_chunks_avail = max(cfg["test_samples"] // nc, 1)
    if cfg["chunks_per_point"] > n_test_chunks_avail:
        raise RuntimeError(
            f"chunks_per_point={cfg['chunks_per_point']} exceeds available "
            f"test chunks ({n_test_chunks_avail}) — increase test_samples."
        )

    sensitivity_rows = []
    fp_scan_rows = []
    all_eff_rows = []
    mismatch_rows = []

    # ---- 2. Per-mass analysis ---------------------------------------- #
    for mass in cfg["pbh_masses_solar"]:
        print(f"\n================ MASS {mass:.3e} M_sun ================")
        inject_hat = get_template(mass, cfg)
        e_hat = float(np.sum(np.abs(inject_hat) ** 2))
        # peak SNR -> optimal matched-filter SNR conversion (see docs):
        #   rho_opt = peak_snr * noise_std_amp * sqrt(E_hat) / sigma_q
        conv = cfg["noise_std_amp_injection"] * np.sqrt(e_hat) / sigma_q
        print(f"    injected template: L={inject_hat.size}, E_hat={e_hat:.1f}, "
              f"rho_opt = {conv:.3f} x peak_snr")

        # Template variants: perfect match + optional "close" templates.
        variants = [("matched", mass)]
        if cfg.get("run_mismatch_study", False):
            for f in cfg["mismatch_factors"]:
                variants.append((f"mismatch_x{f:.4g}", mass * f))

        mass_tag = f"{mass:.2e}"
        variant_results = []   # (label, table, fit) for the comparison plot

        for label, mass_t in variants:
            print(f"    --- template variant '{label}' "
                  f"(template mass {mass_t:.3e} M_sun) ---")
            entry = calibrated_template(mass_t)
            filter_hat = entry["template_hat"]
            t_main, sigma_t = entry["t_main"], entry["sigma_t"]
            calib = entry["calib"]
            match = template_match(inject_hat, filter_hat)
            print(f"        match M = {match:.4f} | "
                  f"threshold({cfg['primary_fp_per_year']} FP/yr) = "
                  f"{t_main:.4f} +/- {sigma_t:.4f} [{calib['model']}] "
                  f"(tail chi2_red={calib['chi2_red']:.2f}, "
                  f"max noise score={calib['score_max']:.3f})")

            # Predicted sigmoid midpoint in peak SNR (rho_opt ~ threshold):
            x0_pred = t_main / max(conv * match, 1e-9)
            snr_list = build_snr_grid(x0_pred, cfg)
            print(f"        predicted x0 ~ {x0_pred:.3g} peak SNR | "
                  f"injection grid: {snr_list[0]:.3g} .. {snr_list[-1]:.3g} "
                  f"({len(snr_list)} points)")

            # -- Injection grid (identical injections across variants;
            #    unit-cached -> resumable)
            events_df, n_noise_total, n_noise_above = collect_events(
                mass, label, inject_hat, filter_hat, sigma_q,
                snr_list, t_main, cfg, cache_dir,
            )
            events_df["rho_opt"] = events_df["target_peak_snr"] * conv * match
            events_df.to_csv(os.path.join(
                csv_dir, f"event_scores_mass_{mass_tag}_{label}.csv"), index=False)

            # realized FP cross-check on sidelobe-guarded noise windows
            realized_fp_frac = (n_noise_above / n_noise_total) if n_noise_total else 0.0
            realized_fp_per_year = realized_fp_frac * wpy
            print(f"        realized FP (sidelobe-guarded): "
                  f"{n_noise_above} / {n_noise_total} "
                  f"(~{realized_fp_per_year:.3g} FP/yr -- 0 expected)")

            # -- Efficiency + threshold-systematic band
            table, fit = snr95_at_threshold(events_df, t_main, cfg)
            table_lo, fit_lo = snr95_at_threshold(events_df, t_main - sigma_t, cfg)
            table_hi, fit_hi = snr95_at_threshold(events_df, t_main + sigma_t, cfg)

            table["mass_solar"] = mass
            table["template_mass_solar"] = mass_t
            table["variant"] = label
            table["match"] = match
            table["threshold"] = t_main
            table["threshold_err"] = sigma_t
            table["rho_opt_conversion"] = conv
            table.to_csv(os.path.join(
                csv_dir, f"efficiency_curve_mass_{mass_tag}_{label}.csv"), index=False)
            all_eff_rows.append(table)
            variant_results.append((label, match, table, fit))

            snr95 = fit["snr_at_target"]
            sys_plus = (fit_hi["snr_at_target"] - snr95) if fit_hi["fit_ok"] else np.nan
            sys_minus = (snr95 - fit_lo["snr_at_target"]) if fit_lo["fit_ok"] else np.nan
            print(f"        SNR95(peak) = {snr95:.3f} "
                  f"+/- {fit['snr_at_target_err']:.3f} (stat) "
                  f"+{sys_plus:.3f}/-{sys_minus:.3f} (threshold sys)")

            row = {
                "mass_solar": mass,
                "variant": label,
                "template_mass_solar": mass_t,
                "match": match,
                "template_length": int(entry["template_hat"].size),
                "threshold": t_main, "threshold_err": sigma_t,
                "predicted_x0_peak_snr": x0_pred,
                "snr_grid_min": snr_list[0], "snr_grid_max": snr_list[-1],
                "snr95_peak": snr95,
                "snr95_stat_err": fit["snr_at_target_err"],
                "snr95_sys_plus": sys_plus, "snr95_sys_minus": sys_minus,
                "snr95_mf_rho": snr95 * conv * match if np.isfinite(snr95) else np.nan,
                "rho_opt_conversion": conv,
                "fit_x0": fit["x0"], "fit_k": fit["k"],
                "fit_chi2_red": fit["chi2_red"],
                "calib_chi2_red": calib["chi2_red"],
                "realized_fp_per_year_check": realized_fp_per_year,
            }
            mismatch_rows.append(row)

            if label == "matched":
                plot_efficiency_curve(
                    table, fit, (table_lo, table_hi), mass, t_main, cfg,
                    os.path.join(curve_dir, f"efficiency_curve_mass_{mass_tag}.png"),
                )
                sensitivity_rows.append(row)

                # systematic tables saved separately (matched analysis)
                for stag, tab, ft in (("minus_sigma", table_lo, fit_lo),
                                      ("plus_sigma", table_hi, fit_hi)):
                    tab = tab.copy()
                    tab["mass_solar"] = mass
                    tab["snr_at_target"] = ft["snr_at_target"]
                    tab.to_csv(os.path.join(
                        sys_dir, f"efficiency_threshold_{stag}_mass_{mass_tag}.csv"),
                        index=False)

                # -- FP/year scan (matched template, same tail fit)
                for fp in cfg["fp_scan_targets"]:
                    try:
                        t_fp, s_fp = threshold_for_fp_per_year(calib, fp, wpy)
                        _, fit_fp = snr95_at_threshold(events_df, t_fp, cfg)
                        fp_scan_rows.append({
                            "mass_solar": mass, "fp_per_year": fp,
                            "threshold": t_fp, "threshold_err": s_fp,
                            "snr95_peak": fit_fp["snr_at_target"],
                            "snr95_stat_err": fit_fp["snr_at_target_err"],
                            "snr95_mf_rho": fit_fp["snr_at_target"] * conv,
                        })
                    except Exception as exc:
                        print(f"      [warn] FP scan target {fp}/yr failed: {exc!r}")

        # -- per-mass matched-vs-mismatched comparison plot
        if len(variant_results) > 1:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            snr_min = min(float(t["target_peak_snr"].min())
                          for _, _, t, _ in variant_results)
            snr_max = max(float(t["target_peak_snr"].max())
                          for _, _, t, _ in variant_results)
            xx = np.linspace(snr_min, snr_max, 300)
            for label, match, tab, ft in variant_results:
                yerr = [np.maximum(tab["efficiency"] - tab["eff_ci_lower"], 0),
                        np.maximum(tab["eff_ci_upper"] - tab["efficiency"], 0)]
                eb = ax.errorbar(tab["target_peak_snr"], tab["efficiency"],
                                 yerr=yerr, fmt="o", ms=4, capsize=3,
                                 label=f"{label} (M={match:.3f})")
                if ft["fit_ok"]:
                    ax.plot(xx, sigmoid_func(xx, ft["x0"], ft["k"]), "--",
                            color=eb.lines[0].get_color(), alpha=0.7)
            ax.axhline(cfg["target_efficiency"], color="gray", ls=":", alpha=0.6)
            ax.set_xlabel("injected PEAK SNR", size=13)
            ax.set_ylabel("event detection efficiency", size=13)
            ax.set_title(f"Matched vs close templates | mass={mass:.2e} M_sun")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(
                curve_dir, f"mismatch_comparison_mass_{mass_tag}.png"), dpi=200)
            plt.close(fig)

    # ---- 3. Global outputs ------------------------------------------- #
    eff_all = pd.concat(all_eff_rows, ignore_index=True)
    eff_all.to_csv(os.path.join(out_dir, "efficiency_data_raw_all.csv"), index=False)

    sens = pd.DataFrame(sensitivity_rows)
    sens.to_csv(os.path.join(out_dir, "sensitivity_summary.csv"), index=False)

    # all variants (matched + mismatched) in one summary
    mism = pd.DataFrame(mismatch_rows)
    mism.to_csv(os.path.join(out_dir, "mismatch_summary.csv"), index=False)

    # sensitivity curve: SNR95 vs mass with stat + systematic errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ok = sens["snr95_peak"].notna()
    ax.errorbar(sens["mass_solar"][ok], sens["snr95_peak"][ok],
                yerr=sens["snr95_stat_err"][ok],
                fmt="s", capsize=4, ms=6, color="tab:blue",
                label=f"perfect template (stat, {cfg['target_efficiency']*100:.0f}% eff)")
    ax.fill_between(
        sens["mass_solar"][ok],
        (sens["snr95_peak"] - sens["snr95_sys_minus"].fillna(0))[ok],
        (sens["snr95_peak"] + sens["snr95_sys_plus"].fillna(0))[ok],
        alpha=0.25, color="tab:orange", label=r"threshold $\pm 1\sigma$ systematic",
    )
    # overlay the "close template" variants
    for label, grp in mism[mism["variant"] != "matched"].groupby("variant"):
        grp = grp.dropna(subset=["snr95_peak"]).sort_values("mass_solar")
        if len(grp):
            ax.errorbar(grp["mass_solar"], grp["snr95_peak"],
                        yerr=grp["snr95_stat_err"], fmt="o--", ms=4, capsize=3,
                        alpha=0.8, label=f"close template: {label}")
    ax.set_xscale("log")
    ax.set_xlabel(r"PBH mass ($M_\odot$)", size=13)
    ax.set_ylabel(f"peak SNR for {cfg['target_efficiency']*100:.0f}% efficiency", size=13)
    ax.set_title(f"Matched-filter sensitivity (FP < {cfg['primary_fp_per_year']}/yr, "
                 "event-based detection)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_curve.png"), dpi=300)
    plt.close(fig)

    # FP/year scan: SNR95 vs allowed FP/year (+ linear-in-log10 trend)
    fp_df = pd.DataFrame(fp_scan_rows)
    fp_df.to_csv(os.path.join(fp_dir, "snr95_vs_fp_per_year.csv"), index=False)
    if not fp_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        trend_rows = []
        for mass, grp in fp_df.groupby("mass_solar"):
            grp = grp.dropna(subset=["snr95_peak"]).sort_values("fp_per_year")
            if len(grp) < 2:
                continue
            ax.errorbar(grp["fp_per_year"], grp["snr95_peak"],
                        yerr=grp["snr95_stat_err"], fmt="o-", capsize=3, ms=4,
                        label=f"{mass:.1e} $M_\\odot$")
            # trend: SNR95 = alpha + beta * log10(FP/yr)
            coeff = np.polyfit(np.log10(grp["fp_per_year"]), grp["snr95_peak"], 1)
            xx = np.logspace(np.log10(grp["fp_per_year"].min()),
                             np.log10(grp["fp_per_year"].max()), 100)
            ax.plot(xx, coeff[1] + coeff[0] * np.log10(xx), "--", alpha=0.5)
            trend_rows.append({"mass_solar": mass,
                               "trend_intercept_alpha": coeff[1],
                               "trend_slope_beta_per_decade": coeff[0]})
        ax.set_xscale("log")
        ax.axvline(cfg["primary_fp_per_year"], color="k", ls=":", alpha=0.6,
                   label=f"primary target ({cfg['primary_fp_per_year']}/yr)")
        ax.set_xlabel("allowed FP per year", size=13)
        ax.set_ylabel(f"peak SNR for {cfg['target_efficiency']*100:.0f}% efficiency", size=13)
        ax.set_title("Matched filter: SNR95 vs FP/year allowance "
                     "(dashed: linear-in-log10 trend)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fp_dir, "snr95_vs_fp_per_year.png"), dpi=300)
        plt.close(fig)
        pd.DataFrame(trend_rows).to_csv(
            os.path.join(fp_dir, "snr95_fp_trend_fits.csv"), index=False)

    print(f"\n--- Matched-filter benchmark complete. Results in {out_dir} ---")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small smoke-test run (2 masses, fewer events)")
    parser.add_argument("--mass-indices", type=str, default=None,
                        help="comma-separated indices into pbh_masses_solar — "
                             "lets several processes fill the shared result "
                             "cache in parallel (run once WITHOUT this flag "
                             "at the end to assemble all outputs)")
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(QUICK_OVERRIDES)
    cfg["placement_pbh_masses_solar"] = list(cfg["pbh_masses_solar"])
    if args.mass_indices:
        idx = [int(i) for i in args.mass_indices.split(",")]
        cfg["pbh_masses_solar"] = [cfg["pbh_masses_solar"][i] for i in idx]
    main(cfg)
