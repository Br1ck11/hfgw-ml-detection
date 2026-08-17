"""
Clustered false-positive DIAGNOSTICS (D).

IMPORTANT: this metric is a *secondary diagnostic only*. The main result
metrics remain (a) detection efficiency vs injected peak SNR and (b) SNR95
at fixed WINDOW-based FP/year. Nothing here feeds the main efficiency
curves or threshold calibration — it is written to a separate CSV
(`clustered_fp_diagnostics.csv`) for later inspection.

Cluster definition (mechanical):
    1. y_pred = score >= threshold
    2. restrict to noise-only windows (label == 0)
    3. contiguous runs of positive predictions form one cluster
    4. optionally merge clusters separated by <= merge_gap_windows
    5. each cluster counts as ONE false trigger
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

CLUSTERED_FP_COLUMNS = [
    "model_name", "mass", "snr", "threshold", "threshold_source",
    "num_noise_windows", "num_false_positive_windows",
    "num_false_positive_clusters",
    "fp_windows_per_year", "fp_clusters_per_year",
    "mean_cluster_length_windows", "median_cluster_length_windows",
    "max_cluster_length_windows",
    "merge_gap_windows", "window_size", "step_size", "sampling_rate",
]


def _find_clusters(flags: np.ndarray, merge_gap_windows: int = 0):
    """
    Return list of (start, end_exclusive) runs of True in `flags`,
    after merging runs separated by <= merge_gap_windows False entries.
    """
    flags = np.asarray(flags).astype(np.int8)
    if flags.size == 0:
        return []
    diffs = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    clusters = list(zip(starts.tolist(), ends.tolist()))
    if merge_gap_windows > 0 and len(clusters) > 1:
        merged = [clusters[0]]
        for s, e in clusters[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= merge_gap_windows:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        clusters = merged
    return clusters


def compute_clustered_fp_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    window_size: int,
    step_size: int,
    sampling_rate: float,
    merge_gap_windows: int = 0,
    model_name: str = "",
    mass: float = float("nan"),
    snr: float = float("nan"),
    threshold_source: str = "calibrated",
) -> dict:
    """
    Compute clustered-FP diagnostics for one (model, mass, snr, threshold)
    operating point. Window order in `y_true` / `scores` must follow the
    time order of the sliding windows (un-shuffled eval datasets do).

    Returns one row (dict) matching CLUSTERED_FP_COLUMNS.
    """
    y_true = np.asarray(y_true).reshape(-1) > 0.5
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_pred = scores >= threshold

    noise_mask = ~y_true
    num_noise = int(np.sum(noise_mask))

    # Restrict to noise-only windows. Removing signal windows splices the
    # sequence; with un-shuffled windows this is a conservative, mechanical
    # definition (clusters cannot span an injected event).
    fp_flags = y_pred[noise_mask]
    num_fp_windows = int(np.sum(fp_flags))

    clusters = _find_clusters(fp_flags, merge_gap_windows=merge_gap_windows)
    lengths = np.array([e - s for s, e in clusters], dtype=np.float64)
    num_clusters = len(clusters)

    seconds_per_year = 31_536_000
    total_samples_per_year = seconds_per_year * float(sampling_rate)
    windows_per_year = (total_samples_per_year - window_size) / step_size + 1

    fpr_windows = num_fp_windows / num_noise if num_noise > 0 else 0.0
    cluster_rate = num_clusters / num_noise if num_noise > 0 else 0.0

    return {
        "model_name": model_name,
        "mass": mass,
        "snr": snr,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "num_noise_windows": num_noise,
        "num_false_positive_windows": num_fp_windows,
        "num_false_positive_clusters": num_clusters,
        "fp_windows_per_year": fpr_windows * windows_per_year,
        "fp_clusters_per_year": cluster_rate * windows_per_year,
        "mean_cluster_length_windows": float(np.mean(lengths)) if num_clusters else 0.0,
        "median_cluster_length_windows": float(np.median(lengths)) if num_clusters else 0.0,
        "max_cluster_length_windows": float(np.max(lengths)) if num_clusters else 0.0,
        "merge_gap_windows": int(merge_gap_windows),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "sampling_rate": float(sampling_rate),
    }


def compute_clustered_fp_diagnostics_multi(
    runs,
    threshold: float,
    window_size: int,
    step_size: int,
    sampling_rate: float,
    merge_gap_windows: int = 0,
    model_name: str = "",
    mass: float = float("nan"),
    snr: float = float("nan"),
    threshold_source: str = "calibrated",
) -> dict:
    """
    Same as `compute_clustered_fp_diagnostics`, but aggregates several
    independent runs (list of (y_true, scores) pairs) into ONE row.
    Clusters never span run boundaries.
    """
    tot_noise = 0
    tot_fp = 0
    lengths = []
    for y_true, scores in runs:
        y_true = np.asarray(y_true).reshape(-1) > 0.5
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        noise_mask = ~y_true
        fp_flags = (scores >= threshold)[noise_mask]
        tot_noise += int(noise_mask.sum())
        tot_fp += int(fp_flags.sum())
        for s, e in _find_clusters(fp_flags, merge_gap_windows=merge_gap_windows):
            lengths.append(e - s)

    lengths = np.asarray(lengths, dtype=np.float64)
    num_clusters = int(lengths.size)

    seconds_per_year = 31_536_000
    total_samples_per_year = seconds_per_year * float(sampling_rate)
    windows_per_year = (total_samples_per_year - window_size) / step_size + 1

    fpr_windows = tot_fp / tot_noise if tot_noise > 0 else 0.0
    cluster_rate = num_clusters / tot_noise if tot_noise > 0 else 0.0

    return {
        "model_name": model_name,
        "mass": mass,
        "snr": snr,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "num_noise_windows": tot_noise,
        "num_false_positive_windows": tot_fp,
        "num_false_positive_clusters": num_clusters,
        "fp_windows_per_year": fpr_windows * windows_per_year,
        "fp_clusters_per_year": cluster_rate * windows_per_year,
        "mean_cluster_length_windows": float(np.mean(lengths)) if num_clusters else 0.0,
        "median_cluster_length_windows": float(np.median(lengths)) if num_clusters else 0.0,
        "max_cluster_length_windows": float(np.max(lengths)) if num_clusters else 0.0,
        "merge_gap_windows": int(merge_gap_windows),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "sampling_rate": float(sampling_rate),
    }


def append_clustered_fp_row(csv_path: str, row: dict) -> None:
    """Append one diagnostics row to `clustered_fp_diagnostics.csv`."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLUSTERED_FP_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
