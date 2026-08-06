"""
Inspect pure-noise vs noise+signal window scores at one mass/SNR point.

This is a lightweight diagnostic companion to efficiency_curve.py. It does not
run threshold calibration or event-based efficiency fitting. It simply builds:

* one pure-noise validation dataset
* one signal-injected validation dataset at a chosen PBH mass and SNR

and then plots the classifier/detector score plus a cosine similarity score for
three groups:

* pure_noise
* injected_noise windows
* injected_signal windows

The cosine reference defaults to the empirical mean pure-noise window. A scalar
constant equal to the z-scored noise mean is usually a near-zero vector and is
therefore not a meaningful cosine reference.
"""

from __future__ import annotations

# --- auto-inserted by reorganize.py: resolve project-root imports --------- #
# This script lives in a subfolder; add the project root and the script
# folders to sys.path so `import vae`, `import data_pre_processing` and
# cross-script imports (e.g. `from train import cfg`) keep working when the
# script is run directly from anywhere.
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
for _p in (
    _PROJECT_ROOT,
    _os.path.join(_PROJECT_ROOT, "scripts", "training"),
    _os.path.join(_PROJECT_ROOT, "scripts", "evaluation"),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ------------------------------------------------------------------------- #


import argparse
import os
import shutil
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from efficiency_curve import (
    CUSTOM_OBJECTS,
    _align_model_and_preprocessing,
    _compute_detection_score,
    _format_float_tag,
    _get_detection_score_label,
    _latent_outputs,
    _normalise_detection_score_mode,
    _predict_logits,
    _release_dataset_resources,
    _score_mode_needs_latents,
    pre_processing_with_memmap,
)

try:
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


M_SOLAR = 1.988e30


# -------------------------------------------------------------------------- #
# Configuration matching the current efficiency_curve.py defaults
# -------------------------------------------------------------------------- #

# Anchored to the project root (bootstrap) so the script works from any CWD.
MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "runs_continued", "WithEncoder",
    "reproduce_Model_2_dec_clas_both_sampling_continued", "checkpoints", "best.keras",
)

NORMALIZATION_PARAMS = {"mean_value": 5.1753e-5, "std_dev_value": 2.7052e-5}
NORMALIZATION_MODE = "zscore"

DIAGNOSTIC_MASS_SOLAR = 1e-8
DIAGNOSTIC_SNR = 3.0
DIAGNOSTIC_NUM_SIGNALS_VAL = 100

DETECTION_SCORE_MODE = "logit"
DETECTION_SCORE_CONFIG = {
    "latent_dim": 0,
    "selected_latent_dims": [0, 2, 3, 7, 11],
    "reduction": "l2",
    "use_abs": True,
}

COSINE_REFERENCE_MODE = "noise_mean_window"
# Options:
#   "noise_mean_window"    empirical mean vector over pure-noise windows
#   "constant_one"         constant all-ones reference, i.e. mean/DC direction
#   "constant_noise_mean"  constant scalar mean of pure-noise windows

# Anchored to the project root (bootstrap) so the script works from any CWD.
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "window_score_diagnostics")

PREP_CONFIG = {
    "filepath_suffixes": ["19.20.36.730"],
    "filepath_template": os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
    "num_samples_to_read_per_file": 200000,
    "offset": 0,
    "window_size": 1024,
    "step_size": 1024 // 30,
    "train_ratio": 0.01,
    "val_ratio": 0.98,
    "test_ratio": 0.01,
    "dtype": "float32",
    "use_amps": True,
    "use_I_Q": False,
    "normalization_type": "zscore",
    "global_mean_input": 5.1753e-5,
    "global_std_input": 2.7052e-5,
    "calculate_stats": False,
    "signal_injection_probability": 1.0,
    "num_signals_to_inject_per_segment": {"train": 0, "val": 1, "test": 0},
    "custom_noise_std": 2.7052e-5,
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _build_dataset_config(
    base_config: Dict,
    *,
    inject_signals: bool,
    mass_kg: float,
    snr: float,
    num_signals_val: int,
    memmap_dir: str,
) -> Dict:
    config = dict(base_config)
    config.pop("test_file_suffixes", None)
    config.pop("num_samples_to_read_per_file_threshold_calibration", None)
    config.update({
        "inject_signals": bool(inject_signals),
        "snr_based_injection": True,
        "m_PBH_injection_list": [float(mass_kg)],
        "amplitude_spectrum_range": [float(snr)],
        "num_signals_to_inject_per_segment": {
            "train": 0,
            "val": int(num_signals_val if inject_signals else 0),
            "test": 0,
        },
        "memmap_dir": memmap_dir,
        "return_tf_datasets": True,
        "tf_batch_size": 512,
        "tf_shuffle": False,
        "tf_repeat": False,
    })
    return config


def _collect_scores(
    model,
    dataset,
    detection_score_mode,
    score_context=None,
    collect_latents=None,
):
    xs, ys, logits, scores = [], [], [], []
    z_means, z_logvars = [], []
    need_latents = (
        bool(collect_latents)
        if collect_latents is not None
        else _score_mode_needs_latents(detection_score_mode)
    )

    for x_batch, y_batch in dataset:
        logits_batch = _predict_logits(model, x_batch)
        if need_latents:
            z_mean_batch, z_logvar_batch = _latent_outputs(model, x_batch)
        else:
            z_mean_batch, z_logvar_batch = None, None

        score_batch = _compute_detection_score(
            logits_batch,
            z_mean_batch,
            z_logvar_batch,
            detection_score_mode=detection_score_mode,
            score_context=score_context,
            detection_score_config=DETECTION_SCORE_CONFIG,
        )

        xs.append(np.asarray(x_batch))
        ys.append(np.asarray(y_batch).reshape(-1))
        logits.append(np.asarray(logits_batch).reshape(-1))
        scores.append(np.asarray(score_batch).reshape(-1))
        if z_mean_batch is not None:
            z_means.append(z_mean_batch)
            z_logvars.append(z_logvar_batch)

    if not xs:
        raise RuntimeError("Diagnostic dataset yielded no windows.")

    result = {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "logit": np.concatenate(logits, axis=0),
        "model_score": np.concatenate(scores, axis=0),
    }
    if z_means:
        result["z_mean"] = np.concatenate(z_means, axis=0)
        result["z_log_var"] = np.concatenate(z_logvars, axis=0)
    return result


def _build_cosine_reference(noise_x: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "noise_mean_window":
        ref = np.mean(noise_x, axis=0)
    elif mode == "constant_one":
        ref = np.ones_like(noise_x[0])
    elif mode == "constant_noise_mean":
        ref = np.full_like(noise_x[0], float(np.mean(noise_x)))
    else:
        raise ValueError(
            "Unsupported cosine reference mode. Choose from "
            "['noise_mean_window', 'constant_one', 'constant_noise_mean']."
        )

    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    if ref_norm < 1e-8:
        raise ValueError(
            f"Cosine reference mode '{mode}' produced a near-zero vector. "
            "For z-scored noise, use 'noise_mean_window' or 'constant_one'."
        )
    return ref


def _cosine_scores(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x_flat = x.reshape(x.shape[0], -1)
    ref_flat = reference.reshape(-1)
    denom = np.linalg.norm(x_flat, axis=1) * np.linalg.norm(ref_flat)
    denom = np.where(denom > 1e-12, denom, np.nan)
    return (x_flat @ ref_flat) / denom


def _plot_group_hist(df, column, xlabel, title, save_path, bins=80):
    fig, ax = plt.subplots(figsize=(9, 5))
    for group, color in [
        ("pure_noise", "tab:blue"),
        ("injected_noise", "tab:gray"),
        ("injected_signal", "tab:red"),
    ]:
        values = df.loc[df["group"] == group, column].dropna().to_numpy()
        if values.size:
            ax.hist(values, bins=bins, alpha=0.5, label=f"{group} (n={values.size})", color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_scatter(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for group, color, alpha in [
        ("pure_noise", "tab:blue", 0.25),
        ("injected_noise", "tab:gray", 0.25),
        ("injected_signal", "tab:red", 0.75),
    ]:
        sub = df[df["group"] == group]
        if not sub.empty:
            ax.scatter(
                sub["cosine_score"],
                sub["model_score"],
                s=9,
                alpha=alpha,
                label=group,
                color=color,
            )
    ax.set_xlabel("cosine score")
    ax.set_ylabel("model detector score")
    ax.set_title("Cosine score vs model score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _save_summary(df: pd.DataFrame, output_dir: str) -> None:
    summary = (
        df.groupby("group")[["model_score", "logit", "cosine_score"]]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    summary.to_csv(os.path.join(output_dir, "score_summary_by_group.csv"), index=False)


def run_diagnostic(args) -> str:
    mass_kg = float(args.mass_solar) * M_SOLAR
    prep_config = dict(PREP_CONFIG)

    model_path, prep_config, normalization_params, _ = _align_model_and_preprocessing(
        args.model_path,
        prep_config,
        NORMALIZATION_PARAMS,
        NORMALIZATION_MODE,
    )
    if normalization_params.get("mean_value") is not None:
        prep_config["global_mean_input"] = normalization_params["mean_value"]
    if normalization_params.get("std_dev_value") is not None:
        prep_config["global_std_input"] = normalization_params["std_dev_value"]

    score_mode = _normalise_detection_score_mode(args.detection_score_mode)
    mass_tag = _format_float_tag(args.mass_solar)
    snr_tag = _format_float_tag(args.snr)
    output_dir = _ensure_dir(
        os.path.join(args.output_dir, f"mass_{mass_tag}_Msol_snr_{snr_tag}")
    )

    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    noise_memmap = os.path.join(output_dir, "memmaps_pure_noise")
    signal_memmap = os.path.join(output_dir, "memmaps_signal")

    noise_cfg = _build_dataset_config(
        prep_config,
        inject_signals=False,
        mass_kg=mass_kg,
        snr=args.snr,
        num_signals_val=args.num_signals_val,
        memmap_dir=noise_memmap,
    )
    signal_cfg = _build_dataset_config(
        prep_config,
        inject_signals=True,
        mass_kg=mass_kg,
        snr=args.snr,
        num_signals_val=args.num_signals_val,
        memmap_dir=signal_memmap,
    )

    _, noise_ds, _ = pre_processing_with_memmap(**noise_cfg)
    _, signal_ds, _ = pre_processing_with_memmap(**signal_cfg)

    score_context = None
    if _score_mode_needs_latents(score_mode):
        noise_latent = _collect_scores(model, noise_ds, "logit", collect_latents=True)
        score_context = _compute_score_context_from_noise(noise_latent, score_mode)
        noise_ds = _release_dataset_resources(dataset_obj=noise_ds)
        _, noise_ds, _ = pre_processing_with_memmap(**noise_cfg)

    noise = _collect_scores(model, noise_ds, score_mode, score_context=score_context)
    signal = _collect_scores(model, signal_ds, score_mode, score_context=score_context)

    reference = _build_cosine_reference(noise["x"], args.cosine_reference_mode)
    noise_cos = _cosine_scores(noise["x"], reference)
    signal_cos = _cosine_scores(signal["x"], reference)

    signal_mask = signal["y"] > 0.5
    frames = [
        pd.DataFrame({
            "group": "pure_noise",
            "label": 0,
            "logit": noise["logit"],
            "model_score": noise["model_score"],
            "cosine_score": noise_cos,
        }),
        pd.DataFrame({
            "group": np.where(signal_mask, "injected_signal", "injected_noise"),
            "label": signal["y"].astype(int),
            "logit": signal["logit"],
            "model_score": signal["model_score"],
            "cosine_score": signal_cos,
        }),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["snr"] = float(args.snr)
    df["mass_solar"] = float(args.mass_solar)
    df["cosine_reference_mode"] = args.cosine_reference_mode
    df["detection_score_mode"] = score_mode
    df.to_csv(os.path.join(output_dir, "window_scores.csv"), index=False)

    np.save(os.path.join(output_dir, "cosine_reference.npy"), reference)
    _save_summary(df, output_dir)

    score_label = _get_detection_score_label(score_mode, score_context)
    _plot_group_hist(
        df,
        "model_score",
        score_label,
        f"Model score distributions | mass={args.mass_solar:.1e} M_solar | SNR={args.snr:g}",
        os.path.join(output_dir, "model_score_histogram.png"),
    )
    _plot_group_hist(
        df,
        "cosine_score",
        "cosine similarity",
        f"Cosine score distributions | reference={args.cosine_reference_mode}",
        os.path.join(output_dir, "cosine_score_histogram.png"),
    )
    _plot_scatter(df, os.path.join(output_dir, "cosine_vs_model_score.png"))

    noise_ds = _release_dataset_resources(dataset_obj=noise_ds)
    signal_ds = _release_dataset_resources(dataset_obj=signal_ds)
    for path in (noise_memmap, signal_memmap):
        if os.path.exists(path):
            shutil.rmtree(path)
    return output_dir


def _compute_score_context_from_noise(noise_data, score_mode):
    from efficiency_curve import _build_detection_score_context

    return _build_detection_score_context(
        noise_data.get("z_mean"),
        noise_data.get("z_log_var"),
        detection_score_mode=score_mode,
        detection_score_config=DETECTION_SCORE_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--mass-solar", type=float, default=DIAGNOSTIC_MASS_SOLAR)
    parser.add_argument("--snr", type=float, default=DIAGNOSTIC_SNR)
    parser.add_argument("--num-signals-val", type=int, default=DIAGNOSTIC_NUM_SIGNALS_VAL)
    parser.add_argument("--detection-score-mode", default=DETECTION_SCORE_MODE)
    parser.add_argument("--cosine-reference-mode", default=COSINE_REFERENCE_MODE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = run_diagnostic(args)
    print(f"Saved diagnostic outputs under: {output_dir}")


if __name__ == "__main__":
    main()
