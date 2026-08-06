"""
Post-training comparison script for ablation runs.

Why this exists
---------------
The ablation sweep answers a causal training question, but the most useful
post-hoc diagnostic is visual:

* how expressive is each latent space?
* does the decoder reconstruct the injected signal, or only the noise mean?
* do those conclusions change when we rerun on a fresh stochastic-noise
  realization?

This script builds one shared evaluation dataset, runs every finished
ablation checkpoint on that same data, and writes:

* per-model latent scatter / latent-dim stats / logit histogram
* cross-model latent-space comparison grids
* shared signal/noise reconstruction overlays
* selected-example latent heatmaps for every mode
* a compact CSV summary of logit separation, latent gap, and recon error

Usage
-----
    python compare_ablation_analysis.py

Edit the CONFIGURATION block below for:
* which ablation root to scan
* which files / mass / SNR to inject for evaluation
* whether reconstructions should decode from z_mean or a sampled z
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


import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import Sampling, VAEClassifier, VAEClassifierAblation, QuadratureConv1D
from vae.analysis import (
    plot_latent_scatter,
    plot_latent_dim_stats,
    plot_logit_histogram,
)
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap


# ===================================================================== #
# Configuration
# ===================================================================== #

# Anchored to the project root (bootstrap) so the script works from any CWD.
ABLATION_ROOT = os.path.join(_PROJECT_ROOT, "runs_ablation", "vae_cls_ablation")

# If None, auto-discover all finished checkpoints under ABLATION_ROOT and use
# the mode order recorded in ablation_config.json when available.
TARGET_MODES = None

ANALYSIS_FILE_SUFFIXES = ["19.23.28.791"]
ANALYSIS_PBH_MASSES_SOLAR = [1e-8]
ANALYSIS_SNR_VALUES = [5.0]
ANALYSIS_NUM_SIGNALS_VAL = 100
ANALYSIS_NUM_SAMPLES = 1_120_000
ANALYSIS_MAX_BATCHES = 40
EVAL_BATCH_SIZE = 1024

# Leave these as None in normal use to load the saved training stats from
# stats_dir / memmap_dir. Set them explicitly if you want to analyze an older
# run whose shared memmaps stats may since have been overwritten.
ANALYSIS_GLOBAL_MEAN = None
ANALYSIS_GLOBAL_STD = None
ANALYSIS_CUSTOM_NOISE_STD = None

# Decoder visualisation mode:
#   "mean"   -> decode from z_mean (stable, deterministic)
#   "sample" -> decode from one sampled z (matches the training path more closely)
DECODER_RECON_INPUT = "mean"

# Fresh noise realizations are the point of rerunning this script. The seed only
# fixes the stochastic classifier/reconstruction sampling inside the script.
ANALYSIS_RANDOM_SEED = 42


CUSTOM_OBJECTS = {
    "Sampling": Sampling,
    "VAEClassifier": VAEClassifier,
    "VAEClassifierAblation": VAEClassifierAblation,
    "QuadratureConv1D": QuadratureConv1D,
}


# ===================================================================== #
# Helpers
# ===================================================================== #

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_ablation_root(root_path: str) -> str:
    root = os.path.abspath(root_path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Ablation root does not exist: {root_path}")
    return root


def _load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _resolve_stats_dir(saved_cfg: dict, script_dir: str) -> str:
    stats_dir = saved_cfg.get("stats_dir")
    if stats_dir in (None, "", "null"):
        stats_dir = saved_cfg.get("memmap_dir", os.path.join(_PROJECT_ROOT, "memmaps"))
    if not os.path.isabs(stats_dir):
        stats_dir = os.path.abspath(os.path.join(script_dir, stats_dir))
    return stats_dir


def _load_saved_normalization_stats(saved_cfg: dict, script_dir: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    stats_dir = _resolve_stats_dir(saved_cfg, script_dir)
    mean_path = os.path.join(stats_dir, "global_mean.npy")
    std_path = os.path.join(stats_dir, "global_std.npy")
    std_mag_path = os.path.join(stats_dir, "global_std_mag.npy")

    if not (os.path.isfile(mean_path) and os.path.isfile(std_path)):
        return None, None, None

    global_mean = float(np.load(mean_path))
    global_std = float(np.load(std_path))
    global_std_mag = float(np.load(std_mag_path)) if os.path.isfile(std_mag_path) else None
    return global_mean, global_std, global_std_mag


def _resolve_dtype(dtype_value):
    if dtype_value is None:
        return np.float32
    if dtype_value == "float32":
        return np.float32
    if dtype_value == "float64":
        return np.float64
    return dtype_value


def _discover_mode_dirs(ablation_root: str, target_modes: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    config_path = os.path.join(ablation_root, "ablation_config.json")
    mode_order = []
    if os.path.isfile(config_path):
        mode_order = _load_json(config_path).get("MODES", [])

    if target_modes is None:
        candidate_modes = mode_order if mode_order else sorted(os.listdir(ablation_root))
    else:
        candidate_modes = list(target_modes)

    discovered = []
    for mode in candidate_modes:
        ckpt_path = os.path.join(ablation_root, mode, "checkpoints", "best.keras")
        if os.path.isfile(ckpt_path):
            discovered.append((mode, ckpt_path))

    if target_modes is None and not mode_order:
        extra_dirs = sorted(
            d for d in os.listdir(ablation_root)
            if os.path.isdir(os.path.join(ablation_root, d))
            and d not in {mode for mode, _ in discovered}
        )
        for mode in extra_dirs:
            ckpt_path = os.path.join(ablation_root, mode, "checkpoints", "best.keras")
            if os.path.isfile(ckpt_path):
                discovered.append((mode, ckpt_path))

    if not discovered:
        raise FileNotFoundError(
            f"No finished ablation checkpoints found under {ablation_root}."
        )
    return discovered


def _build_preprocessing_config(ablation_cfg: dict) -> dict:
    script_dir = _PROJECT_ROOT  # patched by reorganize.py: anchor to project root
    pbh_masses_kg = [m * float(ablation_cfg.get("M_solar", 1.988e30)) for m in ANALYSIS_PBH_MASSES_SOLAR]

    saved_mean, saved_std, saved_std_mag = _load_saved_normalization_stats(
        ablation_cfg, script_dir
    )

    analysis_mean = ANALYSIS_GLOBAL_MEAN if ANALYSIS_GLOBAL_MEAN is not None else saved_mean
    analysis_std = ANALYSIS_GLOBAL_STD if ANALYSIS_GLOBAL_STD is not None else saved_std
    analysis_noise_std = (
        ANALYSIS_CUSTOM_NOISE_STD
        if ANALYSIS_CUSTOM_NOISE_STD is not None
        else saved_std_mag
    )

    prep_config = {
        "filepath_suffixes": ANALYSIS_FILE_SUFFIXES,
        "filepath_template": ablation_cfg["filepath_template"],
        "num_samples_to_read_per_file": ANALYSIS_NUM_SAMPLES,
        "offset": int(ablation_cfg.get("offset", 0)),
        "window_size": int(ablation_cfg["window_size"]),
        "step_size": int(ablation_cfg["step_size"]),
        "train_ratio": 0.01,
        "val_ratio": 0.98,
        "test_ratio": 0.01,
        "dtype": _resolve_dtype(ablation_cfg.get("dtype", "float32")),
        "normalization_type": ablation_cfg.get("normalization_type", "zscore"),
        "global_mean_input": analysis_mean,
        "global_std_input": analysis_std,
        "calculate_stats": (analysis_mean is None or analysis_std is None),
        "use_amps": bool(ablation_cfg.get("use_amps", True)),
        "use_I_Q": bool(ablation_cfg.get("use_I_Q", False)),
        "inject_signals": True,
        "signal_injection_probability": 1.0,
        "m_PBH_injection_list": pbh_masses_kg,
        "amplitude_spectrum_range": ANALYSIS_SNR_VALUES,
        "num_signals_to_inject_per_segment": {
            "train": 0,
            "val": ANALYSIS_NUM_SIGNALS_VAL,
            "test": 0,
        },
        "snr_based_injection": bool(ablation_cfg.get("snr_based_injection", True)),
        "custom_noise_std": analysis_noise_std,
        "f0_gw": float(ablation_cfg.get("f0_gw", 5.0e9)),
        "Gamma_gw": float(ablation_cfg.get("Gamma_gw", 100e3)),
        "N_gw": int(ablation_cfg.get("N_gw", 32768)),
        "M_solar": float(ablation_cfg.get("M_solar", 1.988e30)),
        "memmap_dir": ablation_cfg.get("memmap_dir", os.path.join(_PROJECT_ROOT, "memmaps")),
        "stats_dir": ablation_cfg.get("stats_dir"),
        "return_tf_datasets": True,
        "tf_batch_size": EVAL_BATCH_SIZE,
        "tf_shuffle": False,
        "tf_repeat": False,
    }
    return prep_config


def _materialize_dataset(dataset, max_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    x_batches, y_batches = [], []
    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break
        x, y = batch[0], batch[1]
        x_batches.append(np.asarray(x))
        y_batches.append(np.asarray(y).reshape(-1))
    if not x_batches:
        raise RuntimeError("Evaluation dataset yielded no batches.")
    x_all = np.concatenate(x_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)
    return x_all, y_all


def _iter_array_batches(x_all: np.ndarray, batch_size: int):
    for start in range(0, len(x_all), batch_size):
        yield x_all[start:start + batch_size]


def _score_model_on_array(model, x_all: np.ndarray, batch_size: int, random_seed: int) -> Dict[str, np.ndarray]:
    tf.random.set_seed(random_seed)
    logits_batches, z_mean_batches, z_logvar_batches = [], [], []

    for x_batch_np in _iter_array_batches(x_all, batch_size):
        x_batch = tf.convert_to_tensor(x_batch_np)
        z_mean, z_log_var = model.encoder(x_batch, training=False)
        logits = model(x_batch, training=False)

        logits_batches.append(np.asarray(logits).reshape(-1))
        z_mean_batches.append(np.asarray(z_mean))
        z_logvar_batches.append(np.asarray(z_log_var))

    return {
        "logits_all": np.concatenate(logits_batches, axis=0),
        "z_mean_all": np.concatenate(z_mean_batches, axis=0),
        "z_logvar_all": np.concatenate(z_logvar_batches, axis=0),
    }


def _decode_examples(
    model,
    x_examples: np.ndarray,
    decoder_recon_input: str,
    random_seed: int,
) -> np.ndarray:
    x_tensor = tf.convert_to_tensor(x_examples)
    z_mean, z_log_var = model.encoder(x_tensor, training=False)
    if decoder_recon_input == "sample":
        tf.random.set_seed(random_seed)
        z_dec = model.sampling([z_mean, z_log_var])
    else:
        z_dec = z_mean
    recon = model.decoder(z_dec, training=False)
    return np.asarray(recon)


def _choose_shared_example_indices(y_all: np.ndarray, logits_by_mode: Dict[str, np.ndarray]) -> Tuple[Optional[int], Optional[int]]:
    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask

    sig_idx = None
    if np.any(sig_mask):
        mode_logits = np.stack([logits[sig_mask] for logits in logits_by_mode.values()], axis=0)
        mean_sig_logits = np.mean(mode_logits, axis=0)
        sig_candidates = np.where(sig_mask)[0]
        sig_idx = int(sig_candidates[np.argmax(mean_sig_logits)])

    noise_idx = None
    if np.any(noise_mask):
        mode_logits = np.stack([logits[noise_mask] for logits in logits_by_mode.values()], axis=0)
        mean_noise_logits = np.mean(mode_logits, axis=0)
        noise_candidates = np.where(noise_mask)[0]
        noise_idx = int(noise_candidates[np.argmin(mean_noise_logits)])

    return sig_idx, noise_idx


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _latent_kl_per_dim(z_mean_all: np.ndarray, z_logvar_all: np.ndarray) -> np.ndarray:
    return 0.5 * np.mean(
        np.square(z_mean_all) + np.exp(z_logvar_all) - 1.0 - z_logvar_all,
        axis=0,
    )


def _plot_reconstruction_overlay(
    x_example: np.ndarray,
    recon_by_mode: Dict[str, np.ndarray],
    title: str,
    save_path: str,
) -> None:
    num_channels = x_example.shape[1]
    zoom_center = int(np.argmax(np.abs(x_example[:, 0])))
    zoom_half_width = min(128, x_example.shape[0] // 4)
    zoom_start = max(0, zoom_center - zoom_half_width)
    zoom_end = min(x_example.shape[0], zoom_center + zoom_half_width)

    if num_channels == 1:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        for ax, x_slice, panel_title in (
            (axes[0], slice(None), "full window"),
            (axes[1], slice(zoom_start, zoom_end), "zoom"),
        ):
            ax.plot(
                np.arange(x_example.shape[0])[x_slice],
                x_example[x_slice, 0],
                color="black",
                linewidth=1.2,
                alpha=0.85,
                label="input",
            )
            for mode, recon in recon_by_mode.items():
                ax.plot(
                    np.arange(x_example.shape[0])[x_slice],
                    recon[x_slice, 0],
                    linewidth=1.3,
                    alpha=0.9,
                    label=mode,
                )
            ax.set_ylabel("amplitude")
            ax.set_title(panel_title)
            ax.grid(alpha=0.3)
        axes[1].set_xlabel("time step")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return

    # I/Q mode: plot derived amplitude because a five-model overlay of both
    # quadratures becomes unreadable quickly.
    input_mag = np.sqrt(np.sum(np.square(x_example), axis=1))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, x_slice, panel_title in (
        (axes[0], slice(None), "full window"),
        (axes[1], slice(zoom_start, zoom_end), "zoom"),
    ):
        ax.plot(
            np.arange(x_example.shape[0])[x_slice],
            input_mag[x_slice],
            color="black",
            linewidth=1.2,
            alpha=0.85,
            label="|input|",
        )
        for mode, recon in recon_by_mode.items():
            recon_mag = np.sqrt(np.sum(np.square(recon), axis=1))
            ax.plot(
                np.arange(x_example.shape[0])[x_slice],
                recon_mag[x_slice],
                linewidth=1.3,
                alpha=0.9,
                label=mode,
            )
        ax.set_ylabel("|x|")
        ax.set_title(panel_title)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("time step")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_selected_latent_heatmaps(
    outputs_by_mode: Dict[str, Dict[str, np.ndarray]],
    example_idx: int,
    title_prefix: str,
    save_path: str,
) -> None:
    mode_names = list(outputs_by_mode.keys())
    z_means = np.stack([outputs_by_mode[mode]["z_mean_all"][example_idx] for mode in mode_names], axis=0)
    sigmas = np.stack(
        [np.exp(0.5 * outputs_by_mode[mode]["z_logvar_all"][example_idx]) for mode in mode_names],
        axis=0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im0 = axes[0].imshow(z_means, aspect="auto", cmap="RdBu_r")
    axes[0].set_ylabel("mode")
    axes[0].set_yticks(np.arange(len(mode_names)))
    axes[0].set_yticklabels(mode_names)
    axes[0].set_title(f"{title_prefix}: z_mean")
    fig.colorbar(im0, ax=axes[0], label="z_mean")

    im1 = axes[1].imshow(sigmas, aspect="auto", cmap="viridis")
    axes[1].set_ylabel("mode")
    axes[1].set_yticks(np.arange(len(mode_names)))
    axes[1].set_yticklabels(mode_names)
    axes[1].set_xlabel("latent dimension")
    axes[1].set_title(f"{title_prefix}: sigma = exp(0.5 * z_log_var)")
    fig.colorbar(im1, ax=axes[1], label="sigma")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_latent_scatter_grid(
    outputs_by_mode: Dict[str, Dict[str, np.ndarray]],
    y_all: np.ndarray,
    save_path: str,
) -> None:
    mode_names = list(outputs_by_mode.keys())
    n = len(mode_names)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask

    for ax, mode in zip(axes.flat, mode_names):
        z = outputs_by_mode[mode]["z_mean_all"]
        z_centered = z - z.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(z_centered, full_matrices=False)
        emb = u[:, :2] * s[:2]

        ax.scatter(
            emb[noise_mask, 0], emb[noise_mask, 1],
            s=6, alpha=0.35, label="noise", c="tab:blue",
        )
        ax.scatter(
            emb[sig_mask, 0], emb[sig_mask, 1],
            s=16, alpha=0.8, label="signal", c="tab:red",
        )
        ax.set_title(mode)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("Latent space comparison (each subplot uses its own PCA basis)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_latent_gap_and_kl(
    outputs_by_mode: Dict[str, Dict[str, np.ndarray]],
    y_all: np.ndarray,
    save_path: str,
) -> None:
    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask
    latent_dim = next(iter(outputs_by_mode.values()))["z_mean_all"].shape[1]
    x = np.arange(latent_dim)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for mode, outputs in outputs_by_mode.items():
        z = outputs["z_mean_all"]
        lv = outputs["z_logvar_all"]
        mean_gap = z[sig_mask].mean(axis=0) - z[noise_mask].mean(axis=0)
        kl_dim = _latent_kl_per_dim(z, lv)
        axes[0].plot(x, mean_gap, marker="o", linewidth=1.3, label=mode)
        axes[1].plot(x, kl_dim, marker="o", linewidth=1.3, label=mode)

    axes[0].set_ylabel("signal-noise mean gap")
    axes[0].set_title("Per-dimension latent mean gap")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_ylabel("per-dim KL")
    axes[1].set_xlabel("latent dimension")
    axes[1].set_title("Per-dimension KL usage")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _collect_summary_rows(
    outputs_by_mode: Dict[str, Dict[str, np.ndarray]],
    y_all: np.ndarray,
    sig_idx: Optional[int],
    noise_idx: Optional[int],
    recon_signal_by_mode: Dict[str, np.ndarray],
    recon_noise_by_mode: Dict[str, np.ndarray],
    x_signal: Optional[np.ndarray],
    x_noise: Optional[np.ndarray],
) -> List[Dict[str, float]]:
    rows = []
    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask

    for mode, outputs in outputs_by_mode.items():
        z = outputs["z_mean_all"]
        lv = outputs["z_logvar_all"]
        logits = outputs["logits_all"]
        mean_sig = z[sig_mask].mean(axis=0) if np.any(sig_mask) else np.zeros(z.shape[1])
        mean_noise = z[noise_mask].mean(axis=0) if np.any(noise_mask) else np.zeros(z.shape[1])
        latent_gap_l2 = float(np.linalg.norm(mean_sig - mean_noise))
        kl_mean = float(np.mean(_latent_kl_per_dim(z, lv)))

        row = {
            "mode": mode,
            "mean_signal_logit": float(np.mean(logits[sig_mask])) if np.any(sig_mask) else np.nan,
            "mean_noise_logit": float(np.mean(logits[noise_mask])) if np.any(noise_mask) else np.nan,
            "max_signal_logit": float(np.max(logits[sig_mask])) if np.any(sig_mask) else np.nan,
            "min_noise_logit": float(np.min(logits[noise_mask])) if np.any(noise_mask) else np.nan,
            "latent_gap_l2": latent_gap_l2,
            "mean_per_dim_kl": kl_mean,
        }

        if sig_idx is not None and x_signal is not None:
            recon_signal = recon_signal_by_mode[mode]
            row["selected_signal_logit"] = float(logits[sig_idx])
            row["selected_signal_prob"] = float(_sigmoid(logits[sig_idx]))
            row["selected_signal_recon_mse"] = float(np.mean(np.square(x_signal - recon_signal)))
        else:
            row["selected_signal_logit"] = np.nan
            row["selected_signal_prob"] = np.nan
            row["selected_signal_recon_mse"] = np.nan

        if noise_idx is not None and x_noise is not None:
            recon_noise = recon_noise_by_mode[mode]
            row["selected_noise_logit"] = float(logits[noise_idx])
            row["selected_noise_prob"] = float(_sigmoid(logits[noise_idx]))
            row["selected_noise_recon_mse"] = float(np.mean(np.square(x_noise - recon_noise)))
        else:
            row["selected_noise_logit"] = np.nan
            row["selected_noise_prob"] = np.nan
            row["selected_noise_recon_mse"] = np.nan

        rows.append(row)

    return rows


# ===================================================================== #
# Main
# ===================================================================== #

if __name__ == "__main__":
    tf.random.set_seed(ANALYSIS_RANDOM_SEED)
    np.random.seed(ANALYSIS_RANDOM_SEED)

    ablation_root = _resolve_ablation_root(ABLATION_ROOT)
    ablation_cfg_path = os.path.join(ablation_root, "ablation_config.json")
    if not os.path.isfile(ablation_cfg_path):
        raise FileNotFoundError(
            f"Missing ablation_config.json under {ablation_root}. "
            "Run train_ablation.py first."
        )
    ablation_cfg = _load_json(ablation_cfg_path)
    mode_dirs = _discover_mode_dirs(ablation_root, TARGET_MODES)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _ensure_dir(os.path.join(ablation_root, f"analysis_compare_{timestamp}"))
    per_model_root = _ensure_dir(os.path.join(output_root, "per_model"))

    prep_config = _build_preprocessing_config(ablation_cfg)
    with open(os.path.join(output_root, "analysis_config_snapshot.json"), "w") as handle:
        json.dump(
            {
                "ABLATION_ROOT": ablation_root,
                "TARGET_MODES": [mode for mode, _ in mode_dirs],
                "ANALYSIS_FILE_SUFFIXES": ANALYSIS_FILE_SUFFIXES,
                "ANALYSIS_PBH_MASSES_SOLAR": ANALYSIS_PBH_MASSES_SOLAR,
                "ANALYSIS_SNR_VALUES": ANALYSIS_SNR_VALUES,
                "ANALYSIS_NUM_SIGNALS_VAL": ANALYSIS_NUM_SIGNALS_VAL,
                "ANALYSIS_NUM_SAMPLES": ANALYSIS_NUM_SAMPLES,
                "ANALYSIS_MAX_BATCHES": ANALYSIS_MAX_BATCHES,
                "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
                "ANALYSIS_GLOBAL_MEAN": ANALYSIS_GLOBAL_MEAN,
                "ANALYSIS_GLOBAL_STD": ANALYSIS_GLOBAL_STD,
                "ANALYSIS_CUSTOM_NOISE_STD": ANALYSIS_CUSTOM_NOISE_STD,
                "DECODER_RECON_INPUT": DECODER_RECON_INPUT,
                "ANALYSIS_RANDOM_SEED": ANALYSIS_RANDOM_SEED,
                "prep_config": prep_config,
            },
            handle,
            indent=2,
            default=str,
        )

    print("--- Preprocessing shared evaluation dataset for ablation comparison ---")
    print(f"    Modes: {[mode for mode, _ in mode_dirs]}")
    print(f"    Files: {ANALYSIS_FILE_SUFFIXES}")
    print(f"    SNRs:  {ANALYSIS_SNR_VALUES}")
    print(f"    Data mode: {'I/Q' if prep_config['use_I_Q'] else 'Amplitude'}")

    _, val_ds, _ = pre_processing_with_memmap(**prep_config)
    x_all, y_all = _materialize_dataset(val_ds, max_batches=ANALYSIS_MAX_BATCHES)
    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask
    print(
        f"    Materialized {len(x_all)} windows "
        f"({int(np.sum(sig_mask))} signal, {int(np.sum(noise_mask))} noise)"
    )

    outputs_by_mode: Dict[str, Dict[str, np.ndarray]] = {}
    logits_by_mode: Dict[str, np.ndarray] = {}

    for mode_idx, (mode, ckpt_path) in enumerate(mode_dirs):
        print(f"--- Loading {mode} from {ckpt_path} ---")
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        outputs = _score_model_on_array(
            model,
            x_all,
            batch_size=EVAL_BATCH_SIZE,
            random_seed=ANALYSIS_RANDOM_SEED + mode_idx,
        )
        outputs["x_all"] = x_all
        outputs["y_all"] = y_all
        outputs_by_mode[mode] = outputs
        logits_by_mode[mode] = outputs["logits_all"]

        model_output_dir = _ensure_dir(os.path.join(per_model_root, mode))
        plot_latent_scatter(outputs, os.path.join(model_output_dir, "latent_scatter.png"))
        plot_latent_dim_stats(outputs, os.path.join(model_output_dir, "latent_dim_stats.png"))
        plot_logit_histogram(outputs, os.path.join(model_output_dir, "logit_histogram.png"))

    sig_idx, noise_idx = _choose_shared_example_indices(y_all, logits_by_mode)
    shared_indices = {"signal_idx": sig_idx, "noise_idx": noise_idx}
    with open(os.path.join(output_root, "shared_example_indices.json"), "w") as handle:
        json.dump(shared_indices, handle, indent=2)

    x_signal = x_all[sig_idx] if sig_idx is not None else None
    x_noise = x_all[noise_idx] if noise_idx is not None else None

    recon_signal_by_mode = {}
    recon_noise_by_mode = {}
    for mode_idx, (mode, ckpt_path) in enumerate(mode_dirs):
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        if sig_idx is not None:
            recon_signal_by_mode[mode] = _decode_examples(
                model,
                x_signal[None, ...],
                decoder_recon_input=DECODER_RECON_INPUT,
                random_seed=ANALYSIS_RANDOM_SEED + 1000 + mode_idx,
            )[0]
        if noise_idx is not None:
            recon_noise_by_mode[mode] = _decode_examples(
                model,
                x_noise[None, ...],
                decoder_recon_input=DECODER_RECON_INPUT,
                random_seed=ANALYSIS_RANDOM_SEED + 2000 + mode_idx,
            )[0]

    if sig_idx is not None and x_signal is not None:
        _plot_reconstruction_overlay(
            x_signal,
            recon_signal_by_mode,
            title=(
                f"Signal reconstruction comparison — SNR={ANALYSIS_SNR_VALUES[0]} "
                f"— decoder input: {DECODER_RECON_INPUT}"
            ),
            save_path=os.path.join(output_root, "compare_signal_reconstruction.png"),
        )
        _plot_selected_latent_heatmaps(
            outputs_by_mode,
            sig_idx,
            title_prefix="Selected signal window",
            save_path=os.path.join(output_root, "compare_signal_latents.png"),
        )

    if noise_idx is not None and x_noise is not None:
        _plot_reconstruction_overlay(
            x_noise,
            recon_noise_by_mode,
            title=(
                f"Noise reconstruction comparison — decoder input: {DECODER_RECON_INPUT}"
            ),
            save_path=os.path.join(output_root, "compare_noise_reconstruction.png"),
        )
        _plot_selected_latent_heatmaps(
            outputs_by_mode,
            noise_idx,
            title_prefix="Selected noise window",
            save_path=os.path.join(output_root, "compare_noise_latents.png"),
        )

    _plot_latent_scatter_grid(
        outputs_by_mode,
        y_all,
        save_path=os.path.join(output_root, "compare_latent_scatter_grid.png"),
    )
    _plot_latent_gap_and_kl(
        outputs_by_mode,
        y_all,
        save_path=os.path.join(output_root, "compare_latent_gap_and_kl.png"),
    )

    summary_rows = _collect_summary_rows(
        outputs_by_mode=outputs_by_mode,
        y_all=y_all,
        sig_idx=sig_idx,
        noise_idx=noise_idx,
        recon_signal_by_mode=recon_signal_by_mode,
        recon_noise_by_mode=recon_noise_by_mode,
        x_signal=x_signal,
        x_noise=x_noise,
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_root, "summary_metrics.csv"), index=False)

    print("\n--- Ablation comparison complete ---")
    print(f"Outputs saved under: {output_root}")
