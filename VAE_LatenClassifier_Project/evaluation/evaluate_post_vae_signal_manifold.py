"""
Post-VAE verification with the clean-signal CNN autoencoder (F).

Loads:
  * a trained VAE-classifier checkpoint,
  * a trained clean-signal CNN autoencoder (train_clean_signal_autoencoder.py),
  * freshly preprocessed validation/test windows (with injections + metadata).

For every window:
  1. classifier_logit          = VAEClassifier(x)
  2. s_vae                     = VAE.decoder(z_mean)        (deterministic recon)
  3. s_proj                    = CleanSignalAutoencoder(s_vae)
  4. Score A: corr_vae_to_projection   = corr(s_vae, s_proj)
       "Does the VAE decoder output already lie close to the clean-signal manifold?"
     Score B: corr_input_to_projection = corr(x_window, s_proj)
       "Does the original data window contain the signal-like projected shape?"

Everything is saved to a diagnostic CSV (post_vae_manifold_diagnostics.csv)
plus distribution plots split by TP / FP / TN / FN.

This score is deliberately NOT made the primary result — inspect the
distributions first. A later combined score
    a*logit + b*corr_A + c*corr_B
is possible but intentionally not tuned here.

Usage:
    python evaluate_post_vae_signal_manifold.py
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
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap
from vae.model import Sampling, VAEClassifier, QuadratureConv1D
from train_clean_signal_autoencoder import CleanSignalCNNAutoencoder

CUSTOM_OBJECTS = {
    "Sampling": Sampling,
    "VAEClassifier": VAEClassifier,
    "QuadratureConv1D": QuadratureConv1D,
}


# ===================================================================== #
# Configuration
# ===================================================================== #

M_SOLAR = 1.988e30

CONFIG = {
    # Anchored to the project root (bootstrap) so the script works from any CWD.
    "vae_model_path": os.path.join(_PROJECT_ROOT, "runs", "reproduce_Model_2_dec_clas_both_sampling", "checkpoints", "best.keras"),
    "clean_ae_model_path": os.path.join(_PROJECT_ROOT, "runs_clean_signal_autoencoder", "baseline", "best_model.keras"),
    "output_dir": os.path.join(_PROJECT_ROOT, "post_vae_manifold_results"),

    # Decision threshold used ONLY to split TP/FP/TN/FN in the plots.
    # None -> logit 0.0. Use your calibrated threshold for realistic FP sets.
    "preset_threshold": None,

    # ---- Evaluation data (mirrors efficiency_curve prep_config) ----
    "prep_config": {
        "filepath_suffixes": ["19.23.28.791"],
        "filepath_template": os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
        "num_samples_to_read_per_file": 2000000,
        "offset": 0,
        "window_size": 1024,
        "step_size": 1024 // 10,
        "train_ratio": 0.01,
        "val_ratio": 0.98,
        "test_ratio": 0.01,
        "dtype": np.float32,
        "use_amps": True,
        "use_I_Q": False,
        "normalization_type": "zscore",
        # Use the SAVED training stats of the VAE run:
        "global_mean_input": 5.1753e-5,
        "global_std_input": 2.7052e-5,
        "calculate_stats": False,
        "inject_signals": True,
        "snr_based_injection": True,
        "signal_injection_probability": 1.0,
        "num_signals_to_inject_per_segment": {"train": 0, "val": 200, "test": 0},
        "m_PBH_injection_list": [1e-8 * M_SOLAR],   # kg
        "amplitude_spectrum_range": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "custom_noise_std": 2.7052e-5,
        "response_mode": "real_lorentzian",
    },

    "batch_size": 1024,
    "eps": 1e-8,
    "temp_memmap_dir": os.path.join(_PROJECT_ROOT, "temp_post_vae_manifold"),
    "cleanup_temp_dir": True,
}


# ===================================================================== #
# Helpers
# ===================================================================== #

def _batch_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample Pearson correlation over flattened channel+time dims."""
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a_c * b_c, axis=1)
    den = np.linalg.norm(a_c, axis=1) * np.linalg.norm(b_c, axis=1) + eps
    return num / den


def _batch_energy(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(a.reshape(len(a), -1).astype(np.float64) ** 2, axis=1))


def _plot_group_distributions(df, column, threshold, out_path):
    """Histogram of `column` for TP / FP / TN / FN at `threshold`."""
    pred_pos = df["classifier_logit"] >= threshold
    label_pos = df["label"] > 0.5
    groups = {
        "TP": df[column][pred_pos & label_pos],
        "FP": df[column][pred_pos & ~label_pos],
        "TN": df[column][~pred_pos & ~label_pos],
        "FN": df[column][~pred_pos & label_pos],
    }
    colors = {"TP": "tab:green", "FP": "tab:red", "TN": "tab:blue", "FN": "tab:orange"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in groups.items():
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=60, alpha=0.55, label=f"{name} (n={len(vals)})",
                color=colors[name], density=True, log=True)
    ax.set_xlabel(column)
    ax.set_ylabel("density (log)")
    ax.set_title(f"{column} by confusion group (threshold={threshold:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===================================================================== #
# Main
# ===================================================================== #

def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    eps = float(cfg["eps"])

    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    # ---- 1. Load both models ----
    print(f"Loading VAE-classifier: {cfg['vae_model_path']}")
    vae = tf.keras.models.load_model(
        cfg["vae_model_path"], custom_objects=CUSTOM_OBJECTS, compile=False
    )
    print(f"Loading clean-signal autoencoder: {cfg['clean_ae_model_path']}")
    clean_ae = tf.keras.models.load_model(
        cfg["clean_ae_model_path"],
        custom_objects={"CleanSignalCNNAutoencoder": CleanSignalCNNAutoencoder},
        compile=False,
    )

    # ---- 2. Preprocess evaluation windows (with injections + metadata) ----
    prep = dict(cfg["prep_config"])
    prep.update({
        "memmap_dir": cfg["temp_memmap_dir"],
        "save_clean_signals": True,
        "save_metadata": True,
        "include_clean_in_datasets": False,
        "return_tf_datasets": True,
        "return_info": True,
        "tf_batch_size": cfg["batch_size"],
        "tf_shuffle": False,
        "tf_repeat": False,
    })
    _, _, _, info = pre_processing_with_memmap(**prep)

    # Access the val split directly through the memmaps so window indices
    # stay perfectly aligned with the window metadata CSV.
    val_shape = info["val_shape"]
    dtype = np.dtype(info["dtype"])
    channels = info["num_channels"]
    X = np.memmap(info["val_norm_path"], mode="r", dtype=dtype, shape=val_shape)
    Y = np.memmap(info["val_lbl_path"], mode="r", dtype=np.bool_, shape=(val_shape[0],))

    # ---- 3. Window + event metadata for the val split ----
    win_meta = pd.read_csv(info["window_metadata_path"])
    win_meta = (
        win_meta[win_meta["split"] == "val"]
        .sort_values("window_index")
        .reset_index(drop=True)
    )
    ev_meta = pd.read_csv(info["event_metadata_path"])
    ev_mass = ev_meta.set_index("event_id")["mass_solar"].to_dict()
    ev_snr_energy = ev_meta.set_index("event_id")["snr_energy"].to_dict()

    # ---- 4. Score every window ----
    n = val_shape[0]
    bs = cfg["batch_size"]
    rows = []
    print(f"Scoring {n} validation windows ...")
    for start in range(0, n, bs):
        end = min(start + bs, n)
        x = np.asarray(X[start:end], dtype=np.float32)
        if channels == 1 and x.ndim == 2:
            x = x[..., None]
        y = np.asarray(Y[start:end]).astype(np.float32)

        z_mean, z_log_var = vae.encoder(x, training=False)
        cls_features = vae.classifier_features(z_mean, z_log_var)
        logits = np.asarray(vae.classifier(cls_features, training=False)).reshape(-1)
        s_vae = np.asarray(vae.decoder(z_mean, training=False))
        s_proj = np.asarray(clean_ae(s_vae, training=False))

        corr_a = _batch_corr(s_vae, s_proj, eps=eps)   # Score A
        corr_b = _batch_corr(x, s_proj, eps=eps)       # Score B

        rec_energy = _batch_energy(s_vae)
        proj_energy = _batch_energy(s_proj)

        for j in range(end - start):
            w = start + j
            meta = win_meta.iloc[w] if w < len(win_meta) else None
            event_id = int(meta["event_id"]) if meta is not None else -1
            rows.append({
                "window_index": w,
                "classifier_logit": float(logits[j]),
                "sigmoid_score": float(1.0 / (1.0 + np.exp(-logits[j]))),
                "vae_reconstruction_energy": float(rec_energy[j]),
                "projection_energy": float(proj_energy[j]),
                "corr_vae_to_projection": float(corr_a[j]),
                "corr_input_to_projection": float(corr_b[j]),
                "label": float(y[j]),
                "event_id": event_id,
                "mass": float(ev_mass.get(event_id, np.nan)),
                "peak_snr": float(meta["peak_snr_window"]) if meta is not None else np.nan,
                "snr_energy": float(ev_snr_energy.get(event_id, np.nan)),
                "window_start_sample": int(meta["window_start_sample"]) if meta is not None else -1,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "post_vae_manifold_diagnostics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows -> {csv_path}")

    # ---- 5. Distribution plots & summary by confusion group ----
    threshold = float(cfg["preset_threshold"]) if cfg["preset_threshold"] is not None else 0.0
    for column in ("corr_vae_to_projection", "corr_input_to_projection",
                   "vae_reconstruction_energy", "projection_energy"):
        _plot_group_distributions(
            df, column, threshold,
            os.path.join(out_dir, f"dist_{column}.png"),
        )

    pred_pos = df["classifier_logit"] >= threshold
    label_pos = df["label"] > 0.5
    summary = []
    for name, mask in (
        ("TP", pred_pos & label_pos), ("FP", pred_pos & ~label_pos),
        ("TN", ~pred_pos & ~label_pos), ("FN", ~pred_pos & label_pos),
    ):
        sub = df[mask]
        summary.append({
            "group": name,
            "count": int(len(sub)),
            "mean_corr_vae_to_projection": float(sub["corr_vae_to_projection"].mean()) if len(sub) else np.nan,
            "mean_corr_input_to_projection": float(sub["corr_input_to_projection"].mean()) if len(sub) else np.nan,
            "mean_vae_reconstruction_energy": float(sub["vae_reconstruction_energy"].mean()) if len(sub) else np.nan,
            "mean_projection_energy": float(sub["projection_energy"].mean()) if len(sub) else np.nan,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "post_vae_manifold_summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    print(
        "\nCheck 5 (H): TPs should on average have larger corr_vae_to_projection "
        "and corr_input_to_projection than FPs. If not, the signal autoencoder "
        "is not useful as a verifier."
    )

    if cfg["cleanup_temp_dir"] and os.path.exists(cfg["temp_memmap_dir"]):
        shutil.rmtree(cfg["temp_memmap_dir"])

    return csv_path


if __name__ == "__main__":
    main()
