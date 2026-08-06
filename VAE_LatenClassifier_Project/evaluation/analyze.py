"""
Standalone analysis / xAI entry-point.

Usage
-----
    python analyze.py

Edit the ANALYSIS CONFIGURATION block below to choose which TIQ files
to analyse, which PBH masses and SNR values to inject, and where the
trained checkpoint lives.

Plots produced
--------------
    latent_scatter.png         — 2D embedding of z_mean, coloured by label
    latent_dim_stats.png       — per-dim mean, std, and KL contribution
    logit_histogram.png        — classifier logit, noise vs signal
    composite_signal.png       — input, encoder acts, latent, decoder acts,
                                 and I/Q/amplitude reconstruction comparisons
    composite_noise.png        — same composite, for a clean-noise window

See `vae/analysis.py` for the plotting internals.
"""

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


import os
import json
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from vae import VAEConfig, build_vae_classifier
from vae.model import Sampling, VAEClassifier, QuadratureConv1D
from vae.analysis import analyse_model

# Import the training config for model architecture (does NOT run training
# because train.py is guarded by `if __name__ == "__main__"`).
from train import cfg

from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap


# ===================================================================== #
#                     ANALYSIS CONFIGURATION                            #
#  Edit these values to control what data the analysis runs on.         #
# ===================================================================== #

# --- Data files (use different files than training for a fair test) ---
ANALYSIS_FILE_SUFFIXES = ["19.23.28.791"]

# --- Signal injection: PBH masses (in solar masses) ---
ANALYSIS_PBH_MASSES_SOLAR = [1e-12]

# --- Signal injection: target SNR values ---
ANALYSIS_SNR_VALUES = [2.0]

# --- How many signals to inject into the validation split ---
ANALYSIS_NUM_SIGNALS_VAL = 1

# --- How many raw samples to read per file ---
ANALYSIS_NUM_SAMPLES = 11200000#0

# True: stop if an injection changes zero stored model-input samples.
# False: continue with a warning. Use False only to diagnose failure behavior;
# such zero-change injections cannot support a detection claim.
ANALYSIS_REJECT_UNREPRESENTABLE_INJECTIONS = True

# --- Model path ---
# Can be a run directory (preferred) or a direct .keras checkpoint path.
# Anchored to the project root (bootstrap) so the script works from any CWD.
ANALYSIS_MODEL_PATH = os.path.join(_PROJECT_ROOT, "runs", "WindowSize4096dec_clas_both_samplingUpdatedLosses_1e_minus_12_IQ", "checkpoints")

# --- Precomputed normalization stats from training ---
# Leave these as None in normal use. The script will load the saved GLOBAL
# training stats from stats_dir / memmap_dir so the analyzed inputs are on the
# same scale as during training.
ANALYSIS_GLOBAL_MEAN = 2.3724e-09 # 5.1753e-05
ANALYSIS_GLOBAL_STD = 4.1307e-05 # 2.7052e-5
ANALYSIS_CUSTOM_NOISE_STD = 2.7061e-05 # 2.7052e-5

# --- max batches to feed into the analysis pipeline --- #
ANALYSIS_MAX_BATCHES = 40

# --- Latent value used for classifier scoring and decoder plots ---
# "z_mean": deterministic posterior mean. This was the previous implicit
#           analyze.py behavior.
# "sampled_z": one stochastic posterior sample z = mean + sigma * epsilon.
ANALYSIS_LATENT_MODE = "sampled_z"

# ===================================================================== #
#                     END OF CONFIGURATION                              #
# ===================================================================== #


M_SOLAR = cfg.M_solar


def _resolve_model_checkpoint_path(model_path: str) -> str:
    """Resolve a run directory or file path to a concrete .keras checkpoint."""
    candidate = os.path.abspath(model_path)
    if os.path.isfile(candidate):
        return candidate
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    checkpoint_candidates = [
        os.path.join(candidate, "checkpoints", "best.keras"),
        os.path.join(candidate, "best.keras"),
        os.path.join(candidate, "checkpoint", "best.keras"),
    ]
    for ckpt_path in checkpoint_candidates:
        if os.path.isfile(ckpt_path):
            return ckpt_path

    raise FileNotFoundError(
        f"Could not resolve a .keras checkpoint from '{model_path}'. "
        "Expected a checkpoint file or a directory containing "
        "'checkpoints/best.keras'."
    )


def _infer_run_dir_from_checkpoint(model_file_path: str) -> str:
    """Infer the training run directory from a resolved checkpoint path."""
    checkpoint_dir = os.path.dirname(model_file_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def _load_saved_run_config(run_dir: str) -> dict:
    """
    Load the saved training config if it exists.

    Regular runs store `config.json` directly in the run directory.
    The ablation sweep currently stores a shared `ablation_config.json` one
    directory higher, so we fall back to that when analysing an ablation mode.
    """
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as handle:
            return json.load(handle)

    ablation_config_path = os.path.join(os.path.dirname(run_dir), "ablation_config.json")
    if os.path.isfile(ablation_config_path):
        with open(ablation_config_path, "r") as handle:
            return json.load(handle)

    return {}


def _resolve_stats_dir(saved_cfg: dict) -> str:
    """
    Resolve the stats directory used during training.

    In this project stats_dir defaults to memmap_dir, and both are usually
    relative to the VAE project root rather than the run directory.
    """
    script_dir = _PROJECT_ROOT  # patched by reorganize.py: anchor to project root
    stats_dir = saved_cfg.get("stats_dir")
    if stats_dir in (None, "", "null"):
        stats_dir = saved_cfg.get("memmap_dir", cfg.memmap_dir)
    if not os.path.isabs(stats_dir):
        stats_dir = os.path.abspath(os.path.join(script_dir, stats_dir))
    return stats_dir


def _load_saved_normalization_stats(
    saved_cfg: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Load GLOBAL training stats written by the preprocessing pipeline."""
    stats_dir = _resolve_stats_dir(saved_cfg)
    mean_path = os.path.join(stats_dir, "global_mean.npy")
    std_path = os.path.join(stats_dir, "global_std.npy")
    std_mag_path = os.path.join(stats_dir, "global_std_mag.npy")

    if not (os.path.isfile(mean_path) and os.path.isfile(std_path)):
        return None, None, None

    global_mean = float(np.load(mean_path))
    global_std = float(np.load(std_path))
    global_std_mag = float(np.load(std_mag_path)) if os.path.isfile(std_mag_path) else None
    return global_mean, global_std, global_std_mag


resolved_ckpt_path = _resolve_model_checkpoint_path(ANALYSIS_MODEL_PATH)
run_dir = _infer_run_dir_from_checkpoint(resolved_ckpt_path)
out_dir = os.path.join(run_dir, "analysis")
saved_cfg = _load_saved_run_config(run_dir)

# Keep the analysis preprocessing aligned with the exact checkpoint being loaded.
for key in (
    "window_size",
    "step_size",
    "use_I_Q",
    "use_amps",
    "normalization_type",
    "snr_based_injection",
    "memmap_dir",
    "stats_dir",
):
    if key in saved_cfg:
        setattr(cfg, key, saved_cfg[key])

saved_mean, saved_std, saved_std_mag = _load_saved_normalization_stats(saved_cfg)
analysis_global_mean = ANALYSIS_GLOBAL_MEAN if ANALYSIS_GLOBAL_MEAN is not None else saved_mean
analysis_global_std = ANALYSIS_GLOBAL_STD if ANALYSIS_GLOBAL_STD is not None else saved_std
analysis_custom_noise_std = (
    ANALYSIS_CUSTOM_NOISE_STD
    if ANALYSIS_CUSTOM_NOISE_STD is not None
    else saved_std_mag
)


# --------------------------------------------------------------------- #
# 1. Build the analysis dataset
# --------------------------------------------------------------------- #

print("--- Preprocessing for analysis ---")
print(f"    Files:  {ANALYSIS_FILE_SUFFIXES}")
print(f"    Masses: {ANALYSIS_PBH_MASSES_SOLAR} M_solar")
print(f"    SNRs:   {ANALYSIS_SNR_VALUES}")
print(f"    Checkpoint: {resolved_ckpt_path}")
print(f"    Data mode: {'I/Q' if cfg.use_I_Q else 'Amplitude'}")
print(f"    Quadrature front-end enabled: {cfg.use_quadrature_frontend}")
if cfg.use_quadrature_frontend:
    print(f"    Quadrature output mode: {cfg.quadrature_output_mode}")
print(f"    Normalization mean: {analysis_global_mean}")
print(f"    Normalization std:  {analysis_global_std}")
print(f"    SNR noise std:      {analysis_custom_noise_std}")
print(f"    Analysis latent mode: {ANALYSIS_LATENT_MODE}")
print(
    "    Zero-change injection policy: "
    + (
        "stop"
        if ANALYSIS_REJECT_UNREPRESENTABLE_INJECTIONS
        else "warn and continue"
    )
)

pbh_masses_kg = [m * M_SOLAR for m in ANALYSIS_PBH_MASSES_SOLAR]
injection_counts = {"train": 0, "val": ANALYSIS_NUM_SIGNALS_VAL, "test": 0}

_, val_ds, _, preprocessing_info = pre_processing_with_memmap(
    filepath_suffixes=ANALYSIS_FILE_SUFFIXES,
    filepath_template=cfg.filepath_template,
    num_samples_to_read_per_file=ANALYSIS_NUM_SAMPLES,
    offset=cfg.offset,
    window_size=cfg.window_size,
    step_size=cfg.step_size,
    train_ratio=0.01, val_ratio=0.98, test_ratio=0.01,
    dtype=cfg.dtype,
    normalization_type=cfg.normalization_type,
    global_mean_input=analysis_global_mean,
    global_std_input=analysis_global_std,
    calculate_stats=(analysis_global_mean is None or analysis_global_std is None),
    use_amps=cfg.use_amps, use_I_Q=cfg.use_I_Q,
    inject_signals=True,
    save_clean_signals=True,
    include_clean_in_datasets=True,
    signal_injection_probability=1.0,
    m_PBH_injection_list=pbh_masses_kg,
    amplitude_spectrum_range=ANALYSIS_SNR_VALUES,
    num_signals_to_inject_per_segment=injection_counts,
    snr_based_injection=cfg.snr_based_injection,
    custom_noise_std=analysis_custom_noise_std,
    f0_gw=cfg.f0_gw, Gamma_gw=cfg.Gamma_gw, N_gw=cfg.N_gw, M_solar=cfg.M_solar,
    memmap_dir=cfg.memmap_dir,
    stats_dir=cfg.stats_dir,
    return_tf_datasets=True,
    tf_batch_size=cfg.tf_batch_size,
    tf_shuffle=False,
    tf_repeat=False,
    random_seed=cfg.random_seed,
    reject_unrepresentable_injections=ANALYSIS_REJECT_UNREPRESENTABLE_INJECTIONS,
    return_info=True,
)


# --------------------------------------------------------------------- #
# 2. Load the full model from the .keras checkpoint.
# --------------------------------------------------------------------- #

if os.path.exists(resolved_ckpt_path):
    model = tf.keras.models.load_model(
        resolved_ckpt_path,
        custom_objects={
            "Sampling": Sampling,
            "VAEClassifier": VAEClassifier,
            "QuadratureConv1D": QuadratureConv1D,
        },
        compile=False,
    )
    print(f"Loaded full model from {resolved_ckpt_path}")
else:
    print(
        f"WARNING: no checkpoint at {resolved_ckpt_path} — "
        "building untrained model for inspection."
    )
    model = build_vae_classifier(cfg)
    for batch in val_ds.take(1):
        _ = model(batch[0], training=False)


# --------------------------------------------------------------------- #
# 3. Run the analysis pipeline.
# --------------------------------------------------------------------- #

paths = analyse_model(
    model,
    val_ds,
    output_dir=out_dir,
    max_batches=ANALYSIS_MAX_BATCHES,
    injection_snr=ANALYSIS_SNR_VALUES[0] if ANALYSIS_SNR_VALUES else None,
    window_metadata_path=preprocessing_info.get("window_metadata_path"),
    event_metadata_path=preprocessing_info.get("event_metadata_path"),
    metadata_split="val",
    latent_mode=ANALYSIS_LATENT_MODE,
)

print("\n--- Analysis complete ---")
for name, p in paths.items():
    print(f"  {name:20s} -> {p}")
