"""
Continue / fine-tune an existing VAE checkpoint with new non-architectural
hyperparameters.

Usage
-----
    python continue_train.py

What this script is for
-----------------------
This script is the VAE analogue of a "continued training" script:

1. Load an existing `.keras` checkpoint (or a run directory containing one).
2. Read the saved config of that run as the immutable architecture baseline.
3. Let you override training/data/loss/optimizer/callback hyperparameters.
4. Recompile the loaded model cleanly and continue training.

What you may change here
------------------------
Anything that does not invalidate the loaded weights, for example:
    * file selection / injection settings
    * batch size
    * learning rate / weight decay / Adam betas
    * focal / reconstruction / KL weights
    * early stopping / detection callback settings
    * output directory and run name

What you may NOT change here
----------------------------
Anything baked into the loaded graph and weight shapes, for example:
    * window_size
    * use_amps / use_I_Q
    * num_filters_per_layer
    * kernel_sizes_per_layer
    * strides_per_layer
    * latent_dim
    * quadrature front-end settings
    * classifier_samples_z

If you need to change those, that is not fine-tuning anymore. Build a new
model and train it from scratch.
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


import os
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAEConfig
from vae.model import Sampling, VAEClassifier, QuadratureConv1D
from vae.ablation_model import VAEClassifierAblation
from vae.callbacks import BetaAnnealing, EventDetectionCallback, ScoreQuantileLogger
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap
from data_pre_processing.sanity_plots import save_pretraining_sanity_plots


# ===================================================================== #
# 1. CONTINUATION CONFIGURATION
# ===================================================================== #

# Can be either:
#   * a direct .keras file
#   * a run directory
#   * a checkpoints directory
# Anchored to the project root (bootstrap) so the script works from any CWD.
MODEL_LOAD_PATH = os.path.join(
    _PROJECT_ROOT,
    "runs", "exp4_cleancorr_bfl_kl_cleanmse_corr_tail",
    "checkpoints", "best.keras",
)


# Freeze whole components before recompiling.
# Typical fine-tuning patterns:
#   * encoder frozen, train classifier only
#   * decoder frozen, adapt classifier + encoder
#   * classifier frozen, adapt generative pathway
FREEZE_COMPONENTS: Dict[str, bool] = {
    "encoder": False,
    "decoder": False,
    "classifier": False,
}


# Set values to None to inherit the value from the loaded run.
# Set a concrete value to override it for fine-tuning.
#
# IMPORTANT:
# Do not add architecture-baked keys here. The script will reject them.
FINETUNE_OVERRIDES: Dict[str, Any] = {
    # ---- Data / preprocessing ---- #
    "filepath_template": None,
    "filepath_suffixes": None,
    "num_samples_to_read_per_file": None,
    "offset": None,
    "sampling_rate_hz": None,
    "step_size": None,
    "train_ratio": None,
    "val_ratio": None,
    "test_ratio": None,
    "dtype": None,
    "memmap_dir": os.path.join(_PROJECT_ROOT, "memmaps_continue_train_freeze_encoder_decoder"),
    "stats_dir": None,
    "normalization_type": None,
    "global_mean_input": None,
    "global_std_input": None,
    "global_min_input": None,
    "global_max_input": None,
    "calculate_stats": None,
    # ---- Signal injection ---- #
    "inject_signals": True,
    "signal_injection_probability": 1.0,
    "snr_based_injection": True,
    "num_signals_to_inject_per_segment": {
        "train": 28000, # 14000
        "val": 5600, # 2800
        "test": 0,
    },
    "m_PBH_injection_list": [1e-8],
    "amplitude_spectrum_range": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5, 5.0, 5.5, 6.0
    ],
    "f0_gw": None,
    "Gamma_gw": None,
    "N_gw": None,
    "custom_noise_std": None,
    "reject_unrepresentable_injections": None,
    "no_overlap_injections": None,
    "no_overlap_margin_samples": None,
    "no_overlap_max_attempts": None,
    # Cavity response of the simulator (A5): "real_lorentzian" or
    # "complex_breit_wigner". None inherits the source run's setting.
    # CAUTION: changing this changes the injected signal family — only do it
    # deliberately (e.g. domain-adaptation fine-tuning) and rename the run.
    "response_mode": None,
    # ---- Clean injected-signal saving + metadata (A1-A4) ---- #
    "save_clean_signals": None,
    "save_metadata": None,
    # Must be True when rec_target_mode == "clean_signal" (datasets then
    # yield (x, y, clean_signal_window) 3-tuples).
    "include_clean_in_datasets": None,
    # Keep preprocessing artifacts isolated under the continued run dir.
    # When True (or inherited True), this takes precedence over "memmap_dir".
    "use_run_specific_memmap_dir": None,
    # ---- tf.data ---- #
    "tf_batch_size": 256,
    "tf_shuffle": None,
    "tf_repeat": None,
    # ---- Losses ---- #
    "focal_gamma": 4.0,
    "focal_alpha": 0.25,
    "focal_weight": 10.0,            # lambda_bfl
    "reconstruction_weight": 0.05,   # lambda_rec
    "kl_beta_start": 1e-5,
    "kl_beta_end": 1e-5,
    "kl_warmup_epochs": 0,
    # ---- Optional loss components (B); None inherits the source run ---- #
    "use_bfl": None,
    "use_kl": None,
    "use_rec": None,
    "use_tail_loss": None,           # B3: logsumexp tail loss on negative logits
    "lambda_tail": None,             # e.g. 0.05
    "tail_beta": None,               # e.g. 10.0
    "tail_margin": None,             # e.g. -2.0
    "rec_target_mode": None,         # B4: "raw_input" or "clean_signal"
    "use_corr_loss": None,           # B6: needs rec_target_mode="clean_signal"
    "use_iq_correlation_loss": None, # phase-invariant complex correlation for I/Q
    "lambda_corr": 0.5,             # e.g. 0.05
    "corr_eps": None,                # e.g. 1e-8
    # ---- Diagnostics ---- #
    "log_score_quantiles": None,     # C: per-epoch logit quantiles CSV
    "save_pretraining_sanity_plots": None,
    "pretraining_sanity_split": None,
    "pretraining_sanity_strict": None,
    "pretraining_sanity_max_candidates": None,
    # ---- Optimizer ---- #
    "learning_rate": 1e-4,
    "weight_decay": None,
    "adam_beta_1": None,
    "adam_beta_2": None,
    "adam_epsilon": None,
    "clipnorm": None,
    # ---- Training ---- #
    "epochs": 100,
    "early_stopping_patience": 20,
    "early_stopping_monitor": None,
    # ---- Event-detection callback ---- #
    "detection_every_epochs": None,
    "detection_target_fp_per_year": None,
    "detection_threshold_sweep_points": None,
    "detection_log_fit_tail_fraction": None,
    "detection_min_tail_points": None,
    # ---- I/O ---- #
    "model_name": None,
    "output_dir": os.path.join(_PROJECT_ROOT, "runs_continued", "WithEncoderUpdatedLoss", "10xSignals"),
    "checkpoint_subdir": None,
    "figures_subdir": None,
    "analysis_subdir": None,
    "random_seed": None,
}


IMMUTABLE_CONFIG_KEYS = {
    "window_size",
    "use_amps",
    "use_I_Q",
    "num_filters_per_layer",
    "kernel_sizes_per_layer",
    "strides_per_layer",
    "activation",
    "use_quadrature_frontend",
    "quadrature_output_mode",
    "encoder_activations",
    "decoder_activations",
    "classifier_activations",
    "latent_dim",
    "classifier_hidden_units",
    "classifier_dropout",
    "classifier_samples_z",
}


# ===================================================================== #
# 2. Helpers
# ===================================================================== #

def _resolve_model_checkpoint_path(model_path: str) -> str:
    """Resolve a file / run dir / checkpoints dir to a concrete .keras file."""
    candidate = os.path.abspath(model_path)
    if os.path.isfile(candidate):
        return candidate
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    checkpoint_candidates = [
        os.path.join(candidate, "best.keras"),
        os.path.join(candidate, "checkpoints", "best.keras"),
        os.path.join(candidate, "checkpoint", "best.keras"),
    ]
    for ckpt_path in checkpoint_candidates:
        if os.path.isfile(ckpt_path):
            return ckpt_path

    raise FileNotFoundError(
        f"Could not resolve a .keras checkpoint from '{model_path}'. "
        "Expected a checkpoint file or a directory containing best.keras."
    )


def _infer_run_dir_from_checkpoint(model_file_path: str) -> str:
    checkpoint_dir = os.path.dirname(model_file_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def _load_saved_run_config(run_dir: str) -> dict:
    """
    Load the saved config for the run we continue from.

    Regular runs store `config.json` in the run directory.
    Ablation runs may only have `ablation_config.json` one directory higher.
    """
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as handle:
            return json.load(handle)

    ablation_config_path = os.path.join(os.path.dirname(run_dir), "ablation_config.json")
    if os.path.isfile(ablation_config_path):
        with open(ablation_config_path, "r") as handle:
            return json.load(handle)

    raise FileNotFoundError(
        f"No config.json found in '{run_dir}' and no ablation_config.json "
        f"found in '{os.path.dirname(run_dir)}'."
    )


def _coerce_dtype(value):
    if value is None:
        return None
    if value is np.float32 or value is np.float64:
        return value
    if isinstance(value, str):
        return np.dtype(value).type
    return value


def _cfg_from_saved_dict(saved_cfg: dict) -> VAEConfig:
    """Materialize a VAEConfig from a saved config dictionary."""
    cfg = VAEConfig()
    valid_keys = set(cfg.__dict__.keys())
    for key, value in saved_cfg.items():
        if key not in valid_keys:
            continue
        if key == "dtype":
            value = _coerce_dtype(value)
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def _apply_overrides(cfg: VAEConfig, overrides: Dict[str, Any]) -> None:
    """Apply mutable continuation overrides and reject architecture changes."""
    for key in overrides:
        if key in IMMUTABLE_CONFIG_KEYS:
            raise ValueError(
                f"'{key}' is architecture-baked and may not appear in "
                "FINETUNE_OVERRIDES. Load a different base model instead."
            )

    valid_keys = set(cfg.__dict__.keys())
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in valid_keys:
            raise ValueError(f"Unknown continuation override key: {key}")
        if key == "dtype":
            value = _coerce_dtype(value)
        setattr(cfg, key, value)

    cfg.__post_init__()


def _anchor_relative_paths(cfg: VAEConfig) -> None:
    """
    Absolutize relative path fields against the project root.

    Saved configs from older runs store paths like './memmaps' or
    'GravNet/Data/...tiq' that only resolve when the CWD is the project
    root. Anchoring them here makes the continuation independent of the
    directory the script is started from (a wrong CWD used to surface as
    '0 injected signals / mean 0 / std 0').
    """
    for key in ("filepath_template", "memmap_dir", "stats_dir", "output_dir"):
        value = getattr(cfg, key, None)
        if isinstance(value, str) and value and not os.path.isabs(value):
            setattr(cfg, key, os.path.normpath(os.path.join(_PROJECT_ROOT, value)))


def _default_continued_model_name(saved_cfg: dict) -> str:
    base_name = saved_cfg.get("model_name", "vae_classifier")
    return f"{base_name}_continued"


def _build_optimizer(cfg: VAEConfig):
    """Create AdamW with the fine-tuning hyperparameters."""
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1,
            beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.clipnorm,
        )
    except AttributeError:
        from tensorflow.keras.optimizers.experimental import AdamW
        return AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1,
            beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.clipnorm,
        )


def _plot_history_curve(history, keys, ylabel, save_path, logy=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in keys:
        if k in history.history:
            ax.plot(history.history[k], label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _apply_component_freezing(model, freeze_components: Dict[str, bool]) -> None:
    """Freeze or unfreeze whole submodels before compile()."""
    valid_keys = {"encoder", "decoder", "classifier"}
    unknown = set(freeze_components.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown FREEZE_COMPONENTS keys: {sorted(unknown)}. "
            f"Valid keys are {sorted(valid_keys)}."
        )

    model.encoder.trainable = not bool(freeze_components.get("encoder", False))
    model.decoder.trainable = not bool(freeze_components.get("decoder", False))
    model.classifier.trainable = not bool(freeze_components.get("classifier", False))


def _count_params(weights) -> int:
    return int(sum(np.prod(w.shape) for w in weights))


# ===================================================================== #
# 3. Main
# ===================================================================== #

if __name__ == "__main__":
    resolved_ckpt_path = _resolve_model_checkpoint_path(MODEL_LOAD_PATH)
    source_run_dir = _infer_run_dir_from_checkpoint(resolved_ckpt_path)
    saved_cfg_dict = _load_saved_run_config(source_run_dir)
    cfg = _cfg_from_saved_dict(saved_cfg_dict)

    if FINETUNE_OVERRIDES.get("model_name") is None:
        FINETUNE_OVERRIDES["model_name"] = _default_continued_model_name(saved_cfg_dict)
    _apply_overrides(cfg, FINETUNE_OVERRIDES)
    _anchor_relative_paths(cfg)

    run_dir = os.path.join(cfg.output_dir, cfg.model_name)
    ckpt_dir = os.path.join(run_dir, cfg.checkpoint_subdir)
    fig_dir = os.path.join(run_dir, cfg.figures_subdir)
    # Keep preprocessing artifacts isolated per continued run (same policy as
    # train.py); takes precedence over the FINETUNE_OVERRIDES["memmap_dir"].
    if cfg.use_run_specific_memmap_dir:
        cfg.memmap_dir = os.path.join(run_dir, "memmaps")
        if cfg.stats_dir is not None:
            cfg.stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        snap = {k: (v if not hasattr(v, "__name__") else v.__name__)
                for k, v in cfg.__dict__.items()}
        snap["continued_from_checkpoint"] = resolved_ckpt_path
        snap["continued_from_run_dir"] = source_run_dir
        snap["immutable_config_keys"] = sorted(IMMUTABLE_CONFIG_KEYS)
        snap["freeze_components"] = FREEZE_COMPONENTS
        json.dump(snap, fh, indent=2, default=str)

    print("--- Continue / fine-tune VAE model ---")
    print(f"    Source checkpoint: {resolved_ckpt_path}")
    print(f"    Source run dir:    {source_run_dir}")
    print(f"    Output run dir:    {run_dir}")
    print(f"    Data mode:         {'I/Q' if cfg.use_I_Q else 'Amplitude'}")
    print(
        "    Classifier path:  "
        + ("shared sampled z" if cfg.classifier_samples_z else "deterministic [z_mean, z_log_var]")
    )

    # The preprocessing helper expects physical masses in kg.
    pbh_masses_kg = [m * cfg.M_solar for m in cfg.m_PBH_injection_list]

    print("\n--- Preprocessing ---")
    train_ds, val_ds, test_ds, preprocessing_info = pre_processing_with_memmap(
        filepath_suffixes=cfg.filepath_suffixes,
        filepath_template=cfg.filepath_template,
        num_samples_to_read_per_file=cfg.num_samples_to_read_per_file,
        offset=cfg.offset,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        dtype=cfg.dtype,
        normalization_type=cfg.normalization_type,
        global_min_input=cfg.global_min_input,
        global_max_input=cfg.global_max_input,
        global_mean_input=cfg.global_mean_input,
        global_std_input=cfg.global_std_input,
        calculate_stats=cfg.calculate_stats,
        use_amps=cfg.use_amps,
        use_I_Q=cfg.use_I_Q,
        inject_signals=cfg.inject_signals,
        signal_injection_probability=cfg.signal_injection_probability,
        m_PBH_injection_list=pbh_masses_kg,
        amplitude_spectrum_range=cfg.amplitude_spectrum_range,
        num_signals_to_inject_per_segment=cfg.num_signals_to_inject_per_segment,
        snr_based_injection=cfg.snr_based_injection,
        custom_noise_std=cfg.custom_noise_std,
        f0_gw=cfg.f0_gw,
        Gamma_gw=cfg.Gamma_gw,
        N_gw=cfg.N_gw,
        M_solar=cfg.M_solar,
        response_mode=cfg.response_mode,
        save_clean_signals=cfg.save_clean_signals,
        save_metadata=cfg.save_metadata,
        include_clean_in_datasets=cfg.include_clean_in_datasets,
        memmap_dir=cfg.memmap_dir,
        stats_dir=cfg.stats_dir,
        return_tf_datasets=True,
        tf_batch_size=cfg.tf_batch_size,
        tf_shuffle=cfg.tf_shuffle,
        tf_repeat=cfg.tf_repeat,
        random_seed=cfg.random_seed,
        reject_unrepresentable_injections=cfg.reject_unrepresentable_injections,
        no_overlap_injections=cfg.no_overlap_injections,
        no_overlap_margin_samples=cfg.no_overlap_margin_samples,
        no_overlap_max_attempts=cfg.no_overlap_max_attempts,
        return_info=True,
    )

    if cfg.save_pretraining_sanity_plots and cfg.inject_signals:
        print("\n--- Saving pre-training data sanity plots ---")
        save_pretraining_sanity_plots(
            preprocessing_info,
            output_dir=os.path.join(fig_dir, "pretraining_checks"),
            split=cfg.pretraining_sanity_split,
            strict=cfg.pretraining_sanity_strict,
            max_candidates=cfg.pretraining_sanity_max_candidates,
        )
    elif cfg.save_pretraining_sanity_plots:
        print(
            "\n--- Skipping pre-training signal plots: inject_signals=False, "
            "so no noise+signal window exists. ---"
        )

    print("\n--- Loading existing model ---")
    model = tf.keras.models.load_model(
        resolved_ckpt_path,
        custom_objects={
            "Sampling": Sampling,
            "VAEClassifier": VAEClassifier,
            "VAEClassifierAblation": VAEClassifierAblation,
            "QuadratureConv1D": QuadratureConv1D,
        },
        compile=False,
    )
    print("Model loaded successfully.")

    # Reapply fine-tuning scalar hyperparameters to the loaded model.
    model.focal_gamma = float(cfg.focal_gamma)
    model.focal_alpha = float(cfg.focal_alpha)
    model.focal_weight = float(cfg.focal_weight)
    model.reconstruction_weight = float(cfg.reconstruction_weight)
    model._init_kl_beta = float(cfg.kl_beta_start)
    model.set_beta(cfg.kl_beta_start)

    # Reapply the optional loss components (B). Old checkpoints that predate
    # these knobs deserialize with the legacy defaults, so setting them here
    # makes the continuation config the single source of truth.
    model.use_bfl = bool(cfg.use_bfl)
    model.use_kl = bool(cfg.use_kl)
    model.use_rec = bool(cfg.use_rec)
    model.use_tail_loss = bool(cfg.use_tail_loss)
    model.lambda_tail = float(cfg.lambda_tail)
    model.tail_beta = float(cfg.tail_beta)
    model.tail_margin = float(cfg.tail_margin)
    model.rec_target_mode = str(cfg.rec_target_mode)
    model.use_corr_loss = bool(cfg.use_corr_loss)
    model.use_iq_correlation_loss = bool(cfg.use_iq_correlation_loss)
    model.lambda_corr = float(cfg.lambda_corr)
    model.corr_eps = float(cfg.corr_eps)
    if cfg.rec_target_mode == "clean_signal" and not cfg.include_clean_in_datasets:
        raise ValueError(
            "rec_target_mode='clean_signal' requires include_clean_in_datasets=True "
            "so the continued training batches are (x, y, clean_signal_window)."
        )
    if cfg.use_iq_correlation_loss:
        if not cfg.use_I_Q or cfg.use_amps:
            raise ValueError(
                "use_iq_correlation_loss=True requires an I/Q checkpoint and "
                "use_I_Q=True, use_amps=False."
            )
        if not cfg.use_corr_loss or cfg.rec_target_mode != "clean_signal":
            raise ValueError(
                "use_iq_correlation_loss=True requires use_corr_loss=True and "
                "rec_target_mode='clean_signal'."
            )

    _apply_component_freezing(model, FREEZE_COMPONENTS)

    optimizer = _build_optimizer(cfg)
    model.compile(optimizer=optimizer)

    # Dry forward pass so summary() is available and shape issues show up early.
    for batch in train_ds.take(1):
        _ = model(batch[0], training=False)
    model.summary(expand_nested=True)
    trainable_params = _count_params(model.trainable_weights)
    frozen_params = _count_params(model.non_trainable_weights)
    print(
        "    Freeze state:      "
        f"encoder={FREEZE_COMPONENTS['encoder']} | "
        f"decoder={FREEZE_COMPONENTS['decoder']} | "
        f"classifier={FREEZE_COMPONENTS['classifier']}"
    )
    print(f"    Trainable params:  {trainable_params}")
    print(f"    Frozen params:     {frozen_params}")

    callbacks = [
        BetaAnnealing(
            model=model,
            start=cfg.kl_beta_start,
            end=cfg.kl_beta_end,
            warmup_epochs=cfg.kl_warmup_epochs,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best.keras"),
            monitor=cfg.early_stopping_monitor,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.early_stopping_monitor,
            patience=cfg.early_stopping_patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if cfg.log_score_quantiles:
        callbacks.append(
            ScoreQuantileLogger(model=model, val_ds=val_ds, output_dir=run_dir)
        )

    if cfg.detection_every_epochs > 0:
        callbacks.append(
            EventDetectionCallback(
                model=model,
                val_ds=val_ds,
                window_size=cfg.window_size,
                step_size=cfg.step_size,
                fs=cfg.sampling_rate_hz,
                every_epochs=cfg.detection_every_epochs,
                target_fp_per_year=cfg.detection_target_fp_per_year,
                sweep_points=cfg.detection_threshold_sweep_points,
                log_fit_tail_fraction=cfg.detection_log_fit_tail_fraction,
                min_tail_points=cfg.detection_min_tail_points,
            )
        )

    print("\n--- Continuing training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n--- Plotting training history ---")
    _plot_history_curve(
        history,
        ["loss", "val_loss"],
        "total loss",
        os.path.join(fig_dir, "history_total_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["focal_loss", "val_focal_loss"],
        "focal loss",
        os.path.join(fig_dir, "history_focal_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["recon_loss", "val_recon_loss"],
        "reconstruction MSE",
        os.path.join(fig_dir, "history_recon_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["kl_loss", "val_kl_loss"],
        "KL divergence",
        os.path.join(fig_dir, "history_kl_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["tail_loss", "val_tail_loss"],
        "logsumexp tail loss",
        os.path.join(fig_dir, "history_tail_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["corr_loss", "val_corr_loss"],
        "correlation loss",
        os.path.join(fig_dir, "history_corr_loss.png"),
        logy=True,
    )
    _plot_history_curve(
        history,
        ["auc", "val_auc"],
        "AUC",
        os.path.join(fig_dir, "history_auc.png"),
        logy=False,
    )
    _plot_history_curve(
        history,
        ["neg_logit_q99", "neg_logit_q999", "neg_logit_max"],
        "negative logit tail",
        os.path.join(fig_dir, "history_neg_logit_tail.png"),
        logy=False,
    )
    _plot_history_curve(
        history,
        ["pos_logit_q10"],
        "pos_logit_q10",
        os.path.join(fig_dir, "history_pos_logit_q10.png"),
        logy=False,
    )
    if "det_event_recall" in history.history:
        _plot_history_curve(
            history,
            ["det_event_recall"],
            "event recall @ target FP/year",
            os.path.join(fig_dir, "history_det_event_recall.png"),
            logy=False,
        )
    if "det_threshold" in history.history:
        _plot_history_curve(
            history,
            ["det_threshold"],
            "operating threshold",
            os.path.join(fig_dir, "history_det_threshold.png"),
            logy=False,
        )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        json.dump(history_dict, fh, indent=2)
    pd.DataFrame(history_dict).to_csv(
        os.path.join(run_dir, "history.csv"),
        index_label="epoch",
    )

    print(f"\n--- Continued training complete. Artefacts in {run_dir} ---")
