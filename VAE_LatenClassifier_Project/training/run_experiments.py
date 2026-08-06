"""
Experiment matrix for the GravNet pipeline update (G).

Experiments (VAE-classifier):
  1. baseline    : BFL + KL + raw_input MSE                      (current behavior)
  2. tail        : BFL + KL + raw_input MSE + logsumexp tail
  3. clean_rec   : BFL + KL + clean_signal MSE + logsumexp tail
  4. clean_corr  : BFL + KL + clean_signal MSE + clean-signal correlation
                   loss + logsumexp tail
  (5. standalone clean-signal autoencoder -> train_clean_signal_autoencoder.py)
  (6. post-VAE verification               -> evaluate_post_vae_signal_manifold.py)

Every run gets its own output directory  ./runs/<experiment_name>  with its
config.json, checkpoints, history, figures and score_quantiles.csv, so results
from different configs (and response modes) cannot be mixed accidentally.

Usage:
    python run_experiments.py baseline
    python run_experiments.py tail
    python run_experiments.py clean_rec
    python run_experiments.py clean_corr
    python run_experiments.py all
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


import dataclasses
import json
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAEConfig, build_vae_classifier
from vae.callbacks import BetaAnnealing, EventDetectionCallback, ScoreQuantileLogger
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap
from data_pre_processing.sanity_plots import save_pretraining_sanity_plots

# Base configuration: reuse the single source of truth in train.py.
from train import cfg as BASE_CFG


# ===================================================================== #
# Experiment definitions (B / G)
# ===================================================================== #

EXPERIMENTS = {
    # G1 — current behavior
    "baseline": dict(
        model_name="exp1_baseline_bfl_kl_rawmse",
        use_tail_loss=False,
        rec_target_mode="raw_input",
        use_corr_loss=False,
        include_clean_in_datasets=False,
    ),
    # G2 — + logsumexp tail loss
    "tail": dict(
        model_name="exp2_tail_bfl_kl_rawmse_tail",
        use_tail_loss=True,
        lambda_tail=0.05,
        tail_beta=10.0,
        tail_margin=-2.0,
        rec_target_mode="raw_input",
        use_corr_loss=False,
        include_clean_in_datasets=False,
    ),
    # G3 — clean-signal reconstruction target + tail
    "clean_rec": dict(
        model_name="exp3_cleanrec_bfl_kl_cleanmse_tail",
        use_tail_loss=True,
        lambda_tail=0.05,
        tail_beta=10.0,
        tail_margin=-2.0,
        rec_target_mode="clean_signal",
        use_corr_loss=False,
        include_clean_in_datasets=True,
        save_clean_signals=True,
    ),
    # G4 — clean reconstruction + shape-correlation loss + tail
    "clean_corr": dict(
        model_name="exp4_cleancorr_bfl_kl_cleanmse_corr_tail",
        use_tail_loss=True,
        lambda_tail=0.05,
        tail_beta=10.0,
        tail_margin=-2.0,
        rec_target_mode="clean_signal",
        use_corr_loss=True,
        lambda_corr=0.05,    # small: correlation fixes shape, not amplitude
        corr_eps=1e-8,
        include_clean_in_datasets=True,
        save_clean_signals=True,
    ),
}


def build_experiment_config(name: str) -> VAEConfig:
    overrides = EXPERIMENTS[name]
    cfg = dataclasses.replace(BASE_CFG, **overrides)
    # Tag output dirs with the response mode so different simulator settings
    # can never be mixed accidentally (A5).
    if cfg.response_mode != "real_lorentzian":
        cfg = dataclasses.replace(
            cfg, model_name=f"{cfg.model_name}__{cfg.response_mode}"
        )
    return cfg


# ===================================================================== #
# Training driver (mirrors train.py, plus clean signals + quantile logging)
# ===================================================================== #

def run_training(cfg: VAEConfig):
    pbh_masses_kg = [m * cfg.M_solar for m in cfg.m_PBH_injection_list]

    run_dir = os.path.join(cfg.output_dir, cfg.model_name)
    ckpt_dir = os.path.join(run_dir, cfg.checkpoint_subdir)
    fig_dir = os.path.join(run_dir, cfg.figures_subdir)
    if cfg.use_run_specific_memmap_dir:
        cfg.memmap_dir = os.path.join(run_dir, "memmaps")
        if cfg.stats_dir is not None:
            cfg.stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({k: (v if not hasattr(v, "__name__") else v.__name__)
                   for k, v in cfg.__dict__.items()}, fh, indent=2, default=str)

    print(f"--- Preprocessing for '{cfg.model_name}' "
          f"(response_mode={cfg.response_mode}, "
          f"rec_target={cfg.rec_target_mode}) ---")
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
        f0_gw=cfg.f0_gw, Gamma_gw=cfg.Gamma_gw, N_gw=cfg.N_gw, M_solar=cfg.M_solar,
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

    model = build_vae_classifier(cfg)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1,
            beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.clipnorm,
        )
    except AttributeError:
        from tensorflow.keras.optimizers.experimental import AdamW
        optimizer = AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1,
            beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.clipnorm,
        )

    model.compile(optimizer=optimizer)
    for batch in train_ds.take(1):
        _ = model(batch[0], training=False)
    model.summary(expand_nested=True)

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
    if cfg.log_score_quantiles:
        callbacks.append(
            ScoreQuantileLogger(model=model, val_ds=val_ds, output_dir=run_dir)
        )

    print(f"\n--- Training '{cfg.model_name}' ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    def _plot_history_curve(keys, ylabel, fname, logy=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        any_key = False
        for k in keys:
            if k in history.history:
                ax.plot(history.history[k], label=k)
                any_key = True
        if not any_key:
            plt.close(fig)
            return
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fname), dpi=200)
        plt.close(fig)

    _plot_history_curve(["loss", "val_loss"], "total loss", "history_total_loss.png", logy=True)
    _plot_history_curve(["focal_loss", "val_focal_loss"], "focal loss", "history_focal_loss.png", logy=True)
    _plot_history_curve(["recon_loss", "val_recon_loss"], "reconstruction MSE", "history_recon_loss.png", logy=True)
    _plot_history_curve(["kl_loss", "val_kl_loss"], "KL divergence", "history_kl_loss.png", logy=True)
    _plot_history_curve(["tail_loss", "val_tail_loss"], "logsumexp tail loss", "history_tail_loss.png", logy=True)
    _plot_history_curve(["corr_loss", "val_corr_loss"], "correlation loss", "history_corr_loss.png", logy=True)
    _plot_history_curve(["auc", "val_auc"], "AUC", "history_auc.png")
    _plot_history_curve(["neg_logit_q99", "neg_logit_q999", "neg_logit_max"],
                        "negative logit tail", "history_neg_logit_tail.png")
    _plot_history_curve(["pos_logit_q10"], "pos_logit_q10", "history_pos_logit_q10.png")
    if "det_event_recall" in history.history:
        _plot_history_curve(["det_event_recall"], "event recall @ target FP/year",
                            "history_det_event_recall.png")
    if "det_threshold" in history.history:
        _plot_history_curve(["det_threshold"], "operating threshold", "history_det_threshold.png")

    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  fh, indent=2)

    print(f"\n--- '{cfg.model_name}' complete. Artefacts in {run_dir} ---")
    return run_dir


# ===================================================================== #
# Entry point
# ===================================================================== #

if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    names = list(EXPERIMENTS.keys()) if requested == "all" else [requested]
    for exp_name in names:
        if exp_name not in EXPERIMENTS:
            raise SystemExit(
                f"Unknown experiment '{exp_name}'. "
                f"Choose from {list(EXPERIMENTS.keys()) + ['all']}."
            )
    for exp_name in names:
        exp_cfg = build_experiment_config(exp_name)
        run_training(exp_cfg)
