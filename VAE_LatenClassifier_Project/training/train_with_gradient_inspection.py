"""
Single-mode training run with the gradient inspector enabled.

Use this for a deep dive on one ablation mode. It mirrors the production
train.py from TrainedModels/.../Model_1/ but:

  * builds a VAEClassifierAblation with a configurable classifier_input_mode
  * attaches a GradientInspector that logs per-layer norms + per-loss
    gradient decomposition + pairwise cosines on the encoder

Pick the mode at the top of the script (or via the MODE environment variable).

Outputs land in:
    runs/<model_name>_<mode>/
        checkpoints/best.keras
        figures/                    (loss / AUC / detection plots)
        grad_logs/
            gradients.csv           one row per logged step
            gradients_summary.json  last logged row
            tb/                     TensorBoard event files
            encoder_loss_share.png  share of encoder grad per loss
            encoder_cosines.png     cos(g_focal, g_recon) etc on encoder
            combined_layer_norms.png
            per_loss_layer_norms.png

Cost
----
Eager mode + 3 extra encoder backward passes every `decompose_every_n_steps`
roughly 1.5x slower than train.py. Set decompose_every_n_steps to a larger
number to reduce overhead. Set it to 0 to skip decomposition entirely (you'd
still get cheap per-layer combined norms).
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
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import (
    VAEConfig,
    BetaAnnealing,
    EventDetectionCallback,
    ScoreQuantileLogger,
    build_vae_classifier_ablation,
    GradientInspector,
    plot_gradient_log,
)
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap
from data_pre_processing.sanity_plots import save_pretraining_sanity_plots


# ===================================================================== #
# 1. Configuration
# ===================================================================== #

# Pick the ablation mode here. Options:
#   "concat_mean_logvar"        — current production behaviour (deterministic)
#   "z_mean"                    — classifier sees z_mean only
#   "shared_sample"             — classifier sees the decoder's sampled z
#   "independent_sample"        — classifier sees μ + σ * ε_indep   (HYPOTHESIS)
#   "shared_sample_detached"    — same z but no encoder-grad path through cls
MODE = os.environ.get("MODE", "independent_sample")

# Inspector cadence. The decomposition pass is the expensive one, so keep it
# rare for long runs and dense for short diagnostic runs.
DECOMPOSE_EVERY_N_STEPS = int(os.environ.get("DECOMPOSE_EVERY", "50"))
LAYER_NORMS_EVERY_N_STEPS = int(os.environ.get("LAYER_NORMS_EVERY", "1"))
WRITE_TENSORBOARD = os.environ.get("WRITE_TB", "1") == "1"
FIT_VERBOSE = int(os.environ.get("FIT_VERBOSE", "1"))

# How long to train. Match your usual schedule, or shrink for diagnostics.
EPOCHS = int(os.environ.get("EPOCHS", "100"))


cfg = VAEConfig(
    # ---- Data ----
    # Paths anchored to _PROJECT_ROOT (from the bootstrap) so the script
    # works from any working directory.
    filepath_suffixes=["19.22.48.163", "19.22.56.276"],
    filepath_template=os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
    memmap_dir=os.path.join(_PROJECT_ROOT, "memmaps"),
    num_samples_to_read_per_file=112000000,
    sampling_rate_hz=14e6,
    window_size=1024,
    step_size=1024 // 10,
    train_ratio=0.84, val_ratio=0.15, test_ratio=0.01,
    use_amps=True, use_I_Q=False,
    normalization_type="zscore",

    # ---- Signal injection ----
    inject_signals=True,
    snr_based_injection=True,
    num_signals_to_inject_per_segment={"train": 14000, "val": 2800, "test": 0},
    m_PBH_injection_list=[1e-8],
    amplitude_spectrum_range=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                              5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    f0_gw=5.0e9, Gamma_gw=100e3, N_gw=32768,
    reject_unrepresentable_injections=True,
    # Cavity response (A5): "real_lorentzian" (legacy) or "complex_breit_wigner"
    response_mode="real_lorentzian",
    # Clean injected-signal saving + metadata (A1-A4)
    save_clean_signals=True,
    save_metadata=True,
    include_clean_in_datasets=False,  # True -> (x, y, clean) batches; required
                                      # for rec_target_mode="clean_signal"
    use_run_specific_memmap_dir=True,

    # ---- Architecture ----
    num_filters_per_layer=[16, 32, 64, 96],
    kernel_sizes_per_layer=[16, 7, 7, 7],
    strides_per_layer=[2, 2, 4, 4],
    activation="silu",
    latent_dim=16,
    classifier_hidden_units=[64, 32],
    classifier_dropout=0.1,

    # Convenience boolean: when MODE='concat_mean_logvar' this knob is
    # redundant. Only matters if you build the *base* VAEClassifier instead
    # of VAEClassifierAblation (see train.py).
    classifier_samples_z=False,

    # ---- Losses ----
    focal_gamma=2.0,
    focal_alpha=0.25,
    focal_weight=1.0,             # lambda_bfl
    reconstruction_weight=0.05,   # lambda_rec
    kl_beta_start=1e-4,
    kl_beta_end=1e-2,
    kl_warmup_epochs=5,
    # ---- Optional loss components (B) ---- #
    use_bfl=True,
    use_kl=True,
    use_rec=True,
    use_tail_loss=False,          # B3: logsumexp tail loss on negative logits
    lambda_tail=0.05,
    tail_beta=10.0,
    tail_margin=-2.0,
    rec_target_mode="raw_input",  # B4: "raw_input" (legacy) or "clean_signal"
    use_corr_loss=False,          # B6: needs rec_target_mode="clean_signal"
    use_iq_correlation_loss=False,  # phase-invariant complex correlation for I/Q
    lambda_corr=0.05,
    corr_eps=1e-8,
    log_score_quantiles=True,     # C: per-epoch logit quantiles CSV
    save_pretraining_sanity_plots=True,

    # ---- AdamW ----
    learning_rate=5e-4,
    weight_decay=1e-4,
    clipnorm=1.0,

    # ---- Training ----
    epochs=EPOCHS,
    early_stopping_patience=20,
    tf_batch_size=1024,

    # ---- Detection callback ----
    detection_every_epochs=1,
    detection_target_fp_per_year=1.0,

    # ---- I/O ----
    model_name="vae_cls_inspect",
    output_dir=os.path.join(_PROJECT_ROOT, "runs"),
)


# ===================================================================== #
# Main
# ===================================================================== #

if __name__ == "__main__":
    M_SOLAR = cfg.M_solar
    pbh_masses_kg = [m * M_SOLAR for m in cfg.m_PBH_injection_list]

    run_dir = os.path.join(cfg.output_dir, f"{cfg.model_name}_{MODE}")
    ckpt_dir = os.path.join(run_dir, cfg.checkpoint_subdir)
    fig_dir = os.path.join(run_dir, cfg.figures_subdir)
    grad_dir = os.path.join(run_dir, "grad_logs")
    # Keep preprocessing artifacts isolated per run (same policy as train.py).
    if cfg.use_run_specific_memmap_dir:
        cfg.memmap_dir = os.path.join(run_dir, "memmaps")
        if cfg.stats_dir is not None:
            cfg.stats_dir = os.path.join(run_dir, "stats")
    for d in (ckpt_dir, fig_dir, grad_dir):
        os.makedirs(d, exist_ok=True)

    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)

    # Persist config + chosen mode for reproducibility.
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        snap = {k: (v if not hasattr(v, "__name__") else v.__name__)
                for k, v in cfg.__dict__.items()}
        snap["MODE"] = MODE
        snap["DECOMPOSE_EVERY_N_STEPS"] = DECOMPOSE_EVERY_N_STEPS
        snap["LAYER_NORMS_EVERY_N_STEPS"] = LAYER_NORMS_EVERY_N_STEPS
        json.dump(snap, fh, indent=2, default=str)

    # ----- Data ---------------------------------------------------- #
    print("--- Preprocessing ---")
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

    # ----- Model + optimizer --------------------------------------- #
    print(f"--- Building model with classifier_input_mode='{MODE}' ---")
    model = build_vae_classifier_ablation(cfg, mode=MODE)

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon, clipnorm=cfg.clipnorm,
        )
    except AttributeError:
        from tensorflow.keras.optimizers.experimental import AdamW
        optimizer = AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon, clipnorm=cfg.clipnorm,
        )
    # IMPORTANT: run_eagerly=True is REQUIRED by GradientInspector. The
    # inspector calls `.numpy()` on gradient tensors inside train_step and
    # uses Python control flow between batches; both need eager execution.
    # Cost is roughly 1.3-2x slower per step on this model size — acceptable
    # for diagnostic runs. For production runs, use the regular train.py
    # without the inspector and you'll get tf.function-traced training back.
    model.compile(optimizer=optimizer, run_eagerly=True)

    # Dry forward pass so summary() works.
    for batch in train_ds.take(1):
        _ = model(batch[0], training=False)
    model.summary(expand_nested=True)

    # ----- Callbacks ----------------------------------------------- #
    beta_cb = BetaAnnealing(
        model=model,
        start=cfg.kl_beta_start, end=cfg.kl_beta_end,
        warmup_epochs=cfg.kl_warmup_epochs,
    )
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "best.keras"),
        monitor=cfg.early_stopping_monitor,
        save_best_only=True, save_weights_only=False,
        mode="min", verbose=1,
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.early_stopping_monitor,
        patience=cfg.early_stopping_patience,
        mode="min", restore_best_weights=True, verbose=1,
    )
    inspector = GradientInspector(
        out_dir=grad_dir,
        decompose_every_n_steps=DECOMPOSE_EVERY_N_STEPS,
        layer_norms_every_n_steps=LAYER_NORMS_EVERY_N_STEPS,
        write_tensorboard=WRITE_TENSORBOARD,
        verbose=1,
    )

    callbacks = [beta_cb, ckpt_cb, early_cb]
    if cfg.log_score_quantiles:
        callbacks.append(
            ScoreQuantileLogger(model=model, val_ds=val_ds, output_dir=run_dir)
        )
    if WRITE_TENSORBOARD:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(grad_dir, "tb"),
                histogram_freq=0,
                write_graph=False,
                update_freq="epoch",
                profile_batch=0,
            )
        )
    callbacks.append(inspector)
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

    # ----- Fit ----------------------------------------------------- #
    print("\n--- Starting training (run_eagerly=True forced by inspector) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=FIT_VERBOSE,
    )

    # ----- Plots --------------------------------------------------- #
    def _plot_history_curve(keys, ylabel, fname, logy=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        for k in keys:
            if k in history.history:
                ax.plot(history.history[k], label=k)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fname), dpi=200)
        plt.close(fig)

    _plot_history_curve(["loss", "val_loss"], "total loss",
                        "history_total_loss.png", logy=True)
    _plot_history_curve(["focal_loss", "val_focal_loss"], "focal loss",
                        "history_focal_loss.png", logy=True)
    _plot_history_curve(["recon_loss", "val_recon_loss"], "recon MSE",
                        "history_recon_loss.png", logy=True)
    _plot_history_curve(["kl_loss", "val_kl_loss"], "KL divergence",
                        "history_kl_loss.png", logy=True)
    _plot_history_curve(["tail_loss", "val_tail_loss"], "logsumexp tail loss",
                        "history_tail_loss.png", logy=True)
    _plot_history_curve(["corr_loss", "val_corr_loss"], "correlation loss",
                        "history_corr_loss.png", logy=True)
    _plot_history_curve(["auc", "val_auc"], "AUC", "history_auc.png")
    _plot_history_curve(["neg_logit_q99", "neg_logit_q999", "neg_logit_max"],
                        "negative logit tail", "history_neg_logit_tail.png")
    _plot_history_curve(["pos_logit_q10"], "pos_logit_q10",
                        "history_pos_logit_q10.png")

    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        history_dict = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        json.dump(history_dict, fh, indent=2)
    pd.DataFrame(history_dict).to_csv(
        os.path.join(run_dir, "history.csv"),
        index_label="epoch",
    )

    # ----- Render gradient-inspector plots ------------------------- #
    grad_csv = os.path.join(grad_dir, "gradients.csv")
    if os.path.exists(grad_csv):
        print("--- Plotting gradient inspector outputs ---")
        plot_gradient_log(grad_csv, grad_dir)

    print(f"\n--- Done. Artefacts in {run_dir} ---")
