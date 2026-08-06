"""
Single entry-point training script for the simplified VAE-classifier.

Usage
-----
    python train.py

All hyperparameters live in `vae/config.py::VAEConfig`. Override any of
them at the top of this file — every knob from AdamW's weight decay to
the exact number of filters in each CNN layer is exposed.

What this script does
---------------------
1. Builds a VAEConfig and fills the signal injection lists in physical units.
2. Calls the existing preprocessing pipeline to get (train, val, test)
   tf.data.Datasets that yield (x, y) pairs.
3. Builds the VAEClassifier and compiles it with AdamW.
4. Trains with three callbacks: beta annealing, checkpointing + early
   stopping, and the event-detection callback that reports physical
   numbers (threshold for 1 FP/year, event recall at that threshold).
5. Plots training curves and saves them alongside the checkpoint.
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
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAEConfig, build_vae_classifier
from vae.callbacks import BetaAnnealing, EventDetectionCallback, ScoreQuantileLogger

# Import the existing preprocessing pipeline unchanged.
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap
from data_pre_processing.sanity_plots import save_pretraining_sanity_plots


# ===================================================================== #
# 1. Build the configuration
# ===================================================================== #

cfg = VAEConfig(
    # ---- Data ----
    # All data/output paths are anchored to _PROJECT_ROOT (set by the path
    # bootstrap above) so the script works from ANY working directory.
    # A bare relative template silently loads nothing when the script is
    # started from scripts/training/.
    filepath_template=os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
    filepath_suffixes=["19.22.48.163", "19.22.56.276", "19.23.04.395", "19.23.12.552"], # , "19.23.04.395", "19.23.12.552"
    num_samples_to_read_per_file=112000000,
    offset=0,
    sampling_rate_hz=14e6,
    window_size=4096,
    step_size=4096 // 10,
    train_ratio=0.84,
    val_ratio=0.15,
    test_ratio=0.01,
    dtype=np.float32,
    memmap_dir=os.path.join(_PROJECT_ROOT, "memmaps"),
    stats_dir=None,
    use_amps=False,
    use_I_Q=True,
    normalization_type="zscore",
    global_mean_input=None,
    global_std_input=None,
    global_min_input=None,
    global_max_input=None,
    calculate_stats=True,

    # ---- Signal injection ---- #
    inject_signals=True,
    signal_injection_probability=1.0,
    snr_based_injection=True,
    num_signals_to_inject_per_segment={
        "train": 7000, # 14000
        "val": 1400, # 2800
        "test": 0,
    },  # roughly 5x more train than val injections
    m_PBH_injection_list=[1e-12],  # solar masses; converted to kg below
    amplitude_spectrum_range=[
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5
    ],  # target SNR
    f0_gw=5.0e9,
    Gamma_gw=100e3,
    N_gw=32768,
    custom_noise_std=None,
    # True: stop if an injection changes zero stored input samples.
    # False: continue, warn, and record the invalid injection in metadata.
    reject_unrepresentable_injections=True,
    # Cavity response (A5): "real_lorentzian" (legacy) or "complex_breit_wigner"
    response_mode="real_lorentzian",
    # Clean injected-signal saving + metadata (A1-A4)
    save_clean_signals=True,
    save_metadata=True,
    include_clean_in_datasets=True,  # True -> datasets yield (x, y, clean)

    # ---- tf.data ---- #
    tf_batch_size=1024,
    tf_shuffle=True,
    tf_repeat=False,

    # ---- Architecture (explicit, per-layer) ---- #
    num_filters_per_layer=[16, 32, 64, 96],
    kernel_sizes_per_layer=[16, 7, 7, 7],
    strides_per_layer=[2, 2, 4, 4],
    activation="silu",
    # Optional per-layer overrides:
    # activation=["silu", "linear", "silu", "silu"],  # encoder stages; decoder mirrors in reverse
    # decoder_activations=["silu", "linear", "silu"], # explicit decoder stages; len n or n-1
    use_quadrature_frontend=False,
    quadrature_output_mode="real_imag",  # "magnitude" or "real_imag"; requires use_I_Q=True
    #classifier_activations=["linear", "silu", "linear"],
    encoder_activations=None,
    decoder_activations=None,
    classifier_activations=None,
    latent_dim=32,
    classifier_hidden_units=[64, 32],
    classifier_dropout=0.1,
    # Classifier path:
    #   False -> deterministic classifier on concat([z_mean, z_log_var])
    #   True  -> classifier and decoder both read the same sampled z
    classifier_samples_z=True,

    # ---- Losses ---- #
    focal_gamma=4.0,
    focal_alpha=0.25,
    focal_weight=10.0,            # lambda_bfl
    reconstruction_weight=0.005,  # lambda_rec
    kl_beta_start=1e-5,
    kl_beta_end=1e-3,
    kl_warmup_epochs=25,
    # ---- Optional loss components (B) ---- #
    use_bfl=True,
    use_kl=True,
    use_rec=True,
    use_tail_loss=True,          # B3: logsumexp tail loss on negative logits
    lambda_tail=0.05,
    tail_beta=10.0,             # Do I care about the whole negative tail or mostly the single worst outliers?
    tail_margin=-2.0,           # From what negative logit value do I start caring? (Take sigmoid of logit and start to penalize all that are above sigma(-2.0))
    rec_target_mode="clean_signal",  # B4: "raw_input" (legacy) or "clean_signal"
    use_corr_loss=True,          # B6: needs rec_target_mode="clean_signal"
    # True only for raw I/Q clean-signal training. Uses phase-invariant complex
    # correlation; MSE still supervises the aligned I and Q amplitudes directly.
    use_iq_correlation_loss=True,
    lambda_corr=0.05,
    corr_eps=1e-8,
    log_score_quantiles=True,     # C: per-epoch logit quantiles -> score_quantiles.csv
    save_pretraining_sanity_plots=True,
    pretraining_sanity_split="val",
    pretraining_sanity_strict=True,
    pretraining_sanity_max_candidates=2048,

    # ---- AdamW ---- #
    learning_rate=5e-4,
    weight_decay=1e-4,
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    adam_epsilon=1e-7,
    clipnorm=1.0,

    # ---- Training ---- #
    epochs=100,
    early_stopping_patience=15,
    early_stopping_monitor="val_loss",

    # ---- Event-detection callback ---- #
    detection_every_epochs=1,
    detection_target_fp_per_year=1.0,
    detection_threshold_sweep_points=200,
    detection_log_fit_tail_fraction=1e-4,
    detection_min_tail_points=4,

    # ---- I/O ----
    model_name="WindowSize4096dec_clas_both_samplingUpdatedLosses_1e_minus_12_IQ",
    output_dir=os.path.join(_PROJECT_ROOT, "runs"),
    checkpoint_subdir="checkpoints",
    figures_subdir="figures",
    analysis_subdir="analysis",
    random_seed=42,
)

# ===================================================================== #
# Everything below only runs when executed directly (python train.py),
# NOT when imported (e.g. `from train import cfg` in analyze.py).
# ===================================================================== #

if __name__ == "__main__":

    # Convert PBH masses to kg (the preprocessing helper expects physical units)
    M_SOLAR = cfg.M_solar
    pbh_masses_kg = [m * M_SOLAR for m in cfg.m_PBH_injection_list]

    # ================================================================= #
    # 2. Directories & reproducibility
    # ================================================================= #

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

    # Persist the config so every run is reproducible.
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({k: (v if not hasattr(v, "__name__") else v.__name__)
                   for k, v in cfg.__dict__.items()}, fh, indent=2, default=str)

    # ================================================================= #
    # 3. Preprocessing  →  tf.data datasets
    # ================================================================= #

    print("--- Training objective ---")
    print(f"    Data mode: {'raw I/Q (2 channels)' if cfg.use_I_Q else 'amplitude (1 channel)'}")
    print(f"    Reconstruction target: {cfg.rec_target_mode}")
    print(
        "    Correlation loss: "
        + (
            "phase-invariant complex I/Q correlation"
            if cfg.use_iq_correlation_loss
            else ("flattened Pearson correlation" if cfg.use_corr_loss else "disabled")
        )
    )
    print("--- Preprocessing (this re-uses your existing data pipeline) ---")
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

    # ================================================================= #
    # 4. Build, compile, and summarise the model
    # ================================================================= #

    model = build_vae_classifier(cfg)
    # AdamW lives in `tf.keras.optimizers` as of TF 2.11+. Fall back gracefully.
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
    # Build the model by running a dry forward pass so summary() works.
    for batch in train_ds.take(1):
        _ = model(batch[0], training=False)
    model.summary(expand_nested=True)

    # ================================================================= #
    # 5. Callbacks
    # ================================================================= #

    beta_cb = BetaAnnealing(
        model=model,
        start=cfg.kl_beta_start,
        end=cfg.kl_beta_end,
        warmup_epochs=cfg.kl_warmup_epochs,
    )

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "best.keras"),
        monitor=cfg.early_stopping_monitor,
        save_best_only=True,
        save_weights_only=False,   # full model — Keras reconstructs the graph on load
        mode="min",
        verbose=1,
    )

    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.early_stopping_monitor,
        patience=cfg.early_stopping_patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [beta_cb, ckpt_cb, early_cb]
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

    # ================================================================= #
    # 6. Fit
    # ================================================================= #

    print("\n--- Starting training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ================================================================= #
    # 7. Plot training history
    # ================================================================= #

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

    _plot_history_curve(
        ["loss", "val_loss"], "total loss", "history_total_loss.png", logy=True
    )
    _plot_history_curve(
        ["focal_loss", "val_focal_loss"], "focal loss", "history_focal_loss.png", logy=True
    )
    _plot_history_curve(
        ["recon_loss", "val_recon_loss"], "reconstruction MSE", "history_recon_loss.png", logy=True
    )
    _plot_history_curve(
        ["kl_loss", "val_kl_loss"], "KL divergence", "history_kl_loss.png", logy=True
    )
    _plot_history_curve(
        ["tail_loss", "val_tail_loss"], "logsumexp tail loss", "history_tail_loss.png", logy=True
    )
    _plot_history_curve(
        ["corr_loss", "val_corr_loss"], "correlation loss", "history_corr_loss.png", logy=True
    )
    _plot_history_curve(["auc", "val_auc"], "AUC", "history_auc.png")

    if "det_event_recall" in history.history:
        _plot_history_curve(
            ["det_event_recall"], "event recall @ target FP/year",
            "history_det_event_recall.png",
        )
    if "det_threshold" in history.history:
        _plot_history_curve(
            ["det_threshold"], "operating threshold",
            "history_det_threshold.png",
        )

    # Save the full history as JSON for later inspection.
    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  fh, indent=2)

    print(f"\n--- Training complete. Artefacts in {run_dir} ---")
