"""
Ablation runner: trains the same VAE-classifier architecture under several
classifier-input modes and produces a side-by-side comparison.

The hypothesis under test
-------------------------
> "When the classifier shares the decoder's reparameterization sample, the
>  classifier loss helps the encoder learn signal-discriminating features,
>  which in turn lets the decoder reconstruct the signal."

The five modes we sweep narrow the question down:

    concat_mean_logvar        production deterministic baseline
    z_mean                    even more deterministic, no σ info to classifier
    shared_sample             original "good" setup (shared ε)
    independent_sample        same noise statistics, independent ε  ← key test
    shared_sample_detached    isolates "noise vs gradient" question

Reading the result
------------------
Compare two pairs:

    independent_sample vs shared_sample
        If recon_loss curves and decoder reconstructions are similar,
        the *noise injection on the classifier* is what matters
        (a regularizer that forces the encoder to spread information).
        If shared_sample is much better, the *shared sample path itself* is
        what matters — i.e. classifier gradients on the same ε realisation
        align with decoder gradients in a way that independent ε can't fake.

    shared_sample vs shared_sample_detached
        Detached cuts the encoder-grad path from the classifier loss while
        keeping the same forward computation. If detached collapses recon
        quality, the classifier-via-encoder gradient pathway is real and
        causal. If detached works fine, the encoder-grad pathway wasn't
        doing the work.

Cost
----
Five training runs sequentially. To cap wall time, the script defaults to
fewer epochs than train.py (`EPOCHS_PER_MODE = 30`) and skips the slow
event-detection callback (`detection_every_epochs=0`) — those are noisy on
short runs. Override at the top of the script.

The gradient inspector is enabled in every run with a coarse cadence so the
extra cost stays low (`decompose_every_n_steps=200`).
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
import gc
import shutil
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
# Sweep configuration
# ===================================================================== #

MODES = (
    "concat_mean_logvar",
    "z_mean",
    "shared_sample",
    "independent_sample",
    "shared_sample_detached",
)

EPOCHS_PER_MODE = int(os.environ.get("EPOCHS_PER_MODE", "15"))
DECOMPOSE_EVERY_N_STEPS = int(os.environ.get("DECOMPOSE_EVERY", "100"))
LAYER_NORMS_EVERY_N_STEPS = int(
    os.environ.get("LAYER_NORMS_EVERY", str(DECOMPOSE_EVERY_N_STEPS))
)
WRITE_TENSORBOARD = os.environ.get("WRITE_TB", "0") == "1"
FIT_VERBOSE = int(os.environ.get("FIT_VERBOSE", "1"))
ENABLE_DETECTION_CB = os.environ.get("ENABLE_DETECTION", "0") == "1"
RUN_DETECTION_EVERY = int(os.environ.get("DETECTION_EVERY", "0"))


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

    # ---- Signal injection ---- #
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
    # Clean injected-signal saving + metadata (A1-A4). The preprocessing is
    # shared across all modes, so these are decided once for the whole sweep.
    save_clean_signals=True,
    save_metadata=True,
    include_clean_in_datasets=False,  # True -> (x, y, clean) batches; required
                                      # for rec_target_mode="clean_signal"
    use_run_specific_memmap_dir=True,  # memmaps under <output_dir>/<model_name>/

    # ---- Architecture ----
    num_filters_per_layer=[16, 32, 64, 96],
    kernel_sizes_per_layer=[16, 7, 7, 7],
    strides_per_layer=[2, 2, 4, 4],
    activation="silu",
    latent_dim=16,
    classifier_hidden_units=[64, 32],
    classifier_dropout=0.1,

    # The ablation sweep uses the more granular classifier_input_mode
    # per run, so this top-level boolean is not consulted by
    # build_vae_classifier_ablation. Listed here only for completeness.
    classifier_samples_z=False,

    # ---- Losses ---- #
    focal_gamma=2.0,
    focal_alpha=0.25,
    focal_weight=1.0,             # lambda_bfl
    reconstruction_weight=0.05,   # lambda_rec
    kl_beta_start=1e-4,
    kl_beta_end=1e-2,
    kl_warmup_epochs=5,
    # ---- Optional loss components (B), applied to EVERY mode in the sweep.
    # Keep them identical across modes so differences stay attributable to
    # the classifier-input routing. ---- #
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
    log_score_quantiles=True,     # C: per-epoch logit quantiles CSV per mode
    save_pretraining_sanity_plots=True,

    # ---- AdamW ---- #
    learning_rate=5e-4,
    weight_decay=1e-4,
    clipnorm=1.0,

    # ---- Training ---- #
    epochs=EPOCHS_PER_MODE,
    early_stopping_patience=max(EPOCHS_PER_MODE // 3, 5),
    tf_batch_size=4096,

    # ---- Detection callback ---- #
    detection_every_epochs=RUN_DETECTION_EVERY if ENABLE_DETECTION_CB else 0,
    detection_target_fp_per_year=1.0,

    # ---- I/O ----
    model_name="vae_cls_ablation",
    output_dir=os.path.join(_PROJECT_ROOT, "runs_ablation"),
)


# ===================================================================== #
# Helpers
# ===================================================================== #

def build_optimizer():
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon, clipnorm=cfg.clipnorm,
        )
    except AttributeError:
        from tensorflow.keras.optimizers.experimental import AdamW
        return AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            beta_1=cfg.adam_beta_1, beta_2=cfg.adam_beta_2,
            epsilon=cfg.adam_epsilon, clipnorm=cfg.clipnorm,
        )


def reseed():
    """Re-seed before each model build so all modes start from the same
    weight init and see the same data ordering."""
    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)


def train_one_mode(mode: str, train_ds, val_ds, ablation_root: str) -> dict:
    print(f"\n================== MODE: {mode} ==================")
    reseed()

    run_dir = os.path.join(ablation_root, mode)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    fig_dir = os.path.join(run_dir, "figures")
    grad_dir = os.path.join(run_dir, "grad_logs")
    for d in (ckpt_dir, fig_dir, grad_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        snap = {k: (v if not hasattr(v, "__name__") else v.__name__)
                for k, v in cfg.__dict__.items()}
        snap["MODE"] = mode
        snap["DECOMPOSE_EVERY_N_STEPS"] = DECOMPOSE_EVERY_N_STEPS
        snap["LAYER_NORMS_EVERY_N_STEPS"] = LAYER_NORMS_EVERY_N_STEPS
        snap["WRITE_TENSORBOARD"] = WRITE_TENSORBOARD
        snap["ENABLE_DETECTION_CB"] = ENABLE_DETECTION_CB
        json.dump(snap, fh, indent=2, default=str)

    model = build_vae_classifier_ablation(cfg, mode=mode)
    # GradientInspector requires eager execution — see its docstring.
    model.compile(optimizer=build_optimizer(), run_eagerly=True)
    for batch in train_ds.take(1):
        _ = model(batch[0], training=False)

    callbacks = [
        BetaAnnealing(
            model=model,
            start=cfg.kl_beta_start, end=cfg.kl_beta_end,
            warmup_epochs=cfg.kl_warmup_epochs,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best.keras"),
            monitor=cfg.early_stopping_monitor,
            save_best_only=True, save_weights_only=False,
            mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.early_stopping_monitor,
            patience=cfg.early_stopping_patience,
            mode="min", restore_best_weights=True, verbose=1,
        ),
    ]
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
    callbacks.append(
        GradientInspector(
            out_dir=grad_dir,
            decompose_every_n_steps=DECOMPOSE_EVERY_N_STEPS,
            layer_norms_every_n_steps=LAYER_NORMS_EVERY_N_STEPS,
            write_tensorboard=WRITE_TENSORBOARD,
            verbose=1,
        )
    )
    if cfg.detection_every_epochs > 0:
        callbacks.append(
            EventDetectionCallback(
                model=model, val_ds=val_ds,
                window_size=cfg.window_size, step_size=cfg.step_size,
                fs=cfg.sampling_rate_hz,
                every_epochs=cfg.detection_every_epochs,
                target_fp_per_year=cfg.detection_target_fp_per_year,
                sweep_points=cfg.detection_threshold_sweep_points,
                log_fit_tail_fraction=cfg.detection_log_fit_tail_fraction,
                min_tail_points=cfg.detection_min_tail_points,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=FIT_VERBOSE,
    )

    history_path = os.path.join(run_dir, "history.json")
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(history_path, "w") as fh:
        json.dump(history_dict, fh, indent=2)
    pd.DataFrame(history_dict).to_csv(
        os.path.join(run_dir, "history.csv"),
        index_label="epoch",
    )

    grad_csv = os.path.join(grad_dir, "gradients.csv")
    if os.path.exists(grad_csv):
        plot_gradient_log(grad_csv, grad_dir)

    # Free TF resources between runs.
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "mode": mode,
        "run_dir": run_dir,
        "history": history_dict,
        "grad_csv": grad_csv if os.path.exists(grad_csv) else None,
    }


# ===================================================================== #
# Main
# ===================================================================== #

if __name__ == "__main__":
    M_SOLAR = cfg.M_solar
    pbh_masses_kg = [m * M_SOLAR for m in cfg.m_PBH_injection_list]

    ablation_root = os.path.join(cfg.output_dir, cfg.model_name)
    os.makedirs(ablation_root, exist_ok=True)

    # Keep preprocessing artifacts isolated for this sweep. The pipeline is
    # built ONCE and shared by every mode, so the memmaps live at the sweep
    # root (not per mode).
    if cfg.use_run_specific_memmap_dir:
        cfg.memmap_dir = os.path.join(ablation_root, "memmaps")
        if cfg.stats_dir is not None:
            cfg.stats_dir = os.path.join(ablation_root, "stats")

    with open(os.path.join(ablation_root, "ablation_config.json"), "w") as fh:
        snap = {k: (v if not hasattr(v, "__name__") else v.__name__)
                for k, v in cfg.__dict__.items()}
        snap["MODES"] = list(MODES)
        snap["EPOCHS_PER_MODE"] = EPOCHS_PER_MODE
        snap["DECOMPOSE_EVERY_N_STEPS"] = DECOMPOSE_EVERY_N_STEPS
        snap["ENABLE_DETECTION_CB"] = ENABLE_DETECTION_CB
        json.dump(snap, fh, indent=2, default=str)

    # Build the data pipeline once and reuse it across modes — same windowing
    # and same injected signals so any difference is attributable to the model.
    print("--- Preprocessing (shared across all modes) ---")
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
        use_amps=cfg.use_amps, use_I_Q=cfg.use_I_Q,
        inject_signals=cfg.inject_signals,
        signal_injection_probability=cfg.signal_injection_probability,
        m_PBH_injection_list=pbh_masses_kg,
        amplitude_spectrum_range=cfg.amplitude_spectrum_range,
        num_signals_to_inject_per_segment=cfg.num_signals_to_inject_per_segment,
        snr_based_injection=cfg.snr_based_injection,
        custom_noise_std=cfg.custom_noise_std,
        f0_gw=cfg.f0_gw, Gamma_gw=cfg.Gamma_gw, N_gw=cfg.N_gw,
        M_solar=cfg.M_solar,
        response_mode=cfg.response_mode,
        save_clean_signals=cfg.save_clean_signals,
        save_metadata=cfg.save_metadata,
        include_clean_in_datasets=cfg.include_clean_in_datasets,
        memmap_dir=cfg.memmap_dir, stats_dir=cfg.stats_dir,
        return_tf_datasets=True,
        tf_batch_size=cfg.tf_batch_size,
        tf_shuffle=cfg.tf_shuffle, tf_repeat=cfg.tf_repeat,
        random_seed=cfg.random_seed,
        reject_unrepresentable_injections=cfg.reject_unrepresentable_injections,
        no_overlap_injections=cfg.no_overlap_injections,
        no_overlap_margin_samples=cfg.no_overlap_margin_samples,
        no_overlap_max_attempts=cfg.no_overlap_max_attempts,
        return_info=True,
    )

    # One sanity-plot pass for the shared pipeline (identical for all modes).
    if cfg.save_pretraining_sanity_plots and cfg.inject_signals:
        print("\n--- Saving pre-training data sanity plots (shared) ---")
        save_pretraining_sanity_plots(
            preprocessing_info,
            output_dir=os.path.join(ablation_root, "pretraining_checks"),
            split=cfg.pretraining_sanity_split,
            strict=cfg.pretraining_sanity_strict,
            max_candidates=cfg.pretraining_sanity_max_candidates,
        )
    elif cfg.save_pretraining_sanity_plots:
        print(
            "\n--- Skipping pre-training signal plots: inject_signals=False, "
            "so no noise+signal window exists. ---"
        )

    # ----- Run each mode ----------------------------------------- #
    results = []
    for mode in MODES:
        try:
            res = train_one_mode(mode, train_ds, val_ds, ablation_root)
            results.append(res)
        except Exception as e:
            print(f"!!! MODE {mode} failed: {e!r}")
            import traceback; traceback.print_exc()
            results.append({"mode": mode, "error": repr(e)})

    # Persist a top-level summary.
    with open(os.path.join(ablation_root, "summary.json"), "w") as fh:
        json.dump(
            [{k: v for k, v in r.items() if k != "history"} for r in results],
            fh, indent=2, default=str,
        )

    # ===================================================================== #
    # Cross-mode comparison plots
    # ===================================================================== #

    # First: print what keys the runs actually produced. If you see something
    # weird here ("history is None", "history is empty", or unexpected names),
    # that's the root cause of empty plots.
    print("\n--- Diagnostic: per-mode history snapshot ---")
    for r in results:
        if r.get("error"):
            print(f"  {r['mode']:30s}  ERROR: {r['error']}")
            continue
        h = r.get("history")
        if h is None:
            print(f"  {r['mode']:30s}  history MISSING")
        elif len(h) == 0:
            print(f"  {r['mode']:30s}  history is EMPTY (model.fit returned no metrics)")
        else:
            n_epochs = max(len(v) for v in h.values()) if h else 0
            print(f"  {r['mode']:30s}  {n_epochs} epochs, keys: {sorted(h.keys())}")

    def _safe_legend(ax) -> bool:
        """Attach a legend only if at least one labeled artist exists."""
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return False
        ax.legend()
        return True

    def history_curve(metric: str, ylabel: str, fname: str, logy: bool = False):
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = 0
        for r in results:
            h = r.get("history") or {}
            if metric in h and len(h[metric]) > 0:
                ax.plot(h[metric], label=r["mode"])
                plotted += 1
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{metric} across classifier-input modes")
        ax.grid(True, alpha=0.3)
        if not _safe_legend(ax):
            print(f"[plot] '{metric}' not present in any run's history — "
                  f"skipping {fname}.")
            plt.close(fig)
            return
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_root, fname), dpi=200)
        plt.close(fig)

    history_curve("recon_loss", "training reconstruction MSE",
                  "compare_train_recon_loss.png", logy=True)
    history_curve("val_recon_loss", "validation reconstruction MSE",
                  "compare_val_recon_loss.png", logy=True)
    history_curve("focal_loss", "training focal loss",
                  "compare_train_focal_loss.png", logy=True)
    history_curve("val_focal_loss", "validation focal loss",
                  "compare_val_focal_loss.png", logy=True)
    history_curve("auc", "training AUC", "compare_train_auc.png")
    history_curve("val_auc", "validation AUC", "compare_val_auc.png")
    history_curve("kl_loss", "KL divergence", "compare_kl_loss.png", logy=True)
    history_curve("tail_loss", "training logsumexp tail loss",
                  "compare_train_tail_loss.png", logy=True)
    history_curve("val_tail_loss", "validation logsumexp tail loss",
                  "compare_val_tail_loss.png", logy=True)
    history_curve("corr_loss", "training correlation loss",
                  "compare_train_corr_loss.png", logy=True)
    history_curve("val_corr_loss", "validation correlation loss",
                  "compare_val_corr_loss.png", logy=True)
    history_curve("neg_logit_q99", "negative logit q99 (val)",
                  "compare_neg_logit_q99.png")
    history_curve("pos_logit_q10", "positive logit q10 (val)",
                  "compare_pos_logit_q10.png")

    def grad_csv_curve(column: str, ylabel: str, title: str, fname: str,
                       ylim=None, hline: float = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = 0
        for r in results:
            if not r.get("grad_csv"):
                continue
            try:
                df = pd.read_csv(r["grad_csv"])
                if column not in df.columns or len(df) == 0:
                    continue
                df = df.sort_values("step")
                # Drop NaN rows (cosine on a zero-norm gradient list comes
                # out as NaN by definition — safe to ignore for plotting).
                series = df[["step", column]].dropna()
                if len(series) == 0:
                    continue
                window = max(len(series) // 50, 1)
                smoothed = series[column].rolling(window, min_periods=1).mean()
                ax.plot(series["step"], smoothed, label=r["mode"])
                plotted += 1
            except Exception as e:
                print(f"Could not plot '{column}' for {r['mode']}: {e!r}")
        if hline is not None:
            ax.axhline(hline, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if not _safe_legend(ax):
            print(f"[plot] gradient column '{column}' not present in any "
                  f"run's grad_logs/gradients.csv — skipping {fname}. "
                  "(This usually means the inspector never reached a "
                  "decompose step; check `decompose_every_n_steps` against "
                  "the actual number of training steps.)")
            plt.close(fig)
            return
        fig.tight_layout()
        fig.savefig(os.path.join(ablation_root, fname), dpi=200)
        plt.close(fig)

    grad_csv_curve(
        column="enc_cos/focal_recon",
        ylabel="cos(g_focal, g_recon) on encoder",
        title="Do focal and recon gradients align on the encoder?",
        fname="compare_encoder_cosine_focal_recon.png",
        ylim=(-1.05, 1.05),
        hline=0.0,
    )
    grad_csv_curve(
        column="enc_share/focal",
        ylabel="focal share of encoder grad norm",
        title="How big a slice of the encoder gradient does the classifier own?",
        fname="compare_encoder_share_focal.png",
        ylim=(-0.02, 1.02),
    )

    print(f"\n--- Ablation complete. Per-mode runs in {ablation_root}/<mode>/ ---")
    print(f"--- Cross-mode comparison plots in {ablation_root}/compare_*.png ---")
