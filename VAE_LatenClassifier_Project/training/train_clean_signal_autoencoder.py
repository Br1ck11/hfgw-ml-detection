"""
Standalone clean-signal CNN autoencoder (E).

This is NOT a denoiser. It is trained ONLY on clean signal shapes — trimmed,
cavity-convolved chirp waveforms placed in otherwise-zero windows — with NO
detector noise. The goal is to learn the manifold/family of valid clean
signal shapes so it can later be used as a projection-based verifier
(see evaluate_post_vae_signal_manifold.py).

Training input : clean_signal_window (optionally augmented in clean space)
Training target: the SAME clean_signal_window (un-augmented)

The model is intentionally small and bottlenecked so it cannot become a
trivial identity map; it should project arbitrary inputs TOWARD the
clean-signal manifold.

All amplitudes live in the NORMALIZED space of the main pipeline:
clean_norm = clean_raw / noise_std, i.e. the window peak equals the
injected peak SNR and the outside-signal baseline is exactly zero.

Usage:
    python train_clean_signal_autoencoder.py
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


import csv
import json
import os

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from data_pre_processing.chirp_BW_conv_signal_generation import get_trimmed_waveform
from vae.losses import correlation_loss


# ===================================================================== #
# Configuration
# ===================================================================== #

CONFIG = {
    # ---- Signal family (must match the main pipeline) ----
    "m_PBH_solar_list": [1e-8],          # PBH masses in SOLAR masses
    "M_solar": 1.988e30,
    "f0_gw": 5.0e9,
    "Gamma_gw": 100e3,
    "N_gw": 32768,
    "relative_threshold_factor": 1e-3,   # SAME trimming rule as injection
    "response_mode": "real_lorentzian",  # or "complex_breit_wigner"

    # ---- Window / amplitude (normalized space: peak == peak SNR) ----
    "window_size": 1024,
    "snr_range": [1.0, 8.0],             # uniform peak-SNR sampling
    "in_channels": 1,                    # 1 = amplitude/envelope

    # ---- Dataset sizes ----
    "num_train": 20000,
    "num_val": 4000,
    "random_seed": 42,

    # ---- Clean-space augmentation (applied to the INPUT only) ----
    "aug_amp_jitter": 0.1,               # multiplicative jitter: U(1-a, 1+a); 0 disables
    "aug_time_shift_max": 0,             # max |shift| in samples (input only); 0 disables
    "aug_gauss_frac_of_peak": 0.02,      # additive Gaussian, std = frac * window peak
    "aug_dropout_prob": 0.02,            # per-sample dropout prob inside signal support
    "aug_nuisance_gamma_frac": 0.0,      # optional cavity-Gamma nuisance variation, e.g. 0.05

    # ---- Model (small, bottlenecked — must NOT be an identity map) ----
    "hidden_channels": 32,
    "latent_channels": 16,               # 8 or 16
    "kernel_size": 31,

    # ---- Loss ----
    "use_corr_loss": False,              # default off for the first baseline
    "lambda_corr_autoencoder": 0.01,     # small (0.01 or 0.05) if enabled
    "corr_eps": 1e-8,

    # ---- Optimization ----
    "batch_size": 256,
    "epochs": 60,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,

    # ---- I/O ----
    # Anchored to the project root (bootstrap) so the script works from any CWD.
    "output_dir": os.path.join(_PROJECT_ROOT, "runs_clean_signal_autoencoder", "baseline"),
    "num_reconstruction_plots": 8,
}


# ===================================================================== #
# Model
# ===================================================================== #

@keras.saving.register_keras_serializable(package="clean_ae")
class CleanSignalCNNAutoencoder(tf.keras.Model):
    """
    Small bottlenecked Conv1D autoencoder over clean signal windows.

    encoder: Conv1d(in, hidden, k) GELU -> Conv1d(hidden, hidden, k) GELU
             -> Conv1d(hidden, latent, k) GELU
    decoder: Conv1d(latent, hidden, k) GELU -> Conv1d(hidden, hidden, k) GELU
             -> Conv1d(hidden, in, 1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 16,
        kernel_size: int = 31,
        name: str = "clean_signal_cnn_autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.latent_channels = int(latent_channels)
        self.kernel_size = int(kernel_size)

        k = self.kernel_size
        self.encoder_layers = [
            layers.Conv1D(self.hidden_channels, k, padding="same", name="enc_conv1"),
            layers.Activation("gelu"),
            layers.Conv1D(self.hidden_channels, k, padding="same", name="enc_conv2"),
            layers.Activation("gelu"),
            layers.Conv1D(self.latent_channels, k, padding="same", name="enc_conv3"),
            layers.Activation("gelu"),
        ]
        self.decoder_layers = [
            layers.Conv1D(self.hidden_channels, k, padding="same", name="dec_conv1"),
            layers.Activation("gelu"),
            layers.Conv1D(self.hidden_channels, k, padding="same", name="dec_conv2"),
            layers.Activation("gelu"),
            layers.Conv1D(self.in_channels, 1, padding="same", name="dec_out"),
        ]

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training) if isinstance(layer, layers.Layer) else layer(x)
        for layer in self.decoder_layers:
            x = layer(x, training=training) if isinstance(layer, layers.Layer) else layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "kernel_size": self.kernel_size,
        })
        return config


# ===================================================================== #
# Clean-signal dataset construction
# ===================================================================== #

def _base_waveforms(cfg):
    """Trimmed clean waveform per mass (peak-normalized amplitude envelopes)."""
    waveforms = []
    for m_solar in cfg["m_PBH_solar_list"]:
        m_kg = m_solar * cfg["M_solar"]
        gamma_values = [cfg["Gamma_gw"]]
        if cfg["aug_nuisance_gamma_frac"] > 0:
            frac = cfg["aug_nuisance_gamma_frac"]
            gamma_values += [
                cfg["Gamma_gw"] * (1.0 - frac),
                cfg["Gamma_gw"] * (1.0 + frac),
            ]
        for gamma in gamma_values:
            w = get_trimmed_waveform(
                m_kg, 1.0, cfg["f0_gw"], gamma, cfg["N_gw"], cfg["M_solar"],
                relative_threshold_factor=cfg["relative_threshold_factor"],
                response_mode=cfg["response_mode"],
            )
            if w.size == 0:
                continue
            env = np.abs(w).astype(np.float64)
            peak = env.max()
            if peak > 0:
                waveforms.append(env / peak)  # peak-normalized envelope
    if not waveforms:
        raise RuntimeError("No valid clean waveforms could be generated.")
    return waveforms


def make_clean_dataset(cfg, num_samples, rng, augment=True):
    """
    Returns (inputs, targets) of shape [N, window, 1] in normalized space.

    Targets: exact clean signal window (trimmed envelope, scaled so its
             peak equals the sampled peak SNR, zeros elsewhere).
    Inputs : optionally augmented copies of the target.
    """
    window = cfg["window_size"]
    waveforms = _base_waveforms(cfg)
    snr_lo, snr_hi = cfg["snr_range"]

    targets = np.zeros((num_samples, window), dtype=np.float32)
    inputs = np.zeros((num_samples, window), dtype=np.float32)

    for i in range(num_samples):
        env = waveforms[rng.integers(len(waveforms))]
        sig_len = env.size
        snr = rng.uniform(snr_lo, snr_hi)
        sig = (env * snr).astype(np.float32)

        if sig_len >= window:
            # Signal longer than the window: place a random crop
            off = rng.integers(0, sig_len - window + 1)
            seg = sig[off:off + window]
            t0 = 0
            targets[i, :] = seg
        else:
            t0 = rng.integers(0, window - sig_len + 1)
            targets[i, t0:t0 + sig_len] = sig

        x = targets[i].copy()
        if augment:
            # 1. random amplitude scaling (multiplicative jitter)
            if cfg["aug_amp_jitter"] > 0:
                x = x * rng.uniform(1.0 - cfg["aug_amp_jitter"], 1.0 + cfg["aug_amp_jitter"])
            # 2. small time shift (input only; target stays put)
            if cfg["aug_time_shift_max"] > 0:
                shift = int(rng.integers(-cfg["aug_time_shift_max"], cfg["aug_time_shift_max"] + 1))
                if shift != 0:
                    x = np.roll(x, shift)
                    if shift > 0:
                        x[:shift] = 0.0
                    else:
                        x[shift:] = 0.0
            # 3. small Gaussian perturbation proportional to the signal peak
            if cfg["aug_gauss_frac_of_peak"] > 0:
                peak = float(x.max())
                if peak > 0:
                    support = x > 0
                    noise = rng.normal(0.0, cfg["aug_gauss_frac_of_peak"] * peak, size=window).astype(np.float32)
                    x = np.where(support, x + noise, x)
            # 4. mild dropout of signal samples
            if cfg["aug_dropout_prob"] > 0:
                drop = (rng.random(window) < cfg["aug_dropout_prob"]) & (x != 0)
                x[drop] = 0.0
        inputs[i] = x

    return inputs[..., None], targets[..., None]


# ===================================================================== #
# Validation metrics
# ===================================================================== #

def validation_metrics(model, x_val, y_val, batch_size, eps=1e-8):
    """val_mse, val_corr, val_peak_error, val_energy_error."""
    preds = model.predict(x_val, batch_size=batch_size, verbose=0)
    p = preds.reshape(len(preds), -1).astype(np.float64)
    t = y_val.reshape(len(y_val), -1).astype(np.float64)

    mse = float(np.mean((p - t) ** 2))

    p_c = p - p.mean(axis=1, keepdims=True)
    t_c = t - t.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(p_c, axis=1) * np.linalg.norm(t_c, axis=1) + eps
    corr = float(np.mean(np.sum(p_c * t_c, axis=1) / denom))

    peak_t = np.max(np.abs(t), axis=1)
    peak_p = np.max(np.abs(p), axis=1)
    peak_err = float(np.mean(np.abs(peak_p - peak_t) / (peak_t + eps)))

    en_t = np.sqrt(np.sum(t ** 2, axis=1))
    en_p = np.sqrt(np.sum(p ** 2, axis=1))
    energy_err = float(np.mean(np.abs(en_p - en_t) / (en_t + eps)))

    return {
        "val_mse": mse,
        "val_corr": corr,
        "val_peak_error": peak_err,
        "val_energy_error": energy_err,
    }


def save_reconstruction_plots(model, x_val, y_val, out_dir, num_plots=8):
    os.makedirs(out_dir, exist_ok=True)
    idx = np.linspace(0, len(x_val) - 1, num_plots).astype(int)
    preds = model.predict(x_val[idx], verbose=0)
    for rank, i in enumerate(idx):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_val[i, :, 0], color="black", lw=0.9, label="clean target")
        ax.plot(x_val[i, :, 0], color="tab:blue", lw=0.7, alpha=0.6, label="input (augmented)")
        ax.plot(preds[rank, :, 0], color="tab:red", lw=0.9, label="reconstruction")
        ax.set_xlabel("time step")
        ax.set_ylabel("normalized amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"val_reconstruction_{rank+1}.png"), dpi=200)
        plt.close(fig)


# ===================================================================== #
# Training
# ===================================================================== #

def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "validation_reconstruction_plots")

    np.random.seed(cfg["random_seed"])
    tf.random.set_seed(cfg["random_seed"])
    rng = np.random.default_rng(cfg["random_seed"])

    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    print("--- Building clean-signal datasets (NO detector noise) ---")
    x_train, y_train = make_clean_dataset(cfg, cfg["num_train"], rng, augment=True)
    x_val, y_val = make_clean_dataset(cfg, cfg["num_val"], rng, augment=True)
    print(f"train: {x_train.shape}, val: {x_val.shape}")

    model = CleanSignalCNNAutoencoder(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        latent_channels=cfg["latent_channels"],
        kernel_size=cfg["kernel_size"],
    )
    _ = model(x_train[:2])  # build
    model.summary()

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
    except AttributeError:
        from tensorflow.keras.optimizers.experimental import AdamW
        optimizer = AdamW(
            learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )

    use_corr = bool(cfg["use_corr_loss"])
    lambda_corr = float(cfg["lambda_corr_autoencoder"])
    corr_eps = float(cfg["corr_eps"])

    def loss_fn(y_true, y_pred):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        if use_corr:
            loss = loss + lambda_corr * correlation_loss(y_pred, y_true, eps=corr_eps)
        return loss

    model.compile(optimizer=optimizer, loss=loss_fn)

    log_path = os.path.join(out_dir, "training_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)
    best_path = os.path.join(out_dir, "best_model.keras")
    last_path = os.path.join(out_dir, "last_model.keras")

    best_val_mse = np.inf
    patience_left = cfg["early_stopping_patience"]
    log_columns = ["epoch", "train_loss", "val_loss",
                   "val_mse", "val_corr", "val_peak_error", "val_energy_error"]

    for epoch in range(1, cfg["epochs"] + 1):
        hist = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=cfg["batch_size"],
            epochs=1,
            verbose=1,
        )
        metrics = validation_metrics(model, x_val, y_val, cfg["batch_size"])
        row = {
            "epoch": epoch,
            "train_loss": float(hist.history["loss"][0]),
            "val_loss": float(hist.history["val_loss"][0]),
            **metrics,
        }
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=log_columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"[epoch {epoch}] val_mse={metrics['val_mse']:.5e} "
            f"val_corr={metrics['val_corr']:.4f} "
            f"val_peak_err={metrics['val_peak_error']:.4f} "
            f"val_energy_err={metrics['val_energy_error']:.4f}"
        )

        model.save(last_path)
        if metrics["val_mse"] < best_val_mse:
            best_val_mse = metrics["val_mse"]
            model.save(best_path)
            patience_left = cfg["early_stopping_patience"]
            print(f"  -> new best (val_mse={best_val_mse:.5e}), saved {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Reload best weights for the final plots
    best_model = tf.keras.models.load_model(
        best_path, custom_objects={"CleanSignalCNNAutoencoder": CleanSignalCNNAutoencoder},
        compile=False,
    )
    save_reconstruction_plots(best_model, x_val, y_val, plot_dir,
                              num_plots=cfg["num_reconstruction_plots"])

    # Sanity check 4 (H): the AE must NOT act as an identity map on noise.
    noise_in = rng.normal(0.0, 1.0, size=(512, cfg["window_size"], 1)).astype(np.float32)
    noise_out = best_model.predict(noise_in, batch_size=cfg["batch_size"], verbose=0)
    passthrough = float(
        np.mean(np.sum(noise_in * noise_out, axis=(1, 2))
                / (np.linalg.norm(noise_in.reshape(512, -1), axis=1)
                   * np.linalg.norm(noise_out.reshape(512, -1), axis=1) + 1e-8))
    )
    print(f"[identity check] mean cosine(noise_in, AE(noise_in)) = {passthrough:.4f} "
          "(should be clearly < 1; high values mean the AE is too close to identity)")
    with open(os.path.join(out_dir, "identity_check.txt"), "w") as fh:
        fh.write(f"mean_cosine_noise_passthrough: {passthrough}\n")

    print(f"--- Done. Artefacts in {out_dir} ---")
    return out_dir


if __name__ == "__main__":
    main()
