"""
xAI / analysis helpers for the VAE-classifier.

This module exposes a single entry point, `analyse_model`, that:

  1. Pulls a batch of (signal, noise) examples from a dataset.
  2. Runs them through the trained model.
  3. Extracts and plots:
        - Input waveforms
        - Encoder Conv1D activations layer-by-layer (heatmaps)
        - Decoder Conv1DTranspose activations layer-by-layer (heatmaps)
        - Decoder reconstruction vs input and clean target, when available
        - Latent z_mean / z_log_var bar plots for both classes
        - A 2D embedding of z_mean for the full val set, coloured by label
        - Per-dimension KL contribution ("active units" diagnostic)
        - Composite multi-panel figure for one signal + one noise example

The idea is a *coherent* picture of what the model is doing, rather than a
collection of disconnected plots. Everything gets written under a single
output directory so a run is trivially shareable.

Why this is actually xAI
------------------------
* Activation maps show *where* in the input time-series the encoder is
  responding — a proxy for what pattern the filter has learned.
* Latent scatter + per-dim mean/var shows whether the two classes are
  separable in latent space, which is the disentanglement claim.
* Per-dim KL contribution identifies "dead" latent dimensions (KL ≈ 0),
  a well-known VAE diagnostic.
* Decoder activations + reconstructions let you ask "what does z
  represent?" — a causal/interventional style explanation.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from .model import QuadratureConv1D


NUM_LOWEST_LOGIT_SIGNAL_PLOTS = 4
VALID_ANALYSIS_LATENT_MODES = ("z_mean", "sampled_z")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _latex_safe_text(text: str) -> str:
    """
    Escape text for Matplotlib when text.usetex=True.

    The analysis plots are often rendered in environments with global LaTeX
    text enabled. Characters like '#', '%', '&', '_' and braces can then blow
    up in titles / labels. When usetex is off, leave the string unchanged.
    """
    if not plt.rcParams.get("text.usetex", False):
        return text

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _call_single_input_model(model, x, training: bool = False):
    """
    Call a single-input Functional model using its saved input structure.

    Depending on how a checkpoint was built, Keras may expect a bare tensor,
    a one-item list, or a dictionary keyed by the input name.
    """
    if len(model.inputs) != 1:
        return model(x, training=training)

    inputs_struct = getattr(model, "_inputs_struct", None)
    input_name = model.inputs[0].name.split(":")[0]
    if isinstance(inputs_struct, dict):
        return model({input_name: x}, training=training)
    if isinstance(inputs_struct, (list, tuple)):
        return model([x], training=training)
    return model(x, training=training)


def _load_signal_bounds_by_window(
    window_metadata_path: Optional[str],
    event_metadata_path: Optional[str],
    split: str,
) -> Dict[int, Tuple[int, int]]:
    """Map dataset window indices to the primary injected event's local bounds."""
    if not window_metadata_path or not event_metadata_path:
        return {}
    if not os.path.isfile(window_metadata_path) or not os.path.isfile(event_metadata_path):
        return {}

    events = {}
    with open(event_metadata_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            events[int(row["event_id"])] = (
                int(row["injection_start_sample"]),
                int(row["injection_end_sample"]),
            )

    bounds_by_window = {}
    with open(window_metadata_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            event_id = int(row["event_id"])
            if event_id not in events:
                continue

            window_start = int(row["window_start_sample"])
            window_end = int(row["window_end_sample"])
            event_start, event_end = events[event_id]
            local_start = max(0, event_start - window_start)
            local_end = min(window_end - window_start, event_end - window_start)
            if local_end > local_start:
                bounds_by_window[int(row["window_index"])] = (local_start, local_end)

    return bounds_by_window


def _add_signal_boundaries(ax, signal_bounds: Optional[Tuple[int, int]]) -> None:
    """Draw the start and exclusive end of an injected signal on a time-series axis."""
    if signal_bounds is None:
        return
    start, end = signal_bounds
    style = dict(color="tab:orange", linestyle="--", linewidth=1.1, alpha=0.95)
    ax.axvline(start, label="signal bounds", **style)
    ax.axvline(end, **style)


def _analysis_latent(vae_model, z_mean, z_log_var, latent_mode: str):
    """Return the latent tensor explicitly selected for analysis inference."""
    if latent_mode not in VALID_ANALYSIS_LATENT_MODES:
        raise ValueError(
            f"latent_mode must be one of {VALID_ANALYSIS_LATENT_MODES}, "
            f"got '{latent_mode}'."
        )
    if latent_mode == "z_mean":
        return z_mean
    return vae_model.sampling([z_mean, z_log_var])


def _collect_examples(
    vae_model,
    dataset,
    max_batches: int = 50,
    signal_search_factor: int = 10,
    latent_mode: str = "z_mean",
) -> Dict[str, Any]:
    """
    Score the dataset and collect arrays used by every downstream plot.

    Returns
    -------
    dict with keys:
        x_all, y_all          : raw windows and labels (numpy)
        z_mean_all, z_logvar_all, logits_all : model outputs
        sig_idx, noise_idx    : one index for a strong-signal example
                                 and one for a clean-noise example
    """
    x_batches, y_batches, clean_batches, window_index_batches = [], [], [], []
    zm_batches, zlv_batches, logit_batches = [], [], []
    signal_seen = False
    noise_seen = False
    max_search_batches = max(max_batches, max_batches * signal_search_factor)

    for i, batch in enumerate(dataset):
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
            clean = batch[2] if len(batch) > 2 else None
        else:
            continue

        y_np = np.asarray(y).reshape(-1)
        batch_has_signal = bool(np.any(y_np > 0.5))
        batch_has_noise = bool(np.any(y_np <= 0.5))

        keep_batch = i < max_batches
        if not keep_batch:
            # After the primary summary window, only keep extra batches if we
            # still need a positive or negative example for the composite plots.
            keep_batch = (
                (not signal_seen and batch_has_signal)
                or (not noise_seen and batch_has_noise)
            )

        if not keep_batch:
            if i + 1 >= max_search_batches and signal_seen and noise_seen:
                break
            if i + 1 >= max_search_batches:
                break
            continue

        z_mean, z_log_var = vae_model.encoder(x, training=False)
        analysis_z = _analysis_latent(vae_model, z_mean, z_log_var, latent_mode)
        cls_features = vae_model.classifier_features(
            z_mean, z_log_var, z=analysis_z
        )
        logits = vae_model.classifier(cls_features, training=False)

        x_batches.append(np.asarray(x))
        y_batches.append(y_np)
        if clean is not None:
            clean_batches.append(np.asarray(clean))
        batch_start = i * len(y_np)
        window_index_batches.append(
            np.arange(batch_start, batch_start + len(y_np), dtype=np.int64)
        )
        zm_batches.append(np.asarray(z_mean))
        zlv_batches.append(np.asarray(z_log_var))
        logit_batches.append(np.asarray(logits).reshape(-1))

        signal_seen = signal_seen or batch_has_signal
        noise_seen = noise_seen or batch_has_noise

        if i + 1 >= max_batches and signal_seen and noise_seen:
            break
        if i + 1 >= max_search_batches:
            break

    if not x_batches:
        raise RuntimeError("Dataset yielded no batches for analysis.")

    x_all = np.concatenate(x_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)
    clean_all = np.concatenate(clean_batches, axis=0) if clean_batches else None
    if clean_all is not None and len(clean_all) != len(x_all):
        raise RuntimeError(
            "Analysis received clean targets for only part of the collected dataset."
        )
    window_indices_all = np.concatenate(window_index_batches, axis=0)
    z_mean_all = np.concatenate(zm_batches, axis=0)
    z_logvar_all = np.concatenate(zlv_batches, axis=0)
    logits_all = np.concatenate(logit_batches, axis=0)

    sig_mask = y_all > 0.5
    noise_mask = ~sig_mask

    if np.any(sig_mask):
        # Keep one separate highest-logit signal example.
        sig_logits = np.where(sig_mask, logits_all, -np.inf)
        sig_idx = int(np.argmax(sig_logits))

        # Additionally select the lowest-scoring signal-labelled windows,
        # regardless of whether their logits are positive or negative.
        lowest_signal_candidates = np.where(sig_mask)[0]
        lowest_signal_candidates = lowest_signal_candidates[
            lowest_signal_candidates != sig_idx
        ]
        lowest_signal_sorted = lowest_signal_candidates[
            np.argsort(logits_all[lowest_signal_candidates])
        ]
        lowest_signal_idxs = [
            int(idx)
            for idx in lowest_signal_sorted[:NUM_LOWEST_LOGIT_SIGNAL_PLOTS]
        ]
    else:
        sig_idx = None
        lowest_signal_idxs = []

    if np.any(noise_mask):
        # Pick the most confidently-classified true negative
        noise_logits = np.where(noise_mask, logits_all, np.inf)
        noise_idx = int(np.argmin(noise_logits))
    else:
        noise_idx = None

    return dict(
        x_all=x_all, clean_all=clean_all, y_all=y_all,
        window_indices_all=window_indices_all,
        z_mean_all=z_mean_all, z_logvar_all=z_logvar_all,
        logits_all=logits_all,
        sig_idx=sig_idx, noise_idx=noise_idx,
        lowest_signal_idxs=lowest_signal_idxs,
        num_signal_windows=int(np.sum(sig_mask)),
        num_noise_windows=int(np.sum(noise_mask)),
        latent_mode=latent_mode,
    )


def _encoder_layer_outputs(vae_model) -> Tuple[tf.keras.Model, List[str]]:
    """Build a probe model returning encoder feature maps, including quadrature front-ends."""
    outs = []
    names = []
    for layer in vae_model.encoder.layers:
        if isinstance(layer, (QuadratureConv1D, layers.Conv1D)):
            outs.append(layer.output)
            names.append(layer.name)
    sub = tf.keras.Model(vae_model.encoder.input, outs, name="enc_probe")
    return sub, names


def _decoder_layer_outputs(vae_model) -> Tuple[tf.keras.Model, List[str]]:
    """Build a multi-output sub-model that returns every decoder transpose layer."""
    outs = []
    names = []
    for layer in vae_model.decoder.layers:
        if isinstance(layer, (layers.Conv1DTranspose, layers.Conv1D)):
            outs.append(layer.output)
            names.append(layer.name)
    sub = tf.keras.Model(vae_model.decoder.input, outs, name="dec_probe")
    return sub, names


# --------------------------------------------------------------------- #
# Individual plots
# --------------------------------------------------------------------- #

def plot_latent_scatter(data, save_path: str) -> None:
    """2D embedding of z_mean coloured by label. Uses PCA if latent_dim > 2."""
    z = data["z_mean_all"]
    y = data["y_all"] > 0.5
    if z.shape[1] >= 2:
        if z.shape[1] == 2:
            emb = z
            xlabel, ylabel = "z_mean[0]", "z_mean[1]"
        else:
            # Cheap PCA via SVD
            zc = z - z.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(zc, full_matrices=False)
            emb = u[:, :2] * s[:2]
            xlabel, ylabel = "PC1", "PC2"

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[~y, 0], emb[~y, 1], s=6, alpha=0.4, label="noise", c="tab:blue")
        ax.scatter(emb[y, 0], emb[y, 1], s=10, alpha=0.7, label="signal", c="tab:red")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title("Latent space (z_mean), validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)


def plot_latent_dim_stats(data, save_path: str) -> None:
    """Per-dimension mean and variance for each class + per-dim KL contribution."""
    z = data["z_mean_all"]
    lv = data["z_logvar_all"]
    y = data["y_all"] > 0.5
    latent_dim = z.shape[1]

    mean_sig = z[y].mean(axis=0) if np.any(y) else np.zeros(latent_dim)
    mean_noi = z[~y].mean(axis=0) if np.any(~y) else np.zeros(latent_dim)
    std_sig = z[y].std(axis=0) if np.any(y) else np.zeros(latent_dim)
    std_noi = z[~y].std(axis=0) if np.any(~y) else np.zeros(latent_dim)

    # Per-dim KL contribution (N(mu,sigma^2) || N(0,1))
    kl_per_dim = 0.5 * np.mean(
        np.square(z) + np.exp(lv) - 1.0 - lv, axis=0
    )

    x = np.arange(latent_dim)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].bar(x - 0.2, mean_sig, width=0.4, label="signal", color="tab:red")
    axes[0].bar(x + 0.2, mean_noi, width=0.4, label="noise", color="tab:blue")
    axes[0].set_ylabel("mean of z_mean")
    axes[0].legend()

    axes[1].bar(x - 0.2, std_sig, width=0.4, label="signal", color="tab:red")
    axes[1].bar(x + 0.2, std_noi, width=0.4, label="noise", color="tab:blue")
    axes[1].set_ylabel("std of z_mean")

    axes[2].bar(x, kl_per_dim, color="tab:purple")
    axes[2].set_ylabel("per-dim KL")
    axes[2].set_xlabel("latent dimension")
    axes[2].set_title("Near-zero KL -> inactive ('dead') dimension")

    fig.suptitle("Latent-space statistics per dimension")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_activations_for_example(
    vae_model,
    x_example: np.ndarray,
    label_str: str,
    save_path: str,
    clean_target: Optional[np.ndarray] = None,
    snr: Optional[float] = None,
    signal_bounds: Optional[Tuple[int, int]] = None,
    latent_mode: str = "z_mean",
) -> None:
    """Composite figure: input, activations, latent, and reconstruction comparisons."""
    enc_probe, enc_names = _encoder_layer_outputs(vae_model)
    dec_probe, dec_names = _decoder_layer_outputs(vae_model)

    x = x_example[None, ...]
    enc_acts = _call_single_input_model(enc_probe, x, training=False)
    if not isinstance(enc_acts, list):
        enc_acts = [enc_acts]
    z_mean, z_log_var = _call_single_input_model(vae_model.encoder, x, training=False)
    analysis_z = _analysis_latent(vae_model, z_mean, z_log_var, latent_mode)
    cls_features = vae_model.classifier_features(
        z_mean, z_log_var, z=analysis_z
    )
    dec_acts = _call_single_input_model(dec_probe, analysis_z, training=False)
    if not isinstance(dec_acts, list):
        dec_acts = [dec_acts]
    recon = _call_single_input_model(
        vae_model.decoder, analysis_z, training=False
    ).numpy()[0]
    if clean_target is not None and clean_target.shape != x_example.shape:
        raise ValueError(
            "clean_target must have the same [time, channels] shape as x_example; "
            f"got {clean_target.shape} and {x_example.shape}."
        )
    logit = float(vae_model.classifier(cls_features, training=False).numpy()[0, 0])
    prob = 1.0 / (1.0 + np.exp(-logit))

    num_channels = x_example.shape[1]
    n_enc = len(enc_acts)
    n_dec = len(dec_acts)
    has_iq_reconstruction_panels = (num_channels == 2)
    reconstruction_rows = 3 if has_iq_reconstruction_panels else 1
    enc_row_start = 1
    latent_row = enc_row_start + n_enc
    dec_row_start = latent_row + 1
    recon_row_start = dec_row_start + n_dec
    n_rows = 2 + reconstruction_rows + n_enc + n_dec
    # rows = input, enc*, latent, dec*, then I/Q/magnitude reconstruction panels
    fig = plt.figure(figsize=(13, 1.6 * n_rows + 1))
    gs = fig.add_gridspec(n_rows, 1)

    # --- Row 0: input ---
    ax = fig.add_subplot(gs[0])
    if num_channels == 1:
        ax.plot(x_example[:, 0], color="black", lw=0.8)
        ax.set_ylabel(_latex_safe_text("input"))
    elif num_channels == 2:
        ax.plot(x_example[:, 0], color="tab:green", lw=0.9, label="I")
        ax.plot(x_example[:, 1], color="tab:blue", lw=0.9, label="Q")
        ax.set_ylabel(_latex_safe_text("I / Q"))
        ax.legend(loc="upper right", fontsize=8)
    else:
        for ch in range(num_channels):
            ax.plot(x_example[:, ch], lw=0.8, label=f"ch{ch}")
        ax.set_ylabel(_latex_safe_text("input"))
        ax.legend(loc="upper right", fontsize=8)
    _add_signal_boundaries(ax, signal_bounds)
    if signal_bounds is not None:
        ax.legend(loc="upper right", fontsize=8)
    if snr is None:
        snr_str = "pure noise"
    elif snr != 0.0 and abs(snr) < 0.1:
        mantissa, exponent = f"{snr:.3e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        snr_str = f"SNR={mantissa}e{int(exponent)}"
    else:
        snr_str = f"SNR={snr:.1f}"
    ax.set_title(
        _latex_safe_text(
            f"{label_str} | {snr_str} | latent={latent_mode} | "
            f"logit={logit:+.3f} | p(signal)={prob:.3f}"
        ),
        fontweight="bold",
    )

    # --- Encoder activation rows ---
    for i, act in enumerate(enc_acts):
        ax = fig.add_subplot(gs[enc_row_start + i])
        a = np.asarray(act)[0].T  # (filters, time)
        im = ax.imshow(a, aspect="auto", cmap="magma")
        ax.set_ylabel(_latex_safe_text(f"enc L{i+1}\n({enc_names[i]})"))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # --- Latent row ---
    ax = fig.add_subplot(gs[latent_row])
    zm = z_mean.numpy()[0]
    zlv = z_log_var.numpy()[0]
    xs = np.arange(len(zm))
    ax.bar(xs - 0.2, zm, width=0.4, color="tab:red", label="z_mean")
    # KORREKTUR: Unicode-Zeichen entfernt, um LaTeX-Fehler zu vermeiden
    ax.bar(xs + 0.2, np.exp(0.5 * zlv), width=0.4, color="tab:blue", label="sigma=exp(0.5*logvar)")
    ax.set_ylabel(_latex_safe_text("latent"))
    ax.legend(loc="upper right", fontsize=8)

    # --- Decoder activation rows ---
    for i, act in enumerate(dec_acts):
        ax = fig.add_subplot(gs[dec_row_start + i])
        a = np.asarray(act)[0].T  # (filters, time)
        im = ax.imshow(a, aspect="auto", cmap="viridis")
        ax.set_ylabel(_latex_safe_text(f"dec L{i+1}\n({dec_names[i]})"))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # --- Reconstruction comparison rows ---
    if num_channels == 1:
        ax = fig.add_subplot(gs[recon_row_start])
        ax.plot(x_example[:, 0], color="black", lw=0.8, label="input", alpha=0.7)
        if clean_target is not None:
            ax.plot(
                clean_target[:, 0], color="tab:green", lw=0.9,
                linestyle="--", label="clean MSE target", alpha=0.9,
            )
        ax.plot(recon[:, 0], color="tab:red", lw=0.8, label="reconstruction", alpha=0.9)
        ax.set_ylabel(_latex_safe_text("recon"))
    elif num_channels == 2:
        channel_specs = (
            (0, "I", "tab:green"),
            (1, "Q", "tab:blue"),
        )
        for row_offset, (channel, channel_name, channel_color) in enumerate(channel_specs):
            channel_ax = fig.add_subplot(gs[recon_row_start + row_offset])
            channel_ax.plot(
                x_example[:, channel], color="black", lw=0.7,
                label=f"{channel_name} noisy input", alpha=0.65,
            )
            if clean_target is not None:
                channel_ax.plot(
                    clean_target[:, channel], color=channel_color, lw=0.9,
                    linestyle="--", label=f"{channel_name} clean MSE target",
                    alpha=0.9,
                )
            channel_ax.plot(
                recon[:, channel], color="tab:red", lw=0.9,
                label=f"{channel_name} decoder output", alpha=0.95,
            )
            _add_signal_boundaries(channel_ax, signal_bounds)
            channel_ax.set_ylabel(_latex_safe_text(channel_name))
            channel_ax.legend(loc="upper right", fontsize=8)

        ax = fig.add_subplot(gs[recon_row_start + 2])
        amp_input = np.sqrt(np.sum(np.square(x_example[:, :2]), axis=1))
        amp_recon = np.sqrt(np.sum(np.square(recon[:, :2]), axis=1))
        ax.plot(
            amp_input, color="black", lw=0.7, label="amplitude noisy input",
            alpha=0.65,
        )
        if clean_target is not None:
            amp_target = np.sqrt(np.sum(np.square(clean_target[:, :2]), axis=1))
            ax.plot(
                amp_target, color="tab:green", lw=0.9, linestyle="--",
                label="amplitude derived from clean I/Q target", alpha=0.9,
            )
        ax.plot(
            amp_recon, color="tab:red", lw=0.9,
            label="amplitude derived from decoder I/Q", alpha=0.95,
        )
        ax.set_ylabel(_latex_safe_text("|I+iQ|"))
    else:
        ax = fig.add_subplot(gs[recon_row_start])
        for ch in range(num_channels):
            ax.plot(x_example[:, ch], lw=0.8, linestyle="--", label=f"ch{ch} input", alpha=0.8)
            ax.plot(recon[:, ch], lw=0.8, label=f"ch{ch} recon", alpha=0.95)
        ax.set_ylabel(_latex_safe_text("recon"))
    _add_signal_boundaries(ax, signal_bounds)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(_latex_safe_text("time step"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_logit_histogram(data, save_path: str) -> None:
    """Per-class distribution of the classifier logit."""
    y = data["y_all"] > 0.5
    l = data["logits_all"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(l[~y], bins=80, alpha=0.6, label="noise", color="tab:blue", log=True)
    if np.any(y):
        ax.hist(l[y], bins=80, alpha=0.6, label="signal", color="tab:red", log=True)
    ax.set_xlabel("classifier logit")
    ax.set_ylabel("count (log)")
    # KORREKTUR: Em-Dash entfernt, um LaTeX-Fehler zu vermeiden
    ax.set_title("Logit distribution - noise vs signal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------- #

def analyse_model(
    vae_model,
    dataset,
    output_dir: str,
    max_batches: int = 50,
    injection_snr: Optional[float] = None,
    window_metadata_path: Optional[str] = None,
    event_metadata_path: Optional[str] = None,
    metadata_split: str = "val",
    latent_mode: str = "z_mean",
) -> Dict[str, str]:
    """Run the whole analysis pipeline and return a dict of saved plot paths."""
    _ensure_dir(output_dir)
    if latent_mode not in VALID_ANALYSIS_LATENT_MODES:
        raise ValueError(
            f"latent_mode must be one of {VALID_ANALYSIS_LATENT_MODES}, "
            f"got '{latent_mode}'."
        )
    print(f"[analyse_model] Latent inference mode: {latent_mode}")
    if latent_mode == "sampled_z":
        print(
            "[analyse_model] sampled_z is stochastic; repeated analysis runs "
            "can produce different classifier scores and reconstructions."
        )
    data = _collect_examples(
        vae_model,
        dataset,
        max_batches=max_batches,
        latent_mode=latent_mode,
    )
    signal_bounds_by_window = _load_signal_bounds_by_window(
        window_metadata_path,
        event_metadata_path,
        metadata_split,
    )
    print(
        "[analyse_model] Loaded exact primary-signal bounds for "
        f"{len(signal_bounds_by_window)} {metadata_split} windows."
    )

    def signal_bounds_for(collected_idx: int) -> Optional[Tuple[int, int]]:
        window_idx = int(data["window_indices_all"][collected_idx])
        return signal_bounds_by_window.get(window_idx)
    print(
        "[analyse_model] Collected "
        f"{data['num_signal_windows']} signal windows and "
        f"{data['num_noise_windows']} noise windows for plotting."
    )
    print(
        "[analyse_model] Selected "
        f"{len(data['lowest_signal_idxs'])} lowest-logit signal windows "
        "in addition to the highest-logit signal window."
    )

    for filename in os.listdir(output_dir):
        is_old_weak_tp = filename.startswith("composite_weak_tp_")
        is_lowest_signal = filename.startswith("composite_lowest_logit_signal_")
        if (is_old_weak_tp or is_lowest_signal) and filename.endswith(".png"):
            os.remove(os.path.join(output_dir, filename))

    paths = {}
    paths["latent_scatter"] = os.path.join(output_dir, "latent_scatter.png")
    plot_latent_scatter(data, paths["latent_scatter"])

    paths["latent_dim_stats"] = os.path.join(output_dir, "latent_dim_stats.png")
    plot_latent_dim_stats(data, paths["latent_dim_stats"])

    paths["logit_histogram"] = os.path.join(output_dir, "logit_histogram.png")
    plot_logit_histogram(data, paths["logit_histogram"])

    if data["sig_idx"] is not None:
        paths["composite_signal"] = os.path.join(output_dir, "composite_signal.png")
        plot_activations_for_example(
            vae_model,
            data["x_all"][data["sig_idx"]],
            "Signal (highest classifier logit)",
            paths["composite_signal"],
            clean_target=(
                data["clean_all"][data["sig_idx"]]
                if data["clean_all"] is not None else None
            ),
            snr=injection_snr,
            signal_bounds=signal_bounds_for(data["sig_idx"]),
            latent_mode=latent_mode,
        )

    for rank, signal_idx in enumerate(data["lowest_signal_idxs"], start=1):
        path_key = f"composite_lowest_logit_signal_{rank}"
        paths[path_key] = os.path.join(
            output_dir,
            f"composite_lowest_logit_signal_{rank}.png",
        )
        plot_activations_for_example(
            vae_model,
            data["x_all"][signal_idx],
            f"Signal (lowest classifier logit #{rank})",
            paths[path_key],
            clean_target=(
                data["clean_all"][signal_idx]
                if data["clean_all"] is not None else None
            ),
            snr=injection_snr,
            signal_bounds=signal_bounds_for(signal_idx),
            latent_mode=latent_mode,
        )

    if data["noise_idx"] is not None:
        paths["composite_noise"] = os.path.join(output_dir, "composite_noise.png")
        plot_activations_for_example(
            vae_model,
            data["x_all"][data["noise_idx"]],
            "Noise (most confident TN)",
            paths["composite_noise"],
            clean_target=(
                data["clean_all"][data["noise_idx"]]
                if data["clean_all"] is not None else None
            ),
            snr=None,  # pure noise — no injected signal
            latent_mode=latent_mode,
        )

    return paths
