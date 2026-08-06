
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

# --- CONFIGURABLE CONSTANTS ---
DETECTION_LOGIT_THRESHOLD = -0.6

# --- IMPORTS FROM YOUR PROJECT ---
try:
    from data_pre_processing.pre_processing_incomplete_backup import pre_processing_with_memmap
except ImportError:
    # Fallback if running in a different structure
    from pre_processing_incomplete_backup import pre_processing_with_memmap
    
from vae.model import Sampling, VAEClassifier, QuadratureConv1D

CUSTOM_OBJECTS = {
    "Sampling": Sampling,
    "VAEClassifier": VAEClassifier,
    "QuadratureConv1D": QuadratureConv1D,
}


def _resolve_model_checkpoint_path(model_path):
    """
    Resolve a user-provided model path to an actual loadable checkpoint file.

    Supported inputs:
    - direct .keras / .h5 file
    - run directory containing checkpoints/best.keras
    - checkpoint directory containing best.keras
    """
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
        "Could not resolve a loadable model checkpoint from "
        f"'{model_path}'. Expected a .keras file or a directory containing "
        "'checkpoints/best.keras'."
    )


def _infer_run_dir_from_checkpoint(model_file_path):
    """Infer the training run directory from a resolved checkpoint file path."""
    checkpoint_dir = os.path.dirname(model_file_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def _load_saved_run_config(run_dir):
    """Load the saved training config.json if it exists."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as handle:
        return json.load(handle)


def _resolve_stats_dir(saved_cfg, default_memmap_dir="./memmaps"):
    """
    Resolve the directory containing saved training stats.

    In this project `stats_dir` defaults to `memmap_dir`, both usually rooted
    at the VAE project directory rather than the training run directory.
    """
    script_dir = _PROJECT_ROOT  # patched by reorganize.py: anchor to project root
    stats_dir = saved_cfg.get("stats_dir")
    if stats_dir in (None, "", "null"):
        stats_dir = saved_cfg.get("memmap_dir", default_memmap_dir)
    if not os.path.isabs(stats_dir):
        stats_dir = os.path.abspath(os.path.join(script_dir, stats_dir))
    return stats_dir


def _load_saved_normalization_stats(saved_cfg):
    """Load GLOBAL training normalization stats if they exist on disk."""
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

def plot_cnn_kernel_weights(model, save_path='Model_insights/'):
    """
    Extracts and plots weights for the first two CNN layers.
    Layer 1 is plotted as 1D wavelets; Layer 2 as 2D feature-integration heatmaps.
    """
    print("--- Generating Kernel Weight Plots ---")
    
    # NEU: Wir greifen auf den Encoder zu, falls es sich um ein VAE handelt
    target_model = model.encoder if hasattr(model, 'encoder') else model
    
    quadrature_layers = [l for l in target_model.layers if isinstance(l, QuadratureConv1D)]
    conv_layers = [l for l in target_model.layers if isinstance(l, tf.keras.layers.Conv1D)]
    
    if len(conv_layers) < 1 and not quadrature_layers:
        print("No Conv1D layers found.")
        return

    os.makedirs(save_path, exist_ok=True)

    if quadrature_layers:
        quad_layer = quadrature_layers[0]
        weights = quad_layer.get_weights()
        kernel_re, kernel_im = weights[0], weights[1]
        num_filters = min(kernel_re.shape[2], 16)

        plt.figure(figsize=(15, 8))
        for i in range(num_filters):
            plt.subplot(4, 4, i + 1)
            plt.plot(kernel_re[:, 0, i], color='tab:green', lw=1.3, label='real')
            plt.plot(kernel_im[:, 0, i], color='tab:blue', lw=1.3, label='imag')
            plt.title(f"Quad Filter {i}", fontsize=9)
            plt.xticks([])
            plt.grid(alpha=0.3)
            if i == 0:
                plt.legend(fontsize=7)
        plt.suptitle("Quadrature Front-End Kernels (I/Q Correlation Templates)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_path}quadrature_weights_l1.png')
        plt.close()

    if len(conv_layers) < 1:
        print("No standard Conv1D layers found after the quadrature front-end.")
        return

    # --- LAYER 1: Raw Signal Filters --- #
    weights_1 = conv_layers[0].get_weights()[0]
    num_f1 = min(weights_1.shape[2], 16) # Plot up to 16
    
    plt.figure(figsize=(15, 8))
    for i in range(num_f1):
        plt.subplot(4, 4, i + 1)
        plt.plot(weights_1[:, 0, i], color='tab:blue', lw=1.5)
        plt.title(f"Filter {i}", fontsize=9)
        plt.xticks([]); plt.grid(alpha=0.3)
    plt.suptitle("Layer 1 CNN Kernels (Raw Signal Templates)", fontsize=14)
    plt.tight_layout(); plt.savefig(f'{save_path}cnn_weights_l1.png')

    if len(conv_layers) < 2:
        print("Only one Conv1D layer found – skipping Layer 2 weight visualization.")
        return

    # --- LAYER 2: Feature Integration Map --- #
    weights_2 = conv_layers[1].get_weights()[0]
    num_f2 = min(weights_2.shape[2], 16)
    
    plt.figure(figsize=(15, 8))
    for i in range(num_f2):
        plt.subplot(4, 4, i + 1)
        # Transpose to show (Previous Filters vs Time)
        plt.imshow(weights_2[:, :, i].T, aspect='auto', cmap='RdBu_r')
        plt.title(f"Filter {i}", fontsize=9)
        plt.axis('off')
    plt.suptitle("Layer 2 CNN Weights (Integration of L1 Features)", fontsize=14)
    plt.tight_layout(); plt.savefig(f'{save_path}cnn_weights_l2.png')
    print(f"CNN Weight plots saved to {save_path}")

def run_full_visualization_insight_pipeline(model_path, data_config):
    resolved_model_path = _resolve_model_checkpoint_path(model_path)
    run_dir = _infer_run_dir_from_checkpoint(resolved_model_path)
    saved_cfg = _load_saved_run_config(run_dir)
    saved_mean, saved_std, saved_std_mag = _load_saved_normalization_stats(saved_cfg)

    # Keep the insight preprocessing aligned with the trained model so the
    # channel count and window geometry do not silently mismatch.
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
        if key in saved_cfg and data_config.get(key) != saved_cfg[key]:
            print(
                f"[CONFIG] Overriding data_config['{key}'] from "
                f"{data_config.get(key)} to {saved_cfg[key]} based on saved run config."
            )
            data_config[key] = saved_cfg[key]

    # Use the saved GLOBAL training stats by default so the insight inputs live
    # on the same scale as the checkpoint saw during training.
    if saved_mean is not None and saved_std is not None:
        if data_config.get("global_mean_input") != saved_mean:
            print(
                f"[CONFIG] Overriding data_config['global_mean_input'] from "
                f"{data_config.get('global_mean_input')} to {saved_mean} based on saved stats."
            )
        if data_config.get("global_std_input") != saved_std:
            print(
                f"[CONFIG] Overriding data_config['global_std_input'] from "
                f"{data_config.get('global_std_input')} to {saved_std} based on saved stats."
            )
        data_config["global_mean_input"] = saved_mean
        data_config["global_std_input"] = saved_std
        data_config["calculate_stats"] = False

    if saved_std_mag is not None:
        if data_config.get("custom_noise_std") != saved_std_mag:
            print(
                f"[CONFIG] Overriding data_config['custom_noise_std'] from "
                f"{data_config.get('custom_noise_std')} to {saved_std_mag} based on saved stats."
            )
        data_config["custom_noise_std"] = saved_std_mag

    # 1. Load Data (Conditioning Loop)
    print("--- Loading Dataset (conditioning on ≥1 signal window) ---")

    max_resample_tries = 50
    val_ds = None
    
    for attempt in range(max_resample_tries):
        _, val_ds, _ = pre_processing_with_memmap(**data_config)

        # Check whether at least one signal window exists
        y_true_check = []
        for _, y_batch in val_ds:
            y_true_check.append(y_batch.numpy().reshape(-1))

        if len(y_true_check) == 0: continue

        y_true_check = np.concatenate(y_true_check)
        n_signal_windows = int(np.sum(y_true_check > 0.5))

        if n_signal_windows > 0:
            print(f"[CONDITIONING] Attempt {attempt+1}: Found {n_signal_windows} signal windows.")
            break
    else:
        raise RuntimeError("Failed to generate a dataset with signal windows after 50 attempts.")

    # ---------------------------------------------------------
    # 2. GLOBAL LOGIT CALCULATION
    # ---------------------------------------------------------
    print("--- Computing global logits (Validation Set) ---")

    model = tf.keras.models.load_model(
        resolved_model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    print(f"[INFO] Loaded model from {resolved_model_path}")
    
    all_logits = []
    all_labels = []

    # First pass: Predict everything to get statistics
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        all_logits.append(preds.reshape(-1))
        all_labels.append(y_batch.numpy().reshape(-1))

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    # ---------------------------------------------------------
    # 3. STATS & HISTOGRAM
    # ---------------------------------------------------------
    max_logit_this_run = float(np.max(all_logits))
    print(f"[INSIGHT] Max logit this run = {max_logit_this_run:.4f}")

    os.makedirs("Model_insights", exist_ok=True)
    plt.figure(figsize=(10, 6))
    signal_mask = all_labels > 0.5
    noise_mask = ~signal_mask

    plt.hist(all_logits[noise_mask], bins=100, alpha=0.7, label="Noise", color="tab:blue", log=True)
    plt.hist(all_logits[signal_mask], bins=100, alpha=0.7, label="Signal", color="tab:green", log=True)
    plt.axvline(DETECTION_LOGIT_THRESHOLD, color="red", linestyle="--", label="Threshold")

    plt.xlabel("Logit output"); plt.ylabel("Counts (log)"); plt.legend()
    plt.savefig("Model_insights/validation_logit_histogram.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. IDENTIFY & FETCH SPECIFIC WINDOWS (Strongest Signal)
    # ---------------------------------------------------------
    print("--- Identifying Strongest Signal Window ---")
    
    target_indices = {} # map: global_index -> description_label

    # A. Find Strongest Signal
    if np.any(signal_mask):
        temp_logits = all_logits.copy()
        temp_logits[~signal_mask] = -np.inf # Mask out noise
        idx_max = np.argmax(temp_logits)
        
        if temp_logits[idx_max] >= DETECTION_LOGIT_THRESHOLD:
            target_indices[idx_max] = "Signal (Strongest Detected)"
            print(f"Selection: Strongest Detected Signal at index {idx_max} (Logit={temp_logits[idx_max]:.3f})")
        else:
            target_indices[idx_max] = "Signal (Best Missed)"
            print(f"Selection: No detection. Showing best Missed Signal at index {idx_max} (Logit={temp_logits[idx_max]:.3f})")
    
    # B. Find a Noise Window (Just pick the first clean noise for comparison)
    if np.any(noise_mask):
        idx_noise = np.where(noise_mask)[0][0]
        target_indices[idx_noise] = "Noise (Reference)"

    # C. Retrieve the actual windows by re-iterating (Deterministic because fixed seed)
    examples = {}
    current_idx = 0
    found_count = 0
    total_targets = len(target_indices)

    for x_batch, _ in val_ds:
        if found_count >= total_targets: break
        
        batch_len = x_batch.shape[0]
        batch_start = current_idx
        batch_end = current_idx + batch_len

        # Check if targets are in this batch
        for target_idx, label in list(target_indices.items()):
            if batch_start <= target_idx < batch_end:
                local_offset = target_idx - batch_start
                win = x_batch.numpy()[local_offset:local_offset+1]
                examples[label] = win
                found_count += 1
        
        current_idx += batch_len

    # ---------------------------------------------------------
    # 5. MODEL RECONSTRUCTION (For Visualization)
    # ---------------------------------------------------------
    plot_cnn_kernel_weights(model)

    print("--- Building Visualization Model ---")
    encoder_target_layers = []
    encoder_layer_info = []

    target_model = model.encoder if hasattr(model, 'encoder') else model

    for layer in target_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer): continue
        
        # Unwrap wrappers
        base_layer = layer
        if isinstance(layer, tf.keras.layers.TimeDistributed): base_layer = layer.layer
        if isinstance(layer, tf.keras.layers.Bidirectional): base_layer = layer.forward_layer

        # Collect relevant layers, including the optional quadrature front-end.
        if isinstance(base_layer, (QuadratureConv1D, tf.keras.layers.Conv1D, tf.keras.layers.LSTM, tf.keras.layers.Dense)):
            encoder_target_layers.append(layer.output)
            if isinstance(base_layer, QuadratureConv1D): lname, units = "QuadratureConv1D", base_layer.filters
            elif isinstance(base_layer, tf.keras.layers.Conv1D): lname, units = "Conv1D", base_layer.filters
            elif isinstance(base_layer, tf.keras.layers.LSTM): lname, units = "LSTM", base_layer.units
            elif isinstance(base_layer, tf.keras.layers.Dense): lname, units = "Dense", base_layer.units
            else: lname, units = base_layer.__class__.__name__, "Pooled"
            
            encoder_layer_info.append({"name": lname, "units": units, "rank": len(layer.output.shape)})

    enc_vis_model = tf.keras.models.Model(
        inputs=target_model.input, outputs=encoder_target_layers
    )

    decoder_target_layers = []
    decoder_layer_info = []
    if hasattr(model, "decoder"):
        for layer in model.decoder.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if isinstance(layer, (tf.keras.layers.Conv1DTranspose, tf.keras.layers.Conv1D, tf.keras.layers.Dense)):
                decoder_target_layers.append(layer.output)
                if isinstance(layer, tf.keras.layers.Conv1DTranspose):
                    lname, units = "Conv1DTranspose", layer.filters
                elif isinstance(layer, tf.keras.layers.Conv1D):
                    lname, units = "Conv1D", layer.filters
                elif isinstance(layer, tf.keras.layers.Dense):
                    lname, units = "Dense", layer.units
                else:
                    lname, units = layer.__class__.__name__, "Pooled"
                decoder_layer_info.append(
                    {"name": lname, "units": units, "rank": len(layer.output.shape)}
                )
        dec_vis_model = tf.keras.models.Model(
            inputs=model.decoder.input, outputs=decoder_target_layers
        )
    else:
        dec_vis_model = None

    # ---------------------------------------------------------
    # 6. PLOT ACTIVATIONS
    # ---------------------------------------------------------
    for label_name, inp in examples.items():
        print(f"Plotting activations for: {label_name}")
        encoder_activations = enc_vis_model.predict(inp, verbose=0)
        if not isinstance(encoder_activations, list):
            encoder_activations = [encoder_activations]

        z_mean, z_log_var = model.encoder(inp, training=False)
        cls_features = model.classifier_features(z_mean, z_log_var)
        logit_val = float(model.classifier(cls_features, training=False).numpy()[0, 0])
        reconstruction = model.decoder(z_mean, training=False).numpy()[0]

        decoder_activations = []
        if dec_vis_model is not None:
            decoder_activations = dec_vis_model.predict(z_mean, verbose=0)
            if not isinstance(decoder_activations, list):
                decoder_activations = [decoder_activations]

        num_encoder_stages = len(encoder_activations)
        num_decoder_stages = len(decoder_activations)
        has_iq = (inp.shape[-1] == 2)
        total_rows = 1 + num_encoder_stages + 1 + num_decoder_stages + 1 + (1 if has_iq else 0)
        fig = plt.figure(figsize=(14, 3 * total_rows))
        gs = fig.add_gridspec(total_rows, 1)
        
        # Plot Input
        ax0 = fig.add_subplot(gs[0])
        if inp.shape[-1] == 1:
            ax0.plot(inp[0, :, 0], color='black', alpha=0.8)
            ax0.set_ylabel("Amplitude")
        elif inp.shape[-1] == 2:
            ax0.plot(inp[0, :, 0], color='tab:green', alpha=0.85, label='I')
            ax0.plot(inp[0, :, 1], color='tab:blue', alpha=0.85, label='Q')
            ax0.set_ylabel("I / Q")
            ax0.legend(loc="upper right", fontsize=8)
        else:
            for ch in range(inp.shape[-1]):
                ax0.plot(inp[0, :, ch], alpha=0.8, label=f'ch{ch}')
            ax0.set_ylabel("Input")
            ax0.legend(loc="upper right", fontsize=8)
        ax0.set_title(f"{label_name} | Logit: {logit_val:.4f}", fontweight='bold')

        # Plot encoder stages
        for i in range(num_encoder_stages):
            ax = fig.add_subplot(gs[i + 1])
            act = encoder_activations[i]
            info = encoder_layer_info[i]
            
            if info["rank"] == 3:
                im = ax.imshow(act[0].T, aspect="auto", cmap="magma" if info["name"] == "Conv1D" else "viridis")
                ax.set_ylabel(f"Filter / Unit")
            else:
                im = ax.imshow(act, aspect="auto", cmap="plasma")
                ax.set_yticks([])
                ax.set_xlabel(f"Neuron Index (0–{act.shape[1]-1})")

            ax.set_title(f"Encoder Layer {i+1}: {info['name']} ({info['units']} units)")
            fig.colorbar(im, ax=ax, label="Activity")

        # Plot latent statistics
        latent_row = 1 + num_encoder_stages
        ax_latent = fig.add_subplot(gs[latent_row])
        latent_x = np.arange(z_mean.shape[1])
        ax_latent.bar(latent_x - 0.2, z_mean.numpy()[0], width=0.4, color='tab:purple', label='z_mean')
        ax_latent.bar(latent_x + 0.2, z_log_var.numpy()[0], width=0.4, color='tab:orange', label='z_log_var')
        ax_latent.set_ylabel("Latent")
        ax_latent.set_title("Latent statistics used by the classifier head")
        ax_latent.legend(loc="upper right", fontsize=8)

        # Plot decoder stages
        decoder_start_row = latent_row + 1
        for i in range(num_decoder_stages):
            ax = fig.add_subplot(gs[decoder_start_row + i])
            act = decoder_activations[i]
            info = decoder_layer_info[i]

            if info["rank"] == 3:
                im = ax.imshow(act[0].T, aspect="auto", cmap="cividis")
                ax.set_ylabel("Filter / Unit")
            else:
                im = ax.imshow(act, aspect="auto", cmap="plasma")
                ax.set_yticks([])
                ax.set_xlabel(f"Neuron Index (0–{act.shape[1]-1})")

            ax.set_title(f"Decoder Layer {i+1}: {info['name']} ({info['units']} units)")
            fig.colorbar(im, ax=ax, label="Activity")

        # Plot reconstruction
        recon_row = decoder_start_row + num_decoder_stages
        ax_recon = fig.add_subplot(gs[recon_row])
        if inp.shape[-1] == 1:
            ax_recon.plot(inp[0, :, 0], color='black', alpha=0.75, label='input')
            ax_recon.plot(reconstruction[:, 0], color='tab:red', alpha=0.85, label='reconstruction')
            ax_recon.set_ylabel("Amplitude")
        else:
            ax_recon.plot(inp[0, :, 0], color='tab:green', alpha=0.35, linestyle='--', label='I input')
            ax_recon.plot(reconstruction[:, 0], color='tab:green', alpha=0.95, label='I recon')
            ax_recon.plot(inp[0, :, 1], color='tab:blue', alpha=0.35, linestyle='--', label='Q input')
            ax_recon.plot(reconstruction[:, 1], color='tab:blue', alpha=0.95, label='Q recon')
            ax_recon.set_ylabel("I / Q")
        ax_recon.set_title("Decoder reconstruction")
        ax_recon.legend(loc="upper right", fontsize=8)

        if has_iq:
            ax_mag = fig.add_subplot(gs[recon_row + 1])
            input_mag = np.sqrt(inp[0, :, 0]**2 + inp[0, :, 1]**2)
            recon_mag = np.sqrt(reconstruction[:, 0]**2 + reconstruction[:, 1]**2)
            ax_mag.plot(input_mag, color='black', alpha=0.5, linestyle='--', label='|input|')
            ax_mag.plot(recon_mag, color='tab:red', alpha=0.9, label='|reconstruction|')
            ax_mag.set_ylabel("|x|")
            ax_mag.set_title("Derived complex magnitude")
            ax_mag.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        safe_label = label_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(f'Model_insights/insight_{safe_label}.png', dpi=300)
        plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    M_solar = 1.988e30

    # ============================================================== #
    #                  INSIGHT CONFIGURATION                         #
    #  Edit these values to control what the insight script runs on. #
    # ============================================================== #

    # --- Trained model path (anchored to the project root) ---
    model_file_path = os.path.join(_PROJECT_ROOT, 'TrainedModels', '1e_minus8_Trained', 'Model_1', 'vae_cls_3conv_16_32_64_ld16', 'checkpoints', 'best.keras') # 'runs/vae_cls_3conv_16_32_64_ld16_complex_quadrature_frontend_real_imag'

    # --- Data files to analyse (use files NOT seen during training) ---
    INSIGHT_FILE_SUFFIXES = ['19.23.28.791']

    # --- PBH masses to inject (in solar masses — converted to kg below) ---
    INSIGHT_PBH_MASSES_SOLAR = [1e-13]

    # --- Target SNR values for injected signals ---
    INSIGHT_SNR_VALUES = [1.0]

    # --- How many signals to inject into the val split ---
    INSIGHT_NUM_SIGNALS_VAL = 1

    # --- How many raw samples to read per file ---
    INSIGHT_NUM_SAMPLES = 300_000

    # --- Precomputed normalisation stats from training ---
    INSIGHT_GLOBAL_MEAN = 5.128e-5 # 4.112e-9
    INSIGHT_GLOBAL_STD  = 2.7052e-5 #4.1293e-5 #
    INSIGHT_CUSTOM_NOISE_STD = 2.705e-5   # for SNR-based amplitude scaling

    # ============================================================== #
    #                  END OF CONFIGURATION                          #
    # ============================================================== #

    data_config = {
        'filepath_suffixes': INSIGHT_FILE_SUFFIXES,
        'filepath_template': os.path.join(_PROJECT_ROOT, 'GravNet', 'Data', 'IQDataFile-2024.04.18.{}.tiq'),
        'num_samples_to_read_per_file': INSIGHT_NUM_SAMPLES,
        'offset': 0,
        'window_size': 1024,
        'step_size': 1024 // 10,
        'use_I_Q': False,
        'use_amps': True,
        'inject_signals': True,
        'signal_injection_probability': 1.0,
        'num_signals_to_inject_per_segment': {'train': 0, 'val': INSIGHT_NUM_SIGNALS_VAL, 'test': 0},
        'train_ratio': 0.01,
        'val_ratio': 0.98,
        'test_ratio': 0.01,
        'dtype': np.float32,
        'normalization_type': 'zscore',
        'global_mean_input': INSIGHT_GLOBAL_MEAN,
        'global_std_input': INSIGHT_GLOBAL_STD,
        'calculate_stats': False,
        'custom_noise_std': INSIGHT_CUSTOM_NOISE_STD,
        'snr_based_injection': True,
        'm_PBH_injection_list': [m * M_solar for m in INSIGHT_PBH_MASSES_SOLAR],
        'amplitude_spectrum_range': INSIGHT_SNR_VALUES,
        'return_tf_datasets': True,
        'tf_batch_size': 512,
        'tf_shuffle': False,
        'tf_repeat': False
    }

    run_full_visualization_insight_pipeline(model_file_path, data_config)
