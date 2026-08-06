
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

# ----- Efficiency curven function logic ------ #
# 1. Load the model, the name of the model should be an input value of the function

# 2. For a given model load data using the "pre_processing_incomplete.py" function script for different files than the training, but then we need the normalization parameters of the training to be input to this function here!

# 3. We need a small calibration set where we will calibrate the threshold that we will be using for the sigmoid output of the classifier at the end of a model, i.e. a threshold value between 0 and 1 where I want one where I won't have any False Positive Rate (FPR). This is bestly doen by first computing what threshold is needed for less than 1 False Positive /year. I think this can be done by starting with some small threshold and going up in with some step size and then fitting a suitable function to the data and extrapolate to find the threshold for less than 1 FP/year

# 4. Use the found threshold value and with it compute the efficiency curves for a Primordial Black Hole (PBH) mass range from including 10^-13 solar masses upt to including 10^-6 solar masses using the signal injection in the "pre_processing_incomplete.py" function. Here we go from a certain SNR like 8 to an SNR like 1, where SNR = Signal-to-Noise Ratio. Here the SNR values should be also an input to this fucntion here.
#   IMPORTANTLY: Signals can generally span over multiple windows. I do a event based detection, which is also implemented as a custom callback in the script "Event_Metrics_Callback.py". This means once given the set threshold one window is detected a a signal and this is true, i.e. we check against a flagged boolean list, coming from the script "pre_processing_incomplete.py", then the whole signal is marked as detected!

# 5. Give the option to run the script for multiple runs per SNR per mass, i.e. also specify how often, to potentially assess the stability of the model

# 6. Fit a sigmoid or S-curve function to the data and use Wilson 95 intervals for the errors of the different efficiency as a function of SNR. Here we should end up with saved plots in a predefined directory (also a function input here) of efficiency on the y-axis and SNR value on the x-axis. Wilson 95 intervals because we have a binomial distribution of "success: having detected" a signal or "failure: having not detected a signal". The number of signal we inject is also a fucntion input here. The values of the s-curve, i.e. the found parameters as well ass the data point should be saved in a .csv file for potential later usage.

# 7. When having this curve and with the errors of the data points, we find a pre-defined point with the given function parameters where we have a predefined efficiency (also an input to this function here). E.g. we set the efficiency to 95% and wanna find what SNR can we detect for this efficiency. Also always include what FP/year we will have with the set efficiency and SNR for all masses. The values should be saved in a .csv file

# 8. The final result should be a combiantions of all the found SNR value we need to detect a given signal coming for a PBH of mass m_PBH. This means the final result should be a plot where m_PBH is on the x-axis with a log scale and the SNR will be on the y-axis with a linear scale. The values of the plot should be saved in a .csv file
    
    
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import os
import json
import shutil
import math
import logging
import gc

# Ensure we use the science style if available
try:
    plt.style.use(['science', 'no-latex'])
except:
    pass

# --- Local Imports ---
from data_pre_processing.pre_processing_incomplete import pre_processing_with_memmap

from vae.model import Sampling, VAEClassifier, QuadratureConv1D
from vae.clustered_fp import (
    compute_clustered_fp_diagnostics_multi,
    append_clustered_fp_row,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CUSTOM_OBJECTS = {
    "Sampling": Sampling,
    "VAEClassifier": VAEClassifier,
    "QuadratureConv1D": QuadratureConv1D,
}


def _resolve_model_checkpoint_path(model_path):
    """
    Resolve a user-provided model path to a concrete loadable checkpoint file.

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
    """Load a saved training config.json if it exists."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as handle:
        return json.load(handle)


def _resolve_stats_dir(saved_cfg, default_memmap_dir="./memmaps"):
    """
    Resolve the directory containing saved training stats.

    In this project stats_dir defaults to memmap_dir and both are usually
    relative to the VAE project directory rather than the run directory.
    """
    script_dir = _PROJECT_ROOT  # patched by reorganize.py: anchor to project root
    stats_dir = saved_cfg.get("stats_dir")
    if stats_dir in (None, "", "null"):
        stats_dir = saved_cfg.get("memmap_dir", default_memmap_dir)
    if not os.path.isabs(stats_dir):
        stats_dir = os.path.abspath(os.path.join(script_dir, stats_dir))
    return stats_dir


def _load_saved_normalization_stats(saved_cfg):
    """Load GLOBAL training mean/std stats if they exist on disk."""
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


def _align_model_and_preprocessing(
    model_path,
    preprocessing_config,
    normalization_params,
    normalization_mode,
):
    """
    Align preprocessing with the saved training run of the checkpoint.

    This prevents silent mismatches between the loaded model and the manual
    settings at the bottom of the script.
    """
    resolved_model_path = _resolve_model_checkpoint_path(model_path)
    run_dir = _infer_run_dir_from_checkpoint(resolved_model_path)
    saved_cfg = _load_saved_run_config(run_dir)

    preprocessing_config = preprocessing_config.copy()
    normalization_params = dict(normalization_params or {})

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
        if key in saved_cfg and preprocessing_config.get(key) != saved_cfg[key]:
            logger.info(
                "[CONFIG] Overriding preprocessing_config['%s'] from %s to %s "
                "based on saved run config.",
                key,
                preprocessing_config.get(key),
                saved_cfg[key],
            )
            preprocessing_config[key] = saved_cfg[key]

    saved_mean, saved_std, saved_std_mag = _load_saved_normalization_stats(saved_cfg)

    if normalization_mode == "zscore" and saved_mean is not None and saved_std is not None:
        manual_mean = normalization_params.get("mean_value")
        manual_std = normalization_params.get("std_dev_value")
        if manual_mean is None and manual_std is None:
            logger.info(
                "[CONFIG] Using saved training z-score stats: mean=%s, std=%s.",
                saved_mean,
                saved_std,
            )
            normalization_params["mean_value"] = saved_mean
            normalization_params["std_dev_value"] = saved_std
            preprocessing_config["global_mean_input"] = saved_mean
            preprocessing_config["global_std_input"] = saved_std
            preprocessing_config["calculate_stats"] = False
        else:
            logger.info(
                "[CONFIG] Keeping user-provided z-score stats (mean=%s, std=%s) "
                "instead of auto-loading saved stats (mean=%s, std=%s).",
                manual_mean,
                manual_std,
                saved_mean,
                saved_std,
            )

    if saved_std_mag is not None and preprocessing_config.get("use_I_Q"):
        if preprocessing_config.get("custom_noise_std") is None:
            logger.info(
                "[CONFIG] Using saved magnitude noise std for I/Q SNR scaling: %s.",
                saved_std_mag,
            )
            preprocessing_config["custom_noise_std"] = saved_std_mag
        else:
            logger.info(
                "[CONFIG] Keeping user-provided custom_noise_std=%s instead of "
                "auto-loading saved magnitude std=%s.",
                preprocessing_config.get("custom_noise_std"),
                saved_std_mag,
            )

    return resolved_model_path, preprocessing_config, normalization_params, saved_cfg

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------

def wilson_score_interval(successes, total, confidence=0.95):
    """ Wilson Score Interval for binomial proportion. """
    if total == 0: return 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

def linear_log_model(t, b, c):
    """ Model: ln(FPR) = b*t + c (Linear fit in Log-Space) """
    return b * t + c

# --- Survivor Functions for FPR fitting ---

def sigmoid_func(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))
    
def lomax_tail_model(t, lam, alpha, t_start, r_start):
    """
    Lomax (Pareto Type II) tail model: 
    FPR = r_start * (1 + (t - t_start) / lam)**(-alpha)
    """
    return r_start * (1 + (t - t_start) / lam)**(-alpha)


def _predict_logits(model, x_batch, classifier_inference_mode="sampled_z"):
    """
    Return flat classifier logits using an explicit latent inference rule.

    ``sampled_z`` is the efficiency-curve default because the shared-sample
    classifier was trained on posterior samples. ``z_mean`` remains available
    as a deterministic diagnostic. For classifiers built on
    concat[z_mean, z_log_var], the head remains deterministic because it does
    not accept a sampled z.
    """
    mode = str(classifier_inference_mode).strip().lower()
    if mode not in {"z_mean", "sampled_z"}:
        raise ValueError(
            "classifier_inference_mode must be 'z_mean' or 'sampled_z', "
            f"got '{classifier_inference_mode}'."
        )
    z_mean, z_log_var = model.encoder(x_batch, training=False)
    inference_z = (
        z_mean if mode == "z_mean"
        else model.sampling([z_mean, z_log_var])
    )
    features = model.classifier_features(z_mean, z_log_var, z=inference_z)
    logits = model.classifier(features, training=False)
    return np.asarray(logits).reshape(-1)


def _latent_outputs(model, x_batch):
    """Return z_mean and z_log_var from the VAE encoder."""
    z_mean, z_log_var = model.encoder(x_batch, training=False)
    return np.asarray(z_mean), np.asarray(z_log_var)


def _get_model_window_size(model):
    """Support both direct model.input_shape and encoder-backed models."""
    if getattr(model, "input_shape", None) is not None:
        return model.input_shape[1]
    if hasattr(model, "encoder") and getattr(model.encoder, "input_shape", None) is not None:
        return model.encoder.input_shape[1]
    return None


def _normalise_detection_score_mode(detection_score_mode):
    """Normalise detector score aliases to a supported mode."""
    mode = (detection_score_mode or "logit").strip().lower()
    aliases = {
        "classifier": "logit",
        "classifier_logit": "logit",
        "logits": "logit",
        "mlp": "logit",
        "mlp_logit": "logit",
        "latent_dim": "latent_dim_abs_zscore",
        "latent_z": "latent_dim_abs_zscore",
        "latent_selected": "latent_selected_zscore",
        "latent_l2": "latent_selected_zscore",
        "latent_kl": "latent_total_kl",
        "kl": "latent_total_kl",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {
        "logit",
        "latent_dim_abs_zscore",
        "latent_selected_zscore",
        "latent_total_kl",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported detection_score_mode='{detection_score_mode}'. "
            f"Choose from {sorted(valid_modes)}."
        )
    return mode


def _score_mode_needs_latents(detection_score_mode):
    """Return True if the chosen detector score depends on latent outputs."""
    return _normalise_detection_score_mode(detection_score_mode) != "logit"


def _score_mode_needs_noise_reference(detection_score_mode):
    """Return True if the chosen detector score needs noise reference stats."""
    mode = _normalise_detection_score_mode(detection_score_mode)
    return mode in {"latent_dim_abs_zscore", "latent_selected_zscore"}


def _resolve_latent_dims(detection_score_config, latent_dim_size, default_dim=0):
    """Resolve user-specified latent dimensions and validate the indices."""
    config = detection_score_config or {}
    dims = config.get("selected_latent_dims")
    if dims is None:
        dims = [config.get("latent_dim", default_dim)]
    elif np.isscalar(dims):
        dims = [dims]
    dims = [int(dim) for dim in dims]

    for dim in dims:
        if dim < 0 or dim >= latent_dim_size:
            raise ValueError(
                f"Latent dimension {dim} is out of bounds for latent size {latent_dim_size}."
            )
    return dims


def _sample_kl_per_window(z_mean, z_log_var, dims=None):
    """Return total KL contribution per window, optionally restricted to dims."""
    kl_terms = 0.5 * (
        np.square(z_mean) + np.exp(z_log_var) - 1.0 - z_log_var
    )
    if dims is not None:
        kl_terms = kl_terms[:, dims]
    return np.sum(kl_terms, axis=1)


def _build_detection_score_context(
    z_mean_noise,
    z_log_var_noise,
    detection_score_mode="logit",
    detection_score_config=None,
):
    """
    Build reusable reference data for detector score modes.

    For latent z-score based modes we estimate the noise-only mean and std of
    each latent dimension once and then reuse them during threshold calibration
    and efficiency evaluation.
    """
    mode = _normalise_detection_score_mode(detection_score_mode)
    config = dict(detection_score_config or {})
    context = {"mode": mode, "config": config}

    latent_dim_size = None
    if z_mean_noise is not None:
        z_mean_noise = np.asarray(z_mean_noise)
        latent_dim_size = z_mean_noise.shape[1]

    if mode == "logit":
        return context

    if mode == "latent_total_kl":
        if latent_dim_size is not None:
            if "selected_latent_dims" in config:
                context["selected_latent_dims"] = _resolve_latent_dims(
                    config, latent_dim_size
                )
            else:
                context["selected_latent_dims"] = list(range(latent_dim_size))
        return context

    noise_mean = config.get("noise_mean")
    noise_std = config.get("noise_std")
    if z_mean_noise is None or z_mean_noise.size == 0:
        if noise_mean is None or noise_std is None:
            raise ValueError(
                f"detection_score_mode='{mode}' requires noise-only latent reference "
                "data or precomputed 'noise_mean' and 'noise_std' in detection_score_config."
            )
        noise_mean = np.asarray(noise_mean, dtype=float)
        noise_std = np.asarray(noise_std, dtype=float)
        latent_dim_size = noise_mean.shape[0]
    else:
        noise_mean = np.mean(z_mean_noise, axis=0)
        noise_std = np.std(z_mean_noise, axis=0)
    noise_std = np.where(noise_std > 1e-8, noise_std, 1e-8)

    selected_dims = _resolve_latent_dims(config, latent_dim_size)
    context["noise_mean"] = noise_mean
    context["noise_std"] = noise_std
    context["selected_latent_dims"] = selected_dims
    context["latent_dim"] = int(config.get("latent_dim", selected_dims[0]))
    context["reduction"] = str(config.get("reduction", "l2")).lower()
    context["use_abs"] = bool(config.get("use_abs", True))
    return context


def _compute_detection_score(
    logits,
    z_mean,
    z_log_var,
    detection_score_mode="logit",
    score_context=None,
    detection_score_config=None,
):
    """Compute the per-window detector score used for thresholding."""
    mode = _normalise_detection_score_mode(detection_score_mode)
    config = dict(detection_score_config or {})

    if mode == "logit":
        if logits is None:
            raise ValueError("logit detection mode requires classifier logits.")
        return np.asarray(logits).reshape(-1)

    if z_mean is None or z_log_var is None:
        raise ValueError(
            f"detection_score_mode='{mode}' requires z_mean and z_log_var."
        )

    z_mean = np.asarray(z_mean)
    z_log_var = np.asarray(z_log_var)
    context = score_context or _build_detection_score_context(
        z_mean,
        z_log_var,
        detection_score_mode=mode,
        detection_score_config=config,
    )

    if mode == "latent_total_kl":
        dims = context.get("selected_latent_dims")
        return _sample_kl_per_window(z_mean, z_log_var, dims=dims)

    if mode == "latent_dim_abs_zscore":
        dim = int(context["latent_dim"])
        noise_mean = context["noise_mean"][dim]
        noise_std = context["noise_std"][dim]
        score = (z_mean[:, dim] - noise_mean) / noise_std
        if context.get("use_abs", True):
            score = np.abs(score)
        return score.reshape(-1)

    if mode == "latent_selected_zscore":
        dims = context["selected_latent_dims"]
        score_matrix = (
            (z_mean[:, dims] - context["noise_mean"][dims])
            / context["noise_std"][dims]
        )
        abs_score_matrix = np.abs(score_matrix)
        reduction = context.get("reduction", "l2")
        if reduction == "l2":
            return np.linalg.norm(abs_score_matrix, axis=1)
        if reduction == "mean_abs":
            return np.mean(abs_score_matrix, axis=1)
        if reduction == "max_abs":
            return np.max(abs_score_matrix, axis=1)
        raise ValueError(
            f"Unsupported latent_selected_zscore reduction='{reduction}'. "
            "Choose from ['l2', 'mean_abs', 'max_abs']."
        )

    raise ValueError(f"Unsupported detection score mode '{mode}'.")


def _get_detection_score_label(detection_score_mode, score_context=None):
    """Return a compact human-readable label for the current detector score."""
    mode = _normalise_detection_score_mode(detection_score_mode)
    context = score_context or {}
    if mode == "logit":
        return "classifier logit"
    if mode == "latent_total_kl":
        dims = context.get("selected_latent_dims")
        if dims is None:
            return "total latent KL"
        return f"latent KL sum over dims {dims}"
    if mode == "latent_dim_abs_zscore":
        dim = context.get("latent_dim", 0)
        return f"|z_mean[{dim}] - mu_noise| / sigma_noise"
    if mode == "latent_selected_zscore":
        dims = context.get("selected_latent_dims")
        reduction = context.get("reduction", "l2")
        return f"selected latent z-score ({reduction}, dims={dims})"
    return mode


def _build_calibration_prep_config(
    prep_config,
    calibration_files,
    temp_dir,
    num_samples_threshold_calibration=None,
):
    """Build the preprocessing config used for noise-only threshold calibration."""
    calib_config = prep_config.copy()
    if num_samples_threshold_calibration is not None:
        calib_config['num_samples_to_read_per_file'] = num_samples_threshold_calibration

    calib_config.pop('num_samples_to_read_per_file_threshold_calibration', None)
    if calibration_files:
        calib_config['filepath_suffixes'] = calibration_files
    calib_config.pop('test_file_suffixes', None)
    calib_config.update({
        'inject_signals': False,
        'memmap_dir': temp_dir,
        'calculate_stats': False,
        'return_tf_datasets': True,
        'tf_repeat': False,
        'tf_shuffle': False,
    })
    return calib_config


def _prepare_detection_score_context(
    model,
    prep_config,
    calibration_files=None,
    temp_dir="./temp_score_context",
    num_samples_threshold_calibration=None,
    detection_score_mode="logit",
    detection_score_config=None,
):
    """Prepare latent noise reference statistics for detector score modes."""
    mode = _normalise_detection_score_mode(detection_score_mode)
    if not _score_mode_needs_noise_reference(mode):
        return _build_detection_score_context(
            None,
            None,
            detection_score_mode=mode,
            detection_score_config=detection_score_config,
        )

    config = dict(detection_score_config or {})
    if "noise_mean" in config and "noise_std" in config:
        return _build_detection_score_context(
            None,
            None,
            detection_score_mode=mode,
            detection_score_config=config,
        )

    calib_config = _build_calibration_prep_config(
        prep_config,
        calibration_files,
        temp_dir,
        num_samples_threshold_calibration=num_samples_threshold_calibration,
    )
    _, val_ds, _ = pre_processing_with_memmap(**calib_config)

    z_mean_all = []
    z_log_var_all = []
    for x_batch, _ in val_ds:
        z_mean_batch, z_log_var_batch = _latent_outputs(model, x_batch)
        z_mean_all.append(z_mean_batch)
        z_log_var_all.append(z_log_var_batch)

    val_ds = _release_dataset_resources(dataset_obj=val_ds)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if not z_mean_all:
        raise RuntimeError(
            "Unable to compute latent noise reference statistics: calibration dataset was empty."
        )

    return _build_detection_score_context(
        np.concatenate(z_mean_all, axis=0),
        np.concatenate(z_log_var_all, axis=0),
        detection_score_mode=mode,
        detection_score_config=detection_score_config,
    )


def _make_threshold_sweep(y_pred, detection_score_mode):
    """Build a detector-score-specific threshold sweep range."""
    mode = _normalise_detection_score_mode(detection_score_mode)
    y_pred = np.asarray(y_pred).reshape(-1)
    score_min = float(np.min(y_pred))
    score_max = float(np.max(y_pred))

    if mode == "logit":
        start_sweep = min(-1.5, score_min)
        end_sweep = max(2.0, score_max * 1.5) if score_max > 0 else 2.0
        if end_sweep <= start_sweep:
            end_sweep = start_sweep + 1.0
        return np.linspace(start_sweep, end_sweep, 100)

    start_sweep = 0.0 if score_min >= 0 else score_min
    if np.isclose(score_max, start_sweep):
        end_sweep = score_max + max(1.0, abs(score_max) * 0.5)
    else:
        end_sweep = score_max + 0.25 * (score_max - start_sweep)
    if end_sweep <= start_sweep:
        end_sweep = start_sweep + 1.0
    return np.linspace(start_sweep, end_sweep, 100)


def summarise_latent_statistics(
    z_mean_all,
    z_logvar_all,
    y_true_all,
    logits_all,
    active_kl_threshold=1e-3,
):
    """Summarise latent statistics for one mass/SNR operating point."""
    y_bool = np.asarray(y_true_all).astype(bool).reshape(-1)
    z_mean_all = np.asarray(z_mean_all)
    z_logvar_all = np.asarray(z_logvar_all)
    logits_all = np.asarray(logits_all).reshape(-1)

    latent_dim = z_mean_all.shape[1]
    sig_mask = y_bool
    noi_mask = ~sig_mask

    def _safe_mean(arr, axis=0):
        return np.mean(arr, axis=axis) if arr.size else np.zeros(latent_dim)

    def _safe_std(arr, axis=0):
        return np.std(arr, axis=axis) if arr.size else np.zeros(latent_dim)

    def _safe_kl(zm, zlv):
        if zm.size == 0:
            return np.zeros(latent_dim)
        return 0.5 * np.mean(np.square(zm) + np.exp(zlv) - 1.0 - zlv, axis=0)

    zm_sig = z_mean_all[sig_mask]
    zm_noi = z_mean_all[noi_mask]
    zlv_sig = z_logvar_all[sig_mask]
    zlv_noi = z_logvar_all[noi_mask]

    mean_sig = _safe_mean(zm_sig)
    mean_noi = _safe_mean(zm_noi)
    std_sig = _safe_std(zm_sig)
    std_noi = _safe_std(zm_noi)
    kl_sig = _safe_kl(zm_sig, zlv_sig)
    kl_noi = _safe_kl(zm_noi, zlv_noi)
    kl_all = _safe_kl(z_mean_all, z_logvar_all)
    mean_gap = mean_sig - mean_noi

    latent_df = pd.DataFrame(
        {
            "latent_dim": np.arange(latent_dim),
            "mean_signal": mean_sig,
            "mean_noise": mean_noi,
            "std_signal": std_sig,
            "std_noise": std_noi,
            "kl_signal": kl_sig,
            "kl_noise": kl_noi,
            "kl_all": kl_all,
            "mean_gap_abs": np.abs(mean_gap),
        }
    )

    signal_latent_norm = np.linalg.norm(zm_sig, axis=1) if zm_sig.size else np.array([])
    noise_latent_norm = np.linalg.norm(zm_noi, axis=1) if zm_noi.size else np.array([])

    summary = {
        "latent_gap_l2": float(np.linalg.norm(mean_gap)),
        "latent_gap_mean_abs": float(np.mean(np.abs(mean_gap))),
        "mean_kl_all": float(np.mean(kl_all)),
        "mean_kl_signal": float(np.mean(kl_sig)),
        "mean_kl_noise": float(np.mean(kl_noi)),
        "active_dims_all": int(np.sum(kl_all > active_kl_threshold)),
        "active_dims_signal": int(np.sum(kl_sig > active_kl_threshold)),
        "active_dims_noise": int(np.sum(kl_noi > active_kl_threshold)),
        "signal_latent_norm_mean": float(np.mean(signal_latent_norm)) if signal_latent_norm.size else 0.0,
        "noise_latent_norm_mean": float(np.mean(noise_latent_norm)) if noise_latent_norm.size else 0.0,
        "signal_logit_mean": float(np.mean(logits_all[sig_mask])) if np.any(sig_mask) else 0.0,
        "noise_logit_mean": float(np.mean(logits_all[noi_mask])) if np.any(noi_mask) else 0.0,
        "num_signal_windows_latent": int(np.sum(sig_mask)),
        "num_noise_windows_latent": int(np.sum(noi_mask)),
    }
    return summary, latent_df


def save_latent_statistics_plot(latent_df, save_path, title):
    """Create a compact latent-statistics plot for one operating point."""
    x = latent_df["latent_dim"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].bar(x - 0.2, latent_df["mean_signal"], width=0.4, label="signal", color="tab:red")
    axes[0].bar(x + 0.2, latent_df["mean_noise"], width=0.4, label="noise", color="tab:blue")
    axes[0].set_ylabel("mean(z_mean)")
    axes[0].legend()

    axes[1].bar(x - 0.2, latent_df["std_signal"], width=0.4, label="signal", color="tab:red")
    axes[1].bar(x + 0.2, latent_df["std_noise"], width=0.4, label="noise", color="tab:blue")
    axes[1].set_ylabel("std(z_mean)")

    axes[2].bar(x - 0.2, latent_df["kl_signal"], width=0.4, label="signal", color="tab:red")
    axes[2].bar(x + 0.2, latent_df["kl_noise"], width=0.4, label="noise", color="tab:blue")
    axes[2].plot(x, latent_df["kl_all"], color="black", marker="o", linewidth=1.0, label="all")
    axes[2].set_ylabel("per-dim KL")
    axes[2].set_xlabel("latent dimension")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _release_dataset_resources(
    dataset_obj=None,
    clear_signal_cache=False,
    clear_tf_session=False,
):
    """
    Best-effort cleanup for memmap-backed tf.data pipelines.

    These efficiency scripts iterate over TensorFlow datasets whose batches are
    loaded from NumPy memmaps via `tf.py_function`. On macOS, deleting the
    backing files while dataset workers may still exist can lead to a bus error.
    This helper centralizes the release order so filesystem cleanup only
    happens after Python and TensorFlow have had a chance to drop references.
    """
    dataset_obj = None

    if clear_signal_cache:
        try:
            signal_cache.clear()
        except Exception:
            pass

    if clear_tf_session:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    gc.collect()
    return None


def _compute_windows_per_year(window_size, step_size, fs_val):
    """Convert sliding-window settings into the number of windows per year."""
    seconds_per_year = 31536000
    total_samples_per_year = seconds_per_year * fs_val
    return (total_samples_per_year - window_size) / step_size + 1


def _build_calibration_dataframe(
    thresholds,
    fpr_fractions,
    fpr_errors_lower,
    fpr_errors_upper,
    fpr_sigmas,
):
    """Package calibration points into a DataFrame for reuse and debugging."""
    return pd.DataFrame({
        "threshold": np.asarray(thresholds),
        "fpr_fraction": np.asarray(fpr_fractions),
        "fpr_err_lower": np.asarray(fpr_errors_lower),
        "fpr_err_upper": np.asarray(fpr_errors_upper),
        "sigma": np.asarray(fpr_sigmas),
        "is_floor": np.asarray(fpr_fractions) == 0,
    })


def _load_calibration_points_dataframe(calibration_points_csv):
    """Load previously saved calibration points for fit debugging."""
    calib_df = pd.read_csv(calibration_points_csv)
    required_cols = {"threshold", "fpr_fraction", "fpr_err_lower", "fpr_err_upper", "sigma"}
    missing_cols = required_cols - set(calib_df.columns)
    if missing_cols:
        raise ValueError(
            f"Calibration points CSV is missing required columns: {sorted(missing_cols)}"
        )
    if "is_floor" not in calib_df.columns:
        calib_df["is_floor"] = calib_df["fpr_fraction"] == 0
    return calib_df


def _solve_threshold_from_linear_fit(fit_slope, fit_intercept, target_fraction):
    """Invert ln(FPR fraction) = slope * threshold + intercept for threshold."""
    if target_fraction <= 0:
        raise ValueError("target_fraction must be strictly positive.")
    if fit_slope == 0 or not np.isfinite(fit_slope):
        raise ValueError("fit_slope must be finite and non-zero.")
    return (np.log(target_fraction) - fit_intercept) / fit_slope


def _coerce_fit_covariance(covariance):
    """Return a validated 2x2 covariance matrix or None when unavailable."""
    if covariance is None:
        return None
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError(
            "Linear-fit covariance must be a finite 2x2 matrix ordered as "
            "[slope, intercept]."
        )
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Linear-fit covariance must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(cov)
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(cov, ord=2)))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError(
            "Linear-fit covariance must be positive semidefinite."
        )
    return cov


def _threshold_with_covariance_from_linear_fit(
    fit_slope,
    fit_intercept,
    target_fraction,
    fit_covariance,
):
    """
    Invert ln(FPR)=b*t+c and propagate the full Cov(b,c) to sigma_t.

        t = (ln f - c) / b
        dt/db = -t/b,  dt/dc = -1/b
        Var(t) = J Cov(b,c) J^T
    """
    threshold = _solve_threshold_from_linear_fit(
        fit_slope, fit_intercept, target_fraction
    )
    cov = _coerce_fit_covariance(fit_covariance)
    if cov is None:
        return float(threshold), np.nan
    jacobian = np.array([-threshold / fit_slope, -1.0 / fit_slope], dtype=float)
    variance = float(jacobian @ cov @ jacobian)
    threshold_err = np.sqrt(max(variance, 0.0))
    return float(threshold), float(threshold_err)


def _build_fp_target_list(
    fp_target_values=None,
    fp_scan_range=None,
):
    """
    Build a sorted list of positive FP/year targets.

    `fp_target_values` can be an explicit list. Otherwise `fp_scan_range` can
    define either a linear or log10-spaced range using start/stop/step.
    """
    if fp_target_values is not None:
        fp_values = np.asarray(fp_target_values, dtype=float)
    elif fp_scan_range is not None:
        space = str(fp_scan_range.get("space", "log10")).lower()
        start = float(fp_scan_range["start"])
        stop = float(fp_scan_range["stop"])
        step = float(fp_scan_range["step"])
        if start <= 0 or stop <= 0:
            raise ValueError("FP scan start/stop must be positive.")
        if step <= 0:
            raise ValueError("FP scan step size must be strictly positive.")

        if space == "linear":
            fp_values = np.arange(start, stop + 0.5 * step, step, dtype=float)
        elif space == "log10":
            log_start = np.log10(start)
            log_stop = np.log10(stop)
            exponents = np.arange(log_start, log_stop + 0.5 * step, step, dtype=float)
            fp_values = np.power(10.0, exponents)
        else:
            raise ValueError("fp_scan_range['space'] must be 'linear' or 'log10'.")
    else:
        raise ValueError("Provide either fp_target_values or fp_scan_range.")

    fp_values = np.unique(np.asarray(fp_values, dtype=float))
    fp_values = fp_values[fp_values > 0]
    if fp_values.size == 0:
        raise ValueError("FP scan target list is empty after filtering positive values.")
    return np.sort(fp_values)


def _build_pbh_mass_list(
    explicit_masses_kg=None,
    mass_scan_range=None,
    M_solar=1.988e30,
):
    """
    Build a PBH mass grid in kg.

    By default the script accepts an explicit list of masses in kg via
    `explicit_masses_kg`. To make log-space scans less error-prone, it also
    supports a structured solar-mass range:

    {
        "space": "log10_exponents",
        "start_exp": -13,
        "stop_exp": -6,
        "num": 8,
    }

    or

    {
        "space": "log10_exponents",
        "start_exp": -13,
        "stop_exp": -6,
        "step_exp": 1,
    }

    If `mass_scan_range` is provided, it takes precedence over the explicit
    list. This avoids the ambiguous `np.logspace(..., 1)` trap where the last
    argument is the number of samples rather than the exponent step size.
    """
    if mass_scan_range is not None:
        cfg = dict(mass_scan_range)
        space = str(cfg.get("space", "log10_exponents")).lower()
        if space != "log10_exponents":
            raise ValueError(
                "mass_scan_range['space'] must currently be 'log10_exponents'."
            )

        start_exp = float(cfg["start_exp"])
        stop_exp = float(cfg["stop_exp"])

        if "num" in cfg:
            num = int(cfg["num"])
            if num <= 0:
                raise ValueError("mass_scan_range['num'] must be a positive integer.")
            masses_solar = np.logspace(start_exp, stop_exp, num=num, dtype=float)
        elif "step_exp" in cfg:
            step_exp = float(cfg["step_exp"])
            if step_exp <= 0:
                raise ValueError("mass_scan_range['step_exp'] must be strictly positive.")
            exponents = np.arange(
                start_exp,
                stop_exp + 0.5 * step_exp,
                step_exp,
                dtype=float,
            )
            masses_solar = np.power(10.0, exponents)
        else:
            raise ValueError(
                "mass_scan_range must define either 'num' or 'step_exp'."
            )

        masses_kg = masses_solar * float(M_solar)
    else:
        if explicit_masses_kg is None:
            raise ValueError(
                "Provide either an explicit PBH mass list in kg or a mass_scan_range."
            )
        masses_kg = np.asarray(explicit_masses_kg, dtype=float)

    masses_kg = np.unique(masses_kg[masses_kg > 0])
    if masses_kg.size == 0:
        raise ValueError("PBH mass grid is empty after filtering positive values.")
    return np.sort(masses_kg)


def _format_float_tag(value):
    """Create a filesystem-safe short tag from a float."""
    return f"{float(value):.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def _derive_threshold_table_from_fit(calibration_result, target_fp_values):
    """
    Convert FP/year targets into thresholds using one shared linear log-fit.

    This is the core consistency step for FP scan mode: every operating point
    comes from the same fitted threshold-to-FPR relation.
    """
    if not calibration_result.get("fit_available", False):
        raise RuntimeError(
            "FP scan mode requires a valid linear log-fit. "
            "Calibration did not produce one, and no manual fit was provided."
        )

    fit_slope = float(calibration_result["fit_slope"])
    fit_intercept = float(calibration_result["fit_intercept"])
    fit_covariance = calibration_result.get("fit_covariance")
    windows_per_year = float(calibration_result["windows_per_year"])
    fp_values = _build_fp_target_list(fp_target_values=target_fp_values)

    rows = []
    for fp_value in fp_values:
        target_fraction = fp_value / windows_per_year
        threshold, threshold_err = _threshold_with_covariance_from_linear_fit(
            fit_slope,
            fit_intercept,
            target_fraction,
            fit_covariance,
        )
        rows.append({
            "target_fp_per_year": float(fp_value),
            "target_fpr_fraction": float(target_fraction),
            "threshold": float(threshold),
            "threshold_err": float(threshold_err),
            "threshold_minus_1sigma": float(threshold - threshold_err)
            if np.isfinite(threshold_err) else np.nan,
            "threshold_plus_1sigma": float(threshold + threshold_err)
            if np.isfinite(threshold_err) else np.nan,
            "fit_slope": fit_slope,
            "fit_intercept": fit_intercept,
        })
    return pd.DataFrame(rows)


def _save_calibration_fit_debug_plots(calibration_result, output_base_dir):
    """
    Save calibration debug plots from a reusable calibration result object.

    Keeping this plotting in one helper lets us reuse it for both live
    calibration and the manual fit-injection path.
    """
    calib_df = calibration_result.get("calibration_df")
    fit_available = calibration_result.get("fit_available", False)
    fit_slope = calibration_result.get("fit_slope")
    fit_intercept = calibration_result.get("fit_intercept")
    fit_covariance = calibration_result.get("fit_covariance")
    fit_chi2_red = calibration_result.get("fit_chi2_red", np.nan)
    score_label = calibration_result.get("score_label", "score")
    selected_target_fraction = calibration_result.get("selected_target_fraction")
    selected_target_fp_per_year = calibration_result.get("selected_target_fp_per_year")
    selected_threshold = calibration_result.get("selected_threshold")
    selected_threshold_err = calibration_result.get("selected_threshold_err", np.nan)
    score_values = calibration_result.get("score_values")

    if selected_target_fp_per_year is not None and np.isfinite(selected_target_fp_per_year):
        target_label = f"Target ({selected_target_fp_per_year:g} FP/yr)"
        threshold_label = (
            f"Thr: {selected_threshold:.4f} ({selected_target_fp_per_year:g} FP/yr)"
            if selected_threshold is not None and np.isfinite(selected_threshold)
            else None
        )
    else:
        target_label = "Target"
        threshold_label = (
            f"Thr: {selected_threshold:.4f}"
            if selected_threshold is not None and np.isfinite(selected_threshold)
            else None
        )

    if calib_df is not None:
        calib_df.to_csv(
            os.path.join(output_base_dir, "threshold_calibration_full_points.csv"),
            index=False,
        )

    fit_summary = {
        key: calibration_result.get(key)
        for key in (
            "fit_source", "fit_available", "fit_slope", "fit_intercept",
            "fit_covariance", "fit_slope_err", "fit_intercept_err",
            "fit_cov_slope_intercept", "fit_chi2_red", "fit_dof",
            "fit_threshold_min", "fit_threshold_max",
            "selected_target_fp_per_year", "selected_target_fraction",
            "selected_threshold", "selected_threshold_err",
        )
    }
    with open(os.path.join(output_base_dir, "threshold_calibration_fit.json"), "w") as handle:
        json.dump(fit_summary, handle, indent=2, default=float)

    if score_values is not None and len(score_values) > 0:
        try:
            plt.figure(figsize=(10, 6))
            bins = np.linspace(np.min(score_values), np.max(score_values), 100)
            plt.hist(
                score_values,
                bins=bins,
                alpha=0.7,
                label=f"Calibration samples (N={len(score_values)})",
                log=True,
            )
            plt.xlabel(f"Score: {score_label}")
            plt.ylabel("Count")
            plt.title(f"Calibration Score Histogram ({score_label})")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_base_dir, "calibration_prediction_histogram.png"),
                dpi=300,
            )
            plt.close()
        except Exception as exc:
            logger.warning(f"Histogram plotting failed: {exc}")

    if calib_df is None and not fit_available:
        return

    try:
        plt.figure(figsize=(10, 6))

        if calib_df is not None:
            t = calib_df["threshold"].to_numpy()
            r = calib_df["fpr_fraction"].to_numpy()
            floor_mask = calib_df["is_floor"].to_numpy(dtype=bool)
            non_zeros = r[r > 0]
            visual_floor = (np.min(non_zeros) * 0.01) if len(non_zeros) > 0 else 1e-13
            y_plot_safe = r.copy()
            y_plot_safe[floor_mask] = visual_floor

            err_low = np.maximum(
                0,
                y_plot_safe - np.maximum(calib_df["fpr_err_lower"].to_numpy(), visual_floor),
            )
            err_high = np.maximum(
                0,
                calib_df["fpr_err_upper"].to_numpy() - y_plot_safe,
            )

            if np.any(~floor_mask):
                plt.errorbar(
                    t[~floor_mask],
                    y_plot_safe[~floor_mask],
                    yerr=[err_low[~floor_mask], err_high[~floor_mask]],
                    fmt='o',
                    color='blue',
                    alpha=0.5,
                    markersize=3,
                    label='Measured (Wilson)',
                )
            if np.any(floor_mask):
                plt.scatter(
                    t[floor_mask],
                    y_plot_safe[floor_mask],
                    color='green',
                    s=10,
                    label='Floor (0 events)',
                )
        else:
            t = None

        if fit_available:
            if t is not None and len(t) > 0:
                fit_x = np.linspace(np.min(t), np.max(t), 300)
            else:
                fit_min = calibration_result.get("fit_threshold_min")
                fit_max = calibration_result.get("fit_threshold_max")
                if fit_min is None or fit_max is None:
                    fit_center = calibration_result.get("selected_threshold", 0.0)
                    fit_min = fit_center - 1.0
                    fit_max = fit_center + 1.0
                fit_x = np.linspace(float(fit_min), float(fit_max), 300)

            fit_y = np.exp(linear_log_model(fit_x, fit_slope, fit_intercept))
            cov = _coerce_fit_covariance(fit_covariance)
            if cov is not None:
                design = np.column_stack([fit_x, np.ones_like(fit_x)])
                log_fit_sigma = np.sqrt(
                    np.maximum(np.einsum("ij,jk,ik->i", design, cov, design), 0.0)
                )
                plt.fill_between(
                    fit_x,
                    np.exp(np.log(fit_y) - log_fit_sigma),
                    np.exp(np.log(fit_y) + log_fit_sigma),
                    color="gray",
                    alpha=0.2,
                    label=r"Linear log-fit $\pm 1\sigma$",
                )
            plt.plot(
                fit_x,
                fit_y,
                'k-.',
                linewidth=2,
                label=f"Linear log-fit ($\\chi^2_{{red}}={fit_chi2_red:.2f}$)",
            )

        if selected_target_fraction is not None and selected_target_fraction > 0:
            plt.axhline(
                selected_target_fraction,
                color='green',
                linestyle=':',
                label=target_label,
            )
        if selected_threshold is not None and np.isfinite(selected_threshold):
            if np.isfinite(selected_threshold_err):
                plt.axvspan(
                    selected_threshold - selected_threshold_err,
                    selected_threshold + selected_threshold_err,
                    color='orange',
                    alpha=0.18,
                    label=r"Threshold fit $\pm 1\sigma$",
                )
            plt.axvline(
                selected_threshold,
                color='orange',
                linestyle='-',
                linewidth=2,
                label=threshold_label,
            )

        plt.yscale('log')
        plt.xlabel('Threshold', size=18)
        plt.ylabel('FPR', size=18)
        plt.title(f"Threshold Calibration ({score_label})", size=20)
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_base_dir, "threshold_calibration_full.png"),
            dpi=300,
        )
        plt.close()

        # Plot residuals in log-space so it is easier to see where the shared
        # linear tail fit is faithful and where it starts to drift.
        if calib_df is not None and fit_available:
            positive_mask = calib_df["fpr_fraction"].to_numpy() > 0
            if np.any(positive_mask):
                t_pos = calib_df.loc[positive_mask, "threshold"].to_numpy()
                log_r_pos = np.log(
                    calib_df.loc[positive_mask, "fpr_fraction"].to_numpy()
                )
                sigma_pos = calib_df.loc[positive_mask, "sigma"].to_numpy()
                log_sigma_pos = sigma_pos / np.exp(log_r_pos)
                residuals = log_r_pos - linear_log_model(
                    t_pos, fit_slope, fit_intercept
                )

                plt.figure(figsize=(10, 4.5))
                plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
                plt.errorbar(
                    t_pos,
                    residuals,
                    yerr=log_sigma_pos,
                    fmt="o",
                    color="tab:purple",
                    alpha=0.7,
                    markersize=3,
                )
                plt.xlabel("Threshold", size=16)
                plt.ylabel("ln(FPR) residual", size=16)
                plt.title(f"Calibration Residuals ({score_label})", size=18)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_base_dir, "threshold_calibration_residuals.png"),
                    dpi=300,
                )
                plt.close()
    except Exception as exc:
        logger.warning(f"Calibration fit plotting failed: {exc}")

# -----------------------------------------------------------------------------
# 2. Calibration Function (With Rayleigh Fit & Fixed Zoom)
# -----------------------------------------------------------------------------
def calibrate_threshold(
    model,
    prep_config,
    target_fp_per_year=1.0,
    calibration_files=None,
    temp_dir="./temp_calib",
    num_samples_threshold_calibration=None,
    detection_score_mode="logit",
    detection_score_config=None,
    return_details=False,
    manual_linear_fit=None,
    calibration_points_csv=None,
    require_linear_fit=False,
):
    detection_score_mode = _normalise_detection_score_mode(detection_score_mode)
    detection_score_config = dict(detection_score_config or {})
    output_base_dir = os.path.dirname(temp_dir)
    window_size = prep_config['window_size']
    step_size = prep_config.get('step_size', 2048)
    fs_val = prep_config.get('fs_val', 14e6)
    windows_per_year = _compute_windows_per_year(window_size, step_size, fs_val)

    if manual_linear_fit is not None:
        logger.info("--- Using Provided Linear Fit: Skipping Live Threshold Calibration ---")
        manual_linear_fit = dict(manual_linear_fit)
        fit_slope = float(manual_linear_fit["slope"])
        fit_intercept = float(manual_linear_fit["intercept"])
        fit_chi2_red = float(manual_linear_fit.get("chi2_red", np.nan))
        fit_dof = manual_linear_fit.get("dof")
        fit_covariance = _coerce_fit_covariance(
            manual_linear_fit.get("covariance", manual_linear_fit.get("fit_covariance"))
        )
        fit_source = "manual"

        calibration_points_csv = (
            calibration_points_csv or manual_linear_fit.get("calibration_points_csv")
        )
        calib_df = (
            _load_calibration_points_dataframe(calibration_points_csv)
            if calibration_points_csv else None
        )

        score_context = _prepare_detection_score_context(
            model,
            prep_config,
            calibration_files=calibration_files,
            temp_dir=os.path.join(output_base_dir, "temp_score_context_manual"),
            num_samples_threshold_calibration=num_samples_threshold_calibration,
            detection_score_mode=detection_score_mode,
            detection_score_config=detection_score_config,
        )
        score_label = _get_detection_score_label(detection_score_mode, score_context)

        target_fraction = (
            target_fp_per_year / windows_per_year if target_fp_per_year is not None else None
        )
        if target_fraction is not None:
            chosen_threshold, chosen_threshold_err = (
                _threshold_with_covariance_from_linear_fit(
                    fit_slope, fit_intercept, target_fraction, fit_covariance
                )
            )
        else:
            chosen_threshold, chosen_threshold_err = np.nan, np.nan

        calibration_result = {
            "score_context": score_context,
            "score_label": score_label,
            "detection_score_mode": detection_score_mode,
            "detection_score_config": detection_score_config,
            "windows_per_year": float(windows_per_year),
            "window_size": window_size,
            "step_size": step_size,
            "fs_val": fs_val,
            "fit_available": True,
            "fit_source": fit_source,
            "fit_slope": fit_slope,
            "fit_intercept": fit_intercept,
            "fit_covariance": fit_covariance.tolist() if fit_covariance is not None else None,
            "fit_slope_err": float(np.sqrt(max(fit_covariance[0, 0], 0.0)))
            if fit_covariance is not None else np.nan,
            "fit_intercept_err": float(np.sqrt(max(fit_covariance[1, 1], 0.0)))
            if fit_covariance is not None else np.nan,
            "fit_cov_slope_intercept": float(fit_covariance[0, 1])
            if fit_covariance is not None else np.nan,
            "fit_chi2_red": fit_chi2_red,
            "fit_dof": fit_dof,
            "fit_threshold_min": manual_linear_fit.get("fit_threshold_min"),
            "fit_threshold_max": manual_linear_fit.get("fit_threshold_max"),
            "selected_target_fp_per_year": target_fp_per_year,
            "selected_target_fraction": target_fraction,
            "selected_threshold": chosen_threshold,
            "selected_threshold_err": chosen_threshold_err,
            "calibration_df": calib_df,
            "score_values": None,
        }

        _save_calibration_fit_debug_plots(calibration_result, output_base_dir)
        if return_details:
            return calibration_result
        return chosen_threshold, score_context

    logger.info("--- Starting Threshold Calibration (Linear Log Tail) ---")

    # 1. Preprocessing
    calib_config = _build_calibration_prep_config(
        prep_config,
        calibration_files,
        temp_dir,
        num_samples_threshold_calibration=num_samples_threshold_calibration,
    )
    _, val_ds, _ = pre_processing_with_memmap(**calib_config)

    # 2. Predict
    logits_all = []
    z_mean_all = []
    z_log_var_all = []
    y_true_all = []

    needs_logits = detection_score_mode == "logit"
    needs_latents = _score_mode_needs_latents(detection_score_mode)
    for x_batch, y_batch in val_ds:
        if needs_logits:
            logits_all.append(
                _predict_logits(
                    model,
                    x_batch,
                    classifier_inference_mode=detection_score_config.get(
                        "classifier_inference_mode", "sampled_z"
                    ),
                )
            )
        if needs_latents:
            z_mean_batch, z_log_var_batch = _latent_outputs(model, x_batch)
            z_mean_all.append(z_mean_batch)
            z_log_var_all.append(z_log_var_batch)
        y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all)
    logits_all = np.concatenate(logits_all) if logits_all else None
    z_mean_all = np.concatenate(z_mean_all, axis=0) if z_mean_all else None
    z_log_var_all = np.concatenate(z_log_var_all, axis=0) if z_log_var_all else None

    score_context = _build_detection_score_context(
        z_mean_all,
        z_log_var_all,
        detection_score_mode=detection_score_mode,
        detection_score_config=detection_score_config,
    )
    score_label = _get_detection_score_label(detection_score_mode, score_context)
    y_pred = _compute_detection_score(
        logits_all,
        z_mean_all,
        z_log_var_all,
        detection_score_mode=detection_score_mode,
        score_context=score_context,
        detection_score_config=detection_score_config,
    )

    # 3. Sweep
    thresholds = _make_threshold_sweep(y_pred, detection_score_mode)
    total_noise_windows_measured = np.sum(y_true == 0)
    fpr_fractions = []
    fpr_errors_lower = []
    fpr_errors_upper = []
    fpr_sigmas = []

    for threshold in thresholds:
        _, _, _, fpr_fraction = calculate_event_recall_and_fpr(
            y_true,
            y_pred,
            threshold,
            window_size,
            step_size,
            fs_val,
        )
        fpr_fractions.append(fpr_fraction)

        n_fp = int(fpr_fraction * total_noise_windows_measured)
        low, high = wilson_score_interval(n_fp, total_noise_windows_measured)
        fpr_errors_lower.append(low)
        fpr_errors_upper.append(high)

        sigma = (high - low) / 2.0
        if sigma == 0:
            sigma = 1e-12
        fpr_sigmas.append(sigma)

    t = np.asarray(thresholds)
    r = np.asarray(fpr_fractions)
    sigma = np.asarray(fpr_sigmas)
    calib_df = _build_calibration_dataframe(
        t,
        r,
        fpr_errors_lower,
        fpr_errors_upper,
        sigma,
    )

    # The fit is intentionally taken from the low-FPR tail only.
    log_fit_start = 0.000001
    linear_fit_mask = (r > 0) & (r <= log_fit_start * np.max(r))
    fit_available = False
    fit_slope = np.nan
    fit_intercept = np.nan
    fit_chi2_red = np.nan
    fit_dof = np.nan
    fit_covariance = None

    MIN_TAIL_POINTS = 4
    num_tail_points = int(np.sum(linear_fit_mask))
    if num_tail_points >= MIN_TAIL_POINTS:
        try:
            popt_lin, pcov_lin = curve_fit(
                linear_log_model,
                t[linear_fit_mask],
                np.log(r[linear_fit_mask]),
                sigma=sigma[linear_fit_mask] / r[linear_fit_mask],
                absolute_sigma=True,
            )
            fit_slope, fit_intercept = [float(val) for val in popt_lin]
            try:
                fit_covariance = _coerce_fit_covariance(pcov_lin)
            except ValueError as covariance_exc:
                fit_covariance = None
                logger.warning(
                    "Linear log-fit parameters are usable, but covariance is "
                    f"unavailable: {covariance_exc}"
                )
            residuals = (
                np.log(r[linear_fit_mask])
                - linear_log_model(t[linear_fit_mask], fit_slope, fit_intercept)
            )
            chi2_lin = np.sum((residuals / (sigma[linear_fit_mask] / r[linear_fit_mask])) ** 2)
            fit_dof = int(np.sum(linear_fit_mask) - 2)
            fit_chi2_red = chi2_lin / fit_dof if fit_dof > 0 else np.nan
            fit_available = True
            logger.info(
                f"Linear log-fit (<= {log_fit_start * 100:.3g}% of max): "
                f"b={fit_slope:.4e}, c={fit_intercept:.4e}, "
                f"chi2_red={fit_chi2_red:.3f}"
            )
        except Exception as exc:
            logger.warning(f"Linear log-fit failed: {exc}")

    if require_linear_fit and not fit_available:
        raise RuntimeError(
            "A valid linear log-fit is required here, but calibration did not produce one."
        )

    target_fraction = target_fp_per_year / windows_per_year
    chosen_threshold = np.nan
    used_fallback_threshold = False

    if fit_available:
        chosen_threshold, chosen_threshold_err = _threshold_with_covariance_from_linear_fit(
            fit_slope,
            fit_intercept,
            target_fraction,
            fit_covariance,
        )
        logger.info(
            "Using Linear Log Extrapolation. Selected Threshold: "
            f"{chosen_threshold:.4f} +/- {chosen_threshold_err:.4f} "
            "(full fit covariance)"
        )
    else:
        chosen_threshold_err = np.nan
        measured_mask = r > 0
        t_meas = t[measured_mask]
        r_meas = r[measured_mask]

        if len(t_meas) >= 1 and np.allclose(r_meas, r_meas[0]):
            manual_increment = 0.0001
            chosen_threshold = t_meas[-1] + manual_increment
            used_fallback_threshold = True
            logger.warning(
                "No FPR drop detected (all FPR fractions are the same). "
                f"Manually setting threshold to plateau point + {manual_increment}: "
                f"{chosen_threshold:.4f}"
            )
        elif len(t_meas) < 2:
            raise RuntimeError(
                "Insufficient measured FP points to determine fallback threshold."
            )
        else:
            order = np.argsort(t_meas)
            t_meas = t_meas[order]
            r_meas = r_meas[order]
            t_last = t_meas[-1]
            r_last = r_meas[-1]
            t_prev = t_meas[-2]
            r_prev = r_meas[-2]
            local_slope = (np.log(r_last) - np.log(r_prev)) / (t_last - t_prev)

            if local_slope >= 0:
                raise RuntimeError(
                    "Non-decreasing FP detected near cutoff; cannot extrapolate safely."
                )

            delta_t = (np.log(target_fraction) - np.log(r_last)) / local_slope
            max_jump = 5.0 * abs(t_last - t_prev)
            delta_t = np.clip(delta_t, 0.0, max_jump)
            chosen_threshold = t_last + delta_t
            used_fallback_threshold = True
            logger.warning(
                "Sharp FP drop-off detected - using last-point slope extrapolation.\n"
                f"  Last measured: t={t_last:.4f}, FP={r_last:.2e}\n"
                f"  Previous:      t={t_prev:.4f}, FP={r_prev:.2e}\n"
                f"  Local slope:   dln(FP)/dt = {local_slope:.3e}\n"
                f"  delta_t:       {delta_t:.4f}\n"
                f"  Chosen threshold = {chosen_threshold:.4f}"
            )

    calibration_result = {
        "score_context": score_context,
        "score_label": score_label,
        "detection_score_mode": detection_score_mode,
        "detection_score_config": detection_score_config,
        "windows_per_year": float(windows_per_year),
        "window_size": window_size,
        "step_size": step_size,
        "fs_val": fs_val,
        "fit_available": fit_available,
        "fit_source": "calibrated",
        "fit_slope": float(fit_slope) if fit_available else np.nan,
        "fit_intercept": float(fit_intercept) if fit_available else np.nan,
        "fit_covariance": fit_covariance.tolist() if fit_covariance is not None else None,
        "fit_slope_err": float(np.sqrt(max(fit_covariance[0, 0], 0.0)))
        if fit_covariance is not None else np.nan,
        "fit_intercept_err": float(np.sqrt(max(fit_covariance[1, 1], 0.0)))
        if fit_covariance is not None else np.nan,
        "fit_cov_slope_intercept": float(fit_covariance[0, 1])
        if fit_covariance is not None else np.nan,
        "fit_chi2_red": fit_chi2_red,
        "fit_dof": fit_dof,
        "fit_threshold_min": float(np.min(t[linear_fit_mask])) if np.any(linear_fit_mask) else None,
        "fit_threshold_max": float(np.max(t[linear_fit_mask])) if np.any(linear_fit_mask) else None,
        "selected_target_fp_per_year": target_fp_per_year,
        "selected_target_fraction": float(target_fraction),
        "selected_threshold": float(chosen_threshold),
        "selected_threshold_err": float(chosen_threshold_err),
        "used_fallback_threshold": used_fallback_threshold,
        "calibration_df": calib_df,
        "score_values": y_pred,
    }

    _save_calibration_fit_debug_plots(calibration_result, output_base_dir)

    val_ds = _release_dataset_resources(dataset_obj=val_ds)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if return_details:
        return calibration_result
    return chosen_threshold, score_context


def _evaluate_threshold_table_on_scores(
    y_true,
    y_scores,
    threshold_table,
    window_size,
    step_size,
    fs_val,
):
    """
    Evaluate one set of window scores against many thresholds.

    The event-detection logic itself remains unchanged; we simply reuse it for
    every threshold derived from the shared FP calibration fit.
    """
    results = []
    for threshold_row in threshold_table.itertuples(index=False):
        detected, events, fp_per_year, fpr_fraction = calculate_event_recall_and_fpr(
            y_true,
            y_scores,
            threshold_row.threshold,
            window_size,
            step_size,
            fs_val,
        )
        results.append({
            "target_fp_per_year": float(threshold_row.target_fp_per_year),
            "target_fpr_fraction": float(threshold_row.target_fpr_fraction),
            "threshold": float(threshold_row.threshold),
            "detected": int(detected),
            "events": int(events),
            "fp_per_year_from_eval": float(fp_per_year),
            "fpr_fraction_from_eval": float(fpr_fraction),
        })
    return results


def _fit_efficiency_curve_summary(
    df_curve,
    target_efficiency_for_summary=0.95,
):
    """
    Fit the sigmoid efficiency curve for one mass and one FP target.

    Returning a compact dictionary keeps the caller simple and makes it easy to
    reuse the same fit logic in both normal and scan-style workflows later.
    """
    fit_result = {
        "fit_midpoint": np.nan,
        "fit_slope": np.nan,
        "fit_midpoint_err": np.nan,
        "fit_slope_err": np.nan,
        "fit_cov_x0_k": np.nan,
        "snr_at_target_eff": np.nan,
        "snr_at_target_eff_err": np.nan,
        "chi2_sigmoid": np.nan,
        "chi2_red_sigmoid": np.nan,
        "fit_success": False,
    }

    if df_curve.empty:
        return fit_result

    try:
        popt, pcov = curve_fit(
            sigmoid_func,
            df_curve["snr"],
            df_curve["efficiency"],
            p0=[5.0, 1.0],
            sigma=df_curve["sigma_val"],
            absolute_sigma=True,
            bounds=([0, 0.1], [50, 20]),
            maxfev=5000,
        )

        x0, k = popt
        sigma_x0 = np.sqrt(pcov[0, 0]) if pcov[0, 0] >= 0 else np.nan
        sigma_k = np.sqrt(pcov[1, 1]) if pcov[1, 1] >= 0 else np.nan
        cov_x0_k = pcov[0, 1]

        y_fit_vals = sigmoid_func(df_curve["snr"].values, *popt)
        residuals = df_curve["efficiency"].values - y_fit_vals
        chi2_sig = np.sum((residuals / df_curve["sigma_val"].values) ** 2)
        dof_sig = len(df_curve) - len(popt)
        chi2_red_sig = chi2_sig / dof_sig if dof_sig > 0 else np.nan

        A = np.log(1 / target_efficiency_for_summary - 1)
        dS_dx0 = 1.0
        dS_dk = A / (k ** 2)
        var_target_snr = (
            dS_dx0 ** 2 * pcov[0, 0]
            + dS_dk ** 2 * pcov[1, 1]
            + 2.0 * dS_dx0 * dS_dk * cov_x0_k
        )
        sigma_target_snr = np.sqrt(var_target_snr) if var_target_snr > 0 else np.nan
        target_snr = x0 - (1 / k) * np.log(1 / target_efficiency_for_summary - 1)

        fit_result.update({
            "fit_midpoint": float(x0),
            "fit_slope": float(k),
            "fit_midpoint_err": sigma_x0,
            "fit_slope_err": sigma_k,
            "fit_cov_x0_k": cov_x0_k,
            "snr_at_target_eff": float(target_snr),
            "snr_at_target_eff_err": sigma_target_snr,
            "chi2_sigmoid": float(chi2_sig),
            "chi2_red_sigmoid": chi2_red_sig,
            "fit_success": True,
        })
    except Exception as exc:
        logger.error(f"Sigmoid fit failed for curve summary: {exc}")

    return fit_result


def _efficiency_stats(detected, total):
    """Return efficiency, Wilson interval, and symmetric fit sigma."""
    efficiency = detected / total if total > 0 else 0.0
    low, high = wilson_score_interval(detected, total)
    sigma_val = ((high - efficiency) + (efficiency - low)) / 2.0
    return efficiency, low, high, max(float(sigma_val), 1e-6)


def _fit_efficiency_variant(df_mass, efficiency_column, sigma_column,
                            target_efficiency_for_summary):
    """Fit an alternate threshold's efficiency curve using shared fit logic."""
    if efficiency_column not in df_mass or sigma_column not in df_mass:
        return _fit_efficiency_curve_summary(
            pd.DataFrame(), target_efficiency_for_summary
        )
    variant = df_mass[["snr", efficiency_column, sigma_column]].copy()
    variant = variant.rename(columns={
        efficiency_column: "efficiency",
        sigma_column: "sigma_val",
    })
    variant = variant.dropna(subset=["efficiency", "sigma_val"])
    return _fit_efficiency_curve_summary(
        variant,
        target_efficiency_for_summary=target_efficiency_for_summary,
    )
    
# -----------------------------------------------------------------------------
# 2. Metric Calculation
# -----------------------------------------------------------------------------

def calculate_event_recall_and_fpr(y_true, y_pred, threshold, window_size, step_size, fs):
    """
    Calculates Event-Based Recall and Window-Based FP/Year.
    Includes precise edge-effect calculation for windows per year.
    """
    y_pred_bool = (y_pred > threshold) # Convert continuous model output y_pred into booleans of True when condition is met and else False
    
    # --- Event-Based Recall --- #
    diffs = np.diff(np.concatenate(([0], y_true.astype(int), [0]))) # convert boolean array y_true into 0s = False and 1s = True and add 0s at the start and end to make sure signals at the start and end are detected properly
    
    # Taking the i-th entry of the diffs array, a signal is considered to start when False --> True, i.e. 0 --> 1 and therefore np.diff(array) = array[i + 1] - array[i], e.g. array = [0, 0, 1, 1, 0, 0], then np.diff(array) = [0, 1, 0, -1, 0]
    # --> NOTE: This code has the flaw that if you inject multiple signal and some overlap, several events will be considered as a single prolonged event
    starts = np.where(diffs == 1)[0] # A signal start when np.diff(array) = 1
    ends = np.where(diffs == -1)[0] # A signal ends when np.diff(array) = -1
    
    num_events = len(starts)
    detected_events = 0
    
    for i in range(num_events):
        # Write a single event as it starts at the i-th time np.diff(array) = 1 and ends at the i-th time np.diff(array) = -1 and consider prediction of model inside of the i-th signal range
        event_slice = y_pred_bool[starts[i]:ends[i]] # slices are start inclusive but end exclusive, i.e. ends[i] is not inclduded in event_slice for an event
        
        # If at least one window is flagged as "1 = True" by the model, one event has been detected
        if np.any(event_slice):
            detected_events += 1
            
    # --- False Positive Rate / Year --- #
    noise_mask = (y_true == False) # Define noise windows as windows where there is no signal
    total_noise_windows = np.sum(noise_mask) # Sum up all noise windows
    fp_windows = np.sum(y_pred_bool & noise_mask) # Use logical "AND" <=> "&" to find FP, i.e. only if y_pred_bool and noise_mask are True, we count an event as a FP
    
    seconds_per_year = 31536000 # 60s * 60 min * 24h * 365 days
    total_samples_per_year = seconds_per_year * fs
    
    # Formula: N_windows = (N_total - Window_Size) / Step_Size + 1
    # This accounts for the edge effect of the window sliding over one year of continuous data
    windows_per_year = (total_samples_per_year - window_size) / step_size + 1
    
    fpr_fraction = fp_windows / total_noise_windows if total_noise_windows > 0 else 0.0
    fp_per_year = fpr_fraction * windows_per_year # FP/window * windows / year = FP / year

    return detected_events, num_events, fp_per_year, fpr_fraction

# -----------------------------------------------------------------------------
# 3. Core Logic: Efficiency
# -----------------------------------------------------------------------------

def generate_efficiency_curves(
    model_path,
    normalization_params,
    normalization_mode,
    test_file_suffixes,
    pbh_mass_list,
    snr_list,
    preprocessing_config,
    output_dir=os.path.join(_PROJECT_ROOT, 'efficiency_vae_results'),
    target_fp_per_year=1.0,
    num_runs_per_point=10,
    target_efficiency_for_summary=0.95,
    num_samples_threshold_calibration=None,
    detection_score_mode="logit",
    detection_score_config=None,
    preset_threshold=None, # If already calibrated, set manual threshold here to skip threshold calibration
    clustered_fp_merge_gap_windows=0,  # D: merge clusters separated by <= N windows (diagnostic CSV only)
):
    """
    Main Driver Function.

    NOTE on metrics: the main results (efficiency curves, SNR95) are based on
    the WINDOW-based FP/year metric, unchanged. A separate
    `clustered_fp_diagnostics.csv` with clustered-trigger FP/year is written
    purely as a secondary diagnostic (section D) and is NOT used anywhere in
    the main plots or threshold calibration.
    """
    detection_score_mode = _normalise_detection_score_mode(detection_score_mode)
    detection_score_config = dict(detection_score_config or {})
    os.makedirs(output_dir, exist_ok=True)
    model_path, preprocessing_config, normalization_params, _ = _align_model_and_preprocessing(
        model_path,
        preprocessing_config,
        normalization_params,
        normalization_mode,
    )
    preprocessing_config['normalization_type'] = normalization_mode
    if normalization_mode == 'zscore' and normalization_params:
        preprocessing_config['global_mean_input'] = normalization_params.get('mean_value')
        preprocessing_config['global_std_input'] = normalization_params.get('std_dev_value')
    elif normalization_mode == 'min_max' and normalization_params:
        preprocessing_config['global_min_input'] = normalization_params.get('min_value')
        preprocessing_config['global_max_input'] = normalization_params.get('max_value')
    
    s_curve_plot_dir = os.path.join(output_dir, "S_curve_fits")
    s_curve_csv_dir = os.path.join(s_curve_plot_dir, "S_curve_fits_csv_files")
    latent_stats_dir = os.path.join(output_dir, "latent_statistics")
    threshold_sys_dir = os.path.join(output_dir, "threshold_systematics")
    os.makedirs(s_curve_plot_dir, exist_ok=True)
    os.makedirs(s_curve_csv_dir, exist_ok=True)
    os.makedirs(latent_stats_dir, exist_ok=True)
    os.makedirs(threshold_sys_dir, exist_ok=True)
    
    # 1. Load Model
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    score_context = None
    model_window = _get_model_window_size(model)

    cfg_window = preprocessing_config['window_size']
    filepath_suffixes = preprocessing_config['filepath_suffixes']
    
    # Manually enforce that chose window size and input window size of trained model match, even though the model might be a time-length agnostic network, like a CNN network
    if model_window is not None and model_window != cfg_window:
        raise ValueError(
            f"Window size mismatch: model was built with {model_window} "
            f"time steps, but config has window_size={cfg_window}. "
            "Use the matching window size or retrain the model."
            )
    
    # 2. Threshold selection (calibrate or use preset)
    if preset_threshold is not None:
        calib_threshold = float(preset_threshold)
        calibration_result = {
            "fit_available": False,
            "fit_source": "preset_threshold",
            "fit_covariance": None,
            "fit_slope": np.nan,
            "fit_intercept": np.nan,
            "fit_slope_err": np.nan,
            "fit_intercept_err": np.nan,
            "fit_cov_slope_intercept": np.nan,
            "selected_threshold": calib_threshold,
            "selected_threshold_err": np.nan,
        }
        score_context = _prepare_detection_score_context(
            model,
            preprocessing_config,
            calibration_files=test_file_suffixes,
            temp_dir=os.path.join(output_dir, "temp_score_context"),
            num_samples_threshold_calibration=num_samples_threshold_calibration,
            detection_score_mode=detection_score_mode,
            detection_score_config=detection_score_config,
        )
        score_label = _get_detection_score_label(detection_score_mode, score_context)
        logger.info(
            f"Step 1/3: Skipping calibration. "
            f"Using preset threshold = {calib_threshold} "
            f"for detector score '{score_label}'"
        )
    else:
        logger.info("Step 1/3: Calibrating Threshold...")
        calibration_result = calibrate_threshold(
            model,
            preprocessing_config,
            target_fp_per_year,
            test_file_suffixes,
            temp_dir=os.path.join(output_dir, "temp_calib"),
            num_samples_threshold_calibration=num_samples_threshold_calibration,
            detection_score_mode=detection_score_mode,
            detection_score_config=detection_score_config,
            return_details=True,
        )
        calib_threshold = float(calibration_result["selected_threshold"])
        score_context = calibration_result["score_context"]
        logger.info(f"Final Calibrated Threshold: {calib_threshold}")

    score_label = _get_detection_score_label(detection_score_mode, score_context)
    threshold_err = float(calibration_result.get("selected_threshold_err", np.nan))
    threshold_systematic_available = np.isfinite(threshold_err) and threshold_err >= 0
    threshold_minus = (
        calib_threshold - threshold_err if threshold_systematic_available else np.nan
    )
    threshold_plus = (
        calib_threshold + threshold_err if threshold_systematic_available else np.nan
    )

    with open(os.path.join(output_dir, "calibration_info.txt"), "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Detection Score Mode: {detection_score_mode}\n")
        f.write(f"Detection Score Label: {score_label}\n")
        f.write(f"Detection Score Config: {detection_score_config}\n")
        f.write(f"Target FP/Year: {target_fp_per_year}\n")
        f.write(f"Threshold Fit Source: {calibration_result.get('fit_source')}\n")
        f.write(f"Threshold Fit Slope: {calibration_result.get('fit_slope')}\n")
        f.write(f"Threshold Fit Intercept: {calibration_result.get('fit_intercept')}\n")
        f.write(f"Threshold Fit Covariance [slope, intercept]: {calibration_result.get('fit_covariance')}\n")
        f.write(f"Threshold Fit Slope Error: {calibration_result.get('fit_slope_err')}\n")
        f.write(f"Threshold Fit Intercept Error: {calibration_result.get('fit_intercept_err')}\n")
        f.write(f"Threshold Fit Cov(slope, intercept): {calibration_result.get('fit_cov_slope_intercept')}\n")
        f.write(
            "Threshold uncertainty interpretation: calibration-fit statistical "
            "uncertainty propagated as a correlated nuisance across the efficiency curve.\n"
        )
        f.write(
            "Covariance limitation: the fitted slope/intercept covariance uses "
            "curve_fit's independent-point approximation, although threshold-sweep "
            "tail counts are nested and therefore correlated.\n"
        )
        if preset_threshold is not None:
            f.write(f"Preset Threshold Used: {calib_threshold}\n")
        else:
            f.write(f"Calibrated Threshold: {calib_threshold}\n")
            f.write(f"Calibrated Threshold Error (1 sigma): {threshold_err}\n")
            f.write(f"Threshold Minus 1 Sigma: {threshold_minus}\n")
            f.write(f"Threshold Plus 1 Sigma: {threshold_plus}\n")

    # 3. Efficiency Loop (Grid Search: Mass x SNR)
    logger.info("Step 2/3: Computing Efficiency Grid...")
    
    results = []
    temp_proc_dir = os.path.join(output_dir, "temp_proc")
    
    for mass in pbh_mass_list:
        logger.info(f"--- Processing Mass: {mass:.2e} M_solar ---")
        
        mass_results = []
        
        for snr in snr_list:
            total_detected = 0
            total_detected_threshold_minus = 0
            total_detected_threshold_plus = 0
            total_events = 0
            max_scores_this_snr = []
            snr_latent_means = []
            snr_latent_logvars = []
            snr_labels = []
            snr_logits = []
            snr_scores = []
            cluster_diag_runs = []  # (y_true, scores) per run — diagnostic only

            for run_i in range(num_runs_per_point):
                run_config = preprocessing_config.copy()
                
                run_config.pop('test_file_suffixes', None)
                
                # REMOVE calibration-only key before calling preprocessing
                run_config.pop('num_samples_to_read_per_file_threshold_calibration', None)
                
                run_config.update({
                    'filepath_suffixes': filepath_suffixes,
                    'memmap_dir': temp_proc_dir,
                    'inject_signals': True,
                    'snr_based_injection': True,
                    'm_PBH_injection_list': [mass],
                    'amplitude_spectrum_range': [snr],
                    'normalization_type': normalization_mode,
                    'custom_noise_std': preprocessing_config.get('custom_noise_std'),
                    'return_tf_datasets': True,
                    'tf_batch_size': 4096,
                    'tf_repeat': False,
                    'tf_shuffle': False,
                    'global_std_input': preprocessing_config.get('global_std_input'),
                })

                _, val_ds, _ = pre_processing_with_memmap(**run_config)
                    
                for x_b, y_b in val_ds.take(1):   # just one batch
                    logger.info(f"First batch shape from dataset: {x_b.shape}")
                
                y_pred_list = []
                y_true_list = []
                for x_b, y_b in val_ds:
                    logits_batch = _predict_logits(
                        model,
                        x_b,
                        classifier_inference_mode=detection_score_config.get(
                            "classifier_inference_mode", "sampled_z"
                        ),
                    )
                    z_mean_batch, z_logvar_batch = _latent_outputs(model, x_b)
                    scores_batch = _compute_detection_score(
                        logits_batch,
                        z_mean_batch,
                        z_logvar_batch,
                        detection_score_mode=detection_score_mode,
                        score_context=score_context,
                        detection_score_config=detection_score_config,
                    )
                    labels_batch = y_b.numpy().reshape(-1)

                    y_pred_list.append(scores_batch)
                    y_true_list.append(labels_batch)
                    snr_logits.append(logits_batch)
                    snr_labels.append(labels_batch)
                    snr_scores.append(scores_batch)
                    snr_latent_means.append(z_mean_batch)
                    snr_latent_logvars.append(z_logvar_batch)
                
                y_p = np.concatenate(y_pred_list)
                y_t = np.concatenate(y_true_list)
                cluster_diag_runs.append((y_t.copy(), y_p.copy()))

                max_scores_this_snr.append(np.max(y_p))
                
                detected, events, _, _  = calculate_event_recall_and_fpr(
                    y_t, y_p, calib_threshold,
                    run_config['window_size'],
                    run_config['step_size'],
                    run_config.get('fs_val', 14e6)
                )
                
                total_detected += detected
                total_events += events
                if threshold_systematic_available:
                    detected_minus, events_minus, _, _ = calculate_event_recall_and_fpr(
                        y_t, y_p, threshold_minus,
                        run_config['window_size'],
                        run_config['step_size'],
                        run_config.get('fs_val', 14e6),
                    )
                    detected_plus, events_plus, _, _ = calculate_event_recall_and_fpr(
                        y_t, y_p, threshold_plus,
                        run_config['window_size'],
                        run_config['step_size'],
                        run_config.get('fs_val', 14e6),
                    )
                    if events_minus != events or events_plus != events:
                        raise RuntimeError(
                            "Threshold replay changed the number of ground-truth "
                            "events; event bookkeeping is inconsistent."
                        )
                    total_detected_threshold_minus += detected_minus
                    total_detected_threshold_plus += detected_plus

                # Release the dataset before the next run reuses the same
                # memmap directory. We intentionally do not delete the backing
                # files here because tf.data may still hold worker threads.
                val_ds = _release_dataset_resources(dataset_obj=val_ds)

            # Avoid `clear_session()` inside the scan while we continue using
            # the same model object. Repeated teardown/reuse can destabilize
            # long inference jobs on macOS/Metal.
            _release_dataset_resources(clear_signal_cache=True)

            # Save max-score-per-run data for this SNR
            maxscore_df = pd.DataFrame({
                "run": np.arange(len(max_scores_this_snr)),
                "max_score": max_scores_this_snr,
                "snr": snr,
                "mass": mass,
                "detection_score_mode": detection_score_mode,
            })

            maxscore_outdir = os.path.join(output_dir, "max_scores_per_run")
            os.makedirs(maxscore_outdir, exist_ok=True)

            maxscore_df.to_csv(
                os.path.join(
                    maxscore_outdir,
                    f"max_scores_mass_{mass:.2e}_snr_{snr:.2f}.csv"
                ),
                index=False
            )

            # --- D: clustered FP diagnostics — SEPARATE CSV ONLY --- #
            # Main efficiency/SNR95 metrics below remain window-FP based.
            try:
                mass_solar_diag = mass / 1.988e30
                cluster_row = compute_clustered_fp_diagnostics_multi(
                    cluster_diag_runs,
                    threshold=calib_threshold,
                    window_size=preprocessing_config['window_size'],
                    step_size=preprocessing_config.get('step_size', 2048),
                    sampling_rate=preprocessing_config.get('fs_val', 14e6),
                    merge_gap_windows=clustered_fp_merge_gap_windows,
                    model_name=os.path.basename(os.path.normpath(model_path)),
                    mass=mass_solar_diag,
                    snr=float(snr),
                    threshold_source=(
                        "preset" if preset_threshold is not None else "calibrated"
                    ),
                )
                append_clustered_fp_row(
                    os.path.join(output_dir, "clustered_fp_diagnostics.csv"),
                    cluster_row,
                )
            except Exception as exc:
                logger.warning(f"Clustered FP diagnostics failed: {exc}")

            efficiency, low, high, sigma_val = _efficiency_stats(
                total_detected, total_events
            )
            if threshold_systematic_available:
                eff_threshold_minus, eff_threshold_minus_low, eff_threshold_minus_high, sigma_threshold_minus = (
                    _efficiency_stats(total_detected_threshold_minus, total_events)
                )
                eff_threshold_plus, eff_threshold_plus_low, eff_threshold_plus_high, sigma_threshold_plus = (
                    _efficiency_stats(total_detected_threshold_plus, total_events)
                )
            else:
                eff_threshold_minus = eff_threshold_minus_low = eff_threshold_minus_high = np.nan
                eff_threshold_plus = eff_threshold_plus_low = eff_threshold_plus_high = np.nan
                sigma_threshold_minus = sigma_threshold_plus = np.nan

            latent_summary = {}
            latent_df = None
            score_summary = {}
            if snr_latent_means:
                labels_all = np.concatenate(snr_labels, axis=0).reshape(-1).astype(bool)
                scores_all = np.concatenate(snr_scores, axis=0).reshape(-1)

                score_summary = {
                    "detection_score_mode": detection_score_mode,
                    "threshold_used": float(calib_threshold),
                    "signal_score_mean": float(np.mean(scores_all[labels_all])) if np.any(labels_all) else 0.0,
                    "noise_score_mean": float(np.mean(scores_all[~labels_all])) if np.any(~labels_all) else 0.0,
                    "signal_score_std": float(np.std(scores_all[labels_all])) if np.any(labels_all) else 0.0,
                    "noise_score_std": float(np.std(scores_all[~labels_all])) if np.any(~labels_all) else 0.0,
                }

                latent_summary, latent_df = summarise_latent_statistics(
                    np.concatenate(snr_latent_means, axis=0),
                    np.concatenate(snr_latent_logvars, axis=0),
                    labels_all,
                    np.concatenate(snr_logits, axis=0),
                )
                latent_csv_name = f"latent_stats_mass_{mass:.2e}_snr_{snr:.2f}.csv"
                latent_plot_name = f"latent_stats_mass_{mass:.2e}_snr_{snr:.2f}.png"
                latent_df.to_csv(os.path.join(latent_stats_dir, latent_csv_name), index=False)
                save_latent_statistics_plot(
                    latent_df,
                    os.path.join(latent_stats_dir, latent_plot_name),
                    title=f"Latent statistics | mass={mass / 1.988e30:.2e} M_solar | SNR={snr:.2f}",
                )
            
            logger.info(f"   SNR {snr}: Eff={efficiency:.4f} ({total_detected}/{total_events})")
            
            entry = {
                'mass': mass,
                'snr': snr,
                'efficiency': efficiency,
                'eff_ci_lower': low,
                'eff_ci_upper': high,
                'sigma_val': sigma_val,
                'detected': total_detected,
                'total_events': total_events,
                'threshold_used': float(calib_threshold),
                'threshold_err': threshold_err,
                'threshold_minus_1sigma': threshold_minus,
                'threshold_plus_1sigma': threshold_plus,
                'detected_threshold_minus_1sigma': total_detected_threshold_minus
                if threshold_systematic_available else np.nan,
                'detected_threshold_plus_1sigma': total_detected_threshold_plus
                if threshold_systematic_available else np.nan,
                'efficiency_threshold_minus_1sigma': eff_threshold_minus,
                'efficiency_threshold_minus_1sigma_ci_lower': eff_threshold_minus_low,
                'efficiency_threshold_minus_1sigma_ci_upper': eff_threshold_minus_high,
                'sigma_threshold_minus_1sigma': sigma_threshold_minus,
                'efficiency_threshold_plus_1sigma': eff_threshold_plus,
                'efficiency_threshold_plus_1sigma_ci_lower': eff_threshold_plus_low,
                'efficiency_threshold_plus_1sigma_ci_upper': eff_threshold_plus_high,
                'sigma_threshold_plus_1sigma': sigma_threshold_plus,
            }
            entry.update(score_summary)
            entry.update(latent_summary)
            results.append(entry)
            mass_results.append(entry)

        # 4. Fit Sigmoid for this Mass (Weighted)
        df_mass = pd.DataFrame(mass_results)
        df_mass['fit_midpoint'] = np.nan
        df_mass['fit_slope'] = np.nan
        df_mass['snr_at_target_eff'] = np.nan
        
        popt = None
        chi2_sig = np.nan
        chi2_red_sig = np.nan
        dof_sig = np.nan
        try:
            popt, pcov = curve_fit(
                sigmoid_func,
                df_mass['snr'],
                df_mass['efficiency'],
                p0=[5.0, 1.0],
                sigma=df_mass['sigma_val'],
                absolute_sigma=True,
                bounds=([0, 0.1], [50, 20]),
                maxfev=5000
            )

            # --- Chi^2 and reduced Chi^2 for sigmoid fit ---
            y_fit_vals = sigmoid_func(df_mass['snr'].values, *popt)
            residuals = df_mass['efficiency'].values - y_fit_vals
            chi2_sig = np.sum((residuals / df_mass['sigma_val'].values) ** 2)
            dof_sig = len(df_mass) - len(popt)
            chi2_red_sig = chi2_sig / dof_sig if dof_sig > 0 else np.nan

            # --- Store fit parameter uncertainties ---
            x0, k = popt
            sigma_x0 = np.sqrt(pcov[0, 0]) if pcov[0, 0] >= 0 else np.nan
            sigma_k  = np.sqrt(pcov[1, 1]) if pcov[1, 1] >= 0 else np.nan
            cov_x0_k = pcov[0, 1]

            # --- Uncertainty propagation for SNR at target efficiency ---
            A = np.log(1/target_efficiency_for_summary - 1)

            # Partial derivatives
            dS_dx0 = 1.0
            dS_dk = A / (k**2)

            # Covariance terms
            var_x0 = pcov[0, 0]
            var_k = pcov[1, 1]

            var_target_snr = (
                dS_dx0**2 * var_x0 +
                dS_dk**2 * var_k +
                2.0 * dS_dx0 * dS_dk * cov_x0_k
            )

            sigma_target_snr = np.sqrt(var_target_snr) if var_target_snr > 0 else np.nan

            target_snr = x0 - (1/k) * np.log(1/target_efficiency_for_summary - 1)

            logger.info(f"   Sigmoid Fit: Midpoint={x0:.2f}, Slope={k:.2f}")
            logger.info(f"   Required SNR for {target_efficiency_for_summary*100}%: {target_snr:.2f}")

            df_mass['fit_midpoint'] = x0
            df_mass['fit_slope'] = k
            df_mass['fit_midpoint_err'] = sigma_x0
            df_mass['fit_slope_err'] = sigma_k
            df_mass['fit_cov_x0_k'] = cov_x0_k
            df_mass['snr_at_target_eff'] = target_snr
            df_mass['snr_at_target_eff_err'] = sigma_target_snr
            df_mass['chi2_sigmoid'] = chi2_sig
            df_mass['chi2_red_sigmoid'] = chi2_red_sig

            for r in results:
                if r['mass'] == mass and r['snr'] in df_mass['snr'].values:
                    r['fit_midpoint'] = x0
                    r['fit_slope'] = k
                    r['fit_midpoint_err'] = sigma_x0
                    r['fit_slope_err'] = sigma_k
                    r['fit_cov_x0_k'] = cov_x0_k
                    r['snr_at_target_eff'] = target_snr
                    r['snr_at_target_eff_err'] = sigma_target_snr
                    r['chi2_sigmoid'] = chi2_sig
                    r['chi2_red_sigmoid'] = chi2_red_sig

        except Exception as e:
            logger.error(f"   Curve fit failed for mass {mass}: {e}")

        # Refit the complete efficiency curve at threshold +/- sigma_t. These
        # shifts are a coherent calibration systematic, not another binomial
        # statistical error, so they are stored separately and asymmetrically.
        fit_threshold_minus = _fit_efficiency_variant(
            df_mass,
            "efficiency_threshold_minus_1sigma",
            "sigma_threshold_minus_1sigma",
            target_efficiency_for_summary,
        )
        fit_threshold_plus = _fit_efficiency_variant(
            df_mass,
            "efficiency_threshold_plus_1sigma",
            "sigma_threshold_plus_1sigma",
            target_efficiency_for_summary,
        )
        nominal_target_snr = (
            float(df_mass["snr_at_target_eff"].iloc[0])
            if "snr_at_target_eff" in df_mass
            and len(df_mass)
            and np.isfinite(df_mass["snr_at_target_eff"].iloc[0])
            else np.nan
        )
        snr_target_threshold_minus = fit_threshold_minus["snr_at_target_eff"]
        snr_target_threshold_plus = fit_threshold_plus["snr_at_target_eff"]
        snr_sys_minus = (
            nominal_target_snr - snr_target_threshold_minus
            if np.isfinite(nominal_target_snr) and np.isfinite(snr_target_threshold_minus)
            else np.nan
        )
        snr_sys_plus = (
            snr_target_threshold_plus - nominal_target_snr
            if np.isfinite(nominal_target_snr) and np.isfinite(snr_target_threshold_plus)
            else np.nan
        )
        df_mass["snr_at_target_eff_threshold_minus_1sigma"] = snr_target_threshold_minus
        df_mass["snr_at_target_eff_threshold_plus_1sigma"] = snr_target_threshold_plus
        df_mass["snr_at_target_eff_threshold_sys_minus"] = snr_sys_minus
        df_mass["snr_at_target_eff_threshold_sys_plus"] = snr_sys_plus
        for result_row in results:
            if result_row["mass"] == mass:
                result_row["snr_at_target_eff_threshold_minus_1sigma"] = snr_target_threshold_minus
                result_row["snr_at_target_eff_threshold_plus_1sigma"] = snr_target_threshold_plus
                result_row["snr_at_target_eff_threshold_sys_minus"] = snr_sys_minus
                result_row["snr_at_target_eff_threshold_sys_plus"] = snr_sys_plus

        df_mass[[
            "mass", "snr", "threshold_used", "threshold_err",
            "threshold_minus_1sigma", "threshold_plus_1sigma",
            "efficiency", "efficiency_threshold_minus_1sigma",
            "efficiency_threshold_plus_1sigma",
        ]].to_csv(
            os.path.join(
                threshold_sys_dir,
                f"threshold_systematic_mass_{mass:.2e}.csv",
            ),
            index=False,
        )

        # --- SAVE INDIVIDUAL CSV --- #
        csv_filename = f"efficiency_curve_mass_{mass:.2e}.csv"
        df_mass.to_csv(os.path.join(s_curve_csv_dir, csv_filename), index=False)
        
        # --- PLOT INDIVIDUAL S-CURVE --- #
        plt.figure(figsize=(8, 5))
        lower_err = np.maximum(0, df_mass['efficiency'] - df_mass['eff_ci_lower'])
        upper_err = np.maximum(0, df_mass['eff_ci_upper'] - df_mass['efficiency'])
        
        plt.errorbar(
            df_mass['snr'], df_mass['efficiency'],
            yerr=[lower_err, upper_err],
            fmt='o', label='Data', color='black', capsize=3
        )
        if threshold_systematic_available:
            lower_band = np.minimum(
                df_mass["efficiency_threshold_minus_1sigma"],
                df_mass["efficiency_threshold_plus_1sigma"],
            )
            upper_band = np.maximum(
                df_mass["efficiency_threshold_minus_1sigma"],
                df_mass["efficiency_threshold_plus_1sigma"],
            )
            plt.fill_between(
                df_mass["snr"],
                lower_band,
                upper_band,
                color="tab:orange",
                alpha=0.25,
                label=r"Threshold fit $\pm 1\sigma$ systematic",
            )
        
        if popt is not None:
            x_fit = np.linspace(min(snr_list), max(snr_list), 100)
            y_fit = sigmoid_func(x_fit, *popt)
            plt.plot(
                x_fit,
                y_fit,
                'r--',
                label=(
                    f'Sigmoid Fit '
                    f'($\\chi^2_{{red}}={chi2_red_sig:.2f}$, dof={dof_sig})'
                )
            )
            plt.axhline(target_efficiency_for_summary, color='gray', linestyle=':', alpha=0.5, label=f'{target_efficiency_for_summary*100}% Eff')
            plt.axvline(df_mass['snr_at_target_eff'].iloc[0], color='gray', linestyle=':', alpha=0.5)

        plt.xlabel("SNR", size=18)
        plt.ylabel("Efficiency", size=18)
        plt.title(
            f"Efficiency Curve (Mass {mass:.2e} $M_{{\\odot}}$, score={detection_score_mode})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = f"efficiency_curve_mass_{mass:.2e}.png"
        plt.savefig(os.path.join(s_curve_plot_dir, plot_filename), dpi=300)
        plt.close()

        # --- Explicit cleanup to prevent memory accumulation across masses ---
        val_ds = _release_dataset_resources(dataset_obj=locals().get('val_ds'), clear_signal_cache=True)

    # 5. Global Results & Plots
    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(output_dir, "efficiency_data_raw_all.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    for mass in pbh_mass_list:
        subset = df_all[df_all['mass'] == mass]
        if subset.empty: continue
        
        lower_err = np.maximum(0, subset['efficiency'] - subset['eff_ci_lower'])
        upper_err = np.maximum(0, subset['eff_ci_upper'] - subset['efficiency'])
        
        plt.errorbar(
            subset['snr'], subset['efficiency'],
            yerr=[lower_err, upper_err],
            fmt='o', label=f'{mass:.1e} $M_{{\odot}}$'
        )
        
        if 'fit_midpoint' in subset.columns and not np.isnan(subset.iloc[0]['fit_midpoint']):
            x_fit = np.linspace(min(snr_list), max(snr_list), 100)
            mid = subset.iloc[0]['fit_midpoint']
            slope = subset.iloc[0]['fit_slope']
            y_fit = sigmoid_func(x_fit, mid, slope)
            plt.plot(
                x_fit,
                y_fit,
                '--',
                alpha=0.5,
                label=(
                    f'{mass:.1e} $M_{{\\odot}}$ fit '
                    f'($\\chi^2_{{red}}={subset.iloc[0]["chi2_red_sigmoid"]:.2f}$)'
                )
            )

    plt.xlabel("SNR", size=18)
    plt.ylabel("Detection Efficiency", size=18)
    plt.title(
        f"Efficiency vs SNR (All Masses, FP = {target_fp_per_year}/yr, score={detection_score_mode})"
    )
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_curves_combined.png"), dpi=300)
    plt.close()

    # --- Combined latent diagnostic curves --- #
    plt.figure(figsize=(10, 6))
    for mass in pbh_mass_list:
        subset = df_all[df_all['mass'] == mass]
        if subset.empty or 'latent_gap_l2' not in subset.columns:
            continue
        plt.plot(subset['snr'], subset['latent_gap_l2'], 'o-', label=f'{mass:.1e} $M_{{\\odot}}$')
    plt.xlabel("SNR", size=18)
    plt.ylabel("||mean(z_mean)_signal - mean(z_mean)_noise||_2", size=14)
    plt.title("Latent Mean Separation vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_mean_separation_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for mass in pbh_mass_list:
        subset = df_all[df_all['mass'] == mass]
        if subset.empty or 'mean_kl_all' not in subset.columns:
            continue
        plt.plot(subset['snr'], subset['mean_kl_all'], 'o-', label=f'{mass:.1e} $M_{{\\odot}}$')
    plt.xlabel("SNR", size=18)
    plt.ylabel("Mean per-dim KL", size=14)
    plt.title("Latent KL vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_mean_kl_curves.png"), dpi=300)
    plt.close()

    # --- Sensitivity Curve --- #
    summary_data = df_all.groupby('mass').first().reset_index()
    summary_data = summary_data.dropna(subset=['snr_at_target_eff'])

    if not summary_data.empty:
        X_data = np.log10(summary_data['mass'])
        Y_data = summary_data['snr_at_target_eff']
        Y_err = summary_data.get('snr_at_target_eff_err', None)

        plt.figure(figsize=(10, 6))
        if Y_err is not None and not Y_err.isna().all():
            plt.errorbar(
                summary_data['mass'],
                Y_data,
                yerr=Y_err,
                fmt='s',
                capsize=4,
                markersize=6,
                label=f'Data ({target_efficiency_for_summary*100:.1f}% eff)'
            )
        else:
            plt.semilogx(
                summary_data['mass'],
                Y_data,
                's',
                markersize=6,
                label=f'Data ({target_efficiency_for_summary*100:.1f}% eff)'
            )
        if {
            "snr_at_target_eff_threshold_sys_minus",
            "snr_at_target_eff_threshold_sys_plus",
        }.issubset(summary_data.columns):
            sys_minus = summary_data["snr_at_target_eff_threshold_sys_minus"]
            sys_plus = summary_data["snr_at_target_eff_threshold_sys_plus"]
            valid_sys = sys_minus.notna() & sys_plus.notna()
            if valid_sys.any():
                plt.fill_between(
                    summary_data.loc[valid_sys, "mass"],
                    (
                        summary_data.loc[valid_sys, "snr_at_target_eff"]
                        - sys_minus.loc[valid_sys]
                    ),
                    (
                        summary_data.loc[valid_sys, "snr_at_target_eff"]
                        + sys_plus.loc[valid_sys]
                    ),
                    color="tab:orange",
                    alpha=0.25,
                    label=r"Threshold fit $\pm 1\sigma$ systematic",
                )

        plt.xlabel("PBH Mass ($M_{\odot}$)")
        plt.ylabel(f"SNR required for {target_efficiency_for_summary*100}% Efficiency")
        plt.xscale('log')
        plt.title(
            f"Model Sensitivity Curve (FP < {target_fp_per_year}/yr, score={detection_score_mode})"
        )
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "sensitivity_curve.png"), dpi=300)
        plt.close()

        summary_data[
            [
                'mass',
                'snr_at_target_eff',
                'snr_at_target_eff_err',
                'snr_at_target_eff_threshold_minus_1sigma',
                'snr_at_target_eff_threshold_plus_1sigma',
                'snr_at_target_eff_threshold_sys_minus',
                'snr_at_target_eff_threshold_sys_plus',
                'threshold_used',
                'threshold_err',
                'fit_midpoint',
                'fit_midpoint_err',
                'fit_slope',
                'fit_slope_err',
                'fit_cov_x0_k',
                'chi2_sigmoid',
                'chi2_red_sigmoid'
            ]
        ].to_csv(
            os.path.join(output_dir, "sensitivity_summary.csv"), index=False
        )
    else:
        logger.warning("No valid data for Sensitivity Curve plotting.")

    # Remove the memmap directory only once the full workflow is done.
    _release_dataset_resources(clear_signal_cache=True, clear_tf_session=True)
    if os.path.exists(temp_proc_dir):
        shutil.rmtree(temp_proc_dir)
    
    logger.info("Processing Complete. Results saved.")
    return {
        "raw_df": df_all,
        "summary_df": summary_data.copy() if not summary_data.empty else pd.DataFrame(),
        "output_dir": output_dir,
        "threshold": float(calib_threshold),
        "threshold_err": threshold_err,
        "calibration_result": calibration_result,
        "score_label": score_label,
        "detection_score_mode": detection_score_mode,
    }


def generate_fp_rate_scan(
    model_path,
    normalization_params,
    normalization_mode,
    test_file_suffixes,
    pbh_mass_list,
    snr_list,
    preprocessing_config,
    output_dir=os.path.join(_PROJECT_ROOT, "fp_rate_scan_results"),
    fp_target_values=None,
    fp_scan_range=None,
    num_runs_per_point=10,
    target_efficiency_for_summary=0.95,
    num_samples_threshold_calibration=None,
    detection_score_mode="logit",
    detection_score_config=None,
    manual_linear_fit=None,
    calibration_points_csv=None,
):
    """
    Scan many FP/year targets from one shared threshold calibration fit.

    The key design choice is that the calibration fit is computed only once.
    From that fit we derive every threshold in the FP scan, and then we reuse
    the same per-window detector scores for all thresholds at a given mass/SNR.
    """
    detection_score_mode = _normalise_detection_score_mode(detection_score_mode)
    detection_score_config = dict(detection_score_config or {})
    fp_targets = _build_fp_target_list(
        fp_target_values=fp_target_values,
        fp_scan_range=fp_scan_range,
    )
    target_eff_label = f"{target_efficiency_for_summary * 100:.1f}%"
    target_eff_tag = f"eff{target_efficiency_for_summary * 100:.1f}".replace(".", "p")

    os.makedirs(output_dir, exist_ok=True)
    model_path, preprocessing_config, normalization_params, _ = _align_model_and_preprocessing(
        model_path,
        preprocessing_config,
        normalization_params,
        normalization_mode,
    )
    preprocessing_config["normalization_type"] = normalization_mode
    if normalization_mode == "zscore" and normalization_params:
        preprocessing_config["global_mean_input"] = normalization_params.get("mean_value")
        preprocessing_config["global_std_input"] = normalization_params.get("std_dev_value")
    elif normalization_mode == "min_max" and normalization_params:
        preprocessing_config["global_min_input"] = normalization_params.get("min_value")
        preprocessing_config["global_max_input"] = normalization_params.get("max_value")

    threshold_table_dir = os.path.join(output_dir, "threshold_tables")
    per_mass_root_dir = os.path.join(output_dir, "per_mass")
    os.makedirs(threshold_table_dir, exist_ok=True)
    os.makedirs(per_mass_root_dir, exist_ok=True)

    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    model_window = _get_model_window_size(model)
    cfg_window = preprocessing_config["window_size"]
    filepath_suffixes = preprocessing_config["filepath_suffixes"]
    if model_window is not None and model_window != cfg_window:
        raise ValueError(
            f"Window size mismatch: model was built with {model_window} "
            f"time steps, but config has window_size={cfg_window}. "
            "Use the matching window size or retrain the model."
        )

    # We calibrate once and reuse that one fit for the full FP scan.
    calibration_result = calibrate_threshold(
        model,
        preprocessing_config,
        target_fp_per_year=float(fp_targets[0]),
        calibration_files=test_file_suffixes,
        temp_dir=os.path.join(output_dir, "temp_calib"),
        num_samples_threshold_calibration=num_samples_threshold_calibration,
        detection_score_mode=detection_score_mode,
        detection_score_config=detection_score_config,
        return_details=True,
        manual_linear_fit=manual_linear_fit,
        calibration_points_csv=calibration_points_csv,
        require_linear_fit=True,
    )
    score_context = calibration_result["score_context"]
    score_label = calibration_result["score_label"]

    threshold_table = _derive_threshold_table_from_fit(calibration_result, fp_targets)
    threshold_table.to_csv(
        os.path.join(threshold_table_dir, "fp_scan_threshold_table.csv"),
        index=False,
    )

    # This plot makes the one-fit-many-threshold mapping easy to inspect.
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        threshold_table["target_fp_per_year"],
        threshold_table["threshold"],
        yerr=threshold_table["threshold_err"]
        if threshold_table["threshold_err"].notna().any() else None,
        fmt="o-",
        capsize=3,
    )
    plt.xscale("log")
    plt.xlabel("Target FP/year", size=16)
    plt.ylabel("Threshold", size=16)
    plt.title("Threshold Derived From Shared FP Fit")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(threshold_table_dir, "threshold_vs_fp_from_fit.png"),
        dpi=300,
    )
    plt.close()

    with open(os.path.join(output_dir, "fp_scan_info.txt"), "w") as handle:
        masses_kg = np.asarray(pbh_mass_list, dtype=float)
        masses_solar = masses_kg / float(preprocessing_config.get("M_solar", 1.988e30))
        handle.write(f"Model: {model_path}\n")
        handle.write(f"Detection Score Mode: {detection_score_mode}\n")
        handle.write(f"Detection Score Label: {score_label}\n")
        handle.write(f"Detection Score Config: {detection_score_config}\n")
        handle.write(f"Target Efficiency Summary Level: {target_efficiency_for_summary}\n")
        handle.write(f"PBH Masses (kg): {list(masses_kg)}\n")
        handle.write(f"PBH Masses (solar): {list(masses_solar)}\n")
        handle.write(f"Fit Source: {calibration_result['fit_source']}\n")
        handle.write(f"Fit Slope: {calibration_result['fit_slope']}\n")
        handle.write(f"Fit Intercept: {calibration_result['fit_intercept']}\n")
        handle.write(f"Fit Covariance [slope, intercept]: {calibration_result['fit_covariance']}\n")
        handle.write(f"Fit Slope Error: {calibration_result['fit_slope_err']}\n")
        handle.write(f"Fit Intercept Error: {calibration_result['fit_intercept_err']}\n")
        handle.write(f"Fit Cov(slope, intercept): {calibration_result['fit_cov_slope_intercept']}\n")
        handle.write(f"Fit chi2_red: {calibration_result['fit_chi2_red']}\n")
        handle.write(f"Windows/Year: {calibration_result['windows_per_year']}\n")
        handle.write(
            "Threshold uncertainty interpretation: calibration-fit statistical "
            "uncertainty propagated as a correlated nuisance across the efficiency curve.\n"
        )
        handle.write(
            "Covariance limitation: the fitted slope/intercept covariance uses "
            "curve_fit's independent-point approximation, although threshold-sweep "
            "tail counts are nested and therefore correlated.\n"
        )

    logger.info("Step 2/3: Computing FP scan efficiency grid...")
    results = []
    summary_rows = []
    temp_proc_dir = os.path.join(output_dir, "temp_proc")

    for mass in pbh_mass_list:
        mass_solar = mass / float(preprocessing_config.get("M_solar", 1.988e30))
        mass_tag = _format_float_tag(mass_solar)
        mass_dir = os.path.join(per_mass_root_dir, f"mass_{mass_tag}_Msol")
        latent_stats_dir = os.path.join(mass_dir, "latent_statistics")
        per_fp_curve_dir = os.path.join(mass_dir, "per_fp_efficiency_curves")
        maxscore_outdir = os.path.join(mass_dir, "max_scores_per_run")
        os.makedirs(mass_dir, exist_ok=True)
        os.makedirs(latent_stats_dir, exist_ok=True)
        os.makedirs(per_fp_curve_dir, exist_ok=True)
        os.makedirs(maxscore_outdir, exist_ok=True)
        logger.info(f"--- FP Scan for Mass: {mass_solar:.2e} M_solar ({mass:.2e} kg) ---")
        mass_rows = []

        with open(os.path.join(mass_dir, "mass_info.txt"), "w") as handle:
            handle.write(f"Mass (kg): {mass:.16e}\n")
            handle.write(f"Mass (solar): {mass_solar:.16e}\n")
            handle.write(f"Detection Score Mode: {detection_score_mode}\n")
            handle.write(f"Detection Score Label: {score_label}\n")
            handle.write(f"Target Efficiency Summary Level: {target_efficiency_for_summary}\n")
            handle.write(f"FP Targets (/year): {list(np.asarray(fp_targets, dtype=float))}\n")

        for snr in snr_list:
            # Keep one accumulator per target FP so we can reuse the same window
            # scores for every threshold derived from the shared calibration fit.
            threshold_aggregates = {
                float(row.target_fp_per_year): {
                    "detected": 0,
                    "detected_threshold_minus": 0,
                    "detected_threshold_plus": 0,
                    "events": 0,
                    "threshold": float(row.threshold),
                    "threshold_err": float(row.threshold_err),
                    "threshold_minus": float(row.threshold_minus_1sigma),
                    "threshold_plus": float(row.threshold_plus_1sigma),
                    "target_fpr_fraction": float(row.target_fpr_fraction),
                }
                for row in threshold_table.itertuples(index=False)
            }

            max_scores_this_snr = []
            snr_latent_means = []
            snr_latent_logvars = []
            snr_labels = []
            snr_logits = []
            snr_scores = []

            for run_i in range(num_runs_per_point):
                run_config = preprocessing_config.copy()
                run_config.pop("test_file_suffixes", None)
                run_config.pop("num_samples_to_read_per_file_threshold_calibration", None)
                run_config.update({
                    "filepath_suffixes": filepath_suffixes,
                    "memmap_dir": temp_proc_dir,
                    "inject_signals": True,
                    "snr_based_injection": True,
                    "m_PBH_injection_list": [mass],
                    "amplitude_spectrum_range": [snr],
                    "normalization_type": normalization_mode,
                    "custom_noise_std": preprocessing_config.get("custom_noise_std"),
                    "return_tf_datasets": True,
                    "tf_batch_size": 4096,
                    "tf_repeat": False,
                    "tf_shuffle": False,
                    "global_std_input": preprocessing_config.get("global_std_input"),
                })

                _, val_ds, _ = pre_processing_with_memmap(**run_config)

                if run_i == 0:
                    for x_b, _ in val_ds.take(1):
                        logger.info(f"First batch shape from dataset: {x_b.shape}")

                y_score_list = []
                y_true_list = []
                for x_b, y_b in val_ds:
                    logits_batch = _predict_logits(
                        model,
                        x_b,
                        classifier_inference_mode=detection_score_config.get(
                            "classifier_inference_mode", "sampled_z"
                        ),
                    )
                    z_mean_batch, z_logvar_batch = _latent_outputs(model, x_b)
                    scores_batch = _compute_detection_score(
                        logits_batch,
                        z_mean_batch,
                        z_logvar_batch,
                        detection_score_mode=detection_score_mode,
                        score_context=score_context,
                        detection_score_config=detection_score_config,
                    )
                    labels_batch = y_b.numpy().reshape(-1)

                    y_score_list.append(scores_batch)
                    y_true_list.append(labels_batch)
                    snr_logits.append(logits_batch)
                    snr_labels.append(labels_batch)
                    snr_scores.append(scores_batch)
                    snr_latent_means.append(z_mean_batch)
                    snr_latent_logvars.append(z_logvar_batch)

                y_scores = np.concatenate(y_score_list)
                y_true = np.concatenate(y_true_list)
                max_scores_this_snr.append(np.max(y_scores))

                # The event-based efficiency logic is intentionally unchanged.
                # We simply replay it for every threshold in the shared table.
                threshold_eval_rows = _evaluate_threshold_table_on_scores(
                    y_true,
                    y_scores,
                    threshold_table,
                    run_config["window_size"],
                    run_config["step_size"],
                    run_config.get("fs_val", 14e6),
                )
                for eval_row in threshold_eval_rows:
                    target_fp = eval_row["target_fp_per_year"]
                    threshold_aggregates[target_fp]["detected"] += eval_row["detected"]
                    threshold_aggregates[target_fp]["events"] += eval_row["events"]
                main_events_by_fp = {
                    row["target_fp_per_year"]: row["events"]
                    for row in threshold_eval_rows
                }
                for threshold_row in threshold_table.itertuples(index=False):
                    if not np.isfinite(threshold_row.threshold_err):
                        continue
                    detected_minus, events_minus, _, _ = calculate_event_recall_and_fpr(
                        y_true, y_scores, threshold_row.threshold_minus_1sigma,
                        run_config["window_size"], run_config["step_size"],
                        run_config.get("fs_val", 14e6),
                    )
                    detected_plus, events_plus, _, _ = calculate_event_recall_and_fpr(
                        y_true, y_scores, threshold_row.threshold_plus_1sigma,
                        run_config["window_size"], run_config["step_size"],
                        run_config.get("fs_val", 14e6),
                    )
                    target_fp = float(threshold_row.target_fp_per_year)
                    if (
                        events_minus != main_events_by_fp[target_fp]
                        or events_plus != main_events_by_fp[target_fp]
                    ):
                        raise RuntimeError(
                            "Threshold replay changed the number of ground-truth "
                            "events during FP scan."
                        )
                    threshold_aggregates[target_fp]["detected_threshold_minus"] += detected_minus
                    threshold_aggregates[target_fp]["detected_threshold_plus"] += detected_plus

                # Release the dataset before the next run reuses the same
                # memmap directory. We intentionally keep the files around
                # until the full scan is done to avoid use-after-delete issues
                # in background tf.data workers.
                val_ds = _release_dataset_resources(dataset_obj=val_ds)

            # Avoid `clear_session()` while we continue using the same loaded
            # model. Repeated teardown/reuse can crash long-running inference
            # loops on macOS/Metal.
            _release_dataset_resources(clear_signal_cache=True)

            maxscore_df = pd.DataFrame({
                "run": np.arange(len(max_scores_this_snr)),
                "max_score": max_scores_this_snr,
                "snr": snr,
                "mass": mass,
                "detection_score_mode": detection_score_mode,
            })
            maxscore_df.to_csv(
                os.path.join(
                    maxscore_outdir,
                    f"max_scores_snr_{snr:.2f}.csv",
                ),
                index=False,
            )

            latent_summary = {}
            latent_df = None
            score_summary = {}
            if snr_latent_means:
                # These summaries do not depend on the chosen threshold, so we
                # compute them once per mass/SNR operating point.
                labels_all = np.concatenate(snr_labels, axis=0).reshape(-1).astype(bool)
                scores_all = np.concatenate(snr_scores, axis=0).reshape(-1)

                score_summary = {
                    "detection_score_mode": detection_score_mode,
                    "signal_score_mean": float(np.mean(scores_all[labels_all])) if np.any(labels_all) else 0.0,
                    "noise_score_mean": float(np.mean(scores_all[~labels_all])) if np.any(~labels_all) else 0.0,
                    "signal_score_std": float(np.std(scores_all[labels_all])) if np.any(labels_all) else 0.0,
                    "noise_score_std": float(np.std(scores_all[~labels_all])) if np.any(~labels_all) else 0.0,
                }

                latent_summary, latent_df = summarise_latent_statistics(
                    np.concatenate(snr_latent_means, axis=0),
                    np.concatenate(snr_latent_logvars, axis=0),
                    labels_all,
                    np.concatenate(snr_logits, axis=0),
                )
                latent_csv_name = f"latent_stats_snr_{snr:.2f}.csv"
                latent_plot_name = f"latent_stats_snr_{snr:.2f}.png"
                latent_df.to_csv(os.path.join(latent_stats_dir, latent_csv_name), index=False)
                save_latent_statistics_plot(
                    latent_df,
                    os.path.join(latent_stats_dir, latent_plot_name),
                    title=f"Latent statistics | mass={mass_solar:.2e} $M_{{\\odot}}$ | SNR={snr:.2f}",
                )

            for target_fp, agg in threshold_aggregates.items():
                efficiency, low, high, sigma_val = _efficiency_stats(
                    agg["detected"], agg["events"]
                )
                if np.isfinite(agg["threshold_err"]):
                    eff_minus, eff_minus_low, eff_minus_high, sigma_minus = _efficiency_stats(
                        agg["detected_threshold_minus"], agg["events"]
                    )
                    eff_plus, eff_plus_low, eff_plus_high, sigma_plus = _efficiency_stats(
                        agg["detected_threshold_plus"], agg["events"]
                    )
                else:
                    eff_minus = eff_minus_low = eff_minus_high = sigma_minus = np.nan
                    eff_plus = eff_plus_low = eff_plus_high = sigma_plus = np.nan

                entry = {
                    "mass": mass,
                    "snr": snr,
                    "target_fp_per_year": target_fp,
                    "target_fpr_fraction": agg["target_fpr_fraction"],
                    "threshold": agg["threshold"],
                    "threshold_err": agg["threshold_err"],
                    "threshold_minus_1sigma": agg["threshold_minus"],
                    "threshold_plus_1sigma": agg["threshold_plus"],
                    "efficiency": efficiency,
                    "eff_ci_lower": low,
                    "eff_ci_upper": high,
                    "sigma_val": sigma_val,
                    "detected": agg["detected"],
                    "detected_threshold_minus_1sigma": agg["detected_threshold_minus"]
                    if np.isfinite(agg["threshold_err"]) else np.nan,
                    "detected_threshold_plus_1sigma": agg["detected_threshold_plus"]
                    if np.isfinite(agg["threshold_err"]) else np.nan,
                    "total_events": agg["events"],
                    "efficiency_threshold_minus_1sigma": eff_minus,
                    "efficiency_threshold_minus_1sigma_ci_lower": eff_minus_low,
                    "efficiency_threshold_minus_1sigma_ci_upper": eff_minus_high,
                    "sigma_threshold_minus_1sigma": sigma_minus,
                    "efficiency_threshold_plus_1sigma": eff_plus,
                    "efficiency_threshold_plus_1sigma_ci_lower": eff_plus_low,
                    "efficiency_threshold_plus_1sigma_ci_upper": eff_plus_high,
                    "sigma_threshold_plus_1sigma": sigma_plus,
                    "fit_slope": calibration_result["fit_slope"],
                    "fit_intercept": calibration_result["fit_intercept"],
                    "fit_chi2_red": calibration_result["fit_chi2_red"],
                }
                entry.update(score_summary)
                entry.update(latent_summary)
                results.append(entry)
                mass_rows.append(entry)

        df_mass_all = pd.DataFrame(mass_rows)
        df_mass_all.to_csv(
            os.path.join(mass_dir, "fp_scan_raw.csv"),
            index=False,
        )

        mass_summary_rows = []
        for target_fp in fp_targets:
            # Build one efficiency-vs-SNR curve for this specific FP budget and
            # extract the requested summary point from the sigmoid fit.
            df_curve = df_mass_all[
                np.isclose(df_mass_all["target_fp_per_year"], target_fp)
            ].sort_values("snr").copy()
            fp_tag = _format_float_tag(target_fp)
            df_curve.to_csv(
                os.path.join(
                    per_fp_curve_dir,
                    f"efficiency_curve_fp_{fp_tag}.csv",
                ),
                index=False,
            )
            fit_summary = _fit_efficiency_curve_summary(
                df_curve,
                target_efficiency_for_summary=target_efficiency_for_summary,
            )
            fit_threshold_minus = _fit_efficiency_variant(
                df_curve,
                "efficiency_threshold_minus_1sigma",
                "sigma_threshold_minus_1sigma",
                target_efficiency_for_summary,
            )
            fit_threshold_plus = _fit_efficiency_variant(
                df_curve,
                "efficiency_threshold_plus_1sigma",
                "sigma_threshold_plus_1sigma",
                target_efficiency_for_summary,
            )
            nominal_snr = fit_summary["snr_at_target_eff"]
            threshold_minus_snr = fit_threshold_minus["snr_at_target_eff"]
            threshold_plus_snr = fit_threshold_plus["snr_at_target_eff"]
            fit_summary["snr_at_target_eff_threshold_minus_1sigma"] = threshold_minus_snr
            fit_summary["snr_at_target_eff_threshold_plus_1sigma"] = threshold_plus_snr
            fit_summary["snr_at_target_eff_threshold_sys_minus"] = (
                nominal_snr - threshold_minus_snr
                if np.isfinite(nominal_snr) and np.isfinite(threshold_minus_snr)
                else np.nan
            )
            fit_summary["snr_at_target_eff_threshold_sys_plus"] = (
                threshold_plus_snr - nominal_snr
                if np.isfinite(nominal_snr) and np.isfinite(threshold_plus_snr)
                else np.nan
            )
            if fit_summary["fit_success"]:
                logger.info(
                    f"Mass {mass_solar:.2e} M_solar, FP/year {target_fp:.3e}: "
                    f"SNR@{target_efficiency_for_summary*100:.1f}% = "
                    f"{fit_summary['snr_at_target_eff']:.3f}"
                )

            # Save the actual efficiency curve for this FP target. This makes
            # it easy to inspect whether the flat SNR@95 summary comes from the
            # underlying data or from the summary fit.
            plt.figure(figsize=(8, 5))
            lower_err = np.maximum(0, df_curve["efficiency"] - df_curve["eff_ci_lower"])
            upper_err = np.maximum(0, df_curve["eff_ci_upper"] - df_curve["efficiency"])
            plt.errorbar(
                df_curve["snr"],
                df_curve["efficiency"],
                yerr=[lower_err, upper_err],
                fmt="o",
                color="black",
                capsize=3,
                label="Data",
            )
            if df_curve["efficiency_threshold_minus_1sigma"].notna().any():
                lower_band = np.minimum(
                    df_curve["efficiency_threshold_minus_1sigma"],
                    df_curve["efficiency_threshold_plus_1sigma"],
                )
                upper_band = np.maximum(
                    df_curve["efficiency_threshold_minus_1sigma"],
                    df_curve["efficiency_threshold_plus_1sigma"],
                )
                plt.fill_between(
                    df_curve["snr"],
                    lower_band,
                    upper_band,
                    color="tab:orange",
                    alpha=0.25,
                    label=r"Threshold fit $\pm 1\sigma$ systematic",
                )

            if fit_summary["fit_success"]:
                x_fit = np.linspace(float(np.min(snr_list)), float(np.max(snr_list)), 200)
                y_fit = sigmoid_func(
                    x_fit,
                    fit_summary["fit_midpoint"],
                    fit_summary["fit_slope"],
                )
                plt.plot(
                    x_fit,
                    y_fit,
                    "r--",
                    label=(
                        "Sigmoid Fit "
                        f"($\\chi^2_{{red}}={fit_summary['chi2_red_sigmoid']:.2f}$)"
                    ),
                )
                plt.axhline(
                    target_efficiency_for_summary,
                    color="gray",
                    linestyle=":",
                    alpha=0.5,
                    label=f"{target_eff_label} Eff",
                )
                if np.isfinite(fit_summary["snr_at_target_eff"]):
                    plt.axvline(
                        fit_summary["snr_at_target_eff"],
                        color="gray",
                        linestyle=":",
                        alpha=0.5,
                    )

            plt.xlabel("SNR", size=16)
            plt.ylabel("Efficiency", size=16)
            plt.title(
                f"Efficiency Curve | mass={mass_solar:.2e} $M_{{\\odot}}$ | FP/year={target_fp:g}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    per_fp_curve_dir,
                    f"efficiency_curve_fp_{fp_tag}.png",
                ),
                dpi=300,
            )
            plt.close()

            summary_entry = {
                "mass": mass,
                "target_fp_per_year": float(target_fp),
                "threshold": float(df_curve["threshold"].iloc[0]) if not df_curve.empty else np.nan,
                "threshold_err": float(df_curve["threshold_err"].iloc[0]) if not df_curve.empty else np.nan,
                "target_fpr_fraction": float(df_curve["target_fpr_fraction"].iloc[0]) if not df_curve.empty else np.nan,
            }
            summary_entry.update(fit_summary)
            mass_summary_rows.append(summary_entry)
            summary_rows.append(summary_entry)

        df_mass_summary = pd.DataFrame(mass_summary_rows).sort_values("target_fp_per_year")
        df_mass_summary.to_csv(
            os.path.join(mass_dir, "fp_scan_summary.csv"),
            index=False,
        )

        plt.figure(figsize=(9, 5))
        subset = df_mass_summary.dropna(subset=["snr_at_target_eff"])
        if not subset.empty:
            y_err = subset["snr_at_target_eff_err"]
            if not y_err.isna().all():
                plt.errorbar(
                    subset["target_fp_per_year"],
                    subset["snr_at_target_eff"],
                    yerr=y_err,
                    fmt="o-",
                    capsize=3,
                )
            else:
                plt.plot(
                    subset["target_fp_per_year"],
                    subset["snr_at_target_eff"],
                    "o-",
                )
            if {
                "snr_at_target_eff_threshold_sys_minus",
                "snr_at_target_eff_threshold_sys_plus",
            }.issubset(subset.columns):
                valid_sys = (
                    subset["snr_at_target_eff_threshold_sys_minus"].notna()
                    & subset["snr_at_target_eff_threshold_sys_plus"].notna()
                )
                if valid_sys.any():
                    plt.fill_between(
                        subset.loc[valid_sys, "target_fp_per_year"],
                        (
                            subset.loc[valid_sys, "snr_at_target_eff"]
                            - subset.loc[valid_sys, "snr_at_target_eff_threshold_sys_minus"]
                        ),
                        (
                            subset.loc[valid_sys, "snr_at_target_eff"]
                            + subset.loc[valid_sys, "snr_at_target_eff_threshold_sys_plus"]
                        ),
                        color="tab:orange",
                        alpha=0.25,
                        label=r"Threshold fit $\pm 1\sigma$ systematic",
                    )
        plt.xscale("log")
        plt.xlabel("Target FP/year", size=16)
        plt.ylabel(
            f"SNR required for {target_eff_label} efficiency",
            size=16,
        )
        plt.title(
            f"SNR@{target_eff_label} vs FP/year (Mass {mass_solar:.2e} $M_{{\\odot}}$)"
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                mass_dir,
                f"snr_at_{target_eff_tag}_vs_fp.png",
            ),
            dpi=300,
        )
        plt.close()

    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(output_dir, "fp_scan_efficiency_raw_all.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["mass", "target_fp_per_year"])
    summary_df.to_csv(
        os.path.join(output_dir, f"fp_scan_snr_at_{target_eff_tag}_summary_all.csv"),
        index=False,
    )

    plt.figure(figsize=(10, 6))
    for mass in pbh_mass_list:
        mass_solar = mass / float(preprocessing_config.get("M_solar", 1.988e30))
        subset = summary_df[
            (summary_df["mass"] == mass)
            & (~summary_df["snr_at_target_eff"].isna())
        ].sort_values("target_fp_per_year")
        if subset.empty:
            continue
        plt.plot(
            subset["target_fp_per_year"],
            subset["snr_at_target_eff"],
            "o-",
            label=f"{mass_solar:.1e} $M_{{\\odot}}$",
        )
    plt.xscale("log")
    plt.xlabel("Target FP/year", size=18)
    plt.ylabel(
        f"SNR required for {target_eff_label} efficiency",
        size=18,
    )
    plt.title(f"SNR@{target_eff_label} vs FP/year (All Masses)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"snr_at_{target_eff_tag}_vs_fp_all_masses.png"),
        dpi=300,
    )
    plt.close()

    _release_dataset_resources(clear_signal_cache=True, clear_tf_session=True)
    if os.path.exists(temp_proc_dir):
        shutil.rmtree(temp_proc_dir)

    logger.info("FP rate scan processing complete.")
    return {
        "raw_df": df_all,
        "summary_df": summary_df,
        "threshold_table": threshold_table,
        "calibration_result": calibration_result,
        "output_dir": output_dir,
    }


def _get_score_spec_name(detection_score_mode, detection_score_config=None, name=None):
    """Build a filesystem-friendly detector score name."""
    if name:
        return str(name)
    mode = _normalise_detection_score_mode(detection_score_mode)
    config = detection_score_config or {}
    if mode == "logit":
        return "logit"
    if mode == "latent_dim_abs_zscore":
        return f"latent_dim_{int(config.get('latent_dim', 0))}"
    if mode == "latent_selected_zscore":
        dims = config.get("selected_latent_dims", [])
        dims_str = "_".join(str(int(dim)) for dim in dims)
        reduction = str(config.get("reduction", "l2")).lower()
        return f"latent_selected_{reduction}_{dims_str}" if dims_str else f"latent_selected_{reduction}"
    if mode == "latent_total_kl":
        dims = config.get("selected_latent_dims")
        if dims is None:
            return "latent_total_kl"
        dims_str = "_".join(str(int(dim)) for dim in dims)
        return f"latent_total_kl_{dims_str}"
    return mode


def _get_score_spec_label(detection_score_mode, detection_score_config=None, label=None):
    """Build a compact legend label for a detector score."""
    if label:
        return str(label)
    mode = _normalise_detection_score_mode(detection_score_mode)
    config = detection_score_config or {}
    if mode == "logit":
        return "logit"
    if mode == "latent_dim_abs_zscore":
        return f"latent dim {int(config.get('latent_dim', 0))}"
    if mode == "latent_selected_zscore":
        dims = config.get("selected_latent_dims", [])
        reduction = str(config.get("reduction", "l2")).lower()
        return f"latent dims {dims} ({reduction})"
    if mode == "latent_total_kl":
        dims = config.get("selected_latent_dims")
        if dims is None:
            return "latent total KL"
        return f"latent KL {dims}"
    return mode


def _build_default_comparison_score_specs(
    include_logit=True,
    latent_dims=None,
    latent_use_abs=True,
    preset_thresholds=None,
    shared_detection_score_config=None,
):
    """Build a default logit-vs-latent-dims detector comparison spec list."""
    preset_thresholds = preset_thresholds or {}
    shared_detection_score_config = dict(shared_detection_score_config or {})
    specs = []

    if include_logit:
        specs.append({
            "name": "logit",
            "label": "logit",
            "detection_score_mode": "logit",
            "detection_score_config": {},
            "preset_threshold": preset_thresholds.get("logit"),
        })

    for dim in latent_dims or []:
        spec_name = f"latent_dim_{int(dim)}"
        spec_config = shared_detection_score_config.copy()
        spec_config.update({
            "latent_dim": int(dim),
            "use_abs": bool(latent_use_abs),
        })
        specs.append({
            "name": spec_name,
            "label": f"latent dim {int(dim)}",
            "detection_score_mode": "latent_dim_abs_zscore",
            "detection_score_config": spec_config,
            "preset_threshold": preset_thresholds.get(spec_name),
        })

    return specs


def generate_efficiency_curve_comparison(
    model_path,
    normalization_params,
    normalization_mode,
    test_file_suffixes,
    pbh_mass_list,
    snr_list,
    preprocessing_config,
    comparison_score_specs,
    output_dir=os.path.join(_PROJECT_ROOT, "efficiency_score_comparison"),
    target_fp_per_year=1.0,
    num_runs_per_point=10,
    target_efficiency_for_summary=0.95,
    num_samples_threshold_calibration=None,
):
    """
    Run multiple detector-score variants and save merged comparison outputs.

    This is intentionally a wrapper around the single-detector pipeline so the
    event-efficiency logic stays identical for every detector variant.
    """
    if not comparison_score_specs:
        raise ValueError("comparison_score_specs must contain at least one detector spec.")

    os.makedirs(output_dir, exist_ok=True)

    comparison_frames = []
    comparison_summary_frames = []
    threshold_rows = []

    for spec in comparison_score_specs:
        spec_mode = _normalise_detection_score_mode(spec.get("detection_score_mode", "logit"))
        spec_config = dict(spec.get("detection_score_config", {}))
        spec_name = _get_score_spec_name(spec_mode, spec_config, name=spec.get("name"))
        spec_label = _get_score_spec_label(spec_mode, spec_config, label=spec.get("label"))
        spec_threshold = spec.get("preset_threshold")
        spec_output_dir = os.path.join(output_dir, spec_name)

        logger.info(
            f"=== Detector Comparison: {spec_label} "
            f"(mode={spec_mode}, preset_threshold={spec_threshold}) ==="
        )

        run_result = generate_efficiency_curves(
            model_path,
            normalization_params,
            normalization_mode,
            test_file_suffixes,
            pbh_mass_list,
            snr_list,
            preprocessing_config,
            output_dir=spec_output_dir,
            target_fp_per_year=target_fp_per_year,
            num_runs_per_point=num_runs_per_point,
            target_efficiency_for_summary=target_efficiency_for_summary,
            num_samples_threshold_calibration=num_samples_threshold_calibration,
            detection_score_mode=spec_mode,
            detection_score_config=spec_config,
            preset_threshold=spec_threshold,
        )

        raw_df = run_result["raw_df"].copy()
        raw_df["score_name"] = spec_name
        raw_df["score_label"] = spec_label
        raw_df["score_mode"] = spec_mode
        comparison_frames.append(raw_df)

        summary_df = run_result["summary_df"].copy()
        if not summary_df.empty:
            summary_df["score_name"] = spec_name
            summary_df["score_label"] = spec_label
            summary_df["score_mode"] = spec_mode
            summary_df["threshold"] = run_result["threshold"]
            summary_df["threshold_err"] = run_result["threshold_err"]
            comparison_summary_frames.append(summary_df)

        threshold_rows.append({
            "score_name": spec_name,
            "score_label": spec_label,
            "score_mode": spec_mode,
            "threshold": run_result["threshold"],
            "threshold_err": run_result["threshold_err"],
            "output_dir": spec_output_dir,
        })

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    comparison_df.to_csv(
        os.path.join(output_dir, "comparison_efficiency_raw_all.csv"),
        index=False,
    )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        os.path.join(output_dir, "comparison_thresholds.csv"),
        index=False,
    )

    comparison_summary_df = (
        pd.concat(comparison_summary_frames, ignore_index=True)
        if comparison_summary_frames else pd.DataFrame()
    )
    if not comparison_summary_df.empty:
        comparison_summary_df.to_csv(
            os.path.join(output_dir, "comparison_sensitivity_summary.csv"),
            index=False,
        )

    score_order = threshold_df["score_name"].tolist()

    for mass in pbh_mass_list:
        subset_mass = comparison_df[comparison_df["mass"] == mass]
        if subset_mass.empty:
            continue

        plt.figure(figsize=(10, 6))
        for score_name in score_order:
            subset = subset_mass[subset_mass["score_name"] == score_name].sort_values("snr")
            if subset.empty:
                continue
            label = subset["score_label"].iloc[0]
            lower_err = np.maximum(0, subset["efficiency"] - subset["eff_ci_lower"])
            upper_err = np.maximum(0, subset["eff_ci_upper"] - subset["efficiency"])
            plt.errorbar(
                subset["snr"],
                subset["efficiency"],
                yerr=[lower_err, upper_err],
                fmt='o-',
                capsize=3,
                label=label,
            )

        plt.xlabel("SNR", size=18)
        plt.ylabel("Detection Efficiency", size=18)
        plt.title(f"Detector Comparison (Mass {mass:.2e} $M_{{\\odot}}$)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"comparison_efficiency_mass_{mass:.2e}.png"),
            dpi=300,
        )
        plt.close()

    if not comparison_summary_df.empty:
        plt.figure(figsize=(10, 6))
        for score_name in score_order:
            subset = comparison_summary_df[
                comparison_summary_df["score_name"] == score_name
            ].sort_values("mass")
            if subset.empty:
                continue
            label = subset["score_label"].iloc[0]
            y_err = subset["snr_at_target_eff_err"] if "snr_at_target_eff_err" in subset.columns else None
            if y_err is not None and not y_err.isna().all():
                plt.errorbar(
                    subset["mass"],
                    subset["snr_at_target_eff"],
                    yerr=y_err,
                    fmt='o-',
                    capsize=3,
                    label=label,
                )
            else:
                plt.semilogx(
                    subset["mass"],
                    subset["snr_at_target_eff"],
                    'o-',
                    label=label,
                )

        plt.xlabel("PBH Mass ($M_{\\odot}$)")
        plt.ylabel(f"SNR required for {target_efficiency_for_summary*100}% Efficiency")
        plt.xscale('log')
        plt.title(f"Detector Sensitivity Comparison (FP < {target_fp_per_year}/yr)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "comparison_sensitivity_curve.png"),
            dpi=300,
        )
        plt.close()

    logger.info("Detector comparison processing complete.")
    return {
        "comparison_df": comparison_df,
        "comparison_summary_df": comparison_summary_df,
        "threshold_df": threshold_df,
        "output_dir": output_dir,
    }

# -----------------------------------------------------------------------------
# Configuration & Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define configuration
    M_solar = 1.988e30
    
    config = {
        # Anchored to the project root (bootstrap) so the script works from
        # any working directory.
        'model_path': os.path.join(_PROJECT_ROOT, 'runs_continued', 'WithEncoder', 'reproduce_Model_2_dec_clas_both_sampling_continued', 'checkpoints', 'best.keras'),
        
        # NOTE: file_suffixes here is not being used for efficiency generation
        # because we will use 'test_file_suffixes' from prep_config instead.
        'file_suffixes': ['19.23.28.791'],
        
        'normalization_params': {'mean_value': 5.1753e-5, 'std_dev_value': 2.7052e-5},
        'normalization_mode': 'zscore',
        # Explicit PBH masses in kg. If `mass_scan_range` is provided below, it
        # overrides this list. Keeping the explicit list here is useful for
        # one-off single-mass studies.
        'm_PBH_list': [1e-13 * M_solar],
        # Recommended for log-space studies: define the PBH mass grid in solar
        # masses via exponents, then let the helper convert to kg.
        'mass_scan_range': {
            'space': 'log10_exponents',
            'start_exp': -13,
            'stop_exp': -6,
            'num': 8,
            # Alternative:
            # 'step_exp': 1,
        },
        'snr_range': np.arange(1, 10, 1)[::-1],
        'target_fp': 0.9999,
        'target_efficiency_for_summary': 0.95,
        'num_runs_per_point': 100, # x runs x 1 signal = x events
        'preset_threshold': None, # set a float here to skip the full calibration chain
        'detection_score_mode': 'logit', # 'logit', 'latent_dim_abs_zscore', 'latent_selected_zscore', 'latent_total_kl'
        'detection_score_config': {
            # Efficiency uses the same sampled-z classifier input regime as
            # training. Set "z_mean" only for a deterministic diagnostic.
            'classifier_inference_mode': 'sampled_z',
            'latent_dim': 0, # used by 'latent_dim_abs_zscore'
            'selected_latent_dims': [0, 2, 3, 7, 11], # used by 'latent_selected_zscore' or restricted KL
            'reduction': 'l2', # for 'latent_selected_zscore': 'l2', 'mean_abs', or 'max_abs'
            'use_abs': True, # for 'latent_dim_abs_zscore'
            # Optional: if you set 'preset_threshold' and also provide precomputed
            # 'noise_mean' and 'noise_std' arrays here, latent z-score modes can skip
            # the noise-reference preparation pass as well.
            # 'noise_mean': [...],
            # 'noise_std': [...],
        },
        'run_fp_scan': True,
        'fp_scan_output_dir': os.path.join(_PROJECT_ROOT, 'fp_rate_scan_results'),
        'fp_target_values': [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0], # explicit list, e.g. [0.1, 0.3, 1.0, 3.0, 10.0]
        'fp_scan_range': None,
        # Example alternative if you want an automatically generated scan:
        # 'fp_scan_range': {
        #     'start': 1e-1,
        #     'stop': 1e3,
        #     'step': 0.25,
        #     'space': 'log10', # 'log10' or 'linear'
        # },
        'manual_linear_fit': None,
        
        #{
         #   'slope': -4.2933941100624144,
          #  'intercept': -6.984897564164243,
           # 'covariance': [[var_slope, cov_slope_intercept],
           #                [cov_slope_intercept, var_intercept]],
           # 'chi2_red': 0.20323951687944866,
            #'fit_threshold_min': 0.5245962,
            #'fit_threshold_max': 2.2961178,
        #},
        'calibration_points_csv': os.path.join(_PROJECT_ROOT, 'efficiency_vae_results_saved', 'threshold_calibration_full_points.csv'),

        'run_score_comparison': False,
        'comparison_output_dir': os.path.join(_PROJECT_ROOT, 'efficiency_score_comparison'),
        'comparison_include_logit': False,
        'comparison_latent_dims': list(range(16)),
        'comparison_latent_use_abs': False,
        'comparison_preset_thresholds': {
            # 'logit': 5.4083,
            # 'latent_dim_0': 8.0,
        },
        'comparison_shared_detection_score_config': {
            # Optional shared config for every compared latent-dim detector.
            # This is the place to add precomputed noise_mean / noise_std arrays
            # if you want latent z-score comparison runs to skip the noise-reference
            # preparation pass as well.
            # 'noise_mean': [...],
            # 'noise_std': [...],
        },
        
        # Preprocessing Config
        'prep_config': {
            'filepath_suffixes': ['19.20.04.270'], #
            'test_file_suffixes': ['19.20.36.730', '19.20.12.385', '19.20.20.500', '19.22.23.418'], # '19.23.28.791'
            'filepath_template': os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.{}.tiq"),
            'num_samples_to_read_per_file': 200000, # 300000
            'num_samples_to_read_per_file_threshold_calibration': 112000000, # 112000000
            'offset': 0,
            'window_size': 1024,
            'step_size': 1024 // 30,
            'train_ratio': 0.01,
            'val_ratio': 0.98,
            'test_ratio': 0.01,
            'dtype': 'float32',
            'use_amps': True,
            'use_I_Q': False,
            'normalization_type': 'zscore',
            'global_mean_input': 5.1753e-5,
            'global_std_input': 2.7052e-5,
            'calculate_stats': False,
            'signal_injection_probability': 1.0,
            'num_signals_to_inject_per_segment': {'train': 0, 'val': 1, 'test': 0},
            'custom_noise_std': 2.7052e-5,
            # True: stop if an injection changes zero stored model-input
            # samples. False: warn and continue for diagnostics only.
            'reject_unrepresentable_injections': True,
            #'fs_val': 14e6,
        }
    }

    pbh_mass_list = _build_pbh_mass_list(
        explicit_masses_kg=config.get('m_PBH_list'),
        mass_scan_range=config.get('mass_scan_range'),
        M_solar=M_solar,
    )
    logger.info(
        "PBH mass grid (solar masses): %s",
        list(pbh_mass_list / M_solar),
    )
    
    if config.get('run_fp_scan'):
        generate_fp_rate_scan(
            config['model_path'],
            config['normalization_params'],
            config['normalization_mode'],
            config['prep_config']['test_file_suffixes'],
            pbh_mass_list,
            config['snr_range'],
            config['prep_config'],
            output_dir=config['fp_scan_output_dir'],
            fp_target_values=config.get('fp_target_values'),
            fp_scan_range=config.get('fp_scan_range'),
            num_runs_per_point=config['num_runs_per_point'],
            target_efficiency_for_summary=config['target_efficiency_for_summary'],
            num_samples_threshold_calibration=config['prep_config']['num_samples_to_read_per_file_threshold_calibration'],
            detection_score_mode=config['detection_score_mode'],
            detection_score_config=config['detection_score_config'],
            manual_linear_fit=config.get('manual_linear_fit'),
            calibration_points_csv=config.get('calibration_points_csv'),
        )
    elif config.get('run_score_comparison'):
        comparison_score_specs = _build_default_comparison_score_specs(
            include_logit=config.get('comparison_include_logit', True),
            latent_dims=config.get('comparison_latent_dims', []),
            latent_use_abs=config.get('comparison_latent_use_abs', True),
            preset_thresholds=config.get('comparison_preset_thresholds', {}),
            shared_detection_score_config=config.get('comparison_shared_detection_score_config', {}),
        )
        generate_efficiency_curve_comparison(
            config['model_path'],
            config['normalization_params'],
            config['normalization_mode'],
            config['prep_config']['test_file_suffixes'],
            pbh_mass_list,
            config['snr_range'],
            config['prep_config'],
            comparison_score_specs=comparison_score_specs,
            output_dir=config['comparison_output_dir'],
            target_fp_per_year=config['target_fp'],
            num_runs_per_point=config['num_runs_per_point'],
            target_efficiency_for_summary=config['target_efficiency_for_summary'],
            num_samples_threshold_calibration=config['prep_config']['num_samples_to_read_per_file_threshold_calibration'],
        )
    else:
        generate_efficiency_curves(
            config['model_path'],
            config['normalization_params'],
            config['normalization_mode'],
            # PASSING TEST SUFFIXES DIRECTLY:
            config['prep_config']['test_file_suffixes'],
            pbh_mass_list,
            config['snr_range'],
            config['prep_config'],
            target_fp_per_year=config['target_fp'],
            num_runs_per_point=config['num_runs_per_point'],
            target_efficiency_for_summary=config['target_efficiency_for_summary'],
            num_samples_threshold_calibration=config['prep_config']['num_samples_to_read_per_file_threshold_calibration'],
            detection_score_mode=config['detection_score_mode'],
            detection_score_config=config['detection_score_config'],
            preset_threshold=config['preset_threshold'],
        )
