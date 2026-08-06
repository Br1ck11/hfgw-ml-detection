"""
Paired-input audit for the GravNet injection and reconstruction pipeline.

The audit uses one fixed real noise window and compares it with the exact same
window after injecting a centered waveform at several target peak SNRs. This
isolates the effect of the injection from noise realization and selection bias.
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


import argparse
import csv
import os

import numpy as np
import tensorflow as tf

from data_pre_processing.chirp_BW_conv_signal_generation import get_trimmed_waveform
from data_pre_processing.tiq_data_loader import load_tiq_data_segment
from vae.model import QuadratureConv1D, Sampling, VAEClassifier


M_SOLAR = 1.988e30


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _deterministic_outputs(model, x):
    x_tf = tf.convert_to_tensor(x[None, ..., None], dtype=tf.float32)
    z_mean, z_log_var = model.encoder(x_tf, training=False)
    features = model.classifier_features(z_mean, z_log_var)
    logit = model.classifier(features, training=False)
    recon = model.decoder(z_mean, training=False)
    return (
        float(np.asarray(logit).reshape(-1)[0]),
        np.asarray(recon)[0, :, 0],
        np.asarray(z_mean)[0],
        np.asarray(z_log_var)[0],
    )


def _shared_sample_outputs(model, z_mean, z_log_var, seed=(17, 29)):
    eps = tf.random.stateless_normal(np.shape(z_mean), seed=seed, dtype=tf.float32)
    z = tf.convert_to_tensor(z_mean, dtype=tf.float32) + tf.exp(
        0.5 * tf.convert_to_tensor(z_log_var, dtype=tf.float32)
    ) * eps
    features = model.classifier_features(
        tf.convert_to_tensor(z_mean[None, :], dtype=tf.float32),
        tf.convert_to_tensor(z_log_var[None, :], dtype=tf.float32),
        z=z[None, :],
    )
    logit = model.classifier(features, training=False)
    recon = model.decoder(z[None, :], training=False)
    return float(np.asarray(logit).reshape(-1)[0]), np.asarray(recon)[0, :, 0]


def run_audit(args):
    noise, fs = load_tiq_data_segment(args.tiq_file, args.offset, args.window_size)
    if noise is None or len(noise) != args.window_size:
        raise RuntimeError("Could not load the requested fixed noise window.")

    model = tf.keras.models.load_model(
        args.checkpoint,
        custom_objects={
            "Sampling": Sampling,
            "VAEClassifier": VAEClassifier,
            "QuadratureConv1D": QuadratureConv1D,
        },
        compile=False,
    )

    waveform = get_trimmed_waveform(
        args.mass_solar * M_SOLAR,
        1.0,
        args.f0_gw,
        args.gamma_gw,
        args.n_gw,
        M_SOLAR,
        relative_threshold_factor=1e-3,
        response_mode=args.response_mode,
    )
    if waveform.size > args.window_size:
        raise ValueError(
            f"Trimmed waveform has {waveform.size} samples but window has "
            f"{args.window_size}."
        )

    start = (args.window_size - waveform.size) // 2
    end = start + waveform.size
    noise = np.asarray(noise, dtype=np.complex128)
    noise_amp64 = np.abs(noise)
    noise_stored = noise_amp64.astype(np.float32)
    noise_norm = ((noise_stored - args.global_mean) / args.global_std).astype(np.float32)

    noise_logit, noise_recon, noise_zm, noise_zlv = _deterministic_outputs(model, noise_norm)
    noise_sample_logit, noise_sample_recon = _shared_sample_outputs(
        model, noise_zm, noise_zlv
    )

    rows = []
    for target_snr in args.snrs:
        signal = np.zeros(args.window_size, dtype=np.complex128)
        scaled = waveform.copy()
        peak = np.max(np.abs(scaled))
        if peak > 0:
            scaled *= target_snr * args.noise_std / peak
        signal[start:end] = scaled

        injected = noise + signal
        injected_amp64 = np.abs(injected)
        injected_stored = injected_amp64.astype(np.float32)
        injected_norm = (
            (injected_stored - args.global_mean) / args.global_std
        ).astype(np.float32)
        clean_target = (np.abs(signal) / args.global_std).astype(np.float32)

        logit, recon, z_mean, z_log_var = _deterministic_outputs(model, injected_norm)
        sample_logit, sample_recon = _shared_sample_outputs(model, z_mean, z_log_var)

        input_delta = injected_norm - noise_norm
        raw_effective_delta = injected_amp64 - noise_amp64
        stored_effective_delta = injected_stored - noise_stored
        rows.append({
            "target_peak_snr": target_snr,
            "waveform_samples": waveform.size,
            "complex_changed_samples": int(np.count_nonzero(injected != noise)),
            "amplitude_float64_changed_samples": int(
                np.count_nonzero(injected_amp64 != noise_amp64)
            ),
            "amplitude_float32_changed_samples": int(
                np.count_nonzero(injected_stored != noise_stored)
            ),
            "normalized_float32_changed_samples": int(np.count_nonzero(input_delta)),
            "effective_amp64_peak_snr": float(
                np.max(np.abs(raw_effective_delta)) / args.noise_std
            ),
            "effective_stored_peak_snr": float(
                np.max(np.abs(stored_effective_delta)) / args.noise_std
            ),
            "normalized_input_delta_l2": float(np.linalg.norm(input_delta)),
            "deterministic_noise_logit": noise_logit,
            "deterministic_injected_logit": logit,
            "deterministic_logit_delta": logit - noise_logit,
            "deterministic_recon_delta_l2": float(np.linalg.norm(recon - noise_recon)),
            "deterministic_recon_clean_corr": _correlation(recon, clean_target),
            "noise_recon_clean_corr": _correlation(noise_recon, clean_target),
            "shared_sample_noise_logit": noise_sample_logit,
            "shared_sample_injected_logit": sample_logit,
            "shared_sample_logit_delta_common_eps": sample_logit - noise_sample_logit,
            "shared_sample_recon_delta_l2_common_eps": float(
                np.linalg.norm(sample_recon - noise_sample_recon)
            ),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "paired_low_snr_audit.csv")
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Noise dtype: {noise.dtype}; sampling rate: {fs}")
    print(f"Waveform length: {waveform.size}; insertion: [{start}, {end})")
    print(f"Saved: {output_path}")
    print("\nTarget SNR | stored changed | input L2 | deterministic logit delta | recon delta L2")
    for row in rows:
        print(
            f"{row['target_peak_snr']:10.3e} | "
            f"{row['normalized_float32_changed_samples']:14d} | "
            f"{row['normalized_input_delta_l2']:8.3e} | "
            f"{row['deterministic_logit_delta']:+25.3e} | "
            f"{row['deterministic_recon_delta_l2']:14.3e}"
        )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    # Defaults anchored to the project root (bootstrap) so the script works
    # from any working directory.
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_PROJECT_ROOT, "runs", "reproduce_Model_2_dec_clas_both_samplingUpdatedLosses", "checkpoints", "best.keras"),
    )
    parser.add_argument(
        "--tiq-file",
        default=os.path.join(_PROJECT_ROOT, "GravNet", "Data", "IQDataFile-2024.04.18.19.23.28.791.tiq"),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--mass-solar", type=float, default=1e-8)
    parser.add_argument("--noise-std", type=float, default=2.7052e-5)
    parser.add_argument("--global-mean", type=float, default=5.1753e-5)
    parser.add_argument("--global-std", type=float, default=2.7052e-5)
    parser.add_argument("--f0-gw", type=float, default=5.0e9)
    parser.add_argument("--gamma-gw", type=float, default=100e3)
    parser.add_argument("--n-gw", type=int, default=32768)
    parser.add_argument("--response-mode", default="real_lorentzian")
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[0.0, 1e-19, 1e-12, 1e-9, 1e-7, 1e-6, 1e-3, 1e-2, 0.1, 1.0, 3.0],
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_PROJECT_ROOT, "runs", "reproduce_Model_2_dec_clas_both_samplingUpdatedLosses", "pipeline_audit"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
