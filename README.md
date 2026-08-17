# GravNet — Machine Learning for High-Frequency Gravitational-Wave Detection

Detecting simulated **high-frequency gravitational waves (HFGWs)** — such as those
expected from mergers of **sub-solar-mass primordial black holes (PBHs)** — in noisy
**microwave-cavity** detector data (SUPAX-type experiments), using deep learning.

Signals couple to an electromagnetic cavity mode via the inverse Gertsenshtein
effect and are buried in detector noise. This repository collects the
machine-learning classifiers developed to decide, per data window, whether such a
signal is present, together with the reports documenting each study.

> **Status:** research code targeting real detector `.tiq` recordings that are not
> redistributed here. The simulation, model, training and evaluation code are the
> shareable part; raw data, training runs and checkpoints are git-ignored.

---

## Why this problem is hard

- The induced strains are tiny and real signals are expected to sit below the noise level.
- Each waveform is a **chirp × cavity (Breit–Wigner) response**: a frequency sweep
  set by the PBH mass, shaped by the cavity's resonance, injected into real I/Q
  detector noise of the SUPAX experiment at a controlled signal-to-noise ratio (SNR).
- Conventional interferometric methods lose sensitivity in the high-frequency band,
  motivating cavity experiments — and learned detectors that exploit the full
  waveform structure across many orders of magnitude in PBH mass.

## The studies in this repository

Both share the same signal simulation and evaluation philosophy:

1. **CNN–LSTM classifier** (`CNN_LSTM_Classifier/`) — a hybrid
   convolutional–recurrent binary signal/noise classifier.
   See `Time_Series_Analysis_Research_Project.pdf`.

2. **VAE latent-classifier** (`VAE_LatenClassifier_Project/`) — a variational
   autoencoder whose classifier operates directly on a single, explicit,
   low-dimensional **latent code**, shaped simultaneously by a generative decoder
   (reconstructing the *clean* injected signal) and by physics-motivated loss
   terms. The guiding idea is *architectural locality*: one well-defined place
   where all the compressed, inspectable information about a window resides. Two
   variants are compared — a **latent-only** classifier and a
   **decoder-into-classifier** variant that also sees a stop-gradient
   reconstruction. See `VAEProjectPaper.pdf`.

The base report `Base_report_differentiating_HFGWs_using_ML.pdf` introduces the
overall HFGW-with-ML problem that both studies build on. It was the first report of the ML approach used.

## Selected findings (VAE study)

- Detection is quantified by **SNR₉₅**: the injected peak SNR required for 95%
  event-detection efficiency, calibrated across false-positive (FP) rates from
  0.1 to 100 per year and PBH masses from 10⁻¹² to 10⁻⁶ M⊙.
- The **latent-only** variant requires lower SNR across the mass range than the
  decoder-into-classifier variant (which only wins at the lightest mass).
- The **hardest mass to detect, ≈ 10⁻¹⁰ M⊙**, coincides with a cavity time-scale
  matching Δt = τ (signal duration equal to the cavity ring-down time), and the
  same feature is imprinted in the latent representation.

---

## Repository layout

```
GravNet/
├── Base_report_differentiating_HFGWs_using_ML.pdf   Problem overview report
│
├── CNN_LSTM_Classifier/                CNN–LSTM baseline classifier
│   ├── CNN_LSTM_classifier_model.py
│   ├── Custom_callbacks.py
│   └── Time_Series_Analysis_Research_Project.pdf     CNN–LSTM report
│
└── VAE_LatenClassifier_Project/        VAE latent-classifier study
    ├── VAEProjectPaper.pdf                            VAE report
    ├── vae/                             Model, losses, config, analysis
    │   ├── model.py        encoder → (z_mean, z_log_var) → z → decoder,
    │   │                   classifier head on the latent, optional recon branch
    │   ├── losses.py       focal, KL, log-sum-exp tail, correlation, mass-regression
    │   ├── config.py       single dataclass holding every data/arch/loss/optim knob
    │   ├── analysis.py     latent PCA, activation probes, logit histograms (xAI)
    │   ├── callbacks.py, ablation_model.py, gradient_inspector.py, clustered_fp.py
    │   └── ...
    ├── data_pre_processing/             Shared pipeline
    │   ├── chirp_BW_conv_signal_generation.py   waveform generator (chirp × cavity)
    │   ├── pre_processing_incomplete.py         windowing, injection, normalisation,
    │   │                                        clean-signal targets, memmap caching
    │   ├── tiq_data_loader.py, window_data.py, stats.py, sanity_plots.py
    │   └── GWSignalFigures/
    ├── training/
    │   ├── train.py                     single training run
    │   ├── run_experiments.py           loss-configuration experiment matrix
    │   ├── continue_train.py, train_ablation.py,
    │   ├── train_clean_signal_autoencoder.py, train_with_gradient_inspection.py
    └── evaluation/
        ├── efficiency_curve.py          efficiency vs peak SNR, SNR95 @ FP/year (main metric)
        ├── analyze.py                   detector xAI + metrics
        ├── evaluate_post_vae_signal_manifold.py, window_score_diagnostic.py,
        └── compare_ablation_analysis.py, full_model_insight.py, audit_low_snr_pipeline.py
```

Raw detector data, training runs and checkpoints are not tracked (see
`.gitignore`).

## Installation

The code uses **TensorFlow/Keras**, together with scientific-computing,
plotting and `.tiq`-file utilities. Create a virtual environment and install
the required packages:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow keras tensorflow-probability numpy scipy pandas matplotlib iqtools
```

`tensorflow-probability` is used by the CNN–LSTM loss implementation, while
`iqtools` is required to load the detector recordings. The optional
[`SciencePlots`](https://github.com/garrettj403/SciencePlots) package enables
the plotting style used by the waveform-generation script:

```bash
python -m pip install SciencePlots
```

A GPU is recommended for training. Evaluation can run on CPU, although the
larger evaluation scans may take considerably longer.

## Quickstart (VAE study)

The scripts resolve project-relative paths, so run them from inside
`VAE_LatenClassifier_Project/`:

```bash
cd VAE_LatenClassifier_Project

# Train a VAE-classifier from scratch
python training/train.py

# Run the loss-configuration experiment matrix
python training/run_experiments.py all

# Evaluate: efficiency curves and SNR95 vs false-positive rate (main metric)
python evaluation/efficiency_curve.py
```

All model, data, loss and optimiser settings live in a single config dataclass
(`vae/config.py`). The checked-in training and evaluation scripts contain
example experiment configurations. They are not guaranteed to reproduce the
exact configurations or numerical results reported in the accompanying papers.

## Data

Training and evaluation use real detector I/Q recordings (complex I/Q at 14 MHz),
which are **not** included in the repository. The synthetic-signal generation,
injection, training and evaluation code are fully contained here.

## Reports

Each study ships with its written report:

- `Base_report_differentiating_HFGWs_using_ML.pdf` — HFGW-with-ML problem overview.
- `CNN_LSTM_Classifier/Time_Series_Analysis_Research_Project.pdf` — CNN–LSTM study.
- `VAE_LatenClassifier_Project/VAEProjectPaper.pdf` — VAE latent-classifier study.

## Acknowledgements

This work builds on the SUPAX / GravNet cavity-experiment effort. Parts of the
analysis and documentation were prepared with the assistance of AI tools such as Claude Code, Codex and Gemini.


## License

Released under the [MIT License](LICENSE) — free to use, modify and redistribute
with attribution. If you use this code in academic work, a citation of the
relevant report is appreciated.
