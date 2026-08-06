"""
Central configuration for the simplified VAE-classifier.

Every knob that the training / analysis pipeline exposes lives here. The
main training script instantiates a VAEConfig, optionally overrides fields,
and passes the object to the rest of the pipeline. This keeps the entry-point
tiny while letting you control *everything*, from the AdamW weight decay down
to the exact number of filters in each convolutional layer.

Design notes
------------
* `num_filters_per_layer` is an explicit list, e.g. [16, 32, 64]. The length
  of this list defines the number of convolutional stages in the encoder
  (and, mirrored, in the decoder). This is strictly more flexible than
  `num_filters_first * factor**i`.
* `kernel_sizes_per_layer` and `strides_per_layer` are the same length as
  `num_filters_per_layer`. Defaults broadcast a single value if you pass one.
* Paths are relative so the project is portable between machines.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np


@dataclass
class VAEConfig:
    # -------------------------------------------------------------- #
    # Data
    # -------------------------------------------------------------- #
    filepath_template: str = "GravNet/Data/IQDataFile-2024.04.18.{}.tiq"
    filepath_suffixes: List[str] = field(
        default_factory=lambda: ["19.22.48.163", "19.22.56.276"]
    )
    num_samples_to_read_per_file: int = 112000000
    offset: int = 0
    sampling_rate_hz: float = 14e6

    # Windowing
    window_size: int = 1024
    step_size: int = 1024 // 10  # 90% overlap by default

    # Data split
    train_ratio: float = 0.84
    val_ratio: float = 0.15
    test_ratio: float = 0.01

    # Storage
    dtype: Any = np.float32
    memmap_dir: str = "./memmaps"
    # Keep preprocessing artifacts isolated per run. Shared memmap directories
    # can be silently overwritten by calibration, analysis, or another training
    # process, breaking the correspondence between plots and trained data.
    use_run_specific_memmap_dir: bool = True
    stats_dir: Optional[str] = None           # defaults to memmap_dir
    use_amps: bool = True                     # amplitude channel (1) vs I/Q (2)
    use_I_Q: bool = False
    normalization_type: str = "zscore"        # 'zscore' or 'min_max'

    # Precomputed normalization stats (optional; if None, preprocessing computes)
    global_mean_input: Optional[float] = None
    global_std_input: Optional[float] = None
    global_min_input: Optional[float] = None
    global_max_input: Optional[float] = None
    calculate_stats: bool = True

    # -------------------------------------------------------------- #
    # Signal injection
    # -------------------------------------------------------------- #
    inject_signals: bool = True
    signal_injection_probability: float = 1.0
    snr_based_injection: bool = True
    num_signals_to_inject_per_segment: Dict[str, int] = field(
        default_factory=lambda: {"train": 14000, "val": 2800, "test": 0}
    )
    # Solar-mass scaling is applied in train.py so the user writes natural units
    m_PBH_injection_list: List[float] = field(
        default_factory=lambda: [1e-8]
    )
    amplitude_spectrum_range: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    )  # Interpreted as target SNR when snr_based_injection=True
    f0_gw: float = 5.0e9
    Gamma_gw: float = 100e3
    N_gw: int = 32768
    M_solar: float = 1.988e30
    custom_noise_std: Optional[float] = None  # override global std for injection
    # If True, stop when an injection changes zero samples after conversion to
    # the stored model-input dtype. If False, continue with a warning and record
    # the zero-change injection in event metadata.
    reject_unrepresentable_injections: bool = True

    # --- No-overlap injection mode --- #
    # When True, injected signals keep a clear gap of at least
    # no_overlap_margin_samples between their supports. The default margin
    # (None -> window_size) guarantees that NO sliding window contains samples
    # of two different events — a window of length W can only bridge two
    # events whose gap is < W, regardless of step_size. Essential for
    # multi-mass I/Q training where a heavy-PBH spike inside the same window
    # could cancel/contaminate a light-PBH oscillation and corrupt the
    # clean-signal reconstruction/correlation targets. Injections that cannot
    # be placed (segment too densely packed) are skipped and reported.
    no_overlap_injections: bool = False
    no_overlap_margin_samples: Optional[int] = None   # None -> window_size
    no_overlap_max_attempts: int = 200

    # Cavity response of the simulator (A5):
    #   "real_lorentzian"      — existing/legacy behavior (default)
    #   "complex_breit_wigner" — complex single-pole response H = 1/(Δf + iΓ/2)
    response_mode: str = "real_lorentzian"

    # Clean injected-signal saving (A1) and metadata (A2-A4)
    save_clean_signals: bool = True
    save_metadata: bool = True
    # When True, the tf datasets yield (x, y, clean_signal_window) 3-tuples.
    # Required for rec_target_mode = "clean_signal".
    include_clean_in_datasets: bool = False

    # tf.data
    tf_batch_size: int = 1024
    tf_shuffle: bool = True
    tf_repeat: bool = False

    # -------------------------------------------------------------- #
    # Model architecture
    # -------------------------------------------------------------- #
    # Explicit per-layer control. Lists must all have the same length.
    num_filters_per_layer: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_sizes_per_layer: List[int] = field(default_factory=lambda: [5, 5, 5])
    strides_per_layer: List[int] = field(default_factory=lambda: [2, 2, 2])
    # Global hidden-layer activation fallback. May also be a per-stage list
    # for the encoder; the decoder then mirrors it in reverse by default.
    activation: Union[str, List[str]] = "gelu"
    # Optional phase-aware replacement for the very first encoder layer.
    # This only makes sense for I/Q input (use_I_Q=True, use_amps=False).
    use_quadrature_frontend: bool = False
    quadrature_output_mode: str = "magnitude"  # or "real_imag"
    # Optional explicit per-layer overrides. Encoder/classifier lengths must
    # match the number of stages exactly. Decoder may have length n or n - 1;
    # if n - 1 is given, the final decoder stage remains linear.
    encoder_activations: Optional[List[str]] = None
    decoder_activations: Optional[List[str]] = None
    classifier_activations: Optional[List[str]] = None
    latent_dim: int = 16

    # Classifier head on deterministic latent statistics
    classifier_hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    classifier_dropout: float = 0.1

    # If True, the classifier sees the *sampled* z = μ + σ * ε that the
    # decoder also sees (i.e. the reparameterization sample is shared).
    # If False (current production behavior), the classifier sees
    # concat[z_mean, z_log_var] directly — fully deterministic, no
    # sampling noise on the classifier path.
    # This is the boolean form of the more granular classifier_input_mode
    # in vae.ablation_model.VAEClassifierAblation:
    #     False → "concat_mean_logvar"
    #     True  → "shared_sample"
    classifier_samples_z: bool = False

    # -------------------------------------------------------------- #
    # Losses
    # -------------------------------------------------------------- #
    # Binary focal loss on the classifier head
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    focal_weight: float = 1.0

    # Auxiliary reconstruction loss on the decoder output (MSE).
    # Keeps the decoder meaningful so xAI plots are interpretable.
    # Set reconstruction_weight to 0.0 to disable.
    reconstruction_weight: float = 0.05

    # KL divergence weight (beta) with linear warm-up
    kl_beta_start: float = 1e-4
    kl_beta_end: float = 1e-2
    kl_warmup_epochs: int = 5

    # -------------------------------------------------------------- #
    # Loss components (B): every term is optional and individually logged.
    #   loss_total = lambda_bfl  * loss_bfl            (use_bfl,  focal_weight)
    #              + lambda_tail * loss_logsumexp_tail (use_tail_loss)
    #              + lambda_kl   * loss_kl             (use_kl, kl_beta schedule)
    #              + lambda_rec  * loss_rec            (use_rec, reconstruction_weight)
    #              + lambda_corr * loss_corr           (use_corr_loss)
    # focal_weight == lambda_bfl and reconstruction_weight == lambda_rec
    # (kept under their existing names for backwards compatibility).
    # -------------------------------------------------------------- #
    use_bfl: bool = True
    use_kl: bool = True
    use_rec: bool = True

    # B3: logsumexp tail loss on negative classifier logits
    use_tail_loss: bool = False
    lambda_tail: float = 0.05
    tail_beta: float = 10.0
    tail_margin: float = -2.0

    # B4: reconstruction target — "raw_input" (legacy) or "clean_signal"
    rec_target_mode: str = "raw_input"

    # B6: correlation loss on clean-signal reconstruction (positives only)
    use_corr_loss: bool = False
    # When True, clean I/Q reconstruction uses phase-invariant normalized
    # complex correlation instead of flattening I and Q into one real vector.
    # Requires use_I_Q=True, use_corr_loss=True, and clean-signal targets.
    use_iq_correlation_loss: bool = False
    lambda_corr: float = 0.05
    corr_eps: float = 1e-8

    # --- Optional auxiliary mass-estimation head --- #
    # Small MLP on the same latent features as the classifier, predicting the
    # normalized log10(m / M_sun) of signal windows (Huber loss, positives
    # only). With lambda_mass = 0 the head receives exactly zero gradient and
    # does nothing. Requires per-window mass targets in the datasets:
    # save_mass_targets=True and include_mass_in_datasets=True (which itself
    # requires include_clean_in_datasets=True).
    use_mass_head: bool = False
    lambda_mass: float = 0.0
    mass_head_hidden: List[int] = field(default_factory=lambda: [32])
    mass_huber_delta: float = 1.0
    # Normalization of the regression target:
    #   target = (log10(m/M_sun) - mass_norm_center) / mass_norm_scale
    # Defaults map the 1e-12..1e-6 range to [-1, +1].
    mass_norm_center: float = -9.0
    mass_norm_scale: float = 3.0
    # Preprocessing side: save per-window log10-mass memmaps and include them
    # as the 4th dataset tensor (x, y, clean, log10_mass).
    save_mass_targets: bool = False
    include_mass_in_datasets: bool = False

    # C: per-epoch classifier-logit quantile logging
    log_score_quantiles: bool = True

    # Pre-training data sanity plots. These verify the exact raw injected
    # component plus representative normalized noise and noise+signal windows
    # before model.fit is allowed to start.
    save_pretraining_sanity_plots: bool = True
    pretraining_sanity_split: str = "val"
    pretraining_sanity_strict: bool = True
    pretraining_sanity_max_candidates: int = 2048

    # -------------------------------------------------------------- #
    # Optimizer (AdamW)
    # -------------------------------------------------------------- #
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_epsilon: float = 1e-7
    clipnorm: Optional[float] = 1.0

    # -------------------------------------------------------------- #
    # Training
    # -------------------------------------------------------------- #
    epochs: int = 30
    early_stopping_patience: int = 8
    early_stopping_monitor: str = "val_loss"

    # -------------------------------------------------------------- #
    # Event-detection callback
    # -------------------------------------------------------------- #
    detection_every_epochs: int = 1           # 0 to disable
    detection_target_fp_per_year: float = 1.0
    detection_threshold_sweep_points: int = 200
    detection_log_fit_tail_fraction: float = 1e-4  # fit tail of FPR curve
    detection_min_tail_points: int = 4

    # -------------------------------------------------------------- #
    # I/O
    # -------------------------------------------------------------- #
    model_name: str = "vae_classifier"
    output_dir: str = "./runs"
    checkpoint_subdir: str = "checkpoints"
    figures_subdir: str = "figures"
    analysis_subdir: str = "analysis"
    random_seed: int = 42

    # -------------------------------------------------------------- #
    # Helpers
    # -------------------------------------------------------------- #
    @property
    def input_shape(self) -> Tuple[int, int]:
        channels = 2 if self.use_I_Q else 1
        return (self.window_size, channels)

    def __post_init__(self):
        # Broadcast / validate architecture lists
        n = len(self.num_filters_per_layer)
        if len(self.kernel_sizes_per_layer) == 1:
            self.kernel_sizes_per_layer = self.kernel_sizes_per_layer * n
        if len(self.strides_per_layer) == 1:
            self.strides_per_layer = self.strides_per_layer * n
        assert len(self.kernel_sizes_per_layer) == n, (
            "kernel_sizes_per_layer length must match num_filters_per_layer"
        )
        assert len(self.strides_per_layer) == n, (
            "strides_per_layer length must match num_filters_per_layer"
        )
        if isinstance(self.activation, list):
            assert len(self.activation) == n, (
                "activation list length must match num_filters_per_layer"
            )
        else:
            assert isinstance(self.activation, str), (
                "activation must be either a string or a list of strings"
            )
        if self.encoder_activations is not None:
            assert len(self.encoder_activations) == n, (
                "encoder_activations length must match num_filters_per_layer"
            )
        if self.decoder_activations is not None:
            assert len(self.decoder_activations) in (n - 1, n), (
                "decoder_activations length must be num_filters_per_layer "
                "or num_filters_per_layer - 1"
            )
        if self.classifier_activations is not None:
            assert len(self.classifier_activations) == len(self.classifier_hidden_units), (
                "classifier_activations length must match classifier_hidden_units"
            )
        assert self.quadrature_output_mode in ("magnitude", "real_imag"), (
            "quadrature_output_mode must be 'magnitude' or 'real_imag'"
        )
        if self.use_quadrature_frontend:
            assert self.use_I_Q and not self.use_amps, (
                "use_quadrature_frontend=True requires use_I_Q=True and use_amps=False"
            )
        assert self.use_amps ^ self.use_I_Q, (
            "Pick exactly one of use_amps / use_I_Q"
        )
        assert self.response_mode in ("real_lorentzian", "complex_breit_wigner"), (
            "response_mode must be 'real_lorentzian' or 'complex_breit_wigner'"
        )
        assert self.rec_target_mode in ("raw_input", "clean_signal"), (
            "rec_target_mode must be 'raw_input' or 'clean_signal'"
        )
        assert self.pretraining_sanity_split in ("train", "val", "test"), (
            "pretraining_sanity_split must be 'train', 'val', or 'test'"
        )
        assert self.pretraining_sanity_max_candidates > 0, (
            "pretraining_sanity_max_candidates must be positive"
        )
        if self.use_corr_loss and self.rec_target_mode != "clean_signal":
            raise ValueError(
                "use_corr_loss=True requires rec_target_mode='clean_signal' — "
                "the correlation loss is defined against the clean injected signal."
            )
        if self.use_iq_correlation_loss:
            if not self.use_I_Q or self.use_amps:
                raise ValueError(
                    "use_iq_correlation_loss=True requires use_I_Q=True and "
                    "use_amps=False."
                )
            if not self.use_corr_loss or self.rec_target_mode != "clean_signal":
                raise ValueError(
                    "use_iq_correlation_loss=True requires use_corr_loss=True "
                    "and rec_target_mode='clean_signal'."
                )
        if self.rec_target_mode == "clean_signal" and not self.include_clean_in_datasets:
            raise ValueError(
                "rec_target_mode='clean_signal' requires include_clean_in_datasets=True "
                "(and save_clean_signals=True) so the dataset yields clean_signal_window."
            )
        # --- Mass-head consistency --- #
        if self.use_mass_head and not self.include_mass_in_datasets:
            raise ValueError(
                "use_mass_head=True requires include_mass_in_datasets=True so "
                "batches carry the per-window log10-mass target."
            )
        if self.include_mass_in_datasets and not self.save_mass_targets:
            raise ValueError(
                "include_mass_in_datasets=True requires save_mass_targets=True."
            )
        if self.include_mass_in_datasets and not self.include_clean_in_datasets:
            raise ValueError(
                "include_mass_in_datasets=True requires include_clean_in_datasets=True "
                "(the 4-tuple dataset form is (x, y, clean, log10_mass))."
            )
        assert self.mass_norm_scale > 0, "mass_norm_scale must be positive"
        assert self.no_overlap_max_attempts > 0, (
            "no_overlap_max_attempts must be positive"
        )
        if self.no_overlap_margin_samples is not None:
            assert self.no_overlap_margin_samples >= 0, (
                "no_overlap_margin_samples must be >= 0"
            )
