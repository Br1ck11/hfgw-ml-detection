"""
Simplified VAE-classifier package for HFGW cavity signal detection.

Modules
-------
config     : One dataclass holding every hyperparameter the pipeline exposes.
model      : CNN encoder + reparameterization + CNN decoder + classifier head.
losses     : Binary focal loss and Gaussian KL divergence (analytic form).
callbacks  : Beta-annealing and event-based detection threshold callback.
analysis   : xAI helpers for encoder/decoder activations and latent space.
"""

from .config import VAEConfig
from .model import build_vae_classifier, VAEClassifier, QuadratureConv1D
from .losses import (
    binary_focal_loss,
    kl_divergence_standard_normal,
    logsumexp_tail_loss,
    correlation_loss,
    complex_correlation_loss,
)
from .callbacks import BetaAnnealing, EventDetectionCallback, ScoreQuantileLogger
from .clustered_fp import compute_clustered_fp_diagnostics
from .ablation_model import (
    VAEClassifierAblation,
    build_vae_classifier_ablation,
)
from .gradient_inspector import GradientInspector, plot_gradient_log

__all__ = [
    "VAEConfig",
    "build_vae_classifier",
    "VAEClassifier",
    "QuadratureConv1D",
    "binary_focal_loss",
    "kl_divergence_standard_normal",
    "logsumexp_tail_loss",
    "correlation_loss",
    "complex_correlation_loss",
    "BetaAnnealing",
    "EventDetectionCallback",
    "ScoreQuantileLogger",
    "compute_clustered_fp_diagnostics",
    "VAEClassifierAblation",
    "build_vae_classifier_ablation",
    "GradientInspector",
    "plot_gradient_log",
]
