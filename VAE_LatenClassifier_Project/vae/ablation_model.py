"""
Ablation variant of the VAE-classifier.

This subclass exposes a single new knob — `classifier_input_mode` — that
controls what the classifier head sees. Everything else (encoder, decoder,
loss weights, serialization, metrics) is inherited unchanged.

Modes
-----
"concat_mean_logvar"
    The current production behaviour. Classifier sees concat([z_mean, z_log_var]).
    Fully deterministic. Encoder gradients flow only via μ and log σ².
    Classifier head input dim = 2 * latent_dim.

"z_mean"
    Classifier sees z_mean alone. Deterministic, no σ information.
    Classifier head input dim = latent_dim.

"shared_sample"
    Classifier and decoder both receive the *same* sampled z = μ + σ * ε.
    The reparameterization sample is shared, so classifier-loss gradients
    flow back through ε * σ (the same path the decoder uses).
    Classifier head input dim = latent_dim.

"independent_sample"     ← THE NOISE-INJECTION HYPOTHESIS TEST
    Classifier sees z_cls = μ + σ * ε_cls with ε_cls ~ N(0, I) drawn
    *independently* of the decoder's ε. Same noise statistics as the
    decoder path, but the random draws are uncorrelated. If reconstruction
    quality matches "shared_sample", the benefit comes from noise injection
    on the classifier (regularization). If only "shared_sample" works, the
    shared-sample path itself matters — the classifier and decoder must see
    identical noise realizations for the synergy to appear.
    Classifier head input dim = latent_dim.

"shared_sample_detached"
    Classifier sees the *same* sampled z as the decoder, but wrapped in
    tf.stop_gradient. Classification loss therefore cannot move encoder
    weights at all. Use this to confirm that the classifier really is
    helping the encoder: if recon quality drops to the level of a plain
    VAE, the classifier-via-encoder pathway is what was carrying the signal.
    Classifier head input dim = latent_dim.

How to use
----------
    from vae.ablation_model import build_vae_classifier_ablation
    model = build_vae_classifier_ablation(cfg, mode="independent_sample")

The returned model is a drop-in replacement for VAEClassifier and works
with the existing BetaAnnealing / EventDetectionCallback / model.fit.

Gradient inspector hookup
-------------------------
If `model.gradient_logger` is set to a vae.gradient_inspector.GradientInspector
instance and `model.run_eagerly = True`, the train_step will compute per-loss
gradients on the trainable model every logged step and hand them to the
inspector. The inspector decides what to write to disk based on its own
cadence flags.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
import keras

from .model import (
    VAEClassifier,
    Sampling,
    build_cnn_encoder,
    build_cnn_decoder,
    build_classifier_head,
    _unpack,
)
from .losses import (
    binary_focal_loss,
    kl_divergence_standard_normal,
    logsumexp_tail_loss,
    correlation_loss,
    complex_correlation_loss,
)


_VALID_MODES = (
    "concat_mean_logvar",
    "z_mean",
    "shared_sample",
    "independent_sample",
    "shared_sample_detached",
)


@keras.saving.register_keras_serializable(package="vae")
class VAEClassifierAblation(VAEClassifier):
    """
    VAEClassifier whose classifier-input pathway is configurable.

    Parameters
    ----------
    classifier_input_mode : str
        One of `_VALID_MODES`. See the module docstring.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        classifier: tf.keras.Model,
        classifier_input_mode: str = "concat_mean_logvar",
        **kwargs,
    ):
        if classifier_input_mode not in _VALID_MODES:
            raise ValueError(
                f"classifier_input_mode must be one of {_VALID_MODES}, "
                f"got {classifier_input_mode!r}"
            )
        super().__init__(
            encoder=encoder, decoder=decoder, classifier=classifier, **kwargs
        )
        self.classifier_input_mode = classifier_input_mode
        # Independent sampling layer for the "independent_sample" mode.
        self._sampling_cls = Sampling(name="sampling_classifier")
        # Slot for the gradient inspector (stays None unless explicitly set).
        self.gradient_logger = None

    # -- serialization (preserve mode across save/load) ---------------- #
    def get_config(self):
        config = super().get_config()
        config.update({"classifier_input_mode": self.classifier_input_mode})
        return config

    # -- forward ------------------------------------------------------- #
    def _classifier_input(self, z_mean, z_log_var, z_shared, training):
        """Build the tensor the classifier consumes for the current mode."""
        mode = self.classifier_input_mode
        if mode == "concat_mean_logvar":
            return tf.concat([z_mean, z_log_var], axis=-1)
        if mode == "z_mean":
            return z_mean
        if mode == "shared_sample":
            return z_shared
        if mode == "independent_sample":
            # Independent ε for the classifier path, same μ and σ.
            return self._sampling_cls([z_mean, z_log_var])
        if mode == "shared_sample_detached":
            # Same realization, but no gradient flows back into the encoder.
            return tf.stop_gradient(z_shared)
        raise ValueError(f"Unknown classifier_input_mode: {mode}")

    def _expected_classifier_input_dim(self, latent_dim: int) -> int:
        if self.classifier_input_mode == "concat_mean_logvar":
            return 2 * latent_dim
        return latent_dim

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z = self.sampling([z_mean, z_log_var])
        cls_in = self._classifier_input(z_mean, z_log_var, z, training)
        return self.classifier(cls_in, training=training)

    # -- losses + train step ------------------------------------------ #
    def _compute_losses(self, x, y_true, training, clean=None, mass=None):
        # `mass` accepted for signature compatibility with the parent; the
        # ablation model has no mass head, so the term is always zero.
        # Mirrors the parent's optional loss components (B), but routes the
        # classifier input through `_classifier_input` (the ablation knob):
        #   loss_total = lambda_bfl  * BFL          (use_bfl)
        #              + lambda_tail * logsumexp    (use_tail_loss)
        #              + lambda_kl   * KL           (use_kl, annealed kl_beta)
        #              + lambda_rec  * MSE          (use_rec, rec_target_mode)
        #              + lambda_corr * (1 - corr)   (use_corr_loss, positives)
        z_mean, z_log_var = self.encoder(x, training=training)
        z = self.sampling([z_mean, z_log_var])
        cls_in = self._classifier_input(z_mean, z_log_var, z, training)
        logits = self.classifier(cls_in, training=training)
        recon = self.decoder(z, training=training)

        y_true_f = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)

        focal = (
            binary_focal_loss(
                y_true_f, logits,
                gamma=self.focal_gamma, alpha=self.focal_alpha, from_logits=True,
            )
            if self.use_bfl else zero
        )

        tail = (
            logsumexp_tail_loss(
                logits, y_true_f, beta=self.tail_beta, margin=self.tail_margin,
            )
            if self.use_tail_loss else zero
        )

        rec_target = self._resolve_rec_target(x, clean)
        recon_loss = (
            tf.reduce_mean(tf.square(rec_target - recon))
            if self.use_rec else zero
        )

        corr = (
            (
                complex_correlation_loss(
                    recon, rec_target, labels=y_true_f, eps=self.corr_eps
                )
                if self.use_iq_correlation_loss
                else correlation_loss(
                    recon, rec_target, labels=y_true_f, eps=self.corr_eps
                )
            )
            if (self.use_corr_loss and self.rec_target_mode == "clean_signal")
            else zero
        )

        kl = kl_divergence_standard_normal(z_mean, z_log_var) if self.use_kl else zero

        total = (
            self.focal_weight * focal
            + self.lambda_tail * tail
            + self.reconstruction_weight * recon_loss
            + self.lambda_corr * corr
            + self.kl_beta * kl
        )
        return total, focal, recon_loss, kl, tail, corr, zero, logits

    def train_step(self, data):
        """
        Same as the parent train_step, but uses a *persistent* tape when a
        gradient inspector is attached so we can re-query gradients for each
        loss term without re-running the forward pass.

        IMPORTANT: when an inspector is attached, the model must run eagerly
        (`model.run_eagerly = True`). The inspector sets that flag for you in
        its `on_train_begin` callback; if you bypass `model.fit` you have to
        flip it yourself.
        """
        x, y, clean = _unpack(data)
        inspector = self.gradient_logger
        do_log = inspector is not None and inspector.should_log_step()
        do_decompose = do_log and inspector.should_decompose_step()
        # Persistent tapes are only needed on the rare steps where we run the
        # per-loss decomposition. Combined gradient norms can reuse `grads`
        # from the ordinary training backward pass and do not need a
        # persistent tape.
        persistent = do_decompose

        with tf.GradientTape(persistent=persistent) as tape:
            total, focal, recon, kl, tail, corr, mass_l, logits = self._compute_losses(
                x, y, training=True, clean=clean
            )

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        if do_log:
            if do_decompose:
                # Each call costs ~one extra full-model backward pass. Persistent
                # tape lets us reuse the forward graph across the three calls.
                # We differentiate the *raw* losses (recorded inside the tape)
                # and apply loss weights to the resulting gradients in Python.
                # This is mathematically identical to differentiating the
                # weighted losses, but avoids relying on tape behavior for
                # ops performed after the `with` block exits.
                trainable_vars = self.trainable_weights
                g_focal_raw = tape.gradient(focal, trainable_vars)
                g_recon_raw = tape.gradient(recon, trainable_vars)
                g_kl_raw = tape.gradient(kl, trainable_vars)

                def _scale(gs, w):
                    return [None if g is None else g * w for g in gs]

                g_focal = _scale(g_focal_raw, self.focal_weight)
                g_recon = _scale(g_recon_raw, self.reconstruction_weight)
                g_kl = _scale(g_kl_raw, self.kl_beta)
                # NOTE: the inspector's decomposition stays focal/recon/kl —
                # the optional tail/corr terms are part of `all_grads` but are
                # not decomposed separately (the inspector API is 3-way).
            else:
                g_focal = g_recon = g_kl = None
            inspector.log(
                trainable_vars=self.trainable_weights,
                g_focal=g_focal,
                g_recon=g_recon,
                g_kl=g_kl,
                all_grads=grads,
            )

        if persistent:
            del tape  # release persistent tape resources

        self._update_trackers(total, focal, recon, kl, tail, corr, mass_l, y, logits)
        return {m.name: m.result() for m in self.metrics}


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #

def build_vae_classifier_ablation(
    cfg,
    mode: str = "concat_mean_logvar",
    name_suffix: Optional[str] = None,
) -> VAEClassifierAblation:
    """
    Build a VAEClassifierAblation from a VAEConfig and a classifier-input mode.

    The encoder/decoder activations resolve in exactly the same way as
    `vae.model.build_vae_classifier`, so per-layer activation lists work too.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_MODES}, got {mode!r}"
        )

    encoder_activation = cfg.encoder_activations
    if encoder_activation is None:
        encoder_activation = cfg.activation

    decoder_activation = cfg.decoder_activations
    if decoder_activation is None:
        if cfg.encoder_activations is not None:
            decoder_activation = list(reversed(cfg.encoder_activations))
        elif isinstance(cfg.activation, list):
            decoder_activation = list(reversed(cfg.activation))
        else:
            decoder_activation = cfg.activation

    classifier_activation = cfg.classifier_activations
    if classifier_activation is None:
        if isinstance(cfg.activation, list):
            classifier_activation = cfg.activation[0]
        else:
            classifier_activation = cfg.activation

    encoder = build_cnn_encoder(
        input_shape=cfg.input_shape,
        num_filters_per_layer=cfg.num_filters_per_layer,
        kernel_sizes_per_layer=cfg.kernel_sizes_per_layer,
        strides_per_layer=cfg.strides_per_layer,
        latent_dim=cfg.latent_dim,
        activation=encoder_activation,
        use_quadrature_frontend=cfg.use_quadrature_frontend,
        quadrature_output_mode=cfg.quadrature_output_mode,
    )
    decoder = build_cnn_decoder(
        output_shape=cfg.input_shape,
        num_filters_per_layer=cfg.num_filters_per_layer,
        kernel_sizes_per_layer=cfg.kernel_sizes_per_layer,
        strides_per_layer=cfg.strides_per_layer,
        latent_dim=cfg.latent_dim,
        activation=decoder_activation,
    )
    classifier_input_dim = (
        2 * cfg.latent_dim if mode == "concat_mean_logvar" else cfg.latent_dim
    )
    classifier = build_classifier_head(
        input_dim=classifier_input_dim,
        hidden_units=cfg.classifier_hidden_units,
        dropout=cfg.classifier_dropout,
        activation=classifier_activation,
    )
    base_name = cfg.model_name
    if name_suffix:
        base_name = f"{base_name}_{name_suffix}"
    else:
        base_name = f"{base_name}_{mode}"

    return VAEClassifierAblation(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        classifier_input_mode=mode,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        focal_weight=cfg.focal_weight,
        reconstruction_weight=cfg.reconstruction_weight,
        kl_beta=cfg.kl_beta_start,
        # Optional loss components (B) — same knobs as build_vae_classifier.
        use_bfl=getattr(cfg, "use_bfl", True),
        use_kl=getattr(cfg, "use_kl", True),
        use_rec=getattr(cfg, "use_rec", True),
        use_tail_loss=getattr(cfg, "use_tail_loss", False),
        lambda_tail=getattr(cfg, "lambda_tail", 0.05),
        tail_beta=getattr(cfg, "tail_beta", 10.0),
        tail_margin=getattr(cfg, "tail_margin", -2.0),
        rec_target_mode=getattr(cfg, "rec_target_mode", "raw_input"),
        use_corr_loss=getattr(cfg, "use_corr_loss", False),
        use_iq_correlation_loss=getattr(cfg, "use_iq_correlation_loss", False),
        lambda_corr=getattr(cfg, "lambda_corr", 0.05),
        corr_eps=getattr(cfg, "corr_eps", 1e-8),
        name=base_name,
    )
