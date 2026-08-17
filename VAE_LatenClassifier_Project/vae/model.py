"""
Simplified VAE-classifier model — encoder, decoder, sampling, and head
in a single, readable file.

Architecture (forward pass)
---------------------------
    x  ──>  [optional quadrature front-end]  ──>  CNN encoder  ──>  Flatten  ──>  (z_mean, z_log_var)
                                                 │
                                                 ├── concat([z_mean, z_log_var])
                                                 │              │
                                                 │              ▼
                                                 │      Classifier head (MLP)
                                                 │              │
                                                 │              ▼
                                                 │      window-level logit
                                                 │
                                                 ▼   (reparameterization)
                                                 z  ~  N(z_mean, exp(z_log_var))
                                                 │
                                                 ▼
                                           CNN decoder
                                                 │
                                                 ▼
                                           reconstruction

Losses (computed in `train_step` / `test_step`):
    L = focal_w * BinaryFocal(y, p_hat)
      + recon_w  * MSE(x, x_hat)          (auxiliary, keeps decoder useful)
      + beta     * KL(q(z|x) || N(0, I))

Serialization
-------------
Custom classes (Sampling, VAEClassifier) are registered with
@keras.saving.register_keras_serializable so that saving/loading via
the `.keras` format works out of the box — Keras can reconstruct the
computational graph without needing external references.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import math

import tensorflow as tf
import keras
from tensorflow.keras import layers

from .losses import (
    binary_focal_loss,
    kl_divergence_standard_normal,
    logsumexp_tail_loss,
    correlation_loss,
    complex_correlation_loss,
    masked_mass_regression_loss,
)


# --------------------------------------------------------------------- #
# Activations & Initializers
# --------------------------------------------------------------------- #

def _activation(name: str) -> layers.Layer:
    """Return a fresh activation layer — one instance per call site."""
    key = name.lower()
    if key in ("linear", "identity", "none"):
        return layers.Activation("linear")
    if key in ("silu", "swish"):
        return layers.Activation("swish")
    if key == "gelu":
        return layers.Activation("gelu")
    if key == "relu":
        return layers.ReLU()
    if key in ("leaky_relu", "leakyrelu"):
        try:
            return layers.LeakyReLU(negative_slope=0.01)
        except TypeError:
            return layers.LeakyReLU(alpha=0.01)
    if key == "elu":
        return layers.ELU()
    if key == "selu":
        return layers.Activation("selu")
    if key == "tanh":
        return layers.Activation("tanh")
    if key == "sigmoid":
        return layers.Activation("sigmoid")
    raise ValueError(f"Unknown activation '{name}'")


def _kernel_initializer(activation: str) -> str:
    """
    Return the weight initializer matched to the activation function.

    * he_normal     — rectifier-family (ReLU, LeakyReLU, ELU, GELU, SiLU)
    * glorot_normal — symmetric around zero (Tanh, Sigmoid)
    * lecun_normal  — self-normalizing (SELU)
    * glorot_uniform — linear outputs (fallback)
    """
    key = activation.lower()
    if key in ("relu", "leaky_relu", "leakyrelu", "elu", "gelu", "silu", "swish"):
        return "he_normal"
    if key in ("tanh", "sigmoid"):
        return "glorot_normal"
    if key == "selu":
        return "lecun_normal"
    return "glorot_uniform"


def _resolve_activation_list(
    spec: str | List[str],
    expected_len: int,
    context: str,
) -> List[str]:
    """Broadcast a single activation or validate an explicit per-layer list."""
    if expected_len == 0:
        return []
    if isinstance(spec, str):
        return [spec] * expected_len
    if len(spec) != expected_len:
        raise ValueError(
            f"{context} activation list must have length {expected_len}, "
            f"got {len(spec)}."
        )
    return list(spec)


def _resolve_decoder_activation_list(
    spec: str | List[str],
    num_stages: int,
) -> tuple[List[str], bool]:
    """
    Return per-stage decoder activations and whether to preserve legacy string
    behavior (activation only on the first n - 1 Conv1DTranspose stages).
    """
    if num_stages == 0:
        return [], False
    if isinstance(spec, str):
        return [spec] * num_stages, True

    activations = list(spec)
    if len(activations) == num_stages - 1:
        activations.append("linear")
    elif len(activations) != num_stages:
        raise ValueError(
            "decoder activation list must have length "
            f"{num_stages} or {num_stages - 1}, got {len(activations)}."
        )
    return activations, False


# --------------------------------------------------------------------- #
# Sampling layer (reparameterization trick)
# --------------------------------------------------------------------- #

@keras.saving.register_keras_serializable(package="vae")
class QuadratureConv1D(layers.Layer):
    """
    Complex/quadrature 1D front-end for I/Q data.

    Given input channels [I, Q], the layer learns a complex filter
    h = h_re + i h_im and computes the paired responses

        y_re = conv(I, h_re) - conv(Q, h_im)
        y_im = conv(I, h_im) + conv(Q, h_re)

    The output can either preserve the paired real/imaginary responses or
    collapse them into a phase-invariant magnitude.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        output_mode: str = "magnitude",
        eps: float = 1e-8,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding).upper()
        self.use_bias = bool(use_bias)
        self.output_mode = str(output_mode).lower()
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        if self.output_mode not in {"magnitude", "real_imag"}:
            raise ValueError(
                "QuadratureConv1D output_mode must be 'magnitude' or 'real_imag'."
            )

    def build(self, input_shape):
        if input_shape[-1] != 2:
            raise ValueError(
                f"QuadratureConv1D expects 2 input channels [I, Q], got {input_shape[-1]}."
            )

        kernel_shape = (self.kernel_size, 1, self.filters)
        self.kernel_re = self.add_weight(
            name="kernel_re",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.kernel_im = self.add_weight(
            name="kernel_im",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
        )

        if self.use_bias:
            self.bias_re = self.add_weight(
                name="bias_re",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
            self.bias_im = self.add_weight(
                name="bias_im",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs):
        i_chan = inputs[..., 0:1]
        q_chan = inputs[..., 1:2]

        ii = tf.nn.conv1d(i_chan, self.kernel_re, stride=self.strides, padding=self.padding)
        qq = tf.nn.conv1d(q_chan, self.kernel_im, stride=self.strides, padding=self.padding)
        iq = tf.nn.conv1d(i_chan, self.kernel_im, stride=self.strides, padding=self.padding)
        qi = tf.nn.conv1d(q_chan, self.kernel_re, stride=self.strides, padding=self.padding)

        y_re = ii - qq
        y_im = iq + qi

        if self.use_bias:
            y_re = y_re + self.bias_re
            y_im = y_im + self.bias_im

        if self.output_mode == "real_imag":
            return tf.concat([y_re, y_im], axis=-1)

        power = tf.square(y_re) + tf.square(y_im)
        return tf.sqrt(power + self.eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "output_mode": self.output_mode,
            "eps": self.eps,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

@keras.saving.register_keras_serializable(package="vae")
class Sampling(layers.Layer):
    """z = z_mean + exp(0.5 * z_log_var) * eps,  eps ~ N(0, I)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------------- #
# Encoder / Decoder builders
# --------------------------------------------------------------------- #

def build_cnn_encoder(
    input_shape: Tuple[int, int],
    num_filters_per_layer: List[int],
    kernel_sizes_per_layer: List[int],
    strides_per_layer: List[int],
    latent_dim: int,
    activation: str | List[str] = "silu",
    use_quadrature_frontend: bool = False,
    quadrature_output_mode: str = "magnitude",
    name: str = "encoder",
) -> tf.keras.Model:
    """Build the Conv1D encoder returning (z_mean, z_log_var)."""
    activations = _resolve_activation_list(
        activation, len(num_filters_per_layer), "encoder"
    )
    inp = tf.keras.Input(shape=input_shape, name="encoder_input")
    x = inp
    start_idx = 0

    if use_quadrature_frontend:
        if input_shape[-1] != 2:
            raise ValueError(
                "Quadrature front-end requires I/Q input with exactly 2 channels. "
                "Set use_I_Q=True and use_amps=False in the config."
            )
        first_act_name = activations[0]
        if quadrature_output_mode == "magnitude":
            quad_init = _kernel_initializer(first_act_name)
        else:
            # Preserve the raw paired real/imaginary responses before later
            # layers mix them, so keep the front-end itself linearly initialized.
            quad_init = "glorot_uniform"

        x = QuadratureConv1D(
            filters=num_filters_per_layer[0],
            kernel_size=kernel_sizes_per_layer[0],
            strides=strides_per_layer[0],
            padding="same",
            use_bias=True,
            output_mode=quadrature_output_mode,
            kernel_initializer=quad_init,
            bias_initializer="zeros",
            name="enc_quad_conv1",
        )(x)

        if quadrature_output_mode == "magnitude":
            x = _activation(first_act_name)(x)

        start_idx = 1

    for i in range(start_idx, len(num_filters_per_layer)):
        f = num_filters_per_layer[i]
        k = kernel_sizes_per_layer[i]
        s = strides_per_layer[i]
        act_name = activations[i]
        x = layers.Conv1D(
            filters=f, kernel_size=k, strides=s, padding="same",
            kernel_initializer=_kernel_initializer(act_name),
            bias_initializer="zeros",
            name=f"enc_conv{i+1}",
        )(x)
        x = _activation(act_name)(x)
    x = layers.Flatten(name="enc_flatten")(x)
    # Linear projection — Glorot (no rectifier follows)
    z_mean = layers.Dense(
        latent_dim, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", name="z_mean",
    )(x)
    z_log_var = layers.Dense(
        latent_dim, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", name="z_log_var",
    )(x)
    return tf.keras.Model(inp, [z_mean, z_log_var], name=name)


def build_cnn_decoder(
    output_shape: Tuple[int, int],
    num_filters_per_layer: List[int],
    kernel_sizes_per_layer: List[int],
    strides_per_layer: List[int],
    latent_dim: int,
    activation: str | List[str] = "silu",
    name: str = "decoder",
) -> tf.keras.Model:
    """
    Build the Conv1DTranspose decoder mirroring the encoder in reverse.
    """
    decoder_activations, use_legacy_activation_pattern = _resolve_decoder_activation_list(
        activation, len(num_filters_per_layer)
    )
    window_size, channels = output_shape
    total_stride = 1
    for s in strides_per_layer:
        total_stride *= s
    bottleneck_timesteps = math.ceil(window_size / total_stride)
    bottleneck_filters = num_filters_per_layer[-1]

    latent_inp = tf.keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(
        bottleneck_timesteps * bottleneck_filters,
        kernel_initializer=_kernel_initializer(decoder_activations[0]),
        bias_initializer="zeros",
        name="dec_dense_expand",
    )(latent_inp)
    x = layers.Reshape((bottleneck_timesteps, bottleneck_filters), name="dec_reshape")(x)

    rev_strides = list(reversed(strides_per_layer))
    rev_kernels = list(reversed(kernel_sizes_per_layer))
    rev_filters = list(reversed(num_filters_per_layer))

    for i, (s, k, act_name) in enumerate(
        zip(rev_strides, rev_kernels, decoder_activations)
    ):
        target_filters = rev_filters[i + 1] if i + 1 < len(rev_filters) else rev_filters[-1]
        x = layers.Conv1DTranspose(
            filters=target_filters, kernel_size=k, strides=s, padding="same",
            kernel_initializer=_kernel_initializer(act_name),
            bias_initializer="zeros",
            name=f"dec_convT{i+1}",
        )(x)
        if use_legacy_activation_pattern:
            if i < len(rev_strides) - 1:
                x = _activation(act_name)(x)
        else:
            x = _activation(act_name)(x)

    # Final projection — linear output, Glorot init
    out = layers.Conv1D(
        filters=channels, kernel_size=1, padding="same",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation="linear", name="dec_output",
    )(x)
    return tf.keras.Model(latent_inp, out, name=name)


def build_classifier_head(
    input_dim: int,
    hidden_units: List[int],
    dropout: float,
    activation: str | List[str] = "silu",
    name: str = "classifier_head",
) -> tf.keras.Model:
    """Small MLP mapping deterministic latent statistics -> single logit."""
    activations = _resolve_activation_list(
        activation, len(hidden_units), "classifier"
    )
    feature_inp = tf.keras.Input(shape=(input_dim,), name="cls_input")
    x = feature_inp
    for i, (u, act_name) in enumerate(zip(hidden_units, activations)):
        x = layers.Dense(
            u, kernel_initializer=_kernel_initializer(act_name), bias_initializer="zeros",
            name=f"cls_dense{i+1}",
        )(x)
        x = _activation(act_name)(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"cls_drop{i+1}")(x)
    # Output logit — linear, Glorot
    logit = layers.Dense(
        1, activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="cls_logit",
    )(x)
    return tf.keras.Model(feature_inp, logit, name=name)


# --------------------------------------------------------------------- #
# The VAE-classifier wrapper
# --------------------------------------------------------------------- #

@keras.saving.register_keras_serializable(package="vae")
class VAEClassifier(tf.keras.Model):
    """
    VAE + classifier head with a custom train/test step.

    Decorated with @register_keras_serializable so the .keras format
    can serialise + deserialise the full model without manual custom_objects.
    """

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        classifier: tf.keras.Model,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_weight: float = 1.0,
        reconstruction_weight: float = 0.05,
        kl_beta: float = 1e-4,
        classifier_samples_z: bool = False,
        use_bfl: bool = True,
        use_kl: bool = True,
        use_rec: bool = True,
        use_tail_loss: bool = False,
        lambda_tail: float = 0.05,
        tail_beta: float = 10.0,
        tail_margin: float = -2.0,
        rec_target_mode: str = "raw_input",
        use_corr_loss: bool = False,
        use_iq_correlation_loss: bool = False,
        lambda_corr: float = 0.05,
        corr_eps: float = 1e-8,
        mass_head: tf.keras.Model = None,
        use_mass_head: bool = False,
        lambda_mass: float = 0.0,
        mass_huber_delta: float = 1.0,
        mass_norm_center: float = -9.0,
        mass_norm_scale: float = 3.0,
        name: str = "vae_classifier",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.sampling = Sampling(name="sampling")

        # Store scalar hyper-params for get_config / serialization
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_weight = float(focal_weight)          # == lambda_bfl
        self.reconstruction_weight = float(reconstruction_weight)  # == lambda_rec
        self._init_kl_beta = float(kl_beta)
        # When True, the classifier reads the *sampled* z (shared with the
        # decoder). When False, it reads concat[z_mean, z_log_var] — the
        # current production behavior.
        self.classifier_samples_z = bool(classifier_samples_z)

        # --- Optional loss components (B) --- #
        self.use_bfl = bool(use_bfl)
        self.use_kl = bool(use_kl)
        self.use_rec = bool(use_rec)
        self.use_tail_loss = bool(use_tail_loss)
        self.lambda_tail = float(lambda_tail)
        self.tail_beta = float(tail_beta)
        self.tail_margin = float(tail_margin)
        if rec_target_mode not in ("raw_input", "clean_signal"):
            raise ValueError(
                "rec_target_mode must be 'raw_input' or 'clean_signal', "
                f"got '{rec_target_mode}'."
            )
        self.rec_target_mode = str(rec_target_mode)
        self.use_corr_loss = bool(use_corr_loss)
        self.use_iq_correlation_loss = bool(use_iq_correlation_loss)
        if self.use_iq_correlation_loss:
            output_channels = self.decoder.output_shape[-1]
            if output_channels != 2:
                raise ValueError(
                    "use_iq_correlation_loss=True requires a two-channel I/Q "
                    f"decoder output; received {output_channels} channel(s)."
                )
            if not self.use_corr_loss or self.rec_target_mode != "clean_signal":
                raise ValueError(
                    "use_iq_correlation_loss=True requires use_corr_loss=True "
                    "and rec_target_mode='clean_signal'."
                )
        self.lambda_corr = float(lambda_corr)
        self.corr_eps = float(corr_eps)

        # --- Optional auxiliary mass-estimation head --- #
        # A small regression head on the latent features predicting the
        # normalized log10 PBH mass of signal windows. With lambda_mass == 0
        # the term is still computed but scaled by zero, so the head receives
        # exactly zero gradient ("does nothing") while staying in the graph.
        self.mass_head = mass_head
        self.use_mass_head = bool(use_mass_head)
        if self.use_mass_head and self.mass_head is None:
            raise ValueError(
                "use_mass_head=True requires a mass_head sub-model "
                "(built automatically by build_vae_classifier)."
            )
        self.lambda_mass = float(lambda_mass)
        self.mass_huber_delta = float(mass_huber_delta)
        self.mass_norm_center = float(mass_norm_center)
        self.mass_norm_scale = float(mass_norm_scale)

        # beta is a Variable so a callback can anneal it.
        self.kl_beta = tf.Variable(
            float(kl_beta), trainable=False, dtype=tf.float32, name="kl_beta"
        )

        # Metric trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.focal_tracker = tf.keras.metrics.Mean(name="focal_loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.tail_tracker = tf.keras.metrics.Mean(name="tail_loss")
        self.corr_tracker = tf.keras.metrics.Mean(name="corr_loss")
        self.mass_tracker = tf.keras.metrics.Mean(name="mass_loss")
        self.beta_tracker = tf.keras.metrics.Mean(name="kl_beta")
        self.auc_tracker = tf.keras.metrics.AUC(name="auc", from_logits=True)

    # -- serialization ---------------------------------------------- #
    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "classifier": keras.saving.serialize_keras_object(self.classifier),
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "focal_weight": self.focal_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "kl_beta": self._init_kl_beta,
            "classifier_samples_z": self.classifier_samples_z,
            "use_bfl": self.use_bfl,
            "use_kl": self.use_kl,
            "use_rec": self.use_rec,
            "use_tail_loss": self.use_tail_loss,
            "lambda_tail": self.lambda_tail,
            "tail_beta": self.tail_beta,
            "tail_margin": self.tail_margin,
            "rec_target_mode": self.rec_target_mode,
            "use_corr_loss": self.use_corr_loss,
            "use_iq_correlation_loss": self.use_iq_correlation_loss,
            "lambda_corr": self.lambda_corr,
            "corr_eps": self.corr_eps,
            "mass_head": (
                keras.saving.serialize_keras_object(self.mass_head)
                if self.mass_head is not None else None
            ),
            "use_mass_head": self.use_mass_head,
            "lambda_mass": self.lambda_mass,
            "mass_huber_delta": self.mass_huber_delta,
            "mass_norm_center": self.mass_norm_center,
            "mass_norm_scale": self.mass_norm_scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        classifier = keras.saving.deserialize_keras_object(config.pop("classifier"))
        mass_head_cfg = config.pop("mass_head", None)
        mass_head = (
            keras.saving.deserialize_keras_object(mass_head_cfg)
            if mass_head_cfg is not None else None
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
            mass_head=mass_head,
            **config,
        )

    # -- public helpers --------------------------------------------- #
    def set_beta(self, value: float) -> None:
        self.kl_beta.assign(float(value))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.focal_tracker,
            self.recon_tracker,
            self.kl_tracker,
            self.tail_tracker,
            self.corr_tracker,
            self.mass_tracker,
            self.beta_tracker,
            self.auc_tracker,
        ]

    # -- forward ---------------------------------------------------- #
    def classifier_features(self, z_mean, z_log_var, z=None):
        """
        Build the classifier input.

        If `classifier_samples_z` is True and a sampled `z` is provided, the
        classifier reads the sample directly. This is the "shared sample"
        regime — the classifier sees the same z = μ + σ * ε that the decoder
        sees, and classifier-loss gradients flow back through the
        reparameterization noise the same way the decoder's do.

        If `classifier_samples_z` is False, the classifier reads concat[μ, log σ²]
        deterministically (current production behavior). For compatibility with
        older checkpoints whose classifier head only expects `latent_dim` inputs,
        we also fall back to plain `z_mean`.
        """
        expected_dim = self.classifier.input_shape[-1]
        latent_dim = z_mean.shape[-1]
        if expected_dim is None or latent_dim is None:
            raise ValueError("Classifier and encoder must have known latent dimensions.")
        expected_dim = int(expected_dim)
        latent_dim = int(latent_dim)

        if self.classifier_samples_z:
            if expected_dim != latent_dim:
                raise ValueError(
                    "classifier_samples_z=True requires the classifier head to "
                    f"expect `latent_dim` inputs ({latent_dim}); got {expected_dim}."
                )
            if z is None:
                # Analysis/eval path without an explicit sample: use z_mean
                # (the posterior point estimate) so the classifier output is
                # deterministic. Training-path always provides an explicit z.
                return z_mean
            return z

        if expected_dim == 2 * latent_dim:
            return tf.concat([z_mean, z_log_var], axis=-1)
        if expected_dim == latent_dim:
            return z_mean
        raise ValueError(
            "Classifier input dimension does not match encoder outputs: "
            f"expected {expected_dim}, latent_dim={latent_dim}."
        )

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        if self.classifier_samples_z:
            z = self.sampling([z_mean, z_log_var])
            cls_features = self.classifier_features(z_mean, z_log_var, z=z)
        else:
            cls_features = self.classifier_features(z_mean, z_log_var)
        logits = self.classifier(cls_features, training=training)
        return logits

    def encode(self, inputs):
        return self.encoder(inputs, training=False)

    def decode(self, z):
        return self.decoder(z, training=False)

    def predict_proba(self, inputs):
        logits = self(inputs, training=False)
        return tf.sigmoid(logits)

    # -- loss ------------------------------------------------------- #
    def _resolve_rec_target(self, x, clean):
        """Pick the reconstruction target according to rec_target_mode (B4)."""
        if self.rec_target_mode == "clean_signal":
            if clean is None:
                raise ValueError(
                    "rec_target_mode='clean_signal' but the dataset did not "
                    "provide clean_signal_window. Run preprocessing with "
                    "save_clean_signals=True and include_clean_in_datasets=True "
                    "so batches are (x, y, clean_signal_window) 3-tuples."
                )
            return clean
        return x

    def _compute_losses(self, x, y_true, training, clean=None, mass=None):
        z_mean, z_log_var = self.encoder(x, training=training)
        z = self.sampling([z_mean, z_log_var])
        # Classifier reads either the shared sampled z or concat[μ, log σ²]
        # depending on `classifier_samples_z`. The decoder always reads z.
        cls_features = self.classifier_features(z_mean, z_log_var, z=z)
        logits = self.classifier(cls_features, training=training)
        recon = self.decoder(z, training=training)

        y_true_f = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)

        # B1: Classification loss (window-level binary focal loss)
        focal = (
            binary_focal_loss(
                y_true_f, logits,
                gamma=self.focal_gamma, alpha=self.focal_alpha, from_logits=True,
            )
            if self.use_bfl else zero
        )

        # B3: logsumexp tail loss on negative logits
        tail = (
            logsumexp_tail_loss(
                logits, y_true_f, beta=self.tail_beta, margin=self.tail_margin,
            )
            if self.use_tail_loss else zero
        )

        # B4/B5: Reconstruction (MSE) against the selected target
        rec_target = self._resolve_rec_target(x, clean)
        recon_loss = (
            tf.reduce_mean(tf.square(rec_target - recon))
            if self.use_rec else zero
        )

        # B6: correlation loss on positive/signal windows only
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

        # B2: KL (analytic, N(0,I) prior)
        kl = kl_divergence_standard_normal(z_mean, z_log_var) if self.use_kl else zero

        # Auxiliary mass regression (signal windows only). Always computed
        # when the head exists, scaled by lambda_mass — lambda_mass = 0 gives
        # exactly zero gradients to the head without disconnecting it from
        # the graph (avoids None-gradient errors in apply_gradients).
        if self.use_mass_head and self.mass_head is not None:
            if mass is None:
                raise ValueError(
                    "use_mass_head=True but the dataset did not provide the "
                    "per-window log-mass target. Run preprocessing with "
                    "save_mass_targets=True and include_mass_in_datasets=True "
                    "so batches are (x, y, clean, log10_mass) 4-tuples."
                )
            mass_pred = self.mass_head(cls_features, training=training)
            mass_target_norm = (
                (tf.cast(mass, tf.float32) - self.mass_norm_center)
                / self.mass_norm_scale
            )
            mass_loss = masked_mass_regression_loss(
                mass_pred, mass_target_norm, y_true_f,
                delta=self.mass_huber_delta,
            )
        else:
            mass_loss = zero

        total = (
            self.focal_weight * focal
            + self.lambda_tail * tail
            + self.reconstruction_weight * recon_loss
            + self.lambda_corr * corr
            + self.lambda_mass * mass_loss
            + self.kl_beta * kl
        )
        return total, focal, recon_loss, kl, tail, corr, mass_loss, logits

    # -- train / eval ----------------------------------------------- #
    def train_step(self, data):
        x, y, clean, mass = _unpack(data)
        with tf.GradientTape() as tape:
            total, focal, recon, kl, tail, corr, mass_l, logits = self._compute_losses(
                x, y, training=True, clean=clean, mass=mass
            )
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._update_trackers(total, focal, recon, kl, tail, corr, mass_l, y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, clean, mass = _unpack(data)
        total, focal, recon, kl, tail, corr, mass_l, logits = self._compute_losses(
            x, y, training=False, clean=clean, mass=mass
        )
        self._update_trackers(total, focal, recon, kl, tail, corr, mass_l, y, logits)
        return {m.name: m.result() for m in self.metrics}

    def _update_trackers(self, total, focal, recon, kl, tail, corr, mass_l, y, logits):
        self.total_loss_tracker.update_state(total)
        self.focal_tracker.update_state(focal)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.tail_tracker.update_state(tail)
        self.corr_tracker.update_state(corr)
        self.mass_tracker.update_state(mass_l)
        self.beta_tracker.update_state(self.kl_beta)
        y_f = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        self.auc_tracker.update_state(y_f, logits)

    def predict_log10_mass(self, inputs):
        """
        De-normalized log10(m / M_sun) prediction per window. Only meaningful
        for windows the classifier flags as signal; requires use_mass_head.
        """
        if self.mass_head is None:
            raise ValueError("This model was built without a mass head.")
        z_mean, z_log_var = self.encoder(inputs, training=False)
        cls_features = self.classifier_features(z_mean, z_log_var)
        pred_norm = self.mass_head(cls_features, training=False)
        return pred_norm * self.mass_norm_scale + self.mass_norm_center


def _unpack(data):
    """
    Accept (x, y, clean, log10_mass), (x, y, clean), (x, y), or just x.

    The 3-tuple form is produced by the preprocessing pipeline when
    include_clean_in_datasets=True; the 4-tuple form additionally carries the
    per-window log10 PBH mass (include_mass_in_datasets=True; sentinel value
    for noise windows — always masked by the labels in the loss).
    """
    if isinstance(data, (tuple, list)):
        if len(data) == 2:
            return data[0], data[1], None, None
        if len(data) == 3:
            return data[0], data[1], data[2], None
        if len(data) == 4:
            return data[0], data[1], data[2], data[3]
    return data, tf.zeros(tf.shape(data)[:1], dtype=tf.float32), None, None


# --------------------------------------------------------------------- #
# One-shot factory
# --------------------------------------------------------------------- #

def build_vae_classifier(cfg) -> VAEClassifier:
    """Build a VAEClassifier straight from a VAEConfig."""
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
    # The classifier-head input dim depends on the routing choice:
    #   classifier_samples_z=True  → head reads z (latent_dim)
    #   classifier_samples_z=False → head reads concat[μ, log σ²] (2 * latent_dim)
    classifier_input_dim = (
        cfg.latent_dim if cfg.classifier_samples_z else 2 * cfg.latent_dim
    )
    classifier = build_classifier_head(
        input_dim=classifier_input_dim,
        hidden_units=cfg.classifier_hidden_units,
        dropout=cfg.classifier_dropout,
        activation=classifier_activation,
    )

    # Optional auxiliary mass-regression head: same latent features as the
    # classifier, small MLP, single linear output (normalized log10 mass).
    mass_head = None
    if getattr(cfg, "use_mass_head", False):
        mass_head = build_classifier_head(
            input_dim=classifier_input_dim,
            hidden_units=getattr(cfg, "mass_head_hidden", [32]),
            dropout=0.0,
            activation=classifier_activation,
            name="mass_head",
        )

    return VAEClassifier(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        focal_weight=cfg.focal_weight,
        reconstruction_weight=cfg.reconstruction_weight,
        kl_beta=cfg.kl_beta_start,
        classifier_samples_z=cfg.classifier_samples_z,
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
        mass_head=mass_head,
        use_mass_head=getattr(cfg, "use_mass_head", False),
        lambda_mass=getattr(cfg, "lambda_mass", 0.0),
        mass_huber_delta=getattr(cfg, "mass_huber_delta", 1.0),
        mass_norm_center=getattr(cfg, "mass_norm_center", -9.0),
        mass_norm_scale=getattr(cfg, "mass_norm_scale", 3.0),
        name=cfg.model_name,
    )
