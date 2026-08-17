"""
Loss functions for the simplified VAE-classifier.

Two building blocks:

1. `binary_focal_loss`
   Focal loss from Lin et al. (2017), "Focal Loss for Dense Object Detection".
   In short: standard BCE up-weights the rarer class (alpha) and down-weights
   easy examples by a factor (1 - p_t)^gamma. For spike-like signals where
   *most* windows are noise, focal loss prevents the easy negatives from
   dominating the gradient, which is exactly the failure mode that pushed a
   pure-MSE VAE to learn the mean.

2. `kl_divergence_standard_normal`
   Analytic KL divergence between a Gaussian with per-dim diagonal covariance
   and the standard normal prior. Averaged over the batch.

Both return scalar tensors.
"""

import tensorflow as tf


def binary_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    from_logits: bool = True,
) -> tf.Tensor:
    """
    Binary focal loss, reduced to a scalar (mean over the batch).

    Parameters
    ----------
    y_true : 0/1 labels, broadcastable to y_pred.
    y_pred : logits if `from_logits=True`, otherwise probabilities in [0, 1].
    gamma  : focusing parameter. Larger → more focus on hard examples.
    alpha  : class balance for the positive class. 0.25 usually works well
             when positives are rare.
    """
    y_true = tf.cast(y_true, tf.float32)

    if from_logits:
        # Stable log-sigmoid form.
        # BCE = max(x,0) - x*z + log(1 + exp(-|x|))
        logits = tf.cast(y_pred, tf.float32)
        prob = tf.sigmoid(logits)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    else:
        prob = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        ce = -(y_true * tf.math.log(prob) + (1.0 - y_true) * tf.math.log(1.0 - prob))

    # p_t = p if y=1 else 1-p
    p_t = y_true * prob + (1.0 - y_true) * (1.0 - prob)
    # alpha_t = alpha if y=1 else (1-alpha)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

    modulating = tf.pow(1.0 - p_t, gamma)
    loss = alpha_t * modulating * ce
    return tf.reduce_mean(loss)


def kl_divergence_standard_normal(
    z_mean: tf.Tensor, z_log_var: tf.Tensor
) -> tf.Tensor:
    """
    KL( N(mu, sigma^2) || N(0, 1) ) summed over latent dims, averaged over batch.

        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """
    kl_per_sample = -0.5 * tf.reduce_sum(
        1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    return tf.reduce_mean(kl_per_sample)


def logsumexp_tail_loss(
    logits: tf.Tensor,
    labels: tf.Tensor,
    beta: float = 10.0,
    margin: float = -2.0,
) -> tf.Tensor:
    """
    Smooth-max tail loss on NEGATIVE-class classifier logits.

    Focuses the gradient on the largest (worst) negative logits without hard
    top-k selection — exactly the windows that dominate the FP/year tail.

        excess = softplus(neg_logits - margin)
        loss   = logsumexp(beta * excess) / beta - log(N_neg) / beta
               = (1/beta) * log( mean( exp(beta * excess) ) )      [stable form]

    Returns 0 if the batch contains no negatives.
    """
    logits = tf.reshape(tf.cast(logits, tf.float32), [-1])
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1])

    neg_mask = labels < 0.5
    neg_logits = tf.boolean_mask(logits, neg_mask)
    n_neg = tf.shape(neg_logits)[0]

    def _loss():
        excess = tf.math.softplus(neg_logits - margin)
        # stable log-mean-exp
        lse = tf.reduce_logsumexp(beta * excess)
        return lse / beta - tf.math.log(tf.cast(n_neg, tf.float32)) / beta

    return tf.cond(
        n_neg > 0,
        _loss,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def correlation_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    labels: tf.Tensor = None,
    eps: float = 1e-8,
) -> tf.Tensor:
    """
    1 - Pearson correlation between reconstruction and target, averaged over
    samples. If `labels` is given, ONLY positive/signal samples (label == 1)
    contribute; returns 0 if the batch has no positives.

    pred / target: [B, T, C] (or any [B, ...]); flattened over all non-batch
    dims, per-sample mean subtracted before the correlation.
    """
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    batch = tf.shape(pred)[0]
    p = tf.reshape(pred, [batch, -1])
    t = tf.reshape(target, [batch, -1])

    if labels is not None:
        labels_f = tf.reshape(tf.cast(labels, tf.float32), [-1])
        pos_mask = labels_f > 0.5
        p = tf.boolean_mask(p, pos_mask)
        t = tf.boolean_mask(t, pos_mask)

    n_samples = tf.shape(p)[0]

    def _loss():
        p_c = p - tf.reduce_mean(p, axis=1, keepdims=True)
        t_c = t - tf.reduce_mean(t, axis=1, keepdims=True)
        num = tf.reduce_sum(p_c * t_c, axis=1)
        den = tf.norm(p_c, axis=1) * tf.norm(t_c, axis=1) + eps
        corr = num / den
        return tf.reduce_mean(1.0 - corr)

    return tf.cond(
        n_samples > 0,
        _loss,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def complex_correlation_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    labels: tf.Tensor = None,
    eps: float = 1e-8,
) -> tf.Tensor:
    """
    Phase-invariant normalized complex correlation loss for I/Q waveforms.

    `pred` and `target` must have shape [B, T, 2], with channels ordered as
    [I, Q]. The per-sample complex means are removed before computing

        loss = 1 - |sum_t pred(t) * conj(target(t))|
                   / (||pred||_2 ||target||_2).

    The magnitude makes the score invariant to one global phase rotation while
    still requiring the reconstructed oscillatory complex waveform to match.
    If `labels` is provided, only positive/signal samples contribute.
    """
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)
    tf.debugging.assert_rank(pred, 3, message="I/Q correlation expects [B, T, 2].")
    tf.debugging.assert_rank(target, 3, message="I/Q correlation expects [B, T, 2].")
    tf.debugging.assert_equal(
        tf.shape(pred)[-1],
        2,
        message="I/Q correlation requires exactly two channels ordered [I, Q].",
    )
    tf.debugging.assert_equal(
        tf.shape(target)[-1],
        2,
        message="I/Q correlation requires exactly two channels ordered [I, Q].",
    )

    p = tf.complex(pred[..., 0], pred[..., 1])
    t = tf.complex(target[..., 0], target[..., 1])

    if labels is not None:
        labels_f = tf.reshape(tf.cast(labels, tf.float32), [-1])
        pos_mask = labels_f > 0.5
        p = tf.boolean_mask(p, pos_mask)
        t = tf.boolean_mask(t, pos_mask)

    n_samples = tf.shape(p)[0]

    def _loss():
        p_c = p - tf.reduce_mean(p, axis=1, keepdims=True)
        t_c = t - tf.reduce_mean(t, axis=1, keepdims=True)
        numerator = tf.abs(tf.reduce_sum(p_c * tf.math.conj(t_c), axis=1))
        p_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(p_c)), axis=1))
        t_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(t_c)), axis=1))
        corr = numerator / (p_norm * t_norm + tf.cast(eps, tf.float32))
        corr = tf.clip_by_value(corr, 0.0, 1.0)
        return tf.reduce_mean(1.0 - corr)

    return tf.cond(
        n_samples > 0,
        _loss,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def masked_mass_regression_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    labels: tf.Tensor,
    delta: float = 1.0,
) -> tf.Tensor:
    """
    Huber regression loss on the (normalized) log10 PBH mass, evaluated ONLY
    on signal windows (label == 1). Noise windows have no defined mass and
    never contribute. Returns 0 if the batch contains no positives.

    pred   : [B, 1] mass-head output (normalized log-mass units)
    target : [B] or [B, 1] normalized log-mass (sentinel for noise windows,
             which is irrelevant because of the label mask)
    labels : [B] or [B, 1] window labels
    """
    pred = tf.reshape(tf.cast(pred, tf.float32), [-1])
    target = tf.reshape(tf.cast(target, tf.float32), [-1])
    labels_f = tf.reshape(tf.cast(labels, tf.float32), [-1])

    pos_mask = labels_f > 0.5
    p = tf.boolean_mask(pred, pos_mask)
    t = tf.boolean_mask(target, pos_mask)
    n_pos = tf.shape(p)[0]

    def _loss():
        err = tf.abs(p - t)
        quad = tf.minimum(err, delta)
        return tf.reduce_mean(0.5 * tf.square(quad) + delta * (err - quad))

    return tf.cond(
        n_pos > 0,
        _loss,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
