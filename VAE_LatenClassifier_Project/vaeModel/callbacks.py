"""
Callbacks for the simplified VAE-classifier.

Two callbacks:

1. BetaAnnealing
   Linearly ramps `kl_beta` from `start` to `end` over `warmup_epochs`,
   holding at `end` afterwards. Same idea as the original
   BetaAnnealingCallback — just a bit cleaner and epoch-indexed from 0.

2. EventDetectionCallback
   At the end of every N epochs, runs the model over a provided validation
   dataset, sweeps thresholds, and computes:

       * event-based recall (the "≥1 window detected → event detected" logic
         from efficiency_curve.py)
       * FP per year, extrapolated from the FPR-vs-threshold curve by a
         linear fit in log-FPR space.

   The callback then picks the threshold that realises `target_fp_per_year`
   and logs both the threshold and the event recall at that threshold into
   the Keras `logs` dict so they show up in `history` / TensorBoard.

   Cost
   ----
   Running it every epoch is *doable* but expensive — one full forward pass
   over val_ds plus some NumPy post-processing. In practice I default to
   running it every epoch on a modest val set, and recommend bumping
   `detection_every_epochs` to 5–10 on larger ones. If you just want to
   watch training without the extra cost, set it to 0 to disable entirely.

Cheaper option built in
-----------------------
As a bonus, `val_auc` (from the model's own metric tracker) already gives
you a cheap scalar that correlates with detection quality. This callback is
on top of that — it gives you the *physically meaningful* number
(FP/year → event recall) rather than a summary statistic.
"""

from __future__ import annotations

import csv
import os
from typing import Optional
import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit


class BetaAnnealing(tf.keras.callbacks.Callback):
    """Linear warm-up of kl_beta from start → end over `warmup_epochs`."""

    def __init__(self, model, start: float, end: float, warmup_epochs: int):
        super().__init__()
        self._vae = model
        self.start = float(start)
        self.end = float(end)
        self.warmup_epochs = max(1, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            new_beta = self.end
        else:
            frac = epoch / max(1, self.warmup_epochs - 1) if self.warmup_epochs > 1 else 1.0
            new_beta = self.start + (self.end - self.start) * frac
        self._vae.set_beta(new_beta)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["kl_beta"] = float(self._vae.kl_beta.numpy())


# --------------------------------------------------------------------- #
# Score-quantile logging callback (C)
# --------------------------------------------------------------------- #

SCORE_QUANTILE_COLUMNS = [
    "epoch",
    "num_neg", "num_pos",
    "neg_logit_mean", "neg_logit_std",
    "neg_logit_q50", "neg_logit_q90", "neg_logit_q95",
    "neg_logit_q99", "neg_logit_q999", "neg_logit_max",
    "pos_logit_mean", "pos_logit_std",
    "pos_logit_q01", "pos_logit_q05", "pos_logit_q10",
    "pos_logit_q50", "pos_logit_q90",
    "gap_pos_q10_minus_neg_q99",
    "gap_pos_q05_minus_neg_q999",
]


def compute_score_quantile_row(y_true: np.ndarray, logits: np.ndarray, epoch: int) -> dict:
    """
    Per-class classifier-logit statistics for one epoch.

    The point of these numbers is to watch whether the logsumexp tail loss
    pulls down the extreme NEGATIVE quantiles (q99 / q999 / max) without
    collapsing the low POSITIVE quantiles (q10 / q05).
    """
    y = np.asarray(y_true).reshape(-1) > 0.5
    l = np.asarray(logits, dtype=np.float64).reshape(-1)
    neg = l[~y]
    pos = l[y]

    def _q(arr, q):
        return float(np.quantile(arr, q)) if arr.size else float("nan")

    def _stat(arr, fn):
        return float(fn(arr)) if arr.size else float("nan")

    row = {
        "epoch": int(epoch),
        "num_neg": int(neg.size),
        "num_pos": int(pos.size),
        "neg_logit_mean": _stat(neg, np.mean),
        "neg_logit_std": _stat(neg, np.std),
        "neg_logit_q50": _q(neg, 0.50),
        "neg_logit_q90": _q(neg, 0.90),
        "neg_logit_q95": _q(neg, 0.95),
        "neg_logit_q99": _q(neg, 0.99),
        "neg_logit_q999": _q(neg, 0.999),
        "neg_logit_max": _stat(neg, np.max),
        "pos_logit_mean": _stat(pos, np.mean),
        "pos_logit_std": _stat(pos, np.std),
        "pos_logit_q01": _q(pos, 0.01),
        "pos_logit_q05": _q(pos, 0.05),
        "pos_logit_q10": _q(pos, 0.10),
        "pos_logit_q50": _q(pos, 0.50),
        "pos_logit_q90": _q(pos, 0.90),
    }
    row["gap_pos_q10_minus_neg_q99"] = row["pos_logit_q10"] - row["neg_logit_q99"]
    row["gap_pos_q05_minus_neg_q999"] = row["pos_logit_q05"] - row["neg_logit_q999"]
    return row


class ScoreQuantileLogger(tf.keras.callbacks.Callback):
    """
    At the end of every epoch, scores `val_ds`, computes per-class logit
    quantiles, appends them to `<output_dir>/score_quantiles.csv`, and also
    injects the key numbers into the Keras `logs` dict.
    """

    def __init__(self, model, val_ds, output_dir: str, verbose: int = 1):
        super().__init__()
        self._vae = model
        self.val_ds = val_ds
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "score_quantiles.csv")
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        # Fresh file per training run
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_all, logit_all = [], []
        for batch in self.val_ds:
            if not isinstance(batch, (tuple, list)):
                continue
            x, y = batch[0], batch[1]
            out = self._vae(x, training=False)
            logit_all.append(np.asarray(out).reshape(-1))
            y_all.append(np.asarray(y).reshape(-1))
        if not logit_all:
            return

        row = compute_score_quantile_row(
            np.concatenate(y_all), np.concatenate(logit_all), epoch + 1
        )

        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SCORE_QUANTILE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        for key in (
            "neg_logit_q99", "neg_logit_q999", "neg_logit_max",
            "pos_logit_q10", "gap_pos_q10_minus_neg_q99",
        ):
            logs[key] = row[key]

        if self.verbose:
            print(
                f"\n[ScoreQuantileLogger] epoch {epoch+1}: "
                f"neg q99={row['neg_logit_q99']:.3f} "
                f"q999={row['neg_logit_q999']:.3f} "
                f"max={row['neg_logit_max']:.3f} | "
                f"pos q10={row['pos_logit_q10']:.3f} | "
                f"gap(q10-q99)={row['gap_pos_q10_minus_neg_q99']:.3f}"
            )


# --------------------------------------------------------------------- #
# Event-detection callback
# --------------------------------------------------------------------- #

def _linear_log_model(t, b, c):
    """ln(FPR) = b*t + c — straight line in log-FPR vs threshold."""
    return b * t + c


def _event_recall_and_fpr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):
    """
    Event-based recall + window-level FPR fraction.

    Mirrors `calculate_event_recall_and_fpr` from efficiency_curve.py but
    trimmed to the numbers the callback actually needs.
    """
    y_pred_bool = y_pred > threshold
    # Pad with zeros to catch events at the boundaries
    padded = np.concatenate(([0], y_true.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    num_events = len(starts)
    detected = 0
    for s, e in zip(starts, ends):
        if np.any(y_pred_bool[s:e]):
            detected += 1

    event_recall = detected / num_events if num_events > 0 else 0.0

    noise_mask = ~(y_true.astype(bool))
    total_noise = int(np.sum(noise_mask))
    fp_windows = int(np.sum(y_pred_bool & noise_mask))
    fpr_fraction = fp_windows / total_noise if total_noise > 0 else 0.0
    return event_recall, num_events, detected, fpr_fraction


class EventDetectionCallback(tf.keras.callbacks.Callback):
    """
    Computes an operating threshold for a target FP/year and reports the
    resulting event-based recall on the validation set.

    Logged values (in the Keras `logs` dict — they will flow into
    `history.history`):

        det_threshold       : chosen threshold at target FP/year
        det_event_recall    : event-based recall at that threshold
        det_fp_per_year     : realised FP/year at the chosen threshold
        det_num_events      : number of true events in val set
    """

    def __init__(
        self,
        model,
        val_ds,
        window_size: int,
        step_size: int,
        fs: float,
        every_epochs: int = 1,
        target_fp_per_year: float = 1.0,
        sweep_points: int = 200,
        log_fit_tail_fraction: float = 1e-4,
        min_tail_points: int = 4,
        verbose: int = 1,
    ):
        super().__init__()
        self._vae = model
        self.val_ds = val_ds
        self.window_size = window_size
        self.step_size = step_size
        self.fs = fs
        self.every_epochs = max(0, int(every_epochs))
        self.target_fp_per_year = target_fp_per_year
        self.sweep_points = sweep_points
        self.log_fit_tail_fraction = log_fit_tail_fraction
        self.min_tail_points = min_tail_points
        self.verbose = verbose

        seconds_per_year = 31_536_000
        total_samples_per_year = seconds_per_year * self.fs
        self._windows_per_year = (
            (total_samples_per_year - self.window_size) / self.step_size + 1
        )

    # -- main hook -------------------------------------------------- #
    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        if self.every_epochs == 0:
            return
        if (epoch + 1) % self.every_epochs != 0:
            return

        # 1. Score the val set once. Logits (from_logits=True).
        y_true_all, logits_all = [], []
        for batch in self.val_ds:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                continue
            out = self._vae(x, training=False)
            logits_all.append(out.numpy().reshape(-1))
            y_true_all.append(np.asarray(y).reshape(-1))

        if not logits_all:
            return
        y_true = np.concatenate(y_true_all).astype(np.int8)
        logits = np.concatenate(logits_all).astype(np.float64)

        if np.sum(y_true) == 0:
            # Nothing to measure recall against.
            if self.verbose:
                print("[EventDetectionCallback] No positive events in val set — skipped.")
            return

        # 2. Sweep thresholds from just below min noise logit to above max.
        lo = float(np.min(logits)) - 0.5
        hi = float(np.max(logits)) + 0.5
        thresholds = np.linspace(lo, hi, self.sweep_points)

        fpr_fractions = np.empty_like(thresholds)
        recalls = np.empty_like(thresholds)
        for i, t in enumerate(thresholds):
            rec, n_ev, det, fpr = _event_recall_and_fpr(
                y_true, logits, t
            )
            fpr_fractions[i] = fpr
            recalls[i] = rec

        # 3. Extrapolate to the target FP/year.
        target_fraction = self.target_fp_per_year / self._windows_per_year
        chosen_threshold = self._extrapolate_threshold(
            thresholds, fpr_fractions, target_fraction
        )

        # 4. Event recall at the chosen threshold.
        rec_at_chosen, n_events, detected, fpr_at_chosen = _event_recall_and_fpr(
            y_true, logits, chosen_threshold
        )
        fp_per_year_at_chosen = fpr_at_chosen * self._windows_per_year

        logs["det_threshold"] = float(chosen_threshold)
        logs["det_event_recall"] = float(rec_at_chosen)
        logs["det_fp_per_year"] = float(fp_per_year_at_chosen)
        logs["det_num_events"] = float(n_events)

        if self.verbose:
            print(
                f"\n[EventDetectionCallback] epoch {epoch+1}: "
                f"threshold={chosen_threshold:.4f} | "
                f"event_recall={rec_at_chosen:.3f} "
                f"({detected}/{n_events}) | "
                f"FP/year={fp_per_year_at_chosen:.3f}"
            )

    # -- helpers ---------------------------------------------------- #
    def _extrapolate_threshold(self, t, r, target_fraction):
        """Linear fit in log(FPR) over the low-FPR tail, then invert."""
        max_r = np.max(r)
        if max_r <= 0:
            return float(t[-1])

        tail_mask = (r > 0) & (r <= self.log_fit_tail_fraction * max_r)
        if np.sum(tail_mask) >= self.min_tail_points:
            try:
                popt, _ = curve_fit(
                    _linear_log_model,
                    t[tail_mask],
                    np.log(r[tail_mask]),
                )
                b, c = popt
                if b < 0:
                    return float((np.log(target_fraction) - c) / b)
            except Exception:
                pass

        # Fallback: use the last two measured points with r > 0.
        pos = np.where(r > 0)[0]
        if len(pos) >= 2:
            t_last, r_last = t[pos[-1]], r[pos[-1]]
            t_prev, r_prev = t[pos[-2]], r[pos[-2]]
            slope = (np.log(r_last) - np.log(r_prev)) / (t_last - t_prev)
            if slope < 0:
                return float(t_last + (np.log(target_fraction) - np.log(r_last)) / slope)

        # Last resort: the highest threshold we actually measured.
        return float(t[-1])
