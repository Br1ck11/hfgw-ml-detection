"""
Gradient inspection callback for the ablation VAE-classifier.

What it logs
------------
1. Per-layer L2 gradient norms of the *combined* (post-weighting) gradient
   that the optimizer actually applies. Cheap; logged every step by default.

2. Per-layer L2 gradient norms of the **per-loss decomposition on the
   encoder**, broken into the three terms:
       w_focal * focal_loss     →  g_focal
       w_recon * recon_loss     →  g_recon
       beta    * kl_loss        →  g_kl
   These let you see exactly how much each loss is moving each encoder layer.

3. Pairwise cosine similarity between (g_focal, g_recon, g_kl) on the
   *flattened encoder gradient vector*. Tells you whether two loss terms
   are pulling the encoder in the same direction or fighting each other.

   Cosine ≈ +1     same direction (loss B is helping loss A's update)
   Cosine ≈  0     orthogonal     (loss B has no first-order effect on A)
   Cosine ≈ -1     opposite        (loss B is *undoing* loss A's update)

   This is exactly the diagnostic for the "shared sample → classifier helps
   the decoder" hypothesis: when the decoder really benefits from the
   classifier, you should see cos(g_focal, g_recon) on the encoder be
   meaningfully positive. In the deterministic concat[μ, log σ²] mode it
   typically drops toward zero because the classifier no longer touches the
   reparameterization path.

4. Total norms (one number per loss) and the ratio of each term's norm to
   the combined norm. Catches a loss that's vanishing or dominating.

5. Per-loss norms on all trainable layers plus coarse encoder / decoder /
   classifier summaries. This shows whether a given loss mostly stays in its
   "own" branch or actually reaches the shared trunk.

Outputs
-------
* `gradients.csv`            one row per logged step (long, all scalars)
* `gradients_summary.json`   final-row summary at end of training
* TensorBoard scalars in `<out_dir>/tb/` if you pass `tb_writer=True`

Cost & how to enable
--------------------
Per-loss decomposition is roughly N additional full-model backward passes
(N = 3 here) per logged step. To keep this cheap, set
`decompose_every_n_steps` to e.g. 50 — you still get a clean curve.

`run_eagerly` is forced to True while the inspector is active, because the
inspector needs Python control flow inside `train_step`. The cost on this
model size is small but non-zero (roughly 1.3-2x slower per step). For
production runs, just don't attach the inspector.

Wire-up:
    inspector = GradientInspector(out_dir=run_dir + "/grad_logs",
                                   decompose_every_n_steps=50,
                                   layer_norms_every_n_steps=1)
    model.fit(..., callbacks=[..., inspector])

The callback attaches itself to `model.gradient_logger` automatically and
sets `model.run_eagerly = True`.
"""

from __future__ import annotations

import os
import csv
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _layer_key(var) -> str:
    """
    Group trainable variables by layer using the first path segment.

    In newer Keras versions `var.name` is often just `"kernel"` or `"bias"`,
    which is too coarse for layer-wise diagnostics. The full identifier lives
    in `var.path`, e.g. `enc_conv1/kernel` or `z_mean/bias`. We prefer that and
    only fall back to `name` if `path` is unavailable.

    This gives keys like `enc_conv1`, `z_mean`, `cls_dense2`, etc. — the actual
    layer granularity we want in the gradient plots.
    """
    var_path = getattr(var, "path", None)
    if isinstance(var_path, str) and var_path:
        return var_path.split("/", 1)[0]
    return var.name.split("/", 1)[0]


def _per_layer_norms(grads, vars_) -> Dict[str, float]:
    """L2 norm per layer. None gradients (var unaffected by loss) → 0."""
    by_layer: Dict[str, float] = defaultdict(float)
    for g, v in zip(grads, vars_):
        if g is None:
            continue  # no contribution to this layer
        sq = float(tf.reduce_sum(tf.square(g)).numpy())
        by_layer[_layer_key(v)] += sq
    return {k: math.sqrt(s) for k, s in by_layer.items()}


def _scope_key(layer_name: str) -> str:
    """
    Map layer names onto coarse model components.

    The latent projection heads belong to the encoder scope because they are
    part of the shared representation pathway.
    """
    if layer_name.startswith("enc_") or layer_name in {"z_mean", "z_log_var"}:
        return "encoder"
    if layer_name.startswith("dec_"):
        return "decoder"
    if layer_name.startswith("cls_"):
        return "classifier"
    return "other"


def _select_scope(grads, vars_, scope: str):
    """Return gradients and variables that belong to one coarse scope."""
    scope_grads = []
    scope_vars = []
    for g, v in zip(grads, vars_):
        if _scope_key(_layer_key(v)) != scope:
            continue
        scope_grads.append(g)
        scope_vars.append(v)
    return scope_grads, scope_vars


def _global_norm(norms: Dict[str, float]) -> float:
    """Combine per-layer norms into one global L2 norm."""
    return math.sqrt(sum(v ** 2 for v in norms.values()))


def _flat(grads, vars_) -> np.ndarray:
    """Flatten a (possibly None-containing) gradient list. None → zeros."""
    parts = []
    for g, v in zip(grads, vars_):
        if g is None:
            parts.append(np.zeros(int(np.prod(v.shape)), dtype=np.float32))
        else:
            parts.append(g.numpy().reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------- #
# Inspector
# --------------------------------------------------------------------- #

class GradientInspector(tf.keras.callbacks.Callback):
    """
    Keras callback that logs per-layer gradient norms and per-loss gradient
    decomposition, with encoder-focused alignment diagnostics.
    """

    LOSS_NAMES = ("focal", "recon", "kl")

    def __init__(
        self,
        out_dir: str,
        decompose_every_n_steps: int = 50,
        layer_norms_every_n_steps: int = 1,
        encoder_layer_prefix: str = "enc",
        write_tensorboard: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.decompose_every = int(decompose_every_n_steps)
        self.layer_norms_every = int(layer_norms_every_n_steps)
        self.encoder_layer_prefix = encoder_layer_prefix
        self.write_tensorboard = write_tensorboard
        self.verbose = verbose

        os.makedirs(self.out_dir, exist_ok=True)
        self._csv_path = os.path.join(self.out_dir, "gradients.csv")
        self._summary_path = os.path.join(self.out_dir, "gradients_summary.json")
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: List[str] = []
        self._tb_writer = None

        # Step counter is Python-side. Bumped in on_train_batch_end so that
        # `should_log_step()` is queried *before* each train_step runs.
        self._step = 0
        self._do_layer_norms_now = False
        self._do_decompose_now = False
        self._last_row: Dict[str, float] = {}

    # ---- Keras lifecycle --------------------------------------------- #
    def on_train_begin(self, logs=None):
        # The inspector calls `.numpy()` on gradient tensors and runs Python
        # control flow between train_steps. Both require eager execution.
        # In newer Keras versions, `run_eagerly` is locked in at compile time
        # and cannot be flipped after the fact, so we just *verify* it here
        # and raise a clear error if the user forgot.
        if not getattr(self.model, "run_eagerly", False):
            # One last attempt to flip it — works on some Keras versions.
            try:
                self.model.run_eagerly = True
            except Exception:
                pass
        if not getattr(self.model, "run_eagerly", False):
            raise RuntimeError(
                "GradientInspector requires the model to run eagerly, but "
                "`model.run_eagerly` is False. Compile the model with "
                "`run_eagerly=True` BEFORE calling model.fit:\n"
                "    model.compile(optimizer=..., run_eagerly=True)\n"
                "Without this the inspector cannot call .numpy() on "
                "gradient tensors inside train_step."
            )
        self.model.gradient_logger = self

        if self.write_tensorboard:
            self._tb_writer = tf.summary.create_file_writer(
                os.path.join(self.out_dir, "tb")
            )

        if self.verbose:
            print(
                f"[GradientInspector] writing to {self.out_dir} "
                f"(decompose every {self.decompose_every} steps, "
                f"layer norms every {self.layer_norms_every} steps, "
                f"run_eagerly=True)"
            )

    def on_train_batch_begin(self, batch, logs=None):
        # Flags are sampled at the start of the next train_step.
        self._do_layer_norms_now = (
            self.layer_norms_every > 0
            and (self._step % self.layer_norms_every == 0)
        )
        self._do_decompose_now = (
            self.decompose_every > 0
            and (self._step % self.decompose_every == 0)
        )

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1

    def on_train_end(self, logs=None):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        if self._tb_writer is not None:
            self._tb_writer.flush()
        if self._last_row:
            with open(self._summary_path, "w") as fh:
                json.dump(self._last_row, fh, indent=2)
        # Detach so a subsequent run doesn't accidentally inherit us.
        try:
            self.model.gradient_logger = None
        except Exception:
            pass

    # ---- Probes called from the model's train_step ------------------- #
    def should_log_step(self) -> bool:
        return self._do_layer_norms_now or self._do_decompose_now

    def should_decompose_step(self) -> bool:
        return self._do_decompose_now

    def log(
        self,
        trainable_vars: Sequence[tf.Variable],
        g_focal: Optional[Sequence[Optional[tf.Tensor]]],
        g_recon: Optional[Sequence[Optional[tf.Tensor]]],
        g_kl: Optional[Sequence[Optional[tf.Tensor]]],
        all_grads: Sequence[Optional[tf.Tensor]],
    ) -> None:
        """
        Called from inside the ablation model's train_step. Computes scalar
        diagnostics and writes them to CSV + (optionally) TensorBoard.
        """
        row: Dict[str, float] = {"step": float(self._step)}

        # --- 1. Combined gradient: per-layer norms (cheap) ---------- #
        if self._do_layer_norms_now:
            combined_norms = _per_layer_norms(all_grads, trainable_vars)
            for layer_name, n in combined_norms.items():
                row[f"combined_norm/{layer_name}"] = n
            row["combined_norm/global"] = _global_norm(combined_norms)

        # --- 2. Per-loss decomposition across the trainable model ---- #
        if self._do_decompose_now and g_focal is not None:
            for loss_name, g in zip(self.LOSS_NAMES, (g_focal, g_recon, g_kl)):
                # Full-model per-layer norms
                norms = _per_layer_norms(g, trainable_vars)
                for layer_name, n in norms.items():
                    row[f"all_{loss_name}_norm/{layer_name}"] = n
                row[f"all_{loss_name}_norm/global"] = _global_norm(norms)

                # Coarse branch summaries
                for scope_name in ("encoder", "decoder", "classifier", "other"):
                    scope_grads, scope_vars = _select_scope(g, trainable_vars, scope_name)
                    if not scope_vars:
                        continue
                    scope_norms = _per_layer_norms(scope_grads, scope_vars)
                    row[f"scope_{loss_name}_norm/{scope_name}"] = _global_norm(scope_norms)

                # Preserve encoder-only per-layer view for the original
                # classifier-vs-reconstruction interaction hypothesis.
                enc_grads, enc_vars = _select_scope(g, trainable_vars, "encoder")
                enc_norms = _per_layer_norms(enc_grads, enc_vars)
                for layer_name, n in enc_norms.items():
                    row[f"enc_{loss_name}_norm/{layer_name}"] = n
                row[f"enc_{loss_name}_norm/global"] = _global_norm(enc_norms)

            # --- 3. Pairwise cosine similarity on flattened encoder grad
            focal_enc_grads, encoder_vars = _select_scope(g_focal, trainable_vars, "encoder")
            recon_enc_grads, _ = _select_scope(g_recon, trainable_vars, "encoder")
            kl_enc_grads, _ = _select_scope(g_kl, trainable_vars, "encoder")
            f_focal = _flat(focal_enc_grads, encoder_vars)
            f_recon = _flat(recon_enc_grads, encoder_vars)
            f_kl = _flat(kl_enc_grads, encoder_vars)
            row["enc_cos/focal_recon"] = _cosine(f_focal, f_recon)
            row["enc_cos/focal_kl"] = _cosine(f_focal, f_kl)
            row["enc_cos/recon_kl"] = _cosine(f_recon, f_kl)

            # --- 4. Loss-share on the encoder
            denom = (
                np.linalg.norm(f_focal)
                + np.linalg.norm(f_recon)
                + np.linalg.norm(f_kl)
                + 1e-12
            )
            row["enc_share/focal"] = float(np.linalg.norm(f_focal) / denom)
            row["enc_share/recon"] = float(np.linalg.norm(f_recon) / denom)
            row["enc_share/kl"] = float(np.linalg.norm(f_kl) / denom)

        self._write_row(row)
        self._last_row = row

        # TensorBoard scalars
        if self._tb_writer is not None and len(row) > 1:
            with self._tb_writer.as_default(step=self._step):
                for k, v in row.items():
                    if k == "step":
                        continue
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    tf.summary.scalar(k, v)
            # Flush occasionally so TB stays fresh.
            if self._step % max(self.layer_norms_every * 50, 1) == 0:
                self._tb_writer.flush()

    # ---- CSV writing ------------------------------------------------ #
    def _write_row(self, row: Dict[str, float]) -> None:
        if self._csv_file is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_columns = sorted(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns
            )
            self._csv_writer.writeheader()
        else:
            new_keys = [k for k in row.keys() if k not in self._csv_columns]
            if new_keys:
                # Re-open with extended schema. Keras can give us a wider row
                # later (e.g. on the first decompose step) than on step 0.
                self._csv_file.flush()
                self._csv_file.close()

                old_rows: List[Dict[str, float]] = []
                if os.path.exists(self._csv_path):
                    with open(self._csv_path, "r") as fh:
                        old_rows = list(csv.DictReader(fh))

                self._csv_columns = sorted(set(self._csv_columns) | set(new_keys))
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._csv_columns
                )
                self._csv_writer.writeheader()
                for r in old_rows:
                    self._csv_writer.writerow(
                        {k: r.get(k, "") for k in self._csv_columns}
                    )

        self._csv_writer.writerow(
            {k: row.get(k, "") for k in self._csv_columns}
        )
        # Flush every 100 logged rows so partial logs survive a crash.
        if self._step % 100 == 0:
            self._csv_file.flush()


# --------------------------------------------------------------------- #
# Plotting helper (post-hoc)
# --------------------------------------------------------------------- #

def plot_gradient_log(
    csv_path: str,
    out_dir: str,
    smoothing_window: int = 50,
) -> None:
    """
    Render a small set of comparison plots from a gradients.csv file.

    Saves up to six PNGs (skipping any whose required columns are absent
    or empty — that just means the run didn't reach a decompose step):
        * encoder_loss_share.png         — focal vs recon vs kl share over time
        * encoder_cosines.png            — pairwise cosines on encoder grads
        * combined_layer_norms.png       — global + per-layer combined norms
        * per_loss_layer_norms.png       — encoder layer-wise breakdown
        * per_loss_all_layer_norms.png   — full-model layer-wise breakdown
        * model_component_loss_norms.png — encoder / decoder / classifier norms
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df.sort_values("step").reset_index(drop=True)

    combined_layer_names = sorted(
        c.split("/", 1)[1]
        for c in df.columns
        if c.startswith("combined_norm/")
    )
    if combined_layer_names and set(combined_layer_names) <= {"global", "kernel", "bias"}:
        print(
            "[plot_gradient_log] Warning: this gradients.csv appears to come from "
            "the old grouping bug, so layer-wise plots are collapsed into generic "
            "'kernel' and 'bias' buckets. Re-run training to get true per-layer logs."
        )

    def _smooth(s):
        if smoothing_window > 1 and len(s) > smoothing_window:
            return s.rolling(smoothing_window, min_periods=1).mean()
        return s

    def _safe_legend(ax, **kwargs) -> bool:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return False
        ax.legend(**kwargs)
        return True

    def _has_signal(columns):
        """Return only the columns that actually carry numeric data."""
        good = []
        for c in columns:
            if c not in df.columns:
                continue
            if df[c].dropna().shape[0] > 0:
                good.append(c)
        return good

    # --- Loss share -------------------------------------------------- #
    share_cols = _has_signal(
        [c for c in df.columns if c.startswith("enc_share/")]
    )
    if share_cols:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for c in share_cols:
            sub = df[["step", c]].dropna()
            ax.plot(sub["step"], _smooth(sub[c]), label=c.split("/", 1)[1])
        ax.set_xlabel("training step")
        ax.set_ylabel("fraction of encoder gradient norm")
        ax.set_title("Encoder gradient share by loss term")
        ax.grid(True, alpha=0.3)
        if _safe_legend(ax):
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "encoder_loss_share.png"), dpi=180)
        plt.close(fig)

    # --- Cosines ---------------------------------------------------- #
    cos_cols = _has_signal(
        [c for c in df.columns if c.startswith("enc_cos/")]
    )
    if cos_cols:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for c in cos_cols:
            sub = df[["step", c]].dropna()
            ax.plot(sub["step"], _smooth(sub[c]), label=c.split("/", 1)[1])
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("training step")
        ax.set_ylabel("cosine similarity (encoder grads)")
        ax.set_title("Are loss-pair gradients pulling the encoder together?")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        if _safe_legend(ax):
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "encoder_cosines.png"), dpi=180)
        plt.close(fig)

    # --- Combined per-layer norms ----------------------------------- #
    cn_cols = _has_signal(
        sorted(c for c in df.columns if c.startswith("combined_norm/"))
    )
    if cn_cols:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for c in cn_cols:
            sub = df[["step", c]].dropna()
            ax.plot(sub["step"], _smooth(sub[c]),
                    label=c.split("/", 1)[1], alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("L2 norm")
        ax.set_title("Combined gradient L2 norm per layer")
        ax.grid(True, which="both", alpha=0.25)
        if _safe_legend(ax, fontsize=7, ncol=2):
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "combined_layer_norms.png"), dpi=180)
        plt.close(fig)

    # --- Per-loss × per-encoder-layer ------------------------------- #
    loss_layer_cols = {
        loss: _has_signal(
            sorted(c for c in df.columns if c.startswith(f"enc_{loss}_norm/"))
        )
        for loss in GradientInspector.LOSS_NAMES
    }
    if any(loss_layer_cols.values()):
        fig, axes = plt.subplots(
            len(GradientInspector.LOSS_NAMES), 1,
            figsize=(9, 9), sharex=True,
        )
        any_plotted = False
        for ax, loss in zip(axes, GradientInspector.LOSS_NAMES):
            for c in loss_layer_cols[loss]:
                sub = df[["step", c]].dropna()
                ax.plot(sub["step"], _smooth(sub[c]),
                        label=c.split("/", 1)[1], alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylabel(f"{loss} grad norm")
            ax.grid(True, which="both", alpha=0.25)
            if _safe_legend(ax, fontsize=6, ncol=2):
                any_plotted = True
        axes[-1].set_xlabel("training step")
        fig.suptitle("Per-loss gradient L2 norm on encoder layers")
        if any_plotted:
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "per_loss_layer_norms.png"), dpi=180)
        plt.close(fig)

    # --- Per-loss × all trainable layers ---------------------------- #
    all_loss_layer_cols = {
        loss: _has_signal(
            sorted(c for c in df.columns if c.startswith(f"all_{loss}_norm/"))
        )
        for loss in GradientInspector.LOSS_NAMES
    }
    if any(all_loss_layer_cols.values()):
        fig, axes = plt.subplots(
            len(GradientInspector.LOSS_NAMES), 1,
            figsize=(10, 10), sharex=True,
        )
        any_plotted = False
        for ax, loss in zip(axes, GradientInspector.LOSS_NAMES):
            for c in all_loss_layer_cols[loss]:
                sub = df[["step", c]].dropna()
                ax.plot(sub["step"], _smooth(sub[c]),
                        label=c.split("/", 1)[1], alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylabel(f"{loss} grad norm")
            ax.grid(True, which="both", alpha=0.25)
            if _safe_legend(ax, fontsize=6, ncol=3):
                any_plotted = True
        axes[-1].set_xlabel("training step")
        fig.suptitle("Per-loss gradient L2 norm across all trainable layers")
        if any_plotted:
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "per_loss_all_layer_norms.png"), dpi=180)
        plt.close(fig)

    # --- Per-loss × coarse model components ------------------------- #
    scope_cols = {
        loss: _has_signal(
            sorted(c for c in df.columns if c.startswith(f"scope_{loss}_norm/"))
        )
        for loss in GradientInspector.LOSS_NAMES
    }
    if any(scope_cols.values()):
        fig, axes = plt.subplots(
            len(GradientInspector.LOSS_NAMES), 1,
            figsize=(9, 8), sharex=True,
        )
        any_plotted = False
        for ax, loss in zip(axes, GradientInspector.LOSS_NAMES):
            for c in scope_cols[loss]:
                sub = df[["step", c]].dropna()
                ax.plot(sub["step"], _smooth(sub[c]),
                        label=c.split("/", 1)[1], alpha=0.8)
            ax.set_yscale("log")
            ax.set_ylabel(f"{loss} grad norm")
            ax.grid(True, which="both", alpha=0.25)
            if _safe_legend(ax, fontsize=7, ncol=2):
                any_plotted = True
        axes[-1].set_xlabel("training step")
        fig.suptitle("Per-loss gradient L2 norm by model component")
        if any_plotted:
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "model_component_loss_norms.png"), dpi=180)
        plt.close(fig)
