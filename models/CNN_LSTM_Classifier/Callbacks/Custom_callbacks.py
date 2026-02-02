import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


class ValidationAnalysisCallback(tf.keras.callbacks.Callback):
    """
    Unified validation callback that:
      - runs ONE fused model.predict() on the full validation dataset
      - computes:
          * event-level recall
          * false positive rate (FPR)
          * max noise margin
          * prediction histograms
    """

    def __init__(
        self,
        validation_data,
        threshold=0.0,
        save_dir="Trained_Models/Validation_Analysis",
        mode_label="Amp",
        histogram_bins=100,
        run_every_n_epochs=1,
    ):
        super().__init__()

        self.validation_data = validation_data
        self.threshold = threshold
        self.save_dir = save_dir
        self.mode_label = mode_label
        self.histogram_bins = histogram_bins
        self.run_every_n_epochs = run_every_n_epochs

        os.makedirs(self.save_dir, exist_ok=True)

        # Cache ground-truth labels once (cheap, CPU-only)
        self._y_true = np.concatenate(
            [y.numpy() for _, y in self.validation_data],
            axis=0
        ).astype(bool)

        # Track best separation (lower is better)
        self.best_max_noise_margin = np.inf

    # ------------------------------------------------------------
    # Helper: event-level recall
    # ------------------------------------------------------------
    @staticmethod
    def _compute_event_recall(y_true, y_pred_bool):
        diffs = np.diff(np.concatenate(([0], y_true.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return 0.0

        detected = 0
        for s, e in zip(starts, ends):
            if np.any(y_pred_bool[s:e]):
                detected += 1

        return detected / len(starts)

    # ------------------------------------------------------------
    # Helper: false positive rate
    # ------------------------------------------------------------
    @staticmethod
    def _compute_fpr(y_true, y_pred_bool):
        noise_mask = ~y_true
        total_noise = np.sum(noise_mask)
        if total_noise == 0:
            return 0.0
        return np.sum(y_pred_bool[noise_mask]) / total_noise

    # ------------------------------------------------------------
    # Helper: max noise margin
    # ------------------------------------------------------------
    @staticmethod
    def _compute_max_noise_margin(y_true, y_pred):
        noise_preds = y_pred[~y_true]
        if len(noise_preds) == 0:
            return 0.0
        return float(np.max(noise_preds))

    # ------------------------------------------------------------
    # Save best model checkpoint
    # ------------------------------------------------------------
    def _save_best_model(self, epoch, metric_value):
        out_path = os.path.join(
            self.save_dir,
            f"best_separation_epoch_{epoch:04d}.keras"
        )
        self.model.save(out_path)
        print(
            f"[INFO] Saved new best-separation model "
            f"(max_noise_margin={metric_value:.6e})"
        )

    # ------------------------------------------------------------
    # Helper: histogram plot
    # ------------------------------------------------------------
    def _plot_histograms(self, y_true, y_pred, epoch):
        plt.figure(figsize=(8, 5))

        plt.hist(
            y_pred[~y_true],
            bins=self.histogram_bins,
            alpha=0.6,
            color="cornflowerblue",
            label="Noise",
            density=True,
            log=True,
        )

        plt.hist(
            y_pred[y_true],
            bins=self.histogram_bins,
            alpha=0.6,
            color="limegreen",
            label="Signal",
            density=True,
            log=True,
        )
        
        """
        plt.axvline(
            self.threshold,
            color="k",
            linestyle="--",
            label="Threshold",
        )
        """

        plt.xlabel("Model output")
        plt.ylabel("Density")
        plt.yscale("log")
        plt.title(f"Validation prediction histogram ({self.mode_label})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            self.save_dir, f"hist_epoch_{epoch:04d}.png"
        )
        plt.savefig(out_path)
        plt.close()

    # ------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.run_every_n_epochs != 0:
            return

        # --------------------------------------------------------
        # ONE fused inference pass
        # --------------------------------------------------------
        y_pred = (
            self.model
            .predict(self.validation_data, verbose=0)
            .flatten()
        )

        y_true = self._y_true
        y_pred_bool = y_pred > self.threshold

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------
        event_recall = self._compute_event_recall(y_true, y_pred_bool)
        fpr = self._compute_fpr(y_true, y_pred_bool)
        max_noise_margin = self._compute_max_noise_margin(y_true, y_pred)

        if np.abs(max_noise_margin) < np.abs(self.best_max_noise_margin):
            self.best_max_noise_margin = max_noise_margin
            self._save_best_model(epoch, max_noise_margin)

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        print("\n--- Validation analysis ---")
        print(f" val_event_recall : {event_recall:.4f}")
        print(f" val_fpr          : {fpr:.6f}")
        print(f" max_noise_margin : {max_noise_margin:.6e}")

        if logs is not None:
            logs["val_event_recall"] = event_recall
            logs["val_fpr"] = fpr
            logs["val_max_noise_margin"] = max_noise_margin

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------
        self._plot_histograms(y_true, y_pred, epoch)
