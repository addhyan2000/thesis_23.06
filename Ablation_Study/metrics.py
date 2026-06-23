"""
metrics.py — Evaluation Metrics & Result Persistence for the Ablation Study
============================================================================

Computes the three metrics required by the thesis brief and writes them to
clearly named, per-configuration files:

    • Accuracy            (overall)
    • Macro-F1            (unweighted mean of per-class F1 — robust to imbalance)
    • Confusion Matrix    (raw counts, rows = true, cols = predicted)
    • Per-class precision / recall / F1 (for completeness)

Design
------
``MetricsComputer`` is a small stateless helper (pure functions) so it is easy
to unit test. ``ResultWriter`` owns all filesystem side effects: it writes a
per-config JSON, a confusion-matrix ``.npy`` (+ optional PNG), and appends a row
to a master CSV summary that aggregates every configuration in one place.

sklearn is used when available (it is already a project dependency); a NumPy
fallback keeps the module importable even if sklearn is missing.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # Preferred path — sklearn is already used elsewhere in the repo.
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix as _sk_confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - defensive fallback
    _HAVE_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    """Bundle of metrics for a single evaluated configuration/fold."""

    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    per_class_f1: List[float] = field(default_factory=list)
    per_class_precision: List[float] = field(default_factory=list)
    per_class_recall: List[float] = field(default_factory=list)
    num_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation (pure functions)
# ─────────────────────────────────────────────────────────────────────────────
class MetricsComputer:
    """Stateless computation of accuracy, macro-F1, and confusion matrix."""

    @staticmethod
    def compute(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        num_classes: int,
    ) -> EvalResult:
        """
        Compute all metrics from integer label sequences.

        Parameters
        ----------
        y_true, y_pred : sequence of int
            Ground-truth and predicted class indices (same length).
        num_classes : int
            Total number of classes — fixes the confusion-matrix dimensions even
            when some class is absent from a particular validation fold.
        """
        y_true = np.asarray(list(y_true), dtype=int)
        y_pred = np.asarray(list(y_pred), dtype=int)
        labels = list(range(num_classes))

        if len(y_true) == 0:
            return EvalResult(0.0, 0.0, [[0] * num_classes for _ in range(num_classes)],
                              [0.0] * num_classes, [0.0] * num_classes,
                              [0.0] * num_classes, 0)

        if _HAVE_SKLEARN:
            acc = float(accuracy_score(y_true, y_pred))
            macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                                      average="macro", zero_division=0))
            cm = _sk_confusion_matrix(y_true, y_pred, labels=labels)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=None, zero_division=0
            )
            per_p, per_r, per_f = prec.tolist(), rec.tolist(), f1.tolist()
        else:  # NumPy fallback
            acc = float((y_true == y_pred).mean())
            cm = MetricsComputer._numpy_confusion(y_true, y_pred, num_classes)
            per_p, per_r, per_f = MetricsComputer._numpy_prf(cm)
            macro_f1 = float(np.mean(per_f)) if per_f else 0.0

        return EvalResult(
            accuracy=acc,
            macro_f1=macro_f1,
            confusion_matrix=np.asarray(cm, dtype=int).tolist(),
            per_class_f1=[float(v) for v in per_f],
            per_class_precision=[float(v) for v in per_p],
            per_class_recall=[float(v) for v in per_r],
            num_samples=int(len(y_true)),
        )

    # ── NumPy fallbacks (only used if sklearn is unavailable) ──
    @staticmethod
    def _numpy_confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    @staticmethod
    def _numpy_prf(cm: np.ndarray):
        prec, rec, f1 = [], [], []
        for c in range(cm.shape[0]):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            prec.append(p); rec.append(r); f1.append(f)
        return prec, rec, f1

    @staticmethod
    def average_results(results: List[EvalResult], num_classes: int) -> EvalResult:
        """
        Aggregate per-fold results (used for LOSO): accuracy/F1 are averaged and
        confusion matrices are summed.
        """
        if not results:
            return EvalResult(0.0, 0.0, [[0] * num_classes for _ in range(num_classes)])
        acc = float(np.mean([r.accuracy for r in results]))
        macro_f1 = float(np.mean([r.macro_f1 for r in results]))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for r in results:
            cm += np.asarray(r.confusion_matrix, dtype=int)
        per_p, per_r, per_f = MetricsComputer._numpy_prf(cm)
        return EvalResult(
            accuracy=acc, macro_f1=macro_f1,
            confusion_matrix=cm.tolist(),
            per_class_f1=[float(v) for v in per_f],
            per_class_precision=[float(v) for v in per_p],
            per_class_recall=[float(v) for v in per_r],
            num_samples=int(sum(r.num_samples for r in results)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
class ResultWriter:
    """
    Owns all filesystem writes for ablation results.

    Layout (under ``output_root``)::

        results/
          summary.csv                         ← one row per configuration
          config_8_proposed_unified/
            metrics.json                       ← full EvalResult + toggle flags
            confusion_matrix.npy               ← raw counts
            confusion_matrix.png               ← heatmap (if matplotlib present)
    """

    def __init__(self, output_root: Path, class_names: List[str]) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.summary_csv = self.output_root / "summary.csv"

    def save_config_result(
        self,
        config_name: str,
        toggles: Dict[str, bool],
        result: EvalResult,
        train_state=None,
        data_flow: str = "",
        extra: Optional[Dict] = None,
    ) -> Path:
        """Write per-config artefacts and append to summary CSV."""
        cfg_dir = self.output_root / config_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Final Results JSON ──
        payload = {
            "config_name": config_name,
            "toggles": toggles,
            "class_names": self.class_names,
            "metrics": result.to_dict(),
        }
        if extra:
            payload["extra"] = extra
        (cfg_dir / "final_results.json").write_text(json.dumps(payload, indent=2))

        # ── 2. Confusion matrix artefacts ──
        cm = np.asarray(result.confusion_matrix, dtype=int)
        np.save(cfg_dir / "confusion_matrix.npy", cm)
        self._maybe_save_cm_png(cm, cfg_dir / "confusion_matrix.png", config_name)

        # ── 3. Append to master summary ──
        self._append_summary(config_name, toggles, result)

        if train_state is not None:
            # ── 4. Training Metrics CSV ──
            if train_state.epoch_metrics:
                import csv
                csv_path = cfg_dir / "training_metrics.csv"
                with csv_path.open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "duration_sec"])
                    writer.writeheader()
                    writer.writerows(train_state.epoch_metrics)

            # ── 5. Checkpoint ──
            if train_state.best_state_dict is not None:
                chk_dir = cfg_dir / "checkpoints"
                chk_dir.mkdir(exist_ok=True)
                import torch
                torch.save(train_state.best_state_dict, chk_dir / "best_model.pth")

            # ── 6. Configuration Summary TXT ──
            summary_lines = [
                f"Configuration: {config_name}",
                "=" * 40,
                "",
                "Toggles:",
            ]
            for k, v in toggles.items():
                summary_lines.append(f"  {k}: {v}")
            summary_lines.append("")
            summary_lines.append("Hardware Metrics:")
            summary_lines.append(f"  Total Train Time: {train_state.total_train_time_sec:.2f} seconds")
            summary_lines.append(f"  Peak VRAM: {train_state.peak_vram_mb:.1f} MB")
            summary_lines.append("")
            summary_lines.append("Data Flow:")
            summary_lines.append(data_flow)
            
            (cfg_dir / "configuration_summary.txt").write_text("\n".join(summary_lines))

        return cfg_dir

    def _append_summary(self, config_name: str, toggles: Dict[str, bool], result: EvalResult) -> None:
        header = ["config_name", "use_evm", "use_simam", "use_cnn", "use_transformer",
                  "accuracy", "macro_f1", "num_samples"]
        row = {
            "config_name": config_name,
            "use_evm": toggles.get("use_evm", ""),
            "use_simam": toggles.get("use_simam", ""),
            "use_cnn": toggles.get("use_cnn", ""),
            "use_transformer": toggles.get("use_transformer", ""),
            "accuracy": f"{result.accuracy:.4f}",
            "macro_f1": f"{result.macro_f1:.4f}",
            "num_samples": result.num_samples,
        }
        existing_rows: List[dict] = []
        if self.summary_csv.exists():
            with self.summary_csv.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for existing in reader:
                    existing_rows.append(existing)

        replaced = False
        for idx, existing in enumerate(existing_rows):
            if existing.get("config_name") == config_name:
                existing_rows[idx] = row
                replaced = True
                break
        if not replaced:
            existing_rows.append(row)

        with self.summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(existing_rows)

    def _maybe_save_cm_png(self, cm: np.ndarray, path: Path, title: str) -> None:
        """Save a confusion-matrix heatmap if matplotlib is importable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return  # silently skip — JSON + .npy already capture the data

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(self.class_names, fontsize=8)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    y_true = [0, 1, 2, 0, 1, 2, 0, 0]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 2]
    res = MetricsComputer.compute(y_true, y_pred, num_classes=3)
    print(f"acc={res.accuracy:.3f} macro_f1={res.macro_f1:.3f}")
    print(f"cm={res.confusion_matrix}")
