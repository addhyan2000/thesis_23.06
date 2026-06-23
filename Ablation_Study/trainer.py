"""
trainer.py — Clean Single-Task Trainer for the Ablation Study
==============================================================

A minimal, debuggable training loop for emotion classification. Compared with
the Stage 3 ``AdversarialTrainer`` this class removes:

    • the Gradient Reversal Layer schedule,
    • the identity (subject) loss,
    • the SupCon projection loss and the XBM memory queue,
    • gradient accumulation bookkeeping for the triple-objective.

What remains is a single forward → loss → backward → step loop with optional
AMP and gradient clipping, plus validation that returns predictions/labels for
the metrics module to score.

The trainer is deliberately loss-agnostic and model-agnostic: it receives an
already-constructed model, criterion, optimizer, and loaders.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainState:
    """Lightweight record of the best validation epoch and training statistics."""
    best_val_acc: float = 0.0
    best_val_f1: float = 0.0
    best_epoch: int = -1
    best_state_dict: Optional[dict] = None
    epoch_metrics: List[dict] = field(default_factory=list)
    total_train_time_sec: float = 0.0
    peak_vram_mb: float = 0.0


class AblationTrainer:
    """
    Single-task classification trainer.

    Parameters
    ----------
    model : nn.Module
        The ``AblationMERModel`` (or any model returning ``[B, num_classes]``).
    criterion : nn.Module
        Classification loss (Focal or CrossEntropy).
    optimizer : torch.optim.Optimizer
        Optimiser (AdamW recommended).
    device : torch.device
        Compute device.
    num_epochs : int
        Training epochs.
    scheduler : optional
        LR scheduler stepped once per epoch.
    gradient_clip_norm : float or None
        Global grad-norm clip value; None disables clipping.
    use_amp : bool
        Enable mixed-precision autocast on CUDA.
    logger : logging.Logger or None
        Logger; module logger used if None.
    select_metric : str
        Which validation metric selects the best checkpoint: "acc" or "f1".
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        scheduler=None,
        gradient_clip_norm: Optional[float] = 1.0,
        use_amp: bool = False,
        logger: Optional[logging.Logger] = None,
        select_metric: str = "f1",
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = use_amp and device.type == "cuda"
        self._log = logger or logging.getLogger("AblationTrainer")
        self.select_metric = select_metric

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:  # backward compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.state = TrainState()

    # ────────────────────────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainState:
        """Run the full training loop, tracking the best validation checkpoint."""
        import time
        start_time = time.time()
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_one_epoch(train_loader)

            val_loss, y_true, y_pred = self.evaluate(val_loader)
            val_acc = _accuracy(y_true, y_pred)
            val_f1 = _macro_f1(y_true, y_pred, self._num_classes())

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            self.state.epoch_metrics.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "duration_sec": epoch_time
            })

            self._log.info(
                "epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_macroF1=%.4f | time=%.1fs",
                epoch, self.num_epochs, train_loss, val_loss, val_acc, val_f1, epoch_time
            )

            score = val_f1 if self.select_metric == "f1" else val_acc
            best = (self.state.best_val_f1 if self.select_metric == "f1"
                    else self.state.best_val_acc)
            if score >= best:
                self.state.best_val_acc = val_acc
                self.state.best_val_f1 = val_f1
                self.state.best_epoch = epoch
                self.state.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        self.state.total_train_time_sec = time.time() - start_time
        if self.device.type == "cuda":
            self.state.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        self._log.info(
            "Best epoch %d -> val_acc=%.4f | val_macroF1=%.4f | Peak VRAM: %.1f MB",
            self.state.best_epoch, self.state.best_val_acc, self.state.best_val_f1, self.state.peak_vram_mb
        )
        return self.state

    # ────────────────────────────────────────────────────────────────────────
    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running, n = 0.0, 0
        for tensors, labels in loader:
            tensors = tensors.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(tensors)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                
                # Check for NaN gradients before stepping
                has_nan = False
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan = True
                        break
                        
                if has_nan:
                    self._log.warning("NaN gradient detected! Skipping optimizer step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    continue
                    
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * labels.size(0)
            n += labels.size(0)
        return running / max(n, 1)

    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, List[int], List[int]]:
        """Run validation; return ``(mean_loss, y_true, y_pred)``."""
        self.model.eval()
        running, n = 0.0, 0
        y_true: List[int] = []
        y_pred: List[int] = []
        for tensors, labels in loader:
            tensors = tensors.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(tensors)
                loss = self.criterion(logits, labels)
            running += loss.item() * labels.size(0)
            n += labels.size(0)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
        return running / max(n, 1), y_true, y_pred

    # ────────────────────────────────────────────────────────────────────────
    def load_best(self) -> None:
        """Restore the best checkpoint weights into the model (in place)."""
        if self.state.best_state_dict is not None:
            self.model.load_state_dict(self.state.best_state_dict)

    def _num_classes(self) -> int:
        # The classifier's final Linear out_features == num_classes.
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Could not infer num_classes from model.")

    def _autocast(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny metric helpers (kept local so the trainer can log without importing the
# full metrics module; the orchestrator uses metrics.MetricsComputer for the
# authoritative scoring + persistence).
# ─────────────────────────────────────────────────────────────────────────────
def _accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def _macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    if not y_true:
        return 0.0
    f1s = []
    for c in range(num_classes):
        tp = sum(int(t == c and p == c) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != c and p == c) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == c and p != c) for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)
