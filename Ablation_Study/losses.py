"""
losses.py — Decoupled Classification Loss for the Ablation Study
=================================================================

The primary (and ONLY) training objective here is single-task emotion
classification. The Stage 3/4 auxiliary objectives — Supervised Contrastive
Loss (SupCon), the Cross-Batch Memory (XBM) queue, and the adversarial identity
loss — are intentionally removed.

Two interchangeable losses are provided:

    • ``FocalLoss``       — down-weights easy examples; good for class imbalance.
    • ``nn.CrossEntropyLoss`` — standard baseline (via the factory).

``build_loss`` returns the configured criterion given an ``ExperimentConfig``
and optional class weights, so the trainer stays loss-agnostic.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017): ``FL = -α_t (1-p_t)^γ log(p_t)``.

    With ``gamma=0`` and no alpha this reduces exactly to cross-entropy.

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights ``[num_classes]`` (e.g. inverse frequency). None → equal.
    gamma : float
        Focusing parameter (default 2.0). Higher → more focus on hard examples.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    label_smoothing : float
        Smoothing factor in ``[0, 1)``.

    Shape: logits ``[B, C]``, targets ``[B]`` → scalar (or ``[B]`` if "none").
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 2, f"Expected [B, C] logits, got {tuple(logits.shape)}"
        assert targets.dim() == 1, f"Expected [B] targets, got {tuple(targets.shape)}"

        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)                 # [B, C]
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = log_pt.exp()                                        # [B]
        focal_weight = (1.0 - pt) ** self.gamma                  # [B]

        if self.label_smoothing > 0 and num_classes > 1:
            smooth_loss = -log_probs.mean(dim=1)                 # uniform term
            nll_loss = -log_pt
            loss = ((1.0 - self.label_smoothing) * nll_loss
                    + self.label_smoothing * smooth_loss)
        else:
            loss = -log_pt

        loss = focal_weight * loss
        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        a = "None" if self.alpha is None else f"[{', '.join(f'{v:.3f}' for v in self.alpha)}]"
        return f"gamma={self.gamma}, alpha={a}, label_smoothing={self.label_smoothing}"


def build_loss(exp, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Construct the classification criterion from an ``ExperimentConfig``.

    Parameters
    ----------
    exp : ExperimentConfig
        Provides ``loss_type``, ``focal_gamma``, ``label_smoothing``,
        ``use_class_weights``.
    class_weights : torch.Tensor or None
        Optional ``[num_classes]`` weights (applied only if
        ``exp.use_class_weights`` is True).
    """
    alpha = class_weights if (exp.use_class_weights and class_weights is not None) else None

    if exp.loss_type == "focal":
        return FocalLoss(
            alpha=alpha,
            gamma=exp.focal_gamma,
            reduction="mean",
            label_smoothing=exp.label_smoothing,
        )
    if exp.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=alpha,
            label_smoothing=exp.label_smoothing,
        )
    raise ValueError(f"Unknown loss_type: {exp.loss_type!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))

    fl = FocalLoss(gamma=0.0)
    ce = nn.CrossEntropyLoss()
    assert torch.allclose(fl(logits, targets), ce(logits, targets), atol=1e-5)
    print("✓ Focal(gamma=0) == CrossEntropy")

    fl2 = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 2.0, 3.0]), label_smoothing=0.05)
    print(f"✓ Focal(gamma=2, alpha, ls=0.05) = {fl2(logits, targets).item():.4f}")
