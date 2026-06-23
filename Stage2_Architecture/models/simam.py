"""
simam.py — 3D Parameter-Free Simple Attention Module (SimAM)
=============================================================

Reference Paper:
    Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
    for Convolutional Neural Networks", ICML 2021.

Mathematical Formulation:
    For a neuron activation t in feature map X ∈ ℝ^{C × D × H × W}:

        e_t(w_t, b_t, y_i) = (t - μ)² / [ 4·( σ² + λ ) ] + 0.5

    where:
        μ  = spatial mean of X over (D, H, W)
        σ² = spatial variance = (1/n) · Σ (x_i - μ)²
        n  = D × H × W - 1  (denominator uses Bessel-like correction)
        λ  = small constant (1e-4) for numerical stability

    The energy e_t is passed through a sigmoid gate to produce the
    attention weight, which element-wise rescales the input tensor.

Key Properties:
    1. ZERO learnable parameters — does not increase model complexity.
    2. Extends naturally from 2D (H, W) to 3D (D, H, W) by adjusting
       the spatial dimensions over which statistics are computed.
    3. Acts as a neuron-level attention: each activation in the feature
       map receives its own importance weight.

Adaptation for 3D:
    The original SimAM operates on 4D tensors [B, C, H, W]. We extend
    it to 5D tensors [B, C, D, H, W] by computing statistics over the
    spatio-temporal dimensions (D, H, W) instead of just (H, W).

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAM3D(nn.Module):
    """
    3D Parameter-Free SimAM Attention Module.

    Computes neuron-level importance weights based on an energy function
    derived from neuroscience principles (inter-neuron suppression).

    This module contains **zero learnable parameters**.

    Args:
        e_lambda (float): Regularization constant for numerical stability
                          in the energy denominator. Default: 1e-4.

    Shape:
        Input:  [B, C, D, H, W]  — 5D spatio-temporal feature map
        Output: [B, C, D, H, W]  — Same shape, attention-weighted

    Example:
        >>> simam = SimAM3D(e_lambda=1e-4)
        >>> x = torch.randn(2, 16, 32, 56, 56)
        >>> out = simam(x)
        >>> assert out.shape == (2, 16, 32, 56, 56)
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute SimAM energy and apply sigmoid gating.

        Shape Transformation:
            Input:  [B, C, D, H, W]
            Output: [B, C, D, H, W]   (same shape — element-wise rescaling)

        Mathematical Steps:
            1. Compute spatial mean μ over dims (2, 3, 4) = (D, H, W)
               μ shape: [B, C, 1, 1, 1]
            2. Compute (x - μ)² for each neuron
            3. Compute energy:
               e = (x - μ)² / { 4 · [ Σ(x - μ)² / n + λ ] } + 0.5
               where n = D × H × W - 1
            4. Apply sigmoid(e) as attention weight
            5. Return x * sigmoid(e)
        """
        # ── Validate input dimensionality ──
        assert x.dim() == 5, (
            f"SimAM3D expects 5D input [B, C, D, H, W], got {x.dim()}D "
            f"with shape {x.shape}"
        )

        b, c, d, h, w = x.size()

        # n = total spatial elements per channel minus 1 (Bessel-like)
        n = d * h * w - 1

        # ── Step 1: Squared deviation from spatial mean ──
        # Mean computed over spatio-temporal dims (D=2, H=3, W=4)
        # x_minus_mu_sq shape: [B, C, D, H, W]
        x_minus_mu_sq = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2).contiguous()

        # ── Step 2: Compute energy score ──
        # Numerator:   (x - μ)²                          → [B, C, D, H, W]
        # Denominator: 4 · (variance_estimate + λ)       → [B, C, 1, 1, 1]
        # Energy:      neuron-level scalar                → [B, C, D, H, W]
        variance_estimate = x_minus_mu_sq.sum(dim=[2, 3, 4], keepdim=True) / n
        denom = torch.clamp(4 * (variance_estimate + self.e_lambda), min=1e-7)
        energy = x_minus_mu_sq / denom + 0.5

        # ── Step 3: Sigmoid-gated attention ──
        # Neurons with higher energy (further from mean) get higher weight
        attention = self.sigmoid(energy)

        # ── Step 4: Element-wise rescaling ──
        return x * attention


# ─────────────────────────────────────────────────────────────────────
# Standalone verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("SimAM3D — Parameter-Free 3D Attention Module Verification")
    print("=" * 65)

    module = SimAM3D(e_lambda=1e-4)
    print(f"\nModule: {module}")
    print(f"Learnable parameters: {sum(p.numel() for p in module.parameters())}")

    dummy = torch.randn(2, 16, 32, 56, 56)
    out = module(dummy)
    print(f"\nInput shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(out.shape)}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("\n✓ SimAM3D verification passed.")
