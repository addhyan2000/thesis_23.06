"""
hybrid_model.py — Full Hybrid MER Architecture Orchestrator
=============================================================

Architecture: STSTNet-SimAM-SLSTT Hybrid Model for Micro-Expression Recognition.

This is the top-level nn.Module that orchestrates the complete forward pass:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    HybridMERModel Pipeline                      │
    │                                                                 │
    │   [B, 3, 32, 224, 224]                                          │
    │         │                                                       │
    │         ▼                                                       │
    │   ┌──────────────────────────────────┐                          │
    │   │  STSTNetBackbone3D               │                          │
    │   │  (Split → 3×CNN → 3×SimAM       │                          │
    │   │   → Concat)                      │                          │
    │   └──────────────┬───────────────────┘                          │
    │                  │ [B, 96, 32, 112, 112]                        │
    │                  ▼                                              │
    │   ┌──────────────────────────────────┐                          │
    │   │  AdaptiveAvgPool3d(1, 1, 1)      │ ← Crush spatial dims     │
    │   │  (temporal dim preserved)         │                          │
    │   └──────────────┬───────────────────┘                          │
    │                  │ [B, 96, 32, 1, 1]                            │
    │                  ▼                                              │
    │          Reshape → [B, 32, 96]                                  │
    │                  │                                              │
    │                  ▼                                              │
    │   ┌──────────────────────────────────┐                          │
    │   │  SLSTTTransformer                │                          │
    │   │  (PE → TransformerEncoder        │                          │
    │   │   → Temporal Pool)               │                          │
    │   └──────────────┬───────────────────┘                          │
    │                  │ [B, 96]                                      │
    │                  ▼                                              │
    │   ┌──────────────────────────────────┐                          │
    │   │  Classification Head             │                          │
    │   │  (LayerNorm → Linear → 3)        │                          │
    │   └──────────────┬───────────────────┘                          │
    │                  │ [B, 3]                                       │
    │                  ▼                                              │
    │            Logits (Positive, Negative, Surprise)                 │
    └─────────────────────────────────────────────────────────────────┘

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .ststnet_backbone import STSTNetBackbone3D
from .slstt_transformer import SLSTTTransformer


class HybridMERModel(nn.Module):
    """
    Hybrid Micro-Expression Recognition Model.

    Combines three key innovations:
        1. STSTNet Backbone — Modality-aware 3D-CNN for spatial features
        2. SimAM Attention  — Parameter-free neuron-level attention
        3. SLSTT Transformer — Long-range temporal dependency modelling

    Args:
        num_classes             (int):   Number of output classes (default: 3).
        cnn_mid_channels        (int):   First Conv3D output channels (default: 16).
        cnn_out_channels        (int):   Second Conv3D output channels per stream
                                         (default: 32). Total after concat: 32×3=96.
        cnn_dropout             (float): CNN stream dropout rate (default: 0.3).
        simam_lambda            (float): SimAM regularization constant (default: 1e-4).
        transformer_nhead       (int):   Number of attention heads (default: 8).
        transformer_num_layers  (int):   Transformer encoder depth (default: 4).
        transformer_dim_ff      (int):   FFN hidden dimension (default: 256).
        transformer_dropout     (float): Transformer dropout rate (default: 0.1).
        pool_strategy           (str):   "mean" or "cls" (default: "mean").

    Shape:
        Input:  [B, 3, 32, 224, 224]  — 3-modality spatio-temporal volume
        Output: [B, num_classes]      — Class logits

    Example:
        >>> model = HybridMERModel(num_classes=3)
        >>> x = torch.randn(2, 3, 32, 224, 224)
        >>> logits = model(x)
        >>> assert logits.shape == (2, 3)
    """

    def __init__(
        self,
        num_classes: int = 3,
        cnn_mid_channels: int = 16,
        cnn_out_channels: int = 32,
        cnn_dropout: float = 0.3,
        simam_lambda: float = 1e-4,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 4,
        transformer_dim_ff: int = 256,
        transformer_dropout: float = 0.1,
        pool_strategy: Literal["mean", "cls"] = "mean",
    ) -> None:
        super().__init__()

        # ── Derived dimensions ──
        # After 3-stream concat: cnn_out_channels × 3 = d_model for transformer
        self.d_model = cnn_out_channels * 3   # 32 × 3 = 96

        # ────────────────────────────────────────────────────────────
        # Component 1: STSTNet Backbone (Split → CNN → SimAM → Concat)
        # ────────────────────────────────────────────────────────────
        self.backbone = STSTNetBackbone3D(
            in_channels_per_stream=1,
            mid_channels=cnn_mid_channels,
            out_channels_per_stream=cnn_out_channels,
            dropout_p=cnn_dropout,
            simam_lambda=simam_lambda,
        )

        # ────────────────────────────────────────────────────────────
        # Component 2: Adaptive Spatial Pooling
        # ────────────────────────────────────────────────────────────
        # Crush spatial dims (H, W) → (1, 1) while preserving temporal D=32
        # Output size: (D=32, H=1, W=1) — temporal dim stays at 32
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(32, 1, 1))

        # ────────────────────────────────────────────────────────────
        # Component 3: SLSTT Transformer
        # ────────────────────────────────────────────────────────────
        self.transformer = SLSTTTransformer(
            d_model=self.d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_ff,
            dropout=transformer_dropout,
            max_seq_len=128,
            pool_strategy=pool_strategy,
        )

        # ────────────────────────────────────────────────────────────
        # Component 4: Classification Head
        # ────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, num_classes),
        )

        # ── Weight Initialization ──
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights using best practices for each layer type.

        Strategy:
            - Conv3D: Kaiming He initialization (ReLU-aware)
            - Linear: Xavier uniform
            - BatchNorm: weight=1, bias=0
            - LayerNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: Backbone → Spatial Pool → Reshape → Transformer → Head.

        Shape Transformation (step-by-step):

            1. Input:
               [B, 3, 32, 224, 224]

            2. STSTNetBackbone3D (Split → 3×CNN → 3×SimAM → Concat):
               [B, 3, 32, 224, 224] → [B, 96, 32, 112, 112]

            3. AdaptiveAvgPool3d (crush spatial, preserve temporal):
               [B, 96, 32, 112, 112] → [B, 96, 32, 1, 1]

            4. Reshape (flatten spatial, swap to sequence format):
               [B, 96, 32, 1, 1] → squeeze → [B, 96, 32]
                                 → permute → [B, 32, 96]
               Now we have 32 time steps, each a 96-dim feature vector.

            5. SLSTTTransformer (PE → Encoder → Pool):
               [B, 32, 96] → [B, 96]

            6. Classification Head (LayerNorm → Linear):
               [B, 96] → [B, 3]

        Args:
            x (torch.Tensor): Input tensor [B, 3, 32, 224, 224].

        Returns:
            torch.Tensor: Class logits [B, num_classes].
        """
        # ── Validate input ──
        assert x.dim() == 5, (
            f"HybridMERModel expects 5D input [B, 3, T, H, W], "
            f"got {x.dim()}D with shape {x.shape}"
        )

        # ── Step 1: Backbone — Split → CNN → SimAM → Concat ──
        # [B, 3, 32, 224, 224] → [B, 96, 32, 112, 112]
        features = self.backbone(x)

        # ── Step 2: Adaptive Spatial Pooling ──
        # [B, 96, 32, 112, 112] → [B, 96, 32, 1, 1]
        features = self.spatial_pool(features)

        # ── Step 3: Reshape to sequence format ──
        # Squeeze the spatial dims: [B, 96, 32, 1, 1] → [B, 96, 32]
        features = features.squeeze(-1).squeeze(-1)

        # Permute to [B, T, d_model]: [B, 96, 32] → [B, 32, 96]
        # Now each of the 32 time steps has a 96-dimensional feature vector
        features = features.permute(0, 2, 1).contiguous()

        # ── Step 4: Transformer Encoder ──
        # [B, 32, 96] → [B, 96]
        temporal_repr = self.transformer(features)

        # ── Step 5: Classification Head ──
        # [B, 96] → [B, num_classes]
        logits = self.classifier(temporal_repr)

        return logits

    def count_parameters(self) -> dict:
        """
        Count parameters by component for thesis documentation.

        Returns:
            dict: Parameter counts per component and total.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        def _count_trainable(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        breakdown = {
            "backbone_total": _count(self.backbone),
            "backbone_trainable": _count_trainable(self.backbone),
            "transformer_total": _count(self.transformer),
            "transformer_trainable": _count_trainable(self.transformer),
            "classifier_total": _count(self.classifier),
            "classifier_trainable": _count_trainable(self.classifier),
            "model_total": _count(self),
            "model_trainable": _count_trainable(self),
        }
        return breakdown


# ─────────────────────────────────────────────────────────────────────
# Main Block — End-to-End Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  HybridMERModel — End-to-End Architecture Verification")
    print("  STSTNet-3D  →  SimAM-3D  →  SLSTT Transformer  →  Classifier")
    print("=" * 70)

    # ── Configuration ──
    BATCH_SIZE = 2
    NUM_CLASSES = 3
    INPUT_SHAPE = (BATCH_SIZE, 3, 32, 224, 224)

    print(f"\nInput shape:    {list(INPUT_SHAPE)}")
    print(f"Expected output: [{BATCH_SIZE}, {NUM_CLASSES}]")

    # ── Instantiate Model ──
    model = HybridMERModel(
        num_classes=NUM_CLASSES,
        cnn_mid_channels=16,
        cnn_out_channels=32,
        cnn_dropout=0.3,
        simam_lambda=1e-4,
        transformer_nhead=8,
        transformer_num_layers=4,
        transformer_dim_ff=256,
        transformer_dropout=0.1,
        pool_strategy="mean",
    )

    # ── Parameter Count ──
    param_breakdown = model.count_parameters()
    print(f"\n{'─' * 50}")
    print(f"  Parameter Breakdown")
    print(f"{'─' * 50}")
    print(f"  Backbone (3×CNN + 0 SimAM): {param_breakdown['backbone_trainable']:>10,}")
    print(f"  Transformer (SLSTT):        {param_breakdown['transformer_trainable']:>10,}")
    print(f"  Classifier Head:            {param_breakdown['classifier_trainable']:>10,}")
    print(f"{'─' * 50}")
    print(f"  TOTAL Trainable:            {param_breakdown['model_trainable']:>10,}")
    print(f"{'─' * 50}")

    # ── Forward Pass ──
    print(f"\nRunning forward pass with dummy tensor...")
    dummy_input = torch.randn(*INPUT_SHAPE)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape:  {list(dummy_input.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output values (example): {output[0].tolist()}")

    # ── Assertion ──
    expected_shape = [BATCH_SIZE, NUM_CLASSES]
    assert list(output.shape) == expected_shape, (
        f"SHAPE MISMATCH: Expected {expected_shape}, got {list(output.shape)}"
    )

    print(f"\n{'=' * 70}")
    print(f"  ✓ VERIFICATION PASSED")
    print(f"    Input:  {list(INPUT_SHAPE)}")
    print(f"    Output: {list(output.shape)}")
    print(f"{'=' * 70}")

    sys.exit(0)
