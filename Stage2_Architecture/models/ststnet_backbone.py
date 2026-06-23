"""
ststnet_backbone.py — Three-Branch Shallow 3D-CNN Feature Extractor
====================================================================

Architecture Overview:
    This module implements a modality-aware, three-stream 3D-CNN inspired by
    the original STSTNet (Shallow Triple Stream Three-Dimensional CNN), adapted
    for volumetric (spatio-temporal) input.

    Each modality stream (Horizontal Flow u, Vertical Flow v, Optical Strain os)
    is processed by an **independent, unshared** shallow 3D-CNN with the
    following layer sequence:

        Conv3D → BatchNorm3D → ReLU → Conv3D → BatchNorm3D → ReLU → MaxPool3D

    Key Design Decisions:
        1. **Unshared weights**: Each stream has its own learnable filters,
           allowing the network to specialise kernels for each modality.
        2. **Temporal preservation**: All temporal (depth) operations use
           kernel_size=1, stride=1, padding=0 along the temporal axis so
           the 32-frame temporal dimension is preserved for downstream
           Transformer processing.
        3. **Spatial downsampling**: MaxPool3D reduces spatial dimensions by 2×
           (224 → 112) while leaving temporal depth untouched.
        4. **Shallow depth**: Only 2 Conv3D layers per stream to avoid
           overfitting on the small micro-expression datasets.

    After all three streams are processed, they are concatenated along the
    channel dimension and passed through SimAM (externally) before fusion.

Flow (per stream):
    [B, 1, 32, 224, 224]
        → Conv3D(1→16, k=1×3×3, p=0×1×1)  → [B, 16, 32, 224, 224]
        → BN3D → ReLU
        → Conv3D(16→32, k=1×3×3, p=0×1×1) → [B, 32, 32, 224, 224]
        → BN3D → ReLU
        → MaxPool3D(k=1×2×2, s=1×2×2)     → [B, 32, 32, 112, 112]

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .simam import SimAM3D


class _SingleStreamCNN(nn.Module):
    """
    A single-modality shallow 3D-CNN stream.

    Consists of two Conv3D layers with BatchNorm and ReLU, followed by a
    MaxPool3D that downsamples spatial dimensions while preserving temporal
    depth.

    Args:
        in_channels  (int): Number of input channels (1 for a single modality).
        mid_channels (int): Number of output channels from the first Conv3D.
        out_channels (int): Number of output channels from the second Conv3D.
        dropout_p  (float): Dropout probability applied after each BN+ReLU block.

    Shape:
        Input:  [B, in_channels, 32, 224, 224]
        Output: [B, out_channels, 32, 112, 112]
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 16,
        out_channels: int = 32,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Layer 1: Spatial feature extraction ──
        # Kernel (1, 3, 3): no temporal mixing, 3×3 spatial receptive field
        # Padding (0, 1, 1): preserves H, W; no temporal padding
        self.conv1 = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,  # BN absorbs the bias
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)

        # ── Layer 2: Deeper spatial features ──
        self.conv2 = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # ── Activation + Regularisation ──
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p)

        # ── Spatial-only pooling ──
        # Kernel (1, 2, 2): temporal dim untouched, spatial halved
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single modality stream.

        Shape Transformation (step-by-step):
            Input:   [B, 1, 32, 224, 224]

            Conv1:   [B, 1, 32, 224, 224]  → [B, 16, 32, 224, 224]
            BN1:     [B, 16, 32, 224, 224] → [B, 16, 32, 224, 224]
            ReLU:    [B, 16, 32, 224, 224] → [B, 16, 32, 224, 224]
            Drop:    [B, 16, 32, 224, 224] → [B, 16, 32, 224, 224]

            Conv2:   [B, 16, 32, 224, 224] → [B, 32, 32, 224, 224]
            BN2:     [B, 32, 32, 224, 224] → [B, 32, 32, 224, 224]
            ReLU:    [B, 32, 32, 224, 224] → [B, 32, 32, 224, 224]
            Drop:    [B, 32, 32, 224, 224] → [B, 32, 32, 224, 224]

            Pool:    [B, 32, 32, 224, 224] → [B, 32, 32, 112, 112]

            Output:  [B, 32, 32, 112, 112]
        """
        # ── Block 1 ──
        x = self.conv1(x)       # [B, 1, 32, 224, 224] → [B, 16, 32, 224, 224]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ── Block 2 ──
        x = self.conv2(x)       # [B, 16, 32, 224, 224] → [B, 32, 32, 224, 224]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ── Spatial downsampling ──
        x = self.pool(x)        # [B, 32, 32, 224, 224] → [B, 32, 32, 112, 112]

        return x


class STSTNetBackbone3D(nn.Module):
    """
    Three-Branch Shallow 3D-CNN Backbone with SimAM Attention.

    Splits a 3-channel input into three single-channel modality streams,
    processes each through an independent _SingleStreamCNN, applies
    parameter-free SimAM attention to each stream, and concatenates
    the results along the channel dimension.

    Args:
        in_channels_per_stream (int): Channels per modality (default: 1).
        mid_channels           (int): First Conv3D output channels (default: 16).
        out_channels_per_stream(int): Second Conv3D output channels (default: 32).
        dropout_p            (float): Dropout probability (default: 0.3).
        simam_lambda         (float): SimAM regularization constant (default: 1e-4).

    Shape:
        Input:  [B, 3, 32, 224, 224]   — 3-channel spatio-temporal volume
        Output: [B, 96, 32, 112, 112]  — Concatenated (32×3=96 channels)

    Architecture Diagram:
        ┌──────────┐
        │ Input 3ch │ [B, 3, 32, 224, 224]
        └────┬─────┘
             │ split along channel dim
        ┌────┴────┬────────────┐
        ▼         ▼            ▼
      Stream_u  Stream_v   Stream_os
      (1ch)     (1ch)      (1ch)
        │         │            │
        ▼         ▼            ▼
      CNN_u     CNN_v      CNN_os       ← Unshared weights
        │         │            │
        ▼         ▼            ▼
      SimAM_u   SimAM_v    SimAM_os     ← Parameter-free
        │         │            │
        └────┬────┴────────────┘
             │ concat along channel dim
             ▼
        [B, 96, 32, 112, 112]
    """

    def __init__(
        self,
        in_channels_per_stream: int = 1,
        mid_channels: int = 16,
        out_channels_per_stream: int = 32,
        dropout_p: float = 0.3,
        simam_lambda: float = 1e-4,
    ) -> None:
        super().__init__()

        self.out_channels_total = out_channels_per_stream * 3

        # ── Three independent CNN streams (unshared weights) ──
        self.stream_u = _SingleStreamCNN(
            in_channels_per_stream, mid_channels, out_channels_per_stream, dropout_p
        )
        self.stream_v = _SingleStreamCNN(
            in_channels_per_stream, mid_channels, out_channels_per_stream, dropout_p
        )
        self.stream_os = _SingleStreamCNN(
            in_channels_per_stream, mid_channels, out_channels_per_stream, dropout_p
        )

        # ── Three independent SimAM modules (0 parameters each) ──
        self.simam_u = SimAM3D(e_lambda=simam_lambda)
        self.simam_v = SimAM3D(e_lambda=simam_lambda)
        self.simam_os = SimAM3D(e_lambda=simam_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Split → CNN × 3 → SimAM × 3 → Concatenate.

        Shape Transformation (step-by-step):
            Input:      [B, 3, 32, 224, 224]

            Split:
              x_u:      [B, 1, 32, 224, 224]   (channel 0: horizontal flow)
              x_v:      [B, 1, 32, 224, 224]   (channel 1: vertical flow)
              x_os:     [B, 1, 32, 224, 224]   (channel 2: optical strain)

            CNN per stream:
              feat_u:   [B, 32, 32, 112, 112]
              feat_v:   [B, 32, 32, 112, 112]
              feat_os:  [B, 32, 32, 112, 112]

            SimAM per stream (same shape — element-wise attention):
              attn_u:   [B, 32, 32, 112, 112]
              attn_v:   [B, 32, 32, 112, 112]
              attn_os:  [B, 32, 32, 112, 112]

            Concatenate along channel dim:
              Output:   [B, 96, 32, 112, 112]

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, 32, 224, 224].

        Returns:
            torch.Tensor: Fused features of shape [B, 96, 32, 112, 112].
        """
        assert x.dim() == 5 and x.size(1) == 3, (
            f"STSTNetBackbone3D expects [B, 3, D, H, W], got shape {x.shape}"
        )

        # ── Step 1: Modality Split ──
        # Each slice: [B, 1, 32, 224, 224]
        x_u  = x[:, 0:1, :, :, :].contiguous()   # Horizontal optical flow
        x_v  = x[:, 1:2, :, :, :].contiguous()   # Vertical optical flow
        x_os = x[:, 2:3, :, :, :].contiguous()   # Optical strain

        # ── Step 2: Independent 3D-CNN Feature Extraction ──
        feat_u  = self.stream_u(x_u)     # [B, 1, 32, 224, 224] → [B, 32, 32, 112, 112]
        feat_v  = self.stream_v(x_v)     # [B, 1, 32, 224, 224] → [B, 32, 32, 112, 112]
        feat_os = self.stream_os(x_os)   # [B, 1, 32, 224, 224] → [B, 32, 32, 112, 112]

        # ── Step 3: SimAM Attention (parameter-free) ──
        attn_u  = self.simam_u(feat_u)    # [B, 32, 32, 112, 112] → same
        attn_v  = self.simam_v(feat_v)    # [B, 32, 32, 112, 112] → same
        attn_os = self.simam_os(feat_os)  # [B, 32, 32, 112, 112] → same

        # ── Step 4: Channel-wise Concatenation ──
        # [B, 32, 32, 112, 112] × 3 → [B, 96, 32, 112, 112]
        fused = torch.cat([attn_u, attn_v, attn_os], dim=1)

        return fused


# ─────────────────────────────────────────────────────────────────────
# Standalone verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("STSTNetBackbone3D — Three-Branch 3D-CNN Verification")
    print("=" * 65)

    model = STSTNetBackbone3D()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy = torch.randn(2, 3, 32, 224, 224)
    out = model(dummy)
    print(f"\nInput shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(out.shape)}")

    expected = [2, 96, 32, 112, 112]
    assert list(out.shape) == expected, f"Expected {expected}, got {list(out.shape)}"
    print(f"\n✓ STSTNetBackbone3D verification passed. Output: {expected}")
