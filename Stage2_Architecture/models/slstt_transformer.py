"""
slstt_transformer.py — Positional Encoding + Transformer Encoder (SLSTT)
=========================================================================

Architecture Overview:
    SLSTT = Sequence-Level Spatio-Temporal Transformer

    This module receives a temporal sequence of feature vectors
    [B, T, d_model] (where T=32 frames) and models long-range temporal
    dependencies across all frames using a standard Transformer Encoder
    with sinusoidal positional encoding.

Components:
    1. SinusoidalPositionalEncoding:
       Injects absolute position information into the feature sequence
       using fixed sinusoidal functions (Vaswani et al., 2017).
       No learnable parameters — deterministic encoding.

    2. SLSTTTransformer:
       Wraps nn.TransformerEncoder with configurable depth, width,
       and attention heads. Includes the positional encoding step
       and a configurable pooling strategy for the output.

Flow:
    [B, 32, d_model]
        → Positional Encoding     → [B, 32, d_model]
        → Transformer Encoder     → [B, 32, d_model]
        → Temporal Pooling (mean) → [B, d_model]

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Generates a [1, max_len, d_model] buffer of sinusoidal position
    embeddings that is added to the input sequence. The encoding is
    NOT learnable — it is registered as a persistent buffer.

    Mathematical Definition:
        PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model   (int):   Dimensionality of the feature vectors.
        max_len   (int):   Maximum sequence length supported (default: 128).
        dropout_p (float): Dropout applied after adding PE (default: 0.1).

    Shape:
        Input:  [B, T, d_model]  where T ≤ max_len
        Output: [B, T, d_model]  (same shape, with position info added)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 128,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        # ── Compute sinusoidal encoding matrix ──
        pe = torch.zeros(max_len, d_model)                          # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()    # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                            # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd indices

        # Register as buffer (not a parameter — no gradients)
        # Shape: [1, max_len, d_model] for easy broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input sequence.

        Shape Transformation:
            Input:  [B, T, d_model]
            + PE:   [1, T, d_model]  (broadcast over batch)
            Output: [B, T, d_model]

        Args:
            x (torch.Tensor): Sequence tensor [B, T, d_model].

        Returns:
            torch.Tensor: Position-encoded sequence [B, T, d_model].
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]    # Add PE (broadcast over batch)
        return self.dropout(x)


class SLSTTTransformer(nn.Module):
    """
    Sequence-Level Spatio-Temporal Transformer (SLSTT).

    Receives a temporal sequence [B, T, d_model] and applies:
        1. Sinusoidal positional encoding
        2. Multi-layer Transformer Encoder with multi-head self-attention
        3. Temporal pooling to obtain a fixed-size representation [B, d_model]

    The Transformer Encoder uses pre-norm (layer norm before attention)
    for training stability, which is the default in modern implementations.

    Args:
        d_model        (int):   Feature dimension of each time step (default: 96).
        nhead          (int):   Number of attention heads (default: 8).
                                Must divide d_model evenly.
        num_layers     (int):   Number of Transformer encoder layers (default: 4).
        dim_feedforward(int):   Hidden dimension of the FFN (default: 256).
        dropout        (float): Dropout rate for attention and FFN (default: 0.1).
        max_seq_len    (int):   Maximum sequence length for PE (default: 128).
        pool_strategy  (str):   Temporal pooling: "mean" or "cls" (default: "mean").

    Shape:
        Input:  [B, T, d_model]   — T = 32 temporal frames
        Output: [B, d_model]      — Fixed-size sequence representation

    Example:
        >>> transformer = SLSTTTransformer(d_model=96, nhead=8, num_layers=4)
        >>> x = torch.randn(2, 32, 96)
        >>> out = transformer(x)
        >>> assert out.shape == (2, 96)
    """

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pool_strategy: Literal["mean", "cls"] = "mean",
    ) -> None:
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Got d_model % nhead = {d_model % nhead}."
        )

        self.d_model = d_model
        self.pool_strategy = pool_strategy

        # ── Positional Encoding ──
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout_p=dropout,
        )

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",        # GELU for smoother gradients
            batch_first=True,         # Input: [B, T, d_model]
            norm_first=True,          # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),   # Final LayerNorm after all layers
            enable_nested_tensor=False,   # AMP compatibility
        )

        # ── Optional CLS token (only used if pool_strategy == "cls") ──
        if pool_strategy == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Positional Encoding → Transformer → Pooling.

        Shape Transformation (pool_strategy="mean"):
            Input:          [B, 32, d_model]
            + Pos Encoding: [B, 32, d_model]
            Transformer:    [B, 32, d_model]
            Mean Pool:      [B, d_model]

        Shape Transformation (pool_strategy="cls"):
            Input:          [B, 32, d_model]
            + CLS prepend:  [B, 33, d_model]
            + Pos Encoding: [B, 33, d_model]
            Transformer:    [B, 33, d_model]
            CLS extract:    [B, d_model]

        Args:
            x (torch.Tensor): Temporal sequence [B, T, d_model].

        Returns:
            torch.Tensor: Pooled representation [B, d_model].
        """
        assert x.dim() == 3, (
            f"SLSTTTransformer expects [B, T, d_model], got {x.dim()}D "
            f"with shape {x.shape}"
        )

        # ── Optional: Prepend CLS token ──
        if self.pool_strategy == "cls" and self.cls_token is not None:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)                   # [B, T+1, d_model]

        # ── Step 1: Add positional encoding ──
        x = self.pos_encoder(x)          # [B, T, d_model] → [B, T, d_model]

        # ── Step 2: Transformer Encoder ──
        x = self.transformer_encoder(x)  # [B, T, d_model] → [B, T, d_model]

        # ── Step 3: Temporal Pooling ──
        if self.pool_strategy == "cls":
            # Use the CLS token's output (first position)
            pooled = x[:, 0, :]           # [B, T+1, d_model] → [B, d_model]
        else:
            # Mean pool over the temporal dimension
            pooled = x.mean(dim=1)        # [B, T, d_model] → [B, d_model]

        return pooled


# ─────────────────────────────────────────────────────────────────────
# Standalone verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("SLSTTTransformer — Transformer Encoder Verification")
    print("=" * 65)

    for strategy in ["mean", "cls"]:
        print(f"\n── Pool Strategy: {strategy} ──")
        model = SLSTTTransformer(
            d_model=96, nhead=8, num_layers=4,
            dim_feedforward=256, pool_strategy=strategy,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        dummy = torch.randn(2, 32, 96)
        out = model(dummy)
        print(f"Input shape:  {list(dummy.shape)}")
        print(f"Output shape: {list(out.shape)}")

        assert out.shape == (2, 96), f"Expected [2, 96], got {list(out.shape)}"
        print(f"✓ Passed ({strategy})")

    print("\n✓ SLSTTTransformer verification complete.")
