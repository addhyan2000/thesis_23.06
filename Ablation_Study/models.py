"""
models.py — Modular, Toggleable MER Architecture for the Ablation Study
========================================================================

This file rebuilds the Stage 2 network as ONE configurable ``nn.Module`` whose
forward pass conditionally executes or skips each component based on boolean
flags supplied at construction time. The Stage 3/4 machinery (Gradient Reversal
Layer, identity head, SupCon projection head, XBM queue) is intentionally NOT
present here — this is a clean classification-only model.

The network is decomposed into four debuggable, independently testable pieces:

    1. SimAM3D                 — parameter-free 3D attention (unchanged math).
    2. SingleStream3DCNN       — one shallow 3D-CNN branch (Conv3D k=(1,3,3)).
    3. ThreeStreamCNNBackbone  — three unshared branches + optional SimAM.
    4. RawPatchEmbedding       — the "no-CNN" fallback spatial stem.
    5. SLSTTTransformer        — temporal Transformer encoder (unchanged).
    6. TemporalPooling         — the "no-Transformer" fallback temporal stem.

``AblationMERModel`` wires these together and routes the tensor through the
correct path depending on (use_simam, use_cnn, use_transformer).

────────────────────────────────────────────────────────────────────────────
HOW THE TENSOR FLOWS WHEN A BLOCK IS TOGGLED OFF
────────────────────────────────────────────────────────────────────────────
Input is always ``[B, 3, 32, 224, 224]`` (3 motion channels, 32 frames, 224²).

Stage A — Spatial stem → produces a temporal sequence ``[B, 32, d_model]``:
    • use_cnn = True :
          ThreeStreamCNNBackbone → [B, 96, 32, 112, 112]
          AdaptiveAvgPool3d(32,1,1) → [B, 96, 32, 1, 1]
          squeeze + permute        → [B, 32, 96]
      (SimAM is applied *inside* the backbone, per stream, only if use_simam.)
    • use_cnn = False (raw flattened patches):
          AdaptiveAvgPool3d(32,P,P) on the raw input → [B, 3, 32, P, P]
          reshape per frame                          → [B, 32, 3*P*P]
          Linear projection                          → [B, 32, d_model]
      (SimAM is skipped entirely — there is no CNN feature map to attend over.)

Stage B — Temporal encoder → collapses the 32 steps to ``[B, d_model]``:
    • use_transformer = True : SLSTT (PE → encoder → pool) → [B, d_model]
    • use_transformer = False: mean/max pool over the time axis → [B, d_model]

Stage C — Classifier head: LayerNorm → Linear → ``[B, num_classes]`` (always on).

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

import math
from typing import Dict, Literal

import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# 1.  SimAM3D — Parameter-free 3D attention (math preserved from Stage 2).
# ═════════════════════════════════════════════════════════════════════════════
class SimAM3D(nn.Module):
    """
    3D Parameter-Free SimAM attention (Yang et al., ICML 2021), adapted to 5D.

    Computes a neuron-level energy over the spatio-temporal dims (D, H, W) and
    gates the input with ``sigmoid(energy)``. Contains ZERO learnable params.

    Shape: ``[B, C, D, H, W] → [B, C, D, H, W]`` (element-wise rescaling).
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"SimAM3D expects 5D input, got {x.dim()}D {tuple(x.shape)}"
        _, _, d, h, w = x.size()
        n = d * h * w - 1                                            # Bessel-like
        x_minus_mu_sq = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        variance_estimate = x_minus_mu_sq.sum(dim=[2, 3, 4], keepdim=True) / n
        energy = x_minus_mu_sq / (4 * (variance_estimate + self.e_lambda)) + 0.5
        return x * self.sigmoid(energy)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SingleStream3DCNN — one shallow modality branch.
# ═════════════════════════════════════════════════════════════════════════════
class SingleStream3DCNN(nn.Module):
    """
    A single-modality shallow 3D-CNN branch.

    Two ``Conv3d(k=(1,3,3))`` layers (spatial-only kernels that preserve the
    32-frame temporal axis) with BatchNorm/ReLU/Dropout, then a spatial-only
    ``MaxPool3d(k=(1,2,2))`` that halves H and W.

    Shape: ``[B, 1, 32, 224, 224] → [B, out_channels, 32, 112, 112]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 16,
        out_channels: int = 32,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        # Conv kernel (1,3,3): no temporal mixing; 3x3 spatial receptive field.
        # Padding (0,1,1): keeps H, W; no temporal padding (T stays at 32).
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3),
                               stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 3, 3),
                               stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))   # → [B,16,32,224,224]
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))   # → [B,32,32,224,224]
        x = self.pool(x)                                       # → [B,32,32,112,112]
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ThreeStreamCNNBackbone — three unshared branches + optional SimAM.
# ═════════════════════════════════════════════════════════════════════════════
class ThreeStreamCNNBackbone(nn.Module):
    """
    Three unshared 3D-CNN streams (u-flow, v-flow, optical strain) with an
    OPTIONAL SimAM attention applied per stream before concatenation.

    The ``use_simam`` flag is the only difference between Variable B on/off.
    When False, the SimAM modules are simply not created and not called — the
    raw CNN features are concatenated directly.

    Shape: ``[B, 3, 32, 224, 224] → [B, 3*out_channels, 32, 112, 112]``.
    """

    def __init__(
        self,
        use_simam: bool,
        mid_channels: int = 16,
        out_channels_per_stream: int = 32,
        dropout_p: float = 0.3,
        simam_lambda: float = 1e-4,
    ) -> None:
        super().__init__()
        self.use_simam = use_simam
        self.out_channels_total = out_channels_per_stream * 3

        # ── Three independent (unshared) CNN streams ──
        self.stream_u = SingleStream3DCNN(1, mid_channels, out_channels_per_stream, dropout_p)
        self.stream_v = SingleStream3DCNN(1, mid_channels, out_channels_per_stream, dropout_p)
        self.stream_os = SingleStream3DCNN(1, mid_channels, out_channels_per_stream, dropout_p)

        # ── Optional SimAM per stream (Variable B) ──
        if self.use_simam:
            self.simam_u = SimAM3D(e_lambda=simam_lambda)
            self.simam_v = SimAM3D(e_lambda=simam_lambda)
            self.simam_os = SimAM3D(e_lambda=simam_lambda)
        else:
            self.simam_u = self.simam_v = self.simam_os = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5 and x.size(1) == 3, (
            f"ThreeStreamCNNBackbone expects [B, 3, D, H, W], got {tuple(x.shape)}"
        )
        # ── Split channels into the three physical motion modalities ──
        x_u = x[:, 0:1]    # horizontal flow  [B,1,32,224,224]
        x_v = x[:, 1:2]    # vertical flow
        x_os = x[:, 2:3]   # optical strain

        # ── Independent feature extraction (unshared weights) ──
        feat_u = self.stream_u(x_u)     # [B,32,32,112,112]
        feat_v = self.stream_v(x_v)
        feat_os = self.stream_os(x_os)

        # ── Optional parameter-free attention ──
        if self.use_simam:
            feat_u = self.simam_u(feat_u)
            feat_v = self.simam_v(feat_v)
            feat_os = self.simam_os(feat_os)

        # ── Concatenate along the channel dim → [B, 96, 32, 112, 112] ──
        return torch.cat([feat_u, feat_v, feat_os], dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  RawPatchEmbedding — the "no-CNN" fallback spatial stem (Variable C off).
# ═════════════════════════════════════════════════════════════════════════════
class RawPatchEmbedding(nn.Module):
    """
    Spatial stem used when the 3D-CNN backbone is disabled.

    Instead of learning spatial filters, each frame's 3 motion channels are
    average-pooled to a tiny ``grid x grid`` map, flattened into a per-frame
    vector, and linearly projected to ``d_model``. This yields a temporal
    sequence ``[B, T, d_model]`` directly comparable to the CNN path, so the
    downstream temporal encoder is unchanged.

    Shape:
        ``[B, 3, T, 224, 224]``
          → AdaptiveAvgPool3d((T, grid, grid)) → ``[B, 3, T, grid, grid]``
          → reshape per frame                  → ``[B, T, 3*grid*grid]``
          → Linear                             → ``[B, T, d_model]``
    """

    def __init__(self, in_channels: int, sequence_length: int, grid: int, d_model: int) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.grid = grid
        self.flat_dim = in_channels * grid * grid
        # Keep temporal dim = T; crush spatial dims to grid x grid.
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(sequence_length, grid, grid))
        self.proj = nn.Linear(self.flat_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, _, _ = x.shape
        x = self.spatial_pool(x)                  # [B, C, T, grid, grid]
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, T, C, grid, grid]
        x = x.view(b, t, self.flat_dim)           # [B, T, C*grid*grid]
        x = self.proj(x)                          # [B, T, d_model]
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SLSTT Transformer — temporal encoder (math preserved from Stage 2).
# ═════════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017). No params."""

    def __init__(self, d_model: int, max_len: int = 128, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SLSTTTransformer(nn.Module):
    """
    Sequence-Level Spatio-Temporal Transformer encoder.

    ``[B, T, d_model]`` → positional encoding → pre-norm Transformer encoder →
    temporal pooling → ``[B, d_model]``.
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
            f"d_model ({d_model}) must be divisible by nhead ({nhead})."
        )
        self.d_model = d_model
        self.pool_strategy = pool_strategy

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        try:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )
        if pool_strategy == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"SLSTT expects [B, T, d_model], got {tuple(x.shape)}"
        if self.pool_strategy == "cls" and self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if self.pool_strategy == "cls":
            return x[:, 0, :]
        return x.mean(dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TemporalPooling — the "no-Transformer" fallback temporal stem (D off).
# ═════════════════════════════════════════════════════════════════════════════
class TemporalPooling(nn.Module):
    """
    Collapses the temporal axis with a simple statistic when the Transformer is
    disabled. ``[B, T, d_model] → [B, d_model]`` via mean or max over T.

    This is the deliberately weak temporal baseline: it has no parameters and
    no notion of frame order, isolating the contribution of the Transformer.
    """

    def __init__(self, mode: Literal["mean", "max"] = "mean") -> None:
        super().__init__()
        assert mode in ("mean", "max")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "max":
            return x.max(dim=1).values
        return x.mean(dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  AblationMERModel — the configurable orchestrator.
# ═════════════════════════════════════════════════════════════════════════════
class AblationMERModel(nn.Module):
    """
    Single configurable MER model whose graph is decided by three model-level
    booleans. (The EVM toggle is data-level and handled by the dataset, not
    here.)

    Parameters
    ----------
    num_classes : int
        Output emotion classes (config-driven; no hardcoded value).
    use_simam : bool
        Variable B — apply SimAM inside the CNN backbone. Ignored (with a
        warning logged by the caller) if ``use_cnn`` is False.
    use_cnn : bool
        Variable C — use the 3D-CNN backbone; otherwise raw patch embedding.
    use_transformer : bool
        Variable D — use the SLSTT Transformer; otherwise temporal pooling.
    in_channels, sequence_length : int
        Input tensor geometry (3 and 32 by default). Kept configurable.
    d_model : int
        Temporal feature width. With the CNN on, this MUST equal
        ``cnn_out_channels * 3`` so the concat width matches.
    ... (remaining args mirror ExperimentConfig fields) ...

    Forward
    -------
    ``[B, in_channels, T, H, W] → [B, num_classes]``
    """

    def __init__(
        self,
        num_classes: int,
        use_simam: bool,
        use_cnn: bool,
        use_transformer: bool,
        in_channels: int = 3,
        sequence_length: int = 32,
        d_model: int = 96,
        cnn_mid_channels: int = 16,
        cnn_out_channels: int = 32,
        cnn_dropout: float = 0.3,
        simam_lambda: float = 1e-4,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 4,
        transformer_dim_ff: int = 256,
        transformer_dropout: float = 0.1,
        pool_strategy: Literal["mean", "cls"] = "mean",
        raw_patch_grid: int = 4,
        temporal_pool: Literal["mean", "max"] = "mean",
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.use_simam = use_simam and use_cnn   # SimAM requires a CNN feature map
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.sequence_length = sequence_length

        # ── Stage A: spatial stem → sequence [B, T, d_model] ──────────────────
        if use_cnn:
            # The CNN concat width fixes d_model; enforce consistency.
            cnn_width = cnn_out_channels * 3
            assert cnn_width == d_model, (
                f"With use_cnn=True, d_model must equal cnn_out_channels*3 "
                f"({cnn_width}); got d_model={d_model}."
            )
            self.backbone = ThreeStreamCNNBackbone(
                use_simam=self.use_simam,
                mid_channels=cnn_mid_channels,
                out_channels_per_stream=cnn_out_channels,
                dropout_p=cnn_dropout,
                simam_lambda=simam_lambda,
            )
            # Crush spatial dims, preserve T → [B, d_model, T, 1, 1].
            self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(sequence_length, 1, 1))
            self.raw_embed = None
        else:
            # No CNN: raw flattened patches → linear projection to d_model.
            self.backbone = None
            self.spatial_pool = None
            self.raw_embed = RawPatchEmbedding(
                in_channels=in_channels,
                sequence_length=sequence_length,
                grid=raw_patch_grid,
                d_model=d_model,
            )

        # ── Stage B: temporal encoder → [B, d_model] ──────────────────────────
        if use_transformer:
            self.temporal = SLSTTTransformer(
                d_model=d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_ff,
                dropout=transformer_dropout,
                max_seq_len=max(128, sequence_length + 1),
                pool_strategy=pool_strategy,
            )
        else:
            self.temporal = TemporalPooling(mode=temporal_pool)

        # ── Stage C: classifier head (always present) ─────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(d_model, num_classes),
        )

        self.d_model = d_model
        self._init_weights()

    # ────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Kaiming for Conv3d, Xavier for Linear, ones/zeros for norm layers."""
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

    # ────────────────────────────────────────────────────────────────────────
    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage A router: turn the raw 5D volume into a ``[B, T, d_model]``
        temporal sequence, using either the CNN backbone or raw patch stem.
        """
        if self.use_cnn:
            feats = self.backbone(x)              # [B, 96, T, 112, 112]
            feats = self.spatial_pool(feats)      # [B, 96, T, 1, 1]
            feats = feats.squeeze(-1).squeeze(-1) # [B, 96, T]
            return feats.permute(0, 2, 1)         # [B, T, 96]
        # No-CNN branch: raw flattened patches already yield [B, T, d_model].
        return self.raw_embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, (
            f"AblationMERModel expects 5D input [B, C, T, H, W], got {tuple(x.shape)}"
        )
        # Stage A: spatial stem → temporal sequence [B, T, d_model]
        seq = self._to_sequence(x)
        # Stage B: temporal encoder/pool → [B, d_model]
        temporal_repr = self.temporal(seq)
        # Stage C: classifier → [B, num_classes]
        return self.classifier(temporal_repr)

    # ────────────────────────────────────────────────────────────────────────
    def active_components(self) -> Dict[str, bool]:
        """Return the set of components active in this instance (for logging)."""
        return {
            "cnn_backbone": self.use_cnn,
            "simam_attention": self.use_simam,
            "transformer": self.use_transformer,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Total / trainable parameter counts for thesis documentation."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def describe_data_flow(self) -> str:
        """Returns a multi-line string describing the tensor transformations."""
        lines = []
        lines.append("Stage A - Spatial Stem:")
        if self.use_cnn:
            lines.append("  [B, 3, 32, 224, 224] -> ThreeStreamCNNBackbone -> [B, 96, 32, 112, 112]")
            if self.use_simam:
                lines.append("  (SimAM attention applied internally to feature maps)")
            lines.append("  [B, 96, 32, 112, 112] -> AdaptiveAvgPool3d(32, 1, 1) -> [B, 96, 32, 1, 1]")
            lines.append("  [B, 96, 32, 1, 1] -> squeeze & permute -> [B, 32, 96]")
        else:
            lines.append(f"  [B, 3, 32, 224, 224] -> RawPatchEmbedding -> [B, 32, {self.d_model}]")
        
        lines.append("Stage B - Temporal Encoder:")
        if self.use_transformer:
            lines.append(f"  [B, 32, {self.d_model}] -> SLSTTTransformer -> [B, {self.d_model}]")
        else:
            lines.append(f"  [B, 32, {self.d_model}] -> TemporalPooling -> [B, {self.d_model}]")
            
        lines.append("Stage C - Classifier Head:")
        lines.append(f"  [B, {self.d_model}] -> LayerNorm -> Dropout -> Linear -> [B, num_classes]")
        return "\n".join(lines)


def build_model(ablation, exp) -> AblationMERModel:
    """
    Factory: construct an ``AblationMERModel`` from an ``AblationConfig`` and an
    ``ExperimentConfig`` (see ``ablation_config.py``).

    Keeping construction in one place means the orchestrator never has to know
    the model's argument list — it just passes the two config objects.
    """
    return AblationMERModel(
        num_classes=exp.num_classes,
        use_simam=ablation.use_simam,
        use_cnn=ablation.use_cnn,
        use_transformer=ablation.use_transformer,
        in_channels=exp.in_channels,
        sequence_length=exp.sequence_length,
        d_model=exp.d_model,
        cnn_mid_channels=exp.cnn_mid_channels,
        cnn_out_channels=exp.cnn_out_channels,
        cnn_dropout=exp.cnn_dropout,
        simam_lambda=exp.simam_lambda,
        transformer_nhead=exp.transformer_nhead,
        transformer_num_layers=exp.transformer_num_layers,
        transformer_dim_ff=exp.transformer_dim_ff,
        transformer_dropout=exp.transformer_dropout,
        pool_strategy=exp.pool_strategy,
        raw_patch_grid=exp.raw_patch_grid,
        temporal_pool=exp.temporal_pool,
        classifier_dropout=exp.classifier_dropout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone verification — exercises ALL 8 toggle combinations.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("  AblationMERModel — forward-pass smoke test over toggle combos")
    print("=" * 78)

    dummy = torch.randn(2, 3, 32, 224, 224)
    combos = [
        (s, c, t)
        for c in (False, True)        # use_cnn
        for s in (False, True)        # use_simam
        for t in (False, True)        # use_transformer
    ]
    for use_simam, use_cnn, use_transformer in combos:
        if use_simam and not use_cnn:
            continue  # degenerate: SimAM needs a CNN feature map
        model = AblationMERModel(
            num_classes=3,
            use_simam=use_simam,
            use_cnn=use_cnn,
            use_transformer=use_transformer,
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        params = model.count_parameters()["trainable"]
        tag = f"SimAM={int(use_simam)} CNN={int(use_cnn)} SLSTT={int(use_transformer)}"
        assert out.shape == (2, 3), f"{tag}: bad output {tuple(out.shape)}"
        print(f"  {tag:<30} -> out {tuple(out.shape)} | params {params:,}")

    print("\nOK: All valid toggle combinations produce [2, 3] logits.")
