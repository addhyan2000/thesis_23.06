"""
models — Neural Network Modules for Stage 2
=============================================

Submodules:
    simam               : Parameter-free 3D SimAM attention module
    ststnet_backbone     : Three-branch shallow 3D-CNN feature extractor
    slstt_transformer   : Positional encoding + Transformer encoder
    hybrid_model        : Full pipeline orchestrator (Split → CNN → Attn → Transformer → Head)
"""

from .simam import SimAM3D
from .ststnet_backbone import STSTNetBackbone3D
from .slstt_transformer import SLSTTTransformer
from .hybrid_model import HybridMERModel

__all__ = [
    "SimAM3D",
    "STSTNetBackbone3D",
    "SLSTTTransformer",
    "HybridMERModel",
]
