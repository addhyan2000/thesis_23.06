"""
Stage2_Architecture — Hybrid Neural Architecture for Micro-Expression Recognition
==================================================================================

This package implements the complete Stage 2 pipeline:
    STSTNet (3D-CNN) → SimAM (3D Attention) → SLSTT (Transformer Encoder)

The network ingests 5D tensors of shape [Batch, 3, 32, 224, 224] where:
    - Channel 0: Horizontal Optical Flow (u)
    - Channel 1: Vertical Optical Flow (v)
    - Channel 2: Optical Strain (os)

Output: [Batch, 3] logits for {Positive, Negative, Surprise}.

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""
