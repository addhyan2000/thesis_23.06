"""
main_stage2.py — Stage 2: Hybrid Neural Architecture Orchestrator
===================================================================

This script is the primary entry point for the Stage 2 Hybrid Micro-Expression
Recognition (MER) architecture. It validates the end-to-end integration of:
    1. STSTNet-SimAM Backbone (3D-CNN + Attention)
    2. SLSTT Transformer (Sequence Modelling)
    3. Three-class Classification Head

Key Features:
    - Layer-by-layer shape and parameter auditing for thesis documentation.
    - Resource-aware (CUDA/MPS/CPU) device management.
    - Integrated logging with Stage 1 infrastructure.
    - Configurable architecture via CLI.

Usage:
    # Basic architecture verification (Dry Run)
    python -m Stage2_Architecture.main_stage2 --mode verify

    # Custom hyperparameter check
    python -m Stage2_Architecture.main_stage2 --cnn_out 64 --trans_heads 4

Author  : Addhyan
Stage   : 2 — Hybrid Neural Architecture
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
# 1. PATH SETUP — Ensure Stage 1 and Stage 2 are importable
# ─────────────────────────────────────────────────────────────────────
# Resolve Thesis3/ root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Stage1_DataPipeline"))

# ── Stage 1 Utilities ──
from utils.logger import get_logger          # type: ignore[import]
from config import OUTPUT_CFG                # type: ignore[import]

# ── Stage 2 Model ──
from Stage2_Architecture.models.hybrid_model import HybridMERModel

# ─────────────────────────────────────────────────────────────────────
# 2. LOGGING INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "Stage2_Architecture" / "logs"
log = get_logger(
    "main_stage2",
    log_dir=LOG_DIR,
    log_filename="main_stage2.log",
)


# ─────────────────────────────────────────────────────────────────────
# 3. ARCHITECTURE AUDITING UTILITY
# ─────────────────────────────────────────────────────────────────────

class ArchAuditor:
    """
    Cleaner for thesis-grade model summaries.
    Provides layer-by-layer shape tracing and parameter counting.
    """

    @staticmethod
    def print_summary(model: nn.Module, input_shape: tuple) -> None:
        """
        Prints a structured table of model layers with output shapes
        and parameter counts.
        """
        log.info("=" * 85)
        log.info(f"{'Component/Layer':<40} │ {'Output Shape':<25} │ {'Params':<10}")
        log.info("-" * 85)

        device = next(model.parameters()).device
        x = torch.zeros(*input_shape).to(device)

        hooks = []
        summary = []

        def register_hook(module: nn.Module, name: str):
            def hook(m, input, output):
                # Count params in this specific submodule
                params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                # Handle list/tuple outputs (e.g. transformer)
                if isinstance(output, (list, tuple)):
                    out_shape = str(list(output[0].shape))
                else:
                    out_shape = str(list(output.shape))
                
                summary.append((name, out_shape, params))

            hooks.append(module.register_forward_hook(hook))

        # Register hooks for major sub-blocks
        register_hook(model.backbone, "Backbone (STSTNet-SimAM)")
        register_hook(model.spatial_pool, "Adaptive Spatial Pool")
        register_hook(model.transformer, "SLSTT Transformer")
        register_hook(model.classifier, "Classification Head")

        # Run forward pass to trigger hooks
        model.eval()
        with torch.no_grad():
            model(x)

        # Print the gathered statistics
        for name, shape, params in summary:
            log.info(f"{name:<40} │ {shape:<25} │ {params:>10,}")

        # Remove hooks
        for h in hooks:
            h.remove()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("-" * 85)
        log.info(f"{'Total Trainable Parameters':<40} │ {'':<25} │ {total_params:>10,}")
        log.info("=" * 85)


# ─────────────────────────────────────────────────────────────────────
# 4. CORE PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Enforce reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"RNG seeds fixed to {seed} (Deterministic mode).")


def get_device() -> torch.device:
    """Detect available hardware accelerator."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Hardware: GPU detected ({torch.cuda.get_device_name(0)})")
        # Log GPU capacity
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU Capacity: {total_mem:.2f} GB VRAM available.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Hardware: Apple Silicon GPU (MPS) detected.")
    else:
        device = torch.device("cpu")
        log.warning("Hardware: No GPU found. Falling back to CPU (Not recommended for 3D-CNNs).")
    return device


# ─────────────────────────────────────────────────────────────────────
# 5. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Hybrid MER Architecture Orchestrator")
    
    # -- Execution Mode --
    parser.add_argument("--mode", type=str, default="verify", choices=["verify", "train"],
                        help="Mode: 'verify' for dry run, 'train' for actual processing.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")

    # -- Model Hyperparameters --
    parser.add_argument("--classes", type=int, default=3, help="Number of emotion classes.")
    parser.add_argument("--cnn_mid", type=int, default=16, help="Intermediate CNN channels.")
    parser.add_argument("--cnn_out", type=int, default=32, help="Final CNN channels per stream.")
    parser.add_argument("--trans_heads", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--trans_layers", type=int, default=4, help="Transformer encoder depth.")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"],
                        help="Temporal pooling strategy.")

    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  STAGE 2: HYBRID ARCHITECTURE INITIALIZATION")
    log.info("=" * 70)

    # 1. Setup Environment
    set_seed(args.seed)
    device = get_device()

    # 2. Instantiate Model
    log.info("Building HybridMERModel architecture...")
    try:
        model = HybridMERModel(
            num_classes=args.classes,
            cnn_mid_channels=args.cnn_mid,
            cnn_out_channels=args.cnn_out,
            transformer_nhead=args.trans_heads,
            transformer_num_layers=args.trans_layers,
            pool_strategy=args.pool
        ).to(device)
        log.info("  [SUCCESS] Model constructed and moved to device.")
    except Exception as e:
        log.critical(f"  [FAILURE] Model instantiation failed: {e}")
        sys.exit(1)

    # 3. Model Auditing (For Thesis Documentation)
    input_shape = (1, 3, 32, 224, 224)
    ArchAuditor.print_summary(model, input_shape)

    # 4. Verified Forward Pass (Verify Integration)
    if args.mode == "verify":
        log.info("\nStarting Architecture Verification (Dry Run)...")
        dummy_input = torch.randn(*input_shape).to(device)
        
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        elapsed = (time.time() - start_time) * 1000

        log.info(f"  Forward pass completed in {elapsed:.2f}ms.")
        log.info(f"  Input Shape  : {list(dummy_input.shape)}")
        log.info(f"  Output Shape : {list(logits.shape)}")
        log.info(f"  Logits Peak  : {logits.abs().max().item():.4f}")
        
        # Verify non-nan
        if torch.isnan(logits).any():
            log.error("  [ALERT] NaNs detected in output! Check weight initialization.")
        else:
            log.info("  [PASS] Numerical stability verified.")

    elif args.mode == "train":
        log.info("-" * 70)
        log.info("TRAINING MODE INITIATED")
        log.info("-" * 70)
        log.warning("NOTE: Dataset loaders for Stage 2 are currently being unified.")
        log.warning("Integration with Master Dataset CSV is step 2 of this stage.")
        # TODO: Integrate Stage 1 Processed npy files here.

    log.info("\n" + "=" * 70)
    log.info("  ✓ STAGE 2 INITIALIZATION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
