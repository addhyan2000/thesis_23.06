"""
main_step2.py — Entry Point for Stage 1 / Step 2
================================================

Orchestrates the conversion of cropped micro-expression frames into
homogenized multi-modal tensor cubes (shape: 3x32x224x224).
This script directly kicks off the Multiprocessing Execution pool
via `tensor_pipeline_manager`.

Usage:
    python main_step2.py

Author : Addhyan
Stage  : 1 — Data Pipeline / Step 2
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import OUTPUT_CFG
from step2_extraction.tensor_pipeline_manager import TensorPipelineManager
from utils.logger import get_logger


def _handle_force_reset(log, output_subdir: str) -> None:
    """Clear Step 2 checkpoint and tensor outputs for a clean re-run."""
    ckpt = OUTPUT_CFG.checkpoint_dir / "step2_state.json"
    if ckpt.exists():
        ckpt.unlink()
        log.info("  [DEL] Deleted checkpoint: %s", ckpt)

    tensors_dir = OUTPUT_CFG.processed_root / output_subdir
    if tensors_dir.exists():
        for npy in tensors_dir.glob("*.npy"):
            npy.unlink()
            log.info("  [DEL] Deleted tensor: %s", npy.name)
    log.info("Step 2 force reset complete for output_subdir=%s.", output_subdir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 / Step 2 - Tensor interpolation and modality synthesis."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="ProcessPool worker count. Use a low value for CPU-only systems.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="tensors",
        help="Output directory under Processed_Data (e.g., tensors or tensors_raw).",
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default="CASME_II",
        help="Dataset tag filter from master CSV. Use 'all' to disable.",
    )
    parser.add_argument(
        "--expression_filter",
        type=str,
        default="micro-expression",
        help="Expression filter from master CSV. Use 'all' to disable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete Step 2 checkpoint and existing tensors in output_subdir before running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    log = get_logger("main_step2")
    log.info("=" * 60)
    log.info("STAGE 1 / STEP 2 : Tensor Interpolation & Modality Synthesis")
    log.info("=" * 60)

    if args.force:
        log.info("--force flag detected. Resetting Step 2 checkpoint and tensors...")
        _handle_force_reset(log, args.output_subdir)

    try:
        pipeline_mgr = TensorPipelineManager(
            max_workers=max(1, int(args.max_workers)),
            output_subdir=args.output_subdir,
            dataset_filter=None if args.dataset_filter.lower() == "all" else args.dataset_filter,
            expression_filter=None if args.expression_filter.lower() == "all" else args.expression_filter,
        )
        code = pipeline_mgr.run()
        if code == 0:
            log.info("Pipeline Execution Reached End-Of-Life Sub-Routine Successfully.")
        sys.exit(code)
    except KeyboardInterrupt:
        log.warning("Received KeyboardInterrupt - Pipeline terminated by administrative user interjection.")
        sys.exit(130)
    except Exception as e:
        log.critical("Uncaught architectural level exception escaped to root: %s", e, exc_info=True)
        sys.exit(1)
