"""
main_step1.py — Entry Point for Stage 1 / Step 1
=================================================

This script is the single entry point for the Metadata Unification
pipeline.  It performs the following:

    1.  Validates that all required raw dataset files exist.
    2.  Instantiates the ``PipelineManager``.
    3.  Calls ``PipelineManager.run()`` which handles:
        a.  CASME II metadata unification  (checkpoint-guarded)
        b.  CAS(ME)^2 metadata unification (checkpoint-guarded)
        c.  Merge & CSV export             (checkpoint-guarded)
    4.  Exits cleanly with a summary.

Usage::

    cd Stage1_DataPipeline
    python main_step1.py

    # To force a full re-run (reset all checkpoints):
    python main_step1.py --force

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 1 — Metadata Unification
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure the Stage1 package root is on sys.path ──────────────────
# This allows ``from config import …`` to work when the script is
# invoked as ``python main_step1.py`` from within Stage1_DataPipeline/.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import CASME_II_CFG, CASME2_SQ_CFG, OUTPUT_CFG    # noqa: E402
from pipeline_manager import PipelineManager                   # noqa: E402
from utils.logger import get_logger                             # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 / Step 1 - Unified Metadata Pipeline.  "
            "Reads CASME II and CAS(ME)^2 ground-truth labels, "
            "unifies emotion categories, calculates sequence lengths, "
            "and outputs master_thesis_labels.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Delete existing checkpoints and intermediate files, "
            "forcing a full re-run from scratch."
        ),
    )
    parser.add_argument(
        "--dataset_mode",
        choices=["casme2_only", "both"],
        default="casme2_only",
        help=(
            "Dataset scope for Step 1. 'casme2_only' builds CSV from CASME II only; "
            "'both' includes CAS(ME)^2 as well."
        ),
    )
    parser.add_argument(
        "--casme2_excel",
        type=Path,
        default=None,
        help="Override path to CASME2-coding-20140508.xlsx",
    )
    parser.add_argument(
        "--casme2_frames_root",
        type=Path,
        default=None,
        help=(
            "Override media root. avi mode: subXX/{Filename}.avi  |  "
            "images mode: subXX/{Filename}/ with jpg/png frames (CASME-II Cropped)."
        ),
    )
    parser.add_argument(
        "--casme2_media_mode",
        choices=["avi", "images"],
        default=None,
        help="avi = .avi clips | images = Cropped image folders per clip.",
    )
    return parser.parse_args()


def _apply_path_overrides(args: argparse.Namespace) -> None:
    """Patch CASME_II_CFG when GUI or CLI supplies custom dataset paths."""
    if not args.casme2_excel and not args.casme2_frames_root and not args.casme2_media_mode:
        return
    from dataclasses import replace
    import config as stage1_config

    updates = {}
    if args.casme2_excel:
        updates["excel_path"] = Path(args.casme2_excel).resolve()
    if args.casme2_frames_root:
        updates["frames_root"] = Path(args.casme2_frames_root).resolve()
    if args.casme2_media_mode:
        updates["media_mode"] = args.casme2_media_mode
    stage1_config.CASME_II_CFG = replace(stage1_config.CASME_II_CFG, **updates)


def _validate_prerequisites(log, include_casme2_squared: bool) -> bool:
    """
    Verify that all required raw dataset files exist before we start.

    Returns
    -------
    bool
        ``True`` if all prerequisites are satisfied.
    """
    import config as stage1_config

    casme2_cfg = stage1_config.CASME_II_CFG
    casme2_sq_cfg = stage1_config.CASME2_SQ_CFG
    all_ok = True

    # ── CASME II Excel ──────────────────────────────────────────────
    if not casme2_cfg.excel_path.exists():
        log.error("MISSING: CASME II Excel at %s", casme2_cfg.excel_path)
        all_ok = False
    else:
        log.info("  [OK] CASME II Excel:     %s", casme2_cfg.excel_path)

    # ── CASME II frames root ────────────────────────────────────────
    if not casme2_cfg.frames_root.is_dir():
        log.error("MISSING: CASME II frames at %s", casme2_cfg.frames_root)
        all_ok = False
    else:
        log.info("  [OK] CASME II Frames:    %s", casme2_cfg.frames_root)

    if include_casme2_squared:
        # ── CAS(ME)^2 Excel ────────────────────────────────────────
        if not casme2_sq_cfg.excel_path.exists():
            log.error("MISSING: CAS(ME)^2 Excel at %s", casme2_sq_cfg.excel_path)
            all_ok = False
        else:
            log.info("  [OK] CAS(ME)^2 Excel:    %s", casme2_sq_cfg.excel_path)

        # ── CAS(ME)^2 frames root ──────────────────────────────────
        if not casme2_sq_cfg.frames_root.is_dir():
            log.error("MISSING: CAS(ME)^2 frames at %s", casme2_sq_cfg.frames_root)
            all_ok = False
        else:
            log.info("  [OK] CAS(ME)^2 Frames:   %s", casme2_sq_cfg.frames_root)
    else:
        log.info("  [SKIP] CAS(ME)^2 checks skipped (CASME-II-only mode).")

    return all_ok


def _handle_force_reset(log) -> None:
    """
    Delete checkpoint and intermediate files for a clean re-run.

    This is a destructive operation — we only delete *our own*
    generated artefacts, never the raw dataset files.
    """
    import shutil

    targets = [
        OUTPUT_CFG.checkpoint_path,
        OUTPUT_CFG.master_csv_path,
        OUTPUT_CFG.processed_root / "intermediates",
    ]

    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
                log.info("  [DEL] Deleted directory: %s", target)
            else:
                target.unlink()
                log.info("  [DEL] Deleted file: %s", target)
        else:
            log.debug("  (not found, skipping): %s", target)

    log.info("Force reset complete.  All checkpoints cleared.")


def _should_auto_force(args: argparse.Namespace, log) -> bool:
    """Force refresh when GUI Excel row count differs from cached metadata."""
    if args.force or not args.casme2_excel:
        return False

    import config as stage1_config

    excel_path = stage1_config.CASME_II_CFG.excel_path
    intermediate = OUTPUT_CFG.processed_root / "intermediates" / "casme_ii_unified.csv"
    master_csv = OUTPUT_CFG.master_csv_path

    if not excel_path.is_file():
        return False

    try:
        import pandas as pd

        excel_rows = len(pd.read_excel(excel_path, engine="openpyxl"))
    except Exception as exc:
        log.debug("Could not read Excel for auto-force check: %s", exc)
        return False

    cached_rows = None
    if intermediate.is_file():
        try:
            cached_rows = len(pd.read_csv(intermediate))
        except Exception:
            pass
    elif master_csv.is_file():
        try:
            cached_rows = len(pd.read_csv(master_csv))
        except Exception:
            pass

    if cached_rows is None:
        return False

    if excel_rows != cached_rows:
        log.warning(
            "Excel has %d rows but cached metadata has %d — auto force refresh.",
            excel_rows,
            cached_rows,
        )
        return True
    return False


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    log = get_logger("main_step1")
    _apply_path_overrides(args)

    log.info("=" * 70)
    log.info("Micro-Expression Recognition - Thesis Data Pipeline")
    log.info("Stage 1 / Step 1: Unified Metadata Generation")
    log.info("=" * 70)

    # ── Force reset if requested ────────────────────────────────────
    if args.force:
        log.info("--force flag detected. Resetting all checkpoints...")
        _handle_force_reset(log)
    elif _should_auto_force(args, log):
        log.info("Stale metadata cache detected. Resetting all checkpoints...")
        _handle_force_reset(log)

    include_casme2_squared = args.dataset_mode == "both"

    # ── Pre-flight checks ───────────────────────────────────────────
    log.info("Running pre-flight checks...")
    if not _validate_prerequisites(log, include_casme2_squared):
        log.error(
            "Pre-flight checks FAILED.  Please ensure all raw dataset "
            "files are present before running the pipeline."
        )
        sys.exit(1)
    log.info("All pre-flight checks passed.\n")

    # ── Run the pipeline ────────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        manager = PipelineManager(include_casme2_squared=include_casme2_squared)
        master_df = manager.run()
    except Exception:
        log.exception(
            "Pipeline terminated with an unrecoverable error.  "
            "Fix the issue and re-run - completed blocks will be skipped "
            "thanks to the checkpoint system."
        )
        sys.exit(2)

    elapsed = time.perf_counter() - t_start
    log.info("Total wall-clock time: %.2f seconds.", elapsed)

    # ── Final human-readable summary ────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("OUTPUT FILE:  %s", OUTPUT_CFG.master_csv_path)
    log.info("TOTAL ROWS:   %d", len(master_df))
    log.info("COLUMNS:      %s", master_df.columns.tolist())
    log.info("=" * 70)
    log.info("Step 1 complete.  Ready for Step 2 (Optical Flow extraction).")


if __name__ == "__main__":
    main()
