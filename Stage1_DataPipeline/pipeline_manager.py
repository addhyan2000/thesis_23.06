"""
pipeline_manager.py — Fault-Tolerant Pipeline Orchestrator
==========================================================

``PipelineManager`` is the central orchestrator for Step 1.  It wires
together the ``MetadataUnifier`` classes and the ``CheckpointManager``
to produce the final ``master_thesis_labels.csv`` with full crash
resilience.

Execution Flow:
    1.  Load / create checkpoint state.
    2.  Block "CASME_II_metadata":
            • If already completed → skip.
            • Otherwise → run ``CASMEIIUnifier.unify()``, persist
              intermediate DataFrame, mark block completed.
    3.  Block "CASME2_Squared_metadata":
            • Same pattern.
    4.  Block "merge_and_export":
            • Concatenate the two DataFrames.
            • Sort by (Dataset, Subject_ID, Video_ID).
            • Export to ``Processed_Data/master_thesis_labels.csv``.
            • Print summary statistics.
            • Mark block completed.

If the script crashes at block 2, restarting will skip block 1
entirely and resume from block 2.

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 1 — Metadata Unification
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    CASME_II_CFG,
    CASME2_SQ_CFG,
    OUTPUT_CFG,
    SCHEMA,
)
from metadata_unifier import CASMEIIUnifier, CASME2SquaredUnifier
from utils.checkpoint_manager import CheckpointManager
from utils.logger import get_logger


class PipelineManager:
    """
    Orchestrates the Step 1 metadata-unification pipeline with
    checkpoint-based fault tolerance.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Override for the checkpoint file location.
    output_csv_path : Path, optional
        Override for the master CSV output location.
    """

    # ── Logical block names (must be stable across runs) ────────────
    BLOCK_CASME_II: str = "CASME_II_metadata"
    BLOCK_CASME2_SQ: str = "CASME2_Squared_metadata"
    BLOCK_MERGE: str = "merge_and_export"

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        output_csv_path: Optional[Path] = None,
        include_casme2_squared: bool = True,
    ) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._output_csv = output_csv_path or OUTPUT_CFG.master_csv_path
        self._include_casme2_squared = include_casme2_squared
        self._ckpt = CheckpointManager(
            checkpoint_path or OUTPUT_CFG.checkpoint_path
        )

        # ── Intermediate storage for fault tolerance ────────────────
        # If CASME II was already processed in a prior run, we reload
        # its DataFrame from the intermediate CSV rather than
        # re-reading the Excel.
        self._intermediate_dir = OUTPUT_CFG.processed_root / "intermediates"
        self._intermediate_dir.mkdir(parents=True, exist_ok=True)

        self._casme2_path = self._intermediate_dir / "casme_ii_unified.csv"
        self._casme2sq_path = self._intermediate_dir / "casme2_sq_unified.csv"

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full Step 1 pipeline.

        Returns
        -------
        pd.DataFrame
            The merged, unified master DataFrame (also saved to CSV).
        """
        self._log.info("=" * 70)
        self._log.info("STAGE 1 / STEP 1 - Unified Metadata Pipeline")
        self._log.info("=" * 70)
        self._log.info("Checkpoint status:\n%s", self._ckpt.summary())

        # ── Block 1: CASME II ───────────────────────────────────────
        casme_ii_df = self._process_block(
            block_name=self.BLOCK_CASME_II,
            unifier_factory=lambda: CASMEIIUnifier(),
            intermediate_path=self._casme2_path,
        )

        # ── Block 2: CAS(ME)^2 (optional) ───────────────────────────
        if self._include_casme2_squared:
            casme2_sq_df = self._process_block(
                block_name=self.BLOCK_CASME2_SQ,
                unifier_factory=lambda: CASME2SquaredUnifier(),
                intermediate_path=self._casme2sq_path,
            )
        else:
            self._log.info("Skipping CAS(ME)^2 block (CASME-II-only mode enabled).")
            casme2_sq_df = pd.DataFrame(columns=SCHEMA.ordered_columns())

        # ── Block 3: Merge & Export ─────────────────────────────────
        master_df = self._merge_and_export(casme_ii_df, casme2_sq_df)

        self._log.info("=" * 70)
        self._log.info("PIPELINE COMPLETE")
        self._log.info("=" * 70)
        self._log.info("Master CSV: %s", self._output_csv)
        self._log.info("Total samples: %d", len(master_df))
        self._log.info("Checkpoint status:\n%s", self._ckpt.summary())

        return master_df

    # ─────────────────────────────────────────────────────────────────
    #  Private Helpers
    # ─────────────────────────────────────────────────────────────────

    def _process_block(
        self,
        block_name: str,
        unifier_factory,
        intermediate_path: Path,
    ) -> pd.DataFrame:
        """
        Process a single dataset block with checkpoint support.

        If the block is already completed AND an intermediate CSV
        exists, we reload from CSV (fast).  Otherwise, we run the
        unifier, save the intermediate, and mark the block completed.

        Parameters
        ----------
        block_name : str
            Logical block identifier for the checkpoint.
        unifier_factory : callable
            Zero-argument callable returning a ``BaseMetadataUnifier``.
        intermediate_path : Path
            Path to save/load the intermediate CSV file.

        Returns
        -------
        pd.DataFrame
            The unified DataFrame for this dataset.
        """
        # ── Fast path: already done ─────────────────────────────────
        if self._ckpt.is_completed(block_name):
            if intermediate_path.exists():
                self._log.info(
                    "Loading cached intermediate: %s", intermediate_path
                )
                return pd.read_csv(str(intermediate_path))
            else:
                self._log.warning(
                    "Block '%s' marked completed but intermediate file "
                    "not found at %s.  Re-processing.",
                    block_name, intermediate_path,
                )
                # Fall through to re-process.

        # ── Slow path: run the unifier ──────────────────────────────
        try:
            unifier = unifier_factory()
            df = unifier.unify()

            # Persist intermediate (CSV — zero external dependencies)
            df.to_csv(str(intermediate_path), index=False, encoding="utf-8")
            self._log.info(
                "Saved intermediate CSV: %s  (%d rows)",
                intermediate_path, len(df),
            )

            self._ckpt.mark_completed(block_name)
            return df

        except Exception as exc:
            self._ckpt.mark_failed(block_name, reason=str(exc))
            self._log.exception(
                "FATAL - Block '%s' failed.  The pipeline will exit.  "
                "Fix the issue and re-run; completed blocks will be skipped.",
                block_name,
            )
            raise

    def _merge_and_export(
        self, casme_ii_df: pd.DataFrame, casme2_sq_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge both DataFrames and export the master CSV.

        Parameters
        ----------
        casme_ii_df : pd.DataFrame
            Unified CASME II data.
        casme2_sq_df : pd.DataFrame
            Unified CAS(ME)^2 data.

        Returns
        -------
        pd.DataFrame
            The concatenated, sorted master DataFrame.
        """
        if self._ckpt.is_completed(self.BLOCK_MERGE):
            if self._output_csv.exists():
                self._log.info(
                    "Loading existing master CSV: %s", self._output_csv
                )
                return pd.read_csv(str(self._output_csv))
            else:
                self._log.warning(
                    "Merge block marked completed but CSV not found.  "
                    "Re-generating."
                )

        try:
            self._log.info("Merging CASME II (%d) + CAS(ME)^2 (%d)...",
                           len(casme_ii_df), len(casme2_sq_df))

            master_df = pd.concat(
                [casme_ii_df, casme2_sq_df],
                axis=0,
                ignore_index=True,
            )

            # ── Sort for reproducibility ────────────────────────────
            sort_cols = [SCHEMA.dataset, SCHEMA.subject_id, SCHEMA.video_id]
            master_df.sort_values(sort_cols, inplace=True)
            master_df.reset_index(drop=True, inplace=True)

            # ── Ensure output directory exists ──────────────────────
            self._output_csv.parent.mkdir(parents=True, exist_ok=True)

            # ── Export ──────────────────────────────────────────────
            master_df.to_csv(str(self._output_csv), index=False, encoding="utf-8")
            self._log.info(
                "Master CSV exported: %s  (%d rows)", self._output_csv, len(master_df)
            )

            # ── Summary statistics ──────────────────────────────────
            self._print_summary(master_df)

            self._ckpt.mark_completed(self.BLOCK_MERGE)
            return master_df

        except Exception as exc:
            self._ckpt.mark_failed(self.BLOCK_MERGE, reason=str(exc))
            self._log.exception("FATAL — Merge & export failed.")
            raise

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Log comprehensive summary statistics for the thesis."""
        s = SCHEMA
        self._log.info("-" * 70)
        self._log.info("MASTER DATASET SUMMARY")
        self._log.info("-" * 70)

        # ── Per-dataset counts ──────────────────────────────────────
        dataset_counts = df[s.dataset].value_counts()
        self._log.info("Samples per dataset:\n%s", dataset_counts.to_string())

        # ── Per-dataset emotion distribution ────────────────────────
        for dataset_name, group in df.groupby(s.dataset):
            dist = group[s.unified_emotion].value_counts()
            self._log.info(
                "Emotion distribution for %s:\n%s",
                dataset_name, dist.to_string(),
            )

        # ── Expression type distribution (relevant for CAS(ME)^2) ──
        expr_dist = df[s.expression_type].value_counts()
        self._log.info("Expression type distribution:\n%s", expr_dist.to_string())

        # ── Sequence length statistics per dataset ──────────────────
        for dataset_name, group in df.groupby(s.dataset):
            valid = group[group[s.sequence_length] > 0][s.sequence_length]
            if valid.empty:
                self._log.warning(
                    "No valid sequence lengths for %s.", dataset_name
                )
                continue
            self._log.info(
                "Sequence Length for %s:  "
                "min=%d  max=%d  mean=%.1f  median=%.1f  std=%.1f",
                dataset_name,
                valid.min(), valid.max(), valid.mean(),
                valid.median(), valid.std(),
            )

        # ── FPS breakdown ───────────────────────────────────────────
        fps_dist = df[s.fps].value_counts()
        self._log.info("FPS distribution:\n%s", fps_dist.to_string())

        # ── Frames existence ────────────────────────────────────────
        found = df[s.frames_exist].sum()
        total = len(df)
        self._log.info(
            "Frame directories found: %d / %d (%.1f%%)",
            found, total, 100.0 * found / total if total > 0 else 0,
        )

        self._log.info("-" * 70)
