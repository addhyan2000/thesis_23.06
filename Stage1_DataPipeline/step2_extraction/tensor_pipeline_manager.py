"""
tensor_pipeline_manager.py — Multiprocessing Orchestrator
=========================================================

Reads the master CSV and coordinates parallel extraction of
Farnebäck optical flow and optical strain tensors.

Design Constraints Satisfied:
    - Multiprocessing via ProcessPoolExecutor (max_workers=12).
    - CV2 threading disabled in the worker via FlowStrainExtractor.
    - Checkpoints correctly resume aborted runs.
    - Resulting (3, 32, 224, 224) tensors saved natively.

Author: Addhyan
Stage : 1 — Data Pipeline / Step 2
"""

from __future__ import annotations

import concurrent.futures
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import OUTPUT_CFG, SCHEMA
from step2_extraction.flow_strain_extractor import FlowStrainExtractor
from step2_extraction.temporal_interpolator import TemporalInterpolator
from utils.checkpoint_manager import CheckpointManager
from utils.logger import get_logger

# Pre-instantiate logger for process space (optional, we use local mostly)
_log = get_logger(__name__)


def process_video_worker(row: Dict[str, Any], tensors_dir_str: str) -> Dict[str, Any]:
    """
    Worker function executed in isolated child processes.

    Parameters
    ----------
    row : dict
        A row representing a video metadata entry from pandas.
    tensors_dir_str : str
        Absolute path to the outputs directory.

    Returns
    -------
    dict
        Execution status summary.
    """
    dataset = row[SCHEMA.dataset]
    video_id = row[SCHEMA.video_id]
    onset = int(row[SCHEMA.onset_frame])
    offset = int(row[SCHEMA.offset_frame])
    frames_dir = row[SCHEMA.frames_dir]

    # Instantiate locally to guarantee memory/lock safety inside standard Python spawn
    # The FlowStrainExtractor class disables OpenCV threading upon init.
    ti = TemporalInterpolator(target_length=33, spatial_size=(224, 224))
    fse = FlowStrainExtractor()

    try:
        # 1. Uniformly sample frames
        frames = ti.load_frames(
            frames_dir=frames_dir,
            onset=onset,
            offset=offset,
            dataset=dataset
        )
        if frames is None:
            return {"status": "error", "dataset": dataset, "video_id": video_id, 
                    "error": f"Failed to load frames spanning {onset} -> {offset} from {frames_dir}"}

        # 2. Extract modalities
        tensor = fse.extract(frames)
        if tensor is None:
            return {"status": "error", "dataset": dataset, "video_id": video_id, 
                    "error": "Extractor returned None"}

        # 3. Save as .npy
        out_filename = f"{dataset}_{video_id}.npy"
        out_path = Path(tensors_dir_str) / out_filename
        np.save(str(out_path), tensor)

        del frames
        del tensor
        import gc
        gc.collect()

        return {
            "status": "success",
            "dataset": dataset,
            "video_id": video_id,
            "save_path": str(out_path)
        }

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        import gc
        gc.collect()
        return {"status": "error", "dataset": dataset, "video_id": video_id, "error": err_msg}


class TensorPipelineManager:
    """
    Orchestrates the asynchronous processing of all valid Micro-Expression videos.

    Parameters
    ----------
    max_workers : int
        Maximum number of parallel child processes. Must be tuned exactly
        for Core i7-12650H architecture (12 threads requested by prompt constraint).
    """

    def __init__(
        self,
        max_workers: int = 12,
        output_subdir: str = "tensors",
        dataset_filter: str | None = "CASME_II",
        expression_filter: str | None = "micro-expression",
    ) -> None:
        self.max_workers = max_workers
        self.log = get_logger(self.__class__.__name__)
        self.dataset_filter = dataset_filter
        self.expression_filter = expression_filter

        self.master_csv_path = OUTPUT_CFG.master_csv_path

        # Tensors output directory inside Processed_Data
        self.tensors_dir = OUTPUT_CFG.processed_root / output_subdir
        self.tensors_dir.mkdir(parents=True, exist_ok=True)

        # Isolated step 2 checkpoint
        step2_ckpt_path = OUTPUT_CFG.checkpoint_dir / "step2_state.json"
        self.checkpoint = CheckpointManager(step2_ckpt_path)

    def _tensor_path(self, dataset: str, video_id: str) -> Path:
        return self.tensors_dir / f"{dataset}_{video_id}.npy"

    def _reconcile_stale_checkpoints(self, valid_df: pd.DataFrame) -> int:
        """Clear completed checkpoints whose .npy file is missing on disk."""
        stale = 0
        for _, row in valid_df.iterrows():
            dataset = row[SCHEMA.dataset]
            video_id = row[SCHEMA.video_id]
            block_key = f"{dataset}_{video_id}"
            if self.checkpoint.get_status(block_key) == "completed" and not self._tensor_path(dataset, video_id).is_file():
                self.checkpoint.mark_pending(block_key)
                stale += 1
        if stale:
            self.log.warning(
                "Reconciled %d stale checkpoint(s): marked completed but tensor file missing.",
                stale,
            )
        return stale

    def _count_tensors_on_disk(self, valid_df: pd.DataFrame) -> int:
        return sum(
            1
            for _, row in valid_df.iterrows()
            if self._tensor_path(row[SCHEMA.dataset], row[SCHEMA.video_id]).is_file()
        )

    def run(self) -> int:
        """Main entry point. Returns process exit code (0 = success)."""
        self.log.info("Starting Temporal Interpolation & Optimal Flow Extraction.")
        
        if not self.master_csv_path.exists():
            self.log.error("Master CSV not found at %s. Please run Step 1 metadata unification.", 
                           self.master_csv_path)
            return 1

        df = pd.read_csv(self.master_csv_path)

        if self.dataset_filter:
            df = df[df[SCHEMA.dataset] == self.dataset_filter]
            self.log.info("Applied dataset filter '%s' -> %d rows.", self.dataset_filter, len(df))

        if self.expression_filter:
            df = df[df[SCHEMA.expression_type] == self.expression_filter]
            self.log.info(
                "Applied expression filter '%s' -> %d rows.",
                self.expression_filter,
                len(df),
            )

        # Subset exactly according to requirements: 
        # Frames_Exist == True and Sequence_Length > 2
        cond_exist = self._as_bool_series(df[SCHEMA.frames_exist])
        cond_len = df[SCHEMA.sequence_length] > 2
        
        valid_df = df[cond_exist & cond_len]
        self.log.info("Validation complete. Found %d valid clips out of %d available in CSV.", 
                      len(valid_df), len(df))

        self._reconcile_stale_checkpoints(valid_df)

        # Filter out completed chunks
        tasks_queue = []
        for _, row in valid_df.iterrows():
            dataset = row[SCHEMA.dataset]
            video_id = row[SCHEMA.video_id]
            block_key = f"{dataset}_{video_id}"

            if not self.checkpoint.is_completed(block_key):
                tasks_queue.append(row.to_dict())

        self.log.info("Clips queued for Optical Flow extractions (ignoring already successful runs): %d", len(tasks_queue))

        if not tasks_queue:
            self.log.info("No clips pending. Ensure the checkpoint file matches the truth or run the CSV sync step.")
            self._sync_csv_with_checkpoint()
            on_disk = self._count_tensors_on_disk(valid_df)
            if len(valid_df) > 0 and on_disk < len(valid_df):
                self.log.error(
                    "Step 2 incomplete: %d/%d tensor files on disk in %s. "
                    "Re-run with --force or copy missing tensors as CASME_II_{Video_ID}.npy.",
                    on_disk,
                    len(valid_df),
                    self.tensors_dir,
                )
                return 2
            if len(valid_df) > 0:
                self.log.info(
                    "All %d valid clip tensor(s) present on disk (%s).",
                    on_disk,
                    self.tensors_dir,
                )
            return 0

        # Start execution pool
        success_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {}
            for row in tasks_queue:
                dataset = row[SCHEMA.dataset]
                video_id = row[SCHEMA.video_id]
                block_key = f"{dataset}_{video_id}"

                _fut = executor.submit(process_video_worker, row, str(self.tensors_dir))
                future_to_key[_fut] = block_key

            # Track progress synchronously inside the master event loop handling exceptions
            completed = 0
            total = len(tasks_queue)
            for future in concurrent.futures.as_completed(future_to_key):
                block_key = future_to_key[future]
                completed += 1

                try:
                    res = future.result()
                    if res["status"] == "success":
                        self.checkpoint.mark_completed(block_key)
                        success_count += 1
                        self.log.info("[%d/%d] SUCCESS - %s saved to %s", 
                                      completed, total, block_key, res["save_path"])
                    else:
                        self.checkpoint.mark_failed(block_key, reason=res["error"])
                        self.log.error("[%d/%d] FAILED - %s: %s", 
                                       completed, total, block_key, res["error"])

                except Exception as loop_exc:
                    self.checkpoint.mark_failed(block_key, reason=str(loop_exc))
                    self.log.error("[%d/%d] CRITICAL ERROR - %s generated an internal exception: %s", 
                                   completed, total, block_key, loop_exc)

        self.log.info("Multiprocessing phase completed. Synchronizing statuses strictly onto the core dataset CSV.")
        self._sync_csv_with_checkpoint()
        on_disk = self._count_tensors_on_disk(valid_df)
        self.log.info(
            "Batch extraction complete. %d new tensor(s) written; %d/%d on disk.",
            success_count,
            on_disk,
            len(valid_df),
        )

        if on_disk == 0:
            self.log.error("No tensor files found in %s.", self.tensors_dir)
            return 1
        if on_disk < len(valid_df):
            self.log.error(
                "Partial extraction: %d/%d tensor files on disk. Check log for failed clips.",
                on_disk,
                len(valid_df),
            )
            return 1
        return 0

    def _sync_csv_with_checkpoint(self) -> None:
        """Reads checkpoint state and globally stamps status on the master CSV."""
        df = pd.read_csv(self.master_csv_path)
        modified_count = 0

        for idx, row in df.iterrows():
            dataset = row[SCHEMA.dataset]
            video_id = row[SCHEMA.video_id]
            block_key = f"{dataset}_{video_id}"

            if self.checkpoint.is_completed(block_key):
                if not self._as_bool_value(row[SCHEMA.of_processed]):
                    df.at[idx, SCHEMA.of_processed] = True
                    modified_count += 1
            else:
                if self._as_bool_value(row[SCHEMA.of_processed]):
                    df.at[idx, SCHEMA.of_processed] = False
                    modified_count += 1

        if modified_count > 0:
            df.to_csv(self.master_csv_path, index=False)
            self.log.info("Reflected %d new tracking updates accurately across output data file: %s", 
                          modified_count, OUTPUT_CFG.master_csv_filename)
        else:
            self.log.info("Dataset sync complete. No divergent OF_Processed statuses flagged.")

    @staticmethod
    def _as_bool_series(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        lowered = series.astype(str).str.strip().str.lower()
        return lowered.isin({"true", "1", "yes", "y"})

    @staticmethod
    def _as_bool_value(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "y"}
