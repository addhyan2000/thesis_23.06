"""
metadata_unifier.py — Dataset-Specific Metadata Extraction & Unification
=========================================================================

This module contains two concrete unifier classes:

1.  ``CASMEIIUnifier``   — reads ``CASME2-coding-20140508.xlsx``
2.  ``CASME2SquaredUnifier`` — reads ``CAS(ME)^2code_final.xlsx``

Both inherit from ``BaseMetadataUnifier`` and implement a common
``unify()`` interface that returns a ``pandas.DataFrame`` with the
unified column schema defined in ``config.py``.

Key Responsibilities:
    • Read the raw Excel files with pandas (safely, with error handling)
    • Map native column names / indices → unified schema
    • Apply the emotion mapping (e.g. "disgust" → "Negative")
    • Calculate ``Sequence_Length = Offset_Frame − Onset_Frame + 1``
    • Resolve the absolute frame directory for each sample
    • Verify that the frame directory exists on disk
    • Gracefully handle missing data / unexpected labels

Why Two Classes?
    The Excel files have *fundamentally different structures*:
    CASME II has a proper header row; CAS(ME)^2 has none (the first
    data row was consumed as a header by default pandas behaviour).
    Trying to unify both with if/else spaghetti would be fragile.
    Two classes with a shared ABC is cleaner and more testable.

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 1 — Metadata Unification
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    CASME_II_CFG,
    CASME2_SQ_CFG,
    SCHEMA,
    UNIFIED_EMOTION_MAP,
    CASMEIIConfig,
    CASME2SquaredConfig,
    UnifiedSchema,
)
from utils.logger import get_logger


# =====================================================================
#  Abstract Base Class
# =====================================================================

class BaseMetadataUnifier(abc.ABC):
    """
    Abstract base for dataset-specific metadata unifiers.

    Subclasses must implement ``_read_raw()`` and ``_transform()``.
    The public ``unify()`` method orchestrates the pipeline:
    read → transform → validate → return.
    """

    def __init__(self, schema: UnifiedSchema, emotion_map: Dict[str, str]) -> None:
        self._schema = schema
        self._emotion_map = emotion_map
        self._log = get_logger(self.__class__.__name__)

    # ── Template Method ─────────────────────────────────────────────

    def unify(self) -> pd.DataFrame:
        """
        Execute the full unification pipeline.

        Returns
        -------
        pd.DataFrame
            DataFrame conforming to ``UnifiedSchema.ordered_columns()``.

        Raises
        ------
        FileNotFoundError
            If the source Excel file does not exist.
        """
        self._log.info("═" * 70)
        self._log.info("Starting unification for: %s", self.__class__.__name__)
        self._log.info("═" * 70)

        raw_df = self._read_raw()
        self._log.info("Raw DataFrame loaded.  Shape: %s", raw_df.shape)

        unified_df = self._transform(raw_df)
        self._log.info("Transformation complete.  Shape: %s", unified_df.shape)

        self._validate(unified_df)
        self._log.info(
            "Validation passed.  Final sample count: %d", len(unified_df)
        )

        return unified_df

    # ── Abstract hooks ──────────────────────────────────────────────

    @abc.abstractmethod
    def _read_raw(self) -> pd.DataFrame:
        """Read the raw Excel file and return an unprocessed DataFrame."""

    @abc.abstractmethod
    def _transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns → unified schema, apply emotion mapping, etc."""

    # ── Shared helpers ──────────────────────────────────────────────

    def _map_emotion(self, raw_emotion: str) -> str:
        """
        Map a raw emotion label to the unified category.

        Parameters
        ----------
        raw_emotion : str
            Original label from the Excel file (case-insensitive).

        Returns
        -------
        str
            Unified label ("Positive", "Negative", "Surprise", "Others").

        Notes
        -----
        If the label is not found in ``UNIFIED_EMOTION_MAP``, we log a
        WARNING and default to ``"Others"`` rather than crashing, since one
        unmapped label should not abort hundreds of successfully mapped rows.
        """
        key = str(raw_emotion).strip().lower()
        mapped = self._emotion_map.get(key)
        if mapped is None:
            self._log.warning(
                "UNMAPPED emotion label '%s' — defaulting to 'Others'.  "
                "Consider adding it to UNIFIED_EMOTION_MAP in config.py.",
                raw_emotion,
            )
            return "Others"
        return mapped

    def _calculate_sequence_length(
        self, onset: int, offset: int, video_id: str = ""
    ) -> int:
        """
        Calculate the number of frames in an expression clip.

        Formula: ``Offset − Onset + 1``  (inclusive of both endpoints).

        Parameters
        ----------
        onset : int
            Onset frame number.
        offset : int
            Offset frame number.
        video_id : str, optional
            Used only for logging context if the value is invalid.

        Returns
        -------
        int
            Sequence length, or ``-1`` if the calculation is invalid
            (e.g. offset == 0 meaning "unknown" in CAS(ME)^2).
        """
        try:
            onset_int = int(onset)
            offset_int = int(offset)
        except (ValueError, TypeError):
            self._log.warning(
                "Non-numeric Onset/Offset for '%s': onset=%s, offset=%s",
                video_id, onset, offset,
            )
            return -1

        if offset_int <= 0:
            self._log.debug(
                "Offset is 0 or negative for '%s' — marking sequence length "
                "as -1 (unknown).",
                video_id,
            )
            return -1

        if offset_int < onset_int:
            self._log.warning(
                "Offset (%d) < Onset (%d) for '%s' — sequence length "
                "will be negative; marking as -1.",
                offset_int, onset_int, video_id,
            )
            return -1

        length = offset_int - onset_int + 1
        self._log.debug(
            "Calculated Sequence Length: %d  (Onset=%d, Offset=%d, Video='%s')",
            length, onset_int, offset_int, video_id,
        )
        return length

    def _check_frames_directory(self, frames_dir: Path) -> bool:
        """Return True if the target exists (.avi file or image folder with frames)."""
        if not frames_dir.exists():
            self._log.debug("Frames/Video NOT found: %s", frames_dir)
            return False
        if frames_dir.is_file():
            return True
        if frames_dir.is_dir():
            exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            return any(frames_dir.glob(ext) for ext in exts)
        return False

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame conforms to the unified schema.

        Checks:
            1. All expected columns are present.
            2. No completely empty columns.
            3. Emotion mapping coverage report.
        """
        expected_cols = set(self._schema.ordered_columns())
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        if missing:
            self._log.error("Missing columns in unified output: %s", missing)
            raise ValueError(f"Missing columns: {missing}")

        # ── Report emotion distribution ─────────────────────────────
        emotion_col = self._schema.unified_emotion
        if emotion_col in df.columns:
            dist = df[emotion_col].value_counts()
            self._log.info("Unified emotion distribution:\n%s", dist.to_string())

        # ── Report sequence length statistics ───────────────────────
        seq_col = self._schema.sequence_length
        if seq_col in df.columns:
            valid = df[df[seq_col] > 0][seq_col]
            if not valid.empty:
                self._log.info(
                    "Sequence Length stats (valid only):  "
                    "min=%d  max=%d  mean=%.1f  median=%.1f  std=%.1f",
                    valid.min(), valid.max(), valid.mean(),
                    valid.median(), valid.std(),
                )

        # ── Report missing frames directories ──────────────────────
        exist_col = self._schema.frames_exist
        if exist_col in df.columns:
            missing_count = (~df[exist_col]).sum()
            if missing_count > 0:
                self._log.warning(
                    "%d / %d samples have missing frame directories.",
                    missing_count, len(df),
                )


# =====================================================================
#  CASME II Unifier
# =====================================================================

class CASMEIIUnifier(BaseMetadataUnifier):
    """
    Reads and unifies CASME II metadata (200 fps, 255 samples, 26 subjects).

    Source: ``CASME2-coding-20140508.xlsx``

    Column Mapping:
        Subject         → Subject_ID
        Filename        → Video_ID
        OnsetFrame      → Onset_Frame
        ApexFrame       → Apex_Frame
        OffsetFrame     → Offset_Frame
        Estimated Emotion → Raw_Emotion → (mapped) → Unified_Emotion
        Action Units    → Action_Units

    Frame Directory Pattern:
        ``DATASETS/CASME II/Cropped/sub{XX}/{Filename}/``
        where XX is the zero-padded Subject number.
    """

    def __init__(
        self,
        cfg: Optional[CASMEIIConfig] = None,
        schema: UnifiedSchema = SCHEMA,
        emotion_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(schema, emotion_map or UNIFIED_EMOTION_MAP)
        import config as stage1_config

        self._cfg = cfg if cfg is not None else stage1_config.CASME_II_CFG

    def _read_raw(self) -> pd.DataFrame:
        """
        Read ``CASME2-coding-20140508.xlsx`` with pandas.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame exactly as it appears in the Excel file.

        Raises
        ------
        FileNotFoundError
            If the Excel file does not exist at the configured path.
        """
        excel_path = self._cfg.excel_path

        if not excel_path.exists():
            msg = f"CASME II Excel file not found at: {excel_path}"
            self._log.error(msg)
            raise FileNotFoundError(msg)

        self._log.info("Reading CASME II Excel: %s", excel_path)
        df = pd.read_excel(str(excel_path), engine="openpyxl")
        self._log.info(
            "Successfully read %d rows × %d columns from CASME II Excel.",
            len(df), len(df.columns),
        )
        self._log.debug("Raw columns: %s", df.columns.tolist())
        return df

    def _transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw CASME II DataFrame into the unified schema.

        Steps:
            1. Strip whitespace from text columns.
            2. Map each raw emotion to the unified category.
            3. Calculate sequence length (Offset − Onset + 1).
            4. Resolve absolute path to the frame directory.
            5. Check whether the frame directory exists.
            6. Reorder columns to match ``UnifiedSchema``.
        """
        cfg = self._cfg
        s = self._schema

        rows = []
        for idx, row in raw_df.iterrows():
            subject_id = int(row[cfg.col_subject])
            video_id = str(row[cfg.col_filename]).strip()
            onset = int(row[cfg.col_onset])
            offset = int(row[cfg.col_offset])
            raw_emotion = str(row[cfg.col_emotion]).strip().lower()

            # ── Apex may be non-numeric (some cells contain notes) ──
            try:
                apex = int(row[cfg.col_apex])
            except (ValueError, TypeError):
                apex = np.nan
                self._log.debug(
                    "Non-numeric ApexFrame for sub%02d/%s: '%s'",
                    subject_id, video_id, row[cfg.col_apex],
                )

            # ── Action Units ────────────────────────────────────────
            action_units = str(row[cfg.col_action_units]).strip()

            # ── Emotion mapping ─────────────────────────────────────
            unified_emotion = self._map_emotion(raw_emotion)

            # ── Sequence length ─────────────────────────────────────
            seq_len = self._calculate_sequence_length(onset, offset, video_id)

            # ── Frame directory ─────────────────────────────────────
            subject_folder = (
                f"{cfg.subject_folder_prefix}"
                f"{subject_id:0{cfg.subject_folder_pad}d}"
            )
            if getattr(cfg, "media_mode", "avi") == "images":
                frames_dir = cfg.frames_root / subject_folder / video_id
            else:
                frames_dir = cfg.frames_root / subject_folder / f"{video_id}.avi"
            frames_exist = self._check_frames_directory(frames_dir)

            rows.append({
                s.dataset:          cfg.dataset_tag,
                s.subject_id:       subject_id,
                s.video_id:         video_id,
                s.onset_frame:      onset,
                s.apex_frame:       apex,
                s.offset_frame:     offset,
                s.raw_emotion:      raw_emotion,
                s.unified_emotion:  unified_emotion,
                s.action_units:     action_units,
                s.expression_type:  "micro-expression",   # CASME II is all micro
                s.sequence_length:  seq_len,
                s.fps:              cfg.fps,
                s.frames_dir:       str(frames_dir),
                s.frames_exist:     frames_exist,
                s.of_processed:     False,
            })

            self._log.debug(
                "CASME_II │ sub%02d │ %-12s │ Onset=%4d  Offset=%4d  "
                "SeqLen=%3d │ %s → %s │ Frames=%s",
                subject_id, video_id, onset, offset, seq_len,
                raw_emotion, unified_emotion,
                "✓" if frames_exist else "✗",
            )

        unified_df = pd.DataFrame(rows, columns=s.ordered_columns())

        self._log.info(
            "CASME II unification complete.  %d samples processed.", len(unified_df)
        )
        return unified_df


# =====================================================================
#  CAS(ME)^2 Unifier
# =====================================================================

class CASME2SquaredUnifier(BaseMetadataUnifier):
    """
    Reads and unifies CAS(ME)^2 metadata (30 fps, 357 samples, 22 subjects).

    Source: ``CAS(ME)^2code_final.xlsx``  (NO HEADER ROW)

    Column Index Mapping:
        0 → Subject_ID
        1 → Video_ID
        2 → Onset_Frame
        3 → Apex_Frame
        4 → Offset_Frame       (may be 0 = unknown)
        5 → Action_Units
        6 → Emotion_Category   (negative / positive / surprise / others)
        7 → Expression_Type    (macro-expression / micro-expression)
        8 → Emotion            (anger, sadness, fear, …)

    Frame Directory Pattern:
        ``DATASETS/cropped/{SubjectID}/{VideoID}/``
        where SubjectID is the raw integer (e.g. "15", "16").

    IMPORTANT:
        This dataset contains BOTH macro- and micro-expressions.
        We preserve the ``Expression_Type`` column so downstream
        stages can filter for micro-only if desired.
    """

    def __init__(
        self,
        cfg: Optional[CASME2SquaredConfig] = None,
        schema: UnifiedSchema = SCHEMA,
        emotion_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(schema, emotion_map or UNIFIED_EMOTION_MAP)
        import config as stage1_config

        self._cfg = cfg if cfg is not None else stage1_config.CASME2_SQ_CFG

    def _read_raw(self) -> pd.DataFrame:
        """
        Read ``CAS(ME)^2code_final.xlsx`` WITHOUT a header row.

        The Excel file was released without column headers.  The first
        row (Subject=1, Video=anger1_1, …) is genuine data.
        We therefore pass ``header=None`` to avoid losing it.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with integer column indices (0–8).

        Raises
        ------
        FileNotFoundError
            If the Excel file does not exist at the configured path.
        """
        excel_path = self._cfg.excel_path

        if not excel_path.exists():
            msg = f"CAS(ME)^2 Excel file not found at: {excel_path}"
            self._log.error(msg)
            raise FileNotFoundError(msg)

        self._log.info("Reading CAS(ME)^2 Excel (header=None): %s", excel_path)
        df = pd.read_excel(str(excel_path), header=None, engine="openpyxl")
        self._log.info(
            "Successfully read %d rows × %d columns from CAS(ME)^2 Excel.",
            len(df), len(df.columns),
        )
        return df

    def _transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw CAS(ME)^2 DataFrame into the unified schema.

        Steps:
            1. Access columns by integer index (no headers).
            2. Handle Offset == 0 (unknown) gracefully.
            3. Map raw emotion to unified category.
            4. Preserve Expression_Type for downstream filtering.
            5. Resolve frame directory paths.
            6. Check frame directory existence.
        """
        cfg = self._cfg
        s = self._schema

        rows = []
        skipped_offset_zero = 0

        for idx, row in raw_df.iterrows():
            subject_id = int(row[cfg.idx_subject])
            video_id = str(row[cfg.idx_video_id]).strip()
            onset = int(row[cfg.idx_onset])
            offset = int(row[cfg.idx_offset])
            raw_emotion = str(row[cfg.idx_emotion]).strip().lower()
            expression_type = str(row[cfg.idx_expression_type]).strip().lower()
            action_units = str(row[cfg.idx_action_units]).strip()

            # ── Apex frame ──────────────────────────────────────────
            try:
                apex = int(row[cfg.idx_apex])
            except (ValueError, TypeError):
                apex = np.nan

            # ── Emotion mapping ─────────────────────────────────────
            unified_emotion = self._map_emotion(raw_emotion)

            # ── Sequence length (offset=0 means unknown) ────────────
            seq_len = self._calculate_sequence_length(onset, offset, video_id)
            if offset == 0:
                skipped_offset_zero += 1

            # ── Emotion category from column 6 (for reference) ──────
            # We use the fine-grained emotion (col 8) for mapping,
            # but log the coarse category (col 6) for cross-validation.
            emotion_category = str(row[cfg.idx_emotion_category]).strip().lower()
            self._log.debug(
                "CAS(ME)^2 cross-check: raw_emotion='%s' → "
                "unified='%s', coarse_category='%s'",
                raw_emotion, unified_emotion, emotion_category,
            )

            # ── Frame directory (DYNAMIC RESOLUTION) ────────────────
            # The Excel uses Subject 1, 2, 3, but the disk uses 15, 16, 17.
            # We dynamically hunt down the correct folder using the Video_ID.
            frames_dir = cfg.frames_root / str(subject_id) / f"{video_id}.avi" # Fallback
            
            if cfg.frames_root.exists():
                for folder in cfg.frames_root.iterdir():
                    if folder.is_dir():
                        target_path = folder / f"{video_id}.avi"
                        if target_path.exists():
                            frames_dir = target_path
                            break
                            
            frames_exist = self._check_frames_directory(frames_dir)

            rows.append({
                s.dataset:          cfg.dataset_tag,
                s.subject_id:       subject_id,
                s.video_id:         video_id,
                s.onset_frame:      onset,
                s.apex_frame:       apex,
                s.offset_frame:     offset,
                s.raw_emotion:      raw_emotion,
                s.unified_emotion:  unified_emotion,
                s.action_units:     action_units,
                s.expression_type:  expression_type,
                s.sequence_length:  seq_len,
                s.fps:              cfg.fps,
                s.frames_dir:       str(frames_dir),
                s.frames_exist:     frames_exist,
                s.of_processed:     False,
            })

            self._log.debug(
                "CASME2_SQ │ s%02d │ %-15s │ %s │ Onset=%5d  Offset=%5d  "
                "SeqLen=%4d │ %s → %s │ Frames=%s",
                subject_id, video_id, expression_type,
                onset, offset, seq_len,
                raw_emotion, unified_emotion,
                "✓" if frames_exist else "✗",
            )

        unified_df = pd.DataFrame(rows, columns=s.ordered_columns())

        self._log.info(
            "CAS(ME)^2 unification complete.  %d samples processed  "
            "(%d with unknown offsets).",
            len(unified_df), skipped_offset_zero,
        )
        return unified_df
