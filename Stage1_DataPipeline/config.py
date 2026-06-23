"""
config.py — Central Configuration for Stage 1: Data Pipeline
=============================================================

This module defines all path constants, column-name mappings, emotion
unification dictionaries, and dataset-specific parameters used across
the entire Stage 1 pipeline.  Every other module imports from here,
ensuring a single source of truth that is trivially auditable by the
thesis committee.

Design Decision:
    We use a dataclass-based configuration rather than a loose dict so
    the IDE can auto-complete every field and mypy can type-check paths.
    All paths are resolved relative to a single PROJECT_ROOT so the
    pipeline is portable across machines.

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 1 — Metadata Unification
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────
# 1.  PROJECT ROOT  — Every other path is derived from this.
# ─────────────────────────────────────────────────────────────────────
# Resolve to the parent of Stage1_DataPipeline/, i.e. Thesis3/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# 2.  RAW DATASET PATHS  (READ-ONLY — never write here)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CASMEIIConfig:
    """Paths and column schema for CASME II (200 fps)."""

    # ── Filesystem locations ──
    excel_path: Path = PROJECT_ROOT / "DATASETS" / "CASME II" / "CASME2-coding-20140508.xlsx"
    frames_root: Path = PROJECT_ROOT / "Processed_Data" / "Raw_Videos_Magnified" / "CASME2"

    # ── Native Excel column names (as they appear in the spreadsheet) ──
    col_subject: str = "Subject"
    col_filename: str = "Filename"           # e.g. "EP02_01f"
    col_onset: str = "OnsetFrame"
    col_apex: str = "ApexFrame"
    col_offset: str = "OffsetFrame"
    col_action_units: str = "Action Units"
    col_emotion: str = "Estimated Emotion"

    # ── Dataset-specific constants ──
    fps: int = 200
    dataset_tag: str = "CASME_II"

    # ── Subject folder pattern: sub{XX} where XX is zero-padded ──
    subject_folder_prefix: str = "sub"
    subject_folder_pad: int = 2              # sub01 … sub26

    # "avi"    → {frames_root}/subXX/{Filename}.avi
    # "images" → {frames_root}/subXX/{Filename}/  (jpg/png frame folders, CASME-II Cropped)
    media_mode: str = "avi"


@dataclass(frozen=True)
class CASME2SquaredConfig:
    """
    Paths and column schema for CAS(ME)^2 (30 fps).

    CRITICAL NOTE:
        The Excel file ``CAS(ME)^2code_final.xlsx`` ships **without a
        header row**.  The very first row (Subject=1, Video=anger1_1,
        Onset=557, …) *is* data, not a header.  We therefore read it
        with ``header=None`` and reference columns by integer index.

    Column Index Mapping (verified via _inspect_data.py):
        0 → Subject ID   (int, 1–22)
        1 → Video ID     (str, e.g. "anger1_1")
        2 → Onset Frame  (int)
        3 → Apex Frame   (int)
        4 → Offset Frame (int)       ← may be 0 when unknown
        5 → Action Units (str/mixed)
        6 → Emotion Category (str: negative / positive / surprise / others)
        7 → Expression Type  (str: macro-expression / micro-expression)
        8 → Emotion          (str: anger, sadness, fear, …)
    """

    # ── Filesystem locations ──
    excel_path: Path = PROJECT_ROOT / "DATASETS" / "CAS ME^2" / "CAS(ME)^2code_final.xlsx"
    frames_root: Path = PROJECT_ROOT / "Processed_Data" / "Raw_Videos_Magnified" / "CASME_SQUARED"

    # ── Column **indices** (no header row) ──
    idx_subject: int = 0
    idx_video_id: int = 1
    idx_onset: int = 2
    idx_apex: int = 3
    idx_offset: int = 4
    idx_action_units: int = 5
    idx_emotion_category: int = 6
    idx_expression_type: int = 7
    idx_emotion: int = 8

    # ── Dataset-specific constants ──
    fps: int = 30
    dataset_tag: str = "CASME2_Squared"

    # ── Subject folders use raw numbers (e.g. "15", "16", …) ──
    subject_folder_prefix: str = ""
    subject_folder_pad: int = 0


# ─────────────────────────────────────────────────────────────────────
# 3.  OUTPUT PATHS  (all artefacts are written here)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OutputConfig:
    """
    Directories and file paths for processed outputs.
    These directories will be created at runtime if they do not exist.
    """

    processed_root: Path = PROJECT_ROOT / "Processed_Data"
    master_csv_filename: str = "master_thesis_labels.csv"
    checkpoint_dir: Path = PROJECT_ROOT / "Processed_Data" / "checkpoints"
    checkpoint_filename: str = "processed_state.json"
    log_dir: Path = PROJECT_ROOT / "Stage1_DataPipeline" / "logs"
    log_filename: str = "pipeline_step1.log"

    @property
    def master_csv_path(self) -> Path:
        return self.processed_root / self.master_csv_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_filename

    @property
    def log_path(self) -> Path:
        return self.log_dir / self.log_filename


# ─────────────────────────────────────────────────────────────────────
# 4.  UNIFIED EMOTION MAPPING
# ─────────────────────────────────────────────────────────────────────
#
#  Raw emotion labels across both datasets are heterogeneous:
#    CASME II  → happiness, others, disgust, repression, surprise,
#                fear, sadness
#    CAS(ME)^2 → anger, sadness, fear, disgust, happiness, surprise,
#                helpless, pain, confused, happyiness (sic), sympathy
#
#  We collapse them into three thesis-level categories plus an
#  "Others" bin.  The mapping is intentionally exhaustive so that
#  any unmapped label causes a loud KeyError at runtime rather than
#  being silently dropped.
#
#  Academic justification:
#    • "Positive"  = enjoyment / amusement micro-expressions
#    • "Negative"  = aversive valence (anger, disgust, fear, sadness)
#    • "Surprise"  = surprise (valence-neutral, high arousal)
#    • "Others"    = ambiguous / unclassifiable
# ─────────────────────────────────────────────────────────────────────

UNIFIED_EMOTION_MAP: Dict[str, str] = {
    # ── Positive ──
    "happiness":    "Positive",
    "happyiness":   "Positive",   # Typo present in CAS(ME)^2 Excel

    # ── Negative ──
    "disgust":      "Negative",
    "sadness":      "Negative",
    "fear":         "Negative",
    "anger":        "Negative",
    "repression":   "Negative",   # CASME II treats this as suppressed negative

    # ── Surprise (valence-neutral) ──
    "surprise":     "Surprise",

    # ── Others / Ambiguous ──
    "others":       "Others",
    "helpless":     "Others",
    "pain":         "Others",
    "confused":     "Others",
    "sympathy":     "Others",
}


# ─────────────────────────────────────────────────────────────────────
# 5.  UNIFIED OUTPUT COLUMN SCHEMA
# ─────────────────────────────────────────────────────────────────────
#  Every row in master_thesis_labels.csv will have exactly these
#  columns, in this order.
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UnifiedSchema:
    """Column names for the master CSV output."""

    dataset: str        = "Dataset"            # "CASME_II" | "CASME2_Squared"
    subject_id: str     = "Subject_ID"         # int (original numbering)
    video_id: str       = "Video_ID"           # str (EP02_01f | anger1_1)
    onset_frame: str    = "Onset_Frame"        # int
    apex_frame: str     = "Apex_Frame"         # int | NaN
    offset_frame: str   = "Offset_Frame"       # int
    raw_emotion: str    = "Raw_Emotion"        # str (original label)
    unified_emotion: str = "Unified_Emotion"   # str (Positive/Negative/Surprise/Others)
    action_units: str   = "Action_Units"       # str
    expression_type: str = "Expression_Type"   # str (micro-expression / macro-expression / N/A)
    sequence_length: str = "Sequence_Length"    # int (Offset − Onset + 1)
    fps: str            = "FPS"                # int (200 | 30)
    frames_dir: str     = "Frames_Directory"   # str (absolute path to image folder)
    frames_exist: str   = "Frames_Exist"       # bool (whether the folder exists on disk)

    # ── Optical-flow processing tracker (for Step 2) ──
    of_processed: str   = "OF_Processed"       # bool (False initially)

    def ordered_columns(self) -> List[str]:
        """Return column names in canonical order for CSV output."""
        return [
            self.dataset,
            self.subject_id,
            self.video_id,
            self.onset_frame,
            self.apex_frame,
            self.offset_frame,
            self.raw_emotion,
            self.unified_emotion,
            self.action_units,
            self.expression_type,
            self.sequence_length,
            self.fps,
            self.frames_dir,
            self.frames_exist,
            self.of_processed,
        ]


# ─────────────────────────────────────────────────────────────────────
# 6.  CONVENIENCE — Pre-instantiated config singletons
# ─────────────────────────────────────────────────────────────────────
CASME_II_CFG     = CASMEIIConfig()
CASME2_SQ_CFG    = CASME2SquaredConfig()
OUTPUT_CFG       = OutputConfig()
SCHEMA           = UnifiedSchema()
