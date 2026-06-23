"""
temporal_interpolator.py — TIM-Inspired Frame Sampling & Loading
================================================================

Implements ``TemporalInterpolator``, which solves the fundamental
problem of heterogeneous temporal length across datasets.

Problem Statement:
    Preprocessed EVM `.avi` videos have varying frame counts (e.g., 56 frames).
    The downstream Spatio-Temporal Transformer requires a fixed-length tensor input.

Solution (Temporal Interpolation Model — TIM):
    We sample exactly L=33 uniformly-spaced frame *indices*. This
    yields 33 frames, which produce 32 consecutive pairs for optical-
    flow computation — matching the required temporal dimension T=32.

    We read the entire video into RAM, buffer it, then compute indices
    from 0 to (total_frames - 1).

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 2 — Extraction
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _natural_sort_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _frame_index_from_name(name: str) -> Optional[int]:
    """Try to parse a frame index from a filename stem (e.g. img73.jpg -> 73)."""
    stem = Path(name).stem
    digits = re.findall(r"\d+", stem)
    if not digits:
        return None
    return int(digits[-1])

class TemporalInterpolator:
    """
    Generates L uniformly-spaced frame indices and loads/resizes them from `.avi`.

    Parameters
    ----------
    target_length : int
        Number of frames to sample.  Default is 33 (yielding 32 pairs
        for optical flow).
    spatial_size : Tuple[int, int]
        Target (height, width) for resized frames.  Default (224, 224).

    Notes
    -----
    This class is designed to be instantiated once and called many
    times — it carries no mutable state between calls.  It is safe
    to use from ``ProcessPoolExecutor`` worker processes.
    """

    def __init__(
        self,
        target_length: int = 33,
        spatial_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if target_length < 2:
            raise ValueError(
                f"target_length must be >= 2 (got {target_length}).  "
                f"Need at least 2 frames to form 1 optical-flow pair."
            )
        self._L = target_length
        self._H, self._W = spatial_size

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def compute_indices(
        self, onset: int, offset: int
    ) -> np.ndarray:
        """
        Compute L uniformly-spaced frame indices in [onset, offset].

        Uses ``numpy.linspace`` for maximally uniform spacing, then
        rounds to the nearest integer.  Duplicate indices *are*
        permitted when the clip is shorter than L (supersampling).

        Parameters
        ----------
        onset : int
            First frame number of the expression clip.
        offset : int
            Last frame number of the expression clip.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(L,)`` with frame indices.
        """
        if offset < onset:
            raise ValueError(
                f"offset ({offset}) must be >= onset ({onset})."
            )

        indices = np.linspace(onset, offset, num=self._L)
        indices = np.round(indices).astype(np.int64)
        return indices

    def load_frames(
        self,
        frames_dir: str,
        onset: int,
        offset: int,
        dataset: str,
    ) -> Optional[np.ndarray]:
        """
        Load L frames from a .avi file or a folder of jpg/png images.

        For `.avi` (pre-cropped magnified clips): reads the full file and
        uniformly samples L frames.

        For image folders (CASME-II Cropped): reads jpg/png in
        ``{root}/subXX/{Filename}/``, selects the onset–offset range using
        frame numbers parsed from filenames when possible, then samples L frames.
        """
        filepath = Path(frames_dir)
        if not filepath.exists():
            logger.error("Media path not found: %s", filepath)
            return None

        if filepath.is_dir():
            return self._load_from_image_folder(filepath, onset, offset)
        if filepath.is_file():
            return self._load_from_video(filepath)
        logger.error("Invalid media path: %s", filepath)
        return None

    def _load_from_video(self, filepath: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            logger.error("Failed to open video file: %s", filepath)
            return None

        all_frames: List[np.ndarray] = []
        cv2.setNumThreads(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(
                    gray_frame,
                    (self._W, self._H),
                    interpolation=cv2.INTER_LINEAR,
                )
                all_frames.append(resized_frame.astype(np.float32))
        finally:
            cap.release()

        if not all_frames:
            logger.error("Video file is empty: %s", filepath)
            return None

        all_frames_np = np.stack(all_frames, axis=0)
        indices = self.compute_indices(0, len(all_frames_np) - 1)
        clamped_indices = np.clip(indices, 0, len(all_frames_np) - 1)
        return all_frames_np[clamped_indices]

    def _load_from_image_folder(
        self,
        folder: Path,
        onset: int,
        offset: int,
    ) -> Optional[np.ndarray]:
        image_files = sorted(
            [
                p.name for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            ],
            key=_natural_sort_key,
        )
        if not image_files:
            logger.error("No images found in folder: %s", folder)
            return None

        indexed: List[tuple[int, str]] = []
        for name in image_files:
            idx = _frame_index_from_name(name)
            if idx is not None:
                indexed.append((idx, name))

        selected_files: List[str]
        if indexed and offset >= onset:
            by_idx = {i: n for i, n in indexed}
            clip_names = [by_idx[i] for i in range(onset, offset + 1) if i in by_idx]
            if len(clip_names) >= 2:
                selected_files = clip_names
            else:
                seq_len = offset - onset + 1
                if len(image_files) == seq_len:
                    selected_files = image_files
                elif len(image_files) > offset:
                    selected_files = image_files[max(0, onset - 1):offset]
                else:
                    selected_files = image_files
        else:
            seq_len = offset - onset + 1
            if len(image_files) == seq_len:
                selected_files = image_files
            elif len(image_files) > offset:
                selected_files = image_files[max(0, onset - 1):offset]
            else:
                selected_files = image_files

        frames: List[np.ndarray] = []
        cv2.setNumThreads(0)
        for name in selected_files:
            img = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Failed to read image: %s", folder / name)
                continue
            resized = cv2.resize(
                img, (self._W, self._H), interpolation=cv2.INTER_LINEAR
            )
            frames.append(resized.astype(np.float32))

        if len(frames) < 2:
            logger.error(
                "Need at least 2 frames in %s (got %d)", folder, len(frames)
            )
            return None

        stack = np.stack(frames, axis=0)
        indices = self.compute_indices(0, len(stack) - 1)
        clamped = np.clip(indices, 0, len(stack) - 1)
        return stack[clamped]
