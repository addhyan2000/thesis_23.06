"""
dataset.py — Generalised Single-Dataset Loader for the Ablation Study
======================================================================

A clean PyTorch ``Dataset`` that reads the Stage 1 ``master_thesis_labels.csv``
and the precomputed ``[3, 32, 224, 224]`` motion tensors. It is deliberately
stripped of the Stage 3/4 complexity:

    • NO identity/subject label returned for the network (subject info is kept
      ONLY for building subject-disjoint validation splits).
    • NO joint macro+micro validation logic.
    • NO hardcoded subject ranges, dataset unification, or class-map overrides.

Everything that used to be implicit is now an explicit constructor argument:

    • ``tensor_dir``        — which precomputed tensor set to read (EVM or raw).
    • ``dataset_filter``    — restrict to ONE dataset tag (e.g. "CASME_II"),
                              or None to use all rows.
    • ``expression_filter`` — restrict to "micro-expression" etc., or None.
    • ``emotion_map``       — the class definition; ``num_classes`` is derived
                              from its size. Emotions absent from the map (e.g.
                              "Others") are simply skipped.
    • ``sequence_length``   — temporal length T the model expects (default 32).

Each sample returns ``(tensor, emotion_label)``. The raw subject id is stored
internally so ``subject_disjoint_split`` / ``loso_folds`` can guarantee that no
subject appears in both train and validation.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Column names from the Stage 1 unified schema (kept here so this module has
# no import-time dependency on Stage 1 source).
COL_DATASET = "Dataset"
COL_SUBJECT = "Subject_ID"
COL_VIDEO = "Video_ID"
COL_UNIFIED_EMOTION = "Unified_Emotion"
COL_RAW_EMOTION = "Raw_Emotion"
COL_EXPRESSION_TYPE = "Expression_Type"


class MERAblationDataset(Dataset):
    """
    Generalised micro-expression dataset for the ablation study.

    Parameters
    ----------
    csv_path : Path
        Path to ``master_thesis_labels.csv``.
    tensor_dir : Path
        Directory of ``.npy`` tensors. Choose the EVM or raw directory upstream
        depending on Variable A — this class does not know about EVM.
    emotion_map : dict[str, int]
        Unified-emotion-string or raw-emotion-string → integer-label. Defines the
        class set; rows with emotions not in this map are skipped.
    label_mode : str
        ``"grouped"`` reads ``Unified_Emotion``; ``"individual"`` reads
        ``Raw_Emotion`` (lower-cased).
    dataset_filter : str or None
        If set, keep only rows where ``Dataset == dataset_filter`` (single-dataset
        focus, e.g. "CASME_II"). None keeps all rows.
    expression_filter : str or None
        If set, keep only rows where ``Expression_Type == expression_filter``.
    sequence_length : int
        Temporal length T expected by the model (default 32). Tensors longer than
        T are cropped (centre for eval, random when ``augment=True``).
    augment : bool
        Enable light train-time augmentation (random temporal crop + horizontal
        flip with u-flow sign negation).
    logger : logging.Logger or None
        Optional logger; a module logger is used if None.
    """

    def __init__(
        self,
        csv_path: Path,
        tensor_dir: Path,
        emotion_map: Dict[str, int],
        dataset_filter: Optional[str] = None,
        expression_filter: Optional[str] = None,
        label_mode: str = "grouped",
        sequence_length: int = 32,
        augment: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        self._log = logger or logging.getLogger("MERAblationDataset")

        self._csv_path = Path(csv_path)
        self._tensor_dir = Path(tensor_dir)
        self.emotion_map = dict(emotion_map)
        self._label_mode = label_mode
        self._dataset_filter = dataset_filter
        self._expression_filter = expression_filter
        self._sequence_length = sequence_length
        self._augment = augment

        self.samples: List[dict] = []
        self.subject_map: Dict[int, int] = {}
        self._build_samples()

        self.num_classes = len(self.emotion_map)
        self.num_subjects = len(self.subject_map)

        self._log.info(
            "Dataset ready: %d samples | %d classes | %d subjects | tensors=%s",
            len(self.samples), self.num_classes, self.num_subjects, self._tensor_dir,
        )

    # ────────────────────────────────────────────────────────────────────────
    #  Construction
    # ────────────────────────────────────────────────────────────────────────
    def _build_samples(self) -> None:
        """Read CSV → filter → map labels → verify tensors exist on disk."""
        if not self._csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self._csv_path}")
        if not self._tensor_dir.exists():
            raise FileNotFoundError(
                f"Tensor directory not found: {self._tensor_dir}. "
                f"Generate it with the Stage 1 Step-2 pipeline before running."
            )

        df = pd.read_csv(self._csv_path, encoding="utf-8-sig")
        self._log.info("Loaded CSV: %d rows", len(df))

        # ── Single-dataset focus (abstracts away multi-dataset unification) ──
        if self._dataset_filter is not None:
            df = df[df[COL_DATASET] == self._dataset_filter].copy()
            self._log.info("Dataset filter '%s' -> %d rows", self._dataset_filter, len(df))

        # ── Expression-type focus (micro vs macro) ──
        if self._expression_filter is not None:
            df = df[df[COL_EXPRESSION_TYPE] == self._expression_filter].copy()
            self._log.info("Expression filter '%s' -> %d rows", self._expression_filter, len(df))

        df.reset_index(drop=True, inplace=True)
        if len(df) == 0:
            self._log.warning("No rows remain after filtering — empty dataset!")
            return

        # ── Contiguous subject mapping (raw id → 0..N-1), for split logic only ──
        unique_subjects = sorted(df[COL_SUBJECT].unique().tolist())
        self.subject_map = {raw: idx for idx, raw in enumerate(unique_subjects)}

        matched = missing = unknown = 0
        for _, row in df.iterrows():
            dataset_tag = str(row[COL_DATASET])
            video_id = str(row[COL_VIDEO])
            if self._label_mode == "individual":
                if COL_RAW_EMOTION not in row.index:
                    unknown += 1
                    continue
                emotion = str(row[COL_RAW_EMOTION]).strip().lower()
            else:
                emotion = str(row[COL_UNIFIED_EMOTION]).strip()
            subject_id = int(row[COL_SUBJECT])

            tensor_path = self._tensor_dir / f"{dataset_tag}_{video_id}.npy"
            if not tensor_path.exists():
                missing += 1
                continue
            if emotion not in self.emotion_map:
                unknown += 1
                continue

            self.samples.append({
                "tensor_path": tensor_path,
                "label": self.emotion_map[emotion],
                "raw_subject_id": subject_id,
                "emotion": emotion,
                "video_id": video_id,
            })
            matched += 1

        self._log.info(
            "Tensor matching: matched=%d missing=%d unknown_emotion=%d",
            matched, missing, unknown,
        )

    # ────────────────────────────────────────────────────────────────────────
    #  PyTorch interface
    # ────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return ``(tensor, label)``.

        ``tensor`` is float32 of shape ``[C, sequence_length, H, W]``. Subject id
        is deliberately NOT returned — the network never sees identity.
        """
        sample = self.samples[idx]
        tensor = torch.from_numpy(np.load(str(sample["tensor_path"]))).float()

        # ── Temporal length normalisation to exactly sequence_length ──
        T = tensor.size(1)
        L = self._sequence_length
        if T > L:
            start = (torch.randint(0, T - L + 1, (1,)).item()
                     if self._augment else (T - L) // 2)
            tensor = tensor[:, start:start + L, :, :]
        elif T < L:
            # Pad by repeating the last frame so short clips still run.
            pad = tensor[:, -1:, :, :].repeat(1, L - T, 1, 1)
            tensor = torch.cat([tensor, pad], dim=1)

        # ── Light augmentation (train only) ──
        if self._augment and random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[3])   # horizontal flip
            tensor[0] *= -1                         # negate u-flow direction

        return tensor, sample["label"]

    # ────────────────────────────────────────────────────────────────────────
    #  Class-imbalance helper
    # ────────────────────────────────────────────────────────────────────────
    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights of shape ``[num_classes]``."""
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for s in self.samples:
            counts[s["label"]] += 1.0
        total = counts.sum()
        return total / (self.num_classes * counts.clamp(min=1.0))

    def get_label_distribution(self) -> Dict[int, int]:
        """Return ``{label: count}`` for quick inspection/logging."""
        dist: Dict[int, int] = {}
        for s in self.samples:
            dist[s["label"]] = dist.get(s["label"], 0) + 1
        return dist

    # ────────────────────────────────────────────────────────────────────────
    #  Subject-disjoint split utilities (strict — no identity leakage)
    # ────────────────────────────────────────────────────────────────────────
    def get_unique_subject_ids(self) -> List[int]:
        """Sorted unique raw subject ids present in the dataset."""
        return sorted({s["raw_subject_id"] for s in self.samples})

    def get_indices_for_subjects(self, subject_ids: List[int]) -> List[int]:
        """All sample indices whose raw subject id is in ``subject_ids``."""
        wanted = set(subject_ids)
        return [i for i, s in enumerate(self.samples) if s["raw_subject_id"] in wanted]

    def subject_disjoint_split(
        self,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        Single subject-disjoint train/val split.

        Holds out ``ceil(val_fraction * num_subjects)`` whole subjects for
        validation so that no subject appears in both partitions.

        Returns ``(train_indices, val_indices)``.
        """
        subjects = self.get_unique_subject_ids()
        rng = random.Random(seed)
        shuffled = list(subjects)
        rng.shuffle(shuffled)

        n_val = max(1, int(round(val_fraction * len(subjects))))
        val_subjects = shuffled[:n_val]
        train_subjects = shuffled[n_val:]

        train_idx = self.get_indices_for_subjects(train_subjects)
        val_idx = self.get_indices_for_subjects(val_subjects)

        assert not (set(train_subjects) & set(val_subjects)), "Subject overlap!"
        self._log.info(
            "Subject-disjoint split (seed=%d): train=%d subj/%d samp | val=%d subj/%d samp | val_subjects=%s",
            seed, len(train_subjects), len(train_idx),
            len(val_subjects), len(val_idx), sorted(val_subjects),
        )
        return train_idx, val_idx

    def loso_folds(self) -> List[Tuple[int, List[int], List[int]]]:
        """
        Leave-One-Subject-Out folds.

        Returns a list of ``(held_out_subject_id, train_indices, val_indices)``
        — one fold per subject. Use when ``validation_protocol == "loso"``.
        """
        folds = []
        for sid in self.get_unique_subject_ids():
            val_idx = self.get_indices_for_subjects([sid])
            train_idx = [i for i in range(len(self.samples)) if i not in set(val_idx)]
            folds.append((sid, train_idx, val_idx))
        return folds


# ─────────────────────────────────────────────────────────────────────────────
# Standalone verification
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from ablation_config import ExperimentConfig  # type: ignore

    exp = ExperimentConfig()
    try:
        ds = MERAblationDataset(
            csv_path=exp.csv_path,
            tensor_dir=exp.tensor_dir_for(use_evm=True),
            emotion_map=exp.emotion_map,
            dataset_filter=exp.dataset_filter,
            expression_filter=exp.expression_filter,
            sequence_length=exp.sequence_length,
        )
    except FileNotFoundError as err:
        print(f"[skip] {err}")
        sys.exit(0)

    print(f"len={len(ds)} num_classes={ds.num_classes} num_subjects={ds.num_subjects}")
    print(f"label distribution: {ds.get_label_distribution()}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"sample0 tensor={tuple(x.shape)} label={y}")
        tr, va = ds.subject_disjoint_split(exp.val_fraction, exp.seed)
        print(f"split: train={len(tr)} val={len(va)}")
