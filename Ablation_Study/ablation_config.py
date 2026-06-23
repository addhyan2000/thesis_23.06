"""
ablation_config.py — Experiment Configuration for the Stage 1 + Stage 2 Ablation
=================================================================================

This module centralises *all* tunable knobs for the ablation study so that the
model code, the dataset code, and the training loop never hard-code anything.

It defines two dataclasses:

    1. ``AblationConfig``
       The four scientific toggles that define a single experiment cell in the
       ablation matrix:

           A. use_evm          → Stage 1 preprocessing (motion magnification)
           B. use_simam        → Stage 2 parameter-free spatial attention
           C. use_cnn          → Stage 2 3D-CNN spatial feature extractor
           D. use_transformer  → Stage 2 SLSTT temporal sequencer

       NOTE on the scope of each toggle:
           • ``use_evm`` is a *data-level* switch. EVM is applied offline in
             Stage 1, producing a different set of ``.npy`` motion tensors.
             Toggling it therefore selects between two tensor directories
             (magnified vs. raw); it does NOT change the network graph.
           • ``use_simam``, ``use_cnn``, ``use_transformer`` are *model-level*
             switches that conditionally add/remove modules from the forward
             pass (see ``models.py``).

    2. ``ExperimentConfig``
       Dataset paths, the two tensor directories (EVM vs. raw), the emotion
       class map, optimisation hyper-parameters, and output locations. Every
       value here is explicit and overridable — there are NO hidden multi-
       dataset assumptions, subject-range constants, or class-map overrides.

The canonical 8-cell ``ABLATION_MATRIX`` (Phases I–IV from the thesis brief)
is constructed at the bottom of this file.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal


LabelMode = Literal["grouped", "individual"]

GROUPED_EMOTION_MAP: Dict[str, int] = {
    "Negative": 0,
    "Positive": 1,
    "Surprise": 2,
}

INDIVIDUAL_EMOTION_MAP: Dict[str, int] = {
    "happiness": 0,
    "disgust": 1,
    "sadness": 2,
    "fear": 3,
    "repression": 4,
    "surprise": 5,
}


def build_emotion_map(label_mode: LabelMode) -> Dict[str, int]:
    """Return the class map for grouped (3-class) or individual emotion labels."""
    if label_mode == "individual":
        return dict(INDIVIDUAL_EMOTION_MAP)
    return dict(GROUPED_EMOTION_MAP)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PROJECT ROOT — everything is derived from this single anchor.
# ─────────────────────────────────────────────────────────────────────────────
# Ablation_Study/ lives directly under the repository root, so the parent of
# this file's directory is the project root (same convention as Stage 1/3).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# 2.  THE FOUR SCIENTIFIC TOGGLES — one ablation cell.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AblationConfig:
    """
    A single experiment cell in the 2x2x2x2 ablation matrix.

    Each instance fully describes which components are active for one training
    run. The ``name`` and ``phase`` fields are purely for bookkeeping so that
    results are written to clearly labelled, unique files.

    Parameters
    ----------
    name : str
        Human-readable unique identifier, e.g. ``"config_8_proposed_unified"``.
        Used to name log files, checkpoints, and metric dumps.
    phase : str
        Thesis phase grouping (``"I"``, ``"II"``, ``"III"``, ``"IV"``).
    use_evm : bool
        Variable A — Stage 1 Eulerian Video Magnification.
        Selects the EVM-magnified tensor directory when True, raw otherwise.
    use_simam : bool
        Variable B — Stage 2 SimAM spatial attention (only meaningful when
        ``use_cnn`` is True, since SimAM rescales CNN feature maps).
    use_cnn : bool
        Variable C — Stage 2 three-stream 3D-CNN spatial feature extractor.
        When False, the model falls back to raw flattened spatial patches.
    use_transformer : bool
        Variable D — Stage 2 SLSTT Transformer temporal encoder.
        When False, the model falls back to simple mean/max temporal pooling.
    """

    name: str
    phase: str
    use_evm: bool
    use_simam: bool
    use_cnn: bool
    use_transformer: bool

    # ── Pretty, log-friendly description of the active components ──
    def describe(self) -> str:
        """Return a compact ``[A|B|C|D]`` style component summary string."""

        def tag(flag: bool, label: str) -> str:
            return f"+{label}" if flag else f"-{label}"

        return " ".join([
            tag(self.use_evm, "EVM"),
            tag(self.use_simam, "SimAM"),
            tag(self.use_cnn, "CNN3D"),
            tag(self.use_transformer, "SLSTT"),
        ])

    @property
    def folder_name(self) -> str:
        def tag(flag: bool, label: str) -> str:
            return f"WITH_{label}" if flag else f"no_{label}"
        return f"{self.name}__{tag(self.use_evm, 'evm')}__{tag(self.use_simam, 'simam')}__{tag(self.use_cnn, '3dcnn')}__{tag(self.use_transformer, 'transformer')}"

    # ── Validity guard: SimAM has no meaning without a CNN feature map ──
    def is_valid(self) -> bool:
        """
        Return True if the toggle combination is architecturally meaningful.

        SimAM rescales 3D-CNN feature maps; with the CNN disabled there is no
        feature map to attend over, so ``use_simam=True, use_cnn=False`` is
        considered degenerate. The 8 thesis configs never hit this case, but
        the orchestrator checks it defensively.
        """
        if self.use_simam and not self.use_cnn:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GLOBAL EXPERIMENT CONFIGURATION — paths, classes, hyper-parameters.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """
    All non-toggle settings shared across every ablation cell.

    This is intentionally a *mutable* dataclass (not frozen) so a thin CLI in
    ``run_ablation_experiments.py`` can override any field at launch time
    without editing source. Nothing here is dataset-specific beyond the values
    you explicitly pass — there are no hidden subject ranges or class overrides.

    Dataset generalisation
    -----------------------
    ``dataset_filter`` restricts the master CSV to a single dataset tag (e.g.
    ``"CASME_II"``). Set it to ``None`` to use every row in the CSV. This is the
    one knob that abstracts away multi-dataset unification: the rest of the code
    never assumes more than one dataset.
    """

    # ── Data sources ─────────────────────────────────────────────────────────
    csv_path: Path = PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"

    # Variable A is data-level: two precomputed tensor sets must exist on disk.
    #   • tensor_dir_evm : optical flow/strain computed on EVM-MAGNIFIED frames
    #   • tensor_dir_raw : optical flow/strain computed on RAW (non-magnified) frames
    # Generate each by running the Stage 1 Step-2 pipeline once with EVM on and
    # once with EVM off, writing to these two directories respectively.
    tensor_dir_evm: Path = PROJECT_ROOT / "Processed_Data" / "tensors"
    tensor_dir_raw: Path = PROJECT_ROOT / "Processed_Data" / "tensors_raw"

    # ── Dataset generalisation knobs ──────────────────────────────────────────
    dataset_filter: str | None = "CASME_II"        # single-dataset focus
    expression_filter: str | None = "micro-expression"

    # ── Class definition (config-driven; NO hardcoded class overrides) ────────
    # Default: the 3 thesis classes. "Others" is intentionally excluded by not
    # listing it here, so num_classes is derived from this map's size.
    # Switch with --label_mode grouped|individual on the CLI.
    label_mode: LabelMode = "grouped"
    emotion_map: Dict[str, int] = field(default_factory=lambda: dict(GROUPED_EMOTION_MAP))

    # ── Spatiotemporal tensor geometry (kept fixed per the thesis brief) ──────
    in_channels: int = 3            # u-flow, v-flow, optical strain
    sequence_length: int = 32       # interpolated temporal frames (T)
    spatial_size: int = 224         # H == W

    # ── Backbone / transformer dimensions (kept intact) ───────────────────────
    cnn_mid_channels: int = 16
    cnn_out_channels: int = 32      # ×3 streams = 96 = d_model
    cnn_dropout: float = 0.3
    simam_lambda: float = 1e-4
    d_model: int = 96               # transformer width; also raw-patch proj dim
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_ff: int = 256
    transformer_dropout: float = 0.1
    pool_strategy: Literal["mean", "cls"] = "mean"

    # When the 3D-CNN is OFF, raw frames are average-pooled to this small
    # spatial grid then flattened into per-frame "patch" vectors.
    raw_patch_grid: int = 4         # → 3 * 4 * 4 = 48-dim, projected to d_model
    # When the Transformer is OFF, this pooling collapses the time axis.
    temporal_pool: Literal["mean", "max"] = "mean"
    classifier_dropout: float = 0.3

    # ── Loss (decoupled; no SupCon, no XBM, no identity/GRL) ──────────────────
    loss_type: Literal["focal", "cross_entropy"] = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    use_class_weights: bool = True

    # ── Optimisation ──────────────────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    use_amp: bool = True

    # ── Validation protocol (strict subject-disjoint) ─────────────────────────
    # "holdout"  → single subject-disjoint train/val split (fast; default)
    # "loso"     → full leave-one-subject-out cross-validation (slow; thorough)
    validation_protocol: Literal["holdout", "loso"] = "holdout"
    val_fraction: float = 0.2
    seed: int = 42

    # ── Output locations ──────────────────────────────────────────────────────
    output_root: Path = PROJECT_ROOT / "Ablation_Study" / "results"
    log_dir: Path = PROJECT_ROOT / "Ablation_Study" / "logs"

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def num_classes(self) -> int:
        """Number of emotion classes — derived from the emotion map size."""
        return len(self.emotion_map)

    @property
    def class_names(self) -> List[str]:
        """Class names ordered by their integer label (for confusion matrices)."""
        return [name for name, _ in sorted(self.emotion_map.items(), key=lambda kv: kv[1])]

    def tensor_dir_for(self, use_evm: bool) -> Path:
        """Return the tensor directory matching the EVM toggle (Variable A)."""
        return self.tensor_dir_evm if use_evm else self.tensor_dir_raw


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THE 8-CELL ABLATION MATRIX (Phases I–IV from the thesis brief).
# ─────────────────────────────────────────────────────────────────────────────
# Each row is (name, phase, EVM, SimAM, CNN, Transformer).
_ORIGINAL_MATRIX = [
    AblationConfig("config_1_pure_base",        "I",   False, False, False, False),
    AblationConfig("config_2_temporal_only",    "I",   False, False, False, True),
    AblationConfig("config_3_spatial_only",     "I",   False, False, True,  False),
    AblationConfig("config_4_motion_amp_base",  "II",  True,  False, False, False),
    AblationConfig("config_5_attention_base",   "II",  False, True,  True,  False),
    AblationConfig("config_6_full_stage2_noevm","III", False, True,  True,  True),
    AblationConfig("config_7_full_no_attention","III", True,  False, True,  True),
    AblationConfig("config_8_proposed_unified", "IV",  True,  True,  True,  True),
]

_known_configs = {(c.use_evm, c.use_simam, c.use_cnn, c.use_transformer): c for c in _ORIGINAL_MATRIX}

import itertools

ABLATION_MATRIX: List[AblationConfig] = []
_config_idx = 9
for evm, simam, cnn, trans in itertools.product([False, True], repeat=4):
    key = (evm, simam, cnn, trans)
    if key in _known_configs:
        ABLATION_MATRIX.append(_known_configs[key])
    else:
        name = f"config_{_config_idx}_permutation"
        _config_idx += 1
        ABLATION_MATRIX.append(AblationConfig(name, "Other", evm, simam, cnn, trans))


def get_ablation_matrix() -> List[AblationConfig]:
    """Return a fresh copy of the canonical 8-cell ablation matrix."""
    return list(ABLATION_MATRIX)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone sanity print
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("  Ablation Matrix — 8 configurations across 4 phases")
    print("=" * 78)
    header = f"{'#':<3}{'name':<30}{'phase':<7}{'components':<32}{'valid'}"
    print(header)
    print("-" * 78)
    for i, cfg in enumerate(ABLATION_MATRIX, 1):
        print(f"{i:<3}{cfg.name:<30}{cfg.phase:<7}{cfg.describe():<32}{cfg.is_valid()}")
    print("-" * 78)

    exp = ExperimentConfig()
    print(f"\nnum_classes   : {exp.num_classes}")
    print(f"class_names   : {exp.class_names}")
    print(f"EVM tensors   : {exp.tensor_dir_for(True)}")
    print(f"RAW tensors   : {exp.tensor_dir_for(False)}")
