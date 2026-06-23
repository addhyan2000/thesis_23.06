"""
Ablation_Study — Isolated Stage 1 + Stage 2 Ablation for MER
=============================================================

A self-contained package that strips the Stage 3/4 domain-adversarial and
transfer-learning machinery (GRL, identity head, SupCon, XBM) and reduces the
project to a clean, toggleable study of how each Stage 1/Stage 2 component
contributes to Micro-Expression Recognition.

Public modules
--------------
    ablation_config : AblationConfig, ExperimentConfig, ABLATION_MATRIX
    models          : AblationMERModel (+ build_model factory) and sub-modules
    dataset         : MERAblationDataset (generalised, single-dataset)
    losses          : FocalLoss, build_loss
    metrics         : MetricsComputer, ResultWriter, EvalResult
    trainer         : AblationTrainer
    run_ablation_experiments : orchestrating CLI over the 8-config matrix
"""

from .ablation_config import (
    ABLATION_MATRIX,
    AblationConfig,
    ExperimentConfig,
    get_ablation_matrix,
)
from .dataset import MERAblationDataset
from .losses import FocalLoss, build_loss
from .metrics import EvalResult, MetricsComputer, ResultWriter
from .models import AblationMERModel, build_model
from .trainer import AblationTrainer

__all__ = [
    "ABLATION_MATRIX",
    "AblationConfig",
    "ExperimentConfig",
    "get_ablation_matrix",
    "MERAblationDataset",
    "FocalLoss",
    "build_loss",
    "EvalResult",
    "MetricsComputer",
    "ResultWriter",
    "AblationMERModel",
    "build_model",
    "AblationTrainer",
]
