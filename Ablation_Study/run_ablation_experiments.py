"""
run_ablation_experiments.py — Ablation Orchestrator (Stage 1 + Stage 2)
========================================================================

Drives the full 8-cell ablation matrix end-to-end:

    for each AblationConfig in the matrix:
        1. pick the tensor directory implied by Variable A (EVM on/off),
        2. build a single-dataset MERAblationDataset,
        3. make a STRICT subject-disjoint validation split (holdout or LOSO),
        4. build the conditional AblationMERModel for (SimAM, CNN, Transformer),
        5. train with a decoupled Focal/CE loss (no GRL/SupCon/XBM),
        6. evaluate the best checkpoint and write Accuracy / Macro-F1 /
           Confusion Matrix to unique per-config files.

Everything is configuration-driven via ``ExperimentConfig`` (see
``ablation_config.py``); a thin CLI exposes the most common overrides.

USAGE
-----
    # Run the entire matrix on CASME II micro-expressions:
    python Ablation_Study/run_ablation_experiments.py

    # Run a single configuration by name:
    python Ablation_Study/run_ablation_experiments.py --only config_8_proposed_unified

    # Quick smoke test (few epochs, tiny subset) to validate plumbing:
    python Ablation_Study/run_ablation_experiments.py --epochs 1 --max_samples 16

DATA REQUIREMENT FOR VARIABLE A (EVM)
-------------------------------------
This script needs TWO precomputed tensor sets:
    • exp.tensor_dir_evm — flow/strain on EVM-magnified frames (use_evm=True)
    • exp.tensor_dir_raw — flow/strain on raw frames           (use_evm=False)
Generate each by running the Stage 1 Step-2 pipeline once with EVM enabled and
once with it disabled. If a required directory is missing, that configuration is
skipped with a clear warning (the rest still run).

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ── Local imports (run as a script from anywhere) ──
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ablation_config import (  # noqa: E402
    AblationConfig,
    ExperimentConfig,
    build_emotion_map,
    get_ablation_matrix,
)
from dataset import MERAblationDataset  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import EvalResult, MetricsComputer, ResultWriter  # noqa: E402
from models import build_model  # noqa: E402
from trainer import AblationTrainer  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class AblationOrchestrator:
    """
    Runs the ablation matrix and persists per-configuration metrics.

    The orchestrator caches one dataset per (tensor_dir) so the EVM and raw
    variants are each loaded at most once, then reused across configurations.
    """

    def __init__(
        self,
        exp: ExperimentConfig,
        max_samples: Optional[int] = None,
        device_preference: str = "auto",
    ) -> None:
        self.exp = exp
        self.max_samples = max_samples
        self.device = self._resolve_device(device_preference)

        exp.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = self._make_logger(exp.log_dir / "ablation.log")
        self.writer = ResultWriter(exp.output_root, exp.class_names)

        # Dataset cache keyed by resolved tensor directory.
        self._dataset_cache: dict = {}

        self._log.info("=" * 78)
        self._log.info("  Ablation Orchestrator")
        self._log.info("  device=%s | label_mode=%s | classes=%s | dataset_filter=%s | expr_filter=%s",
                       self.device, exp.label_mode, exp.class_names, exp.dataset_filter, exp.expression_filter)
        self._log.info("  protocol=%s | epochs=%d | batch=%d | loss=%s",
                       exp.validation_protocol, exp.epochs, exp.batch_size, exp.loss_type)
        self._log.info("=" * 78)

    @staticmethod
    def _resolve_device(device_preference: str) -> torch.device:
        pref = device_preference.lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── logging ──────────────────────────────────────────────────────────────
    @staticmethod
    def _make_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("AblationOrchestrator")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        return logger

    # ── dataset (cached per tensor dir) ───────────────────────────────────────
    def _get_dataset(self, use_evm: bool) -> Optional[MERAblationDataset]:
        tensor_dir = self.exp.tensor_dir_for(use_evm)
        if tensor_dir in self._dataset_cache:
            return self._dataset_cache[tensor_dir]

        if not tensor_dir.exists():
            self._log.warning(
                "Tensor dir for use_evm=%s does not exist: %s — configs needing it will be skipped.",
                use_evm, tensor_dir,
            )
            self._dataset_cache[tensor_dir] = None
            return None

        ds = MERAblationDataset(
            csv_path=self.exp.csv_path,
            tensor_dir=tensor_dir,
            emotion_map=self.exp.emotion_map,
            dataset_filter=self.exp.dataset_filter,
            expression_filter=self.exp.expression_filter,
            label_mode=self.exp.label_mode,
            sequence_length=self.exp.sequence_length,
            augment=False,  # base dataset; augmentation toggled per-split below
            logger=self._log,
        )
        self._dataset_cache[tensor_dir] = ds
        return ds

    # ── loaders for one split ──────────────────────────────────────────────────
    def _make_loaders(
        self,
        dataset: MERAblationDataset,
        train_idx: List[int],
        val_idx: List[int],
    ) -> tuple[DataLoader, DataLoader]:
        # Optional subsetting for fast smoke tests.
        if self.max_samples is not None:
            train_idx = train_idx[: self.max_samples]
            val_idx = val_idx[: max(1, self.max_samples // 4)]

        # A separate augmented "view" of the same dataset for the train split.
        # Both instances read the same CSV/tensor_dir with identical filters, so
        # their internal ``samples`` ordering is deterministic and aligned — the
        # assertion below makes that contract explicit (indices must map 1:1).
        train_ds = MERAblationDataset(
            csv_path=self.exp.csv_path,
            tensor_dir=dataset._tensor_dir,
            emotion_map=self.exp.emotion_map,
            dataset_filter=self.exp.dataset_filter,
            expression_filter=self.exp.expression_filter,
            label_mode=self.exp.label_mode,
            sequence_length=self.exp.sequence_length,
            augment=True,
            logger=self._log,
        )
        assert len(train_ds) == len(dataset), (
            "Augmented train view and eval view disagree on sample count "
            f"({len(train_ds)} vs {len(dataset)}); index alignment would break."
        )
        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=self.exp.batch_size,
            shuffle=True, num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=self.exp.batch_size,
            shuffle=False, num_workers=0, drop_last=False,
        )
        return train_loader, val_loader

    # ── train + eval one (config, split) pair ─────────────────────────────────
    def _train_eval_split(
        self,
        ablation: AblationConfig,
        dataset: MERAblationDataset,
        train_idx: List[int],
        val_idx: List[int],
    ) -> EvalResult:
        set_seed(self.exp.seed)
        train_loader, val_loader = self._make_loaders(dataset, train_idx, val_idx)

        model = build_model(ablation, self.exp)
        self._log.info("  components=%s | trainable_params=%d",
                       model.active_components(), model.count_parameters()["trainable"])

        class_weights = dataset.get_class_weights().to(self.device)
        criterion = build_loss(self.exp, class_weights=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.exp.lr, weight_decay=self.exp.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.exp.epochs, eta_min=1e-7,
        )

        trainer = AblationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            num_epochs=self.exp.epochs,
            scheduler=scheduler,
            gradient_clip_norm=self.exp.gradient_clip_norm,
            use_amp=self.exp.use_amp,
            logger=self._log,
            select_metric="f1",
        )
        train_state = trainer.fit(train_loader, val_loader)
        trainer.load_best()

        _, y_true, y_pred = trainer.evaluate(val_loader)
        result = MetricsComputer.compute(y_true, y_pred, self.exp.num_classes)
        return result, train_state, model.describe_data_flow()

    # ── run one configuration (handles holdout vs LOSO) ────────────────────────
    def run_config(self, ablation: AblationConfig) -> Optional[EvalResult]:
        self._log.info("-" * 78)
        self._log.info("CONFIG %s [phase %s] | %s", ablation.name, ablation.phase, ablation.describe())

        if not ablation.is_valid():
            self._log.warning("  Skipping degenerate combo (SimAM without CNN).")
            return None

        dataset = self._get_dataset(ablation.use_evm)
        if dataset is None or len(dataset) == 0:
            self._log.warning("  Skipping — dataset unavailable/empty for use_evm=%s.", ablation.use_evm)
            return None

        if self.exp.validation_protocol == "loso":
            fold_results: List[EvalResult] = []
            train_state_final = None
            data_flow_final = ""
            for sid, train_idx, val_idx in dataset.loso_folds():
                self._log.info("  LOSO fold — held-out subject %s", sid)
                fold_res, train_state_final, data_flow_final = self._train_eval_split(ablation, dataset, train_idx, val_idx)
                fold_results.append(fold_res)
            result = MetricsComputer.average_results(fold_results, self.exp.num_classes)
        else:  # holdout
            train_idx, val_idx = dataset.subject_disjoint_split(
                self.exp.val_fraction, self.exp.seed,
            )
            result, train_state_final, data_flow_final = self._train_eval_split(ablation, dataset, train_idx, val_idx)

        self._log.info("  RESULT %s -> acc=%.4f | macroF1=%.4f",
                       ablation.name, result.accuracy, result.macro_f1)

        self.writer.save_config_result(
            config_name=ablation.folder_name,
            toggles={
                "use_evm": ablation.use_evm,
                "use_simam": ablation.use_simam,
                "use_cnn": ablation.use_cnn,
                "use_transformer": ablation.use_transformer,
            },
            result=result,
            train_state=train_state_final,
            data_flow=data_flow_final,
            extra={"phase": ablation.phase, "protocol": self.exp.validation_protocol, "label_mode": self.exp.label_mode},
        )
        return result

    # ── run the whole matrix (or a filtered subset) ────────────────────────────
    def run_matrix(
        self,
        only: Optional[str] = None,
        phase: Optional[str] = None,
        configs: Optional[List[str]] = None,
    ) -> int:
        matrix = get_ablation_matrix()
        if only is not None:
            matrix = [c for c in matrix if c.name == only]
        if phase is not None:
            matrix = [c for c in matrix if c.phase == phase]
        if configs:
            wanted = {name.strip() for name in configs}
            matrix = [c for c in matrix if c.name in wanted]

        if not matrix:
            self._log.error("No configurations matched the filter (only=%s phase=%s).", only, phase)
            return 1

        completed = 0
        for ablation in matrix:
            try:
                if self.run_config(ablation) is not None:
                    completed += 1
            except Exception as err:  # keep going so one failure doesn't kill the sweep
                self._log.exception("  CONFIG %s failed: %s", ablation.name, err)
            finally:
                # ── Clear GPU memory after each run to prevent accumulation ──
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._log.info("=" * 78)
        self._log.info("  Ablation sweep complete. Summary CSV: %s", self.writer.summary_csv)
        self._log.info("=" * 78)

        if completed == 0:
            self._log.error(
                "No configurations completed successfully (all skipped or failed)."
            )
            return 2
        if completed < len(matrix):
            self._log.warning(
                "Partial sweep: %d/%d configurations completed.",
                completed,
                len(matrix),
            )
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 + Stage 2 ablation sweep.")
    p.add_argument("--csv_path", type=Path, default=None)
    p.add_argument("--tensor_dir_evm", type=Path, default=None)
    p.add_argument("--tensor_dir_raw", type=Path, default=None)
    p.add_argument("--dataset_filter", type=str, default=None,
                   help="Single dataset tag, e.g. CASME_II. Pass 'all' to disable.")
    p.add_argument("--expression_filter", type=str, default=None,
                   help="e.g. micro-expression. Pass 'all' to disable.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss_type", choices=["focal", "cross_entropy"], default=None)
    p.add_argument("--protocol", choices=["holdout", "loso"], default=None)
    p.add_argument("--val_fraction", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--only", type=str, default=None, help="Run only this config name.")
    p.add_argument("--phase", type=str, default=None, help="Run only this phase (I/II/III/IV).")
    p.add_argument("--max_samples", type=int, default=None, help="Cap samples for a smoke test.")
    p.add_argument("--output_root", type=Path, default=None)
    p.add_argument("--label_mode", choices=["grouped", "individual"], default=None,
                   help="grouped=Positive/Negative/Surprise; individual=raw CASME-II emotions.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Run only these config names (space-separated).")
    return p.parse_args()


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Apply CLI overrides on top of the dataclass defaults."""
    exp = ExperimentConfig()
    if args.csv_path is not None:
        exp.csv_path = args.csv_path
    if args.tensor_dir_evm is not None:
        exp.tensor_dir_evm = args.tensor_dir_evm
    if args.tensor_dir_raw is not None:
        exp.tensor_dir_raw = args.tensor_dir_raw
    if args.dataset_filter is not None:
        exp.dataset_filter = None if args.dataset_filter.lower() == "all" else args.dataset_filter
    if args.expression_filter is not None:
        exp.expression_filter = None if args.expression_filter.lower() == "all" else args.expression_filter
    if args.epochs is not None:
        exp.epochs = args.epochs
    if args.batch_size is not None:
        exp.batch_size = args.batch_size
    if args.lr is not None:
        exp.lr = args.lr
    if args.loss_type is not None:
        exp.loss_type = args.loss_type
    if args.protocol is not None:
        exp.validation_protocol = args.protocol
    if args.val_fraction is not None:
        exp.val_fraction = args.val_fraction
    if args.seed is not None:
        exp.seed = args.seed
    if args.label_mode is not None:
        exp.label_mode = args.label_mode
        exp.emotion_map = build_emotion_map(args.label_mode)
    if args.output_root is not None:
        exp.output_root = args.output_root
    if args.device == "cpu":
        exp.use_amp = False
    elif args.device == "cuda":
        exp.use_amp = True
    return exp


def main() -> int:
    args = parse_args()
    exp = build_experiment_config(args)
    set_seed(exp.seed)
    orchestrator = AblationOrchestrator(
        exp,
        max_samples=args.max_samples,
        device_preference=args.device,
    )
    return orchestrator.run_matrix(only=args.only, phase=args.phase, configs=args.configs)


if __name__ == "__main__":
    raise SystemExit(main())
