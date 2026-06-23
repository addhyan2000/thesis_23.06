"""
smoke_ablation_cpu.py
=====================

Creates a tiny synthetic CASME-II-like dataset and runs a CPU-only ablation
smoke test to verify the initial project wiring end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_data(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    tensor_dir_evm = root / "tensors"
    tensor_dir_raw = root / "tensors_raw"
    tensor_dir_evm.mkdir(parents=True, exist_ok=True)
    tensor_dir_raw.mkdir(parents=True, exist_ok=True)

    emotions = ["Negative", "Positive", "Surprise"]
    rows = []
    sample_idx = 0
    for subject_id in (1, 2, 3):
        for clip_idx in (1, 2, 3):
            video_id = f"clip_{subject_id}_{clip_idx}"
            emotion = emotions[sample_idx % len(emotions)]
            sample_idx += 1

            base = np.random.default_rng(seed=subject_id * 100 + clip_idx).normal(
                loc=0.0,
                scale=1.0,
                size=(3, 32, 64, 64),
            ).astype(np.float32)
            raw_tensor = base
            evm_tensor = (base * 1.05).astype(np.float32)

            np.save(tensor_dir_raw / f"CASME_II_{video_id}.npy", raw_tensor)
            np.save(tensor_dir_evm / f"CASME_II_{video_id}.npy", evm_tensor)

            rows.append(
                {
                    "Dataset": "CASME_II",
                    "Subject_ID": subject_id,
                    "Video_ID": video_id,
                    "Raw_Emotion": emotion.lower() if emotion != "Negative" else "disgust",
                    "Unified_Emotion": emotion,
                    "Expression_Type": "micro-expression",
                }
            )

    csv_path = root / "master_thesis_labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path, tensor_dir_evm, tensor_dir_raw


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    smoke_root = project_root / "tmp" / "smoke_ablation"
    results_root = smoke_root / "results"

    csv_path, tensor_dir_evm, tensor_dir_raw = build_synthetic_data(smoke_root)

    cmd = [
        sys.executable,
        str(project_root / "Ablation_Study" / "run_ablation_experiments.py"),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--csv_path",
        str(csv_path),
        "--tensor_dir_evm",
        str(tensor_dir_evm),
        "--tensor_dir_raw",
        str(tensor_dir_raw),
        "--dataset_filter",
        "CASME_II",
        "--expression_filter",
        "micro-expression",
        "--output_root",
        str(results_root),
        "--configs",
        "config_1_pure_base",
        "config_3_spatial_only",
        "config_7_full_no_attention",
        "config_8_proposed_unified",
    ]

    completed = subprocess.run(cmd, cwd=project_root, check=False)
    if completed.returncode != 0:
        print("Smoke run failed.")
        return completed.returncode

    summary_csv = results_root / "summary.csv"
    if not summary_csv.exists():
        print("Smoke run finished but summary.csv is missing.")
        return 2

    summary_df = pd.read_csv(summary_csv)
    expected = {
        "config_1_pure_base__no_evm__no_simam__no_3dcnn__no_transformer",
        "config_3_spatial_only__no_evm__no_simam__WITH_3dcnn__no_transformer",
        "config_7_full_no_attention__WITH_evm__no_simam__WITH_3dcnn__WITH_transformer",
        "config_8_proposed_unified__WITH_evm__WITH_simam__WITH_3dcnn__WITH_transformer",
    }
    got = set(summary_df["config_name"].tolist())
    missing = expected - got
    if missing:
        print(f"Smoke run incomplete. Missing configs in summary.csv: {sorted(missing)}")
        return 3

    print(f"Smoke run passed. Results: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
