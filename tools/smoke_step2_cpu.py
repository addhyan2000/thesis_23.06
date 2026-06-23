"""
smoke_step2_cpu.py
==================

Runs a small Stage1 Step2 smoke test by generating one synthetic .avi sample,
building a minimal master CSV, and verifying that one tensor file is produced.

Restores master CSV and Step 2 checkpoint state after the run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def write_synthetic_video(path: Path, num_frames: int = 40, size: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for t in range(num_frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            cv2.circle(frame, (10 + t % 40, 20 + (t // 2) % 40), 8, (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    processed_root = project_root / "Processed_Data"
    processed_root.mkdir(parents=True, exist_ok=True)
    csv_path = processed_root / "master_thesis_labels.csv"
    backup_csv = processed_root / "master_thesis_labels.csv.backup_smoke"
    step2_ckpt = processed_root / "checkpoints" / "step2_state.json"
    backup_ckpt = processed_root / "checkpoints" / "step2_state.json.backup_smoke"

    smoke_root = project_root / "tmp" / "smoke_step2"
    video_path = smoke_root / "videos" / "sub01" / "clip_1.avi"
    write_synthetic_video(video_path)

    row = {
        "Dataset": "CASME_II",
        "Subject_ID": 1,
        "Video_ID": "clip_1",
        "Onset_Frame": 1,
        "Apex_Frame": 20,
        "Offset_Frame": 40,
        "Raw_Emotion": "happiness",
        "Unified_Emotion": "Positive",
        "Action_Units": "N/A",
        "Expression_Type": "micro-expression",
        "Sequence_Length": 40,
        "FPS": 30,
        "Frames_Directory": str(video_path),
        "Frames_Exist": True,
        "OF_Processed": False,
    }
    df = pd.DataFrame([row])

    had_original_csv = csv_path.exists()
    if had_original_csv:
        shutil.copy2(csv_path, backup_csv)
    had_original_ckpt = step2_ckpt.exists()
    if had_original_ckpt:
        shutil.copy2(step2_ckpt, backup_ckpt)

    try:
        df.to_csv(csv_path, index=False, encoding="utf-8")

        cmd = [
            sys.executable,
            str(project_root / "Stage1_DataPipeline" / "main_step2.py"),
            "--force",
            "--max_workers",
            "1",
            "--output_subdir",
            "smoke_tensors",
            "--dataset_filter",
            "CASME_II",
            "--expression_filter",
            "micro-expression",
        ]
        completed = subprocess.run(cmd, cwd=project_root, check=False)
        if completed.returncode != 0:
            print("Stage1 Step2 smoke run failed.")
            return completed.returncode

        output_tensor = processed_root / "smoke_tensors" / "CASME_II_clip_1.npy"
        if not output_tensor.exists():
            print(f"Expected tensor not found: {output_tensor}")
            return 2

        tensor = np.load(output_tensor)
        if tensor.shape != (3, 32, 224, 224):
            print(f"Unexpected tensor shape: {tensor.shape}")
            return 3

        print(f"Stage1 Step2 smoke test passed. Tensor: {output_tensor}")
        return 0
    finally:
        if had_original_csv and backup_csv.exists():
            shutil.move(backup_csv, csv_path)
        elif (not had_original_csv) and csv_path.exists():
            csv_path.unlink()
        if backup_csv.exists():
            backup_csv.unlink()

        if had_original_ckpt and backup_ckpt.exists():
            shutil.move(backup_ckpt, step2_ckpt)
        elif (not had_original_ckpt) and step2_ckpt.exists():
            step2_ckpt.unlink()
        if backup_ckpt.exists():
            backup_ckpt.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
