"""
smoke_step1_cpu.py
==================

Runs a CASME-II-only smoke test for Stage1 Step1 by creating a minimal
synthetic CASME-II Excel file and validating master CSV generation.

Does NOT leave poisoned Processed_Data checkpoints behind — all Step 1
artefacts created for the smoke run are restored or removed in ``finally``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _backup(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    if path.is_dir():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(path, backup)
    else:
        shutil.copy2(path, backup)
    return backup


def _restore(path: Path, backup: Path | None) -> None:
    if backup is None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()
    if backup.is_dir():
        shutil.copytree(backup, path)
    else:
        shutil.move(str(backup), str(path))


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    processed_root = project_root / "Processed_Data"

    dataset_dir = project_root / "DATASETS" / "CASME II"
    excel_path = dataset_dir / "CASME2-coding-20140508.xlsx"
    backup_excel = dataset_dir / "CASME2-coding-20140508.xlsx.backup_smoke"

    frames_root = project_root / "Processed_Data" / "Raw_Videos_Magnified" / "CASME2" / "sub01"
    frame_video = frames_root / "clip_1.avi"
    backup_video = frames_root / "clip_1.avi.backup_smoke"

    master_csv = processed_root / "master_thesis_labels.csv"
    checkpoint = processed_root / "checkpoints" / "processed_state.json"
    intermediates = processed_root / "intermediates"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    had_excel = excel_path.exists()
    had_video = frame_video.exists()
    backup_master = _backup(master_csv, ".backup_smoke")
    backup_ckpt = _backup(checkpoint, ".backup_smoke")
    backup_inter = _backup(intermediates, ".backup_smoke")

    if had_excel:
        shutil.copy2(excel_path, backup_excel)
    if had_video:
        shutil.copy2(frame_video, backup_video)

    try:
        casme_df = pd.DataFrame(
            [
                {
                    "Subject": 1,
                    "Filename": "clip_1",
                    "OnsetFrame": 1,
                    "ApexFrame": 15,
                    "OffsetFrame": 30,
                    "Action Units": "AU12",
                    "Estimated Emotion": "happiness",
                }
            ]
        )
        casme_df.to_excel(excel_path, index=False)
        frame_video.write_bytes(b"SMOKE")

        cmd = [
            sys.executable,
            str(project_root / "Stage1_DataPipeline" / "main_step1.py"),
            "--dataset_mode",
            "casme2_only",
            "--force",
        ]
        completed = subprocess.run(cmd, cwd=project_root, check=False)
        if completed.returncode != 0:
            print("Stage1 Step1 smoke run failed.")
            return completed.returncode

        csv_path = processed_root / "master_thesis_labels.csv"
        if not csv_path.exists():
            print(f"master_thesis_labels.csv not found: {csv_path}")
            return 2

        out_df = pd.read_csv(csv_path)
        if out_df.empty:
            print("master_thesis_labels.csv is empty.")
            return 3
        if not (out_df["Dataset"] == "CASME_II").all():
            print("Unexpected non-CASME_II dataset rows found.")
            return 4

        print(f"Stage1 Step1 smoke test passed. CSV: {csv_path}")
        return 0
    finally:
        if had_excel and backup_excel.exists():
            shutil.move(backup_excel, excel_path)
        elif (not had_excel) and excel_path.exists():
            excel_path.unlink()
        if backup_excel.exists():
            backup_excel.unlink()

        if had_video and backup_video.exists():
            shutil.move(backup_video, frame_video)
        elif (not had_video) and frame_video.exists():
            frame_video.unlink()
        if backup_video.exists():
            backup_video.unlink()

        _restore(master_csv, backup_master)
        _restore(checkpoint, backup_ckpt)
        _restore(intermediates, backup_inter)
        for stale in (backup_master, backup_ckpt, backup_inter):
            if stale is not None and stale.exists():
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
