"""
build_test_dataset.py — Create a small CASME-II test pack aligned with Excel rows.

Output folder (default): MER_TestDataset_5epoch/
  CASME2-coding-20140508.xlsx   (13-clip subset)
  Cropped/subXX/{Filename}/img{frame}.jpg   (images mode)
  Video/subXX/{Filename}.avi                  (avi mode)

Usage:
  python tools/build_test_dataset.py
  python tools/build_test_dataset.py --output E:/path/MER_TestDataset_5epoch
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pick_subset(df: pd.DataFrame, subjects: list[int]) -> pd.DataFrame:
    """Pick a small subject-balanced subset for quick 5-epoch GPU tests."""
    picked_rows = []
    seen: set[tuple] = set()

    def add_row(row) -> None:
        key = (int(row["Subject"]), str(row["Filename"]))
        if key not in seen:
            seen.add(key)
            picked_rows.append(row)

    for subject in subjects:
        subdf = df[df["Subject"] == subject]
        for emotion in ("happiness", "disgust", "surprise", "others", "repression"):
            rows = subdf[subdf["Estimated Emotion"] == emotion]
            if len(rows):
                add_row(rows.iloc[0])
        for _, row in subdf.head(3).iterrows():
            add_row(row)

    out = pd.DataFrame(picked_rows).drop_duplicates(subset=["Subject", "Filename"])
    return out.reset_index(drop=True)


def _make_frame(
    base: np.ndarray,
    rng: np.random.Generator,
    frame_idx: int,
    emotion: str,
) -> np.ndarray:
    noise = rng.integers(0, 25, size=base.shape, dtype=np.uint8)
    img = np.clip(base.astype(np.int16) + noise + (frame_idx % 7), 0, 255).astype(np.uint8)
    if emotion == "happiness":
        img = np.clip(img.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    elif emotion == "disgust":
        img = np.clip(img.astype(np.int16) - 6, 0, 255).astype(np.uint8)
    return img


def generate_clip_frames(
    row: pd.Series,
    image_size: tuple[int, int] = (128, 128),
) -> tuple[list[int], list[np.ndarray]]:
    """Build grayscale frames for onset..offset (images) and 0..offset (video)."""
    subject = int(row["Subject"])
    video_id = str(row["Filename"]).strip()
    onset = int(row["OnsetFrame"])
    offset = int(row["OffsetFrame"])
    emotion = str(row["Estimated Emotion"]).strip().lower()

    rng = np.random.default_rng(abs(hash((subject, video_id))) % (2**32))
    base = rng.integers(40, 180, size=image_size, dtype=np.uint8)

    video_end = max(offset, onset)
    frame_indices = list(range(0, video_end + 1))
    frames = [_make_frame(base, rng, idx, emotion) for idx in frame_indices]
    return frame_indices, frames


def write_synthetic_images(
    out_root: Path,
    row: pd.Series,
    image_size: tuple[int, int] = (128, 128),
) -> int:
    subject = int(row["Subject"])
    video_id = str(row["Filename"]).strip()
    onset = int(row["OnsetFrame"])
    offset = int(row["OffsetFrame"])

    clip_dir = out_root / "Cropped" / f"sub{subject:02d}" / video_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    _, all_frames = generate_clip_frames(row, image_size)
    count = 0
    for frame_idx in range(onset, offset + 1):
        if frame_idx >= len(all_frames):
            break
        path = clip_dir / f"img{frame_idx:04d}.jpg"
        cv2.imwrite(str(path), all_frames[frame_idx])
        count += 1
    return count


def write_synthetic_video(
    out_root: Path,
    row: pd.Series,
    fps: int = 200,
    image_size: tuple[int, int] = (128, 128),
) -> int:
    subject = int(row["Subject"])
    video_id = str(row["Filename"]).strip()

    video_dir = out_root / "Video" / f"sub{subject:02d}"
    video_dir.mkdir(parents=True, exist_ok=True)
    avi_path = video_dir / f"{video_id}.avi"

    frame_indices, frames = generate_clip_frames(row, image_size)
    if not frames:
        return 0

    h, w = frames[0].shape
    writer = cv2.VideoWriter(
        str(avi_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (w, h),
        isColor=False,
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(avi_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (w, h),
            isColor=False,
        )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {avi_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()
    return len(frames)


def build(output: Path, source_excel: Path, subjects: list[int]) -> None:
    if not source_excel.is_file():
        raise FileNotFoundError(f"Source Excel not found: {source_excel}")

    df = pd.read_excel(source_excel, engine="openpyxl")
    subset = pick_subset(df, subjects)

    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("Cropped", "Video"):
        path = output / subdir
        if path.exists():
            shutil.rmtree(path)

    excel_out = output / "CASME2-coding-20140508.xlsx"
    subset.to_excel(excel_out, index=False, engine="openpyxl")

    total_images = 0
    total_video_frames = 0
    for _, row in subset.iterrows():
        total_images += write_synthetic_images(output, row)
        total_video_frames += write_synthetic_video(output, row)

    cropped_root = output / "Cropped"
    video_root = output / "Video"

    readme = output / "README.txt"
    readme.write_text(
        "\n".join([
            "MER Test Dataset (5-epoch quick run)",
            "====================================",
            f"Clips: {len(subset)}",
            f"  Images: {total_images} jpg frames in Cropped/",
            f"  Video:  {len(subset)} avi files in Video/ ({total_video_frames} total frames)",
            "",
            "Shared Excel:",
            "  CASME2-coding-20140508.xlsx",
            "",
            "--- Option A: image folders (CASME-II Cropped layout) ---",
            f"  Excel:      {excel_out}",
            f"  Media root: {cropped_root}",
            "  Media type: images",
            "",
            "--- Option B: avi clips ---",
            f"  Excel:      {excel_out}",
            f"  Media root: {video_root}",
            "  Media type: avi",
            "",
            "Then: Preprocess -> GPU grouped (epochs=5)",
            "",
            "CLI Step 1 (images):",
            f'  python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only '
            f'--casme2_excel "{excel_out}" '
            f'--casme2_frames_root "{cropped_root}" '
            f'--casme2_media_mode images',
            "",
            "CLI Step 1 (avi):",
            f'  python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only '
            f'--casme2_excel "{excel_out}" '
            f'--casme2_frames_root "{video_root}" '
            f'--casme2_media_mode avi',
        ]),
        encoding="utf-8",
    )

    print(f"Created test dataset: {output}")
    print(f"  Excel rows:   {len(subset)}")
    print(f"  Images:       {total_images}")
    print(f"  Video clips:  {len(subset)}  ({total_video_frames} frames)")
    print(f"  Excel:        {excel_out}")
    print(f"  Cropped:      {cropped_root}")
    print(f"  Video:        {video_root}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build small CASME-II test dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root.parent / "MER_TestDataset_5epoch",
    )
    parser.add_argument(
        "--source_excel",
        type=Path,
        default=project_root.parent / "CASME Ⅱ" / "CASME2-coding-20140508.xlsx",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    build(args.output.resolve(), args.source_excel.resolve(), args.subjects)


if __name__ == "__main__":
    main()
