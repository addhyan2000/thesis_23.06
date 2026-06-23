"""Quick check that the GPU project has the latest verification fixes."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    checks: list[tuple[str, Path, str]] = [
        (
            "run_gui.py has fresh preprocess + RESULT banner",
            root / "tools" / "run_gui.py",
            "metadata CSV (fresh)",
        ),
        (
            "main_step2.py has --force",
            root / "Stage1_DataPipeline" / "main_step2.py",
            "--force",
        ),
        (
            "main_step1.py auto-force on stale cache",
            root / "Stage1_DataPipeline" / "main_step1.py",
            "_should_auto_force",
        ),
        (
            "ablation fails when all configs skipped",
            root / "Ablation_Study" / "run_ablation_experiments.py",
            "completed == 0",
        ),
        (
            "smoke_step1 restores Processed_Data",
            root / "tools" / "smoke_step1_cpu.py",
            "backup_master",
        ),
    ]

    ok = True
    print("MER GPU deployment verification")
    print("Project:", root)
    print("-" * 60)
    for label, path, needle in checks:
        if not path.is_file():
            print(f"FAIL  {label}\n      missing: {path}")
            ok = False
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if needle in text:
            print(f"OK    {label}")
        else:
            print(f"FAIL  {label}\n      missing marker: {needle!r}")
            ok = False
    print("-" * 60)
    if ok:
        print("All checks passed.")
        return 0
    print("Some checks failed — copy the latest MER_Client_GPU files.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
