"""
run_ablation_gpu.py
===================
Run ablation experiments with GPU defaults (CUDA + AMP).
Extra CLI args are forwarded to run_ablation_experiments.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    runner = project_root / "Ablation_Study" / "run_ablation_experiments.py"

    cmd = [
        sys.executable,
        str(runner),
        "--device",
        "cuda",
        "--batch_size",
        "4",
        *sys.argv[1:],
    ]
    print("[GPU MODE]", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=project_root)
    if completed.returncode != 0:
        return completed.returncode

    try:
        import torch
    except ImportError:
        return completed.returncode

    if not torch.cuda.is_available():
        print(
            "\nWarning: GPU mode requested but CUDA is not available. "
            "Install requirements.txt on a CUDA machine:\n"
            "  pip install -r requirements.txt"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
