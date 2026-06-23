# MER Initial Version — GPU Client Package

**Start here.** This package is for the client machine with **NVIDIA GPU + CUDA 12.6**.

---

## What you received

| Item | Description |
|---|---|
| Stage 1 pipeline | CASME-II metadata + optical flow/strain tensor extraction |
| Ablation study | 12 modular configs (EVM / 3D-CNN / SimAM / SLSTT) |
| GPU training | CUDA + mixed precision via `tools/run_ablation_gpu.py` |
| Smoke tests | Verify wiring without CASME-II dataset |

**Not included in this initial version:** full 26-fold LOSO, dissertation, pre-trained weights.

---

## System requirements

| Requirement | Recommendation |
|---|---|
| OS | Windows 10/11 |
| Python | **3.11.x** (3.10 or 3.12 also OK) |
| NVIDIA driver | CUDA **12.6** (already on your machine) |
| GPU | Any CUDA-capable NVIDIA GPU (8 GB+ VRAM recommended) |
| Dataset | CASME-II Excel + cropped video folders (you provide) |

---

## Quick setup (5 minutes)

Open PowerShell in this folder and run:

```powershell
.\setup_gpu.ps1
```

Works with `python` on PATH **or** the Windows `py` launcher (`py --version`).

Or manually with `py` (when `python` is not recognized):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe tools\run_gui.py
```

**Expected:** `CUDA available: True` and your GPU name printed.

If CUDA is false, see **Troubleshooting** in `GPU_MODE.md`.

---

## Verify the system works (no dataset)

```powershell
.\verify_gpu.ps1
```

Or step by step:

```powershell
.\.venv\Scripts\activate
python tools/smoke_step1_cpu.py
python tools/smoke_step2_cpu.py
python tools/smoke_ablation_cpu.py
python tools/run_ablation_gpu.py --epochs 1 --max_samples 16 --configs config_1_pure_base
```

All steps should finish without errors.

---

## Run on real CASME-II data

### 1) Place dataset files

Put in `DATASETS/CASME II/`:

- `CASME2-coding-20140508.xlsx`
- Cropped/magnified CASME-II video folders (as referenced by Step 1)

### 2) Step 1 — metadata CSV

```powershell
python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only
```

Output: `Processed_Data/master_thesis_labels.csv`

### 3) Step 2 — tensor extraction

EVM path:

```powershell
python Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors --dataset_filter CASME_II --expression_filter micro-expression
```

Raw path:

```powershell
python Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors_raw --dataset_filter CASME_II --expression_filter micro-expression
```

### 4) Ablation — GPU training

Quick verification (4 key configs):

```powershell
python tools/run_ablation_gpu.py --epochs 5 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
```

Full 12-config matrix:

```powershell
python tools/run_ablation_gpu.py
```

---

## Success checklist

- [ ] `python tools/check_environment.py` → CUDA available: **True**
- [ ] 3 smoke tests pass
- [ ] Step 1 creates `Processed_Data/master_thesis_labels.csv`
- [ ] Step 2 creates `.npy` tensors (shape `(3, 32, 224, 224)`)
- [ ] `Ablation_Study/results/summary.csv` has rows for your configs
- [ ] Each config folder has `final_results.json` and `confusion_matrix.npy`

---

## Output files to share back

After your first real-data run, please share:

- `Ablation_Study/results/summary.csv`
- `Ablation_Study/logs/ablation.log` (if present)
- Screenshots or copy of `python tools/check_environment.py` output

---

## More documentation

| File | Purpose |
|---|---|
| `GPU_MODE.md` | Full GPU install, commands, troubleshooting |
| `INITIAL_VERSION.md` | Complete runbook |
| `CLIENT_INITIAL_VERSION.md` | Scope and delivery summary |

---

## Install file reference

| File | Purpose |
|---|---|
| `requirements.txt` | **Only install file** — PyTorch CUDA 12.6 + all dependencies |
