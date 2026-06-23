# GPU Mode Guide

Use this on a machine with **NVIDIA GPU + CUDA 12.6**.

Recommended: **Python 3.11**.

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\check_environment.py
```

Or launch the test GUI (includes setup note):

```powershell
.\.venv\Scripts\python.exe tools\run_gui.py
```

Expected: `CUDA available: True` and GPU name printed.

## Preprocessing (CPU)

GPU training needs precomputed tensors:

```powershell
.\.venv\Scripts\python.exe Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only
.\.venv\Scripts\python.exe Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors --dataset_filter CASME_II --expression_filter micro-expression
.\.venv\Scripts\python.exe Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors_raw --dataset_filter CASME_II --expression_filter micro-expression
```

Or use **Real preprocessing** in `tools/run_gui.py`.

## GPU ablation

```powershell
.\.venv\Scripts\python.exe tools/run_ablation_gpu.py --label_mode grouped --epochs 5 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
```

## Troubleshooting

| Issue | Action |
|---|---|
| `CUDA available: False` | Reinstall: `pip install -r requirements.txt`, update NVIDIA driver |
| Out of memory | `--batch_size 2` |
| Missing tensors | Run Step 1 + Step 2 or GUI preprocessing |
