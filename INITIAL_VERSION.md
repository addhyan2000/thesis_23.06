# Initial Version Guide

This is the main runbook for the first delivery.

For client-facing scope and testing checklist, see `CLIENT_INITIAL_VERSION.md`.

---

## Choose your mode

| Mode | Install | Run ablation with |
|---|---|---|
| CPU (no GPU) | `pip install -r requirements-cpu.txt` | `python tools/run_ablation_cpu.py ...` |
| GPU (CUDA) | `pip install -r requirements.txt` | `python tools/run_ablation_gpu.py ...` |

Detailed mode docs:

- `CPU_MODE.md`
- `GPU_MODE.md`

Check setup anytime:

```powershell
python tools/check_environment.py
```

---

## 1) Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Then install **one** mode file (CPU or GPU).

---

## 2) Prepare data

Place in `DATASETS/CASME II/`:

- `CASME2-coding-20140508.xlsx`
- Preprocessed/cropped CASME-II video folders referenced by Step 1

---

## 3) Stage 1 Step 1 (metadata CSV)

```powershell
python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only
```

Output: `Processed_Data/master_thesis_labels.csv`

---

## 4) Stage 1 Step 2 (tensor extraction)

EVM tensors:

```powershell
python Stage1_DataPipeline/main_step2.py --max_workers 4 --output_subdir tensors --dataset_filter CASME_II --expression_filter micro-expression
```

Raw tensors:

```powershell
python Stage1_DataPipeline/main_step2.py --max_workers 4 --output_subdir tensors_raw --dataset_filter CASME_II --expression_filter micro-expression
```

Outputs:

- `Processed_Data/tensors/*.npy`
- `Processed_Data/tensors_raw/*.npy`

Each tensor shape: `(3, 32, 224, 224)`.

---

## 5) Ablation experiments

### CPU quick check

```powershell
python tools/run_ablation_cpu.py --epochs 2 --max_samples 16 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
```

### GPU quick check (client machine)

Grouped labels (3-class):

```powershell
python tools/run_ablation_gpu.py --label_mode grouped --epochs 5 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
```

Individual emotions:

```powershell
python tools/run_ablation_gpu.py --label_mode individual --epochs 5 --configs config_8_proposed_unified --output_root Ablation_Study/results_individual
```

Generate plots:

```powershell
python tools/plot_ablation_results.py
python tools/compare_with_literature.py
```

### Full matrix

```powershell
python tools/run_ablation_cpu.py
# or
python tools/run_ablation_gpu.py
```

Results:

- `Ablation_Study/results/summary.csv`
- per-config folders under `Ablation_Study/results/`

---

## 6) Verify system works (no dataset)

```powershell
python tools/smoke_step1_cpu.py
python tools/smoke_step2_cpu.py
python tools/smoke_ablation_cpu.py
```

All three should end with `passed`.

---

## 7) Verification checklist

- [ ] `python tools/check_environment.py` runs
- [ ] 3 smoke tests pass
- [ ] Step1 CSV generated
- [ ] Step2 `.npy` tensors generated
- [ ] Ablation `summary.csv` created
- [ ] At least 4 key configs produce `final_results.json`
