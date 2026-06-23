# June 19 Delivery — GPU Test Guide

Complete checklist to verify **all June 19 requirements** on your GPU PC.

**Requirements:** Python 3.11, NVIDIA GPU, CUDA 12.6 (or compatible driver).

---

## Step 0 — Copy project to GPU PC

Copy either:
- `E:\Rcognition\MER_Client_GPU.zip` (unzip), or
- `E:\Rcognition\MER_InitialProject\` folder

Also copy CASME-II dataset into `DATASETS/CASME II/`:
- `CASME2-coding-20140508.xlsx`
- Cropped/magnified video frame folders

---

## Step 1 — Install (one time)

```powershell
cd MER_InitialProject
.\setup_gpu.ps1
```

Expected: `CUDA available: True`

---

## Step 2 — Wiring tests (no real dataset)

```powershell
.\.venv\Scripts\activate
python tools/smoke_step1_cpu.py
python tools/smoke_step2_cpu.py
python tools/smoke_ablation_cpu.py
```

All three must print `passed`.

---

## Step 3 — Real data preprocessing

```powershell
python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only
python Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors --dataset_filter CASME_II --expression_filter micro-expression
python Stage1_DataPipeline/main_step2.py --max_workers 8 --output_subdir tensors_raw --dataset_filter CASME_II --expression_filter micro-expression
```

Check:
- `Processed_Data/master_thesis_labels.csv` exists
- `.npy` files in `Processed_Data/tensors/` and `tensors_raw/`

---

## Step 4 — June 19 verification experiments (4 key configs)

### 4a) Grouped labels (Positive / Negative / Surprise)

```powershell
python tools/run_ablation_gpu.py --label_mode grouped --epochs 5 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
```

### 4b) Individual emotion labels

```powershell
python tools/run_ablation_gpu.py --label_mode individual --epochs 5 --configs config_1_pure_base config_8_proposed_unified --output_root Ablation_Study/results_individual
```

---

## Step 5 — Generate plots + literature comparison

```powershell
python tools/plot_ablation_results.py
python tools/plot_ablation_results.py --results_root Ablation_Study/results_individual
python tools/compare_with_literature.py
```

Outputs:
- `Ablation_Study/results/plots/accuracy_macro_f1_bar.png`
- `Ablation_Study/results/plots/per_class_f1_grouped.png`
- `Ablation_Study/results/literature_comparison.csv`

---

## Step 6 — Optional full 12-config sweep

```powershell
python tools/run_ablation_gpu.py --label_mode grouped --epochs 30
python tools/plot_ablation_results.py
```

---

## June 19 success checklist

| Requirement | How to verify |
|---|---|
| Preprocessing pipeline | Step 3 completes, `.npy` tensors created |
| EVM toggle | Configs 4/7/8 use EVM tensors; 1/3 use raw |
| 3D-CNN + SimAM + Transformer toggles | 12 configs in `ablation_config.py` |
| 4 key verification configs | Step 4a produces 4 rows in `summary.csv` |
| Grouped + individual labels | Step 4a + 4b both run |
| Metrics (acc, F1, confusion matrix) | `summary.csv`, `final_results.json`, CM PNG |
| Comparison plots | Step 5 PNG files |
| Literature comparison table | `literature_comparison.csv` |
| GPU training | `CUDA available: True` in check_environment |

---

## One-click script (GPU PC)

```powershell
.\client_delivery\run_june19_gpu_test.ps1
```

Runs smoke tests + grouped 4-config ablation (5 epochs) + plots.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| CUDA False | `pip install -r requirements.txt` |
| Out of memory | `--batch_size 2` |
| Missing tensors | Re-run Step 2; check CSV `Frames_Directory` paths |
| Empty dataset | Confirm micro-expression filter matches your CSV |
| Individual labels empty | Ensure CSV has `Raw_Emotion` column from Step 1 |
