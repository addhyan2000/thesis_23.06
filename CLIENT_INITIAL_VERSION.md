# Initial Version — Client Delivery Summary

This document explains what the **initial version** includes, what it does **not** include yet, and exactly how you (or your client) can verify the system works.

---

## What this initial version is

This is **Stage 1 code delivery** for the CASME-II micro-expression research project.

It is a **modular, reproducible pipeline** that lets you:

1. Prepare CASME-II labels and motion tensors (flow + strain, optional EVM path).
2. Train/evaluate a **toggle-based MER model** (3D-CNN, SimAM, SLSTT Transformer).
3. Run **ablation configurations** and collect metrics (accuracy, macro F1, confusion matrix).

### Included in this delivery

| Component | Status |
|---|---|
| CASME-II metadata unification (Step 1) | Included |
| Optical flow + strain tensor extraction (Step 2) | Included |
| Modular ablation model (12 toggle combinations) | Included |
| Subject-disjoint holdout validation | Included |
| CPU mode (no GPU required) | Included |
| GPU mode (CUDA training) | Included (separate install/run path) |
| Synthetic smoke tests (no dataset needed) | Included |
| Flexible labels (grouped / individual) | Included | `--label_mode grouped` or `individual` |
| Comparison plots | Included | `python tools/plot_ablation_results.py` |
| Literature comparison table | Included | `python tools/compare_with_literature.py` |
| Dissertation writing | **Not in initial version** |
| Final paper-matching accuracy targets (~70%) | **Not guaranteed in smoke tests**; requires full dataset + longer training |

---

## Two separate runtime modes

| Mode | Install file | Ablation runner | When to use |
|---|---|---|---|
| **CPU** | `requirements-cpu.txt` | `python tools/run_ablation_cpu.py ...` | Your PC (no GPU), quick wiring checks |
| **GPU** | `requirements.txt` | `python tools/run_ablation_gpu.py ...` | Client machine with NVIDIA GPU |

Detailed commands:

- CPU: `CPU_MODE.md`
- GPU: `GPU_MODE.md`

---

## How to see the system works (fastest path)

From `MER_InitialProject`:

### Step A — Check environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-cpu.txt
python tools/check_environment.py
```

### Step B — Run synthetic smoke tests (no CASME-II dataset required)

```powershell
python tools/smoke_step1_cpu.py
python tools/smoke_step2_cpu.py
python tools/smoke_ablation_cpu.py
```

If all 3 print `passed`, the pipeline wiring is correct.

### Step C — Run real data (client side)

1. Put dataset files in `DATASETS/CASME II/`.
2. Run Step 1 + Step 2 preprocessing.
3. Run 4 verification configs (CPU or GPU).

See `INITIAL_VERSION.md` for full real-data commands.

---

## What success looks like

After smoke tests:

- Step1 creates `Processed_Data/master_thesis_labels.csv`.
- Step2 creates tensor `.npy` with shape `(3, 32, 224, 224)`.
- Ablation writes:
  - `Ablation_Study/results/summary.csv`
  - per-config `final_results.json`
  - per-config `confusion_matrix.npy`

After real-data quick run:

- `summary.csv` contains rows for configs like:
  - `config_1_pure_base`
  - `config_3_spatial_only`
  - `config_7_full_no_attention`
  - `config_8_proposed_unified`

---

## Suggested client reply (copy/paste)

> Initial version delivered in `MER_InitialProject`.
>
> It includes CASME-II preprocessing, modular ablation model toggles (EVM / 3D-CNN / SimAM / Transformer), and experiment logging.
>
> CPU and GPU modes are separated:
> - CPU install: `requirements-cpu.txt`
> - GPU install: `requirements.txt`
>
> To verify quickly without dataset:
> 1) `python tools/check_environment.py`
> 2) run the 3 smoke scripts in `tools/`
>
> For real CASME-II runs, follow `INITIAL_VERSION.md` (Step1 -> Step2 -> ablation).
>
> Full LOSO (26 folds) is intentionally deferred for your GPU-side validation in the next phase.

---

## Project folders (clean initial scope)

```
MER_InitialProject/
├── Stage1_DataPipeline/   # preprocessing
├── Stage2_Architecture/   # optional model verification
├── Ablation_Study/        # experiments + results
├── tools/                 # smoke tests + cpu/gpu runners
├── DATASETS/CASME II/     # input data location
└── Processed_Data/        # generated outputs
```
