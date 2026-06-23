# Test GUI Guide

## Launch

```text
.\.venv\Scripts\python.exe tools\run_gui.py
```

## Two test modes

| Mode | What you need | Buttons |
|---|---|---|
| **Smoke (no data)** | Nothing | Smoke (no data), Environment |
| **Real data** | Excel + `.avi` media root | Preprocess, GPU grouped, … |

## Real-data paths

| Field | Purpose |
|---|---|
| **Excel (.xlsx)** | `CASME2-coding-20140508.xlsx` |
| **Media root** | Depends on **Media type** (see below) |
| **Media type** | `avi` or `images` |

### Media type: `images` (CASME-II Cropped)

```text
{Media root}/          e.g. DATASETS/CASME II/Cropped/
  sub01/
    EP02_01f/
      img73.jpg
      img74.jpg
      ...
  sub02/
    ...
```

Excel row `Subject=1`, `Filename=EP02_01f` → folder `Cropped/sub01/EP02_01f/`

Step 2 reads jpg/png inside that folder and uses OnsetFrame–OffsetFrame from Excel.

### Media type: `avi` (magnified clips)

```text
{Media root}/          e.g. Processed_Data/Raw_Videos_Magnified/CASME2/
  sub01/EP02_01f.avi
  sub02/...
```

### Auto-fill

**Browse CASME-II folder (auto-fill)** detects:
- Excel file
- `Cropped/` → sets **images** mode
- `.avi` tree → sets **avi** mode

## Verification checklist (9 items)

All nine map to test buttons:

1. GPU env — Environment  
2. Smoke S1/S2/ablation — Smoke (no data)  
3. Preprocess — Preprocess (real data)  
4. GPU 3-cls / GPU indiv. — GPU buttons  
5. Plots / Literature — respective buttons  
6. **▶ RUN ALL** — runs the full sequence  

## Window

- Opens maximized  
- Drag the divider above the log to resize the log panel  
