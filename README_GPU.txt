MER Initial Version — GPU Client Package
=========================================

CONTENTS
--------
- CASME-II preprocessing pipeline (Stage 1)
- Modular ablation model + 12 toggle configs
- GPU training scripts (CUDA 12.6)
- Test GUI (June 19 checklist)
- Synthetic smoke tests (no dataset required)

FIRST STEPS
-----------
1. Install Python 3.11 (https://www.python.org/downloads/)
2. Unzip this folder
3. Open PowerShell in the unzipped folder
4. py -3.11 -m venv .venv
5. .\.venv\Scripts\python.exe -m pip install -r requirements.txt
6. .\.venv\Scripts\python.exe tools\run_gui.py

MAIN GUIDE
----------
Open START_HERE_GPU.md (or README.md — same content)

DATASET
-------
Select paths in the GUI, or place files in DATASETS/CASME II/

WHAT TO SEND BACK
-----------------
After first real-data GPU run:
- Ablation_Study/results/summary.csv
- Environment check output (screenshot or text)

SUPPORT FILES
-------------
GPU_MODE.md               — troubleshooting
JUNE19_GUI.md             — GUI test guide
INITIAL_VERSION.md        — full runbook

REQUIREMENTS
------------
requirements.txt          — pip install -r requirements.txt

LAUNCH GUI
----------
.\.venv\Scripts\python.exe tools\run_gui.py
