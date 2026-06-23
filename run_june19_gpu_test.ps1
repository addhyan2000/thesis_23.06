# June 19 GPU verification — smoke tests + 4-config grouped ablation + plots

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot | Out-Null
if (Test-Path "..\Stage1_DataPipeline") {
    Set-Location ..
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Run .\client_delivery\setup_gpu.ps1 first." -ForegroundColor Red
    exit 1
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "=== Smoke tests ===" -ForegroundColor Cyan
& $py tools/smoke_step1_cpu.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py tools/smoke_step2_cpu.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py tools/smoke_ablation_cpu.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$csv = "Processed_Data\master_thesis_labels.csv"
if (-not (Test-Path $csv)) {
    Write-Host "`nNo real data CSV found. Skipping GPU ablation (smoke tests passed)." -ForegroundColor Yellow
    Write-Host "Place CASME-II data and run Step 1+2, then re-run this script." -ForegroundColor Yellow
    exit 0
}

Write-Host "`n=== GPU ablation (4 key configs, grouped labels) ===" -ForegroundColor Cyan
& $py tools/run_ablation_gpu.py --label_mode grouped --epochs 5 --configs config_1_pure_base config_3_spatial_only config_7_full_no_attention config_8_proposed_unified
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Plots + literature comparison ===" -ForegroundColor Cyan
& $py tools/plot_ablation_results.py
& $py tools/compare_with_literature.py

Write-Host "`n=== JUNE 19 GPU TEST COMPLETE ===" -ForegroundColor Green
Write-Host "Check Ablation_Study/results/summary.csv and results/plots/"
