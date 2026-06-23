# MER Initial Version — GPU verification (smoke tests + 1-epoch GPU ablation)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv not found. Run .\setup_gpu.ps1 first." -ForegroundColor Red
    exit 1
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "=== Step 1: Environment ===" -ForegroundColor Cyan
& $py tools/check_environment.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 2: Pipeline smoke tests (no dataset) ===" -ForegroundColor Cyan
& $py tools/smoke_step1_cpu.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py tools/smoke_step2_cpu.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py tools/smoke_ablation_cpu.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 3: GPU ablation mini-run ===" -ForegroundColor Cyan
& $py tools/run_ablation_gpu.py --epochs 1 --max_samples 16 --configs config_1_pure_base
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== ALL GPU VERIFICATION PASSED ===" -ForegroundColor Green
