# MER Initial Version — GPU environment setup (Windows)
# Works with `python` on PATH or the Windows `py` launcher (py -3.11).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

. "$PSScriptRoot\resolve_python.ps1"

Write-Host "=== MER GPU Setup ===" -ForegroundColor Cyan

$pyCmd = Get-MerPythonCommand
if (-not $pyCmd) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host "Install Python 3.11 from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Or ensure the 'py' launcher works:  py --version" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using: $($pyCmd.Display)" -ForegroundColor Green

$version = Invoke-MerPython @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Python version: $version"

if ($version -notmatch "^(3\.(10|11|12))") {
    Write-Host "WARNING: Python 3.10–3.12 recommended. Found $version" -ForegroundColor Yellow
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    Invoke-MerPython @("-m", "venv", ".venv") | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "ERROR: venv creation failed — missing $venvPy" -ForegroundColor Red
    exit 1
}

Write-Host "Installing dependencies into .venv ..." -ForegroundColor Green
& $venvPy -m pip install --upgrade pip

$reqFile = "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Host "ERROR: requirements.txt not found in project root." -ForegroundColor Red
    exit 1
}

& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nRunning environment check..." -ForegroundColor Green
& $venvPy tools/check_environment.py

Write-Host "`nSetup complete." -ForegroundColor Cyan
Write-Host "Use the venv Python for all commands (no need for 'python' on PATH):" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\activate" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe tools/check_environment.py" -ForegroundColor White
Write-Host "Then run:" -ForegroundColor Cyan
Write-Host "  .\verify_gpu.ps1" -ForegroundColor White
