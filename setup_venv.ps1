$ErrorActionPreference = "Stop"

$venvPath = ".venv"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Virtual environment is ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
