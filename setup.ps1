<#
.SYNOPSIS
    One-command setup & launch for AquaSpectra - maize water-stress detection.

.DESCRIPTION
    Designed for agriculture researchers (no developer experience needed).
    This script will:
      1. Check that Python 3.10+ is installed.
      2. Create an isolated virtual environment (.venv).
      3. Install AquaSpectra and all dependencies.
      4. (Optional) Generate a synthetic demo dataset and run the full pipeline.
      5. (Optional) Launch the web app in your browser.

.PARAMETER Demo
    Generate a synthetic dataset and run the full analysis pipeline end-to-end
    (no real imagery needed) so you can see results immediately.

.PARAMETER Launch
    Start the AquaSpectra web app after setup (default: on).

.PARAMETER NoLaunch
    Skip launching the web app (just set everything up).

.PARAMETER Port
    Port for the web app (default 8501).

.EXAMPLE
    .\setup.ps1
    Full setup, then open the web app.

.EXAMPLE
    .\setup.ps1 -Demo
    Set up, build a demo dataset, run the analysis, then open the web app.

.EXAMPLE
    .\setup.ps1 -NoLaunch
    Just install everything; don't open the app.
#>
[CmdletBinding()]
param(
    [switch]$Demo,
    [switch]$Launch = $true,
    [switch]$NoLaunch,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  AquaSpectra - Maize Crop Water-Stress Detection (setup)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. Python ---
Write-Step "Checking for Python 3.10+"
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        try {
            $ver = & $cmd -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($ver -and [version]$ver -ge [version]"3.10") { $python = $cmd; break }
        } catch { }
    }
}
if (-not $python) {
    Write-Warn "Python 3.10 or newer was not found."
    Write-Warn "Please install it from https://www.python.org/downloads/ (tick"
    Write-Warn "'Add Python to PATH' during install), then re-run this script."
    exit 1
}
Write-Info "Found Python $ver via '$python'."

# ------------------------------------------------------------ 2. venv setup ---
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment (.venv)"
    & $python -m venv .venv
} else {
    Write-Step "Virtual environment already exists - reusing it"
}

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip --quiet

# ----------------------------------------------------------- 3. install app ---
Write-Step "Installing AquaSpectra and dependencies (this can take a few minutes)"
& $venvPython -m pip install -e ".[ui]" --quiet
Write-Info "Installation complete."

# ------------------------------------------------------------- 4. demo run ----
if ($Demo) {
    Write-Step "Generating synthetic demo dataset (126 band TIFs + plots)"
    & $venvPython tools\make_synthetic_data.py

    Write-Step "Running the analysis pipeline on the demo data"
    & $venvPython -m aquaspectra.cli run -c config.test.yaml

    Write-Step "Generating the demo stress map"
    & $venvPython -m aquaspectra.cli predict -c config.test.yaml
    Write-Info "Demo results saved in the 'outputs' folder."
}

# --------------------------------------------------------------- 5. launch ----
if ($NoLaunch) { $Launch = $false }

if ($Launch) {
    Write-Step "Starting the AquaSpectra web app"
    Write-Info "Open this address in your browser if it does not open automatically:"
    Write-Info "    http://localhost:$Port"
    Write-Info "Press Ctrl+C in this window to stop the app."
    Write-Host ""
    & $venvPython -m streamlit run app.py --server.port $Port
} else {
    Write-Host ""
    Write-Step "Setup finished. To start the web app later, run:"
    Write-Info  ".\run.ps1"
    Write-Step "Or use the command line, e.g.:"
    Write-Info  ".\.venv\Scripts\python.exe -m aquaspectra.cli run -c config.yaml"
}
