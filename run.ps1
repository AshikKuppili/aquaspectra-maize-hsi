<#
.SYNOPSIS
    Launch the AquaSpectra web app (assumes setup.ps1 has already been run).

.PARAMETER Port
    Port for the web app (default 8501).

.EXAMPLE
    .\run.ps1
#>
[CmdletBinding()]
param([int]$Port = 8501)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Environment not found. Please run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting AquaSpectra web app at http://localhost:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
& $venvPython -m streamlit run app.py --server.port $Port
