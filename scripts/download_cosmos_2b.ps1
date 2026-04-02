# Descarga Cosmos-Reason2-2B en models\Cosmos-Reason2-2B
# Requiere: hf auth login con token que permita repos gated (ver HUGGINGFACE_SETUP.md)
# Uso: .\scripts\download_cosmos_2b.ps1

$ErrorActionPreference = "Stop"
# PSScriptRoot = ...\Nvidia_COSMOS\scripts  ->  project root = parent
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$hf = Join-Path $root ".venv\Scripts\hf.exe"
if (-not (Test-Path $hf)) {
    Write-Host "No existe $hf — crea el venv e instala huggingface_hub: pip install huggingface_hub"
    exit 1
}

Write-Host "Comprobando sesión Hugging Face..."
& $hf auth whoami

$dest = Join-Path $root "models\Cosmos-Reason2-2B"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Write-Host "Descargando nvidia/Cosmos-Reason2-2B hacia $dest (varios GB, puede tardar)..."
& $hf download nvidia/Cosmos-Reason2-2B --local-dir $dest

Write-Host ""
Write-Host "Listo. En .env pon:"
Write-Host "MOCK_COSMOS=0"
Write-Host "COSMOS_MODEL=$($dest -replace '\\','/')"
