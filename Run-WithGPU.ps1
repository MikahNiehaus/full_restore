# GPU Accelerated Full Restoration Pipeline Launcher
# This script activates the virtual environment and runs the full restoration pipeline with GPU acceleration

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "DeOldify Full Restoration Pipeline with GPU Acceleration" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

Write-Host "`nChecking for CUDA GPU..."

# Check for NVIDIA GPU
try {
    $gpuCheck = nvidia-smi
    $gpuName = ($gpuCheck | Select-String "NVIDIA" | Select-Object -First 1).ToString()
    Write-Host "NVIDIA GPU detected: $gpuName" -ForegroundColor Green
}
catch {
    Write-Host "[WARNING] NVIDIA GPU not detected or drivers not properly installed." -ForegroundColor Yellow
    Write-Host "          Pipeline may run in CPU-only mode (very slow)." -ForegroundColor Yellow
    Write-Host "`nPress any key to continue anyway or Ctrl+C to abort..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

Write-Host "`nActivating virtual environment..."

# Activate the virtual environment if it exists
if (Test-Path "venv310\Scripts\Activate.ps1") {
    & .\venv310\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
}
else {
    Write-Host "[WARNING] Virtual environment not found at venv310\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "          Will try to continue without activating environment." -ForegroundColor Yellow
}

Write-Host "`nStarting full restoration pipeline with GPU acceleration..." -ForegroundColor Green
Write-Host ""
Write-Host "* GPU will be used for DeOldify colorization" -ForegroundColor Cyan
Write-Host "* GPU will be used for image restoration" -ForegroundColor Cyan  
Write-Host "* Audio enhancements enabled" -ForegroundColor Cyan
Write-Host "`nProcessing any videos found in the 'inputs' directory..."

# Run the simple_run.py script with GPU acceleration
python simple_run.py --do-enhance

Write-Host "`nPipeline completed. Press any key to exit."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Deactivate the virtual environment if it was activated
if ($env:VIRTUAL_ENV) {
    deactivate
}
