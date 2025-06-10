# PowerShell script to run tests and the fixed pipeline
# This is the PowerShell equivalent of run_fixed.bat

Write-Host "Running tests to verify fixes..." -ForegroundColor Green
python test_fixes.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed, but continuing anyway..." -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
}

# Make sure the environment is activated
if (Test-Path "venv310\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    . .\venv310\Scripts\Activate.ps1
}

# Run the pipeline with full restoration and GPU acceleration
Write-Host "Starting full restoration pipeline with GPU acceleration..." -ForegroundColor Green
python simple_run.py --do-enhance

Write-Host "`nPipeline completed. Press Enter to exit." -ForegroundColor Cyan
Read-Host
