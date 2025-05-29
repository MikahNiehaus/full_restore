# Setup and run image restoration and colorization
Write-Host "Setting up image restoration and colorization environment..." -ForegroundColor Cyan

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptPath

# Activate virtual environment
& "$scriptPath\venv\Scripts\Activate.ps1"

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install watchdog Pillow

# Setup colorizer module
Write-Host "Setting up colorizer module..." -ForegroundColor Cyan
python setup_colorizer.py

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Cyan
New-Item -Path "inputs" -ItemType Directory -Force | Out-Null
New-Item -Path "outputs" -ItemType Directory -Force | Out-Null
New-Item -Path "processed" -ItemType Directory -Force | Out-Null

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Please put your image files into the 'inputs' folder to process them."

Write-Host "`nStarting automatic monitoring of the inputs folder..." -ForegroundColor Cyan
Write-Host "Any new image placed in the 'inputs' folder will be automatically processed."
Write-Host "Press Ctrl+C to stop the monitoring process.`n"

# Run the auto watch script
python auto_watch_simple.py

Read-Host "Press Enter to exit"
