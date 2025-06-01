# This script downloads and installs ffmpeg automatically

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Define installation directory
$installDir = "C:\ffmpeg"
if (-not $isAdmin) {
    $installDir = "$env:USERPROFILE\ffmpeg"
}

Write-Host "This script will download and install ffmpeg to $installDir" -ForegroundColor Cyan
$confirm = Read-Host "Continue? (y/n)"

if ($confirm -ne 'y') {
    Write-Host "Installation cancelled." -ForegroundColor Yellow
    exit 0
}

# Create temp directory for download
$tempDir = "$env:TEMP\ffmpeg_install"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Download ffmpeg
$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zipFile = "$tempDir\ffmpeg.zip"

Write-Host "Downloading ffmpeg... This may take a few minutes." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $url -OutFile $zipFile -UseBasicParsing
}
catch {
    Write-Host "Failed to download ffmpeg: $_" -ForegroundColor Red
    exit 1
}

# Extract the ZIP file
Write-Host "Extracting ffmpeg..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force
    
    # Find the extracted directory (it has version number in the name)
    $extractedDir = (Get-ChildItem -Path $tempDir -Directory | Where-Object { $_.Name -like "ffmpeg-*" })[0]
    
    # Create the installation directory if it doesn't exist
    if (-not (Test-Path $installDir)) {
        New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    }
    
    # Copy the files to the installation directory
    Copy-Item -Path "$($extractedDir.FullName)\bin\*" -Destination "$installDir\bin\" -Recurse -Force
    
    # Clean up
    Remove-Item -Recurse -Force $tempDir
    
    Write-Host "ffmpeg has been installed to $installDir\bin\" -ForegroundColor Green
}
catch {
    Write-Host "Failed to extract or install ffmpeg: $_" -ForegroundColor Red
    exit 1
}

# Add to PATH for current session
$env:PATH = "$installDir\bin;$env:PATH"

# Offer to add to system PATH permanently
if ($isAdmin) {
    $addToPath = Read-Host "Would you like to add ffmpeg to your system PATH permanently? (y/n)"
    if ($addToPath -eq 'y') {
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Machine)
        $newPath = "$currentPath;$installDir\bin"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, [EnvironmentVariableTarget]::Machine)
        Write-Host "ffmpeg has been added to your system PATH." -ForegroundColor Green
    }
}
else {
    Write-Host "To add ffmpeg to your system PATH permanently:" -ForegroundColor Yellow
    Write-Host "1. Run PowerShell as Administrator" -ForegroundColor Yellow
    Write-Host "2. Run the add_ffmpeg_permanent.ps1 script" -ForegroundColor Yellow
}

Write-Host "`nVerifying installation..." -ForegroundColor Cyan
try {
    $output = & "$installDir\bin\ffmpeg.exe" -version
    Write-Host "ffmpeg is working properly!" -ForegroundColor Green
    Write-Host "You can now use ffmpeg in your current session." -ForegroundColor Green
}
catch {
    Write-Host "Failed to run ffmpeg. There might be an issue with the installation." -ForegroundColor Red
}
