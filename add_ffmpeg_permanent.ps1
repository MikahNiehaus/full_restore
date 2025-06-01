# This script permanently adds ffmpeg to the system PATH

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "This script needs to be run as Administrator to modify the system PATH." -ForegroundColor Red
    Write-Host "Please close this window and run PowerShell as Administrator, then run this script again." -ForegroundColor Yellow
    exit 1
}

# Common locations where ffmpeg might be installed
$possibleLocations = @(
    "C:\Program Files\ffmpeg\bin",
    "C:\ffmpeg\bin",
    "C:\Program Files (x86)\ffmpeg\bin",
    "$env:USERPROFILE\ffmpeg\bin",
    "$env:LOCALAPPDATA\Programs\ffmpeg\bin"
)

$ffmpegPath = $null

# Check each location for ffmpeg.exe
foreach ($location in $possibleLocations) {
    if (Test-Path "$location\ffmpeg.exe") {
        $ffmpegPath = $location
        break
    }
}

# If not found in common locations, ask user for the path
if (-not $ffmpegPath) {
    Write-Host "Could not find ffmpeg in common locations." -ForegroundColor Yellow
    $customPath = Read-Host "Please enter the full path to the folder containing ffmpeg.exe"
    
    if (Test-Path "$customPath\ffmpeg.exe") {
        $ffmpegPath = $customPath
    } else {
        Write-Host "Could not find ffmpeg.exe at the specified location." -ForegroundColor Red
        exit 1
    }
}

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Machine)

# Check if ffmpeg path is already in the PATH
if ($currentPath -split ";" -contains $ffmpegPath) {
    Write-Host "ffmpeg is already in your system PATH." -ForegroundColor Green
    exit 0
}

# Add ffmpeg to system PATH
$newPath = "$currentPath;$ffmpegPath"
[Environment]::SetEnvironmentVariable("PATH", $newPath, [EnvironmentVariableTarget]::Machine)

Write-Host "ffmpeg has been permanently added to your system PATH." -ForegroundColor Green
Write-Host "You may need to restart your PowerShell session or computer for changes to take effect." -ForegroundColor Yellow
Write-Host "After restarting, verify by running: ffmpeg -version" -ForegroundColor Cyan
