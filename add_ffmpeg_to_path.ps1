# This script adds ffmpeg to the PATH for the current PowerShell session

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

# Add ffmpeg to the current session PATH
$env:PATH = "$ffmpegPath;$env:PATH"

Write-Host "ffmpeg has been added to PATH for the current session." -ForegroundColor Green
Write-Host "You can now use ffmpeg commands directly." -ForegroundColor Green
Write-Host "This change is temporary and will be lost when you close this PowerShell window." -ForegroundColor Yellow
Write-Host "To verify, try running: ffmpeg -version" -ForegroundColor Cyan
