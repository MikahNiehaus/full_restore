# This script searches your system for ffmpeg installations

Write-Host "Searching for ffmpeg installations on your system..." -ForegroundColor Cyan

# Common directories to search
$searchDirs = @(
    "C:\Program Files",
    "C:\Program Files (x86)",
    "C:\",
    "$env:USERPROFILE",
    "$env:LOCALAPPDATA\Programs"
)

$foundLocations = @()

foreach ($dir in $searchDirs) {
    if (Test-Path $dir) {
        Write-Host "Searching in $dir..." -ForegroundColor Yellow
        $results = Get-ChildItem -Path $dir -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue
        
        foreach ($result in $results) {
            $foundLocations += $result.DirectoryName
            Write-Host "Found ffmpeg at: $($result.DirectoryName)" -ForegroundColor Green
        }
    }
}

if ($foundLocations.Count -eq 0) {
    Write-Host "No ffmpeg installations found in common locations." -ForegroundColor Red
    Write-Host "If you've installed ffmpeg elsewhere, you'll need to manually locate it." -ForegroundColor Yellow
    
    $downloadPrompt = Read-Host "Would you like instructions on how to download and install ffmpeg? (y/n)"
    if ($downloadPrompt -eq 'y') {
        Write-Host "`nFFMPEG INSTALLATION INSTRUCTIONS:" -ForegroundColor Cyan
        Write-Host "1. Visit https://ffmpeg.org/download.html" -ForegroundColor White
        Write-Host "2. Download the Windows build (choose the 'Windows builds' link)" -ForegroundColor White
        Write-Host "3. Extract the downloaded zip file to a permanent location (e.g., C:\ffmpeg)" -ForegroundColor White
        Write-Host "4. The ffmpeg.exe file should be in the 'bin' subdirectory" -ForegroundColor White
        Write-Host "5. Run the add_ffmpeg_to_path.ps1 or add_ffmpeg_permanent.ps1 script to add it to your PATH" -ForegroundColor White
    }
} else {
    Write-Host "`nFound $($foundLocations.Count) ffmpeg installation(s)" -ForegroundColor Green
    Write-Host "To add ffmpeg to your PATH, run one of the following scripts:" -ForegroundColor Cyan
    Write-Host "- add_ffmpeg_to_path.ps1 (temporary, for current session)" -ForegroundColor White
    Write-Host "- add_ffmpeg_permanent.ps1 (permanent, requires admin rights)" -ForegroundColor White
}
