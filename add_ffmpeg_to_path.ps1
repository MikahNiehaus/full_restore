# This script adds C:\ffmpeg\bin to the user PATH if not already present
$ffmpegPath = "C:\ffmpeg\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*${ffmpegPath}*") {
    $newPath = if ($currentPath) { "$currentPath;$ffmpegPath" } else { $ffmpegPath }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "[INFO] C:\ffmpeg\bin added to your user PATH. Please restart your terminal or VS Code."
} else {
    Write-Host "[INFO] C:\ffmpeg\bin is already in your user PATH."
}
