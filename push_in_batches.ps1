# This script pushes changes to a remote Git repository in smaller batches.
# If the push fails due to too many requests, it waits for 30 minutes before retrying.

# Refine the script to explicitly handle uncommitted changes and push one file at a time
# Function to push a single file
function Push-SingleFile {
    param (
        [string]$FilePath,
        [string]$Branch = "main"
    )

    Write-Host "Attempting to push file '$FilePath' to branch '$Branch'..."

    try {
        # Clear the staging area to ensure only one file is staged
        git reset

        # Stage the file
        git add $FilePath

        # Commit the file
        git commit -m "Pushing file: $FilePath"

        # Push the changes
        git push origin $Branch --force

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Push succeeded for file: $FilePath"
            return $true
        } else {
            Write-Host "Push failed for file: $FilePath. Exit code: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Host "An error occurred during the push for file: ${FilePath}: $_"
        return $false
    }
}

# Function to push a batch of files
function Push-Batch {
    param (
        [array]$FilePaths,
        [string]$Branch = "main"
    )

    Write-Host "Attempting to push batch of files: $($FilePaths -join ', ') to branch '$Branch'..."

    try {
        # Clear the staging area to ensure only the batch is staged
        git reset

        # Stage the batch of files
        foreach ($file in $FilePaths) {
            git add $file
        }

        # Commit the batch
        git commit -m "Pushing batch of files: $($FilePaths -join ', ')"
        # Push the changes
        git push origin $Branch --force

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Push succeeded for batch: $($FilePaths -join ', ')"
            return $true
        } else {
            Write-Host "Push failed for batch: $($FilePaths -join ', '). Exit code: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Host "An error occurred during the push for batch: $($FilePaths -join ', '): $_"
        return $false
    }
}

# Function to check if a file is too large
function Is-FileTooLarge {
    param (
        [string]$FilePath,
        [int]$MaxSizeMB = 10
    )

    $fileInfo = Get-Item $FilePath -ErrorAction SilentlyContinue
    if ($fileInfo -and $fileInfo.Length -gt ($MaxSizeMB * 1MB)) {
        return $true
    }
    return $false
}

# Initialize the Git repository if not already initialized
if (-not (Test-Path -Path ".git")) {
    Write-Host "Initializing a new Git repository..."
    git init
    git remote add origin https://github.com/MikahNiehaus/full_restore.git
    git checkout -b main
}

# Function to check if a file is ignored by .gitignore
function Is-FileIgnored {
    param (
        [string]$FilePath
    )

    $result = git check-ignore -q $FilePath
    return ($LASTEXITCODE -eq 0)
}

# Get all files in the workspace
$allFiles = Get-ChildItem -Recurse -File | ForEach-Object { $_.FullName }

# Filter files based on .gitignore
$filteredFiles = @()
foreach ($file in $allFiles) {
    if (-not (Is-FileIgnored -FilePath $file)) {
        $filteredFiles += $file
    } else {
        Write-Host "Skipping file '$file' as it is ignored by .gitignore."
    }
}

# Push files in batches of 10
$batchSize = 10
$maxFileSizeMB = 100

$filteredFiles = @()
foreach ($file in $uncommittedFiles) {
    if (-not (Is-FileTooLarge -FilePath $file -MaxSizeMB $maxFileSizeMB)) {
        $filteredFiles += $file
    } else {
        Write-Host "Skipping file '$file' as it exceeds the size limit of $maxFileSizeMB MB. Removing from Git cache."
        git rm --cached $file
    }
}

for ($i = 0; $i -lt $filteredFiles.Count; $i += $batchSize) {
    $batch = $filteredFiles[$i..([math]::Min($i + $batchSize - 1, $filteredFiles.Count - 1))]
    $success = $false

    while (-not $success) {
        $success = Push-Batch -FilePaths $batch

        if (-not $success) {
            Write-Host "Push failed for batch: $($batch -join ', '). Checking for too many requests..."

            # Check if the error is related to too many requests
            if ($LASTEXITCODE -eq 22) {
                Write-Host "Too many requests. Waiting for 30 minutes before retrying..."
                Start-Sleep -Seconds 1800
            } else {
                Write-Host "Retrying immediately for batch: $($batch -join ', ')..."
            }
        }
    }
}

Write-Host "All uncommitted changes pushed successfully!"
