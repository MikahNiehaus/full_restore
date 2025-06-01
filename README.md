# DeOldify Full Restore

A comprehensive video restoration and colorization pipeline using DeOldify with advanced audio processing.

## Features

- **Continuous Monitoring**: Automatically processes videos placed in an input folder
- **AI Image Restoration**: Applies advanced AI techniques to restore and improve frames before colorization
- **Maximum Quality Colorization**: Uses DeOldify with maximum quality settings (render_factor=40) for best results
- **Advanced Audio Processing**: Extracts, enhances, and synchronizes audio tracks
- **Automatic YouTube Upload**: Uploads processed videos to YouTube with customizable title and description
- **Upload Organization**: Organizes videos in separate folders for successful and failed uploads
- **Complete Pipeline**: Processes videos from start to finish with no manual intervention needed
- **Clean Processing**: Moves processed videos and cleans up temporary files automatically

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- FFmpeg (must be installed and on your system PATH)
- Google Cloud Project with YouTube Data API v3 enabled (for YouTube uploads)
- Other dependencies listed in requirements.txt

## Installation

1. Install Python dependencies:
```
pip install -r requirements.txt
```

2. Install FFmpeg:

   ### Windows:
   - Download the FFmpeg build from [ffmpeg.org](https://ffmpeg.org/download.html) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (recommended: git-full build)
   - Extract the zip file to a permanent location (e.g., `C:\Program Files\ffmpeg`)
   - Add FFmpeg to your system PATH:
     - Right-click on "This PC" or "My Computer" and select "Properties"
     - Click on "Advanced system settings"
     - Click on "Environment Variables"
     - Under "System variables", find the "Path" variable, select it and click "Edit"
     - Click "New" and add the path to the FFmpeg `bin` folder (e.g., `C:\Program Files\ffmpeg\bin`)
     - Click "OK" on all dialog boxes
   - Verify installation by opening a new command prompt and typing: `ffmpeg -version`

   ### macOS:
   - Using Homebrew: `brew install ffmpeg`
   - Verify installation: `ffmpeg -version`

   ### Linux:
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - CentOS/RHEL: `sudo yum install ffmpeg ffmpeg-devel`
   - Verify installation: `ffmpeg -version`

## Usage

### Basic Usage

Simply run the full restoration pipeline:

```
python full_restore.py
```

This will:
1. Monitor the `inputs` folder for new video files
2. Colorize each video with DeOldify at maximum quality
3. Enhance the audio track
4. Save processed videos to the `outputs` folder
5. Move original videos to the `processed` folder

### Advanced Options

```
python full_restore.py --help
```

Available options:
- `--input-dir`, `-i`: Input directory to monitor (default: "inputs")
- `--output-dir`, `-o`: Output directory for processed videos (default: "outputs")
- `--processed-dir`, `-p`: Directory for original videos after processing (default: "processed")
- `--render-factor`, `-r`: DeOldify render factor (10-45) (default: 40)
- `--poll-interval`: Seconds between checks for new videos (default: 10)
- `--no-audio-enhance`: Disable audio enhancement
- `--no-restore-frames`: Disable AI image restoration

## YouTube Upload Functionality

This project automatically uploads processed videos to YouTube and organizes them into appropriate folders:

1. First, set up the YouTube API:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the YouTube Data API v3 in your project
   - Create OAuth credentials for a Desktop app
   - Download the client secret JSON and place it in the `YouTubeApi` folder

2. Authorize the application (required only once):
```
# Using the batch file
authorize_youtube.bat

# OR using Python directly
python YouTubeApi\authorize.py
# OR if you have issues with the regular authorization
python YouTubeApi\simple_authorize.py
```

3. Run the full pipeline with automatic uploads:
```
python simple_run.py
# OR
python full_restore.py
```

4. The system will:
   - Process each video with DeOldify and audio enhancement
   - Upload the processed video to YouTube
   - Move successful uploads to the `outputs/uploaded` folder
   - Move failed uploads to the `outputs/failed_upload` folder

**Note**: New Google Cloud projects have restrictions when first created. You may need to:
- Add your Google account as a test user in the OAuth consent screen
- Wait a short period (sometimes up to an hour) for the API access to become fully available
- If facing issues, try with `simple_authorize.py` which uses a different authorization method

## Examples

Process videos with a specific quality setting:
```
python full_restore.py --render-factor 35
```

Use different directories:
```
python full_restore.py --input-dir my_videos --output-dir colorized --processed-dir originals
```

Skip audio enhancement:
```
python full_restore.py --no-audio-enhance
```

Skip AI image restoration:
```
python full_restore.py --no-restore-frames
```

## How It Works

1. **Video Frame Extraction**: Each video is split into individual frames
2. **Audio Extraction**: Audio track is separated for enhancement
3. **AI Image Restoration**: Frames are processed to remove scratches, reduce noise, and enhance details
4. **Colorization**: DeOldify colorizes each restored frame with maximum quality
5. **Audio Enhancement**: Noise reduction and audio restoration
6. **Reassembly**: Frames are reassembled into a video
7. **Audio Muxing**: Enhanced audio is synchronized with the colorized video
8. **YouTube Upload**: The final video is automatically uploaded to YouTube (if enabled)
9. **Cleanup**: Temporary files are removed

## Notes

- Higher render_factor values provide better quality but require more GPU memory
- If you encounter CUDA out of memory errors, try reducing the render_factor value
- Processing large videos can take significant time

# ffmpeg Setup Helper Scripts

This collection of scripts helps you set up and configure ffmpeg on your Windows system.

## PowerShell Execution Policy

Before running the scripts, you may need to adjust your PowerShell execution policy. By default, Windows restricts running unsigned PowerShell scripts for security reasons. You have several options:

### Option 1: Run scripts in the current session only
Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```
This will allow script execution only in the current PowerShell session.

### Option 2: Temporarily bypass the execution policy for a single script
```powershell
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg.ps1
```

### Option 3: Change execution policy permanently (less secure)
Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
This allows local scripts to run without digital signatures.

## Available Scripts

### 1. `find_ffmpeg.ps1`
Searches your system for existing ffmpeg installations.
- Run: `.\find_ffmpeg.ps1`

### 2. `install_ffmpeg.ps1`
Downloads and installs ffmpeg automatically.
- Run: `.\install_ffmpeg.ps1`

### 3. `add_ffmpeg_to_path.ps1`
Adds ffmpeg to your PATH temporarily (for the current PowerShell session).
- Run: `.\add_ffmpeg_to_path.ps1`

### 4. `add_ffmpeg_permanent.ps1`
Adds ffmpeg to your system PATH permanently.
- Run as Administrator: `.\add_ffmpeg_permanent.ps1`

## Manual ffmpeg Installation

If you prefer to install ffmpeg manually:

1. Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) (use the Windows builds)
2. Extract the zip file to a location like `C:\ffmpeg`
3. Add the bin folder to your PATH:
   - Press Win+X and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the path to the ffmpeg bin folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all windows
4. Restart any open command prompts or PowerShell windows
5. Verify installation with `ffmpeg -version`

## Common Warnings and Troubleshooting

### NumExpr and pkg_resources Warnings

You may see warnings like:

```
NumExpr defaulting to 12 threads.
C:\Users\...\pkg_resources is deprecated as an API...
```

These are normal and can be safely ignored:

- **NumExpr defaulting to X threads**: This is just informational, showing that NumExpr (a numerical expression evaluator) is using multiple CPU threads for better performance.

- **pkg_resources is deprecated**: This is a warning from one of the dependencies (fastai) using an older Python packaging API. This doesn't affect functionality and is just a notice for developers.

These warnings are related to the underlying libraries and don't indicate problems with your setup or the application's functionality.

### Audio Enhancement Errors

If you see an error like:
```
[ERROR] Audio enhancement failed: The length of the input vector x must be greater than padlen, which is 27.
[WARNING] Audio enhancement failed, using original audio
```

This means:
- The audio in the video is too short or empty for the enhancement algorithms
- The system is automatically falling back to using the original audio
- Your video will still be processed with the original audio (no enhancement)

This typically happens with very short videos or videos with no significant audio content. Possible solutions:

1. **For videos with very short audio**: 
   - Use the `--no-audio-enhance` flag to skip audio enhancement entirely
   - If you need to process multiple short videos, consider batch processing:
   ```
   python full_restore.py --no-audio-enhance
   ```

2. **To modify the audio processing parameters**:
   You can edit `audio_enhancer.py` to make the equalizer more tolerant of short audio files:
   - Find the `apply_equalization` method
   - Reduce the filter order (e.g., change order=4 to order=2)
   - Or conditionally skip equalization for very short audio files

Most videos should process without this error, as the system will automatically handle longer audio tracks.
