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

2. Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or using your package manager

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
