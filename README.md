# DeOldify Full Restore

A comprehensive video restoration and colorization pipeline using DeOldify with advanced audio processing.

## Features

- **Continuous Monitoring**: Automatically processes videos placed in an input folder
- **Maximum Quality Colorization**: Uses DeOldify with maximum quality settings (render_factor=40) for best results
- **Advanced Audio Processing**: Extracts, enhances, and synchronizes audio tracks
- **Complete Pipeline**: Processes videos from start to finish with no manual intervention needed
- **Clean Processing**: Moves processed videos and cleans up temporary files automatically

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- FFmpeg (must be installed and on your system PATH)
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

## How It Works

1. **Video Frame Extraction**: Each video is split into individual frames
2. **Audio Extraction**: Audio track is separated for enhancement
3. **Colorization**: DeOldify colorizes each frame with maximum quality
4. **Audio Enhancement**: Noise reduction and audio restoration
5. **Reassembly**: Frames are reassembled into a video
6. **Audio Muxing**: Enhanced audio is synchronized with the colorized video
7. **Cleanup**: Temporary files are removed

## Notes

- Higher render_factor values provide better quality but require more GPU memory
- If you encounter CUDA out of memory errors, try reducing the render_factor value
- Processing large videos can take significant time
