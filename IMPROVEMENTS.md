# DeOldify Photo and Video Restoration Pipeline

This project provides tools for restoring and colorizing old photos and videos using DeOldify and Real-ESRGAN models.

## Features

- **Robust CPU Fallback**: Automatically switches to CPU mode when GPU is unavailable or has insufficient memory
- **Enhanced Colorization**: Improved color saturation and brightness adjustments
- **Video Processing**: Frame-by-frame restoration and colorization for videos
- **Audio Enhancement**: Audio restoration for old video soundtracks
- **Upscaling**: High-quality upscaling using Real-ESRGAN

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- FFmpeg
- DeOldify
- Real-ESRGAN

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/full_restore.git
cd full_restore
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download models:
```
python setup_colorizer.py
```

## Usage

### Basic Usage

Process an image:
```
python run_improved.py --input path/to/your/photo.jpg
```

Process a video:
```
python run_improved.py --input path/to/your/video.mp4
```

### Advanced Options

```
python run_improved.py --input path/to/file --output path/to/output --scale 4 --artistic --cpu
```

Options:
- `--input`: Path to input file
- `--output`: Path to output directory
- `--scale`: Upscaling factor (default: 4)
- `--artistic`: Use artistic colorization model
- `--cpu`: Force CPU mode
- `--video`: Process as video even if file has image extension
- `--image`: Process as image even if file has video extension

## CPU Fallback

The pipeline features robust CPU fallback with the following behavior:

1. Automatically detects when GPU is unavailable
2. Reduces render factor to optimize CPU processing
3. Applies enhanced colorization for CPU mode
4. Multiple fallback levels to ensure successful processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeOldify](https://github.com/jantic/DeOldify)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
