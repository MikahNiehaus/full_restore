# Image and Video Restoration Pipeline

## Overview
This project provides a unified image and video restoration pipeline that follows a specific order of operations:
1. **Restore** - Upscale and enhance details in the image using AI models or OpenCV
2. **Colorize** - Add color to grayscale or faded images using DeOldify
3. **Enhance** - Make final adjustments to brightness, contrast, and sharpness

## Pipeline Components
The pipeline consists of these main components:
- `unified_image_processor.py` - Core class that handles image processing and provides the correct pipeline
- `video_restore_and_colorize.py` - Video processing using the unified image processor
- `fix_realesrgan_import.py` - Helper script to fix import issues with RealESRGAN

## Implementation Details

### ImageProcessor Class
The `ImageProcessor` class provides the following key methods:
- `restore_image()` - Uses RealESRGAN or OpenCV for image restoration
- `colorize_image()` - Uses DeOldify for colorization with fallback options
- `enhance_image()` - Final image adjustments using OpenCV and PIL
- `process_image()` - Complete pipeline for single image processing
- `process_frames()` - Batch processing for video frames

### Processing Order
The pipeline now follows the correct order:
1. **Restore** - First, the image is upscaled and details are enhanced
2. **Colorize** - Then, the restored image is colorized
3. **Enhance** - Finally, adjustments are made to improve the visual quality

### Error Handling
- Each step has proper error handling and fallback mechanisms
- If any step fails, the pipeline continues with the output from the previous step
- Only one final output file is produced per image/frame

## Testing
The pipeline has been tested with:
- Individual images
- Video frames
- Error cases and fallback mechanisms

## Usage
For image processing:
```python
from unified_image_processor import ImageProcessor

processor = ImageProcessor("outputs")
result = processor.process_image("input.jpg", scale=2)
```

For video processing:
```
python video_restore_and_colorize.py
```

## Future Improvements
- Real-ESRGAN import handling could be further improved
- Additional enhancement options could be added
- Performance optimizations for batch processing
- More configurability of the pipeline parameters
