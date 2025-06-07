"""
Test script to verify that DeOldify is using GPU acceleration for colorization
"""
import torch
import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Check if CUDA is available
logging.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logging.info(f"CUDA is available! Using GPU acceleration")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    if hasattr(torch.backends, 'cudnn'):
        logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    logging.warning("CUDA is not available. Using CPU only.")
    
# Path to the video to test (update this if needed)
video_path = str(Path('inputs').absolute() / "test_video.mp4")
if len(sys.argv) > 1:
    video_path = sys.argv[1]

# Check if the video exists
if not os.path.exists(video_path):
    logging.warning(f"Video not found at {video_path}")
    logging.info("Please provide a valid video path as an argument or place a test_video.mp4 in the inputs folder")
    # Look for any video in the inputs directory
    input_dir = Path('inputs')
    if input_dir.exists():
        video_files = list(input_dir.glob('*.mp4'))
        if video_files:
            video_path = str(video_files[0])
            logging.info(f"Found video: {video_path}")
        else:
            logging.error("No .mp4 files found in inputs directory")
            sys.exit(1)

# Add DeOldify to the Python path
sys.path.append(os.path.abspath('DeOldify'))

# Import DeOldify after CUDA check
try:
    from deoldify import device
    device.set(device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    from deoldify.visualize import get_video_colorizer
    
    logging.info("Successfully imported DeOldify")
    logging.info(f"DeOldify device set to: {device.device}")
    logging.info("Starting colorization test...")
    
    # Get the colorizer with GPU acceleration
    colorizer = get_video_colorizer(render_factor=21)
    
    # Colorize the video
    logging.info(f"Colorizing video: {video_path}")
    render_factor = 21  # Default value, adjust if needed
    result_path = colorizer.colorize_from_file_name(
        file_name=video_path, 
        render_factor=render_factor,
        watermarked=False
    )
    
    logging.info(f"Colorization complete! Result saved to: {result_path}")
    
except Exception as e:
    logging.error(f"Error during colorization: {str(e)}", exc_info=True)
    sys.exit(1)
