#!/usr/bin/env python3
"""
Test script to verify that DeOldify is using GPU acceleration for colorization
"""
import torch
import logging
import sys
import os
import time
from pathlib import Path

# Add DeOldify to Python path
sys.path.append(os.path.abspath('DeOldify'))

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

# Import DeOldify after CUDA check
try:
    # First import device and set it to GPU
    from deoldify import device
    from deoldify.device_id import DeviceId
    # Try both ways to set the device
    try:
        # Try using DeviceId enum first - it's the most reliable method
        if torch.cuda.is_available():
            device.set(DeviceId.GPU0)
            logging.info(f"Set device to GPU0 using DeviceId enum")
        else:
            device.set(DeviceId.CPU)
            logging.info(f"Set device to CPU using DeviceId enum")
    except Exception as e:
        logging.warning(f"Setting device with DeviceId failed: {e}")
        # Fall back to direct method
        try:
            import torch
            if torch.cuda.is_available():
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                logging.info("Set CUDA_VISIBLE_DEVICES=0 directly")
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                logging.info("Set CUDA_VISIBLE_DEVICES='' directly")
        except Exception as e2:
            logging.error(f"Both device setting methods failed: {e2}")
    # Now import the visualize module for colorization
    from deoldify.visualize import get_video_colorizer
    
    logging.info("Successfully imported DeOldify")
    logging.info(f"DeOldify device set to: {device.current()}")
    
    # Check if GPU is really being used by PyTorch
    if torch.cuda.is_available():
        # Print GPU memory usage before loading model
        logging.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    logging.info("Starting colorization test...")
    # Get the colorizer with GPU acceleration
    start_time = time.time()
    colorizer = get_video_colorizer(render_factor=21)
    end_time = time.time()
    logging.info(f"Colorizer initialization took {end_time - start_time:.2f} seconds")
    
    # Check GPU memory after model is loaded to confirm GPU usage
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated after model loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"GPU memory reserved after model loading: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Check if model is on GPU by examining a parameter
        import inspect
        for name, param in colorizer.vis.filter.filters[0].learn.model.named_parameters():
            logging.info(f"Model parameter '{name}' is on device: {param.device}")
            break
    
    # Colorize the video
    logging.info(f"Colorizing video: {video_path}")
    start_time = time.time()
    result_path = colorizer.colorize_from_file_name(
        file_name=video_path, 
        render_factor=21,
        watermarked=False
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    logging.info(f"Colorization complete in {processing_time:.2f} seconds!")
    logging.info(f"Result saved to: {result_path}")
    
except Exception as e:
    logging.error(f"Error during colorization: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
