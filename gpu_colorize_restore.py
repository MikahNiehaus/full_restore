#!/usr/bin/env python3
"""
GPU-accelerated colorization and restoration tester.
This script checks if GPU is being properly used for both colorization and restoration.
"""
import os
import sys
import time
import torch
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Add DeOldify directory to sys.path
sys.path.append(os.path.abspath('DeOldify'))

# First check GPU availability
logging.info(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    logging.info(f"CUDA is available! Using GPU acceleration")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    logging.info(f"cuDNN benchmark mode enabled: {torch.backends.cudnn.benchmark}")
else:
    logging.warning("CUDA is not available. Using CPU only.")
    sys.exit(1)

# First set DeOldify device to GPU
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    
    device.set(DeviceId.GPU0)
    logging.info(f"DeOldify device set to: {device.current()}")
    
    # Test that it's properly set
    is_gpu = device.is_gpu()
    logging.info(f"DeOldify using GPU: {is_gpu}")
    
    if not is_gpu:
        logging.error("DeOldify is not using GPU despite setting to GPU0!")
        sys.exit(1)
    
except Exception as e:
    logging.error(f"Error setting DeOldify device: {e}", exc_info=True)
    sys.exit(1)

# Test DeOldify colorizer
try:
    from deoldify.visualize import get_video_colorizer
    logging.info("Successfully imported DeOldify colorizer")
    
    colorizer = get_video_colorizer(render_factor=40)  # Use high quality
    logging.info("Successfully created DeOldify colorizer")
    
    # Check that the model is on GPU
    try:
        device_type = next(colorizer.vis.filter.filters[0].learn.model.parameters()).device.type
        logging.info(f"DeOldify colorizer model device: {device_type}")
        
        if device_type != 'cuda':
            logging.error("DeOldify colorizer model is not on CUDA device!")
        else:
            logging.info("DeOldify colorizer is correctly using GPU")
    except Exception as e:
        logging.error(f"Error checking colorizer device: {e}", exc_info=True)
except Exception as e:
    logging.error(f"Error creating DeOldify colorizer: {e}", exc_info=True)

# Test image restorer
try:
    from image_restorer import ImageRestorer
    logging.info("Successfully imported ImageRestorer")
    
    # Create restorer with explicit GPU device
    cuda_device = torch.device('cuda')
    restorer = ImageRestorer(device=cuda_device)
    
    # Check device type
    logging.info(f"ImageRestorer device: {restorer.device}")
    
    if restorer.device.type != 'cuda':
        logging.error("ImageRestorer is not using CUDA device!")
    else:
        logging.info("ImageRestorer is correctly using GPU")
except Exception as e:
    logging.error(f"Error testing ImageRestorer: {e}", exc_info=True)

print("\n=== GPU USAGE TEST RESULTS ===")
print("If you see no errors above and both components report 'correctly using GPU',")
print("then the system is properly configured to use GPU acceleration!")
print("You can now use 'python simple_run.py' to process videos with GPU acceleration.")
