#!/usr/bin/env python3
"""
Test script to verify that image restoration is using GPU acceleration properly
"""
import torch
import logging
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

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
        logging.info(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
else:
    logging.warning("CUDA is not available. Using CPU only.")
    
# Path to an image to test (update this if needed)
image_path = str(Path('inputs').absolute() / "test_image.jpg")
if len(sys.argv) > 1:
    image_path = sys.argv[1]

# Check if the image exists
if not os.path.exists(image_path):
    logging.warning(f"Image not found at {image_path}")
    logging.info("Please provide a valid image path as an argument or place a test_image.jpg in the inputs folder")
    # Look for any image in the inputs directory
    input_dir = Path('inputs')
    if input_dir.exists():
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(input_dir.glob(f'*{ext}')))
        if image_files:
            image_path = str(image_files[0])
            logging.info(f"Found image: {image_path}")
        else:
            logging.error("No image files found in inputs directory")
            sys.exit(1)

# Import the image restorer
try:
    from image_restorer import ImageRestorer
    
    # Print GPU memory usage before loading model
    if torch.cuda.is_available():
        logging.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    logging.info("Starting restoration test...")
    
    # Initialize the image restorer - explicitly set device to GPU if available
    start_time = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    restorer = ImageRestorer(device=device)
    end_time = time.time()
    logging.info(f"Restorer initialization took {end_time - start_time:.2f} seconds")
    
    # Check GPU memory after model is loaded
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated after model loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"GPU memory reserved after model loading: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Restore the image
    logging.info(f"Restoring image: {image_path}")
    output_path = Path("result_images") / f"{Path(image_path).stem}_restored{Path(image_path).suffix}"
    output_path.parent.mkdir(exist_ok=True)
    
    start_time = time.time()
    # Process the image
    restored_image = restorer.restore_image(
        image_path=image_path, 
        output_path=str(output_path),
        log_pipeline=True
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    logging.info(f"Restoration complete in {processing_time:.2f} seconds!")
    logging.info(f"Result saved to: {output_path}")
    
    # Check final GPU memory usage
    if torch.cuda.is_available():
        logging.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logging.info(f"Final GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    # Display performance summary
    h, w = restored_image.shape[:2]
    pixels = h * w
    logging.info(f"Image dimensions: {w}x{h} ({pixels/1000000:.2f} megapixels)")
    pixels_per_second = pixels / processing_time
    logging.info(f"Processing speed: {pixels_per_second/1000000:.2f} megapixels/second")
    
except Exception as e:
    logging.error(f"Error during restoration: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
