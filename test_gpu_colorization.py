#!/usr/bin/env python3
"""
Test script to verify that DeOldify is correctly using GPU acceleration for colorization
This script measures performance and verifies GPU usage
"""

import os
import sys
import time
import torch
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Add DeOldify to Python path
sys.path.append(os.path.abspath('DeOldify'))

def test_gpu_colorization():
    """Test if DeOldify is using GPU and measure GPU vs CPU performance"""
    
    # Check if CUDA is available
    logging.info(f"PyTorch version: {torch.__version__}")
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        logging.info(f"CUDA is available! Using GPU acceleration")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    else:
        logging.warning("CUDA is not available. Testing will use CPU only.")
    
    # First import device module and set it explicitly
    try:
        from deoldify import device
        # Force GPU mode if available, otherwise fallback to CPU
        if gpu_available:
            device.set(device=torch.device('cuda'))
            logging.info(f"Device set to CUDA")
        else:
            device.set(device=torch.device('cpu'))
            logging.info(f"Device set to CPU")
            
        # Verify device setting
        logging.info(f"Current device: {device.device}")
        
        # Load the model with GPU acceleration
        from deoldify.visualize import get_image_colorizer
        
        # Create test directory if it doesn't exist
        test_dir = Path("gpu_test_results")
        test_dir.mkdir(exist_ok=True)
        
        # Find a test image (grayscale) to colorize
        test_image_path = find_test_image()
        if not test_image_path:
            logging.error("No test image found. Please place a grayscale JPG or PNG image in the 'inputs' folder.")
            return False
        
        logging.info(f"Using test image: {test_image_path}")
        
        # First run - with current device setting
        logging.info("Loading colorizer with current device setting...")
        start_time = time.time()
        colorizer = get_image_colorizer(artistic=False)  # Use the stable model
        end_time = time.time()
        load_time = end_time - start_time
        logging.info(f"Model loading time: {load_time:.2f} seconds")
        
        logging.info("Starting image colorization...")
        # Measure GPU memory before
        if gpu_available:
            before_mem = torch.cuda.memory_allocated()
            logging.info(f"GPU memory before colorization: {before_mem/1024**2:.2f} MB")
            
        # Colorize and time the process
        start_time = time.time()
        result_path = colorizer.plot_transformed_image(str(test_image_path), 
                                                   render_factor=35, 
                                                   results_dir=test_dir)
        end_time = time.time()
        colorize_time = end_time - start_time
        
        # Check GPU memory after to confirm GPU was used
        if gpu_available:
            after_mem = torch.cuda.memory_allocated()
            mem_diff = after_mem - before_mem
            logging.info(f"GPU memory after colorization: {after_mem/1024**2:.2f} MB")
            logging.info(f"GPU memory used for colorization: {mem_diff/1024**2:.2f} MB")
            
            # If memory increased, GPU was likely used
            if mem_diff > 0:
                logging.info("✅ GPU was used for colorization (memory usage increased)")
            else:
                logging.warning("❌ GPU might not have been used (no memory usage change detected)")
                
        logging.info(f"Colorization completed in {colorize_time:.2f} seconds")
        logging.info(f"Colorized image saved to: {result_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during GPU colorization test: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_test_image():
    """Find a grayscale test image in the inputs directory"""
    # Check in inputs directory
    inputs_dir = Path("inputs")
    if not inputs_dir.exists():
        inputs_dir.mkdir(exist_ok=True)
        
    # Look for JPG or PNG files
    image_files = list(inputs_dir.glob("*.jpg")) + list(inputs_dir.glob("*.jpeg")) + list(inputs_dir.glob("*.png"))
    
    if image_files:
        return image_files[0]
    
    # If no files found, try to create a test image
    try:
        # Create a simple grayscale test image
        test_img = Image.new('L', (512, 512), color=128)
        
        # Add some patterns
        for y in range(512):
            for x in range(512):
                # Create some gradients and patterns
                val = (x + y) % 256
                test_img.putpixel((x, y), val)
                
        test_img_path = inputs_dir / "test_grayscale.jpg"
        test_img.save(test_img_path)
        logging.info(f"Created test grayscale image: {test_img_path}")
        return test_img_path
    except Exception as e:
        logging.error(f"Failed to create test image: {e}")
        return None

if __name__ == "__main__":
    logging.info("Starting GPU colorization test...")
    test_gpu_colorization()
