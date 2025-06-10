#!/usr/bin/env python3
"""
Simple test script for DeOldify GPU colorization
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

def test_colorization():
    # Check CUDA availability
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA is available! Using GPU acceleration")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        device_name = torch.cuda.get_device_name(0)
    else:
        logging.warning("CUDA is not available. Using CPU only.")
        device_name = "CPU"
    
    # Set device
    try:
        from deoldify import device
        from deoldify.device_id import DeviceId
        
        # Set the device to GPU0 if available, otherwise CPU
        if torch.cuda.is_available():
            device.set(DeviceId.GPU0)
            logging.info("Device set to GPU0 successfully")
        else:
            device.set(DeviceId.CPU)
            logging.info("Device set to CPU")
            
        logging.info(f"DeOldify device current: {device.current()}")
    except Exception as e:
        logging.error(f"Error setting device: {str(e)}")
        return False
    
    # Import and initialize the video colorizer
    try:
        from deoldify.visualize import get_stable_video_colorizer
        
        start_time = time.time()
        logging.info("Initializing video colorizer model...")
        
        # Check GPU memory before model load
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated(0) / (1024**2)
            logging.info(f"GPU memory before model load: {before_mem:.2f} MB")
        
        # Get the colorizer
        colorizer = get_stable_video_colorizer(render_factor=21)
        
        # Check GPU memory after model load to confirm GPU usage
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated(0) / (1024**2)
            logging.info(f"GPU memory after model load: {after_mem:.2f} MB")
            logging.info(f"GPU memory increase: {after_mem - before_mem:.2f} MB")
            
        end_time = time.time()
        logging.info(f"Colorizer loaded in {end_time - start_time:.2f} seconds")
        
        # Check if model is on GPU
        for name, param in colorizer.vis.filter.filters[0].learn.model.named_parameters():
            logging.info(f"Model parameter '{name}' device: {param.device}")
            break
            
        # Look for test image to colorize
        test_image_path = None
        for test_dir in ['inputs', 'test_images', '.']:
            for ext in ['.jpg', '.png', '.jpeg']:
                candidates = list(Path(test_dir).glob(f"*{ext}"))
                if candidates:
                    test_image_path = str(candidates[0])
                    break
            if test_image_path:
                break
                
        if not test_image_path:
            logging.warning("No test image found. Creating a sample image...")
            # Create a simple grayscale test image
            import numpy as np
            from PIL import Image
            img = np.zeros((256, 256), dtype=np.uint8)
            img[64:192, 64:192] = 200  # White square in the middle
            test_image = Image.fromarray(img)
            test_image_path = "test_gray.jpg"
            test_image.save(test_image_path)
            
        # Test colorizing a single image
        logging.info(f"Colorizing test image: {test_image_path}")
        start_time = time.time()
        result_path = colorizer.vis.plot_transformed_image(
            path=test_image_path,
            render_factor=21,
            watermarked=False
        )
        end_time = time.time()
        logging.info(f"Image colorization completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Result saved to: {result_path}")
        
        # Check if result exists
        if Path(result_path).exists():
            logging.info(f"Successfully colorized image using {device_name}")
            return True
        else:
            logging.error("Failed to save colorized image")
            return False
            
    except Exception as e:
        logging.error(f"Error in colorization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
if __name__ == "__main__":
    test_colorization()
