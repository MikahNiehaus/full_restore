#!/usr/bin/env python3
"""
GPU accelerator module for DeOldify and image restoration
This module forces GPU usage for both DeOldify colorization and image restoration
when imported at the beginning of your script or pipeline.
"""
import os
import sys
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class GPUAccelerator:
    """GPU acceleration manager for DeOldify and image restoration"""
    
    @staticmethod
    def setup_gpu():
        """Setup GPU acceleration for all modules"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available! GPU acceleration cannot be enabled.")
            logging.warning("Please check your PyTorch installation and GPU drivers.")
            return False
        
        # Display GPU information
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logging.info(f"GPU Accelerator: Using {gpu_name}")
        logging.info(f"CUDA Version: {cuda_version}")
        logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Enable cuDNN for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            logging.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            logging.info("cuDNN benchmark mode enabled for optimal performance")
        
        # Add DeOldify to path if needed
        deoldify_path = Path(__file__).parent / 'DeOldify'
        if deoldify_path.exists() and str(deoldify_path) not in sys.path:
            sys.path.append(str(deoldify_path))
            
        # Force DeOldify to use GPU
        try:
            from deoldify import device
            from deoldify.device_id import DeviceId
            device.set(DeviceId.GPU0)
            logging.info(f"DeOldify device set to GPU0 successfully")
        except ImportError:
            logging.warning("Could not import DeOldify. If you're using DeOldify, import this module first.")
        except Exception as e:
            logging.warning(f"Failed to set DeOldify device: {e}")
        
        return True

# Auto-initialize when imported
gpu_available = GPUAccelerator.setup_gpu()
