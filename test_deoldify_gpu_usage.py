#!/usr/bin/env python3
"""
Test script to verify that DeOldify is using GPU for both colorization and restoration
"""
import os
import sys
import torch
import traceback
from pathlib import Path

# Import our GPU accelerator to force both components to use GPU
from force_gpu import gpu_available

# Configure path for DeOldify
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeOldify'))

def print_separator():
    print("\n" + "="*60 + "\n")

# Main test function
def test_gpu_usage():
    print_separator()
    print("TESTING GPU USAGE FOR DEOLDIFY AND IMAGE RESTORATION")
    print_separator()
    
    # Step 1: Check if PyTorch can see the GPU
    print("Step 1: Checking PyTorch GPU access")
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! GPU acceleration cannot be enabled.")
        print("Please check your PyTorch installation and GPU drivers.")
        return False
    
    # Show GPU information
    device_count = torch.cuda.device_count()
    cuda_version = torch.version.cuda
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
    
    print(f"SUCCESS: CUDA is available")
    print(f"CUDA version: {cuda_version}")
    print(f"GPU device: {device_name}")
    print(f"Number of CUDA devices: {device_count}")
    
    # Step 2: Check if DeOldify is using GPU
    print_separator()
    print("Step 2: Checking if DeOldify is using GPU")
    
    try:
        from deoldify import device
        from deoldify.device_id import DeviceId
        
        print(f"DeOldify device current setting: {device.current()}")
        is_gpu = device.is_gpu()
        
        if is_gpu:
            print(f"SUCCESS: DeOldify is using GPU")
        else:
            print(f"WARNING: DeOldify is not using GPU, attempting to set it now")
            device.set(DeviceId.GPU0)
            is_gpu = device.is_gpu()
            if is_gpu:
                print(f"SUCCESS: DeOldify is now using GPU")
            else:
                print(f"ERROR: Failed to set DeOldify to use GPU")
    except Exception as e:
        print(f"ERROR: Failed to check DeOldify GPU usage: {e}")
        traceback.print_exc()
    
    # Step 3: Create a colorizer and check if its model is on GPU
    print_separator()
    print("Step 3: Checking if DeOldify colorizer model is on GPU")
    
    try:
        from deoldify.visualize import get_video_colorizer
        
        colorizer = get_video_colorizer(render_factor=40)
        
        # Check if model is on GPU by examining one of its parameters
        device_type = next(colorizer.vis.filter.filters[0].learn.model.parameters()).device.type
        
        if device_type == 'cuda':
            print(f"SUCCESS: DeOldify colorizer model is on GPU (device: {device_type})")
        else:
            print(f"ERROR: DeOldify colorizer model is on {device_type}, not on GPU!")
    except Exception as e:
        print(f"ERROR: Failed to check colorizer model device: {e}")
        traceback.print_exc()
    
    # Step 4: Check image restorer
    print_separator()
    print("Step 4: Checking if Image Restorer is using GPU")
    
    try:
        from image_restorer import ImageRestorer
        
        # Create with explicit GPU device
        cuda_device = torch.device('cuda')
        restorer = ImageRestorer(device=cuda_device)
        
        print(f"Image Restorer device: {restorer.device}")
        
        if restorer.device.type == 'cuda':
            print(f"SUCCESS: Image Restorer is using GPU (device: {restorer.device})")
        else:
            print(f"ERROR: Image Restorer is using {restorer.device}, not GPU!")
    except Exception as e:
        print(f"ERROR: Failed to check Image Restorer device: {e}")
        traceback.print_exc()
    
    print_separator()
    print("GPU USAGE TEST COMPLETED")
    print_separator()
    
    return True

if __name__ == "__main__":
    test_gpu_usage()
