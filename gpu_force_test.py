#!/usr/bin/env python3
"""
GPU Force Test for DeOldify and Image Restorer

This script ensures that both DeOldify colorizer and Image Restorer 
are using GPU acceleration. It'll test:
1. PyTorch CUDA availability
2. DeOldify device setting
3. DeOldify colorizer model device
4. Image Restorer device
5. End-to-end processing with GPU

Usage:
    python gpu_force_test.py [test_image]
"""

import os
import sys
import time
import torch
import traceback
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Configure paths
current_dir = Path(__file__).parent
deoldify_dir = current_dir / "DeOldify"
sys.path.append(str(current_dir))
sys.path.append(str(deoldify_dir))

def print_section(title):
    """Print a formatted section title"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def test_gpu_usage():
    """Test GPU usage for all components"""
    print_section("FORCING GPU USAGE FOR ALL COMPONENTS")
    
    # Step 1: Check PyTorch GPU access
    print_section("Step 1: Checking PyTorch GPU access")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"SUCCESS: CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        # Enable cuDNN auto-tuning for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN benchmark mode enabled for optimal performance")
    else:
        print(f"ERROR: CUDA is not available. This test requires a GPU.")
        return False
    
    # Step 2: Configure DeOldify to use GPU
    print_section("Step 2: Setting DeOldify device to GPU")
    
    try:
        # IMPORTANT: This must be done before any other DeOldify imports
        from deoldify import device
        from deoldify.device_id import DeviceId
        
        # Force DeOldify to use GPU0
        device.set(DeviceId.GPU0)
        print(f"DeOldify device set to GPU0 successfully")
        print(f"DeOldify device current setting: {device.current()}")
        
        if device.current() == DeviceId.GPU0:
            print(f"SUCCESS: DeOldify is using GPU")
        else:
            print(f"ERROR: DeOldify is not using GPU")
            return False
    except Exception as e:
        print(f"ERROR: Failed to set DeOldify device to GPU: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Check if DeOldify colorizer model is on GPU
    print_section("Step 3: Checking DeOldify colorizer model")
    
    try:
        from deoldify.visualize import get_video_colorizer
        
        # Get the stable colorizer with high quality settings
        colorizer = get_video_colorizer(render_factor=40)  # Maximum quality
        
        # Check if the model is on GPU
        if hasattr(colorizer, "vis") and hasattr(colorizer.vis, "learn"):
            if hasattr(colorizer.vis.learn, "model"):
                model_device = next(colorizer.vis.learn.model.parameters()).device
                print(f"SUCCESS: DeOldify colorizer model is on {model_device} (device: {model_device.type})")
                
                if model_device.type == "cuda":
                    print(f"DeOldify colorizer is using GPU acceleration!")
                else:
                    print(f"ERROR: DeOldify colorizer model is not on CUDA device!")
                    return False
            else:
                print(f"ERROR: Cannot find model in colorizer.vis.learn")
                return False
        else:
            print(f"ERROR: Unexpected DeOldify colorizer structure")
            return False
    except Exception as e:
        print(f"ERROR: Failed to check DeOldify colorizer model: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Check if Image Restorer is using GPU
    print_section("Step 4: Testing Image Restorer GPU acceleration")
    
    try:
        from image_restorer import ImageRestorer
        
        # Force GPU usage
        restorer = ImageRestorer(device=torch.device('cuda'))
        
        # Check restorer device
        if hasattr(restorer, "device"):
            print(f"Image Restorer device: {restorer.device}")
            if restorer.device.type == "cuda":
                print(f"SUCCESS: Image Restorer is using GPU acceleration!")
            else:
                print(f"ERROR: Image Restorer is not using GPU!")
                return False
        else:
            print(f"ERROR: Image Restorer has no device attribute")
            return False
    except Exception as e:
        print(f"ERROR: Failed to check Image Restorer device: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Process a test image with both components
    print_section("Step 5: Testing end-to-end processing with GPU")
    
    # Find a test image (use the first image file in the inputs directory or DeOldify tests)
    test_image_path = None
    
    # Try to find an image in the inputs directory
    inputs_dir = Path("inputs")
    if inputs_dir.exists():
        for ext in ['.png', '.jpg', '.jpeg']:
            images = list(inputs_dir.glob(f"*{ext}"))
            if images:
                test_image_path = images[0]
                print(f"Found test image: {test_image_path}")
                break
    
    # If no test image found, try DeOldify test images
    if test_image_path is None:
        test_img = deoldify_dir / "test_images" / "image.png"
        if test_img.exists():
            test_image_path = test_img
            print(f"Using DeOldify test image: {test_image_path}")
    
    if test_image_path is None:
        print(f"WARNING: No test image found. Skipping end-to-end test.")
        return True
    
    # Process the test image
    try:
        import cv2
        from deoldify.visualize import ModelImageVisualizer, ColorizerFilter
        
        # Set output directory
        output_dir = Path("gpu_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Restore the image
        print("Testing Image Restorer with GPU...")
        start_time = time.time()
        restored_path = output_dir / "restored_image.png"
        
        # Load and restore image
        image = cv2.imread(str(test_image_path))
        if image is None:
            print(f"ERROR: Could not load test image {test_image_path}")
            return False
        
        restored = restorer.restore_image(image, str(restored_path))
        print(f"Image restoration completed in {time.time() - start_time:.2f} seconds")
        
        # 2. Colorize the image
        print("Testing DeOldify Colorizer with GPU...")
        start_time = time.time()
        colorized_path = output_dir / "colorized_image.png"
        
        # Get the visualizer from the colorizer and colorize directly
        vis = colorizer.vis
        colorized = vis.get_transformed_image(
            str(restored_path), 
            render_factor=40,
            watermarked=False
        )
        colorized.save(str(colorized_path))
        print(f"Image colorization completed in {time.time() - start_time:.2f} seconds")
        
        print(f"SUCCESS: End-to-end processing completed using GPU acceleration")
        print(f"Results saved to: {output_dir}")
        
        # Print CUDA memory stats
        print("\nCUDA Memory Stats:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"ERROR: Failed in end-to-end image processing: {e}")
        traceback.print_exc()
        return False
    
    print_section("GPU USAGE TEST COMPLETED SUCCESSFULLY")
    print("Both DeOldify colorizer and Image Restorer are using GPU acceleration!")
    return True

def main():
    print(f"=" * 60)
    print(f"GPU ACCELERATION TEST FOR DEOLDIFY AND IMAGE RESTORATION")
    print(f"=" * 60)
    
    # Check if NVIDIA GPU is available with CUDA
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU found. This test requires GPU acceleration.")
        sys.exit(1)
    
    # Print GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU Accelerator: Using {gpu_name}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
    print(f"[INFO] GPU Memory: {gpu_mem:.2f} GB")
    
    if torch.backends.cudnn.is_available():
        print(f"[INFO] cuDNN Version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] cuDNN benchmark mode enabled for optimal performance")
    
    # Run the GPU usage test
    try:
        success = test_gpu_usage()
        if not success:
            print("\nWARNING: Not all components are using GPU acceleration.")
            print("Please check the error messages above for details.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
