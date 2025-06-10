#!/usr/bin/env python3
"""
GPU Test Script for Image Restorer Module

This script tests the GPU acceleration functionality of the image restorer.
It loads a sample image, processes it with GPU, and verifies the performance.
"""
import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed image restorer
from image_restorer_fixed import ImageRestorer

def test_gpu_restoration(image_path=None, output_dir=None):
    """Test the GPU-accelerated image restoration"""
    print("\n===== Testing GPU-accelerated Image Restoration =====")
    
    # Find a test image if not provided
    if not image_path:
        # Look in common directories for test images
        possible_dirs = ['inputs', 'temp_video_frames', '.']
        found = False
        
        for dir_path in possible_dirs:
            if not Path(dir_path).exists():
                continue
                
            images = list(Path(dir_path).glob("*.jpg")) + list(Path(dir_path).glob("*.png"))
            if images:
                image_path = str(images[0])
                print(f"Using test image: {image_path}")
                found = True
                break
                
        if not found:
            print("No test images found. Creating a test image...")
            # Create a simple test image
            test_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (100, 100), (400, 400), (255, 255, 255), -1)
            cv2.circle(test_img, (250, 250), 100, (0, 0, 0), -1)
            # Add some noise
            noise = np.random.normal(0, 25, test_img.shape).astype(np.uint8)
            test_img = cv2.add(test_img, noise)
            image_path = "test_image.jpg"
            cv2.imwrite(image_path, test_img)
            print(f"Created test image: {image_path}")
    
    # Create output directory if not provided
    if not output_dir:
        output_dir = "restorer_test_results"
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    # Ensure image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return False
    
    # Test GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"GPU detected: {device_name}")
        print(f"CUDA version: {cuda_version}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Warning: CUDA not available. Testing will use CPU which will be much slower.")
        print("If you have a CUDA-compatible GPU, please ensure your drivers and PyTorch CUDA are properly installed.")
    
    # List models to test
    models_to_test = ['RealESRGAN_x4plus']
    if Path("c:/prj/full_restore/models/realesr-general-x4v3.pth").exists():
        models_to_test.append('realesr-general-x4v3')
    
    # Test with each model
    results = {}
    for model_name in models_to_test:
        print(f"\nTesting with model: {model_name}")
        
        # Initialize restorer with GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_str = "GPU"
        else:
            device = torch.device('cpu')
            device_str = "CPU"
            
        # Create restorer with specified device
        restorer = ImageRestorer(
            device=device,
            model_name=model_name,
            denoise_strength=0.5
        )
        
        # Time the restoration process
        start_time = time.time()
        
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_{model_name}_restored{Path(image_path).suffix}")
        try:
            # Process the image
            print(f"Processing image with {device_str}...")
            restorer.restore_image(
                image_path=image_path,
                output_path=output_path,
                save_sharpened_path=os.path.join(output_dir, f"{Path(image_path).stem}_{model_name}_sharpened{Path(image_path).suffix}")
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify output was created
            if Path(output_path).exists():
                filesize = Path(output_path).stat().st_size / (1024 * 1024)  # MB
                print(f"Success! Restored image saved to: {output_path}")
                print(f"File size: {filesize:.2f} MB")
                print(f"Processing time: {processing_time:.2f} seconds using {device_str}")
                
                # Calculate throughput
                input_image = cv2.imread(image_path)
                h, w = input_image.shape[:2]
                mpixels = (h * w) / 1000000
                throughput = mpixels / processing_time
                
                print(f"Image size: {w}x{h} ({mpixels:.2f} megapixels)")
                print(f"Throughput: {throughput:.2f} MP/s")
                
                results[model_name] = {
                    "success": True,
                    "device": device_str,
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "output_path": output_path
                }
            else:
                print(f"Failed: Output file not created at {output_path}")
                results[model_name] = {
                    "success": False,
                    "device": device_str,
                    "error": "Output file not created"
                }
        except Exception as e:
            print(f"Error during restoration with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                "success": False,
                "device": device_str,
                "error": str(e)
            }
    
    # Print summary
    print("\n===== GPU Restoration Test Summary =====")
    for model_name, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"{model_name}: {status} on {result['device']}")
        if result["success"]:
            print(f"  Time: {result['processing_time']:.2f}s, Throughput: {result['throughput']:.2f} MP/s")
        else:
            print(f"  Error: {result['error']}")
    
    # Overall success if at least one model worked
    overall_success = any(result["success"] for result in results.values())
    print(f"\nOverall GPU restoration test: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    test_gpu_restoration(image_path, output_dir)
