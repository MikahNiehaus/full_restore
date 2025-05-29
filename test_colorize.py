"""
Simple test script for the colorization functionality with proper CPU fallback
"""
import os
import sys
import torch
from unified_image_processor import ImageProcessor

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Create output directory
    os.makedirs("test_colorize_outputs", exist_ok=True)
      # Use a test image from DeOldify
    input_path = os.path.join("DeOldify", "test_images", "WaltWhitman.jpg")
    if not os.path.exists(input_path):
        print(f"Test image not found at: {input_path}")
        # Try alternative paths
        alt_paths = [
            os.path.join("inputs", "sample.jpg"),
            os.path.join("dummy", "test.jpg"),
            "dummy_image.jpg"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                input_path = alt_path
                print(f"Using alternative image: {input_path}")
                break
        else:
            print("No test images found. Cannot continue.")
            return
        
    # Setup processor and output path
    processor = ImageProcessor(output_dir="test_colorize_outputs")
    output_path = os.path.join("test_colorize_outputs", "WaltWhitman_colorized.png")
    
    # Try colorizing with default settings
    print("\n===== Testing with GPU if available =====")
    processor.colorize_image(input_path, output_path, artistic=True, render_factor=35)
    
    # Test explicit CPU fallback
    print("\n===== Testing with explicit CPU fallback =====")
    force_cpu_output = os.path.join("test_colorize_outputs", "WaltWhitman_colorized_cpu.png")
    
    # Temporarily modify the colorize_image method to force CPU
    original_colorize = processor.colorize_image
    
    # Create a wrapper function that forces CPU
    def force_cpu_colorize(input_path, output_path, artistic=True, render_factor=35):
        # Import needed modules
        from deoldify import device
        from deoldify.device_id import DeviceId
        
        # Force CPU device
        print("Explicitly setting device to CPU...")
        device.set(device=DeviceId.CPU)
        
        # Call the original method which will now use CPU
        return original_colorize(input_path, output_path, artistic, render_factor)
    
    # Replace with our CPU-forcing version
    processor.colorize_image = force_cpu_colorize
    
    # Test with CPU
    processor.colorize_image(input_path, force_cpu_output, artistic=True, render_factor=25)
    
    print("\n===== Testing with fallback method =====")
    fallback_output = os.path.join("test_colorize_outputs", "WaltWhitman_colorized_fallback.png")
    processor._fallback_colorize(input_path, fallback_output)
    
    print("\nAll outputs saved to test_colorize_outputs directory")

if __name__ == "__main__":
    main()
