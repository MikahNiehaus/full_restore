"""
Test script for the unified image processor to verify our CPU fallback mechanism
"""
import os
import sys
import time
from unified_image_processor import ImageProcessor
from PIL import Image
import shutil

def main():
    # Create a processor instance
    processor = ImageProcessor(output_dir="test_outputs")
    os.makedirs("test_outputs", exist_ok=True)
    
    # Look for test images
    test_images_dir = os.path.join("DeOldify", "test_images")
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        return
        
    # Get a sample image
    sample_images = [
        os.path.join(test_images_dir, "WaltWhitman.jpg"), 
        os.path.join(test_images_dir, "migrant_mother.jpg"),
        os.path.join(test_images_dir, "Chief.jpg")
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            # Create output paths
            base_name = os.path.basename(img_path)
            name_only = os.path.splitext(base_name)[0]
            
            colorized_path = os.path.join("test_outputs", f"{name_only}_colorized.png")
            
            print(f"\n===== Testing with image: {base_name} =====")
            # Make a copy of the image to ensure we're not overwriting the original
            temp_path = os.path.join("test_outputs", base_name)
            shutil.copy(img_path, temp_path)
            
            # Test colorization
            print("Testing colorization...")
            start_time = time.time()
            colorize_result = processor.colorize_image(temp_path, colorized_path)
            elapsed = time.time() - start_time
            
            print(f"Colorization {'succeeded' if colorize_result else 'failed'} in {elapsed:.2f} seconds")
            
            if colorize_result:
                # Check if the file exists and looks colorized
                try:
                    if os.path.exists(colorized_path):
                        img = Image.open(colorized_path)
                        print(f"Output image mode: {img.mode}")
                        print(f"Output image size: {img.size}")
                        print(f"Output image saved to: {colorized_path}")
                    else:
                        print(f"Error: Output file {colorized_path} not found")
                except Exception as e:
                    print(f"Error checking colorized image: {e}")

if __name__ == "__main__":
    main()
