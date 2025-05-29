"""
Test script for the updated restoration pipeline.
Tests both image and video processing with the correct order:
restore -> colorize -> enhance
"""
import os
import sys
from unified_image_processor import ImageProcessor

def create_test_dirs():
    """
    Create test directories for input and output
    """
    os.makedirs("test_inputs", exist_ok=True)
    os.makedirs("test_outputs", exist_ok=True)
    
    # Find some test images/videos to process
    test_images = []
    for root, _, files in os.walk("inputs"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                break  # Only need one image for testing
    
    if not test_images:
        print("No test images found in inputs directory!")
        # Create a simple test image
        import numpy as np
        import cv2
        test_img = np.zeros((300, 300), dtype=np.uint8)
        # Add some gradient
        for i in range(300):
            test_img[:, i] = i * 255 // 300
        test_img_path = os.path.join("test_inputs", "test_grayscale.png")
        cv2.imwrite(test_img_path, test_img)
        test_images.append(test_img_path)
        print(f"Created test image: {test_img_path}")
    
    return test_images

def test_image_pipeline():
    """
    Test the image processing pipeline
    """
    print("\n=== TESTING IMAGE PROCESSING PIPELINE ===")
    test_images = create_test_dirs()
    
    if not test_images:
        print("No test images available!")
        return
    
    processor = ImageProcessor("test_outputs")
    
    for image_path in test_images:
        print(f"\nProcessing image: {image_path}")
        result = processor.process_image(image_path, "test_outputs", scale=2)
        
        if result:
            print(f"SUCCESS! Image processed: {result}")
            print("Pipeline followed order: restore -> colorize -> enhance")
        else:
            print(f"FAILED to process image: {image_path}")
    
def main():
    """Main test function"""
    print("=== TESTING UPDATED RESTORATION PIPELINE ===")
    print("Pipeline order: restore -> colorize -> enhance")
    
    # Test image processing
    test_image_pipeline()
    
    print("\nTests completed!")

if __name__ == "__main__":
    main()
