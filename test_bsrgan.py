"""
Test script for the BSRGAN upscaler
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Import the BSRGAN upscaler
from bsrgan_upscaler import BSRGANUpscaler

# Create a test image if not provided
test_input = "inputs/test_bsrgan.png"
os.makedirs("inputs", exist_ok=True)

# Create a simple test image (200x200)
if not os.path.exists(test_input):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Draw some shapes for testing detail enhancement
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 3)
    cv2.circle(img, (100, 100), 40, (0, 255, 0), 2)
    cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 2)
    cv2.putText(img, "Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(test_input, img)
    print(f"Created test image: {test_input}")

# Load the test image
img = cv2.imread(test_input)
if img is None:
    print(f"Failed to read image: {test_input}")
    sys.exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Input image shape: {img_rgb.shape}")

# Initialize the BSRGAN upscaler
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    upscaler = BSRGANUpscaler(device=device)
    
    # Enhance the image
    print("Enhancing image with BSRGAN...")
    output, _ = upscaler.enhance(img_rgb)
    
    # Save the result
    output_path = "bsrgan_test_output.png"
    print(f"Output image shape: {output.shape}")
    output_img = Image.fromarray(output)
    output_img.save(output_path)
    print(f"Saved enhanced image to: {output_path}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Test OpenCV fallback upscaling for comparison
    print("Falling back to OpenCV upscaling for comparison...")
    height, width = img.shape[:2]
    scale = 4  # Same scale as BSRGAN
    upscaled = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    
    # Apply detail enhancement
    enhanced_frame = cv2.detailEnhance(upscaled, sigma_s=10, sigma_r=0.15)
    
    # Sharpen the image a bit
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)
    
    cv2.imwrite("opencv_test_output.png", enhanced_frame)
    print("Saved OpenCV enhanced image to: opencv_test_output.png")
