"""
Test script to verify Real-ESRGAN integration is working.
"""
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch

# Import the patch first
print("Loading compatibility patch...")
import importlib.util
spec = importlib.util.spec_from_file_location("realesrgan_patch", 
                                            os.path.join(os.path.dirname(__file__), "realesrgan_patch.py"))
realesrgan_patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(realesrgan_patch)
print("[INFO] Applied Real-ESRGAN compatibility patch")

# Set paths
test_input = "inputs/test.png"
test_output_ai = "test_output_ai.png"
test_output_cv = "test_output_cv.png"

# Create test image if it doesn't exist
if not os.path.exists(test_input):
    os.makedirs("inputs", exist_ok=True)
    # Create a simple test image (200x200)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Draw some shapes for testing detail enhancement
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 3)
    cv2.circle(img, (100, 100), 40, (0, 255, 0), 2)
    cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 2)
    cv2.putText(img, "Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(test_input, img)
    print(f"Created test image: {test_input}")

print(f"Reading test image: {test_input}")
img = cv2.imread(test_input)
if img is None:
    print(f"Failed to read image: {test_input}")
    sys.exit(1)

# Convert to RGB (PIL format) for Real-ESRGAN
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

try:
    print("Importing Real-ESRGAN modules...")
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    # Set up the model
    print("Setting up Real-ESRGAN model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Model path
    model_path = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights', 'realesr-general-x4v3.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize upsampler with parameters
    upsampler = RealESRGANer(
        scale=4,
        model=model,
        model_path=model_path,
        dni_weight=None,
        device=device,
        tile=400  # Tiling helps with memory consumption
    )
    
    # Process the image
    print("Upscaling with Real-ESRGAN...")
    img_array = np.array(pil_img)
    output, _ = upsampler.enhance(img_array)
    
    # Save the result
    output_img = Image.fromarray(output)
    output_img.save(test_output_ai, "PNG")
    print(f"AI upscaled image saved to: {test_output_ai}")
    
    # Success!
    print("Real-ESRGAN upscaling successful!")
    
except Exception as e:
    print(f"Real-ESRGAN failed: {e}")
    import traceback
    traceback.print_exc()
    
# Also try OpenCV upscaling for comparison
print("Upscaling with OpenCV for comparison...")
height, width = img.shape[:2]
upscaled = cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(test_output_cv, upscaled)
print(f"OpenCV upscaled image saved to: {test_output_cv}")
