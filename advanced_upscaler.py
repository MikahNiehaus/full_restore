"""
Advanced image upscaler that uses OpenCV-based super-resolution for high-quality results
when AI upscaling can't be used. This file provides a drop-in replacement for the Real-ESRGAN upscaling.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import time

class AdvancedUpscaler:
    """Advanced image upscaling using multiple enhancement techniques"""
    
    def __init__(self, scale=4, use_edsr=True):
        """Initialize the upscaler
        
        Args:
            scale: Scaling factor (2, 3, or 4)
            use_edsr: Whether to use EDSR super-resolution model if available
        """
        self.scale = scale
        self.use_edsr = use_edsr
        self.sr_model = None
        
        # Try to load OpenCV's super-resolution models if available
        if use_edsr:
            try:
                self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
                
                # Define model path based on scale
                model_path = self._get_model_path(self.scale)
                
                # Download models if they don't exist
                if not os.path.exists(model_path):
                    print(f"[INFO] Downloading super-resolution model for {self.scale}x upscaling...")
                    self._download_models(self.scale)
                
                # Load the model
                if os.path.exists(model_path):
                    print(f"[INFO] Loading super-resolution model: {model_path}")
                    self.sr_model.readModel(model_path)
                    self.sr_model.setModel("edsr", self.scale)
                    print("[INFO] Super-resolution model loaded successfully")
                else:
                    print(f"[WARNING] Super-resolution model not found at: {model_path}")
                    self.use_edsr = False
            except Exception as e:
                print(f"[WARNING] Failed to initialize super-resolution: {e}")
                self.use_edsr = False
    
    def _get_model_path(self, scale):
        """Get the path to the super-resolution model"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Return the path based on scale
        return os.path.join(models_dir, f"EDSR_x{scale}.pb")
    
    def _download_models(self, scale):
        """Download the super-resolution models"""
        import urllib.request
        
        # URLs for different models
        model_urls = {
            2: "https://github.com/fannymonori/TF-EDSR/raw/master/EDSR_x2.pb",
            3: "https://github.com/fannymonori/TF-EDSR/raw/master/EDSR_x3.pb",
            4: "https://github.com/fannymonori/TF-EDSR/raw/master/EDSR_x4.pb"
        }
        
        if scale in model_urls:
            model_path = self._get_model_path(scale)
            try:
                urllib.request.urlretrieve(model_urls[scale], model_path)
                print(f"[INFO] Downloaded super-resolution model to: {model_path}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to download model: {e}")
                return False
        else:
            print(f"[ERROR] No model available for scale factor {scale}")
            return False
    
    def upscale(self, img):
        """Upscale an RGB image
        
        Args:
            img: RGB image as numpy array (h, w, 3)
            
        Returns:
            Upscaled RGB image as numpy array
        """
        # Make a copy of the image to avoid modifying the original
        img_copy = img.copy()
        
        # Try using EDSR model if available
        if self.use_edsr and self.sr_model is not None:
            try:
                print("[INFO] Upscaling image with EDSR super-resolution...")
                start_time = time.time()
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                
                # Apply super-resolution
                result_bgr = self.sr_model.upsample(img_bgr)
                
                # Convert back to RGB
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                
                print(f"[INFO] Super-resolution completed in {time.time() - start_time:.2f} seconds")
                return result
            except Exception as e:
                print(f"[WARNING] Super-resolution failed: {e}")
                print("[INFO] Falling back to advanced OpenCV upscaling")
        
        # Fallback to advanced OpenCV upscaling
        print("[INFO] Using advanced OpenCV upscaling...")
        start_time = time.time()
        
        # Step 1: Apply bilateral filter to preserve edges while reducing noise
        img_filtered = cv2.bilateralFilter(img_copy, 7, 50, 50)
        
        # Step 2: Apply detail enhancement
        img_details = cv2.detailEnhance(img_filtered, sigma_s=10, sigma_r=0.15)
        
        # Step 3: Upscale using Lanczos (higher quality than bicubic) or INTER_CUBIC
        h, w = img_details.shape[:2]
        img_upscaled = cv2.resize(img_details, (w * self.scale, h * self.scale), 
                                 interpolation=cv2.INTER_LANCZOS4)
        
        # Step 4: Apply edge-preserving filter to reduce artifacts
        img_upscaled = cv2.edgePreservingFilter(img_upscaled, flags=cv2.RECURS_FILTER, sigma_s=0.2, sigma_r=0.2)
        
        # Step 5: Apply unsharp mask for sharpening
        blur = cv2.GaussianBlur(img_upscaled, (0, 0), 3.0)
        img_upscaled = cv2.addWeighted(img_upscaled, 1.5, blur, -0.5, 0)
        
        print(f"[INFO] Advanced upscaling completed in {time.time() - start_time:.2f} seconds")
        
        return img_upscaled

# Function to use as a drop-in replacement for RealESRGANer.enhance
def enhance(img, scale=4):
    """Enhance an image using advanced upscaling (drop-in replacement for RealESRGANer)
    
    Args:
        img: RGB image as numpy array
        scale: Scaling factor
        
    Returns:
        Tuple of (enhanced_image, None) to match RealESRGANer API
    """
    upscaler = AdvancedUpscaler(scale=scale)
    result = upscaler.upscale(img)
    return result, None

# Test function
def test_upscaler(input_path, output_path, scale=4):
    """Test the upscaler on a single image"""
    from PIL import Image
    import cv2
    
    # Load test image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read image: {input_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Input image shape: {img_rgb.shape}")
    
    # Initialize upscaler
    upscaler = AdvancedUpscaler(scale=scale)
    
    # Enhance image
    print("Enhancing image...")
    output = upscaler.upscale(img_rgb)
    
    # Save result
    print(f"Output image shape: {output.shape}")
    output_img = Image.fromarray(output)
    output_img.save(output_path)
    print(f"Saved enhanced image to: {output_path}")
    
    return output

# Create a test image if not provided
def create_test_image(output_path):
    """Create a test image with various features to test upscaling"""
    import cv2
    import numpy as np
    
    # Create a simple test image (200x200)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    # Draw some shapes for testing detail enhancement
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 3)
    cv2.circle(img, (100, 100), 40, (0, 255, 0), 2)
    cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 2)
    cv2.putText(img, "Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add some gradients
    for i in range(200):
        cv2.line(img, (0, i), (50, i), (i, 200-i, i), 1)
    
    cv2.imwrite(output_path, img)
    print(f"Created test image: {output_path}")
    
    return img

if __name__ == "__main__":
    # Create a test image if not provided
    test_input = "inputs/test.png"
    if not os.path.exists(test_input):
        os.makedirs("inputs", exist_ok=True)
        create_test_image(test_input)
    
    # Test the upscaler
    test_upscaler(test_input, "test_output_advanced_upscaler.png", scale=4)
