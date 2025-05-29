"""
Real-ESRGAN model loader with proper architecture for realesr-general-x4v3.pth weights
"""

import os
import sys
import torch
import numpy as np
from torch import nn
from collections import OrderedDict

# Fix imports by applying the patch first
try:
    # Run the patch before importing Real-ESRGAN
    import importlib.util
    spec = importlib.util.spec_from_file_location("realesrgan_patch", 
                                                os.path.join(os.path.dirname(__file__), "realesrgan_patch.py"))
    realesrgan_patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realesrgan_patch)
    print("[INFO] Applied Real-ESRGAN compatibility patch")
except Exception as e:
    print(f"[WARNING] Could not apply Real-ESRGAN compatibility patch: {e}")

# Define a simple network architecture that matches the weights
class SimpleRealESRGANModel(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4):
        super(SimpleRealESRGANModel, self).__init__()
        self.scale = scale
        self.body = nn.ModuleList()
        
        # Create a simple sequential model with the same weight names
        # This model structure matches the weight keys in realesr-general-x4v3.pth
        for i in range(67):  # Based on examining the weight keys
            if i % 3 == 0 or i % 3 == 2:  # Create conv layers with bias
                self.body.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            else:  # Create conv layers without bias (just weights)
                self.body.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
                
        # First convolution to convert from input channels
        self.conv_first = nn.Conv2d(num_in_ch, 64, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        
        # Final convolution
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(64, num_out_ch, kernel_size=3, padding=1)
        
        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        # Process through body
        body_feat = feat
        for i, layer in enumerate(self.body):
            if i % 3 == 1:  # Apply activation after certain layers
                body_feat = self.lrelu(layer(body_feat))
            else:
                body_feat = layer(body_feat)
                
        # Residual connection
        feat = feat + body_feat
        
        # Upsampling
        feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
        
        # Final convolutions
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        return out

class RealESRGANer:
    """A wrapper for RealESRGAN model that applies super-resolution on images"""

    def __init__(self, scale, model, model_path, device=None, tile=0):
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Initialize the model
        self.model = model.to(self.device)
        
        # Load pre-trained weights
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the model has 'params' key (Real-ESRGAN format)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
            
        self.model.eval()
        self.tile_size = tile
        
        # For large images, we need tiling to avoid memory issues
        self.use_tiling = tile > 0

    def enhance(self, img):
        """Enhance an RGB image with shape (h, w, c)"""
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, c, h, w)
        
        with torch.no_grad():
            if self.use_tiling and max(img.shape[2:]) > self.tile_size:
                # Use tiling for large images
                output = self._tile_process(img)
            else:
                # Process the whole image at once
                output = self.model(img)
                
        # Convert back to numpy array
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (h, w, c)
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        
        return output, None

    def _tile_process(self, img):
        """Process large images tile by tile to avoid memory issues"""
        batch, channel, height, width = img.shape
        output = torch.zeros((batch, channel, height * self.scale, width * self.scale), device=self.device)
        
        # Set tile size and overlap
        tile = self.tile_size
        overlap = 32  # Overlap between tiles to avoid boundary artifacts
        
        # Process each tile
        for h in range(0, height, tile - overlap):
            h_end = min(h + tile, height)
            h_start = max(0, h_end - tile)
            
            for w in range(0, width, tile - overlap):
                w_end = min(w + tile, width)
                w_start = max(0, w_end - tile)
                
                # Extract tile
                tile_img = img[:, :, h_start:h_end, w_start:w_end]
                
                # Process tile
                tile_output = self.model(tile_img)
                
                # Calculate output coordinates
                out_h_start = h_start * self.scale
                out_h_end = h_end * self.scale
                out_w_start = w_start * self.scale
                out_w_end = w_end * self.scale
                
                # Merge tile back
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_output
                
        return output

# Test function
def test_real_esrgan_model(input_path, output_path):
    """Test the model on a single image"""
    from PIL import Image
    import cv2
    
    # Load test image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read image: {input_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Input image shape: {img_rgb.shape}")
    
    # Initialize model
    model = SimpleRealESRGANModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load weights
    model_path = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights', 'realesr-general-x4v3.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Initialize the enhancer
    upsampler = RealESRGANer(
        scale=4,
        model=model,
        model_path=model_path,
        tile=400,
        device=device
    )
    
    # Enhance image
    print("Enhancing image...")
    output, _ = upsampler.enhance(img_rgb)
    
    # Save result
    print(f"Output image shape: {output.shape}")
    output_img = Image.fromarray(output)
    output_img.save(output_path)
    print(f"Saved enhanced image to: {output_path}")
    
    return output

if __name__ == "__main__":
    # Create a test image if not provided
    test_input = "inputs/test.png"
    if not os.path.exists(test_input):
        os.makedirs("inputs", exist_ok=True)
        import cv2
        # Create a simple test image (200x200)
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Draw some shapes for testing detail enhancement
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 3)
        cv2.circle(img, (100, 100), 40, (0, 255, 0), 2)
        cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 2)
        cv2.putText(img, "Test", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(test_input, img)
        print(f"Created test image: {test_input}")
    
    # Test the model
    test_real_esrgan_model(test_input, "test_output_realesrgan.png")
