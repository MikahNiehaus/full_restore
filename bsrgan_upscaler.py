"""
BSRGAN upscaler implementation for image and video restoration.
This model is designed for blind super-resolution - handling various degradations
like noise, blur, and compression artifacts.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F

# Define RRDBNet architecture (simplified for BSRGAN)
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        # First conv layer
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Body layers
        trunk = []
        for _ in range(nb):
            trunk.append(ResidualDenseBlock(nf))
        trunk.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.trunk = nn.Sequential(*trunk)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # HR conv layers
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.trunk(feat)
        feat = feat + body_feat
        
        # Upsampling
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # Final conv layers
        out = self.lrelu(self.HRconv(feat))
        out = self.conv_last(out)
        return out


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


class BSRGANUpscaler:
    """A wrapper class for BSRGAN that applies super-resolution on images"""

    def __init__(self, model_path=None, device=None, scale=4, tile=0):
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.tile_size = tile
        self.use_tiling = tile > 0
        
        # Initialize the model
        self.model = RRDBNet(scale=scale).to(self.device)
        
        # Download the model if needed and not provided
        if model_path is None:
            model_path = self._download_model()
            
        # Load pre-trained weights
        self._load_model(model_path)
        self.model.eval()

    def _download_model(self):
        """Download BSRGAN model if not available locally"""
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'BSRGAN.pth')
        
        if not os.path.exists(model_path):
            import gdown
            
            try:
                print(f"Downloading BSRGAN model to {model_path}...")
                # BSRGAN model public URL (this is a direct link to the model file)
                # Note: This URL may need to be updated if the file is moved
                url = 'https://github.com/yangxy/BSRGAN/raw/main/model_zoo/BSRGAN.pth'
                gdown.download(url, model_path, quiet=False)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                # Provide a secondary download method if gdown fails
                import requests
                print("Trying alternative download method...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    print("Model downloaded successfully.")
                else:
                    print(f"Failed to download model. Please download it manually from {url}")
                    model_path = None
        else:
            print(f"BSRGAN model already exists at {model_path}")
            
        return model_path

    def _load_model(self, model_path):
        """Load the BSRGAN model from a .pth file"""
        try:
            loadnet = torch.load(model_path, map_location=self.device)
            
            # Handle different model saving formats
            if 'params' in loadnet:
                self.model.load_state_dict(loadnet['params'], strict=True)
            elif 'model_state_dict' in loadnet:
                self.model.load_state_dict(loadnet['model_state_dict'], strict=True)
            else:
                self.model.load_state_dict(loadnet, strict=True)
                
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def enhance(self, img):
        """Enhance an RGB image with shape (h, w, c)"""
        if img.shape[2] > 3:
            # Handle alpha channel if present
            has_alpha = True
            alpha = img[:, :, 3]
            img = img[:, :, :3]
        else:
            has_alpha = False
        
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
        
        # Reattach alpha channel if present
        if has_alpha:
            # Resize alpha to match new dimensions
            alpha_resized = cv2.resize(
                alpha, 
                (output.shape[1], output.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            output = np.concatenate([output, alpha_resized[:, :, np.newaxis]], axis=2)
        
        return output, None

    def _tile_process(self, img):
        """Process large images tile by tile to avoid memory issues"""
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        
        output = torch.zeros((batch, channel, output_height, output_width), device=self.device)
        
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
def test_bsrgan_upscaler(input_path, output_path, scale=4):
    """Test the BSRGAN upscaler on a single image"""
    import cv2
    from PIL import Image
    
    # Load test image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {input_path}")
        return
        
    # Convert BGR to RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Handle BGRA (with alpha channel)
        tmp = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        img = np.concatenate([tmp, img[:, :, 3:]], axis=2)
    
    print(f"Input image shape: {img.shape}")
    
    # Initialize upscaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    upscaler = BSRGANUpscaler(device=device, scale=scale, tile=800)
    
    # Enhance image
    print("Enhancing image with BSRGAN...")
    output, _ = upscaler.enhance(img)
    
    # Save result
    print(f"Output image shape: {output.shape}")
    
    # Convert numpy array to PIL Image and save
    if output.shape[2] == 4:  # With alpha channel
        output_img = Image.fromarray(output, 'RGBA')
    else:
        output_img = Image.fromarray(output)
        
    output_img.save(output_path)
    print(f"Saved enhanced image to: {output_path}")
    
    return output


if __name__ == "__main__":
    # Create a test image if not provided
    import argparse
    
    parser = argparse.ArgumentParser(description="BSRGAN Upscaler")
    parser.add_argument("-i", "--input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path", default="bsrgan_output.png")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscaling factor (default: 4)")
    args = parser.parse_args()
    
    if args.input:
        test_bsrgan_upscaler(args.input, args.output, args.scale)
    else:
        # Create test image if no input provided
        test_input = "inputs/test_bsrgan.png"
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
        
        test_bsrgan_upscaler(test_input, "bsrgan_output.png")
