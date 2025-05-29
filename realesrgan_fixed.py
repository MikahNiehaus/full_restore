"""
Real-ESRGAN model loader that properly loads the realesr-general-x4v3.pth weights.
The model architecture is explicitly created to match the weight structure.
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

# Define a SRVGGNetCompact model that exactly matches the weights structure
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='leakyrelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # First conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # Activation
        self.body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Main body
        for i in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if (i + 1) % 3 == 0:  # Add activation only after 3rd, 6th, 9th layers etc.
                self.body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Upsampling
        if self.upscale == 4:
            self.body.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            self.body.append(nn.PixelShuffle(2))
            self.body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.body.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            self.body.append(nn.PixelShuffle(2))
            self.body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif self.upscale == 2:
            self.body.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            self.body.append(nn.PixelShuffle(2))
            self.body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise ValueError(f'Unsupported upscale factor: {self.upscale}')

        # Output conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.body):
            out = layer(out)
        return out

    def load_state_dict_custom(self, state_dict, strict=True):
        """Load state dict directly into the body modules"""
        own_state = self.state_dict()
        # Map model weights to the body modules
        for i, (name, param) in enumerate(own_state.items()):
            key_name = name.replace('body.', '', 1)  # Remove 'body.' prefix
            mapped_key = f'body.{i}.{key_name}' if '.' in key_name else f'body.{i}'
            
            # Debugging
            print(f"Mapping {name} -> {mapped_key}")
            
            if mapped_key in state_dict:
                param.copy_(state_dict[mapped_key])
            elif strict:
                raise KeyError(f'Missing key in state dict: {mapped_key}')

class RealESRGANer:
    """A wrapper for RealESRGAN model that applies super-resolution on images"""

    def __init__(self, scale, model_path, device=None, tile=0):
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Load model weights
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the model has 'params' key (Real-ESRGAN format)
        if 'params' in loadnet:
            state_dict = loadnet['params']
        else:
            state_dict = loadnet
            
        # Count number of convolutional layers in body
        conv_count = 0
        for k in state_dict.keys():
            if k.startswith('body.') and k.endswith('.weight') and len(state_dict[k].shape) == 4:
                conv_count += 1
        
        # Remove conv_first, upconv1, upconv2, etc.
        body_conv_count = conv_count - 3  # subtract first conv and 2 upconv layers
        
        print(f"Total conv layers: {conv_count}, Body conv layers: {body_conv_count}")
        
        # Initialize the model
        self.model = SRVGGNetCompact(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64,
            num_conv=body_conv_count,
            upscale=scale
        ).to(self.device)
        
        # Direct loading from state dict - bypass standard PyTorch loading
        for name in self.model.state_dict():
            # Map the model's parameter names to the state dict
            # This is the trickiest part - we need to match our model's parameter names
            # with the parameter names in the state dict
            parts = name.split('.')
            if len(parts) <= 2:  # It's a direct body parameter
                weight_key = name
            else:  # It's a nested parameter
                index = int(parts[1])
                sub_key = '.'.join(parts[2:])
                weight_key = f"body.{index}.{sub_key}"
                
            if weight_key in state_dict:
                self.model.state_dict()[name].copy_(state_dict[weight_key])
            else:
                print(f"Warning: {weight_key} not found in state dict")
        
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

def create_model_from_weights(model_path):
    """Create a model with the architecture that matches the weights"""
    # Load weights
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if the model has 'params' key (Real-ESRGAN format)
    if 'params' in loadnet:
        state_dict = loadnet['params']
    else:
        state_dict = loadnet
        
    # Count number of convolutional layers in body
    conv_count = 0
    for k in state_dict.keys():
        if k.startswith('body.') and k.endswith('.weight') and len(state_dict[k].shape) == 4:
            conv_count += 1
    
    # Remove conv_first, upconv1, upconv2, etc.
    body_conv_count = conv_count - 3  # subtract first conv and 2 upconv layers
    
    print(f"Total conv layers: {conv_count}, Body conv layers: {body_conv_count}")
    
    # Check input/output channels of first and last conv
    first_conv_weight = state_dict['body.0.weight']  # First conv weight
    last_conv_weight = None
    for k in reversed(sorted(state_dict.keys())):
        if k.endswith('.weight') and len(state_dict[k].shape) == 4:
            last_conv_weight = state_dict[k]
            break
    
    in_channels = first_conv_weight.shape[1]
    out_channels = last_conv_weight.shape[0] if last_conv_weight is not None else 3
    
    print(f"Input channels: {in_channels}, Output channels: {out_channels}")
    
    # Create model
    model = SRVGGNetCompact(
        num_in_ch=in_channels,
        num_out_ch=out_channels,
        num_feat=64,
        num_conv=body_conv_count,
        upscale=4
    )
    
    return model

# Test function
def test_real_esrgan(input_path, output_path):
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
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set model path
    model_path = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights', 'realesr-general-x4v3.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Initialize the enhancer
    upsampler = RealESRGANer(
        scale=4,
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
    test_real_esrgan(test_input, "test_output_fixed_realesrgan.png")
