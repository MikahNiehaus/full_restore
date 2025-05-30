#!/usr/bin/env python3
"""
AI Image Restoration Module for DeOldify Colorization Pipeline

This module uses advanced neural networks to restore old images/frames before colorization:
1. Noise reduction
2. Scratch/damage removal
3. Detail enhancement
4. Resolution improvement

It can be used as a pre-processing step before DeOldify colorization for better results.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Configure path for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ImageRestorer:
    """
    Advanced image restoration class that applies multiple restoration techniques
    to improve image quality before colorization.
    """
    
    def __init__(self, device=None, models_dir='models'):
        """
        Initialize the image restorer with models
        
        Args:
            device: PyTorch device (GPU/CPU)
            models_dir: Directory containing model weights
        """
        self.models_dir = Path(models_dir)
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"[INFO] Image restorer using device: {self.device}")
        
        # Initialize models
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """Load necessary models for restoration"""
        try:
            # Import Real-ESRGAN for super-resolution
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Model paths
            self.sr_model_path = self.models_dir / 'RealESRGAN_x4plus.pth'
            
            # Check if model exists
            if not self.sr_model_path.exists():
                print(f"[WARNING] SR model not found at {self.sr_model_path}")
                print("[INFO] Downloading RealESRGAN model...")
                self._download_model('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
                                   self.sr_model_path)
            
            # Create RealESRGAN model
            self.sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.sr_enhancer = RealESRGANer(
                scale=4,
                model_path=str(self.sr_model_path),
                model=self.sr_model,
                tile=0,  # Tile size, 0 for no tile
                tile_pad=10,
                pre_pad=0,
                half=True,  # Use half precision to save memory
                device=self.device
            )
            
            self.models_loaded = True
            print("[INFO] Image restoration models loaded successfully")
            
        except ImportError as e:
            print(f"[WARNING] Could not import required modules for restoration: {e}")
            print("[INFO] Install Real-ESRGAN: pip install realesrgan")
            self.models_loaded = False
        except Exception as e:
            print(f"[ERROR] Failed to load restoration models: {e}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
            
    def _download_model(self, url, output_path):
        """Download a model file"""
        try:
            import requests
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as file, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
                    
            return True
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            return False
    
    def _denoise_image(self, image):
        """Apply adaptive denoising to the image"""
        try:
            # Convert to grayscale to detect noise level
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Estimate noise level
            noise_sigma = np.std(gray)
            
            # Apply denoising with parameters based on noise level
            if noise_sigma > 20:
                # Heavy noise
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            elif noise_sigma > 10:
                # Medium noise
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 5, 15)
            else:
                # Light or no noise
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 3, 9)
                
            return denoised
        except Exception as e:
            print(f"[WARNING] Denoising failed: {e}")
            return image
    
    def _remove_scratches(self, image):
        """Remove scratches and artifacts from the image"""
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Create mask for inpainting (detect bright scratches)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            # Apply morphology to connect nearby scratches
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Inpaint to remove scratches
            restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
            
            return restored
        except Exception as e:
            print(f"[WARNING] Scratch removal failed: {e}")
            return image
    
    def _enhance_details(self, image):
        """Enhance image details while preserving structure"""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
            
            return enhanced
        except Exception as e:
            print(f"[WARNING] Detail enhancement failed: {e}")
            return image
    
    def _super_resolve(self, image):
        """Apply super-resolution to increase image resolution"""
        if not self.models_loaded:
            print("[WARNING] Super-resolution models not loaded, skipping")
            return image
        
        try:
            # Process with Real-ESRGAN
            output, _ = self.sr_enhancer.enhance(image, outscale=2)
            return output
        except Exception as e:
            print(f"[WARNING] Super-resolution failed: {e}")
            return image
    
    def restore_image(self, image_path, output_path=None, scale=2.0):
        """
        Restore an image using multiple techniques
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            scale: Scale factor for super-resolution
            
        Returns:
            numpy.ndarray: Restored image
        """
        try:
            # Load image
            if isinstance(image_path, str) or isinstance(image_path, Path):
                image = cv2.imread(str(image_path))
            else:
                # Assume it's already a numpy array
                image = image_path
                
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            # Skip restoration for very small images
            if image.shape[0] < 100 or image.shape[1] < 100:
                print("[WARNING] Image too small for full restoration pipeline")
                return image
                
            print("[INFO] Applying restoration pipeline...")
            
            # 1. Remove scratches and artifacts
            image = self._remove_scratches(image)
            
            # 2. Denoise the image
            image = self._denoise_image(image)
            
            # 3. Enhance details
            image = self._enhance_details(image)
            
            # 4. Super-resolution (if model is loaded)
            if self.models_loaded:
                image = self._super_resolve(image)
            
            # Save output if path provided
            if output_path:
                os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
                cv2.imwrite(str(output_path), image)
                print(f"[INFO] Restored image saved to: {output_path}")
            
            return image
            
        except Exception as e:
            print(f"[ERROR] Image restoration failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original image or None if it failed
            if isinstance(image_path, str) or isinstance(image_path, Path):
                return cv2.imread(str(image_path))
            return image_path
            
    def restore_frames(self, input_dir, output_dir):
        """
        Restore all frames in a directory
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save restored frames
            
        Returns:
            list: Paths to restored frames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = sorted([f for f in input_dir.glob('*.png') or input_dir.glob('*.jpg')])
        
        if not image_files:
            print(f"[WARNING] No image files found in {input_dir}")
            return []
            
        print(f"[INFO] Restoring {len(image_files)} frames...")
        restored_paths = []
        
        # Process each frame
        for img_path in tqdm(image_files, desc="Restoring frames"):
            output_path = output_dir / img_path.name
            self.restore_image(img_path, output_path)
            restored_paths.append(output_path)
            
        print(f"[INFO] Restored {len(restored_paths)} frames to {output_dir}")
        return restored_paths

def restore_image(image_path, output_path=None):
    """
    Convenience function to restore a single image
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (optional)
        
    Returns:
        numpy.ndarray: Restored image
    """
    restorer = ImageRestorer()
    return restorer.restore_image(image_path, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Restore old images before colorization")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", "-o", help="Output image file or directory")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Determine device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Initialize restorer
    restorer = ImageRestorer(device=device)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_dir():
        output_dir = args.output if args.output else input_path.parent / (input_path.name + "_restored")
        restorer.restore_frames(input_path, output_dir)
    else:
        output_path = args.output if args.output else input_path.parent / (input_path.stem + "_restored" + input_path.suffix)
        restorer.restore_image(input_path, output_path)
