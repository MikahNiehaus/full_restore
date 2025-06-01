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
        
        # Always try GPU first, fallback to CPU if not available
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("[INFO] ImageRestorer: Using GPU (cuda)")
            else:
                self.device = torch.device('cpu')
                print("[WARNING] ImageRestorer: CUDA not available, using CPU fallback")
        else:
            self.device = device
            print(f"[INFO] ImageRestorer: Using user-specified device: {self.device}")
        
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
    
    def restore_image(self, image_path, output_path=None, scale=2.0, log_pipeline=True):
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
    
    def restore_frames(self, input_dir, output_dir, batch_size=32):
        """
        Restore all frames in a directory in batches for efficiency.
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save restored frames
            batch_size: Number of frames to process in a batch (default: 32)
            
        Returns:
            list: Paths to restored frames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
        
        if not image_files:
            print(f"[WARNING] No image files found in {input_dir}")
            return []
        
        print(f"[INFO] Restoring {len(image_files)} frames in batches of {batch_size}...")
        restored_paths = []
        for i in tqdm(range(0, len(image_files), batch_size), desc="Restoring frames"):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            for img_path in batch_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    batch_images.append((img_path, img))
                else:
                    print(f"[WARNING] Failed to load image: {img_path}")
            # Only log once per batch
            if batch_images:
                self.restore_image(batch_images[0][1], None, log_pipeline=True)
                # Actually restore all images in batch, but only log for the first
                for img_path, img in batch_images:
                    restored = self.restore_image(img, None, log_pipeline=False)
                    output_path = output_dir / img_path.name
                    cv2.imwrite(str(output_path), restored)
                    restored_paths.append(output_path)
        print(f"[INFO] Restored {len(restored_paths)} frames to {output_dir}")
        return restored_paths
    
    def enhance_frames(self, input_dir, output_dir, batch_size=32, outscale=2):
        """
        Enhance all frames in a directory using Real-ESRGAN super-resolution only.
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save enhanced frames
            batch_size: Number of frames to process in a batch (default: 32)
            outscale: Upscale factor (default: 2)
            
        Returns:
            list: Paths to enhanced frames
        """
        if not self.models_loaded:
            print("[WARNING] Super-resolution models not loaded, skipping enhancement")
            return []

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
        if not image_files:
            print(f"[WARNING] No image files found in {input_dir}")
            return []

        print(f"[INFO] Enhancing {len(image_files)} frames with Real-ESRGAN in batches of {batch_size}...")
        enhanced_paths = []
        for i in tqdm(range(0, len(image_files), batch_size), desc="Enhancing frames"):
            batch_files = image_files[i:i+batch_size]
            for img_path in batch_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    try:
                        output, _ = self.sr_enhancer.enhance(img, outscale=outscale)
                        output_path = output_dir / img_path.name
                        cv2.imwrite(str(output_path), output)
                        enhanced_paths.append(output_path)
                    except Exception as e:
                        print(f"[WARNING] Enhancement failed for {img_path}: {e}")
                else:
                    print(f"[WARNING] Failed to load image: {img_path}")
        print(f"[INFO] Enhanced {len(enhanced_paths)} frames to {output_dir}")
        return enhanced_paths

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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for frame restoration (default: 32)")
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
        restorer.restore_frames(input_path, output_dir, batch_size=args.batch_size)
    else:
        output_path = args.output if args.output else input_path.parent / (input_path.stem + "_restored" + input_path.suffix)
        restorer.restore_image(input_path, output_path)
