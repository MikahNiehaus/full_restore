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
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from realesrgan_adapter import RealESRGANAdapter

# Configure path for local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ImageRestorer:
    """
    Advanced image restoration class that applies multiple restoration techniques
    to improve image quality before colorization.
    """
    def __init__(self, device=None, models_dir='models', model_name='RealESRGAN_x4plus', model_path=None, tile=0, tile_pad=10, pre_pad=0, outscale=2, fp32=False, gpu_id=None, alpha_upsampler='realesrgan', denoise_strength=1.0, force_device=None):
        """
        Initialize the image restorer with models and Real-ESRGAN options
        
        Args:
            device: PyTorch device (GPU/CPU)
            models_dir: Directory containing model weights
            model_name: Real-ESRGAN model name
            model_path: Path to model weights (optional)
            tile, tile_pad, pre_pad, outscale, fp32, gpu_id, alpha_upsampler, denoise_strength: Real-ESRGAN options
            force_device: Override device detection and force a specific device (cuda or cpu)
        """
        self.models_dir = Path(models_dir)
        
        # Setup device with improved handling
        if force_device is not None:
            # Force a specific device regardless of availability
            self.device = torch.device(force_device)
            print(f"[INFO] ImageRestorer: Forcing device: {self.device}")
        elif device is not None:
            # User specified device
            self.device = device
            print(f"[INFO] ImageRestorer: Using user-specified device: {self.device}")
        else:
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"[INFO] ImageRestorer: Using GPU acceleration with {torch.cuda.get_device_name(0)}")
                print(f"[INFO] ImageRestorer: GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                print(f"[INFO] ImageRestorer: CUDA Version: {torch.version.cuda}")
                # Set optimal CUDA performance settings
                torch.backends.cudnn.benchmark = True
                print("[INFO] ImageRestorer: Enabled cuDNN benchmark mode for optimal performance")
            else:
                self.device = torch.device('cpu')
                print("[WARNING] ImageRestorer: CUDA not available, using CPU fallback (much slower)")
            
        # Set optimal tile size for GPU memory
        adjusted_tile = tile
        if self.device.type == 'cuda' and tile == 0:
            # Automatically set tile size based on GPU VRAM
            # This helps prevent CUDA out of memory errors
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem_gb > 10:  # High-end GPU with lots of VRAM
                adjusted_tile = 0  # No tiling needed
            elif gpu_mem_gb > 6:  # Mid-range GPU
                adjusted_tile = 1000  # Large tile size
            elif gpu_mem_gb > 4:  # Low VRAM GPU
                adjusted_tile = 800  # Medium tile size
            else:  # Very low VRAM
                adjusted_tile = 400  # Small tile size
            print(f"[INFO] ImageRestorer: Auto-set tile size to {adjusted_tile} based on {gpu_mem_gb:.1f}GB GPU VRAM")
        
        # Initialize models
        self.models_loaded = False
        self.load_models(
            model_name=model_name,
            model_path=model_path,
            tile=adjusted_tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            outscale=outscale,
            fp32=fp32,
            gpu_id=gpu_id,
            alpha_upsampler=alpha_upsampler,
            denoise_strength=denoise_strength
        )
        
    def load_models(self, model_name='RealESRGAN_x4plus', model_path=None, tile=0, tile_pad=10, pre_pad=0, outscale=2, fp32=False, gpu_id=None, alpha_upsampler='realesrgan', denoise_strength=1.0):
        """Load necessary models for restoration, with flexible Real-ESRGAN options."""
        try:
            # Model path logic
            if not model_path:
                if model_name == 'RealESRGAN_x4plus':
                    model_path = Path(r"c:/prj/full_restore/models/RealESRGAN_x4plus.pth")
                elif model_name == 'RealESRNet_x4plus':
                    model_path = Path(r"c:/prj/full_restore/models/RealESRNet_x4plus.pth")
                elif model_name == 'RealESRGAN_x4plus_anime_6B':
                    model_path = Path(r"c:/prj/full_restore/models/RealESRGAN_x4plus_anime_6B.pth")
                elif model_name == 'RealESRGAN_x2plus':
                    model_path = Path(r"c:/prj/full_restore/models/RealESRGAN_x2plus.pth")
                elif model_name == 'realesr-animevideov3':
                    model_path = Path(r"c:/prj/full_restore/models/realesr-animevideov3.pth")
                elif model_name == 'realesr-general-x4v3':
                    model_path = Path(r"c:/prj/full_restore/models/realesr-general-x4v3.pth")
                else:
                    raise FileNotFoundError(f"Unknown model name: {model_name} and no model_path provided.")
            if not Path(model_path).exists():
                raise FileNotFoundError(f"SR model not found at {model_path}. Please ensure the weights file exists.")
            self.sr_enhancer = RealESRGANAdapter(
                model_name=model_name,
                model_path=model_path,
                device=self.device,
                half=not fp32,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                outscale=outscale,
                fp32=fp32,
                gpu_id=gpu_id,
                alpha_upsampler=alpha_upsampler,
                denoise_strength=denoise_strength
            )
            self.models_loaded = True
            print(f"[INFO] Image restoration models loaded successfully (model: {model_name})")
        except ImportError as e:
            print(f"[WARNING] Could not import required modules for restoration: {e}")
            print("[INFO] Install Real-ESRGAN: pip install realesrgan")
            self.models_loaded = False
        except Exception as e:
            print(f"[ERROR] Failed to load restoration models: {e}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
            
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
        """Enhance image details with very strong sharpening before super-resolution"""
        try:
            # Strong sharpening kernel (increase center value for more effect)
            strong_kernel = np.array([
                [-2, -2, -2],
                [-2, 17, -2],
                [-2, -2, -2]
            ])
            # Apply strong sharpening
            sharpened = cv2.filter2D(image, -1, strong_kernel)
            # Optionally blend less with the original for even more sharpness
            enhanced = cv2.addWeighted(image, 0.2, sharpened, 0.8, 0)
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
            # Clear CUDA cache before processing to free up memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
                # Optimize processing by ensuring efficient memory usage
                # Check available memory and set tile size dynamically if needed
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                free_memory_mb = free_memory / (1024**2)
                
                # If memory is tight, use tiling automatically
                h, w = image.shape[:2]
                pixels = h * w
                # Estimate memory need: ~4 bytes per pixel × 3 channels × 4 for processing overhead
                est_memory_need = pixels * 3 * 4 * 4 / (1024**2)  # in MB
                
                if est_memory_need > free_memory_mb * 0.8:  # If we need more than 80% of free memory
                    print(f"[INFO] Large image detected ({w}x{h}), memory optimization active")
                    # Calculate a safe tile size
                    safe_tile_size = int(np.sqrt((free_memory_mb * 0.7 * 1024**2) / (3 * 4 * 4)))
                    print(f"[INFO] Using automatic tile size of {safe_tile_size}px")
                    self.sr_enhancer.upsampler.tile = min(safe_tile_size, 1024)  # Cap at 1024px tiles
                else:
                    # Use full image processing for better quality when memory allows
                    if self.sr_enhancer.upsampler.tile != 0:
                        print("[INFO] Sufficient GPU memory, processing without tiling")
                        self.sr_enhancer.upsampler.tile = 0
            
            # Use the adapter for enhancement
            start_time = time.time()
            output, _ = self.sr_enhancer.enhance(image, outscale=2)
            end_time = time.time()
            
            if self.device.type == 'cuda':
                print(f"[INFO] Super-resolution completed in {end_time - start_time:.2f} seconds using GPU")
                # Report GPU memory usage
                print(f"[INFO] GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
                print(f"[INFO] GPU memory reserved: {torch.cuda.memory_reserved(0) / (1024**2):.1f} MB")
                # Clear cache again after processing
                torch.cuda.empty_cache()
            else:
                print(f"[INFO] Super-resolution completed in {end_time - start_time:.2f} seconds using CPU")
                
            return output
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[ERROR] CUDA out of memory during super-resolution. Try:")
                print("  1. Reduce image size")
                print("  2. Increase tile size (for tiled processing)")
                print("  3. Use CPU (much slower)")
                # Fallback to CPU processing
                try:
                    print("[INFO] Attempting fallback to CPU for this image...")
                    # Store current device
                    original_device = self.device
                    # Temporarily switch to CPU
                    self.device = torch.device('cpu')
                    self.sr_enhancer.upsampler.device = torch.device('cpu')
                    self.sr_enhancer.upsampler.model.to('cpu')
                    
                    # Try processing on CPU
                    start_time = time.time()
                    output, _ = self.sr_enhancer.enhance(image, outscale=2)
                    end_time = time.time()
                    print(f"[INFO] CPU fallback completed in {end_time - start_time:.2f} seconds")
                    
                    # Switch back to original device
                    self.device = original_device
                    self.sr_enhancer.upsampler.device = original_device
                    self.sr_enhancer.upsampler.model.to(original_device)
                    
                    return output
                except Exception as cpu_e:
                    print(f"[WARNING] CPU fallback also failed: {cpu_e}")
                    return image
            else:
                print(f"[WARNING] Super-resolution failed: {e}")
                return image
        except Exception as e:
            print(f"[WARNING] Super-resolution failed: {e}")
            return image
    
    def restore_image(self, image_path, output_path=None, scale=2.0, log_pipeline=True, save_sharpened_path=None, skip_enhance=False):
        """
        Restore an image using multiple techniques
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            scale: Scale factor for super-resolution
            save_sharpened_path: Optional path to save the sharpened (pre-Real-ESRGAN) image
            skip_enhance: If True, skip the Real-ESRGAN enhancement step
        
        Returns:
            numpy.ndarray: Restored image
        """
        try:
            # Load image
            if isinstance(image_path, str) or isinstance(image_path, Path):
                image = cv2.imread(str(image_path))
            else:
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
            
            # --- Strong sharpening ---
            image_sharp = self._enhance_details(image)
            if save_sharpened_path:
                os.makedirs(os.path.dirname(str(save_sharpened_path)), exist_ok=True)
                cv2.imwrite(str(save_sharpened_path), image_sharp)
                print(f"[INFO] Sharpened image saved to: {save_sharpened_path}")
            # --- Real-ESRGAN enhancement ---
            if not skip_enhance and self.models_loaded:
                image_enhanced = self._super_resolve(image_sharp)
            else:
                image_enhanced = image_sharp
            if output_path:
                os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
                cv2.imwrite(str(output_path), image_enhanced)
                print(f"[INFO] Restored image saved to: {output_path}")
            return image_enhanced
        except Exception as e:
            print(f"[ERROR] Image restoration failed: {e}")
            import traceback
            traceback.print_exc()
            if isinstance(image_path, str) or isinstance(image_path, Path):
                return cv2.imread(str(image_path))
            return image_path
    
    def restore_frames(self, input_dir, output_dir):
        """
        Restore all frames in a directory using DeOldify-based restoration only (no Real-ESRGAN enhancement).
        
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
        image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
        
        if not image_files:
            print(f"[WARNING] No image files found in {input_dir}")
            return []
        
        print(f"[INFO] Restoring {len(image_files)} frames with DeOldify (no Real-ESRGAN)...")
        restored_paths = []
        for img_path in tqdm(image_files, desc="Restoring frames"):
            img = cv2.imread(str(img_path))
            if img is not None:
                # Only DeOldify-based restoration: remove scratches, denoise, sharpen, but skip Real-ESRGAN
                restored = self.restore_image(img, None, log_pipeline=True, skip_enhance=True)
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), restored)
                restored_paths.append(output_path)
            else:
                print(f"[WARNING] Failed to load image: {img_path}")
        print(f"[INFO] Restored {len(restored_paths)} frames to {output_dir}")
        return restored_paths
    
    def enhance_frames(self, input_dir, output_dir, outscale=2):
        """
        Enhance all frames in a directory using Real-ESRGAN super-resolution only (no DeOldify restoration).
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save enhanced frames
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

        print(f"[INFO] Enhancing {len(image_files)} frames with Real-ESRGAN (no DeOldify)...")
        enhanced_paths = []
        for img_path in tqdm(image_files, desc="Enhancing frames"):
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
    parser = argparse.ArgumentParser(description="Restore or enhance images using DeOldify or Real-ESRGAN.")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", "-o", help="Output image file or directory")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], help="Device to use (cuda/cpu)")
    parser.add_argument("--mode", choices=["restore", "enhance"], default="restore", help="Pipeline mode: 'restore' for DeOldify, 'enhance' for Real-ESRGAN")
    parser.add_argument("--outscale", type=int, default=2, help="Upscale factor for enhancement (Real-ESRGAN only)")
    parser.add_argument("--model_name", type=str, default="RealESRGAN_x4plus", help="Real-ESRGAN model name (e.g. RealESRGAN_x4plus, RealESRNet_x4plus, RealESRGAN_x4plus_anime_6B, RealESRGAN_x2plus, realesr-animevideov3, realesr-general-x4v3)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Real-ESRGAN model weights (optional)")
    parser.add_argument("--tile", type=int, default=0, help="Tile size for Real-ESRGAN (default: 0)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding for Real-ESRGAN (default: 10)")
    parser.add_argument("--pre_pad", type=int, default=0, help="Pre padding for Real-ESRGAN (default: 0)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 precision for Real-ESRGAN (default: fp16)")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU device id for Real-ESRGAN (default: None)")
    parser.add_argument("--alpha_upsampler", type=str, default="realesrgan", help="Alpha upsampler for Real-ESRGAN (default: realesrgan)")
    parser.add_argument("--denoise_strength", type=float, default=1.0, help="Denoise strength for realesr-general-x4v3 (default: 1.0)")
    args = parser.parse_args()
    # Determine device
    device = None
    if args.device:
        device = torch.device(args.device)
    # Initialize restorer with all Real-ESRGAN options
    restorer = ImageRestorer(
        device=device,
        model_name=args.model_name,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        outscale=args.outscale,
        fp32=args.fp32,
        gpu_id=args.gpu_id,
        alpha_upsampler=args.alpha_upsampler,
        denoise_strength=args.denoise_strength
    )
    # Process input
    input_path = Path(args.input)
    if input_path.is_dir():
        output_dir = args.output if args.output else input_path.parent / (input_path.name + ("_restored" if args.mode=="restore" else "_enhanced"))
        if args.mode == "restore":
            restorer.restore_frames(input_path, output_dir)
        else:
            restorer.enhance_frames(input_path, output_dir, outscale=args.outscale)
    else:
        output_path = args.output if args.output else input_path.parent / (input_path.stem + ("_restored" if args.mode=="restore" else "_enhanced") + input_path.suffix)
        if args.mode == "restore":
            restorer.restore_image(input_path, output_path)
        else:
            img = cv2.imread(str(input_path))
            if img is not None:
                output, _ = restorer.sr_enhancer.enhance(img, outscale=args.outscale)
                cv2.imwrite(str(output_path), output)
                print(f"[INFO] Enhanced image saved to: {output_path}")
            else:
                print(f"[ERROR] Failed to load image: {input_path}")
