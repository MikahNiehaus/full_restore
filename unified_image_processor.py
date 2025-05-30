"""
A unified image processor class that can be used for both individual images and video frames.
This follows OOP principles and can be used in both pipelines.
"""
import os
import sys
import shutil
import traceback
from PIL import Image, ImageEnhance
import cv2
import numpy as np

class ImageProcessor:
    """
    A unified class for image processing operations that works for both
    individual images and video frames.
    """
    def __init__(self, output_dir="outputs"):
        """
        Initialize the image processor.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Add DeOldify to the path if not already there
        deoldify_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
        if deoldify_dir not in sys.path:
            sys.path.insert(0, deoldify_dir)
            
    def enhance_image(self, input_path, output_path, scale=2, brighten_factor=1.15):
        """
        Subtle enhancement: gentle upscaling, light detail enhancement, and mild sharpening.
        Prevents over-sharpening and white/pixellated artifacts.
        
        Args:
            input_path (str): Path to the input image
            output_path (str): Path to save the enhanced image
            scale (int): Scale factor for upscaling
            brighten_factor (float): Factor to increase brightness (>1.0 brightens, <1.0 darkens)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"[ENHANCE] Starting enhancement for: {input_path}")
            print(f"[ENHANCE] Output path: {output_path}, Scale: {scale}, Brighten: {brighten_factor}")
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"[ENHANCE][ERROR] Could not read image: {input_path}")
                return False
                
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"[ENHANCE] Image loaded and converted to RGB.")
            
            # Gentle bilateral filter
            img_filtered = cv2.bilateralFilter(img_rgb, 5, 40, 40)
            print(f"[ENHANCE] Bilateral filter applied.")
            
            # Light detail enhancement (lower sigma)
            img_details = cv2.detailEnhance(img_filtered, sigma_s=5, sigma_r=0.08)
            print(f"[ENHANCE] Detail enhancement applied.")
            
            # Mild upscaling (if scale > 1)
            h, w = img_details.shape[:2]
            if scale > 1:
                img_upscaled = cv2.resize(img_details, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                print(f"[ENHANCE] Image upscaled to {(w*scale, h*scale)}.")
            else:
                img_upscaled = img_details
                print(f"[ENHANCE] No upscaling applied.")
            
            # Very gentle unsharp mask (reduce strength)
            blur = cv2.GaussianBlur(img_upscaled, (0, 0), 1.0)
            img_upscaled = cv2.addWeighted(img_upscaled, 1.10, blur, -0.08, 0)  # Lowered from 1.5/-0.5
            print(f"[ENHANCE] Unsharp mask applied.")
            
            # Use PIL for mild brightness/contrast
            pil_img = Image.fromarray(img_upscaled)
            enhancer = ImageEnhance.Brightness(pil_img)
            brightened = enhancer.enhance(brighten_factor)
            contrast = ImageEnhance.Contrast(brightened)
            final_img = contrast.enhance(1.05)  # Lowered from 1.1
            
            # Save the final enhanced image
            final_img.save(output_path)
            print(f"[ENHANCE] Enhanced image saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ENHANCE][ERROR] Error enhancing image: {e}")
            traceback.print_exc()
            return False
            
    def colorize_image(self, input_path, output_path, color_model='stable', render_factor=25, color_boost=1.0):
        """
        Colorize an image using DeOldify, always forcing CPU usage.
        
        Args:
            input_path (str): Path to the input image
            output_path (str): Path to save the colorized image
            color_model (str): 'stable', 'artistic', or 'both' (default: 'stable')
            render_factor (int): Render factor for DeOldify (default: 25 for CPU)
            color_boost (float): Color saturation multiplier (NOT USED - removed to prevent orange tint)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"\n===============================================")
            print(f"DEBUG: Colorizing image with DeOldify ({color_model} model)")
            print(f"DEBUG: Input: {input_path}")
            print(f"DEBUG: Output: {output_path}")
            print(f"DEBUG: Render factor: {render_factor}")
            print(f"===============================================\n")

            # Import DeOldify
            from deoldify.visualize import get_image_colorizer
            from deoldify import device
            from deoldify.device_id import DeviceId
            
            # Always force CPU
            device.set(device=DeviceId.CPU)
            force_cpu = True
            
            # Check if the models exist in the custom directory
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'My PTH'))
            print(f"DEBUG: Looking for models in: {model_dir}")
            
            # Set environment variable for DeOldify to find models
            os.environ['DEOLDIFY_MODELS'] = model_dir
            print(f"DEBUG: Set DEOLDIFY_MODELS environment variable to: {os.environ['DEOLDIFY_MODELS']}")
            
            models_to_try = []
            if color_model == 'both':
                models_to_try = ['stable', 'artistic']
            elif color_model == 'artistic':
                models_to_try = ['artistic']
            else:
                models_to_try = ['stable']
            
            print(f"DEBUG: Will try these models in order: {models_to_try}")
            
            last_success = False
            for model in models_to_try:
                try:
                    # Check if model file exists
                    if model == 'stable':
                        model_path = os.path.join(model_dir, 'ColorizeStable_gen.pth')
                    else:
                        model_path = os.path.join(model_dir, 'ColorizeArtistic_gen.pth')
                        
                    if not os.path.exists(model_path):
                        print(f"DEBUG: ERROR - Model file not found: {model_path}")
                        continue
                    
                    print(f"DEBUG: Using DeOldify {model} model from: {model_path}")
                    print(f"DEBUG: File exists? {os.path.exists(model_path)}")
                    print(f"DEBUG: File size: {os.path.getsize(model_path)}")
                    
                    # Monkey-patch the model loading to use our custom path
                    import fastai
                    original_load = fastai.basic_train.load_learner
                    
                    def patched_load_learner(*args, **kwargs):
                        print(f"DEBUG: Patched load_learner called with:")
                        print(f"DEBUG:   args: {args}")
                        print(f"DEBUG:   kwargs: {kwargs}")
                        
                        if 'path' in kwargs and 'models' in str(kwargs['path']):
                            print(f"DEBUG:   Replacing kwargs path with: {model_dir}")
                            kwargs['path'] = model_dir
                        elif len(args) > 0 and 'models' in str(args[0]):
                            print(f"DEBUG:   Replacing args[0] with: {model_dir}")
                            args = list(args)
                            args[0] = model_dir
                            args = tuple(args)
                        
                        print(f"DEBUG:   Final args: {args}")
                        print(f"DEBUG:   Final kwargs: {kwargs}")
                        return original_load(*args, **kwargs)
                    
                    fastai.basic_train.load_learner = patched_load_learner
                    
                    # Get colorizer and process the image
                    print(f"DEBUG: Getting image colorizer for {model} model...")
                    colorizer = get_image_colorizer(artistic=(model == 'artistic'))
                    print(f"DEBUG: Colorizer object created: {colorizer}")
                    
                    print(f"DEBUG: Processing image with {model} model...")
                    colorized_path = colorizer.plot_transformed_image(
                        input_path,
                        render_factor=render_factor,
                        watermarked=False,
                        post_process=True,
                        results_dir=os.path.dirname(output_path),
                        force_cpu=force_cpu
                    )
                    print(f"DEBUG: Colorized path returned: {colorized_path}")
                    
                    # Restore original function
                    fastai.basic_train.load_learner = original_load
                    
                    if isinstance(colorized_path, str) and os.path.exists(colorized_path):
                        print(f"DEBUG: Colorized file exists at: {colorized_path}")
                        
                        if os.path.abspath(colorized_path) != os.path.abspath(output_path):
                            shutil.copy(colorized_path, output_path)
                            print(f"DEBUG: Copied to final output path: {output_path}")
                        
                        # REMOVED: Color boosting - might be causing orange tint
                        # No color enhancement or contrast adjustment
                        
                        print(f"DEBUG: ✓ Colorization completed, image saved to: {output_path}")
                        last_success = True
                        break
                    else:
                        print(f"DEBUG: ERROR - Failed to generate valid colorized output with {model} model")
                        if colorized_path:
                            print(f"DEBUG:   Path returned: {colorized_path}")
                            print(f"DEBUG:   Exists? {os.path.exists(str(colorized_path))}")
                        
                except Exception as e:
                    print(f"DEBUG: ERROR - Error with {model} model: {str(e)}")
                    print(f"DEBUG: Traceback:")
                    traceback.print_exc()
                    if 'models' in str(e) or 'not found' in str(e).lower():
                        print(f"DEBUG: This appears to be a model loading error!")
                        print(f"DEBUG: Models dir: {model_dir}")
                        print(f"DEBUG: Model file: {model_path}")
                        print(f"DEBUG: File exists? {os.path.exists(model_path)}")
            
            if last_success:
                print(f"DEBUG: Colorization completed successfully!")
                return True
            
            print(f"DEBUG: ERROR - All colorization models failed.")
            # Do NOT save any fallback or orange image. Raise error.
            raise RuntimeError(f"DeOldify colorization failed for {input_path} (no fallback allowed)")
            
        except Exception as e:
            print(f"DEBUG: FATAL ERROR - Error during DeOldify colorization: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"DeOldify colorization failed for {input_path}: {e}")
            
    def restore_image(self, input_path, output_path, scale=2):
        """
        Restore an image using AI upscaling and detail enhancement.
        
        Args:
            input_path (str): Path to the input image
            output_path (str): Path to save the restored image
            scale (int): Scale factor for upscaling
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Restoring image with scale factor {scale}...")
            
            # Try to use Real-ESRGAN for upscaling
            try:                # Import Real-ESRGAN
                realesrgan_path = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN')
                if realesrgan_path not in sys.path:
                    sys.path.append(realesrgan_path)
                import torch
                try:
                    from realesrgan import RealESRGANer
                except ImportError:
                    # Try adding parent directory to path in case it's in the parent module
                    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    if parent_dir not in sys.path:
                        sys.path.append(parent_dir)
                    from realesrgan import RealESRGANer
                
                # Find model weights
                possible_weight_dirs = [
                    os.path.join(os.path.dirname(__file__), 'weights'),
                    os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights'),
                    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Real-ESRGAN', 'weights')
                ]
                
                model_path = None
                for weights_dir in possible_weight_dirs:
                    potential_path = os.path.join(weights_dir, 'realesr-general-x4v3.pth')
                    if os.path.exists(potential_path):
                        model_path = potential_path
                        break
                        
                if not model_path:
                    raise FileNotFoundError(f"Real-ESRGAN model file not found")
                
                print(f"Using Real-ESRGAN for restoration with model: {model_path}")
                
                # Initialize upsampler
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=model_path,
                    dni_weight=None,
                    device=device,
                    tile=400  # Use tiling to reduce memory usage
                )
                
                # Process image
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img)
                
                # Save output
                cv2.imwrite(output_path, output)
                print(f"AI restored image saved to: {output_path}")
                return True
                
            except Exception as e:
                print(f"Real-ESRGAN restoration failed: {e}")
                print("Using OpenCV for restoring (fallback method)...")
                
                # OpenCV fallback restoration
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError(f"Could not read image: {input_path}")
                
                # Apply bilateral filter to preserve edges while reducing noise
                img_filtered = cv2.bilateralFilter(img, 9, 75, 75)
                
                # Apply detail enhancement
                img_details = cv2.detailEnhance(img_filtered, sigma_s=10, sigma_r=0.15)
                
                # Upscale
                h, w = img_details.shape[:2]
                img_upscaled = cv2.resize(img_details, (w * scale, h * scale), 
                                        interpolation=cv2.INTER_LANCZOS4)
                
                # Apply unsharp mask for sharpening
                blur = cv2.GaussianBlur(img_upscaled, (0, 0), 3.0)
                img_upscaled = cv2.addWeighted(img_upscaled, 1.5, blur, -0.5, 0)
                
                # Save the result
                cv2.imwrite(output_path, img_upscaled)
                print(f"OpenCV restored image saved to: {output_path}")
                return True
                
        except Exception as e:
            print(f"Error during image restoration: {e}")
            traceback.print_exc()
            return False

    def process_image(self, input_path, output_dir=None, scale=2, do_restore=True, do_colorize=True, do_enhance=True, color_boost=1.0, color_model='stable'):
        """
        Process a single image following these steps: restore -> colorize -> enhance.
        Produces a single final output file.
        
        Args:
            input_path (str): Path to the input image
            output_dir (str): Directory to save output files
            scale (int): Scale factor for upscaling
            do_restore (bool): Whether to run the restore step
            do_colorize (bool): Whether to run the colorize step
            do_enhance (bool): Whether to run the enhance step
            color_boost (float): Color saturation multiplier
            color_model (str): 'stable', 'artistic', or 'both' (default: 'stable')
            
        Returns:
            str: Path to the final processed image or None if failed
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.exists(input_path):
                print(f"Input file not found: {input_path}")
                return None
            base_name = os.path.basename(input_path)
            file_name_wo_ext = os.path.splitext(base_name)[0]
            final_path = os.path.join(output_dir, f"{file_name_wo_ext}_restored.png")
            # Step 1: Restore
            if do_restore:
                print(f"Step 1: Restoring image quality...")
                restore_success = self.restore_image(input_path, final_path, scale=scale)
                if not restore_success:
                    print("Restoration failed. Using original image.")
                    shutil.copy(input_path, final_path)
            else:
                shutil.copy(input_path, final_path)
            # Step 2: Colorize
            if do_colorize:
                print(f"Step 2: Colorizing with DeOldify...")
                colorize_success = self.colorize_image(final_path, final_path, color_model=color_model, color_boost=color_boost)
                if not colorize_success:
                    print("[FATAL] Colorization failed. No fallback will be used. Aborting pipeline.")
                    raise RuntimeError("DeOldify colorization failed and no fallback is allowed.")
            # Step 3: Enhance
            if do_enhance:
                print(f"Step 3: Final enhancement...")
                enhance_success = self.enhance_image(final_path, final_path, scale=1, brighten_factor=1.2)
                if not enhance_success:
                    print("Final enhancement failed. Using colorized image as final.")
            return final_path
        except Exception as e:
            print(f"Error processing image {input_path}: {e}")
            traceback.print_exc()
            return None
            
    def process_video_frame(self, frame_path, output_dir, scale=2):
        """
        Process a video frame using the same pipeline as process_image:
        restore -> colorize -> enhance, producing one final output.
        
        Args:
            frame_path (str): Path to the frame image
            output_dir (str): Directory to save output files
            scale (int): Scale factor for upscaling
            
        Returns:
            str: Path to the processed frame or None if failed
        """
        return self.process_image(frame_path, output_dir, scale)
        
    def enhance_frames(self, frame_paths, output_dir, scale=2):
        """
        Enhance multiple video frames.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_dir (str): Directory to save enhanced frames
            scale (int): Scale factor for upscaling
            
        Returns:
            list: List of paths to enhanced frames
        """
        os.makedirs(output_dir, exist_ok=True)
        enhanced_paths = []
        
        for frame_path in frame_paths:
            base_name = os.path.basename(frame_path)
            enhanced_path = os.path.join(output_dir, base_name)
            
            success = self.enhance_image(frame_path, enhanced_path, scale)
            if success:
                enhanced_paths.append(enhanced_path)
            else:
                # On failure, copy original frame
                shutil.copy(frame_path, enhanced_path)
                enhanced_paths.append(enhanced_path)
                
        return enhanced_paths
        
    def colorize_frames(self, frame_paths, output_dir, color_model='stable', render_factor=35, color_boost=1.0):
        """
        Colorize multiple video frames.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_dir (str): Directory to save colorized frames
            color_model (str): 'stable', 'artistic', or 'both' (default: 'stable')
            render_factor (int): Render factor for DeOldify
            color_boost (float): Color saturation multiplier
        
        Returns:
            list: List of paths to colorized frames
        """
        os.makedirs(output_dir, exist_ok=True)
        colorized_paths = []
        
        for frame_path in frame_paths:
            base_name = os.path.basename(frame_path)
            colorized_path = os.path.join(output_dir, base_name)
            
            success = self.colorize_image(frame_path, colorized_path, color_model=color_model, render_factor=render_factor, color_boost=color_boost)
            if success:
                colorized_paths.append(colorized_path)
            else:
                # On failure, do NOT copy original or restored frame. Raise error.
                raise RuntimeError(f"Colorization failed for frame: {frame_path}")
        
        return colorized_paths

    def process_frames(self, frame_paths, output_dir, scale=2, color_model='stable', color_boost=1.0):
        """
        Process multiple video frames with our complete pipeline:
        restore -> colorize -> enhance.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_dir (str): Directory to save processed frames
            scale (int): Scale factor for upscaling
            color_model (str): 'stable', 'artistic', or 'both' (default: 'stable')
            color_boost (float): Color saturation multiplier
        
        Returns:
            list: List of paths to processed frames
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                print(f"Processing frame {i+1}/{len(frame_paths)}...")
                base_name = os.path.basename(frame_path)
                file_name_wo_ext = os.path.splitext(base_name)[0]
                final_path = os.path.join(output_dir, f"{file_name_wo_ext}_restored.png")
                
                # Process the frame with our complete pipeline
                result_path = self.process_image(frame_path, output_dir, scale, color_model=color_model, color_boost=color_boost)
                
                if result_path and os.path.exists(result_path):
                    processed_paths.append(result_path)
                else:
                    # On failure, do NOT copy original or restored frame. Raise error.
                    raise RuntimeError(f"Processing failed for frame: {frame_path}")
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                raise  # Propagate error, do not fallback or copy original
        
        return processed_paths

# Simple test function if executed directly
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified image processor for both images and video frames")
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-o", "--output", help="Output directory (default: ./outputs)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor (default: 2)")
    args = parser.parse_args()
    
    processor = ImageProcessor(args.output)
    result = processor.process_image(args.input, args.output, args.scale)
    
    if result:
        print(f"Success! Final image: {result}")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main()
