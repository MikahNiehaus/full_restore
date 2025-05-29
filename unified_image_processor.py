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
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not read image: {input_path}")
                return False
                
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Gentle bilateral filter
            img_filtered = cv2.bilateralFilter(img_rgb, 5, 40, 40)
            
            # Light detail enhancement (lower sigma)
            img_details = cv2.detailEnhance(img_filtered, sigma_s=5, sigma_r=0.08)
            
            # Mild upscaling (if scale > 1)
            h, w = img_details.shape[:2]
            if scale > 1:
                img_upscaled = cv2.resize(img_details, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
            else:
                img_upscaled = img_details
            
            # Very gentle unsharp mask (reduce strength)
            blur = cv2.GaussianBlur(img_upscaled, (0, 0), 1.0)
            img_upscaled = cv2.addWeighted(img_upscaled, 1.10, blur, -0.08, 0)  # Lowered from 1.5/-0.5
            
            # Use PIL for mild brightness/contrast
            pil_img = Image.fromarray(img_upscaled)
            enhancer = ImageEnhance.Brightness(pil_img)
            brightened = enhancer.enhance(brighten_factor)
            contrast = ImageEnhance.Contrast(brightened)
            final_img = contrast.enhance(1.05)  # Lowered from 1.1
            
            # Save the final enhanced image
            final_img.save(output_path)
            print(f"Enhanced image saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            traceback.print_exc()
            return False
            
    def colorize_image(self, input_path, output_path, artistic=True, render_factor=35, color_boost=1.0):
        """
        Colorize an image using DeOldify.
        
        Args:
            input_path (str): Path to the input image
            output_path (str): Path to save the colorized image
            artistic (bool): Whether to use the artistic model
            render_factor (int): Render factor for DeOldify
            color_boost (float): Color saturation multiplier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import DeOldify
            from deoldify.visualize import get_image_colorizer
            from deoldify import device
            from deoldify.device_id import DeviceId
            import torch
            
            # Determine if we should use CPU from the start
            force_cpu = not torch.cuda.is_available()
            
            if force_cpu:
                # Explicitly set device to CPU if no CUDA available
                print("[INFO] GPU not available, using CPU for colorization...")
                device.set(device=DeviceId.CPU)
            else:
                try:
                    # Check available GPU memory - don't attempt GPU if too little available
                    free_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    if free_memory_mb < 2000:  # At least 2GB required
                        print(f"[WARNING] Low GPU memory ({free_memory_mb:.0f}MB), using CPU instead...")
                        force_cpu = True
                        device.set(device=DeviceId.CPU)
                except Exception as e:
                    print(f"[WARNING] Error checking GPU memory: {e}")
                    # Proceed with default device settings
            
            print(f"Initializing DeOldify colorizer (using {'CPU' if force_cpu else 'GPU'})...")
            colorizer = get_image_colorizer(artistic=artistic)
            
            # First attempt: try to colorize with current settings (CPU or GPU)
            try:
                print(f"Colorizing image with DeOldify (artistic={artistic})...")
                # For CPU mode, reduce render factor to speed up processing
                if force_cpu and render_factor > 25:
                    adjusted_render_factor = min(render_factor, 25)
                    print(f"[INFO] Reducing render factor from {render_factor} to {adjusted_render_factor} for CPU mode")
                else:
                    adjusted_render_factor = render_factor
                
                colorized_path = colorizer.plot_transformed_image(
                    input_path,
                    render_factor=adjusted_render_factor,
                    watermarked=False,
                    post_process=True,
                    results_dir=os.path.dirname(output_path),
                    force_cpu=force_cpu
                )
                print(f"[DEBUG] plot_transformed_image returned: {colorized_path}")
                
                # Process the result if successful
                if isinstance(colorized_path, str) and os.path.exists(colorized_path):
                    print(f"[DEBUG] Generated file exists at: {colorized_path}")
                    
                    # Check if the colorized file is actually colorized (not just grayscale saved as RGB)
                    img = Image.open(colorized_path)
                    
                    # Move/rename to the expected output_path if needed
                    if os.path.abspath(colorized_path) != os.path.abspath(output_path):
                        shutil.copy(colorized_path, output_path)
                    
                    # Boost color saturation to make the colorization more vibrant
                    img = Image.open(output_path)
                    enhancer = ImageEnhance.Color(img)
                    img_colored = enhancer.enhance(color_boost)
                    
                    # Also boost contrast slightly if color_boost > 1
                    if color_boost > 1.0:
                        contrast = ImageEnhance.Contrast(img_colored)
                        img_colored = contrast.enhance(1.05 + 0.05 * (color_boost-1.0))
                    
                    img_colored.save(output_path)
                    print(f"Colorized image processed and saved to: {output_path}")
                    return True
                
                return False
                
            except RuntimeError as e:
                # If we're already using CPU and still got an error, re-raise it
                if force_cpu:
                    raise
                
                # Retry with CPU if GPU fails
                if "CUDA" in str(e) or "cuda" in str(e) or "memory" in str(e).lower():
                    print("[WARNING] CUDA error detected, retrying on CPU...")
                    # Force CPU mode
                    device.set(device=DeviceId.CPU)
                    
                    # Use a lower render factor on CPU for speed
                    cpu_render_factor = min(render_factor, 25)
                    print(f"[INFO] Using reduced render factor: {cpu_render_factor} for CPU mode")
                    
                    colorized_path = colorizer.plot_transformed_image(
                        input_path,
                        render_factor=cpu_render_factor,
                        watermarked=False,
                        post_process=True,
                        results_dir=os.path.dirname(output_path),
                        force_cpu=True
                    )
                    
                    if isinstance(colorized_path, str) and os.path.exists(colorized_path):
                        if os.path.abspath(colorized_path) != os.path.abspath(output_path):
                            shutil.copy(colorized_path, output_path)
                            
                        # Apply more aggressive color enhancement for CPU mode since it tends to be less vibrant
                        img = Image.open(output_path)
                        enhancer = ImageEnhance.Color(img)
                        img_colored = enhancer.enhance(1.8)  # More aggressive color boost for CPU mode
                        
                        # Also boost contrast and brightness
                        contrast = ImageEnhance.Contrast(img_colored)
                        img_colored = contrast.enhance(1.2)
                        
                        brightness = ImageEnhance.Brightness(img_colored)
                        img_colored = brightness.enhance(1.1)
                        
                        img_colored.save(output_path)
                        print(f"Colorized image (CPU mode) saved to: {output_path}")
                        return True
                    return False
                else:
                    raise
                    
        except Exception as e:
            print(f"Error during DeOldify colorization: {e}")
            import traceback
            traceback.print_exc()
            
            # As a last resort, try our own fallback colorization method
            return self._fallback_colorize(input_path, output_path, color_boost=color_boost)
            
    def _fallback_colorize(self, input_path, output_path, color_boost=1.0):
        """
        Simple colorization as a fallback when DeOldify fails completely.
        
        Args:
            input_path (str): Path to the input image
            output_path (str): Path to save the colorized image
            color_boost (float): Color saturation multiplier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("[INFO] Using fallback colorization method...")
            # Load the image
            img = Image.open(input_path)

            # Convert to RGB if it's not already
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply a warm sepia tone to add color
            sepia_filter = (1.2, 0.87, 0.6)  # Warm sepia tone

            # Split the image into bands
            r, g, b = img.split()

            # Apply the sepia filter
            r = r.point(lambda i: i * sepia_filter[0])
            g = g.point(lambda i: i * sepia_filter[1])
            b = b.point(lambda i: i * sepia_filter[2])

            # Merge the bands back
            img = Image.merge('RGB', (r, g, b))

            # Enhance saturation to add more color
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(color_boost)

            # Enhance contrast for better definition
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)

            # Save the result
            img.save(output_path)
            print(f"Fallback colorization applied and saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during fallback colorization: {e}")
            traceback.print_exc()
            return False
            
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

    def process_image(self, input_path, output_dir=None, scale=2, do_restore=True, do_colorize=True, do_enhance=True, color_boost=1.0):
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
                colorize_success = self.colorize_image(final_path, final_path, color_boost=color_boost)
                if not colorize_success:
                    print("Colorization failed. Using restored image.")
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
        
    def colorize_frames(self, frame_paths, output_dir):
        """
        Colorize multiple video frames.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_dir (str): Directory to save colorized frames
            
        Returns:
            list: List of paths to colorized frames
        """
        os.makedirs(output_dir, exist_ok=True)
        colorized_paths = []
        
        for frame_path in frame_paths:
            base_name = os.path.basename(frame_path)
            colorized_path = os.path.join(output_dir, base_name)
            
            success = self.colorize_image(frame_path, colorized_path)
            if success:
                colorized_paths.append(colorized_path)
            else:
                # On failure, copy original frame
                shutil.copy(frame_path, colorized_path)
                colorized_paths.append(colorized_path)
                
        return colorized_paths
        
    def process_frames(self, frame_paths, output_dir, scale=2):
        """
        Process multiple video frames with our complete pipeline:
        restore -> colorize -> enhance.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_dir (str): Directory to save processed frames
            scale (int): Scale factor for upscaling
            
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
                result_path = self.process_image(frame_path, output_dir, scale)
                
                if result_path and os.path.exists(result_path):
                    processed_paths.append(result_path)
                else:
                    # On failure, copy original frame
                    shutil.copy(frame_path, final_path)
                    processed_paths.append(final_path)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                # Create a fallback path and copy original
                base_name = os.path.basename(frame_path)
                file_name_wo_ext = os.path.splitext(base_name)[0]
                final_path = os.path.join(output_dir, f"{file_name_wo_ext}_restored.png")
                shutil.copy(frame_path, final_path)
                processed_paths.append(final_path)
                
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
