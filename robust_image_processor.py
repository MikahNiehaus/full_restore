"""
A robust and simplified implementation of the restoration and colorization process
with reliable fallbacks and proper error handling.
"""
import os
import sys
import shutil
import argparse
from PIL import Image
import traceback
import cv2
import numpy as np

def enhance_image(input_path, output_path, scale=2):
    """
    Enhance an image using OpenCV-based upscaling and detail enhancement.
    This is a reliable fallback method when AI upscaling fails.
    """
    try:
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Could not read image: {input_path}")
            return False
            
        # Upscale
        height, width = img.shape[:2]
        upscaled = cv2.resize(img, (width * scale, height * scale), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        # Apply detail enhancement
        enhanced = cv2.detailEnhance(filtered, sigma_s=10, sigma_r=0.15)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Save the enhanced image
        cv2.imwrite(output_path, sharpened)
        print(f"Enhanced image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error enhancing image: {e}")
        traceback.print_exc()
        return False

def colorize_with_deoldify(input_path, output_path, artistic=True, render_factor=35, force_cpu=False):
    """
    Colorize an image using DeOldify if available.
    """
    try:
        # First make sure DeOldify is in the path
        deoldify_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
        if deoldify_dir not in sys.path:
            sys.path.insert(0, deoldify_dir)
        
        # Import DeOldify
        from deoldify.visualize import get_image_colorizer
        
        colorizer = get_image_colorizer(artistic=artistic)
        
        # Try GPU, fallback to CPU if needed
        try:
            colorized_path = colorizer.plot_transformed_image(
                input_path,
                render_factor=render_factor,
                watermarked=False,
                post_process=True,
                results_dir=os.path.dirname(output_path),
                force_cpu=force_cpu
            )
            
            # If a path is returned, rename to our desired output path
            if isinstance(colorized_path, str) and os.path.exists(colorized_path):
                if os.path.abspath(colorized_path) != os.path.abspath(output_path):
                    shutil.move(colorized_path, output_path)
                return True
            elif hasattr(colorized_path, 'save'):
                # If an image is returned, save it
                colorized_path.save(output_path)
                return True
                
            return False
        except RuntimeError as e:
            if (not force_cpu) and ("CUDA" in str(e) or "cuda" in str(e)):
                print("[WARNING] CUDA error detected, retrying on CPU...")
                return colorize_with_deoldify(input_path, output_path, artistic, render_factor, force_cpu=True)
            else:
                raise
    except Exception as e:
        print(f"Error during DeOldify colorization: {e}")
        traceback.print_exc()
        return False

def process_image(input_path, output_dir=None, scale=2):
    """
    Process a single image: colorize and enhance it.
    Returns the path to the final processed image or None if failed.
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return None
        
        base_name = os.path.basename(input_path)
        file_name_wo_ext = os.path.splitext(base_name)[0]
        
        # Prepare output paths
        colorized_path = os.path.join(output_dir, f"{file_name_wo_ext}_colorized.png")
        enhanced_path = os.path.join(output_dir, f"{file_name_wo_ext}_enhanced.png")
        final_path = os.path.join(output_dir, f"{file_name_wo_ext}_restored.png")
        
        # Step 1: First try to colorize with DeOldify
        print(f"Step 1: Colorizing {base_name}...")
        colorize_success = colorize_with_deoldify(input_path, colorized_path)
        
        if not colorize_success:
            print("Colorization failed. Proceeding with original image.")
            # If colorization fails, use the original image
            img = Image.open(input_path)
            img.save(colorized_path)
        
        # Step 2: Enhance the colorized image
        print(f"Step 2: Enhancing image quality...")
        enhance_success = enhance_image(colorized_path, enhanced_path, scale)
        
        if enhance_success:
            # Copy enhanced as final output
            shutil.copy(enhanced_path, final_path)
            print(f"Final restored image saved to: {final_path}")
            return final_path
        else:
            # If enhancement fails, use colorized as final
            shutil.copy(colorized_path, final_path)
            print(f"Enhancement failed. Using colorized image as final: {final_path}")
            return final_path
            
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Robust image restoration and colorization")
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-o", "--output", help="Output directory (default: ./outputs)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor (default: 2)")
    args = parser.parse_args()
    
    print(f"\nProcessing: {args.input}")
    final_path = process_image(args.input, args.output, args.scale)
    
    if final_path:
        print(f"\nSuccess! Final image: {final_path}")
    else:
        print("\nProcessing failed.")

if __name__ == "__main__":
    main()
