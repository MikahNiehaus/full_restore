import os
import sys
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageChops
import numpy as np

# Apply torchvision compatibility patch for Real-ESRGAN
try:
    # First try to get patch from current directory
    import importlib.util
    patch_paths = [
        os.path.join(os.path.dirname(__file__), "realesrgan_patch.py"),  # Same directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "realesrgan_patch.py"),  # Absolute path to same dir
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "realesrgan_patch.py")  # Parent directory
    ]
    
    patch_loaded = False
    for patch_path in patch_paths:
        if os.path.exists(patch_path):
            spec = importlib.util.spec_from_file_location("realesrgan_patch", patch_path)
            realesrgan_patch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(realesrgan_patch)
            print(f"[INFO] Applied Real-ESRGAN compatibility patch from {patch_path}")
            patch_loaded = True
            break
    
    if not patch_loaded:
        print("[WARNING] Could not find realesrgan_patch.py in any expected locations")
except Exception as e:
    print(f"[WARNING] Could not apply Real-ESRGAN compatibility patch: {e}")

def restore_old_photo(input_path, output_dir=None, scale=8):
    """
    AI-powered restoration for old photos: first upscale with Real-ESRGAN (large amount), then colorize with DeOldify.
    """
    # Import needed modules inside the function to avoid scope issues
    import os
    import sys
    import torch
    import inspect
    import cv2  # Added import for OpenCV
    
    try:
        from realesrgan import RealESRGANer
        # Make sure input_path is an absolute path
        if not os.path.isabs(input_path):
            input_path = os.path.abspath(input_path)
        print(f"Full input path: {input_path}")
        if not os.path.exists(input_path):
            print(f"Error: File does not exist: {input_path}")
            return None
            
        # Open the image using PIL
        img = Image.open(input_path).convert('RGB')
        
        # Prepare output directory
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)
        
        upscaled_path = os.path.join(output_dir, f"{file_name_without_ext}_upscaled.png")
        final_output_path = os.path.join(output_dir, f"{file_name_without_ext}_restored.png")        # Step 1: Use Real-ESRGAN for strong AI upscaling
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # Look for weights in several possible locations
            possible_weight_dirs = [
                os.path.join(os.path.dirname(__file__), 'weights'),  # Current directory/weights
                os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights'),  # Current directory/Real-ESRGAN/weights
                os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Real-ESRGAN', 'weights')  # ../Real-ESRGAN/weights
            ]
            
            model_path = None
            for weights_dir in possible_weight_dirs:
                potential_path = os.path.join(weights_dir, 'realesr-general-x4v3.pth')
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
                    
            if not model_path:
                raise FileNotFoundError(f"Real-ESRGAN model file not found in any of: {possible_weight_dirs}")
            
            print(f"[INFO] Using Real-ESRGAN AI upscaling with model: {model_path}")
            
            # Initialize the upsampler with the direct model path
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                dni_weight=None,
                device=device,
                tile=400  # Use tiling to reduce memory usage
            )
            
            # Convert PIL to numpy array
            np_img = np.array(img)
            
            # Enhance using Real-ESRGAN
            output, _ = upsampler.enhance(np_img)
            
            # Convert back to PIL and save
            upscaled_img = Image.fromarray(output)
            
        except Exception as e:
            print(f"[WARNING] Real-ESRGAN AI upscaling failed: {e}")
            print("[INFO] Using OpenCV for upscaling (fallback method)")
            
            # Convert PIL image to cv2
            np_img = np.array(img)
            cv_img = np_img[:, :, ::-1].copy()  # RGB to BGR for OpenCV
            
            # Use OpenCV to upscale
            height, width = cv_img.shape[:2]
            upscaled = cv2.resize(cv_img, (width * scale, height * scale), 
                                interpolation=cv2.INTER_CUBIC)
            
            # Apply detail enhancement
            upscaled = cv2.detailEnhance(upscaled, sigma_s=10, sigma_r=0.15)
            
            # Convert back to PIL and save
            upscaled_img = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        
        # Save the upscaled image
        upscaled_img.save(upscaled_path, "PNG")
        upscaled_img.save(upscaled_path, "PNG")
        print(f"[AI] Upscaled image saved to: {upscaled_path}")
        # Step 2: Colorize the upscaled image with DeOldify
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '../DeOldify'))
            from deoldify.visualize import get_image_colorizer
            colorizer = get_image_colorizer(artistic=True)
            colorized = colorizer.plot_transformed_image(
                upscaled_path,
                render_factor=35,
                watermarked=False,
                post_process=True,
                results_dir=output_dir,
                force_cpu=False
            )
            # colorizer returns the path, but we want to ensure the final output is named consistently
            if isinstance(colorized, str):
                # If a path is returned, rename/move to final_output_path
                import shutil
                shutil.move(colorized, final_output_path)
            elif hasattr(colorized, 'save'):
                colorized.save(final_output_path)
            print(f"[AI] Colorized image saved to: {final_output_path}")
        except Exception as e:
            print(f"[WARNING] DeOldify colorization failed: {e}")
            upscaled_img.save(final_output_path, "PNG")
            print(f"[FALLBACK] Saved only upscaled image as restored: {final_output_path}")
        return final_output_path
    except ImportError:
        print("[ERROR] Real-ESRGAN is not installed. Please install realesrgan-pytorch.")
        return None
    except Exception as e:
        print(f"Error restoring photo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Restore old photographs.")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor (default: 2)")

    args = parser.parse_args()

    # Check if input is a directory or file
    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')):
                file_path = os.path.join(args.input, filename)
                restore_old_photo(file_path, args.output, args.scale)
    else:
        # Process a single image
        restore_old_photo(args.input, args.output, args.scale)