import os
import sys
import requests
import shutil
from pathlib import Path

def setup_deoldify(download_models=True):
    """
    Set up DeOldify environment for colorization
    """
    # Define models directory
    deoldify_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify')
    models_dir = os.path.join(deoldify_dir, 'models')

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Create simpler version of DeOldify colorizer
    print("Creating simplified DeOldify colorization module...")

    # Create a simple colorizer module
    colorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_colorizer.py')

    # Define the content of the simple colorizer
    colorizer_content = '''import os
import sys
import numpy as np
from PIL import Image, ImageEnhance

# Try to import DeOldify if available
DEOLDIFY_AVAILABLE = False
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify'))
    from deoldify.visualize import get_image_colorizer
    DEOLDIFY_AVAILABLE = True
except Exception as e:
    print(f"DeOldify not fully available: {e}")
    DEOLDIFY_AVAILABLE = False

def get_colorizer(artistic=True):
    """
    Get the DeOldify colorizer if available, otherwise return None
    """
    if DEOLDIFY_AVAILABLE:
        try:
            return get_image_colorizer(artistic=artistic)
        except Exception as e:
            print(f"Error loading DeOldify colorizer: {e}")
            return None
    return None

def colorize_image(input_path, output_path=None, artistic=True, render_factor=35):
    """
    Colorize an image using DeOldify if available, otherwise use a simple colorization method
    """
    try:
        if DEOLDIFY_AVAILABLE:
            # Try using DeOldify
            colorizer = get_colorizer(artistic=artistic)
            if colorizer:
                # Use DeOldify for colorization
                print("Using DeOldify for advanced colorization...")
                result = colorizer.get_transformed_image(input_path, render_factor=render_factor, watermarked=False)

                if output_path:
                    result.save(output_path)
                    print(f"DeOldify colorization saved to: {output_path}")
                return result

        # Fallback to simple colorization
        print("Using simplified colorization method...")
        return simple_colorize(input_path, output_path)

    except Exception as e:
        print(f"Error during colorization: {e}")
        return None

def simple_colorize(input_path, output_path=None):
    """
    Simple colorization as a fallback when DeOldify is not available
    """
    try:
        # Load the image
        img = Image.open(input_path)

        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply a slight sepia tone to warm up the image
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
        img = enhancer.enhance(1.5)  # Increase saturation

        # Enhance contrast for better definition
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Save if output path provided
        if output_path:
            img.save(output_path)
            print(f"Simple colorization saved to: {output_path}")

        return img
    except Exception as e:
        print(f"Error during simple colorization: {e}")
        return None

if __name__ == "__main__":
    # Test the colorization
    import argparse

    parser = argparse.ArgumentParser(description="Colorize B&W images")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-a", "--artistic", action="store_true", help="Use artistic mode for DeOldify")
    parser.add_argument("-r", "--render-factor", type=int, default=35, help="Render factor for DeOldify (10-45)")

    args = parser.parse_args()

    colorize_image(args.input, args.output, args.artistic, args.render_factor)
'''

    # Write the simple colorizer module
    with open(colorizer_path, 'w') as f:
        f.write(colorizer_content)

    print(f"Created simplified colorizer module: {colorizer_path}")

    # Download models if requested
    if download_models:
        print("Note: Due to file size constraints, DeOldify models can't be downloaded automatically.")
        print("For full DeOldify functionality, please follow these steps:")
        print("1. Visit the DeOldify repository: https://github.com/jantic/DeOldify")
        print("2. Follow the instructions to download the pre-trained models")
        print(f"3. Place the downloaded models in: {models_dir}")
        print("4. The simplified colorization method will be used until the models are properly set up.")

    # Create directories needed for our restoration and colorization process
    os.makedirs('inputs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('processed', exist_ok=True)

    print("\nSetup complete! You can now use the restoration and colorization scripts.")
    print("1. Place images in the 'inputs' folder")
    print("2. Run the auto_watch_simple.py script to automatically process new images")
    print("3. Results will be saved in the 'outputs' folder")

if __name__ == "__main__":
    setup_deoldify()
