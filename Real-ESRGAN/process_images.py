import os
import sys
from PIL import Image, ImageEnhance, ImageFilter

def convert_webp_to_png(input_path, output_dir=None):
    """
    Convert a WEBP image to PNG format
    """
    try:
        # Open the WEBP image
        img = Image.open(input_path)

        # Get the filename without extension
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        # Create the output path
        output_path = os.path.join(output_dir, f"{file_name_without_ext}.png")

        # Save as PNG
        img.save(output_path, "PNG")

        print(f"Successfully converted {input_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def enhance_image(input_path, output_dir=None, scale=2, output_suffix="enhanced"):
    """
    Enhance an image using PIL (Pillow) library
    Works with multiple image formats including WEBP
    """
    try:
        # Open the image using PIL
        img = Image.open(input_path)

        # Apply a slight sharpen filter
        img = img.filter(ImageFilter.SHARPEN)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)

        # Enhance color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

        # Resize the image to simulate super resolution
        # Use LANCZOS resampling for high quality
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

        # Get the filename without extension
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        # Create the output path
        output_path = os.path.join(output_dir, f"{file_name_without_ext}_{output_suffix}.png")

        # Save the enhanced image
        img.save(output_path, "PNG")

        print(f"Successfully enhanced {input_path} and saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return None

def process_image(input_path, output_dir=None, scale=2, convert_only=False):
    """
    Process an image - either convert from WEBP to PNG, or enhance it, or both
    """
    # Detect if it's a WEBP file
    is_webp = input_path.lower().endswith('.webp')

    if is_webp and convert_only:
        # Just convert the WEBP to PNG
        return convert_webp_to_png(input_path, output_dir)
    else:
        # Enhance the image directly
        return enhance_image(input_path, output_dir, scale)

def process_directory(input_dir, output_dir=None, scale=2, convert_only=False, file_types=None):
    """
    Process all images in a directory
    """
    if file_types is None:
        file_types = ['.webp', '.png', '.jpg', '.jpeg']

    if output_dir is None:
        output_dir = input_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    processed_files = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        # Check if the file has a supported extension
        if any(file_path.lower().endswith(ext) for ext in file_types):
            result_path = process_image(file_path, output_dir, scale, convert_only)
            if result_path:
                processed_files.append(result_path)

    return processed_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and enhance images including WEBP files.")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor for enhancement (default: 2)")
    parser.add_argument("-c", "--convert-only", action="store_true", help="Only convert WEBP to PNG without enhancement")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")

    args = parser.parse_args()

    # Check if input is a directory or file
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.scale, args.convert_only)
    else:
        process_image(args.input, args.output, args.scale, args.convert_only)
