import os
import sys
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import time
import traceback
from datetime import datetime

def simulate_colorization(img):
    """
    Simulate colorization by enhancing colors
    This is a fallback when DeOldify is not available
    """
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

    return img

def restore_and_colorize(input_path, output_dir=None, processed_dir=None, scale=2):
    """
    Restore and add color to an old black and white image
    """
    try:
        # Step 1: Restore the image using our custom restoration
        print(f"\nProcessing: {os.path.basename(input_path)}")
        print("Step 1: Restoring image quality...")

        from restore_photo import restore_old_photo

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

        os.makedirs(output_dir, exist_ok=True)        # Make sure input_path is absolute
        if not os.path.isabs(input_path):
            input_path = os.path.abspath(input_path)
            
        # Get the restored image path
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        restored_path = os.path.join(output_dir, f"{file_name_without_ext}_restored.png")
        
        # Perform restoration
        print(f"Restoring from: {input_path}")
        restored_path = restore_old_photo(input_path, output_dir, scale)
        
        if not restored_path or not os.path.exists(restored_path):
            print("Restoration failed. Attempting to use original image for colorization.")
            # If restoration fails, use the original image for colorization
            restored_path = input_path        # Step 2: Add color to the restored image
        print("Step 2: Adding color to the image...")
        
        # Try to use the simple_colorizer module
        try:
            from simple_colorizer import colorize_image
            
            # Save the colorized version
            colorized_path = os.path.join(output_dir, f"{file_name_without_ext}_restored_colorized.png")
            
            # Colorize the image
            print(f"Colorizing image: {restored_path}")
            colorized_img = colorize_image(restored_path, colorized_path, artistic=True)
            
            if colorized_img:
                print(f"Colorized image saved to: {colorized_path}")
            else:
                print("Colorization module failed, falling back to simple colorization")
                restored_img = Image.open(restored_path)
                colorized_img = simulate_colorization(restored_img)
                colorized_img.save(colorized_path)
                print(f"Simple colorized image saved to: {colorized_path}")
                
        except ImportError as e:
            print(f"Colorization module not found: {e}")
            print("Using built-in colorization method")
            
            # Load the restored image
            restored_img = Image.open(restored_path)
            
            # Apply our color enhancement
            colorized_img = simulate_colorization(restored_img)
            
            # Save the colorized version
            colorized_path = os.path.join(output_dir, f"{file_name_without_ext}_restored_colorized.png")
            colorized_img.save(colorized_path)
            print(f"Colorized image saved to: {colorized_path}")

        # Step 3: Create a comparison with original, restored, and colorized versions
        print("Step 3: Creating comparison image...")

        # Load the original image
        original = Image.open(input_path)

        # Resize to match dimensions
        width, height = restored_img.size
        original = original.resize((width, height), Image.LANCZOS)

        # Create a new image with all three side by side
        gap = 20
        comparison = Image.new('RGB', (width * 3 + gap * 2, height + 40), (255, 255, 255))

        # Paste the images
        comparison.paste(original, (0, 40))
        comparison.paste(restored_img, (width + gap, 40))
        comparison.paste(colorized_img, (width * 2 + gap * 2, 40))

        # Add text labels
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(comparison)
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fall back to default font
            font = ImageFont.load_default()

        draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
        draw.text((width + gap + 10, 10), "Restored", fill=(0, 0, 0), font=font)
        draw.text((width * 2 + gap * 2 + 10, 10), "Colorized", fill=(0, 0, 0), font=font)

        # Save the comparison
        comparison_path = os.path.join(output_dir, f"{file_name_without_ext}_full_comparison.png")
        comparison.save(comparison_path)
        print(f"Full comparison image saved to: {comparison_path}")

        # Move the original file to processed directory if specified
        if processed_dir:
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, base_name)
            shutil.move(input_path, processed_path)
            print(f"Original image moved to: {processed_path}")

        return colorized_path

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        traceback.print_exc()
        return None

def process_input_directory(input_dir="inputs", output_dir="outputs", processed_dir="processed", scale=2):
    """
    Process all images in the input directory
    """
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Get all image files
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')):
            image_files.append(os.path.join(input_dir, file))

    total_files = len(image_files)
    if total_files == 0:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {total_files} image files to process")
    start_time = time.time()

    # Process each image
    for i, file_path in enumerate(image_files):
        print(f"\n[{i+1}/{total_files}] Processing: {os.path.basename(file_path)}")
        restore_and_colorize(file_path, output_dir, processed_dir, scale)

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Processed {total_files} images in {elapsed_time:.1f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Restore and colorize old photographs")
    parser.add_argument("-i", "--input", default="inputs", help="Input directory (default: inputs)")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("-p", "--processed", default="processed", help="Directory to move processed originals (default: processed)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor for restoration (default: 2)")
    parser.add_argument("-f", "--file", help="Process a single file instead of directory")

    args = parser.parse_args()

    if args.file:
        # Process a single file
        restore_and_colorize(args.file, args.output, args.processed, args.scale)
    else:
        # Process directory
        process_input_directory(args.input, args.output, args.processed, args.scale)
