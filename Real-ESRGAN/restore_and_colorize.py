import os
import sys
import shutil
from PIL import Image
import time
import traceback
from datetime import datetime

# Add DeOldify to the path (robust to script location)
deoldify_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeOldify'))
if deoldify_dir not in sys.path:
    sys.path.insert(0, deoldify_dir)
print('PYTHONPATH sys.path:', sys.path)

# Add root to path for unified_image_processor
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from unified_image_processor import ImageProcessor

def check_deoldify_available():
    """Check if DeOldify is properly installed and available"""
    try:
        from deoldify.visualize import get_image_colorizer
        return True
    except ImportError:
        print("DeOldify is not properly set up. Make sure it's installed correctly.")
        return False

def restore_and_colorize(input_path, output_dir=None, processed_dir=None, scale=2, artistic=True, force_cpu=False):
    """
    Colorize first, then restore the colorized image using the unified processor.
    """
    try:
        print(f"\nProcessing: {os.path.basename(input_path)}")
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Use the unified image processor for both steps
        processor = ImageProcessor(output_dir)
        final_path = processor.process_image(input_path, output_dir, scale)

        if final_path and os.path.exists(final_path):
            print(f"[SUCCESS] Final restored image saved to: {final_path}")
        else:
            print("[ERROR] Restoration failed. Aborting without fallback.")
            raise RuntimeError("Restoration and colorization failed - no fallback allowed")

        # Optionally move the original file to processed_dir
        if processed_dir:
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, base_name)
            shutil.move(input_path, processed_path)
            print(f"Original image moved to: {processed_path}")

        # Optionally create a comparison image (original, colorized, restored)
        try:
            colorized_path = os.path.join(output_dir, f"{file_name_without_ext}.png")
            restored_path = os.path.join(output_dir, f"{file_name_without_ext}_restored.png")
            original = Image.open(input_path)
            colorized = Image.open(colorized_path)
            restored = Image.open(restored_path)
            width, height = colorized.size
            original = original.resize((width, height), Image.LANCZOS)
            restored = restored.resize((width, height), Image.LANCZOS)
            gap = 20
            comparison = Image.new('RGB', (width * 3 + gap * 2, height), (255, 255, 255))
            comparison.paste(original, (0, 0))
            comparison.paste(colorized, (width + gap, 0))
            comparison.paste(restored, (width * 2 + gap * 2, 0))
            comparison_path = os.path.join(output_dir, f"{file_name_without_ext}_comparison.png")
            comparison.save(comparison_path)
            print(f"Comparison image saved to: {comparison_path}")
        except Exception as e:
            print(f"[WARNING] Could not create comparison image: {e}")

    except Exception as e:
        print(f"[ERROR] Exception in restore_and_colorize: {e}")
        traceback.print_exc()

def process_input_directory(input_dir="inputs", output_dir="outputs", processed_dir="processed", scale=2, artistic=True):
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
        restore_and_colorize(file_path, output_dir, processed_dir, scale, artistic=True)

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Processed {total_files} images in {elapsed_time:.1f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Restore and colorize old photographs using unified processor")
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-o", "--output", help="Output directory (default: ./outputs)")
    parser.add_argument("-p", "--processed", help="Directory to move processed originals (default: processed)")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor (default: 2)")
    args = parser.parse_args()
    restore_and_colorize(args.input, args.output, args.processed, args.scale)
