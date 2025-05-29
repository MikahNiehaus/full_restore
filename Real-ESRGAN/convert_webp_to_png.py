from PIL import Image
import sys
import os

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_webp_to_png.py input_webp_path [output_directory]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    convert_webp_to_png(input_path, output_dir)
