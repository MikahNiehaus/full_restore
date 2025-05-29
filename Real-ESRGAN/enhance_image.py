import os
import sys
from PIL import Image, ImageEnhance, ImageFilter

def enhance_image(input_path, output_dir=None, scale=2):
    """
    Enhance an image using PIL (Pillow) library
    as a fallback when Real-ESRGAN has compatibility issues
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
        
        # Resize the image to simulate super resolution (optional)
        # Use LANCZOS resampling for high quality
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.LANCZOS)        # Get the filename without extension
        base_name = os.path.basename(input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        # Create the output path
        output_path = os.path.join(output_dir, f"{file_name_without_ext}_enhanced.png")

        # Save the enhanced image
        img.save(output_path, "PNG")

        print(f"Successfully enhanced {input_path} and saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhance_image.py input_image_path [output_directory]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    enhance_image(input_path, output_dir)
