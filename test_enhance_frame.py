import os
import torch
from enhanced_deoldify import enhance_image
from PIL import Image

def test_enhance_frame(skip_enhance=False):
    # Path to the input frame
    input_frame_path = r"C:\\prj\\full_restore\\dummy\\frame_00013.png"
    # Path to save the sharpened frame (pre-Real-ESRGAN)
    sharpened_frame_path = r"C:\\prj\\full_restore\\result_images\\frame_00013_sharpened.png"
    # Path to save the enhanced frame
    output_frame_path = r"C:\\prj\\full_restore\\result_images\\frame_00013_enhanced.png"

    # Ensure the input frame exists
    if not os.path.exists(input_frame_path):
        print(f"Input frame not found: {input_frame_path}")
        return

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enhance the frame
    try:
        enhance_image(input_frame_path, output_frame_path, device=device, sharpened_path=sharpened_frame_path, skip_enhance=skip_enhance)
        print(f"Sharpened image saved to: {sharpened_frame_path}")
        if not skip_enhance:
            print(f"Enhanced image saved to: {output_frame_path}")
        else:
            print(f"Enhancement skipped. Only sharpened image saved.")
    except Exception as e:
        print(f"Enhancement failed: {e}")

if __name__ == "__main__":
    # Set skip_enhance=True to skip enhancement
    test_enhance_frame(skip_enhance=False)
