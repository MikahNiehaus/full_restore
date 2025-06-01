"""
Test script to check if DeOldify colorization works with the current torch_safety_patch.
This will attempt to load the DeOldify colorizer and colorize a single test frame.
"""

import sys
from pathlib import Path

# Patch torch before importing DeOldify
import torch_safety_patch  # noqa: F401

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))
from deoldify.visualize import get_video_colorizer

import cv2
import torch

def test_colorization():
    print("[TEST] Starting DeOldify colorization test...")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Current device: cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print("[INFO] Running on CPU")
    colorizer = get_video_colorizer(render_factor=21)
    test_image_path = Path(__file__).parent / 'result_images' / 'test_frame.png'
    if not test_image_path.exists():
        print(f"[ERROR] Test image not found: {test_image_path}")
        return False
    try:
        result = colorizer.vis.get_transformed_image(
            str(test_image_path),
            render_factor=21,
            watermarked=False,
            post_process=True
        )
        output_path = Path(__file__).parent / 'result_images' / 'test_frame_colorized.png'
        result.save(str(output_path))
        print(f"[SUCCESS] Colorization completed. Output saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Colorization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_colorization()
