import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify'))
from deoldify.visualize import get_image_colorizer
from restore_photo import restore_old_photo

INPUT_DIR = 'inputs'
RESTORED_DIR = 'restored'
OUTPUT_DIR = 'outputs'
os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')

# 1. Restore all images
restored_files = []
for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(image_exts):
        in_path = os.path.join(INPUT_DIR, fname)
        print(f"Restoring: {fname}")
        restored_path = restore_old_photo(in_path, RESTORED_DIR)
        restored_files.append(restored_path)

# 2. Colorize all restored images
colorizer = get_image_colorizer(artistic=False)
for restored_path in restored_files:
    fname = os.path.basename(restored_path)
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_colorized.png")
    print(f"Colorizing: {fname}")
    result = colorizer.get_transformed_image(restored_path, render_factor=35, watermarked=False)
    result.save(out_path)
print("Batch image restoration and colorization complete.")
