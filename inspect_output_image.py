from PIL import Image
import sys
import os

output_path = os.path.join('outputs', 'frame_000003.png')
if not os.path.exists(output_path):
    print(f"File not found: {output_path}")
    sys.exit(1)

img = Image.open(output_path)
print(f"Image mode: {img.mode}")
print(f"Image size: {img.size}")

# Check if image is grayscale or color
if img.mode in ("L", "1"):
    print("The image is grayscale.")
elif img.mode in ("RGB", "RGBA"):
    # Check if all channels are equal (still grayscale)
    arr = img.convert("RGB")
    pixels = arr.getdata()
    is_gray = all(r == g == b for r, g, b in pixels)
    if is_gray:
        print("The image is RGB but all channels are equal (grayscale).")
    else:
        print("The image is in color (RGB).")
else:
    print("Unknown image mode.")
