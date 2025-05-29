import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeOldify'))
from deoldify.visualize import get_image_colorizer

input_dir = 'restored'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
colorizer = get_image_colorizer(artistic=False)

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp")):
        in_path = os.path.join(input_dir, fname)
        file_name_wo_ext = os.path.splitext(fname)[0]
        colorized_path = os.path.join(output_dir, f"{file_name_wo_ext}_colorized.png")
        print(f"Colorizing: {fname}")
        result = colorizer.get_transformed_image(in_path, render_factor=35, watermarked=False)
        result.save(colorized_path)
