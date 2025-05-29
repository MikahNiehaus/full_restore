import os
from restore_photo import restore_old_photo

input_dir = 'inputs'
output_dir = 'restored'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp")):
        in_path = os.path.join(input_dir, fname)
        print(f"Restoring: {fname}")
        restore_old_photo(in_path, output_dir)
