from pathlib import Path
from shutil import copyfile
from PIL import Image

src = Path("C:/prj/full_restore/outputs/New folder/The-Spider-Woman-50.png")
dst = Path("C:/prj/full_restore/result_images/test_frame.png")

if not src.exists():
    print(f"[ERROR] Source image does not exist: {src}")
else:
    copyfile(src, dst)
    print(f"[INFO] Copied {src} to {dst}")
    # Verify the image can be opened
    try:
        with Image.open(dst) as img:
            img.verify()
        print("[SUCCESS] Image is a valid PNG.")
    except Exception as e:
        print(f"[ERROR] Copied file is not a valid image: {e}")
