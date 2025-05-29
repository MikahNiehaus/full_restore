"""
Utility script to copy the restore_photo.py from Real-ESRGAN to the root directory
This helps avoid import issues
"""
import os
import sys
import shutil

def copy_restore_photo():
    src_path = os.path.join('Real-ESRGAN', 'restore_photo.py')
    dst_path = 'restore_photo.py'
    
    if os.path.exists(src_path):
        print(f"Copying {src_path} to {dst_path}...")
        shutil.copy2(src_path, dst_path)
        print("Done!")
    else:
        print(f"ERROR: Source file not found: {src_path}")

if __name__ == "__main__":
    copy_restore_photo()
