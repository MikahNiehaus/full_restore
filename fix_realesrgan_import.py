"""
This is a helper script to fix RealESRGAN import issues.
"""

import os
import sys
import importlib
from importlib import util

def fix_realesrgan_import():
    """
    Fix the import path for RealESRGAN.
    
    This function adds the Real-ESRGAN directory to the sys.path and applies
    any necessary compatibility patches.
    
    Returns:
        bool: True if successful, False otherwise
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add Real-ESRGAN to path
    realesrgan_paths = [
        os.path.join(current_dir, 'Real-ESRGAN'),
        os.path.join(os.path.dirname(current_dir), 'Real-ESRGAN')
    ]
    
    # Find a valid Real-ESRGAN path
    realesrgan_path = None
    for path in realesrgan_paths:
        if os.path.exists(path) and os.path.isdir(path):
            realesrgan_path = path
            if path not in sys.path:
                sys.path.append(path)
            print(f"Added Real-ESRGAN path: {path}")
            break
    
    if not realesrgan_path:
        print("Real-ESRGAN directory not found")
        return False
        
    # Apply compatibility patch if available
    patch_paths = [
        os.path.join(current_dir, "realesrgan_patch.py"),
        os.path.join(os.path.dirname(current_dir), "realesrgan_patch.py")
    ]
    
    patch_applied = False
    for patch_path in patch_paths:
        if os.path.exists(patch_path):
            try:
                spec = util.spec_from_file_location("realesrgan_patch", patch_path)
                realesrgan_patch = util.module_from_spec(spec)
                spec.loader.exec_module(realesrgan_patch)
                print(f"Applied RealESRGAN compatibility patch from: {patch_path}")
                patch_applied = True
                break
            except Exception as e:
                print(f"Failed to apply patch from {patch_path}: {e}")
    
    # Test importing RealESRGANer
    try:
        from realesrgan import RealESRGANer
        print("Successfully imported RealESRGANer")
        return True
    except ImportError as e:
        print(f"Failed to import RealESRGANer: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error importing RealESRGANer: {e}")
        return False

if __name__ == "__main__":
    success = fix_realesrgan_import()
    print(f"Import fix {'succeeded' if success else 'failed'}")
