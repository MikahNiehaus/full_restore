"""
Enhanced DeOldify module that provides patched versions of DeOldify functionality.
This module handles the dummy directory issue.
"""

import os
import sys
import torch
from pathlib import Path

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)  # Limit CPU threads to 4

# Add DeOldify to path
deoldify_path = Path(__file__).parent / 'DeOldify'
if str(deoldify_path) not in sys.path:
    sys.path.append(str(deoldify_path))

# Import our patch_deoldify and apply patches
print("[INFO] Applying DeOldify patches to bypass dummy directory...")
from patch_deoldify import patch_deoldify
patch_deoldify()

# Now import DeOldify modules
from deoldify.visualize import get_video_colorizer as original_get_video_colorizer

def get_enhanced_colorizer(render_factor=40):
    """
    Get a patched version of the video colorizer that doesn't require
    the dummy directory for initialization.
    
    Args:
        render_factor: Quality factor for the colorization (higher = better quality but slower)
        
    Returns:
        A DeOldify VideoColorizer object ready for colorizing videos
    """
    try:
        print("[INFO] Setting up enhanced video colorizer with render_factor={0}...".format(render_factor))
        colorizer = original_get_video_colorizer(render_factor=render_factor)
        print("[INFO] Video colorizer initialized successfully")
        return colorizer
    except Exception as e:
        print(f"[ERROR] Failed to initialize DeOldify: {e}")
        import traceback
        traceback.print_exc()
        raise
