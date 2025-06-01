"""
Simple DeOldify integration for CPU/GPU mode.
This module configures DeOldify to use CUDA by default, or CPU if --cpu is passed.
"""
import os
import sys
from pathlib import Path
import torch

# Force CPU mode only if --cpu is passed
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
    torch.set_num_threads(4)  # Limit CPU threads
    FORCE_CPU = True
else:
    FORCE_CPU = False

# Add DeOldify to path
deoldify_path = Path(__file__).parent / 'DeOldify'
if str(deoldify_path) not in sys.path:
    sys.path.append(str(deoldify_path))

# Import DeOldify modules with appropriate device configuration
from deoldify import device
from deoldify.device_id import DeviceId
if FORCE_CPU:
    device.set(device=DeviceId.CPU)  # Force CPU mode
else:
    device.set(device=DeviceId.GPU0)  # Use first GPU if available

from deoldify.visualize import get_video_colorizer

def get_video_colorizer_auto(render_factor=21):
    """
    Get a video colorizer configured for CUDA (default) or CPU (if --cpu)
    
    Args:
        render_factor: Quality factor for the colorization (higher = better quality but slower)
        
    Returns:
        A DeOldify VideoColorizer object ready for colorizing videos
    """
    return get_video_colorizer(render_factor=render_factor)
