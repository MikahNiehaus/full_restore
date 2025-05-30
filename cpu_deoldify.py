"""
Simple DeOldify integration for CPU-only mode.
This module configures DeOldify to work in CPU mode.
"""
import os
import sys
from pathlib import Path
import torch

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
torch.set_num_threads(4)  # Limit CPU threads

# Add DeOldify to path
deoldify_path = Path(__file__).parent / 'DeOldify'
if str(deoldify_path) not in sys.path:
    sys.path.append(str(deoldify_path))

# Import DeOldify modules with CPU configuration
from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.CPU)  # Force CPU mode

# Import necessary visualization modules
from deoldify.visualize import get_video_colorizer

def get_cpu_video_colorizer(render_factor=21):
    """
    Get a video colorizer configured for CPU usage
    
    Args:
        render_factor: Quality factor for the colorization (higher = better quality but slower)
        
    Returns:
        A DeOldify VideoColorizer object ready for colorizing videos
    """
    return get_video_colorizer(render_factor=render_factor)
