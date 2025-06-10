#!/usr/bin/env python3
"""
GPU-optimized version of the simple_run.py script
This version explicitly forces GPU usage for both colorization and restoration
"""
import os
import sys
import time
import torch
from pathlib import Path

# If explicit CPU mode is requested, honor it
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print('[INFO] Forcing CPU mode (CUDA disabled)')
    GPU_MODE = False
else:
    # Force GPU mode by default
    GPU_MODE = True
    print('[INFO] Running in GPU-optimized mode')

# Import our PyTorch patch first to fix loading issues
from torch_safety_patch import *

# Import our GPU accelerator to ensure both components use GPU
if GPU_MODE:
    from force_gpu import GPUAccelerator
    GPUAccelerator.setup_gpu()

# Check one more time for GPU
if GPU_MODE and not torch.cuda.is_available():
    print("[WARNING] CUDA is not available despite requesting GPU mode!")
    print("[WARNING] Falling back to CPU mode")
    GPU_MODE = False
elif GPU_MODE:
    # Display GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] Using GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    
    # Enable performance optimizations
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] cuDNN optimization enabled: version {torch.backends.cudnn.version()}")
else:
    print("[INFO] Running in CPU mode")

# Import the simplified watchdog
from simple_watchdog import VideoWatchdog, YOUTUBE_UPLOADER_AVAILABLE

def show_help():
    """Display help information for the script"""
    print("\nDeOldify Full Restore Pipeline (GPU-optimized)")
    print("=========================================")
    print("Options:")
    print("  --help       Show this help message and exit")
    print("  --cpu        Force CPU mode even if GPU is available (much slower)")
    print("  --do-enhance Enable Real-ESRGAN enhancement (default: off)")
    print("\nUsage:")
    print("  python gpu_run.py           # Run with GPU acceleration")
    print("  python gpu_run.py --cpu     # Force CPU mode")
    print("\nPlace video files in the 'inputs' directory to process them automatically.")
    sys.exit(0)

def main():
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
    
    print("[INFO] Starting DeOldify Full Restore Pipeline...")
    
    # Display device mode
    if GPU_MODE:
        print("[INFO] Running in GPU mode - processing will be much faster!")
    else:
        print("[INFO] Running in CPU mode - processing will be slow")
    
    # Parse enhancement flag
    do_enhance = '--do-enhance' in sys.argv
    
    # Create the watchdog with GPU optimization if available
    watchdog = VideoWatchdog(
        inputs_dir='inputs',
        outputs_dir='outputs',
        processed_dir='processed',
        temp_dir='temp_video_frames',
        poll_interval=10,  # Check every 10 seconds for better responsiveness
        do_enhance=do_enhance
    )
    
    # Run the watchdog
    print("[INFO] Watchdog started - monitoring 'inputs' directory for videos...")
    watchdog.run()

if __name__ == '__main__':
    main()
