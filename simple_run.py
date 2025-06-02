import sys
import os

# Handle CPU flag right at the beginning
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print('[INFO] Forcing CPU mode (CUDA disabled)')

# Import our PyTorch patch FIRST before any torch imports
from torch_safety_patch import *

# Check GPU availability and show detailed information
import torch
def check_gpu_availability():
    """Check and log GPU availability information"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
        
        print(f"[INFO] CUDA is available! Using GPU acceleration")
        print(f"[INFO] PyTorch version: {torch.__version__}")
        print(f"[INFO] CUDA version: {cuda_version}")
        print(f"[INFO] GPU device: {device_name}")
        print(f"[INFO] Number of CUDA devices: {device_count}")
        
        # Enable cuDNN auto-tuning for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"[INFO] cuDNN version: {torch.backends.cudnn.version()}")
        
        return True
    else:
        print("[INFO] CUDA is not available, using CPU mode")
        print(f"[INFO] PyTorch version: {torch.__version__}")
        
        if '+cpu' in torch.__version__:
            print("[WARNING] You're using a CPU-only PyTorch build")
            print("[WARNING] For GPU acceleration, reinstall PyTorch with CUDA:")
            print("[WARNING] pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        return False

# Check GPU availability on startup
using_gpu = check_gpu_availability() and '--cpu' not in sys.argv

# Import the simplified watchdog that uses DeOldify directly
from simple_watchdog import VideoWatchdog, YOUTUBE_UPLOADER_AVAILABLE

def show_help():
    """Display help information for the script"""
    print("\nDeOldify Full Restore Pipeline")
    print("=============================")
    print("Options:")
    print("  --help     Show this help message and exit")
    print("  --cpu      Force CPU mode even if CUDA/GPU is available")
    print("  --do-enhance   Enable Real-ESRGAN enhancement (default: off)")
    print("\nUsage:")
    print("  python simple_run.py         # Run with GPU acceleration if available")
    print("  python simple_run.py --cpu   # Force CPU mode")
    print("\nPlace video files in the 'inputs' directory to process them automatically.")
    sys.exit(0)

def main():
    # Check for help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
    
    print("[INFO] Starting DeOldify Full Restore Pipeline...")
    
    # Display device mode
    if using_gpu:
        print("[INFO] Running in GPU mode - colorization will be much faster!")
    else:
        print("[INFO] Running in CPU mode - colorization may be slow")
    
    # Parse enhancement flag
    do_enhance = '--do-enhance' in sys.argv

    # Create watchdog with enhancement flag
    watchdog = VideoWatchdog(
        inputs_dir='inputs',
        outputs_dir='outputs',
        processed_dir='processed',
        temp_dir='temp_video_frames',
        poll_interval=10,  # Check every 10 seconds for better responsiveness
        do_enhance=do_enhance
    )
    
    # Run the watchdog
    watchdog.run()

if __name__ == '__main__':
    main()
