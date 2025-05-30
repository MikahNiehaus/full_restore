# Import our PyTorch patch FIRST before any torch imports
from torch_safety_patch import *

# Import the simplified watchdog that uses DeOldify directly
from simple_watchdog import VideoWatchdog, YOUTUBE_UPLOADER_AVAILABLE

def main():
    print("[INFO] Starting DeOldify Video Watchdog...")
      # Create watchdog with 10 second polling interval
    watchdog = VideoWatchdog(
        inputs_dir='inputs',
        outputs_dir='outputs',
        processed_dir='processed',
        temp_dir='temp_video_frames',
        poll_interval=10  # Check every 10 seconds for better responsiveness
    )
    
    # Run the watchdog
    watchdog.run()

if __name__ == '__main__':
    main()
