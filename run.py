import time
import os
import sys
import traceback
from pathlib import Path
from video_watchdog import VideoWatchdog

def main():
    print("[INFO] Starting DeOldify Video Watchdog...")
    
    # Add DeOldify to Python path
    deoldify_path = Path(__file__).parent / 'DeOldify'
    if deoldify_path.exists():
        print(f"[INFO] Adding DeOldify path to sys.path: {deoldify_path}")
        sys.path.append(str(deoldify_path))
    else:
        print(f"[ERROR] DeOldify directory not found at {deoldify_path}")
    
    # Print some diagnostics
    inputs_dir = Path('inputs')
    outputs_dir = Path('outputs')
    processed_dir = Path('processed')
    temp_dir = Path('temp_video_frames')
    
    # Create directories if they don't exist
    print("[INFO] Checking required directories...")
    for dir_path in [inputs_dir, outputs_dir, processed_dir, temp_dir]:
        if not dir_path.exists():
            print(f"[INFO] Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[INFO] Directory exists: {dir_path}")
    
    # Check for video files in inputs
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    videos = [f for f in inputs_dir.glob('*') if f.is_file() and f.suffix.lower() in video_exts]
    
    if not videos:
        print(f"[WARNING] No video files found in '{inputs_dir}'. Please add videos to process.")
        print(f"[INFO] Supported formats: {', '.join(video_exts)}")
    else:
        print(f"[INFO] Found {len(videos)} videos in inputs directory:")
        for video in videos:
            print(f"       - {video.name}")
    
    # Import check for DeOldify
    try:
        print("[INFO] Testing DeOldify import...")
        from deoldify.visualize import get_video_colorizer
        print("[INFO] DeOldify import successful.")
    except ImportError as e:
        print(f"[ERROR] Failed to import DeOldify: {e}")
        print("[ERROR] Make sure the DeOldify folder is in the same directory as run.py")
        return
    
    # Start watchdog with 10 second polling for better responsiveness
    print("[INFO] Starting watchdog to monitor for new videos...")
    watchdog = VideoWatchdog(
        inputs_dir=str(inputs_dir),
        outputs_dir=str(outputs_dir),
        processed_dir=str(processed_dir),
        temp_dir=str(temp_dir),
        poll_interval=10  # Check every 10 seconds instead of 60 for better responsiveness
    )
    
    try:
        print("[INFO] Entering watchdog main loop. Press Ctrl+C to stop.")
        watchdog.run()
    except KeyboardInterrupt:
        print("\n[INFO] Watchdog stopped by user.")
    except Exception as e:
        print(f"[ERROR] Watchdog error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
