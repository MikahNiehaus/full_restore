import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm
import traceback

# Import our PyTorch patch first to fix loading issues
from torch_safety_patch import *

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))

# Try to import DeOldify - this will verify it's working
try:
    from deoldify.visualize import get_video_colorizer
    print("[INFO] Successfully imported DeOldify")
except Exception as e:
    print(f"[ERROR] Failed to import DeOldify: {e}")
    traceback.print_exc()
    sys.exit(1)

class VideoWatchdog:
    """
    Simple video watchdog that monitors a directory for new videos and processes them with DeOldify.
    """
    def __init__(self, inputs_dir='inputs', outputs_dir='outputs', processed_dir='processed', 
                 temp_dir='temp_video_frames', poll_interval=60):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = Path(processed_dir)
        self.temp_dir = Path(temp_dir)
        self.poll_interval = poll_interval
        self.supported_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        self.ensure_dirs()
        
    def ensure_dirs(self):
        """Ensure all required directories exist"""
        for dir_path in [self.inputs_dir, self.outputs_dir, self.processed_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
            print(f"[INFO] Directory exists or created: {dir_path}")

    def find_next_video(self) -> Optional[Path]:
        """Find the first video file in the inputs directory"""
        for f in self.inputs_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.supported_exts:
                return f
        return None
    
    def extract_audio(self, video_path, audio_path):
        """Extract audio from a video file"""
        cmd = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}" -hide_banner -loglevel error'
        os.system(cmd)
        if audio_path.exists():
            print(f"[INFO] Audio extracted to {audio_path}")
        else:
            print("[WARNING] No audio track found or extraction failed")
    
    def process_video(self, video_path: Path):
        """Process a single video file with DeOldify"""
        print(f"[INFO] Processing video: {video_path.name}")
        
        # Extract audio if available
        temp_audio_path = Path('temp_audio.aac')
        self.extract_audio(video_path, temp_audio_path)
          # Set up DeOldify colorizer
        print("[INFO] Setting up DeOldify colorizer...")
        try:
            # Force PyTorch to use weights_only=False for model loading
            import torch.serialization
            with torch.serialization.safe_globals([slice]):
                colorizer = get_video_colorizer(render_factor=21)  # Default render factor
            
            # Process video directly with DeOldify
            print("[INFO] Colorizing video with DeOldify...")
            result_path = colorizer.colorize_from_file_name(
                str(video_path),
                render_factor=21,
                watermarked=False
            )
            
            # Move result to outputs folder
            if result_path and Path(result_path).exists():
                output_path = self.outputs_dir / Path(result_path).name
                shutil.copy(result_path, output_path)
                print(f"[INFO] Copied colorized video to: {output_path}")
            else:
                print(f"[WARNING] DeOldify didn't return a valid result path")
            
            # Move original video to processed folder
            processed_path = self.processed_dir / video_path.name
            shutil.move(str(video_path), str(processed_path))
            print(f"[INFO] Moved original video to: {processed_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to colorize video: {e}")
            traceback.print_exc()
        
        print(f"[INFO] Finished processing: {video_path.name}")
    
    def run(self):
        """Run the watchdog loop to monitor for videos"""
        print("[INFO] Starting video watchdog...")
        print(f"[INFO] Monitoring directory: {self.inputs_dir}")
        print(f"[INFO] Output directory: {self.outputs_dir}")
        print(f"[INFO] Poll interval: {self.poll_interval} seconds")
        
        while True:
            try:
                # Check for videos
                video = self.find_next_video()
                if video:
                    print(f"[INFO] Found video: {video.name}")
                    self.process_video(video)
                else:
                    print("[INFO] No videos found. Waiting...")
                
                # Wait before next check
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                print("\n[INFO] Watchdog stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] Watchdog error: {e}")
                traceback.print_exc()
                time.sleep(self.poll_interval)  # Wait before retrying

if __name__ == '__main__':
    VideoWatchdog().run()
