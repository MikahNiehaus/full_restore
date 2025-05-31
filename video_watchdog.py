import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm

# Import our PyTorch patch first to fix loading issues
from torch_safety_patch import *

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))
from deoldify.visualize import get_video_colorizer

class VideoWatchdog:
    def __init__(self, inputs_dir='inputs', outputs_dir='outputs', processed_dir='processed', temp_dir='temp_video_frames', poll_interval=60):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = Path(processed_dir)
        self.temp_dir = Path(temp_dir)
        self.poll_interval = poll_interval
        self.supported_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        self.ensure_dirs()

    def ensure_dirs(self):
        self.inputs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)    
    def find_next_video(self) -> Optional[Path]:
         for f in self.inputs_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.supported_exts:
                return f
            return None
    def process_video(self, video_path: Path):
        print(f"[INFO] Processing video: {video_path.name}")
        
        # Set up temp directories
        temp_frames_dir = self.temp_dir
        temp_colorized_dir = Path('temp_colorized_frames')
        temp_colorized_dir.mkdir(exist_ok=True)
        temp_audio_path = Path('temp_audio.aac')
        
        # Extract frames
        print("[INFO] Extracting frames...")
        self.extract_frames(video_path, temp_frames_dir)
        
        # Extract audio if possible
        print("[INFO] Extracting audio...")
        self.extract_audio(video_path, temp_audio_path)
        
        # Setup colorizer
        print("[INFO] Setting up DeOldify colorizer...")
        colorizer = get_video_colorizer(render_factor=21)  # Default quality
        
        # Process video using DeOldify
        print("[INFO] Colorizing video with DeOldify...")
    
        # Call DeOldify's colorize function
        output_path = self.outputs_dir / f"{video_path.stem}_colorized.mp4"
        
        try:
            # Process individual frames
            frame_files = sorted(list(temp_frames_dir.glob('*.png')))
            for i, frame_path in enumerate(tqdm(frame_files, desc="Colorizing frames")):
                # Get the ModelImageVisualizer from the colorizer
                vis = colorizer.vis
                # Colorize individual frame
                colorized_frame = vis.get_transformed_image(
                    frame_path, 
                    render_factor=40,  # Maximum quality
                    watermarked=False,
                    post_process=True
                )
                # Save to colorized frames directory
                colorized_frame_path = temp_colorized_dir / f"frame_{i:05d}.png"
                colorized_frame.save(colorized_frame_path)
            
            # Reassemble video
            print("[INFO] Reassembling video...")
            fps = self.get_video_fps(video_path)
            self.reassemble_video(temp_colorized_dir, temp_audio_path, output_path, fps)
            
            print(f"[INFO] Output video saved to: {output_path}")
            
            # Move original video to processed
            processed_input = self.processed_dir / video_path.name
            shutil.move(str(video_path), str(processed_input))
            print(f"[INFO] Moved input video to: {processed_input}")
            
            # Cleanup temp files
            print("[INFO] Cleaning up temp files...")
            self.cleanup_temp_files([temp_frames_dir, temp_colorized_dir, temp_audio_path])
            
        except Exception as e:
            print(f"[ERROR] Error during video processing: {e}")
            import traceback
            traceback.print_exc()
        
        print("[INFO] Processing complete.")

    def extract_frames(self, video_path: Path, output_dir: Path):
        """Extract frames from video file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any existing frames
        for file in output_dir.glob("*.*"):
            file.unlink()
        
        # Open video file
        video = cv2.VideoCapture(str(video_path))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        success = True
        frame_count = 0
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while success:
                success, image = video.read()
                if success:
                    output_file = output_dir / f"frame_{frame_count:05d}.png"
                    cv2.imwrite(str(output_file), image)
                    frame_count += 1
                    pbar.update(1)
        
        video.release()
        print(f"[INFO] Extracted {frame_count} frames to {output_dir}")
    
    def extract_audio(self, video_path: Path, audio_path: Path):
        """Extract audio from video file"""
        try:
            cmd = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}" -hide_banner -loglevel error'
            os.system(cmd)
            if audio_path.exists():
                print(f"[INFO] Audio extracted to {audio_path}")
            else:
                print("[WARNING] No audio track found or extraction failed")
        except Exception as e:
            print(f"[WARNING] Failed to extract audio: {e}")
    
    def get_video_fps(self, video_path: Path) -> float:
        """Get video FPS"""
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    
    def reassemble_video(self, frames_dir: Path, audio_path: Path, output_path: Path, fps: float = 24.0):
        """Reassemble video from frames and add audio if available"""
        # Get frame dimensions from first frame
        frame_files = sorted(list(frames_dir.glob('*.png')))
        if not frame_files:
            raise ValueError("No frames found to reassemble video")
        
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = output_path.parent / f"temp_{output_path.name}"
        video_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_file in tqdm(frame_files, desc="Assembling video"):
            frame = cv2.imread(str(frame_file))
            video_writer.write(frame)
        
        video_writer.release()
        
        # Add audio if available
        if audio_path.exists():
            print("[INFO] Adding audio to video...")
            final_path = output_path
            cmd = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{final_path}" -hide_banner -loglevel error'
            os.system(cmd)
            temp_video_path.unlink(missing_ok=True)
        else:
            # Just rename the video if no audio
            shutil.move(str(temp_video_path), str(output_path))
            
        print(f"[INFO] Video reassembled and saved to {output_path}")
    
    def cleanup_temp_files(self, paths_to_clean):
        """Clean up temporary files and directories"""
        for path in paths_to_clean:
            try:
                if isinstance(path, Path):
                    if path.is_file() and path.exists():
                        path.unlink()
                    elif path.is_dir() and path.exists():
                        shutil.rmtree(path, ignore_errors=True)
            except Exception as e:
                print(f"[WARNING] Failed to clean up {path}: {e}")
    
    def run(self):
        print("[INFO] VideoWatchdog started. Monitoring for new videos...")
        print(f"[INFO] Inputs directory: {self.inputs_dir}")
        print(f"[INFO] Outputs directory: {self.outputs_dir}")
        print(f"[INFO] Processed directory: {self.processed_dir}")
        print(f"[INFO] Checking for videos every {self.poll_interval} seconds")
        
        while True:
            print(f"[INFO] Checking for videos in {self.inputs_dir}...")
            video = self.find_next_video()
            if video:
                print(f"[INFO] Found video to process: {video}")
                self.process_video(video)
            else:
                print(f"[INFO] No new videos found in {self.inputs_dir}. Waiting {self.poll_interval} seconds...")
            time.sleep(self.poll_interval)

if __name__ == '__main__':
    VideoWatchdog().run()
