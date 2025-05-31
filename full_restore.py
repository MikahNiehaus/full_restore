#!/usr/bin/env python3
"""
DeOldify Full Restore Pipeline

This script provides a comprehensive pipeline for colorizing and enhancing old videos:
1. Continuously monitors an input folder for new videos
2. Processes videos one at a time (to avoid GPU memory issues)
3. Uses DeOldify for maximum quality colorization (render_factor=40)
4. Extracts, enhances, and synchronizes audio
5. Moves processed videos to output folder
6. Cleans up temporary files

Requirements:
- PyTorch
- DeOldify
- OpenCV
- FFmpeg
- SciPy
- SoundFile

Usage:
    python full_restore.py
"""

import os
import sys
import time
import shutil
import traceback
from pathlib import Path
from typing import Optional, List, Union
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
import cv2
from tqdm import tqdm
import numpy as np

# Import our PyTorch patch first to fix loading issues with PyTorch 2.6+
from torch_safety_patch import *

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))
from deoldify.visualize import get_video_colorizer

# Try to import YouTube uploader
try:
    from YouTubeApi.youtube_uploader import YouTubeUploader
    from YouTubeApi.secrets_manager import get_client_secrets_file
    YOUTUBE_UPLOADER_AVAILABLE = True
    print("[INFO] YouTube upload functionality is available")
except ImportError:
    YOUTUBE_UPLOADER_AVAILABLE = False
    print("[WARNING] YouTube upload functionality not available")

# Import audio enhancer functions
try:
    from audio_enhancer import extract_audio, AudioEnhancer, mux_audio_to_video
    AUDIO_ENHANCER_AVAILABLE = True
except ImportError:
    print("[WARNING] audio_enhancer.py not found, audio enhancement will be disabled")
    AUDIO_ENHANCER_AVAILABLE = False
    
# Import image restorer functions
try:
    from image_restorer import ImageRestorer
    IMAGE_RESTORER_AVAILABLE = True
    print("[INFO] AI image restoration module loaded and available")
except ImportError:
    IMAGE_RESTORER_AVAILABLE = False
    print("[WARNING] image_restorer.py not found, image restoration will be disabled")

class VideoProcessor:     
    def __init__(
        self, 
        inputs_dir='inputs', 
        outputs_dir='outputs', 
        processed_dir='processed', 
        temp_dir='temp_video_frames',
        render_factor=40,
        enhance_audio=True,
        restore_frames=True
    ):
        """
        Initialize the video processor
        
        Args:
            inputs_dir (str): Directory to monitor for new videos
            outputs_dir (str): Directory to save processed videos
            processed_dir (str): Directory to move original videos after processing
            temp_dir (str): Directory to store temporary files
            render_factor (int): DeOldify render factor (10-45), higher is better quality
            enhance_audio (bool): Whether to enhance audio
            restore_frames (bool): Whether to apply AI image restoration before colorization
        """
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.processed_dir = Path(processed_dir)
        self.temp_dir = Path(temp_dir)
        self.temp_colorized_dir = Path('temp_colorized_frames')
        self.temp_restored_dir = Path('temp_restored_frames')
        self.temp_audio_dir = Path('temp_audio')
        self.supported_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        self.render_factor = render_factor
        self.enhance_audio = enhance_audio and AUDIO_ENHANCER_AVAILABLE
        self.restore_frames = restore_frames and IMAGE_RESTORER_AVAILABLE
        self.ensure_dirs()    
    def ensure_dirs(self):
        """Ensure all required directories exist"""
        for dir_path in [self.inputs_dir, self.outputs_dir, self.processed_dir, 
                         self.temp_dir, self.temp_colorized_dir, self.temp_restored_dir, self.temp_audio_dir]:
            dir_path.mkdir(exist_ok=True)
            print(f"[INFO] Directory exists or created: {dir_path}")
            
    def find_next_video(self) -> Optional[Path]:
        """Find the next video file to process"""
        for f in self.inputs_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.supported_exts:
                return f
        return None
    
    def extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """
        Extract frames from video file
        
        Args:
            video_path (Path): Path to video file
            output_dir (Path): Directory to save frames
            
        Returns:
            int: Number of frames extracted
        """
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
        return frame_count
    
    def colorize_frames(self, frame_dir: Path, output_dir: Path) -> List[Path]:
        """
        Colorize video frames using DeOldify
        
        Args:
            frame_dir (Path): Directory containing frames
            output_dir (Path): Directory to save colorized frames
            
        Returns:
            List[Path]: List of colorized frame paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any existing frames
        for file in output_dir.glob("*.*"):
            file.unlink()
            
        # Setup colorizer with maximum quality
        print(f"[INFO] Setting up DeOldify colorizer with render_factor={self.render_factor}...")
        colorizer = get_video_colorizer(render_factor=self.render_factor)
        
        # Get frames and sort them
        frame_files = sorted(list(frame_dir.glob('*.png')))
        colorized_paths = []
        
        # Process individual frames
        print("[INFO] Colorizing video frames...")
        for i, frame_path in enumerate(tqdm(frame_files, desc="Colorizing frames")):
            # Get the ModelImageVisualizer from the colorizer
            vis = colorizer.vis
            # Colorize individual frame with maximum quality
            colorized_frame = vis.get_transformed_image(
                str(frame_path), 
                render_factor=self.render_factor,
                watermarked=False,
                post_process=True
            )
            # Save to colorized frames directory
            colorized_frame_path = output_dir / f"frame_{i:05d}.png"
            colorized_frame.save(str(colorized_frame_path))
            colorized_paths.append(colorized_frame_path)
            
        return colorized_paths
    
    def get_video_fps(self, video_path: Path) -> float:
        """Get video FPS"""
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    
    def reassemble_video(self, frames_dir: Path, audio_path: Union[Path, None], output_path: Path, fps: float = 24.0) -> Path:
        """
        Reassemble video from frames and add audio if available
        
        Args:
            frames_dir (Path): Directory containing frames
            audio_path (Path): Path to audio file (or None)
            output_path (Path): Path to save output video
            fps (float): Frames per second
            
        Returns:
            Path: Path to assembled video
        """
        # Get frame dimensions from first frame
        frame_files = sorted(list(frames_dir.glob('*.png')))
        if not frame_files:
            raise ValueError("No frames found to reassemble video")
        
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Create video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = output_path.parent / f"temp_{output_path.name}"
        video_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_file in tqdm(frame_files, desc="Assembling video"):
            frame = cv2.imread(str(frame_file))
            video_writer.write(frame)
        
        video_writer.release()
        
        # Add audio if available
        if audio_path and audio_path.exists():
            print("[INFO] Adding audio to video...")
            final_path = output_path
            cmd = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -b:a 320k -strict experimental "{final_path}" -hide_banner -loglevel error'
            os.system(cmd)
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        else:
            # Just rename the video if no audio
            shutil.move(str(temp_video_path), str(output_path))
            
        print(f"[INFO] Video reassembled and saved to {output_path}")
        return output_path
    
    def process_audio(self, video_path: Path) -> tuple:
        """
        Extract and enhance audio from video
        
        Args:
            video_path (Path): Path to video file
            
        Returns:
            tuple: (original_audio_path, enhanced_audio_path)
        """
        temp_audio_path = self.temp_audio_dir / f"{video_path.stem}_original.wav"
        enhanced_audio_path = self.temp_audio_dir / f"{video_path.stem}_enhanced.wav"
        
        # Extract audio using the audio_enhancer module if available
        if AUDIO_ENHANCER_AVAILABLE:
            audio_extracted = extract_audio(str(video_path), str(temp_audio_path))
        else:
            # Fallback to basic FFmpeg extraction
            cmd = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a -acodec pcm_s16le "{temp_audio_path}" -hide_banner -loglevel error'
            os.system(cmd)
            audio_extracted = temp_audio_path.exists()
            
        if not audio_extracted:
            print("[WARNING] No audio track found or extraction failed")
            return None, None
            
        # Enhance audio if enabled and available
        if self.enhance_audio and audio_extracted and temp_audio_path.exists():
            try:
                print("[INFO] Enhancing audio...")
                enhancer = AudioEnhancer(verbose=True)
                audio_enhanced = enhancer.enhance_audio(str(temp_audio_path), str(enhanced_audio_path))
                
                if audio_enhanced and enhanced_audio_path.exists():
                    print(f"[INFO] Audio enhancement successful: {enhanced_audio_path}")
                    return temp_audio_path, enhanced_audio_path
            except Exception as e:
                print(f"[WARNING] Audio enhancement failed: {e}")
                
        # Return original audio if enhancement failed or is disabled
        return temp_audio_path, None
    
    def restore_frames(self, input_dir: Path, output_dir: Path) -> List[Path]:
        """
        Apply AI image restoration to frames before colorization
        
        Args:
            input_dir (Path): Directory containing input frames
            output_dir (Path): Directory to save restored frames
            
        Returns:
            List[Path]: List of restored frame paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any existing frames
        for file in output_dir.glob("*.*"):
            file.unlink()
            
        if not self.restore_frames or not IMAGE_RESTORER_AVAILABLE:
            print("[INFO] Image restoration skipped (disabled or not available)")
            # Simply copy the original frames to the output
            restored_paths = []
            frame_files = sorted(list(input_dir.glob('*.png')))
            for i, frame_path in enumerate(frame_files):
                output_path = output_dir / frame_path.name
                shutil.copy(str(frame_path), str(output_path))
                restored_paths.append(output_path)
            return restored_paths
            
        # Initialize image restorer
        print("[INFO] Setting up AI image restorer...")
        restorer = ImageRestorer()
        
        # Process frames with restorer
        print("[INFO] Restoring video frames...")
        restored_paths = restorer.restore_frames(input_dir, output_dir)
        
        return restored_paths
    
    def process_video(self, video_path: Path):
        """
        Process a single video with DeOldify colorization and audio enhancement
        
        Args:
            video_path (Path): Path to video file
        """
        print(f"[INFO] Processing video: {video_path.name}")
        
        try:
            # Extract frames
            print("[INFO] Extracting frames...")
            self.extract_frames(video_path, self.temp_dir)
            
            # Process audio
            print("[INFO] Processing audio...")
            original_audio, enhanced_audio = self.process_audio(video_path)
            
            # Restore frames if enabled
            restored_frames = self.restore_frames(self.temp_dir, self.temp_restored_dir)
            
            # Colorize frames
            colorized_frames = self.colorize_frames(self.temp_restored_dir, self.temp_colorized_dir)
            
            # Get video FPS
            fps = self.get_video_fps(video_path)
            
            # First assemble without audio
            temp_output_path = self.outputs_dir / f"{video_path.stem}_colorized_no_audio.mp4"
            self.reassemble_video(self.temp_colorized_dir, None, temp_output_path, fps)
            
            # Final output with enhanced audio if available
            output_path = self.outputs_dir / f"{video_path.stem}_colorized.mp4"
            
            # Use enhanced audio if available, otherwise use original
            final_audio = enhanced_audio if enhanced_audio and enhanced_audio.exists() else original_audio
            
            if final_audio and final_audio.exists():
                # Use the muxing function from audio_enhancer if available
                if AUDIO_ENHANCER_AVAILABLE:
                    mux_audio_to_video(str(temp_output_path), str(final_audio), str(output_path))
                else:
                    cmd = f'ffmpeg -y -i "{temp_output_path}" -i "{final_audio}" -c:v copy -c:a aac -strict experimental "{output_path}" -hide_banner -loglevel error'
                    os.system(cmd)
                
                # Remove temporary video
                if os.path.exists(temp_output_path):
                        os.unlink(temp_output_path)
                else:
                    # Just rename the video if no audio
                    shutil.move(str(temp_output_path), str(output_path))
                
            # Move original video to processed directory
            processed_input = self.processed_dir / video_path.name
            shutil.move(str(video_path), str(processed_input))
            print(f"[INFO] Moved input video to: {processed_input}")
              # Upload to YouTube if available
            if YOUTUBE_UPLOADER_AVAILABLE:
                try:
                    print("[INFO] Uploading video to YouTube...")
                    # Generate title based on original filename
                    video_title = f"{video_path.stem} colorized enhanced restored"
                    
                    # Prepare description
                    video_description = """I created this video using AI to restore and enhance a historical recording. My goal is to make these moments from the past feel more real and accessible so we never forget the people and stories they hold.

This project is still a work in progress, and I'm sharing everything I'm learning on GitHub so others can get involved or do their own restorations. 

Check out the repository here to do it yourself: https://github.com/MikahNiehaus/full_restore

Check out my LinkedIn to learn more about me:
https://www.linkedin.com/in/mikahniehaus/

Thanks for watching and let's keep history alive together.

#AI #History #VideoRestoration #DeepLearning #AIColorization #HistoricalFootage"""
                    
                    # Get client secret file from environment variables or file
                    client_secret = get_client_secrets_file()
                    
                    # Create uploader and upload video
                    if client_secret.exists():
                        uploader = YouTubeUploader(client_secret)
                        video_id = uploader.upload_video(
                            str(output_path),
                            title=video_title,
                            description=video_description,
                            privacy_status="unlisted"  # Not publicly listed but accessible with link
                        )
                        
                        if video_id:
                            print(f"[INFO] Successfully uploaded to YouTube: https://youtu.be/{video_id}")
                        else:
                            print("[WARNING] YouTube upload failed")
                    else:
                        print(f"[WARNING] YouTube client secrets not found. Check your .env file.")
                except Exception as e:
                    print(f"[WARNING] YouTube upload error: {e}")
                    traceback.print_exc()
            
            print(f"[INFO] Processing complete. Output video: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Error during video processing: {e}")
            traceback.print_exc()
            return None
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    def cleanup_temp_files(self):
        """Clean up temporary files and directories"""
        print("[INFO] Cleaning up temporary files...")
        
        # Clean frames directories but keep the parent folders
        for file in self.temp_dir.glob("*.*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"[WARNING] Failed to delete temporary file {file}: {e}")
                
        for file in self.temp_colorized_dir.glob("*.*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"[WARNING] Failed to delete temporary file {file}: {e}")
                
        # Clean restored frames directory
        for file in self.temp_restored_dir.glob("*.*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"[WARNING] Failed to delete temporary file {file}: {e}")
                
        # Clean audio files
        for file in self.temp_audio_dir.glob("*.*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"[WARNING] Failed to delete temporary file {file}: {e}")
    
    def run(self, poll_interval=60):
        """
        Run the watchdog loop to continuously monitor for videos
        
        Args:
            poll_interval (int): Seconds between checks for new videos
        """
        print("[INFO] Full Restore Video Processor started")
        print(f"[INFO] Monitoring directory: {self.inputs_dir}")
        print(f"[INFO] Output directory: {self.outputs_dir}")
        print(f"[INFO] Processed originals directory: {self.processed_dir}")          
        print(f"[INFO] DeOldify render factor: {self.render_factor}")
        print(f"[INFO] Audio enhancement: {'Enabled' if self.enhance_audio else 'Disabled'}")
        print(f"[INFO] AI image restoration: {'Enabled' if self.restore_frames else 'Disabled'}")
        print(f"[INFO] YouTube upload: {'Enabled' if YOUTUBE_UPLOADER_AVAILABLE else 'Disabled'}")
        print(f"[INFO] Poll interval: {poll_interval} seconds")
        
        # Initial check for existing videos in input folder
        video = self.find_next_video()
        if video:
            print(f"[INFO] Found existing video to process: {video}")
            self.process_video(video)
        
        # Start monitoring loop
        while True:
            try:
                print(f"[INFO] Checking for videos in {self.inputs_dir}...")
                video = self.find_next_video()
                if video:
                    print(f"[INFO] Found video to process: {video}")
                    self.process_video(video)
                else:
                    print(f"[INFO] No new videos found. Waiting {poll_interval} seconds...")
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                print("\n[INFO] Process stopped by user.")
                break
            except Exception as e:
                print(f"[ERROR] Watchdog error: {e}")
                traceback.print_exc()
                # Wait before retrying
                time.sleep(poll_interval)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeOldify Full Restore - Video Colorization and Enhancement Pipeline")
    parser.add_argument("--input-dir", "-i", default="inputs", help="Input directory to monitor for videos")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory for processed videos")
    parser.add_argument("--processed-dir", "-p", default="processed", help="Directory to move original videos after processing")
    parser.add_argument("--render-factor", "-r", type=int, default=40, help="DeOldify render factor (10-45), higher is better quality")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between checks for new videos (how often to look for new videos)")
    parser.add_argument("--no-audio-enhance", action="store_true", help="Disable audio enhancement")
    parser.add_argument("--no-restore-frames", action="store_true", help="Disable AI image restoration")
    
    args = parser.parse_args()
    
    # Validate render factor
    if args.render_factor < 10 or args.render_factor > 45:
        print(f"[WARNING] Invalid render factor {args.render_factor}. Using default of 40.")
        args.render_factor = 40
        
    # Create and run the processor - always enable audio enhancement and image restoration by default
    processor = VideoProcessor(
        inputs_dir=args.input_dir,
        outputs_dir=args.output_dir,
        processed_dir=args.processed_dir,
        render_factor=args.render_factor,
        enhance_audio=True,  # Always enabled by default
        restore_frames=True  # Always enabled by default
    )
    
    processor.run(poll_interval=args.poll_interval)

if __name__ == "__main__":
    main()
