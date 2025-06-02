import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm
import traceback
import json

# Import our PyTorch patch first to fix loading issues
from torch_safety_patch import *

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))

# Import YouTube uploader if available
try:
    from YouTubeApi.youtube_uploader import YouTubeUploader
    YOUTUBE_UPLOADER_AVAILABLE = True
    print("[INFO] YouTube upload functionality available")
except ImportError:
    YOUTUBE_UPLOADER_AVAILABLE = False
    print("[INFO] YouTube upload functionality not available - skipping video uploads")

# Import image restorer if available
try:
    from image_restorer import ImageRestorer
    IMAGE_RESTORER_AVAILABLE = True
    print("[INFO] AI image restoration module loaded and available")
except ImportError:
    IMAGE_RESTORER_AVAILABLE = False
    print("[WARNING] AI image restoration module not available - skipping restoration steps")

# Import audio enhancer if available
try:
    from audio_enhancer import AudioEnhancer, extract_audio, mux_audio_to_video
    AUDIO_ENHANCER_AVAILABLE = True
    print("[INFO] Audio enhancement module loaded and available")
except ImportError:
    AUDIO_ENHANCER_AVAILABLE = False
    print("[WARNING] Audio enhancer module not available - skipping audio enhancement")

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
                 temp_dir='temp_video_frames', poll_interval=60, do_enhance=False):
        self.inputs_dir = Path(inputs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.uploaded_dir = Path(outputs_dir) / 'uploaded'
        self.failed_upload_dir = Path(outputs_dir) / 'failed_upload'
        self.processed_dir = Path(processed_dir)
        self.temp_dir = Path(temp_dir)
        self.poll_interval = poll_interval
        self.do_enhance = do_enhance
        self.supported_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        self.ensure_dirs()
        
    def ensure_dirs(self):
        """Ensure all required directories exist"""
        for dir_path in [self.inputs_dir, self.outputs_dir, self.processed_dir, self.temp_dir,
                        self.uploaded_dir, self.failed_upload_dir]:
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

    def move_to_uploaded(self, video_path: Path):
        """Move video to the uploaded directory"""
        if not video_path.exists():
            print(f"[WARNING] Cannot move non-existent video to uploaded folder: {video_path}")
            return
        
        # Create destination path
        dest_path = self.uploaded_dir / video_path.name
        
        try:
            # Handle case if file already exists in destination
            if dest_path.exists():
                print(f"[WARNING] File already exists in uploaded folder: {dest_path}")
                # Add timestamp to avoid overwriting
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = self.uploaded_dir / f"{video_path.stem}_{timestamp}{video_path.suffix}"
            
            # Move the file
            shutil.move(str(video_path), str(dest_path))
            print(f"[INFO] Moved successfully uploaded video to: {dest_path}")
        except Exception as e:
            print(f"[ERROR] Failed to move video to uploaded folder: {e}")
            
    def move_to_failed_upload(self, video_path: Path):
        """Move video to the failed_upload directory"""
        if not video_path.exists():
            print(f"[WARNING] Cannot move non-existent video to failed_upload folder: {video_path}")
            return
        
        # Create destination path
        dest_path = self.failed_upload_dir / video_path.name
        
        try:
            # Handle case if file already exists in destination
            if dest_path.exists():
                print(f"[WARNING] File already exists in failed_upload folder: {dest_path}")
                # Add timestamp to avoid overwriting
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = self.failed_upload_dir / f"{video_path.stem}_{timestamp}{video_path.suffix}"
            
            # Move the file
            shutil.move(str(video_path), str(dest_path))
            print(f"[INFO] Moved failed upload video to: {dest_path}")
        except Exception as e:
            print(f"[ERROR] Failed to move video to failed_upload folder: {e}")
    
    def reassemble_video(self, frames_dir: Path, audio_path: Path, output_path: Path, fps: float = 24.0):
        """Reassemble video from frames and add audio if available"""
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
            if AUDIO_ENHANCER_AVAILABLE:
                mux_audio_to_video(str(temp_video_path), str(audio_path), str(final_path))
            else:
                cmd = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -b:a 320k -strict experimental "{final_path}" -hide_banner -loglevel error'
                os.system(cmd)
            
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        else:
            # Just rename the video if no audio
            shutil.move(str(temp_video_path), str(output_path))
            
        print(f"[INFO] Video reassembled and saved to {output_path}")
    def clean_temp_directories(self):
        """Clean up all temporary directories and recreate them"""
        temp_dirs = [
            Path('temp_video_frames'),
            Path('temp_restored_frames'),
            Path('temp_enhanced_frames'),
            Path('temp_colorized_frames')
        ]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                print(f"[INFO] Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(exist_ok=True)
            print(f"[INFO] Recreated temporary directory: {temp_dir}")

    def log_json(self, video_name, stage, status, message):
        """Log results to YouTubeApi/logs.json with more detail"""
        log_path = Path(__file__).parent / "YouTubeApi" / "logs.json"
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    try:
                        logs = json.load(f)
                    except Exception:
                        logs = {}
            else:
                logs = {}
            if video_name not in logs:
                logs[video_name] = {}
            logs[video_name][stage] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "message": message
            }
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to log to logs.json: {e}")

    def process_video(self, video_path: Path):
        """Process a single video file with DeOldify"""
        print(f"[INFO] Processing video: {video_path.name}")

        # Clean temporary directories before processing
        self.clean_temp_directories()

        output_path = None
        temp_frames_dir = Path('temp_video_frames')
        temp_restored_dir = Path('temp_restored_frames')
        temp_enhanced_dir = Path('temp_enhanced_frames')
        temp_colorized_dir = Path('temp_colorized_frames')

        # Ensure temp directories exist
        for temp_dir in [temp_frames_dir, temp_restored_dir, temp_enhanced_dir, temp_colorized_dir]:
            temp_dir.mkdir(exist_ok=True)
            
        # Extract frames if image restoration is available
        frames_extracted = False
        
        # Extract audio for enhancement if available
        temp_audio_path = Path('temp_audio.wav')  # Use .wav for better quality
        enhanced_audio_path = None
        
        # AUDIO
        audio_status = "SUCCESS"
        audio_message = ""
        # Extract and enhance audio if available
        if AUDIO_ENHANCER_AVAILABLE:
            print("[INFO] Extracting and enhancing audio...")
            audio_extracted = extract_audio(str(video_path), str(temp_audio_path))
            
            if audio_extracted and temp_audio_path.exists():
                try:
                    enhanced_audio_path = Path('temp_audio_enhanced.wav')
                    enhancer = AudioEnhancer(verbose=True)
                    audio_enhanced = enhancer.enhance_audio(str(temp_audio_path), str(enhanced_audio_path))
                    
                    if not audio_enhanced or not enhanced_audio_path.exists():
                        enhanced_audio_path = temp_audio_path
                        audio_status = "WARNING"
                        audio_message = "Audio enhancement failed, using original audio"
                        print("[WARNING] Audio enhancement failed, using original audio")
                except Exception as e:
                    audio_status = "ERROR"
                    audio_message = f"Audio enhancement error: {e}"
                    print(f"[WARNING] Audio enhancement error: {e}")
                    enhanced_audio_path = temp_audio_path
            else:
                audio_status = "WARNING"
                audio_message = "No audio track found or extraction failed"
                print("[WARNING] No audio track found or extraction failed")
        else:
            # Basic audio extraction
            self.extract_audio(video_path, temp_audio_path)
            audio_status = "INFO"
            audio_message = "Audio enhancer not available, used basic extraction"
        
        self.log_json(video_path.name, "audio", audio_status, audio_message)

        # RESTORE & COLORIZE
        restore_status = "SUCCESS"
        restore_message = ""
        enhance_status = "SUCCESS"
        enhance_message = ""
        colorize_status = "SUCCESS"
        colorize_message = ""
        if IMAGE_RESTORER_AVAILABLE:
            try:
                print("[INFO] Extracting frames for AI restoration...")
                # Extract frames
                video = cv2.VideoCapture(str(video_path))
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)
                
                # Clear any existing frames
                for file in temp_frames_dir.glob("*.*"):
                    file.unlink()
                
                # Extract frames
                success = True
                frame_count = 0
                with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                    while success:
                        success, image = video.read()
                        if success:
                            output_file = temp_frames_dir / f"frame_{frame_count:05d}.png"
                            cv2.imwrite(str(output_file), image)
                            frame_count += 1
                            pbar.update(1)
                
                video.release()
                frames_extracted = True
                
                # Restore frames
                print("[INFO] Applying AI image restoration...")
                restorer = ImageRestorer()
                try:
                    restorer.restore_frames(temp_frames_dir, temp_restored_dir)
                except Exception as e:
                    restore_status = "ERROR"
                    restore_message = f"Image restoration error: {e}"
                    print(f"[ERROR] Error during image restoration: {e}")

                # Enhance frames (Real-ESRGAN) if enabled
                if self.do_enhance:
                    print("[INFO] Enhancing restored frames with Real-ESRGAN (super-resolution)...")
                    try:
                        restorer.enhance_frames(temp_restored_dir, temp_enhanced_dir)
                        frames_for_color = temp_enhanced_dir
                    except Exception as e:
                        enhance_status = "ERROR"
                        enhance_message = f"Enhancement error: {e}"
                        print(f"[ERROR] Error during enhancement: {e}")
                        frames_for_color = temp_restored_dir
                else:
                    print("[INFO] Skipping Real-ESRGAN enhancement (super-resolution) as requested.")
                    frames_for_color = temp_restored_dir

                # Set up DeOldify colorizer for processed frames
                print("[INFO] Setting up DeOldify colorizer...")
                colorizer = get_video_colorizer(render_factor=40)  # Maximum quality

                # Colorize frames (enhanced or just restored)
                print("[INFO] Colorizing frames with DeOldify...")
                frame_files = sorted(list(frames_for_color.glob('*.png')))
                for i, frame_path in enumerate(tqdm(frame_files, desc="Colorizing frames")):
                    vis = colorizer.vis
                    try:
                        colorized_frame = vis.get_transformed_image(
                            str(frame_path), 
                            render_factor=40,  # Maximum quality
                            watermarked=False,
                            post_process=True
                        )
                        colorized_frame_path = temp_colorized_dir / f"frame_{i:05d}.png"
                        colorized_frame.save(str(colorized_frame_path))
                    except Exception as e:
                        colorize_status = "ERROR"
                        colorize_message = f"Colorization error: {e}"
                        print(f"[ERROR] Error during colorization: {e}")
                output_path = self.outputs_dir / f"{video_path.stem}_restored_enhanced_colorized.mp4"
                print("[INFO] Reassembling video with enhanced frames...")
                self.reassemble_video(temp_colorized_dir, enhanced_audio_path or temp_audio_path, output_path, fps)
            except Exception as e:
                restore_status = "ERROR"
                restore_message = f"Error during image restoration process: {e}"
                print(f"[ERROR] Error during image restoration process: {e}")
                traceback.print_exc()
                frames_extracted = False
        
        self.log_json(video_path.name, "restore", restore_status, restore_message)
        self.log_json(video_path.name, "enhance", enhance_status, enhance_message)
        self.log_json(video_path.name, "colorize", colorize_status, colorize_message)

        # If image restoration failed or is not available, use standard DeOldify directly
        if not frames_extracted:
            print("[INFO] Using standard DeOldify colorization...")
            # Force PyTorch to use weights_only=False for model loading
            import torch.serialization
            with torch.serialization.safe_globals([slice]):
                colorizer = get_video_colorizer(render_factor=40)  # Maximum quality
            
            # Process video directly with DeOldify
            print("[INFO] Colorizing video with DeOldify...")
            result_path = colorizer.colorize_from_file_name(
                str(video_path),
                render_factor=40,  # Maximum quality for best results
                watermarked=False
            )
            
            # Move result to outputs folder
            if result_path and Path(result_path).exists():
                output_path = self.outputs_dir / Path(result_path).name
                shutil.copy(result_path, output_path)
                print(f"[INFO] Copied colorized video to: {output_path}")
                
                # Upload to YouTube if available
                if YOUTUBE_UPLOADER_AVAILABLE and output_path.exists():
                    self.upload_to_youtube(video_path, output_path)
            else:
                print(f"[WARNING] DeOldify didn't return a valid result path")
            
        # Always upload to YouTube if enabled and output_path exists
        if YOUTUBE_UPLOADER_AVAILABLE and output_path and Path(output_path).exists():
            self.upload_to_youtube(video_path, output_path)

        # Move original video to processed folder (always, after processing)
        if video_path.exists():
            processed_path = self.processed_dir / video_path.name
            shutil.move(str(video_path), str(processed_path))
            print(f"[INFO] Moved original video to: {processed_path}")
        print(f"[INFO] Finished processing: {video_path.name}")

    def get_client_secret_file(self):
        """Find the first client_secret_*.json file in the YouTubeApi directory."""
        api_dir = Path(__file__).parent / "YouTubeApi"
        for f in api_dir.glob("client_secret_*.json"):
            return f
        return None

    def upload_to_youtube(self, original_video_path: Path, colorized_video_path: Path):
        """Upload processed video to YouTube, retrying every hour if quota exceeded."""
        import time
        try:
            print("[INFO] Uploading video to YouTube...")
            client_secret = self.get_client_secret_file()
            if not client_secret or not client_secret.exists():
                print(f"[WARNING] YouTube client secret file not found in {client_secret}")
                self.move_to_failed_upload(colorized_video_path)
                return
            uploader = YouTubeUploader(client_secret)
            # Sanitize video title for YouTube (remove problematic characters)
            import re
            def sanitize_title(title):
                # Replace curly quotes and dashes with ASCII equivalents
                title = title.replace('‘', "'").replace('’', "'")
                title = title.replace('“', '"').replace('”', '"')
                title = title.replace('–', '-').replace('—', '-')
                # Remove any other non-ASCII characters
                title = re.sub(r'[^\x00-\x7F]+', '', title)
                # Collapse whitespace
                title = re.sub(r'\s+', ' ', title).strip()
                return title

            video_title = sanitize_title(f"{original_video_path.stem} colorized enhanced restored")
            video_description = """I created this video using AI to restore and enhance a historical recording. My goal is to make these moments from the past feel more real and accessible so we never forget the people and stories they hold.

This project is still a work in progress, and I'm sharing everything I'm learning on GitHub so others can get involved or do their own restorations.

Check out the repository here to do it yourself:
github.com/MikahNiehaus/full_restore

Check out my LinkedIn to learn more about me:
linkedin.com/in/mikahniehaus/

Thanks for watching and let's keep history alive together.
"""
            while True:
                video_id, error_message = uploader.upload_video(
                    str(colorized_video_path),
                    title=video_title,
                    description=video_description,
                    privacy_status="public"  # Make video public by default
                )
                if video_id:
                    print(f"[INFO] Successfully uploaded to YouTube: https://youtu.be/{video_id}")
                    self.move_to_uploaded(colorized_video_path)
                    break
                elif error_message and 'exceeded the number of videos' in error_message.lower():
                    print("[WARNING] YouTube upload quota exceeded. Waiting 1 hour before retrying...")
                    self.log_json(colorized_video_path.name, "upload", "QUOTA_EXCEEDED", error_message)
                    time.sleep(3600)
                else:
                    print(f"[WARNING] YouTube upload failed for another reason: {error_message}")
                    self.log_json(colorized_video_path.name, "upload", "ERROR", error_message or "Unknown error")
                    self.move_to_failed_upload(colorized_video_path)
                    break
        except Exception as e:
            print(f"[ERROR] YouTube upload error: {e}")
            traceback.print_exc()
            self.move_to_failed_upload(colorized_video_path)

    def run(self):
        """Run the watchdog loop to monitor for videos"""
        print("[INFO] Starting video watchdog...")
        print(f"[INFO] Monitoring directory: {self.inputs_dir}")
        print(f"[INFO] Output directory: {self.outputs_dir}")
        print(f"[INFO] Poll interval: {self.poll_interval} seconds")
        print(f"[INFO] AI image restoration: {'Enabled' if IMAGE_RESTORER_AVAILABLE else 'Disabled'}")
        print(f"[INFO] Real-ESRGAN enhancement: {'Enabled' if self.do_enhance else 'Disabled'}")
        print(f"[INFO] Audio enhancement: {'Enabled' if AUDIO_ENHANCER_AVAILABLE else 'Disabled'}")
        print(f"[INFO] YouTube upload: {'Enabled' if YOUTUBE_UPLOADER_AVAILABLE else 'Disabled'}")
        
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
