import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeOldify'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Real-ESRGAN'))

# Apply torchvision compatibility patch for Real-ESRGAN
try:
    # Run the patch before importing Real-ESRGAN
    import importlib.util
    spec = importlib.util.spec_from_file_location("realesrgan_patch", 
                                                os.path.join(os.path.dirname(__file__), "realesrgan_patch.py"))
    realesrgan_patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realesrgan_patch)
    print("[INFO] Applied Real-ESRGAN compatibility patch")
except Exception as e:
    print(f"[WARNING] Could not apply Real-ESRGAN compatibility patch: {e}")

# Force DeOldify to use CUDA if available
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    device.set(device=DeviceId.GPU0)
    print("[INFO] DeOldify set to use CUDA (GPU0)")
except Exception as e:
    print(f"[WARNING] Could not set DeOldify device to CUDA: {e}")

import cv2
from tqdm import tqdm
from deoldify.visualize import get_image_colorizer
# Import restore_old_photo function at runtime to avoid import errors
# We'll import it when needed within the restore_frames function
from PIL import Image
import subprocess
import numpy as np
import argparse

INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'
TEMP_FRAMES_DIR = 'temp_video_frames'
TEMP_COLOR_DIR = 'temp_colorized_frames'
TEMP_RESTORED_DIR = 'temp_restored_frames'
TEMP_ENHANCED_DIR = 'temp_enhanced_frames'

# Find the first video file in the inputs directory
def find_video_file():
    for fname in os.listdir(INPUTS_DIR):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return os.path.join(INPUTS_DIR, fname)
    return None

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    frame_paths = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frame_path = os.path.join(frames_dir, f"frame_{count:06d}.png")
        cv2.imwrite(frame_path, image)
        frame_paths.append(frame_path)
        count += 1
    vidcap.release()
    return frame_paths

def colorize_frames(frame_paths, color_dir, color_boost=1.0):
    # Use our unified image processor to colorize frames
    os.makedirs(color_dir, exist_ok=True)
    colorized_paths = []
    
    # Import the unified image processor here to avoid circular imports
    from unified_image_processor import ImageProcessor
    
    try:
        # Initialize the processor
        processor = ImageProcessor(color_dir)
        
        # Use the processor to colorize all frames
        colorized_paths = processor.colorize_frames(frame_paths, color_dir)
        
        if not colorized_paths:
            raise Exception("No frames were colorized")
            
    except Exception as e:
        print(f"[ERROR] Unified image colorization failed: {e}")
        print("[WARNING] Falling back to direct colorization")
        
        try:
            # Traditional colorization as fallback
            colorizer = get_image_colorizer(artistic=True)
            for frame_path in tqdm(frame_paths, desc='Colorizing frames (fallback)'):
                out_path = os.path.join(color_dir, os.path.basename(frame_path))
                # Colorize the frame
                colorizer.plot_transformed_image(
                    frame_path,
                    render_factor=35,
                    watermarked=False,
                    post_process=True,
                    results_dir=color_dir,
                    force_cpu=True  # Force CPU to avoid potential CUDA errors
                )
                
                # --- Add more color (increase saturation) and brightness ---
                try:
                    from PIL import Image, ImageEnhance
                    img = Image.open(out_path)
                    # Color enhancement
                    enhancer = ImageEnhance.Color(img)
                    img_colored = enhancer.enhance(color_boost)  # Increase saturation by color_boost%
                    
                    # Brightness enhancement
                    brightness = ImageEnhance.Brightness(img_colored)
                    img_brightened = brightness.enhance(1.2)  # Increase brightness by 20%
                    
                    img_brightened.save(out_path)
                except Exception as e:
                    print(f"[WARNING] Could not enhance colors for {out_path}: {e}")
                colorized_paths.append(out_path)
                
        except Exception as e:
            print(f"[ERROR] Fallback colorization failed: {e}")
            print("[WARNING] Using original frames without colorization")
            
            # If all colorization methods fail, use original frames
            for frame_path in frame_paths:
                out_path = os.path.join(color_dir, os.path.basename(frame_path))
                import shutil
                shutil.copy(frame_path, out_path)
                colorized_paths.append(out_path)
                
    return colorized_paths

def restore_frames(colorized_paths, restored_dir):
    os.makedirs(restored_dir, exist_ok=True)
    restored_paths = []
    
    # Use our unified image processor for restoration
    from unified_image_processor import ImageProcessor
    
    try:
        # Initialize the processor
        processor = ImageProcessor(restored_dir)
        
        # Process each colorized frame
        for color_path in tqdm(colorized_paths, desc='Restoring frames'):
            try:
                base_name = os.path.basename(color_path)
                file_name_wo_ext = os.path.splitext(base_name)[0]
                restored_path = os.path.join(restored_dir, f"{file_name_wo_ext}_restored.png")
                
                # Enhance the colorized image (we're using enhancement as restoration)
                success = processor.enhance_image(
                    color_path,
                    restored_path,
                    scale=1,  # Don't upscale again, just enhance
                    brighten_factor=1.3  # Brighten the image to fix the "too dark" issue
                )
                
                if success and os.path.exists(restored_path):
                    print(f"Successfully restored: {color_path} -> {restored_path}")
                    restored_paths.append(restored_path)
                else:
                    # If restoration fails, just copy the colorized file
                    import shutil
                    shutil.copy(color_path, restored_path)
                    print(f"Restoration failed, using colorized image: {restored_path}")
                    restored_paths.append(restored_path)
                    
            except Exception as e:
                print(f"Error restoring frame {color_path}: {e}")
                # Create fallback path
                base_name = os.path.basename(color_path)
                file_name_wo_ext = os.path.splitext(base_name)[0]
                fallback_path = os.path.join(restored_dir, f"{file_name_wo_ext}_restored.png")
                
                # Copy the colorized file as the restored version
                import shutil
                shutil.copy(color_path, fallback_path)
                print(f"Exception fallback: copied {color_path} -> {fallback_path}")
                restored_paths.append(fallback_path)
    
    except Exception as e:
        print(f"ERROR in restoration process: {e}")
        # Fallback to simple copy
        for color_path in tqdm(colorized_paths, desc='Simple frame restoration (fallback)'):
            base_name = os.path.basename(color_path)
            file_name_wo_ext = os.path.splitext(base_name)[0]
            fallback_path = os.path.join(restored_dir, f"{file_name_wo_ext}_restored.png")
            
            # Just copy the colorized file as the restored version
            import shutil
            shutil.copy(color_path, fallback_path)
            print(f"Simple restore: copied {color_path} -> {fallback_path}")
            restored_paths.append(fallback_path)
    
    if not restored_paths:
        print("WARNING: No frames were successfully restored!")
    
    return restored_paths

def enhance_frames(frame_paths, enhanced_dir, scale=4):
    # Use our unified image processor to enhance frames
    os.makedirs(enhanced_dir, exist_ok=True)
    enhanced_paths = []
    
    # Import the unified image processor here to avoid circular imports
    from unified_image_processor import ImageProcessor
    
    try:
        print(f"[INFO] Using unified image processor with scale factor: {scale}")
        
        # Initialize the processor
        processor = ImageProcessor(enhanced_dir)
        
        # Use the processor to enhance all frames
        enhanced_paths = processor.enhance_frames(frame_paths, enhanced_dir, scale)
        
        if not enhanced_paths:
            raise Exception("No frames were enhanced")
            
    except Exception as e:
        print(f"[ERROR] Unified image enhancement failed: {e}")
        print("[WARNING] Falling back to simple OpenCV upscaling")
        
        import numpy as np
        # Fallback to simple OpenCV enhancement
        for frame_path in tqdm(frame_paths, desc='Enhancing frames (OpenCV fallback)'):
            enhanced_frame_path = os.path.join(enhanced_dir, os.path.basename(frame_path))
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"[WARNING] Could not read file: {frame_path}, skipping...")
                continue
            
            # Simple upscale
            height, width = frame.shape[:2]
            upscaled = cv2.resize(frame, (width * scale, height * scale), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            # Brighten the image - addresses the "too dark" issue
            brightness = 30  # Increase brightness by 30 (0-255)
            upscaled = cv2.convertScaleAbs(upscaled, alpha=1.2, beta=brightness)
            
            cv2.imwrite(enhanced_frame_path, upscaled)
            enhanced_paths.append(enhanced_frame_path)
    
    return enhanced_paths

def frames_to_video(frame_paths, output_path, fps=24):
    if not frame_paths:
        print("[ERROR] No frames provided to create video")
        return
        
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get frame size from the first valid frame
    frame = None
    for path in frame_paths:
        if os.path.exists(path):
            frame = cv2.imread(path)
            if frame is not None:
                break
    
    if frame is None:
        print(f"[ERROR] Could not read any valid frames from {len(frame_paths)} paths")
        print(f"First few paths: {frame_paths[:3]}")
        return
    
    # Set up video writer
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    valid_frames = 0
    for frame_path in tqdm(frame_paths, desc='Writing video'):
        if not os.path.exists(frame_path):
            print(f"[WARNING] Frame file not found: {frame_path}")
            continue
            
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARNING] Could not read frame: {frame_path}")
            continue
            
        video.write(img)
        valid_frames += 1
    
    video.release()
    
    if valid_frames == 0:
        print(f"[ERROR] No valid frames were written to {output_path}")
    else:
        print(f"[SUCCESS] Created video with {valid_frames} frames at {output_path}")

def extract_audio(video_path, audio_path):
    # Extract audio using ffmpeg
    ffmpeg_cmd = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    
    # Check if the video has an audio stream
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 
                 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    try:
        result = subprocess.run(probe_cmd, check=False, capture_output=True, text=True)
        has_audio = result.stdout.strip() == 'audio'
    except Exception:
        # In case of error, assume no audio
        has_audio = False
        
    if not has_audio:
        print(f"[WARNING] No audio stream found in {video_path}")
        # Create a silent audio file of the same duration as the video
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                       '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        try:
            result = subprocess.run(duration_cmd, check=False, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            print(f"[INFO] Creating silent audio of duration {duration} seconds")
            cmd = [
                ffmpeg_cmd, '-y', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo', 
                '-t', str(duration), '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
            ]
        except Exception:
            print("[WARNING] Could not determine video duration, using default 5 seconds")
            cmd = [
                ffmpeg_cmd, '-y', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo', 
                '-t', '5', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
            ]
    else:
        # Normal extraction of audio
        cmd = [
            ffmpeg_cmd, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
        ]
        
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg and add it to your PATH.")
        raise

def enhance_audio_with_audio_enhancer(input_wav, output_wav):
    # Use audio_enhancer.py to denoise and upsample audio
    cmd = [
        sys.executable, 'audio_enhancer.py',
        '-i', input_wav,
        '-o', output_wav
    ]
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    import subprocess
    subprocess.run(cmd, check=True)

def mux_audio_to_video(video_path, audio_path, output_path):
    # Combine enhanced audio with video using ffmpeg
    ffmpeg_cmd = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    cmd = [
        ffmpeg_cmd, '-y', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0', output_path
    ]
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg and add it to your PATH.")
        raise

def get_video_fps(video_path):
    """Get the original FPS of the video using ffprobe."""
    import subprocess
    try:
        ffprobe_cmd = 'ffprobe.exe' if os.name == 'nt' else 'ffprobe'
        cmd = [
            ffprobe_cmd, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
        if '/' in output:
            num, denom = output.split('/')
            fps = float(num) / float(denom)
        else:
            fps = float(output)
        return fps
    except Exception as e:
        print(f"[WARNING] Could not determine FPS, defaulting to 24. Error: {e}")
        return 24.0

def main():
    parser = argparse.ArgumentParser(description="Video restore and colorize pipeline")
    parser.add_argument('--no-restore', action='store_true', help='Disable restoration step')
    parser.add_argument('--no-colorize', action='store_true', help='Disable colorization step')
    parser.add_argument('--no-enhance', action='store_true', help='Disable enhancement step')
    parser.add_argument('--no-audio-improve', action='store_true', help='Disable audio improvement step')
    parser.add_argument('--color-boost', type=float, default=1.0, help='Set color boost amount for colorization (e.g., 4 for extra color, 1 for normal)')
    args = parser.parse_args()
    
    video_path = find_video_file()
    if not video_path:
        print('No video file found in inputs/')
        return
    print(f'Processing video: {video_path}')
    # 1. Extract frames
    frame_paths = extract_frames(video_path, TEMP_FRAMES_DIR)
    
    # 2. Process frames with the correct pipeline: restore -> colorize -> enhance
    from unified_image_processor import ImageProcessor
    processor = ImageProcessor(TEMP_RESTORED_DIR)
    print(f"Processing frames with pipeline: restore -> colorize -> enhance")
    progress = tqdm(total=len(frame_paths), desc='Restoring frames', unit='frame')
    processed_paths = []
    for frame_path in frame_paths:
        result = processor.process_image(
            frame_path, TEMP_RESTORED_DIR, scale=2,
            do_restore=not args.no_restore,
            do_colorize=not args.no_colorize,
            do_enhance=not args.no_enhance,
            color_boost=args.color_boost
        )
        processed_paths.append(result)
        progress.update(1)
    progress.close()
    
    # 5. Assemble video (silent) with correct FPS
    fps = get_video_fps(video_path)
    temp_video_path = os.path.join(OUTPUTS_DIR, os.path.splitext(os.path.basename(video_path))[0] + '_temp_silent.mp4')
    frames_to_video(processed_paths, temp_video_path, fps=fps)
    # 6. Extract, enhance, and mux audio
    if not args.no_audio_improve:
        temp_audio_path = os.path.join(OUTPUTS_DIR, 'temp_audio.wav')
        enhanced_audio_path = os.path.join(OUTPUTS_DIR, 'enhanced_audio.wav')
        try:
            extract_audio(video_path, temp_audio_path)
            enhance_audio_with_audio_enhancer(temp_audio_path, enhanced_audio_path)
            final_output_path = os.path.join(OUTPUTS_DIR, os.path.splitext(os.path.basename(video_path))[0] + '_enhanced_colorized_restored_audio.mp4')
            mux_audio_to_video(temp_video_path, enhanced_audio_path, final_output_path)
            print(f'Final output video with enhanced audio saved to: {final_output_path}')
        except Exception as e:
            print(f'[WARNING] Audio enhancement or muxing failed: {e}')
            print(f'Using original audio as fallback.')
            fallback_audio_path = os.path.join(OUTPUTS_DIR, 'original_audio.wav')
            extract_audio(video_path, fallback_audio_path)
            final_output_path = os.path.join(OUTPUTS_DIR, os.path.splitext(os.path.basename(video_path))[0] + '_enhanced_colorized_restored_original_audio.mp4')
            mux_audio_to_video(temp_video_path, fallback_audio_path, final_output_path)
            print(f'Final output video with original audio saved to: {final_output_path}')
    else:
        print('[INFO] Skipping audio improvement step.')
    
    # Cleanup temp dirs and files
    import shutil
    for temp_dir in [TEMP_FRAMES_DIR, TEMP_RESTORED_DIR]:  # Only cleanup frames and final processed
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    for temp_audio in ['temp_audio.wav', 'enhanced_audio.wav', 'original_audio.wav']:
        temp_audio_path = os.path.join(OUTPUTS_DIR, temp_audio)
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
    # Remove the silent temp video
    if os.path.exists(temp_video_path):
        try:
            os.remove(temp_video_path)
        except Exception:
            pass

if __name__ == '__main__':
    main()
