#!/usr/bin/env python3
"""
Clean video colorization pipeline that uses the clean DeOldify implementation.
This version never produces orange/sepia fallback images. If colorization fails, it errors out.
Features:
- Maximum quality colorization using highest render factor
- Auto-move processed videos to 'processed' directory
- Continuous processing mode to monitor for new videos
- Clean temporary directories between processing
"""
import os
import sys
import cv2
import time
import shutil
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path

# Add DeOldify to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeOldify'))

# Set model directory to use 'models' folder instead of 'My PTH'
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
os.environ['DEOLDIFY_MODELS'] = models_dir
print(f"Setting DEOLDIFY_MODELS environment variable to: {models_dir}")    # Import our clean colorization implementation
sys.path.append(os.path.join(os.path.dirname(
    __file__), 'simple_colorize_clean'))
try:
    from simple_colorize import colorize_image
except ImportError:
    from simple_colorize_clean.simple_colorize import colorize_image

# Print status of model files
def check_model_files():
    models_dir = os.environ.get('DEOLDIFY_MODELS', os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
    stable_path = os.path.join(models_dir, 'ColorizeStable_gen.pth')
    artistic_path = os.path.join(models_dir, 'ColorizeArtistic_gen.pth')
    
    print(f"Checking DeOldify model files in: {models_dir}")
    if os.path.exists(stable_path):
        size_mb = os.path.getsize(stable_path) / (1024 * 1024)
        print(f"✓ Stable model exists: {size_mb:.2f}MB")
    else:
        print(f"✗ Stable model missing!")
        
    if os.path.exists(artistic_path):
        size_mb = os.path.getsize(artistic_path) / (1024 * 1024)
        print(f"✓ Artistic model exists: {size_mb:.2f}MB")
    else:
        print(f"✗ Artistic model missing!")

# Check models at startup
check_model_files()

# Directory constants
INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'
PROCESSED_DIR = 'processed'
TEMP_FRAMES_DIR = 'temp_video_frames'
TEMP_COLOR_DIR = 'temp_colorized_frames'

# Max render factor for best quality
MAX_RENDER_FACTOR = 45


def find_video_file():
    """Find the first video file in the inputs directory"""
    for fname in os.listdir(INPUTS_DIR):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return os.path.join(INPUTS_DIR, fname)
    return None


def move_to_processed(video_path):
    """Move processed video to the processed directory"""
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Get the filename
    filename = os.path.basename(video_path)
    processed_path = os.path.join(PROCESSED_DIR, filename)

    # If a file with the same name exists in the processed directory, add a timestamp
    if os.path.exists(processed_path):
        name, ext = os.path.splitext(filename)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        processed_path = os.path.join(
            PROCESSED_DIR, f"{name}_{timestamp}{ext}")

    # Move the file
    print(f"Moving {video_path} to {processed_path}")
    shutil.move(video_path, processed_path)
    return processed_path


def extract_frames(video_path, frames_dir):
    """Extract frames from a video file"""
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    frame_paths = []

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            success, image = vidcap.read()
            if not success:
                break
            frame_path = os.path.join(frames_dir, f"frame_{count:06d}.png")
            cv2.imwrite(frame_path, image)
            frame_paths.append(frame_path)
            count += 1
            pbar.update(1)

    vidcap.release()
    print(f"Extracted {len(frame_paths)} frames to {frames_dir}")
    return frame_paths


def colorize_frames(frame_paths, color_dir, render_factor=MAX_RENDER_FACTOR, model='stable'):
    """
    Colorize frames using our clean DeOldify implementation.
    This will raise an exception if colorization fails.
    NOTE: Only the 'stable' model is reliably working with the current version.
    """
    # Force the stable model since we know it works
    if model != 'stable':
        print(f"WARNING: The {model} model may not work correctly. Forcing 'stable' model for reliable colorization.")
        model = 'stable'
        
    os.makedirs(color_dir, exist_ok=True)
    colorized_paths = []

    print(
        f"Starting colorization of {len(frame_paths)} frames using model: {model}")
    print(f"Using render factor {render_factor} for maximum quality")
    print(f"Output directory for colorized frames: {color_dir}")

    # Process in smaller batches to avoid overwhelming the system
    batch_size = 10
    total_frames = len(frame_paths)
    
    # Process first frame separately to catch issues early
    if total_frames > 0:
        print("Processing first frame as a test...")
        frame_path = frame_paths[0]
        base_name = os.path.basename(frame_path)
        file_name_wo_ext = os.path.splitext(base_name)[0]
        out_path = os.path.join(color_dir, f"{file_name_wo_ext}_colorized.png")

        try:
            # Use our clean colorization function - it will raise exceptions if it fails
            colorized_path = colorize_image(
                frame_path,
                out_path,
                model='stable',  # Force the stable model
                render_factor=render_factor
            )
            
            colorized_paths.append(colorized_path)
            print(f"First frame colorized successfully: {colorized_path}")
            if os.path.exists(colorized_path):
                print(f"First frame colorization confirmed successful.")
            else:
                print(f"WARNING: First frame output file does not exist: {colorized_path}")
        except Exception as e:
            print(f"Error colorizing first frame: {str(e)}")
            raise  # Re-raise to stop processing
      # Process the rest of the frames
    remaining_frames = frame_paths[1:] if total_frames > 0 else []
    for i in tqdm(range(0, len(remaining_frames), batch_size), desc='Colorizing frame batches'):
        batch_frames = remaining_frames[i:i+batch_size]
        for frame_path in tqdm(batch_frames, desc='Frames in batch'):
            base_name = os.path.basename(frame_path)
            file_name_wo_ext = os.path.splitext(base_name)[0]
            out_path = os.path.join(color_dir, f"{file_name_wo_ext}_colorized.png")

            # Use our clean colorization function - always use stable model
            try:
                colorized_path = colorize_image(
                    frame_path,
                    out_path,
                    model='stable',  # Always use stable model 
                    render_factor=render_factor
                )
                colorized_paths.append(colorized_path)
            except Exception as e:
                print(f"Error colorizing frame {frame_path}: {str(e)}")
                # Try again with a lower render factor if it fails
                try:
                    lower_factor = max(25, render_factor - 10)
                    print(f"Retrying with lower render factor: {lower_factor}")
                    colorized_path = colorize_image(
                        frame_path,
                        out_path,
                        model='stable',
                        render_factor=lower_factor
                    )
                    colorized_paths.append(colorized_path)
                except Exception as retry_e:
                    print(f"Retry also failed: {str(retry_e)}")
                    raise

    print(f"Colorization completed. {len(colorized_paths)} frames were colorized.")
    return colorized_paths


def frames_to_video(frame_paths, output_path, fps=24):
    """Create a video from a list of image frames"""
    if not frame_paths:
        raise ValueError("No frames provided to create video")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get frame size from the first frame
    frame = cv2.imread(frame_paths[0])
    if frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    # Set up video writer
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_path in tqdm(frame_paths, desc='Writing video'):
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame file not found: {frame_path}")

        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        video.write(img)

    video.release()
    print(f"Video created successfully at {output_path}")


def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg"""
    ffmpeg_cmd = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

    # Check if the video has an audio stream
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
                 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    try:
        result = subprocess.run(probe_cmd, check=False,
                                capture_output=True, text=True)
        has_audio = result.stdout.strip() == 'audio'
    except Exception:
        # In case of error, assume no audio
        has_audio = False

    if not has_audio:
        print(f"No audio stream found in {video_path}")
        # Create a silent audio file of the same duration as the video
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        try:
            result = subprocess.run(
                duration_cmd, check=False, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            print(f"Creating silent audio of duration {duration} seconds")
            cmd = [
                ffmpeg_cmd, '-y', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', str(duration), '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
            ]
        except Exception:
            print("Could not determine video duration, using default 5 seconds")
            cmd = [
                ffmpeg_cmd, '-y', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', '5', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
            ]
    else:
        # Normal extraction of audio
        cmd = [
            ffmpeg_cmd, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
        ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def mux_audio_to_video(video_path, audio_path, output_path):
    """Combine audio with video using ffmpeg"""
    ffmpeg_cmd = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    cmd = [
        ffmpeg_cmd, '-y', '-i', video_path, '-i', audio_path, '-c:v', 'copy',
        '-c:a', 'aac', '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0',
        output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def get_video_fps(video_path):
    """Get the original FPS of the video"""
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
        print(f"Could not determine FPS, using default of 24. Error: {e}")
        return 24.0


def clean_temp_directories():
    """Clean up all temporary directories and recreate them"""
    # Clean up old temp directories if they exist
    for temp_dir in [TEMP_FRAMES_DIR, TEMP_COLOR_DIR]:
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Create new temp directories
    os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
    os.makedirs(TEMP_COLOR_DIR, exist_ok=True)
    print("Temporary directories cleaned and recreated.")


def process_video(video_path, args):
    """Process a single video with the given arguments"""
    print(f'\n==== Processing video: {video_path} ====')

    # Create output directories
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Always clean temp directories before processing
    clean_temp_directories()

    try:
        # 1. Extract frames
        frame_paths = extract_frames(video_path, TEMP_FRAMES_DIR)
        if args.no_colorize:
            print("Colorization disabled, using original frames")
            colorized_paths = frame_paths
        else:
            # 2. Colorize frames (with either one model or both)
            if args.both:
                print(
                    f"Processing with different render factors of the stable model (artistic model currently not working)")
                # Process with stable model at first render factor
                stable_dir = os.path.join(TEMP_COLOR_DIR, 'stable')
                os.makedirs(stable_dir, exist_ok=True)
                stable_paths = colorize_frames(
                    frame_paths,
                    stable_dir,
                    render_factor=args.render_factor,
                    model='stable'
                )

                # Process with stable model at a different render factor (higher quality)
                high_quality_dir = os.path.join(TEMP_COLOR_DIR, 'high_quality')
                os.makedirs(high_quality_dir, exist_ok=True)
                higher_render_factor = min(45, args.render_factor + 5)  # Increase render factor but cap at 45
                high_quality_paths = colorize_frames(
                    frame_paths,
                    high_quality_dir,
                    render_factor=higher_render_factor,
                    model='stable'
                )

                # Use higher quality results as the final output
                colorized_paths = high_quality_paths
                print(f"Using higher quality stable model results (render factor: {higher_render_factor}) as final output")
            else:
                # Process with single model (always the stable model)
                print(
                    f"Processing with stable model (render factor: {args.render_factor})")
                colorized_paths = colorize_frames(
                    frame_paths,
                    TEMP_COLOR_DIR,
                    render_factor=args.render_factor,
                    model='stable'  # Always use stable model
                )

        # 3. Create video with correct FPS
        fps = get_video_fps(video_path)

        # Create appropriate output names
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Set proper output path if not provided
        if not args.output:
            model_name = "both" if args.both else args.model
            args.output = os.path.join(
                OUTPUTS_DIR, f"{base_name}_{model_name}_rf{args.render_factor}.mp4")

        # Process audio
        temp_video_path = os.path.join(OUTPUTS_DIR, "temp_colorized_video.mp4")
        frames_to_video(colorized_paths, temp_video_path, fps=fps)

        # 4. Handle audio processing
        temp_audio_path = os.path.join(OUTPUTS_DIR, 'temp_audio.wav')
        enhanced_audio_path = os.path.join(OUTPUTS_DIR, 'enhanced_audio.wav')

        # Extract audio from original video
        extract_audio(video_path, temp_audio_path)

        # Improve audio if requested
        if not args.no_audio_improve and os.path.exists(os.path.join(os.path.dirname(__file__), 'audio_enhancer.py')):
            try:
                print("Enhancing audio...")
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__),
                                 'audio_enhancer.py'),
                    '-i', temp_audio_path,
                    '-o', enhanced_audio_path
                ]
                subprocess.run(cmd, check=True)
                audio_to_use = enhanced_audio_path
                print("Audio enhancement successful")
            except Exception as e:
                print(f"Audio enhancement failed: {e}")
                audio_to_use = temp_audio_path
        else:
            # Use original audio
            audio_to_use = temp_audio_path

        # 5. Combine video and audio
        mux_audio_to_video(temp_video_path, audio_to_use, args.output)
        print(f'Final colorized video saved to: {args.output}')

        # Create additional output if both models were used
        if args.both and 'stable_paths' in locals():
            stable_video_path = os.path.join(
                OUTPUTS_DIR, f"{base_name}_stable_rf{args.render_factor}.mp4")
            stable_temp_path = os.path.join(
                OUTPUTS_DIR, "temp_stable_video.mp4")
            frames_to_video(stable_paths, stable_temp_path, fps=fps)
            mux_audio_to_video(
                stable_temp_path, audio_to_use, stable_video_path)
            print(f'Stable model video saved to: {stable_video_path}')

            # Clean up extra temp file
            if os.path.exists(stable_temp_path):
                os.remove(stable_temp_path)
                # Verify that output files exist before considering the process successful
        if not os.path.exists(args.output):
            raise RuntimeError(f"Output file {args.output} was not created")

        if args.both and 'stable_paths' in locals():
            stable_video_path = os.path.join(
                OUTPUTS_DIR, f"{base_name}_stable_rf{args.render_factor}.mp4")
            if not os.path.exists(stable_video_path):
                print(
                    f"WARNING: Stable model output {stable_video_path} was not created")

        # Cleanup temp files
        print("Cleaning up temporary files...")
        for temp_file in [temp_video_path, temp_audio_path, enhanced_audio_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Move the input video to processed folder only after successful processing
        processed_path = move_to_processed(video_path)
        print(f"Video moved to {processed_path}")

        print("Video processing complete!")
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Processing failed - no fallback colorization will be attempted.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean video colorization pipeline")
    parser.add_argument('--render-factor', '-r', type=int, default=MAX_RENDER_FACTOR,
                        help=f'Render factor for colorization (default: {MAX_RENDER_FACTOR} for maximum quality)')
    parser.add_argument('--single-model', action='store_true',
                        help='Process with a single model instead of both stable and artistic')
    parser.add_argument('--model', '-m', choices=['stable', 'artistic'], default='stable',
                        help='Colorization model when using single model (default: stable) - Note: currently only stable model works reliably')
    parser.add_argument('--no-restore', action='store_true',
                        help='Disable restoration step')
    parser.add_argument('--enhance', action='store_true',
                        help='Enable enhancement (disabled by default)')
    parser.add_argument('--no-colorize', action='store_true',
                        help='Disable colorization step')
    parser.add_argument('--no-audio-improve', action='store_true',
                        help='Disable audio improvement step')
    parser.add_argument(
        '--input', '-i', help='Input video file (defaults to first video in inputs/)')
    parser.add_argument(
        '--output', '-o', help='Output video file (defaults to auto-generated name)')
    parser.add_argument('--single-run', action='store_true',
                        help='Run once instead of continuous mode')
    parser.add_argument('--check-interval', type=int, default=60,
                        help='Interval in seconds to check for new videos in continuous mode (default: 60)')
    args = parser.parse_args()

    # Create output and processed directories
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # Set defaults for both/single model
    args.both = not args.single_model    # Single video mode or continuous mode
    if not args.single_run:  # Default to continuous mode
        print("\n==== STARTING CONTINUOUS VIDEO PROCESSING MODE ====")
        print(f"Using render factor {args.render_factor} for maximum quality")
        if args.both:
            print("Processing with both stable and artistic models")
        else:
            print(f"Processing with {args.model} model")
        print(f"Checking for new videos every {args.check_interval} seconds")
        print("Press Ctrl+C to exit\n")
        
        while True:
            # Find a video to process
            video_path = args.input if args.input else find_video_file()
            
            if video_path:
                # Reset output path for each video
                args.output = None
                
                # Process the video
                success = process_video(video_path, args)
                
                if success:
                    print(f"Successfully processed: {video_path}")
                    # Reset input path only after successful processing
                    args.input = None
                else:
                    print(f"Failed to process: {video_path}")
                    print(f"Waiting {args.check_interval} seconds before retrying...")
                    time.sleep(args.check_interval)
                    # Do not reset input path after failure so we retry the same video
            else:
                print(f"No videos found in inputs directory. Checking again in {args.check_interval} seconds...")
                time.sleep(args.check_interval)
                
                # Always clean temp directories after each attempt
                clean_temp_directories()
    else:
        # Regular single-video mode
        video_path = args.input if args.input else find_video_file()
        if not video_path:
            print('No video file found in inputs/')
            return 1

        # Process the video
        success = process_video(video_path, args)

        # Always clean temp directories at the end
        clean_temp_directories()

        return 0 if success else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Cleaning up...")
        clean_temp_directories()
        print("Cleanup complete. Exiting.")
        sys.exit(0)
