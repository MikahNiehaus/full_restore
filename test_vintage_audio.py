#!/usr/bin/env python3
"""
Test script for vintage audio enhancement on old films

This script processes a video file with the VintageAudioEnhancer 
to improve audio quality with vintage-specific filters.
"""
import os
import sys
from pathlib import Path
import logging
import time
import shutil
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Check if ffmpeg is available
try:
    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    logging.info("FFmpeg is available")
except (subprocess.SubprocessError, FileNotFoundError):
    logging.error("FFmpeg is not available. Please install FFmpeg and make sure it's in your PATH.")
    sys.exit(1)

# Import the vintage audio enhancer
try:
    from vintage_audio_enhancer import VintageAudioEnhancer, enhance_audio_from_video
except ImportError:
    logging.error("Could not import VintageAudioEnhancer - make sure vintage_audio_enhancer.py is in the current directory")
    sys.exit(1)

def test_vintage_audio_enhancement(video_path=None):
    """Test the vintage audio enhancement on a video file"""
    
    # Find a test video if not provided
    if not video_path:
        inputs_dir = Path("inputs")
        if inputs_dir.exists():
            videos = list(inputs_dir.glob("*.mp4"))
            if videos:
                video_path = videos[0]
                logging.info(f"Using video: {video_path}")
            else:
                logging.error("No .mp4 files found in 'inputs' directory")
                sys.exit(1)
        else:
            logging.error("No video provided and 'inputs' directory not found")
            sys.exit(1)
    else:
        video_path = Path(video_path)
        
    # Ensure video exists
    if not video_path.exists():
        logging.error(f"Video not found: {video_path}")
        sys.exit(1)
      # Create output folder and ensure it exists
    output_dir = Path("enhanced_audio_tests")
    
    # First, try to remove the directory if it exists but is problematic
    if output_dir.exists():
        try:
            for file in output_dir.iterdir():
                try:
                    file.unlink()
                except:
                    pass
        except:
            pass
    
    # Now create the directory
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        else:
            logging.info(f"Output directory already exists: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory: {e}")
        
        # Try an alternative approach with os.makedirs
        try:
            os.makedirs("enhanced_audio_tests", exist_ok=True)
            logging.info("Created output directory using os.makedirs")
        except Exception as e2:
            logging.error(f"Failed again to create output directory: {e2}")
            sys.exit(1)
    
    # Process the video
    logging.info(f"Processing video: {video_path}")
    temp_audio_path = output_dir / "temp_original.wav"
    enhanced_audio_path = output_dir / f"{video_path.stem}_vintage_enhanced.wav"
    result_video_path = output_dir / f"{video_path.stem}_audio_enhanced.mp4"
      # Create the temp directory for audio extraction
    temp_audio_path = Path("temp_audio.wav")  # Use root directory for simplicity
    enhanced_audio_path = output_dir / f"{video_path.stem}_vintage_enhanced.wav"
    result_video_path = output_dir / f"{video_path.stem}_audio_enhanced.mp4"    # Extract audio with ffmpeg while preserving timing information
    logging.info("Extracting original audio...")
    # Use absolute paths
    video_abs_path = video_path.absolute()
    temp_abs_path = temp_audio_path.absolute()
    
    # First get video information (frame rate, duration)
    video_info_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,duration -of default=noprint_wrappers=1 "{video_abs_path}"'
    video_info = subprocess.check_output(video_info_cmd, shell=True, text=True)
    
    # Parse frame rate
    frame_rate = None
    duration = None
    for line in video_info.split('\n'):
        if line.startswith('r_frame_rate='):
            # Format is typically "numerator/denominator"
            parts = line.split('=')[1].split('/')
            if len(parts) == 2:
                try:
                    frame_rate = float(parts[0]) / float(parts[1])
                    logging.info(f"Video frame rate: {frame_rate} fps")
                except (ValueError, ZeroDivisionError):
                    logging.warning(f"Could not parse frame rate: {line}")
        elif line.startswith('duration='):
            try:
                duration = float(line.split('=')[1])
                logging.info(f"Video duration: {duration} seconds")
            except ValueError:
                logging.warning(f"Could not parse duration: {line}")
      # We already have absolute paths from earlier
    
    # Extract audio with precise timing preservation
    extract_cmd = f'ffmpeg -y -i "{video_abs_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_abs_path}"'
    logging.info(f"Running command: {extract_cmd}")
    extract_result = os.system(extract_cmd)
    
    if extract_result != 0 or not temp_audio_path.exists():
        logging.error("Failed to extract audio from video")
        logging.error(f"Check if ffmpeg is installed and working. Command used: {extract_cmd}")
        sys.exit(1)
    
    # Verify audio was extracted correctly
    if temp_audio_path.stat().st_size == 0:
        logging.error("Extracted audio file is empty")
        sys.exit(1)
    
    audio_info_cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels -of default=noprint_wrappers=1 "{temp_abs_path}"'
    try:
        audio_info = subprocess.check_output(audio_info_cmd, shell=True, text=True)
        logging.info(f"Extracted audio properties: {audio_info.strip()}")
    except subprocess.SubprocessError:
        logging.warning("Could not retrieve audio properties")
    
    logging.info(f"Successfully extracted audio to: {temp_audio_path}")
      # Enhance audio
    logging.info("Enhancing audio with vintage-specific processing...")
    start_time = time.time()
    enhancer = VintageAudioEnhancer(verbose=True)
    
    # Create parent directory for enhanced audio if it doesn't exist
    if not enhanced_audio_path.parent.exists():
        enhanced_audio_path.parent.mkdir(parents=True, exist_ok=True)
    success = enhancer.enhance_vintage_audio(str(temp_audio_path), str(enhanced_audio_path))
    elapsed = time.time() - start_time
    
    if success and enhanced_audio_path.exists():
        logging.info(f"Audio enhancement completed in {elapsed:.2f} seconds")
        
        # Try to create waveform visualization to verify audio enhancement
        try:
            import matplotlib.pyplot as plt
            import soundfile as sf
            
            # Load both original and enhanced audio for comparison
            orig_data, orig_sr = sf.read(str(temp_audio_path))
            enhanced_data, enhanced_sr = sf.read(str(enhanced_audio_path))
            
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            # Plot original audio waveform
            plt.subplot(2, 1, 1)
            plt.title("Original Audio Waveform")
            plt.plot(orig_data)
            
            # Plot enhanced audio waveform
            plt.subplot(2, 1, 2)
            plt.title("Enhanced Audio Waveform")
            plt.plot(enhanced_data)
            
            # Save plot
            plot_path = output_dir / f"{video_path.stem}_waveform_comparison.png"
            plt.tight_layout()
            plt.savefig(str(plot_path))
            logging.info(f"Saved waveform comparison to: {plot_path}")
        except Exception as e:
            logging.warning(f"Could not create waveform visualization: {e}")
          # Create enhanced video with the new audio
        logging.info("Creating video with enhanced audio...")
        # Using absolute paths for reliability
        video_abs_path = video_path.absolute()
        enhanced_abs_path = enhanced_audio_path.absolute()
        result_abs_path = result_video_path.absolute()
        
        # Create parent directory for result video if it doesn't exist
        if not result_video_path.parent.exists():
            result_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use special flags to ensure perfect audio-video sync
        # -vsync 0: preserve the original timestamps
        # -async 1: audio sync method that minimizes drift
        # -af "aresample=async=1": resample audio to maintain sync
        mux_cmd = (f'ffmpeg -y -i "{video_abs_path}" -i "{enhanced_abs_path}" '
                  f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k '
                  f'-vsync 0 -af "aresample=async=1" '
                  f'-metadata:s:a:0 "sync_audio=true" '
                  f'-shortest "{result_abs_path}"')
        logging.info(f"Running mux command: {mux_cmd}")
        mux_result = os.system(mux_cmd)
        
        if mux_result == 0 and result_video_path.exists():
            logging.info(f"Enhanced video created: {result_video_path}")
        else:
            logging.error("Failed to create video with enhanced audio")
            logging.error(f"Mux command exit code: {mux_result}")
              # Try with original audio as fallback
            logging.info("Trying with original audio as fallback...")
            fallback_mux_cmd = (f'ffmpeg -y -i "{video_abs_path}" -i "{temp_abs_path}" '
                              f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k '
                              f'-vsync 0 -af "aresample=async=1" '
                              f'-metadata:s:a:0 "sync_audio=true" '
                              f'-shortest "{result_abs_path}"')
            fallback_result = os.system(fallback_mux_cmd)
            
            if fallback_result == 0 and result_video_path.exists():
                logging.info(f"Video created with original audio: {result_video_path}")
            else:
                logging.error("Failed to create video with any audio")
    else:
        logging.error("Audio enhancement failed")
          # Try with original audio as fallback
        logging.info("Trying with original audio as fallback...")
        video_abs_path = video_path.absolute()
        temp_abs_path = temp_audio_path.absolute()
        result_abs_path = result_video_path.absolute()
        
        # Create parent directory for result video if it doesn't exist
        if not result_video_path.parent.exists():
            result_video_path.parent.mkdir(parents=True, exist_ok=True)
            
        fallback_mux_cmd = (f'ffmpeg -y -i "{video_abs_path}" -i "{temp_abs_path}" '
                          f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k '
                          f'-vsync 0 -af "aresample=async=1" '
                          f'-metadata:s:a:0 "sync_audio=true" '
                          f'-shortest "{result_abs_path}"')
        fallback_result = os.system(fallback_mux_cmd)
        
        if fallback_result == 0 and result_video_path.exists():
            logging.info(f"Video created with original audio: {result_video_path}")
        else:
            logging.error("Failed to create video with any audio")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_vintage_audio_enhancement(sys.argv[1])
    else:
        test_vintage_audio_enhancement()
