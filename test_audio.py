#!/usr/bin/env python3
"""
Test script for audio enhancement on MP4 files
This script checks the audio enhancement functionality by:
1. Extracting audio from a video file
2. Enhancing the audio
3. Re-muxing the audio back to the video
4. Comparing the before/after results

Usage:
    python test_audio.py -i input_video.mp4

Options:
    -i, --input   Input video file to test (required)
    -v, --verbose Enable verbose output
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import audio enhancer module
try:
    from audio_enhancer import AudioEnhancer, extract_audio, mux_audio_to_video
except ImportError:
    print("[ERROR] Could not import audio_enhancer module. Make sure it exists and is in the Python path.")
    sys.exit(1)

def test_audio_enhancement(input_file, verbose=False):
    """
    Test the audio enhancement process on a video file
    
    Args:
        input_file (str): Path to input video file
        verbose (bool): Enable verbose output
        
    Returns:
        bool: True if successful, False otherwise
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return False
        
    # Create temp directory for test files
    temp_dir = Path("temp_audio_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Define temporary files
    temp_audio_orig = temp_dir / "original_audio.wav"
    temp_audio_enhanced = temp_dir / "enhanced_audio.wav"
    output_video = temp_dir / f"enhanced_{input_path.name}"
    
    print(f"[INFO] Testing audio enhancement on: {input_path}")
    start_time = time.time()
    
    # Step 1: Extract audio
    print("[INFO] Step 1: Extracting audio from video...")
    if not extract_audio(str(input_path), str(temp_audio_orig)):
        print("[ERROR] Failed to extract audio from video")
        return False
    
    # Step 2: Enhance the audio
    print("[INFO] Step 2: Enhancing audio...")
    enhancer = AudioEnhancer(verbose=verbose)
    if not enhancer.enhance_audio(str(temp_audio_orig), str(temp_audio_enhanced)):
        print("[ERROR] Failed to enhance audio")
        return False
    
    # Step 3: Mux enhanced audio back to video
    print("[INFO] Step 3: Adding enhanced audio to video...")
    if not mux_audio_to_video(str(input_path), str(temp_audio_enhanced), str(output_video)):
        print("[ERROR] Failed to add enhanced audio to video")
        return False
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    print(f"[INFO] Audio enhancement test completed in {elapsed_time:.2f} seconds")
    
    # Print file locations
    print("\nTest Output Files:")
    print(f"Original Audio: {temp_audio_orig}")
    print(f"Enhanced Audio: {temp_audio_enhanced}")
    print(f"Enhanced Video: {output_video}")
    
    # Open the enhanced video with the default player if available
    try:
        print("\n[INFO] Attempting to play enhanced video...")
        if os.name == 'nt':  # Windows
            os.startfile(output_video)
        elif os.name == 'posix':  # macOS, Linux
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{output_video}"')
            else:  # Linux
                os.system(f'xdg-open "{output_video}"')
        print("[INFO] Video should open in your default player")
    except Exception as e:
        print(f"[INFO] Could not open video automatically: {e}")
        print(f"[INFO] Please open manually: {output_video}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test audio enhancement functionality")
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    success = test_audio_enhancement(args.input, args.verbose)
    
    if success:
        print("\n[SUCCESS] Audio enhancement test completed successfully")
        return 0
    else:
        print("\n[ERROR] Audio enhancement test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
