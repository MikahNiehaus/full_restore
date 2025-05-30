#!/usr/bin/env python3
"""
DeOldify Full Restore - Single Video Processor

This script processes a single video file with the DeOldify Full Restore pipeline.
It colorizes the video using DeOldify at maximum quality and enhances the audio.

Usage:
    python process_single_video.py input_video.mp4 [output_video.mp4]
"""

import os
import sys
import argparse
from pathlib import Path

# Import the main processor
from full_restore import VideoProcessor

def process_single_video(
    input_path, 
    output_path=None, 
    render_factor=40, 
    enhance_audio=True
):
    """
    Process a single video file
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save output video (optional)
        render_factor (int): DeOldify render factor (10-45)
        enhance_audio (bool): Whether to enhance audio
        
    Returns:
        str: Path to output video file
    """
    input_path = Path(input_path)
    
    # Create temporary directories
    temp_dir = Path('temp_video_frames')
    temp_dir.mkdir(exist_ok=True)
    
    # Set up output path if not provided
    if output_path is None:
        output_path = Path('outputs') / f"{input_path.stem}_colorized.mp4"
    else:
        output_path = Path(output_path)
        
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Processing video: {input_path}")
    print(f"[INFO] Output will be saved to: {output_path}")
    print(f"[INFO] Render factor: {render_factor}")
    print(f"[INFO] Audio enhancement: {'Enabled' if enhance_audio else 'Disabled'}")
    
    # Create processor with temporary paths
    processor = VideoProcessor(
        inputs_dir='temp_inputs',
        outputs_dir=output_path.parent,
        processed_dir='temp_processed',
        temp_dir='temp_video_frames',
        render_factor=render_factor,
        enhance_audio=enhance_audio
    )
    
    # Create a temporary input directory and copy the video there
    temp_input_dir = Path('temp_inputs')
    temp_input_dir.mkdir(exist_ok=True)
    
    import shutil
    temp_input_path = temp_input_dir / input_path.name
    shutil.copy(input_path, temp_input_path)
    
    # Process the video
    try:
        result_path = processor.process_video(temp_input_path)
        
        # If the output path is different from the default, rename the result
        if result_path and result_path.exists() and str(result_path) != str(output_path):
            shutil.move(str(result_path), str(output_path))
            print(f"[INFO] Renamed output to: {output_path}")
    finally:
        # Clean up temporary directories
        if temp_input_dir.exists():
            shutil.rmtree(temp_input_dir, ignore_errors=True)
        if Path('temp_processed').exists():
            shutil.rmtree(Path('temp_processed'), ignore_errors=True)
    
    if output_path.exists():
        print(f"[INFO] Processing complete: {output_path}")
        return str(output_path)
    else:
        print("[ERROR] Processing failed - output file not found")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process a single video with DeOldify Full Restore")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", nargs="?", help="Output video file path (optional)")
    parser.add_argument("--render-factor", "-r", type=int, default=40, help="DeOldify render factor (10-45)")
    parser.add_argument("--no-audio-enhance", action="store_true", help="Disable audio enhancement")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return 1
    
    # Process video
    result = process_single_video(
        args.input, 
        args.output, 
        args.render_factor, 
        not args.no_audio_enhance
    )
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
