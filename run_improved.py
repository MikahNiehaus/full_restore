"""
Main script for running the full image and video restoration pipeline
with improved CPU fallback for colorization.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import time

def main():
    """Main entry point for the full restoration pipeline."""
    
    parser = argparse.ArgumentParser(description="Full restoration pipeline for old photos and videos")
    parser.add_argument("-i", "--input", help="Input file or directory path")
    parser.add_argument("-o", "--output", help="Output directory path")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Scale factor for upscaling")
    parser.add_argument("--artistic", action="store_true", help="Use artistic colorization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--video", action="store_true", help="Process as video even if image extension")
    parser.add_argument("--image", action="store_true", help="Process as image even if video extension")
    args = parser.parse_args()
    
    # Set up input/output paths
    if args.input:
        input_path = os.path.abspath(args.input)
    else:
        # Look for files in the inputs directory
        inputs_dir = os.path.join(os.getcwd(), "inputs")
        if not os.path.exists(inputs_dir):
            os.makedirs(inputs_dir)
            print(f"No input provided, and inputs directory empty.")
            print(f"Please place files in {inputs_dir} directory and run again.")
            return
            
        files = os.listdir(inputs_dir)
        if not files:
            print(f"No files found in {inputs_dir} directory.")
            return
            
        # Use the first file in the directory
        input_path = os.path.join(inputs_dir, files[0])
        print(f"No input specified, using: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return
        
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.join(os.getcwd(), "outputs")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if input is a video or image
    is_video = False
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        # Override based on flags
        if args.video:
            is_video = True
        elif args.image:
            is_video = False
    
    # Process based on type
    if is_video:
        process_video(input_path, output_dir, args.scale, args.artistic, args.cpu)
    else:
        process_image(input_path, output_dir, args.scale, args.artistic, args.cpu)

def process_image(input_path, output_dir, scale=4, artistic=True, force_cpu=False):
    """Process a single image with the unified processor."""
    try:
        from unified_image_processor import ImageProcessor
        
        print(f"Processing image: {input_path}")
        start_time = time.time()
        
        # Initialize the processor
        processor = ImageProcessor(output_dir=output_dir)
        
        # Process the image
        result_path = processor.process_image(
            input_path=input_path,
            output_dir=output_dir,
            scale=scale
        )
        
        elapsed = time.time() - start_time
        print(f"Image processing completed in {elapsed:.2f} seconds")
        
        if result_path and os.path.exists(result_path):
            print(f"Final image saved to: {result_path}")
        else:
            print("Image processing failed!")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def process_video(input_path, output_dir, scale=4, artistic=True, force_cpu=False):
    """Process a video with the video restoration pipeline."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeOldify'))
        # Use the optimized video processor
        from video_restore_and_colorize import process_video as video_processor
        
        print(f"Processing video: {input_path}")
        start_time = time.time()
        
        # Process the video
        result_path = video_processor(
            input_path=input_path,
            output_dir=output_dir,
            scale=scale,
            artistic=artistic,
            force_cpu=force_cpu
        )
        
        elapsed = time.time() - start_time
        print(f"Video processing completed in {elapsed:.2f} seconds")
        
        if result_path and os.path.exists(result_path):
            print(f"Final video saved to: {result_path}")
        else:
            print("Video processing failed!")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
