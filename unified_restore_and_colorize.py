"""
Unified pipeline for restoring, colorizing, and enhancing both images and videos using OOP.
Uses the same logic for both, frame-by-frame for videos.
"""
import os
import sys
import cv2
from pathlib import Path
from unified_image_processor import ImageProcessor
from PIL import Image
import shutil

def is_image_file(path):
    return str(path).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))

def is_video_file(path):
    return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

def process_image(input_path, output_dir, scale=2):
    processor = ImageProcessor(output_dir)
    result = processor.process_image(input_path, output_dir, scale)
    if result:
        print(f"[SUCCESS] Image processed: {result}")
    else:
        print(f"[ERROR] Failed to process image: {input_path}")

def process_video(input_path, output_dir, scale=2, fps=None):
    processor = ImageProcessor(output_dir)
    temp_frames = os.path.join(output_dir, "temp_frames")
    temp_restored = os.path.join(output_dir, "temp_restored")
    os.makedirs(temp_frames, exist_ok=True)
    os.makedirs(temp_restored, exist_ok=True)
    
    # Extract frames
    vidcap = cv2.VideoCapture(input_path)
    frame_paths = []
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frame_path = os.path.join(temp_frames, f"frame_{count:06d}.png")
        cv2.imwrite(frame_path, image)
        frame_paths.append(frame_path)
        count += 1
    vidcap.release()
    if not frame_paths:
        print("[ERROR] No frames extracted from video.")
        return
    print(f"[INFO] Extracted {len(frame_paths)} frames.")
    
    # Process each frame
    restored_paths = []
    for frame_path in frame_paths:
        restored_path = os.path.join(temp_restored, os.path.basename(frame_path))
        result = processor.process_image(frame_path, temp_restored, scale)
        if result:
            restored_paths.append(result)
        else:
            # Fallback: copy original frame
            shutil.copy(frame_path, restored_path)
            restored_paths.append(restored_path)
    print(f"[INFO] Processed {len(restored_paths)} frames.")
    
    # Assemble video
    first_frame = cv2.imread(restored_paths[0])
    height, width, _ = first_frame.shape
    if fps is None:
        # Try to get FPS from input video
        vcap = cv2.VideoCapture(input_path)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        vcap.release()
        if not fps or fps < 1:
            fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(output_dir, f"{Path(input_path).stem}_restored_colorized_enhanced.mp4")
    video_writer = cv2.VideoWriter(out_video_path, fourcc, int(fps), (width, height))
    for frame_path in restored_paths:
        img = cv2.imread(frame_path)
        video_writer.write(img)
    video_writer.release()
    print(f"[SUCCESS] Video saved: {out_video_path}")
    
    # Cleanup temp dirs
    shutil.rmtree(temp_frames, ignore_errors=True)
    shutil.rmtree(temp_restored, ignore_errors=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified restore, colorize, and enhance for images and videos.")
    parser.add_argument("-i", "--input", required=True, help="Input image or video file")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Scale factor (default: 2)")
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    scale = args.scale
    
    if is_image_file(input_path):
        process_image(input_path, output_dir, scale)
    elif is_video_file(input_path):
        process_video(input_path, output_dir, scale)
    else:
        print("[ERROR] Input file must be an image or video.")

if __name__ == "__main__":
    main()
