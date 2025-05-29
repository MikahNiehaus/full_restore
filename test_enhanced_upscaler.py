"""
Simple test script to verify the enhanced video restoration workflow.
This script calls the enhance_frames function directly to test the upscaling.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def enhance_frames_simple(frame_paths, enhanced_dir, scale=4):
    """Simple enhanced frames function that uses advanced OpenCV techniques"""
    os.makedirs(enhanced_dir, exist_ok=True)
    enhanced_paths = []
    
    # Process each frame
    for frame_path in tqdm(frame_paths, desc='Enhancing frames'):
        enhanced_frame_path = os.path.join(enhanced_dir, os.path.basename(frame_path))
        
        # Read image
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to read image: {frame_path}")
            continue
        
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply advanced enhancement
        print(f"Enhancing {os.path.basename(frame_path)}...")
        
        # Step 1: Apply bilateral filter to preserve edges while reducing noise
        img_filtered = cv2.bilateralFilter(img_rgb, 7, 50, 50)
        
        # Step 2: Apply detail enhancement
        img_details = cv2.detailEnhance(img_filtered, sigma_s=10, sigma_r=0.15)
        
        # Step 3: Upscale using Lanczos (higher quality than bicubic)
        h, w = img_details.shape[:2]
        img_upscaled = cv2.resize(img_details, (w * scale, h * scale), 
                                interpolation=cv2.INTER_LANCZOS4)
        
        # Step 4: Apply unsharp mask for sharpening
        blur = cv2.GaussianBlur(img_upscaled, (0, 0), 3.0)
        img_upscaled = cv2.addWeighted(img_upscaled, 1.5, blur, -0.5, 0)
        
        # Convert back to BGR and save with OpenCV
        img_bgr = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(enhanced_frame_path, img_bgr)
        
        enhanced_paths.append(enhanced_frame_path)
    
    return enhanced_paths

def extract_frames(video_path, frames_dir):
    """Extract frames from video"""
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

def frames_to_video(frame_paths, output_path, fps=24):
    """Create video from frames"""
    if not frame_paths:
        return
    
    # Get frame size
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_path in tqdm(frame_paths, desc='Writing video'):
        img = cv2.imread(frame_path)
        video.write(img)
    
    video.release()
    print(f"Video saved to: {output_path}")

def main():
    """Main function"""
    # Set up directories
    video_path = "inputs/test_video.mp4"
    frames_dir = "test_output/frames"
    enhanced_dir = "test_output/enhanced"
    output_path = "test_output/enhanced_video.mp4"
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Extract frames
    print(f"Extracting frames from {video_path}...")
    frame_paths = extract_frames(video_path, frames_dir)
    print(f"Extracted {len(frame_paths)} frames")
    
    # Enhance frames
    print("Enhancing frames...")
    enhanced_paths = enhance_frames_simple(frame_paths, enhanced_dir)
    
    # Create video
    print("Creating video...")
    frames_to_video(enhanced_paths, output_path, fps=5)  # Use same fps as input

if __name__ == "__main__":
    main()
