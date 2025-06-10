#!/usr/bin/env python3
"""
Full Restoration Pipeline Test Script with GPU Optimization

This script runs the entire image restoration, colorization, and enhancement pipeline
with GPU acceleration properly configured. It verifies that both:
1. The image_restorer.py with GPU works correctly
2. The DeOldify colorizer with GPU works correctly 

It tests the complete pipeline end-to-end with proper fallbacks.
"""

import os
import sys
import time
from pathlib import Path
import torch
import cv2
import numpy as np
import shutil
import subprocess

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from both fixed (new) modules and DeOldify
from image_restorer_fixed import ImageRestorer
from vintage_audio_enhancer import VintageAudioEnhancer
# Import DeOldify components
sys.path.append('./DeOldify')
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer
except ImportError:
    print("[ERROR] Failed to import DeOldify modules. Make sure the DeOldify directory exists and is properly set up.")
    sys.exit(1)

def run_full_pipeline_test(image_path=None, video_path=None, output_dir=None):
    """Test the full restoration pipeline with GPU optimization"""
    print("\n===== Testing Full Restoration Pipeline with GPU Optimization =====")
    
    # Use either image or video test
    if image_path is None and video_path is None:
        # Look for test files
        if Path("inputs").exists():
            images = list(Path("inputs").glob("*.jpg")) + list(Path("inputs").glob("*.png"))
            videos = list(Path("inputs").glob("*.mp4"))
            
            if images:
                image_path = str(images[0])
                print(f"Using test image: {image_path}")
            elif videos:
                video_path = str(videos[0])
                print(f"Using test video: {video_path}")
        
        # If still no test file, create a test image
        if image_path is None and video_path is None:
            print("No test files found. Creating a test image...")
            test_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (100, 100), (400, 400), (200, 200, 200), -1)
            cv2.circle(test_img, (250, 250), 100, (100, 100, 100), -1)
            # Add some noise to simulate an old photo
            noise = np.random.normal(0, 15, test_img.shape).astype(np.uint8)
            test_img = cv2.add(test_img, noise)
            image_path = "test_vintage_image.jpg"
            cv2.imwrite(image_path, test_img)
            print(f"Created test image: {image_path}")
    
    # Create output directory
    if not output_dir:
        output_dir = "full_pipeline_test_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Test GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {device_name}")
        print(f"CUDA version: {cuda_version}")
        print(f"GPU memory: {gpu_memory:.2f} GB")
        use_gpu = True
    else:
        print("Warning: CUDA not available. Testing will use CPU which will be much slower.")
        print("If you have a CUDA-compatible GPU, please ensure your drivers and PyTorch CUDA are properly installed.")
        use_gpu = False
    
    # Process based on whether we have an image or video
    if image_path:
        return test_image_pipeline(image_path, output_dir, use_gpu)
    elif video_path:
        return test_video_pipeline(video_path, output_dir, use_gpu)
    else:
        print("[ERROR] No test file available.")
        return False

def test_image_pipeline(image_path, output_dir, use_gpu=True):
    """Test the full image pipeline: restore > colorize > enhance"""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        return False
        
    print(f"\nProcessing image: {image_path}")
    
    # Set device for DeOldify
    if use_gpu:
        device.set(DeviceId.GPU0)
        print("[INFO] Set DeOldify to use GPU")
    else:
        device.set(DeviceId.CPU)
        print("[INFO] Set DeOldify to use CPU")
    
    try:
        # Step 1: Restore the image
        print("\n[Step 1/3] Restoring image...")
        torch_device = torch.device('cuda' if use_gpu else 'cpu')
        restorer = ImageRestorer(device=torch_device)
        
        restored_path = Path(output_dir) / f"{image_path.stem}_restored{image_path.suffix}"
        start_time = time.time()
        restorer.restore_image(
            image_path=str(image_path),
            output_path=str(restored_path)
        )
        restore_time = time.time() - start_time
        print(f"Restoration completed in {restore_time:.2f} seconds")
        
        # Step 2: Colorize the restored image
        print("\n[Step 2/3] Colorizing image...")
        colorizer = get_image_colorizer(artistic=True)
        colorized_path = Path(output_dir) / f"{image_path.stem}_colorized{image_path.suffix}"
        
        start_time = time.time()
        colorizer.plot_transformed_image(
            path=str(restored_path),
            render_factor=35,
            compare=False,
            watermarked=False,
            output_path=str(colorized_path)
        )
        colorize_time = time.time() - start_time
        print(f"Colorization completed in {colorize_time:.2f} seconds")
        
        # Step 3: Enhance the colorized image
        print("\n[Step 3/3] Enhancing image...")
        enhanced_path = Path(output_dir) / f"{image_path.stem}_enhanced{image_path.suffix}"
        
        start_time = time.time()
        # Use the GPU-optimized super-resolution
        restorer.sr_enhancer.outscale = 2  # 2x upscaling
        colorized_img = cv2.imread(str(colorized_path))
        output, _ = restorer.sr_enhancer.enhance(colorized_img)
        cv2.imwrite(str(enhanced_path), output)
        enhance_time = time.time() - start_time
        print(f"Enhancement completed in {enhance_time:.2f} seconds")
        
        # Calculate and print total time
        total_time = restore_time + colorize_time + enhance_time
        print(f"\nFull pipeline completed in {total_time:.2f} seconds")
        print(f"Results saved in {output_dir}:")
        print(f"  - Restored: {restored_path}")
        print(f"  - Colorized: {colorized_path}")
        print(f"  - Enhanced: {enhanced_path}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_pipeline(video_path, output_dir, use_gpu=True):
    """Test video processing pipeline with the vintage audio enhancer"""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return False
        
    print(f"\nProcessing video: {video_path}")
    
    # First check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("[INFO] FFmpeg is available")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[ERROR] FFmpeg is not available. Please install FFmpeg and make sure it's in your PATH.")
        return False
    
    # Create temporary directories
    temp_frames_dir = Path(output_dir) / "temp_frames"
    temp_restored_dir = Path(output_dir) / "temp_restored"
    temp_colorized_dir = Path(output_dir) / "temp_colorized"
    
    for dir_path in [temp_frames_dir, temp_restored_dir, temp_colorized_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set device for DeOldify
    if use_gpu:
        device.set(DeviceId.GPU0)
        print("[INFO] Set DeOldify to use GPU")
    else:
        device.set(DeviceId.CPU)
        print("[INFO] Set DeOldify to use CPU")
    
    try:
        # Step 1: Extract audio
        print("\n[Step 1/6] Extracting audio...")
        temp_audio_path = Path(output_dir) / "temp_audio.wav"
        extract_cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio_path}"'
        subprocess.run(extract_cmd, shell=True, check=True)
        print(f"Audio extracted to: {temp_audio_path}")
        
        # Step 2: Extract frames
        print("\n[Step 2/6] Extracting video frames...")
        extract_frames_cmd = f'ffmpeg -y -i "{video_path}" -vf "fps=24" "{temp_frames_dir / "%04d.jpg"}"'
        subprocess.run(extract_frames_cmd, shell=True, check=True)
        frames = sorted(list(temp_frames_dir.glob("*.jpg")))
        print(f"Extracted {len(frames)} frames")
        
        if len(frames) == 0:
            print("[ERROR] No frames extracted from video")
            return False
            
        # Step 3: Restore frames
        print("\n[Step 3/6] Restoring frames...")
        torch_device = torch.device('cuda' if use_gpu else 'cpu')
        restorer = ImageRestorer(device=torch_device)
        
        # Process just a subset of frames for testing
        test_frames = frames[:min(24, len(frames))]  # Process max 1 second (24 frames)
        for i, frame in enumerate(test_frames):
            restored_path = temp_restored_dir / frame.name
            restorer.restore_image(
                image_path=str(frame),
                output_path=str(restored_path)
            )
            if (i+1) % 5 == 0:
                print(f"Restored {i+1}/{len(test_frames)} frames")
        
        # Step 4: Colorize frames
        print("\n[Step 4/6] Colorizing frames...")
        colorizer = get_image_colorizer(artistic=True)
        
        for i, frame in enumerate(test_frames):
            restored_path = temp_restored_dir / frame.name
            colorized_path = temp_colorized_dir / frame.name
            
            try:
                colorizer.plot_transformed_image(
                    path=str(restored_path),
                    render_factor=35,
                    compare=False,
                    watermarked=False,
                    output_path=str(colorized_path)
                )
            except Exception as ce:
                print(f"[WARNING] Error colorizing frame {frame.name}: {ce}")
                # Copy the restored frame as fallback
                shutil.copy(str(restored_path), str(colorized_path))
                
            if (i+1) % 5 == 0:
                print(f"Colorized {i+1}/{len(test_frames)} frames")
        
        # Step 5: Enhance and process audio
        print("\n[Step 5/6] Enhancing audio...")
        enhanced_audio_path = Path(output_dir) / "enhanced_audio.wav"
        
        try:
            audio_enhancer = VintageAudioEnhancer(verbose=True)
            enhance_success = audio_enhancer.enhance_vintage_audio(
                str(temp_audio_path), 
                str(enhanced_audio_path)
            )
            
            if not enhance_success:
                print("[WARNING] Audio enhancement failed, using original audio")
                enhanced_audio_path = temp_audio_path
        except Exception as ae:
            print(f"[WARNING] Error enhancing audio: {ae}")
            enhanced_audio_path = temp_audio_path
        
        # Step 6: Create output video from processed frames
        print("\n[Step 6/6] Creating output video...")
        output_video_path = Path(output_dir) / f"{video_path.stem}_restored_colorized.mp4"
        
        create_video_cmd = (
            f'ffmpeg -y -framerate 24 -i "{temp_colorized_dir / "%04d.jpg"}" '
            f'-i "{enhanced_audio_path}" -c:v libx264 -preset medium -crf 18 '
            f'-c:a aac -b:a 192k -shortest -pix_fmt yuv420p '
            f'-vsync 0 -af "aresample=async=1" '
            f'-metadata:s:a:0 "sync_audio=true" '
            f'"{output_video_path}"'
        )
        subprocess.run(create_video_cmd, shell=True, check=True)
        
        print(f"\nFull video pipeline test completed")
        print(f"Output video saved to: {output_video_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Video pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command line arguments
    image_path = None
    video_path = None
    output_dir = None
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        if test_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = test_path
        elif test_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = test_path
        else:
            print(f"Unrecognized file format: {test_path}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    success = run_full_pipeline_test(image_path, video_path, output_dir)
    if success:
        print("\n✅ Full pipeline test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Full pipeline test FAILED")
        sys.exit(1)
