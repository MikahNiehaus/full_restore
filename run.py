import os
import sys
import traceback
from pathlib import Path
import argparse
from tqdm import tqdm

def has_audio_files(directory):
    audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')
    return any(f.lower().endswith(audio_exts) for f in os.listdir(directory))

def has_video_files(directory):
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    return any(f.lower().endswith(video_exts) for f in os.listdir(directory))

def has_image_files(directory):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    return any(f.lower().endswith(image_exts) for f in os.listdir(directory))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full restoration pipeline runner")
    parser.add_argument('--no-restore', action='store_true', help='Disable image/video restoration step')
    parser.add_argument('--no-colorize', action='store_true', help='Disable colorization step')
    parser.add_argument('--no-enhance', action='store_true', help='Disable enhancement step')
    parser.add_argument('--no-audio-improve', action='store_true', help='Disable audio improvement step')
    parser.add_argument('--color-boost', type=float, default=1.0, help='Set color boost amount for colorization (e.g., 4 for extra color, 1 for normal)')
    args = parser.parse_args()

    INPUTS_DIR = 'inputs'
    OUTPUTS_DIR = 'outputs'
    
    # Count total tasks for progress bar
    total_tasks = 0
    audio_files = [f for f in os.listdir(INPUTS_DIR) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"))]
    video_files = [f for f in os.listdir(INPUTS_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))]
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(INPUTS_DIR) if f.lower().endswith(image_exts)]
    total_tasks += len(audio_files) if not args.no_audio_improve else 0
    total_tasks += len(video_files)
    total_tasks += len(image_files)
    if total_tasks == 0:
        print(f"No supported audio, video, or image files found in '{INPUTS_DIR}'.")
        sys.exit(0)
    progress = tqdm(total=total_tasks, desc='Full Pipeline Progress', unit='file')
    ran = False

    # Test audio pipeline
    if not args.no_audio_improve:
        print("[INFO] Running audio pipeline test...")
        audio_test_result = os.system(f'{sys.executable} test_audio_enhancer.py')
        if audio_test_result == 0:
            print("[INFO] Audio pipeline test passed.")
        else:
            print("[WARNING] Audio pipeline test failed.")

    # Process audio files
    if has_audio_files(INPUTS_DIR) and not args.no_audio_improve:
        print("[INFO] Found audio file(s). Running audio_enhancer.py...")
        for fname in audio_files:
            in_path = os.path.join(INPUTS_DIR, fname)
            out_path = os.path.join(OUTPUTS_DIR, f"{Path(fname).stem}_enhanced.wav")
            os.system(f'{sys.executable} audio_enhancer.py -i "{in_path}" -o "{out_path}"')
            progress.update(1)
        ran = ran or bool(audio_files)

    # Process video files
    for fname in video_files:
        print(f"[INFO] Processing video: {fname}")
        video_flags = ''
        if args.no_restore:
            video_flags += ' --no-restore'
        if args.no_colorize:
            video_flags += ' --no-colorize'
        if args.no_enhance:
            video_flags += ' --no-enhance'
        if args.no_audio_improve:
            video_flags += ' --no-audio-improve'
        if args.color_boost > 1:
            video_flags += f' --color-boost {int(args.color_boost)}'  # Ensure integer value for multiple passes
        os.system(f'{sys.executable} video_restore_and_colorize.py{video_flags}')
        progress.update(1)
        ran = True

    # Process image files using our unified image processor
    if image_files:
        print("[INFO] Found image file(s). Running Real-ESRGAN/restore_and_colorize.py for each...")
        # Make sure PYTHONPATH includes DeOldify
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = f"{os.environ['PYTHONPATH']};{os.path.join(os.getcwd(), 'DeOldify')}"
        else:
            os.environ['PYTHONPATH'] = f"{os.path.join(os.getcwd(), 'DeOldary')}"
        from unified_image_processor import ImageProcessor
        processor = ImageProcessor(OUTPUTS_DIR)
        for fname in image_files:
            in_path = os.path.join(INPUTS_DIR, fname)
            print(f"[INFO] Processing image: {fname}")
            try:
                result_path = processor.process_image(
                    in_path, OUTPUTS_DIR, scale=2,
                    do_restore=not args.no_restore,
                    do_colorize=not args.no_colorize,
                    do_enhance=not args.no_enhance,
                    color_boost=int(args.color_boost)  # Ensure integer value for multiple passes
                )
                if result_path:
                    print(f"[SUCCESS] Image processed successfully: {result_path}")
                else:
                    print(f"[ERROR] Failed to process image: {fname}")
            except Exception as e:
                print(f"[ERROR] Exception while processing {fname}: {e}")
                traceback.print_exc()
            progress.update(1)
        ran = True

    progress.close()
    if not ran:
        print(f"No supported audio, video, or image files found in '{INPUTS_DIR}'.")
    else:
        print("Processing complete. Check the outputs directory.")
