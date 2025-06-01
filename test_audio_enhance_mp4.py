#!/usr/bin/env python3
"""
Test script: Enhance audio in an MP4 and mux it back.
Usage:
    python test_audio_enhance_mp4.py input.mp4 [output.mp4]

This will extract, enhance, and mux the improved audio back into a new mp4 file.
"""
import sys
import os
from pathlib import Path
from audio_enhancer import process_mp4_audio

def main():
    # Always use the first video in inputs directory
    inputs_dir = Path('inputs')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    candidates = [f for f in inputs_dir.iterdir() if f.is_file() and f.suffix.lower() in video_exts]
    if not candidates:
        print("[ERROR] No video files found in 'inputs' directory.")
        return 1
    input_mp4 = str(candidates[0])
    print(f"[INFO] Using first video in inputs: {input_mp4}")
    output_mp4 = None  # Let the enhancer pick the default output name
    result = process_mp4_audio(input_mp4, output_mp4)
    if result:
        print(f"[SUCCESS] Enhanced audio muxed to: {result}")
        return 0
    else:
        print("[FAIL] Audio enhancement or muxing failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
