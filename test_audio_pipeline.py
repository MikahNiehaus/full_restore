import os
import sys
import traceback
from pathlib import Path

def print_status(msg):
    print(f"[TEST] {msg}")

def check_imports():
    print_status("Checking required audio libraries...")
    try:
        import librosa
        import soundfile as sf
        from pydub import AudioSegment
        import noisereduce as nr
        import torchaudio
        import ffmpeg
        print_status("All required audio libraries are installed.")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False

def check_ffmpeg():
    print_status("Checking ffmpeg availability...")
    from shutil import which
    if which('ffmpeg') is None:
        print("[ERROR] ffmpeg not found on PATH. Please install ffmpeg and add it to your system PATH.")
        return False
    print_status("ffmpeg is available on PATH.")
    return True

def test_audio_enhancer_on_sample():
    print_status("Testing audio_enhancer.py on a sample file...")
    import subprocess
    # Use a short test audio file or generate one if not present
    test_audio = Path("test_audio/test.wav")
    test_audio.parent.mkdir(exist_ok=True)
    if not test_audio.exists():
        import numpy as np
        import soundfile as sf
        sr = 16000
        t = np.linspace(0, 1, sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(str(test_audio), y, sr)
        print_status(f"Generated test audio: {test_audio}")
    output_audio = test_audio.parent / "test_enhanced.wav"
    try:
        result = subprocess.run([
            sys.executable, "audio_enhancer.py",
            "-i", str(test_audio),
            "-o", str(output_audio),
            "--upsample"
        ], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if output_audio.exists():
            print_status(f"Enhanced audio file created: {output_audio}")
            return True
        else:
            print("[ERROR] Enhanced audio file was not created.")
            return False
    except Exception as e:
        print(f"[ERROR] Exception running audio_enhancer.py: {e}")
        traceback.print_exc()
        return False

def main():
    all_ok = True
    if not check_imports():
        all_ok = False
    if not check_ffmpeg():
        all_ok = False
    if not test_audio_enhancer_on_sample():
        all_ok = False
    if all_ok:
        print("\n[SUCCESS] Audio pipeline is fully set up and working!")
    else:
        print("\n[FAIL] Audio pipeline setup is incomplete or broken.")

if __name__ == "__main__":
    main()
