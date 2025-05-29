import os
import sys
import subprocess
import wave
import contextlib

def test_audio_enhancer():
    print("[TEST] Checking audio_enhancer.py pipeline...")
    # Generate a test WAV file
    test_wav = 'test_audio.wav'
    duration = 1  # seconds
    freq = 440  # Hz
    try:
        import numpy as np
        import soundfile as sf
        samplerate = 16000
        t = np.linspace(0, duration, int(samplerate * duration), False)
        note = 0.5 * np.sin(2 * np.pi * freq * t)
        sf.write(test_wav, note, samplerate)
    except Exception as e:
        print(f"[ERROR] Could not generate test audio: {e}")
        return False

    # Run audio_enhancer.py
    enhanced_wav = 'test_audio_enhanced.wav'
    cmd = [sys.executable, 'audio_enhancer.py', '-i', test_wav, '-o', enhanced_wav]
    print(f"[TEST] Running: {' '.join(cmd)})")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"[ERROR] audio_enhancer.py failed: {result.stderr.decode()}")
        return False
    if not os.path.exists(enhanced_wav):
        print(f"[ERROR] Enhanced audio file not created.")
        return False
    # Check duration
    with contextlib.closing(wave.open(enhanced_wav, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration_out = frames / float(rate)
        if abs(duration_out - duration) > 0.1:
            print(f"[ERROR] Output audio duration mismatch: {duration_out}s")
            return False
    print("[TEST] Audio enhancement pipeline works!")
    os.remove(test_wav)
    os.remove(enhanced_wav)
    return True

if __name__ == '__main__':
    ok = test_audio_enhancer()
    sys.exit(0 if ok else 1)
