import os
import subprocess
import wave
import contextlib

INPUT_WAV = 'test_audio/test_input.wav'
OUTPUT_WAV = 'test_audio/test_output.wav'

def generate_dummy_wav(path, duration_sec=1, framerate=44100):
    import numpy as np
    import struct
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.linspace(0, duration_sec, int(framerate * duration_sec), False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        for s in tone:
            wf.writeframes(struct.pack('<h', int(s * 32767)))

def is_valid_wav(path):
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            return wf.getnframes() > 0
    except Exception:
        return False

def test_audio_enhancer():
    if not os.path.exists(INPUT_WAV):
        print(f"[INFO] Generating dummy WAV at {INPUT_WAV}")
        generate_dummy_wav(INPUT_WAV)
    if os.path.exists(OUTPUT_WAV):
        os.remove(OUTPUT_WAV)
    cmd = [
        'python', 'audio_enhancer.py',
        '-i', INPUT_WAV,
        '-o', OUTPUT_WAV
    ]
    print(f"[TEST] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert os.path.exists(OUTPUT_WAV), "Output WAV was not created!"
    assert is_valid_wav(OUTPUT_WAV), "Output WAV is not valid!"
    print("[SUCCESS] Audio enhancer pipeline works!")

if __name__ == '__main__':
    test_audio_enhancer()
