import os
import subprocess
import wave
import contextlib

INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'

# Find the first video file in the inputs directory
def find_video_file():
    for fname in os.listdir(INPUTS_DIR):
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return os.path.join(INPUTS_DIR, fname)
    return None

def is_valid_wav(path):
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            return wf.getnframes() > 0
    except Exception:
        return False

def main():
    video_path = find_video_file()
    if not video_path:
        print('[ERROR] No video file found in inputs/.')
        return
    print(f'[INFO] Using video: {video_path}')
    orig_audio_path = os.path.join(OUTPUTS_DIR, 'test_extracted_audio.wav')
    enhanced_audio_path = os.path.join(OUTPUTS_DIR, 'test_enhanced_audio.wav')
    # Extract audio
    ffmpeg_cmd = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    cmd = [
        ffmpeg_cmd, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', orig_audio_path
    ]
    print(f'[INFO] Extracting audio...')
    subprocess.run(cmd, check=True)
    if not os.path.exists(orig_audio_path) or not is_valid_wav(orig_audio_path):
        print('[ERROR] Failed to extract valid audio.')
        return
    print(f'[INFO] Extracted audio saved to: {orig_audio_path}')
    # Enhance audio
    print(f'[INFO] Enhancing audio...')
    cmd = [
        os.sys.executable, 'audio_enhancer.py',
        '-i', orig_audio_path,
        '-o', enhanced_audio_path
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(enhanced_audio_path) or not is_valid_wav(enhanced_audio_path):
        print('[ERROR] Failed to enhance audio.')
        return
    print(f'[INFO] Enhanced audio saved to: {enhanced_audio_path}')
    print('\n[SUCCESS] Both original and enhanced audio are ready for listening in the outputs/ folder.')
    print('You can now listen to:')
    print(f'  - Original: {orig_audio_path}')
    print(f'  - Enhanced: {enhanced_audio_path}')
    print('Compare them to verify the enhancement pipeline.')

if __name__ == '__main__':
    main()
