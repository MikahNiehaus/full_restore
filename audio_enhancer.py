import os
import sys
import traceback
from pathlib import Path

# --- Robust Audio Enhancer Template ---
# Supports: audio or video input, denoising, upsampling, enhancement, and output
# Handles missing dependencies and provides clear errors

def ensure_ffmpeg_on_path():
    """Warn if ffmpeg is not on PATH."""
    from shutil import which
    if which('ffmpeg') is None:
        print("[WARNING] ffmpeg not found on PATH. Please install ffmpeg and add it to your system PATH.")

try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    import noisereduce as nr
    import torchaudio
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}. Please install all requirements.")
    sys.exit(1)

ensure_ffmpeg_on_path()

def enhance_audio(input_path, output_path=None, denoise=True, upsample=False):
    """
    Enhance audio from input_path and save to output_path.
    Supports denoising and upsampling. Handles audio or video input.
    """
    try:
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced.wav"
        # Extract audio if input is video
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            from ffmpeg import input as ffmpeg_input, output as ffmpeg_output, run as ffmpeg_run
            audio_temp = input_path.parent / f"{input_path.stem}_temp_audio.wav"
            ffmpeg_run(ffmpeg_output(ffmpeg_input(str(input_path)), str(audio_temp), acodec='pcm_s16le', ac=1, ar='16000'))
            audio_path = audio_temp
        else:
            audio_path = input_path
        # Load audio (preserve stereo if present)
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)
        import numpy as np
        # If stereo, average to mono for processing clarity
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        # Gentle but effective denoising (prop_decrease=0.7, freq_mask_smooth_hz=150)
        if denoise:
            y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7, freq_mask_smooth_hz=150)
        # Upsample (if requested)
        if upsample and sr < 32000:
            y = librosa.resample(y, orig_sr=sr, target_sr=32000)
            sr = 32000
        import scipy.signal
        # --- Highpass filter to remove rumble (cutoff 70 Hz) ---
        def highpass_filter(y, sr, cutoff=70.0, order=2):
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = scipy.signal.butter(order, normal_cutoff, btype='high')
            return scipy.signal.lfilter(b, a, y)
        y = highpass_filter(y, sr)
        # --- Subtle high-shelf EQ for clarity (gain 2 dB at 3500 Hz) ---
        def gentle_highshelf(y, sr, gain_db=2.0, freq=3500.0):
            nyq = 0.5 * sr
            freq = freq / nyq
            b, a = scipy.signal.iirfilter(
                N=2, Wn=freq, btype='high', ftype='butter')
            y_filt = scipy.signal.lfilter(b, a, y)
            gain = 10**(gain_db/20)
            y_boosted = y + (y_filt * (gain - 1) * 0.18)
            return y_boosted
        y = gentle_highshelf(y, sr)
        # --- De-esser to reduce sibilance/ringing (attenuate above 6kHz, more aggressive) ---
        def strong_de_esser(y, sr, threshold=0.12, freq=6000.0, reduction=0.7):
            nyq = 0.5 * sr
            freq = freq / nyq
            b, a = scipy.signal.butter(2, freq, btype='high')
            sibilant = scipy.signal.lfilter(b, a, y)
            mask = np.abs(sibilant) > threshold
            y[mask] = y[mask] - reduction * sibilant[mask]
            return y
        y = strong_de_esser(y, sr)
        # --- Strong multi-stage de-esser and lowpass to fully remove high ringing ---
        def multi_stage_de_esser_and_lowpass(y, sr, de_ess_freq=6000.0, de_ess_thresh=0.10, de_ess_reduction=0.8, lowpass_freq=7500.0):
            nyq = 0.5 * sr
            # Stage 1: Strong de-esser
            freq = de_ess_freq / nyq
            b, a = scipy.signal.butter(2, freq, btype='high')
            sibilant = scipy.signal.lfilter(b, a, y)
            mask = np.abs(sibilant) > de_ess_thresh
            y[mask] = y[mask] - de_ess_reduction * sibilant[mask]
            # Stage 2: Gentle lowpass filter to remove remaining high frequencies
            lp_freq = lowpass_freq / nyq
            b_lp, a_lp = scipy.signal.butter(2, lp_freq, btype='low')
            y = scipy.signal.lfilter(b_lp, a_lp, y)
            return y
        y = multi_stage_de_esser_and_lowpass(y, sr)
        # --- Optional: Dynamic range compression for speech intelligibility ---
        def soft_compress(y, threshold=0.18, ratio=3.0):
            return np.sign(y) * np.where(np.abs(y) < threshold, np.abs(y), threshold + (np.abs(y) - threshold) / ratio)
        y = soft_compress(y)
        # --- Loudness normalization (RMS) ---
        rms = np.sqrt(np.mean(y**2))
        target_rms = 0.14  # normalize a bit more aggressively for clarity
        if rms > 0:
            y = y * (target_rms / rms)
        # --- Brickwall limiter at 0.9 to prevent any peaks/ringing ---
        y = np.clip(y, -0.9, 0.9)
        # Save enhanced audio
        sf.write(str(output_path), y, sr)
        print(f"Enhanced audio saved to: {output_path}")
        # Clean up temp audio
        if 'audio_temp' in locals() and audio_path.exists():
            audio_path.unlink()
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Audio enhancement failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhance audio from audio or video files.")
    parser.add_argument("-i", "--input", required=True, help="Input audio or video file")
    parser.add_argument("-o", "--output", help="Output audio file (default: *_enhanced.wav)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--upsample", action="store_true", help="Upsample audio to 32kHz if lower")
    args = parser.parse_args()
    enhance_audio(args.input, args.output, denoise=not args.no_denoise, upsample=args.upsample)
