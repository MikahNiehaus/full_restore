#!/usr/bin/env python3
"""
Audio Enhancer for DeOldify Video Restoration Pipeline

This module provides comprehensive audio processing for old videos:
1. Audio extraction from video files
2. Audio enhancement (noise reduction, normalization)
3. Audio restoration (EQ, compression)
4. Audio synchronization with processed videos

Usage:
    python audio_enhancer.py -i input.wav -o output.wav
"""

import os
import sys
import argparse
import wave
import numpy as np
import traceback
import soundfile as sf
from scipy import signal
from pathlib import Path

class AudioEnhancer:
    """
    Class for enhancing and restoring old audio recordings
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message):
        """Log messages if verbose mode is enabled"""
        if self.verbose:
            print(message)
            
    def load_audio(self, input_path):
        """
        Load audio file using soundfile
        
        Args:
            input_path (str): Path to input audio file
            
        Returns:
            tuple: audio data as numpy array, sample rate
        """
        try:
            self.log(f"[INFO] Loading audio file: {input_path}")
            audio_data, sample_rate = sf.read(input_path)
            self.log(f"[INFO] Audio loaded: {len(audio_data)} samples, {sample_rate}Hz")
            return audio_data, sample_rate
        except Exception as e:
            print(f"[ERROR] Failed to load audio: {e}")
            traceback.print_exc()
            return None, None
            
    def save_audio(self, output_path, audio_data, sample_rate):
        """
        Save audio data to file
        
        Args:
            output_path (str): Path to save audio file
            audio_data (numpy.ndarray): Audio data array
            sample_rate (int): Sample rate in Hz
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.log(f"[INFO] Saving audio to: {output_path}")
            sf.write(output_path, audio_data, sample_rate)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save audio: {e}")
            traceback.print_exc()
            return False
            
    def reduce_noise(self, audio_data, sample_rate, reduction_strength=0.8):
        """
        Apply noise reduction to audio
        
        Args:
            audio_data (numpy.ndarray): Input audio data
            sample_rate (int): Sample rate in Hz
            reduction_strength (float): Strength of noise reduction (0.0-1.0)
            
        Returns:
            numpy.ndarray: Noise-reduced audio data
        """
        self.log("[INFO] Applying noise reduction...")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            working_data = np.mean(audio_data, axis=1)
        else:
            working_data = audio_data.copy()
        
        # Find noise profile from the first 1000ms of audio
        noise_len = min(int(sample_rate), len(working_data))
        noise_sample = working_data[:noise_len]
        
        # Compute noise power spectral density
        nperseg = min(512, noise_len)
        noise_psd = signal.welch(noise_sample, sample_rate, nperseg=nperseg)[1]
        
        # Apply spectral subtraction
        denoised_audio = working_data.copy()
        
        # Process in chunks to avoid memory issues
        chunk_size = 8192
        for i in range(0, len(working_data), chunk_size):
            chunk = working_data[i:i+chunk_size]
            
            # Short-time Fourier transform
            f, t, stft_data = signal.stft(chunk, sample_rate, nperseg=nperseg)
            
            # Spectral subtraction
            for j in range(stft_data.shape[1]):
                stft_data[:, j] = stft_data[:, j] * (1 - reduction_strength * noise_psd / (np.abs(stft_data[:, j])**2 + 1e-10))
            
            # Inverse STFT
            _, chunk_denoised = signal.istft(stft_data, sample_rate, nperseg=nperseg)
            
            # Copy denoised chunk
            end_idx = min(i + len(chunk_denoised), len(denoised_audio))
            denoised_audio[i:end_idx] = chunk_denoised[:end_idx-i]
        
        # Convert back to original format (mono/stereo)
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            result = np.repeat(denoised_audio[:, np.newaxis], audio_data.shape[1], axis=1)
        else:
            result = denoised_audio
            
        return result
    
    def normalize_audio(self, audio_data, target_level=-3.0):
        """
        Normalize audio level
        
        Args:
            audio_data (numpy.ndarray): Input audio data
            target_level (float): Target peak level in dB
            
        Returns:
            numpy.ndarray: Normalized audio data
        """
        self.log(f"[INFO] Normalizing audio to {target_level}dB...")
        
        # Calculate current peak
        current_peak = np.max(np.abs(audio_data))
        
        # Calculate target peak (converting from dB)
        target_peak = 10 ** (target_level / 20)
        
        # Apply gain if needed
        if current_peak > 0:
            gain = target_peak / current_peak
            normalized_audio = audio_data * gain
        else:
            normalized_audio = audio_data
            
        return normalized_audio
    
    def apply_equalization(self, audio_data, sample_rate):
        """
        Apply EQ to enhance speech frequencies and reduce unwanted frequencies
        
        Args:
            audio_data (numpy.ndarray): Input audio data
            sample_rate (int): Sample rate in Hz
            
        Returns:
            numpy.ndarray: Equalized audio data
        """
        self.log("[INFO] Applying equalization...")

        # Create a bandpass filter to enhance speech frequencies (300Hz-3400Hz)
        nyquist = sample_rate / 2
        low = 250 / nyquist
        high = 3800 / nyquist
        
        try:
            # First try standard 4th order filter
            b, a = signal.butter(4, [low, high], btype='band')
            # Apply filter
            equalized_audio = signal.filtfilt(b, a, audio_data)
        except ValueError as e:
            # If we get an error about input vector length, try a lower order filter
            if "The length of the input vector x must be greater than padlen" in str(e):
                self.log("[INFO] Audio too short for standard filter, using lower order filter...")
                try:
                    # Try a 2nd order filter with less padding requirements
                    b, a = signal.butter(2, [low, high], btype='band')
                    equalized_audio = signal.filtfilt(b, a, audio_data)
                except ValueError:
                    # If that still fails, just return the original audio
                    self.log("[INFO] Audio too short for EQ filters, returning original audio...")
                    return audio_data
            else:
                # If it's a different error, re-raise it
                raise
        
        # Try to apply high frequency boost if audio is long enough
        try:
            # Slightly boost high frequencies for clarity
            high_boost = 3000 / nyquist
            b_high, a_high = signal.butter(4, high_boost, btype='high')
            high_boosted = signal.filtfilt(b_high, a_high, equalized_audio) * 0.3
            
            # Mix with original signal
            equalized_audio = equalized_audio + high_boosted
        except ValueError:
            # If high boost fails, just use the bandpass filtered audio
            self.log("[INFO] Skipping high frequency boost for short audio...")
            pass
        
        return equalized_audio
    
    def enhance_audio(self, input_path, output_path):
        """
        Perform full audio enhancement process
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to save enhanced audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load audio
            audio_data, sample_rate = self.load_audio(input_path)
            if audio_data is None:
                return False
                
            # Check if audio is too short (less than 100ms)
            if len(audio_data) < sample_rate / 10:
                self.log("[WARNING] Audio is very short, may have limited enhancement effects")
            
            try:
                # Step 1: Noise reduction
                audio_data = self.reduce_noise(audio_data, sample_rate)
            except Exception as e:
                self.log(f"[WARNING] Noise reduction failed: {e}, using original audio")
                # Reload the original audio data
                audio_data, sample_rate = self.load_audio(input_path)
            
            try:
                # Step 2: Equalization for better voice/music clarity
                audio_data = self.apply_equalization(audio_data, sample_rate)
            except Exception as e:
                self.log(f"[WARNING] Equalization failed: {e}, using previously processed audio")
            
            try:
                # Step 3: Normalize audio levels
                audio_data = self.normalize_audio(audio_data)
            except Exception as e:
                self.log(f"[WARNING] Normalization failed: {e}, using previously processed audio")
            
            # Save enhanced audio
            return self.save_audio(output_path, audio_data, sample_rate)
            
        except Exception as e:
            print(f"[ERROR] Audio enhancement failed: {e}")
            traceback.print_exc()
            # If enhancement completely fails, try to just copy the original file
            try:
                self.log("[INFO] Attempting to copy original audio as fallback...")
                import shutil
                shutil.copy(input_path, output_path)
                return os.path.exists(output_path)
            except:
                return False

def extract_audio(video_path, output_path):
    """
    Extract audio from video file using FFmpeg
    
    Args:
        video_path (str): Path to video file
        output_path (str): Path to save extracted audio
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"[INFO] Extracting audio from: {video_path}")
        cmd = f'ffmpeg -y -i "{video_path}" -q:a 0 -vn "{output_path}" -hide_banner -loglevel error'
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(output_path):
            print(f"[INFO] Audio extracted to: {output_path}")
            return True
        else:
            print("[WARNING] FFmpeg extraction failed")
            return False
    except Exception as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        traceback.print_exc()
        return False

def mux_audio_to_video(video_path, audio_path, output_path):
    """
    Add audio track to video file
    
    Args:
        video_path (str): Path to video file
        audio_path (str): Path to audio file
        output_path (str): Path to output video file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"[INFO] Adding audio to video: {output_path}")
        cmd = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{output_path}" -hide_banner -loglevel error'
        result = os.system(cmd)
        
        if result == 0 and os.path.exists(output_path):
            print(f"[INFO] Video with audio saved to: {output_path}")
            return True
        else:
            print("[WARNING] FFmpeg muxing failed")
            return False
    except Exception as e:
        print(f"[ERROR] Audio muxing failed: {e}")
        traceback.print_exc()
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio enhancer for old video/audio restoration")
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output enhanced audio file")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Enhance audio
    enhancer = AudioEnhancer()
    success = enhancer.enhance_audio(args.input, args.output)
    
    if success:
        print(f"[INFO] Audio enhancement complete: {args.output}")
        return 0
    else:
        print(f"[ERROR] Audio enhancement failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
