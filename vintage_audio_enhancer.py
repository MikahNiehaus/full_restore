#!/usr/bin/env python3
"""
Vintage Audio Enhancer Module for DeOldify colorization pipeline

This module enhances the audio of old black and white films with a focus on:
1. Removing hiss and noise
2. Gentle equalization to improve dialog clarity
3. Mild compression to balance volume levels
4. Volume normalization

Used in conjunction with DeOldify to provide both visual and audial restoration.
"""

import os
import sys
import numpy as np
import traceback
from pathlib import Path
import logging

# Try to import soundfile for audio processing
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Try to import scipy for signal processing
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class VintageAudioEnhancer:
    """Class for enhancing audio from vintage movies and recordings"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def log(self, message):
        """Log messages if verbose mode is enabled"""
        if self.verbose:
            logging.info(message)
    
    def enhance_vintage_audio(self, input_path, output_path, minimal_processing=False):
        """
        Enhance audio from vintage recordings with period-appropriate improvements
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            minimal_processing: If True, only apply basic equalization and normalization for very short audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not SOUNDFILE_AVAILABLE or not SCIPY_AVAILABLE:
            self.log("[WARNING] Required audio libraries not available, skipping enhancement")
            self.log("[INFO] To install required libraries: pip install soundfile scipy")
            return False
        
        # Convert to Path objects if needed
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        
        # Ensure input file exists
        if not input_path.exists():
            self.log(f"[ERROR] Input audio file not found: {input_path}")
            return False
            
        # Create output directory if needed
        if not output_path.parent.exists():
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.log(f"[INFO] Created output directory: {output_path.parent}")
            except Exception as e:
                self.log(f"[ERROR] Failed to create output directory: {e}")
                return False
        
        # Make a copy of the input file as a first fallback option
        try:
            import shutil
            fallback_path = str(output_path).replace(".wav", "_fallback.wav")
            shutil.copy2(str(input_path), fallback_path)
            self.log(f"[INFO] Created fallback copy of original audio")
        except Exception as e:
            self.log(f"[WARNING] Could not create fallback copy: {e}")
            
        try:
            # Load the audio file
            self.log(f"[INFO] Loading audio file: {input_path}")
            audio_data, sample_rate = sf.read(str(input_path))
            if audio_data is None or sample_rate is None:
                self.log("[ERROR] Failed to load audio data")
                # Try to use the fallback copy
                if os.path.exists(fallback_path):
                    shutil.copy2(fallback_path, str(output_path))
                    self.log("[INFO] Using original audio as fallback")
                    return True
                return False                # Convert to mono if stereo (most vintage recordings were mono)
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)
                      # Check audio length - if it's too short, use minimal processing
                if len(audio_data) < 4096:
                    self.log("[INFO] Audio is very short, using minimal processing")
                    minimal_processing = True
                
                # Save original audio as a fallback
                original_audio = np.copy(audio_data)
                
                # 1. Apply noise reduction (mild to preserve character)
                if not minimal_processing:
                    try:
                        # Find noise profile from the first 500ms where usually there's no speech
                        noise_len = min(int(sample_rate * 0.5), len(audio_data))
                        noise_profile = audio_data[:noise_len]
                        noise_power = np.mean(noise_profile**2)
                        
                        # Apply spectral subtraction for noise reduction
                        if noise_power > 0:
                            # Split into frames
                            frame_size = min(2048, len(audio_data) // 4)  # Adaptive frame size
                            if frame_size < 256:  # Too short for effective processing
                                frame_size = len(audio_data)
                                
                            hop_size = frame_size // 2
                            
                            # Create output array
                            output = np.zeros_like(audio_data)
                            
                            for i in range(0, max(1, len(audio_data) - frame_size), hop_size):
                                # Get frame
                                frame = audio_data[i:i+frame_size]
                                
                                # Compute FFT
                                spectrum = np.fft.rfft(frame)
                                power = np.abs(spectrum)**2
                                
                                # Apply spectral subtraction
                                power = np.maximum(power - noise_power * 0.6, 0.01 * power)
                                
                                # Apply to spectrum and inverse FFT
                                enhanced_spectrum = spectrum * np.sqrt(power / (np.abs(spectrum)**2 + 1e-10))
                                enhanced_frame = np.fft.irfft(enhanced_spectrum)
                                
                                # Overlap-add
                                output[i:i+frame_size] += enhanced_frame * np.hanning(frame_size)
                            
                            # Normalize output
                            audio_data = output / (np.max(np.abs(output)) + 1e-10) * np.max(np.abs(audio_data) + 1e-10)
                    except Exception as e:
                        self.log(f"[WARNING] Noise reduction failed: {e}, using original audio")
                        audio_data = original_audio
                    
                    # 2. Apply vintage-appropriate EQ (emphasize midrange for dialog clarity)
                    # Design a bandpass filter that emphasizes speech frequencies
                    if SCIPY_AVAILABLE:
                        try:
                            # Adaptive filter order based on audio length
                            if len(audio_data) < 1000:
                                filter_order = 1
                            elif len(audio_data) < 5000:
                                filter_order = 2
                            else:
                                filter_order = 3
                                
                            # Create bandpass filter for speech (300Hz - 3000Hz)
                            b1, a1 = signal.butter(filter_order, [300/(sample_rate/2), 3000/(sample_rate/2)], 'bandpass')
                            speech_enhanced = signal.filtfilt(b1, a1, audio_data)
                            
                            # Mix with original to keep some character
                            audio_data = audio_data * 0.3 + speech_enhanced * 0.7
                            
                            # Gentle high cut to reduce harshness (common in old recordings)
                            b2, a2 = signal.butter(filter_order, 7000/(sample_rate/2), 'lowpass')
                            audio_data = signal.filtfilt(b2, a2, audio_data)
                        except Exception as e:
                            self.log(f"[WARNING] EQ processing failed: {e}, falling back to original audio")
                            audio_data = original_audio
            
            # 3. Apply gentle compression
            # Simple compressor: soft knee, ratio 2:1 above -20dB
            threshold = 10 ** (-20 / 20)
            ratio = 2.0
            def soft_compressor(x, threshold, ratio):
                absx = np.abs(x)
                over = absx > threshold
                x[over] = np.sign(x[over]) * (threshold + (absx[over] - threshold) / ratio)
                return x
            
            audio_data = soft_compressor(audio_data, threshold, ratio)
            
            # 4. Normalize volume
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
              # Save enhanced audio
            try:
                sf.write(str(output_path), audio_data, sample_rate)
                self.log(f"[INFO] Enhanced vintage audio saved to: {output_path}")
                
                # Validate the file was created successfully
                if output_path.exists() and output_path.stat().st_size > 0:
                    self.log(f"[INFO] Verified enhanced audio file: {output_path.stat().st_size} bytes")
                    return True
                else:
                    self.log(f"[ERROR] Enhanced audio file was not created or is empty")
                    return False
                    
            except Exception as e:
                self.log(f"[ERROR] Failed to save enhanced audio: {e}")
                return False
            
        except Exception as e:
            self.log(f"[ERROR] Vintage audio enhancement failed: {e}")
            traceback.print_exc()
            return False

def enhance_audio_from_video(video_path, output_audio_path=None, temp_audio_path=None, create_output_dir=True):
    """
    Extract and enhance audio from a video file
    
    Args:
        video_path: Path to video file
        output_audio_path: Path to save enhanced audio (if None, will use video_stem + '_enhanced.wav')
        temp_audio_path: Path to save temporary audio (if None, will use 'temp_audio.wav')
        create_output_dir: Whether to create output directory if it doesn't exist
        
    Returns:
        str: Path to enhanced audio file, or None if failed
    """
    # Convert string paths to Path objects if needed
    video_path = Path(video_path) if isinstance(video_path, str) else video_path
    
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return None
        
    # Set default paths if not provided
    if temp_audio_path is None:
        temp_audio_path = Path("temp_audio.wav")
    else:
        temp_audio_path = Path(temp_audio_path) if isinstance(temp_audio_path, str) else temp_audio_path
    
    if output_audio_path is None:
        output_audio_path = Path(f"{video_path.stem}_enhanced.wav")
    else:
        output_audio_path = Path(output_audio_path) if isinstance(output_audio_path, str) else output_audio_path
    
    # Create output directory if needed
    if create_output_dir and output_audio_path.parent != Path('.'):
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Extract audio from video
    try:
        logging.info(f"Extracting audio from: {video_path}")
        cmd = f'ffmpeg -y -i "{str(video_path)}" -q:a 0 -vn -acodec pcm_s16le "{str(temp_audio_path)}" -hide_banner -loglevel error'
        logging.info(f"Running command: {cmd}")
        result = os.system(cmd)
        
        if result != 0 or not temp_audio_path.exists():
            logging.warning("FFmpeg audio extraction failed")
            logging.warning("Check if ffmpeg is installed and working properly")
            return None
            
        # Enhance the audio
        enhancer = VintageAudioEnhancer()
        success = enhancer.enhance_vintage_audio(str(temp_audio_path), str(output_audio_path))
        
        if success and output_audio_path.exists():
            logging.info(f"Enhanced audio saved to: {output_audio_path}")
            return str(output_audio_path)
        else:
            logging.warning("Audio enhancement failed, using original audio")
            return str(temp_audio_path) if temp_audio_path.exists() else None
            
    except Exception as e:
        logging.error(f"Audio processing failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        enhanced_audio = enhance_audio_from_video(video_path)
        if enhanced_audio:
            print(f"Enhanced audio saved to: {enhanced_audio}")
    else:
        print("Usage: python vintage_audio_enhancer.py <video_file>")
