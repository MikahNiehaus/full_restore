import os
from pathlib import Path
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

class RealESRGANAdapter:
    """
    Adapter for Real-ESRGAN super-resolution using the RealESRGANer OOP interface.
    Encapsulates all model loading and enhancement logic, supporting flexible model selection and options.
    """
    def __init__(self, model_name='RealESRGAN_x4plus', model_path=None, device=None, half=True, tile=0, tile_pad=10, pre_pad=0, outscale=2, fp32=False, gpu_id=None, alpha_upsampler='realesrgan', denoise_strength=1.0):
        self.model_name = model_name
        self.outscale = outscale
        self.alpha_upsampler = alpha_upsampler
        self.denoise_strength = denoise_strength
        
        # Handle device configuration with better logging
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"[INFO] RealESRGAN: Using GPU acceleration (CUDA device: {torch.cuda.get_device_name(0)})")
            else:
                self.device = torch.device('cpu')
                print("[WARNING] RealESRGAN: CUDA not available, using CPU (much slower)")
        else:
            self.device = device
            print(f"[INFO] RealESRGAN: Using specified device: {device}")
            
        # Handle precision (half vs full precision)
        self.half = half and self.device.type == 'cuda'  # Only use half precision with CUDA
        if half and self.device.type == 'cuda':
            print("[INFO] RealESRGAN: Using half precision (FP16) for faster processing")
        elif not half and self.device.type == 'cuda':
            print("[INFO] RealESRGAN: Using full precision (FP32) for higher accuracy")
        elif half and self.device.type == 'cpu':
            print("[INFO] RealESRGAN: Half precision not available on CPU, using full precision")
            self.half = False
            
        # Store other parameters
        self.gpu_id = gpu_id if self.device.type == 'cuda' else None
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        
        # Model selection logic (mirrors official script)
        netscale = 4
        model = None
        dni_weight = None
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            if not model_path:
                model_path = 'models/RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            if not model_path:
                model_path = 'models/RealESRNet_x4plus.pth'
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            if not model_path:
                model_path = 'models/RealESRGAN_x4plus_anime_6B.pth'
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            if not model_path:
                model_path = 'models/RealESRGAN_x2plus.pth'
        elif model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            if not model_path:
                model_path = 'models/realesr-animevideov3.pth'
        elif model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            if not model_path:
                model_path = 'models/realesr-general-x4v3.pth'
            # Denoise strength logic (for general-x4v3)
            if denoise_strength != 1:
                wdn_model_path = str(model_path).replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
                model_path = [model_path, wdn_model_path]
                dni_weight = [denoise_strength, 1 - denoise_strength]
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        # Absolute path (always use string for model_path)
        if isinstance(model_path, Path):
            model_path = str(model_path)
        elif isinstance(model_path, list):
            model_path = [str(p) if isinstance(p, Path) else p for p in model_path]
            
        # Check if model file exists
        if isinstance(model_path, list):
            for mp in model_path:
                if not os.path.isfile(mp):
                    raise FileNotFoundError(f"Model file not found: {mp}")
        else:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        # Instantiate RealESRGANer
        try:
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=self.half,
                device=self.device,
                gpu_id=self.gpu_id
            )
            print(f"[INFO] RealESRGAN model '{model_name}' loaded successfully on {self.device}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[WARNING] CUDA out of memory while loading model. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.half = False
                self.gpu_id = None
                self.upsampler = RealESRGANer(
                    scale=netscale,
                    model_path=model_path,
                    dni_weight=dni_weight,
                    model=model,
                    tile=tile,
                    tile_pad=tile_pad,
                    pre_pad=pre_pad,
                    half=False,
                    device=self.device,
                    gpu_id=None
                )
                print(f"[INFO] RealESRGAN model '{model_name}' loaded on CPU as fallback")
            else:
                raise e

    def enhance(self, image, outscale=None, alpha_upsampler=None):
        """
        Enhance an image using Real-ESRGAN super-resolution.
        Args:
            image: Input image (numpy array)
            outscale: Output scale factor (optional, overrides default)
            alpha_upsampler: Alpha upsampler (optional, overrides default)
        Returns:
            Enhanced image (numpy array), image mode
        """
        try:
            # Try to enhance with current settings
            return self.upsampler.enhance(
                image,
                outscale=outscale if outscale is not None else self.outscale,
                alpha_upsampler=alpha_upsampler if alpha_upsampler is not None else self.alpha_upsampler
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.device.type == 'cuda':
                print(f"[WARNING] CUDA out of memory during enhancement. Trying with tiling...")
                
                # Try with tiling first
                old_tile = self.upsampler.tile
                self.upsampler.tile = 512  # Use a reasonable tile size
                
                try:
                    result = self.upsampler.enhance(
                        image,
                        outscale=outscale if outscale is not None else self.outscale,
                        alpha_upsampler=alpha_upsampler if alpha_upsampler is not None else self.alpha_upsampler
                    )
                    print("[INFO] Successfully processed with tiling")
                    return result
                except RuntimeError:
                    print(f"[WARNING] Tiling approach also failed. Falling back to CPU...")
                    
                    # Fallback to CPU
                    original_device = self.device
                    self.device = torch.device('cpu')
                    self.upsampler.device = torch.device('cpu')
                    self.upsampler.model = self.upsampler.model.to('cpu')
                    self.upsampler.half = False
                    
                    try:
                        result = self.upsampler.enhance(
                            image,
                            outscale=outscale if outscale is not None else self.outscale,
                            alpha_upsampler=alpha_upsampler if alpha_upsampler is not None else self.alpha_upsampler
                        )
                        
                        # Restore original device for future calls
                        self.device = original_device
                        self.upsampler.device = original_device
                        self.upsampler.model = self.upsampler.model.to(original_device)
                        self.upsampler.half = self.half
                        self.upsampler.tile = old_tile
                        
                        print("[INFO] Successfully processed with CPU fallback")
                        return result
                    except Exception as cpu_e:
                        print(f"[ERROR] CPU fallback also failed: {cpu_e}")
                        raise cpu_e
            else:
                # Re-raise original error if not CUDA OOM
                raise e
