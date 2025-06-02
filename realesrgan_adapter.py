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
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.half = not fp32 if not half else half
        self.gpu_id = gpu_id
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
        # Instantiate RealESRGANer
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
            gpu_id=gpu_id
        )

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
        return self.upsampler.enhance(
            image,
            outscale=outscale if outscale is not None else self.outscale,
            alpha_upsampler=alpha_upsampler if alpha_upsampler is not None else self.alpha_upsampler
        )
