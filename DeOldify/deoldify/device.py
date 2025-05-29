# filepath: deoldify/device.py - Optimized for GPU usage
import torch
from fastai.core import *
from enum import Enum
from .device_id import DeviceId

# Track the current device being used
_current_device = None

def _is_gpu_available():
    """Return True if a GPU is available."""
    try:
        return torch.cuda.is_available()
    except:
        return False

def _set_nvidia_gpu(device_id: DeviceId):
    """Set the NVIDIA GPU as active."""
    import torch
    cuda_id = int(device_id.value)
    torch.cuda.set_device(cuda_id)
    print(f'[INFO] CUDA GPU {cuda_id} set as active device')
    
    # Apply performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print(f'[INFO] CUDNN optimizations enabled')
    return torch.device(f'cuda:{cuda_id}')

def _set_cpu():
    """Set the CPU as active."""
    print('[INFO] CPU set as active device (No GPU available)')
    return torch.device('cpu')

def set(device: DeviceId = None):
    """Set the device. If no device is specified and CUDA is available,
    default to first NVIDIA GPU.
    """
    global _current_device
    
    if device is not None and device != DeviceId.CPU and _is_gpu_available():
        _current_device = _set_nvidia_gpu(device)
    else:
        if _is_gpu_available() and device is None:
            print(f'[AUTO] No device specified. Using GPU (CUDA) by default.')
            _current_device = _set_nvidia_gpu(DeviceId.GPU0)
        else:
            _current_device = _set_cpu()

def get():
    """Get the current device."""
    global _current_device
    
    if _current_device is None:
        set()
    
    return _current_device

def is_gpu():
    """Check if the current device is a GPU."""
    global _current_device
    
    if _current_device is None:
        set()
    
    return _current_device.type == 'cuda'
