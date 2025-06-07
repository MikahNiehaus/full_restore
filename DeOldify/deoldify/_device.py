import os
from enum import Enum
from .device_id import DeviceId

#NOTE:  This must be called first before any torch imports in order to work properly!

class DeviceException(Exception):
    pass

class _Device:
    def __init__(self):
        self.set(DeviceId.CPU)

    def is_gpu(self):
        ''' Returns `True` if the current device is GPU, `False` otherwise. '''
        return self.current() is not DeviceId.CPU
  
    def current(self):
        return self._current_device

    def set(self, device):     
        import torch
        if isinstance(device, DeviceId):
            # Original code path using DeviceId enum
            if device == DeviceId.CPU:
                os.environ['CUDA_VISIBLE_DEVICES']=''
            else:
                os.environ['CUDA_VISIBLE_DEVICES']=str(device.value)
                torch.backends.cudnn.benchmark=False
            self._current_device = device
        elif isinstance(device, torch.device):
            # New code path for torch.device objects
            if device.type == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES']=''
                self._current_device = DeviceId.CPU
            else:
                # For CUDA devices, set the specific index if available
                if hasattr(device, 'index') and device.index is not None:
                    os.environ['CUDA_VISIBLE_DEVICES']=str(device.index)
                else:
                    # Default to first GPU if no index specified
                    os.environ['CUDA_VISIBLE_DEVICES']='0'
                torch.backends.cudnn.benchmark=False
                # Map to closest DeviceId enum
                self._current_device = DeviceId.GPU
        else:
            raise ValueError(f"Unsupported device type: {type(device)}")
            
        return self._current_device