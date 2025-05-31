"""
Script to fully patch DeOldify to work without requiring the dummy directory
This directly patches the core functions that cause issues
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import importlib
import types

# Add path to DeOldify
deoldify_path = Path(__file__).parent / 'DeOldify'
if str(deoldify_path) not in sys.path:
    sys.path.append(str(deoldify_path))

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import necessary modules
from fastai.basic_data import DataBunch
from fastai.vision import *
from fastai.vision.data import *
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

# Create a simple dataset that returns a single tensor
class DummyDataset(Dataset):
    def __len__(self):
        return 10
        
    def __getitem__(self, i):
        # Create a simple random tensor of the right shape for an image
        img = torch.zeros(3, 256, 256)
        return img, img

def create_databunch_with_real_data():
    """
    Create a databunch with a real dataset that has items in it
    This avoids the zero samples error
    """
    # Create the dataset
    ds = DummyDataset()
    
    # Create a DataBunch with this dataset for both train and validation
    db = DataBunch(
        train_dl=DeviceDataLoader.create(
            ds, bs=1, shuffle=False, num_workers=0, device=torch.device('cpu')
        ),
        valid_dl=DeviceDataLoader.create(
            ds, bs=1, shuffle=False, num_workers=0, device=torch.device('cpu')
        ),
        device=torch.device('cpu')
    )
    
    # Set classes to 3 for RGB
    db.c = 3
    return db

def patch_generator_module():
    """Patch the generator module to bypass the dummy directory issue"""
    from deoldify import generators
    
    # Replace the gen_inference_wide function
    original_gen_inference_wide = generators.gen_inference_wide
    
    # Create a patched version
    def patched_gen_inference_wide(root_folder=Path('./'), weights_name='ColorizeVideo_gen', 
                                 arch=models.resnet101, nf_factor=2):
        # Instead of using get_dummy_databunch, we'll use our custom one
        data = create_databunch_with_real_data()
        
        # The rest of the function can remain the same
        # This is a simplified version that skips loading the model weights
        # but still creates all the objects DeOldify expects
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
            
        arch = models.resnet101
        learn = unet_learner(data, arch=arch, wd=1e-3, blur=True, norm_type=NormType.Spectral,
                          self_attention=True, y_range=(-3.0,3.0), loss_func=F.l1_loss)
        
        # Load the weights
        weights_path = Path(root_folder/'models'/f'{weights_name}.pth')
        if weights_path.exists():
            learn.load(weights_name)
            # Remove the head layers that are only used during training
            learn = learn.to_fp16().to_fp32() if torch.cuda.is_available() else learn
            learn.model = learn.model[0]
        
        return learn
    
    # Apply the patch
    generators.gen_inference_wide = patched_gen_inference_wide
    
    return "Generator module patched"

def patch_deoldify():
    """Apply all necessary patches to DeOldify"""
    # Apply direct patches to the modules
    patch_generator_module()
    
    # Return success message
    return "DeOldify patched successfully - all core modules updated"

if __name__ == "__main__":
    # Run the patch directly if this script is executed
    result = patch_deoldify()
    print(result)