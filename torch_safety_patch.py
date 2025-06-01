"""
Patch to fix PyTorch 2.6+ loading issues with DeOldify models.

This adds the necessary globals to PyTorch's safe list to allow loading
DeOldify models with weights_only=True (the new default in PyTorch 2.6+).
"""

import torch
import torch.serialization
import builtins

# Add the necessary globals to PyTorch's safe list (if available)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice, getattr(builtins, 'slice', slice)])

# Also add a monkey patch for torch.load to always use weights_only=False for DeOldify
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    """Patched version of torch.load that forces weights_only=False for DeOldify models"""
    # Ensure pickle_module is not None, default to built-in pickle if necessary
    if pickle_module is None:
        import pickle
        pickle_module = pickle

    # Remove unsupported arguments like 'weights_only'
    if 'weights_only' in pickle_load_args:
        del pickle_load_args['weights_only']

    return original_torch_load(f, map_location, pickle_module, **pickle_load_args)

# Replace torch.load with our patched version
torch.load = patched_torch_load

print("[INFO] PyTorch serialization patched to allow loading DeOldify models")
