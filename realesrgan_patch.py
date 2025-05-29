"""
Patch for Real-ESRGAN imports using newer versions of torchvision.
This creates a compatibility layer for the missing or moved functions.
"""
import os
import sys
import importlib.util
import types

# Check if torchvision is installed
try:
    import torchvision
    torchvision_version = torchvision.__version__
    print(f"[INFO] Found torchvision version {torchvision_version}")
except ImportError:
    print("[ERROR] torchvision not found. Please install it first.")
    sys.exit(1)

# Create a missing module if needed
if not hasattr(torchvision.transforms, 'functional_tensor'):
    # Create a new module
    functional_tensor_module = types.ModuleType('functional_tensor')
    
    # Add missing functions by importing them from new locations
    try:
        # In newer torchvision, rgb_to_grayscale is in transforms.functional
        from torchvision.transforms.functional import rgb_to_grayscale as new_rgb_to_grayscale
        functional_tensor_module.rgb_to_grayscale = new_rgb_to_grayscale
        print("[INFO] Successfully patched rgb_to_grayscale from functional")
    except ImportError:
        try:
            # It might be in _functional_tensor in some versions
            from torchvision.transforms._functional_tensor import rgb_to_grayscale as new_rgb_to_grayscale
            functional_tensor_module.rgb_to_grayscale = new_rgb_to_grayscale
            print("[INFO] Successfully patched rgb_to_grayscale from _functional_tensor")
        except ImportError:
            print("[WARNING] Could not find rgb_to_grayscale in any expected location")
    
    # Inject our module into torchvision.transforms
    torchvision.transforms.functional_tensor = functional_tensor_module
    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_module
    print("[INFO] Created compatibility module 'torchvision.transforms.functional_tensor'")

# Try to import RealESRGANer after patching - just to verify the patch works
try:
    from realesrgan import RealESRGANer
    print("[SUCCESS] RealESRGANer imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import RealESRGANer: {e}")
    import traceback
    traceback.print_exc()
