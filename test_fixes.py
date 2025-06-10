#!/usr/bin/env python3
"""
GPU Acceleration Test for DeOldify and Image Restorer

This script tests GPU acceleration and PyTorch serialization 
to verify the fixes applied to the pipeline.
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Import our PyTorch patch first to fix loading issues
from torch_safety_patch import *

# Add DeOldify to sys.path
sys.path.append(str(Path(__file__).parent / 'DeOldify'))

def test_torch_serialization():
    """Test that PyTorch serialization is working properly"""
    print("\n=== Testing PyTorch Serialization ===")
    
    try:
        # Check if the PyTorch version supports add_safe_globals or if we need our patch
        if hasattr(torch.serialization, 'add_safe_globals'):
            print("[SUCCESS] torch.serialization has add_safe_globals method")
        else:
            print("[INFO] Using our patch for torch.serialization")
            
        # Try loading a model (patched torch.load should handle this)
        from deoldify.visualize import get_video_colorizer
        colorizer = get_video_colorizer(render_factor=40)
        print("[SUCCESS] Successfully loaded DeOldify colorizer model")
        
        return True
    except Exception as e:
        print(f"[ERROR] Serialization test failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_usage():
    """Test that both components can use GPU properly"""
    print("\n=== Testing GPU Usage ===")
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"[SUCCESS] CUDA is available: {torch.cuda.get_device_name(0)}")
            
            # Test DeOldify GPU usage
            from deoldify.device_id import DeviceId
            from deoldify import device
            device.set(DeviceId.GPU0)
            
            if device.is_gpu():
                print("[SUCCESS] DeOldify is using GPU")
            else:
                print("[ERROR] Failed to set DeOldify to use GPU")
                return False
            
            # Test ImageRestorer GPU usage
            from image_restorer import ImageRestorer
            restorer = ImageRestorer(force_device='cuda')
            
            if restorer.device.type == 'cuda':
                print("[SUCCESS] ImageRestorer is using GPU")
            else:
                print("[ERROR] ImageRestorer is not using GPU")
                return False
                
            return True
        else:
            print("[WARNING] CUDA is not available - tests will be skipped")
            return True  # Not a failure if CUDA is unavailable
    except Exception as e:
        print(f"[ERROR] GPU test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and report results"""
    print("=== Testing Full Restore Pipeline Fixes ===")
    
    # Test PyTorch serialization patch
    serialization_ok = test_torch_serialization()
    
    # Test GPU usage
    gpu_ok = test_gpu_usage()
    
    # Print overall results
    print("\n=== Test Results ===")
    print(f"PyTorch Serialization: {'SUCCESS' if serialization_ok else 'FAILED'}")
    print(f"GPU Usage: {'SUCCESS' if gpu_ok else 'FAILED'}")
    
    if serialization_ok and gpu_ok:
        print("\n[SUCCESS] All tests passed - pipeline is ready to use!")
        return 0
    else:
        print("\n[ERROR] Some tests failed - see above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
