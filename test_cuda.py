#!/usr/bin/env python3
"""
Script to test CUDA support for DeOldify and PyTorch
"""
import os
import sys
import time

print("[INFO] Testing CUDA support for DeOldify")

# Check if CUDA_VISIBLE_DEVICES environment variable is set
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"[WARNING] CUDA_VISIBLE_DEVICES is set to: '{os.environ['CUDA_VISIBLE_DEVICES']}'")
    print("[WARNING] This may prevent GPU usage. Empty string or '-1' disables CUDA.")
else:
    print("[INFO] CUDA_VISIBLE_DEVICES not set (good for GPU usage)")

# Import PyTorch and check CUDA availability
print("\n[INFO] Checking PyTorch CUDA support...")
import torch

print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
print(f"[INFO] CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"[INFO] cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.is_available() else 'N/A'}")
print(f"[INFO] Number of CUDA devices: {torch.cuda.device_count()}")

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"\n[INFO] CUDA Device {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Capability: {torch.cuda.get_device_capability(i)}")
        print(f"    Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")

# Run a simple test to verify GPU computation
print("\n[INFO] Running GPU computation test...")
if torch.cuda.is_available():
    # Create tensors on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    
    # Time matrix multiplication
    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    gpu_time = time.time() - start
    print(f"[INFO] GPU computation time: {gpu_time:.4f} seconds")
    
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    
    start = time.time()
    for _ in range(10):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"[INFO] CPU computation time: {cpu_time:.4f} seconds")
    print(f"[INFO] Speed ratio (CPU/GPU): {cpu_time/gpu_time:.2f}x")
    
    if cpu_time/gpu_time < 1.5:
        print("[WARNING] GPU not significantly faster than CPU. May not be utilized correctly.")
    else:
        print("[INFO] GPU computation confirmed working.")
else:
    print("[INFO] CUDA not available. Cannot run GPU computation test.")

# Check for DeOldify specific configuration
print("\n[INFO] Checking DeOldify configuration...")
try:
    sys.path.append("DeOldify")
    from deoldify import device
    from deoldify.device_id import DeviceId
    
    print("[INFO] DeOldify device module imported successfully")
    
    # Check if we can set the device to GPU
    try:
        device.set(device=DeviceId.GPU0)
        print("[INFO] Successfully set DeOldify device to GPU0")
        current_device = "GPU" if torch.cuda.current_device() == 0 else "CPU"
        print(f"[INFO] Current PyTorch device: {current_device}")
    except Exception as e:
        print(f"[ERROR] Failed to set DeOldify device to GPU: {e}")
except ImportError:
    print("[ERROR] Could not import DeOldify modules")

print("\n[INFO] CUDA test completed")
