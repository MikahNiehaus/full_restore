#!/usr/bin/env python3
"""
Script to diagnose NVIDIA and CUDA setup issues
"""
import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return the output"""
    try:
        result = subprocess.run(command, shell=True, check=False, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
        return result.stdout
    except Exception as e:
        return f"Error running command: {e}"

print("=== NVIDIA & CUDA Diagnosis Tool ===")
print(f"OS: {platform.platform()}")
print(f"Python: {platform.python_version()}")

# Check for NVIDIA drivers
print("\n=== NVIDIA Driver Info ===")
nvidia_smi = run_command("nvidia-smi")
if "NVIDIA-SMI" in nvidia_smi:
    print(nvidia_smi)
else:
    print("NVIDIA drivers not found or not working correctly.")
    print("Make sure NVIDIA drivers are installed and working.")

# Check CUDA installation
print("\n=== CUDA Installation ===")
nvcc_version = run_command("nvcc --version")
if "release" in nvcc_version:
    print(nvcc_version)
else:
    print("NVCC (CUDA compiler) not found in PATH or not installed.")
    
# Check for CUDA environment variables
print("\n=== CUDA Environment Variables ===")
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home:
    print(f"CUDA_HOME/CUDA_PATH: {cuda_home}")
else:
    print("CUDA_HOME/CUDA_PATH not set")

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible_devices is not None:
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
else:
    print("CUDA_VISIBLE_DEVICES not set (all GPUs available)")

# Check environment PATH for CUDA binaries
system_path = os.environ.get("PATH", "")
print("\n=== CUDA in PATH ===")
cuda_in_path = False
for path in system_path.split(os.pathsep):
    if "cuda" in path.lower():
        print(f"Found CUDA in PATH: {path}")
        cuda_in_path = True
if not cuda_in_path:
    print("No CUDA directories found in PATH")

# Try to locate CUDA installation directories
print("\n=== Looking for CUDA Installation ===")
common_cuda_paths = [
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    "C:\\Program Files\\NVIDIA\\CUDA",
    "/usr/local/cuda",
    "/opt/cuda"
]

for cuda_path in common_cuda_paths:
    if os.path.exists(cuda_path):
        print(f"Found potential CUDA installation: {cuda_path}")
        versions = [d for d in os.listdir(cuda_path) if os.path.isdir(os.path.join(cuda_path, d))]
        if versions:
            print(f"  Versions found: {', '.join(versions)}")

print("\n=== Recommendations ===")
if "NVIDIA-SMI" not in nvidia_smi:
    print("1. Install or update NVIDIA drivers from https://www.nvidia.com/Download/index.aspx")
if "release" not in nvcc_version:
    print("2. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
if not cuda_in_path:
    print("3. Add CUDA bin directory to your PATH environment variable")
if not cuda_home:
    print("4. Set CUDA_HOME environment variable to your CUDA installation directory")

print("\nAfter fixing these issues, try installing PyTorch with CUDA support again:")
