#!/usr/bin/env python3
"""
Script to reinstall PyTorch with CUDA support for DeOldify Full Restore
"""
import os
import sys
import subprocess
import platform

def get_cuda_version():
    """Try to detect the most appropriate CUDA version to use"""
    try:
        # Try to run nvidia-smi to get CUDA version
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and 'CUDA Version:' in result.stdout:
            # Extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    major, minor = cuda_version.split('.')[:2]
                    return f"{major}.{minor}"
        return None
    except Exception:
        return None

def main():
    print("PyTorch CUDA Installation Helper")
    print("===============================")
    
    # Check current PyTorch installation
    try:
        import torch
        current_version = torch.__version__
        has_cuda = torch.cuda.is_available()
        
        print(f"\nCurrent PyTorch version: {current_version}")
        print(f"CUDA available: {has_cuda}")
        
        if has_cuda:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
            
            print("\n[INFO] You already have a CUDA-enabled PyTorch installation!")
            choice = input("Do you still want to reinstall? (y/N): ").lower()
            if choice != 'y':
                print("Exiting without changes.")
                return
    except ImportError:
        print("\n[INFO] PyTorch is not currently installed.")
    
    # Determine best CUDA version
    detected_cuda = get_cuda_version()
    print("\nDetecting appropriate CUDA version...")
    
    if detected_cuda:
        print(f"[INFO] Detected CUDA version: {detected_cuda}")
        if float(detected_cuda) >= 12.0:
            recommended = "cu121"
        elif float(detected_cuda) >= 11.8:
            recommended = "cu118"
        elif float(detected_cuda) >= 11.7:
            recommended = "cu117"
        else:
            recommended = "cu117"  # Fallback for older CUDA versions
    else:
        print("[INFO] Unable to detect CUDA version.")
        recommended = "cu121"  # Default to latest
    
    # Provide CUDA version options
    print("\nSelect PyTorch CUDA version:")
    print("1. CUDA 12.1 (newest, for recent GPUs)")
    print("2. CUDA 11.8 (more compatible with older GPUs)")
    print("3. CUDA 11.7 (for legacy support)")
    print("4. CPU-only (no GPU acceleration)")
    
    choice = input(f"\nEnter selection [1-4] (default: {1 if recommended == 'cu121' else 2 if recommended == 'cu118' else 3}): ")
    
    # Set appropriate PyTorch installation command
    if choice == "4":
        install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        version_str = "CPU-only"
    elif choice == "3":
        install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
        version_str = "CUDA 11.7"
    elif choice == "2":
        install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        version_str = "CUDA 11.8"
    else:  # Default to CUDA 12.1
        install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        version_str = "CUDA 12.1"
    
    # Uninstall existing PyTorch
    print(f"\n[INFO] Uninstalling existing PyTorch packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    
    # Install PyTorch with selected CUDA version
    print(f"\n[INFO] Installing PyTorch with {version_str} support...")
    subprocess.run(install_cmd, shell=True)
    
    # Verify installation
    print("\n[INFO] Verifying installation...")
    try:
        import torch
        print(f"Installed PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
    except ImportError:
        print("[ERROR] Failed to import the newly installed PyTorch.")

    print("\n[INFO] Installation process completed.")
    print("If GPU mode is still not working, please verify your NVIDIA drivers are installed correctly.")
    print("Run 'python test_cuda.py' to check your CUDA configuration.")

if __name__ == "__main__":
    main()
