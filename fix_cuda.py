#!/usr/bin/env python3
"""
Script to fix CUDA usage in simple_watchdog.py by adding explicit CUDA device selection
"""
import os
import sys
from pathlib import Path

print("[INFO] Fixing CUDA support in simple_watchdog.py...")

# Path to the simple_watchdog.py file
watchdog_path = Path('simple_watchdog.py')

if not watchdog_path.exists():
    print(f"[ERROR] {watchdog_path} not found!")
    sys.exit(1)

# Read the current content
with open(watchdog_path, 'r') as f:
    content = f.read()

# Check if our fix is already applied
if 'if torch.cuda.is_available():' in content and 'torch.cuda.set_device(0)' in content:
    print("[INFO] CUDA fix already applied to simple_watchdog.py")
    sys.exit(0)

# Find the right spot to insert the CUDA setup code
import_section_end = content.find('# Import YouTube uploader if available')
if import_section_end == -1:
    import_section_end = content.find('try:')
    if import_section_end == -1:
        print("[ERROR] Could not find a suitable location to insert CUDA setup code")
        sys.exit(1)

# Add explicit CUDA setup after the imports but before anything else
cuda_setup = """
# Ensure CUDA is used if available
import torch
if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print("[INFO] CUDA is available, using GPU")
    torch.cuda.set_device(0)  # Use the first GPU
    # Enable cuDNN auto-tuning
    torch.backends.cudnn.benchmark = True
else:
    print("[INFO] CUDA not available or disabled, using CPU")

"""

# Insert the CUDA setup code
modified_content = content[:import_section_end] + cuda_setup + content[import_section_end:]

# Create a backup
backup_path = watchdog_path.with_suffix('.py.bak')
with open(backup_path, 'w') as f:
    f.write(content)
print(f"[INFO] Created backup at {backup_path}")

# Write the modified content
with open(watchdog_path, 'w') as f:
    f.write(modified_content)

print("[INFO] Applied CUDA fix to simple_watchdog.py")
print("[INFO] Now the pipeline will explicitly use GPU if available")
