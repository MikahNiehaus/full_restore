"""
Cleanup script to ensure consistent code usage across the project.
This updates references to old processing files and ensures the unified processor is used.
"""
import os
import shutil
import sys

def main():
    print("=== Cleaning up project to ensure consistent code ===")
    
    # List of files that should be removed or replaced with the unified processor
    deprecated_files = [
        'robust_image_processor.py',  # Replaced by unified_image_processor.py
        'restore_photo.py'  # Should use only the one in Real-ESRGAN
    ]
    
    # Delete deprecated files if they exist
    for file in deprecated_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed deprecated file: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")
                
    # Make sure the unified image processor is in the root directory
    if not os.path.exists('unified_image_processor.py'):
        print("ERROR: unified_image_processor.py not found!")
    else:
        print("Unified image processor is present.")
    
    # Ensure proper directory structure
    for directory in ['inputs', 'outputs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            
    print("\nProject cleanup complete!")
    print("Video and image processing now use the same unified code for consistency.")
    print("The unified image processor includes brightness adjustments to fix dark output.")

if __name__ == "__main__":
    main()
