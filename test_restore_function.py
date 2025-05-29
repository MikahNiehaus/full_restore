"""
Test script to verify the behavior of the restore_old_photo function
"""
import os
import sys
sys.path.append('Real-ESRGAN')  # Add the Real-ESRGAN directory to the path
from restore_photo import restore_old_photo

def main():
    # Create test directories
    os.makedirs('test_restore', exist_ok=True)
    
    # Test the function with a sample image
    input_path = 'inputs/test_video.mp4_frame_000000.png'  # Assuming this file exists
    output_dir = 'test_restore'
    
    print(f"Testing restore_old_photo with:\nInput: {input_path}\nOutput dir: {output_dir}")
    restored_path = restore_old_photo(input_path, output_dir, scale=2)
    
    print(f"Function returned: {restored_path}")
    print(f"Does path exist? {os.path.exists(restored_path) if restored_path else 'N/A'}")
    
    # List files in output directory
    print("\nFiles in output directory:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
