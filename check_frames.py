"""
Utility script to check frame directories and debug any missing files
"""
import os
import sys

def check_directory(dir_path, file_pattern=None):
    """Check directory contents and print stats"""
    print(f"\n=== Checking directory: {dir_path} ===")
    
    if not os.path.exists(dir_path):
        print(f"Directory does not exist: {dir_path}")
        return 0
        
    files = os.listdir(dir_path)
    print(f"Total files: {len(files)}")
    
    if file_pattern:
        matching = [f for f in files if file_pattern in f]
        print(f"Files matching '{file_pattern}': {len(matching)}")
        
    # Print first few files
    if files:
        print("\nSample files:")
        for i, f in enumerate(sorted(files)[:5]):
            full_path = os.path.join(dir_path, f)
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else "DIR"
            print(f"  {i+1}. {f} ({size} bytes)")
            
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more files")
    
    return len(files)

def main():
    # Check all temporary directories
    frame_dirs = [
        'temp_video_frames',
        'temp_enhanced_frames',
        'temp_colorized_frames',
        'temp_restored_frames',
    ]
    
    for dir_path in frame_dirs:
        count = check_directory(dir_path)
        if count == 0:
            print(f"WARNING: Directory '{dir_path}' is empty or doesn't exist!")

    # Check outputs directory
    check_directory('outputs')
    
if __name__ == "__main__":
    main()
