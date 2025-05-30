# Simple script to fix indentation issues in simple_watchdog.py
import re

def fix_indentation():
    # Read the file
    with open('simple_watchdog.py', 'r') as f:
        content = f.read()
    
    # Fix the specific indentation issues
    content = re.sub(r'print\("\[WARNING\] No audio track found or extraction failed"\)\s+def move_to_uploaded',
                    'print("[WARNING] No audio track found or extraction failed")\n\n    def move_to_uploaded', 
                    content)
    
    content = re.sub(r'print\(f"\[INFO\] Finished processing: \{video_path\.name\}"\)\s+def upload_to_youtube',
                    'print(f"[INFO] Finished processing: {video_path.name}")\n\n    def upload_to_youtube', 
                    content)
    
    content = re.sub(r'traceback\.print_exc\(\)\s+self\.move_to_failed_upload\(colorized_video_path\)\s+def run',
                    'traceback.print_exc()\n            self.move_to_failed_upload(colorized_video_path)\n\n    def run', 
                    content)
    
    # Write the fixed content back
    with open('simple_watchdog.py', 'w') as f:
        f.write(content)
    
    print("Indentation issues fixed in simple_watchdog.py")

if __name__ == "__main__":
    fix_indentation()
