import sys
import re

# Read the file content
file_path = "c:\\prj\\full_restore\\DeOldify\\deoldify\\visualize.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix the first issue - missing newline after result_path
content = re.sub(r"\)\s+logging\.info", r")\n        logging.info", content)

# Fix the second issue - missing newline between methods
content = re.sub(r"\)\s+def colorize_from_url", r")\n\n    def colorize_from_url", content)

# Fix the third issue - missing newline after download_video_from_url
content = re.sub(r"_download_video_from_url\(source_url, source_path\)\s+return", 
                 r"_download_video_from_url(source_url, source_path)\n        return", content)

# Fix the fourth issue - missing newline between url and filename methods
content = re.sub(r"\)\s+def colorize_from_file_name", r")\n\n    def colorize_from_file_name", content)

# Modify colorize_from_file_name to handle input paths better
pattern = r"def colorize_from_file_name\([^)]*\)[^:]*:.*?# Try in inputs/ directory.*?source_path = self\.source_folder / simple_filename"
replacement = """def colorize_from_file_name(
        self, file_name: str, render_factor: int = None, watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        # Analyze if file_name is already a full path
        file_path = Path(file_name)
        
        # Check if it exists as given (absolute path or relative to current directory)
        if file_path.exists():
            source_path = file_path
        # If it doesn't exist as-is, try just the filename part in different locations
        else:
            # Get just the filename if it's a path
            simple_filename = file_path.name
            
            # Try in inputs/ directory first
            inputs_path = Path('inputs') / simple_filename
            if inputs_path.exists():
                source_path = inputs_path
            # Try in video/source/ as last resort
            else:
                source_path = self.source_folder / simple_filename"""

# Write the modified content back to the file
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixes applied to visualize.py")
