#!/usr/bin/env python3
"""
Script to fix indentation and syntax issues in visualize.py
"""
import re
import os
from pathlib import Path

def fix_visualize_py():
    file_path = Path('DeOldify/deoldify/visualize.py')
    
    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the colorize_from_url method - make sure there's proper spacing and newlines
    content = re.sub(r'def colorize_from_url\s*\(\s*self,\s*source_url,\s*file_name:\s*str,\s*render_factor:\s*int\s*=\s*None,\s*post_process:\s*bool\s*=\s*True,\s*watermarked:\s*bool\s*=\s*True,\s*\)\s*->\s*Path:\s*source_path', 
                     'def colorize_from_url(\n        self,\n        source_url,\n        file_name: str,\n        render_factor: int = None,\n        post_process: bool = True,\n        watermarked: bool = True,\n    ) -> Path:\n        source_path', 
                     content)
    
    # Fix any ) at the end of a line followed immediately by a new method definition
    content = re.sub(r'(\))\s+(def\s+\w+)', r'\1\n\n    \2', content)
    
    # Fix the return statement after download_video_from_url
    content = re.sub(r'self\._download_video_from_url\(source_url,\s*source_path\)\s*return', 
                     'self._download_video_from_url(source_url, source_path)\n        return', 
                     content)
    
    # Replace the colorize_from_file_name method with our more robust implementation
    new_method = '''    def colorize_from_file_name(
        self, file_name: str, render_factor: int = None, watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        # Try multiple possible paths in order of priority
        paths_to_try = []
        file_path = Path(file_name)
        
        # 1. First try as-is (might be a full path or relative path)
        paths_to_try.append(file_path)
        
        # 2. Check if path contains "inputs/" already and fix double-path issues
        if "inputs" in str(file_path):
            # Handle case where inputs/ might be duplicated (e.g., "inputs/inputs/file.mp4")
            clean_name = file_path.name
            fixed_path = Path('inputs') / clean_name
            if fixed_path != file_path:  # Avoid duplicating the same path
                paths_to_try.append(fixed_path)
        
        # 3. Try in inputs/ directory using just the filename
        simple_filename = file_path.name
        inputs_path = Path('inputs') / simple_filename
        if inputs_path not in paths_to_try:
            paths_to_try.append(inputs_path)
            
        # 4. Try in video/source/ as last resort
        source_path = self.source_folder / simple_filename
        paths_to_try.append(source_path)
        
        # Try each path until one exists
        for path in paths_to_try:
            if path.exists():
                return self._colorize_from_path(
                    path, render_factor=render_factor, post_process=post_process, watermarked=watermarked
                )
                
        # If we get here, no path worked
        print(f"[DEBUG] File not found. Tried the following paths:")
        for i, path in enumerate(paths_to_try):
            print(f"  {i+1}. {path}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        raise Exception(f'Video not found. Tried multiple paths including inputs/ and {self.source_folder}/')'''
    
    # Find the old method - pattern needs to match the entire method including the signature and content
    pattern = r'def colorize_from_file_name\(.*?\).*?->.*?Path:.*?source_path,.*?render_factor=render_factor.*?\)'
    content = re.sub(pattern, new_method, content, flags=re.DOTALL)
    
    # Fix any instances where an os.system() call is missing a newline before logging
    content = re.sub(r'(\))\s+(logging\.)', r'\1\n        \2', content)
    
    # Fix any remaining issue with missing newlines between end of one method and start of another
    content = re.sub(r'(return .*?)\s+(def\s+\w+)', r'\1\n\n    \2', content)
    
    # Write the corrected file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed indentation and syntax issues in {file_path}")

if __name__ == "__main__":
    fix_visualize_py()
