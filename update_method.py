from pathlib import Path
import os

original_file = "c:\\prj\\full_restore\\DeOldify\\deoldify\\visualize.py"
replacement_method = "c:\\prj\\full_restore\\colorize_from_file_name.py"
new_file = "c:\\prj\\full_restore\\DeOldify\\deoldify\\visualize.py.new"

# Read the replacement method
with open(replacement_method, 'r') as f:
    new_method = f.read()

# Flag to track if we're in the method we want to replace
in_method = False
method_indentation = ""
current_method_name = ""

# Read and process the original file
with open(original_file, 'r') as f_in, open(new_file, 'w') as f_out:
    for line in f_in:
        # Check if this is the start of a method definition
        if line.strip().startswith('def '):
            method_name = line.strip().split('(')[0].split(' ')[1]
            current_method_name = method_name
            method_indentation = line[:line.find('def')]
            
            # If this is the method we want to replace
            if method_name == 'colorize_from_file_name':
                in_method = True
                # Write the replacement method with proper indentation
                for replacement_line in new_method.splitlines():
                    f_out.write(method_indentation + replacement_line + '\n')
            else:
                in_method = False
                f_out.write(line)
        # If we're not in the method to replace or it's not a method definition
        elif not in_method or not line.strip():
            # Check if we're exiting the method to replace
            if in_method and (line.strip().startswith('def ') or 
                               (line.strip() and line.startswith(method_indentation) and 
                                not line.startswith(method_indentation + ' '))):
                in_method = False
                
            if not in_method:
                f_out.write(line)

# Replace the original file with our new version
os.replace(new_file, original_file)
print("File successfully updated with new method!")
