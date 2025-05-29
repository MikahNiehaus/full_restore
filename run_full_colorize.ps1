# PowerShell script to run restore_and_colorize.py from the Real-ESRGAN directory
$scriptPath = "c:\prj\full_restore\Real-ESRGAN\restore_and_colorize.py"
$pythonPath = "C:\Users\grayw\AppData\Local\Programs\Python\Python38\python.exe"

# Run the script
& $pythonPath $scriptPath @args