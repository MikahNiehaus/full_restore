@echo off
echo Starting automatic image restoration and colorization...
echo Place images in the "inputs" folder to process them automatically.
echo Processed images will appear in the "outputs" folder.
echo.
cd /d "%~dp0"
call venv\Scripts\activate.bat
python auto_watch_simple.py
pause
