@echo off
echo Setting up image restoration and colorization environment...
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Installing required packages...
pip install watchdog Pillow

echo Setting up colorizer module...
python setup_colorizer.py

echo Creating necessary directories...
mkdir -p inputs outputs processed

echo Setup complete!
echo Please put your image files into the 'inputs' folder to process them.
echo.
echo Starting automatic monitoring of the inputs folder...
echo Any new image placed in the 'inputs' folder will be automatically processed.
echo Press Ctrl+C to stop the monitoring process.
echo.

python auto_watch_simple.py

pause
