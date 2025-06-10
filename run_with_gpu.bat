@echo off
REM GPU Accelerated Full Restoration Pipeline Launcher
REM This script activates the virtual environment and runs the full restoration pipeline with GPU acceleration

echo ======================================================================
echo DeOldify Full Restoration Pipeline with GPU Acceleration
echo ======================================================================

echo.
echo Checking for CUDA GPU...

REM Check for NVIDIA GPU
nvidia-smi > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU not detected or drivers not properly installed.
    echo           Pipeline may run in CPU-only mode (very slow).
    echo.
    pause
)

echo.
echo Activating virtual environment...

REM Activate the virtual environment if it exists
if exist "venv310\Scripts\activate.bat" (
    call venv310\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found at venv310\Scripts\activate.bat
    echo           Will try to continue without activating environment.
    echo.
)

echo.
echo Starting full restoration pipeline with GPU acceleration...
echo.
echo * GPU will be used for DeOldify colorization
echo * GPU will be used for image restoration
echo * Audio enhancements enabled
echo.
echo Processing any videos found in the 'inputs' directory...
echo.

REM Run the simple_run.py script with GPU acceleration
python simple_run.py --do-enhance

echo.
echo Pipeline completed. Press any key to exit.
pause

REM Deactivate the virtual environment if it was activated
if defined VIRTUAL_ENV (
    deactivate
)
