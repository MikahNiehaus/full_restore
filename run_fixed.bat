@echo off
REM Run test script to verify fixes
echo Running tests to verify fixes...
python test_fixes.py

if %errorlevel% neq 0 (
    echo Tests failed, but continuing anyway...
    pause
)

REM Make sure the environment is activated
if exist "venv310\Scripts\activate.bat" (
    call venv310\Scripts\activate.bat
)

REM Run the pipeline with full restoration and GPU acceleration
echo Starting full restoration pipeline with GPU acceleration...
python simple_run.py --do-enhance

echo.
echo Pipeline completed. Press any key to exit.
pause
