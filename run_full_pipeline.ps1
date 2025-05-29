# Step 1: Activate restoration venv and run restoration
& "C:\path\to\restoration\venv\Scripts\Activate.ps1"
python batch_restore.py
deactivate

# Step 2: Activate DeOldify venv and run colorization
& "C:\path\to\deoldify\venv\Scripts\Activate.ps1"
python batch_colorize.py
deactivate
