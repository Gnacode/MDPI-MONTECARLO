@echo off
REM Create and provision the Python virtual environment (Windows)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Environment ready. Activate later with: .venv\Scripts\activate.bat
