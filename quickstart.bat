@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

python -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Python is not installed or not on PATH.
  echo Please install Python 3.10+ and try again.
  pause
  exit /b 1
)

python -m pip --version >nul 2>&1
if errorlevel 1 (
  echo Pip is not available. Please reinstall Python with pip enabled.
  pause
  exit /b 1
)

echo Installing requirements if needed...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  pause
  exit /b 1
)

echo Launching OCreator...
python ocreator.py
pause
