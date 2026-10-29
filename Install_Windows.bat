@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
echo EDC-AOPRisk GUI - first-time setup
set "EDC_PY="
py -3.13 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 set "EDC_PY=py -3.13"
if not defined EDC_PY (
  py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
  if not errorlevel 1 set "EDC_PY=py -3.12"
)
if not defined EDC_PY (
  python -c "import sys; assert sys.version_info[:2] in [(3,12),(3,13)] and sys.maxsize > 2**32" >nul 2>&1
  if not errorlevel 1 set "EDC_PY=python"
)
if not defined EDC_PY (
  echo Install 64-bit Python 3.12 or 3.13 from https://www.python.org/downloads/windows/
  echo Enable the Python launcher / Add Python to PATH, then run this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  %EDC_PY% -m venv ".venv"
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" selfcheck.py
if errorlevel 1 goto failed
echo Setup completed. Double-click Start_Windows.bat to open the GUI.
pause
exit /b 0
:failed
echo Setup failed. Review the messages above. No model has been changed.
pause
exit /b 1
