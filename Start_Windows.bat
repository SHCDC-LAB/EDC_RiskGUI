@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First run Install_Windows.bat to configure the environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import tkinter; import rdkit; import sklearn; import lightgbm" >nul 2>&1
if errorlevel 1 (
  echo Missing dependencies. Run Install_Windows.bat again.
  pause
  exit /b 1
)
echo Checking feature and numerical consistency...
".venv\Scripts\python.exe" runtime_guard.py > Startup_Diagnostic.log 2>&1
if errorlevel 1 (
  echo Numerical self-check failed. Prediction was not started.
  echo See Startup_Runtime.json and Startup_Diagnostic.log in this folder.
  type Startup_Diagnostic.log
  pause
  exit /b 1
)
start "EDC-AOPRisk" ".venv\Scripts\pythonw.exe" "app.py"
