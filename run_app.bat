@echo off
setlocal
cd /d "%~dp0"
title JPM Chooser Option Pricing Tool
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

set "PYTHON_CMD="

if exist "D:\Pythonstudy\python.exe" (
    set "PYTHON_CMD=D:\Pythonstudy\python.exe"
    goto launch
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto launch
)

py -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    goto launch
)

echo.
echo ERROR: A usable Python installation could not be found.
echo Install Python or update PYTHON_CMD in run_app.bat.
echo.
pause
exit /b 1

:launch
echo Starting JPM Chooser Option Pricing Tool...
echo Python: %PYTHON_CMD%
echo Project: %CD%
echo.
"%PYTHON_CMD%" -m streamlit run "app\app.py" --server.headless false

if errorlevel 1 (
    echo.
    echo ERROR: The Streamlit application could not start.
    echo Review the message above, then press any key to close this window.
    pause
    exit /b 1
)

endlocal
