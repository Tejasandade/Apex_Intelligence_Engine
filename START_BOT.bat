@echo off
:: ============================================================
:: APEX INTELLIGENCE ENGINE — ONE-CLICK LAUNCHER
:: Just double-click this file to start the bot.
:: Automatically activates venv and starts apex_v7.py
:: ============================================================

title APEX Intelligence Engine v7.2

set ROOT=%~dp0
cd /d "%ROOT%"
set PYTHON=%ROOT%venv\Scripts\python.exe

echo.
echo  ====================================================
echo   APEX INTELLIGENCE ENGINE v7.2
echo   Multi-Asset Engine (BTC, ETH, Gold)
echo   Pluggable Prop Firm & MT5 Execution Ready
echo  ====================================================
echo.

:: Check Python exists
if not exist "%PYTHON%" (
    echo  ERROR: venv not found. Run this first:
    echo    python -m venv venv
    echo    venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo  Starting bot... (Press Ctrl+C to stop)
echo.

"%PYTHON%" "%ROOT%scripts\apex_v7.py"

echo.
echo  Bot stopped. Press any key to exit.
pause

