@echo off
title APEX INDIAN INTELLIGENCE ENGINE v1.0 — NIFTY & BANKNIFTY OPTIONS
color 0B
cd /d "%~dp0"

echo ======================================================================
echo    APEX INDIAN INTELLIGENCE ENGINE v1.0 — NSE OPTIONS CORE
echo    Underlying: NIFTY & BANKNIFTY Options (CE / PE)
echo    Broker Bridge: Angel One SmartAPI (Fully Automated TOTP Auth)
echo ======================================================================
echo.

:run_loop
echo [%date% %time%] Launching Apex Indian Engine in PAPER mode...
python -m apex_indian_engine.run_indian --mode paper --capital 100000 %*

echo.
echo [%date% %time%] Indian Engine process stopped.
echo Auto-restarting in 5 seconds (Press Ctrl+C to stop)...
timeout /t 5 >nul
goto run_loop

