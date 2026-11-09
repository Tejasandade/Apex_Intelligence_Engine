@echo off
title APEX INDIAN INTELLIGENCE ENGINE v1.0 — LIVE BROKER EXECUTION (ANGEL ONE)
color 0C
cd /d "%~dp0"

echo ======================================================================
echo    APEX INDIAN INTELLIGENCE ENGINE v1.0 — LIVE BROKER TRADING
echo    Underlying: NIFTY & BANKNIFTY Options (CE / PE)
echo    Broker: Angel One SmartAPI (Live Order Execution)
echo    Capital: Rs. 1,00,000 | Max Daily Loss: 3%% (Rs. 3,000)
echo ======================================================================
echo.
echo  [WARNING] LIVE TRADING MODE ACTIVE! Real orders will be sent to Angel One.
echo  Press Ctrl+C immediately to abort, or any key to confirm and start.
pause

:run_loop
echo [%date% %time%] Launching Apex Indian Engine in LIVE mode...
python -m apex_indian_engine.run_indian --mode live --capital 100000 %*

echo.
echo [%date% %time%] Indian Engine process stopped.
echo Auto-restarting in 5 seconds (Press Ctrl+C to abort)...
timeout /t 5 >nul
goto run_loop
