@echo off
title APEX v8 Multi-Symbol Paper Trader
cd /d "%~dp0"
echo.
echo ============================================================
echo   APEX INTELLIGENCE ENGINE v8.0 - MULTI-SYMBOL MODE
echo   Paper Trading Only - Scalp + Swing
echo ============================================================
echo.

:loop
echo [%date% %time%] Starting apex_v8_multi.py...
venv\Scripts\python.exe scripts\apex_v8_multi.py
echo.
echo [%date% %time%] Process exited. Restarting in 10s... (Ctrl+C to stop)
timeout /t 10 /nobreak
goto loop

