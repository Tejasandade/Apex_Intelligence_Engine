@echo off
:: ============================================================
:: APEX INTELLIGENCE ENGINE — PROCESS GUARDIAN
:: Starts apex_v7.py and auto-restarts it on crash.
:: Logs all output to logs\apex_guardian.log with timestamps.
:: ============================================================

setlocal
set ROOT=%~dp0..
set VENV=%ROOT%\venv\Scripts\activate.bat
set SCRIPT=%ROOT%\scripts\apex_v7.py
set LOGFILE=%ROOT%\logs\apex_guardian.log
set PYTHON=%ROOT%\venv\Scripts\python.exe

:: Create logs directory if it doesn't exist
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

echo [%date% %time%] APEX GUARDIAN starting... >> "%LOGFILE%"
echo [%date% %time%] Script: %SCRIPT% >> "%LOGFILE%"

:LOOP
echo [%date% %time%] ============================================== >> "%LOGFILE%"
echo [%date% %time%] Starting apex_v7.py... >> "%LOGFILE%"
echo.
echo  APEX GUARDIAN: Starting apex_v7.py at %time%...
echo  (Logs: %LOGFILE%)
echo.

:: Run the bot — stdout and stderr go to both console and log
"%PYTHON%" "%SCRIPT%" 2>&1 | powershell -Command "$input | ForEach-Object { $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; \"$ts | $_\" | Tee-Object -FilePath '%LOGFILE%' -Append }"

set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Process exited with code: %EXIT_CODE% >> "%LOGFILE%"

if %EXIT_CODE% == 0 (
    echo  APEX GUARDIAN: Clean exit (code 0). Stopping guardian.
    echo [%date% %time%] Clean exit — guardian stopped. >> "%LOGFILE%"
    goto END
)

echo  APEX GUARDIAN: Crashed with code %EXIT_CODE%. Restarting in 10 seconds...
echo [%date% %time%] CRASH detected (code=%EXIT_CODE%). Restarting in 10s... >> "%LOGFILE%"
timeout /t 10 /nobreak >nul
goto LOOP

:END
echo [%date% %time%] APEX GUARDIAN stopped. >> "%LOGFILE%"
endlocal

