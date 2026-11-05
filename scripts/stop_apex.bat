@echo off
:: Stop the APEX bot process gracefully
echo Stopping APEX Intelligence Engine...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq apex_v7*" 2>nul
:: Fallback: kill any python.exe running apex_v7.py
for /f "tokens=1" %%p in ('wmic process where "commandline like '%%apex_v7%%'" get processid ^| findstr /r "[0-9]"') do (
    echo Killing PID %%p
    taskkill /F /PID %%p 2>nul
)
echo Done.

