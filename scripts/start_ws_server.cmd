@echo off
cd /d D:\Apex_Intelligence_Engine
call venv\Scripts\python.exe -m uvicorn src.dashboard.backend.ws_server:app --host 127.0.0.1 --port 8080
