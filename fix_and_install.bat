@echo off
echo ========================================================
echo Apex Intelligence Engine — Deep Learning Setup Fix
echo ========================================================
echo.
echo Cleaning corrupted pip cache and broken packages...
python -m pip cache purge
rmdir /s /q "%APPDATA%\Python\Python312\site-packages\~orch" 2>nul
rmdir /s /q "%APPDATA%\Python\Python312\site-packages\~unctorch" 2>nul
echo.
echo Installing PyTorch for AMD DirectML...
echo (This may take 5-15 minutes depending on download speed)
echo.

set PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
python -m pip install numpy scipy xgboost --force-reinstall --user
python -m pip install torch torch-directml torchvision --no-cache-dir --user

echo.
echo ========================================================
echo Installation Complete! 
echo You can now run the backtester:
echo python scripts\run_backtest_oos.py --symbol ethusdt
echo ========================================================
pause
