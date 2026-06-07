@echo off
echo Starting Apex Intelligence Engine V5 Dashboard...
echo Crypto Dashboard: http://localhost:8000/?port=8765
echo India Dashboard:  http://localhost:8000/?port=8766
python -m http.server 8000 --directory dashboard
