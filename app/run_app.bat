@echo off
REM Launcher for the Silkworm Disease Predictor web app.
REM Uses Python 3.11 explicitly (the 'python' on PATH is Python 2.7 from MGLTools).
cd /d "%~dp0"
py -3 app.py
pause
