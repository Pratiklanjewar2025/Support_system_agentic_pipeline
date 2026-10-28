@echo off
title AdaRAG-Ops Technical Support Dashboard
cd /d "%~dp0"
echo =========================================================================
echo    Starting AdaRAG-Ops AI Technical Support Autonomous Dashboard...
echo =========================================================================
if exist ".\.venv\Scripts\python.exe" (
    ".\.venv\Scripts\python.exe" run_dashboard.py
) else (
    python run_dashboard.py
)
pause
