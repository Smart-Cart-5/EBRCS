@echo off
REM Windows wrapper for db/export_price_seed.sh
setlocal

where bash >nul 2>&1
if errorlevel 1 (
    echo [ERROR] bash not found. Install Git for Windows or run in WSL.
    exit /b 1
)

bash "%~dp0export_price_seed.sh" %*
exit /b %errorlevel%
