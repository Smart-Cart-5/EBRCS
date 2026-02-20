@echo off
REM Windows wrapper for db/start_local_mysql.sh
setlocal

where bash >nul 2>&1
if errorlevel 1 (
    echo [ERROR] bash not found. Install Git for Windows or run in WSL.
    exit /b 1
)

bash "%~dp0start_local_mysql.sh" %*
exit /b %errorlevel%
