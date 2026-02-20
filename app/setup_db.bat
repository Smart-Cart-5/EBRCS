@echo off
REM Bootstrap/check DB schema used by EBRCS web app.
REM Usage:
REM   setup_db.bat
REM   setup_db.bat --check

setlocal enabledelayedexpansion

set "APP_DIR=%~dp0"
set "PROJECT_ROOT=%APP_DIR%.."

if not exist "%APP_DIR%backend\.venv\Scripts\activate.bat" (
    echo ❌ Backend virtual environment not found. Run setup_venv.bat first.
    exit /b 1
)

if exist "%PROJECT_ROOT%\.env" (
    echo Loading environment from ..\.env...
    for /f "usebackq tokens=1,* delims==" %%A in ("%PROJECT_ROOT%\.env") do (
        set "KEY=%%A"
        if not "!KEY!"=="" if not "!KEY:~0,1!"=="#" set "%%A=%%B"
    )
)

call "%APP_DIR%backend\.venv\Scripts\activate.bat"
set "PYTHONPATH=%APP_DIR%;%PROJECT_ROOT%"

python -m backend.db_bootstrap %*
exit /b %errorlevel%
