@echo off
REM One-shot web app setup: venv + dependencies + DB bootstrap.

setlocal
set "APP_DIR=%~dp0"

cd /d "%APP_DIR%"
call setup_venv.bat || exit /b 1
call setup_db.bat || exit /b 1

echo.
echo ✅ All setup steps completed.
echo Next:
echo   run_web.bat
exit /b 0
