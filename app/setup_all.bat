@echo off
REM One-shot web app setup: venv + dependencies + DB bootstrap.

setlocal
set "APP_DIR=%~dp0"
set "PROJECT_ROOT=%APP_DIR%.."

echo.
echo 🔐 HuggingFace 토큰 자동 설정 확인 중...
powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_DIR%setup_hf_token.ps1" -ProjectRoot "%PROJECT_ROOT%"
if errorlevel 1 exit /b 1

cd /d "%APP_DIR%"
call setup_venv.bat || exit /b 1
call setup_db.bat || exit /b 1

echo.
echo ✅ All setup steps completed.
echo Next:
echo   run_web.bat
exit /b 0
