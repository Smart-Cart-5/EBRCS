@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Find REPO_ROOT by walking up from this script directory (max 6 levels) ---
for %%I in ("%~dp0.") do set "CUR=%%~fI"
set "REPO_ROOT="

for /L %%N in (0,1,6) do (
  if exist "!CUR!\checkout_core\" (
    set "REPO_ROOT=!CUR!"
    goto :ROOT_FOUND
  )
  for %%P in ("!CUR!\..") do set "CUR=%%~fP"
)

echo [ERROR] Cannot locate repo root (checkout_core not found).
pause
exit /b 1

:ROOT_FOUND
set "APP_DIR=%REPO_ROOT%\app"
set "BACKEND_DIR=%APP_DIR%\backend"

if not exist "%BACKEND_DIR%\main.py" (
  echo [ERROR] Cannot find backend main.py: "%BACKEND_DIR%\main.py"
  echo         REPO_ROOT=%REPO_ROOT%
  pause
  exit /b 1
)

REM --- Find venv (prefer repo root .venv) ---
set "VENV_DIR="
if exist "%REPO_ROOT%\.venv\Scripts\activate.bat" set "VENV_DIR=%REPO_ROOT%\.venv"
if exist "%REPO_ROOT%\venv\Scripts\activate.bat"  set "VENV_DIR=%REPO_ROOT%\venv"
if exist "%APP_DIR%\.venv\Scripts\activate.bat"   set "VENV_DIR=%APP_DIR%\.venv"
if exist "%APP_DIR%\venv\Scripts\activate.bat"    set "VENV_DIR=%APP_DIR%\venv"

if "%VENV_DIR%"=="" (
  echo [ERROR] Cannot find venv activate.bat
  echo Create:
  echo   cd /d "%REPO_ROOT%"
  echo   python -m venv .venv
  pause
  exit /b 1
)

REM --- Write runtime env file for the backend runner ---
set "ENV_FILE=%~dp0_run_backend_env.cmd"
> "%ENV_FILE%" (
  echo @echo off
  echo set "REPO_ROOT=%REPO_ROOT%"
  echo set "APP_DIR=%APP_DIR%"
  echo set "VENV_DIR=%VENV_DIR%"
)

REM --- Launch backend window (call separate cmd to avoid escaping issues) ---
start "EBRCS Backend" cmd /k call "%~dp0_run_backend.cmd"

echo [DONE] Launched backend window.
echo        REPO_ROOT=%REPO_ROOT%
echo        VENV_DIR=%VENV_DIR%
endlocal
exit /b 0