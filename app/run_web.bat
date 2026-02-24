@echo off
REM Run backend (FastAPI) and frontend (Vite dev) concurrently for local development.
REM Usage: run_web.bat

REM Resolve script/app directory regardless of invocation cwd
set APP_DIR=%~dp0
for %%I in ("%APP_DIR%..") do set PROJECT_ROOT=%%~fI

REM Load .env if exists
if exist "%PROJECT_ROOT%\.env" (
    echo Loading environment from %PROJECT_ROOT%\.env...
    for /f "delims=" %%a in ('type "%PROJECT_ROOT%\.env" ^| findstr /r "="') do set %%a
)

if exist "%APP_DIR%.env" (
    echo Loading environment from %APP_DIR%.env...
    for /f "delims=" %%a in ('type "%APP_DIR%.env" ^| findstr /r "="') do set %%a
)

REM Activate backend virtual environment
if not exist "%APP_DIR%backend\.venv\Scripts\activate.bat" (
    echo Backend virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

echo Activating backend virtual environment...
call "%APP_DIR%backend\.venv\Scripts\activate.bat"

REM Fix OpenMP duplicate library issue
set KMP_DUPLICATE_LIB_OK=TRUE

REM Stop existing dev server windows to avoid duplicates
taskkill /FI "WINDOWTITLE eq EBRCS Backend" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq EBRCS Frontend" /T /F >nul 2>&1

echo === EBRCS Web App (Local Dev) ===
echo Backend : http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C in each window to stop the servers
echo.

REM Start backend in new window
start "EBRCS Backend" cmd /k "cd /d %APP_DIR% && call backend\.venv\Scripts\activate.bat && set PYTHONPATH=%PROJECT_ROOT% && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a bit for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend in new window
start "EBRCS Frontend" cmd /k "cd /d %APP_DIR%frontend && npm run dev"

echo.
echo Backend and Frontend started in separate windows
echo Close the terminal windows to stop the servers
pause
exit /b 0
