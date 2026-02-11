@echo off
REM Run backend (FastAPI) and frontend (Vite dev) concurrently for local development.
REM Usage: run_web.bat

REM Load .env if exists
if exist .env (
    echo Loading environment from .env...
    for /f "delims=" %%a in ('type .env ^| findstr /v "^#"') do set %%a
)

REM Activate backend virtual environment
if not exist backend\.venv\Scripts\activate.bat (
    echo ⚠️  Backend virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

echo Activating backend virtual environment...
call backend\.venv\Scripts\activate.bat

REM Fix OpenMP duplicate library issue
set KMP_DUPLICATE_LIB_OK=TRUE

echo === EBRCS Web App (Local Dev) ===
echo Backend : http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C in each window to stop the servers
echo.

REM Get project root directory
set PROJECT_ROOT=%cd%\..

REM Start backend in new window
start "EBRCS Backend" cmd /k "cd /d %cd% && call backend\.venv\Scripts\activate.bat && set PYTHONPATH=%PROJECT_ROOT% && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a bit for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend in new window
start "EBRCS Frontend" cmd /k "cd /d %cd%\frontend && npm run dev"

echo.
echo ✅ Backend and Frontend started in separate windows
echo Close the terminal windows to stop the servers
pause
