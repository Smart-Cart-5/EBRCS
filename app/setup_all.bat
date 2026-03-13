@echo off
REM One-shot full setup: local Docker MySQL + venv + dependencies + DB bootstrap + seed import.
REM
REM Usage:
REM   cd app && setup_all.bat
REM
REM Options:
REM   --skip-db    Docker MySQL 기동 및 시드 import 건너뜀 (EC2 공용 DB 사용 시)

setlocal enabledelayedexpansion
set "APP_DIR=%~dp0"
set "PROJECT_ROOT=%APP_DIR%.."
set "SKIP_DB=false"

:parse_args
if "%~1"=="--skip-db" ( set "SKIP_DB=true" & shift & goto parse_args )
if not "%~1"=="" ( echo Unknown option: %~1 & exit /b 1 )

REM ── Step 0: Docker Desktop 확인 및 설치 ───────────────────────────────────────
echo.
echo ^> Step 0: Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo   Docker not found. Installing Docker Desktop via winget...
    winget install --id Docker.DockerDesktop -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo   winget installation failed. Please install Docker Desktop manually:
        echo   https://www.docker.com/products/docker-desktop/
        exit /b 1
    )
    echo   Docker Desktop installed. Starting...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
)

REM Docker 데몬 대기
docker info >nul 2>&1
if errorlevel 1 (
    echo   Waiting for Docker daemon (up to 60s)...
    set /a _dc=0
    :wait_docker
    docker info >nul 2>&1
    if not errorlevel 1 goto docker_ready
    set /a _dc+=1
    if !_dc! geq 30 (
        echo   Docker did not start in time. Please start Docker Desktop manually and re-run.
        exit /b 1
    )
    timeout /t 2 /nobreak >nul
    goto wait_docker
    :docker_ready
    echo   Docker daemon is ready.
) else (
    echo   Docker is running.
)
echo.

REM ── Step 1: 로컬 Docker MySQL 기동 + 스키마 + 시드 ───────────────────────────
if "%SKIP_DB%"=="false" (
    echo.
    echo ^> Step 1/3: Setting up local Docker MySQL...
    call "%PROJECT_ROOT%db\setup_local_db.bat" || exit /b 1
    echo.
) else (
    echo.
    echo ^> Step 1/3: Skipped --skip-db. Using DATABASE_URL from .env.
    echo.
)

REM ── Step 2: Python venv + 의존성 설치 ────────────────────────────────────────
echo ^> Step 2/3: Setting up Python venv and Node dependencies...
cd /d "%APP_DIR%"
call setup_venv.bat || exit /b 1
echo.

REM ── Step 3: DB 스키마 최종 확인 ──────────────────────────────────────────────
echo ^> Step 3/3: Verifying DB schema...
call setup_db.bat || exit /b 1
echo.

echo ================================================================
echo   All setup steps completed.
echo.
echo   Next step:
echo     run_web.bat
echo ================================================================
exit /b 0
