@echo off
setlocal

REM Start local MySQL via Docker for EBRCS (Windows).

where docker >nul 2>nul
if errorlevel 1 (
  echo ❌ docker not found. Install Docker Desktop first.
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo ❌ docker compose not found. Enable Docker Compose in Docker Desktop.
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker-compose.mysql.yml"

echo Starting local MySQL container...
docker compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 exit /b 1

echo Waiting for MySQL readiness...
set /a retries=60

:wait_loop
docker compose -f "%COMPOSE_FILE%" exec -T mysql mysqladmin ping -h 127.0.0.1 -uroot -proot1234 --silent >nul 2>nul
if not errorlevel 1 goto ready

set /a retries-=1
if %retries% LEQ 0 (
  echo ❌ MySQL container did not become ready in time.
  echo Check logs: docker compose -f "%COMPOSE_FILE%" logs mysql
  exit /b 1
)

timeout /t 2 /nobreak >nul
goto wait_loop

:ready
echo ✅ Local MySQL is ready.
echo.
echo Use this in .env:
echo DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db
exit /b 0
