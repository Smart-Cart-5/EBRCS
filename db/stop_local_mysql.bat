@echo off
setlocal

REM Stop local MySQL Docker container for EBRCS.

where docker >nul 2>nul
if errorlevel 1 (
  echo ❌ docker not found.
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo ❌ docker compose not found.
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker-compose.mysql.yml"

if /I "%~1"=="--purge" (
  echo Stopping local MySQL and removing volume...
  docker compose -f "%COMPOSE_FILE%" down -v
) else (
  echo Stopping local MySQL ^(volume kept^) ...
  docker compose -f "%COMPOSE_FILE%" down
)

if errorlevel 1 exit /b 1
echo ✅ Done.
exit /b 0
