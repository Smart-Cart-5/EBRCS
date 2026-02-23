@echo off
setlocal EnableDelayedExpansion

REM Import products/product_prices seed into local Docker MySQL without local mysql client.
REM Usage:
REM   import_price_seed.bat --seed db\seeds\price_seed_latest.sql
REM   import_price_seed.bat --seed db\seeds\price_seed_latest.sql.gz

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%docker-compose.mysql.yml"
set "SEED_FILE="
set "APPEND_MODE=false"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--seed" (
  set "SEED_FILE=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--append" (
  set "APPEND_MODE=true"
  shift
  goto parse_args
)
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage
echo ❌ Unknown option: %~1
goto usage

:args_done
if "%SEED_FILE%"=="" goto usage

if not exist "%SEED_FILE%" (
  if exist "%CD%\%SEED_FILE%" (
    set "SEED_FILE=%CD%\%SEED_FILE%"
  ) else (
    echo ❌ Seed file not found: %SEED_FILE%
    exit /b 1
  )
)

where docker >nul 2>nul
if errorlevel 1 (
  echo ❌ docker not found. Install Docker Desktop first.
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo ❌ docker compose not found.
  exit /b 1
)

echo Ensuring local MySQL container is running...
docker compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 exit /b 1

echo Uploading seed file to container...
docker cp "%SEED_FILE%" ebrcs-local-mysql:/tmp/price_seed_input
if errorlevel 1 (
  echo ❌ Failed to copy seed file to container.
  exit /b 1
)

if /I "%APPEND_MODE%"=="false" (
  echo Truncating existing tables ^(replace mode^) ...
  docker compose -f "%COMPOSE_FILE%" exec -T mysql mysql -uroot -proot1234 item_db -e "SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE product_prices; TRUNCATE TABLE products; SET FOREIGN_KEY_CHECKS=1;"
  if errorlevel 1 (
    echo ❌ Failed to truncate tables. Run app\setup_db.bat first.
    exit /b 1
  )
)

echo Importing seed...
set "EXT=%SEED_FILE:~-3%"
if /I "%EXT%"==".gz" (
  docker compose -f "%COMPOSE_FILE%" exec -T mysql sh -lc "gzip -dc /tmp/price_seed_input | mysql -uroot -proot1234 item_db"
) else (
  docker compose -f "%COMPOSE_FILE%" exec -T mysql sh -lc "mysql -uroot -proot1234 item_db < /tmp/price_seed_input"
)
if errorlevel 1 (
  echo ❌ Seed import failed.
  exit /b 1
)

echo Verifying row counts...
docker compose -f "%COMPOSE_FILE%" exec -T mysql mysql -uroot -proot1234 item_db -e "SELECT 'products' AS table_name, COUNT(*) AS rows_count FROM products UNION ALL SELECT 'product_prices' AS table_name, COUNT(*) AS rows_count FROM product_prices;"

echo ✅ Price seed imported.
exit /b 0

:usage
echo Import EBRCS price seed into local Docker MySQL.
echo.
echo Usage:
echo   import_price_seed.bat --seed db\seeds\price_seed_latest.sql
echo   import_price_seed.bat --seed db\seeds\price_seed_latest.sql.gz
echo   import_price_seed.bat --seed db\seeds\price_seed_latest.sql --append
exit /b 1
