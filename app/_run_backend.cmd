@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM --- Load env written by run_web.bat ---
call "%~dp0_run_backend_env.cmd"

REM --- Safety: prevent PYTHONHOME path-eval crashes ---
set "PYTHONHOME="
set "PYTHONPATH=%REPO_ROOT%"

REM ============================================================
REM  Runtime toggles (affect ONLY this cmd window)
REM  - Change values here without touching system env
REM ============================================================

REM Data dir
REM You said 1536-dim embeddings/labels are in repo_root\data2, so keep DATA_DIR as data2.
set "DATA_DIR=%REPO_ROOT%\data2"

REM Core modes:
set "EVENT_MODE=false"
set "WARP_MODE=false"

REM Turn profiling ON to see [PROF] breakdown logs (decode/yolo/embed/faiss/send...)
set "ENABLE_PROFILING=true"

set "SEARCH_DEBUG_LOG=true"
set "SEARCH_DEBUG_SAVE_CROP=true"
set "SEARCH_DEBUG_CROP_DIR=%REPO_ROOT%\debug_crops"
set "SEARCH_DEBUG_SAVE_INTERVAL_MS=1000"


REM YOLO:
set "USE_YOLO=true"
REM set "YOLO_MODEL_PATH=%REPO_ROOT%\smartcart_hand_yolo11_best_arg_best.pt"
REM set "YOLO_CONF_THRESHOLD=0.5"

REM ============================================================
REM  Realtime performance tuning (new)
REM ============================================================

REM Backpressure: keep queue tiny so we always process the latest frame
set "CHECKOUT_QUEUE_MAXSIZE=1"

REM Throttle heavy stages (embedding/FAISS/voting) to every Nth frame
REM - Faster / more realtime: 8~12
REM - Balanced: 5~7
REM - More frequent recognition: 3~4 (slower)
set "SEARCH_EVERY_N_FRAMES=6"

REM ============================================================
REM  Responsive preset
REM ============================================================
set "FAISS_TOP_K=5"
set "SEARCH_COOLDOWN_MS_PRECONFIRM=20"
set "SEARCH_COOLDOWN_MS=450"
set "SEARCH_POST_CONFIRM_WINDOW_MS=1800"

set "SEARCH_CROP_MIN_SIDE=320"
set "SEARCH_CROP_PAD_RATIO=0.25"
set "SEARCH_CROP_MIN_SIDE_AFTER_EXPAND=220"
set "SEARCH_CROP_ASPECT_MIN=0.45"
set "SEARCH_CROP_ASPECT_MAX=2.2"

set "UNKNOWN_SCORE_THRESHOLD=0.43"
set "UNKNOWN_GAP_THRESHOLD=0.015"
set "STABLE_RESULT_FRAMES=2"
set "HIGH_CONFIDENCE_THRESHOLD=0.65"
set "HIGH_CONFIDENCE_THRESHOLD_OCCLUDED=0.82"

set "SEARCH_MASK_HAND=true"
set "HAND_OVERLAP_IOU=0.50"

REM Optional tuning (override defaults only if needed):
REM set "DETECT_EVERY_N_FRAMES=6"
REM set "MAX_PRODUCTS_PER_FRAME=3"
REM set "VOTE_WINDOW_SIZE=5"
REM set "VOTE_MIN_SAMPLES=3"

REM ============================================================

cd /d "%APP_DIR%"
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo [INFO] PYTHON=%VIRTUAL_ENV%
echo [INFO] EVENT_MODE=%EVENT_MODE%  WARP_MODE=%WARP_MODE%  PROFILING=%ENABLE_PROFILING%  USE_YOLO=%USE_YOLO%
echo [INFO] DATA_DIR=%DATA_DIR%
echo [INFO] CHECKOUT_QUEUE_MAXSIZE=%CHECKOUT_QUEUE_MAXSIZE%  SEARCH_EVERY_N_FRAMES=%SEARCH_EVERY_N_FRAMES%
python -c "import sys; print('[PYTHON]', sys.executable); import checkout_core; print('[OK] checkout_core import')"
echo.

REM --- Uvicorn (watch both app/ and checkout_core/ for reload) ---
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload ^
  --reload-dir "%APP_DIR%" ^
  --reload-dir "%REPO_ROOT%\checkout_core"

echo.
echo [ERROR] Uvicorn exited with code %ERRORLEVEL%
pause
