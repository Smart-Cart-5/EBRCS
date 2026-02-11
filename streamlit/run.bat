@echo off
REM Streamlit 실행 스크립트 (Windows)

if exist .env (
    echo Loading environment from .env...
    for /f "delims=" %%a in ('type .env ^| findstr /v "^#"') do set %%a
)

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

streamlit run app.py
