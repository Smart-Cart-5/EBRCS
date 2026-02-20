@echo off
REM 웹앱 백엔드 가상환경 설정 스크립트 (Windows)

echo 🔧 웹앱 백엔드 가상환경 설정 시작...

REM backend 디렉토리로 이동
cd backend

REM 가상환경 생성
if not exist .venv (
    echo 📦 가상환경 생성 중...
    python -m venv .venv
) else (
    echo ✓ 가상환경이 이미 존재합니다.
)

REM 가상환경 활성화
echo 🔌 가상환경 활성화 중...
call .venv\Scripts\activate.bat

REM 의존성 설치
echo 📥 패키지 설치 중...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Frontend npm 설치
echo.
echo 📦 Frontend 패키지 설치 중...
cd ..\frontend
call npm install

cd ..

echo.
echo ✅ 웹앱 환경 설정 완료!
echo.
echo 사용법:
echo   setup_db.bat  ^(DB 스키마 초기화/검증^)
echo   run_web.bat
pause
