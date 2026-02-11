#!/bin/bash
# EBRCS 웹앱 종료 스크립트

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🛑 EBRCS 웹앱 종료 중..."

# PID 파일에서 프로세스 종료
if [ -f "$APP_DIR/logs/backend.pid" ]; then
    BACKEND_PID=$(cat "$APP_DIR/logs/backend.pid")
    kill $BACKEND_PID 2>/dev/null || true
    echo "  ✓ Backend 종료 (PID: $BACKEND_PID)"
    rm "$APP_DIR/logs/backend.pid"
fi

if [ -f "$APP_DIR/logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$APP_DIR/logs/frontend.pid")
    kill $FRONTEND_PID 2>/dev/null || true
    echo "  ✓ Frontend 종료 (PID: $FRONTEND_PID)"
    rm "$APP_DIR/logs/frontend.pid"
fi

# 혹시 모를 남은 프로세스 정리
pkill -f "uvicorn backend.main:app" || true
pkill -f "vite preview" || true

echo "✅ 종료 완료"
