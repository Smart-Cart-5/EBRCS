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

sleep 2

# 포트 기반 강제 종료
PORT_PID=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    echo "  ✓ 포트 8000 점유 프로세스 강제 종료 (PID: $PORT_PID)"
    kill -9 $PORT_PID || true
fi

PORT_PID=$(lsof -ti:5173 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    echo "  ✓ 포트 5173 점유 프로세스 강제 종료 (PID: $PORT_PID)"
    kill -9 $PORT_PID || true
fi

echo "✅ 종료 완료"
