#!/bin/bash
# systemd 서비스 설정 스크립트

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "⚙️  systemd 서비스 설정"
echo "====================="
echo ""

# 서비스 파일 복사
echo "📋 서비스 파일 복사 중..."
sudo cp "$SCRIPT_DIR/ebrcs.service" /etc/systemd/system/

# systemd 리로드
echo "🔄 systemd 리로드 중..."
sudo systemctl daemon-reload

# 서비스 활성화
echo "✅ 서비스 활성화 중..."
sudo systemctl enable ebrcs.service

# 서비스 시작
echo "🚀 서비스 시작 중..."
sudo systemctl start ebrcs.service

# 상태 확인
sleep 3
sudo systemctl status ebrcs.service --no-pager

echo ""
echo "====================="
echo "✅ systemd 설정 완료!"
echo ""
echo "📝 서비스 관리 명령어:"
echo "  시작:   sudo systemctl start ebrcs"
echo "  종료:   sudo systemctl stop ebrcs"
echo "  재시작: sudo systemctl restart ebrcs"
echo "  상태:   sudo systemctl status ebrcs"
echo "  로그:   sudo journalctl -u ebrcs -f"
echo ""
