#!/bin/bash
# 배포 관련 파일만 정리 (로컬 실행 파일은 모두 유지)

set -e

echo "🧹 배포 관련 파일 정리 스크립트"
echo "======================================"
echo ""
echo "✅ 유지할 것:"
echo "  - Streamlit 데모 (app.py, pages/)"
echo "  - React 웹앱 (frontend/, backend/)"
echo "  - 실행 스크립트 (run.sh, run_web.sh)"
echo "  - Docker 설정 (Dockerfile, docker-compose.yml)"
echo "  - 코어 로직 (checkout_core/)"
echo ""
echo "❌ 삭제할 것:"
echo "  - Hugging Face Spaces 배포 파일"
echo "  - GitHub Pages 배포 파일"
echo "  - 배포 관련 문서"
echo "  - DB 마이그레이션 (미사용)"
echo "  - 테스트 파일"
echo ""
read -p "계속하시겠습니까? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 1
fi

# Git 백업
echo ""
echo "📦 현재 상태 백업 중..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    BACKUP_BRANCH="backup-before-cleanup-$(date +%Y%m%d-%H%M%S)"
    git add -A
    git commit -m "Backup before deployment files cleanup" || true
    git branch "$BACKUP_BRANCH"
    echo "✅ 백업 브랜치: $BACKUP_BRANCH"
else
    echo "⚠️  Git 저장소가 아닙니다. 백업을 건너뜁니다."
fi

# 삭제 시작
echo ""
echo "🗑️  배포 관련 파일 삭제 중..."

# Hugging Face Spaces
rm -f deploy_hf_spaces.sh
rm -f HF_SPACES_MANUAL.md
rm -f README_HF_SPACE.md
rm -f .gitattributes
rm -f .gitignore_hf
rm -f Dockerfile.streamlit
rm -rf hf_space_temp
echo "  ✓ Hugging Face Spaces 배포 파일 삭제"

# GitHub Pages
rm -rf .github/workflows
rm -f frontend/.env.production
echo "  ✓ GitHub Pages 배포 파일 삭제"

# 배포 문서
rm -f docs/DEPLOYMENT_GUIDE.md
rm -f docs/DEPLOYMENT_SUMMARY.md
echo "  ✓ 배포 관련 문서 삭제"

# DB 마이그레이션 (미사용)
rm -rf backend/migrations
rm -f docs/DATABASE_ERD.md
echo "  ✓ DB 마이그레이션 파일 삭제"

# 테스트/분석 문서
rm -f test_incremental_update.py
rm -f docs/INCREMENTAL_UPDATE_SUMMARY.md
echo "  ✓ 테스트 파일 삭제"

# 정리 스크립트 자체도 삭제 (선택)
if [ -f "cleanup_for_streamlit.sh" ]; then
    rm -f cleanup_for_streamlit.sh
    echo "  ✓ 잘못된 정리 스크립트 삭제"
fi

# 완료
echo ""
echo "======================================"
echo "✅ 정리 완료!"
echo ""
echo "📂 남은 구조:"
echo ""
echo "EBRCS_streaming/"
echo "├── 🎨 Streamlit 데모"
echo "│   ├── app.py"
echo "│   ├── mobile_app.py"
echo "│   ├── pages/"
echo "│   ├── pages_mobile/"
echo "│   ├── run.sh"
echo "│   └── run_mobile.sh"
echo "├── ⚛️  React 웹앱"
echo "│   ├── frontend/"
echo "│   ├── backend/"
echo "│   └── run_web.sh"
echo "├── 🧠 공통 코어"
echo "│   ├── checkout_core/"
echo "│   └── data/"
echo "└── 🐳 Docker"
echo "    ├── Dockerfile"
echo "    └── docker-compose.yml"
echo ""
echo "🚀 실행 방법:"
echo "  ./run.sh           # Streamlit 데스크톱"
echo "  ./run_mobile.sh    # Streamlit 모바일"
echo "  ./run_web.sh       # React 웹앱"
echo ""
echo "🔄 복구 방법:"
echo "  git checkout $BACKUP_BRANCH"
echo ""
