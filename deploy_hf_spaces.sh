#!/bin/bash
# Hugging Face Spaces 배포 스크립트

set -e  # 에러 발생 시 중단

echo "🚀 Hugging Face Spaces 배포 준비"
echo "================================"
echo ""

# 1. 사용자 정보 입력
read -p "Hugging Face 사용자명 입력: " HF_USERNAME
read -p "Space 이름 입력 (예: ebrcs-checkout): " SPACE_NAME

SPACE_REPO="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

echo ""
echo "📦 배포 정보:"
echo "  Space URL: $SPACE_REPO"
echo ""

# 2. 임시 디렉토리 생성
TEMP_DIR="hf_space_temp"
if [ -d "$TEMP_DIR" ]; then
    echo "⚠️  기존 임시 디렉토리 삭제 중..."
    rm -rf "$TEMP_DIR"
fi

mkdir -p "$TEMP_DIR"
echo "✅ 임시 디렉토리 생성: $TEMP_DIR"

# 3. 필요한 파일 복사
echo ""
echo "📋 파일 복사 중..."

# Streamlit 앱
cp app.py "$TEMP_DIR/"
echo "  ✓ app.py"

# 모바일 앱 (선택 사항)
if [ -f "mobile_app.py" ]; then
    cp mobile_app.py "$TEMP_DIR/"
    echo "  ✓ mobile_app.py"
fi

# 코어 로직
cp -r checkout_core "$TEMP_DIR/"
echo "  ✓ checkout_core/"

# 페이지
if [ -d "pages" ]; then
    cp -r pages "$TEMP_DIR/"
    echo "  ✓ pages/"
fi

if [ -d "pages_mobile" ]; then
    cp -r pages_mobile "$TEMP_DIR/"
    echo "  ✓ pages_mobile/"
fi

# UI 테마
if [ -f "ui_theme.py" ]; then
    cp ui_theme.py "$TEMP_DIR/"
    echo "  ✓ ui_theme.py"
fi

if [ -f "mobile_nav.py" ]; then
    cp mobile_nav.py "$TEMP_DIR/"
    echo "  ✓ mobile_nav.py"
fi

# 데이터 (Git LFS 필요)
if [ -d "data" ]; then
    cp -r data "$TEMP_DIR/"
    echo "  ✓ data/ (경고: 대용량 파일, Git LFS 필요)"
fi

# 패키지 의존성
cp requirements.txt "$TEMP_DIR/"
echo "  ✓ requirements.txt"

# README (Space 설정)
cp README_HF_SPACE.md "$TEMP_DIR/README.md"
echo "  ✓ README.md (Space 설정)"

# .gitattributes (Git LFS)
cp .gitattributes "$TEMP_DIR/"
echo "  ✓ .gitattributes (Git LFS)"

# .gitignore
if [ -f ".gitignore_hf" ]; then
    cp .gitignore_hf "$TEMP_DIR/.gitignore"
    echo "  ✓ .gitignore"
fi

# 4. Git 저장소 초기화
echo ""
echo "🔧 Git 저장소 초기화 중..."
cd "$TEMP_DIR"

git init
git lfs install

# 5. 원격 저장소 추가
echo ""
echo "🌐 Hugging Face Space 연결 중..."
echo "  주의: 이제 Hugging Face 로그인이 필요합니다."
echo "  토큰 생성: https://huggingface.co/settings/tokens"
echo ""

git remote add origin "$SPACE_REPO"

# 6. 커밋
echo ""
echo "📝 커밋 생성 중..."
git add .
git commit -m "Initial deployment to Hugging Face Spaces"

# 7. Push
echo ""
echo "⬆️  Hugging Face Spaces에 업로드 중..."
echo "  (Git LFS 파일이 있으면 시간이 걸릴 수 있습니다)"
echo ""

git push -u origin main

# 8. 완료
cd ..
echo ""
echo "================================"
echo "✅ 배포 완료!"
echo ""
echo "🌐 Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""
echo "⚙️  다음 단계:"
echo "  1. Space 페이지에서 Settings → Variables and secrets 클릭"
echo "  2. New secret 추가:"
echo "     Name: HF_TOKEN"
echo "     Value: (당신의 Hugging Face token)"
echo "  3. Restart Space"
echo ""
echo "📊 Space가 빌드되는 동안 1-2분 기다려주세요."
echo "   GPU 할당: Settings → Hardware → T4 small (무료)"
echo ""
echo "🗑️  임시 파일 정리: rm -rf $TEMP_DIR"
echo ""
