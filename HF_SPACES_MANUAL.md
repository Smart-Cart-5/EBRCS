# 🤗 Hugging Face Spaces 배포 매뉴얼

## 🎯 배포 전 체크리스트

- [ ] Hugging Face 계정 생성 (https://huggingface.co)
- [ ] Git LFS 설치 (`brew install git-lfs` 또는 https://git-lfs.github.com)
- [ ] Hugging Face 토큰 생성 (https://huggingface.co/settings/tokens)

---

## 🚀 방법 1: 자동 스크립트 (추천)

### 1단계: 스크립트 실행

```bash
./deploy_hf_spaces.sh
```

**입력 사항:**
- Hugging Face 사용자명 (예: `user`)
- Space 이름 (예: `ebrcs-checkout`)

### 2단계: Git 인증

Push 시 로그인 요청이 나타나면:
- Username: `your_hf_username`
- Password: `hf_xxxxxxxxx` (토큰 사용)

### 3단계: HF_TOKEN 설정

1. Space 페이지 이동: `https://huggingface.co/spaces/yourusername/ebrcs-checkout`
2. **Settings** 클릭
3. **Variables and secrets** 클릭
4. **New secret** 클릭
   - **Name**: `HF_TOKEN`
   - **Value**: 당신의 Hugging Face 읽기 토큰
     ```
     hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```
5. **Save** 클릭
6. Space 자동 재시작 대기 (~2-3분)

### 4단계: GPU 활성화 (선택 사항, 성능 향상)

1. Settings → **Hardware**
2. **T4 small** 선택 (무료)
3. **Change hardware** 클릭

---

## 🚀 방법 2: 수동 배포

### 1단계: Hugging Face Space 생성

1. https://huggingface.co/new-space 접속
2. 정보 입력:
   - **Owner**: 본인 계정
   - **Space name**: `ebrcs-checkout`
   - **License**: MIT
   - **Select the Space SDK**: **Streamlit**
   - **Space hardware**: T4 small (무료 GPU)
   - **Space visibility**: Public

3. **Create Space** 클릭

### 2단계: 로컬에서 Git 클론

```bash
# Space Git 저장소 클론
git clone https://huggingface.co/spaces/yourusername/ebrcs-checkout
cd ebrcs-checkout

# Git LFS 초기화
git lfs install
```

### 3단계: 파일 복사

```bash
# Streamlit 앱
cp ../EBRCS_streaming/app.py .

# 코어 로직
cp -r ../EBRCS_streaming/checkout_core .

# 페이지
cp -r ../EBRCS_streaming/pages .
cp -r ../EBRCS_streaming/pages_mobile .

# UI
cp ../EBRCS_streaming/ui_theme.py .
cp ../EBRCS_streaming/mobile_nav.py .

# 데이터 (Git LFS 자동 처리)
cp -r ../EBRCS_streaming/data .

# 패키지
cp ../EBRCS_streaming/requirements.txt .

# Space 설정
cp ../EBRCS_streaming/README_HF_SPACE.md README.md
cp ../EBRCS_streaming/.gitattributes .
```

### 4단계: README.md 헤더 확인

`README.md` 파일 맨 위에 다음이 있는지 확인:

```yaml
---
title: EBRCS Smart Checkout
emoji: 🛒
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---
```

### 5단계: Git Push

```bash
git add .
git commit -m "Deploy EBRCS checkout system"
git push
```

### 6단계: HF_TOKEN 설정

Space 페이지 → Settings → Variables and secrets → New secret:
- **Name**: `HF_TOKEN`
- **Value**: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

## 🔐 HF_TOKEN 처리 방법

### Option A: Space Secrets (추천 ⭐)

**장점:** 안전, 코드에 노출 안 됨

1. Space Settings → Variables and secrets
2. New secret:
   - Name: `HF_TOKEN`
   - Value: 당신의 토큰
3. 코드에서 자동 사용:
   ```python
   # checkout_core/inference.py는 자동으로 secrets에서 가져옴
   token = st.secrets.get("HF_TOKEN")
   ```

### Option B: .streamlit/secrets.toml (로컬 테스트용)

**주의:** Private Space에서만 사용

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
EOF

# .gitignore에 추가 (토큰 노출 방지)
echo ".streamlit/secrets.toml" >> .gitignore
```

---

## 🎨 Space 커스터마이징

### 1. 앱 아이콘 변경

README.md 헤더 수정:
```yaml
emoji: 🛒  # 원하는 이모지로 변경
```

### 2. 테마 색상 변경

```yaml
colorFrom: orange  # 시작 색상
colorTo: red       # 끝 색상
```

### 3. Space 고정 (Pinned)

```yaml
pinned: true  # 프로필 상단에 고정
```

---

## 🐛 트러블슈팅

### 1. Git LFS 에러

**문제:**
```
Error: this repository is over its data quota
```

**해결:**
```bash
# Git LFS 재설치
git lfs uninstall
git lfs install
git lfs track "*.npy"
git lfs track "*.bin"
git add .gitattributes
git commit -m "Setup Git LFS"
git push
```

### 2. 모델 로딩 실패

**문제:**
```
OSError: You are trying to access a gated repo
```

**해결:**
1. Space Settings → Secrets에 `HF_TOKEN` 추가
2. Hugging Face에서 모델 접근 권한 요청:
   - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
   - "Request access" 클릭

### 3. OOM (Out of Memory)

**문제:**
```
Killed
```

**해결:**
1. Settings → Hardware → T4 small 선택
2. 또는 requirements.txt에서 `faiss-cpu` 대신 `faiss-gpu` 사용

### 4. Space 빌드 실패

**문제:**
```
ERROR: Could not find a version that satisfies...
```

**해결:**
`requirements.txt`에서 버전 고정 제거:
```diff
- fastapi>=0.115.0
+ fastapi
```

---

## 📊 배포 후 확인 사항

### 1. Space 상태 확인

Space 페이지에서:
- ✅ **Running** - 정상 작동 중
- ⚠️ **Building** - 빌드 중 (1-3분 대기)
- ❌ **Runtime error** - 로그 확인 필요

### 2. 로그 확인

Space 페이지 하단 **Logs** 탭:
```
Loading models...
✓ DINOv3 loaded
✓ CLIP loaded
✓ FAISS index loaded
Streamlit app running at port 7860
```

### 3. 성능 테스트

1. 앱 접속
2. "체크아웃 시작" 클릭
3. 카메라 또는 영상 업로드
4. 추론 속도 확인:
   - CPU: 2-5초/프레임
   - T4 GPU: 200-500ms/프레임

---

## 🚀 고급 설정

### 1. 자동 재시작 비활성화

Space가 15분 idle 후 sleep되지 않게:
- Settings → Hardware → **Always on** (유료, $0.60/hr)

### 2. Private Space

민감한 데이터가 있는 경우:
- Settings → Visibility → **Private**

### 3. Duplicate Space

다른 사용자가 복제할 수 있게:
- Space 카드에 "Duplicate this Space" 버튼 자동 생성

---

## 📞 도움말

- Hugging Face 공식 문서: https://huggingface.co/docs/hub/spaces
- Streamlit Spaces 가이드: https://huggingface.co/docs/hub/spaces-sdks-streamlit
- Git LFS 문서: https://git-lfs.github.com

---

## ✅ 배포 완료!

Space URL: `https://huggingface.co/spaces/yourusername/ebrcs-checkout`

**다음 단계:**
- [ ] README 업데이트 (사용법 추가)
- [ ] 스크린샷 추가
- [ ] 데모 영상 업로드
- [ ] 커뮤니티 공유
