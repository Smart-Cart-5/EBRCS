# 🚀 GitHub Pages + 무료 백엔드 배포 가이드

## 📋 배포 구조

```
┌───────────────────────────┐         ┌────────────────────────────┐
│ GitHub Pages (무료)        │         │ 백엔드 서버 (선택)          │
│ https://username.github.io│ ─────>  │ https://your-api.com       │
│ /EBRCS_streaming/         │  API    │                            │
│                           │         │ FastAPI + PyTorch + FAISS  │
│ React Frontend (정적)      │         │                            │
└───────────────────────────┘         └────────────────────────────┘
     무료 ✅                                유료 or 제한적 무료
```

---

## 🎯 Option 1: GitHub Pages + Render.com (무료) ⭐ 권장

### 백엔드 배포: Render.com

#### 1. Render.com 계정 생성
- https://render.com 가입 (GitHub 계정 연동)

#### 2. New Web Service 생성
```
Name: ebrcs-api
Runtime: Docker
Region: Oregon (US West)
Branch: main
Docker Command: (기본값 사용)

Environment Variables:
  - HF_TOKEN: your_huggingface_token
  - HUGGINGFACE_HUB_TOKEN: your_huggingface_token

Instance Type: Free
```

#### 3. 배포 URL 확인
- 배포 완료 후 URL: `https://ebrcs-api.onrender.com`
- 첫 배포는 10-15분 소요 (모델 다운로드)

⚠️ **Render 무료 티어 제한:**
- 15분 idle 시 sleep (재접속 시 ~1분 wake-up)
- CPU only (GPU 없음) → 추론 느림 (프레임당 2-5초)
- 메모리: 512MB → 모델 로딩 시 OOM 가능

#### 4. Frontend 환경 변수 업데이트

`frontend/.env.production` 수정:
```bash
VITE_API_BASE_URL=https://ebrcs-api.onrender.com
```

### Frontend 배포: GitHub Pages

#### 1. Repository Settings
GitHub 저장소 → Settings → Pages:
- Source: Deploy from a branch
- Branch: `gh-pages` / (root)

#### 2. GitHub Actions 배포 스크립트

`.github/workflows/deploy.yml` 생성:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        working-directory: frontend
        run: npm ci

      - name: Build
        working-directory: frontend
        run: npm run build
        env:
          VITE_API_BASE_URL: https://ebrcs-api.onrender.com

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

#### 3. Base URL 설정

`frontend/vite.config.ts` 수정:
```typescript
export default defineConfig({
  base: '/EBRCS_streaming/',  // 저장소 이름과 동일
  plugins: [react()],
  // ...
})
```

#### 4. 배포
```bash
git add .
git commit -m "Setup GitHub Pages deployment"
git push origin main
```

배포 완료 후 접속: `https://username.github.io/EBRCS_streaming/`

---

## 🎯 Option 2: Hugging Face Spaces (GPU 무료) ⭐⭐ 최고 성능

### 장점
- ✅ 무료 GPU (NVIDIA T4)
- ✅ 빠른 추론 (프레임당 200-500ms)
- ✅ 공개 링크 자동 생성

### 단점
- ❌ Streamlit/Gradio UI만 지원 (React 불가)
- ❌ 15분 idle 시 sleep

### 배포 방법

#### 1. Hugging Face 계정 생성
- https://huggingface.co 가입

#### 2. New Space 생성
```
Name: ebrcs-checkout
SDK: Streamlit
Hardware: T4 small (GPU, 무료)
```

#### 3. 파일 업로드
Space Git 저장소에 push:
```bash
# Space clone
git clone https://huggingface.co/spaces/yourusername/ebrcs-checkout
cd ebrcs-checkout

# 기존 Streamlit 파일 복사
cp ../EBRCS_streaming/app.py .
cp ../EBRCS_streaming/mobile_app.py .
cp -r ../EBRCS_streaming/checkout_core .
cp -r ../EBRCS_streaming/data .
cp -r ../EBRCS_streaming/pages .
cp ../EBRCS_streaming/requirements.txt .

# README.md 작성 (Space 설명)
cat > README.md <<EOF
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

# EBRCS 스마트 체크아웃 시스템

AI 기반 실시간 상품 인식 체크아웃 데모
EOF

# Push
git add .
git commit -m "Initial deployment"
git push
```

#### 4. 접속
- URL: `https://huggingface.co/spaces/yourusername/ebrcs-checkout`
- GPU 로딩: 첫 접속 시 ~2-3분

---

## 🎯 Option 3: Google Colab (임시 데모용)

### 사용 시나리오
- 발표/시연 시에만 켜기
- 12시간 세션 제한

### 배포 스크립트

`deploy_colab.ipynb` 생성:
```python
# 1. 저장소 클론
!git clone https://github.com/yourusername/EBRCS_streaming.git
%cd EBRCS_streaming

# 2. 패키지 설치
!pip install -r requirements.txt

# 3. 백엔드 실행 (백그라운드)
!nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 4. ngrok 설치 및 실행
!pip install pyngrok
from pyngrok import ngrok

# ngrok 토큰 설정 (https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok.set_auth_token("your_ngrok_token")

# 터널 생성
public_url = ngrok.connect(8000)
print(f"🚀 백엔드 URL: {public_url}")

# 5. Frontend 빌드 및 실행
%cd frontend
!npm install
!npm run build

# Vite preview 서버 실행
!npx vite preview --host 0.0.0.0 --port 3000 &

# Frontend 터널
public_frontend = ngrok.connect(3000)
print(f"🌐 Frontend URL: {public_frontend}")
```

**실행:**
1. Google Colab에서 노트북 열기
2. Runtime → Change runtime type → GPU 선택
3. 셀 실행
4. 출력된 URL 공유

---

## 📊 비교표

| 옵션 | Frontend | Backend | GPU | 속도 | 비용 | 난이도 |
|------|----------|---------|-----|------|------|--------|
| **Render + GitHub Pages** | GitHub Pages | Render.com | ❌ | 느림 (2-5초/프레임) | 무료 | ⭐⭐☆ |
| **Hugging Face Spaces** | Streamlit | HF Spaces | ✅ T4 | 빠름 (200-500ms/프레임) | 무료 | ⭐☆☆ |
| **Google Colab** | Colab | Colab | ✅ T4 | 빠름 | 무료 (12h 제한) | ⭐⭐⭐ |

---

## 🛠️ 트러블슈팅

### Render.com OOM (Out of Memory)

**문제:** 모델 로딩 시 메모리 부족
```
MemoryError: Unable to allocate array
```

**해결:**
1. DINOv3 모델을 경량 버전으로 변경:
   ```python
   # checkout_core/inference.py
   DINO_MODEL_NAME = "facebook/dinov2-base"  # vitl16 → base (1.5GB → 300MB)
   ```

2. 또는 Railway.app ($5 크레딧) 사용

### GitHub Pages CORS 에러

**문제:**
```
Access to fetch at 'https://ebrcs-api.onrender.com/api/sessions'
from origin 'https://username.github.io' has been blocked by CORS
```

**해결:**
Backend CORS 설정 확인 ([backend/main.py:81-87](../backend/main.py#L81-L87)):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["https://username.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Hugging Face Spaces Sleep

**문제:** 15분 idle 후 sleep, 재접속 시 1분+ 대기

**해결:**
1. Space Settings → Hardware → Always on (Paid, $0.60/hr)
2. 또는 무료로 사용하고 wake-up 기다리기

---

## 🎓 최종 추천

### 프로젝트 발표용
→ **Hugging Face Spaces** (GPU 무료, 빠름, Streamlit UI)

### React 웹앱 공개용
→ **GitHub Pages + Render** (느리지만 무료)

### 데모 시연용
→ **Google Colab** (12시간 제한, 발표 당일만 켜기)

---

## 📚 참고 자료

- [GitHub Pages 공식 문서](https://docs.github.com/en/pages)
- [Render.com 무료 티어](https://docs.render.com/free)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [ngrok 가이드](https://ngrok.com/docs/getting-started)
