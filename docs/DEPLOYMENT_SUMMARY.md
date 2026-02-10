# GitHub Pages 배포 요약 (빠른 시작)

## 🎯 3가지 선택지

### 1️⃣ Hugging Face Spaces (추천 ⭐⭐⭐⭐⭐)

**장점:** GPU 무료, 빠름, 간단
**단점:** Streamlit UI (React 웹앱 아님)

```bash
# 1. https://huggingface.co 가입
# 2. New Space 생성 (SDK: Streamlit, Hardware: T4 small)
# 3. Git push
git clone https://huggingface.co/spaces/yourusername/ebrcs-checkout
cd ebrcs-checkout
cp ../EBRCS_streaming/app.py .
cp -r ../EBRCS_streaming/checkout_core .
cp -r ../EBRCS_streaming/data .
cp ../EBRCS_streaming/requirements.txt .
git add .
git commit -m "Deploy"
git push
```

**접속:** `https://huggingface.co/spaces/yourusername/ebrcs-checkout`

---

### 2️⃣ GitHub Pages + Render (무료, 느림)

**장점:** React 웹앱 그대로 사용
**단점:** CPU only (추론 2-5초/프레임), 15분 idle sleep

#### A. 백엔드 배포 (Render.com)

```bash
# 1. https://render.com 가입
# 2. New Web Service
#    - Runtime: Docker
#    - Branch: main
#    - Instance Type: Free
# 3. Environment Variables 추가:
#    HF_TOKEN=your_huggingface_token
# 4. 배포 완료 후 URL 복사: https://ebrcs-api.onrender.com
```

#### B. Frontend 배포 (GitHub Pages)

```bash
# 1. 백엔드 URL 설정
echo "VITE_API_BASE_URL=https://ebrcs-api.onrender.com" > frontend/.env.production

# 2. GitHub 저장소 Settings → Pages
#    Source: Deploy from a branch
#    Branch: gh-pages

# 3. Push (GitHub Actions 자동 배포)
git add .
git commit -m "Setup GitHub Pages"
git push origin main
```

**접속:** `https://yourusername.github.io/EBRCS_streaming/`

---

### 3️⃣ Google Colab (발표/데모용)

**장점:** GPU 무료, 빠름
**단점:** 12시간 제한, 수동 실행

```python
# Colab 노트북에서 실행:

# 1. 저장소 클론
!git clone https://github.com/yourusername/EBRCS_streaming.git
%cd EBRCS_streaming

# 2. 백엔드 실행
!pip install -r requirements.txt
!nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 3. ngrok 터널
!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("your_token")  # https://dashboard.ngrok.com
public_url = ngrok.connect(8000)
print(f"🚀 접속 URL: {public_url}")
```

---

## 🚨 현재 작업 완료 상태

### ✅ 완료된 작업
- [x] Frontend API URL 환경 변수 지원 ([client.ts](../frontend/src/api/client.ts))
- [x] GitHub Actions 워크플로우 ([.github/workflows/deploy.yml](../.github/workflows/deploy.yml))
- [x] Vite base path 설정 ([vite.config.ts](../frontend/vite.config.ts))
- [x] 배포 가이드 문서 ([DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md))

### 📝 다음 단계 (사용자가 선택)

#### Option A: Hugging Face Spaces 배포
1. HF 계정 생성
2. New Space 생성 (T4 GPU)
3. `app.py` 업로드

#### Option B: GitHub Pages + Render
1. Render.com에 백엔드 배포
2. `frontend/.env.production`에 URL 추가
3. GitHub에 push (자동 배포)

#### Option C: Colab 데모
1. Colab 노트북 생성
2. 위 스크립트 실행
3. 발표 시에만 사용

---

## 📚 상세 문서

전체 설명은 [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) 참고

## 🆘 문제 해결

| 문제 | 해결 |
|------|------|
| Render OOM | DINOv2-base 사용 (경량화) |
| CORS 에러 | backend CORS 설정 확인 |
| GitHub Pages 404 | base path 확인 (`/EBRCS_streaming/`) |
| Render sleep | 15분마다 ping 또는 paid plan |

---

**다음 작업:** 위 3가지 옵션 중 하나 선택하여 배포 진행
