# 임베딩 기반 리테일 체크아웃 시스템

DINOv3 + CLIP 임베딩과 FAISS 검색을 결합해, 웹캠으로 들어오는 상품을 실시간 인식/집계하는 Streamlit 앱입니다.

## 주요 기능
- `Add Product`: 상품명 + 이미지(1~3장)로 임베딩 DB 등록
- `Checkout`: 웹캠 기반 실시간 인식, ROI(관심영역) 설정, 수량 집계
- `Validate Bill`: 인식된 장바구니 항목 검수/수정

## 아키텍처 요약
1. OpenCV 배경차분으로 움직이는 물체를 탐지
2. 탐지된 영역에서 DINOv3 + CLIP 임베딩 생성
3. 가중 결합 임베딩(`DINO 0.7`, `CLIP 0.3`)으로 FAISS 최근접 검색
4. 임계값 이상 매칭 결과만 카운트

## 저장소 구조
```text
Embedding-Based-Retail-Checkout-System/
├─ app.py
├─ pages/
│  ├─ 1_Add_Product.py
│  ├─ 2_Checkout.py
│  └─ 3_Validate_Bill.py
├─ data/
│  ├─ adapter_config.json
│  ├─ adapter_model.safetensors   # git ignore
│  ├─ embeddings.npy              # git ignore
│  ├─ labels.npy                  # git ignore
│  └─ faiss_index.bin             # git ignore
├─ requirements.txt
├─ run.sh
├─ .env.example
└─ .gitignore
```

## 요구 사항
- macOS/Linux (Python 3.10+ 권장)
- 웹캠
- 인터넷 연결 (최초 실행 시 Hugging Face 모델 다운로드)

## 빠른 시작 (처음부터)
### 1) 저장소 클론 및 진입
```bash
git clone <your-repo-url>
cd Embedding-Based-Retail-Checkout-System
```

### 2) 가상환경 생성/활성화
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) 환경변수 설정 (`.env`)
아래처럼 예시 파일을 복사한 뒤 토큰 값을 채워주세요.

```bash
cp .env.example .env
```

`.env` 파일 내용:

```bash
HF_TOKEN=your_hf_token
HUGGINGFACE_HUB_TOKEN=your_hf_token
```

- 둘 중 하나만 있어도 동작합니다.
- private/gated 모델 접근 이슈를 줄이려면 둘 다 동일 값으로 넣는 것을 권장합니다.

### 5) 실행
```bash
./run.sh
```
또는
```bash
streamlit run app.py
```

## 실행 후 사용 순서 (중요)
1. `Add Product` 페이지에서 상품을 먼저 1개 이상 등록합니다.
2. `Checkout` 페이지에서 실시간 인식을 시작합니다.
3. 필요 시 사이드바에서 ROI를 설정해 특정 영역 진입 이벤트만 카운트합니다.
4. `Validate Bill` 페이지에서 수량 수정/결제 확정을 진행합니다.

## 데이터/모델 파일 정책 (.gitignore 반영)
GitHub에 올리지 않도록 설정된 파일:
- `.env`, `.env.*`, `.streamlit/secrets.toml`
- `.venv/`, `__pycache__/` 등 로컬 캐시
- `faiss_index.bin`
- `data/embeddings.npy`
- `data/labels.npy`
- `data/faiss_index.bin`
- `data/adapter_model.safetensors`

즉, 임베딩 DB/FAISS 인덱스/대용량 LoRA 가중치는 로컬에서 생성·관리하고 저장소에는 코드 중심으로 올립니다.

## GitHub 업로드 전 체크리스트
```bash
git status
```
확인 포인트:
- 위 ignore 대상 파일이 `Changes to be committed`에 없어야 함
- 코드/문서 파일(`app.py`, `pages/`, `README.md`, `.gitignore` 등)만 올라가야 함

이미 추적 중인 대용량 파일이 있다면(예: 루트 `faiss_index.bin`):
```bash
git rm --cached faiss_index.bin
```

## 문제 해결
- `Embedding load error: embeddings.npy not found.`
  - `Add Product`에서 상품을 먼저 등록해 `data/embeddings.npy`, `data/labels.npy`를 생성하세요.

- `카메라를 열 수 없습니다.`
  - 다른 앱에서 카메라를 점유 중인지 확인하고, OS 카메라 권한을 허용하세요.

- `DINO LoRA 로드 실패 (베이스 모델 사용)`
  - `data/adapter_model.safetensors`가 없거나 불일치한 경우입니다.
  - 없어도 베이스 모델로 동작합니다.

## 참고
- 이 프로젝트는 Hugging Face 모델/가중치를 사용하므로, 사용 전 각 모델 라이선스를 확인하세요.
