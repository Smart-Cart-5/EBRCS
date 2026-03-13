# 장보GO — 로컬 실행 가이드

> `git clone` 후 이 문서만 보고 웹앱을 띄울 수 있도록 정리된 가이드입니다.

---

## 사전 요구사항

| 도구 | 버전 | 확인 명령 |
|------|------|----------|
| Python | 3.11+ | `python3 --version` |
| Node.js | 20.19.0+ | `node --version` |
| Git | 2.0+ | `git --version` |

> **Docker Desktop**은 `setup_all` 스크립트가 자동으로 설치하고 기동합니다.
> - macOS: Homebrew(`brew`)가 필요합니다. 없으면 [brew.sh](https://brew.sh) 먼저 설치.
> - Windows: `winget`이 필요합니다 (Windows 10 1709+ 기본 포함).

---

## 0단계 — data 파일 준비 (필수)

아래 파일들은 팀원에게 전달받아 프로젝트 루트 `data/` 폴더에 넣어주세요.

| 파일 | 설명 |
|------|------|
| `embeddings.npy` | 상품 임베딩 DB (필수) |
| `labels.npy` | 상품 라벨 (필수) |
| `adapter_model.safetensors` | LoRA 가중치 (선택, 없으면 기본 DINO로 동작) |
| `db/seeds/full_seed_latest.sql.gz` | DB 시드 데이터 (필수) |

> `adapter_config.json`과 `faiss_index.bin`은 자동 처리됩니다 (Git 포함 / 서버 시작 시 자동 생성).

---

## 1단계 — 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 항목 수정:

| 변수 | 설명 | 예시 |
|------|------|------|
| `HF_TOKEN` | HuggingFace 토큰 | `hf_xxxx...` |
| `SECRET_KEY` | JWT 비밀 키 | 아래 명령으로 생성 |
| `DATA_DIR` | `data/` 폴더 절대경로 **(필수)** | `/Users/yourname/EBRCS/data` |
| `DATABASE_URL` | DB 연결 (기본값 유지 권장) | 아래 참고 |
| `DB_VIEWER_USER` | Adminer API 로그인 ID | `admin` |
| `DB_VIEWER_PASSWORD` | Adminer API 로그인 PW | `admin` |

**SECRET_KEY 생성:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**DATABASE_URL (기본값 권장):**
```env
# 로컬 Docker MySQL — setup_all이 자동으로 띄워줌
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/mydb
```

---

## 2단계 — 원스텝 세팅 (최초 1회)

### 🍎 macOS / 🐧 Linux

```bash
cd app
./setup_all.sh
```

### 🪟 Windows (CMD)

```cmd
cd app
setup_all.bat
```

**내부 동작:**

```
[1/3] db/setup_local_db → Docker MySQL 기동(port 3307) + 스키마 생성 + 시드 import
[2/3] setup_venv        → Python venv 생성 + pip install + npm install
[3/3] setup_db          → DB 스키마 최종 검증
```

> `db/seeds/full_seed_latest.sql.gz` 가 없으면 시드 import는 스킵됩니다.
> 팀원에게 파일을 받아 `db/seeds/` 에 넣으세요.

---

## 3단계 — 웹앱 실행

### 🍎 macOS / 🐧 Linux

```bash
# app/ 디렉토리에서
./run_web.sh
```

### 🪟 Windows (CMD)

```cmd
run_web.bat
```

**접속 주소:**

| URL | 설명 |
|-----|------|
| http://localhost:5173 | 웹앱 (Frontend) |
| http://localhost:8000/docs | API 문서 (Swagger) |
| http://localhost:8081 | DB 뷰어 (Adminer) |

---

## DB 뷰어 — Adminer (localhost:8081)

Docker MySQL과 함께 자동으로 실행됩니다. 테이블 조회·수정·SQL 실행 가능.

| 항목 | 값 |
|------|-----|
| System | MySQL |
| Server | `mysql` |
| Username | `ebrcs_app` |
| Password | `ebrcs_pass` |
| Database | `mydb` |

---

## 관리자 계정 만들기

웹 UI에서 일반 계정 생성 후 아래 명령으로 role 변경:

### 🍎 macOS / 🐧 Linux

```bash
cd app
source backend/.venv/bin/activate
python -c "
from backend.database import SessionLocal
from backend import models
db = SessionLocal()
user = db.query(models.User).filter(models.User.username == 'admin').first()
if user:
    user.role = 'admin'; db.commit()
    print(f'✅ {user.username} → admin 변경 완료')
else:
    print('❌ 사용자 없음')
db.close()
"
```

### 🪟 Windows (CMD)

```cmd
cd app
backend\.venv\Scripts\activate
python -c "from backend.database import SessionLocal; from backend import models; db = SessionLocal(); user = db.query(models.User).filter(models.User.username == 'admin').first(); user.role = 'admin' if user else None; db.commit() if user else None; print('✅ 완료' if user else '❌ 없음'); db.close()"
```

---

## 공용 EC2 DB 사용 (팀 데이터 공유)

모든 팀원이 동일한 DB를 공유하고 싶을 때:

1. `.env`의 `DATABASE_URL`을 EC2 주소로 변경
2. `--skip-db` 옵션으로 Docker MySQL 기동 건너뜀

```bash
# macOS/Linux
cd app && ./setup_all.sh --skip-db && ./run_web.sh

# Windows
cd app && setup_all.bat --skip-db && run_web.bat
```

| 구분 | 로컬 Docker MySQL | 공용 EC2 MySQL |
|------|-----------------|--------------|
| `DATABASE_URL` | `127.0.0.1:3307/mydb` | `<EC2_IP>:3306/mydb` |
| `setup_all` 옵션 | (기본값) | `--skip-db` |
| Docker 필요 | ✅ | ❌ |
| 데이터 공유 | 개인 독립 | 팀 전체 공유 |

---

## 자주 쓰는 명령

```bash
# Docker MySQL만 중지
./db/stop_local_mysql.sh

# 시드 데이터 다시 import
./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz

# DB 스키마만 재검증
cd app && ./setup_db.sh
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `FileNotFoundError: embeddings.npy` | `DATA_DIR` 미설정 | `.env`에 `DATA_DIR` 절대경로 추가 |
| `SIGSEGV` (macOS 시작 즉시 종료) | OpenMP 충돌 | `run_web.sh`에 이미 처리됨 — `.env`에 넣지 말 것 |
| `Access denied for user 'ebrcs_app'` | Docker MySQL 볼륨 오염 | `docker volume rm ebrcs_mysql_data` 후 재실행 |
| `DB viewer credentials are not configured` | `.env`에 `DB_VIEWER_USER/PASSWORD` 없음 | `.env`에 두 변수 추가 |
| Docker MySQL 포트 3307 충돌 | 기존 컨테이너 실행 중 | `docker ps`로 확인 후 종료 |
