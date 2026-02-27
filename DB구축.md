# DB 구축 및 로컬 실행 파이프라인

이 문서는 팀원이 저장소를 clone한 뒤, EC2 서버와 동일한 DB 상태로 로컬에서 웹앱을 실행하는 절차를 정리합니다.

## 목표

- `db/seeds/full_seed_latest.sql.gz` 하나만 있으면
- 로컬 Docker MySQL + 로컬 웹앱으로 EC2와 동일한 상태 재현 가능

## 시드 파일에 포함된 테이블

| 테이블 | 설명 |
|---|---|
| `products` | 상품 목록 |
| `product_prices` | 상품 가격 |
| `product_discounts` | 할인 정보 |
| `store_corners` | 매장 코너 레이아웃 |
| `category_corner_map` | 카테고리 → 코너 매핑 |
| `users` | 사용자 계정 (smoke test 계정 제외) |
| `purchase_history` | 구매 이력 |

## Django 사용자 참고

- Django `makemigrations`처럼 마이그레이션 파일을 자동 생성하지 않습니다.
- 이 저장소에서 `migrate` 역할은 `cd app && ./setup_db.sh`입니다.
- 모델 스키마를 바꿀 때는 `app/backend/models.py`와 `app/backend/db_bootstrap.py`를 함께 수정해야 합니다.

## 사전 준비

- Git
- Docker + Docker Compose
- Python 3.11+
- Node.js 20.19+

## 로컬 초기 설정 (최초 1회)

### 1) 저장소 clone

```bash
git clone https://github.com/Smart-Cart-5/EBRCS.git
cd EBRCS
```

### 2) 환경 파일 준비

```bash
cp .env.example .env
```

`.env`에서 `DATABASE_URL`을 Docker 로컬 MySQL로 설정 (DB명 `mydb`는 EC2와 동일):

```env
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/mydb
```

### 3) 로컬 MySQL 컨테이너 기동

```bash
./db/start_local_mysql.sh
```

### 4) 앱 환경 + DB 스키마 생성

```bash
cd app
./setup_venv.sh
./setup_db.sh
cd ..
```

### 5) 전체 시드 임포트

```bash
./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz
```

### 6) 웹앱 실행

```bash
cd app
./run_web.sh
```

접속:
- Frontend: `http://localhost:5173`
- Backend Docs: `http://127.0.0.1:8000/docs`
- **Adminer (DB 웹 UI)**: `http://localhost:8081`
  - System: MySQL
  - Server: `mysql`
  - Username: `ebrcs_app`
  - Password: `ebrcs_pass`
  - Database: `mydb`

---

## EC2에서 최신 시드 내보내기 (데이터 업데이트 시)

EC2 서버에서 실행:

```bash
# 전체 시드 내보내기 (smoke test 계정 자동 제외)
./db/export_full_seed.sh --output ./db/seeds/full_seed_latest.sql.gz

# 커밋 후 push
git add db/seeds/full_seed_latest.sql.gz
git commit -m "db: update full seed"
git push
```

로컬에서 최신 시드 반영:

```bash
git pull
./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz
```

---

## 종료

웹앱 종료:

```bash
cd app
./stop_web.sh
```

MySQL 컨테이너 종료 (데이터 보존):

```bash
./db/stop_local_mysql.sh
```

데이터까지 완전 삭제 후 재시작하려면:

```bash
./db/stop_local_mysql.sh --purge
./db/start_local_mysql.sh
cd app && ./setup_db.sh && cd ..
./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz
```

---

## 스크립트 참고

| 스크립트 | 용도 |
|---|---|
| `db/export_full_seed.sh` | EC2 → 시드 파일로 전체 7개 테이블 내보내기 |
| `db/import_full_seed.sh` | 시드 파일 → 로컬 MySQL로 전체 7개 테이블 가져오기 |
| `db/start_local_mysql.sh` | Docker MySQL 컨테이너 시작 |
| `db/stop_local_mysql.sh` | Docker MySQL 컨테이너 종료 |
| `app/setup_db.sh` | DB 스키마 생성 (테이블 없으면 생성) |
