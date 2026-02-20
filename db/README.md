# DB Seed Workflow

가격 DB(`products`, `product_prices`)를 팀 단위로 재현하기 위한 스크립트 모음입니다.

원칙:
- 스키마(테이블 구조)는 Git 포함
- 실데이터(가격 덤프)는 Git 미포함

## ✅ 체크리스트

- [ ] `cd app && ./setup_db.sh`로 스키마 준비
- [ ] 운영 DB(EC2)에서 `./db/export_price_seed.sh`로 시드 덤프 생성
- [ ] 시드 파일(`.sql`/`.sql.gz`)을 팀 스토리지(S3/Drive/사내 저장소)에 업로드
- [ ] 각 개발자는 DB 타겟 선택
- [ ] 선택한 DB에 `./db/import_price_seed.sh --seed <file>` 실행
- [ ] `cd app && ./setup_db.sh --check`로 필수 테이블 확인 후 실행

## 1) 시드 내보내기 (운영 DB/EC2)

```bash
cd /path/to/EBRCS
./db/export_price_seed.sh
```

기본 출력 경로:
- `db/seeds/price_seed_<YYYYMMDD_HHMMSS>.sql.gz`

출력 경로 지정:

```bash
./db/export_price_seed.sh --output ./db/seeds/price_seed_latest.sql.gz
```

## 2) 시드 가져오기 (개발자 환경)

```bash
cd /path/to/EBRCS
./db/import_price_seed.sh --seed ./db/seeds/price_seed_latest.sql.gz
```

기본 동작:
- `products`, `product_prices`를 비우고(replace) 시드 재적재

추가 적재 모드:

```bash
./db/import_price_seed.sh --seed ./db/seeds/price_seed_latest.sql.gz --append
```

## 3) DB 타겟 선택

### A. EC2 공유 DB 사용

- `.env`의 `DATABASE_URL`을 EC2 MySQL로 설정
- 주의: EC2(또는 DB 인스턴스)가 꺼져 있으면 연결 불가

### B. 로컬 Docker MySQL 사용 (권장)

```bash
cd /path/to/EBRCS
./db/start_local_mysql.sh
```

`.env` 예시:

```env
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db
```

종료:

```bash
./db/stop_local_mysql.sh
```

데이터까지 삭제:

```bash
./db/stop_local_mysql.sh --purge
```

## 4) 검증 및 실행

```bash
cd app
./setup_db.sh --check
./run_web.sh
```
