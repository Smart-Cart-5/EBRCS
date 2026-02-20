# DB Workflow (MySQL)

`products`, `product_prices` 시드를 팀 단위로 공유/적용하기 위한 문서입니다.

원칙:
- 스키마(테이블 구조): Git 포함
- 시드 덤프(`.sql`, `.sql.gz`): Git 제외 (`db/seeds/*`)

## 1) 로컬 DB 시작

### macOS / Linux

```bash
./db/start_local_mysql.sh
```

### Windows (CMD/PowerShell)

```bat
db\start_local_mysql.bat
```

기본 `DATABASE_URL`:

```env
DATABASE_URL=mysql+pymysql://ebrcs_app:ebrcs_pass@127.0.0.1:3307/item_db
```

종료:

- macOS / Linux: `./db/stop_local_mysql.sh`
- Windows: `db\stop_local_mysql.bat`

데이터까지 삭제:

- macOS / Linux: `./db/stop_local_mysql.sh --purge`
- Windows: `db\stop_local_mysql.bat --purge`

## 2) 스키마 준비

- macOS / Linux: `cd app && ./setup_db.sh`
- Windows: `cd app && setup_db.bat`

## 3) 시드 가져오기 (개발 환경)

- macOS / Linux:

```bash
./db/import_price_seed.sh --seed ./db/seeds/price_seed_latest.sql.gz
```

- Windows:

```bat
db\import_price_seed.bat --seed ./db/seeds/price_seed_latest.sql.gz
```

기본은 replace 모드(`products`, `product_prices` 비우고 적재)입니다.

append 모드:

- macOS / Linux: `./db/import_price_seed.sh --seed <file> --append`
- Windows: `db\import_price_seed.bat --seed <file> --append`

## 4) 시드 내보내기 (운영/공유 DB)

- macOS / Linux: `./db/export_price_seed.sh`
- Windows: `db\export_price_seed.bat`

## 5) 최종 검증

- macOS / Linux:

```bash
cd app
./setup_db.sh --check
./run_web.sh
```

- Windows:

```bat
cd app
setup_db.bat --check
run_web.bat
```
