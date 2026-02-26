#!/usr/bin/env bash
# Import full DB seed (products, product_prices, product_discounts, users, purchase_history) into MySQL.
#
# Usage:
#   ./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz
#   ./db/import_full_seed.sh --seed ./db/seeds/full_seed.sql --append
#   ./db/import_full_seed.sh --seed ./db/seeds/full_seed_latest.sql.gz --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SEED_FILE=""
APPEND_MODE="false"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Import EBRCS full DB seed into MySQL.

Options:
  --seed <path>     Input seed file (.sql or .sql.gz).
  --append          Keep existing data and append imported rows.
                    Default: truncates all 5 tables before importing.
  --dry-run         Print resolved DB target and command without executing.
  -h, --help        Show this help.

Tables affected (child → parent truncate order):
  purchase_history, users, product_discounts, product_prices, products
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --seed)
            [ "$#" -lt 2 ] && { echo "❌ --seed requires a value."; exit 1; }
            SEED_FILE="$2"; shift 2 ;;
        --append)
            APPEND_MODE="true"; shift ;;
        --dry-run)
            DRY_RUN="true"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "❌ Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ -z "$SEED_FILE" ]; then
    echo "❌ --seed is required."; usage; exit 1
fi

if [[ "$SEED_FILE" != /* ]]; then
    SEED_FILE="$PROJECT_ROOT/$SEED_FILE"
fi

if [ ! -f "$SEED_FILE" ]; then
    echo "❌ Seed file not found: $SEED_FILE"; exit 1
fi

if ! command -v mysql >/dev/null 2>&1; then
    echo "❌ mysql client not found. Install MySQL client first."; exit 1
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ python/python3 not found."; exit 1
fi

if [[ "$SEED_FILE" == *.gz ]] && ! command -v gzip >/dev/null 2>&1; then
    echo "❌ gzip not found but seed file is .gz."; exit 1
fi

# ── Parse DATABASE_URL from .env ──────────────────────────────────────────────
parse_output="$(
PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import os, sys
from pathlib import Path
from urllib.parse import urlparse, unquote

project_root = Path(os.environ["PROJECT_ROOT"])
env_path = project_root / ".env"
if env_path.exists():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value

database_url = os.getenv("DATABASE_URL", "").strip()
if not database_url:
    print("❌ DATABASE_URL is not set.", file=sys.stderr); sys.exit(2)

if database_url.startswith("mysql://"):
    database_url = "mysql+pymysql://" + database_url[len("mysql://"):]

parsed = urlparse(database_url)
scheme = parsed.scheme.lower()
if not scheme.startswith("mysql"):
    print(f"❌ DATABASE_URL backend must be mysql, got: {scheme}", file=sys.stderr); sys.exit(3)

db_name = (parsed.path or "").lstrip("/")
if not db_name:
    print("❌ DATABASE_URL must include database name.", file=sys.stderr); sys.exit(4)

db_user = unquote(parsed.username or "")
if not db_user:
    print("❌ DATABASE_URL must include username.", file=sys.stderr); sys.exit(5)

db_password = unquote(parsed.password or "")
db_host = parsed.hostname or "127.0.0.1"
db_port = parsed.port or 3306

print(f"DB_HOST={db_host}")
print(f"DB_PORT={db_port}")
print(f"DB_USER={db_user}")
print(f"DB_PASSWORD={db_password}")
print(f"DB_NAME={db_name}")
PY
)"

while IFS= read -r line; do
    [ -n "$line" ] && export "$line"
done <<< "$parse_output"

mysql_cmd=(
    mysql
    "--host=$DB_HOST"
    "--port=$DB_PORT"
    "--user=$DB_USER"
)

echo "Resolved target DB: mysql://$DB_USER@${DB_HOST}:${DB_PORT}/$DB_NAME"
echo "Seed file: $SEED_FILE"
echo "Import mode: $( [ "$APPEND_MODE" = "true" ] && echo "append" || echo "replace (truncate all tables first)" )"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "Dry-run — would run:"
    echo "  1. CREATE DATABASE IF NOT EXISTS \`$DB_NAME\`"
    if [ "$APPEND_MODE" != "true" ]; then
        echo "  2. TRUNCATE purchase_history, users, product_discounts, product_prices, products"
    fi
    if [[ "$SEED_FILE" == *.gz ]]; then
        echo "  3. gzip -dc $SEED_FILE | mysql $DB_NAME"
    else
        echo "  3. mysql $DB_NAME < $SEED_FILE"
    fi
    exit 0
fi

# ── Ensure DB exists ──────────────────────────────────────────────────────────
MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" -e \
    "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# ── Check schema is present ───────────────────────────────────────────────────
required_count="$(
    MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" -Nse \
    "SELECT COUNT(*) FROM information_schema.tables
     WHERE table_schema='$DB_NAME'
     AND table_name IN ('products','product_prices','product_discounts','users','purchase_history');"
)"

if [ "${required_count:-0}" -lt 5 ]; then
    echo "❌ Required tables are missing in $DB_NAME."
    echo "   Run: cd app && ./setup_db.sh"
    exit 1
fi

# ── Optionally truncate (FK-safe order: child → parent) ───────────────────────
if [ "$APPEND_MODE" != "true" ]; then
    echo "Truncating tables..."
    MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" "$DB_NAME" -e \
        "SET FOREIGN_KEY_CHECKS=0;
         TRUNCATE TABLE purchase_history;
         TRUNCATE TABLE users;
         TRUNCATE TABLE product_discounts;
         TRUNCATE TABLE product_prices;
         TRUNCATE TABLE products;
         SET FOREIGN_KEY_CHECKS=1;"
fi

# ── Import seed ───────────────────────────────────────────────────────────────
echo "Importing seed data..."
if [[ "$SEED_FILE" == *.gz ]]; then
    gzip -dc "$SEED_FILE" | MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" "$DB_NAME"
else
    MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" "$DB_NAME" < "$SEED_FILE"
fi

echo "✅ Full seed imported."
echo ""
echo "Row counts after import:"
MYSQL_PWD="$DB_PASSWORD" "${mysql_cmd[@]}" "$DB_NAME" -e \
    "SELECT 'products'          AS tbl, COUNT(*) AS cnt FROM products
     UNION ALL SELECT 'product_prices',   COUNT(*) FROM product_prices
     UNION ALL SELECT 'product_discounts',COUNT(*) FROM product_discounts
     UNION ALL SELECT 'users',            COUNT(*) FROM users
     UNION ALL SELECT 'purchase_history', COUNT(*) FROM purchase_history;"
