#!/usr/bin/env bash
# Import products/product_prices seed data into MySQL.
#
# Usage:
#   ./db/import_price_seed.sh --seed ./db/seeds/price_seed_20260220.sql.gz
#   ./db/import_price_seed.sh --seed ./db/seeds/price_seed.sql --append
#   ./db/import_price_seed.sh --seed ./db/seeds/price_seed.sql.gz --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SEED_FILE=""
APPEND_MODE="false"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Import EBRCS price seed into MySQL.

Options:
  --seed <path>     Input seed file (.sql or .sql.gz).
  --append          Keep existing products/product_prices and append imported rows.
                    Default behavior truncates products/product_prices first.
  --dry-run         Print resolved DB target and command without executing.
  -h, --help        Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --seed)
            if [ "$#" -lt 2 ]; then
                echo "❌ --seed requires a value."
                exit 1
            fi
            SEED_FILE="$2"
            shift 2
            ;;
        --append)
            APPEND_MODE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$SEED_FILE" ]; then
    echo "❌ --seed is required."
    usage
    exit 1
fi

if [[ "$SEED_FILE" != /* ]]; then
    SEED_FILE="$PROJECT_ROOT/$SEED_FILE"
fi

if [ ! -f "$SEED_FILE" ]; then
    echo "❌ Seed file not found: $SEED_FILE"
    exit 1
fi

if ! command -v mysql >/dev/null 2>&1; then
    echo "❌ mysql client not found. Install MySQL client first."
    exit 1
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ python/python3 not found."
    exit 1
fi

if [[ "$SEED_FILE" == *.gz ]] && ! command -v gzip >/dev/null 2>&1; then
    echo "❌ gzip not found but seed file is .gz."
    exit 1
fi

parse_output="$(
PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import os
import sys
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
    print("❌ DATABASE_URL is not set. Configure it in .env.", file=sys.stderr)
    sys.exit(2)

if database_url.startswith("mysql://"):
    database_url = "mysql+pymysql://" + database_url[len("mysql://"):]

parsed = urlparse(database_url)
scheme = parsed.scheme.lower()
if not scheme.startswith("mysql"):
    print(f"❌ DATABASE_URL backend must be mysql, got: {scheme or 'unknown'}", file=sys.stderr)
    sys.exit(3)

db_name = (parsed.path or "").lstrip("/")
if not db_name:
    print("❌ DATABASE_URL must include database name.", file=sys.stderr)
    sys.exit(4)

db_user = unquote(parsed.username or "")
if not db_user:
    print("❌ DATABASE_URL must include username.", file=sys.stderr)
    sys.exit(5)

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

mysql_base_cmd=(
    mysql
    "--host=$DB_HOST"
    "--port=$DB_PORT"
    "--user=$DB_USER"
)

echo "Resolved target DB: mysql://$DB_USER@${DB_HOST}:${DB_PORT}/$DB_NAME"
echo "Seed file: $SEED_FILE"
echo "Import mode: $( [ "$APPEND_MODE" = "true" ] && echo "append" || echo "replace (truncate products/product_prices)" )"

if [ "$DRY_RUN" = "true" ]; then
    printf 'Dry-run command (create db): MYSQL_PWD=*** '
    printf '%q ' "${mysql_base_cmd[@]}"
    printf '%q\n' "-e CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    if [ "$APPEND_MODE" != "true" ]; then
        printf 'Dry-run command (truncate): MYSQL_PWD=*** '
        printf '%q ' "${mysql_base_cmd[@]}"
        printf '%q %q\n' "$DB_NAME" "-e SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE product_prices; TRUNCATE TABLE products; SET FOREIGN_KEY_CHECKS=1;"
    fi
    if [[ "$SEED_FILE" == *.gz ]]; then
        printf 'Dry-run command (import): gzip -dc %q | MYSQL_PWD=*** ' "$SEED_FILE"
        printf '%q ' "${mysql_base_cmd[@]}"
        printf '%q\n' "$DB_NAME"
    else
        printf 'Dry-run command (import): MYSQL_PWD=*** '
        printf '%q ' "${mysql_base_cmd[@]}"
        printf '%q < %q\n' "$DB_NAME" "$SEED_FILE"
    fi
    exit 0
fi

MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

required_count="$(
    MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" -Nse \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='$DB_NAME' AND table_name IN ('products','product_prices');"
)"

if [ "${required_count:-0}" -lt 2 ]; then
    echo "❌ Required tables are missing in $DB_NAME (products, product_prices)."
    echo "   Run: cd app && ./setup_db.sh"
    exit 1
fi

if [ "$APPEND_MODE" != "true" ]; then
    MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" "$DB_NAME" -e \
        "SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE product_prices; TRUNCATE TABLE products; SET FOREIGN_KEY_CHECKS=1;"
fi

if [[ "$SEED_FILE" == *.gz ]]; then
    gzip -dc "$SEED_FILE" | MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" "$DB_NAME"
else
    MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" "$DB_NAME" < "$SEED_FILE"
fi

echo "✅ Price seed imported."

MYSQL_PWD="$DB_PASSWORD" "${mysql_base_cmd[@]}" "$DB_NAME" -e \
    "SELECT 'products' AS table_name, COUNT(*) AS rows_count FROM products
     UNION ALL
     SELECT 'product_prices' AS table_name, COUNT(*) AS rows_count FROM product_prices;"
