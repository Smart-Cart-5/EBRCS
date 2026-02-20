#!/usr/bin/env bash
# Export products/product_prices data from MySQL as seed SQL.
#
# Usage:
#   ./db/export_price_seed.sh
#   ./db/export_price_seed.sh --output ./db/seeds/price_seed.sql.gz
#   ./db/export_price_seed.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_OUTPUT="$SCRIPT_DIR/seeds/price_seed_$(date +%Y%m%d_%H%M%S).sql.gz"
OUTPUT_FILE="$DEFAULT_OUTPUT"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Export EBRCS price seed from MySQL.

Options:
  --output <path>   Output file path (.sql or .sql.gz). Default: db/seeds/price_seed_<timestamp>.sql.gz
  --dry-run         Print resolved DB target and command without executing.
  -h, --help        Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            if [ "$#" -lt 2 ]; then
                echo "❌ --output requires a value."
                exit 1
            fi
            OUTPUT_FILE="$2"
            shift 2
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

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "⚠️  .env not found at project root. Using current shell environment only."
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ python/python3 not found."
    exit 1
fi

if ! command -v mysqldump >/dev/null 2>&1; then
    echo "❌ mysqldump not found. Install MySQL client first."
    exit 1
fi

if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="$PROJECT_ROOT/$OUTPUT_FILE"
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

mkdir -p "$(dirname "$OUTPUT_FILE")"

dump_cmd=(
    mysqldump
    "--host=$DB_HOST"
    "--port=$DB_PORT"
    "--user=$DB_USER"
    "--single-transaction"
    "--set-gtid-purged=OFF"
    "--no-tablespaces"
    "--skip-comments"
    "--no-create-info"
    "--skip-triggers"
    "$DB_NAME"
    "products"
    "product_prices"
)

echo "Resolved source DB: mysql://$DB_USER@${DB_HOST}:${DB_PORT}/$DB_NAME"
echo "Output file: $OUTPUT_FILE"

if [ "$DRY_RUN" = "true" ]; then
    printf 'Dry-run command: MYSQL_PWD=*** '
    printf '%q ' "${dump_cmd[@]}"
    if [[ "$OUTPUT_FILE" == *.gz ]]; then
        printf '| gzip -c > %q\n' "$OUTPUT_FILE"
    else
        printf '> %q\n' "$OUTPUT_FILE"
    fi
    exit 0
fi

if [[ "$OUTPUT_FILE" == *.gz ]]; then
    MYSQL_PWD="$DB_PASSWORD" "${dump_cmd[@]}" | gzip -c > "$OUTPUT_FILE"
else
    MYSQL_PWD="$DB_PASSWORD" "${dump_cmd[@]}" > "$OUTPUT_FILE"
fi

echo "✅ Price seed exported."
ls -lh "$OUTPUT_FILE"
