#!/usr/bin/env bash

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$APP_DIR/setup_adminer.sh"

assert_eq() {
    local actual="$1"
    local expected="$2"
    local label="$3"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $label expected='$expected' actual='$actual'" >&2
        exit 1
    fi
}

tmp_env="$(mktemp)"
trap 'rm -f "$tmp_env"' EXIT

cat > "$tmp_env" <<'EOF'
DB_VIEWER_USER = admin
DB_VIEWER_PASSWORD=admin # trailing comment
ADMINER_BASIC_USER=<새아이디>
ADMINER_BASIC_PASSWORD="quoted#value"
ADMINER_BASIC_PASSWORD=final-value
EOF

# shellcheck disable=SC1090
source "$SCRIPT_PATH"

assert_eq "$(env_file_lookup DB_VIEWER_USER "$tmp_env")" "admin" "db viewer user parse"
assert_eq "$(env_file_lookup DB_VIEWER_PASSWORD "$tmp_env")" "admin" "db viewer pass parse"
assert_eq "$(env_file_lookup ADMINER_BASIC_USER "$tmp_env")" "<새아이디>" "angle bracket parse"
assert_eq "$(env_file_lookup ADMINER_BASIC_PASSWORD "$tmp_env")" "final-value" "latest value wins"

echo "PASS test_setup_adminer_env.sh"
