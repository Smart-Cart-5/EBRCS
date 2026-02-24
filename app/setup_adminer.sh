#!/usr/bin/env bash
# Configure Adminer without Docker:
# - installs php-cli if missing
# - downloads Adminer single-file app
# - runs Adminer as a systemd service bound to localhost
# - protects /adminer/ via Nginx basic auth

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$APP_DIR")"
LOG_DIR="$APP_DIR/logs"
SETUP_LOG="$LOG_DIR/setup_adminer.log"

ADMINER_DIR="$APP_DIR/adminer"
ADMINER_FILE="$ADMINER_DIR/index.php"
ADMINER_PORT="${ADMINER_PORT:-18080}"
ADMINER_DOWNLOAD_URL="${ADMINER_DOWNLOAD_URL:-https://www.adminer.org/latest.php}"

SERVICE_NAME="ebrcs-adminer.service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
NGINX_SITE="/etc/nginx/sites-available/ebrcs"
HTPASSWD_FILE="/etc/nginx/.htpasswd_adminer"

mkdir -p "$LOG_DIR"
: > "$SETUP_LOG"

log() {
    printf '%s\n' "$*"
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$SETUP_LOG"
}

run_logged() {
    "$@" >> "$SETUP_LOG" 2>&1
}

ensure_command() {
    local cmd="$1"
    local pkg="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log "Installing missing package: $pkg"
        if ! run_logged sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"; then
            log "Package install retry with apt update (hooks disabled): $pkg"
            run_logged sudo apt-get \
                -o APT::Update::Post-Invoke-Success::= \
                -o APT::Update::Post-Invoke::= \
                -o DPkg::Post-Invoke::= \
                update || true
            run_logged sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
        fi
    fi
}

ensure_php_mysql_extension() {
    if php -m | grep -qiE '^(mysqli|pdo_mysql)$'; then
        return 0
    fi

    log "Installing missing PHP MySQL extension: php-mysql"
    if ! run_logged sudo DEBIAN_FRONTEND=noninteractive apt-get install -y php-mysql; then
        log "PHP MySQL extension install retry with apt update (hooks disabled)."
        run_logged sudo apt-get \
            -o APT::Update::Post-Invoke-Success::= \
            -o APT::Update::Post-Invoke::= \
            -o DPkg::Post-Invoke::= \
            update || true
        run_logged sudo DEBIAN_FRONTEND=noninteractive apt-get install -y php-mysql
    fi
}

env_file_lookup() {
    local key="$1"
    local env_file="$2"
    local value

    value="$(
        awk -v k="$key" '
            BEGIN { FS="=" }
            $0 ~ "^[[:space:]]*" k "[[:space:]]*=" {
                line = $0
                sub(/^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*/, "", line)
                v = line
            }
            END { print v }
        ' "$env_file"
    )"

    value="${value%$'\r'}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:-1}"
    elif [[ "$value" == \'*\' ]]; then
        value="${value:1:-1}"
    else
        value="${value%%#*}"
        value="$(printf '%s' "$value" | sed 's/[[:space:]]*$//')"
    fi

    printf '%s' "$value"
}

load_env() {
    local env_file="$PROJECT_ROOT/.env"
    [ -f "$env_file" ] || return 0

    local value

    value="$(env_file_lookup DB_VIEWER_USER "$env_file")"
    [ -n "$value" ] && DB_VIEWER_USER="$value"

    value="$(env_file_lookup DB_VIEWER_PASSWORD "$env_file")"
    [ -n "$value" ] && DB_VIEWER_PASSWORD="$value"

    value="$(env_file_lookup ADMINER_BASIC_USER "$env_file")"
    [ -n "$value" ] && ADMINER_BASIC_USER="$value"

    value="$(env_file_lookup ADMINER_BASIC_PASSWORD "$env_file")"
    [ -n "$value" ] && ADMINER_BASIC_PASSWORD="$value"
}

write_htpasswd() {
    local user="$1"
    local pass="$2"
    local hash
    hash="$(openssl passwd -6 "$pass")"
    printf '%s:%s\n' "$user" "$hash" | sudo tee "$HTPASSWD_FILE" >/dev/null
    if getent group www-data >/dev/null 2>&1; then
        run_logged sudo chown root:www-data "$HTPASSWD_FILE"
    else
        run_logged sudo chown root:root "$HTPASSWD_FILE"
    fi
    run_logged sudo chmod 640 "$HTPASSWD_FILE"
}

install_adminer_service() {
    local php_bin
    php_bin="$(command -v php)"

    cat <<EOF | sudo tee "$SERVICE_FILE" >/dev/null
[Unit]
Description=EBRCS Adminer
After=network.target

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$ADMINER_DIR
ExecStart=$php_bin -S 127.0.0.1:$ADMINER_PORT -t $ADMINER_DIR
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    run_logged sudo systemctl daemon-reload
    run_logged sudo systemctl enable "$SERVICE_NAME"
    run_logged sudo systemctl restart "$SERVICE_NAME"
}

main() {
    log "Configuring Adminer (no Docker)."
    log "Log file: $SETUP_LOG"

    ensure_command curl curl
    ensure_command openssl openssl
    ensure_command php php-cli
    ensure_command nginx nginx
    ensure_php_mysql_extension

    load_env

    local auth_user auth_pass
    auth_user="${ADMINER_BASIC_USER:-${DB_VIEWER_USER:-adminer}}"
    auth_pass="${ADMINER_BASIC_PASSWORD:-${DB_VIEWER_PASSWORD:-}}"
    if [ -z "$auth_pass" ]; then
        auth_pass="$(openssl rand -base64 18 | tr -d '=+/[:space:]' | cut -c1-16)"
        log "Generated ADMINER basic-auth password (not found in .env)."
        log "ADMINER_BASIC_USER=$auth_user"
        log "ADMINER_BASIC_PASSWORD=$auth_pass"
        log "Add these to $PROJECT_ROOT/.env to persist custom credentials."
    fi

    mkdir -p "$ADMINER_DIR"
    log "Downloading Adminer from: $ADMINER_DOWNLOAD_URL"
    run_logged curl -fsSL "$ADMINER_DOWNLOAD_URL" -o "${ADMINER_FILE}.tmp"
    mv "${ADMINER_FILE}.tmp" "$ADMINER_FILE"
    chmod 644 "$ADMINER_FILE"

    log "Writing Nginx basic-auth credentials."
    write_htpasswd "$auth_user" "$auth_pass"

    log "Installing Adminer systemd service."
    install_adminer_service

    log "Installing Nginx site config from repository."
    run_logged sudo cp "$PROJECT_ROOT/nginx/ebrcs.conf" "$NGINX_SITE"
    run_logged sudo nginx -t
    run_logged sudo systemctl reload nginx

    run_logged curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:${ADMINER_PORT}/"

    local service_state
    service_state="$(sudo systemctl is-active "$SERVICE_NAME" 2>> "$SETUP_LOG" || true)"
    if [ "$service_state" != "active" ]; then
        log "ERROR: $SERVICE_NAME is not active ($service_state)."
        run_logged sudo systemctl status "$SERVICE_NAME" --no-pager -l
        exit 1
    fi

    log ""
    log "Adminer setup complete."
    log "URL: https://<EC2_PUBLIC_IP>/adminer/"
    log "Basic Auth user: $auth_user"
    log "Basic Auth password: $auth_pass"
    log "Service status: sudo systemctl status $SERVICE_NAME --no-pager"
    log "Setup log: $SETUP_LOG"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
