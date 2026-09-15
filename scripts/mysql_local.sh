#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env.mysql.local"
CONTAINER="neuro-preprocess-mysql"
IMAGE="mysql:8.4"
VOLUME="neuro-preprocess-mysql-data"

create_env() {
  if [[ -f "$ENV_FILE" ]]; then
    return
  fi
  umask 077
  {
    printf 'MYSQL_ROOT_PASSWORD=%s\n' "$(openssl rand -hex 24)"
    printf 'MYSQL_DATABASE=neuro_preprocess\n'
    printf 'MYSQL_USER=neuro_agent\n'
    printf 'MYSQL_PASSWORD=%s\n' "$(openssl rand -hex 24)"
    printf 'MYSQL_PORT=3306\n'
  } > "$ENV_FILE"
}

load_env() {
  create_env
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  export NEURO_AGENT_MYSQL_PASSWORD="$MYSQL_PASSWORD"
}

wait_until_ready() {
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" mysqladmin ping -h 127.0.0.1 -uroot -p"$MYSQL_ROOT_PASSWORD" --silent >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  docker logs --tail 80 "$CONTAINER"
  echo "MySQL did not become ready within 120 seconds." >&2
  exit 1
}

start() {
  load_env
  if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    docker start "$CONTAINER" >/dev/null
  else
    docker run -d \
      --name "$CONTAINER" \
      --restart unless-stopped \
      -p "127.0.0.1:${MYSQL_PORT}:3306" \
      -e MYSQL_ROOT_PASSWORD="$MYSQL_ROOT_PASSWORD" \
      -e MYSQL_DATABASE="$MYSQL_DATABASE" \
      -e MYSQL_USER="$MYSQL_USER" \
      -e MYSQL_PASSWORD="$MYSQL_PASSWORD" \
      -v "$VOLUME:/var/lib/mysql" \
      "$IMAGE" >/dev/null
  fi
  wait_until_ready
  "$PROJECT_ROOT/.venv/bin/python" -m neuro_preprocess_agent.cli mysql-init \
    --mysql-config "$PROJECT_ROOT/configs/mysql.docker.json"
  echo "MySQL is ready at 127.0.0.1:${MYSQL_PORT}."
  echo "Load the app password with: set -a; source $ENV_FILE; set +a; export NEURO_AGENT_MYSQL_PASSWORD=\"\$MYSQL_PASSWORD\""
}

case "${1:-status}" in
  start|setup) start ;;
  stop) docker stop "$CONTAINER" >/dev/null && echo "MySQL stopped." ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' ;;
  logs) docker logs --tail 100 -f "$CONTAINER" ;;
  shell)
    load_env
    docker exec -it "$CONTAINER" mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE"
    ;;
  *) echo "Usage: $0 {start|stop|status|logs|shell}" >&2; exit 2 ;;
esac
