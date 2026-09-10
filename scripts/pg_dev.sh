#!/usr/bin/env bash
# Local dev Postgres for robin (conda env "pg", cluster ~/.local/share/robin-postgres, port 5433).
# Usage: scripts/pg_dev.sh start|stop|status
set -e
PGDATA="$HOME/.local/share/robin-postgres"
case "${1:-status}" in
  start)  conda run -n pg pg_ctl -D "$PGDATA" -o "-p 5433 -c listen_addresses=127.0.0.1" -l "$PGDATA/server.log" start ;;
  stop)   conda run -n pg pg_ctl -D "$PGDATA" stop ;;
  status) conda run -n pg pg_ctl -D "$PGDATA" status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
