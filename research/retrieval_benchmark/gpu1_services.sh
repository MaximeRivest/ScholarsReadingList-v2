#!/usr/bin/env bash
# Pause/resume the services that share GPU 1, so benchmarks can use the whole card.
# Usage: ./gpu1_services.sh stop | start | status
set -euo pipefail
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
SYSTEM=(chattering-semantic kokoro-tts parakeet-server)   # system services (sudo)
USER_UNITS=(qwen3-embed-4b)                                # user service
case "${1:-status}" in
  stop)  sudo systemctl stop "${SYSTEM[@]}"; systemctl --user stop "${USER_UNITS[@]}" ;;
  start) systemctl --user start "${USER_UNITS[@]}"; sudo systemctl start "${SYSTEM[@]}" ;;
  status) ;;
  *) echo "usage: $0 stop|start|status" >&2; exit 2 ;;
esac
for u in "${SYSTEM[@]}"; do printf '%-22s %s\n' "$u" "$(systemctl is-active "$u" || true)"; done
for u in "${USER_UNITS[@]}"; do printf '%-22s %s\n' "$u" "$(systemctl --user is-active "$u" || true)"; done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
