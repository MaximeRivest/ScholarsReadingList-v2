#!/usr/bin/env bash
# Controls the persistent, declarative service. A deliberate pause survives reboot.
set -euo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
unit=openalex-embeddings
out=/mnt/fast/openalex-embeddings/denseon-english-abstracts
case "${1:-status}" in
  start|resume)
    mountpoint -q /mnt/fast || { echo 'The embedding SSD is not mounted.' >&2; exit 1; }
    .venv/bin/python -c 'from pathlib import Path; import embed_content as e; p=Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts"); p.mkdir(parents=True,exist_ok=True); (p/"PAUSED").unlink(missing_ok=True); e.sync_dir(p)'
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    systemctl --user start "$unit"
    ;;
  pause|stop)
    mountpoint -q /mnt/fast || { echo 'The embedding SSD is not mounted.' >&2; exit 1; }
    .venv/bin/python -c 'from pathlib import Path; import embed_content as e; e.atomic_json(Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/PAUSED"), {"requested_utc":e.now()})'
    systemctl --user stop "$unit"
    ;;
  status)
    systemctl --user status "$unit" --no-pager || true
    if test -f "$out/status.json"; then .venv/bin/python embed_content.py status --output "$out"; fi
    ;;
  logs) journalctl --user -u "$unit" -f ;;
  *) echo "Usage: $0 start|resume|pause|stop|status|logs" >&2; exit 2 ;;
esac
