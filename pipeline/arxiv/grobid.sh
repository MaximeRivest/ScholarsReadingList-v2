#!/usr/bin/env bash
# GROBID (reference extraction from PDFs) for refs.py, on 127.0.0.1:8070, as a user service.
#   pipeline/arxiv/grobid.sh up | down | status
# CPU-only image (grobid 0.9.1-crf); 30 papers at once (/mnt/bulk/arxiv/grobid/grobid.yaml), at most 28 cores, and
# half the CPU weight of other work so the site's services keep first claim.
set -euo pipefail
IMG=docker.io/grobid/grobid:0.9.1-crf
case "${1:-status}" in
  up)
    podman rm -f grobid-pilot >/dev/null 2>&1 || true
    systemctl --user stop arxiv-grobid-server 2>/dev/null || true; systemctl --user reset-failed arxiv-grobid-server 2>/dev/null || true
    systemd-run --user -q --unit=arxiv-grobid-server -p Restart=on-failure \
      podman run --rm --name grobid -p 127.0.0.1:8070:8070 --cpus 28 --cpu-shares 512 --memory 48g \
      -v /mnt/bulk/arxiv/grobid/grobid.yaml:/opt/grobid/grobid-home/config/grobid.yaml:ro "$IMG"
    for _ in $(seq 1 60); do curl -sf -m 5 127.0.0.1:8070/api/isalive >/dev/null && { echo "GROBID up"; exit 0; }; sleep 3; done
    echo "GROBID did not start" >&2; exit 1 ;;
  down) systemctl --user stop arxiv-grobid-server; echo stopped ;;
  status) systemctl --user is-active arxiv-grobid-server; curl -sf -m 5 127.0.0.1:8070/api/isalive && echo " alive" || echo "not answering" ;;
esac
