#!/usr/bin/env bash
# Upload the export folder to the Hugging Face Hub, at a capped speed so the public site
# (which leaves this machine through the same home connection) keeps its bandwidth.
#
#   pipeline/hf/upload.sh start      # start (or resume) the upload as a user service, capped at $RATE
#   pipeline/hf/upload.sh status     # progress, current speed
#   pipeline/hf/upload.sh stop       # stop the upload and remove the speed cap
#   pipeline/hf/upload.sh uncap      # keep uploading, remove the cap (e.g. at night)
#
# HOLD_CARD=1 start: everything but README.md (the card, with the licence), to publish it later with
#   hf upload $REPO $OUT/README.md README.md --repo-type dataset
# Re-running start resumes: files already on the Hub are not sent again.
# The cap: packets of the hf-upload service's cgroup get a firewall mark; a traffic-control class
# limits marked packets to $RATE. Everything else on the interface is untouched. Both are runtime
# rules (gone after a reboot), removed by stop/uncap.
set -euo pipefail

OUT=${OUT:-/mnt/bulk/hf-export/openalex-denseon}
REPO=${REPO:-maximerivest/openalex-denseon-embeddings}
RATE=${RATE:-70mbit}          # measured uplink ~117 Mbit/s (2026-10-12); leave the rest to the site
UNIT=hf-upload
MARK=0x48
IF=$(ip route get 1.1.1.1 | grep -oP 'dev \K\S+')
CGROUP="user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/$UNIT.service"
HF="uvx --from huggingface_hub[hf_xet]==2.2.0 hf"
IPT="sudo nix shell nixpkgs#iptables -c iptables"

cap() {
  $IPT -t mangle -C OUTPUT -m cgroup --path "$CGROUP" -j MARK --set-mark $MARK 2>/dev/null ||
    $IPT -t mangle -A OUTPUT -m cgroup --path "$CGROUP" -j MARK --set-mark $MARK
  sudo tc qdisc replace dev "$IF" root handle 1: htb default 10
  sudo tc class replace dev "$IF" parent 1: classid 1:10 htb rate 10gbit
  sudo tc class replace dev "$IF" parent 1: classid 1:20 htb rate "$RATE" ceil "$RATE"
  sudo tc qdisc replace dev "$IF" parent 1:10 handle 10: fq_codel
  sudo tc qdisc replace dev "$IF" parent 1:20 handle 20: fq_codel
  sudo tc filter replace dev "$IF" parent 1: protocol all prio 1 handle $MARK fw flowid 1:20
  echo "capped $UNIT at $RATE on $IF"
}

uncap() {
  sudo tc qdisc del dev "$IF" root 2>/dev/null || true
  while $IPT -t mangle -D OUTPUT -m cgroup --path "$CGROUP" -j MARK --set-mark $MARK 2>/dev/null; do :; done
  echo "speed cap removed"
}

case "${1:-status}" in
  start)
    test -f "$OUT/README.md" || { echo "no $OUT/README.md: run pipeline/hf/card.py first"; exit 1; }
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --nice=10 -p IOSchedulingClass=idle \
      --setenv=HF_XET_HIGH_PERFORMANCE=0 -- \
      $HF upload "$REPO" "$OUT" . --repo-type dataset \
        --exclude "work/*" --exclude "*.tmp" --exclude "verify.json" --exclude ".cache/*" \
        ${HOLD_CARD:+--exclude README.md} \
        --commit-message "Release $(python3 -c "import json;print(json.load(open('$OUT/export.json'))['release'])")"
    sleep 2
    cap
    ;;
  status)
    systemctl --user status "$UNIT" --no-pager | sed -n 1,5p || true
    journalctl --user -u "$UNIT" --no-pager | tail -5
    sudo tc -s class show dev "$IF" 2>/dev/null | grep -A2 "1:20" | head -3 || true
    ;;
  stop)
    systemctl --user stop "$UNIT" || true
    uncap
    ;;
  uncap) uncap ;;
  cap) cap ;;
  *) echo "usage: $0 start|status|stop|uncap|cap"; exit 2 ;;
esac
