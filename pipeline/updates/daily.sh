#!/usr/bin/env bash
# The daily update: new OpenAlex papers -> details corpus -> vectors. The nightly release (03:30) then puts them
# in search and on the map. Run by srl-updates.timer (~/Projects/os/machines/hm/scholarsreadinglist-data.nix).
#
#   pipeline/updates/daily.sh            fetch, corpus (+ gc), embed
#   pipeline/updates/daily.sh status     what is held, last runs
#
# Each step is resumable and safe to re-run; a step that fails stops the rest (the next run continues).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
UPD=/mnt/fast/scholarsreadinglist/updates

if [ "${1:-}" = status ]; then
  for b in "$UPD"/*/; do
    echo "base $(basename "$b")"
    c=$(ls -d "$b"corpus/2*/ 2>/dev/null | tail -1)
    [ -n "$c" ] && python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(f\"  corpus {m['created_utc']}: {m['works']:,} papers, published {m['dates']}, record facts for {m.get('quality_facts')}\")" "$c/manifest.json"
    ls -t "$b"runs/*.json 2>/dev/null | head -3 | while read -r r; do
      python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"  run {d['run']}: {d['written']:,} new, \${d['spent_usd']} spent, stopped: {d['stopped']}\")" "$r"
    done
  done
  n=$(find /mnt/fast/openalex-embeddings/denseon-english-abstracts/shards -path '*/rg-u*/part-*/commit.json' | wc -l)
  echo "daily shards: $n"
  exit 0
fi

uv run -q pipeline/updates/fetch.py
search/.venv/bin/python pipeline/updates/corpus.py --gc
pipeline/embeddings/.venv/bin/python pipeline/updates/embed.py
