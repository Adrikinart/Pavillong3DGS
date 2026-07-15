#!/usr/bin/env bash
# Sync any retained node-local scratch workspaces back to persistent storage and
# report leftovers. ScratchContext normally cleans itself on success; this helps
# recover data from a crashed/preempted stage that retained its scratch dir.
set -uo pipefail
SCRATCH_ROOT="${1:-/var/tmp/$USER/v2g}"
echo "scanning retained scratch under $SCRATCH_ROOT"
if [ ! -d "$SCRATCH_ROOT" ]; then echo "none"; exit 0; fi
find "$SCRATCH_ROOT" -maxdepth 2 -type d 2>/dev/null | while read -r d; do
  n=$(find "$d" -type f 2>/dev/null | wc -l)
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  echo "  $d  files=$n size=$sz"
done
echo "Inspect the paths above; copy anything needed, then remove with: rm -rf $SCRATCH_ROOT/<id>"
