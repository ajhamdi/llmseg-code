#!/usr/bin/env bash
# Download the COCO images that RefCOCO, LVIS and COCO evaluation run on.
#
#   bash scripts/download_coco_images.sh                 # train2014 val2017 lvis
#   bash scripts/download_coco_images.sh val2017 lvis    # only what LVIS / COCO need
#
# Keys:
#   train2014  all RefCOCO / RefCOCO+ / RefCOCOg splits use COCO train2014   (12.6 GB, 82,783 images)
#   val2017    COCO val2017 (COCO evaluation + part of LVIS val)             (0.8 GB, 5,000 images)
#   lvis       the 15,000 train2017 images referenced by LVIS v1 val (per-image download,
#              instead of the full 18 GB train2017 zip; needs lvis_v1_val.json first)
#
# Uses aria2c when installed (faster, resumable), curl otherwise.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="${LLMSEG_DATA:-data/benchmarks}"
IMG="$ROOT/coco/images"
PY="${PYTHON:-python}"
mkdir -p "$IMG"
KEYS=("$@")
[ ${#KEYS[@]} -eq 0 ] && KEYS=(train2014 val2017 lvis)
log() { echo "[$(date +%H:%M:%S)] $*"; }

get_zip() {  # get_zip <name>
  local name="$1" zip="$ROOT/coco/$1.zip"
  if [ -f "$IMG/.$name.done" ]; then log "$name: already present"; return 0; fi
  log "$name: downloading"
  if command -v aria2c >/dev/null; then
    aria2c -c -x8 -s8 --console-log-level=warn -d "$ROOT/coco" -o "$name.zip" \
      "http://images.cocodataset.org/zips/$name.zip" || return 1
  else
    curl -fL --retry 3 -C - -o "$zip" "http://images.cocodataset.org/zips/$name.zip" || return 1
  fi
  log "$name: extracting"
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$IMG" \
    && touch "$IMG/.$name.done" && rm -f "$zip"
}

for key in "${KEYS[@]}"; do
  case "$key" in
    train2014|val2017) get_zip "$key" ;;
    lvis)
      ann="$ROOT/lvis/lvis_v1_val.json"
      [ -s "$ann" ] || { log "lvis: run 'bash scripts/download_benchmarks.sh lvis' first"; continue; }
      get_zip val2017
      mkdir -p "$IMG/train2017"
      list="$(cd "$ROOT/coco" && pwd)/lvis_train2017_urls.txt"
      "$PY" - "$ann" "$list" "$IMG/train2017" <<'PY'
import json, os, sys
ann, out, have = sys.argv[1:]
urls = [im["coco_url"] for im in json.load(open(ann))["images"] if "/train2017/" in im["coco_url"]]
todo = [u for u in urls if not os.path.exists(os.path.join(have, u.rsplit("/", 1)[-1]))]
open(out, "w").write("\n".join(todo) + ("\n" if todo else ""))
print(f"lvis: {len(urls) - len(todo)}/{len(urls)} train2017 images present, {len(todo)} to fetch")
PY
      if [ -s "$list" ]; then
        if command -v aria2c >/dev/null; then
          aria2c -c -j16 --console-log-level=warn -i "$list" -d "$IMG/train2017"
        else
          (cd "$IMG/train2017" && xargs -n 1 -P 16 curl -fsSLO --retry 3 < "$list")
        fi
      fi
      rm -f "$list" ;;
    *) log "unknown key '$key' (train2014 val2017 lvis)" ;;
  esac
done
log "done -> $IMG"
