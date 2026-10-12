#!/usr/bin/env bash
# Download the evaluation annotations (and the self-contained ReasonSeg / ADE20K images).
# Only evaluation splits are fetched; nothing here is used for training.
#
#   bash scripts/download_benchmarks.sh                       # everything below
#   bash scripts/download_benchmarks.sh reasonseg refcoco     # a subset
#
# Keys:
#   reasonseg  Ricky06662/ReasonSeg_{val,test}  (images + masks inside the parquet)       ~1.5 GB
#   refcoco    jxu124/refcoco{,plus,g}           (expressions + masks; images = COCO train2014)
#   saco       facebook/SACo-Gold                (GATED: accept the terms on Hugging Face and
#                                                 export HF_TOKEN; images via download_saco_images.py)
#   ade20k     ADEChallengeData2016, validation split only (images + labels)              ~0.9 GB zip
#   lvis       LVIS v1 val annotations           (images = COCO val2017 + train2017 subset)
#   coco       COCO 2017 val instance annotations (images = COCO val2017)
#
# Images that are not bundled: bash scripts/download_coco_images.sh, python scripts/download_saco_images.py.
# Target: $LLMSEG_DATA (default: data/benchmarks). Re-running skips what is already there.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -f .env ] && { set -a; . ./.env; set +a; }
ROOT="${LLMSEG_DATA:-data/benchmarks}"
PY="${PYTHON:-python}"
mkdir -p "$ROOT"
KEYS=("$@")
[ ${#KEYS[@]} -eq 0 ] && KEYS=(reasonseg refcoco saco ade20k lvis coco)
log() { echo "[$(date +%H:%M:%S)] $*"; }

hf_dataset() {  # hf_dataset <repo> <subdir> [ignore-pattern ...]
  local repo="$1" out="$ROOT/$2"; shift 2
  log "  $repo -> $out"
  "$PY" - "$repo" "$out" "$@" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, out, ignore = sys.argv[1], sys.argv[2], sys.argv[3:]
try:
    snapshot_download(repo, repo_type="dataset", local_dir=out, ignore_patterns=ignore or None,
                      token=os.getenv("HF_TOKEN") or None, max_workers=8)
except Exception as e:
    sys.exit(f"  FAILED {repo}: {type(e).__name__}: {str(e)[:300]}")
PY
}

fetch() {  # fetch <url> <file>
  [ -s "$2" ] && return 0
  log "  downloading $1"
  curl -fL --retry 3 -C - -o "$2" "$1"
}

unzip_some() {  # unzip_some <zip> <dest> [glob ...]  (python zipfile, no unzip binary needed)
  "$PY" - "$@" <<'PY'
import fnmatch, sys, zipfile
zpath, dest, globs = sys.argv[1], sys.argv[2], sys.argv[3:]
with zipfile.ZipFile(zpath) as z:
    names = [n for n in z.namelist() if not globs or any(fnmatch.fnmatch(n, g) for g in globs)]
    for n in names:
        z.extract(n, dest)
print(f"  extracted {len(names)} files into {dest}")
PY
}

for key in "${KEYS[@]}"; do
  case "$key" in
    reasonseg)
      log "ReasonSeg (val + test)"
      hf_dataset Ricky06662/ReasonSeg_val  reasonseg/val
      hf_dataset Ricky06662/ReasonSeg_test reasonseg/test ;;
    refcoco)
      log "RefCOCO / RefCOCO+ / RefCOCOg (evaluation splits)"
      hf_dataset jxu124/refcoco     refcoco/refcoco     "*train*"
      hf_dataset jxu124/refcocoplus refcoco/refcocoplus "*train*"
      hf_dataset jxu124/refcocog    refcoco/refcocog    "*train*" ;;
    saco)
      log "SA-Co/Gold annotations (gated: needs HF_TOKEN with the facebook/SACo-Gold terms accepted)"
      [ -z "${HF_TOKEN:-}" ] && log "  WARNING: HF_TOKEN is not set"
      hf_dataset facebook/SACo-Gold saco/gold ;;
    saco-silver)
      log "SA-Co/Silver annotations (gated, optional; not used in the paper)"
      hf_dataset facebook/SACo-Silver saco/silver ;;
    ade20k)
      log "ADE20K validation"
      if [ -d "$ROOT/ade20k/ADEChallengeData2016/annotations/validation" ]; then
        log "  already present"
      else
        mkdir -p "$ROOT/ade20k"
        fetch "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" "$ROOT/ade20k.zip" \
          && unzip_some "$ROOT/ade20k.zip" "$ROOT/ade20k" \
               'ADEChallengeData2016/images/validation/*' 'ADEChallengeData2016/annotations/validation/*' \
               'ADEChallengeData2016/*.txt' \
          && rm -f "$ROOT/ade20k.zip"
      fi ;;
    lvis)
      log "LVIS v1 val annotations"
      mkdir -p "$ROOT/lvis"
      if [ ! -s "$ROOT/lvis/lvis_v1_val.json" ]; then
        fetch "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" "$ROOT/lvis_val.zip" \
          && unzip_some "$ROOT/lvis_val.zip" "$ROOT/lvis" && rm -f "$ROOT/lvis_val.zip"
      fi ;;
    coco)
      log "COCO 2017 val instance annotations"
      mkdir -p "$ROOT/coco"
      if [ ! -s "$ROOT/coco/annotations/instances_val2017.json" ]; then
        fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$ROOT/coco_ann.zip" \
          && unzip_some "$ROOT/coco_ann.zip" "$ROOT/coco" 'annotations/instances_val2017.json' \
          && rm -f "$ROOT/coco_ann.zip"
      fi ;;
    *) log "unknown key '$key' (reasonseg refcoco saco saco-silver ade20k lvis coco)" ;;
  esac
done
log "done -> $ROOT"
