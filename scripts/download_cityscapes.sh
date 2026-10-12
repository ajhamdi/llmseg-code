#!/usr/bin/env bash
# Cityscapes validation split (500 images) for the semantic-segmentation evaluation.
#
# Cityscapes needs a free account at https://www.cityscapes-dataset.com (accept the license).
# Either
#   (a) download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip yourself and put them in
#       data/benchmarks/cityscapes/, or
#   (b) let the official downloader fetch them:
#         pip install cityscapesscripts
#         CITYSCAPES_USERNAME=... CITYSCAPES_PASSWORD=... bash scripts/download_cityscapes.sh
# then this script extracts only the val split.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="${LLMSEG_DATA:-data/benchmarks}/cityscapes"
PY="${PYTHON:-python}"
mkdir -p "$ROOT"
ZIPS=(gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip)

for z in "${ZIPS[@]}"; do
  [ -s "$ROOT/$z" ] && continue
  if command -v csDownload >/dev/null; then
    (cd "$ROOT" && csDownload "$z") || { echo "csDownload $z failed (check credentials)"; exit 1; }
  else
    echo "missing $ROOT/$z: download it from https://www.cityscapes-dataset.com/downloads/"
    echo "(or pip install cityscapesscripts and set CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD)"
    exit 1
  fi
done

"$PY" - "$ROOT" <<'PY'
import sys, zipfile
root = sys.argv[1]
for z, prefix in (("gtFine_trainvaltest.zip", "gtFine/val/"),
                  ("leftImg8bit_trainvaltest.zip", "leftImg8bit/val/")):
    with zipfile.ZipFile(f"{root}/{z}") as f:
        names = [n for n in f.namelist() if n.startswith(prefix)]
        for n in names:
            f.extract(n, root)
    print(f"{z}: extracted {len(names)} files of {prefix}")
PY
echo "Cityscapes val ready in $ROOT (the zips can be deleted)"
