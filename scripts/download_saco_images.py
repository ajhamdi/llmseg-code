#!/usr/bin/env python
"""Download the SA-Co images from Roboflow and lay them out to match the Hugging Face annotations.

The facebook/SACo-Gold (and -Silver) annotation files reference images by relative
path but carry no pixels. Meta mirrors the images on the Roboflow workspaces
``sa-co-gold`` / ``sa-co-silver``. This script downloads those projects and links
each image to ``data/benchmarks/saco/<split>/images/<file_name>``.

Needs a (free) Roboflow API key and the annotations first:

    bash scripts/download_benchmarks.sh saco
    pip install roboflow
    ROBOFLOW_API_KEY=... python scripts/download_saco_images.py gold
    ROBOFLOW_API_KEY=... python scripts/download_saco_images.py silver   # optional, not in the paper

Re-running skips projects already downloaded. Silver's food_rec domain has no
Roboflow mirror and is skipped.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("LLMSEG_DATA", REPO / "data" / "benchmarks"))

GOLD_PROJECTS = [  # one per domain; images are identical across the a/b/c annotation replicas
    "gold-metaclip-merged-a-release-test",
    "gold-attributes-merged-a-release-test",
    "gold-crowded-merged-a-release-test",
    "gold-wiki-common-merged-a-release-test",
    "gold-fg-food-merged-a-release-test",
    "gold-fg-sports-equipment-merged-a-release-test",
    "gold-sa-1b-merged-a-release-test",
]
SILVER_PROJECTS = [
    "inaturalist-2017", "fathomnet-kmz5d", "geode", "national-gallery-of-art", "bdd100k-gwmh6",
    "sa-v", "ego4d-w7fiu", "droid-cfual", "yt-temporal-1b",
]

_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.\w+$", re.I)
_EXT_TAIL = re.compile(r"_(jpe?g|png)$", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _NON_ALNUM.sub("_", s.lower()).strip("_")


def roboflow_key(name: str) -> str:
    """'<stem>_jpg.rf.<hash>.jpg' -> normalized original stem."""
    return norm(_EXT_TAIL.sub("", _RF_SUFFIX.sub("", name)))


def annotation_key(file_name: str) -> str:
    return norm(os.path.splitext(file_name)[0])


def latest_version(project):
    def num(v):
        try:
            return int(str(getattr(v, "version", "")).rstrip("/").split("/")[-1])
        except ValueError:
            return 0
    return max(project.versions(), key=num)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "gold"
    if split not in ("gold", "silver"):
        sys.exit("usage: download_saco_images.py [gold|silver]")
    key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("RF_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY (free at https://roboflow.com)")
    root = DATA / "saco" / split
    ann_files = sorted(root.glob("*.json"))
    if not ann_files:
        sys.exit(f"no annotations in {root}: run  bash scripts/download_benchmarks.sh saco  first")
    by_key = {}
    for f in ann_files:
        for im in json.loads(f.read_text())["images"]:
            by_key[annotation_key(im["file_name"])] = im["file_name"]
    print(f"{split}: {len(set(by_key.values()))} images referenced by the annotations", flush=True)

    from roboflow import Roboflow
    ws = Roboflow(api_key=key).workspace(f"sa-co-{split}")
    img_dir, rf_dir = root / "images", root / "_roboflow"
    linked = set()
    for slug in GOLD_PROJECTS if split == "gold" else SILVER_PROJECTS:
        dest = rf_dir / slug
        try:
            if not (dest / ".downloaded").exists():
                print(f"[download] {slug}", flush=True)
                latest_version(ws.project(slug)).download("coco", location=str(dest), overwrite=True)
                (dest / ".downloaded").write_text("ok\n")
        except Exception as e:  # noqa: BLE001 (keep going with the other projects)
            print(f"[failed] {slug}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        n = 0
        for img in glob.glob(str(dest / "**" / "*.*"), recursive=True):
            fn = by_key.get(roboflow_key(os.path.basename(img)))
            if not fn or not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            out = img_dir / fn
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                os.symlink(os.path.abspath(img), out)
            linked.add(fn)
            n += 1
        print(f"   {slug}: {n} images linked", flush=True)
    total = len(set(by_key.values()))
    print(f"{split}: {len(linked)}/{total} images available ({100 * len(linked) / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
