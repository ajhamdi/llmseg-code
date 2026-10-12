#!/usr/bin/env python
"""Download ODinW13 (Hugging Face ``kcz358/ODinW-13``) and write one COCO folder per subset.

    data/benchmarks/odinw/<Subset>/images/*.jpg
    data/benchmarks/odinw/<Subset>/annotations.coco.json

ODinW is a detection benchmark (boxes only); the parquet stores [x1, y1, x2, y2]
boxes, converted here to COCO [x, y, w, h].

    python scripts/download_odinw.py
"""
from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.getenv("LLMSEG_DATA", REPO / "data" / "benchmarks")) / "odinw"


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image

    hf = OUT / "_hf"
    snapshot_download("kcz358/ODinW-13", repo_type="dataset", local_dir=str(hf),
                      allow_patterns=["data/*.parquet", "README.md"])
    files = sorted((hf / "data").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet downloaded into {hf}")
    d = {}
    for f in files:
        part = pq.read_table(f).to_pydict()
        for k, v in part.items():
            d.setdefault(k, []).extend(v)
    by_subset = defaultdict(list)
    for i, name in enumerate(d["dataset_name"]):
        by_subset[name].append(i)

    for subset, idxs in sorted(by_subset.items()):
        sdir = OUT / subset
        if (sdir / "annotations.coco.json").exists():
            print(f"  {subset}: already present", flush=True)
            continue
        (sdir / "images").mkdir(parents=True, exist_ok=True)
        cats = list(d["categories"][idxs[0]])
        cat_id = {c: k + 1 for k, c in enumerate(cats)}
        coco = {"images": [], "annotations": [],
                "categories": [{"id": k + 1, "name": c} for k, c in enumerate(cats)]}
        for iid, i in enumerate(idxs, 1):
            im = Image.open(io.BytesIO(d["image"][i]["bytes"])).convert("RGB")
            fn = f"{iid:06d}.jpg"
            im.save(sdir / "images" / fn, "JPEG", quality=95)
            coco["images"].append({"id": iid, "file_name": fn, "width": im.width, "height": im.height})
            for lab, box in zip(d["labels"][i], d["bboxes"][i]):
                x1, y1, x2, y2 = (float(v) for v in box)
                if lab in cat_id and x2 > x1 and y2 > y1:
                    coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid,
                                                "category_id": cat_id[lab], "bbox": [x1, y1, x2 - x1, y2 - y1],
                                                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        (sdir / "annotations.coco.json").write_text(json.dumps(coco))
        print(f"  {subset:26s} images={len(coco['images']):5d} boxes={len(coco['annotations']):6d} "
              f"classes={len(cats)}", flush=True)
    print(f"ODinW13 ready in {OUT}")


if __name__ == "__main__":
    main()
