#!/usr/bin/env python
"""Any COCO-format dataset (one annotation json + one image dir): category-prompted detection.

ODinW13 and RF-100VL are collections of small COCO detection datasets; this
harness runs ONE of them and ``run_apzero.py`` loops it over a collection.
Items are (image, category) pairs, the query being the category name.

Live metrics use the GT boxes filled as rectangles when an annotation has no
segmentation (these are box-only datasets); the reported metric is box AP from
the per-instance masks, computed afterwards by ``score_ap.py --ann <json>``.

    python evals/run_cocodet.py --ann <dir>/annotations.coco.json --images <dir>/images \\
        --name odinw/<subset> --dry-run --n 5
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402


def category_query(cat: dict) -> str:
    return (cat.get("name") or "").replace("_", " ").replace("-", " ").strip()


def gt_masks_boxfill(anns, h, w):
    """Per-instance GT masks; a box-only annotation becomes its filled rectangle."""
    out = []
    for a in anns:
        if a.get("iscrowd"):
            continue
        if a.get("segmentation"):
            m = gtmask.ann_to_mask(a["segmentation"], h, w)
        elif a.get("bbox"):
            x, y, bw, bh = a["bbox"]
            m = np.zeros((h, w), dtype=bool)
            m[max(0, int(y)):min(h, int(round(y + bh))), max(0, int(x)):min(w, int(round(x + bw)))] = True
        else:
            continue
        if m.any():
            out.append(m)
    return out


def build(ann_path: Path, images_dir: Path, negatives: int, seed: int):
    d = json.loads(ann_path.read_text())
    cats = {int(c["id"]): c for c in d["categories"]}
    by_pair = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd"):
            by_pair[(int(a["image_id"]), int(a["category_id"]))].append(a)
    positives = defaultdict(list)
    for iid, cid in by_pair:
        positives[iid].append(cid)
    all_cat_ids = sorted(cats)
    items, missing = [], 0
    for im in d["images"]:
        iid = int(im["id"])
        pos = sorted(positives.get(iid, []))
        if not pos and not negatives:
            continue
        path = images_dir / im["file_name"]
        if not path.exists():
            missing += 1
            continue
        chosen = [(cid, True) for cid in pos]
        if negatives:
            pool = [c for c in all_cat_ids if c not in set(pos)]
            rng = random.Random(f"{seed}:{iid}")
            chosen += [(c, False) for c in rng.sample(pool, min(negatives, len(pool)))]
        for cid, is_pos in chosen:
            items.append({"key": f"{iid}_c{cid}", "image_id": iid, "cat_id": cid, "file": str(path),
                          "h": int(im.get("height") or 0), "w": int(im.get("width") or 0),
                          "query": category_query(cats[cid]), "positive": is_pos})
    del d
    gc.collect()
    return items, by_pair, missing


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ann", required=True, help="COCO annotation json")
    ap.add_argument("--images", required=True, help="image directory of that json")
    ap.add_argument("--name", required=True, help="benchmark name in the output path, e.g. odinw/<subset>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--negatives", type=int, default=0, metavar="K",
                    help="unannotated categories asked per image (default 0, as in the paper)")
    C.add_common_args(ap)


def run(args) -> None:
    ann, images = Path(args.ann), Path(args.images)
    if not ann.exists() or not images.is_dir():
        raise SystemExit(f"missing {ann if not ann.exists() else images}")
    all_items, by_pair, missing = build(ann, images, args.negatives, args.seed)
    if not all_items:
        raise SystemExit(f"no runnable items for {ann}")
    items = C.select(all_items, args.n, args.shuffle, args.seed)
    out = C.out_dir(args, args.name, args.split,
                    extra_parts=(f"neg{args.negatives}" if args.negatives else "",))

    def process(it, work):
        shape = (it["h"], it["w"])
        gt_masks = gt_masks_boxfill(by_pair.get((it["image_id"], it["cat_id"]), []), *shape)
        gt = gtmask.union(gt_masks, shape)
        if work is None:
            return (f"{it['key']}: {'POS' if it['positive'] else 'NEG'} instances={len(gt_masks)} "
                    f"cat={it['query']!r} GT-vs-GT IoU={C.iou(gt, gt)[2]:.3f}")
        res = C.run_pipeline(work, Path(it["file"]), it["query"], args)
        rec = {"key": it["key"], "image_id": it["image_id"], "cat_id": it["cat_id"],
               "query": it["query"], "positive": it["positive"], "gt_px": int(gt.sum()),
               "gt_instances": len(gt_masks), "shape": list(shape)}
        return C.score_both_stages(rec, res, gt, gt_masks)

    def summarize(records):
        s = {"n_scored": len(records)}
        s.update(C.presence_f1(records, "seg", group_key="cat_id"))
        s.update(C.presence_f1(records, "rec", group_key="cat_id"))
        s.update(C.iou_summary(records, positives_only=True))
        return s

    C.run_benchmark(items, process, out, args, summarize, args.name, args.split,
                    header_extra=f"ann={ann} missing_images={missing}",
                    provenance={"gt_backend": gtmask.backend(), "ann": str(ann),
                                "images": str(images), "negatives": args.negatives})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)
    run(args)


if __name__ == "__main__":
    main()
