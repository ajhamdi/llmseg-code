#!/usr/bin/env python
"""LVIS v1 val / COCO val2017: category-prompted instance segmentation, metric cgF1 (+ AP).

One item = one (image, category) pair, the query being the category name. With
``--negatives K`` each image also gets K categories verified absent:

    lvis  sampled from the image's ``neg_category_ids`` (LVIS is federated, so
          only those are known to be absent)
    coco  sampled from the categories without an annotation (COCO is exhaustive)

Live metrics: IL_MCC / pmF1 / cgF1 (as in run_saco.py, grouped per category),
gIoU / cIoU over positives, rejection rate on negatives. Box / mask AP is computed
afterwards from the per-instance masks: ``python evals/score_ap.py --results <out> --dataset lvis``.

Data: data/benchmarks/lvis/lvis_v1_val.json, data/benchmarks/coco/annotations/instances_val2017.json
      images under data/benchmarks/coco/images/{val2017,train2017}/
      (scripts/download_benchmarks.sh lvis coco; scripts/download_coco_images.sh val2017 lvis)

    python evals/run_instseg.py --dataset lvis --negatives 3 --dry-run --n 5
    python evals/run_instseg.py --dataset lvis --negatives 3 --n 1500 --shuffle --parallel 8
    python evals/run_instseg.py --dataset coco --negatives 3 --n 1500 --shuffle --parallel 8
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402

COCO_IMAGES = C.DATA / "coco" / "images"
SOURCES = {
    "lvis": C.DATA / "lvis" / "lvis_v1_val.json",
    "coco": C.DATA / "coco" / "annotations" / "instances_val2017.json",
}


def category_query(cat: dict) -> str:
    """'aerosol_can' -> 'aerosol can'."""
    return (cat.get("name") or "").replace("_", " ").replace("-", " ").strip()


def image_file(dataset: str, im: dict) -> Path:
    """LVIS val spans val2017 and train2017; only ``coco_url`` says which."""
    if dataset == "lvis":
        parts = (im.get("coco_url") or "").rstrip("/").split("/")
        if len(parts) >= 2:
            return COCO_IMAGES / parts[-2] / parts[-1]
    return COCO_IMAGES / "val2017" / im["file_name"]


def build(dataset: str, negatives: int, seed: int):
    """-> (items, {(image_id, cat_id): [ann, ...]}, n_missing_images)."""
    src = C.require(SOURCES[dataset], f"bash scripts/download_benchmarks.sh {dataset}")
    d = json.loads(src.read_text())
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
        path = image_file(dataset, im)
        if not path.exists():
            missing += 1
            continue
        chosen = [(cid, True) for cid in pos]
        if negatives:
            pool = ([int(c) for c in (im.get("neg_category_ids") or [])] if dataset == "lvis"
                    else [c for c in all_cat_ids if c not in set(pos)])
            rng = random.Random(f"{seed}:{iid}")
            chosen += [(c, False) for c in rng.sample(pool, min(negatives, len(pool)))]
        for cid, is_pos in chosen:
            items.append({"key": f"{iid:012d}_c{cid:04d}", "image_id": iid, "cat_id": cid,
                          "file": str(path), "h": int(im["height"]), "w": int(im["width"]),
                          "query": category_query(cats[cid]), "positive": is_pos})
    del d
    gc.collect()
    return items, by_pair, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="lvis", choices=sorted(SOURCES))
    ap.add_argument("--negatives", type=int, default=3, metavar="K",
                    help="verified-absent categories asked per image (paper: 3)")
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)

    all_items, by_pair, missing = build(args.dataset, args.negatives, args.seed)
    if not all_items:
        sys.exit(f"no images on disk for {args.dataset}\n  -> bash scripts/download_coco_images.sh val2017 lvis")
    items = C.select(all_items, args.n, args.shuffle, args.seed)
    split = "val2017" if args.dataset == "coco" else "v1_val"
    out = C.out_dir(args, args.dataset, split, extra_parts=(f"neg{args.negatives}",))

    def process(it, work):
        shape = (it["h"], it["w"])
        gt_masks = gtmask.anns_to_masks(by_pair.get((it["image_id"], it["cat_id"]), []), *shape)
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
        s = {"n_scored": len(records),
             "n_positive": sum(1 for r in records if r.get("gt_px", 0) > 0),
             "n_negative": sum(1 for r in records if r.get("gt_px", 0) == 0)}
        s.update(C.presence_f1(records, "seg", group_key="cat_id"))
        s.update(C.presence_f1(records, "rec", group_key="cat_id"))
        s.update(C.iou_summary(records, positives_only=True))
        s.update(C.rejection_summary(records))
        return s

    C.run_benchmark(items, process, out, args, summarize, args.dataset, split,
                    header_extra=f"negatives={args.negatives} missing_images={missing}",
                    provenance={"gt_backend": gtmask.backend(), "negatives": args.negatives,
                                "images_missing": missing,
                                "instance_source": "connected components of the union mask"})


if __name__ == "__main__":
    main()
