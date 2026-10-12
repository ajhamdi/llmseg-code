#!/usr/bin/env python
"""ADE20K (validation): class-prompted semantic segmentation, metric gIoU (+ mIoU).

One item = one (image, class name) pair: "wall", "sky", "building". By default
only the classes present in the image are asked (``--negatives 0``), so the
result is a class-oracle upper bound and NOT the published 150-class ADE20K mIoU
(that would need 150 queries per image). ``--negatives K`` adds K absent classes
per image, which penalises hallucinated classes. Classes covering less than
``--min-class-pct`` of the image are not asked.

Reported: gIoU / cIoU over all items, and mIoU with per-class intersection and
union accumulated over the run before dividing.

Data: data/benchmarks/ade20k/ADEChallengeData2016/{images,annotations}/validation/
      (scripts/download_benchmarks.sh ade20k)

    python evals/run_ade20k.py --dry-run --n 5
    python evals/run_ade20k.py --n 1000 --shuffle --parallel 8
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ADE = C.DATA / "ade20k" / "ADEChallengeData2016"
N_CLASSES = 150


def load_class_names() -> dict:
    """class id (1..150) -> first synonym of objectInfo150.txt ("building, edifice" -> "building")."""
    names = {}
    for line in (ADE / "objectInfo150.txt").read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5:
            names[int(parts[0])] = parts[4].split(",")[0].strip()
    if len(names) != N_CLASSES:
        sys.exit(f"objectInfo150.txt gave {len(names)} classes, expected {N_CLASSES}")
    return names


def read_label(path: Path) -> np.ndarray:
    from PIL import Image
    a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def build_items(names: dict, negatives: int, min_frac: float, seed: int) -> list:
    img_dir, ann_dir = ADE / "images" / "validation", ADE / "annotations" / "validation"
    items = []
    for lp in sorted(ann_dir.glob("*.png")):
        ip = img_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        lab = read_label(lp)
        ids, counts = np.unique(lab, return_counts=True)
        present = {int(i) for i, c in zip(ids, counts) if i != 0 and c >= min_frac * lab.size}
        chosen = [(cid, True) for cid in sorted(present)]
        if negatives:
            absent = [c for c in range(1, N_CLASSES + 1) if c not in present]
            rng = random.Random(f"{seed}:{lp.stem}")
            chosen += [(c, False) for c in rng.sample(absent, min(negatives, len(absent)))]
        for cid, is_pos in chosen:
            items.append({"key": f"{lp.stem}_c{cid:03d}", "image": str(ip), "label": str(lp),
                          "class_id": cid, "query": names[cid], "positive": is_pos})
    return items


def per_class_miou(records, prefix):
    acc = defaultdict(lambda: [0, 0])
    for r in records:
        acc[r["class_id"]][0] += r.get(f"{prefix}_inter", 0)
        acc[r["class_id"]][1] += r.get(f"{prefix}_union", 0)
    ious = [i / u for i, u in acc.values() if u > 0]
    return round(float(np.mean(ious)), 4) if ious else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--negatives", type=int, default=0, metavar="K",
                    help="absent classes asked per image (default 0 = class-oracle, as in the paper)")
    ap.add_argument("--min-class-pct", type=float, default=0.1, metavar="PCT",
                    help="skip classes covering less than PCT %% of the image (default 0.1)")
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)
    C.require(ADE, "bash scripts/download_benchmarks.sh ade20k")

    items = C.select(build_items(load_class_names(), args.negatives, args.min_class_pct / 100.0, args.seed),
                     args.n, args.shuffle, args.seed)
    extra = (f"neg{args.negatives}" if args.negatives else "",
             f"min{args.min_class_pct:g}" if args.min_class_pct != 0.1 else "")
    out = C.out_dir(args, "ade20k", "validation", extra_parts=extra)

    def process(it, work):
        gt = read_label(Path(it["label"])) == it["class_id"]
        if work is None:
            return (f"{it['key']}: {'POS' if it['positive'] else 'NEG'} gt_px={int(gt.sum())} "
                    f"class={it['query']!r} GT-vs-GT IoU={C.iou(gt, gt)[2]:.3f}")
        res = C.run_pipeline(work, Path(it["image"]), it["query"], args)
        rec = {"key": it["key"], "class_id": it["class_id"], "query": it["query"],
               "positive": it["positive"], "gt_px": int(gt.sum()), "shape": list(gt.shape)}
        return C.score_both_stages(rec, res, gt)

    def summarize(records):
        s = {"n_scored": len(records), **C.iou_summary(records),
             "seg_mIoU": per_class_miou(records, "seg"), "rec_mIoU": per_class_miou(records, "rec")}
        s.update(C.rejection_summary(records))
        return s

    C.run_benchmark(items, process, out, args, summarize, "ade20k", "validation",
                    header_extra=f"negatives={args.negatives}",
                    provenance={"negatives": args.negatives, "min_class_pct": args.min_class_pct})


if __name__ == "__main__":
    main()
