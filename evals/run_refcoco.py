#!/usr/bin/env python
"""RefCOCO / RefCOCO+ / RefCOCOg: referring expression segmentation, metric gIoU and cIoU.

One item = one (referring expression, target object) pair. A ref carries several
sentences and each is scored separately (the standard protocol). All splits of
all three datasets use COCO train2014 images.

Data: data/benchmarks/refcoco/{refcoco,refcocoplus,refcocog}/data/*.parquet
      data/benchmarks/coco/images/train2014/
      (scripts/download_benchmarks.sh refcoco; scripts/download_coco_images.sh train2014)

    python evals/run_refcoco.py --dataset refcoco --split val --dry-run --n 5
    python evals/run_refcoco.py --dataset refcoco --split testA --n 500 --shuffle --parallel 8
    python evals/run_refcoco.py --dataset refcocog --split test --n 500 --shuffle --parallel 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402

DATASETS = ("refcoco", "refcocoplus", "refcocog")
IMAGE_DIR = C.DATA / "coco" / "images" / "train2014"


def open_table(dataset: str, split: str):
    """Rows of one split, selected by the ``split`` column (not the file name)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    ddir = C.DATA / "refcoco" / dataset / "data"
    files = sorted(ddir.glob("*.parquet"))
    if not files:
        sys.exit(f"no parquet under {ddir}\n  -> bash scripts/download_benchmarks.sh refcoco")
    t = pa.concat_tables([pq.read_table(f) for f in files])
    sel = t.filter(pc.equal(t.column("split"), split))
    if sel.num_rows == 0:
        sys.exit(f"{dataset} has no split {split!r}; available: {sorted(set(t.column('split').to_pylist()))}")
    return sel


def build_items(table) -> list:
    items = []
    cols = zip(table.column("ann_id").to_pylist(), table.column("image_id").to_pylist(),
               table.column("sentences").to_pylist())
    for idx, (aid, iid, sents) in enumerate(cols):
        texts = [t for t in ((s.get("sent") or s.get("raw") or "").strip() for s in (sents or [])) if t]
        for si, text in enumerate(texts):
            items.append({"key": f"{idx:05d}_{aid}_s{si}", "idx": idx, "ann_id": aid,
                          "image_id": iid, "query": text})
    return items


def gt_for(table, idx: int):
    row = table.slice(idx, 1).to_pydict()
    info = json.loads(row["raw_image_info"][0])
    ann = json.loads(row["raw_anns"][0])
    h, w = int(info["height"]), int(info["width"])
    return gtmask.ann_to_mask(ann.get("segmentation"), h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="refcoco", choices=DATASETS)
    ap.add_argument("--split", default="val", help="val | testA | testB (refcoco, refcocoplus); val | test (refcocog)")
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)
    C.require(IMAGE_DIR, "bash scripts/download_coco_images.sh train2014")

    table = open_table(args.dataset, args.split)
    split = f"{args.dataset}_{args.split}"
    items = C.select(build_items(table), args.n, args.shuffle, args.seed)
    out = C.out_dir(args, "refcoco", split)

    def process(it, work):
        gt = gt_for(table, it["idx"])
        img = IMAGE_DIR / f"COCO_train2014_{int(it['image_id']):012d}.jpg"
        if work is None:
            return (f"{it['key']}: gt={gt.shape} gt_px={int(gt.sum())} "
                    f"img={'ok' if img.exists() else 'MISSING'} query={it['query'][:60]!r} "
                    f"GT-vs-GT IoU={C.iou(gt, gt)[2]:.3f}")
        if not img.exists():
            raise FileNotFoundError(img)
        res = C.run_pipeline(work, img, it["query"], args)
        rec = {"key": it["key"], "image_id": it["image_id"], "ann_id": it["ann_id"],
               "query": it["query"], "gt_px": int(gt.sum()), "shape": list(gt.shape)}
        return C.score_both_stages(rec, res, gt)

    def summarize(records):
        return {"n_scored": len(records), **C.iou_summary(records)}

    C.run_benchmark(items, process, out, args, summarize, "refcoco", split,
                    provenance={"gt_backend": gtmask.backend()})


if __name__ == "__main__":
    main()
