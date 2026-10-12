#!/usr/bin/env python
"""ReasonSeg (val / test): reasoning segmentation, metric gIoU and cIoU.

The query is an implicit description ("something used for playing music") rather
than a name. Every parquet row is self-contained: image bytes, query, GT mask.
A few rows have an empty GT mask (the object is absent); they are scored, not
skipped: an empty prediction scores IoU 1 there.

    gIoU = mean per-item IoU,  cIoU = sum(I) / sum(U)  (both stages: seg_*, rec_*)

Data: data/benchmarks/reasonseg/{val,test}/data/*.parquet  (scripts/download_benchmarks.sh reasonseg)

    python evals/run_reasonseg.py --split val --dry-run --n 5
    python evals/run_reasonseg.py --split val --parallel 8          # full val (200 items)
    python evals/run_reasonseg.py --split test --parallel 8         # full test (779 items)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def open_table(split: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ddir = C.DATA / "reasonseg" / split / "data"
    files = sorted(ddir.glob("*.parquet"))
    if not files:
        sys.exit(f"no parquet under {ddir}\n  -> bash scripts/download_benchmarks.sh reasonseg")
    return pa.concat_tables([pq.read_table(f) for f in files])


def row_key(table, idx: int) -> str:
    """Checkpoint key read from the id columns only (no image / mask decode)."""
    aid = table.column("ann_id")[idx].as_py() or table.column("image_id")[idx].as_py() or ""
    return f"{idx:04d}_{aid}" if aid else f"{idx:04d}"


def materialize(table, idx: int) -> dict:
    row = table.slice(idx, 1).to_pydict()
    mask = np.asarray(row["mask"][0], dtype=bool)
    if mask.ndim != 2:
        mask = mask.reshape(row["img_height"][0], row["img_width"][0])
    return {"image_bytes": row["image"][0]["bytes"], "query": (row["text"][0] or "").strip(),
            "gt": mask, "image_id": row["image_id"][0], "ann_id": row["ann_id"][0]}


def write_image(dst: Path, data: bytes) -> Path:
    ext = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
    p = dst / f"image.{ext}"
    p.write_bytes(data)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)

    table = open_table(args.split)
    items = C.select([{"key": row_key(table, i), "idx": i} for i in range(table.num_rows)],
                     args.n, args.shuffle, args.seed)
    out = C.out_dir(args, "reasonseg", args.split)

    def process(it, work):
        row = materialize(table, it["idx"])
        gt = row["gt"]
        if work is None:
            return (f"{it['key']}: gt={gt.shape} gt_px={int(gt.sum())} "
                    f"query={row['query'][:60]!r} GT-vs-GT IoU={C.iou(gt, gt)[2]:.3f}")
        img = write_image(work, row["image_bytes"])
        res = C.run_pipeline(work, img, row["query"], args)
        rec = {"key": it["key"], "image_id": row["image_id"], "ann_id": row["ann_id"],
               "query": row["query"], "gt_px": int(gt.sum()), "shape": list(gt.shape)}
        return C.score_both_stages(rec, res, gt)

    def summarize(records):
        return {"n_scored": len(records), **C.iou_summary(records)}

    C.run_benchmark(items, process, out, args, summarize, "reasonseg", args.split)


if __name__ == "__main__":
    main()
