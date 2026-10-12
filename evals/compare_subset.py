#!/usr/bin/env python
"""Compare two runs of the same benchmark on exactly their shared items.

Item keys are built by the loaders from the dataset alone, so two runs (two
models, two providers, two configurations, or any external method that writes the
same ``results.jsonl`` fields) can be compared after the fact on the
intersection of their keys, whatever ``--n`` / ``--shuffle`` each used.
Benchmarks with sampled negatives (LVIS, COCO, ADE20K) only match when both
runs used the same ``--negatives`` and ``--seed``.

Every metric is recomputed from the per-item fields of ``results.jsonl`` (no
masks needed); with ``--ann`` box / mask AP is also re-scored on the shared items.

    python evals/compare_subset.py --a results/reasonseg/gemini-3.7-flash/val \\
                                   --b results/reasonseg/claude-sonnet-5-5/val
    python evals/compare_subset.py --a <lvis run A> --b <lvis run B> --ann data/benchmarks/lvis/lvis_v1_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load(run_dir: Path) -> dict:
    p = run_dir / "results.jsonl"
    if not p.exists():
        sys.exit(f"no results.jsonl in {run_dir}")
    return {r["key"]: r for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))}


def metrics(records: list) -> dict:
    if not records:
        return {}
    out = {}
    if isinstance(records[0].get("rec_inter"), list):          # semantic segmentation (per-class lists)
        for p in ("seg", "rec"):
            inter = np.sum([r[f"{p}_inter"] for r in records], axis=0)
            union = np.sum([r[f"{p}_union"] for r in records], axis=0)
            v = [i / u for i, u in zip(inter, union) if u]
            out[f"{p}_mIoU"] = round(float(np.mean(v)), 4) if v else None
        return out
    has_negatives = any(r.get("gt_px", 1) == 0 for r in records)
    out.update(C.iou_summary(records))
    if has_negatives:
        out.update(C.iou_summary(records, positives_only=True))
    if any("rec_tp" in r for r in records):
        group = "cat_id" if "cat_id" in records[0] else "query"
        for p in ("seg", "rec"):
            out.update({k: v for k, v in C.presence_f1(records, p, group).items() if not k.endswith("presence")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="run directory A")
    ap.add_argument("--b", required=True, help="run directory B")
    ap.add_argument("--ann", help="COCO GT json: also compute box / mask AP on the shared items")
    ap.add_argument("--stage", default="rec", choices=["rec", "seg"], help="stage used for AP")
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    keys = set(A) & set(B)
    print(f"A: {len(A)} items   B: {len(B)} items   shared: {len(keys)}")
    if not keys:
        sys.exit("no shared keys (different benchmark, --negatives or --seed?)")
    mA, mB = metrics([A[k] for k in sorted(keys)]), metrics([B[k] for k in sorted(keys)])
    if args.ann:
        import score_ap
        for m, d in ((mA, args.a), (mB, args.b)):
            s = score_ap.score(Path(d), Path(args.ann), args.stage, "mask", keys)
            m["box_AP"], m["mask_AP"] = s["box_AP"].get("AP"), s["mask_AP"].get("AP")
    names = list(dict.fromkeys([*mA, *mB]))
    w = max(map(len, names))
    print(f"\n{'metric'.ljust(w)}  {'A':>10}  {'B':>10}")
    for k in names:
        print(f"{k.ljust(w)}  {str(mA.get(k, '-')):>10}  {str(mB.get(k, '-')):>10}")


if __name__ == "__main__":
    main()
