#!/usr/bin/env python
"""SA-Co/Gold (and Silver): promptable concept segmentation, metric cgF1 / pmF1.

One item = one (image, noun phrase) pair; most pairs are negatives (the phrase
is absent and the right answer is an empty mask), so the metric gates instance
quality on the present/absent decision:

    IL_MCC  Matthews correlation of present/absent over all pairs
    pmF1    instance F1 at IoU >= 0.5, pooled per noun phrase over positive
            pairs, macro-averaged over phrases (instances = connected
            components of the predicted union, the same for every method)
    cgF1    100 * IL_MCC * pmF1
    plus gIoU / cIoU over positives and the rejection rate on negatives.

Gold ships 7 domains x 3 annotation replicas (a/b/c) of the same images; the paper
uses replica a. Items whose image is missing on disk are dropped and counted.

Data: data/benchmarks/saco/{gold,silver}/*.json + .../images/<file_name>
      (scripts/download_benchmarks.sh saco; scripts/download_saco_images.py gold)

    python evals/run_saco.py --split gold --domain all --dry-run --n 5
    python evals/run_saco.py --split gold --domain all --n 2500 --shuffle --parallel 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402

SACO = C.DATA / "saco"
GOLD_RE = re.compile(r"^gold_(?P<domain>.+)_merged_(?P<rep>[abc])_release_test$")
SILVER_RE = re.compile(r"^silver_(?P<domain>.+?)(?:_merged)?_test$")


def domain_files(split: str, replica: str) -> dict:
    out = {}
    for p in sorted((SACO / split).glob("*.json")):
        m = (GOLD_RE if split == "gold" else SILVER_RE).match(p.stem)
        if m and (split != "gold" or m.group("rep") == replica):
            out[m.group("domain")] = p
    return out


def build_items(split: str, domains: dict):
    items, missing = [], 0
    img_root = SACO / split / "images"
    for dom, path in domains.items():
        for im in json.loads(path.read_text())["images"]:
            p = img_root / im["file_name"]
            if not p.exists():
                missing += 1
                continue
            items.append({"key": f"{dom}_{im['id']}", "domain": dom, "image_id": int(im["id"]),
                          "file": str(p), "query": (im.get("text_input") or "").strip(),
                          "h": int(im["height"]), "w": int(im["width"])})
    return items, missing


def load_annotations(domains: dict, items: list) -> dict:
    """{domain: {image_id: [ann, ...]}} for the selected items only (the files are large)."""
    keep = defaultdict(set)
    for it in items:
        keep[it["domain"]].add(it["image_id"])
    index = {}
    for dom, path in domains.items():
        if not keep[dom]:
            continue
        idx = defaultdict(list)
        for a in json.loads(path.read_text())["annotations"]:
            if int(a["image_id"]) in keep[dom]:
                idx[int(a["image_id"])].append(a)
        index[dom] = idx
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="gold", choices=["gold", "silver"])
    ap.add_argument("--domain", default="all", help="domain name, comma-separated list, or 'all'")
    ap.add_argument("--replica", default="a", choices=["a", "b", "c"], help="Gold annotation replica")
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)

    available = domain_files(args.split, args.replica)
    if not available:
        sys.exit(f"no {args.split} label files under {SACO / args.split}\n"
                 f"  -> bash scripts/download_benchmarks.sh saco")
    want = sorted(available) if args.domain == "all" else [d.strip() for d in args.domain.split(",")]
    unknown = [d for d in want if d not in available]
    if unknown:
        sys.exit(f"unknown domain(s) {unknown}; {args.split} has: {', '.join(sorted(available))}")
    domains = {d: available[d] for d in want}

    all_items, missing = build_items(args.split, domains)
    if not all_items:
        sys.exit(f"no images on disk for {want}\n  -> python scripts/download_saco_images.py {args.split}")
    items = C.select(all_items, args.n, args.shuffle, args.seed)
    anns = load_annotations(domains, items)
    split = f"{args.split}_{'-'.join(want) if args.domain != 'all' else 'all'}" + \
        (f"_{args.replica}" if args.split == "gold" else "")
    out = C.out_dir(args, "saco", split)

    def process(it, work):
        shape = (it["h"], it["w"])
        gt_masks = gtmask.anns_to_masks(anns.get(it["domain"], {}).get(it["image_id"], []), *shape)
        gt = gtmask.union(gt_masks, shape)
        if work is None:
            return (f"{it['key']}: {'POS' if gt.any() else 'NEG'} instances={len(gt_masks)} "
                    f"np={it['query'][:40]!r} GT-vs-GT IoU={C.iou(gt, gt)[2]:.3f}")
        res = C.run_pipeline(work, Path(it["file"]), it["query"], args)
        rec = {"key": it["key"], "domain": it["domain"], "image_id": it["image_id"],
               "query": it["query"], "gt_px": int(gt.sum()), "gt_instances": len(gt_masks),
               "shape": list(shape)}
        return C.score_both_stages(rec, res, gt, gt_masks)

    def summarize(records):
        s = {"n_scored": len(records),
             "n_positive": sum(1 for r in records if r.get("gt_px", 0) > 0),
             "n_negative": sum(1 for r in records if r.get("gt_px", 0) == 0)}
        s.update(C.presence_f1(records, "seg"))
        s.update(C.presence_f1(records, "rec"))
        s.update(C.iou_summary(records, positives_only=True))
        s.update(C.rejection_summary(records))
        return s

    C.run_benchmark(items, process, out, args, summarize, "saco", split,
                    header_extra=f"domains={','.join(want)} missing_images={missing}",
                    provenance={"gt_backend": gtmask.backend(), "images_missing": missing,
                                "domains": want, "replica": args.replica,
                                "instance_source": "connected components of the union mask"})


if __name__ == "__main__":
    main()
