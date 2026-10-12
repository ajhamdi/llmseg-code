#!/usr/bin/env python
"""Zero-shot detection over a whole collection (ODinW13 or RF-100VL): runs every subset.

Each subset is one ``run_cocodet.py`` run (checkpointed separately, so a killed
sweep resumes per subset). Afterwards ``score_apzero.py`` turns each subset's
per-instance masks into box AP and averages them into the collection's AP0.

Paper protocol: ODinW13 every item of every subset (``--n 0``), RF-100VL the
first 20 items of each of the 100 test subsets (``--n 20``).

Data: data/benchmarks/odinw/<subset>/{annotations.coco.json,images/}
      data/benchmarks/rf100vl/<subset>/test/{_annotations.coco.json,*.jpg}
      (python scripts/download_odinw.py; python scripts/download_rf100vl.py)

    python evals/run_apzero.py --collection odinw --dry-run --n 3 --limit 2
    python evals/run_apzero.py --collection odinw --parallel 8
    python evals/run_apzero.py --collection rf100vl --n 20 --parallel 8
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import run_cocodet  # noqa: E402

PAPER_N = {"odinw": 0, "rf100vl": 20}


def subsets(collection: str):
    """Yield (subset name, annotation json, image dir) of a collection's eval split."""
    base = C.DATA / collection
    if not base.is_dir():
        return
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if collection == "odinw":
            ann, imgs = d / "annotations.coco.json", d / "images"
        else:
            ann, imgs = d / "test" / "_annotations.coco.json", d / "test"
        if ann.exists() and imgs.is_dir():
            yield d.name, ann, imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collection", required=True, choices=sorted(PAPER_N))
    ap.add_argument("--limit", type=int, default=0, help="only the first K subsets (smoke tests)")
    ap.add_argument("--negatives", type=int, default=0, metavar="K")
    C.add_common_args(ap)
    ap.set_defaults(n=None)
    args = ap.parse_args()
    C.resolve_args(ap, args)
    if args.n is None:
        args.n = PAPER_N[args.collection]
    if args.out:
        ap.error("--out is per subset; results go to results/<collection>/<subset>/<model>/test")

    subs = list(subsets(args.collection))[: args.limit or None]
    if not subs:
        sys.exit(f"no subsets under {C.DATA / args.collection}\n"
                 f"  -> python scripts/download_{args.collection}.py")
    print(f"[apzero] {args.collection}: {len(subs)} subsets, n={args.n or 'all'} per subset", flush=True)
    for i, (name, ann, imgs) in enumerate(subs, 1):
        print(f"\n=== [{i}/{len(subs)}] {args.collection}/{name} ===", flush=True)
        sub = copy.copy(args)
        sub.ann, sub.images, sub.name, sub.split = str(ann), str(imgs), f"{args.collection}/{name}", "test"
        try:
            run_cocodet.run(sub)
        except SystemExit as e:
            print(f"  [skip] {name}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001 (one broken subset must not stop the sweep)
            print(f"  [error] {name}: {e!r}", flush=True)
    print(f"\n[apzero] done. Score with: python evals/score_apzero.py --collection {args.collection}"
          + (f" --model {args.model}" if args.model else ""), flush=True)


if __name__ == "__main__":
    main()
