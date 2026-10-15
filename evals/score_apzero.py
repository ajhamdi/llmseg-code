#!/usr/bin/env python
"""Collection AP0 (ODinW13 / RF-100VL): mean over subsets of each subset's box AP.

Run after ``run_apzero.py``. Writes results/<collection>/AP0_<model>_<stage>.json.

    python evals/score_apzero.py --collection odinw --model gemini-3.8-flash
    python evals/score_apzero.py --collection rf100vl --model gemini-3.8-flash --stage seg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import score_ap  # noqa: E402
from run_apzero import subsets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collection", required=True, choices=["odinw", "rf100vl"])
    ap.add_argument("--model", required=True, help="model name used for the run (its results directory)")
    ap.add_argument("--run-dir", default="test",
                    help="run directory name under <subset>/<model>/ (default 'test'; e.g. 'test__it0')")
    ap.add_argument("--stage", default="rec", choices=["rec", "seg"])
    ap.add_argument("--box-source", default="mask", choices=["mask", "detect"])
    a = ap.parse_args()

    rows = []
    for name, ann, _ in subsets(a.collection):
        rdir = C.RESULTS / a.collection / name / a.model.replace("/", "_") / a.run_dir
        if not (rdir / "results.jsonl").exists():
            print(f"  [missing] {name}: no results.jsonl", flush=True)
            continue
        try:
            d = score_ap.score(rdir, ann, a.stage, a.box_source)
        except SystemExit as e:
            print(f"  [skip] {name}: {e}", flush=True)
            continue
        box_ap = (d.get("box_AP") or {}).get("AP")
        rows.append({"subset": name, "AP": box_ap, "n_ap_images": d["n_ap_images"],
                     "n_ap_total_images": d["n_ap_total_images"]})
        print(f"  {name:42s} box AP={box_ap}  ({d['n_ap_images']}/{d['n_ap_total_images']} images covered)",
              flush=True)

    valid = [r["AP"] for r in rows if r["AP"] is not None]
    ap0 = round(100.0 * sum(valid) / len(valid), 2) if valid else None
    print(f"\n{a.collection} AP0 = {ap0}  (mean box AP over {len(valid)} subsets, stage={a.stage})")
    out = C.RESULTS / a.collection / f"AP0_{a.model.replace('/', '_')}_{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"collection": a.collection, "model": a.model, "stage": a.stage,
                               "box_source": a.box_source, "AP0": ap0,
                               "n_subsets_scored": len(valid), "per_subset": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
