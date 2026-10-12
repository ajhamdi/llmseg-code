#!/usr/bin/env python
"""Box / mask AP of a finished category-prompted run (LVIS, COCO, ODinW, RF-100VL, any COCO json).

AP needs ranked, scored per-instance proposals. Every pipeline run writes them to
``work/<item>/instances.json``: one entry per instance with its stage-2 and final
mask (COCO RLE), its box and its detection confidence (the ranking score). This
script scores them with pycocotools COCOeval (``bbox`` and ``segm``) against the
dataset GT, and also reports pmF1 / cgF1 on these true instances (the live
summary uses connected components of the union mask instead).

Coverage: an image enters AP only if every GT category in it was queried by the
run, otherwise an un-queried category would count as missed. ``n_ap_images`` /
``n_ap_total_images`` reports the fraction. On LVIS this is COCO-style AP over the
covered images, not the federated LVIS AP. Note that with ``--save`` pruning
only PNGs are removed; instances.json is always kept.

    python evals/score_ap.py --results results/lvis/<model>/v1_val__neg3__shuf0 --dataset lvis
    python evals/score_ap.py --results results/odinw/<subset>/<model>/test --ann <subset json>
    python evals/score_ap.py --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402

SOURCES = {
    "lvis": C.DATA / "lvis" / "lvis_v1_val.json",
    "coco": C.DATA / "coco" / "annotations" / "instances_val2017.json",
}
DEFAULT_SCORE = 0.5   # ranking score of an instance whose detection carried no confidence


def _pycoco():
    try:
        from pycocotools import mask as M
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return COCO, COCOeval, M
    except ImportError:
        sys.exit("AP needs pycocotools: pip install -r requirements-evals.txt")


def tight_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def encode_rle(mask, M) -> dict:
    rle = M.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def proposals(work: Path, stage: str, box_source: str = "mask") -> list:
    """[{mask, box [x1,y1,x2,y2], score}] of one item from its instances.json."""
    p = work / "instances.json"
    if not p.exists():
        return []
    out = []
    for inst in json.loads(p.read_text()).get("instances", []):
        rle = inst.get(f"{stage}_rle")
        if not rle:
            continue
        m = gtmask.ann_to_mask(rle, *rle["size"])
        tb = tight_box(m)
        if tb is None:
            continue
        box = inst["box"] if (box_source == "detect" and inst.get("box")) else tb
        conf = inst.get("confidence")
        try:
            score = float(conf)
        except (TypeError, ValueError):
            score = DEFAULT_SCORE
        out.append({"mask": m, "box": [float(v) for v in box], "score": score})
    return out


def coco_ap(gt_dict, dt_list, img_ids, cat_ids, iou_type) -> dict:
    COCO, COCOeval, _ = _pycoco()
    if not dt_list:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0, "n_dt": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dt_list), iou_type)
        ev.params.imgIds = sorted(img_ids)
        ev.params.catIds = sorted(cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"AP": round(float(s[0]), 4), "AP50": round(float(s[1]), 4),
            "AP75": round(float(s[2]), 4), "AR100": round(float(s[8]), 4), "n_dt": len(dt_list)}


def load_gt(src: Path):
    d = json.loads(src.read_text())
    images = {int(im["id"]): im for im in d["images"]}
    cats = [{"id": int(c["id"]), "name": c.get("name", str(c["id"]))} for c in d["categories"]]
    anns_by_img, anns_by_pair, cats_by_img = defaultdict(list), defaultdict(list), defaultdict(set)
    for n, a in enumerate(d["annotations"], 1):
        if a.get("iscrowd"):
            continue
        iid, cid = int(a["image_id"]), int(a["category_id"])
        a = dict(a)
        a.setdefault("id", n)
        a.setdefault("iscrowd", 0)
        if "area" not in a and a.get("bbox"):
            a["area"] = float(a["bbox"][2] * a["bbox"][3])
        anns_by_img[iid].append(a)
        anns_by_pair[(iid, cid)].append(a)
        cats_by_img[iid].add(cid)
    return images, cats, anns_by_img, anns_by_pair, cats_by_img


def score(results: Path, ann: Path, stage: str = "rec", box_source: str = "mask", keys=None) -> dict:
    _, _, M = _pycoco()
    recs = [json.loads(line) for line in (results / "results.jsonl").read_text().splitlines() if line.strip()]
    recs = [r for r in recs if "image_id" in r and "cat_id" in r and (keys is None or r["key"] in keys)]
    if not recs:
        raise SystemExit(f"no (image, category) records in {results / 'results.jsonl'}")
    images, cats, anns_by_img, anns_by_pair, cats_by_img = load_gt(ann)

    queried = defaultdict(set)
    for r in recs:
        queried[int(r["image_id"])].add(int(r["cat_id"]))
    covered = {i for i in queried if cats_by_img[i] <= queried[i]}
    gt_dict = {"images": [images[i] for i in covered],
               "annotations": [a for i in covered for a in anns_by_img[i]], "categories": cats}
    cat_ids = {int(r["cat_id"]) for r in recs if int(r["image_id"]) in covered}

    dt, per_cat = [], defaultdict(lambda: [0, 0, 0])
    tp = tn = fp = fn = 0
    for r in recs:
        iid, cid = int(r["image_id"]), int(r["cat_id"])
        props = proposals(results / "work" / r["key"], stage, box_source)
        if iid in covered:
            for p in props:
                b = p["box"]
                dt.append({"image_id": iid, "category_id": cid, "score": p["score"],
                           "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                           "segmentation": encode_rle(p["mask"], M)})
        gt_present, pred_present = r.get("gt_px", 0) > 0, bool(props)
        tp += gt_present and pred_present
        fn += gt_present and not pred_present
        fp += pred_present and not gt_present
        tn += not gt_present and not pred_present
        if gt_present and r.get("shape"):
            gts = gtmask.anns_to_masks(anns_by_pair.get((iid, cid), []), *r["shape"])
            t, f_p, f_n = C.match_f1([p["mask"] for p in props], gts)
            acc = per_cat[cid]
            acc[0] += t
            acc[1] += f_p
            acc[2] += f_n

    has_masks = any(a.get("segmentation") for a in gt_dict["annotations"])
    f1s = []
    for t, f_p, f_n in per_cat.values():
        prec, rec_ = (t / (t + f_p) if t + f_p else 0.0), (t / (t + f_n) if t + f_n else 0.0)
        f1s.append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
    il_mcc = C.mcc(int(tp), int(tn), int(fp), int(fn))
    pmf1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    out = {"results": str(results), "ann": str(ann), "stage": stage, "box_source": box_source,
           "n_records": len(recs), "n_ap_images": len(covered), "n_ap_total_images": len(queried),
           "box_AP": (coco_ap(gt_dict, dt, covered, cat_ids, "bbox") if covered
                      else {"skipped": "no image had all of its GT categories queried"}),
           "mask_AP": (coco_ap(gt_dict, dt, covered, cat_ids, "segm") if covered and has_masks
                       else {"skipped": "no fully queried image" if not covered
                             else "GT has no masks (box-only dataset)"}),
           "instance_IL_MCC": il_mcc, "instance_pmF1": pmf1,
           "instance_cgF1": round(100.0 * il_mcc * pmf1, 2)}
    (results / f"ap_summary_{stage}_{box_source}.json").write_text(json.dumps(out, indent=2))
    return out


def selftest() -> None:
    """Prediction == GT must give AP ~ 1.0 (validates the COCOeval plumbing, no data needed)."""
    _, _, M = _pycoco()
    boxes = {1: [(10, 10, 40, 40)], 2: [(50, 50, 90, 80), (5, 60, 20, 90)]}
    anns, dt = [], []
    for iid, bs in boxes.items():
        for x1, y1, x2, y2 in bs:
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "iscrowd": 0,
                         "area": (x2 - x1) * (y2 - y1), "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]})
            m = np.zeros((100, 100), np.uint8)
            m[y1:y2, x1:x2] = 1
            dt.append({"image_id": iid, "category_id": 1, "score": 1.0,
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": encode_rle(m, M)})
    gt = {"images": [{"id": i, "width": 100, "height": 100} for i in boxes], "annotations": anns,
          "categories": [{"id": 1, "name": "thing"}]}
    b, s = coco_ap(gt, dt, [1, 2], [1], "bbox"), coco_ap(gt, dt, [1, 2], [1], "segm")
    ok = b["AP"] > 0.99 and s["AP"] > 0.99
    print(f"box AP {b['AP']}  mask AP {s['AP']}  ->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", help="a finished run directory (contains results.jsonl and work/)")
    ap.add_argument("--dataset", choices=sorted(SOURCES), help="built-in GT (lvis / coco)")
    ap.add_argument("--ann", help="any COCO GT json (ODinW / RF-100VL subsets, ...)")
    ap.add_argument("--stage", choices=["rec", "seg"], default="rec",
                    help="rec = final masks (default), seg = before refinement")
    ap.add_argument("--box-source", choices=["mask", "detect"], default="mask",
                    help="mask: tight box of the instance mask (default); detect: the stage-1 box")
    ap.add_argument("--keys", help="file of item keys (one per line): score only that subset")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if not args.results or not (args.dataset or args.ann):
        ap.error("need --results and one of --dataset / --ann (or --selftest)")
    keys = ({k.strip() for k in Path(args.keys).read_text().splitlines() if k.strip()}
            if args.keys else None)
    ann = Path(args.ann) if args.ann else C.require(SOURCES[args.dataset], "download the annotations first")
    print(json.dumps(score(Path(args.results), ann, args.stage, args.box_source, keys), indent=2))


if __name__ == "__main__":
    main()
