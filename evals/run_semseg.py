#!/usr/bin/env python
"""Cityscapes (val): 19-class semantic segmentation, metric mIoU.

One item = one image. The pipeline is run ONCE per image with all 19 class names
as the query ("road, sidewalk, building, ..."); every detected instance carries a
class label, and the labelled per-instance masks are composited into one label
map (``compose_label_map``). mIoU accumulates per-class intersection and union
over all validation pixels (GT trainId 255 ignored) before dividing.

Data: data/benchmarks/cityscapes/{leftImg8bit,gtFine}/val/
      (needs a free Cityscapes account; see scripts/download_cityscapes.sh)

    python evals/run_semseg.py --dry-run --n 3
    python evals/run_semseg.py --parallel 8                  # full val, 500 images
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gtmask  # noqa: E402

# (class name, Cityscapes labelId, trainId), from the official cityscapesScripts labels.py
CLASSES = [
    ("road", 7, 0), ("sidewalk", 8, 1), ("building", 11, 2), ("wall", 12, 3),
    ("fence", 13, 4), ("pole", 17, 5), ("traffic light", 19, 6), ("traffic sign", 20, 7),
    ("vegetation", 21, 8), ("terrain", 22, 9), ("sky", 23, 10), ("person", 24, 11),
    ("rider", 25, 12), ("car", 26, 13), ("truck", 27, 14), ("bus", 28, 15),
    ("train", 31, 16), ("motorcycle", 32, 17), ("bicycle", 33, 18),
]
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
THINGS = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
IGNORE = 255
NAMES = [c[0] for c in CLASSES]
NC = len(CLASSES)
IS_THING = np.array([n in THINGS for n in NAMES], dtype=bool)
ROOT = C.DATA / "cityscapes"
MAX_INSTANCES = 100      # instance cap for dense street scenes (as in the paper)


def train_lut() -> np.ndarray:
    lut = np.full(256, IGNORE, dtype=np.int16)
    for _, label_id, train_id in CLASSES:
        lut[label_id] = train_id
    return lut


def build_items() -> list:
    img_root = C.require(ROOT / "leftImg8bit" / "val", "see scripts/download_cityscapes.sh")
    items = []
    for img in sorted(img_root.rglob("*_leftImg8bit.png")):
        base = img.name[: -len("_leftImg8bit.png")]
        gt = ROOT / "gtFine" / "val" / img.parent.name / f"{base}_gtFine_labelIds.png"
        if gt.exists():
            items.append({"key": base, "img": str(img), "gt": str(gt)})
    return items


def per_class_iu(pred: np.ndarray, gt: np.ndarray):
    """-> (inter[NC], union[NC]) over valid pixels (gt != 255); -1 predictions only cost recall."""
    valid = gt != IGNORE
    g, p = gt[valid], pred[valid]
    inter = np.array([np.count_nonzero((g == c) & (p == c)) for c in range(NC)], dtype=np.int64)
    union = np.array([np.count_nonzero((g == c) | (p == c)) for c in range(NC)], dtype=np.int64)
    return inter, union


def match_class(label: str) -> int:
    lab = (label or "").strip().lower()
    for i, n in enumerate(NAMES):
        if lab == n:
            return i
    for i, n in enumerate(NAMES):
        if n in lab or lab in n:
            return i
    return -1


def compose_label_map(instances, shape) -> np.ndarray:
    """Labelled instance masks -> HxW trainId map (-1 = nothing predicted).

    Painted back to front so the frontmost surface wins each pixel: all "stuff"
    instances, then all "thing" instances, larger area first within each group
    (a car on the road, a pole in front of a building).
    """
    pred = np.full(shape, -1, dtype=np.int16)
    for ci, m in sorted(instances, key=lambda cm: (bool(IS_THING[cm[0]]), -int(cm[1].sum()))):
        pred[m] = ci
    return pred


def label_maps(instances_json: Path, shape):
    """-> (stage-2 map, final map) from the pipeline's instances.json."""
    empty = np.full(shape, -1, dtype=np.int16)
    if not instances_json.exists():
        return empty, empty
    insts = json.loads(instances_json.read_text()).get("instances", [])
    maps = []
    for stage in ("seg", "rec"):
        labelled = []
        for inst in insts:
            ci, rle = match_class(inst.get("label")), inst.get(f"{stage}_rle")
            if ci >= 0 and rle:
                m = gtmask.ann_to_mask(rle, *rle["size"])
                if m.shape == shape:
                    labelled.append((ci, m))
        maps.append(compose_label_map(labelled, shape))
    return maps[0], maps[1]


def colorize(train_id: np.ndarray) -> np.ndarray:
    out = np.zeros((*train_id.shape, 3), dtype=np.uint8)
    for t, c in enumerate(PALETTE):
        out[train_id == t] = c
    return out


def miou(records, prefix):
    if not records:
        return None, {}
    inter = np.sum([r[f"{prefix}_inter"] for r in records], axis=0)
    union = np.sum([r[f"{prefix}_union"] for r in records], axis=0)
    per_class = {NAMES[c]: (round(float(inter[c] / union[c]), 4) if union[c] else None) for c in range(NC)}
    valid = [inter[c] / union[c] for c in range(NC) if union[c]]
    return (round(float(np.mean(valid)), 4) if valid else None), per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_args(ap)
    args = ap.parse_args()
    C.resolve_args(ap, args)

    from PIL import Image
    lut = train_lut()
    items = C.select(build_items(), args.n, args.shuffle, args.seed)
    out = C.out_dir(args, "cityscapes", "val")
    query = ", ".join(NAMES)

    def process(it, work):
        gt = lut[np.array(Image.open(it["gt"]))]
        if work is None:
            i_, u_ = per_class_iu(gt.astype(np.int16), gt)
            return f"{it['key']}: gt={gt.shape} classes_present={int((u_ > 0).sum())} self-IoU ok={bool((i_ == u_).all())}"
        res = C.run_pipeline(work, Path(it["img"]), query, args,
                             extra=("--max-instances", str(MAX_INSTANCES)))
        seg_map, rec_map = label_maps(res["instances_json"], gt.shape)
        if args.iterations == 0:
            rec_map = seg_map
        rec = {"key": it["key"], "status": res["status"], "t_total": res["t_total"]}
        for prefix, pred in (("seg", seg_map), ("rec", rec_map)):
            i_, u_ = per_class_iu(pred, gt)
            rec[f"{prefix}_inter"], rec[f"{prefix}_union"] = i_.tolist(), u_.tolist()
        if res.get("err"):
            rec["err"] = res["err"]
        Image.fromarray(np.where(rec_map < 0, IGNORE, rec_map).astype(np.uint8)).save(work / "pred_trainid.png")
        Image.fromarray(colorize(rec_map)).save(work / "pred_color.png")
        return rec

    def summarize(records):
        seg_miou, _ = miou(records, "seg")
        rec_miou, per_class = miou(records, "rec")
        return {"n_scored": len(records), "seg_mIoU": seg_miou, "rec_mIoU": rec_miou,
                "rec_per_class_IoU": per_class}

    C.run_benchmark(items, process, out, args, summarize, "cityscapes", "val",
                    header_extra=f"classes={NC}", provenance={"max_instances": MAX_INSTANCES})


if __name__ == "__main__":
    main()
