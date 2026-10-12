#!/usr/bin/env python3
"""Training-free text-prompted segmentation with a multimodal LLM as the only vision model.

    stage 1  detect   image + query          -> N instance boxes (+ morphology)
    stage 2  segment  image + boxes          -> per-instance closed B-splines -> mask
    stage 3  refine   overlay of the mask    -> focus boxes -> +/- B-splines -> mask
                      (up to T passes, early stop on a small mask residual)

Usage:
    python pipeline.py data/toy/cat.jpg "the cat" --out-dir output/cat
    python pipeline.py data/toy/taxis.jpeg "the taxi" --model gemini-3.7-flash --iterations 8
    python pipeline.py data/toy/dog.jpeg "the dog" --model claude-sonnet-5-5
    python pipeline.py data/toy/car.jpg "the car" --model gpt-5.6-terra
    python pipeline.py data/toy/clock.jpeg "the clock" --model google/gemini-3.7-flash   # OpenRouter

Every stage writes a JSON record plus overlay / mask PNGs into ``--out-dir``.
The final mask is ``rec_mask.png`` (``seg_mask.png`` is the stage-2 mask);
``instances.json`` holds every instance's label, confidence, box and stage-2 /
final mask (COCO RLE), which is what instance-level metrics (AP) read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import geometry as G
import llm
import prompts as P
from llm import log

MAX_INSTANCES = 200          # N_max, instances kept per image (highest confidence)
DEDUPE_IOU = 0.9             # near-duplicate detections are merged
CONVERGE_EPS = 0.005         # residual threshold, fraction of the instance-box union
CONVERGE_PATIENCE = 3        # consecutive quiet passes before stopping
FALLBACK_MASK_COLOR = "green"


# --------------------------------------------------------------------------- #
# Reply parsing shared by the batched stages                                  #
# --------------------------------------------------------------------------- #
def _valid_splines(splines) -> list:
    if not isinstance(splines, list):
        return []
    return [s for s in splines if isinstance(s, dict) and len(s.get("control_points", [])) >= 3]


def parse_id_keyed_splines(payload: dict, expected_ids: list, keys) -> dict:
    """id -> splines from an id-keyed reply (positional match if ids were dropped)."""
    entries = next((payload[k] for k in keys
                    if isinstance(payload, dict) and isinstance(payload.get(k), list)), [])
    by_id = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        try:
            sid = int(entry.get("id", i))
        except (TypeError, ValueError):
            sid = i
        by_id[sid] = _valid_splines(entry.get("splines"))
    if expected_ids and not any(e in by_id for e in expected_ids) and len(entries) == len(expected_ids):
        by_id = {e: _valid_splines(entries[i].get("splines") if isinstance(entries[i], dict) else None)
                 for i, e in enumerate(expected_ids)}
    return {e: by_id.get(e, []) for e in expected_ids}


def _salvage_id_keyed(text: str) -> dict:
    """Recover the complete {id, splines} entries (and shared scalars) of a cut-off reply."""
    payload: dict = {}
    for key, pat, cast in (("mask_color", r'"mask_color"\s*:\s*"([^"]+)"', str),
                           ("degree", r'"degree"\s*:\s*(\d+)', int)):
        m = re.search(pat, text)
        if m:
            payload[key] = cast(m.group(1))
    entries: dict = {}
    for obj in llm.extract_json_objects(text):
        if isinstance(obj, dict) and "id" in obj and "splines" in obj:
            try:
                entries.setdefault(int(obj["id"]), obj)
            except (TypeError, ValueError):
                continue
    items = [entries[k] for k in sorted(entries)]
    payload["instances"] = payload["refinements"] = items
    return payload


def parse_batched_reply(text: str, what: str) -> dict:
    """Strict parse, else structure repair (if complete), else truncation salvage."""
    try:
        return llm.extract_json(text)
    except ValueError:
        payload = _salvage_id_keyed(text)
        n = len(payload["instances"])
        repaired = llm.parse_repaired_json(text, only_if_complete=True)
        if isinstance(repaired, dict) and len(repaired.get("instances", [])) > n:
            log(f"[{what}] malformed reply; repaired JSON structure")
            return repaired
        if not n:
            raise
        log(f"[{what}] truncated reply; salvaged {n} complete entries")
        return payload


def format_instances(instances: list, model: str, size, include_morphology: bool) -> str:
    """Instance lines for a prompt, each with its box restated as explicit bounds."""
    W, H = size
    lines = []
    for inst in instances:
        mx1, my1, mx2, my2 = (round(v) for v in llm.to_model_box(inst["box"], model, W, H))
        lines.append(f'- id={inst["id"]} label="{inst.get("label", "object")}" '
                     f'{llm.box_field_name(model)}={llm.format_model_box(inst["box"], model, W, H)}')
        lines.append(f'    every x,y you return for id={inst["id"]} must satisfy: '
                     f'{mx1} <= x <= {mx2}  and  {my1} <= y <= {my2}')
        if include_morphology:
            lines.append(f'    morphology: {json.dumps(inst.get("morphology", "n/a"))}')
    return "\n".join(lines)


def merged_label(instances: list) -> str:
    labels: list = []
    for inst in instances:
        lb = str(inst.get("label", "") or "").strip()
        if lb and lb not in labels:
            labels.append(lb)
    if not labels:
        return "object"
    head = " / ".join(labels[:3])
    return head + " / ..." if len(labels) > 3 else head


def resolve_mask_color(name):
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in P.MASK_COLORS:
        return key, P.MASK_COLORS[key]
    for k in P.MASK_COLORS:
        if key and (k in key or key in k):
            return k, P.MASK_COLORS[k]
    return FALLBACK_MASK_COLOR, P.MASK_COLORS[FALLBACK_MASK_COLOR]


# --------------------------------------------------------------------------- #
# Stage 1: detect                                                             #
# --------------------------------------------------------------------------- #
def parse_detections(text: str) -> list:
    """Detections from a reply; a truncated/malformed reply keeps its complete entries."""
    def keep(objs):
        return [o for o in objs if isinstance(o, dict) and llm.normalize_box_key(o)
                and ("label" in o or "confidence" in o)]
    try:
        data = llm.extract_json(text)
        return [d for d in data.get("detections", []) if isinstance(d, dict)]
    except ValueError:
        dets = keep(llm.extract_json_objects(text))
        if not dets:
            repaired = llm.parse_repaired_json(text)
            if isinstance(repaired, dict) and isinstance(repaired.get("detections"), list):
                dets = keep(repaired["detections"])
        if not dets:
            raise
        log(f"[detect] malformed reply; salvaged {len(dets)} detection(s)")
        return dets


def _confidence(d) -> float:
    try:
        return float(d.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _reading_order(d):
    try:
        return (float(d["box"][1]), float(d["box"][0]))
    except (TypeError, ValueError, IndexError, KeyError):
        return (0.0, 0.0)


def stage_detect(image: Path, query: str, model: str, size) -> list:
    W, H = size
    prompt = P.PROMPT_DETECT.format(
        image_path=str(image.resolve()),
        coord_space=llm.coord_space_instructions(model, W, H),
        box_field=llm.box_field_name(model),
        box_order=llm.box_order_literal(model),
        box_format=llm.box_format_instructions(model),
        query=query,
    )
    dets = parse_detections(llm.query(prompt, image, model, "detections"))
    llm.normalize_boxes(dets, model, W, H)

    kept: list = []
    for det in dets:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        dup = next((i for i, prev in enumerate(kept) if G.box_iou(box, prev["box"]) >= DEDUPE_IOU), None)
        if dup is None:
            kept.append(det)
        elif _confidence(det) > _confidence(kept[dup]):
            kept[dup] = det
    kept = sorted(kept, key=_reading_order)
    for i, det in enumerate(kept):
        det["id"] = i
    log(f"[detect] {len(kept)} instance(s)")
    return kept


# --------------------------------------------------------------------------- #
# Stage 2: segment                                                            #
# --------------------------------------------------------------------------- #
def select_instances(detections: list, max_instances: int = MAX_INSTANCES) -> list:
    dets = [d for d in detections if isinstance(d, dict) and d.get("box")]
    dets = sorted(dets, key=_confidence, reverse=True)[:max_instances]
    dets = sorted(dets, key=_reading_order)
    return [{"id": i, "label": d.get("label", "object"), "box": [float(v) for v in d["box"]],
             "morphology": d.get("morphology"), "confidence": d.get("confidence")}
            for i, d in enumerate(dets)]


def stage_segment(image: Path, instances: list, model: str, size):
    """One call traces every instance. Returns (instances with splines, degree, color name)."""
    W, H = size
    label = merged_label(instances)
    prompt = P.PROMPT_SEGMENT.format(
        image_path=str(image.resolve()),
        coord_space=llm.coord_space_instructions(model, W, H),
        label=label,
        n_instances=len(instances),
        instances_block=format_instances(instances, model, size, include_morphology=True),
        box_field=llm.box_field_name(model),
        box_order=llm.box_order_literal(model),
        point_order_note=llm.point_order_note(model),
        color_scope_note=P.color_scope_note(len(instances)),
        output_budget_note=P.output_budget_note(len(instances), "instances"),
        degree_guidance=P.segment_degree_guidance(),
        mask_color_options=P.mask_color_options(),
    )
    parsed = parse_batched_reply(llm.query(prompt, image, model, "B-spline fit"), "segment")

    ids = [inst["id"] for inst in instances]
    by_id = parse_id_keyed_splines(parsed, ids, keys=("instances", "objects", "refinements"))
    if not any(by_id.values()) and parsed.get("splines"):   # flat single-object reply
        by_id = {e: [] for e in ids}
        by_id[ids[0]] = _valid_splines(parsed.get("splines"))
    llm.rescale_spline_groups([by_id[e] for e in ids], model, W, H)
    for inst in instances:
        if llm.fix_transposed_points(by_id[inst["id"]], inst["box"], model):
            log(f"[segment] instance {inst['id']}: control points were [y, x]; transposed")

    degree = parsed.get("degree")
    if degree is None:
        degree = next((s.get("degree") for e in ids for s in by_id[e]
                       if s.get("degree") is not None), None)
    try:
        degree = int(degree)
    except (TypeError, ValueError):
        degree = 3
    degree = degree if degree in (1, 3) else 3

    for inst in instances:
        inst["splines"] = by_id[inst["id"]]
        for s in inst["splines"]:
            s["degree"] = degree
    color, _ = resolve_mask_color(parsed.get("mask_color"))
    return instances, degree, color


# --------------------------------------------------------------------------- #
# Stage 3: refine                                                             #
# --------------------------------------------------------------------------- #
def focus_boxes(overlay: Path, label: str, instances: list, model: str, size, color: str) -> list:
    """3a: typed (positive / negative) focus boxes on the current overlay."""
    W, H = size
    prompt = P.PROMPT_FOCUS.format(
        overlay_path=str(overlay.resolve()),
        label=label,
        coord_space=llm.coord_space_instructions(model, W, H),
        instances_block=format_instances(instances, model, size, include_morphology=False),
        box_field=llm.box_field_name(model),
        box_order=llm.box_order_literal(model),
        focus_boxes_per_instance=P.FOCUS_BOXES_PER_INSTANCE,
        max_focus_boxes=P.FOCUS_BOXES_PER_INSTANCE * max(1, len(instances)),
        focus_spread_note=P.focus_spread_note(len(instances)),
        mask_color=color,
    )
    reply = llm.extract_json(llm.query(prompt, overlay, model, "focus boxes"))
    boxes = reply.get("focus_boxes", []) if isinstance(reply, dict) else []
    boxes = [b for b in boxes if isinstance(b, dict)]
    llm.normalize_boxes(boxes, model, W, H)
    return boxes


def focus_splines(overlay: Path, label: str, pending: list, model: str, size, color: str) -> dict:
    """3b: one call returns cubic loops for every surviving focus box -> {box id: splines}."""
    W, H = size
    lines = [f'- id={b["id"]} instance_id={b["instance_id"]} type="{b["type"]}" '
             f'{llm.box_field_name(model)}={llm.format_model_box(b["box"], model, W, H)}'
             f'  shape: {b.get("shape", "") or "(shape not given)"}' for b in pending]
    prompt = P.PROMPT_FOCUS_SPLINE.format(
        overlay_path=str(overlay.resolve()),
        coord_space=llm.coord_space_instructions(model, W, H),
        label=label,
        n_boxes=len(pending),
        focus_boxes_block="\n".join(lines),
        box_order=llm.box_order_literal(model),
        point_order_note=llm.point_order_note(model),
        mask_color=color,
        degree=3,
        output_budget_note=P.output_budget_note(len(pending), "focus boxes"),
        degree_guidance=P.focus_degree_guidance(),
    )
    reply = llm.query(prompt, overlay, model, f"focus splines ({len(pending)} box(es))")
    payload = parse_batched_reply(reply, "refine")
    by_id = parse_id_keyed_splines(payload, [b["id"] for b in pending],
                                   keys=("refinements", "focus_boxes", "boxes"))
    for b in pending:
        splines = by_id[b["id"]]
        llm.rescale_spline_groups([splines], model, W, H)
        llm.fix_transposed_points(splines, b["box"], model)
        for s in splines:
            s["degree"] = 3
    return by_id


def stage_refine(image: Path, instances: list, model: str, size, color: str,
                 seg_overlay: Path, out: Path, iterations: int, record: dict) -> dict:
    """Up to ``iterations`` passes of 3a -> effect gate -> 3b -> compose -> postprocess."""
    W, H = size
    rgb = P.MASK_COLORS[color]
    label = merged_label(instances)
    boxes = [inst["box"] for inst in instances]
    ids = [inst["id"] for inst in instances]
    for inst in instances:
        inst["mask"] = G.mask_from_splines(inst["splines"], inst["box"], size)
        inst["lock"] = np.zeros((H, W), dtype=np.float32)
    by_inst = {inst["id"]: inst for inst in instances}
    union = lambda: G.union_masks([inst["mask"] for inst in instances], size)  # noqa: E731
    denom = G.box_union_area_px(boxes, size) or (W * H)

    overlay, quiet_streak = seg_overlay, 0
    record.update(passes=[], iterations=0, converged=False, complete=False)
    for it in range(1, iterations + 1):
        before = union()
        try:
            proposed = focus_boxes(overlay, label, instances, model, size, color)
        except ValueError as e:
            log(f"[refine] pass {it}: unparseable focus-box reply ({e}); quiet pass")
            proposed = []

        pending = []
        for fb in proposed:
            if len(fb.get("box") or []) != 4:
                continue
            idx = G.assign_instance(fb["box"], instances, claimed_id=fb.get("instance_id"))
            fbox = G.clamp_box(fb["box"], instances[idx]["box"], size) if idx is not None else None
            if fbox is None:
                continue
            pending.append({"type": "negative" if fb.get("type") == "negative" else "positive",
                            "box": fbox, "shape": fb.get("shape", ""),
                            "instance_id": int(instances[idx]["id"])})

        kept = []   # effect gate against the owning instance's mask and lock
        for item in pending:
            inst = by_inst[item["instance_id"]]
            n, area = G.achievable_px(item["type"], item["box"], inst["mask"], inst["lock"], size)
            if n >= max(G.REFINE_EFFECT_MIN_PX, int(np.ceil(G.REFINE_EFFECT_MIN_FRAC * max(area, 1)))):
                item["id"] = len(kept)
                kept.append(item)
        log(f"[refine] pass {it}: {len(proposed)} focus box(es) proposed, {len(kept)} kept")

        applied = []
        if kept:
            try:
                splines = focus_splines(overlay, label, kept, model, size, color)
            except ValueError as e:
                log(f"[refine] pass {it}: unparseable spline reply ({e}); nothing applied")
                splines = {item["id"]: [] for item in kept}
            applied = [{k: item[k] for k in ("type", "box", "shape", "instance_id")}
                       | {"splines": splines[item["id"]]} for item in kept if splines[item["id"]]]
            grouped = defaultdict(list)
            for rec in applied:
                grouped[rec["instance_id"]].append(rec)
            for iid, refs in grouped.items():
                inst = by_inst[iid]
                inst["mask"], inst["lock"], pos = G.apply_locked_refinements(
                    inst["mask"], inst["lock"], refs, size)
                inst["mask"] = G.bridge_positive_islands(inst["mask"], inst["box"], pos)

        overlay = out / f"recseg_iter{it}.png"
        G.render_overlay(image, union(), overlay, rgb, boxes=boxes, ids=ids)
        for inst in instances:
            inst["mask"] = G.postprocess_mask(inst["mask"], inst["box"])
        overlay = out / f"recseg_iter{it}_post.png"
        G.render_overlay(image, union(), overlay, rgb, boxes=boxes, ids=ids)

        residual = float(np.logical_xor(before, union()).sum()) / float(denom)
        quiet = (not kept) or residual < CONVERGE_EPS
        quiet_streak = quiet_streak + 1 if quiet else 0
        log(f"[refine] pass {it}: residual={residual:.4f} quiet_streak={quiet_streak}")
        record["passes"].append({"index": it, "refinements": applied,
                                 "mask_symdiff_frac": round(residual, 6)})
        record["iterations"] = it
        if quiet_streak >= CONVERGE_PATIENCE:
            record["converged"] = True
            log(f"[refine] converged after {it} pass(es)")
            break
    record["complete"] = True
    return record


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_instances(path: Path, instances: list, seg_masks: list, rec_masks: list, size) -> None:
    """Per-instance record: label, detect confidence, box, stage-2 and final mask (RLE)."""
    W, H = size
    _write(path, {"width": W, "height": H, "instances": [
        {"id": inst["id"], "label": inst.get("label"), "confidence": inst.get("confidence"),
         "box": inst["box"], "seg_rle": G.mask_to_rle(sm), "rec_rle": G.mask_to_rle(rm)}
        for inst, sm, rm in zip(instances, seg_masks, rec_masks)]})


def run(image: Path, query: str, model: str, out: Path, iterations: int,
        max_instances: int = MAX_INSTANCES) -> int:
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as im:
        size = im.size
    W, H = size
    empty = np.zeros((H, W), dtype=bool)
    try:
        detections = stage_detect(image, query, model, size)
    except (TimeoutError, RuntimeError, ValueError) as e:
        log(f"[detect] error: {e}")
        return 1
    _write(out / "detect.json", {"query": query, "model": model, "width": W, "height": H,
                                 "detections": detections})
    if not detections:
        for name in ("seg_mask.png", "rec_mask.png"):
            Image.fromarray(empty.astype(np.uint8)).save(out / name)
        _write_instances(out / "instances.json", [], [], [], size)
        log("[detect] nothing matched the query; empty mask written")
        return 0

    instances = select_instances(detections, max_instances)
    try:
        instances, degree, color = stage_segment(image, instances, model, size)
    except (TimeoutError, RuntimeError, ValueError) as e:
        log(f"[segment] error: {e}")
        return 1
    if not any(inst["splines"] for inst in instances):
        log("[segment] error: no instance came back with a usable outline")
        return 1
    rgb = P.MASK_COLORS[color]
    per_masks = [G.mask_from_splines(inst["splines"], inst["box"], size) for inst in instances]
    seg_mask = G.union_masks(per_masks, size)
    seg_overlay = out / "seg.png"
    G.render_overlay(image, seg_mask, seg_overlay, rgb,
                     boxes=[inst["box"] for inst in instances], ids=[inst["id"] for inst in instances],
                     instance_masks=per_masks if len(per_masks) > 1 else None,
                     palette_colors=list(P.MASK_COLORS.values()), mask_out=out / "seg_mask.png")
    record = {"query": query, "model": model, "width": W, "height": H, "degree": degree,
              "mask_color": color, "coverage_pct": round(100.0 * float(seg_mask.mean()), 2),
              "instances": [{k: inst[k] for k in ("id", "label", "box", "morphology", "splines")}
                            for inst in instances]}
    _write(out / "seg.json", record)
    if iterations < 1:
        _write_instances(out / "instances.json", instances, per_masks, per_masks, size)
        return 0

    try:
        record = stage_refine(image, instances, model, size, color, seg_overlay, out,
                              iterations, record)
    except (TimeoutError, RuntimeError) as e:
        log(f"[refine] aborted, keeping the passes completed so far: {e}")
    final = G.union_masks([inst["mask"] for inst in instances], size)
    G.render_overlay(image, final, out / "recseg.png", rgb,
                     instance_masks=[inst["mask"] for inst in instances] if len(instances) > 1 else None,
                     palette_colors=list(P.MASK_COLORS.values()), mask_out=out / "rec_mask.png")
    record["coverage_pct"] = round(100.0 * float(final.mean()), 2)
    _write(out / "recseg.json", record)
    _write_instances(out / "instances.json", instances, per_masks,
                     [inst["mask"] for inst in instances], size)
    log(f"[done] final mask -> {out / 'rec_mask.png'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image", help="input image")
    ap.add_argument("query", help="text query, e.g. 'the cat'")
    ap.add_argument("--model", default=None,
                    help="vision model; the provider is inferred from the name (see --provider). "
                         "Default: the first provider (in this order) with a key set, with its default model "
                         f"({', '.join(f'{p}={m}' for p, m in llm.DEFAULT_MODELS.items())})")
    ap.add_argument("--provider", default="auto", choices=("auto", *llm.PROVIDERS),
                    help="API to call (default auto: 'a/b' slug -> openrouter, gemini-*/gemma-* -> "
                         "gemini, claude-* -> anthropic, gpt-*/o* -> openai)")
    ap.add_argument("--iterations", type=int, default=8,
                    help="max refinement passes T (default: 8; 0 = stop after stage 2)")
    ap.add_argument("--timeout", type=float, default=llm.DEFAULT_TIMEOUT,
                    help=f"seconds per vision call (default: {llm.DEFAULT_TIMEOUT})")
    ap.add_argument("--max-instances", type=int, default=MAX_INSTANCES,
                    help=f"instances kept per image, highest confidence first (default: {MAX_INSTANCES})")
    ap.add_argument("--out-dir", help="output directory (default: output/<image stem>)")
    args = ap.parse_args()
    image = Path(args.image)
    if not image.exists():
        log(f"error: image not found: {image}")
        return 1
    try:
        model, provider = llm.resolve_model(args.model, args.provider)
    except (RuntimeError, ValueError) as e:
        log(f"error: {e}")
        return 2
    llm.configure(provider=provider, timeout=args.timeout)
    out = Path(args.out_dir) if args.out_dir else Path("output") / image.stem
    return run(image, args.query, model, out, args.iterations, args.max_instances)


if __name__ == "__main__":
    sys.exit(main())
