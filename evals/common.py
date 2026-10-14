"""Shared core of every benchmark harness in ``evals/``.

A harness script is three things:

    1. a LOADER   dataset files -> list of items, each an (image, text query, GT) triple
                  with a stable string ``key``
    2. a SCORER   predicted masks + GT -> one record dict
    3. a SUMMARY  records -> that benchmark's published metric

Everything else (running the pipeline, checkpointing, resuming, parallelism,
config-tagged output directories) is identical across benchmarks and lives here.

The pipeline is run as a subprocess of ``pipeline.py`` (the same command line a
user runs on one image), so every benchmark measures exactly the released code:

    python pipeline.py <image> <query> --model M --provider P --iterations T --out-dir <work>
                       [--topology] [--refine-box] [--no-postprocess]

Each item is scored on BOTH masks the pipeline writes: ``seg_mask.png`` (stage 2,
before refinement; metric prefix ``seg_``) and ``rec_mask.png`` (final, after
refinement; prefix ``rec_``). A missing mask file is scored as an empty prediction,
which is the right answer on items whose ground truth is empty.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIPELINE = REPO / "pipeline.py"
DATA = Path(os.getenv("LLMSEG_DATA", REPO / "data" / "benchmarks")).expanduser()
RESULTS = Path(os.getenv("LLMSEG_RESULTS", REPO / "results")).expanduser()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import llm  # noqa: E402  (provider resolution only; the pipeline itself runs as a subprocess)

DEFAULT_ITERATIONS = 8      # T of the paper's main results
DEFAULT_TIMEOUT = 400       # seconds per vision call


# --------------------------------------------------------------------------- #
# Command line shared by every harness                                        #
# --------------------------------------------------------------------------- #
def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--model", default=None,
                    help="vision model (default: the default model of the first provider with a key set)")
    ap.add_argument("--provider", default="auto", choices=("auto", *llm.PROVIDERS),
                    help="API to call; auto infers it from --model (see llm.py)")
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                    help=f"refinement passes T (default {DEFAULT_ITERATIONS}); 0 scores the stage-2 mask")
    ap.add_argument("--topology", action="store_true",
                    help="pipeline.py --topology (topology conditioning; the ablation reference)")
    ap.add_argument("--refine-box", action="store_true",
                    help="pipeline.py --refine-box (detect-box adjustment pre-pass)")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="pipeline.py --no-postprocess (skip the per-pass morphological cleanup)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per vision call")
    ap.add_argument("--n", type=int, default=0, help="number of items (0 = all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="seeded shuffle before taking --n (use it on large, ordered annotation files)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="items in flight at once (each runs its own pipeline subprocess)")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/<bench>/<model>/<split>[__<config>])")
    ap.add_argument("--force", action="store_true", help="ignore the checkpoint and redo every item")
    ap.add_argument("--dry-run", action="store_true",
                    help="load items and score GT against itself; no model calls, nothing written")
    ap.add_argument("--save", type=int, default=200, metavar="N",
                    help="keep images/overlays for the first N items only, JSON for the rest "
                         "(metrics never need the PNGs); <0 keeps everything")
    return ap


def resolve_args(ap: argparse.ArgumentParser, args) -> None:
    """Validate shared flags and fill in the resolved model / provider."""
    if args.parallel < 1:
        ap.error("--parallel must be >= 1")
    if args.iterations < 0:
        ap.error("--iterations must be >= 0")
    if args.dry_run:
        args.model = args.model or "dry-run"
        return
    try:
        args.model, args.provider = llm.resolve_model(args.model, args.provider)
    except (RuntimeError, ValueError) as e:
        ap.error(str(e))


def config_tag(args, extra_parts=()) -> str:
    """Output-dir suffix for a non-default configuration ("" for the default).

    Results are checkpointed per item key, so two configurations must never share
    a directory or one would resume the other's rows. ``--n`` and ``--parallel``
    are deliberately absent: a bigger --n on the same configuration extends the
    same run instead of starting a new one.
    """
    parts = [p for p in extra_parts if p]
    if args.iterations != DEFAULT_ITERATIONS:
        parts.append(f"it{args.iterations}")
    for flag, part in (("topology", "topo"), ("refine_box", "boxadj"), ("no_postprocess", "nopost")):
        if getattr(args, flag, False):
            parts.append(part)
    if args.shuffle:
        parts.append(f"shuf{args.seed}")
    return "__".join(parts)


def out_dir(args, bench: str, split: str, extra_parts=()) -> Path:
    if args.out:
        return Path(args.out)
    tag = config_tag(args, extra_parts)
    model = args.model.replace("/", "_")
    return RESULTS / bench / model / (split + (f"__{tag}" if tag else ""))


def select(items: list, n: int, shuffle: bool, seed: int) -> list:
    if shuffle:
        items = list(items)
        random.Random(seed).shuffle(items)
    return items[:n] if n else list(items)


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"missing {path}\n  -> {hint}")
    return path


# --------------------------------------------------------------------------- #
# Running the pipeline                                                        #
# --------------------------------------------------------------------------- #
def pipeline_flags(args) -> list:
    """The design-choice switches forwarded to pipeline.py."""
    return [f for f, on in (("--topology", args.topology), ("--refine-box", args.refine_box),
                            ("--no-postprocess", args.no_postprocess)) if on]


def run_pipeline(work: Path, img: Path, query: str, args, extra=()) -> dict:
    """detect -> segment -> refine for one (image, query). Never raises.

    Returns the status plus the two mask paths to score. ``rec_mask`` is the
    final mask; it equals ``seg_mask`` when refinement was skipped (T = 0).
    """
    cmd = [sys.executable, str(PIPELINE), str(img), query, "--model", args.model,
           "--provider", args.provider, "--iterations", str(args.iterations),
           "--timeout", str(args.timeout), "--out-dir", str(work), *pipeline_flags(args), *extra]
    cap = (2 * args.iterations + 2 + int(args.refine_box)) * args.timeout + 120
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=cap)
        (work / "run.log").write_text(r.stderr or "")
        ok, err = r.returncode == 0, ("" if r.returncode == 0 else (r.stderr or r.stdout)[-800:])
    except subprocess.TimeoutExpired:
        ok, err = False, f"pipeline exceeded the {cap}s wall cap"
    seg_mask = work / "seg_mask.png"
    rec_mask = seg_mask if args.iterations == 0 else work / "rec_mask.png"
    if not rec_mask.exists():
        rec_mask = seg_mask
    return {"status": "ok" if ok else "failed", "err": err, "t_total": round(time.time() - t0, 1),
            "seg_mask": seg_mask, "rec_mask": rec_mask, "instances_json": work / "instances.json"}


# --------------------------------------------------------------------------- #
# Scoring primitives                                                          #
# --------------------------------------------------------------------------- #
def load_pred(path: Path, shape) -> np.ndarray:
    from PIL import Image
    if not Path(path).exists():
        return np.zeros(shape, dtype=bool)
    m = np.array(Image.open(path).convert("L")) > 0
    if m.shape != tuple(shape):
        m = np.array(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return m


def iou(pred: np.ndarray, gt: np.ndarray):
    """-> (intersection, union, IoU). Empty prediction on empty GT scores 1.0."""
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return inter, union, (1.0 if union == 0 else inter / union)


def gmean(records, field):
    vals = [r[field] for r in records if field in r]
    return round(float(np.mean(vals)), 4) if vals else None


def cumulative_iou(records, inter_f, union_f):
    inter = sum(r.get(inter_f, 0) for r in records)
    union = sum(r.get(union_f, 0) for r in records)
    return round(inter / union, 4) if union else None


def precision_at(records, field, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    vals = [r[field] for r in records if field in r]
    if not vals:
        return {}
    return {f"Pr@{t}": round(float(np.mean([v >= t for v in vals])), 4) for t in thresholds}


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation of present / absent decisions (the IL_MCC of cgF1)."""
    num = tp * tn - fp * fn
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return round(num / den, 4) if den else 0.0


def match_f1(pred_masks, gt_masks, thr: float = 0.5):
    """Greedy one-to-one IoU matching at ``thr`` -> (tp, fp, fn)."""
    if not gt_masks and not pred_masks:
        return 0, 0, 0
    pairs = []
    for pi, p in enumerate(pred_masks):
        for gi, g in enumerate(gt_masks):
            v = iou(p, g)[2]
            if v >= thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for _, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    tp = len(used_p)
    return tp, len(pred_masks) - tp, len(gt_masks) - tp


MIN_INSTANCE_AREA_FRAC = 1e-4   # smaller connected components are speckle, not instances


def pred_instances(mask: np.ndarray) -> list:
    """Instance proxies of a union mask: its connected components.

    Exact when instances are separate, merges touching ones (a lower bound on
    pmF1 for crowded images). Used identically for every method in the paper's
    cgF1 / pmF1 columns; ``score_ap.py`` reports the per-instance variant too.
    """
    import gtmask
    return gtmask.connected_components(mask, min_px=max(16, int(MIN_INSTANCE_AREA_FRAC * mask.size)))


def presence_f1(records, prefix: str, group_key: str = "query") -> dict:
    """IL_MCC, pmF1 and cgF1 = 100 * IL_MCC * pmF1 for one stage ('seg' or 'rec').

    IL_MCC  present/absent correlation over all items (positives and negatives).
    pmF1    instance TP/FP/FN pooled per prompt over positive items, F1 per
            prompt, macro-averaged over prompts.
    """
    tp = tn = fp = fn = 0
    per_prompt = defaultdict(lambda: [0, 0, 0])
    for r in records:
        gt_present = r.get("gt_px", 0) > 0
        pred_present = r.get(f"{prefix}_px", 0) > 0
        if gt_present and pred_present:
            tp += 1
        elif gt_present:
            fn += 1
        elif pred_present:
            fp += 1
        else:
            tn += 1
        if gt_present:
            acc = per_prompt[r.get(group_key, "")]
            for i, k in enumerate(("tp", "fp", "fn")):
                acc[i] += int(r.get(f"{prefix}_{k}", 0))
    f1s = []
    for t, f_p, f_n in per_prompt.values():
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    il_mcc = mcc(tp, tn, fp, fn)
    pmf1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    return {f"{prefix}_IL_MCC": il_mcc, f"{prefix}_pmF1": pmf1,
            f"{prefix}_cgF1": round(100.0 * il_mcc * pmf1, 2),
            f"{prefix}_presence": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def score_both_stages(rec: dict, res: dict, gt: np.ndarray, gt_masks=None) -> dict:
    """IoU (and, given ``gt_masks``, instance TP/FP/FN) of the stage-2 and final masks."""
    for prefix in ("seg", "rec"):
        pred = load_pred(res[f"{prefix}_mask"], gt.shape)
        i_, u_, v_ = iou(pred, gt)
        rec.update({f"{prefix}_iou": round(v_, 4), f"{prefix}_inter": i_,
                    f"{prefix}_union": u_, f"{prefix}_px": int(pred.sum())})
        if gt_masks is not None:
            t, f_p, f_n = match_f1(pred_instances(pred), gt_masks) if gt_masks else (0, 0, 0)
            rec.update({f"{prefix}_tp": t, f"{prefix}_fp": f_p, f"{prefix}_fn": f_n})
    rec.update(status=res["status"], t_total=res["t_total"])
    if res.get("err"):
        rec["err"] = res["err"]
    return rec


def iou_summary(records, positives_only: bool = False) -> dict:
    """gIoU / cIoU (+ Pr@k) of both stages."""
    sfx = "_pos" if positives_only else ""
    recs = [r for r in records if r.get("gt_px", 0) > 0] if positives_only else records
    out = {}
    for p in ("seg", "rec"):
        out[f"{p}_gIoU{sfx}"] = gmean(recs, f"{p}_iou")
        out[f"{p}_cIoU{sfx}"] = cumulative_iou(recs, f"{p}_inter", f"{p}_union")
    out.update({f"rec_{k}{sfx}": v for k, v in precision_at(recs, "rec_iou").items()})
    return out


def rejection_summary(records) -> dict:
    neg = [r for r in records if r.get("gt_px", 0) == 0]
    if not neg:
        return {}
    return {f"{p}_reject_rate": round(float(np.mean([r[f"{p}_px"] == 0 for r in neg])), 4)
            for p in ("seg", "rec")}


# --------------------------------------------------------------------------- #
# The run loop: checkpoint, resume, parallelism, summary                      #
# --------------------------------------------------------------------------- #
# A record whose error carries one of these is a billing / auth / daily-quota
# cut-off, not a measurement of the item: it is not cached, and the run stops so
# a later resume retries every remaining item.
_CUTOFF_SIGNS = ("PerDay", "monthly spending cap", "PERMISSION_DENIED", "insufficient_quota",
                 "credit balance is too low", "Insufficient credits", "API_KEY_INVALID",
                 "call failed (401)", "call failed (403)")


class Cutoff(RuntimeError):
    pass


def run_benchmark(items, process, out: Path, args, summarize, bench: str, split: str,
                  header_extra: str = "", provenance=None) -> None:
    """Drive ``process(item, work_dir) -> record`` over ``items``, checkpointed.

    ``results.jsonl`` gets one line per finished item and doubles as the
    checkpoint: a re-run skips every cached key. ``summary.json`` is rewritten
    after every item. An exception inside ``process`` is not a measurement, so
    the item is not recorded and a re-run retries it. A pipeline failure IS a
    measurement (scored on whatever masks it left) and is recorded.
    """
    if args.dry_run:
        print(f"[{bench}/{split}] DRY RUN items={len(items)} {header_extra}", flush=True)
        for i, it in enumerate(items, 1):
            try:
                print(f"  [{i}/{len(items)}] " + process(it, None), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(items)}] {it['key']}: ERROR {exc!r}", flush=True)
        return

    work_root = out / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = out / "results.jsonl", out / "summary.json"
    done = {}
    if results_path.exists() and not args.force:
        for line in results_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except (ValueError, KeyError):
                pass
    if args.force and results_path.exists():
        results_path.unlink()
    records = [done[it["key"]] for it in items if it["key"] in done]
    pending = [it for it in items if it["key"] not in done]
    save_keys = None if args.save < 0 else {it["key"] for it in items[:args.save]}
    print(f"[{bench}/{split}] provider={args.provider} model={args.model} "
          f"iterations={args.iterations} items={len(items)} cached={len(records)} "
          f"{header_extra} out={out}", flush=True)

    def write(final=False):
        summ = {"benchmark": f"{bench}/{split}", "model": args.model, "provider": args.provider,
                "iterations": args.iterations, "timeout": args.timeout,
                "topology": args.topology, "refine_box": args.refine_box,
                "postprocess": not args.no_postprocess,
                "n_items_selected": len(items), "n_records": len(records),
                "n_failed": sum(1 for r in records if r.get("status") == "failed")}
        summ.update(provenance or {})
        try:
            summ.update(summarize(records) or {})
        except Exception as exc:  # noqa: BLE001 (a metric bug must not lose the data)
            summ["summarize_error"] = repr(exc)
        summ["complete"] = final and len(records) == len(items)
        summary_path.write_text(json.dumps(summ, indent=2))
        if final:
            print(json.dumps(summ, indent=2), flush=True)

    def prune(key):
        w = work_root / key
        if w.is_dir():
            for p in w.iterdir():
                if p.is_file() and p.suffix.lower() != ".json":
                    p.unlink(missing_ok=True)

    n_done = [0]

    def emit(rec):
        if any(s in str(rec.get("err") or "") for s in _CUTOFF_SIGNS):
            raise Cutoff(f"{rec['key']}: {str(rec['err'])[-300:]}")
        with results_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        records.append(rec)
        n_done[0] += 1
        write()
        if save_keys is not None and rec["key"] not in save_keys:
            prune(rec["key"])
        print(f"  [{n_done[0]}/{len(pending)}] {rec['key']}: {rec.get('status')} "
              f"seg={rec.get('seg_iou', '-')} rec={rec.get('rec_iou', '-')} ({rec.get('t_total')}s) "
              f"q={str(rec.get('query', ''))[:45]!r}", flush=True)

    def work_for(it):
        w = work_root / it["key"]
        w.mkdir(parents=True, exist_ok=True)
        return w

    def handle(key, exc) -> bool:
        """-> True when the whole run must stop."""
        if isinstance(exc, Cutoff):
            print(f"  STOP: provider cut-off (billing / auth / daily quota), item not cached: {exc}\n"
                  f"  Fix the account, then re-run the same command to resume.", flush=True)
            return True
        print(f"  {key}: harness error {exc!r} (not recorded; re-run to retry)", flush=True)
        return False

    try:
        if args.parallel > 1 and pending:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = {ex.submit(process, it, work_for(it)): it["key"] for it in pending}
                for fut in as_completed(futs):
                    try:
                        emit(fut.result())
                    except Exception as exc:  # noqa: BLE001 (per-item isolation)
                        if handle(futs[fut], exc):
                            for f in futs:
                                f.cancel()
                            break
        else:
            for it in pending:
                try:
                    emit(process(it, work_for(it)))
                except Exception as exc:  # noqa: BLE001
                    if handle(it["key"], exc):
                        break
    finally:
        write(final=True)
