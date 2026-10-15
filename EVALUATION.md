# Evaluation

The harness in `evals/` runs the released `pipeline.py` on every item of a benchmark, scores the
stage-2 mask (`seg_*` metrics, before refinement) and the final mask (`rec_*`, after refinement), and
reports each benchmark's standard metric. It works with every provider (`--model` / `--provider` as for
the pipeline).

## 1. Download the benchmarks

Everything goes to `data/benchmarks/` (override with `LLMSEG_DATA=/path`). Only evaluation splits
are downloaded. All scripts are resumable.

| benchmark | what to run | needs | size |
|---|---|---|---|
| ReasonSeg val / test | `bash scripts/download_benchmarks.sh reasonseg` | | 1.5 GB |
| RefCOCO / RefCOCO+ / RefCOCOg | `bash scripts/download_benchmarks.sh refcoco`<br>`bash scripts/download_coco_images.sh train2014` | | 13 GB |
| SA-Co/Gold | `bash scripts/download_benchmarks.sh saco`<br>`python scripts/download_saco_images.py gold` | `HF_TOKEN` (accept the [facebook/SACo-Gold](https://huggingface.co/datasets/facebook/SACo-Gold) terms), `ROBOFLOW_API_KEY`, `pip install roboflow` | 3 GB |
| LVIS v1 val | `bash scripts/download_benchmarks.sh lvis`<br>`bash scripts/download_coco_images.sh lvis` | | 3.5 GB |
| COCO val2017 | `bash scripts/download_benchmarks.sh coco`<br>`bash scripts/download_coco_images.sh val2017` | | 0.8 GB |
| ADE20K val | `bash scripts/download_benchmarks.sh ade20k` | | 0.9 GB |
| Cityscapes val | `bash scripts/download_cityscapes.sh` | free [Cityscapes](https://www.cityscapes-dataset.com) account | 12 GB zips |
| ODinW13 | `python scripts/download_odinw.py` | | 0.5 GB |
| RF-100VL | `python scripts/download_rf100vl.py` | `ROBOFLOW_API_KEY`, `pip install rf100vl` | 41 GB |

Resulting layout:

```
data/benchmarks/
├── reasonseg/{val,test}/data/*.parquet           # images + masks inside the parquet
├── refcoco/{refcoco,refcocoplus,refcocog}/data/*.parquet
├── coco/annotations/instances_val2017.json
├── coco/images/{train2014,val2017,train2017}/    # train2017: only the LVIS-referenced subset
├── lvis/lvis_v1_val.json
├── saco/gold/*.json + saco/gold/images/
├── ade20k/ADEChallengeData2016/{images,annotations}/validation/
├── cityscapes/{leftImg8bit,gtFine}/val/
├── odinw/<subset>/{annotations.coco.json,images/}
└── rf100vl/<subset>/test/{_annotations.coco.json,*.jpg}
```

Respect each dataset's license; none of them is redistributed here.

## 2. Run the harnesses

Every harness first accepts `--dry-run`, which loads the items and scores the ground truth against
itself without any model call. Use it to check a download:

```bash
python evals/run_reasonseg.py --dry-run --n 5
```

| benchmark | command (one paper configuration each) |
|---|---|
| ReasonSeg | `python evals/run_reasonseg.py --split val` (and `--split test`) |
| RefCOCO/+/g | `python evals/run_refcoco.py --dataset refcoco --split testA --n 500 --shuffle` |
| SA-Co/Gold | `python evals/run_saco.py --split gold --domain all --n 2500 --shuffle` |
| LVIS / COCO | `python evals/run_instseg.py --dataset lvis --negatives 3 --n 1500 --shuffle`<br>`python evals/score_ap.py --dataset lvis --results <run dir>` |
| ADE20K | `python evals/run_ade20k.py --n 1000 --shuffle` |
| Cityscapes | `python evals/run_semseg.py` |
| ODinW13 / RF-100VL | `python evals/run_apzero.py --collection odinw`<br>`python evals/score_apzero.py --collection odinw --model <model>` |
| any COCO json | `python evals/run_cocodet.py --ann <json> --images <dir> --name <name>` |

Options shared by every harness:

| option | default | meaning |
|---|---|---|
| `--model`, `--provider` | as for `pipeline.py` | vision backend |
| `--iterations` | `8` | refinement budget T (`0` = stage 2 only) |
| `--topology`, `--refine-box`, `--no-postprocess` | off | forwarded to `pipeline.py` (`python pipeline.py --help`) |
| `--n` | `0` (all) | number of items |
| `--shuffle`, `--seed` | off, `0` | seeded shuffle before taking `--n` (the paper's sampled subsets) |
| `--parallel` | `1` | items in flight (mind your provider's rate limits) |
| `--timeout` | `400` | seconds per vision call |
| `--save` | `200` | keep overlays / masks for the first N items, JSON only afterwards |
| `--force` | off | ignore the checkpoint and redo everything |
| `--out` | `results/<benchmark>/<model>/<split>[__<config>]` | output directory |

Runs are **checkpointed per item**: re-running the same command resumes. A larger `--n` extends the
same run. A non-default `--iterations`, `--shuffle`, `--negatives` or design switch gets its own
directory (e.g. `val__it12__topo`), so two
configurations never mix. A run stops cleanly (without caching the item) on a billing, auth or daily
quota error; fix the account and re-run to resume. Pipeline failures are recorded and scored on
whatever mask they produced (usually empty).

To compare two runs (two models, two providers, two settings) on exactly their shared items:

```bash
python evals/compare_subset.py --a results/reasonseg/gemini-3.8-flash/val \
                               --b results/reasonseg/claude-sonnet-5-5/val
```

## 3. Metrics and outputs

| benchmark | metric (as in the paper) | where |
|---|---|---|
| ReasonSeg, RefCOCO/+/g | gIoU (mean per-item IoU), cIoU (ΣI / ΣU) | `summary.json`: `rec_gIoU`, `rec_cIoU` |
| SA-Co/Gold, LVIS, COCO | cgF1 = 100 · IL_MCC · pmF1 | `summary.json`: `rec_cgF1`, `rec_pmF1`, `rec_IL_MCC` |
| LVIS, COCO | box / mask AP (COCO-style, over fully queried images) | `score_ap.py` → `ap_summary_rec_mask.json` |
| ADE20K | class-oracle gIoU / cIoU | `summary.json`: `rec_gIoU` |
| Cityscapes | 19-class mIoU | `summary.json`: `rec_mIoU` |
| ODinW13, RF-100VL | AP0 = mean per-subset box AP | `score_apzero.py` → `results/<collection>/AP0_<model>_rec.json` |

Notes on the protocol:

* Items whose ground truth is empty are scored, not skipped: an empty prediction scores IoU 1.
* For cgF1, IL_MCC is the Matthews correlation of present / absent decisions over all items, and
  pmF1 is instance F1 at IoU 0.5 pooled per prompt over positive items and macro-averaged over prompts.
  Instances are the connected components of the predicted union mask, as for every method in the
  paper. `score_ap.py` also reports pmF1 / cgF1 on the true per-instance masks (`instance_*`).
* AP ranks each instance by its detection confidence. An image enters AP only if every ground-truth
  category in it was queried; on LVIS this is COCO-style AP over those images, not federated LVIS AP.
* ADE20K asks only the classes present in each image (`--negatives 0`), an oracle-class upper bound,
  not the 150-class ADE20K mIoU. Cityscapes runs one query naming all 19 classes per image and
  composites the labelled instances into a label map.

Each run directory holds `results.jsonl` (one record per item), `summary.json` (rewritten after every
item) and `work/<item>/` (the pipeline outputs for that item).

## 4. Reproduce the paper

```bash
MODEL=gemini-3.7-flash PARALLEL=8 bash scripts/reproduce_paper.sh
ONLY="reasonseg refcoco" bash scripts/reproduce_paper.sh          # a subset of benchmarks
```

This runs every benchmark with the paper's item selection (full ReasonSeg, Cityscapes and ODinW13;
seeded samples of 500 per RefCOCO split, 2,500 SA-Co/Gold, 1,500 LVIS and COCO with 3 negatives per
image, 1,000 ADE20K; the first 20 items of each RF-100VL subset), with T = 8. The selected items are
identical to those of the paper, so results can be compared item by item.

## Main results

LLMSeg with Gemini-3.7-Flash (T = 8). Full tables, baselines and ablations are in the paper.

| benchmark | metric | LLMSeg (w/o refine) | **LLMSeg** | SAM 3 (same items) |
|---|---|---|---|---|
| ReasonSeg val / test | cIoU | 64.6 / 58.2 | **71.5 / 65.8** | 9.9 / 11.3 |
| RefCOCO val / testA / testB | cIoU | 65.5 / 65.7 / 72.9 | **71.2 / 73.8 / 76.0** | 42.0 / 42.8 / 44.2 |
| RefCOCO+ val / testA / testB | cIoU | 59.7 / 61.4 / 61.1 | **67.0 / 69.9 / 64.9** | 33.6 / 37.0 / 32.9 |
| RefCOCOg val / test | cIoU | 64.8 / 69.3 | **70.9 / 74.5** | 40.0 / 39.7 |
| LVIS | cgF1 / box AP / mask AP | 50.4 / 57.1 / 36.9 | **53.9 / 64.2 / 47.2** | 50.1 / 69.1 / 62.2 |
| COCO | cgF1 / box AP / mask AP | 50.4 / 59.6 / 38.0 | **56.1 / 66.6 / 50.5** | 52.4 / 68.0 / 59.2 |
| SA-Co/Gold | cgF1 / pmF1 | | 36.4 / 55.4 | 45.4 / 55.2 |
| ADE20K | gIoU | 56.6 | **59.2** | 57.9 |
| Cityscapes | mIoU | 48.8 | **53.3** | 63.7 |
| ODinW13 | AP0 | 52.0 | **57.2** | 51.5 |
| RF-100VL | AP0 | 30.4 | **31.4** | 14.2 |

Back to the [README](README.md).
