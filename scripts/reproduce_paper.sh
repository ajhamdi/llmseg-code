#!/usr/bin/env bash
# Run every benchmark of the paper with its exact item selection (T = 8, seed 0).
#
#   bash scripts/reproduce_paper.sh                       # default model of the first provider with a key
#   MODEL=gemini-3.7-flash bash scripts/reproduce_paper.sh
#   MODEL=google/gemini-3.7-flash PARALLEL=16 bash scripts/reproduce_paper.sh
#   ONLY="reasonseg refcoco" bash scripts/reproduce_paper.sh
#
# Every run is checkpointed per item: re-running the script resumes where it stopped.
# A pipeline failure is recorded as a (scored) result; delete its line from results.jsonl to retry it.
#
#   benchmark            items                               metric
#   ReasonSeg val/test   all 200 / all 779                   gIoU, cIoU
#   RefCOCO/+/g (8)      500 per split, shuffled seed 0      gIoU, cIoU
#   SA-Co/Gold           2,500 over 7 domains, shuffled      cgF1, pmF1
#   LVIS / COCO          1,500 each, 3 negatives, shuffled   cgF1, pmF1, box/mask AP
#   ADE20K               1,000, shuffled                     gIoU
#   Cityscapes val       all 500                             mIoU
#   ODinW13              every item of the 13 subsets        AP0 (mean box AP)
#   RF-100VL             first 20 items of the 100 subsets   AP0 (mean box AP)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
PAR="${PARALLEL:-8}"
ONLY="${ONLY:-reasonseg cityscapes saco lvis coco ade20k odinw refcoco rf100vl}"
want() { [[ " $ONLY " == *" $1 "* ]]; }
COMMON=(--parallel "$PAR" --iterations 8)
[ -n "${MODEL:-}" ] && COMMON+=(--model "$MODEL")
SHUF=(--shuffle --seed 0)
MODEL_DIR="${MODEL:-$("$PY" -c 'import llm; print(llm.resolve_model(None)[0])')}"

if want reasonseg; then
  "$PY" evals/run_reasonseg.py "${COMMON[@]}" --split val
  "$PY" evals/run_reasonseg.py "${COMMON[@]}" --split test
fi
want cityscapes && "$PY" evals/run_semseg.py "${COMMON[@]}"
want saco && "$PY" evals/run_saco.py "${COMMON[@]}" --split gold --domain all --replica a --n 2500 "${SHUF[@]}"
for ds in lvis coco; do
  if want "$ds"; then
    "$PY" evals/run_instseg.py "${COMMON[@]}" --dataset "$ds" --negatives 3 --n 1500 "${SHUF[@]}"
    split=$([ "$ds" = coco ] && echo val2017 || echo v1_val)
    "$PY" evals/score_ap.py --dataset "$ds" \
      --results "results/$ds/${MODEL_DIR//\//_}/${split}__neg3__shuf0" > /dev/null \
      && echo "AP -> results/$ds/${MODEL_DIR//\//_}/${split}__neg3__shuf0/ap_summary_rec_mask.json"
  fi
done
want ade20k && "$PY" evals/run_ade20k.py "${COMMON[@]}" --n 1000 "${SHUF[@]}"
if want odinw; then
  "$PY" evals/run_apzero.py "${COMMON[@]}" --collection odinw --n 0
  "$PY" evals/score_apzero.py --collection odinw --model "$MODEL_DIR"
fi
if want refcoco; then
  for spec in refcoco:val refcoco:testA refcoco:testB refcocoplus:val refcocoplus:testA \
              refcocoplus:testB refcocog:val refcocog:test; do
    "$PY" evals/run_refcoco.py "${COMMON[@]}" --dataset "${spec%%:*}" --split "${spec##*:}" \
      --n 500 "${SHUF[@]}"
  done
fi
if want rf100vl; then
  "$PY" evals/run_apzero.py "${COMMON[@]}" --collection rf100vl --n 20
  "$PY" evals/score_apzero.py --collection rf100vl --model "$MODEL_DIR"
fi
echo "done. Per-benchmark metrics: results/<benchmark>/<model>/<split>/summary.json"
