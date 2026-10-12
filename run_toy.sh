#!/usr/bin/env bash
# Run the full pipeline (T = 8 refinement passes) on the 16 toy images.
#
#   ./run_toy.sh                               # default model of the first provider with a key set
#   ./run_toy.sh gemini-3.7-flash              # Gemini API      (GEMINI_API_KEY)
#   ./run_toy.sh google/gemini-3.7-flash       # OpenRouter      (OPENROUTER_API_KEY)
#   ./run_toy.sh claude-sonnet-5-5             # Anthropic API   (ANTHROPIC_API_KEY)
#   ./run_toy.sh gpt-5.6-terra                 # OpenAI API      (OPENAI_API_KEY)
#   ITERATIONS=0 ./run_toy.sh                  # stop after stage 2 (no refinement)
#
# Results land in output/toy/<model>/<image>/ (final mask: rec_mask.png, log: run.log).
set -uo pipefail
cd "$(dirname "$0")"
MODEL="${1:-}"
PY="${PYTHON:-python}"
ITERATIONS="${ITERATIONS:-8}"
SLUG="${MODEL:-default}"
OUT="output/toy/${SLUG//\//_}"
MODEL_ARGS=()
[ -n "${MODEL}" ] && MODEL_ARGS=(--model "${MODEL}")

CASES=(
  "sunflower.jpg|the sunflower"
  "child.jpg|the children"
  "clock.jpeg|the clock"
  "person.jpg|the people"
  "screen.jpg|the screen"
  "donut.jpeg|the donut"
  "taxis.jpeg|the taxi"
  "kingfisher.jpg|the bird"
  "toy.jpg|the toy"
  "cat.jpg|the cat"
  "dog.jpeg|the dog"
  "car.jpg|the car"
  "airplane.jpg|the airplane"
  "bicycle.jpg|the bicycle"
  "bananas.jpeg|the bananas"
  "sailboat.jpeg|the sailboat"
)

for case in "${CASES[@]}"; do
  img="${case%%|*}"
  query="${case#*|}"
  dir="${OUT}/${img%.*}"
  mkdir -p "${dir}"
  echo "=== ${img}: \"${query}\""
  "${PY}" pipeline.py "data/toy/${img}" "${query}" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
      --iterations "${ITERATIONS}" --out-dir "${dir}" 2> "${dir}/run.log" \
    || echo "    failed (see ${dir}/run.log)"
done
