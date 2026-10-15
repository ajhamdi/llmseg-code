<div align="center">

# LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines

**Abdullah Hamdi**

King Abdullah University of Science and Technology (KAUST)

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](#citation)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![Providers](https://img.shields.io/badge/API-Gemini%20%7C%20Anthropic%20%7C%20OpenAI%20%7C%20OpenRouter-6f42c1.svg)](#api-keys-and-providers)

<img src="assets/teaser.png" width="95%">

</div>

LLMSeg turns an off-the-shelf multimodal LLM into a text-prompted image segmenter, **with no
training, no segmentation head and no mask decoder**. The model is only ever asked for
coordinates (boxes and B-spline control points) as JSON; everything else (parsing, box
confinement, rasterization, mask composition, stopping) is deterministic code.

This repository contains the full method, the 16 toy images used during development, and the
complete evaluation harness for every benchmark in the paper. Every component runs with an API key
from **any one** of Google Gemini, Anthropic, OpenAI or OpenRouter.

---

## Contents

- [Installation](#installation)
- [API keys and providers](#api-keys-and-providers)
- [Quick start: segment an image](#quick-start-segment-an-image)
- [Toy examples](#toy-examples)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/ajhamdi/llmseg-code.git
cd llmseg-code

pip install -r requirements.txt          # pipeline + toy examples
pip install -r requirements-evals.txt    # + evaluation harness and dataset downloads
```

Or with [uv](https://docs.astral.sh/uv/): `uv sync` (pipeline) or `uv sync --extra evals` (everything),
then prefix commands with `uv run`.

## API keys and providers

You need **one** key. Put it in a `.env` file (copy `.env.example`) or export it:

| provider | environment variable | example `--model` |
|---|---|---|
| Google Gemini API | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-3.8-flash`, `gemma-4-31b-it` |
| OpenRouter | `OPENROUTER_API_KEY` | `google/gemini-3.8-flash`, `qwen/qwen3-vl-235b-a22b-instruct` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5`, `claude-opus-5-5` |
| OpenAI | `OPENAI_API_KEY` | `gpt-5.6-terra` |

The provider is inferred from the model name: a slug containing `/` goes to OpenRouter,
`gemini-*` / `gemma-*` to the Gemini API, `claude-*` to Anthropic, `gpt-*` / `o<digit>*` to OpenAI.
Force it with `--provider {gemini,anthropic,openai,openrouter}`. If you omit `--model`, the first
provider with a key set (in the order Gemini, OpenRouter, OpenAI, Anthropic) is used with the default
model from the table.

Prompts, parsers and geometry are identical for every provider. What differs is handled per model
family in `llm.py`: Gemini, Gemma and Qwen-VL answer on their native 0-1000 grid (Gemini and Gemma with
their native `box_2d = [ymin, xmin, ymax, xmax]` layout), every other model answers in pixels, and
all replies are converted to absolute pixels before any other code sees them.

## Quick start: segment an image

```bash
python pipeline.py data/toy/cat.jpg "the cat" --out-dir output/cat
python pipeline.py data/toy/taxis.jpeg "the taxi" --model gemini-3.8-flash
python pipeline.py data/toy/dog.jpeg "the dog" --model claude-sonnet-5-5
python pipeline.py data/toy/car.jpg "the car" --model gpt-5.6-terra
python pipeline.py data/toy/clock.jpeg "the clock" --model google/gemini-3.8-flash   # OpenRouter
```

The final mask is written to `<out-dir>/rec_mask.png` (default `output/<image stem>/`), alongside
the per-stage JSON records and overlays. Useful options: `--iterations T` (refinement budget,
default 8; `0` stops after stage 2), `--max-instances`, `--timeout`, and the ablation switches
`--topology`, `--refine-box` and `--no-postprocess` (all off by default, which is the paper's
main configuration). See `python pipeline.py --help` for details.

## Toy examples

The 16 development images (one query each) are in `data/toy/`:

```bash
./run_toy.sh                               # default model of the first provider with a key
./run_toy.sh gemini-3.8-flash              # Gemini API
./run_toy.sh google/gemini-3.8-flash       # OpenRouter
./run_toy.sh claude-sonnet-5-5             # Anthropic
./run_toy.sh gpt-5.6-terra                 # OpenAI
ITERATIONS=0 ./run_toy.sh                  # stage 2 only
./run_toy.sh gemini-3.8-flash --topology   # further arguments go to pipeline.py
```

Results land in `output/toy/<model>/<image>/` (final mask `rec_mask.png`, log `run.log`).

To check the geometry without any model call:

```bash
python -c "
import numpy as np
from geometry import mask_from_splines
th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
ctrl = np.column_stack([100 + 60 * np.cos(th), 100 + 60 * np.sin(th)])
m = mask_from_splines([{'role': 'outer', 'degree': 3, 'control_points': ctrl.tolist()}],
                      [0, 0, 200, 200], (200, 200))
print('coverage %', round(m.mean() * 100, 1))"

python -m pytest tests/      # offline tests: routing, coordinates, metrics, a full scripted pipeline run
```

## Evaluation

The `evals/` harness runs the pipeline on ReasonSeg, RefCOCO/+/g, SA-Co/Gold, LVIS, COCO, ADE20K,
Cityscapes, ODinW13 and RF-100VL and reports each benchmark's metric from the paper. The datasets are
not redistributed; download scripts are in `scripts/` (e.g.
`bash scripts/download_benchmarks.sh reasonseg`), and every run is checkpointed and resumable:

```bash
pip install -r requirements-evals.txt
bash scripts/download_benchmarks.sh reasonseg
python evals/run_reasonseg.py --split val --model gemini-3.8-flash
MODEL=gemini-3.8-flash bash scripts/reproduce_paper.sh     # every benchmark, paper item selection
```

See [EVALUATION.md](EVALUATION.md) for the downloads of every benchmark, all harness commands and
options, the metrics, the full reproduction protocol and the
[main results](EVALUATION.md#main-results).

## Citation

```bibtex
@article{hamdi2026llmseg,
  title   = {LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines},
  author  = {Hamdi, Abdullah},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

We evaluate on ReasonSeg, RefCOCO/+/g, SA-Co, LVIS, COCO, ADE20K, Cityscapes, ODinW and RF-100VL,
and follow the image evaluation suite of SAM 3. We thank the authors of these benchmarks for making
them available.
