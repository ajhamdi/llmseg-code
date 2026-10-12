"""Offline tests: no API key and no network needed.

    python -m pytest tests/        (or)        python tests/test_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evals"))

import common as C  # noqa: E402
import geometry as G  # noqa: E402
import gtmask  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402


def test_provider_inference():
    assert llm.infer_provider("google/gemini-3.7-flash") == "openrouter"
    assert llm.infer_provider("qwen/qwen3-vl-235b-a22b-instruct") == "openrouter"
    assert llm.infer_provider("gemini-3.7-flash") == "gemini"
    assert llm.infer_provider("gemma-4-31b-it") == "gemini"
    assert llm.infer_provider("claude-sonnet-5-5") == "anthropic"
    assert llm.infer_provider("gpt-5.6-terra") == "openai"
    assert llm.infer_provider("o4-mini") == "openai"


def test_resolve_model_defaults(monkeypatch=None):
    import os
    saved = {k: os.environ.pop(k, None) for names in llm.KEY_ENV.values() for k in names}
    try:
        os.environ["ANTHROPIC_API_KEY"] = "x"
        assert llm.resolve_model(None) == (llm.DEFAULT_MODELS["anthropic"], "anthropic")
        os.environ["OPENROUTER_API_KEY"] = "x"
        assert llm.resolve_model(None) == (llm.DEFAULT_MODELS["openrouter"], "openrouter")
        assert llm.resolve_model("claude-sonnet-5-5") == ("claude-sonnet-5-5", "anthropic")
        try:
            llm.resolve_model("gpt-5.6-terra")
            raise AssertionError("expected a missing-key error")
        except RuntimeError as e:
            assert "OPENAI_API_KEY" in str(e)
    finally:
        for k in [k for names in llm.KEY_ENV.values() for k in names]:
            os.environ.pop(k, None)
            if saved.get(k) is not None:
                os.environ[k] = saved[k]


def test_coordinate_conventions():
    # Gemini answers box_2d = [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    dets = [{"box_2d": [100, 200, 500, 800], "label": "x"}]
    llm.normalize_boxes(dets, "gemini-3.7-flash", 2000, 1000)
    assert dets[0]["box"] == [400.0, 100.0, 1600.0, 500.0]
    # Pixel models are left alone.
    dets = [{"box": [1, 2, 3, 4]}]
    llm.normalize_boxes(dets, "claude-sonnet-5-5", 100, 100)
    assert dets[0]["box"] == [1.0, 2.0, 3.0, 4.0]


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for p in (0.0, 0.3, 1.0):
        m = rng.random((37, 23)) < p
        rle = G.mask_to_rle(m)
        assert (G.rle_to_mask(rle) == m).all()
        assert (gtmask.ann_to_mask(rle, 37, 23) == m).all()


def test_metrics():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert C.iou(a, a)[2] == 1.0
    assert C.iou(np.zeros_like(a), np.zeros_like(a))[2] == 1.0       # empty vs empty
    assert C.match_f1([a], [a]) == (1, 0, 0)
    recs = [{"query": "q", "gt_px": 5, "rec_px": 5, "rec_tp": 1, "rec_fp": 0, "rec_fn": 0},
            {"query": "q", "gt_px": 0, "rec_px": 0}]
    out = C.presence_f1(recs, "rec")
    assert out["rec_IL_MCC"] == 1.0 and out["rec_pmF1"] == 1.0 and out["rec_cgF1"] == 100.0


def _ellipse(cx, cy, rx, ry, n=16):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th)]).round(1).tolist()


def _fake_model(prompt: str) -> str:
    """Scripted replies for the four prompt types (pixel coordinates, one instance)."""
    if "Find every DISTINCT instance" in prompt:                       # stage 1
        return json.dumps({"detections": [{"label": "cat", "confidence": 0.9, "box": [40, 30, 160, 170],
                                           "morphology": {"aspect": "tall"}}]})
    if "This image shows the CURRENT segmentation" in prompt:          # stage 3a
        return json.dumps({"focus_boxes": [{"instance_id": 0, "type": "positive",
                                            "box": [130, 80, 160, 120], "shape": "bump"}]})
    if "current mask of the" in prompt:                                # stage 3b
        return json.dumps({"refinements": [{"id": 0, "splines": [
            {"role": "outer", "control_points": _ellipse(145, 100, 14, 18, 8)}]}]})
    return json.dumps({"mask_color": "cyan", "degree": 3, "instances": [  # stage 2
        {"id": 0, "splines": [{"role": "outer", "control_points": _ellipse(100, 100, 45, 60)}]}]})


def test_pipeline_end_to_end(tmp_path=None):
    import tempfile
    from PIL import Image
    tmp = Path(tmp_path or tempfile.mkdtemp())
    img = tmp / "img.png"
    Image.fromarray(np.full((200, 200, 3), 128, np.uint8)).save(img)
    real = llm._complete
    llm._complete = lambda provider, model, prompt, path, timeout: _fake_model(prompt)
    try:
        assert pipeline.run(img, "the cat", "claude-sonnet-5-5", tmp / "out", iterations=2) == 0
    finally:
        llm._complete = real
    out = tmp / "out"
    for f in ("detect.json", "seg.json", "recseg.json", "seg_mask.png", "rec_mask.png", "instances.json"):
        assert (out / f).exists(), f
    rec = json.loads((out / "recseg.json").read_text())
    assert rec["complete"] and rec["iterations"] == 2
    assert rec["passes"][0]["refinements"], "the positive focus box should have been applied"
    inst = json.loads((out / "instances.json").read_text())["instances"]
    assert len(inst) == 1 and inst[0]["label"] == "cat" and inst[0]["confidence"] == 0.9
    seg, final = G.rle_to_mask(inst[0]["seg_rle"]), G.rle_to_mask(inst[0]["rec_rle"])
    assert (final == (np.array(Image.open(out / "rec_mask.png")) > 0)).all()
    assert (seg == (np.array(Image.open(out / "seg_mask.png")) > 0)).all()
    assert final.sum() > seg.sum(), "a positive refinement grows the mask"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
