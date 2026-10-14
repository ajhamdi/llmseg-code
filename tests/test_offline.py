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
    """Scripted replies for the five prompt types (pixel coordinates, one instance)."""
    if "Find every DISTINCT instance" in prompt:                       # stage 1
        det = {"label": "cat", "confidence": 0.9, "box": [40, 30, 160, 170],
               "morphology": {"aspect": "tall"}}
        if '"topology"' in prompt:
            det["topology"] = {"components": 1, "holes": 1}
        return json.dumps({"detections": [det]})
    if "SIGNED side delta" in prompt:                                  # stage 3-pre (box adjust)
        return json.dumps({"instances": [{"id": 0, "x1": 0, "x2": 500, "y1": 0, "y2": -10}]})
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


def test_design_switches_end_to_end(tmp_path=None):
    """--topology, --refine-box and --no-postprocess all reach the pipeline and its records."""
    import tempfile
    from PIL import Image
    tmp = Path(tmp_path or tempfile.mkdtemp())
    img = tmp / "img.png"
    Image.fromarray(np.full((200, 200, 3), 128, np.uint8)).save(img)
    prompts_seen = []
    real = llm._complete

    def fake(provider, model, prompt, path, timeout):
        prompts_seen.append(prompt)
        return _fake_model(prompt)
    llm._complete = fake
    try:
        assert pipeline.run(img, "the cat", "claude-sonnet-5-5", tmp / "out", iterations=1,
                            topology=True, refine_box=True, postprocess=False) == 0
    finally:
        llm._complete = real
    out = tmp / "out"
    seg = json.loads((out / "seg.json").read_text())
    assert seg["config"] == {"topology": True, "refine_box": True, "postprocess": False,
                             "iterations": 1, "max_instances": pipeline.MAX_INSTANCES}
    assert seg["instances"][0]["topology"] == {"components": 1, "holes": 1}
    assert any("topology.components" in p for p in prompts_seen), "stage 2 sees the topology"
    rec = json.loads((out / "recseg.json").read_text())
    adj = rec["box_adjust"][0]
    # x2 asks for +500 px but is capped at 20% of the 120 px width; y2 shrinks by 10 px
    assert adj["box_after"] == [40.0, 30.0, 184.0, 160.0], adj
    assert json.loads((out / "instances.json").read_text())["instances"][0]["box"] == adj["box_after"]
    assert (out / "seg_box.png").exists() and not (out / "recseg_iter1_post.png").exists()


def test_topology_gated_postprocess():
    yy, xx = np.mgrid[:120, :120]
    ring = ((yy - 60) ** 2 + (xx - 60) ** 2 <= 50 ** 2) & ((yy - 60) ** 2 + (xx - 60) ** 2 > 4 ** 2)
    box = [0, 0, 120, 120]
    assert G.postprocess_mask(ring, box)[60, 60], "without topology a small hole is filled"
    assert not G.postprocess_mask(ring, box, {"holes": 1})[60, 60], "a reported hole is kept"
    speck = ring.copy()
    speck[2:6, 2:6] = True
    assert not G.postprocess_mask(speck, box)[3, 3]
    assert G.postprocess_mask(speck, box, {"components": 2})[3, 3], "a reported component is kept"


def test_box_adjust_units_and_caps():
    # grid models answer on 0-1000; deltas are rescaled per axis and keep their sign
    d = llm.deltas_to_pixels({"x1": 100, "x2": -50, "y1": 0, "y2": 10}, "gemini-3.7-flash", 400, 200)
    assert d == {"x1": 40.0, "x2": -20.0, "y1": 0.0, "y2": 2.0}
    assert llm.deltas_to_pixels({"x1": 7}, "claude-sonnet-5-5", 400, 200)["x1"] == 7.0
    assert G.adjust_box([10, 10, 110, 60], {"x1": 999, "y1": -999}, (400, 200)) == [0.0, 20.0, 110.0, 60.0]


def test_harness_forwards_switches():
    import argparse
    ap = C.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(["--topology", "--no-postprocess", "--iterations", "12"])
    assert C.pipeline_flags(args) == ["--topology", "--no-postprocess"]
    assert C.config_tag(args) == "it12__topo__nopost"
    assert C.config_tag(ap.parse_args([])) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
