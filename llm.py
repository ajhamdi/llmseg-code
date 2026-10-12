"""Vision calls (four providers), tolerant JSON parsing, per-backend coordinate conventions.

One call shape for every stage: one text prompt + one image in, reply text out.
The provider underneath is picked by ``resolve_provider``:

    provider     API key env var                    example --model
    gemini       GEMINI_API_KEY (or GOOGLE_API_KEY) gemini-3.7-flash
    anthropic    ANTHROPIC_API_KEY                  claude-sonnet-5-5
    openai       OPENAI_API_KEY                     gpt-5.6-terra
    openrouter   OPENROUTER_API_KEY                 google/gemini-3.7-flash

``--provider auto`` (the default) infers it from the model name: a slug with a
``/`` goes to OpenRouter, ``gemini-*`` / ``gemma-*`` to the Gemini API,
``claude-*`` to Anthropic, ``gpt-*`` / ``o<digit>*`` to OpenAI. With no
``--model`` at all, the first provider whose key is set picks its default model.
Prompts, parsers and geometry are identical across providers.

Everything stored or passed between stages is absolute pixels, origin top-left,
boxes as [x1, y1, x2, y2]. What goes *on the wire* to the model is per-backend
(decided by the model family, not the provider) and lives entirely in this module:

  * Gemini / Gemma are trained on a 0-1000 normalized grid and their native box
    is ``box_2d = [ymin, xmin, ymax, xmax]`` (y first). We ask for that native
    layout and convert in code; asking for x-first boxes makes the model
    transpose its own output, which it silently fails to do on some images.
  * Qwen-VL shares the 0-1000 grid but keeps x-first boxes.
  * Every other model is prompted and answered in pixels.

Control points are always [x, y] on every backend.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

try:  # optional: read API keys from a .env file next to this module or in the CWD
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
except ImportError:
    pass

PROVIDERS = ("gemini", "anthropic", "openai", "openrouter")
KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}
# Default model per provider, used when --model is omitted. The paper's main
# results use Gemini-3.7-Flash (native Gemini API, or the same model via OpenRouter).
DEFAULT_MODELS = {
    "gemini": "gemini-3.7-flash",
    "openrouter": "google/gemini-3.7-flash",
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5-5",
}
# Order in which an unset --model looks for a usable key.
DEFAULT_PROVIDER_ORDER = ("gemini", "openrouter", "openai", "anthropic")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TIMEOUT = 400  # seconds per vision call
MAX_RETRIES = 5
MAX_OUTPUT_TOKENS = int(os.getenv("LLMSEG_MAX_OUTPUT_TOKENS", "32000"))  # Anthropic needs an explicit cap
ANTHROPIC_MAX_IMAGE_BYTES = 5 * 1024 * 1024
NORMALIZED_GRID = 1000.0

_provider_override: str | None = None
_timeout = DEFAULT_TIMEOUT


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Provider selection                                                          #
# --------------------------------------------------------------------------- #
def api_key(provider: str) -> str | None:
    for name in KEY_ENV[provider]:
        if os.getenv(name):
            return os.getenv(name)
    return None


def infer_provider(model: str) -> str:
    """Provider implied by a model name (see the module docstring)."""
    name = (model or "").strip().lower()
    if "/" in name:
        return "openrouter"
    if name.startswith(("gemini", "gemma")):
        return "gemini"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "chatgpt")) or re.match(r"^o\d", name):
        return "openai"
    return "openrouter"


def resolve_model(model: str | None, provider: str = "auto") -> tuple[str, str]:
    """(model, provider) after applying defaults. Raises if no API key is available."""
    if provider not in ("auto", *PROVIDERS):
        raise ValueError(f"unknown provider {provider!r}; choose from auto, {', '.join(PROVIDERS)}")
    if not model:
        candidates = DEFAULT_PROVIDER_ORDER if provider == "auto" else (provider,)
        for p in candidates:
            if api_key(p):
                return DEFAULT_MODELS[p], p
        keys = ", ".join(n for p in candidates for n in KEY_ENV[p])
        raise RuntimeError(f"no API key found; set one of: {keys} (or put it in a .env file)")
    p = infer_provider(model) if provider == "auto" else provider
    if not api_key(p):
        hint = ""
        if p == "gemini" and api_key("openrouter"):
            hint = f" (only OPENROUTER_API_KEY is set: try --model google/{model})"
        raise RuntimeError(f"model {model!r} routes to provider '{p}', which needs "
                           f"{' or '.join(KEY_ENV[p])}{hint}")
    return model, p


def configure(provider: str = "auto", timeout: float | None = None) -> None:
    """Process-wide call settings (set once by the command line entry point)."""
    global _provider_override, _timeout
    _provider_override = None if provider in (None, "auto") else provider
    if timeout:
        _timeout = float(timeout)


def provider_for(model: str) -> str:
    return _provider_override or infer_provider(model)


# --------------------------------------------------------------------------- #
# Provider calls                                                              #
# --------------------------------------------------------------------------- #
def _image_bytes(image_path: Path) -> tuple[bytes, str]:
    """The exact file bytes (no re-encode) and their MIME type."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if not (mime and mime.startswith("image/")):
        mime = "image/png"
    return image_path.read_bytes(), mime


def _call_gemini(model: str, prompt: str, image_path: Path, timeout: float) -> str:
    from google import genai
    from google.genai import types

    data, mime = _image_bytes(image_path)
    client = genai.Client(api_key=api_key("gemini"),
                          http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    response = client.models.generate_content(
        model=model, contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt])
    return str(getattr(response, "text", "") or "").strip()


def _anthropic_image(image_path: Path) -> tuple[bytes, str]:
    """Anthropic accepts jpeg/png/gif/webp up to 5 MB; re-encode only when outside that."""
    data, mime = _image_bytes(image_path)
    if mime in ("image/jpeg", "image/png", "image/gif", "image/webp") and len(data) <= ANTHROPIC_MAX_IMAGE_BYTES:
        return data, mime
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        for fmt, kwargs, out_mime in (("PNG", {"optimize": True}, "image/png"),
                                      ("JPEG", {"quality": 92}, "image/jpeg")):
            buf = io.BytesIO()
            im.save(buf, fmt, **kwargs)
            if buf.tell() <= ANTHROPIC_MAX_IMAGE_BYTES:
                return buf.getvalue(), out_mime
    return buf.getvalue(), "image/jpeg"


def _call_anthropic(model: str, prompt: str, image_path: Path, timeout: float) -> str:
    import anthropic

    data, mime = _anthropic_image(image_path)
    client = anthropic.Anthropic(api_key=api_key("anthropic"), timeout=timeout, max_retries=0)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": mime,
                                     "data": base64.b64encode(data).decode("ascii")}},
        {"type": "text", "text": prompt},
    ]
    # Streamed so a long batched reply is not rejected as a too-long non-streaming request.
    with client.messages.stream(model=model, max_tokens=MAX_OUTPUT_TOKENS,
                                messages=[{"role": "user", "content": content}]) as stream:
        message = stream.get_final_message()
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()


def _call_openai_compatible(provider: str, model: str, prompt: str, image_path: Path,
                            timeout: float) -> str:
    """OpenAI Chat Completions; OpenRouter speaks the same API at another base URL."""
    from openai import OpenAI

    data, mime = _image_bytes(image_path)
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    if provider == "openrouter":
        client = OpenAI(api_key=api_key("openrouter"), base_url=OPENROUTER_BASE_URL,
                        timeout=timeout, max_retries=0)
        model = model.removeprefix("openrouter/")
    else:
        client = OpenAI(api_key=api_key("openai"), timeout=timeout, max_retries=0)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}])
    if not response or not getattr(response, "choices", None):
        return ""
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "\n".join(
            str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content or "").strip()


_NON_RETRYABLE = (400, 401, 403, 404)


def _complete(provider: str, model: str, prompt: str, image_path: Path, timeout: float) -> str:
    """One provider call with retries on transient errors."""
    if not api_key(provider):
        raise RuntimeError(f"provider '{provider}' needs {' or '.join(KEY_ENV[provider])}")
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "gemini":
                return _call_gemini(model, prompt, image_path, timeout)
            if provider == "anthropic":
                return _call_anthropic(model, prompt, image_path, timeout)
            return _call_openai_compatible(provider, model, prompt, image_path, timeout)
        except ImportError as exc:
            raise RuntimeError(f"the '{provider}' provider needs its SDK: pip install -r requirements.txt "
                               f"({exc})") from exc
        except Exception as exc:  # noqa: BLE001 (retry any transient provider error)
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status in _NON_RETRYABLE:
                raise RuntimeError(f"{provider} call failed ({status}): {exc}") from exc
            last_exc = exc
            time.sleep(min(2 ** attempt, 8.0) + random.uniform(0.0, 0.2))
    raise RuntimeError(f"{provider} call failed after {MAX_RETRIES} attempts: {last_exc}")


def query(prompt: str, image_path, model: str, what: str, timeout: float | None = None) -> str:
    """One multimodal call (prompt + one image), wall-clock bounded; returns reply text."""
    timeout = float(timeout or _timeout)
    path = Path(image_path)
    if not path.exists():
        raise RuntimeError(f"image not found for vision call: {path}")
    provider = provider_for(model)
    log(f"[llm:{provider}:{model}] requesting {what} (timeout {timeout:.0f}s) ...")
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_complete, provider, model, prompt, path.resolve(), timeout)
    try:
        text = fut.result(timeout=timeout)
    except FuturesTimeout as exc:
        raise TimeoutError(f"{model} exceeded {timeout:.0f}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not text:
        raise RuntimeError(f"model '{model}' returned an empty response")
    return text


# --------------------------------------------------------------------------- #
# Tolerant JSON parsing                                                       #
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict:
    """Pull a JSON object out of model text (fences / surrounding prose allowed)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


_JSON_BAREWORDS = frozenset({"true", "false", "null", "NaN", "Infinity"})


def strip_stray_barewords(text: str) -> str:
    """Drop unquoted identifier tokens a model leaked into its JSON.

    Outside a string the only legal words are the JSON literals and a number's
    e/E exponent; any other alphabetic run is removed so one stray token does
    not make an otherwise complete entry unparseable.
    """
    out: list = []
    in_str = escaped = False
    last = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                last = ch
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            last = ch
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            is_exponent = word[0] in "eE" and last in "0123456789."
            if word in _JSON_BAREWORDS or is_exponent:
                out.append(word)
                last = word[-1]
            i = j
            continue
        out.append(ch)
        if not ch.isspace():
            last = ch
        i += 1
    return "".join(out)


def repair_json_report(text: str):
    """Re-insert closers a malformed reply dropped -> (repaired_text, truncated).

    A closer that does not match the innermost open frame closes the frames
    inside it; an opener where a key is due means the previous sibling never
    closed; anything still open at EOF is closed (and flagged as truncated).
    """
    text = strip_stray_barewords(text)
    out: list = []
    stack: list = []
    in_str = escaped = False
    last = ""
    for ch in text:
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                last = ch
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            last = ch
            continue
        if ch in "{[":
            if last == "," and stack and stack[-1] == "}":
                while out and out[-1].isspace():
                    out.pop()
                if out and out[-1] == ",":
                    out.pop()
                    out.append(stack.pop())
                    out.append(",")
                else:
                    out.append(stack.pop())
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
            last = ch
        elif ch in "}]":
            if not stack:
                continue
            if stack[-1] == ch:
                stack.pop()
                out.append(ch)
                last = ch
            elif ch in stack:
                while stack and stack[-1] != ch:
                    out.append(stack.pop())
                stack.pop()
                out.append(ch)
                last = ch
        else:
            out.append(ch)
            if not ch.isspace():
                last = ch
    truncated = bool(stack) or in_str
    if in_str:
        out.append('"')
    while stack:
        out.append(stack.pop())
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out)), truncated


def parse_repaired_json(text: str, only_if_complete: bool = False):
    """Structure-repair + parse; None when it still fails (or was truncated, if asked)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None
    repaired, truncated = repair_json_report(text[start:])
    if only_if_complete and truncated:
        return None
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json_objects(text: str) -> list:
    """Every balanced JSON object in *text* (innermost first); survives truncation."""
    text = strip_stray_barewords(text)
    objects: list = []
    stack: list = []
    in_str = escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_str:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                objects.append(json.loads(text[start:i + 1]))
            except json.JSONDecodeError:
                pass
    return objects


# --------------------------------------------------------------------------- #
# Coordinate conventions                                                      #
# --------------------------------------------------------------------------- #
def _leaf(model: str) -> str:
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def is_normalized_grid_model(model: str) -> bool:
    """Backends trained on a 0-1000 normalized grid (Gemini, Gemma, Qwen-VL)."""
    return _leaf(model).startswith(("gemini", "gemma", "qwen"))


def is_yfirst_box_model(model: str) -> bool:
    """Backends whose native box is [ymin, xmin, ymax, xmax] (Gemini, Gemma)."""
    return _leaf(model).startswith(("gemini", "gemma"))


def box_field_name(model: str) -> str:
    return "box_2d" if is_yfirst_box_model(model) else "box"


def box_order_literal(model: str) -> str:
    return "[ymin, xmin, ymax, xmax]" if is_yfirst_box_model(model) else "[x1, y1, x2, y2]"


def box_format_instructions(model: str) -> str:
    field, order = box_field_name(model), box_order_literal(model)
    if is_yfirst_box_model(model):
        return (
            f"where {field} is {order} in the coordinate space described above --\n"
            f"the standard box_2d layout, y FIRST then x (ymin < ymax, xmin < xmax,\n"
            f"all values inside the valid range for that space)."
        )
    return (
        f"where {field} is {order} in the coordinate space described above\n"
        f"(x1 < x2, y1 < y2, all values inside the valid range for that space)."
    )


def point_order_note(model: str) -> str:
    if not is_yfirst_box_model(model):
        return ""
    return (
        "- NOTE the two different layouts: every BOX is box_2d = [ymin, xmin, ymax, xmax]\n"
        "  (y first), but every CONTROL POINT is [x, y] (x first). Do not mix them up.\n"
    )


def coord_space_instructions(model: str, width: int, height: int) -> str:
    if is_normalized_grid_model(model):
        g = int(NORMALIZED_GRID)
        return (
            f"The image is {width}x{height} pixels, but ALL coordinates you output "
            f"(boxes, control points, expansions) use a NORMALIZED 0-{g:g} "
            f"grid — NOT absolute pixels.\n"
            f"  x=0 is the left edge, x={g:g} is the right edge;\n"
            f"  y=0 is the top edge,  y={g:g} is the bottom edge;\n"
            f"  origin is TOP-LEFT; x increases right, y increases down.\n"
            f"Any box written in this prompt is already on that same 0-{g:g} "
            f"grid and in the same {box_order_literal(model)} order. Do NOT convert "
            f"to pixels or reorder them yourself."
        )
    return (
        f"The image is {width}x{height} pixels (origin top-left, x right, y down).\n"
        f"ALL coordinates are absolute PIXELS with "
        f"0 <= x <= {width} and 0 <= y <= {height}."
    )


def to_model_box(box, model: str, width, height) -> list:
    """Pixel box -> the model's units, still x-first."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if not is_normalized_grid_model(model):
        return [x1, y1, x2, y2]
    W, H = float(width), float(height)
    g = NORMALIZED_GRID
    return [x1 * g / W, y1 * g / H, x2 * g / W, y2 * g / H]


def format_model_box(box, model: str, width, height) -> str:
    """Pixel box -> the literal array the prompt schema uses (order + units)."""
    x1, y1, x2, y2 = (round(v) for v in to_model_box(box, model, width, height))
    if is_yfirst_box_model(model):
        return f"[{y1}, {x1}, {y2}, {x2}]"
    return f"[{x1}, {y1}, {x2}, {y2}]"


def _needs_grid_rescale(values) -> bool:
    """Grid replies are rescaled; values > 1000 mean the model answered in pixels."""
    vals = [abs(float(v)) for v in values]
    return bool(vals) and max(vals) <= NORMALIZED_GRID


def _box_coords(box):
    try:
        return [float(v) for v in list(box)[:4]]
    except (TypeError, ValueError):
        return []


_BOX_KEYS = ("box_2d", "bbox_2d", "box", "bbox", "box2d")


def normalize_box_key(entry: dict) -> list:
    """Move whichever box key an entry used onto ``box``; return its coords."""
    if not isinstance(entry, dict):
        return []
    for key in _BOX_KEYS:
        coords = _box_coords(entry.get(key))
        if coords and len(coords) == 4:
            if key != "box":
                entry.pop(key, None)
                entry["box"] = coords
            return coords
    return []


def normalize_boxes(entries, model: str, width, height) -> None:
    """In place: every entry's box -> ``box`` as pixel [x1, y1, x2, y2].

    The transpose happens before the scale: the x scale belongs to the x values
    wherever they sit in the reply's array.
    """
    coords_by_entry = [(e, normalize_box_key(e)) for e in entries if isinstance(e, dict)]
    if not is_normalized_grid_model(model):
        return
    flat = [v for _, coords in coords_by_entry for v in coords]
    rescale = _needs_grid_rescale(flat)
    sx, sy = (width / NORMALIZED_GRID, height / NORMALIZED_GRID) if rescale else (1.0, 1.0)
    for entry, coords in coords_by_entry:
        if not coords:
            continue
        a, b, c, d = coords
        x1, y1, x2, y2 = (b, a, d, c) if is_yfirst_box_model(model) else (a, b, c, d)
        entry["box"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def rescale_spline_groups(groups, model: str, width, height) -> None:
    """In place: grid -> pixel control points, decided once across all groups.

    ``groups`` is a list of spline lists. Deciding over every point together
    avoids calling a small group "already pixels" while its siblings are on the grid.
    """
    if not is_normalized_grid_model(model):
        return
    coords = [c for splines in groups for s in splines if isinstance(s, dict)
              for p in (s.get("control_points") or [])
              if isinstance(p, (list, tuple)) and len(p) >= 2 for c in (p[0], p[1])]
    if not _needs_grid_rescale(coords):
        return
    sx, sy = width / NORMALIZED_GRID, height / NORMALIZED_GRID
    for splines in groups:
        for s in splines:
            if not isinstance(s, dict) or not s.get("control_points"):
                continue
            s["control_points"] = [
                [float(p[0]) * sx, float(p[1]) * sy, *list(p[2:])]
                if isinstance(p, (list, tuple)) and len(p) >= 2 else p
                for p in s["control_points"]
            ]


def _inside_frac(pts, box_xyxy, swap: bool) -> float:
    x1, y1, x2, y2 = box_xyxy
    padx, pady = max(2.0, 0.02 * (x2 - x1)), max(2.0, 0.02 * (y2 - y1))
    inside = 0
    for a, b in pts:
        px, py = (b, a) if swap else (a, b)
        if x1 - padx <= px <= x2 + padx and y1 - pady <= py <= y2 + pady:
            inside += 1
    return inside / len(pts)


def fix_transposed_points(splines, box, model: str) -> bool:
    """In place: re-read a loop as [y, x] when only that reading fits its box.

    Points confined to a box can be checked against it: flip only when the
    swapped reading is >= 80% inside and beats the original by >= 50 points.
    """
    if not is_yfirst_box_model(model) or not splines:
        return False
    coords = _box_coords(box)
    if len(coords) != 4:
        return False
    pts = []
    for s in splines:
        if not isinstance(s, dict):
            continue
        for p in s.get("control_points") or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    pass
    if len(pts) < 3:
        return False
    x1, x2 = sorted((coords[0], coords[2]))
    y1, y2 = sorted((coords[1], coords[3]))
    as_is = _inside_frac(pts, (x1, y1, x2, y2), swap=False)
    swapped = _inside_frac(pts, (x1, y1, x2, y2), swap=True)
    if swapped < 0.8 or swapped - as_is < 0.5:
        return False
    for s in splines:
        if not isinstance(s, dict) or not s.get("control_points"):
            continue
        s["control_points"] = [
            [float(p[1]), float(p[0]), *list(p[2:])]
            if isinstance(p, (list, tuple)) and len(p) >= 2 else p
            for p in s["control_points"]
        ]
    return True
