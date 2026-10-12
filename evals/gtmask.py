"""Ground-truth mask decoding for COCO-format benchmarks (RefCOCO, SA-Co, LVIS, COCO, ODinW, RF-100VL).

Masks are stored either as a list of polygons ``[[x0, y0, x1, y1, ...], ...]`` or
as a COCO RLE ``{"counts": [...] | "<str>", "size": [h, w]}``; ``ann_to_mask``
takes either and returns a boolean HxW array.

``pycocotools`` is the reference rasterizer and is used whenever it is installed
(``pip install -r requirements-evals.txt``). A pure-Python fallback is selected
otherwise: its RLE decoder is exact, its polygon fill (PIL) can differ by a
boundary pixel. ``backend()`` is recorded in every summary.
"""
from __future__ import annotations
import numpy as np

try:                                    # reference rasterizer, if the user has it
    from pycocotools import mask as _coco_mask
except Exception:                       # noqa: BLE001 - any import failure means "use fallback"
    _coco_mask = None


def backend() -> str:
    return "pycocotools" if _coco_mask is not None else "pure-python"


# --------------------------------------------------------------------------- #
# pure-Python fallback                                                          #
# --------------------------------------------------------------------------- #
def decode_counts(s: str) -> list:
    """COCO's LEB128-ish compressed-RLE string -> run lengths.

    Ported from cocotools `rleFrString`: 5 payload bits per char (offset by 48),
    bit 0x20 continues the number, bit 0x10 of the final char sign-extends it,
    and from the 3rd run onwards each value is a *delta* against run i-2.
    """
    cnts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts


def _rle_to_mask_py(rle: dict, h: int, w: int) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = decode_counts(counts.decode() if isinstance(counts, bytes) else counts)
    flat = np.zeros(h * w, dtype=bool)
    pos, on = 0, False           # runs alternate, starting with a run of zeros
    for c in counts:
        if on:
            flat[pos:pos + c] = True
        pos += c
        on = not on
    return flat.reshape((h, w), order="F")   # COCO RLE is column-major


def _polys_to_mask_py(polys, h: int, w: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for poly in polys:
        if len(poly) < 6:        # fewer than 3 points cannot bound an area
            continue
        d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1)
    return np.array(img, dtype=bool)


# --------------------------------------------------------------------------- #
def ann_to_mask(seg, h: int, w: int) -> np.ndarray:
    """One annotation's `segmentation` field -> boolean HxW mask.

    Returns an empty mask on absent, malformed, or zero-dimension input rather
    than raising: some detection exports (e.g. RF-100VL) carry degenerate polygons
    or image records with no height/width, which make pycocotools' `frPyObjects`
    throw. A box-only dataset needs no GT mask (its metric is box AP), so an empty
    mask is the correct fallback; valid LVIS/COCO polygons are unaffected.
    """
    if seg is None or h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=bool)
    try:
        if isinstance(seg, dict):                       # RLE
            size = seg.get("size") or [h, w]
            if _coco_mask is not None:
                rle = dict(seg)
                if isinstance(rle["counts"], list):     # uncompressed -> compress first
                    rle = _coco_mask.frPyObjects(rle, size[0], size[1])
                elif isinstance(rle["counts"], str):
                    rle = {"counts": rle["counts"].encode(), "size": size}
                m = _coco_mask.decode(rle).astype(bool)
            else:
                m = _rle_to_mask_py(seg, size[0], size[1])
        else:                                           # polygon list
            polys = seg if (seg and isinstance(seg[0], (list, tuple))) else [seg]
            if _coco_mask is not None:
                rles = _coco_mask.frPyObjects([list(map(float, p)) for p in polys if len(p) >= 6], h, w)
                if not rles:
                    return np.zeros((h, w), dtype=bool)
                m = _coco_mask.decode(_coco_mask.merge(rles)).astype(bool)
            else:
                m = _polys_to_mask_py(polys, h, w)
    except Exception:                                   # malformed segmentation -> no GT mask
        return np.zeros((h, w), dtype=bool)
    if m.shape != (h, w):        # defensive: a mis-sized RLE would corrupt every IoU
        from PIL import Image
        m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST), dtype=bool)
    return m


def anns_to_masks(anns, h: int, w: int, skip_crowd: bool = True) -> list:
    """List of annotations -> list of per-instance boolean masks."""
    out = []
    for a in anns:
        if skip_crowd and a.get("iscrowd"):
            continue
        m = ann_to_mask(a.get("segmentation"), h, w)
        if m.any():
            out.append(m)
    return out


def union(masks, shape) -> np.ndarray:
    u = np.zeros(shape, dtype=bool)
    for m in masks:
        u |= m
    return u


def connected_components(mask: np.ndarray, min_px: int = 1) -> list:
    """Split a union mask into instance proxies.

    Used for the live pmF1, which is computed from the union mask for every method.
    Connected components is exact when instances are spatially separate and
    merges them when they touch.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        c = lab == i
        if int(c.sum()) >= min_px:
            out.append(c)
    return out
