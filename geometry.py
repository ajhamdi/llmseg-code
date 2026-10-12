"""Deterministic geometry: closed B-splines, box-confined rasterization,
signed soft-lock composition, postprocessing, and overlay rendering.

No model calls here. Every mask is built from one primitive, ``spline_region``,
which clamps both the control polygon samples and the resulting polygon to a
confinement box (an instance's detect box in stage 2, a focus box in stage 3),
so a mask can never leak outside its instance box.
"""

from __future__ import annotations

import inspect

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from skimage import morphology

from llm import log

# Rasterization: samples per cubic loop = clamp(round(2 * (box_w + box_h)), 64, 2000).
SPLINE_SAMPLE_BOX_PCT = 200.0
SPLINE_SAMPLES_MIN = 64
SPLINE_SAMPLES_MAX = 2000
DEFAULT_SPLINE_DEGREE = 3

# Signed soft-lock: an opposing edit cancels this much commitment per pass.
REFINE_LOCK_ALPHA = 0.5

# Effect gate: a focus box survives when it can still change
# >= max(MIN_PX, MIN_FRAC * box_area) pixels under the soft-lock.
REFINE_EFFECT_MIN_FRAC = 0.02
REFINE_EFFECT_MIN_PX = 16

# Island bridging: connect a positive-created island to the main blob when the
# gap is <= this % of the instance box's min dimension.
BRIDGE_MAX_DIST_PCT = 15.0

# Postprocessing (per pass): closing/opening radius as % of box min-dim, capped
# at 2 px (and at least 1 px when applied); hole / speck caps as % of object
# area with pixel floors.
POSTPROC_CLOSE_PCT = 1.5
POSTPROC_OPEN_PCT = 2.0
POSTPROC_RADIUS_MAX = 2
POSTPROC_HOLE_PCT = 2.0
POSTPROC_SPECK_PCT = 1.0
POSTPROC_HOLE_FLOOR = 32
POSTPROC_SPECK_FLOOR = 64

MASK_ALPHA = 0.25  # overlay tint opacity: light enough to see real edges underneath


# --------------------------------------------------------------------------- #
# Closed B-splines and box-confined rasterization                             #
# --------------------------------------------------------------------------- #
def eval_closed_bspline(ctrl: np.ndarray, degree: int = 3, n: int = 600) -> np.ndarray:
    """Sample a closed (periodic, uniform-knot) B-spline; degree 1 = the control polygon."""
    ctrl = np.asarray(ctrl, dtype=float)
    count = ctrl.shape[0]
    if count < 3:
        return np.vstack([ctrl, ctrl[:1]])
    degree = int(np.clip(degree, 1, count - 1))
    if degree == 1:
        return np.vstack([ctrl, ctrl[:1]])
    kv = np.arange(-degree, count + degree + 1, dtype=float)
    factor, fraction = divmod(count + degree + 1, count)
    wrapped = np.roll(np.concatenate((ctrl,) * factor + (ctrl[:fraction],)), -1, axis=0)
    spline = BSpline(kv, wrapped, degree, extrapolate=False)
    pts = spline(np.linspace(0, count, n))
    return pts[np.isfinite(pts).all(axis=1)]


def clip_to_box(pts: np.ndarray, box, size) -> np.ndarray:
    x1, y1, x2, y2 = box
    W, H = size
    x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], x1, x2)
    out[:, 1] = np.clip(out[:, 1], y1, y2)
    return out


def samples_for_box(box) -> int:
    x1, y1, x2, y2 = (float(v) for v in box)
    n = int(round(SPLINE_SAMPLE_BOX_PCT / 100.0 * (abs(x2 - x1) + abs(y2 - y1))))
    return int(np.clip(n, SPLINE_SAMPLES_MIN, SPLINE_SAMPLES_MAX))


def spline_region(ctrl, degree, box, size) -> np.ndarray:
    """Rasterize ONE closed B-spline, confined to ``box``, into a boolean image."""
    W, H = size
    poly = eval_closed_bspline(np.asarray(ctrl, float), int(degree), samples_for_box(box))
    poly = clip_to_box(poly, box, size)
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in poly], fill=255)
    return np.array(img, dtype=bool)


def mask_from_splines(splines: list, box, size) -> np.ndarray:
    """Outer loops OR in, then "hole" loops carve out."""
    W, H = size
    mask = np.zeros((H, W), dtype=bool)
    for s in splines:
        if s.get("role", "outer") == "hole" or len(s.get("control_points", [])) < 3:
            continue
        mask |= spline_region(s["control_points"], s.get("degree", DEFAULT_SPLINE_DEGREE), box, size)
    for s in splines:
        if s.get("role", "outer") != "hole" or len(s.get("control_points", [])) < 3:
            continue
        mask &= ~spline_region(s["control_points"], s.get("degree", DEFAULT_SPLINE_DEGREE), box, size)
    return mask


def focus_region(splines: list, focus_box, size) -> np.ndarray:
    """Union of one focus box's loops, each confined to that focus box."""
    W, H = size
    region = np.zeros((H, W), dtype=bool)
    for s in splines:
        if len(s.get("control_points", [])) < 3:
            continue
        region |= spline_region(s["control_points"], s.get("degree", DEFAULT_SPLINE_DEGREE), focus_box, size)
    return region


def union_masks(masks: list, size) -> np.ndarray:
    W, H = size
    out = np.zeros((H, W), dtype=bool)
    for m in masks:
        out |= np.asarray(m, dtype=bool)
    return out


def mask_to_rle(mask: np.ndarray) -> dict:
    """Boolean HxW mask -> uncompressed COCO RLE (column-major runs, zeros first)."""
    mask = np.asarray(mask, dtype=bool)
    flat = mask.ravel(order="F")
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat[0]:
        runs = [0] + runs
    return {"size": list(mask.shape), "counts": runs}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Inverse of ``mask_to_rle``."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, on = 0, False
    for c in rle["counts"]:
        if on:
            flat[pos:pos + c] = True
        pos += c
        on = not on
    return flat.reshape((h, w), order="F")


# --------------------------------------------------------------------------- #
# Boxes                                                                       #
# --------------------------------------------------------------------------- #
def box_iou(a, b) -> float:
    inter = box_overlap_area(a, b)
    if inter <= 0:
        return 0.0
    area = lambda r: abs(float(r[2]) - float(r[0])) * abs(float(r[3]) - float(r[1]))  # noqa: E731
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def box_overlap_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def assign_instance(fbox, instances: list, claimed_id=None):
    """Owning instance of a focus box: largest overlap, the claimed id breaks ties."""
    best_i, best_area = None, 0.0
    for i, inst in enumerate(instances):
        area = box_overlap_area(fbox, inst["box"])
        if area > best_area:
            best_area, best_i = area, i
        elif area > 0 and area == best_area and claimed_id is not None:
            try:
                if int(inst["id"]) == int(claimed_id):
                    best_i = i
            except (TypeError, ValueError):
                pass
    return best_i if best_area > 0 else None


def clamp_box(box, outer, size):
    """Intersect a focus box with its instance box and the image (None if degenerate)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    ox1, oy1, ox2, oy2 = (float(v) for v in outer)
    W, H = size
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(x1, ox1, 0.0)
    y1 = max(y1, oy1, 0.0)
    x2 = min(x2, ox2, float(W))
    y2 = min(y2, oy2, float(H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def box_roi(box, size) -> np.ndarray:
    W, H = size
    x1, y1, x2, y2 = (float(v) for v in box)
    xi1 = int(max(0, min(W, np.floor(min(x1, x2)))))
    yi1 = int(max(0, min(H, np.floor(min(y1, y2)))))
    xi2 = int(max(0, min(W, np.ceil(max(x1, x2)))))
    yi2 = int(max(0, min(H, np.ceil(max(y1, y2)))))
    roi = np.zeros((H, W), dtype=bool)
    if xi2 > xi1 and yi2 > yi1:
        roi[yi1:yi2, xi1:xi2] = True
    return roi


def box_union_area_px(boxes, size) -> int:
    """Pixel area of the union of instance boxes: the only region a pass can change."""
    W, H = size
    region = np.zeros((H, W), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(float(v))) for v in b)
        x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
        y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            region[y1:y2, x1:x2] = True
    return int(region.sum())


# --------------------------------------------------------------------------- #
# Stage 3 composition                                                         #
# --------------------------------------------------------------------------- #
def achievable_px(ftype: str, box, mask: np.ndarray, lock: np.ndarray, size):
    """(pixels this focus box could still change under the soft-lock, box area)."""
    roi = box_roi(box, size)
    area = int(roi.sum())
    if area == 0:
        return 0, 0
    if ftype == "negative":   # carve: free or erodible foreground
        target = mask & ((lock == 0.0) | (lock > 0.0))
    else:                     # add: free or erodible background
        target = (~mask) & ((lock == 0.0) | (lock < 0.0))
    return int((roi & target).sum()), area


def apply_locked_refinements(mask, lock, refinements: list, size, alpha: float = REFINE_LOCK_ALPHA):
    """Compose one pass through the signed commitment field ``lock``.

    Per pixel covered by a refinement of sign s (+1 add, -1 carve):
      free (lock == 0)          -> set the pixel to that side, lock = s
      agreeing (sign(lock) = s) -> lock += s (harder to overturn later)
      opposing                  -> |lock| -= alpha; flip (lock = s) once it reaches 0
    Negatives are applied before positives. Returns (mask, lock, positive_region).
    """
    mask = np.asarray(mask, dtype=bool).copy()
    lock = np.asarray(lock, dtype=float).copy()
    W, H = size
    positive_region = np.zeros((H, W), dtype=bool)

    def apply_region(region, sign):
        target = sign > 0
        free = region & (lock == 0.0)
        agree = region & (np.sign(lock) == sign)
        oppose = region & (np.sign(lock) == -sign)
        mask[free] = target
        lock[free] = float(sign)
        mask[agree] = target
        lock[agree] += float(sign)
        cand = np.abs(lock) - alpha
        survive = oppose & (cand > 0.0)
        flip = oppose & (cand <= 0.0)
        lock[survive] = np.sign(lock[survive]) * cand[survive]
        mask[flip] = target
        lock[flip] = float(sign)

    for want_negative in (True, False):
        for r in refinements:
            if (r.get("type") == "negative") != want_negative:
                continue
            region = focus_region(r.get("splines", []), r["box"], size)
            if not region.any():
                continue
            if not want_negative:
                positive_region |= region
            apply_region(region, -1 if want_negative else +1)
    return mask, lock, positive_region


def _box_min_dim(box) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(1.0, min(abs(x2 - x1), abs(y2 - y1)))


def _morph_radius(box, pct: float) -> int:
    r = int(round(pct / 100.0 * _box_min_dim(box)))
    return int(np.clip(r, 0, POSTPROC_RADIUS_MAX))


def _draw_capsule(mask: np.ndarray, y0, x0, y1, x1, radius: int) -> None:
    H, W = mask.shape
    n = max(int(np.hypot(y1 - y0, x1 - x0)) + 1, 1)
    rr = max(1, int(radius))
    for y, x in zip(np.linspace(y0, y1, n), np.linspace(x0, x1, n)):
        yi, xi = int(round(y)), int(round(x))
        y_lo, y_hi = max(0, yi - rr), min(H, yi + rr + 1)
        x_lo, x_hi = max(0, xi - rr), min(W, xi + rr + 1)
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        mask[y_lo:y_hi, x_lo:x_hi] |= (yy - yi) ** 2 + (xx - xi) ** 2 <= rr * rr


def bridge_positive_islands(mask: np.ndarray, box, positive_region: np.ndarray) -> np.ndarray:
    """Join islands created by this pass's positives to the largest blob with a thin corridor."""
    mask = np.asarray(mask, dtype=bool).copy()
    if not positive_region.any() or not mask.any():
        return mask
    labeled, nlab = ndi.label(mask)
    if nlab <= 1:
        return mask
    areas = ndi.sum(mask, labeled, index=np.arange(1, nlab + 1))
    anchor_label = int(np.argsort(areas)[::-1][0] + 1)
    anchors = labeled == anchor_label
    max_dist = BRIDGE_MAX_DIST_PCT / 100.0 * _box_min_dim(box)
    half_w = max(1, max(3, 2 * _morph_radius(box, POSTPROC_OPEN_PCT) + 3) // 2)

    edge = anchors & ~ndi.binary_erosion(anchors, iterations=1)
    ay, ax = np.nonzero(edge)
    if len(ay) == 0:
        ay, ax = np.nonzero(anchors)
    if len(ay) > 4000:
        step = len(ay) // 4000
        ay, ax = ay[::step], ax[::step]
    tree = cKDTree(np.column_stack([ay, ax]))

    for lab in range(1, nlab + 1):
        if lab == anchor_label:
            continue
        comp = labeled == lab
        if not (comp & positive_region).any():
            continue
        cy, cx = np.nonzero(comp)
        if len(cy) > 2000:
            step = len(cy) // 2000
            cy, cx = cy[::step], cx[::step]
        dists, idxs = tree.query(np.column_stack([cy, cx]), k=1)
        best = int(np.argmin(dists))
        if float(dists[best]) > max_dist:
            continue
        j = int(idxs[best])
        _draw_capsule(mask, int(cy[best]), int(cx[best]), int(ay[j]), int(ax[j]), half_w)
    return mask


# scikit-image >= 0.26 renamed min_size/area_threshold to max_size (inclusive).
_RSH = set(inspect.signature(morphology.remove_small_holes).parameters)
_RSO = set(inspect.signature(morphology.remove_small_objects).parameters)
_OPENING = getattr(morphology, "opening", morphology.binary_opening)
_CLOSING = getattr(morphology, "closing", morphology.binary_closing)


def _remove_small_holes(mask, area_threshold: int):
    if "max_size" in _RSH:
        return morphology.remove_small_holes(mask, max_size=max(0, area_threshold - 1))
    return morphology.remove_small_holes(mask, area_threshold=area_threshold)


def _remove_small_objects(mask, min_size: int):
    if "max_size" in _RSO:
        return morphology.remove_small_objects(mask, max_size=max(0, min_size - 1))
    return morphology.remove_small_objects(mask, min_size=min_size)


def postprocess_mask(mask: np.ndarray, box) -> np.ndarray:
    """Minor closing + opening, then fill small holes and drop small specks."""
    mask = np.asarray(mask, dtype=bool)
    obj = int(mask.sum())
    r_close = max(1, _morph_radius(box, POSTPROC_CLOSE_PCT))
    r_open = max(1, _morph_radius(box, POSTPROC_OPEN_PCT))
    mask = np.asarray(_CLOSING(mask, morphology.disk(r_close)), dtype=bool)
    mask = np.asarray(_OPENING(mask, morphology.disk(r_open)), dtype=bool)
    if obj > 0:
        hole_cap = max(POSTPROC_HOLE_FLOOR, int(round(POSTPROC_HOLE_PCT / 100.0 * obj)))
        mask = np.asarray(_remove_small_holes(mask, hole_cap), dtype=bool)
        speck_cap = max(POSTPROC_SPECK_FLOOR, int(round(POSTPROC_SPECK_PCT / 100.0 * obj)))
        mask = np.asarray(_remove_small_objects(mask, speck_cap), dtype=bool)
    return mask


# --------------------------------------------------------------------------- #
# Overlay                                                                     #
# --------------------------------------------------------------------------- #
def _hue(rgb) -> float:
    r, g, b = (float(c) / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h * 60.0


def _hue_dist(a, b) -> float:
    d = abs(_hue(a) - _hue(b)) % 360.0
    return min(d, 360.0 - d)


def instance_palette(n: int, colors: list, preferred_rgb) -> list:
    """``n`` distinct tints: the preferred one first, then maximin hue distance."""
    pref = tuple(int(v) for v in preferred_rgb)
    chosen = [pref] if pref in colors else []
    remaining = [c for c in colors if c not in chosen]
    if not chosen:
        remaining = sorted(remaining, key=_hue)
        chosen.append(remaining.pop(0))
    while len(chosen) < min(n, len(colors)):
        best = max(remaining, key=lambda c: min(_hue_dist(c, x) for x in chosen))
        remaining.remove(best)
        chosen.append(best)
    return [chosen[i % len(chosen)] for i in range(n)]


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_overlay(image_path, mask, out_path, mask_rgb, boxes=None, ids=None,
                   instance_masks=None, palette_colors=None, mask_out=None) -> None:
    """Translucent tint + red mask contour (+ yellow instance boxes with id chips).

    This is the image the model critiques in stage 3. With several
    ``instance_masks`` each instance gets its own tint from ``palette_colors``.
    Stroke and font scale with the image's long side.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(float)
    scale = max(img.width, img.height) / 700.0
    stroke = max(2, int(round(2 * scale)))

    if instance_masks is not None and len(instance_masks) > 1:
        tints = instance_palette(len(instance_masks), palette_colors, mask_rgb)
        for m, rgb in zip(instance_masks, tints):
            m = np.asarray(m, dtype=bool)
            arr[m] = (1 - MASK_ALPHA) * arr[m] + MASK_ALPHA * np.array(rgb, dtype=float)
        contour_src = union_masks(instance_masks, (img.width, img.height))
    else:
        arr[mask] = (1 - MASK_ALPHA) * arr[mask] + MASK_ALPHA * np.array(mask_rgb, dtype=float)
        contour_src = np.asarray(mask, dtype=bool)
    if contour_src.any():
        edge = contour_src & ~ndi.binary_erosion(contour_src, iterations=stroke)
        arr[edge] = [235, 30, 30]

    overlay = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if boxes:
        draw = ImageDraw.Draw(overlay)
        font = _font(max(12, int(round(14 * scale))))
        pad_x, pad_y = max(4, int(round(6 * scale))), max(2, int(round(4 * scale)))
        tx, ty = max(2, int(round(3 * scale))), max(1, int(round(1 * scale)))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            draw.rectangle([x1, y1, x2, y2], outline=(255, 210, 0), width=stroke)
            label = str(ids[i] if ids is not None else i)
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([x1, y1, x1 + tw + pad_x, y1 + th + pad_y], fill=(255, 210, 0))
            draw.text((x1 + tx, y1 + ty), label, fill=(20, 20, 20), font=font)
    overlay.save(out_path)
    if mask_out is not None:
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(mask_out)
    log(f"[overlay] saved -> {out_path}")
