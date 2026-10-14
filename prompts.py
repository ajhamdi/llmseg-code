"""Prompt templates for the three stages (configuration of the main results).

Templates are ``str.format`` strings: literal JSON braces are doubled. Anything
per-backend (coordinate space, box key and order) is filled in from ``llm``.
"""

# Mask tints stage 2 may choose from (red = mask contour, yellow = detect box,
# so neither is offered).
MASK_COLORS = {
    "lime":         (60, 250, 20),
    "green":        (20, 230, 70),
    "spring_green": (0, 250, 120),
    "mint":         (40, 255, 170),
    "teal":         (0, 210, 180),
    "cyan":         (0, 235, 255),
    "turquoise":    (20, 235, 200),
    "sky_blue":     (50, 185, 255),
    "blue":         (30, 100, 255),
    "periwinkle":   (100, 120, 255),
    "indigo":       (80, 50, 255),
    "violet":       (140, 40, 255),
    "purple":       (175, 40, 255),
    "magenta":      (255, 20, 210),
    "hot_pink":     (255, 40, 160),
    "pink":         (255, 90, 180),
    "coral":        (255, 80, 90),
    "orange":       (255, 120, 20),
    "chartreuse":   (180, 255, 20),
}

# Control-point reference ranges (prompt guidance, not enforced in code).
SEGMENT_CTRL_POINTS_MIN, SEGMENT_CTRL_POINTS_MAX = 16, 64
FOCUS_CTRL_POINTS_MIN, FOCUS_CTRL_POINTS_MAX = 4, 12
FOCUS_BOXES_PER_INSTANCE = 6


# --------------------------------------------------------------------------- #
# Stage 1: detect                                                             #
# --------------------------------------------------------------------------- #
PROMPT_DETECT = """Read the image file at this absolute path: {image_path}

{coord_space}

Find every DISTINCT instance of the following in the image: "{query}"

Return ONLY a JSON object (no prose, no markdown fences) with this exact schema:
{{"detections": [{{
  "id": <int starting at 0>,
  "label": "<short label>",
  "{box_field}": {box_order},
  "confidence": <0..1>,
  "morphology": {{
    "shape": "<overall silhouette in a few words, e.g. 'compact rounded body with two upright protrusions'>",
    "aspect": "<tall|wide|square>",
    "convexity": "<convex|slightly-concave|highly-irregular>",
    "parts": ["<protruding/notable parts, e.g. 'left protrusion'>", "<'base contact'>", "..."]
  }}{topology_schema}
}}]}}

{box_format}
- One entry per DISTINCT instance -- do NOT return duplicate or nested boxes for
  the same object. Order detections in reading order (top-to-bottom, then
  left-to-right); ids will be reassigned by code, but keep them contiguous from 0.
- morphology.aspect must describe the instance as you SEE it, independently of
  the numbers you wrote in {box_field}; it is cross-checked against them.
- Make each box LARGE enough to ENTIRELY encompass that instance -- leave generous
  margin so no part of the silhouette (protrusions, extremities, soft edges,
  head, ears, paws) is clipped by the box. Prefer a roomy box over a tight crop;
  a later stage confines all outline tracing to this box, so an undersized box
  truncates the segmentation.
{form_note}If nothing matches, return {{"detections": []}}.
Output the JSON and nothing else."""


# --------------------------------------------------------------------------- #
# Stage 2: segment (initial per-instance B-splines)                           #
# --------------------------------------------------------------------------- #
PROMPT_SEGMENT = """Read the image file at this absolute path: {image_path}

{coord_space}

We have already DETECTED {n_instances} instance(s) of "{label}" (same query / class).
Each instance has its own bounding box; a later stage confines that instance's
outline to ITS box only.

INSTANCES (trace EVERY one):
{instances_block}

Your job has TWO parts:
  (1) pick ONE shared overlay tint color for the mask (same for all instances),
  (2) CHOOSE a spline degree (1 or 3) shared by every instance of this class, then
      for EACH instance trace its silhouette with piece-wise CLOSED 2D B-splines
      of that degree so the spline(s) tightly SURROUND that instance.

MASK COLOR (part 1):
Pick ONE color name from this fixed list:
  {mask_color_options}
Choose the color that maximizes contrast of a light translucent tint against BOTH
the "{label}" AND its surrounding background, so a later pass can clearly see
what is covered vs not. Rules for the choice:
{color_scope_note}- Prefer a hue that is NOT already dominant in the object or in the nearby
  background.
- Do NOT pick a color that would wash out or blend into the object's own colors
  or hide fine edge detail under the tint.
- Return the name EXACTLY as listed (lowercase, underscores).

SPLINE RULES (part 2):
- Return exactly one entry per instance, keyed by the same integer "id".
- Every control point for instance id=K MUST lie inside THAT instance's
  {box_field}={box_order} listed above.
{point_order_note}
{loops_note}- Use each instance's morphology to place control points on the true outline
  (follow protrusions, concavities, corners, and curves named in the morphology).
- If one instance's box is empty, unreadable, or holds nothing matching the
  "{label}", return an EMPTY "splines" list for that id rather than inventing an
  outline. Still return its id, and still trace every other instance normally.
{output_budget_note}{degree_guidance}

Return ONLY this JSON (no prose, no markdown fences):
{{"mask_color": "<one name from the list>",
  "mask_color_reason": "<short why this hue contrasts with object + background>",
  "degree": <1 or 3>,
  "degree_reason": "<short why this degree fits the object's form>",
  "instances": [
  {{"id": 0, "splines": [
    {{"role": "outer", "degree": <same as top-level degree>, "closed": true,
      "control_points": [[x, y], [x, y], ...]}}
  ]}}
]}}
Output the JSON and nothing else."""


# --------------------------------------------------------------------------- #
# Stage 3-pre (opt-in, --refine-box): one signed per-side detect-box adjustment #
# --------------------------------------------------------------------------- #
PROMPT_BOX_ADJUST = """Read the image file at this absolute path: {overlay_path}

This image shows the CURRENT segmentation of the "{label}":
  - the translucent {mask_color} tint marks the current MASK REGION (coverage),
  - the RED outline is the current mask BOUNDARY,
  - each YELLOW rectangle is one instance's detection box (all later outline work
    for that instance is confined to its box, so anything of that instance outside
    it can NEVER be recovered, and empty room inside wastes the confinement).
    Instance ids are drawn next to each yellow box.
The {mask_color} fill is light on purpose: you can still see the {label}'s real
edges/texture underneath. Use the {mask_color} tint + red outline for coverage;
use the underlying pixels to judge the TRUE extent of each instance.
{coord_space}

INSTANCES (decide side deltas independently for EACH):
{instances_block}

Your task: for EACH instance, decide whether its yellow box is WRONG: too tight
(clips the object), too loose (includes obvious empty background), or shifted
(one side has slack while the opposite side clips). Adjust that box; otherwise
leave it alone.

Return a SIGNED side delta in the SAME coordinate units for each side of EACH
instance:
  - "x1": LEFT edge; positive expands LEFT (decrease left), negative shrinks
    RIGHTWARD (increase left),
  - "x2": RIGHT edge; positive expands RIGHT (increase right), negative shrinks
    LEFTWARD (decrease right),
  - "y1": TOP edge; positive expands UP (decrease top), negative shrinks
    DOWNWARD (increase top),
  - "y2": BOTTOM edge; positive expands DOWN (increase bottom), negative shrinks
    UPWARD (decrease bottom).
Each value may be positive, negative, or zero. Absolute value must not exceed
the per-side cap listed above. Prefer ZERO on every side unless the yellow box
is OBVIOUSLY wrong. Do NOT nudge for soft margins or tiny ambiguity. If a yellow
box already tightly and fully contains its instance, return all zeros for that id.

Hard caps are listed per instance above (same units; |delta| must not exceed).
If a needed move would exceed the cap, move only up to the cap; do not ask for more.

Rules:
- Positive on a side ONLY when object of that instance clearly lies outside that
  edge. Negative on a side ONLY when that edge clearly has empty background slack
  and pulling it in would still leave the object inside.
- You may combine signs across sides (e.g. +x1 and -x2) to shift or resize.
- Leave a small generous margin past the true extent when you do move a side;
  do not hug the silhouette so tightly that a later pass clips it again.
- Default / preferred answer when the box looks fine: all zeros.
- Return exactly one entry per instance id listed above.

Return ONLY this JSON (no prose, no markdown fences):
{{"instances": [
  {{"id": 0, "x1": 0, "x2": 0, "y1": 0, "y2": 0}}
]}}
Output the JSON and nothing else."""


# --------------------------------------------------------------------------- #
# Stage 3a: focus boxes on the current overlay                                #
# --------------------------------------------------------------------------- #
PROMPT_FOCUS = """Read the image file at this absolute path: {overlay_path}

This image shows the CURRENT segmentation of the "{label}":
  - the translucent {mask_color} tint marks the current MASK REGION (coverage),
  - the RED outline is the current mask BOUNDARY -- the main edge to compare,
  - each YELLOW rectangle is one instance's detection box (ids drawn beside them).
The {mask_color} fill is light on purpose: you can still see the {label}'s real
edges/texture underneath. Use the {mask_color} tint to tell what is already
covered; use the red outline + the underlying pixels to judge where the TRUE
edge runs. Do NOT treat the soft tinted edge as the object boundary.
{coord_space}

DETECTION BOXES (each focus box MUST lie inside ONE of these):
{instances_block}

Your task is a COMPARISON: follow the RED outline and, at each part of it, look
through the {mask_color} tint at the actual object pixels to see where the TRUE
edge of the {label} is. Report the places where the two disagree.

Find FOCUS BOXES that mark where the mask is wrong and needs local fixing:
  - type "positive": a box that FULLY encloses an area that IS part of the
    {label} but is NOT tinted / lies OUTSIDE the red outline (MISSED, e.g. a
    protruding part or tip left outside the mask) -> to be ADDED.
  - type "negative": a box that FULLY encloses an area that IS tinted / INSIDE
    the red outline but is actually background (wrongly included, e.g.
    background bleeding beside the silhouette, or a gap that is not the
    {label}) -> to be REMOVED.
Check BOTH error directions along the red outline: missed object (positive)
AND tinted background (negative). Do not only hunt for misses.

Rules:
- Tag every focus box with "instance_id" naming which detection box it belongs to.
- Each focus box {box_order} MUST lie inside that instance's detection box.
- Make each box LARGE enough to ENTIRELY encompass the region to be fixed --
  leave generous margin so no part of the miss / over-include is clipped by the
  box edge. Prefer one roomy box over a tight crop; undersized boxes cause the
  next-step spline to miss the true outline. Propose 1-{focus_boxes_per_instance} boxes
  PER instance (up to {max_focus_boxes} total), one per distinct problem area.
{focus_spread_note}- Anchor each box on what you can actually SEE: it should cover the full gap
  between the red outline (or the tinted coverage) and the true edge of the
  {label} at that spot, plus margin. Place box edges on real image features
  (the object's boundary, a protrusion, the adjacent background), not on the
  soft tint alone.
- A pass may be ALL positive, ALL negative, or a mix -- there is NO requirement to
  include both types. Return only boxes that are genuinely wrong; never add a box
  of the other type just to have one of each.
- Only flag areas that are CLEARLY wrong in the current image; if a spot already
  looks correct, do not propose a box there (earlier fixes already show in the tint).
- For each box, describe the SHAPE of the region to be added/removed in a few
  words, and how intricate its outline is (e.g. "thin curved sliver", "small
  round bump", "jagged notch", "simple triangular wedge"). A later step uses this
  to decide how many control points to spend on the box, so be honest about
  whether the local edge is simple or wiggly.

Return ONLY this JSON (no prose, no markdown fences). Example shape (mix of
types; your list may be all-positive, all-negative, or mixed):
{{"focus_boxes": [
  {{"type": "positive",
    "instance_id": 0,
    "shape": "<shape + complexity of the region to add>",
    "{box_field}": {box_order}}},
  {{"type": "negative",
    "instance_id": 0,
    "shape": "<shape + complexity of the region to remove>",
    "{box_field}": {box_order}}}
]}}
Output the JSON and nothing else."""


# --------------------------------------------------------------------------- #
# Stage 3b: one batched call for the B-splines of every surviving focus box   #
# --------------------------------------------------------------------------- #
PROMPT_FOCUS_SPLINE = """Read the image file at this absolute path: {overlay_path}

{coord_space}
The translucent {mask_color} tint is the current mask of the "{label}"; the RED
outline is its boundary. The tint is light enough that the object's real edges
and texture stay visible -- use the {mask_color} tint to see what is already
covered, and the red outline + underlying pixels to find the TRUE edge. Do NOT
treat the soft tinted edge as the object boundary.

You are given {n_boxes} FOCUS BOX(ES). For EACH box, trace the OUTLINE of the
region that must be fixed with CLOSED degree-{degree} 2D B-spline(s). Control
points for a box MUST stay inside THAT box. Each box lists the SHAPE of the
region to fix -- use it to judge how many control points that box needs.
All boxes and control points use the coordinate space described above.

FOCUS BOXES:
{focus_boxes_block}

Per-box action:
- type "positive": outline the part of the {label} inside the box that is NOT
  tinted / lies OUTSIDE the red outline (missed) — this region will be ADDED.
- type "negative": outline the area inside the box that IS tinted / INSIDE the
  red outline but is actually background / not the {label} — it will be REMOVED.

Rules:
- Return exactly one entry per focus box, keyed by the same integer "id".
- Every control point for box id=K MUST lie inside that box's {box_order}.
{point_order_note}- Fit as many closed loops as the region needs (usually 1).
- If a box has nothing clear to outline, return an empty "splines" list for it.
{output_budget_note}{degree_guidance}

Return ONLY this JSON (no prose, no markdown fences):
{{"refinements": [
  {{"id": 0, "splines": [
    {{"degree": {degree}, "closed": true, "control_points": [[x, y], [x, y], ...]}}
  ]}}
]}}
Output the JSON and nothing else."""


# --------------------------------------------------------------------------- #
# Prompt fragments                                                            #
# --------------------------------------------------------------------------- #
def mask_color_options() -> str:
    return ", ".join(f'"{k}"' for k in MASK_COLORS)


def segment_degree_guidance() -> str:
    """Stage 2: the model chooses degree 1 or 3, then places points accordingly."""
    return (
        "CHOOSE THE SPLINE DEGREE (1 or 3) from the object's form, then place the\n"
        "control points the way that degree requires:\n"
        "- Pick DEGREE 1 (closed polyline) for angular / polygonal / straight-edged\n"
        "  objects, or simple silhouettes where a polygon is enough. Its control\n"
        "  points are vertices ON the true object outline (not a looser control\n"
        "  polygon): place them directly on the edge, ordered around the loop, denser\n"
        "  at corners/bends and sparser on long straight spans.\n"
        "- Pick DEGREE 3 (closed cubic B-spline) for organic / rounded / smoothly\n"
        "  curved silhouettes. Its control points are\n"
        "  the B-spline CONTROL POLYGON (the curve is smoother than and sits inside the\n"
        "  polygon), so push them slightly OUTWARD toward the object edge, ordered\n"
        "  around the loop.\n"
        "- Use the SAME degree for EVERY spline (outer loops and holes). Report it in\n"
        '  the top-level "degree" field AND set "degree" on each spline to that value.\n'
        f"- Choose the NUMBER of control points from the object's morphology and what\n"
        f"  you actually see inside the box -- typically {SEGMENT_CTRL_POINTS_MIN}-"
        f"{SEGMENT_CTRL_POINTS_MAX} per loop. Use MORE points for intricate /\n"
        f"  irregular / many-parted outlines (closer to {SEGMENT_CTRL_POINTS_MAX}) and\n"
        f"  FEWER for simple, smooth, or rounded ones (near {SEGMENT_CTRL_POINTS_MIN}).\n"
        f"  Do not aim for a fixed count -- let the shape's complexity decide within\n"
        f"  that range; under-sampling a complex outline is worse than a few extra\n"
        f"  points on a simple one."
    )


def focus_degree_guidance() -> str:
    """Stage 3b: refinement loops are always cubic, with a small local point budget."""
    return (
        "DEGREE = 3 (closed cubic B-spline):\n"
        "- Control points are the B-spline CONTROL POLYGON (the curve is smoother\n"
        "  than and sits inside the polygon), so push them slightly OUTWARD toward\n"
        "  the object edge.\n"
        "- Order them sequentially around the loop.\n"
        "- Set \"degree\": 3 on EVERY spline in the response.\n"
        f"- A focus box is a SMALL local patch, so use FEWER control points than a\n"
        f"  whole-object outline -- typically {FOCUS_CTRL_POINTS_MIN}-{FOCUS_CTRL_POINTS_MAX}\n"
        f"  per loop. Use each box's reported SHAPE to decide: near\n"
        f"  {FOCUS_CTRL_POINTS_MIN} for a simple bump/notch/sliver, closer to\n"
        f"  {FOCUS_CTRL_POINTS_MAX} only for a genuinely wiggly local edge."
    )


def topology_schema(topology: bool) -> str:
    """Stage 1: the optional topology block of each detection (--topology)."""
    if not topology:
        return ""
    return (
        ',\n  "topology": {\n'
        '    "components": <int number of disconnected pieces of the object silhouette>,\n'
        '    "holes": <int number of enclosed background holes inside the silhouette (genus)>,\n'
        '    "boundary_loops": <int total closed boundary curves = components + holes>,\n'
        '    "notes": "<one short phrase, e.g. \'single blob, no holes\'>"\n'
        '  }'
    )


def form_note(topology: bool) -> str:
    """Stage 1: what the morphology (and, with --topology, topology) fields are for."""
    if not topology:
        return (
            "- morphology describes the object's FORM (shape of its outline) so a later\n"
            "  stage can trace it.\n"
        )
    return (
        "- morphology describes the object's FORM (shape of its outline) so a later stage\n"
        "  can trace it; topology describes its STRUCTURE (how many separate blobs and how\n"
        "  many holes) so a later stage knows how many closed curves to fit.\n"
    )


def loops_note(topology: bool) -> str:
    """Stage 2: how many loops per instance (counted from the image, or from topology)."""
    if not topology:
        return (
            "- Fit as many closed B-splines as that instance's silhouette actually needs:\n"
            "  one per connected piece you can see (role \"outer\"), plus one per interior\n"
            "  hole you can see through (role \"hole\"). Decide the count from the image.\n"
        )
    return (
        "- Fit ONE closed B-spline per connected blob of that instance\n"
        "  (topology.components), plus ONE additional closed B-spline per interior hole\n"
        "  (topology.holes), marking holes with \"role\":\"hole\". Outer silhouette loops use\n"
        "  \"role\":\"outer\".\n"
    )


def color_scope_note(n_instances: int) -> str:
    if int(n_instances) <= 1:
        return ""
    return (
        "- ONE tint is shared by every instance: pick a hue that reads clearly on ALL\n"
        "  of them and against each one's own local background, not just the first.\n"
    )


def focus_spread_note(n_instances: int) -> str:
    if int(n_instances) <= 1:
        return ""
    return (
        f"- Check EVERY one of the {int(n_instances)} instances listed above, not only the\n"
        f"  largest or the most obviously wrong one. Propose up to\n"
        f"  {FOCUS_BOXES_PER_INSTANCE} boxes PER instance (do not dump the whole budget on one).\n"
    )


def output_budget_note(n_items: int, what: str) -> str:
    """Batched replies are long: pin ascending-id order so a truncation loses the tail."""
    n = int(n_items)
    if n <= 1:
        return ""
    return (
        f"- Your whole reply must fit in ONE response, and {n} {what} is a lot of\n"
        f"  coordinates. Emit entries in ASCENDING id order and FINISH each entry before\n"
        f"  starting the next; stay near the LOW end of the control-point range for the\n"
        f"  simple ones so that nothing gets cut off.\n"
    )
