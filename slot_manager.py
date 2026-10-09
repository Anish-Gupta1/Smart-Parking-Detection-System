"""
Parking Slot Manager
- Persistent layout storage (JSON files)
- IoU-based vehicle-to-slot matching
- Three-state occupancy classification: free / occupied / uncertain
"""

import json
import os
import hashlib
import numpy as np

# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden per-call)
# ---------------------------------------------------------------------------
DEFAULT_OCCUPIED_IOU = 0.15       # IoU >= this → occupied
DEFAULT_UNCERTAIN_IOU = 0.05      # IoU >= this but < occupied → uncertain
LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "layouts")

# Ensure layouts directory exists
os.makedirs(LAYOUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------
def _layout_path(layout_name: str) -> str:
    """Return the path to a named layout JSON file."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in layout_name)
    return os.path.join(LAYOUTS_DIR, f"{safe}.json")


def save_layout(layout_name: str, slots: list[dict]) -> str:
    """
    Save a list of slot dicts to a JSON file.

    Each slot dict:
        {
            "id": "P1",
            "x_norm": float,   # 0-1, relative to image width
            "y_norm": float,
            "w_norm": float,
            "h_norm": float,
        }

    Returns the path written.
    """
    path = _layout_path(layout_name)
    with open(path, "w") as f:
        json.dump({"slots": slots}, f, indent=2)
    return path


def load_layout(layout_name: str) -> list[dict] | None:
    """Load saved slots for a layout.  Returns None if not found."""
    path = _layout_path(layout_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("slots", [])


def delete_layout(layout_name: str) -> bool:
    """Delete a saved layout.  Returns True if file existed."""
    path = _layout_path(layout_name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_layouts() -> list[str]:
    """Return names of all saved layouts."""
    layouts = []
    for f in os.listdir(LAYOUTS_DIR):
        if f.endswith(".json"):
            layouts.append(f[:-5])
    return sorted(layouts)


def canvas_to_normalized_slots(canvas_result, canvas_w: int, canvas_h: int, img_w: int, img_h: int) -> list[dict]:
    """
    Extract rectangle objects from the drawable canvas and convert
    to normalized slot dicts. Supports Fabric.js v5 ('rect') and v6 ('Rect'),
    as well as labeled_rect or any bounding-box object.
    """
    slots = []
    if canvas_result is None:
        return slots

    json_data = getattr(canvas_result, "json_data", None)
    if json_data is None and isinstance(canvas_result, dict):
        json_data = canvas_result
    if not json_data:
        return slots

    objects = json_data.get("objects", [])
    scale_x = img_w / canvas_w
    scale_y = img_h / canvas_h

    idx = 1
    for obj in objects:
        obj_type = str(obj.get("type", "")).lower()
        # Accept 'rect', 'labeledrect', or any shape with bounding box width/height
        if "rect" in obj_type or ("width" in obj and "height" in obj):
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
            height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

            # Handle negative width/height if dragged right-to-left or bottom-to-top
            if width < 0:
                left += width
                width = abs(width)
            if height < 0:
                top += height
                height = abs(height)

            # Ignore tiny clicks/accidental dots (< 5px)
            if width < 5 or height < 5:
                continue

            x_px = left * scale_x
            y_px = top * scale_y
            w_px = width * scale_x
            h_px = height * scale_y

            # Clamp normalized coordinates to [0.0, 1.0]
            x_norm = max(0.0, min(1.0, x_px / img_w))
            y_norm = max(0.0, min(1.0, y_px / img_h))
            w_norm = max(0.0, min(1.0 - x_norm, w_px / img_w))
            h_norm = max(0.0, min(1.0 - y_norm, h_px / img_h))

            slots.append({
                "id": f"P{idx}",
                "x_norm": round(x_norm, 4),
                "y_norm": round(y_norm, 4),
                "w_norm": round(w_norm, 4),
                "h_norm": round(h_norm, 4),
            })
            idx += 1

    return slots


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def slots_to_pixels(slots: list[dict], img_w: int, img_h: int) -> list[dict]:
    """Convert normalised slot coords to pixel coords, adding pixel fields."""
    out = []
    for s in slots:
        px = {
            "id": s["id"],
            "x": int(s["x_norm"] * img_w),
            "y": int(s["y_norm"] * img_h),
            "w": int(s["w_norm"] * img_w),
            "h": int(s["h_norm"] * img_h),
            # keep normalised too
            "x_norm": s["x_norm"],
            "y_norm": s["y_norm"],
            "w_norm": s["w_norm"],
            "h_norm": s["h_norm"],
        }
        out.append(px)
    return out


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------
def _iou(boxA: tuple, boxB: tuple) -> float:
    """
    Compute Intersection-over-Union for two (x1, y1, x2, y2) boxes.
    Returns 0.0 – 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def _overlap_ratio(slot_box: tuple, vehicle_box: tuple) -> float:
    """
    Compute what fraction of the *slot* area is covered by the vehicle box.
    This is more useful than IoU for small slots under large vehicles.
    """
    xA = max(slot_box[0], vehicle_box[0])
    yA = max(slot_box[1], vehicle_box[1])
    xB = min(slot_box[2], vehicle_box[2])
    yB = min(slot_box[3], vehicle_box[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    slot_area = (slot_box[2] - slot_box[0]) * (slot_box[3] - slot_box[1])
    return inter / slot_area if slot_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Occupancy classification
# ---------------------------------------------------------------------------
def classify_slots(
    slots_px: list[dict],
    vehicles: list[tuple],
    vehicle_confs: list[float] | None = None,
    occupied_threshold: float = DEFAULT_OCCUPIED_IOU,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_IOU,
) -> list[dict]:
    """
    Classify each slot as 'free', 'occupied', or 'uncertain'.

    Parameters
    ----------
    slots_px : list of dicts with pixel keys  (x, y, w, h, id)
    vehicles : list of (x1, y1, x2, y2) vehicle bounding boxes
    vehicle_confs : optional list of confidence scores parallel to vehicles
    occupied_threshold : minimum overlap ratio to declare occupied
    uncertain_threshold : minimum overlap ratio to declare uncertain

    Returns
    -------
    list of dicts: each slot dict augmented with:
        "status" : "free" | "occupied" | "uncertain"
        "best_overlap" : float
        "best_vehicle_conf" : float or None
    """
    if vehicle_confs is None:
        vehicle_confs = [1.0] * len(vehicles)

    # Track which vehicles have already been matched to a slot
    # (prevents one vehicle from occupying multiple adjacent slots)
    vehicle_assigned = [False] * len(vehicles)

    # First pass: compute best overlap for each slot
    slot_results = []
    for s in slots_px:
        slot_box = (s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"])
        best_overlap = 0.0
        best_iou = 0.0
        best_conf = 0.0
        best_v_idx = -1

        for v_idx, (v_box, v_conf) in enumerate(zip(vehicles, vehicle_confs)):
            iou = _iou(slot_box, v_box)
            overlap = _overlap_ratio(slot_box, v_box)
            # Use the max of IoU and overlap_ratio for matching
            score = max(iou, overlap)

            if score > best_overlap:
                best_overlap = score
                best_iou = iou
                best_conf = v_conf
                best_v_idx = v_idx

        slot_results.append({
            **s,
            "best_overlap": best_overlap,
            "best_iou": best_iou,
            "best_vehicle_conf": best_conf,
            "best_v_idx": best_v_idx,
        })

    # Sort by best_overlap descending so the strongest matches get assigned first
    indexed = list(enumerate(slot_results))
    indexed.sort(key=lambda x: x[1]["best_overlap"], reverse=True)

    for orig_idx, sr in indexed:
        v_idx = sr["best_v_idx"]
        overlap = sr["best_overlap"]

        if overlap >= occupied_threshold and v_idx >= 0 and not vehicle_assigned[v_idx]:
            sr["status"] = "occupied"
            vehicle_assigned[v_idx] = True
        elif overlap >= uncertain_threshold:
            # Could be occupied but the evidence is weak, or vehicle already assigned
            if v_idx >= 0 and vehicle_assigned[v_idx]:
                # Vehicle already matched to a better slot
                sr["status"] = "uncertain"
            else:
                sr["status"] = "uncertain"
        else:
            sr["status"] = "free"

    # Restore original order
    result = [None] * len(slot_results)
    for orig_idx, sr in indexed:
        # Remove internal fields
        sr.pop("best_v_idx", None)
        sr.pop("best_iou", None)
        result[orig_idx] = sr

    return result


def count_statuses(classified: list[dict]) -> dict:
    """Return a summary dict of status counts."""
    counts = {"total": len(classified), "free": 0, "occupied": 0, "uncertain": 0}
    for s in classified:
        status = s.get("status", "uncertain")
        if status in counts:
            counts[status] += 1
    return counts
