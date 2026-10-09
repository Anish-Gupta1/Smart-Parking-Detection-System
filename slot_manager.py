"""
Parking Slot Manager
- Persistent layout storage (JSON files)
- Advanced geometric slot-to-vehicle overlap analysis (slot overlap, vehicle overlap, IoU)
- Exclusive vehicle-to-slot matching preventing false adjacent-slot occupancy
- Secondary slot-crop visual analysis for vehicles missed by YOLO
- Three-state occupancy classification: free / occupied / uncertain
"""

import json
import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_OCCUPIED_OVERLAP = 0.15   # Overlap ratio >= this → occupied
DEFAULT_UNCERTAIN_OVERLAP = 0.05  # Overlap ratio >= this but < occupied → uncertain
LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "layouts")

os.makedirs(LAYOUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------
def _layout_path(layout_name: str) -> str:
    """Return the path to a named layout JSON file."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in layout_name)
    return os.path.join(LAYOUTS_DIR, f"{safe}.json")


def save_layout(layout_name: str, slots: list[dict]) -> str:
    """Save a list of slot dicts to a JSON file."""
    path = _layout_path(layout_name)
    with open(path, "w") as f:
        json.dump({"slots": slots}, f, indent=2)
    return path


def load_layout(layout_name: str) -> list[dict] | None:
    """Load saved slots for a layout. Returns None if not found."""
    path = _layout_path(layout_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("slots", [])


def delete_layout(layout_name: str) -> bool:
    """Delete a saved layout. Returns True if file existed."""
    path = _layout_path(layout_name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_layouts() -> list[str]:
    """Return names of all saved layouts."""
    layouts = []
    if os.path.exists(LAYOUTS_DIR):
        for f in os.listdir(LAYOUTS_DIR):
            if f.endswith(".json"):
                layouts.append(f[:-5])
    return sorted(layouts)


# ---------------------------------------------------------------------------
# Canvas to Normalized Slots
# ---------------------------------------------------------------------------
def canvas_to_normalized_slots(
    canvas_result, canvas_w: int, canvas_h: int, img_w: int, img_h: int
) -> list[dict]:
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
        if "rect" in obj_type or ("width" in obj and "height" in obj):
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
            height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

            if width < 0:
                left += width
                width = abs(width)
            if height < 0:
                top += height
                height = abs(height)

            if width < 5 or height < 5:
                continue

            x_px = left * scale_x
            y_px = top * scale_y
            w_px = width * scale_x
            h_px = height * scale_y

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
            "x_norm": s["x_norm"],
            "y_norm": s["y_norm"],
            "w_norm": s["w_norm"],
            "h_norm": s["h_norm"],
        }
        out.append(px)
    return out


# ---------------------------------------------------------------------------
# Geometric overlap calculations
# ---------------------------------------------------------------------------
def compute_overlap_metrics(slot_box: tuple, vehicle_box: tuple) -> dict:
    """
    Calculate comprehensive overlap metrics between a slot and vehicle:
      - slot_overlap_ratio: intersection_area / slot_area
      - vehicle_overlap_ratio: intersection_area / vehicle_area
      - iou: intersection_area / union_area
      - center_inside: whether vehicle center is inside the slot
    """
    s_x1, s_y1, s_x2, s_y2 = slot_box
    v_x1, v_y1, v_x2, v_y2 = vehicle_box

    inter_x1 = max(s_x1, v_x1)
    inter_y1 = max(s_y1, v_y1)
    inter_x2 = min(s_x2, v_x2)
    inter_y2 = min(s_y2, v_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)

    slot_area = float(max(1, (s_x2 - s_x1) * (s_y2 - s_y1)))
    veh_area = float(max(1, (v_x2 - v_x1) * (v_y2 - v_y1)))
    union_area = float(slot_area + veh_area - inter_area)

    slot_overlap_ratio = inter_area / slot_area
    vehicle_overlap_ratio = inter_area / veh_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    # Center check
    v_cx = (v_x1 + v_x2) / 2.0
    v_cy = (v_y1 + v_y2) / 2.0
    center_inside = (s_x1 <= v_cx <= s_x2) and (s_y1 <= v_cy <= s_y2)

    return {
        "inter_area": inter_area,
        "slot_overlap_ratio": slot_overlap_ratio,
        "vehicle_overlap_ratio": vehicle_overlap_ratio,
        "iou": iou,
        "center_inside": center_inside,
    }


def _iou(boxA: tuple, boxB: tuple) -> float:
    """Legacy helper for IoU."""
    m = compute_overlap_metrics(boxA, boxB)
    return m["iou"]


def _overlap_ratio(slot_box: tuple, vehicle_box: tuple) -> float:
    """Legacy helper for slot overlap ratio."""
    m = compute_overlap_metrics(slot_box, vehicle_box)
    return m["slot_overlap_ratio"]


# ---------------------------------------------------------------------------
# Secondary Slot-Crop Occupancy Classifier
# ---------------------------------------------------------------------------
def analyze_slot_crop(crop: np.ndarray) -> dict:
    """
    Secondary occupancy classifier analyzing the image crop of a configured slot.
    Evaluates edge complexity, gradient distribution, and chromatic saturation.
    Prevents false-free classifications when YOLO misses a vehicle.
    """
    if crop is None or crop.size < 100:
        return {"occupied_score": 0.0, "is_occupied": False, "reason": "empty_crop"}

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Edge complexity using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        edge_mean = float(np.mean(grad_mag))

        # 2. Structural texture variance (Laplacian)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. Color saturation and value variance
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))

        # Normalized feature components
        edge_score = min(1.0, edge_mean / 40.0)
        lap_score = min(1.0, lap_var / 120.0)
        color_score = min(1.0, (sat_mean / 25.0) + (val_std / 50.0))

        # Combined occupancy score [0.0, 1.0]
        occupied_score = 0.40 * edge_score + 0.30 * lap_score + 0.30 * color_score
        is_occupied = occupied_score >= 0.65

        return {
            "occupied_score": round(occupied_score, 3),
            "is_occupied": is_occupied,
            "edge_mean": round(edge_mean, 1),
            "lap_var": round(lap_var, 1),
            "sat_mean": round(sat_mean, 1),
        }
    except Exception:
        return {"occupied_score": 0.0, "is_occupied": False, "reason": "analysis_error"}


# ---------------------------------------------------------------------------
# Occupancy classification engine
# ---------------------------------------------------------------------------
def classify_slots(
    slots_px: list[dict],
    vehicles: list[tuple],
    vehicle_confs: list[float] | None = None,
    frame: np.ndarray | None = None,
    occupied_threshold: float = DEFAULT_OCCUPIED_OVERLAP,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_OVERLAP,
    enable_secondary: bool = True,
) -> list[dict]:
    """
    Classify each parking slot as 'free', 'occupied', or 'uncertain'.

    Parameters
    ----------
    slots_px : list of slot dicts with pixel coords (x, y, w, h, id)
    vehicles : list of (x1, y1, x2, y2) vehicle bounding boxes
    vehicle_confs : list of confidence scores for each vehicle
    frame : optional full-resolution BGR frame for secondary crop classification
    occupied_threshold : minimum overlap ratio to declare occupied
    uncertain_threshold : minimum overlap ratio to declare uncertain
    enable_secondary : whether to use secondary crop classification when YOLO misses

    Returns
    -------
    list of dicts: augmented slot dicts with:
        "status" : "free" | "occupied" | "uncertain"
        "best_overlap" : float (slot overlap ratio)
        "best_vehicle_conf" : float or None
        "evidence" : str explanation
    """
    if vehicle_confs is None:
        vehicle_confs = [1.0] * len(vehicles)

    num_slots = len(slots_px)
    num_vehicles = len(vehicles)

    # 1. Compute pairwise metrics between all slots and vehicles
    # metrics_matrix[s_idx][v_idx]
    metrics_matrix = []
    for s in slots_px:
        s_box = (s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"])
        row = []
        for v_box in vehicles:
            row.append(compute_overlap_metrics(s_box, v_box))
        metrics_matrix.append(row)

    # 2. Optimal Vehicle-to-Slot Matching
    # Each vehicle should primarily occupy its best matching slot.
    # To handle genuine double-parking, a vehicle can occupy multiple slots
    # ONLY if its vehicle_overlap_ratio on both slots is >= 0.35.
    vehicle_primary_slot: dict[int, int] = {}
    DOUBLE_PARK_VEH_THRESHOLD = 0.35

    for v_idx in range(num_vehicles):
        best_s = -1
        best_metric_val = -1.0
        for s_idx in range(num_slots):
            m = metrics_matrix[s_idx][v_idx]
            # Primary criterion: slot_overlap_ratio + IoU
            score = m["slot_overlap_ratio"] + (0.5 * m["iou"])
            if score > best_metric_val:
                best_metric_val = score
                best_s = s_idx
        if best_metric_val > 0.05 and best_s != -1:
            vehicle_primary_slot[v_idx] = best_s

    # 3. Classify each slot
    classified_results = []
    frame_h, frame_w = (frame.shape[:2] if frame is not None else (0, 0))

    for s_idx, s in enumerate(slots_px):
        best_overlap = 0.0
        best_v_overlap = 0.0
        best_iou = 0.0
        best_conf = 0.0
        best_v_idx = -1
        primary_match = False

        for v_idx in range(num_vehicles):
            m = metrics_matrix[s_idx][v_idx]
            overlap = m["slot_overlap_ratio"]
            v_overlap = m["vehicle_overlap_ratio"]
            conf = vehicle_confs[v_idx]

            is_primary = (vehicle_primary_slot.get(v_idx) == s_idx)
            is_genuine_double = (v_overlap >= DOUBLE_PARK_VEH_THRESHOLD)

            # If vehicle belongs to another slot and merely clipped this slot border,
            # do not let it falsely occupy this slot.
            if not is_primary and not is_genuine_double:
                continue

            if overlap > best_overlap:
                best_overlap = overlap
                best_v_overlap = v_overlap
                best_iou = m["iou"]
                best_conf = conf
                best_v_idx = v_idx
                primary_match = is_primary

        status = "free"
        evidence = "No vehicle detected"

        # Check YOLO overlap evidence
        if best_overlap >= occupied_threshold and best_v_idx >= 0:
            status = "occupied"
            evidence = f"YOLO vehicle match (overlap: {best_overlap:.1%}, conf: {best_conf:.0%})"
        elif best_overlap >= uncertain_threshold and best_v_idx >= 0:
            status = "uncertain"
            evidence = f"Borderline overlap ({best_overlap:.1%})"
        else:
            # 4. Secondary Crop Classifier (when YOLO didn't detect an overlapping vehicle)
            if frame is not None and enable_secondary:
                sx = max(0, min(frame_w - 1, s["x"]))
                sy = max(0, min(frame_h - 1, s["y"]))
                sw = max(1, min(frame_w - sx, s["w"]))
                sh = max(1, min(frame_h - sy, s["h"]))
                crop = frame[sy:sy + sh, sx:sx + sw]

                crop_analysis = analyze_slot_crop(crop)
                c_score = crop_analysis.get("occupied_score", 0.0)

                if c_score >= 0.70:
                    status = "occupied"
                    evidence = f"Secondary classifier: vehicle texture detected (score: {c_score:.2f})"
                elif c_score >= 0.52:
                    status = "uncertain"
                    evidence = f"Secondary classifier: ambiguous pattern (score: {c_score:.2f})"
                else:
                    status = "free"
                    evidence = "Confirmed empty pavement"
            else:
                status = "free"
                evidence = "No vehicle overlap"

        classified_results.append({
            **s,
            "status": status,
            "best_overlap": best_overlap,
            "veh_overlap": best_v_overlap,
            "best_vehicle_conf": best_conf if best_v_idx >= 0 else None,
            "evidence": evidence,
        })

    return classified_results


def count_statuses(classified: list[dict]) -> dict:
    """
    Return a summary dict of status counts.
    Strict invariant: total = free + occupied + uncertain.
    """
    counts = {"total": len(classified), "free": 0, "occupied": 0, "uncertain": 0}
    for s in classified:
        status = s.get("status", "uncertain")
        if status in ("free", "occupied", "uncertain"):
            counts[status] += 1
        else:
            counts["uncertain"] += 1
    return counts
