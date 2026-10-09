"""
Tests for slot_manager module:
  - IoU computation
  - Overlap ratio
  - Slot classification logic
  - Layout persistence
  - Count validation (total = free + occupied + uncertain)
"""
import os
import sys
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slot_manager import (
    _iou, _overlap_ratio, classify_slots, count_statuses,
    save_layout, load_layout, delete_layout, slots_to_pixels,
)


def test_iou_no_overlap():
    """Two non-overlapping boxes → IoU = 0"""
    boxA = (0, 0, 10, 10)
    boxB = (20, 20, 30, 30)
    assert _iou(boxA, boxB) == 0.0, f"Expected 0, got {_iou(boxA, boxB)}"
    print("✅ test_iou_no_overlap passed")


def test_iou_perfect_overlap():
    """Identical boxes → IoU = 1"""
    boxA = (10, 10, 50, 50)
    assert _iou(boxA, boxA) == 1.0, f"Expected 1.0, got {_iou(boxA, boxA)}"
    print("✅ test_iou_perfect_overlap passed")


def test_iou_partial():
    """50% overlap area"""
    boxA = (0, 0, 10, 10)  # area = 100
    boxB = (5, 0, 15, 10)  # area = 100, intersection = 50
    iou = _iou(boxA, boxB)
    expected = 50 / 150  # ~0.333
    assert abs(iou - expected) < 0.01, f"Expected ~{expected:.3f}, got {iou:.3f}"
    print("✅ test_iou_partial passed")


def test_overlap_ratio():
    """Slot fully inside vehicle → overlap_ratio = 1.0"""
    slot = (10, 10, 20, 20)   # 10x10 slot
    vehicle = (5, 5, 25, 25)  # 20x20 vehicle covering it
    ratio = _overlap_ratio(slot, vehicle)
    assert ratio == 1.0, f"Expected 1.0, got {ratio}"
    print("✅ test_overlap_ratio passed")


def test_classify_occupied():
    """Vehicle on top of a slot → occupied"""
    slots_px = [{"id": "P1", "x": 10, "y": 10, "w": 40, "h": 40}]
    vehicles = [(15, 15, 45, 45)]
    confs = [0.9]

    result = classify_slots(slots_px, vehicles, confs)
    assert result[0]["status"] == "occupied", f"Got {result[0]['status']}"
    print("✅ test_classify_occupied passed")


def test_classify_free():
    """No vehicles near slot → free"""
    slots_px = [{"id": "P1", "x": 10, "y": 10, "w": 40, "h": 40}]
    vehicles = [(200, 200, 250, 250)]  # far away
    confs = [0.9]

    result = classify_slots(slots_px, vehicles, confs)
    assert result[0]["status"] == "free", f"Got {result[0]['status']}"
    print("✅ test_classify_free passed")


def test_classify_no_vehicles():
    """No vehicles at all → all free"""
    slots_px = [
        {"id": "P1", "x": 0, "y": 0, "w": 30, "h": 30},
        {"id": "P2", "x": 50, "y": 50, "w": 30, "h": 30},
    ]
    result = classify_slots(slots_px, [], [])
    for s in result:
        assert s["status"] == "free", f"{s['id']} got {s['status']}"
    print("✅ test_classify_no_vehicles passed")


def test_single_vehicle_one_slot():
    """A vehicle should only occupy ONE slot, not adjacent ones."""
    slots_px = [
        {"id": "P1", "x": 10, "y": 10, "w": 40, "h": 40},
        {"id": "P2", "x": 45, "y": 10, "w": 40, "h": 40},  # adjacent
    ]
    vehicles = [(12, 12, 48, 48)]  # mostly overlaps P1, slightly touches P2
    confs = [0.9]

    result = classify_slots(slots_px, vehicles, confs)
    occupied_count = sum(1 for s in result if s["status"] == "occupied")
    assert occupied_count == 1, f"Expected 1 occupied, got {occupied_count}"
    assert result[0]["status"] == "occupied", f"P1 should be occupied"
    print("✅ test_single_vehicle_one_slot passed")


def test_count_validation():
    """total = free + occupied + uncertain"""
    slots_px = [
        {"id": "P1", "x": 10, "y": 10, "w": 30, "h": 30},
        {"id": "P2", "x": 50, "y": 10, "w": 30, "h": 30},
        {"id": "P3", "x": 100, "y": 10, "w": 30, "h": 30},
    ]
    vehicles = [(15, 15, 35, 35)]
    confs = [0.8]

    result = classify_slots(slots_px, vehicles, confs)
    counts = count_statuses(result)

    total = counts["free"] + counts["occupied"] + counts["uncertain"]
    assert total == counts["total"], \
        f"Validation failed: {counts['free']}+{counts['occupied']}+{counts['uncertain']} != {counts['total']}"
    print("✅ test_count_validation passed")


def test_layout_persistence():
    """Save, load, delete layout"""
    test_name = "_test_layout_tmp"
    slots = [
        {"id": "P1", "x_norm": 0.1, "y_norm": 0.2, "w_norm": 0.1, "h_norm": 0.15},
        {"id": "P2", "x_norm": 0.3, "y_norm": 0.2, "w_norm": 0.1, "h_norm": 0.15},
    ]

    # Save
    path = save_layout(test_name, slots)
    assert os.path.exists(path), "Layout file not created"

    # Load
    loaded = load_layout(test_name)
    assert loaded is not None, "Layout not loaded"
    assert len(loaded) == 2, f"Expected 2 slots, got {len(loaded)}"
    assert loaded[0]["id"] == "P1"

    # Delete
    deleted = delete_layout(test_name)
    assert deleted, "Delete returned False"
    assert load_layout(test_name) is None, "Layout still exists after delete"

    print("✅ test_layout_persistence passed")


def test_slots_to_pixels():
    """Normalized coords → pixel coords"""
    slots = [{"id": "P1", "x_norm": 0.1, "y_norm": 0.2, "w_norm": 0.1, "h_norm": 0.15}]
    px = slots_to_pixels(slots, 1000, 800)
    assert px[0]["x"] == 100
    assert px[0]["y"] == 160
    assert px[0]["w"] == 100
    assert px[0]["h"] == 120
    print("✅ test_slots_to_pixels passed")


def test_canvas_to_normalized_slots():
    """Verify both Fabric v6 'Rect' and v5 'rect' objects are parsed correctly"""
    from slot_manager import canvas_to_normalized_slots

    class DummyCanvasResult:
        def __init__(self, json_data):
            self.json_data = json_data

    # Fabric v6 uses type="Rect"
    canvas_v6 = DummyCanvasResult({
        "objects": [
            {"type": "Rect", "left": 100, "top": 50, "width": 40, "height": 80, "scaleX": 1, "scaleY": 1},
            {"type": "rect", "left": 200, "top": 50, "width": 40, "height": 80, "scaleX": 1, "scaleY": 1},
            # Negative dragging (right-to-left)
            {"type": "Rect", "left": 350, "top": 150, "width": -50, "height": -60, "scaleX": 1, "scaleY": 1},
            # Tiny accidental click (< 5px) should be ignored
            {"type": "Rect", "left": 10, "top": 10, "width": 2, "height": 2, "scaleX": 1, "scaleY": 1},
        ]
    })

    slots = canvas_to_normalized_slots(canvas_v6, canvas_w=960, canvas_h=540, img_w=1920, img_h=1080)
    assert len(slots) == 3, f"Expected 3 valid slots, got {len(slots)}"
    assert slots[0]["id"] == "P1"
    assert slots[1]["id"] == "P2"
    assert slots[2]["id"] == "P3"
    assert 0.0 <= slots[2]["x_norm"] <= 1.0
    assert 0.0 <= slots[2]["y_norm"] <= 1.0
    print("✅ test_canvas_to_normalized_slots passed")


def test_compute_overlap_metrics():
    """Verify slot overlap ratio and vehicle overlap ratio."""
    from slot_manager import compute_overlap_metrics
    slot = (0, 0, 100, 100)        # area = 10000
    veh = (50, 0, 150, 100)        # area = 10000, inter = 5000
    m = compute_overlap_metrics(slot, veh)
    assert m["slot_overlap_ratio"] == 0.5, f"Expected 0.5, got {m['slot_overlap_ratio']}"
    assert m["vehicle_overlap_ratio"] == 0.5, f"Expected 0.5, got {m['vehicle_overlap_ratio']}"
    assert round(m["iou"], 4) == round(5000 / 15000, 4)
    print("✅ test_compute_overlap_metrics passed")


def test_analyze_slot_crop():
    """Verify crop feature extractor produces scores in [0, 1]."""
    from slot_manager import analyze_slot_crop
    import numpy as np
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    res = analyze_slot_crop(blank)
    assert 0.0 <= res["occupied_score"] <= 1.0
    print("✅ test_analyze_slot_crop passed")


if __name__ == "__main__":
    test_iou_no_overlap()
    test_iou_perfect_overlap()
    test_iou_partial()
    test_overlap_ratio()
    test_classify_occupied()
    test_classify_free()
    test_classify_no_vehicles()
    test_single_vehicle_one_slot()
    test_count_validation()
    test_layout_persistence()
    test_slots_to_pixels()
    test_canvas_to_normalized_slots()
    test_compute_overlap_metrics()
    test_analyze_slot_crop()
    print("\n🎉 All tests passed!")
