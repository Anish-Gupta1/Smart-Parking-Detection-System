"""
Parking Detector Module
- YOLO-based vehicle detection with enhanced overhead/aerial sensitivity
- Multi-class vehicle mapping + aerial proxy detection
- CLAHE contrast enhancement for challenging parking lighting
- Class-agnostic NMS to eliminate overlapping candidate boxes
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.ops as ops

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Model cache
_MODELS: dict[str, YOLO] = {}


def get_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Retrieve or load a YOLO model by weight name."""
    if model_name not in _MODELS:
        _MODELS[model_name] = YOLO(model_name).to(DEVICE)
    return _MODELS[model_name]


# Standard vehicle classes in COCO
STANDARD_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Top-down / aerial vehicle proxies:
# Overhead cameras view vehicles from directly above. In COCO-trained YOLO,
# vehicles seen from bird's-eye view frequently trigger 'cell phone' (metallic rounded rectangle)
# or 'parking meter' with high confidence.
AERIAL_PROXY_CLASSES = {"cell phone", "parking meter", "suitcase"}


def box_overlap(boxA, boxB) -> bool:
    """Return True when two bounding boxes intersect."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB


def enhance_frame_contrast(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on lightness channel to reveal vehicles obscured by harsh shadows,
    glare, or dark asphalt without altering color hue.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def detect_vehicles(
    frame: np.ndarray,
    conf_thres: float = 0.10,
    imgsz: int = 1280,
    aerial_mode: bool = True,
    model_name: str = "yolov8n.pt",
    iou_nms: float = 0.35,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Detect vehicles in an image/frame with high sensitivity for parking lots.

    Parameters
    ----------
    frame : np.ndarray
        Original resolution BGR image.
    conf_thres : float
        Minimum confidence threshold.
    imgsz : int
        YOLO inference resolution (640, 960, 1280, 1920).
    aerial_mode : bool
        If True, includes top-down vehicle proxies filtered by vehicle aspect ratio.
    model_name : str
        YOLO model weights ('yolov8n.pt', 'yolov8s.pt').
    iou_nms : float
        IoU threshold for class-agnostic NMS deduplication.

    Returns
    -------
    vehicles : list of (x1, y1, x2, y2) bounding boxes
    confidences : list of confidence scores in [0.0, 1.0]
    """
    if frame is None or frame.size == 0:
        return [], []

    model = get_model(model_name)
    proc_frame = enhance_frame_contrast(frame)

    # Run inference
    results = model.predict(
        source=proc_frame,
        conf=max(0.02, conf_thres),
        imgsz=imgsz,
        verbose=False,
        device=DEVICE,
    )

    h_img, w_img = frame.shape[:2]
    frame_area = float(h_img * w_img)

    target_classes = set(STANDARD_VEHICLE_CLASSES)
    if aerial_mode:
        target_classes.update(AERIAL_PROXY_CLASSES)

    raw_boxes: list[list[float]] = []
    raw_scores: list[float] = []

    for result in results:
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            cls_name = model.names[cls_idx]

            if cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = bw * bh

            # Filter out tiny noise dots or full-image boxes
            if area < 300 or area > (0.50 * frame_area):
                continue

            # Vehicle aspect ratio validation for proxy classes
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if cls_name in AERIAL_PROXY_CLASSES:
                # Parked vehicles typically have aspect ratio between 1.15 and 4.2
                if not (1.15 <= aspect <= 4.2):
                    continue

            raw_boxes.append([x1, y1, x2, y2])
            raw_scores.append(float(box.conf[0]))

    if not raw_boxes:
        return [], []

    # Apply class-agnostic NMS to merge duplicate detections across classes
    boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=iou_nms)

    final_boxes: list[tuple[int, int, int, int]] = []
    final_scores: list[float] = []

    for idx in keep_indices.tolist():
        b = boxes_tensor[idx].tolist()
        final_boxes.append((int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))))
        final_scores.append(float(scores_tensor[idx]))

    return final_boxes, final_scores


def detect_vehicles_legacy(frame, conf_thres=0.25):
    """Legacy wrapper returning only boxes for backwards compatibility."""
    boxes, _ = detect_vehicles(frame, conf_thres=conf_thres)
    return boxes
