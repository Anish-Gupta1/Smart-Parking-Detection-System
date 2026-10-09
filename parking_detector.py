
from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

model = YOLO("yolov8n.pt").to(DEVICE)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def box_overlap(boxA, boxB):
    """Return True when two bounding boxes intersect."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    return xA < xB and yA < yB


def detect_vehicles(frame, conf_thres=0.20, imgsz=1280):
    """
    Detect vehicles and return (boxes, confidences).

    Parameters
    ----------
    frame : np.ndarray
        BGR image at *original* resolution (do NOT pre-shrink).
    conf_thres : float
        Minimum confidence threshold (lowered from 0.25 to catch more cars).
    imgsz : int
        YOLO inference resolution. Higher = better detection of small cars
        in aerial views, but slower.  1280 is a good balance.

    Returns
    -------
    vehicles : list of (x1, y1, x2, y2) in the frame's coordinate space
    confidences : list of float, parallel to vehicles
    """
    results = model.predict(
        source=frame,
        conf=conf_thres,
        imgsz=imgsz,
        verbose=False,
        device=DEVICE,
    )

    vehicles = []
    confidences = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if model.names[cls] not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            vehicles.append((x1, y1, x2, y2))
            confidences.append(conf)

    return vehicles, confidences


def detect_vehicles_legacy(frame, conf_thres=0.25):
    """Legacy wrapper that returns only boxes (for backward compat)."""
    vehicles, _ = detect_vehicles(frame, conf_thres=conf_thres)
    return vehicles
