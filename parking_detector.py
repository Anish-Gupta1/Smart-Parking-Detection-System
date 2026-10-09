
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


def detect_vehicles(frame, conf_thres=0.25):
    """Detect vehicles and return their (x1, y1, x2, y2) boxes."""
    results = model.predict(
        source=frame,
        conf=conf_thres,
        imgsz=960,
        verbose=False,
        device=DEVICE,
    )

    vehicles = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if model.names[cls] not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            vehicles.append((x1, y1, x2, y2))

    return vehicles
