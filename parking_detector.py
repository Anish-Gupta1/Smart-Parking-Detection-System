from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

model = YOLO("yolov8n.pt").to(DEVICE)

VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]



def box_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB


def detect_vehicles(frame, conf_thres=0.5):
    vehicles = []
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            if box.conf[0] < conf_thres:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append((x1, y1, x2, y2))

    return vehicles
