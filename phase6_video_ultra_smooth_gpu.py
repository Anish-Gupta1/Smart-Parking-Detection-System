import cv2
import time
from detector.auto_slot_detector import auto_detect_slots
from detector.parking_detector import detect_vehicles, box_overlap

VIDEO_PATH = "data/videos/parking.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, first_frame = cap.read()
if not ret:
    print("Video not found")
    exit()

FRAME_W, FRAME_H = 640, 360
first_frame = cv2.resize(first_frame, (FRAME_W, FRAME_H))

# ---- SLOT DETECTION (STABILIZED) ----
slots = auto_detect_slots(first_frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
YOLO_SKIP = 5
last_vehicles = []

# FPS smoothing
prev_time = time.time()
fps_buffer = []

TARGET_FPS = cap.get(cv2.CAP_PROP_FPS)
if TARGET_FPS == 0:
    TARGET_FPS = 25
frame_delay = int(1000 / TARGET_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame_count += 1

    # ---- YOLO (FRAME COPY FOR SMOOTHNESS) ----
    if frame_count % YOLO_SKIP == 0:
        vehicles = detect_vehicles(frame.copy())
        last_vehicles = vehicles
    else:
        vehicles = last_vehicles

    free = 0

    # Draw vehicles
    for x1, y1, x2, y2 in vehicles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Slot logic
    for x, y, w, h in slots:
        slot_box = (x, y, x + w, y + h)
        occupied = False

        for v in vehicles:
            if box_overlap(slot_box, v):
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            free += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ---- SMOOTH FPS CALCULATION ----
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    fps_buffer.append(fps)
    if len(fps_buffer) > 10:
        fps_buffer.pop(0)

    avg_fps = int(sum(fps_buffer) / len(fps_buffer))

    cv2.putText(frame, f"FPS: {avg_fps}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Free Slots: {free}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Parking – GPU Ultra Smooth", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
