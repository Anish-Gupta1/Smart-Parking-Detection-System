import cv2
import time
from detector.auto_slot_detector import auto_detect_slots
from detector.parking_detector import detect_vehicles, box_overlap

# ---------------- VIDEO SETUP ----------------
VIDEO_PATH = "data/videos/parking.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Kill OpenCV buffering (VERY IMPORTANT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Read first frame to init slots
ret, first_frame = cap.read()
if not ret:
    print("❌ Video not found")
    exit()

# Resize aggressively for FPS
FRAME_W, FRAME_H = 640, 360
first_frame = cv2.resize(first_frame, (FRAME_W, FRAME_H))

# ---------------- AUTO SLOT DETECTION (ONCE) ----------------
slots = auto_detect_slots(first_frame)

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------- PERFORMANCE TUNING ----------------
frame_count = 0
YOLO_SKIP = 5          # YOLO every 5th frame
last_vehicles = []

prev_time = time.time()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame_count += 1

    # ---------- YOLO DETECTION (GPU) ----------
    if frame_count % YOLO_SKIP == 0:
        vehicles = detect_vehicles(frame)
        last_vehicles = vehicles
    else:
        vehicles = last_vehicles

    free = 0

    # ---------- DRAW VEHICLES ----------
    for x1, y1, x2, y2 in vehicles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ---------- SLOT CHECK ----------
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

    # ---------- FPS ----------
    now = time.time()
    fps = int(1 / (now - prev_time))
    prev_time = now

    cv2.putText(frame, f"FPS: {fps}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Free Slots: {free}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Parking – GPU Ultra Smooth", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
