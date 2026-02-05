import streamlit as st
import cv2
import tempfile
import time

from detector.auto_slot_detector import auto_detect_slots
from detector.parking_detector import detect_vehicles, box_overlap

st.set_page_config(page_title="Smart Parking System", layout="wide")

st.title("🚗 Smart Parking Detection System")
st.markdown("AI-based parking slot detection using Computer Vision & YOLO")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Video Upload", "Live Webcam"]
)

FRAME_W, FRAME_H = 640, 360

# ======================================================
# 🖼️ IMAGE MODE
# ======================================================
if mode == "Image Upload":
    st.header("🖼️ Image Parking Detection")

    uploaded_img = st.file_uploader("Upload Parking Image", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        file_bytes = uploaded_img.read()
        img_array = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        img = cv2.resize(img_array, (FRAME_W, FRAME_H))

        slots = auto_detect_slots(img)
        vehicles = detect_vehicles(img)

        free = 0

        for v in vehicles:
            cv2.rectangle(img, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 2)

        for x, y, w, h in slots:
            slot_box = (x, y, x + w, y + h)
            occupied = any(box_overlap(slot_box, v) for v in vehicles)

            if occupied:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                free += 1

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        st.success(f"Free Slots: {free}")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

# ======================================================
# 🎥 VIDEO MODE
# ======================================================
elif mode == "Video Upload":
    st.header("🎥 Video Parking Detection")

    uploaded_video = st.file_uploader("Upload Parking Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Cannot read video")
        else:
            first_frame = cv2.resize(first_frame, (FRAME_W, FRAME_H))
            slots = auto_detect_slots(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frame_window = st.image([])

            frame_count = 0
            YOLO_SKIP = 5
            last_vehicles = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                frame_count += 1

                if frame_count % YOLO_SKIP == 0:
                    vehicles = detect_vehicles(frame)
                    last_vehicles = vehicles
                else:
                    vehicles = last_vehicles

                free = 0

                for v in vehicles:
                    cv2.rectangle(frame, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 2)

                for x, y, w, h in slots:
                    slot_box = (x, y, x + w, y + h)
                    occupied = any(box_overlap(slot_box, v) for v in vehicles)

                    if occupied:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                        free += 1

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                frame_window.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

                time.sleep(0.03)

# ======================================================
# 📷 LIVE WEBCAM MODE
# ======================================================
else:
    st.header("📷 Live Webcam Parking Detection")

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
        else:
            first_frame = cv2.resize(first_frame, (FRAME_W, FRAME_H))
            slots = auto_detect_slots(first_frame)

            frame_window = st.image([])

            frame_count = 0
            YOLO_SKIP = 5
            last_vehicles = []

            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                frame_count += 1

                if frame_count % YOLO_SKIP == 0:
                    vehicles = detect_vehicles(frame)
                    last_vehicles = vehicles
                else:
                    vehicles = last_vehicles

                free = 0

                for v in vehicles:
                    cv2.rectangle(frame, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 2)

                for x, y, w, h in slots:
                    slot_box = (x, y, x + w, y + h)
                    occupied = any(box_overlap(slot_box, v) for v in vehicles)

                    if occupied:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                        free += 1

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                frame_window.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

                time.sleep(0.03)

        cap.release()
