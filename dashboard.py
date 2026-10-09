import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import hashlib
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from parking_detector import detect_vehicles
from slot_manager import (
    slots_to_pixels,
    classify_slots,
    count_statuses,
    canvas_to_normalized_slots,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Parking System", layout="wide")
st.title("🚗 Smart Parking Detection System")
st.markdown("AI-based parking slot detection using Computer Vision & YOLO")

# ── Sidebar: Mode & Settings ───────────────────────────────────────────────
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Video Upload", "Live Webcam"],
    key="select_mode",
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Settings")

model_choice = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="yolov8n is faster; yolov8s has higher accuracy on overhead views",
)

aerial_mode = st.sidebar.checkbox(
    "Overhead / Aerial Perspective",
    value=True,
    help="Optimizes vehicle detection for bird's-eye and top-down camera views",
)

secondary_classifier = st.sidebar.checkbox(
    "Secondary Slot-Crop Classifier",
    value=True,
    help="Analyzes image crops inside configured slots to rescue vehicles missed by YOLO",
)

conf_threshold = st.sidebar.slider(
    "YOLO Confidence",
    min_value=0.02,
    max_value=0.70,
    value=0.10,
    step=0.02,
    help="Lower threshold detects more vehicles in aerial views",
)

occupied_threshold = st.sidebar.slider(
    "Occupied Overlap Ratio",
    min_value=0.05,
    max_value=0.60,
    value=0.15,
    step=0.05,
    help="Minimum slot coverage required to mark a slot as occupied",
)

uncertain_threshold = st.sidebar.slider(
    "Uncertain Overlap Ratio",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
    help="Minimum overlap to mark a slot as uncertain if evidence is borderline",
)

yolo_imgsz = st.sidebar.selectbox(
    "YOLO Resolution",
    [640, 960, 1280, 1920],
    index=2,
    help="Higher resolution improves detection of small/distant vehicles in high-res images",
)

# ── Helper styling & drawing ──────────────────────────────────────────────
STATUS_COLORS_BGR = {
    "free": (0, 200, 0),
    "occupied": (0, 0, 220),
    "uncertain": (0, 180, 255),
}

DISPLAY_W = 960


def draw_results(img, classified_slots, vehicles, vehicle_confs):
    """Draw vehicle detections and classified parking slots on the image."""
    overlay = img.copy()

    # Draw detected vehicle bounding boxes
    for (x1, y1, x2, y2), conf in zip(vehicles, vehicle_confs):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            overlay,
            f"{conf:.0%}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 200, 0),
            1,
        )

    # Draw classified parking slots
    for s in classified_slots:
        status = s.get("status", "uncertain")
        color = STATUS_COLORS_BGR.get(status, (128, 128, 128))
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]

        # Semi-transparent fill
        sub_overlay = overlay.copy()
        cv2.rectangle(sub_overlay, (x, y), (x + w, y + h), color, -1)
        alpha = 0.28
        cv2.addWeighted(sub_overlay, alpha, overlay, 1 - alpha, 0, overlay)

        # Crisp slot boundary
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Slot ID tag
        label = s.get("id", "?")
        ov = s.get("best_overlap", 0.0)
        label_text = f"{label} ({ov:.0%})" if status != "free" else label
        cv2.putText(
            overlay,
            label_text,
            (x + 4, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    return overlay


def display_stats(counts):
    """Display summary metrics strictly verifying total = free + occupied + uncertain."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Slots", counts["total"])
    c2.metric("🟢 Free", counts["free"])
    c3.metric("🔴 Occupied", counts["occupied"])
    c4.metric("🟠 Uncertain", counts["uncertain"])


# ---------------------------------------------------------------------------
# Session State Management: Strict Isolation Per-Mode & Per-Input
# ---------------------------------------------------------------------------
# 1. Reset state whenever the mode changes
if st.session_state.get("current_mode") != mode:
    st.session_state["current_mode"] = mode
    st.session_state["input_identity"] = None
    st.session_state["confirmed_slots"] = []
    st.session_state["layout_confirmed"] = False
    st.session_state["analysis_done"] = False
    st.session_state["detection_results"] = None


# ======================================================================
# 🖼️ IMAGE MODE
# ======================================================================
if mode == "Image Upload":
    st.header("🖼️ Image Parking Detection")

    uploaded_img = st.file_uploader(
        "Upload Parking Image",
        type=["jpg", "png", "jpeg"],
        key="img_uploader",
    )

    if uploaded_img is not None:
        file_bytes = uploaded_img.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 2. Reset layout if a new image file is uploaded
        if st.session_state.get("input_identity") != file_hash:
            st.session_state["input_identity"] = file_hash
            st.session_state["confirmed_slots"] = []
            st.session_state["layout_confirmed"] = False
            st.session_state["analysis_done"] = False
            st.session_state["detection_results"] = None

        img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_h, img_w = img_array.shape[:2]

        display_h = int(DISPLAY_W * img_h / img_w)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        bg_image = Image.fromarray(img_rgb).resize((DISPLAY_W, display_h))

        # Workflow step indicator
        is_confirmed = st.session_state.get("layout_confirmed", False)

        # ── STEP A: SLOT CONFIGURATION WORKFLOW ─────────────────────────
        if not is_confirmed:
            st.subheader("📐 Step 1: Mark Parking Slots")
            st.info(
                "Draw **rectangles** around each parking space on the image below. "
                "Once all slots are marked, click **✅ Confirm Layout** to proceed to analysis."
            )

            # Pre-populate any existing in-progress slots
            current_slots = st.session_state.get("confirmed_slots", [])
            initial_drawing = None
            if current_slots:
                initial_objects = [
                    {
                        "type": "Rect",
                        "left": s["x_norm"] * DISPLAY_W,
                        "top": s["y_norm"] * display_h,
                        "width": s["w_norm"] * DISPLAY_W,
                        "height": s["h_norm"] * display_h,
                        "fill": "rgba(0, 200, 0, 0.15)",
                        "stroke": "#00C800",
                        "strokeWidth": 2,
                    }
                    for s in current_slots
                ]
                initial_drawing = {"version": "5.3.0", "objects": initial_objects}

            canvas_result = st_canvas(
                fill_color="rgba(0, 200, 0, 0.15)",
                stroke_width=2,
                stroke_color="#00C800",
                background_image=bg_image,
                drawing_mode="rect",
                height=display_h,
                width=DISPLAY_W,
                initial_drawing=initial_drawing,
                key=f"canvas_img_{file_hash}",
            )

            # Keep slots tracked continuously in session_state
            drawn_slots = canvas_to_normalized_slots(
                canvas_result, DISPLAY_W, display_h, img_w, img_h
            )
            if drawn_slots:
                st.session_state["in_progress_slots"] = drawn_slots

            active_slots = drawn_slots or st.session_state.get("in_progress_slots", [])

            if active_slots:
                st.info(f"✨ **{len(active_slots)} slot(s) drawn** on this image.")
            else:
                st.caption("No slots drawn yet. Click and drag to create parking rectangles.")

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("✅ Confirm Layout", type="primary"):
                    to_confirm = active_slots or canvas_to_normalized_slots(
                        canvas_result, DISPLAY_W, display_h, img_w, img_h
                    )
                    if to_confirm:
                        st.session_state["confirmed_slots"] = to_confirm
                        st.session_state["layout_confirmed"] = True
                        st.session_state["analysis_done"] = False
                        st.success(f"✅ Layout confirmed with {len(to_confirm)} parking spaces!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Please draw at least one parking slot before confirming.")
            with col_btn2:
                if st.button("🔄 Clear Drawing"):
                    st.session_state["in_progress_slots"] = []
                    st.session_state["confirmed_slots"] = []
                    st.rerun()

        # ── STEP B: ANALYSIS & RESULTS WORKFLOW ─────────────────────────
        else:
            confirmed_slots = st.session_state.get("confirmed_slots", [])
            st.subheader("🔍 Step 2: Occupancy Detection")

            # Layout control buttons
            col_ctrl1, col_ctrl2 = st.columns([1, 1])
            with col_ctrl1:
                if st.button("📝 Edit / Redraw Layout"):
                    st.session_state["layout_confirmed"] = False
                    st.session_state["analysis_done"] = False
                    st.rerun()
            with col_ctrl2:
                if st.button("🗑️ Reset Layout for this Image"):
                    st.session_state["confirmed_slots"] = []
                    st.session_state["layout_confirmed"] = False
                    st.session_state["analysis_done"] = False
                    st.session_state["detection_results"] = None
                    st.rerun()

            st.markdown("---")

            # Run analysis
            slots_px = slots_to_pixels(confirmed_slots, img_w, img_h)

            with st.spinner("Analyzing parking lot with YOLO and secondary classifier..."):
                vehicles, vehicle_confs = detect_vehicles(
                    img_array,
                    conf_thres=conf_threshold,
                    imgsz=yolo_imgsz,
                    aerial_mode=aerial_mode,
                    model_name=model_choice,
                )

                classified = classify_slots(
                    slots_px,
                    vehicles,
                    vehicle_confs,
                    frame=img_array,
                    occupied_threshold=occupied_threshold,
                    uncertain_threshold=uncertain_threshold,
                    enable_secondary=secondary_classifier,
                )

            st.session_state["analysis_done"] = True
            st.session_state["detection_results"] = classified

            counts = count_statuses(classified)
            display_stats(counts)

            result_img = draw_results(img_array, classified, vehicles, vehicle_confs)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.expander("📊 Detailed Slot Occupancy Breakdown"):
                for s in classified:
                    icon = {"free": "🟢", "occupied": "🔴", "uncertain": "🟠"}.get(s["status"], "❓")
                    st.write(
                        f"{icon} **{s['id']}**: **{s['status'].upper()}** — {s.get('evidence', '')}"
                    )

            with st.expander(f"🚘 Detected Vehicles ({len(vehicles)} found)"):
                for idx, ((vx1, vy1, vx2, vy2), vconf) in enumerate(zip(vehicles, vehicle_confs)):
                    st.write(
                        f"Vehicle #{idx + 1}: box=({vx1}, {vy1}) → ({vx2}, {vy2}) | Confidence: {vconf:.1%}"
                    )


# ======================================================================
# 🎥 VIDEO MODE
# ======================================================================
elif mode == "Video Upload":
    st.header("🎥 Video Parking Detection")

    uploaded_video = st.file_uploader(
        "Upload Parking Video",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        video_hash = hashlib.md5(video_bytes).hexdigest()

        # Reset layout when a new video is uploaded
        if st.session_state.get("input_identity") != video_hash:
            st.session_state["input_identity"] = video_hash
            st.session_state["confirmed_slots"] = []
            st.session_state["layout_confirmed"] = False
            st.session_state["analysis_done"] = False

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_bytes)
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap.read()

        if not ret:
            st.error("Cannot read video file.")
        else:
            v_h, v_w = first_frame.shape[:2]
            display_h = int(DISPLAY_W * v_h / v_w)
            bg = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)).resize(
                (DISPLAY_W, display_h)
            )

            is_confirmed_v = st.session_state.get("layout_confirmed", False)

            # ── STEP A: CONFIGURE LAYOUT ON REPRESENTATIVE FRAME ──────────
            if not is_confirmed_v:
                st.subheader("📐 Step 1: Mark Parking Spaces on Representative Frame")
                st.info(
                    "Draw rectangles around parking spaces on the frame below. "
                    "This layout will be locked and reused across all video frames."
                )

                canvas_v = st_canvas(
                    fill_color="rgba(0, 200, 0, 0.15)",
                    stroke_width=2,
                    stroke_color="#00C800",
                    background_image=bg,
                    drawing_mode="rect",
                    height=display_h,
                    width=DISPLAY_W,
                    key=f"canvas_v_{video_hash}",
                )

                drawn_slots_v = canvas_to_normalized_slots(
                    canvas_v, DISPLAY_W, display_h, v_w, v_h
                )
                if drawn_slots_v:
                    st.session_state["in_progress_slots_v"] = drawn_slots_v

                active_v_slots = drawn_slots_v or st.session_state.get("in_progress_slots_v", [])

                if active_v_slots:
                    st.info(f"✨ **{len(active_v_slots)} slot(s) drawn** on representative frame.")
                else:
                    st.caption("No slots drawn yet. Click and drag to draw parking rectangles.")

                col_v1, col_v2 = st.columns([1, 1])
                with col_v1:
                    if st.button("✅ Confirm Video Layout", type="primary"):
                        to_confirm_v = active_v_slots or canvas_to_normalized_slots(
                            canvas_v, DISPLAY_W, display_h, v_w, v_h
                        )
                        if to_confirm_v:
                            st.session_state["confirmed_slots"] = to_confirm_v
                            st.session_state["layout_confirmed"] = True
                            st.success(
                                f"✅ Confirmed {len(to_confirm_v)} slots for this video!"
                            )
                            st.rerun()
                        else:
                            st.warning("⚠️ Please draw at least one slot before confirming.")
                with col_v2:
                    if st.button("🔄 Clear Drawing"):
                        st.session_state["in_progress_slots_v"] = []
                        st.session_state["confirmed_slots"] = []
                        st.rerun()

            # ── STEP B: PROCESS VIDEO FRAMES ─────────────────────────────
            else:
                confirmed_slots_v = st.session_state.get("confirmed_slots", [])
                st.subheader("▶️ Step 2: Live Video Occupancy Processing")

                col_vc1, col_vc2 = st.columns([1, 1])
                with col_vc1:
                    if st.button("📝 Edit Video Layout"):
                        st.session_state["layout_confirmed"] = False
                        st.rerun()
                with col_vc2:
                    if st.button("🗑️ Reset Video Layout"):
                        st.session_state["confirmed_slots"] = []
                        st.session_state["layout_confirmed"] = False
                        st.rerun()

                slots_px = slots_to_pixels(confirmed_slots_v, v_w, v_h)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                stats_placeholder = st.empty()
                frame_window = st.image([])

                frame_count = 0
                YOLO_SKIP = 4
                last_vehicles = []
                last_confs = []

                # Temporal smoothing per slot
                status_history = {s["id"]: [] for s in slots_px}
                SMOOTH_WINDOW = 4

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if frame_count % YOLO_SKIP == 0 or not last_vehicles:
                        vehicles, confs = detect_vehicles(
                            frame,
                            conf_thres=conf_threshold,
                            imgsz=yolo_imgsz,
                            aerial_mode=aerial_mode,
                            model_name=model_choice,
                        )
                        last_vehicles = vehicles
                        last_confs = confs
                    else:
                        vehicles = last_vehicles
                        confs = last_confs

                    classified = classify_slots(
                        slots_px,
                        vehicles,
                        confs,
                        frame=frame,
                        occupied_threshold=occupied_threshold,
                        uncertain_threshold=uncertain_threshold,
                        enable_secondary=secondary_classifier,
                    )

                    # Temporal smoothing to prevent flicker
                    for s in classified:
                        sid = s["id"]
                        status_history[sid].append(s["status"])
                        if len(status_history[sid]) > SMOOTH_WINDOW:
                            status_history[sid].pop(0)
                        recent = status_history[sid]
                        s["status"] = max(set(recent), key=recent.count)

                    counts = count_statuses(classified)

                    with stats_placeholder.container():
                        display_stats(counts)

                    result_frame = draw_results(frame, classified, vehicles, confs)
                    frame_window.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    time.sleep(0.02)

                cap.release()


# ======================================================================
# 📷 LIVE WEBCAM MODE
# ======================================================================
else:
    st.header("📷 Live Webcam Parking Detection")
    st.info("Configure your parking layout on the live webcam feed.")

    run = st.checkbox("Start Live Stream")
    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Webcam not accessible.")
        else:
            w_h, w_w = first_frame.shape[:2]
            display_h = int(DISPLAY_W * w_h / w_w)
            bg = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)).resize(
                (DISPLAY_W, display_h)
            )

            is_confirmed_w = st.session_state.get("layout_confirmed", False)

            if not is_confirmed_w:
                st.subheader("📐 Mark Parking Slots on Webcam Stream")
                canvas_w = st_canvas(
                    fill_color="rgba(0, 200, 0, 0.15)",
                    stroke_width=2,
                    stroke_color="#00C800",
                    background_image=bg,
                    drawing_mode="rect",
                    height=display_h,
                    width=DISPLAY_W,
                    key="webcam_canvas",
                )

                drawn_slots_w = canvas_to_normalized_slots(
                    canvas_w, DISPLAY_W, display_h, w_w, w_h
                )
                if st.button("✅ Confirm Webcam Layout", type="primary"):
                    if drawn_slots_w:
                        st.session_state["confirmed_slots"] = drawn_slots_w
                        st.session_state["layout_confirmed"] = True
                        st.rerun()
                    else:
                        st.warning("Draw parking slots before confirming.")
                cap.release()
            else:
                confirmed_slots_w = st.session_state.get("confirmed_slots", [])
                slots_px = slots_to_pixels(confirmed_slots_w, w_w, w_h)

                if st.button("📝 Edit Layout"):
                    st.session_state["layout_confirmed"] = False
                    st.rerun()

                stats_placeholder = st.empty()
                frame_window = st.image([])

                frame_count = 0
                YOLO_SKIP = 4
                last_vehicles = []
                last_confs = []
                status_history = {s["id"]: [] for s in slots_px}
                SMOOTH_WINDOW = 3

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % YOLO_SKIP == 0 or not last_vehicles:
                        vehicles, confs = detect_vehicles(
                            frame,
                            conf_thres=conf_threshold,
                            imgsz=yolo_imgsz,
                            aerial_mode=aerial_mode,
                            model_name=model_choice,
                        )
                        last_vehicles = vehicles
                        last_confs = confs
                    else:
                        vehicles = last_vehicles
                        confs = last_confs

                    classified = classify_slots(
                        slots_px,
                        vehicles,
                        confs,
                        frame=frame,
                        occupied_threshold=occupied_threshold,
                        uncertain_threshold=uncertain_threshold,
                        enable_secondary=secondary_classifier,
                    )

                    for s in classified:
                        sid = s["id"]
                        status_history[sid].append(s["status"])
                        if len(status_history[sid]) > SMOOTH_WINDOW:
                            status_history[sid].pop(0)
                        recent = status_history[sid]
                        s["status"] = max(set(recent), key=recent.count)

                    counts = count_statuses(classified)
                    with stats_placeholder.container():
                        display_stats(counts)

                    result_frame = draw_results(frame, classified, vehicles, confs)
                    frame_window.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    time.sleep(0.03)

                cap.release()
