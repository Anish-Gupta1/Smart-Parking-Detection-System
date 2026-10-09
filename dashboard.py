import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import json
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from parking_detector import detect_vehicles, detect_vehicles_legacy, box_overlap
from slot_manager import (
    save_layout, load_layout, delete_layout, list_layouts,
    slots_to_pixels, classify_slots, count_statuses,
    canvas_to_normalized_slots,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Parking System", layout="wide")
st.title("🚗 Smart Parking Detection System")
st.markdown("AI-based parking slot detection using Computer Vision & YOLO")

# ── Sidebar ───────────────────────────────────────────────────────────────
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Video Upload", "Live Webcam"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider(
    "YOLO Confidence", 0.05, 0.80, 0.20, 0.05,
    help="Lower = detect more vehicles (may include false positives)",
)
occupied_threshold = st.sidebar.slider(
    "Occupied Overlap", 0.05, 0.60, 0.15, 0.05,
    help="Min overlap ratio to classify a slot as occupied",
)
uncertain_threshold = st.sidebar.slider(
    "Uncertain Overlap", 0.01, 0.30, 0.05, 0.01,
    help="Min overlap ratio to classify a slot as uncertain",
)
yolo_imgsz = st.sidebar.selectbox(
    "YOLO Resolution", [640, 960, 1280, 1920], index=2,
    help="Higher = better detection of small/distant cars, slower",
)

# ── Helpers ───────────────────────────────────────────────────────────────
STATUS_COLORS_BGR = {
    "free":      (0, 200, 0),
    "occupied":  (0, 0, 220),
    "uncertain": (0, 180, 255),
}

STATUS_COLORS_HEX = {
    "free":      "#00C800",
    "occupied":  "#DC0000",
    "uncertain": "#FFB400",
}

DISPLAY_W = 960  # Width for canvas display


def draw_results(img, classified_slots, vehicles, vehicle_confs):
    """Draw slots and vehicles on the image and return it."""
    overlay = img.copy()

    # Draw vehicle detections (thin cyan boxes)
    for (x1, y1, x2, y2), conf in zip(vehicles, vehicle_confs):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(overlay, f"{conf:.0%}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # Draw classified parking slots
    for s in classified_slots:
        status = s.get("status", "uncertain")
        color = STATUS_COLORS_BGR.get(status, (128, 128, 128))
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]

        # Semi-transparent fill
        sub_overlay = overlay.copy()
        cv2.rectangle(sub_overlay, (x, y), (x + w, y + h), color, -1)
        alpha = 0.25
        cv2.addWeighted(sub_overlay, alpha, overlay, 1 - alpha, 0, overlay)

        # Border
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Slot ID label
        label = s.get("id", "?")
        cv2.putText(overlay, label, (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return overlay


def display_stats(counts):
    """Display occupancy statistics in columns."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Slots", counts["total"])
    c2.metric("🟢 Free", counts["free"])
    c3.metric("🔴 Occupied", counts["occupied"])
    c4.metric("🟠 Uncertain", counts["uncertain"])



# ======================================================================
# 🖼️ IMAGE MODE
# ======================================================================
if mode == "Image Upload":
    st.header("🖼️ Image Parking Detection")

    uploaded_img = st.file_uploader(
        "Upload Parking Image", type=["jpg", "png", "jpeg"],
        key="img_uploader",
    )

    if uploaded_img is not None:
        file_bytes = uploaded_img.read()
        img_array = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
        img_h, img_w = img_array.shape[:2]

        # Layout management
        st.sidebar.markdown("---")
        st.sidebar.subheader("📐 Layout Management")
        saved = list_layouts()
        layout_name = st.sidebar.text_input(
            "Layout Name", value="default",
            help="Name to save/load this parking layout",
        )

        # Check for saved layout
        saved_slots = load_layout(layout_name)

        # State management for edit mode
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = saved_slots is None

        # Sidebar layout controls
        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button("📝 Edit Layout"):
            st.session_state.edit_mode = True
            st.rerun()
        if col_s2.button("🗑️ Reset Layout"):
            delete_layout(layout_name)
            st.session_state.edit_mode = True
            st.rerun()

        # ── EDIT MODE: Draw parking slots ────────────────────────
        if st.session_state.edit_mode:
            st.subheader("📐 Configure Parking Slots")
            st.info(
                "Draw **rectangles** around each parking space on the image. "
                "Click **Save Layout** when done."
            )

            # Prepare background image for canvas
            display_h = int(DISPLAY_W * img_h / img_w)
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            bg_image = Image.fromarray(img_rgb).resize((DISPLAY_W, display_h))

            # Pre-populate existing slots as initial drawing objects
            initial_drawing = None
            if saved_slots:
                objects = []
                for s in saved_slots:
                    objects.append({
                        "type": "Rect",
                        "left": s["x_norm"] * DISPLAY_W,
                        "top": s["y_norm"] * display_h,
                        "width": s["w_norm"] * DISPLAY_W,
                        "height": s["h_norm"] * display_h,
                        "fill": "rgba(0, 200, 0, 0.15)",
                        "stroke": "#00C800",
                        "strokeWidth": 2,
                    })
                initial_drawing = {"version": "5.3.0", "objects": objects}

            canvas_result = st_canvas(
                fill_color="rgba(0, 200, 0, 0.15)",
                stroke_width=2,
                stroke_color="#00C800",
                background_image=bg_image,
                drawing_mode="rect",
                height=display_h,
                width=DISPLAY_W,
                initial_drawing=initial_drawing,
                key="slot_canvas",
            )

            # Extract drawn slots and persist them across reruns
            state_key = f"drawn_slots_{layout_name}"
            canvas_slots = canvas_to_normalized_slots(
                canvas_result, DISPLAY_W, display_h, img_w, img_h,
            )
            if canvas_slots:
                st.session_state[state_key] = canvas_slots

            active_slots = canvas_slots or st.session_state.get(state_key, [])

            # Slot count indicator
            if active_slots:
                st.info(f"📐 **{len(active_slots)} parking slot(s) drawn!** Click **Save Layout** to apply.")
            else:
                st.caption("Slots drawn: 0 (drag on the image above to draw rectangles)")

            # Save button
            if st.button("💾 Save Layout", type="primary"):
                slots_to_save = active_slots or canvas_to_normalized_slots(
                    canvas_result, DISPLAY_W, display_h, img_w, img_h,
                )
                if slots_to_save:
                    save_layout(layout_name, slots_to_save)
                    st.session_state[state_key] = []
                    st.session_state.edit_mode = False
                    st.success(
                        f"✅ Saved {len(slots_to_save)} parking slots as '{layout_name}'"
                    )
                    st.rerun()
                else:
                    st.warning("No slots drawn! Draw rectangles around parking spaces first.")

        # ── DETECTION MODE: Run YOLO & classify ──────────────────
        else:
            loaded_slots = load_layout(layout_name)
            if not loaded_slots:
                st.warning(
                    "⚠️ No parking layout found. "
                    "Click **Edit Layout** in the sidebar to mark parking spaces."
                )
            else:
                st.subheader("🔍 Detection Results")

                with st.spinner("Running YOLO vehicle detection..."):
                    # Run YOLO on FULL RESOLUTION image (not downsized)
                    vehicles, vehicle_confs = detect_vehicles(
                        img_array,
                        conf_thres=conf_threshold,
                        imgsz=yolo_imgsz,
                    )

                # Convert slots to pixel coords
                slots_px = slots_to_pixels(loaded_slots, img_w, img_h)

                # Classify each slot
                classified = classify_slots(
                    slots_px, vehicles, vehicle_confs,
                    occupied_threshold=occupied_threshold,
                    uncertain_threshold=uncertain_threshold,
                )

                # Statistics
                counts = count_statuses(classified)
                display_stats(counts)

                # Draw results on image
                result_img = draw_results(
                    img_array, classified, vehicles, vehicle_confs,
                )

                # Display
                st.image(
                    cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

                # Details expander
                with st.expander("📊 Slot Details"):
                    for s in classified:
                        emoji = {"free": "🟢", "occupied": "🔴", "uncertain": "🟠"
                                 }.get(s["status"], "❓")
                        ov = s.get("best_overlap", 0)
                        st.write(
                            f"{emoji} **{s['id']}** — {s['status'].upper()} "
                            f"(overlap: {ov:.1%})"
                        )

                with st.expander("🚘 Detected Vehicles"):
                    st.write(f"**{len(vehicles)}** vehicles detected")
                    for i, ((x1, y1, x2, y2), conf) in enumerate(
                        zip(vehicles, vehicle_confs)
                    ):
                        st.write(
                            f"Vehicle {i+1}: ({x1},{y1})→({x2},{y2}) "
                            f"conf={conf:.1%}"
                        )


# ======================================================================
# 🎥 VIDEO MODE
# ======================================================================
elif mode == "Video Upload":
    st.header("🎥 Video Parking Detection")

    # Layout selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Layout for Video")
    layout_name_v = st.sidebar.text_input(
        "Layout Name", value="default", key="video_layout_name",
        help="Use a layout saved from Image mode",
    )

    loaded_slots_v = load_layout(layout_name_v)

    uploaded_video = st.file_uploader(
        "Upload Parking Video", type=["mp4", "avi", "mov"],
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Cannot read video")
        else:
            v_h, v_w = first_frame.shape[:2]

            # If no layout, let user configure from first frame
            if not loaded_slots_v:
                st.warning(
                    "⚠️ No parking layout found. "
                    "Please configure one in **Image Upload** mode first, "
                    "or draw one on the first frame below."
                )
                st.subheader("📐 Configure Layout from First Frame")

                display_h = int(DISPLAY_W * v_h / v_w)
                bg = Image.fromarray(
                    cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                ).resize((DISPLAY_W, display_h))

                canvas_v = st_canvas(
                    fill_color="rgba(0, 200, 0, 0.15)",
                    stroke_width=2,
                    stroke_color="#00C800",
                    background_image=bg,
                    drawing_mode="rect",
                    height=display_h,
                    width=DISPLAY_W,
                    key="video_canvas",
                )

                state_v_key = f"drawn_slots_v_{layout_name_v}"
                canvas_v_slots = canvas_to_normalized_slots(
                    canvas_v, DISPLAY_W, display_h, v_w, v_h,
                )
                if canvas_v_slots:
                    st.session_state[state_v_key] = canvas_v_slots

                active_v_slots = canvas_v_slots or st.session_state.get(state_v_key, [])

                if active_v_slots:
                    st.info(f"📐 **{len(active_v_slots)} slot(s) drawn!** Click **Save Video Layout** to apply.")
                else:
                    st.caption("Slots drawn: 0 (drag on the image above to draw rectangles)")

                if st.button("💾 Save Video Layout", type="primary"):
                    slots_to_save = active_v_slots or canvas_to_normalized_slots(
                        canvas_v, DISPLAY_W, display_h, v_w, v_h,
                    )
                    if slots_to_save:
                        save_layout(layout_name_v, slots_to_save)
                        st.session_state[state_v_key] = []
                        st.success(f"✅ Saved {len(slots_to_save)} slots as '{layout_name_v}'")
                        st.rerun()
                    else:
                        st.warning("Draw some rectangles first!")

            else:
                # Process video with saved layout
                slots_px = slots_to_pixels(loaded_slots_v, v_w, v_h)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                stats_placeholder = st.empty()
                frame_window = st.image([])

                frame_count = 0
                YOLO_SKIP = 5
                last_vehicles = []
                last_confs = []

                # Temporal smoothing: track status history per slot
                status_history = {s["id"]: [] for s in slots_px}
                SMOOTH_WINDOW = 3

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if frame_count % YOLO_SKIP == 0:
                        vehicles, confs = detect_vehicles(
                            frame, conf_thres=conf_threshold, imgsz=yolo_imgsz,
                        )
                        last_vehicles = vehicles
                        last_confs = confs
                    else:
                        vehicles = last_vehicles
                        confs = last_confs

                    classified = classify_slots(
                        slots_px, vehicles, confs,
                        occupied_threshold=occupied_threshold,
                        uncertain_threshold=uncertain_threshold,
                    )

                    # Temporal smoothing
                    for s in classified:
                        sid = s["id"]
                        status_history[sid].append(s["status"])
                        if len(status_history[sid]) > SMOOTH_WINDOW:
                            status_history[sid].pop(0)
                        # Use most common recent status
                        recent = status_history[sid]
                        s["status"] = max(
                            set(recent), key=recent.count,
                        )

                    counts = count_statuses(classified)

                    result_frame = draw_results(
                        frame, classified, vehicles, confs,
                    )

                    with stats_placeholder.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total", counts["total"])
                        c2.metric("🟢 Free", counts["free"])
                        c3.metric("🔴 Occupied", counts["occupied"])
                        c4.metric("🟠 Uncertain", counts["uncertain"])

                    frame_window.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    time.sleep(0.03)

                cap.release()


# ======================================================================
# 📷 LIVE WEBCAM MODE
# ======================================================================
else:
    st.header("📷 Live Webcam Parking Detection")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Layout for Webcam")
    layout_name_w = st.sidebar.text_input(
        "Layout Name", value="default", key="webcam_layout_name",
    )

    loaded_slots_w = load_layout(layout_name_w)

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
        else:
            w_h, w_w = first_frame.shape[:2]

            if not loaded_slots_w:
                st.warning(
                    "⚠️ No layout found. Configure one in **Image Upload** mode first."
                )
                cap.release()
            else:
                slots_px = slots_to_pixels(loaded_slots_w, w_w, w_h)

                stats_placeholder = st.empty()
                frame_window = st.image([])

                frame_count = 0
                YOLO_SKIP = 5
                last_vehicles = []
                last_confs = []

                status_history = {s["id"]: [] for s in slots_px}
                SMOOTH_WINDOW = 3

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if frame_count % YOLO_SKIP == 0:
                        vehicles, confs = detect_vehicles(
                            frame, conf_thres=conf_threshold, imgsz=yolo_imgsz,
                        )
                        last_vehicles = vehicles
                        last_confs = confs
                    else:
                        vehicles = last_vehicles
                        confs = last_confs

                    classified = classify_slots(
                        slots_px, vehicles, confs,
                        occupied_threshold=occupied_threshold,
                        uncertain_threshold=uncertain_threshold,
                    )

                    for s in classified:
                        sid = s["id"]
                        status_history[sid].append(s["status"])
                        if len(status_history[sid]) > SMOOTH_WINDOW:
                            status_history[sid].pop(0)
                        recent = status_history[sid]
                        s["status"] = max(set(recent), key=recent.count)

                    counts = count_statuses(classified)

                    result_frame = draw_results(
                        frame, classified, vehicles, confs,
                    )

                    with stats_placeholder.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total", counts["total"])
                        c2.metric("🟢 Free", counts["free"])
                        c3.metric("🔴 Occupied", counts["occupied"])
                        c4.metric("🟠 Uncertain", counts["uncertain"])

                    frame_window.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    time.sleep(0.03)

                cap.release()
