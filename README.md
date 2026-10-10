# 🚗 AI Smart Parking Detection System

An AI-powered smart parking management system that detects parking slot occupancy in real time from images, recorded videos, and live webcam feeds using Computer Vision and YOLOv8. When a parking lot reaches full capacity, drivers can join an automated waiting queue via the dashboard. When a parking space becomes available, the system automatically selects the next eligible driver in FIFO order and dispatches an email notification via Google Gmail.

---

## 🌟 Key Features

### 1. Vision & Occupancy Detection
- **Multi-Input Support**: Analyze parking spaces using **Images**, **Recorded Surveillance Videos**, or **Live Webcams**.
- **Interactive Slot Annotator**: Directly draw and configure custom parking slot geometries on live webcam footage or uploaded images using an interactive canvas.
- **YOLOv8 Vehicle Detection**: Real-time vehicle detection (cars, motorcycles, buses, trucks) powered by Ultralytics YOLOv8 with overhead/aerial perspective optimization.
- **Hybrid Secondary Classification**: Analyzes slot crops using texture and saturation metrics (Laplacian variance and HSV analysis) to accurately identify parked vehicles even under harsh lighting, shadows, or unusual angles.
- **Temporal Debouncing**: Majority-vote smoothing (`SMOOTH_WINDOW`) prevents frame-to-frame flicker and false status transitions caused by camera noise or passing pedestrians.
- **Strict Invariant Accounting**: Guarantees that `Total Slots = Free + Occupied + Uncertain` at all times.

### 2. Driver Waiting Queue Management
- **Dashboard Queue Management**: Operators can enroll waiting drivers by entering their email address directly from the dashboard.
- **RFC 5322 Format Validation**: Validates driver email syntax prior to acceptance.
- **Duplicate Prevention**: Prevents duplicate active queue entries for the same email address.
- **FIFO Order Processing**: Enforces First-In, First-Out (FIFO) queue priority.
- **Tabbed Queue Views**:
  - **Active Waiting Queue**: Displays queued drivers, queue position (`#1`, `#2`, ...), timestamps, delivery status, assigned slots, and retry counts.
  - **Notification History**: Archives all successfully notified drivers with confirmed delivery timestamps and assigned slot numbers.
- **Persistent Storage**: Queue entries and historical records are persistently stored in `data/parking_queue.json`, surviving server restarts and page refreshes.
- **Operator Controls**: Allows manual removal of drivers and immediate retry for failed email deliveries.

### 3. Automated Gmail Notifications
- **Confirmed Vacancy Trigger**: Automatically detects when a previously occupied parking space transitions to vacant.
- **Atomic Driver Claiming**: Claims the earliest waiting driver and assigns the slot before dispatching the email, preventing race conditions.
- **Official Gmail API & SMTP Support**:
  - **Google Cloud Gmail API v1**: Official Google OAuth 2.0 integration via `google-api-python-client`.
  - **Secure SMTP Fallback**: Supports Gmail App Passwords over TLS (Port 587) and SSL (Port 465).
- **Non-Blocking Asynchronous Dispatch**: Email transmission runs on a background daemon thread, ensuring webcam detection and video playback remain completely uninterrupted.
- **Automatic Diagnostics**: Built-in environment variable inspection detects missing or placeholder credentials and reports sanitized diagnostic logs to the terminal.
- **Verified Notification Reporting**: The dashboard displays confirmation messages only after the email provider accepts delivery.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python`)
- **Object Detection**: Ultralytics YOLOv8 (`yolov8n.pt`, `yolov8s.pt`)
- **Dashboard Framework**: Streamlit
- **Canvas Annotation**: `streamlit-drawable-canvas`
- **Email Services**: Google Gmail API v1 (`google-api-python-client`, `google-auth-oauthlib`) & Python `smtplib`
- **Configuration & Storage**: `python-dotenv`, JSON persistence

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard UI                    │
│   (Mode Select | Live Video/Webcam | Slot Canvas | Queue)     │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
       [Frames / Images]             [Driver Enrollments]
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│    Vehicle Detection Engine  │ │     Queue Manager (FIFO)    │
│  - YOLOv8 Inference          │ │  - RFC Email Validation     │
│  - Aerial Optimization       │ │  - Duplicate Prevention     │
│  - Secondary Crop Classifier │ │  - Atomic Driver Claiming   │
└──────────────┬───────────────┘ │  - Status: waiting/notified │
               │                 └─────────────┬───────────────┘
     [Classified Slots]                        │
               │                               │
               ▼                               │
┌──────────────────────────────┐               │
│   Temporal Smoothing Window  │               │
│  - Majority Vote Smoothing   │               │
│  - Confirmed Vacancy Event   │               │
└──────────────┬───────────────┘               │
               │                               │
   [Occupied -> Free Transition]               │
               │                               │
               └───────────────┬───────────────┘
                               │
                       [Claim Next Driver]
                               │
                               ▼
               ┌───────────────────────────────┐
               │  Async Gmail Notification     │
               │  - Google Gmail API v1 / SMTP │
               │  - Non-blocking daemon worker │
               │  - Template with Slot & Q-ID  │
               └───────────────────────────────┘
```

---

## 📁 Project Structure

```text
ai-parking-system/
├── dashboard.py                  # Main interactive Streamlit application
├── parking_detector.py           # YOLOv8 vehicle detection pipeline & aerial tuning
├── slot_manager.py               # Slot coordinate scaling, IoU overlap & crop classifier
├── queue_manager.py              # FIFO waiting queue engine & persistence (JSON)
├── email_service.py              # Gmail API & SMTP notification service (async)
├── test_gmail_connection.py      # CLI diagnostic tool for Gmail credentials
├── auto_slot_detector.py         # Contour-based automatic slot detector helper
├── phase6_video_ultra_smooth_gpu.py # Standalone OpenCV video test script
├── requirements.txt              # Project Python dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore file (protects credentials & cache)
├── data/
│   └── parking_queue.json        # Persistent queue data and notification history
├── layouts/                      # Saved parking slot layout configurations
└── tests/
    ├── test_slot_manager.py      # Regression tests for geometry & classification
    ├── test_queue_and_email.py   # Unit tests for queue lifecycle & email formatting
    └── test_e2e_vacancy_integration.py # End-to-end simulated vacancy workflow tests
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-parking-system
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration (Gmail Notifications)

The system supports two methods for sending automatic email notifications: **Gmail App Password (SMTP)** or **Google Cloud OAuth 2.0 (Gmail API)**.

Create a `.env` file in the project root by copying the template:
```bash
cp .env.example .env
```

### Option A: Gmail App Password (Quickest Setup)
1. Go to your Google Account: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Ensure **2-Step Verification** is active on your account.
3. Under **App passwords**, create a new password (e.g., named `SmartParking`).
4. Copy the generated 16-character password and update your `.env`:
   ```env
   GMAIL_SENDER_EMAIL=your_email@gmail.com
   GMAIL_APP_PASSWORD=abcdefghijklmnop
   ```

### Option B: Google Cloud OAuth 2.0 (Official Gmail API)
1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Gmail API** under **APIs & Services > Library**.
3. Create OAuth 2.0 Credentials (Desktop Application) with scope `https://www.googleapis.com/auth/gmail.send`.
4. Add the credentials to `.env`:
   ```env
   GMAIL_CLIENT_ID=your_client_id.apps.googleusercontent.com
   GMAIL_CLIENT_SECRET=your_client_secret
   GMAIL_REFRESH_TOKEN=your_oauth_refresh_token
   GMAIL_SENDER_EMAIL=your_email@gmail.com
   ```

### Verify Email Configuration
Run the built-in diagnostic CLI tool to verify that your credentials work without starting the full dashboard:
```bash
python test_gmail_connection.py your_email@gmail.com
```

---

## 💻 Running the Application

### Launch the Dashboard
```bash
streamlit run dashboard.py
```
The application will open in your default browser at `http://localhost:8501`.

### Operating Modes

1. **📷 Live Webcam Mode**:
   - **Step 1**: The live camera initializes. Draw rectangles directly over each parking space on the video feed and click **Confirm Webcam Layout**.
   - **Step 2**: Start the live detection stream. The system classifies slots as Free (Green) or Occupied (Red).
   - **Waiting Queue**: When all slots are full, add waiting drivers by entering their email address. When any vehicle departs, the system detects the vacancy, assigns the slot to the first driver in the queue, and automatically dispatches a Gmail notification.

2. **🎥 Video Upload Mode**:
   - Upload an MP4, AVI, or MOV surveillance video.
   - Mark parking slots on the initial video frame and run playback to monitor occupancy and vacancy transitions frame by frame.

3. **🖼️ Image Upload Mode**:
   - Upload a parking lot image (JPG, PNG).
   - Draw parking spaces to evaluate vehicle detection, overlap ratios, and secondary crop analysis.

---

## 🧪 Running the Test Suite

The project includes unit and integration tests covering geometry, YOLO integration, queue transitions, and notification dispatch:

```bash
# Run all unit and integration tests
python -m unittest discover -s tests -p "test_*.py"

# Run slot geometry and classifier regression tests
python tests/test_slot_manager.py
```

---

## 📄 Notification Template

When a slot becomes vacant, the claimed driver receives an email formatted as follows:

```text
Subject: Parking Slot Available - Smart Parking System

Hello,

A parking slot is now available for you.

Parking Slot: Slot #1
Queue Reference: Q-1001

Please proceed to the parking area and follow the parking attendant's instructions.

This notification was generated automatically by the Smart Parking System.
```
