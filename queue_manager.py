"""
Parking Queue Manager
- Persistent FIFO waiting queue management (JSON storage)
- Atomic driver claiming to prevent duplicate notifications
- Strict status lifecycle: waiting -> processing -> notified | failed
- Active queue separation: notified drivers are moved to history
- Retry tracking and verified last-successful-notification records
"""

import os
import json
import re
from datetime import datetime
from threading import Lock

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QUEUE_FILE = os.path.join(DATA_DIR, "parking_queue.json")
_LOCK = Lock()

os.makedirs(DATA_DIR, exist_ok=True)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> bool:
    """Validate RFC 5322 compatible email format."""
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def _load_raw_data() -> dict:
    """Load raw data dictionary from disk."""
    if not os.path.exists(QUEUE_FILE):
        return {"queue": [], "last_notified": None}
    try:
        with open(QUEUE_FILE, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"queue": [], "last_notified": None}
            if "queue" not in data:
                data["queue"] = []
            return data
    except Exception:
        return {"queue": [], "last_notified": None}


def _save_raw_data(data: dict):
    """Persist data dictionary to disk."""
    with open(QUEUE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_to_queue(email: str) -> tuple[bool, str, dict | None]:
    """
    Validate and add an email to the waiting queue in FIFO order.
    Rejects invalid emails and active duplicates (waiting, processing, or failed).
    """
    clean_email = email.strip() if email else ""
    if not validate_email(clean_email):
        return False, "Invalid email address format.", None

    with _LOCK:
        data = _load_raw_data()
        queue = data["queue"]

        # Check for active duplicates
        for entry in queue:
            if entry["email"].lower() == clean_email.lower() and entry["status"] in ("waiting", "processing", "failed"):
                return False, f"Email '{clean_email}' is already in the active waiting queue ({entry['status']}).", None

        # Generate sequential Queue ID
        seq = len(queue) + 1
        queue_id = f"Q-{1000 + seq}"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "queue_id": queue_id,
            "email": clean_email,
            "created_at": now_str,
            "status": "waiting",  # waiting | processing | notified | failed
            "notified_at": None,
            "assigned_slot": None,
            "error_message": None,
            "retry_count": 0,
        }

        queue.append(new_entry)
        data["queue"] = queue
        _save_raw_data(data)
        return True, f"Added {clean_email} to waiting queue as {queue_id}.", new_entry


def get_active_queue() -> list[dict]:
    """Return active waiting queue entries (waiting, processing, failed) in FIFO order."""
    with _LOCK:
        data = _load_raw_data()
        return [e for e in data["queue"] if e["status"] in ("waiting", "processing", "failed")]


def get_notification_history() -> list[dict]:
    """Return historical successfully notified driver records in reverse chronological order."""
    with _LOCK:
        data = _load_raw_data()
        notified = [e for e in data["queue"] if e["status"] == "notified"]
        return list(reversed(notified))


def get_queue(active_only: bool = False) -> list[dict]:
    """Return queue entries. If active_only is True, excludes already notified entries."""
    if active_only:
        return get_active_queue()
    with _LOCK:
        data = _load_raw_data()
        return data["queue"]


def get_waiting_count() -> int:
    """Return count of drivers currently waiting for a vacant slot."""
    with _LOCK:
        data = _load_raw_data()
        return sum(1 for e in data["queue"] if e["status"] == "waiting")


def remove_from_queue(queue_id: str) -> bool:
    """Remove a driver entry from the queue by queue_id."""
    with _LOCK:
        data = _load_raw_data()
        initial_len = len(data["queue"])
        data["queue"] = [e for e in data["queue"] if e["queue_id"] != queue_id]
        if len(data["queue"]) < initial_len:
            _save_raw_data(data)
            return True
        return False


def claim_next_waiting_driver(slot_id: str) -> dict | None:
    """
    Atomically claim the earliest waiting driver in FIFO order.
    Marks status as 'processing' and records assigned_slot.
    """
    with _LOCK:
        data = _load_raw_data()
        for entry in data["queue"]:
            if entry["status"] == "waiting":
                entry["status"] = "processing"
                entry["assigned_slot"] = slot_id
                _save_raw_data(data)
                return entry
        return None


def mark_driver_notified(queue_id: str, slot_id: str) -> bool:
    """
    Mark a claimed driver as successfully notified by the email provider.
    Transitions status to 'notified', sets timestamp, and records as latest successful notification.
    """
    with _LOCK:
        data = _load_raw_data()
        for entry in data["queue"]:
            if entry["queue_id"] == queue_id:
                entry["status"] = "notified"
                entry["assigned_slot"] = slot_id
                entry["notified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry["error_message"] = None

                # Update confirmed last successful notification
                data["last_notified"] = {
                    "email": entry["email"],
                    "slot_id": slot_id,
                    "queue_id": queue_id,
                    "timestamp": entry["notified_at"],
                }
                _save_raw_data(data)
                return True
        return False


def mark_driver_failed(queue_id: str, error_message: str) -> bool:
    """
    Mark notification as failed.
    Sets status to 'failed', increments retry_count, and stores error reason.
    Keeps the entry in the active queue so the operator can retry or remove it.
    """
    with _LOCK:
        data = _load_raw_data()
        for entry in data["queue"]:
            if entry["queue_id"] == queue_id:
                entry["status"] = "failed"
                entry["error_message"] = str(error_message)
                entry["retry_count"] = entry.get("retry_count", 0) + 1
                _save_raw_data(data)
                return True
        return False


def retry_failed_driver(queue_id: str) -> bool:
    """Reset a failed driver back to waiting status for another attempt."""
    with _LOCK:
        data = _load_raw_data()
        for entry in data["queue"]:
            if entry["queue_id"] == queue_id:
                entry["status"] = "waiting"
                entry["assigned_slot"] = None
                entry["error_message"] = None
                _save_raw_data(data)
                return True
        return False


def get_last_successful_notification() -> dict | None:
    """Retrieve the most recent verified successful notification record from persistence."""
    with _LOCK:
        data = _load_raw_data()
        return data.get("last_notified")


def clear_all_queue() -> bool:
    """Clear all queue entries and reset state (utility for testing/resetting)."""
    with _LOCK:
        _save_raw_data({"queue": [], "last_notified": None})
        return True
