"""
Unit Tests for Queue Manager and Email Notification Service:
- Email validation
- FIFO queue ordering
- Active queue separation and notification history
- Duplicate active entry prevention
- Atomic claiming
- Status transitions: waiting -> processing -> notified / failed
- Verified last-successful-notification tracking
- Placeholder credential detection and diagnostic reporting
- Email template formatting
- Non-blocking async notification dispatch
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queue_manager import (
    validate_email,
    add_to_queue,
    get_queue,
    get_active_queue,
    get_notification_history,
    get_waiting_count,
    remove_from_queue,
    claim_next_waiting_driver,
    mark_driver_notified,
    mark_driver_failed,
    retry_failed_driver,
    get_last_successful_notification,
    clear_all_queue,
)
from email_service import (
    format_notification_body,
    send_parking_notification,
    send_parking_notification_async,
    get_gmail_status,
    is_placeholder_or_empty,
)


class TestQueueManager(unittest.TestCase):
    def setUp(self):
        clear_all_queue()

    def tearDown(self):
        clear_all_queue()

    def test_email_validation(self):
        self.assertTrue(validate_email("user@example.com"))
        self.assertTrue(validate_email("driver.123+tag@gmail.com"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertFalse(validate_email("@missinguser.com"))
        self.assertFalse(validate_email("missingdomain@"))
        self.assertFalse(validate_email(""))

    def test_fifo_ordering(self):
        ok1, _, e1 = add_to_queue("first@example.com")
        ok2, _, e2 = add_to_queue("second@example.com")
        ok3, _, e3 = add_to_queue("third@example.com")

        self.assertTrue(ok1 and ok2 and ok3)
        self.assertEqual(get_waiting_count(), 3)

        q = get_active_queue()
        self.assertEqual(q[0]["email"], "first@example.com")
        self.assertEqual(q[1]["email"], "second@example.com")
        self.assertEqual(q[2]["email"], "third@example.com")

        # Claiming should follow strict FIFO order
        claimed = claim_next_waiting_driver("P1")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["email"], "first@example.com")
        self.assertEqual(claimed["assigned_slot"], "P1")
        self.assertEqual(claimed["status"], "processing")
        self.assertEqual(get_waiting_count(), 2)

    def test_duplicate_prevention(self):
        ok1, _, _ = add_to_queue("driver@example.com")
        self.assertTrue(ok1)

        # Attempt to add duplicate while waiting
        ok2, msg, _ = add_to_queue("driver@example.com")
        self.assertFalse(ok2)
        self.assertIn("already in the active waiting queue", msg)
        self.assertEqual(get_waiting_count(), 1)

    def test_status_lifecycle_and_retry(self):
        add_to_queue("driver@example.com")
        claimed = claim_next_waiting_driver("P3")
        qid = claimed["queue_id"]

        # Mark failed: increments retry_count and stores error
        mark_driver_failed(qid, "SMTP Timeout")
        active = get_active_queue()
        self.assertEqual(active[0]["status"], "failed")
        self.assertEqual(active[0]["error_message"], "SMTP Timeout")
        self.assertEqual(active[0]["retry_count"], 1)
        self.assertEqual(get_waiting_count(), 0)

        # Retry resets status to waiting
        retry_failed_driver(qid)
        self.assertEqual(get_waiting_count(), 1)
        active2 = get_active_queue()
        self.assertEqual(active2[0]["status"], "waiting")

        # Claim again and mark notified
        claimed2 = claim_next_waiting_driver("P4")
        self.assertEqual(claimed2["queue_id"], qid)
        mark_driver_notified(qid, "P4")

        # Driver should now be removed from active queue and placed in history
        self.assertEqual(len(get_active_queue()), 0)
        history = get_notification_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "notified")
        self.assertEqual(history[0]["assigned_slot"], "P4")

        # Verified last notification is tracked
        last_notif = get_last_successful_notification()
        self.assertIsNotNone(last_notif)
        self.assertEqual(last_notif["email"], "driver@example.com")
        self.assertEqual(last_notif["slot_id"], "P4")

    def test_remove_from_queue(self):
        add_to_queue("driver1@example.com")
        add_to_queue("driver2@example.com")
        q = get_active_queue()
        qid1 = q[0]["queue_id"]

        removed = remove_from_queue(qid1)
        self.assertTrue(removed)
        self.assertEqual(len(get_active_queue()), 1)
        self.assertEqual(get_active_queue()[0]["email"], "driver2@example.com")


class TestEmailService(unittest.TestCase):
    def test_template_formatting(self):
        body = format_notification_body("P7", "Q-1005")
        self.assertIn("Parking Slot: P7", body)
        self.assertIn("Queue Reference: Q-1005", body)
        self.assertIn("Smart Parking System", body)

    def test_placeholder_detection(self):
        self.assertTrue(is_placeholder_or_empty("your_16_digit_app_password"))
        self.assertTrue(is_placeholder_or_empty("your_oauth_refresh_token"))
        self.assertTrue(is_placeholder_or_empty(""))
        self.assertTrue(is_placeholder_or_empty(None))
        self.assertFalse(is_placeholder_or_empty("valid_token_xyz_9988"))
        self.assertFalse(is_placeholder_or_empty("kakashi.hunter1@gmail.com"))

    def test_diagnostic_reports_placeholders(self):
        with patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_EMAIL": "kakashi.hunter1@gmail.com",
                "GMAIL_APP_PASSWORD": "your_16_digit_app_password",
                "GMAIL_CLIENT_ID": "cid.apps.googleusercontent.com",
                "GMAIL_CLIENT_SECRET": "sec",
                "GMAIL_REFRESH_TOKEN": "your_oauth_refresh_token",
            },
            clear=True,
        ):
            st = get_gmail_status()
            self.assertFalse(st["configured"])
            self.assertIn("GMAIL_APP_PASSWORD", st["placeholder_vars"])
            self.assertIn("GMAIL_REFRESH_TOKEN", st["placeholder_vars"])

    @patch("email_service._send_via_smtp_fallback")
    def test_send_notification_mock_success(self, mock_smtp):
        mock_smtp.return_value = (True, "Message sent successfully")
        with patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_EMAIL": "sender@gmail.com",
                "GMAIL_APP_PASSWORD": "validapppassword16",
            },
            clear=True,
        ):
            success, msg = send_parking_notification("test@driver.com", "P2", "Q-1001")
            self.assertTrue(success)
            self.assertEqual(msg, "Message sent successfully")

    @patch("email_service.send_parking_notification")
    def test_send_async_non_blocking(self, mock_send):
        mock_send.return_value = (True, "Async sent")
        completed = []

        def on_done(success, message, slot_id, queue_id):
            completed.append((success, message, slot_id, queue_id))

        thread = send_parking_notification_async("async@driver.com", "P5", "Q-1002", on_complete=on_done)
        thread.join(timeout=3.0)

        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0][0])
        self.assertEqual(completed[0][2], "P5")
        self.assertEqual(completed[0][3], "Q-1002")


if __name__ == "__main__":
    unittest.main()
