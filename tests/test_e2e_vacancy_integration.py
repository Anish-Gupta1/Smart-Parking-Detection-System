"""
End-to-End Simulation Test:
Simulates parking occupancy transitions, FIFO claiming, async email dispatch,
active queue separation, history archiving, error handling, and queue reload.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from queue_manager import (
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
    send_parking_notification_async,
    format_notification_body,
    EMAIL_SUBJECT,
)


class TestE2EVacancyIntegration(unittest.TestCase):
    def setUp(self):
        clear_all_queue()

    def tearDown(self):
        clear_all_queue()

    def test_e2e_vacancy_lifecycle(self):
        # 1. Lot is full (0 free spaces), 2 drivers join the waiting queue
        ok1, msg1, entry1 = add_to_queue("driver1@example.com")
        ok2, msg2, entry2 = add_to_queue("driver2@example.com")
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(get_waiting_count(), 2)
        self.assertEqual(len(get_active_queue()), 2)
        self.assertEqual(len(get_notification_history()), 0)

        # 2. Simulate webcam monitoring: Slot 1 transitions from occupied -> free
        slot_prev_status = {"1": "occupied", "2": "occupied"}
        current_status = {"1": "free", "2": "occupied"}

        # Detect transition for Slot 1
        sid = "1"
        if slot_prev_status[sid] == "occupied" and current_status[sid] == "free":
            self.assertGreater(get_waiting_count(), 0)
            claimed = claim_next_waiting_driver(slot_id=f"Slot #{sid}")
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["email"], "driver1@example.com")
            self.assertEqual(claimed["queue_id"], entry1["queue_id"])
            self.assertEqual(claimed["status"], "processing")

            # Async email notification dispatch
            with patch("email_service.send_parking_notification", return_value=(True, "Mocked Gmail success")):
                completed_events = []

                def callback(success, message, slot_lbl, qid):
                    if success:
                        mark_driver_notified(qid, slot_lbl)
                    else:
                        mark_driver_failed(qid, message)
                    completed_events.append((success, slot_lbl, qid))

                thread = send_parking_notification_async(
                    recipient_email=claimed["email"],
                    slot_id=f"Slot #{sid}",
                    queue_id=claimed["queue_id"],
                    on_complete=callback,
                )
                thread.join(timeout=2.0)

                self.assertEqual(len(completed_events), 1)
                self.assertTrue(completed_events[0][0])
                self.assertEqual(completed_events[0][1], "Slot #1")

        # 3. Verify driver 1 is removed from active queue and placed in history
        self.assertEqual(get_waiting_count(), 1)
        self.assertEqual(len(get_active_queue()), 1)
        history = get_notification_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["email"], "driver1@example.com")
        self.assertEqual(history[0]["status"], "notified")
        self.assertEqual(history[0]["assigned_slot"], "Slot #1")

        # Verified last notification is driver1
        last_notif = get_last_successful_notification()
        self.assertIsNotNone(last_notif)
        self.assertEqual(last_notif["email"], "driver1@example.com")

        # 4. Slot 2 now transitions from occupied -> free
        sid2 = "2"
        current_status["2"] = "free"
        if slot_prev_status[sid2] == "occupied" and current_status[sid2] == "free":
            claimed2 = claim_next_waiting_driver(slot_id=f"Slot #{sid2}")
            self.assertIsNotNone(claimed2)
            self.assertEqual(claimed2["email"], "driver2@example.com")

            with patch("email_service.send_parking_notification", return_value=(True, "Mocked Gmail success")):
                thread2 = send_parking_notification_async(
                    recipient_email=claimed2["email"],
                    slot_id=f"Slot #{sid2}",
                    queue_id=claimed2["queue_id"],
                    on_complete=lambda ok, msg, s, q: mark_driver_notified(q, s) if ok else mark_driver_failed(q, msg),
                )
                thread2.join(timeout=2.0)

        # 5. Active waiting queue is now 0; history has both drivers
        self.assertEqual(get_waiting_count(), 0)
        self.assertEqual(len(get_active_queue()), 0)
        history_all = get_notification_history()
        self.assertEqual(len(history_all), 2)

        # Verified last notification is driver2
        last_notif2 = get_last_successful_notification()
        self.assertEqual(last_notif2["email"], "driver2@example.com")
        self.assertEqual(last_notif2["slot_id"], "Slot #2")

        # 6. Another slot becomes free when queue is empty: no claim
        claimed3 = claim_next_waiting_driver(slot_id="Slot #3")
        self.assertIsNone(claimed3)

    def test_failed_notification_preservation_and_retry(self):
        ok, _, entry = add_to_queue("failuser@example.com")
        self.assertTrue(ok)

        claimed = claim_next_waiting_driver(slot_id="Slot #5")
        self.assertIsNotNone(claimed)

        # Simulate failure
        with patch("email_service.send_parking_notification", return_value=(False, "Simulated network timeout")):
            def callback(success, message, slot_lbl, qid):
                if success:
                    mark_driver_notified(qid, slot_lbl)
                else:
                    mark_driver_failed(qid, message)

            thread = send_parking_notification_async(
                recipient_email=claimed["email"],
                slot_id="Slot #5",
                queue_id=claimed["queue_id"],
                on_complete=callback,
            )
            thread.join(timeout=2.0)

        # Verify entry remains in active queue with status 'failed' and error message
        active = get_active_queue()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], "failed")
        self.assertEqual(active[0]["retry_count"], 1)
        self.assertIn("Simulated network timeout", active[0]["error_message"])

        # History must NOT contain failed entry
        self.assertEqual(len(get_notification_history()), 0)

        # Operator retries
        retry_ok = retry_failed_driver(entry["queue_id"])
        self.assertTrue(retry_ok)
        self.assertEqual(get_waiting_count(), 1)
        active_after = get_active_queue()
        self.assertEqual(active_after[0]["status"], "waiting")
        self.assertIsNone(active_after[0]["error_message"])


if __name__ == "__main__":
    unittest.main()
