"""
CLI Diagnostic Tool: Test Gmail Configuration and Connectivity
Run: python test_gmail_connection.py [optional_recipient_email]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from email_service import get_gmail_status, log_gmail_env_status, send_parking_notification


def main():
    print("=" * 65)
    print("🚗 SMART PARKING SYSTEM - GMAIL CONNECTION DIAGNOSTIC")
    print("=" * 65)

    log_gmail_env_status()
    print("-" * 65)

    status = get_gmail_status()

    if not status["configured"]:
        print("❌ GMAIL IS NOT PROPERLY CONFIGURED.")
        print(f"Reason: {status['status_message']}")
        print("\n🔧 HOW TO FIX:")
        print("1. Open .env in your project folder.")
        print("2. Choose ONE of the following methods:")
        print("   OPTION A (Quickest - Gmail App Password):")
        print("     - Visit: https://myaccount.google.com/apppasswords")
        print("     - Enable 2-Step Verification if not already active.")
        print("     - Generate an App Password for 'Mail'.")
        print("     - Set in .env:")
        print("         GMAIL_SENDER_EMAIL=your_actual_gmail@gmail.com")
        print("         GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx (the 16 letters generated)")
        print("\n   OPTION B (OAuth 2.0):")
        print("     - Configure GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in .env.")
        print("=" * 65)
        sys.exit(1)

    print(f"✅ Configuration verified using method: {status['method'].upper()}")
    print(f"   Sender Account: {status['sender_email']}")

    recipient = sys.argv[1] if len(sys.argv) > 1 else status["sender_email"]
    print(f"\n📤 Attempting to send a test notification to: {recipient}...")

    success, message = send_parking_notification(
        recipient_email=recipient,
        slot_id="Test-Slot-1",
        queue_id="TEST-DIAGNOSTIC",
    )

    if success:
        print(f"🎉 SUCCESS: {message}")
        print("Gmail delivery integration is fully working!")
    else:
        print(f"❌ SEND FAILED: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
