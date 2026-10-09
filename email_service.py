"""
Gmail Notification Backend Service
- Official Google Gmail API v1 with OAuth 2.0
- Secure SMTP fallback (port 587 TLS / 465 SSL) using Gmail App Passwords
- Automatic placeholder detection for environment variables
- Structured, sanitized server-side diagnostics (never prints secret tokens/passwords)
- Asynchronous, non-blocking notification dispatch for webcam streams
- Standardized email template with slot ID and queue reference
"""

import os
import re
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from typing import Callable
from dotenv import load_dotenv

# Ensure .env is reliably loaded using absolute path
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE, override=True)

EMAIL_SUBJECT = "Parking Slot Available - Smart Parking System"
EMAIL_BODY_TEMPLATE = """Hello,

A parking slot is now available for you.

Parking Slot: {slot_id}
Queue Reference: {queue_id}

Please proceed to the parking area and follow the parking attendant's instructions.

This notification was generated automatically by the Smart Parking System."""

PLACEHOLDER_SUBSTRINGS = (
    "your_",
    "your-",
    "placeholder",
    "example",
    "16_digit",
    "oauth_refresh",
    "<",
    ">",
    "xxx",
)


def is_placeholder_or_empty(val: str | None) -> bool:
    """Return True if the value is None, empty, or contains a placeholder string."""
    if not val or not isinstance(val, str):
        return True
    cleaned = val.strip().lower()
    if not cleaned:
        return True
    return any(p in cleaned for p in PLACEHOLDER_SUBSTRINGS)


def format_notification_body(slot_id: str, queue_id: str) -> str:
    """Format the official email body template."""
    return EMAIL_BODY_TEMPLATE.format(slot_id=slot_id, queue_id=queue_id)


def get_gmail_status() -> dict:
    """
    Inspect environment variables and return configuration diagnostic.
    Sanitized: NEVER exposes secret values.
    """
    # Load .env without overwriting existing environment variables
    load_dotenv(ENV_FILE, override=False)

    client_id = os.getenv("GMAIL_CLIENT_ID", "")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET", "")
    refresh_token = os.getenv("GMAIL_REFRESH_TOKEN", "")
    sender_email = os.getenv("GMAIL_SENDER_EMAIL", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    has_sender = bool(sender_email) and not is_placeholder_or_empty(sender_email)

    # Check OAuth 2.0 readiness
    oauth_missing = []
    oauth_placeholders = []
    for var_name, var_val in [
        ("GMAIL_CLIENT_ID", client_id),
        ("GMAIL_CLIENT_SECRET", client_secret),
        ("GMAIL_REFRESH_TOKEN", refresh_token),
    ]:
        if not var_val:
            oauth_missing.append(var_name)
        elif is_placeholder_or_empty(var_val):
            oauth_placeholders.append(var_name)

    oauth_ready = has_sender and not oauth_missing and not oauth_placeholders

    # Check SMTP App Password readiness
    smtp_missing = []
    smtp_placeholders = []
    if not sender_email:
        smtp_missing.append("GMAIL_SENDER_EMAIL")
    elif is_placeholder_or_empty(sender_email):
        smtp_placeholders.append("GMAIL_SENDER_EMAIL")

    if not app_password:
        smtp_missing.append("GMAIL_APP_PASSWORD")
    elif is_placeholder_or_empty(app_password):
        smtp_placeholders.append("GMAIL_APP_PASSWORD")

    smtp_ready = not smtp_missing and not smtp_placeholders

    if oauth_ready:
        return {
            "configured": True,
            "method": "oauth",
            "sender_email": sender_email,
            "missing_vars": [],
            "placeholder_vars": [],
            "status_message": f"Gmail OAuth 2.0 configured (Sender: {sender_email})",
        }
    elif smtp_ready:
        return {
            "configured": True,
            "method": "smtp",
            "sender_email": sender_email,
            "missing_vars": [],
            "placeholder_vars": [],
            "status_message": f"Gmail SMTP App Password configured (Sender: {sender_email})",
        }
    else:
        # Determine why it is not ready
        missing = sorted(list(set(oauth_missing if oauth_missing else smtp_missing)))
        placeholders = sorted(list(set(oauth_placeholders + smtp_placeholders)))

        details = []
        if placeholders:
            details.append(f"Placeholders detected in: {', '.join(placeholders)}")
        if missing:
            details.append(f"Missing variables: {', '.join(missing)}")

        status_msg = "; ".join(details) if details else "No Gmail credentials configured in .env."

        return {
            "configured": False,
            "method": "none",
            "sender_email": sender_email if has_sender else None,
            "missing_vars": missing,
            "placeholder_vars": placeholders,
            "status_message": status_msg,
        }


def log_gmail_env_status():
    """Print structured, sanitized server-side log of email configuration."""
    st = get_gmail_status()
    print(f"[GMAIL-DIAGNOSTIC] Status: {'READY' if st['configured'] else 'NOT CONFIGURED'} | Method: {st['method'].upper()}")
    print(f"[GMAIL-DIAGNOSTIC] Sender: {st['sender_email'] if st['sender_email'] else 'Not specified'}")
    if st["placeholder_vars"]:
        print(f"[GMAIL-DIAGNOSTIC] Placeholder variables: {', '.join(st['placeholder_vars'])}")
    if st["missing_vars"]:
        print(f"[GMAIL-DIAGNOSTIC] Missing variables: {', '.join(st['missing_vars'])}")
    print(f"[GMAIL-DIAGNOSTIC] Message: {st['status_message']}")


def _send_via_gmail_api(recipient_email: str, subject: str, body: str) -> tuple[bool, str]:
    """Send email using official Google Gmail API v1 with OAuth 2.0."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        return False, "Google API client packages (google-api-python-client, google-auth) not installed."

    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")
    refresh_token = os.getenv("GMAIL_REFRESH_TOKEN")
    sender_email = os.getenv("GMAIL_SENDER_EMAIL", "me")

    if is_placeholder_or_empty(client_id) or is_placeholder_or_empty(client_secret) or is_placeholder_or_empty(refresh_token):
        return False, "Gmail OAuth 2.0 credentials contain unconfigured placeholder values in .env."

    try:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        )

        service = build("gmail", "v1", credentials=creds, cache_discovery=False)

        message = MIMEText(body)
        message["to"] = recipient_email
        message["from"] = sender_email
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        send_result = service.users().messages().send(userId="me", body={"raw": raw_message}).execute()

        msg_id = send_result.get("id", "confirmed")
        print(f"[GMAIL-API SUCCESS] Email sent to {recipient_email}. Message ID: {msg_id}")
        return True, f"Gmail API sent successfully (Message ID: {msg_id})"
    except Exception as e:
        err_msg = str(e)
        print(f"[GMAIL-API ERROR] Failed sending to {recipient_email}: {err_msg}")
        return False, f"Gmail API failed: {err_msg}"


def _send_via_smtp_fallback(recipient_email: str, subject: str, body: str) -> tuple[bool, str]:
    """Send email using secure Gmail SMTP with App Password."""
    sender_email = os.getenv("GMAIL_SENDER_EMAIL", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    if is_placeholder_or_empty(sender_email) or is_placeholder_or_empty(app_password):
        return False, "Gmail SMTP credentials contain unconfigured placeholder values in .env (GMAIL_APP_PASSWORD)."

    # Remove any spaces that users might copy from Google's 4-character grouping
    clean_app_password = app_password.replace(" ", "")

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Try Port 587 (TLS) first, then Port 465 (SSL)
    last_err = ""
    for port, use_ssl in [(587, False), (465, True)]:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=12) as server:
                    server.login(sender_email, clean_app_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP("smtp.gmail.com", port, timeout=12) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(sender_email, clean_app_password)
                    server.send_message(msg)

            print(f"[GMAIL-SMTP SUCCESS] Email sent to {recipient_email} via port {port}")
            return True, f"Sent email to {recipient_email} via Gmail SMTP (port {port})"
        except smtplib.SMTPAuthenticationError as auth_err:
            msg_auth = (
                f"Gmail SMTP authentication failed: {auth_err.smtp_error.decode() if hasattr(auth_err, 'smtp_error') else str(auth_err)}. "
                "Ensure GMAIL_APP_PASSWORD is a valid 16-character Google App Password (not your primary password) "
                "and 2-Step Verification is active on the Google Account."
            )
            print(f"[GMAIL-SMTP ERROR] {msg_auth}")
            return False, msg_auth
        except Exception as e:
            last_err = str(e)
            continue

    print(f"[GMAIL-SMTP ERROR] Failed to send via SMTP: {last_err}")
    return False, f"Gmail SMTP failed: {last_err}"


def send_parking_notification(recipient_email: str, slot_id: str, queue_id: str) -> tuple[bool, str]:
    """
    Synchronous notification dispatcher.
    Validates configuration, sends via OAuth or SMTP, and returns (success, message).
    """
    subject = EMAIL_SUBJECT
    body = format_notification_body(slot_id=slot_id, queue_id=queue_id)

    status = get_gmail_status()
    if not status["configured"]:
        err_msg = f"Gmail not configured: {status['status_message']}"
        print(f"[GMAIL-DISPATCH REJECTED] {err_msg}")
        return False, err_msg

    if status["method"] == "oauth":
        success, msg = _send_via_gmail_api(recipient_email, subject, body)
        if success:
            return True, msg
        # If OAuth failed and SMTP is configured, try fallback
        if not is_placeholder_or_empty(os.getenv("GMAIL_APP_PASSWORD")):
            print("[GMAIL-DISPATCH] OAuth failed, attempting SMTP fallback...")
            return _send_via_smtp_fallback(recipient_email, subject, body)
        return False, msg

    if status["method"] == "smtp":
        return _send_via_smtp_fallback(recipient_email, subject, body)

    return False, f"No viable email transport available: {status['status_message']}"


def send_parking_notification_async(
    recipient_email: str,
    slot_id: str,
    queue_id: str,
    on_complete: Callable[[bool, str, str, str], None] | None = None,
):
    """
    Non-blocking asynchronous notification dispatcher.
    Runs in a background thread so real-time webcam frame processing is not delayed.
    Calls on_complete(success, message, slot_id, queue_id) upon completion.
    """
    def _worker():
        try:
            success, message = send_parking_notification(recipient_email, slot_id, queue_id)
        except Exception as e:
            success = False
            message = f"Unexpected error during email dispatch: {str(e)}"

        if on_complete:
            try:
                on_complete(success, message, slot_id, queue_id)
            except Exception as cb_err:
                print(f"[GMAIL-CALLBACK ERROR] Exception in on_complete callback: {cb_err}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
