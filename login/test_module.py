"""Test the garmin_login module.

Usage:
    Set environment variables before running:
        $env:GARMIN_EMAIL = "your@email.com"
        $env:GARMIN_PASSWORD = "your_password"
    Or the script will prompt for them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from login.garmin_login import garmin_login

EMAIL = os.getenv("GARMIN_EMAIL", "")
PASSWORD = os.getenv("GARMIN_PASSWORD", "")

if not EMAIL or not PASSWORD:
    EMAIL = input("Garmin email: ").strip()
    PASSWORD = input("Garmin password: ").strip()


def test_credential_login():
    """First login with credentials (saves tokens to project tokens/ dir)."""
    print("=== Test 1: Credential login ===")
    garmin = garmin_login(email=EMAIL, password=PASSWORD)
    print(f"  Display: {garmin.display_name}")
    print(f"  Name:    {garmin.full_name}")
    print("  PASS\n")
    return garmin


def test_token_restore():
    """Restore from saved tokens (no credentials)."""
    print("=== Test 2: Token restore (no password) ===")
    garmin = garmin_login()  # no email/password
    print(f"  Display: {garmin.display_name}")
    print(f"  Name:    {garmin.full_name}")
    print("  PASS\n")
    return garmin


def test_api_call(garmin: "Garmin"):
    """Verify the session actually works by calling an API."""
    from datetime import date
    print("=== Test 3: API call ===")
    try:
        summary = garmin.get_user_summary(date.today().isoformat())
        steps = summary.get("totalSteps", 0)
        cal = summary.get("totalKilocalories", 0)
        print(f"  Steps today: {steps}")
        print(f"  Calories:    {cal:.0f} kcal")
        print("  PASS\n")
    except Exception as e:
        print(f"  FAIL: {e}\n")


if __name__ == "__main__":
    garmin = test_credential_login()
    test_token_restore()
    test_api_call(garmin)
