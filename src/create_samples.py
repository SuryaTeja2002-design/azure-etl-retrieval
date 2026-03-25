# create_samples.py - creates all sample documents for testing

import os
os.makedirs("docs/manuals", exist_ok=True)
os.makedirs("docs/troubleshooting", exist_ok=True)
os.makedirs("docs/policies", exist_ok=True)

# ── 1. deviceA.pdf ────────────────────────────────────────────────────────────
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(0, 10, """Device A User Manual

Section 1: Getting Started
Device A is a smart home controller. To set up, plug the device into a
power source and press the power button for 3 seconds until the LED turns blue.

Section 2: Factory Reset
To reset Device A to factory settings, hold the reset button on the back
for 10 seconds until the LED flashes red. All settings will be erased.

Section 3: Connectivity
Device A supports WiFi 2.4GHz and 5GHz. Open the mobile app and follow
the on-screen pairing instructions. Ensure Bluetooth is enabled during setup.

Section 4: Troubleshooting
If the device does not turn on, check the power cable connection.
If the LED is red, restart the device by unplugging and replugging after 30 seconds.
""")
pdf.output("docs/manuals/deviceA.pdf")
print("created: docs/manuals/deviceA.pdf")

# ── 2. deviceB.pdf ────────────────────────────────────────────────────────────
pdf2 = FPDF()
pdf2.add_page()
pdf2.set_font("Helvetica", size=12)
pdf2.multi_cell(0, 10, """Device B Technical Manual

Section 1: Overview
Device B is an industrial sensor unit used for temperature and humidity monitoring.
It transmits data every 60 seconds over MQTT protocol to a central broker.

Section 2: Installation
Mount Device B on a flat surface using the included screws.
Connect the 12V DC power adapter. The green LED indicates normal operation.

Section 3: Calibration
To calibrate, press and hold the CAL button for 5 seconds.
The device will perform a self-test and calibrate against ambient conditions.

Section 4: Factory Reset
Hold the RESET pin with a paperclip for 8 seconds to restore factory defaults.
All stored readings and network settings will be permanently deleted.
""")
pdf2.output("docs/manuals/deviceB.pdf")
print("created: docs/manuals/deviceB.pdf")

# ── 3. error101.md ────────────────────────────────────────────────────────────
with open("docs/troubleshooting/error101.md", "w") as f:
    f.write("""---
title: Error Code 101 Troubleshooting Guide
category: troubleshooting
severity: medium
---

# Error Code 101 – Connection Timeout

## Description
Error 101 occurs when the device cannot establish a connection to the
central server within the allowed timeout window of 30 seconds.

## Common Causes
- Network firewall blocking outbound port 8883 (MQTT)
- Incorrect WiFi credentials entered during setup
- Server maintenance window in progress
- Device firmware is outdated

## Step-by-Step Resolution

### Step 1: Check Network Connectivity
Ensure your router is online and other devices can access the internet.
Restart the router by unplugging it for 30 seconds.

### Step 2: Verify Credentials
Navigate to Settings > Network > WiFi and re-enter your WiFi password.
Make sure there are no extra spaces in the password field.

### Step 3: Update Firmware
Go to Settings > System > Firmware Update and install any available updates.
The device must be connected to WiFi to download firmware.

### Step 4: Contact Support
If error 101 persists after all steps above, contact support at
support@deviceco.com with your device serial number and firmware version.
""")
print("created: docs/troubleshooting/error101.md")

# ── 4. security.txt ───────────────────────────────────────────────────────────
with open("docs/policies/security.txt", "w") as f:
    f.write("""SECURITY POLICY DOCUMENT
Version 2.3 | Effective Date: January 1, 2026

1. DATA ENCRYPTION
All data transmitted between devices and servers must use TLS 1.2 or higher.
Data at rest must be encrypted using AES-256 encryption standard.
Encryption keys must be rotated every 90 days.

2. ACCESS CONTROL
All user accounts require multi-factor authentication (MFA).
Passwords must be at least 12 characters with mixed case, numbers and symbols.
Inactive accounts are automatically disabled after 60 days.
Administrative access requires separate privileged accounts.

3. INCIDENT RESPONSE
Security incidents must be reported to security@company.com within 1 hour.
A full incident report must be filed within 24 hours of detection.
Critical vulnerabilities must be patched within 72 hours of disclosure.

4. DEVICE SECURITY
All devices must run approved and up-to-date firmware only.
Default passwords must be changed before deployment.
Devices must be registered in the central device management portal.
Physical access to devices must be logged and audited monthly.

5. DATA RETENTION
Customer data is retained for a maximum of 7 years.
Deleted data is permanently purged within 30 days of deletion request.
Backups are stored in geographically separate locations.
""")
print("created: docs/policies/security.txt")

print("\nAll sample documents created successfully!")