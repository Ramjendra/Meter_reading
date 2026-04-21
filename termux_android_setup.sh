#!/data/data/com.termux/files/usr/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Analog Meter Reader — Android setup via Termux (fully offline after setup)
#
# Requirements on the phone:
#   1. Install Termux from F-Droid (NOT Play Store):
#      https://f-droid.org/packages/com.termux/
#   2. Run this script once (needs internet for setup only)
#   3. After setup, the meter_reader.py runs completely offline
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=== Updating Termux packages ==="
pkg update -y && pkg upgrade -y

echo "=== Installing Python ==="
pkg install python -y

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install moondream pillow

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Copy your meter images to Termux storage, then run:"
echo ""
echo "  # Download model once (needs internet, ~700 MB for 0_5b)"
echo "  python meter_reader.py --model 0_5b --dir /sdcard/DCIM/meters/"
echo ""
echo "  # After first run the model is cached — no internet needed"
echo "  python meter_reader.py --model 0_5b 1.jpeg 2.jpeg 3.jpeg 4.jpeg"
echo ""
echo "  # Allow Termux storage access if needed:"
echo "  termux-setup-storage"
