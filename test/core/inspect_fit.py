"""
Inspect FIT file to find pre-calculated values from Garmin device
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from fitparse import FitFile
import gzip
import zipfile
import tempfile
import shutil

fit_path = Path(__file__).parent / "台州市 越野跑_457618021.fit"

# Handle compression
actual_fit_path = fit_path
temp_dir = None
try:
    with open(fit_path, 'rb') as f:
        header = f.read(4)

    if header[:2] == b'\x1f\x8b':
        temp_dir = tempfile.mkdtemp()
        temp_fit_file = Path(temp_dir) / fit_path.stem
        with gzip.open(fit_path, 'rb') as gz:
            with open(temp_fit_file, 'wb') as out:
                out.write(gz.read())
        actual_fit_path = temp_fit_file
        print(f"Detected: GZIP compressed FIT")
    elif header[:4] == b'PK\x03\x04':
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(fit_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
            if extracted_files:
                actual_fit_path = extracted_files[0]
                print(f"Detected: ZIP compressed FIT")
except Exception as e:
    print(f"Warning checking compression: {e}")

fitfile = FitFile(str(actual_fit_path))

print("=" * 70)
print("FIT FILE INSPECTION - Finding Pre-Calculated Values")
print("=" * 70)

# Check all message types
print("\n[1] Available Message Types:")
message_types = set()
for message in fitfile:
    message_types.add(message.type)
print(f"  Message types: {message_types}")

# Reset file
fitfile = FitFile(str(actual_fit_path))

# Check SESSION message (usually contains summary data)
print("\n[2] SESSION Message (Summary Data):")
session_data = {}
for message in fitfile.get_messages('session'):
    print(f"\n  Session Fields:")
    for field in message:
        field_name = field.name
        field_value = field.value
        field_units = field.units

        # Store important values
        if 'total' in field_name.lower() or 'ascent' in field_name.lower() or 'descent' in field_name.lower():
            session_data[field_name] = {'value': field_value, 'units': field_units}

        if field_value is not None:
            units_str = f" {field_units}" if field_units else ""
            print(f"    {field_name:30} = {field_value}{units_str}")
    break  # First session only

# Reset file
fitfile = FitFile(str(actual_fit_path))

# Check LAP messages
print("\n[3] LAP Messages:")
lap_count = 0
for message in fitfile.get_messages('lap'):
    lap_count += 1
    if lap_count == 1:  # Show first lap details
        print(f"\n  First LAP Fields:")
        for field in message:
            if field.value is not None:
                units_str = f" {field.units}" if field.units else ""
                print(f"    {field.name:30} = {field.value}{units_str}")

print(f"\n  Total LAPs: {lap_count}")

# Reset file
fitfile = FitFile(str(actual_fit_path))

# Check ACTIVITY message
print("\n[4] ACTIVITY Message:")
for message in fitfile.get_messages('activity'):
    print(f"\n  ACTIVITY Fields:")
    for field in message:
        if field.value is not None:
            units_str = f" {field.units}" if field.units else ""
            print(f"    {field.name:30} = {field.value}{units_str}")
    break

# Summary of pre-calculated values
print("\n" + "=" * 70)
print("IMPORTANT: Pre-Calculated Values from FIT File")
print("=" * 70)

if session_data:
    print("\nKey Pre-Calculated Fields:")
    for field_name, data in session_data.items():
        value = data['value']
        units = data['units']
        units_str = f" {units}" if units else ""
        print(f"  {field_name:40} = {value}{units_str}")
else:
    print("\n  No session data found - checking all record fields...")

# Cleanup
if temp_dir and os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)

print("\n" + "=" * 70)
