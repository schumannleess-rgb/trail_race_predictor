"""Inspect FIT file to find pre-calculated values from Garmin device"""
import sys
import gzip
import zipfile
import tempfile
import shutil
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fitparse import FitFile

fit_path = Path(r'd:\Garmin\garmin-fitness-v3\projects\race_predictor\trail_race_predictor_v1.2.1\example\台州市 越野跑_457618021.fit')

print("=" * 70)
print("FIT FILE INSPECTION - Finding Pre-Calculated Values")
print("=" * 70)

# Check file header
with open(fit_path, 'rb') as f:
    header = f.read(4)
    print(f"\n[0] File Header Analysis:")
    print(f"    First 4 bytes (hex): {header.hex()}")
    print(f"    First 2 bytes: {header[:2]}")
    
    if header[:2] == b'\x1f\x8b':
        print(f"    Detected: GZIP compressed file")
    elif header[:4] == b'PK\x03\x04':
        print(f"    Detected: ZIP compressed file")
    elif header[:2] == b'.F':
        print(f"    Detected: Raw FIT file")
    else:
        print(f"    Detected: Unknown format")

# Handle compression
actual_fit_path = fit_path
temp_dir = None

try:
    if header[:2] == b'\x1f\x8b':
        temp_dir = tempfile.mkdtemp()
        temp_fit_file = Path(temp_dir) / fit_path.stem
        with gzip.open(fit_path, 'rb') as gz:
            with open(temp_fit_file, 'wb') as out:
                out.write(gz.read())
        actual_fit_path = temp_fit_file
        print(f"\n    Decompressed to: {temp_fit_file}")
    elif header[:4] == b'PK\x03\x04':
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(fit_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
            if extracted_files:
                actual_fit_path = extracted_files[0]
                print(f"\n    Extracted to: {actual_fit_path}")
except Exception as e:
    print(f"\n    Error decompressing: {e}")

# Parse FIT file
try:
    fitfile = FitFile(str(actual_fit_path))

    # Check all message types
    print("\n[1] Available Message Types:")
    message_types = set()
    for message in fitfile:
        message_types.add(message.type)
    print(f"    Message types: {sorted(message_types)}")

    # Reset file
    fitfile = FitFile(str(actual_fit_path))

    # Check SESSION message
    print("\n[2] SESSION Message (Summary Data):")
    session_data = {}
    for message in fitfile.get_messages('session'):
        print(f"\n    Session Fields:")
        for field in message:
            field_name = field.name
            field_value = field.value
            field_units = field.units
            if 'total' in field_name.lower() or 'ascent' in field_name.lower() or 'descent' in field_name.lower() or 'elapsed' in field_name.lower() or 'timer' in field_name.lower():
                session_data[field_name] = {'value': field_value, 'units': field_units}
            if field_value is not None:
                units_str = f" {field_units}" if field_units else ""
                print(f"      {field_name:30} = {field_value}{units_str}")
        break

    # Reset file
    fitfile = FitFile(str(actual_fit_path))

    # Check LAP messages count
    print("\n[3] LAP Messages:")
    lap_count = 0
    for message in fitfile.get_messages('lap'):
        lap_count += 1
    print(f"    Total LAPs: {lap_count}")

    # Reset file
    fitfile = FitFile(str(actual_fit_path))

    # Check RECORD messages count
    print("\n[4] RECORD Messages:")
    record_count = 0
    for message in fitfile.get_messages('record'):
        record_count += 1
    print(f"    Total RECORDs: {record_count}")

    # Summary
    print("\n" + "=" * 70)
    print("IMPORTANT: Pre-Calculated Values from FIT File")
    print("=" * 70)

    if session_data:
        print("\nKey Pre-Calculated Fields:")
        for field_name, data in session_data.items():
            value = data['value']
            units = data['units']
            units_str = f" {units}" if units else ""
            print(f"    {field_name:40} = {value}{units_str}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    required_fields = ['total_ascent', 'total_descent', 'total_distance', 'total_elapsed_time', 'total_timer_time']
    found_fields = [f for f in required_fields if f in session_data]
    missing_fields = [f for f in required_fields if f not in session_data]

    print(f"\n    Required fields found: {len(found_fields)}/{len(required_fields)}")
    if found_fields:
        print(f"    Found: {found_fields}")
    if missing_fields:
        print(f"    Missing: {missing_fields}")

    if len(found_fields) == len(required_fields):
        print("\n    [PASS] FIT file contains all required pre-calculated values")
        
        # Calculate rest ratio
        elapsed = session_data.get('total_elapsed_time', {}).get('value', 0)
        timer = session_data.get('total_timer_time', {}).get('value', 0)
        if elapsed > 0:
            rest_ratio = (elapsed - timer) / elapsed
            print(f"\n    Rest Ratio Calculation:")
            print(f"      total_elapsed_time = {elapsed} s")
            print(f"      total_timer_time   = {timer} s")
            print(f"      rest_time          = {elapsed - timer} s")
            print(f"      rest_ratio         = {rest_ratio:.1%}")
    else:
        print(f"\n    [WARN] FIT file missing {len(missing_fields)} required fields")

except Exception as e:
    print(f"\n[ERROR] Failed to parse FIT file: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n    Cleaned up temp directory")

print("\n" + "=" * 70)
