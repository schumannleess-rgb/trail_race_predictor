"""Analyze RECORD messages in FIT file"""
import sys
import gzip
import zipfile
import tempfile
import shutil
import os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from fitparse import FitFile

fit_path = Path(r'd:\Garmin\garmin-fitness-v3\projects\race_predictor\trail_race_predictor_v1.2.1\example\台州市 越野跑_457618021.fit')

with open(fit_path, 'rb') as f:
    header = f.read(4)

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
    elif header[:4] == b'PK\x03\x04':
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(fit_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = list(Path(temp_dir).rglob('*.fit'))
            if extracted_files:
                actual_fit_path = extracted_files[0]

    fitfile = FitFile(str(actual_fit_path))

    all_field_names = set()
    field_value_counts = defaultdict(int)
    field_null_counts = defaultdict(int)
    field_sample_values = defaultdict(list)
    field_types = {}
    field_units = {}
    timestamps = []

    for message in fitfile.get_messages('record'):
        for field in message:
            field_name = field.name
            field_value = field.value
            field_unit = field.units
            
            all_field_names.add(field_name)
            field_value_counts[field_name] += 1
            
            if field_value is None or (isinstance(field_value, tuple) and all(v is None for v in field_value)):
                field_null_counts[field_name] += 1
            
            if field_name not in field_units and field_unit:
                field_units[field_name] = field_unit
            
            if len(field_sample_values[field_name]) < 3 and field_value is not None:
                if not (isinstance(field_value, tuple) and all(v is None for v in field_value)):
                    field_sample_values[field_name].append(field_value)
                    if field_name not in field_types:
                        field_types[field_name] = type(field_value).__name__
            
            if field_name == 'timestamp':
                timestamps.append(field_value)

    total_records = field_value_counts.get('timestamp', 0)
    
    print("=" * 80)
    print("RECORD MESSAGE ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\nTotal RECORDs: {total_records}")
    print(f"Total unique fields: {len(all_field_names)}")
    
    if timestamps:
        print(f"Time range: {timestamps[0]} ~ {timestamps[-1]}")
        print(f"Duration: {(timestamps[-1] - timestamps[0]).total_seconds() / 3600:.2f} hours")

    print("\n" + "=" * 80)
    print("TIME INTERVAL ANALYSIS")
    print("=" * 80)
    
    if len(timestamps) >= 2:
        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(delta)
        
        interval_counts = Counter(intervals)
        
        print("\nInterval Distribution:")
        print(f"{'Interval (s)':<15} {'Count':>10} {'Percentage':>12}")
        print("-" * 40)
        
        sorted_intervals = sorted(interval_counts.items())
        total = len(intervals)
        
        for interval, count in sorted_intervals[:20]:
            pct = count / total * 100
            print(f"{interval:<15.1f} {count:>10} {pct:>11.1f}%")
        
        print("\nStatistics:")
        print(f"  Min:    {min(intervals):.1f} s")
        print(f"  Max:    {max(intervals):.1f} s")
        print(f"  Mean:   {statistics.mean(intervals):.2f} s")
        print(f"  Median: {statistics.median(intervals):.1f} s")
        print(f"  Mode:   {statistics.mode(intervals):.1f} s")
        
        gaps = [i for i in intervals if i > 5]
        if gaps:
            print(f"\nLarge gaps (>5s): {len(gaps)} occurrences")
            gap_counts = Counter(gaps)
            for gap, count in gap_counts.most_common(10):
                print(f"  {gap:.0f}s gap: {count} times")

    print("\n" + "=" * 80)
    print("ALL FIELDS")
    print("=" * 80)
    
    print(f"\n{'Field Name':<35} {'Type':<12} {'Units':<10} {'Non-Null':>10} {'Null':>8}")
    print("-" * 80)
    
    for field_name in sorted(all_field_names):
        non_null = field_value_counts[field_name] - field_null_counts[field_name]
        null = field_null_counts[field_name]
        ftype = field_types.get(field_name, "unknown")
        funits = field_units.get(field_name, "")
        print(f"{field_name:<35} {ftype:<12} {funits:<10} {non_null:>10} {null:>8}")

    print("\n" + "=" * 80)
    print("SAMPLE VALUES")
    print("=" * 80)
    
    for field_name in sorted(all_field_names):
        samples = field_sample_values[field_name]
        funits = field_units.get(field_name, "")
        if samples:
            print(f"\n{field_name} ({funits}):")
            for i, v in enumerate(samples[:3]):
                if isinstance(v, float):
                    print(f"  [{i+1}] {v:.4f}")
                else:
                    print(f"  [{i+1}] {v}")

finally:
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

print("\n" + "=" * 80)
