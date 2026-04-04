"""
Test script to demonstrate FIT and GPX file processing
Shows BEFORE vs AFTER filtering statistics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
from core.predictor import FeatureExtractor, MLRacePredictor
from core.gpx_filter import GPXFilter
from core.utils import apply_fit_filter, apply_gpx_filter, FilterConfig
import json

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_section(title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.END}\n")

def print_metric(label, value, unit="", color=Colors.END):
    print(f"{color}{label:40}{Colors.END}{Colors.BOLD}{value:>10}{Colors.END} {unit}")

def print_comparison(label, before, after, unit="", threshold=0.05):
    """Print before/after comparison with color coding"""
    change = (after - before) / before if before != 0 else 0
    if abs(change) > threshold:
        color = Colors.GREEN if change < 0 else Colors.RED
        arrow = "↓" if change < 0 else "↑"
        change_pct = f"{change*100:+.1f}%"
    else:
        color = Colors.END
        arrow = "→"
        change_pct = "±0.0%"

    print(f"{label:40}")
    print(f"  {Colors.YELLOW}BEFORE:{Colors.END} {before:>10.1f} {unit}")
    print(f"  {Colors.GREEN}AFTER: {Colors.END} {after:>10.1f} {unit} {color}{arrow} {change_pct}{Colors.END}")


# ============================================================================
# PART 1: FIT FILE PROCESSING
# ============================================================================

print_header("FIT FILE PROCESSING - BEFORE vs AFTER")

fit_path = Path(__file__).parent / "台州市 越野跑_457618021.fit"

print_section("Step 1: Parse FIT File (Raw Data)")
print(f"File: {Colors.BOLD}{fit_path.name}{Colors.END}")
print(f"Size: {fit_path.stat().st_size:,} bytes")

# Extract raw data - handle compressed FIT files
from fitparse import FitFile
import gzip
import zipfile
import tempfile
import shutil

# Check file header for compression
actual_fit_path = fit_path
temp_dir = None
try:
    with open(fit_path, 'rb') as f:
        header = f.read(4)

    if header[:2] == b'\x1f\x8b':
        # GZIP format
        temp_dir = tempfile.mkdtemp()
        temp_fit_file = Path(temp_dir) / fit_path.stem
        with gzip.open(fit_path, 'rb') as gz:
            with open(temp_fit_file, 'wb') as out:
                out.write(gz.read())
        actual_fit_path = temp_fit_file
        print(f"Detected: {Colors.YELLOW}GZIP compressed FIT{Colors.END}")
    elif header[:4] == b'PK\x03\x04':
        # ZIP format
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(fit_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
            if extracted_files:
                actual_fit_path = extracted_files[0]
                print(f"Detected: {Colors.YELLOW}ZIP compressed FIT{Colors.END}")
except Exception as e:
    print(f"Warning checking compression: {e}")

fitfile = FitFile(str(actual_fit_path))

# Get raw records
raw_records = []
raw_timestamps = []
raw_distances = []
raw_elevations = []
raw_heart_rates = []

# Debug: check available fields
all_fields = set()
for record in fitfile.get_messages('record'):
    for field in record:
        all_fields.add(field.name)
    break
print(f"Available fields in FIT: {all_fields}")

for i, record in enumerate(fitfile.get_messages('record')):
    record_data = {}
    for field in record:
        record_data[field.name] = field.value
        if field.name == 'timestamp' and field.value is not None:
            if hasattr(field.value, 'timestamp'):
                raw_timestamps.append(field.value.timestamp())
            else:
                raw_timestamps.append(i * 60)
        elif field.name == 'distance' and field.value is not None:
            raw_distances.append(float(field.value))
        elif field.name in ['altitude', 'enhanced_altitude'] and field.value is not None:
            raw_elevations.append(float(field.value))
        elif field.name in ['heart_rate', 'enhanced_heart_rate'] and field.value is not None:
            raw_heart_rates.append(int(field.value))
    raw_records.append(record_data)

# Ensure lists have same length
min_len = min(len(raw_timestamps), len(raw_distances), len(raw_elevations))
raw_timestamps = raw_timestamps[:min_len]
raw_distances = raw_distances[:min_len]
raw_elevations = raw_elevations[:min_len]

print(f"Records parsed: timestamps={len(raw_timestamps)}, distances={len(raw_distances)}, elevations={len(raw_elevations)}")

print_metric("Total Records", len(raw_records), "points")
print_metric("Total Distance", raw_distances[-1]/1000 if raw_distances else 0, "km")
print_metric("Duration", (raw_timestamps[-1]-raw_timestamps[0])/60 if len(raw_timestamps) > 1 else 0, "min")

print_section("Step 2: Elevation Filtering (Savitzky-Golay)")

# Calculate BEFORE filtering stats
if len(raw_elevations) == 0:
    print(f"{Colors.RED}Error: No elevation data found in FIT file{Colors.END}")
    print("Skipping FIT file processing...")
    # Skip to GPX processing
    skip_fit_processing = True
else:
    skip_fit_processing = False
    raw_ele_arr = np.array(raw_elevations)
    raw_grades = []

    # Check if we have valid distance data
    has_valid_distance = len(raw_distances) > 1 and max(raw_distances) > 100

    if has_valid_distance:
        # Use distance from FIT file
        for i in range(len(raw_elevations) - 1):
            if i+1 < len(raw_distances):
                dist_m = raw_distances[i+1] - raw_distances[i]
                ele_m = raw_elevations[i+1] - raw_elevations[i]
                if dist_m > 0.5:
                    grade = (ele_m / dist_m) * 100
                    raw_grades.append(grade)
    else:
        # Estimate from timestamps (assuming ~2 min/km pace for trail running)
        print(f"{Colors.YELLOW}Warning: No valid distance data, estimating from timestamps{Colors.END}")
        for i in range(len(raw_elevations) - 1):
            if i+1 < len(raw_timestamps):
                time_s = raw_timestamps[i+1] - raw_timestamps[i]
                if time_s > 0:
                    # Estimate distance: ~8.3 min/km = 7.2 km/h = 2 m/s
                    estimated_dist = time_s * 2
                    ele_m = raw_elevations[i+1] - raw_elevations[i]
                    grade = (ele_m / estimated_dist) * 100 if estimated_dist > 0.5 else 0
                    raw_grades.append(grade)

    raw_grades = np.array(raw_grades) if raw_grades else np.array([0])

    raw_gain = np.sum(np.maximum(raw_grades * np.diff(raw_distances[:len(raw_grades)+1]) / 100, 0))
    raw_max_grade = np.max(np.abs(raw_grades))

    print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Raw Elevation Range", np.max(raw_ele_arr) - np.min(raw_ele_arr), "m")
    print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Elevation Std Dev", np.std(raw_ele_arr), "m")
    print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Total Climb", raw_gain, "m")
    print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Max Grade", raw_max_grade, "%")
    print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Extreme Grades (>45%)", np.sum(np.abs(raw_grades) > 45), "count")

    # Apply filtering
    smoothed_elevations, filter_info = apply_fit_filter(raw_elevations, raw_timestamps)

    # Calculate AFTER filtering stats
    smoothed_grades = []

    if has_valid_distance:
        for i in range(len(smoothed_elevations) - 1):
            if i+1 < len(raw_distances):
                dist_m = raw_distances[i+1] - raw_distances[i]
                ele_m = smoothed_elevations[i+1] - smoothed_elevations[i]
                if dist_m > 0.5:
                    grade = (ele_m / dist_m) * 100
                    smoothed_grades.append(grade)
    else:
        for i in range(len(smoothed_elevations) - 1):
            if i+1 < len(raw_timestamps):
                time_s = raw_timestamps[i+1] - raw_timestamps[i]
                if time_s > 0:
                    estimated_dist = time_s * 2
                    ele_m = smoothed_elevations[i+1] - smoothed_elevations[i]
                    grade = (ele_m / estimated_dist) * 100 if estimated_dist > 0.5 else 0
                    smoothed_grades.append(grade)

    smoothed_grades = np.array(smoothed_grades) if smoothed_grades else np.array([0])

    smoothed_gain = np.sum(np.maximum(smoothed_grades * np.diff(raw_distances[:len(smoothed_grades)+1]) / 100, 0))
    smoothed_max_grade = np.max(np.abs(smoothed_grades))

    print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Elevation Range", np.max(smoothed_elevations) - np.min(smoothed_elevations), "m")
    print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Elevation Std Dev", np.std(smoothed_elevations), "m")
    print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Total Climb", smoothed_gain, "m")
    print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Max Grade", smoothed_max_grade, "%")
    print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Extreme Grades (>45%)", np.sum(np.abs(smoothed_grades) > 45), "count")

    print_section("Step 3: Feature Extraction (200m segments)")

    segments = FeatureExtractor.extract_from_fit(fit_path, segment_length_m=200)

    if segments:
        print_metric("Segments Extracted", len(segments), "segments")

        speeds = [s.speed_kmh for s in segments]
        grades = [s.grade_pct for s in segments]

        print_metric("Speed Range", f"{min(speeds):.1f} - {max(speeds):.1f}", "km/h")
        print_metric("Grade Range", f"{min(grades):.1f} - {max(grades):.1f}", "%")
        print_metric("Average Speed", np.mean(speeds), "km/h")
        print_metric("Average Climb Speed", np.mean([s.speed_kmh for s in segments if s.grade_pct > 5]), "km/h")

    print_section("Step 4: FIT Processing Summary")

    print_comparison("Elevation Noise (Std Dev)",
                     np.std(raw_ele_arr), np.std(smoothed_elevations), "m")
    print_comparison("Total Climb",
                     raw_gain, smoothed_gain, "m")
    print_comparison("Max Grade",
                     raw_max_grade, smoothed_max_grade, "%")
    print_comparison("Extreme Grades Count",
                     np.sum(np.abs(raw_grades) > 45), np.sum(np.abs(smoothed_grades) > 45), "")


# ============================================================================
# PART 2: GPX FILE PROCESSING
# ============================================================================

print_header("GPX FILE PROCESSING - BEFORE vs AFTER")

gpx_path = Path(__file__).parent / "2025黄岩九峰大师赛最终版.gpx"

print_section("Step 1: Parse GPX File (Raw Data)")
print(f"File: {Colors.BOLD}{gpx_path.name}{Colors.END}")
print(f"Size: {gpx_path.stat().st_size:,} bytes")

# Parse raw GPX
import xml.etree.ElementTree as ET
tree = ET.parse(gpx_path)
root = tree.getroot()
ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

raw_points = []
for trkpt in root.findall('.//gpx:trkpt', ns):
    ele = trkpt.find('gpx:ele', ns)
    raw_points.append({
        'lat': float(trkpt.get('lat')),
        'lon': float(trkpt.get('lon')),
        'ele': float(ele.text) if ele is not None else 0
    })

# Extract waypoints
checkpoints = []
for wpt in root.findall('.//gpx:wpt', ns):
    name_elem = wpt.find('gpx:name', ns)
    checkpoints.append({
        'name': name_elem.text if name_elem is not None else 'Unknown',
        'lat': float(wpt.get('lat')),
        'lon': float(wpt.get('lon'))
    })

# Calculate raw distances
raw_distances = [0]
for i in range(len(raw_points) - 1):
    import math
    p1, p2 = raw_points[i], raw_points[i+1]
    lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
    lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    dist = 6371000 * 2 * math.asin(math.sqrt(a))
    raw_distances.append(raw_distances[-1] + dist)

raw_distances = np.array(raw_distances)
raw_elevations = np.array([p['ele'] for p in raw_points])

print_metric("Total Track Points", len(raw_points), "points")
print_metric("Total Distance", raw_distances[-1]/1000, "km")
print_metric("Checkpoints (CP)", len(checkpoints), "points")

# Calculate raw grade
raw_gpx_grades = []
for i in range(len(raw_elevations) - 1):
    dist_m = raw_distances[i+1] - raw_distances[i]
    ele_m = raw_elevations[i+1] - raw_elevations[i]
    if dist_m > 0.5:
        grade = (ele_m / dist_m) * 100
        raw_gpx_grades.append(grade)
    else:
        raw_gpx_grades.append(0)
raw_gpx_grades = np.array(raw_gpx_grades)

raw_gpx_gain = np.sum(np.maximum(raw_gpx_grades * (raw_distances[1:] - raw_distances[:-1]) / 100, 0))

print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Raw Elevation Range", np.max(raw_elevations) - np.min(raw_elevations), "m")
print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Raw Total Climb", raw_gpx_gain, "m")
print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Raw Max Grade", np.max(np.abs(raw_gpx_grades)), "%")

print_section("Step 2: GPX Filtering Pipeline")

# Apply GPX filter
smoothed_gpx_elevations, gpx_filter_info = apply_gpx_filter(raw_elevations.tolist(), raw_distances.tolist(), raw_gpx_gain)

# Calculate filtered grade
filtered_gpx_grades = []
for i in range(len(smoothed_gpx_elevations) - 1):
    dist_m = raw_distances[i+1] - raw_distances[i]
    ele_m = smoothed_gpx_elevations[i+1] - smoothed_gpx_elevations[i]
    if dist_m > 0.5:
        grade = (ele_m / dist_m) * 100
        # Apply grade clipping
        grade = max(-45, min(45, grade))
        filtered_gpx_grades.append(grade)
    else:
        filtered_gpx_grades.append(0)
filtered_gpx_grades = np.array(filtered_gpx_grades)

filtered_gpx_gain = np.sum(np.maximum(filtered_gpx_grades * (raw_distances[1:] - raw_distances[:-1]) / 100, 0))

print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Filtered Elevation Range", np.max(smoothed_gpx_elevations) - np.min(smoothed_gpx_elevations), "m")
print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Filtered Total Climb", filtered_gpx_gain, "m")
print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Filtered Max Grade", np.max(np.abs(filtered_gpx_grades)), "%")

print_section("Step 3: GPX Processing Summary")

print_comparison("Total Climb (DEM Noise Reduction)",
                 raw_gpx_gain, filtered_gpx_gain, "m", threshold=0.01)

climbing_loss_m = raw_gpx_gain - filtered_gpx_gain
climbing_loss_pct = (climbing_loss_m / raw_gpx_gain * 100) if raw_gpx_gain > 0 else 0

print_metric("Climbing Loss (Removed)", f"{climbing_loss_m:.0f}m ({climbing_loss_pct:.1f}%)", "", Colors.YELLOW)

# Grade distribution
grade_ranges = {
    'Steep Descent (<-10%)': np.sum(filtered_gpx_grades < -10),
    'Gentle Descent (-10% to -5%)': np.sum((filtered_gpx_grades >= -10) & (filtered_gpx_grades < -5)),
    'Flat (-5% to 5%)': np.sum((filtered_gpx_grades >= -5) & (filtered_gpx_grades < 5)),
    'Gentle Climb (5% to 15%)': np.sum((filtered_gpx_grades >= 5) & (filtered_gpx_grades < 15)),
    'Steep Climb (15% to 30%)': np.sum((filtered_gpx_grades >= 15) & (filtered_gpx_grades < 30)),
    'Extreme Climb (>30%)': np.sum(filtered_gpx_grades >= 30),
}

print_section("Grade Distribution (Filtered)")
spacing = np.mean(raw_distances[1:] - raw_distances[:-1]) if len(raw_distances) > 1 else 1
for range_name, count in grade_ranges.items():
    pct = count / len(filtered_gpx_grades) * 100
    dist_km = count * spacing / 1000
    bar_length = int(pct / 2)
    bar = "#" * bar_length + "-" * (50 - bar_length)
    print(f"  {range_name:25} {Colors.GREEN}{bar}{Colors.END} {pct:5.1f}% ({dist_km:4.1f}km)")


# ============================================================================
# PART 3: GENERATE REPORT
# ============================================================================

print_header("GENERATING REPORT")

report = {
    "timestamp": str(Path(__file__).stat().st_mtime),
    "fit_file": {
        "name": fit_path.name,
        "size_bytes": fit_path.stat().st_size,
        "before": {
            "elevation_std_m": round(float(np.std(raw_ele_arr)), 2),
            "total_climb_m": round(float(raw_gain), 0),
            "max_grade_pct": round(float(raw_max_grade), 1),
            "extreme_grades_count": int(np.sum(np.abs(raw_grades) > 45))
        },
        "after": {
            "elevation_std_m": round(float(np.std(smoothed_elevations)), 2),
            "total_climb_m": round(float(smoothed_gain), 0),
            "max_grade_pct": round(float(smoothed_max_grade), 1),
            "extreme_grades_count": int(np.sum(np.abs(smoothed_grades) > 45))
        },
        "segments_extracted": len(segments) if segments else 0
    },
    "gpx_file": {
        "name": gpx_path.name,
        "size_bytes": gpx_path.stat().st_size,
        "before": {
            "total_climb_m": round(float(raw_gpx_gain), 0),
            "max_grade_pct": round(float(np.max(np.abs(raw_gpx_grades))), 1)
        },
        "after": {
            "total_climb_m": round(float(filtered_gpx_gain), 0),
            "max_grade_pct": round(float(np.max(np.abs(filtered_gpx_grades))), 1)
        },
        "climbing_loss_m": round(float(climbing_loss_m), 0),
        "climbing_loss_pct": round(float(climbing_loss_pct), 1)
    }
}

report_path = Path(__file__).parent / "processing_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"Report saved to: {Colors.BOLD}{report_path}{Colors.END}")

print_header("PROCESSING COMPLETE")
print(f"{Colors.GREEN}[OK]{Colors.END} FIT file processed successfully")
print(f"{Colors.GREEN}[OK]{Colors.END} GPX file processed successfully")
print(f"{Colors.GREEN}[OK]{Colors.END} Report generated: {Colors.BOLD}processing_report.json{Colors.END}")

# Cleanup temp directory
if temp_dir and os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)
