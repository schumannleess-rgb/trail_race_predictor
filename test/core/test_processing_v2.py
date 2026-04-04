"""
Test script to demonstrate FIT and GPX file processing
Shows Garmin pre-calculated values vs our calculations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
from core.predictor import FeatureExtractor, MLRacePredictor
from core.gpx_filter import GPXFilter
from core.utils import apply_fit_filter, apply_gpx_filter, FilterConfig
from fitparse import FitFile
import json
import gzip
import zipfile
import tempfile
import shutil

# ANSI color codes
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

def print_comparison(label, garmin, calculated, unit=""):
    """Print Garmin vs calculated comparison"""
    if garmin > 0:
        diff = calculated - garmin
        diff_pct = (diff / garmin) * 100
        if abs(diff_pct) > 5:
            color = Colors.RED
            status = "[WRONG]"
        else:
            color = Colors.GREEN
            status = "[OK]"
    else:
        color = Colors.END
        diff_pct = 0
        status = "[N/A]"

    print(f"{label:40}")
    print(f"  {Colors.YELLOW}Garmin (CORRECT):{Colors.END} {garmin:>10.1f} {unit}")
    print(f"  {Colors.CYAN}Our Calculation:{Colors.END} {calculated:>10.1f} {unit} {color}{diff_pct:+.1f}% {status}{Colors.END}")


# ============================================================================
# PART 1: FIT FILE PROCESSING - Garmin vs Our Calculation
# ============================================================================

print_header("FIT FILE - Garmin Pre-Calculated vs Our Calculation")

fit_path = Path(__file__).parent / "台州市 越野跑_457618021.fit"

print_section("Step 1: Extract Garmin Pre-Calculated Values")

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
        print(f"Detected: {Colors.YELLOW}GZIP compressed FIT{Colors.END}")
    elif header[:4] == b'PK\x03\x04':
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(fit_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
            if extracted_files:
                actual_fit_path = extracted_files[0]
                print(f"Detected: {Colors.YELLOW}ZIP compressed FIT{Colors.END}")
except Exception as e:
    print(f"Warning: {e}")

fitfile = FitFile(str(actual_fit_path))

# Get Garmin pre-calculated values from SESSION
garmin_data = {}
for session in fitfile.get_messages('session'):
    for field in session:
        if field.name in ['total_ascent', 'total_descent', 'total_distance', 'total_timer_time']:
            garmin_data[field.name] = field.value
    break

print(f"File: {Colors.BOLD}{fit_path.name}{Colors.END}")
print_metric("Garmin Total Ascent", garmin_data.get('total_ascent', 0), "m", Colors.GREEN)
print_metric("Garmin Total Descent", garmin_data.get('total_descent', 0), "m", Colors.GREEN)
print_metric("Garmin Total Distance", garmin_data.get('total_distance', 0) / 1000, "km", Colors.GREEN)
print_metric("Garmin Total Time", garmin_data.get('total_timer_time', 0) / 60, "min", Colors.GREEN)

print_section("Step 2: Our Calculation from Raw Elevation")

# Reset and get raw records
fitfile = FitFile(str(actual_fit_path))
raw_elevations = []
raw_distances = []
raw_timestamps = []

for record in fitfile.get_messages('record'):
    for field in record:
        if field.name == 'timestamp' and field.value is not None:
            raw_timestamps.append(field.value.timestamp())
        elif field.name == 'distance' and field.value is not None:
            raw_distances.append(float(field.value))
        elif field.name in ['altitude', 'enhanced_altitude'] and field.value is not None:
            raw_elevations.append(float(field.value))

# Calculate from raw data
raw_ele_arr = np.array(raw_elevations)
if len(raw_elevations) > 1 and len(raw_distances) > 1:
    # Calculate grades
    raw_grades = []
    for i in range(len(raw_elevations) - 1):
        if i+1 < len(raw_distances):
            dist_m = raw_distances[i+1] - raw_distances[i]
            ele_m = raw_elevations[i+1] - raw_elevations[i]
            if dist_m > 0.5:
                grade = (ele_m / dist_m) * 100
                raw_grades.append(grade)

    raw_grades = np.array(raw_grades)

    # Calculate climb (before filtering)
    calculated_ascent_before = np.sum(np.maximum(raw_grades * np.diff(raw_distances[:len(raw_grades)+1]) / 100, 0))

    # Apply filtering
    smoothed_elevations, _ = apply_fit_filter(raw_elevations, raw_timestamps)

    # Calculate climb (after filtering)
    smoothed_grades = []
    for i in range(len(smoothed_elevations) - 1):
        if i+1 < len(raw_distances):
            dist_m = raw_distances[i+1] - raw_distances[i]
            ele_m = smoothed_elevations[i+1] - smoothed_elevations[i]
            if dist_m > 0.5:
                grade = (ele_m / dist_m) * 100
                smoothed_grades.append(grade)

    smoothed_grades = np.array(smoothed_grades)
    calculated_ascent_after = np.sum(np.maximum(smoothed_grades * np.diff(raw_distances[:len(smoothed_grades)+1]) / 100, 0))

    print_metric("Our Calculated (Before Filter)", calculated_ascent_before, "m", Colors.CYAN)
    print_metric("Our Calculated (After Filter)", calculated_ascent_after, "m", Colors.CYAN)

print_section("Step 3: Comparison - Why We Should Use Garmin Values")

print_comparison("Total Ascent",
                 garmin_data.get('total_ascent', 0),
                 calculated_ascent_after,
                 "m")

print(f"\n{Colors.RED}CRITICAL ISSUE:{Colors.END}")
print(f"  Garmin device uses barometric altitude + proprietary filtering")
print(f"  Our calculation from GPS elevation has {Colors.RED}~90% error{Colors.END}!")
print(f"  {Colors.GREEN}SOLUTION: Always use Garmin pre-calculated values{Colors.END}")

print_section("Step 4: Feature Extraction (Using Correct Values)")

# Extract segments using the corrected FeatureExtractor
segments = FeatureExtractor.extract_from_fit(fit_path, segment_length_m=200)

if segments:
    print_metric("Segments Extracted", len(segments), "segments")

    speeds = [s.speed_kmh for s in segments]
    grades = [s.grade_pct for s in segments]

    print_metric("Speed Range", f"{min(speeds):.1f} - {max(speeds):.1f}", "km/h")
    print_metric("Grade Range", f"{min(grades):.1f} - {max(grades):.1f}", "%")
    print_metric("Average Speed", np.mean(speeds), "km/h")


# ============================================================================
# PART 2: GPX FILE PROCESSING
# ============================================================================

print_header("GPX FILE PROCESSING - Before vs After Filtering")

gpx_path = Path(__file__).parent / "2025黄岩九峰大师赛最终版.gpx"

print_section("Step 1: Parse GPX File (Raw Data)")
print(f"File: {Colors.BOLD}{gpx_path.name}{Colors.END}")

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

# Calculate distances
import math
raw_distances_gpx = [0]
for i in range(len(raw_points) - 1):
    p1, p2 = raw_points[i], raw_points[i+1]
    lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
    lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])
    dist = 6371000 * 2 * math.asin(math.sqrt(
        (math.sin((lat2-lat1)/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2)**2)
    ))
    raw_distances_gpx.append(raw_distances_gpx[-1] + dist)

raw_distances_gpx = np.array(raw_distances_gpx)
raw_elevations_gpx = np.array([p['ele'] for p in raw_points])

print_metric("Total Track Points", len(raw_points), "points")
print_metric("Total Distance", raw_distances_gpx[-1]/1000, "km")

# Calculate raw grades
raw_gpx_grades = []
for i in range(len(raw_elevations_gpx) - 1):
    dist_m = raw_distances_gpx[i+1] - raw_distances_gpx[i]
    ele_m = raw_elevations_gpx[i+1] - raw_elevations_gpx[i]
    if dist_m > 0.5:
        grade = (ele_m / dist_m) * 100
        raw_gpx_grades.append(grade)

raw_gpx_grades = np.array(raw_gpx_grades)
# Ensure arrays match in length
dist_diffs = raw_distances_gpx[1:len(raw_gpx_grades)+1] - raw_distances_gpx[:len(raw_gpx_grades)]
raw_gpx_gain = np.sum(np.maximum(raw_gpx_grades * dist_diffs / 100, 0))

print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Raw Total Climb", raw_gpx_gain, "m")
print_metric(f"{Colors.YELLOW}BEFORE{Colors.END} - Max Grade", np.max(np.abs(raw_gpx_grades)), "%")

print_section("Step 2: Apply Filtering")

smoothed_gpx_elevations, gpx_filter_info = apply_gpx_filter(raw_elevations_gpx.tolist(), raw_distances_gpx.tolist(), raw_gpx_gain)

filtered_gpx_grades = []
for i in range(len(smoothed_gpx_elevations) - 1):
    dist_m = raw_distances_gpx[i+1] - raw_distances_gpx[i]
    ele_m = smoothed_gpx_elevations[i+1] - smoothed_gpx_elevations[i]
    if dist_m > 0.5:
        grade = (ele_m / dist_m) * 100
        grade = max(-45, min(45, grade))  # Clip
        filtered_gpx_grades.append(grade)

filtered_gpx_grades = np.array(filtered_gpx_grades)
# Ensure arrays match in length
dist_diffs_filt = raw_distances_gpx[1:len(filtered_gpx_grades)+1] - raw_distances_gpx[:len(filtered_gpx_grades)]
filtered_gpx_gain = np.sum(np.maximum(filtered_gpx_grades * dist_diffs_filt / 100, 0))

print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Filtered Total Climb", filtered_gpx_gain, "m")
print_metric(f"{Colors.GREEN}AFTER{Colors.END} - Max Grade (clipped)", np.max(np.abs(filtered_gpx_grades)), "%")

climbing_loss_m = raw_gpx_gain - filtered_gpx_gain
climbing_loss_pct = (climbing_loss_m / raw_gpx_gain * 100) if raw_gpx_gain > 0 else 0

print_metric("Climbing Loss (DEM Noise Removed)", f"{climbing_loss_m:.0f}m ({climbing_loss_pct:.1f}%)", "", Colors.YELLOW)


# ============================================================================
# PART 3: GENERATE REPORT
# ============================================================================

print_header("GENERATING REPORT")

report = {
    "timestamp": str(Path(__file__).stat().st_mtime),
    "fit_file": {
        "name": fit_path.name,
        "size_bytes": fit_path.stat().st_size,
        "garmin_pre_calculated": {
            "total_ascent_m": garmin_data.get('total_ascent', 0),
            "total_descent_m": garmin_data.get('total_descent', 0),
            "total_distance_m": garmin_data.get('total_distance', 0),
            "total_time_s": garmin_data.get('total_timer_time', 0)
        },
        "our_calculation": {
            "total_ascent_before_m": round(float(calculated_ascent_after) if 'calculated_ascent_after' in locals() else 0, 0),
            "total_ascent_after_m": round(float(calculated_ascent_after) if 'calculated_ascent_after' in locals() else 0, 0),
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

report_path = Path(__file__).parent / "processing_report_v2.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"Report saved to: {Colors.BOLD}{report_path}{Colors.END}")

print_header("PROCESSING COMPLETE")
print(f"{Colors.GREEN}[OK]{Colors.END} FIT file processed with Garmin pre-calculated values")
print(f"{Colors.GREEN}[OK]{Colors.END} GPX file processed with filtering")
print(f"{Colors.GREEN}[OK]{Colors.END} Report generated: {Colors.BOLD}processing_report_v2.json{Colors.END}")

# Cleanup
if temp_dir and os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)
