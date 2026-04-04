"""
Trail Race Predictor - File Report Generator

Tests core code with example FIT and GPX files,
generates detailed reports for both file types.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from core.predictor import MLRacePredictor, FeatureExtractor, SegmentFeatures
from core.utils import FilterConfig, apply_fit_filter, apply_gpx_filter, GradeAnalyzer

# Example file paths
FIT_FILE = ROOT_DIR / "example" / "台州市 越野跑_457618021.fit"
GPX_FILE = ROOT_DIR / "example" / "2025黄岩九峰大师赛最终版.gpx"
OUTPUT_DIR = ROOT_DIR / "example"
REPORT_ENCODING = "utf-8"


def format_time(minutes: float) -> str:
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


def generate_fit_report(fit_path: Path) -> str:
    """Generate detailed report for FIT training file."""
    print(f"\n{'='*60}")
    print(f"Processing FIT file: {fit_path.name}")
    print(f"{'='*60}")

    # Extract segments from FIT
    segments, rest_ratio = FeatureExtractor.extract_from_fit(fit_path)

    if not segments:
        return f"FIT Report: No segments extracted from {fit_path.name}"

    # Calculate statistics
    speeds = [s.speed_kmh for s in segments]
    grades = [s.grade_pct for s in segments]
    altitudes = [s.absolute_altitude_m for s in segments]

    total_distance = max(s.accumulated_distance_km for s in segments)
    total_ascent = max(s.accumulated_ascent_m for s in segments)

    # Speed statistics
    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    min_speed = min(speeds)

    # Grade distribution
    climbing_segs = [s for s in segments if s.grade_pct > 2]
    flat_segs = [s for s in segments if -2 <= s.grade_pct <= 2]
    descent_segs = [s for s in segments if s.grade_pct < -2]

    climbing_speed = sum(s.speed_kmh for s in climbing_segs) / len(climbing_segs) if climbing_segs else 0
    flat_speed = sum(s.speed_kmh for s in flat_segs) / len(flat_segs) if flat_segs else 0
    descent_speed = sum(s.speed_kmh for s in descent_segs) / len(descent_segs) if descent_segs else 0

    # Grade range distribution
    grade_dist = {
        "steep_descent (< -10%)": len([s for s in segments if s.grade_pct < -10]),
        "descent (-10% ~ -5%)": len([s for s in segments if -10 <= s.grade_pct < -5]),
        "gentle_descent (-5% ~ -2%)": len([s for s in segments if -5 <= s.grade_pct < -2]),
        "flat (-2% ~ 2%)": len([s for s in segments if -2 <= s.grade_pct <= 2]),
        "gentle_climb (2% ~ 5%)": len([s for s in segments if 2 < s.grade_pct <= 5]),
        "climb (5% ~ 10%)": len([s for s in segments if 5 < s.grade_pct <= 10]),
        "steep_climb (> 10%)": len([s for s in segments if s.grade_pct > 10]),
    }

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("  FIT Training File Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  File: {fit_path.name}")
    lines.append(f"  Size: {fit_path.stat().st_size / 1024:.1f} KB")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  1. Summary")
    lines.append("-" * 70)
    lines.append(f"  Total Distance:       {total_distance:.2f} km")
    lines.append(f"  Total Ascent:         {total_ascent:.0f} m")
    lines.append(f"  Elevation Density:    {total_ascent / total_distance:.1f} m/km" if total_distance > 0 else "  N/A")
    lines.append(f"  Total Segments:       {len(segments)}")
    lines.append(f"  Altitude Range:       {min(altitudes):.0f} m ~ {max(altitudes):.0f} m")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  2. Speed Analysis")
    lines.append("-" * 70)
    lines.append(f"  Average Speed:       {avg_speed:.2f} km/h")
    lines.append(f"  Max Speed:           {max_speed:.2f} km/h")
    lines.append(f"  Min Speed:           {min_speed:.2f} km/h")
    lines.append(f"  Speed Std Dev:       {(sum((s - avg_speed)**2 for s in speeds) / len(speeds))**0.5:.2f} km/h")
    lines.append("")
    lines.append(f"  Climbing Speed:      {climbing_speed:.2f} km/h ({len(climbing_segs)} segments)")
    lines.append(f"  Flat Speed:          {flat_speed:.2f} km/h ({len(flat_segs)} segments)")
    lines.append(f"  Descent Speed:       {descent_speed:.2f} km/h ({len(descent_segs)} segments)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  3. Grade Distribution")
    lines.append("-" * 70)
    for name, count in grade_dist.items():
        pct = count / len(segments) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {name:30s} {count:4d} ({pct:5.1f}%) {bar}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  4. Segment Details (first 20)")
    lines.append("-" * 70)
    lines.append(f"  {'#':>3s}  {'Dist(km)':>8s}  {'Grade%':>7s}  {'Alt(m)':>7s}  {'Speed(km/h)':>11s}  {'Ascent(m)':>9s}  Type")
    lines.append(f"  {'---':>3s}  {'--------':>8s}  {'-------':>7s}  {'-------':>7s}  {'-----------':>11s}  {'---------':>9s}  ----")

    for i, seg in enumerate(segments[:20]):
        if seg.grade_pct > 5:
            terrain = "climbing"
        elif seg.grade_pct < -5:
            terrain = "descent"
        else:
            terrain = "flat"
        lines.append(
            f"  {i+1:3d}  {seg.accumulated_distance_km:8.2f}  "
            f"{seg.grade_pct:7.1f}  {seg.absolute_altitude_m:7.0f}  "
            f"{seg.speed_kmh:11.2f}  {seg.accumulated_ascent_m:9.0f}  {terrain}"
        )

    if len(segments) > 20:
        lines.append(f"  ... ({len(segments) - 20} more segments)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("  End of FIT Report")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_gpx_report(gpx_path: Path) -> str:
    """Generate detailed report for GPX route file."""
    print(f"\n{'='*60}")
    print(f"Processing GPX file: {gpx_path.name}")
    print(f"{'='*60}")

    predictor = MLRacePredictor()

    # Parse GPX route
    segments, route_info = predictor.parse_gpx_route(str(gpx_path))

    if not segments:
        return f"GPX Report: No segments parsed from {gpx_path.name}"

    total_distance = route_info["total_distance_km"]
    total_gain = route_info["total_elevation_gain_m"]
    total_loss = route_info["total_elevation_loss_m"]
    elevation_density = route_info["elevation_density"]
    checkpoint_count = route_info["checkpoint_count"]
    filter_info = route_info.get("filter_info", {})

    # Grade statistics
    grades = [s.grade_pct for s in segments]
    altitudes = [s.absolute_altitude_m for s in segments]
    seg_ascent = [s.segment_ascent_m for s in segments]
    seg_descent = [s.segment_descent_m for s in segments]

    # Grade range distribution
    grade_dist = {
        "steep_descent (< -15%)": len([s for s in segments if s.grade_pct < -15]),
        "descent (-15% ~ -8%)": len([s for s in segments if -15 <= s.grade_pct < -8]),
        "gentle_descent (-8% ~ -3%)": len([s for s in segments if -8 <= s.grade_pct < -3]),
        "flat (-3% ~ 3%)": len([s for s in segments if -3 <= s.grade_pct <= 3]),
        "gentle_climb (3% ~ 8%)": len([s for s in segments if 3 < s.grade_pct <= 8]),
        "climb (8% ~ 15%)": len([s for s in segments if 8 < s.grade_pct <= 15]),
        "steep_climb (> 15%)": len([s for s in segments if s.grade_pct > 15]),
    }

    # CP points
    checkpoints = route_info.get("checkpoints", [])

    # Difficulty rating
    if elevation_density > 100:
        difficulty = "EXTREME"
    elif elevation_density > 70:
        difficulty = "HARD"
    elif elevation_density > 40:
        difficulty = "MODERATE"
    else:
        difficulty = "EASY"

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("  GPX Route File Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  File: {gpx_path.name}")
    lines.append(f"  Size: {gpx_path.stat().st_size / 1024:.1f} KB")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  1. Route Summary")
    lines.append("-" * 70)
    lines.append(f"  Total Distance:       {total_distance:.2f} km")
    lines.append(f"  Total Elevation Gain: {total_gain:.0f} m")
    lines.append(f"  Total Elevation Loss: {total_loss:.0f} m")
    lines.append(f"  Elevation Density:    {elevation_density:.1f} m/km")
    lines.append(f"  Difficulty Rating:    {difficulty}")
    lines.append(f"  Total Segments:       {len(segments)}")
    lines.append(f"  Checkpoints (CP):     {checkpoint_count}")
    lines.append(f"  Altitude Range:       {min(altitudes):.0f} m ~ {max(altitudes):.0f} m")
    lines.append(f"  Grade Range:          {min(grades):.1f}% ~ {max(grades):.1f}%")
    lines.append("")

    # Filter info
    lines.append("-" * 70)
    lines.append("  2. Filter Information (Savitzky-Golay)")
    lines.append("-" * 70)
    lines.append(f"  Original Gain:        {filter_info.get('original_gain_m', 0):.0f} m")
    lines.append(f"  Filtered Gain:        {filter_info.get('filtered_gain_m', 0):.0f} m")
    lines.append(f"  Climbing Loss:        {filter_info.get('climbing_loss_m', 0):.0f} m")
    lines.append(f"  Grade Range (after):  {filter_info.get('grade_range', 'N/A')}")
    lines.append("")

    # Checkpoints
    if checkpoints:
        lines.append("-" * 70)
        lines.append("  3. Checkpoints")
        lines.append("-" * 70)
        lines.append(f"  {'Name':20s}  {'Lat':>10s}  {'Lon':>11s}  {'Alt(m)':>7s}")
        lines.append(f"  {'----':20s}  {'---':>10s}  {'---':>11s}  {'---':>7s}")
        for cp in checkpoints:
            lines.append(
                f"  {cp['name']:20s}  {cp['lat']:10.6f}  {cp['lon']:11.6f}  {cp['ele']:7.0f}"
            )
        lines.append("")

    # Grade distribution
    lines.append("-" * 70)
    lines.append(f"  {'4' if checkpoints else '3'}. Grade Distribution")
    lines.append("-" * 70)
    for name, count in grade_dist.items():
        pct = count / len(segments) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {name:30s} {count:4d} ({pct:5.1f}%) {bar}")
    lines.append("")

    # Segment details
    section_num = 5 if checkpoints else 4
    lines.append("-" * 70)
    lines.append(f"  {section_num}. Segment Details (every 5th segment)")
    lines.append("-" * 70)
    lines.append(
        f"  {'#':>3s}  {'Dist(km)':>8s}  {'Grade%':>7s}  "
        f"{'Alt(m)':>7s}  {'Ascent(m)':>9s}  {'Descent(m)':>10s}  {'CP':10s}  Type"
    )
    lines.append(
        f"  {'---':>3s}  {'--------':>8s}  {'-------':>7s}  "
        f"{'-------':>7s}  {'---------':>9s}  {'----------':>10s}  {'--':10s}  ----"
    )

    for i, seg in enumerate(segments):
        if i % 5 != 0 and i != len(segments) - 1:
            continue

        if seg.grade_pct > 15:
            terrain = "steep_up"
        elif seg.grade_pct > 3:
            terrain = "climb"
        elif seg.grade_pct < -15:
            terrain = "steep_down"
        elif seg.grade_pct < -3:
            terrain = "descent"
        else:
            terrain = "flat"

        cp_display = seg.cp_name[:10] if seg.cp_name else "-"
        lines.append(
            f"  {i+1:3d}  {seg.accumulated_distance_km:8.2f}  "
            f"{seg.grade_pct:7.1f}  {seg.absolute_altitude_m:7.0f}  "
            f"{seg.segment_ascent_m:9.0f}  {seg.segment_descent_m:10.0f}  "
            f"{cp_display:10s}  {terrain}"
        )

    lines.append("")

    # JSON export for further analysis
    lines.append("-" * 70)
    lines.append(f"  {section_num + 1}. Route JSON Summary")
    lines.append("-" * 70)
    json_summary = {
        "file": gpx_path.name,
        "route_info": route_info,
        "segments_count": len(segments),
        "grade_distribution": grade_dist,
        "difficulty": difficulty,
    }
    lines.append(json.dumps(json_summary, ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("=" * 70)
    lines.append("  End of GPX Report")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_combined_prediction_report(fit_path: Path, gpx_path: Path) -> str:
    """Generate prediction report using FIT for training and GPX for route."""
    print(f"\n{'='*60}")
    print(f"Combined Prediction: FIT training + GPX route")
    print(f"{'='*60}")

    predictor = MLRacePredictor()

    # Train from FIT file
    print("\nStep 1: Training model from FIT file...")
    success = predictor.train_from_files([str(fit_path)])
    if not success:
        return "Combined Report: Training failed!"

    # Predict from GPX
    print("\nStep 2: Predicting from GPX route...")
    lines = []
    lines.append("=" * 70)
    lines.append("  Combined Prediction Report (FIT + GPX)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Training File: {fit_path.name}")
    lines.append(f"  Route File:    {gpx_path.name}")
    lines.append(f"  Generated:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Training stats
    lines.append("-" * 70)
    lines.append("  1. Training Statistics")
    lines.append("-" * 70)
    stats = predictor.training_stats
    lines.append(f"  Files Used:        {stats.get('file_count', 0)}")
    lines.append(f"  Segments:          {stats.get('segment_count', 0)}")
    lines.append(f"  Avg Speed:         {stats.get('avg_speed', 0):.2f} km/h")
    lines.append(f"  P50 Speed:         {stats.get('p50_speed', 0):.2f} km/h")
    lines.append(f"  P90 Speed:         {stats.get('p90_speed', 0):.2f} km/h")
    lines.append(f"  Effort Range:      {stats.get('effort_range', 1.0):.2f}x")
    lines.append(f"  Avg Rest Ratio:    {stats.get('avg_rest_ratio', 0):.1%}")
    lines.append(f"  Calib. Max Grade:  {stats.get('calibrated_max_grade_pct', 0):.0f}%")
    lines.append("")

    # Feature importance
    lines.append("-" * 70)
    lines.append("  2. Feature Importance")
    lines.append("-" * 70)
    importance = predictor.all_feature_importance
    if importance:
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp / max(importance.values()) * 30)
            lines.append(f"  {feat:30s} {imp:6.0f}  {bar}")
    else:
        lines.append("  (fallback model, no feature importance)")
    lines.append("")

    # Predictions for different effort levels
    lines.append("-" * 70)
    lines.append("  3. Race Predictions")
    lines.append("-" * 70)

    for effort in [0.85, 0.9, 1.0, 1.1, 1.2]:
        try:
            result = predictor.predict_race(str(gpx_path), effort)
            label = {
                0.85: "Conservative",
                0.9: "Cautious",
                1.0: "Average (P50)",
                1.1: "Race Mode",
                1.2: "Full Gas (P90+)",
            }.get(effort, "")

            lines.append(f"")
            lines.append(f"  [{label}] Effort: {effort}x")
            lines.append(f"    Predicted Time:  {result['predicted_time_hm']} ({result['predicted_time_min']:.0f} min)")
            lines.append(f"    Moving Time:     {format_time(result.get('predicted_moving_time_min', result['predicted_time_min']))} ({result.get('predicted_moving_time_min', result['predicted_time_min']):.0f} min)")
            lines.append(f"    Rest Ratio:      {result.get('rest_ratio_used', 0):.1%}")
            lines.append(f"    Average Pace:    {result['predicted_pace_min_km']:.1f} min/km")
            lines.append(f"    Average Speed:   {result['predicted_speed_kmh']:.2f} km/h")
            lines.append(f"    Total Distance:  {result['route_info']['total_distance_km']:.2f} km")
            lines.append(f"    Total Ascent:    {result['route_info']['total_elevation_gain_m']:.0f} m")

            # CP point timings
            seg_preds = result.get("segment_predictions", [])
            cp_entries = [s for s in seg_preds if s.get("cp_name")]
            if cp_entries:
                lines.append(f"    Checkpoint Times:")
                for cp in cp_entries:
                    lines.append(
                        f"      {cp['cp_name']:20s}  @ {cp['distance_km']:.1f}km  "
                        f"{format_time(cp['cumulative_time_min'])}"
                    )

            # Last segment (finish)
            if seg_preds:
                last = seg_preds[-1]
                lines.append(
                    f"      {'FINISH':20s}  @ {last['distance_km'] + result['route_info']['total_distance_km'] - last['distance_km']:.1f}km  "
                    f"{format_time(last['cumulative_time_min'])}"
                )
        except Exception as e:
            lines.append(f"  Effort {effort}: Error - {e}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  End of Combined Report")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Trail Race Predictor - File Report Generator")
    print("=" * 70)

    # Verify files exist
    if not FIT_FILE.exists():
        print(f"ERROR: FIT file not found: {FIT_FILE}")
        return
    if not GPX_FILE.exists():
        print(f"ERROR: GPX file not found: {GPX_FILE}")
        return

    print(f"\nFIT file: {FIT_FILE.name} ({FIT_FILE.stat().st_size / 1024:.1f} KB)")
    print(f"GPX file: {GPX_FILE.name} ({GPX_FILE.stat().st_size / 1024:.1f} KB)")

    # --- Generate FIT report ---
    fit_report = generate_fit_report(FIT_FILE)
    fit_report_path = OUTPUT_DIR / "fit_report.txt"
    with open(fit_report_path, "w", encoding=REPORT_ENCODING) as f:
        f.write(fit_report)
    print(f"\nFIT report saved: {fit_report_path}")

    # --- Generate GPX report ---
    gpx_report = generate_gpx_report(GPX_FILE)
    gpx_report_path = OUTPUT_DIR / "gpx_report.txt"
    with open(gpx_report_path, "w", encoding=REPORT_ENCODING) as f:
        f.write(gpx_report)
    print(f"GPX report saved: {gpx_report_path}")

    # --- Generate combined prediction report ---
    combined_report = generate_combined_prediction_report(FIT_FILE, GPX_FILE)
    combined_report_path = OUTPUT_DIR / "prediction_report.txt"
    with open(combined_report_path, "w", encoding=REPORT_ENCODING) as f:
        f.write(combined_report)
    print(f"Combined prediction report saved: {combined_report_path}")

    print(f"\n{'='*70}")
    print("All reports generated successfully!")
    print(f"  1. {fit_report_path}")
    print(f"  2. {gpx_report_path}")
    print(f"  3. {combined_report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
