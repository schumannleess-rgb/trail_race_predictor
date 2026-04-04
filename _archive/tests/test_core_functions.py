"""
Core Module Function Tests
==========================
Tests all core module functions using:
  - temp/records/*.fit  as training input (FIT records)
  - temp/routes/*.gpx  as prediction map (GPX route)

Usage:
  python test_core_functions.py
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

RECORDS_DIR = PROJECT_ROOT / 'temp' / 'records'
ROUTES_DIR = PROJECT_ROOT / 'temp' / 'routes'
OUTPUT_DIR = PROJECT_ROOT / 'temp' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestRunner:
    """Simple test runner with pass/fail counting."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def run(self, name: str, func):
        """Run a single test function and record the result."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
        try:
            func()
            self.passed += 1
            self.results.append({'name': name, 'status': 'PASS'})
            print(f"[PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            self.results.append({'name': name, 'status': 'FAIL', 'error': str(e)})
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            self.failed += 1
            tb = traceback.format_exc()
            self.results.append({'name': name, 'status': 'ERROR', 'error': str(e), 'traceback': tb})
            print(f"[ERROR] {name}: {e}")
            print(tb)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"SUMMARY: {total} tests, {self.passed} passed, {self.failed} failed")
        print('='*60)
        for r in self.results:
            icon = "OK" if r['status'] == 'PASS' else "!!"
            line = f"  [{icon}] {r['name']}"
            if r['status'] != 'PASS':
                line += f"  -- {r.get('error', '')[:80]}"
            print(line)

        # Save JSON report
        report = {
            'test_time': datetime.now().isoformat(),
            'total': total,
            'passed': self.passed,
            'failed': self.failed,
            'results': self.results,
        }
        report_path = OUTPUT_DIR / 'core_test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nReport saved: {report_path}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Data Discovery
# ---------------------------------------------------------------------------

def collect_fit_files():
    files = list(RECORDS_DIR.glob('*.fit')) + list(RECORDS_DIR.glob('*.FIT'))
    return sorted(set(files))


def collect_gpx_files():
    files = list(ROUTES_DIR.glob('*.gpx')) + list(ROUTES_DIR.glob('*.GPX'))
    return sorted(set(files))


FIT_FILES = collect_fit_files()
GPX_FILES = collect_gpx_files()


# ---------------------------------------------------------------------------
# Test: core.utils — FilterConfig, ElevationFilter, GradeAnalyzer
# ---------------------------------------------------------------------------

def test_filter_config_defaults():
    """FilterConfig should provide GPX and FIT default configurations."""
    from core.utils import FilterConfig

    assert isinstance(FilterConfig.GPX, dict), "FilterConfig.GPX should be a dict"
    assert isinstance(FilterConfig.FIT, dict), "FilterConfig.FIT should be a dict"

    # GPX config
    assert FilterConfig.GPX['resample_required'] is True
    assert FilterConfig.GPX['max_grade_pct'] == 45.0
    assert FilterConfig.GPX['window_size'] == 7

    # FIT config
    assert FilterConfig.FIT['resample_required'] is False
    assert FilterConfig.FIT['max_grade_pct'] == 50.0
    assert FilterConfig.FIT['window_size'] == 7

    print(f"  GPX config: {FilterConfig.GPX}")
    print(f"  FIT config: {FilterConfig.FIT}")


def test_elevation_filter_smooth():
    """ElevationFilter.smooth should return smoothed array of same length."""
    from core.utils import ElevationFilter, FilterConfig
    import numpy as np

    raw = [100, 102, 98, 105, 95, 110, 100, 108, 97, 103,
           101, 99, 106, 94, 111, 100, 107, 96, 104, 102]

    result = ElevationFilter.smooth(raw, FilterConfig.FIT)
    assert len(result) == len(raw), f"Length mismatch: {len(result)} vs {len(raw)}"
    assert isinstance(result, np.ndarray)

    # Noise should be reduced (std of diff should be smaller)
    diff_raw = np.std(np.diff(raw))
    diff_smooth = np.std(np.diff(result))
    print(f"  Raw diff std: {diff_raw:.3f}, Smoothed diff std: {diff_smooth:.3f}")
    assert diff_smooth < diff_raw, "Smoothing should reduce noise"


def test_elevation_filter_calculate_grade():
    """ElevationFilter.calculate_grade should compute grades with clipping."""
    from core.utils import ElevationFilter, FilterConfig
    import numpy as np

    # 1000m distance, 100m climb => 10% grade
    distances = np.array([0, 500, 1000, 1500, 2000])
    elevations = np.array([100, 150, 200, 180, 160])

    grades = ElevationFilter.calculate_grade(elevations, distances, FilterConfig.GPX)
    assert len(grades) == len(elevations)
    assert 8 < grades[0] < 12, f"Expected ~10%, got {grades[0]:.1f}%"

    print(f"  Grades: {grades}")
    print(f"  Grade range: {np.min(grades):.1f}% ~ {np.max(grades):.1f}%")


def test_grade_analyzer_distribution():
    """GradeAnalyzer.analyze_distribution should return grade breakdown."""
    from core.utils import GradeAnalyzer
    import numpy as np

    grades = np.array([0, 2, -1, 8, 12, -8, 25, -3, 4, -15, 0, 3])
    spacing_m = 20

    dist = GradeAnalyzer.analyze_distribution(grades, spacing_m)
    assert 'flat' in dist
    assert 'gentle_climb' in dist
    assert 'steep_climb' in dist
    assert dist['flat']['percentage'] > 0

    print(f"  Distribution: {json.dumps({k: v['percentage'] for k, v in dist.items()}, indent=2)}")


def test_grade_analyzer_climbing_loss():
    """GradeAnalyzer.calculate_climbing_loss should compute loss correctly."""
    from core.utils import GradeAnalyzer

    loss_m, loss_pct = GradeAnalyzer.calculate_climbing_loss(1000, 850)
    assert loss_m == 150
    assert abs(loss_pct - 15.0) < 0.01

    print(f"  Original: 1000m, Filtered: 850m, Loss: {loss_m}m ({loss_pct:.1f}%)")


def test_apply_fit_filter():
    """apply_fit_filter should smooth FIT elevations and return info."""
    from core.utils import apply_fit_filter
    import numpy as np

    # Simulate noisy elevation data (100 points)
    np.random.seed(42)
    raw_elev = 500 + np.cumsum(np.random.randn(100) * 2)

    smoothed, info = apply_fit_filter(raw_elev.tolist())
    assert len(smoothed) == len(raw_elev)
    assert 'points' in info
    assert 'noise_std_m' in info
    assert info['points'] == 100

    print(f"  Points: {info['points']}")
    print(f"  Noise std: {info['noise_std_m']:.2f} m")
    print(f"  Max deviation: {info['max_deviation_m']:.2f} m")


def test_apply_gpx_filter():
    """apply_gpx_filter should smooth GPX elevations and return stats."""
    from core.utils import apply_gpx_filter
    import numpy as np

    elevations = [100, 110, 120, 130, 125, 115, 105, 100, 95, 90]
    distances = list(range(0, 1000, 100))  # 0, 100, 200, ... 900

    smoothed, info = apply_gpx_filter(elevations, distances, original_gain_m=50)
    assert len(smoothed) == len(elevations)
    assert 'points' in info
    assert 'filtered_gain_m' in info
    assert 'grade_range' in info

    print(f"  Points: {info['points']}")
    print(f"  Filtered gain: {info['filtered_gain_m']:.0f} m")
    print(f"  Grade range: {info['grade_range']}")


# ---------------------------------------------------------------------------
# Test: core.utils — FilterConfig.calibrate_from_fit_files
# ---------------------------------------------------------------------------

def test_calibrate_from_fit_files():
    """FilterConfig.calibrate_from_fit_files should auto-calibrate max_grade."""
    from core.utils import FilterConfig

    if not FIT_FILES:
        print("  [SKIP] No FIT files found in temp/records/")
        return

    fit_paths = [str(f) for f in FIT_FILES[:5]]
    calibrated = FilterConfig.calibrate_from_fit_files(fit_paths)

    assert isinstance(calibrated, dict)
    assert 'max_grade_pct' in calibrated
    assert 30 <= calibrated['max_grade_pct'] <= 80, \
        f"Calibrated max_grade out of range: {calibrated['max_grade_pct']}"

    print(f"  Calibrated max_grade_pct: {calibrated['max_grade_pct']:.1f}%")
    print(f"  Window size: {calibrated['window_size']}")
    print(f"  Used {len(fit_paths)} FIT files for calibration")


# ---------------------------------------------------------------------------
# Test: core.gpx_filter — GPXFilter
# ---------------------------------------------------------------------------

def test_gpx_filter_parse():
    """GPXFilter.parse_gpx should parse track points and waypoints."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found in temp/routes/")
        return

    from core.gpx_filter import GPXFilter

    gpx = GPXFilter(str(GPX_FILES[0]))
    data = gpx.parse_gpx()

    assert 'points' in data
    assert 'waypoints' in data
    assert 'distances' in data
    assert 'elevations' in data
    assert len(data['points']) > 0

    print(f"  Track points: {len(data['points'])}")
    print(f"  Waypoints (CPs): {len(data['waypoints'])}")
    print(f"  Total distance: {data['total_distance_km']:.2f} km")
    print(f"  Elevation range: {data['elevations'].min():.0f} ~ {data['elevations'].max():.0f} m")


def test_gpx_filter_full_process():
    """GPXFilter.process should run full pipeline: resample -> smooth -> grade."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found in temp/routes/")
        return

    from core.gpx_filter import GPXFilter
    import numpy as np

    gpx = GPXFilter(str(GPX_FILES[0]))
    result = gpx.process(spacing_m=20, smoothing_method='savgol', window_size=7, max_grade=45.0)

    assert 'distances_m' in result
    assert 'elevations_m' in result
    assert 'grades_pct' in result
    assert 'total_distance_km' in result
    assert 'total_elevation_gain_m' in result

    grades = result['grades_pct']
    assert np.all(grades >= -45.1) and np.all(grades <= 45.1), "Grades should be clipped to ±45%"

    print(f"  Filtered points: {len(result['distances_m'])}")
    print(f"  Total distance: {result['total_distance_km']:.2f} km")
    print(f"  Filtered gain: {result['total_elevation_gain_m']:.0f} m")
    print(f"  Grade range: {np.min(grades):.1f}% ~ {np.max(grades):.1f}%")


def test_gpx_filter_save_filtered_gpx():
    """GPXFilter.save_filtered_gpx should produce a valid GPX file."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found in temp/routes/")
        return

    from core.gpx_filter import GPXFilter

    gpx = GPXFilter(str(GPX_FILES[0]))
    gpx.process(spacing_m=20)

    out_path = str(OUTPUT_DIR / 'test_filtered.gpx')
    gpx.save_filtered_gpx(out_path)
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0

    print(f"  Output: {out_path} ({Path(out_path).stat().st_size / 1024:.1f} KB)")


def test_gpx_filter_save_filtered_json():
    """GPXFilter.save_filtered_json should produce a valid JSON file."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found in temp/routes/")
        return

    from core.gpx_filter import GPXFilter

    gpx = GPXFilter(str(GPX_FILES[0]))
    gpx.process(spacing_m=20)

    out_path = str(OUTPUT_DIR / 'test_filtered.json')
    gpx.save_filtered_json(out_path)
    assert Path(out_path).exists()

    with open(out_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'segments' in data
    assert 'route_info' in data

    print(f"  Output: {out_path}")
    print(f"  Segments: {len(data['segments'])}")
    print(f"  Route distance: {data['route_info']['total_distance_km']:.2f} km")


# ---------------------------------------------------------------------------
# Test: core.predictor — FeatureExtractor (from FIT)
# ---------------------------------------------------------------------------

def test_feature_extractor_from_fit():
    """FeatureExtractor.extract_from_fit should extract segments from FIT files."""
    from core.predictor import FeatureExtractor

    if not FIT_FILES:
        print("  [SKIP] No FIT files found")
        return

    fit_path = FIT_FILES[0]
    segments, rest_ratio = FeatureExtractor.extract_from_fit(fit_path)

    assert isinstance(segments, list)
    assert isinstance(rest_ratio, float)
    assert 0 <= rest_ratio <= 0.5, f"Rest ratio out of range: {rest_ratio}"

    if segments:
        seg = segments[0]
        assert hasattr(seg, 'speed_kmh')
        assert hasattr(seg, 'grade_pct')
        assert hasattr(seg, 'accumulated_distance_km')
        assert hasattr(seg, 'accumulated_ascent_m')
        assert hasattr(seg, 'absolute_altitude_m')
        assert 0 < seg.speed_kmh < 25, f"Unreasonable speed: {seg.speed_kmh}"

    print(f"  File: {fit_path.name}")
    print(f"  Segments: {len(segments)}")
    print(f"  Rest ratio: {rest_ratio:.1%}")
    if segments:
        print(f"  First segment: speed={segments[0].speed_kmh:.2f} km/h, grade={segments[0].grade_pct:.1f}%")


def test_feature_extractor_all_fit_files():
    """FeatureExtractor should work on all FIT training files."""
    from core.predictor import FeatureExtractor

    if not FIT_FILES:
        print("  [SKIP] No FIT files found")
        return

    total_segments = 0
    file_results = []

    for fit_path in FIT_FILES:
        segments, rest_ratio = FeatureExtractor.extract_from_fit(fit_path)
        total_segments += len(segments)
        file_results.append({
            'file': fit_path.name,
            'segments': len(segments),
            'rest_ratio': rest_ratio,
        })
        print(f"  {fit_path.name}: {len(segments)} segments, rest={rest_ratio:.1%}")

    assert total_segments > 0, "No segments extracted from any FIT file"
    print(f"\n  Total: {len(FIT_FILES)} files, {total_segments} segments")


# ---------------------------------------------------------------------------
# Test: core.predictor — LightGBMPredictor
# ---------------------------------------------------------------------------

def test_lightgbm_predictor_train():
    """LightGBMPredictor.train should build a model from segment features."""
    from core.predictor import LightGBMPredictor, FeatureExtractor

    if not FIT_FILES:
        print("  [SKIP] No FIT files found")
        return

    # Extract segments from first few files
    all_segments = []
    for fit_path in FIT_FILES[:3]:
        segments, _ = FeatureExtractor.extract_from_fit(fit_path)
        all_segments.extend(segments)

    if len(all_segments) < 10:
        print(f"  [SKIP] Only {len(all_segments)} segments, need >= 10")
        return

    predictor = LightGBMPredictor()
    success = predictor.train(all_segments)

    assert success, "Training failed"
    assert predictor.is_trained
    assert predictor.model is not None
    assert len(predictor.feature_importance) > 0

    print(f"  Trained on {len(all_segments)} segments")
    print(f"  Feature importance:")
    for feat, imp in sorted(predictor.feature_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.1f}")


# ---------------------------------------------------------------------------
# Test: core.predictor — MLRacePredictor full workflow
# ---------------------------------------------------------------------------

_trained_predictor = None  # cached across tests


def _get_trained_predictor():
    """Train and cache an MLRacePredictor for reuse across tests."""
    global _trained_predictor
    if _trained_predictor is not None:
        return _trained_predictor

    from core.predictor import MLRacePredictor

    assert FIT_FILES, "No FIT files found for training"

    predictor = MLRacePredictor()
    training_paths = [str(f) for f in FIT_FILES]
    success = predictor.train_from_files(training_paths)
    assert success, "train_from_files failed"

    _trained_predictor = predictor
    return predictor


def test_mlrace_predictor_train():
    """MLRacePredictor.train_from_files should succeed with FIT records."""
    predictor = _get_trained_predictor()

    assert predictor.predictor is not None
    assert predictor.predictor.is_trained
    assert len(predictor.training_stats) > 0
    assert len(predictor.all_feature_importance) > 0

    stats = predictor.training_stats
    print(f"  Files: {stats.get('file_count')}")
    print(f"  Segments: {stats.get('segment_count')}")
    print(f"  Avg speed: {stats.get('avg_speed')} km/h")
    print(f"  P50 speed: {stats.get('p50_speed')} km/h")
    print(f"  P90 speed: {stats.get('p90_speed')} km/h")
    print(f"  Rest ratio: {stats.get('avg_rest_ratio', 0):.1%}")


def test_mlrace_predictor_parse_gpx_route():
    """MLRacePredictor.parse_gpx_route should parse GPX into segments."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found")
        return

    predictor = _get_trained_predictor()
    segments, route_info = predictor.parse_gpx_route(str(GPX_FILES[0]))

    assert len(segments) > 0
    assert 'total_distance_km' in route_info
    assert 'total_elevation_gain_m' in route_info
    assert 'total_elevation_loss_m' in route_info
    assert route_info['total_distance_km'] > 0

    print(f"  Segments: {len(segments)}")
    print(f"  Distance: {route_info['total_distance_km']:.2f} km")
    print(f"  Gain: {route_info['total_elevation_gain_m']:.0f} m")
    print(f"  Loss: {route_info['total_elevation_loss_m']:.0f} m")
    print(f"  Density: {route_info['elevation_density']:.1f} m/km")
    print(f"  Checkpoints: {route_info['checkpoint_count']}")

    # Check first segment
    seg = segments[0]
    print(f"\n  First segment: grade={seg.grade_pct:.1f}%, alt={seg.absolute_altitude_m:.0f}m")


def test_mlrace_predictor_predict_race():
    """MLRacePredictor.predict_race should return full prediction results."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found")
        return

    predictor = _get_trained_predictor()

    for effort, label in [(0.85, "Conservative"), (1.0, "Average P50"), (1.1, "Race P90")]:
        result = predictor.predict_race(str(GPX_FILES[0]), effort)

        # Validate result structure
        assert 'predicted_time_min' in result
        assert 'predicted_time_hm' in result
        assert 'predicted_pace_min_km' in result
        assert 'predicted_speed_kmh' in result
        assert 'segment_predictions' in result
        assert 'route_info' in result
        assert 'feature_importance' in result
        assert result['predicted_time_min'] > 0
        assert result['predicted_speed_kmh'] > 0
        assert len(result['segment_predictions']) > 0

        print(f"\n  [{label}] effort={effort}")
        print(f"    Time: {result['predicted_time_hm']} ({result['predicted_time_min']:.0f} min)")
        print(f"    Pace: {result['predicted_pace_min_km']:.1f} min/km")
        print(f"    Speed: {result['predicted_speed_kmh']:.2f} km/h")
        print(f"    Segments: {len(result['segment_predictions'])}")


def test_mlrace_predictor_segment_details():
    """Segment predictions should contain all required fields."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found")
        return

    predictor = _get_trained_predictor()
    result = predictor.predict_race(str(GPX_FILES[0]), 1.0)

    seg = result['segment_predictions'][0]
    required_keys = ['segment', 'distance_km', 'grade_pct', 'altitude_m',
                     'predicted_speed_kmh', 'segment_time_min', 'cumulative_time_min',
                     'grade_type', 'difficulty']
    for key in required_keys:
        assert key in seg, f"Missing key: {key}"

    print(f"  Segment structure (first): {json.dumps(seg, indent=2, default=str)}")

    # Verify cumulative time is monotonically increasing
    times = [s['cumulative_time_min'] for s in result['segment_predictions']]
    for i in range(1, len(times)):
        assert times[i] >= times[i-1], f"Cumulative time not increasing at segment {i}"


def test_mlrace_predictor_effort_scaling():
    """Higher effort_factor should produce faster (lower) predicted times."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found")
        return

    predictor = _get_trained_predictor()

    results = {}
    for effort in [0.85, 1.0, 1.1]:
        r = predictor.predict_race(str(GPX_FILES[0]), effort)
        results[effort] = r['predicted_time_min']

    # Lower effort => longer time
    assert results[0.85] > results[1.0] > results[1.1], \
        f"Effort scaling wrong: {results}"

    print(f"  Effort 0.85: {results[0.85]:.0f} min")
    print(f"  Effort 1.00: {results[1.0]:.0f} min")
    print(f"  Effort 1.10: {results[1.1]:.0f} min")
    print(f"  Range: {results[0.85] - results[1.1]:.0f} min spread")


def test_mlrace_predictor_save_results():
    """Save full prediction results to JSON for inspection."""
    if not GPX_FILES:
        print("  [SKIP] No GPX files found")
        return

    predictor = _get_trained_predictor()

    all_route_results = []
    for gpx_path in GPX_FILES:
        result = predictor.predict_race(str(gpx_path), 1.0)
        route_data = {
            'route_file': gpx_path.name,
            'predicted_time_hm': result['predicted_time_hm'],
            'predicted_time_min': result['predicted_time_min'],
            'predicted_pace_min_km': result['predicted_pace_min_km'],
            'predicted_speed_kmh': result['predicted_speed_kmh'],
            'total_distance_km': result['route_info']['total_distance_km'],
            'total_elevation_gain_m': result['route_info']['total_elevation_gain_m'],
            'segment_count': len(result['segment_predictions']),
        }
        all_route_results.append(route_data)
        print(f"  {gpx_path.name}: {route_data['predicted_time_hm']} "
              f"({route_data['predicted_pace_min_km']:.1f} min/km)")

    output = {
        'test_time': datetime.now().isoformat(),
        'training_files': [f.name for f in FIT_FILES],
        'training_stats': predictor.training_stats,
        'feature_importance': predictor.all_feature_importance,
        'routes': all_route_results,
    }
    out_path = OUTPUT_DIR / 'core_test_prediction_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    assert out_path.exists()
    print(f"\n  Results saved: {out_path}")


# ---------------------------------------------------------------------------
# Test: core.types — Type definitions
# ---------------------------------------------------------------------------

def test_type_definitions():
    """Core type definitions should be importable and usable."""
    from core.types import (
        EffortLevel, ValidationResult, TrainingResult,
        SegmentPrediction, PredictionResult, PerformanceAnalysis
    )

    # EffortLevel enum
    assert EffortLevel.HIGH.value == "high"
    assert EffortLevel.MEDIUM.value == "medium"
    assert EffortLevel.LOW.value == "low"

    # ValidationResult
    vr = ValidationResult(valid=True)
    assert bool(vr) is True
    vr2 = ValidationResult(valid=False, error="bad input")
    assert bool(vr2) is False

    # TrainingResult
    tr = TrainingResult(success=True, stats={'segments': 100})
    assert bool(tr) is True

    # SegmentPrediction
    sp = SegmentPrediction(
        segment_id=1, start_km=0, end_km=0.2, distance_km=0.2,
        grade_pct=5.0, altitude_m=500, predicted_speed_kmh=8.0,
        predicted_time_min=1.5, cumulative_time_min=1.5
    )
    d = sp.to_dict()
    assert d['segment_id'] == 1
    assert d['grade_pct'] == 5.0

    # PredictionResult
    pr = PredictionResult(
        total_time_min=120, total_time_hm="2:00:00", pace_min_km=6.0,
        speed_kmh=10.0, total_distance_km=20, total_ascent_m=1000,
        total_descent_m=800, elevation_density=50.0, segments=[sp],
        feature_importance={'grade': 100}, model_confidence=0.85,
        effort_level='high'
    )
    pr_dict = pr.to_dict()
    assert pr_dict['summary']['total_time_hm'] == "2:00:00"
    assert len(pr_dict['segments']) == 1

    print("  EffortLevel: OK")
    print("  ValidationResult: OK")
    print("  TrainingResult: OK")
    print("  SegmentPrediction: OK")
    print("  PredictionResult: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Core Module Function Tests")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"FIT files: {len(FIT_FILES)} in {RECORDS_DIR}")
    print(f"GPX files: {len(GPX_FILES)} in {ROUTES_DIR}")
    print("=" * 60)

    runner = TestRunner()

    # --- utils ---
    runner.run("FilterConfig defaults", test_filter_config_defaults)
    runner.run("ElevationFilter.smooth", test_elevation_filter_smooth)
    runner.run("ElevationFilter.calculate_grade", test_elevation_filter_calculate_grade)
    runner.run("GradeAnalyzer.analyze_distribution", test_grade_analyzer_distribution)
    runner.run("GradeAnalyzer.calculate_climbing_loss", test_grade_analyzer_climbing_loss)
    runner.run("apply_fit_filter", test_apply_fit_filter)
    runner.run("apply_gpx_filter", test_apply_gpx_filter)
    runner.run("FilterConfig.calibrate_from_fit_files", test_calibrate_from_fit_files)

    # --- gpx_filter ---
    runner.run("GPXFilter.parse_gpx", test_gpx_filter_parse)
    runner.run("GPXFilter.process (full pipeline)", test_gpx_filter_full_process)
    runner.run("GPXFilter.save_filtered_gpx", test_gpx_filter_save_filtered_gpx)
    runner.run("GPXFilter.save_filtered_json", test_gpx_filter_save_filtered_json)

    # --- predictor: feature extraction ---
    runner.run("FeatureExtractor.extract_from_fit (single)", test_feature_extractor_from_fit)
    runner.run("FeatureExtractor.extract_from_fit (all files)", test_feature_extractor_all_fit_files)

    # --- predictor: LightGBM ---
    runner.run("LightGBMPredictor.train", test_lightgbm_predictor_train)

    # --- predictor: MLRacePredictor full workflow ---
    runner.run("MLRacePredictor.train_from_files", test_mlrace_predictor_train)
    runner.run("MLRacePredictor.parse_gpx_route", test_mlrace_predictor_parse_gpx_route)
    runner.run("MLRacePredictor.predict_race", test_mlrace_predictor_predict_race)
    runner.run("MLRacePredictor segment details", test_mlrace_predictor_segment_details)
    runner.run("MLRacePredictor effort scaling", test_mlrace_predictor_effort_scaling)
    runner.run("MLRacePredictor save results", test_mlrace_predictor_save_results)

    # --- types ---
    runner.run("Type definitions", test_type_definitions)

    # --- Summary ---
    success = runner.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
