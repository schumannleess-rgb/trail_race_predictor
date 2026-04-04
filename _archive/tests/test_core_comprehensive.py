"""
Trail Race Predictor - Comprehensive Core Function Tests

Comprehensive test suite for validating all core folder functions:
- Data discovery and file collection
- Filter utils (FilterConfig, ElevationFilter, GradeAnalyzer, apply_fit_filter, apply_gpx_filter)
- GPX filter (GPXFilter.parse_gpx, process, save methods)
- Feature extraction (FeatureExtractor.extract_from_fit)
- ML training (LightGBMPredictor.train)
- Full prediction workflow (MLRacePredictor train, parse, predict)
- Report generation (ReportGenerator HTML and TXT)
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import MLRacePredictor, FeatureExtractor, LightGBMPredictor, SegmentFeatures
from core.gpx_filter import GPXFilter
from core.utils import FilterConfig, ElevationFilter, GradeAnalyzer, apply_fit_filter, apply_gpx_filter
from core.types import PredictionResult
from reports.report_generator import ReportGenerator


# Test configuration
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
RECORDS_DIR = TEMP_DIR / "records"
ROUTES_DIR = TEMP_DIR / "routes"
OUTPUT_DIR = TEMP_DIR / "output"

# Output paths
TEST_REPORT_PATH = OUTPUT_DIR / "core_test_report.json"
PREDICTION_JSON_PATH = OUTPUT_DIR / "prediction_results.json"
HTML_REPORT_PATH = OUTPUT_DIR / "prediction_report.html"
TXT_REPORT_PATH = OUTPUT_DIR / "prediction_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "core_test_summary.txt"


def discover_files() -> Dict[str, List[Path]]:
    """Discover all FIT and GPX files for testing"""
    print("\n[1/7] Discovering data files...")

    # Collect FIT files (Windows glob is case-insensitive, need deduplicate)
    fit_files = list(RECORDS_DIR.glob("*.fit"))
    fit_files.extend(list(RECORDS_DIR.glob("*.FIT")))
    # Try .gz and .zip variants
    fit_files.extend(list(RECORDS_DIR.glob("*.fit.gz")))
    fit_files.extend(list(RECORDS_DIR.glob("*.FIT.gz")))
    fit_files.extend(list(RECORDS_DIR.glob("*.zip")))
    # Deduplicate by lowercase path
    seen = set()
    unique_fit_files = []
    for f in fit_files:
        key = f.name.lower()
        if key not in seen:
            seen.add(key)
            unique_fit_files.append(f)
    fit_files = unique_fit_files

    # Collect GPX files (Windows glob is case-insensitive, need deduplicate)
    gpx_files = list(ROUTES_DIR.glob("*.gpx"))
    gpx_files.extend(list(ROUTES_DIR.glob("*.GPX")))
    # Deduplicate by lowercase path
    seen = set()
    unique_gpx_files = []
    for f in gpx_files:
        key = f.name.lower()
        if key not in seen:
            seen.add(key)
            unique_gpx_files.append(f)
    gpx_files = unique_gpx_files

    print(f"  Found {len(fit_files)} FIT files")
    print(f"  Found {len(gpx_files)} GPX files")

    return {
        'fit_files': fit_files,
        'gpx_files': gpx_files
    }


def test_filter_utils(files: Dict[str, List[Path]]) -> Dict:
    """Test filter utilities"""
    print("\n[2/7] Testing filter utils...")
    results = {
        'FilterConfig': {'passed': False, 'details': ''},
        'ElevationFilter': {'passed': False, 'details': ''},
        'GradeAnalyzer': {'passed': False, 'details': ''},
        'apply_fit_filter': {'passed': False, 'details': ''},
        'apply_gpx_filter': {'passed': False, 'details': ''}
    }

    # Test FilterConfig
    try:
        gpx_config = FilterConfig.GPX
        fit_config = FilterConfig.FIT
        results['FilterConfig']['passed'] = True
        results['FilterConfig']['details'] = f"GPX: {gpx_config.get('window_size')}w/{gpx_config.get('poly_order')}p, FIT: {fit_config.get('window_size')}w/{fit_config.get('poly_order')}p"
        print(f"  FilterConfig: {results['FilterConfig']['details']}")
    except Exception as e:
        results['FilterConfig']['details'] = str(e)

    # Test FilterConfig.calibrate_from_fit_files
    try:
        fit_paths = [str(f) for f in files['fit_files'][:3]]  # Use first 3 files
        if fit_paths:
            calibrated = FilterConfig.calibrate_from_fit_files(fit_paths)
            results['FilterConfig']['details'] += f", calibrated max_grade={calibrated.get('max_grade_pct', 0):.0f}%"
    except Exception as e:
        pass  # Calibration may fail if files are not readable

    # Test ElevationFilter
    try:
        elevations = [100, 101, 102, 101, 100]
        smoothed = ElevationFilter.smooth(elevations, FilterConfig.GPX)
        results['ElevationFilter']['passed'] = len(smoothed) == len(elevations)
        results['ElevationFilter']['details'] = f"smoothed {len(elevations)} points"
    except Exception as e:
        results['ElevationFilter']['details'] = str(e)

    # Test GradeAnalyzer
    try:
        import numpy as np
        grades = np.array([-15, -8, 0, 5, 12, 25])
        distribution = GradeAnalyzer.analyze_distribution(grades, spacing_m=200)
        results['GradeAnalyzer']['passed'] = len(distribution) > 0
        results['GradeAnalyzer']['details'] = f"analyzed {len(grades)} grade points"
    except Exception as e:
        results['GradeAnalyzer']['details'] = str(e)

    # Test apply_fit_filter
    try:
        elevations = [100.0, 101.0, 102.0, 101.5, 100.0] * 20
        smoothed, info = apply_fit_filter(elevations)
        results['apply_fit_filter']['passed'] = 'points' in info
        results['apply_fit_filter']['details'] = f"filtered {info.get('points')} points, noise_std={info.get('noise_std_m')}m"
    except Exception as e:
        results['apply_fit_filter']['details'] = str(e)

    # Test apply_gpx_filter
    try:
        import numpy as np
        elevations = [100.0] * 50
        distances = list(range(0, 1000, 20))  # 50 points, 0-1000m
        smoothed, info = apply_gpx_filter(elevations, distances, original_gain_m=500)
        results['apply_gpx_filter']['passed'] = 'points' in info
        results['apply_gpx_filter']['details'] = f"filtered {info.get('points')} points, gain={info.get('filtered_gain_m')}m"
    except Exception as e:
        results['apply_gpx_filter']['details'] = str(e)

    return results


def test_gpx_filter(files: Dict[str, List[Path]]) -> Dict:
    """Test GPX filter"""
    print("\n[3/7] Testing GPX filter...")
    results = {
        'parse_gpx': {'passed': False, 'details': ''},
        'process': {'passed': False, 'details': ''},
        'save': {'passed': False, 'details': ''}
    }

    if not files.get('gpx_files'):
        results['parse_gpx']['details'] = "No GPX files found"
        return results

    gpx_path = files['gpx_files'][0]
    print(f"  Testing with: {gpx_path.name}")

    try:
        # Test parse_gpx
        gpx_filter = GPXFilter(str(gpx_path))
        raw_data = gpx_filter.parse_gpx()
        results['parse_gpx']['passed'] = raw_data is not None and 'points' in raw_data
        results['parse_gpx']['details'] = f"parsed {len(raw_data.get('points', []))} points, {len(raw_data.get('waypoints', []))} waypoints"
        print(f"    parse_gpx: {results['parse_gpx']['details']}")

        # Test process
        filtered_data = gpx_filter.process(
            spacing_m=20,
            smoothing_method='savgol',
            window_size=7,
            poly_order=2,
            max_grade=45.0
        )
        results['process']['passed'] = filtered_data is not None and 'distances_m' in filtered_data
        results['process']['details'] = f"processed {len(filtered_data.get('distances_m', []))} points, gain={filtered_data.get('total_elevation_gain_m', 0):.0f}m"

        # Test save (temp output)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temp_gpx = OUTPUT_DIR / f"test_filtered_{gpx_path.stem}.gpx"
        gpx_filter.save_filtered_gpx(str(temp_gpx))
        results['save']['passed'] = temp_gpx.exists()
        results['save']['details'] = f"saved to {temp_gpx.name}, size={temp_gpx.stat().st_size} bytes"
        print(f"    process: {results['process']['details']}")
        print(f"    save: {results['save']['details']}")

    except Exception as e:
        results['parse_gpx']['details'] = str(e)
        results['process']['details'] = str(e)
        results['save']['details'] = str(e)

    return results


def test_feature_extraction(files: Dict[str, List[Path]]) -> Dict:
    """Test feature extraction from FIT files"""
    print("\n[4/7] Testing feature extraction...")
    results = {
        'extract_from_fit': {'passed': False, 'details': ''}
    }

    if not files.get('fit_files'):
        results['extract_from_fit']['details'] = "No FIT files found"
        return results

    # Try each FIT file until one works - limit to 2 files for speed
    max_files = 2
    for fit_path in files['fit_files'][:max_files]:
        try:
            segments, duration = FeatureExtractor.extract_from_fit(Path(fit_path))
            if segments and len(segments) > 0:
                results['extract_from_fit']['passed'] = True
                results['extract_from_fit']['details'] = f"extracted {len(segments)} segments from {fit_path.name}, duration={duration:.1f}min"
                print(f"  extract_from_fit: {results['extract_from_fit']['details']}")
                break
        except Exception as e:
            print(f"    Failed on {fit_path.name}: {e}")
            continue

    if not results['extract_from_fit']['passed']:
        results['extract_from_fit']['details'] = "Failed to extract features from all FIT files"

    return results


def test_ml_training(files: Dict[str, List[Path]]) -> Dict:
    """Test ML training"""
    print("\n[5/7] Testing ML training...")
    results = {
        'LightGBMPredictor_train': {'passed': False, 'details': ''}
    }

    if not files.get('fit_files'):
        results['LightGBMPredictor_train']['details'] = "No FIT files found"
        return results

    try:
        # Create predictor and train
        predictor = LightGBMPredictor()

        # Collect segments from all FIT files
        all_segments = []
        for fit_path in files['fit_files']:
            try:
                segments, _ = FeatureExtractor.extract_from_fit(Path(fit_path))
                all_segments.extend(segments)
            except:
                continue

        if len(all_segments) < 10:
            results['LightGBMPredictor_train']['details'] = f"Insufficient segments: {len(all_segments)}"
            return results

        # Train model
        success = predictor.train(all_segments)
        results['LightGBMPredictor_train']['passed'] = success

        if success:
            # Get feature importance
            importance = predictor.get_feature_importance()
            results['LightGBMPredictor_train']['details'] = f"trained with {len(all_segments)} segments, importance={importance}"
            print(f"  LightGBMPredictor.train: {len(all_segments)} segments trained")
        else:
            results['LightGBMPredictor_train']['details'] = "Training failed"

    except Exception as e:
        results['LightGBMPredictor_train']['details'] = str(e)

    return results


def test_full_prediction(files: Dict[str, List[Path]]) -> Dict:
    """Test full prediction workflow"""
    print("\n[6/7] Testing full prediction...")
    results = {
        'train_from_files': {'passed': False, 'details': ''},
        'parse_gpx_route': {'passed': False, 'details': ''},
        'predict_race': {'passed': False, 'details': ''}
    }

    if not files.get('fit_files') or not files.get('gpx_files'):
        results['train_from_files']['details'] = "Missing FIT or GPX files"
        return results

    try:
        predictor = MLRacePredictor()

        # Train from files
        fit_paths = [str(f) for f in files['fit_files']]
        success = predictor.train_from_files(fit_paths)
        results['train_from_files']['passed'] = success
        results['train_from_files']['details'] = f"trained from {len(fit_paths)} files"
        print(f"  train_from_files: trained from {len(fit_paths)} files")

        if not success:
            return results

        # Parse GPX route
        gpx_path = files['gpx_files'][0]
        segments, route_info = predictor.parse_gpx_route(str(gpx_path))
        results['parse_gpx_route']['passed'] = len(segments) > 0
        results['parse_gpx_route']['details'] = f"parsed {len(segments)} segments from {gpx_path.name}"
        print(f"  parse_gpx_route: {len(segments)} segments")

        if not results['parse_gpx_route']['passed']:
            return results

        # Predict race
        prediction = predictor.predict_race(str(gpx_path))
        results['predict_race']['passed'] = prediction is not None and prediction.get('predicted_time_min', 0) > 0
        results['predict_race']['details'] = f"predicted {prediction.get('predicted_time_hm', 'N/A')} for {gpx_path.name}"

        if results['predict_race']['passed']:
            print(f"  predict_race: {prediction.get('predicted_time_hm', 'N/A')}")

            # Save prediction results
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(PREDICTION_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(prediction, f, ensure_ascii=False, indent=2)
            print(f"  Saved prediction to: {PREDICTION_JSON_PATH.name}")

    except Exception as e:
        results['train_from_files']['details'] = str(e)
        results['parse_gpx_route']['details'] = str(e)
        results['predict_race']['details'] = str(e)

    return results


def test_report_generation(prediction: Dict) -> Dict:
    """Test report generation"""
    print("\n[7/7] Testing report generation...")
    results = {
        'HTML_report': {'passed': False, 'details': ''},
        'TXT_report': {'passed': False, 'details': ''}
    }

    if not prediction:
        results['HTML_report']['details'] = "No prediction data"
        results['TXT_report']['details'] = "No prediction data"
        return results

    try:
        # Create prediction result from dict
        result = _create_prediction_result(prediction)

        gpx_name = prediction.get('route_info', {}).get('name', 'test.gpx')
        fit_names = [Path(f).stem for f in prediction.get('training_files', [])]

        # Generate HTML report
        reporter = ReportGenerator(result, gpx_name, fit_names)
        html_path = reporter.generate_html_report(str(HTML_REPORT_PATH))
        results['HTML_report']['passed'] = Path(html_path).exists()
        results['HTML_report']['details'] = f"generated {HTML_REPORT_PATH.name}, size={Path(html_path).stat().st_size} bytes"
        print(f"  generate_html_report: {results['HTML_report']['details']}")

        # Generate TXT report
        txt_content = reporter.generate_txt_report()
        with open(TXT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        results['TXT_report']['passed'] = Path(TXT_REPORT_PATH).exists()
        results['TXT_report']['details'] = f"generated {TXT_REPORT_PATH.name}, size={Path(TXT_REPORT_PATH).stat().st_size} bytes"
        print(f"  generate_txt_report: {results['TXT_report']['details']}")

    except Exception as e:
        results['HTML_report']['details'] = str(e)
        results['TXT_report']['details'] = str(e)

    return results


def _create_prediction_result(prediction: Dict) -> PredictionResult:
    """Create PredictionResult object from prediction dict"""
    summary = {
        'total_time_min': prediction.get('predicted_time_min', 0),
        'total_time_hm': prediction.get('predicted_time_hm', '0:00:00'),
        'pace_min_km': prediction.get('predicted_pace_min_km', 0),
        'speed_kmh': prediction.get('predicted_speed_kmh', 0),
        'total_distance_km': prediction.get('total_distance_km', 0),
        'total_ascent_m': prediction.get('route_info', {}).get('total_elevation_gain_m', 0),
        'total_descent_m': prediction.get('route_info', {}).get('total_elevation_loss_m', 0),
        'elevation_density': prediction.get('route_info', {}).get('elevation_density', 0),
    }

    # Create segment predictions
    # predict_race returns 'segment_predictions', not 'segments'
    segments = []
    for seg in prediction.get('segment_predictions', []):
        from core.types import SegmentPrediction
        sp = SegmentPrediction(
            segment_id=seg.get('segment', 0),
            start_km=0,  # Not provided in segment_predictions
            end_km=0,    # Not provided in segment_predictions
            distance_km=seg.get('distance_km', 0),
            grade_pct=seg.get('grade_pct', 0),
            altitude_m=seg.get('altitude_m', 0),  # Key is 'altitude_m' not 'elevation_m'
            predicted_speed_kmh=seg.get('predicted_speed_kmh', 0),
            predicted_time_min=seg.get('segment_time_min', 0),  # Key is 'segment_time_min'
            cumulative_time_min=seg.get('cumulative_time_min', 0),
            difficulty_level=seg.get('difficulty', 'moderate'),
            grade_type=seg.get('grade_type', '平地')
        )
        segments.append(sp)

    # Create feature importance dict
    feature_importance = prediction.get('feature_importance', {})

    result = PredictionResult(
        total_time_min=summary.get('total_time_min', 0),
        total_time_hm=summary.get('total_time_hm', '0:00:00'),
        pace_min_km=summary.get('pace_min_km', 0),
        speed_kmh=summary.get('speed_kmh', 0),
        total_distance_km=summary.get('total_distance_km', 0),
        total_ascent_m=summary.get('total_ascent_m', 0),
        total_descent_m=summary.get('total_descent_m', 0),
        elevation_density=summary.get('elevation_density', 0),
        segments=segments,
        feature_importance=feature_importance,
        model_confidence=prediction.get('model_confidence', 0.8),
        effort_level=prediction.get('effort_level', 'medium'),
        training_stats=prediction.get('training_stats', {}),
        warnings=prediction.get('warnings', [])
    )

    return result


def run_all_tests() -> Dict:
    """Run all tests and generate report"""
    print("=" * 60)
    print("Comprehensive Core Function Tests")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover files
    files = discover_files()

    # Run tests
    results = {}

    # Test group 1: Filter utils
    results['filter_utils'] = test_filter_utils(files)

    # Test group 2: GPX filter
    results['gpx_filter'] = test_gpx_filter(files)

    # Test group 3: Feature extraction
    results['feature_extraction'] = test_feature_extraction(files)

    # Test group 4: ML training
    results['ml_training'] = test_ml_training(files)

    # Test group 5: Full prediction
    prediction_results = test_full_prediction(files)
    results['full_prediction'] = prediction_results

    # Test group 6: Report generation
    prediction_data = None
    if PREDICTION_JSON_PATH.exists():
        with open(PREDICTION_JSON_PATH, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)

    results['report_generation'] = test_report_generation(prediction_data)

    return results


def generate_summary_report(results: Dict) -> str:
    """Generate human-readable summary report"""
    lines = []

    lines.append("=" * 60)
    lines.append("Comprehensive Core Function Tests - Summary")
    lines.append("=" * 60)
    lines.append("")

    total_tests = 0
    passed_tests = 0

    for group_name, group_results in results.items():
        lines.append(f"\n[{group_name}]")
        for test_name, test_result in group_results.items():
            status = "PASS" if test_result.get('passed') else "FAIL"
            total_tests += 1
            if test_result.get('passed'):
                passed_tests += 1
            lines.append(f"  {test_name}: {status} - {test_result.get('details', '')}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total: {passed_tests}/{total_tests} tests passed")
    lines.append(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    lines.append("=" * 60)

    summary = "\n".join(lines)

    # Save summary
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\nSummary saved to: {SUMMARY_PATH}")

    return summary


def main():
    """Main entry point"""
    try:
        # Run all tests
        results = run_all_tests()

        # Generate summary
        summary = generate_summary_report(results)

        # Save test report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(TEST_REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nTest report saved to: {TEST_REPORT_PATH}")
        print("\n" + summary)

        return 0 if all(
            all(t.get('passed') for t in g.values())
            for g in results.values()
        ) else 1

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())