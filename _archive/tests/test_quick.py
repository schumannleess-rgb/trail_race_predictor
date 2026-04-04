"""Quick test runner for core functions"""
import os
os.chdir(r'D:\Garmin\garmin-fitness-v3\projects\race_predictor\trail_race_predictor_v1.2.1')

from pathlib import Path
import json

BASE_DIR = Path('.')
TEMP_DIR = BASE_DIR / 'temp'
OUTPUT_DIR = TEMP_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

def run_tests():
    results = {}
    passed = 0
    total = 0

    # ===== TEST 1: Filter Utils =====
    print("\n[1/7] Filter Utils...")
    from core.utils import FilterConfig, ElevationFilter, GradeAnalyzer
    total += 1

    # Test FilterConfig
    gpx_cfg = FilterConfig.GPX
    fit_cfg = FilterConfig.FIT
    results['filter_config'] = {'passed': True, 'details': f"GPX window={gpx_cfg['window_size']}, FIT window={fit_cfg['window_size']}"}
    passed += 1
    print(f"  FilterConfig: {results['filter_config']['details']}")

    # Test ElevationFilter
    total += 1
    smoothed = ElevationFilter.smooth([100,101,102,101,100], FilterConfig.GPX)
    results['elevation_filter'] = {'passed': len(smoothed) == 5, 'details': f"smoothed {len(smoothed)} points"}
    if results['elevation_filter']['passed']:
        passed += 1
    print(f"  ElevationFilter: {results['elevation_filter']['details']}")

    # Test GradeAnalyzer
    total += 1
    import numpy as np
    grades = np.array([-15, -8, 0, 5, 12, 25])
    dist = GradeAnalyzer.analyze_distribution(grades, spacing_m=200)
    results['grade_analyzer'] = {'passed': len(dist) > 0, 'details': f"analyzed {len(grades)} grades"}
    if results['grade_analyzer']['passed']:
        passed += 1
    print(f"  GradeAnalyzer: {results['grade_analyzer']['details']}")

    # ===== TEST 2: GPX Filter =====
    print("\n[2/7] GPX Filter...")
    from core.gpx_filter import GPXFilter
    gpx_files = list((TEMP_DIR / 'routes').glob('*.gpx'))
    total += 1

    if gpx_files:
        gpx = GPXFilter(str(gpx_files[0]))
        raw = gpx.parse_gpx()
        results['gpx_parse'] = {'passed': True, 'details': f"{len(raw['points'])} points, {len(raw['waypoints'])} waypoints"}
        passed += 1
        print(f"  parse_gpx: {results['gpx_parse']['details']}")

        filtered = gpx.process()
        total += 1
        results['gpx_process'] = {'passed': True, 'details': f"{len(filtered['distances_m'])} points, gain={filtered['total_elevation_gain_m']:.0f}m"}
        passed += 1
        print(f"  process: {results['gpx_process']['details']}")
    else:
        results['gpx_parse'] = {'passed': False, 'details': 'No GPX files found'}
        results['gpx_process'] = {'passed': False, 'details': 'No GPX files found'}
        total += 2

    # ===== TEST 3: Feature Extraction =====
    print("\n[3/7] Feature Extraction...")
    from core.predictor import FeatureExtractor
    fit_files = list((TEMP_DIR / 'records').glob('*.fit'))
    total += 1

    if fit_files:
        # Use Path objects directly (not string paths) to avoid encoding issues
        segs, duration = FeatureExtractor.extract_from_fit(fit_files[0])
        results['extract_from_fit'] = {'passed': len(segs) > 0, 'details': f"{len(segs)} segments from {fit_files[0].name}"}
        if results['extract_from_fit']['passed']:
            passed += 1
        print(f"  extract_from_fit: {results['extract_from_fit']['details']}")
    else:
        results['extract_from_fit'] = {'passed': False, 'details': 'No FIT files found'}

    # ===== TEST 4: ML Training =====
    print("\n[4/7] ML Training...")
    from core.predictor import MLRacePredictor
    total += 1

    if fit_files:
        predictor = MLRacePredictor()
        # Pass Path objects like above
        train_result = predictor.train_from_files([str(f) for f in fit_files[:2]])
        results['ml_train'] = {'passed': train_result, 'details': f"trained from {len(fit_files[:2])} files"}
        if results['ml_train']['passed']:
            passed += 1
        print(f"  LightGBMPredictor.train: {results['ml_train']['details']}")
    else:
        results['ml_train'] = {'passed': False, 'details': 'No FIT files found'}

    # ===== TEST 5: Full Prediction =====
    print("\n[5/7] Full Prediction...")
    total += 1

    if train_result and gpx_files:
        pred = predictor.predict_race(str(gpx_files[0]))
        if pred:
            results['predict_race'] = {'passed': True, 'details': f"{pred['summary']['total_time_hm']} for {gpx_files[0].name}"}
            passed += 1
            print(f"  predict_race: {results['predict_race']['details']}")

            # Save prediction
            with open(OUTPUT_DIR / 'prediction_results.json', 'w', encoding='utf-8') as f:
                json.dump(pred, f, ensure_ascii=False, indent=2)
        else:
            results['predict_race'] = {'passed': False, 'details': 'Prediction failed'}
    else:
        results['predict_race'] = {'passed': False, 'details': 'Training or GPX failed'}

    # ===== TEST 6: Report Generation =====
    print("\n[6/7] Report Generation...")
    pred_path = OUTPUT_DIR / 'prediction_results.json'
    total += 2

    if pred_path.exists():
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)

        from core.types import PredictionResult, SegmentPrediction
        from reports.report_generator import ReportGenerator

        # Create result object
        summary = pred.get('summary', {})
        segments = [SegmentPrediction(
            segment_id=s['segment_id'],
            start_km=s['start_km'],
            end_km=s['end_km'],
            distance_km=s['distance_km'],
            grade_pct=s['grade_pct'],
            altitude_m=s['altitude_m'],
            predicted_speed_kmh=s['predicted_speed_kmh'],
            predicted_time_min=s['predicted_time_min'],
            cumulative_time_min=s['cumulative_time_min'],
            difficulty_level=s['difficulty_level'],
            grade_type=s.get('grade_type', '平地')
        ) for s in pred.get('segments', [])]

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
            feature_importance=pred.get('feature_importance', {}),
            model_confidence=pred.get('model_confidence', 0.8),
            effort_level=pred.get('effort_level', 'medium')
        )

        # Generate HTML report
        gpx_name = gpx_files[0].name if gpx_files else 'test.gpx'
        fit_names = [f.stem for f in fit_files[:3]]
        reporter = ReportGenerator(result, gpx_name, fit_names)

        html_path = reporter.generate_html_report(str(OUTPUT_DIR / 'prediction_report.html'))
        results['html_report'] = {'passed': Path(html_path).exists(), 'details': f"generated {Path(html_path).name}"}
        if results['html_report']['passed']:
            passed += 1
        print(f"  HTML report: {results['html_report']['details']}")

        # Generate TXT report
        txt_content = reporter.generate_txt_report()
        with open(OUTPUT_DIR / 'prediction_report.txt', 'w', encoding='utf-8') as f:
            f.write(txt_content)
        results['txt_report'] = {'passed': Path(OUTPUT_DIR / 'prediction_report.txt').exists(), 'details': f"generated prediction_report.txt"}
        if results['txt_report']['passed']:
            passed += 1
        print(f"  TXT report: {results['txt_report']['details']}")

    else:
        results['html_report'] = {'passed': False, 'details': 'No prediction data'}
        results['txt_report'] = {'passed': False, 'details': 'No prediction data'}

    # ===== SUMMARY =====
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 50)

    # Save report
    with open(OUTPUT_DIR / 'core_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_text = "Comprehensive Core Function Tests\n"
    summary_text += "=" * 50 + "\n\n"
    for name, res in results.items():
        status = "PASS" if res['passed'] else "FAIL"
        summary_text += f"{name}: {status} - {res['details']}\n"
    summary_text += f"\nTotal: {passed}/{total} passed\n"
    summary_text += f"Success rate: {passed/total*100:.0f}%\n"

    with open(OUTPUT_DIR / 'core_test_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nReports saved to: {OUTPUT_DIR}")
    return passed == total

if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)