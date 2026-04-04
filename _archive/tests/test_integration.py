"""
Integration Test for Trail Race Predictor V1.2

Tests the complete workflow:
1. Load training data from temp/records/
2. Train the ML model
3. Load GPX route from temp/routes/
4. Predict race time
5. Validate results
"""

import sys
import os
from pathlib import Path
import time
import json

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.predictor import MLRacePredictor, FeatureExtractor

# Paths
RECORDS_DIR = Path("temp/records")
ROUTES_DIR = Path("temp/routes")

def test_integration():
    """Run complete integration test"""
    print("=" * 70)
    print("INTEGRATION TEST - Trail Race Predictor V1.2")
    print("=" * 70)

    # Step 1: Collect training files
    print("\n[Step 1/5] Collecting training files...")
    # Fix: Deduplicate files on case-insensitive filesystems (Windows)
    fit_files = list(set(list(RECORDS_DIR.glob("*.fit")) + list(RECORDS_DIR.glob("*.FIT"))))

    if not fit_files:
        print(f"  ERROR: No FIT files found in {RECORDS_DIR}")
        return False

    print(f"  Found {len(fit_files)} training files:")
    for f in sorted(fit_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name} ({size_mb:.1f} MB)")

    training_files = [str(f) for f in fit_files]

    # Step 2: Train the model
    print("\n[Step 2/5] Training ML model...")
    predictor = MLRacePredictor()

    start_time = time.time()
    success = predictor.train_from_files(training_files)
    train_time = time.time() - start_time

    if not success:
        print("  ERROR: Training failed!")
        return False

    print(f"  Training completed in {train_time:.2f}s")
    print(f"  Training stats:")
    for key, value in predictor.training_stats.items():
        print(f"    {key}: {value}")

    # Step 3: Load GPX route
    print("\n[Step 3/5] Loading GPX route...")
    gpx_files = list(ROUTES_DIR.glob("*.gpx"))

    if not gpx_files:
        print(f"  ERROR: No GPX files found in {ROUTES_DIR}")
        return False

    gpx_path = str(gpx_files[0])
    print(f"  Using route: {gpx_files[0].name}")

    # Step 4: Parse route and validate
    print("\n[Step 4/5] Parsing route...")
    try:
        segments, route_info = predictor.parse_gpx_route(gpx_path)
        print(f"  Route parsed successfully:")
        print(f"    Total distance: {route_info['total_distance_km']} km")
        print(f"    Total elevation gain: {route_info['total_elevation_gain_m']} m")
        print(f"    Number of segments: {len(segments)}")
    except Exception as e:
        print(f"  ERROR: Failed to parse route: {e}")
        return False

    # Step 5: Predict race time
    print("\n[Step 5/5] Predicting race time...")

    results = {}
    for effort in [0.9, 1.0, 1.1]:
        try:
            result = predictor.predict_race(gpx_path, effort)
            results[effort] = result
            print(f"\n  Effort Factor: {effort}x")
            print(f"    Predicted time: {result['predicted_time_hm']} ({result['predicted_time_min']} min)")
            print(f"    Average speed: {result['predicted_speed_kmh']} km/h")
            print(f"    Pace: {result['predicted_pace_min_km']} min/km")
        except Exception as e:
            print(f"  ERROR: Prediction failed for effort {effort}: {e}")
            return False

    # Feature importance
    print("\n[Feature Importance]")
    if predictor.all_feature_importance:
        for feature, importance in sorted(predictor.all_feature_importance.items(),
                                          key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.2f}")

    # Save results
    output_path = Path("temp/integration_test_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_name': 'Integration Test',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_files': [f.name for f in fit_files],
            'training_stats': predictor.training_stats,
            'route_info': route_info,
            'predictions': {
                f'effort_{k}': {
                    'time_hm': v['predicted_time_hm'],
                    'time_min': v['predicted_time_min'],
                    'speed_kmh': v['predicted_speed_kmh'],
                    'pace_min_km': v['predicted_pace_min_km']
                }
                for k, v in results.items()
            },
            'feature_importance': predictor.all_feature_importance
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"{'Effort':<10} {'Time':<15} {'Speed (km/h)':<15} {'Pace (min/km)':<15}")
    print("-" * 70)
    for effort, result in results.items():
        print(f"{effort}x{'':<8} {result['predicted_time_hm']:<15} {result['predicted_speed_kmh']:<15.2f} {result['predicted_pace_min_km']:<15.1f}")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST PASSED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
