"""Test: same features, same data, different predictions?"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from core.predictor import MLRacePredictor as OLD
from core_rebuild.predictor import MLRacePredictor as NEW
from core.predictor import SegmentFeatures as OLD_SF
from core_rebuild.predictor.features import SegmentFeatures as NEW_SF

records_dir = Path('temp/records')
fit_files = sorted(records_dir.glob('*.fit'), key=lambda p: p.stat().st_size, reverse=True)[:2]
train_files = [str(f) for f in fit_files]

print("Training OLD...", flush=True)
old = OLD()
old.train_from_files(train_files)

print("Training NEW...", flush=True)
new = NEW()
new.train_from_files(train_files)

print(f"OLD p50={old.predictor.p50_speed:.4f} p90={old.predictor.p90_speed:.4f}", flush=True)
print(f"NEW p50={new._model.p50_speed:.4f} p90={new._model.p90_speed:.4f}", flush=True)

# Same flat segment input
flat_old = OLD_SF(
    speed_kmh=6.0, grade_pct=0.0, rolling_grade_500m=0.0,
    accumulated_distance_km=0.5, accumulated_ascent_m=5.0,
    absolute_altitude_m=100.0, elevation_density=10.0,
    is_climbing=False, is_descending=False
)
flat_new = NEW_SF(
    speed_kmh=6.0, grade_pct=0.0, rolling_grade_500m=0.0,
    accumulated_distance_km=0.5, accumulated_ascent_m=5.0,
    absolute_altitude_m=100.0, elevation_density=10.0,
    is_climbing=False, is_descending=False
)

old_speed = old.predictor.predict_speed(flat_old, 1.0)
new_speed = new._model.predict_speed(flat_new, 1.0)

print(f"OLD speed for flat segment: {old_speed:.4f} km/h", flush=True)
print(f"NEW speed for flat segment: {new_speed:.4f} km/h", flush=True)
print(f"DIFF: {abs(old_speed - new_speed):.4f} km/h", flush=True)
print("SAME MODEL?" , old_speed == new_speed, flush=True)
