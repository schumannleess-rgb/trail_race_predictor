"""
Prove: LightGBM non-determinism without seed causes the 10-min gap.

Strategy: Patch LightGBM params before training to test both scenarios.
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

records_dir = Path('temp/records')
gpx_file = Path('temp/routes/2025黄岩九峰大师赛最终版.gpx')
fit_files = sorted(records_dir.glob('*.fit'), key=lambda p: p.stat().st_size, reverse=True)
train_files = [str(f) for f in fit_files]

print(f"Files: {len(train_files)}", flush=True)

# --- Test 1: WITHOUT seed (simulates committed core, no seed) ---
# We patch the model module before training

print("\n=== WITHOUT seed (committed core behavior) ===", flush=True)

# Save and patch
import core_rebuild.predictor.model as model_module
original_params = model_module.LightGBMPredictor.train

def train_no_seed(self, segments):
    import lightgbm as lgb
    import numpy as np

    if len(segments) < 10:
        return self._train_fallback(segments)

    X, y = self._build_matrices(segments)
    self.max_training_distance = max(s.accumulated_distance_km for s in segments)
    self.max_training_ascent = max(s.accumulated_ascent_m for s in segments)

    self.feature_names = [
        'grade_pct', 'rolling_grade_500m',
        'accumulated_distance_km', 'accumulated_ascent_m',
        'absolute_altitude_m', 'elevation_density',
    ]

    # NO SEED - this is what committed core had (non-deterministic)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data': 1,
        # NOTE: NO seed!
    }

    train_data = lgb.Dataset(X, label=y)
    self.model = lgb.train(params, train_data, num_boost_round=100)

    preds = self.model.predict(X)
    mae = np.mean(np.abs(preds - y))
    rmse = np.sqrt(np.mean((preds - y) ** 2))

    flat_speeds = [y[i] for i in range(len(y)) if -5 <= X[i][0] <= 5]
    src = flat_speeds if flat_speeds else list(y)
    self.p50_speed = float(np.percentile(src, 50))
    self.p90_speed = float(np.percentile(src, 90))

    self.feature_importance = dict(zip(self.feature_names,
                                        [float(v) for v in self.model.feature_importance()]))

    print(f"  Model (no seed): MAE={mae:.2f} km/h, P50={self.p50_speed:.2f}, P90={self.p90_speed:.2f}", flush=True)
    self.is_trained = True
    return True

model_module.LightGBMPredictor.train = train_no_seed

from core_rebuild.predictor import MLRacePredictor
pred_noseed = MLRacePredictor()
pred_noseed.train_from_files(train_files)
r_noseed = pred_noseed.predict_race(str(gpx_file), effort_factor=1.0)
print(f"  Result (no seed): {r_noseed['predicted_time_hm']} ({r_noseed['predicted_time_min']} min)", flush=True)

# Restore
model_module.LightGBMPredictor.train = original_params

# --- Test 2: WITH seed (core_rebuild behavior) ---
print("\n=== WITH random_state=42 (core_rebuild behavior) ===", flush=True)
pred_withseed = MLRacePredictor()
pred_withseed.train_from_files(train_files)
r_withseed = pred_withseed.predict_race(str(gpx_file), effort_factor=1.0)
print(f"  Result (with seed): {r_withseed['predicted_time_hm']} ({r_withseed['predicted_time_min']} min)", flush=True)

# --- Summary ---
print("\n" + "="*60, flush=True)
print("ROOT CAUSE SUMMARY", flush=True)
print("="*60, flush=True)
print(f"  WITHOUT seed (committed core):  {r_noseed['predicted_time_hm']} ({r_noseed['predicted_time_min']} min)", flush=True)
print(f"  WITH seed (core_rebuild):        {r_withseed['predicted_time_hm']} ({r_withseed['predicted_time_min']} min)", flush=True)
print(f"  Historical core (saved file):   6:44:54 (405 min)", flush=True)
print(f"  Historical core_rebuild (saved): 6:34:32 (395 min)", flush=True)
print("", flush=True)
print("  DIFF (no-seed vs with-seed):", flush=True)
print(f"    time_min: {abs(r_noseed['predicted_time_min'] - r_withseed['predicted_time_min'])} min", flush=True)
