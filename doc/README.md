# Trail Race Predictor V5 - Machine Learning Edition

A high-precision越野赛 (trail race) performance prediction system using LightGBM machine learning with unified GPX/FIT data filtering.

## Overview

This predictor uses your historical training records (FIT files) to predict race performance on any trail course (GPX file). It features:

- **Machine Learning**: LightGBM regression with 6 engineered features
- **Dual Filtering System**: Separate signal processing for GPX (race routes) and FIT (training records)
- **VAM Validation**: Vertical Ascension Meter checking prevents unrealistic climbing predictions
- **Extrapolation Penalty**: Handles predictions beyond training data range

## Performance

### Model Accuracy

| Effort Level | MAE | RMSE | Training Segments | Files |
|--------------|-----|------|-------------------|-------|
| 高速 (High) | 0.02 km/h | 0.03 km/h | 47 | 6 |
| 低速 (Low) | 0.07 km/h | 0.08 km/h | 167 | 7 |

### Huangyan Nine Peaks 29km Prediction

| Effort Level | Predicted Time | Pace | Speed |
|--------------|----------------|------|-------|
| 高速 | 5:34:26 | 11.4 min/km | 5.26 km/h |
| 低速 | 6:32:31 | 13.4 min/km | 4.48 km/h |

**Steep Grade Validation (>30%):**
- Segments capped at 2.47-3.15 km/h (realistic hiking speed)
- VAM limited to 1000 m/h (elite athlete level)

## Technologies & Algorithms

### 1. Machine Learning

**Algorithm**: LightGBM (Gradient Boosting Machine)
- Optimized for small datasets (47-167 samples)
- Handles non-linear relationships between terrain and speed
- Feature importance analysis for model interpretability

**Features (6 total):**
```
1. grade_pct           - Instantaneous slope (%)
2. rolling_grade_500m  - Average slope over past 500m
3. accumulated_distance_km - Total distance covered
4. accumulated_ascent_m    - Total elevation gain
5. absolute_altitude_m     - Current altitude
6. elevation_density       - Climbing per distance (m/km)
```

### 2. Signal Processing

**Savitzky-Golay Filter**:
- Preserves elevation curvature while removing noise
- Different parameters for GPX vs FIT data

**GPX Configuration (Race Routes):**
```python
{
    'resample_spacing_m': 20,      # 20m uniform spacing
    'window_size': 7,               # Covers 140m terrain
    'poly_order': 2,                # Preserves curvature
    'max_grade_pct': 45.0,          # Strict clip for DEM noise
    'min_distance_m': 0.5           # Epsilon check
}
```

**FIT Configuration (Training Records):**
```python
{
    'resample_spacing_m': None,     # No resampling (1s intervals)
    'window_size': 7,               # Covers 7-10s motion
    'poly_order': 2,
    'max_grade_pct': 50.0,          # Lenient clip for real efforts
    'min_distance_m': 0.5           # Epsilon check
}
```

### 3. Validation Systems

**VAM (Vertical Ascension Meter) Validation:**
```python
vam = predicted_speed * 10 * grade_pct  # m/hour
if vam > 1000:  # Elite level threshold
    penalty = vam / 1000
    predicted_speed /= penalty
```

**Extrapolation Penalty:**
```python
if distance > max_training_distance:
    penalty = 1 + (excess_ratio - 1) * 0.3
    predicted_speed /= penalty
```

## Installation

```bash
pip install numpy scipy lightgbm
```

## Usage

### Basic Prediction

```python
from predictor import MLRacePredictor

# Initialize with training records directory
predictor = MLRacePredictor('records/')

# Train models from your FIT files
predictor.analyze_and_train()

# Predict race time for a GPX course
result = predictor.predict_race('maps/route.gpx', effort_level='中速')

print(f"Predicted time: {result['predicted_time_hm']}")
print(f"Predicted pace: {result['predicted_pace_min_km']} min/km")
```

### Command Line

```bash
python predictor.py
```

Output:
```
======================================================================
越野赛成绩预测器 V3 - LightGBM 机器学习版
======================================================================

[Step 1/3] 训练 ML 模型...
  Model trained: MAE=0.02 km/h, RMSE=0.03 km/h

[Step 2/3] 训练统计:
  高速: 6 文件, 47 分段, 平均速度=5.94 km/h
  低速: 7 文件, 167 分段, 平均速度=4.90 km/h

[Step 3/3] 预测比赛成绩:

【高速】
  预测时间: 5:34:26 (334分钟)
  预测配速: 11.4 min/km
  平均速度: 5.26 km/h

【低速】
  预测时间: 6:32:31 (393分钟)
  预测配速: 13.4 min/km
  平均速度: 4.48 km/h
```

## File Structure

```
trail_race_predictor_v5/
├── predictor.py           # Main ML predictor
├── utils.py               # Unified filtering utilities
├── gpx_filter.py          # GPX race route filtering
├── records/               # Training FIT files
│   ├── 高速/              # High effort training
│   ├── 中速/              # Medium effort training
│   └── 低速/              # Low effort training
├── maps/                  # Race route GPX files
│   └── 2025黄岩九峰大师赛最终版.gpx
├── prediction_result_v3_ml.json  # Prediction output
└── README.md              # This file
```

## Development Journey

### Problems Encountered & Solutions

#### Problem 1: Grade Calculation Unit Error
**Issue**: Grades showing 3482% instead of 3.48%
**Root Cause**: Mixed units (meters vs kilometers)
```python
# WRONG
avg_grade = (elevation_gain / distance * 100)  # gain(m), dist(km)

# FIXED
avg_grade = (elevation_gain / distance / 10)  # Correct unit conversion
```

#### Problem 2: Distance Double-Counting
**Issue**: Predicted 375 hours instead of 5 hours
**Root Cause**: Using cumulative distance for each segment
```python
# WRONG
for seg in segments:
    segment_time = seg.accumulated_distance_km / speed  # Double counting!

# FIXED
prev_cumulative = 0
for seg in segments:
    segment_distance = seg.accumulated_distance_km - prev_cumulative
    segment_time = segment_distance / speed
    prev_cumulative = seg.accumulated_distance_km
```

#### Problem 3: GPX DEM Noise
**Issue**: Raw GPX data had cliffs and bridges causing false spikes
**Solution**: 3-step filtering process
1. Resample to uniform 20m spacing
2. Savitzky-Golay smoothing (window=7, poly=2)
3. Grade clipping at ±45%

**Result**: 2426m → 1921m climbing (21% noise removed)

#### Problem 4: FIT Sensor Noise
**Issue**: Barometer drift and arm swing causing jitter
**Solution**: Apply SG filter with FIT-specific parameters
- No resampling (preserve 1s intervals)
- Window=7 (7-10 seconds of motion)
- Grade clipping at ±50%

**Result**: Noise std reduced to 0.48m

#### Problem 5: Unrealistic Steep Climb Predictions
**Issue**: 5.54 km/h predicted on 40.5% grade (VAM = 2246 m/h!)
**Root Cause**: VAM calculation missing factor of 10
```python
# WRONG
vam = predicted_speed * grade_pct  # km/h * % = wrong unit!

# FIXED
vam = predicted_speed * 10 * grade_pct  # km/h * 10 * % = m/h
```

**Result**: Steep segments now capped at 2.47-3.15 km/h (realistic hiking speed)

#### Problem 6: Grade Explosion from GPS Drift
**Issue**: Tiny distances (<0.5m) causing infinite grades
**Solution**: Epsilon check in grade calculation
```python
min_distance = config.get('min_distance_m', 0.5)
if dist_m > min_distance:
    grade = (ele_m / dist_m) * 100
else:
    grade = 0  # Prevent explosion
```

#### Problem 7: Low Feature Importance for Rolling Grade
**Issue**: `rolling_grade_500m` only 6-7% importance (5th place)
**Analysis**: Instantaneous grade dominates because:
- Training data lacks sustained climb samples
- Rolling window may be too small (500m)
- Consider increasing to 1000m or adding more training data

### Key Insights

1. **Separate Filtering is Essential**: GPX and FIT require different parameters due to different noise sources
2. **Unit Consistency Matters**: Always verify units when combining distance/elevation/speed
3. **VAM is Critical**: Vertical speed limits prevent unrealistic predictions on extreme terrain
4. **Epsilon Checks**: Protect against edge cases like GPS drift
5. **Small Data ML**: LightGBM works well with 47-167 samples when features are well-engineered

## Feature Importance Analysis

### Current Ranking (High Speed)
1. **grade_pct (38%)** - Instantaneous slope dominates
2. **accumulated_distance_km (30%)** - Fatigue from distance
3. **accumulated_ascent_m (13%)** - Fatigue from climbing
4. **elevation_density (9%)** - Course difficulty
5. **rolling_grade_500m (7%)** - Slope persistence
6. **absolute_altitude_m (2%)** - Altitude effect minimal

### Observations
- Instantaneous grade is 5x more important than rolling grade
- Distance fatigue is more significant than climbing fatigue
- Altitude has minimal impact (training likely at similar elevations)

## Future Improvements

1. **Increase Rolling Window**: Try 1000m instead of 500m for better slope persistence
2. **Add More Training Data**: Include more sustained climb sessions
3. **Terrain Classification**: Separate models for technical vs runnable terrain
4. **Weather Factors**: Add temperature/humidity if available
5. **Recovery Features**: Include recent effort intensity (e.g., speed in last 1km)

## References

- `ml反馈1.txt` - ML feedback and FIT filtering requirements
- `滤波器区别fitgpx.txt` - Parameter differentiation guidance
- `综合反馈1.txt` - Epsilon check, VAM validation, window size recommendations
- `滤波反馈1.txt` - Resampling spacing optimization

## License

MIT License - Feel free to use for personal trail running predictions

## Author

Built with Claude Code for越野赛 performance prediction
