# Trail Race Predictor V5 - Version Snapshot

**Version**: V5.0
**Date**: 2026-03-31
**Status**: Production Ready

## Version Summary

V5 represents a complete rewrite with machine learning, unified filtering, and production-ready validation systems.

## Key Features in V5

### 1. Machine Learning Prediction
- LightGBM regression with 6 engineered features
- MAE: 0.02-0.07 km/h
- Handles non-linear terrain-speed relationships

### 2. Dual Filtering System
- GPX: 20m resampling, 140m window, ±45% clip (DEM noise)
- FIT: No resampling, 7-10s window, ±50% clip (sensor noise)
- Epsilon check prevents grade explosion

### 3. VAM Validation
- Caps vertical climbing speed at 1000 m/h
- Prevents unrealistic predictions on steep terrain
- Fixed critical bug (missing factor of 10)

### 4. Extrapolation Penalty
- Distance penalty: 30% per 10% beyond training range
- Ascent penalty: 20% per 10% beyond training range

## Performance Metrics

### Training Data
- High speed: 6 files, 47 segments
- Low speed: 7 files, 167 segments

### Prediction Results (Huangyan Nine Peaks 29km)
- High effort: 5:34:26 (11.4 min/km)
- Low effort: 6:32:31 (13.4 min/km)
- Steep segments (>30%): 2.47-3.15 km/h (validated)

## Critical Bugs Fixed

1. **Grade Unit Error** - Fixed meter/kilometer confusion
2. **Distance Double-Counting** - Fixed cumulative distance bug
3. **VAM Calculation** - Added missing factor of 10
4. **GPS Drift** - Added epsilon check for small distances
5. **Numpy Serialization** - Fixed JSON export error
6. **Array Shape Mismatch** - Fixed GPX filter dimension bug

## Files in V5

### Core Modules
- `predictor.py` - ML predictor with LightGBM
- `utils.py` - Unified filtering utilities
- `gpx_filter.py` - GPX race route processing
- `README.md` - Complete documentation

### Data Files
- `records/` - Training FIT files (高速/中速/低速)
- `maps/` - Race route GPX files
- `prediction_result_v3_ml.json` - Latest prediction output

### Documentation
- `FIT_FILTERING_INTEGRATION.md` - Filter integration details
- `VERSION_V5.md` - This file

## Feature Importance (V5)

### High Speed Model
1. grade_pct: 38.1%
2. accumulated_distance_km: 30.3%
3. accumulated_ascent_m: 13.3%
4. elevation_density: 8.5%
5. rolling_grade_500m: 7.4%
6. absolute_altitude_m: 2.3%

### Low Speed Model
1. grade_pct: 34.4%
2. accumulated_distance_km: 32.2%
3. accumulated_ascent_m: 19.0%
4. rolling_grade_500m: 6.3%
5. elevation_density: 5.0%
6. absolute_altitude_m: 3.1%

## Configuration

### GPX Filter
```python
{
    'resample_spacing_m': 20,
    'window_size': 7,
    'poly_order': 2,
    'max_grade_pct': 45.0,
    'resample_required': True,
    'min_distance_m': 0.5
}
```

### FIT Filter
```python
{
    'resample_spacing_m': None,
    'window_size': 7,
    'poly_order': 2,
    'max_grade_pct': 50.0,
    'resample_required': False,
    'min_distance_m': 0.5
}
```

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
lightgbm>=3.3.0
```

## Known Limitations

1. **Rolling Grade Importance**: Only 6-7% importance (5th place)
   - Cause: Training data lacks sustained climb samples
   - Impact: Model may underpredict fatigue on continuous climbs

2. **Altitude Effect**: Only 2-3% importance
   - Cause: Training data at similar elevations
   - Impact: May not capture high-altitude performance decline

3. **Small Training Set**: 47-167 segments
   - Mitigation: LightGBM optimized for small data
   - Risk: Overfitting on terrain types not in training

## Future Roadmap

### V5.1 (Short Term)
- [ ] Increase rolling window to 1000m
- [ ] Add terrain classification (technical/runnable)
- [ ] Include more sustained climb training data

### V6.0 (Medium Term)
- [ ] Multi-model ensemble (separate models by terrain type)
- [ ] Weather factor integration (temperature, humidity)
- [ ] Recovery features (recent effort intensity)

### V7.0 (Long Term)
- [ ] Deep learning alternative (LSTM for sequence prediction)
- [ ] Power meter integration (if available)
- [ ] Race-specific calibration factors

## Version History

### V5.0 (2026-03-31)
- Initial ML version with LightGBM
- Unified GPX/FIT filtering
- VAM validation
- Extrapolation penalty
- Complete documentation

### Previous Versions
- V1: Formula-based prediction
- V2: Basic feature extraction
- V3-V4: Development iterations (not released)

## Backup & Restoration

To restore V5 from backup:
1. Copy all `.py` files from backup
2. Restore `records/` and `maps/` directories
3. Verify dependencies: `pip install -r requirements.txt`
4. Test: `python predictor.py`

## Contact & Support

For issues or questions:
- Check README.md for usage
- Review documentation files for technical details
- Verify training data quality and quantity
