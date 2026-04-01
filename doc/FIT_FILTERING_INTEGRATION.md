# FIT Filtering Integration

## Overview

Integrated unified filtering utilities (`utils.py`) into the predictor pipeline to apply Savitzky-Golay smoothing to FIT training records before feature extraction.

## What Changed

### 1. New Unified Filter: `utils.py`

Created a centralized filtering module with separate configurations for GPX and FIT data:

```python
class FilterConfig:
    # GPX configuration (for race routes)
    GPX = {
        'resample_spacing_m': 20,      # 20m resampling
        'window_size': 7,               # Covers 140m terrain
        'poly_order': 2,                # Preserves curvature
        'max_grade_pct': 45.0,          # Strict clip for DEM noise
        'resample_required': True,
        'min_distance_m': 0.5           # Epsilon check threshold
    }

    # FIT configuration (for training records)
    FIT = {
        'resample_spacing_m': None,     # No resampling (keep 1s/point)
        'window_size': 7,               # Covers 7-10s motion (better slope persistence)
        'poly_order': 2,
        'max_grade_pct': 50.0,          # Lenient clip for real performance
        'resample_required': False,
        'min_distance_m': 0.5           # Epsilon check threshold
    }
```

### 2. Updated: `predictor.py`

Modified `FeatureExtractor._extract_from_metrics()` to:

1. **Import filtering utilities**:
   ```python
   from utils import apply_fit_filter, FilterConfig
   ```

2. **Apply FIT filtering** before feature extraction:
   ```python
   # Apply Savitzky-Golay smoothing with FIT configuration
   smoothed_elevations, filter_info = apply_fit_filter(elevations, timestamps)
   ```

3. **Calculate grades from smoothed data**:
   ```python
   grades = []
   for i in range(len(smoothed_ele_arr) - 1):
       dist_m = distances_arr[i + 1] - distances_arr[i]
       ele_m = smoothed_ele_arr[i + 1] - smoothed_ele_arr[i]
       grade = (ele_m / dist_m) * 100
       grade = np.clip(grade, -FilterConfig.FIT['max_grade_pct'], FilterConfig.FIT['max_grade_pct'])
       grades.append(grade)
   ```

### 3. Enhanced Predictions with VAM Validation

Added **VAM (Vertical Ascension Meter)** validation to detect unrealistic climbing speeds:

```python
# VAM validation for steep terrain
if segment.grade_pct > 15:  # Steep climbs
    vam = predicted_speed * segment.grade_pct
    if vam > 1000:  # Beyond elite level
        vam_penalty = vam / 1000
        predicted_speed /= vam_penalty
```

## Improvements Based on Feedback

### 1. Epsilon Check for Grade Calculation

Added minimum distance threshold to prevent grade explosion from GPS drift:

```python
# In ElevationFilter.calculate_grade()
min_distance = config.get('min_distance_m', 0.5)
if dist_m > min_distance:
    grade = (ele_m / dist_m) * 100
else:
    grade = 0  # Prevent division by tiny values
```

**Test Results:**
- Distance = 0.1m, Elevation = 1m → Grade = 0% (prevented)
- Distance = 10m, Elevation = 1m → Grade = 10% (correct)

### 2. FIT Window Size Increased

Changed from `window_size=5` to `window_size=7` to better model "slope persistence" in trail running.

**Rationale:** A 7-second window better captures how runners perceive sustained climbs versus momentary elevation spikes.

### 3. VAM Validation

Added vertical speed checking for steep climbs (>15% grade):
- Normal VAM range: 500-800 m/h for amateurs
- Elite VAM range: 800-1200 m/h
- Beyond 1000 m/h: Apply penalty to prevent unrealistic predictions

## Why Different Parameters?

| Aspect | GPX (Race Routes) | FIT (Training Records) |
|--------|-------------------|------------------------|
| **Data Source** | DEM (Digital Elevation Model) | Barometric sensor |
| **Noise Type** | Cliff/bridge artifacts | Pressure drift, arm swing |
| **Sampling** | Highly uneven | Uniform 1s intervals |
| **Resampling** | Required (20m spacing) | Not needed |
| **Window Size** | 7 points (140m terrain) | 7 points (7-10s motion) |
| **Grade Clip** | Strict ±45% | Lenient ±50% |
| **Min Distance** | 0.5m epsilon check | 0.5m epsilon check |

## Benefits

1. **Cleaner Training Data**: Removes sensor noise from FIT files, preventing ML model from learning incorrect grade-speed relationships

2. **Consistent Pipeline**: Same Savitzky-Golay algorithm and epsilon check for both GPX and FIT

3. **Better Generalization**: Model learns from realistic terrain features rather than sensor artifacts

4. **Preserves Real Performance**: Lenient clipping (±50%) retains legitimate extreme efforts like stairs or steep climbs

5. **VAM Safety Net**: Prevents unrealistic predictions on extreme terrain (e.g., Huangyan Nine Peaks with many stairs)

## Testing

Verified integration with:
- ✓ Config update: FIT window_size now 7, both configs have min_distance_m
- ✓ Epsilon check: Grades below 0.5m distance set to 0
- ✓ VAM validation: Steep climb predictions checked for realistic vertical speed
- ✓ Import test: All modules load correctly

## References

- `ml反馈1.txt`: Request for FIT filtering in predictor
- `滤波器区别fitgpx.txt`: Parameter differentiation guidance
- `综合反馈1.txt`: Epsilon check, VAM validation, window size recommendations
- `utils.py`: Unified filtering implementation
