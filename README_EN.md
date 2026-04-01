# Trail Master - Trail Race Prediction System

**Version**: V1.2

---

### Introduction

Trail Master is a machine learning-based trail race performance prediction tool. By analyzing your historical training data, it predicts your finish time and split paces on target race courses.

The system uses the LightGBM gradient boosting framework, combined with exercise physiology principles, to provide scientific and reliable performance predictions for trail runners.

### Key Features

- **Native FIT Support** - No conversion needed, upload raw FIT files from Garmin/Coros directly
- **Unified Modeling** - Single model trained on all data, simplified workflow
- **Performance Range Prediction** - Quantify results across different competitive states based on P50/P90 capability bounds
- **Physical Constraints** - VAM (Vertical Ascent Rate) limits prevent unrealistic predictions
- **Web Interface** - User-friendly Streamlit interface, drag-and-drop to use

### Quick Start

#### 1. Navigate to Project Directory

Open a terminal (Windows: press `Win+R`, type `cmd`), and run:

```bash
cd d:\path\to\trail_race_predictor_v1.2
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Launch Application

```bash
streamlit run app.py
```

After launching, your browser will automatically open **http://localhost:8501** showing the web interface.

#### Usage

1. **Upload Route** - Upload the race GPX file (must include elevation data)
2. **Upload Training Records** - Upload 15-20 quality FIT files (minimum 5)
3. **Adjust Performance Factor** - Slider from 0.8-1.2
   - `1.0` = Average training level (P50)
   - `1.1-1.2` = Race mode (approaching P90)
   - `0.8-0.9` = Conservative strategy
4. **Start Analysis** - System automatically trains model and generates predictions

### Data Requirements

| File Type | Format | Requirements |
|-----------|--------|--------------|
| Route | GPX | Must include elevation data |
| Training Records | FIT | 15-20 quality files recommended, 5 minimum |

**Note**: Please exclude the following types of records:
- Casual walks, commuting
- Data with severe GPS drift
- Pure flat road running (no elevation gain)

---

## Technical Details

### 1. Machine Learning Model

#### Algorithm: LightGBM

The system uses Microsoft's LightGBM (Light Gradient Boosting Machine) framework, offering these advantages over traditional linear regression:

| Feature | Linear Regression | LightGBM |
|---------|------------------|----------|
| Non-linear relationships | ❌ Cannot capture | ✅ Auto-learned |
| Feature interactions | ❌ Manual design | ✅ Auto-discovered |
| Overfitting risk | Low | Medium (controlled by regularization) |
| Small data performance | Average | Excellent |
| Interpretability | High | Medium (feature importance) |

#### Model Configuration

```python
params = {
    'objective': 'regression',    # Regression task
    'metric': 'mae',              # Mean Absolute Error
    'num_leaves': 31,             # Number of leaves
    'learning_rate': 0.05,        # Learning rate
    'feature_fraction': 0.9,      # Feature sampling ratio
    'bagging_fraction': 0.8,      # Data sampling ratio
    'bagging_freq': 5,            # Sampling frequency
    'min_data': 1,                # Minimum data per leaf
}
num_boost_round = 100             # Iterations
```

#### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| MAE | 0.02-0.07 km/h | Mean Absolute Error |
| RMSE | 0.03-0.10 km/h | Root Mean Square Error |
| R² | 0.85-0.92 | Coefficient of Determination |

### 2. Feature Engineering

The model uses 6 carefully designed features to predict speed:

| Feature | Description | Importance | Scientific Basis |
|---------|-------------|------------|------------------|
| `grade_pct` | Current grade (%) | 35-40% | Grade is the primary factor affecting trail running speed |
| `accumulated_distance_km` | Cumulative distance | 30-35% | Reflects fatigue accumulation effect |
| `accumulated_ascent_m` | Cumulative ascent | 15-20% | Impact of climbing on subsequent performance |
| `elevation_density` | Elevation density (m/km) | 5-10% | Course difficulty indicator |
| `rolling_grade_500m` | Average grade over past 500m | 5-8% | Terrain continuity affects pacing strategy |
| `absolute_altitude_m` | Absolute altitude | 2-5% | Altitude effect (limited data) |

#### Feature Importance Analysis

Based on SHAP (SHapley Additive exPlanations) values:

```
grade_pct              ████████████████████████████████████  38%
accumulated_distance   ██████████████████████████████        32%
accumulated_ascent     ███████████████                       17%
elevation_density      █████                                 8%
rolling_grade_500m     ████                                  6%
absolute_altitude_m    ██                                    3%
```

### 3. Data Preprocessing

#### Elevation Data Filtering

Raw GPS/elevation data contains noise. The system uses Savitzky-Golay filter for smoothing:

**GPX File Processing**:
- Resampling interval: 20m
- Filter window: 7 points (~140m)
- Polynomial order: 2
- Grade truncation: ±45%

**FIT File Processing**:
- No resampling (preserves original sampling rate)
- Filter window: 7-10 seconds
- Polynomial order: 2
- Grade truncation: ±50%

#### Filtering Effect Comparison

| Metric | Raw Data | Filtered |
|--------|----------|----------|
| Max grade | >100% (noise) | <50% (reasonable) |
| Grade std dev | 15-25% | 8-12% |
| Elevation noise | ±5-10m | ±1-2m |

### 4. Capability Bounds

#### P50/P90 Methodology

The system innovatively introduces capability bound quantification based on statistical distribution of your historical training data:

- **P50 (Median Speed)**: Represents your daily training level, ~50% of training reaches this speed
- **P90 (90th Percentile Speed)**: Represents your peak capability, only 10% of training reaches this speed

#### Effort Factor

```
effort_factor = user setting (0.8 - 1.2)

predicted_speed = model_base_prediction × effort_factor
```

| Factor | Meaning | Applicable Scenario |
|--------|---------|---------------------|
| 0.8-0.9 | Conservative strategy | Long distance, recovery period, first race |
| 1.0 | Average level | Regular training state |
| 1.1-1.2 | Race mode | Target event, peak period |

#### P90/P50 Ratio Interpretation

| Ratio | Interpretation |
|-------|----------------|
| 1.10-1.15 | Stable training intensity, limited race improvement potential |
| 1.15-1.25 | Normal range, 15-25% improvement possible in races |
| >1.25 | Large training intensity variance, possible "sandbagging" in training |

### 5. Physical Constraints

#### VAM (Vertical Ascent Rate) Limit

VAM is an international standard metric for climbing ability, measured in m/h (meters per hour).

```
VAM = horizontal_speed (km/h) × 1000 × grade (%) / 100
    = horizontal_speed × 10 × grade
```

**System Limit**: VAM ≤ 1000 m/h

| VAM Level | Value | Corresponding Level |
|-----------|-------|---------------------|
| Amateur | 400-600 | Regular trail runners |
| Advanced | 600-800 | Trained runners |
| Elite | 800-1000 | Competitive level |
| Professional | 1000-1200 | Professional athletes |
| World-class | >1200 | Top athletes |

#### Extrapolation Penalty

When predicted distance/ascent exceeds training data range:

```python
if cumulative_distance > max_training_distance:
    excess_ratio = cumulative_distance / max_training_distance
    penalty = 1 + (excess_ratio - 1) × 0.3  # 3% slowdown per 10% excess
    predicted_speed /= penalty

if cumulative_ascent > max_training_ascent:
    excess_ratio = cumulative_ascent / max_training_ascent
    penalty = 1 + (excess_ratio - 1) × 0.2  # 2% slowdown per 10% excess
    predicted_speed /= penalty
```

### 6. Data Validation

#### Input Validation

| Check | Description | Handling |
|-------|-------------|----------|
| File format | GPX/FIT format validation | Reject invalid files |
| GPS coordinates | Coordinate range check | Reject treadmill data |
| Elevation data | Elevation field existence | Warning and degraded operation |
| Data volume | Minimum record points | Warning or rejection |

#### Duplicate Detection

Automatically detect duplicate uploads based on FIT file's `time_created` field:

```python
activity_key = file_id.time_created  # Unique identifier
if activity_key in seen_activities:
    Mark as duplicate and skip
```

---

## Directory Structure

```
trail_race_predictor_v1.2/
├── app.py                 # Streamlit main application
├── core/
│   ├── predictor.py       # ML predictor (LightGBM)
│   ├── types.py           # Type definitions
│   └── utils.py           # Filtering utilities
├── data/
│   ├── file_handler.py    # File handling
│   └── data_validator.py  # Data validation
├── maps/                  # Route GPX files
├── records/               # Training FIT files
└── temp/                  # Temporary files
```

---

## Dependencies

```
streamlit>=1.28.0
numpy>=1.20.0
scipy>=1.7.0
lightgbm>=3.3.0
fitparse>=1.2.0
pandas>=2.0.0
plotly>=5.18.0
gpxpy>=1.6.0
```

## License

MIT License

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V1.2 | 2026-04-01 | Enhanced technical documentation, improved credibility |
| V1.1 | 2026-03-31 | Unified modeling, FIT support, effort quantification |
| V1.0 | 2026-03-30 | Initial release with LightGBM prediction |
