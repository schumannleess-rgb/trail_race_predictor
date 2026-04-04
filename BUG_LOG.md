# Bug Log - Trail Race Predictor V1.2.1

## Date: 2026-04-02

---

## Critical Bugs Found

### Bug #1: Non-existent method call in main()
**File:** `core/predictor.py:1028`
**Severity:** CRITICAL
**Status:** FIXED

**Description:**
```python
predictor.analyze_and_train()  # This method doesn't exist!
```

The method should be `train_from_files()`, not `analyze_and_train()`.

**Impact:** Code crashes at runtime

---

### Bug #2: Constructor takes no arguments but receives one
**File:** `core/predictor.py:1025`
**Severity:** CRITICAL
**Status:** FIXED

**Description:**
```python
MLRacePredictor(str(records_dir))  # Constructor takes no arguments!
```

The `MLRacePredictor.__init__()` takes no parameters.

**Impact:** TypeError at runtime

---

### Bug #3: Incorrect grade calculation formula
**File:** `core/predictor.py:823`
**Severity:** HIGH
**Status:** FIXED

**Description:**
```python
avg_grade = (seg_gain / seg_dist_km / 10)  # Wrong formula!
```

**Correct formula:**
```python
avg_grade = (seg_gain / seg_distance_m) * 100
```

**Impact:** All grade calculations are incorrect, affecting predictions

---

### Bug #4: Rolling grade window calculation issue
**File:** `core/predictor.py:448, 605`
**Severity:** MEDIUM
**Status:** FIXED

**Description:**
```python
rolling_window = max(1, int(500 / (seg_dist if seg_dist > 0 else 1)))
```

This uses a single segment's distance instead of proper window sizing based on index positions.

**Impact:** Rolling grade calculations are inaccurate

---

### Bug #5: report_generator.py in wrong location
**File:** `core/report_generator.py`
**Severity:** LOW
**Status:** FIXED

**Description:**
`report_generator.py` is not core algorithm code, should be in `reports/` folder.

**Impact:** Code organization issue

---

## Fixed Issues Summary

| Bug # | Severity | File | Issue | Status |
|-------|----------|------|-------|--------|
| 1 | CRITICAL | predictor.py:1028 | Non-existent method call | ✅ FIXED |
| 2 | CRITICAL | predictor.py:1025 | Constructor argument mismatch | ✅ FIXED |
| 3 | HIGH | predictor.py:823,855 | Wrong grade formula | ✅ FIXED |
| 4 | MEDIUM | predictor.py:448,605 | Incorrect rolling grade window | ✅ FIXED |
| 5 | LOW | - | File organization | ✅ FIXED |
| 6 | **CRITICAL** | predictor.py:265+ | **Not using Garmin pre-calculated values** | ✅ FIXED |

---

## Bug #6: Not Using Garmin Pre-Calculated Values
**Date:** 2026-04-02
**Severity:** CRITICAL
**Status:** FIXED

### Problem
The code was calculating total_ascent and total_descent from raw elevation data, which gives **wrong results**:

| Method | Total Climb | Error |
|--------|-------------|-------|
| Garmin Device (CORRECT) | **2,206 m** | - |
| Our calculation (before) | 3,629 m | +64.5% ❌ |
| Our calculation (after) | 4,253 m | +92.8% ❌ |

### Root Cause
FIT files contain pre-calculated values in the **SESSION message**:
- `total_ascent`: Garmin device calculated total climb (barometric altitude)
- `total_descent`: Garmin device calculated total descent
- `total_distance`: Accurate distance from GPS/accelerometer

The code was ignoring these and recalculating from raw elevation data, which includes GPS noise.

### Fix
Extract and use Garmin pre-calculated values from SESSION message:

```python
# Get pre-calculated values from SESSION message
session_data = {}
for session in fitfile.get_messages('session'):
    for field in session:
        if field.name in ['total_ascent', 'total_descent', 'total_distance']:
            session_data[field.name] = field.value
    break

garmin_ascent = session_data.get('total_ascent', 0)  # Use Garmin's value!
```

### Impact
- Training data now uses **correct** elevation values
- ML model trains on accurate terrain data
- Predictions will be more reliable

---

## Fix Details

### Bug #1 & #2 Fix: main() function
**Before:**
```python
predictor = MLRacePredictor(str(records_dir))
if not predictor.analyze_and_train():
```

**After:**
```python
predictor = MLRacePredictor()
training_files = [str(f) for f in fit_files + json_files]
if not predictor.train_from_files(training_files):
```

### Bug #3 Fix: Grade calculation formula
**Before:**
```python
avg_grade = (seg_gain / seg_dist_km / 10)  # Wrong!
```

**After:**
```python
avg_grade = (seg_gain / seg_distance / 100)  # Correct: grade% = rise/run * 100
```

### Bug #4 Fix: Rolling grade window
**Before:**
```python
rolling_window = max(1, int(500 / (seg_dist if seg_dist > 0 else 1)))
```

**After:**
```python
# Properly look back ~500m in the distance array
target_distance_m = 500
rolling_start_idx = i
distance_accumulated = 0
for j in range(i, max(0, i - 1000), -1):
    if j < len(distances):
        distance_accumulated += (distances[j] - distances[j-1]) if j > 0 else 0
        if distance_accumulated >= target_distance_m:
            rolling_start_idx = j
            break
rolling_grade = np.mean(grades[rolling_start_idx:i+1])
```

### Bug #5 Fix: File organization
`core/report_generator.py` → `reports/report_generator.py`

---
