# Bug Analysis Report
## 越野赛预测器 V1.2 代码审查报告

**日期**: 2026-04-02
**审查文件**: `core/predictor.py`, `core/utils.py`, `core/gpx_filter.py`

---

## 摘要

| 类别 | 数量 |
|------|------|
| 已修复 | 2 |
| 仍存在 | 5 |
| 致命 | 1 |
| 显著 | 3 |
| 轻微 | 1 |

---

## Bug 详细状态

### ✅ Bug 1 (致命) - 下坡 grade 永远 ≥ 0
**状态**: **已修复**

**原问题**: `_create_segments()` 用 `seg_gain`（只累加上升量）计算 `avg_grade`

**修复位置**: `core/predictor.py:861`
```python
# 修复后代码
avg_grade = ((elevations[-1] - elevations[0]) / seg_distance) * 100 if seg_distance > 0 else 0
```
现在使用净海拔差（可正可负），下坡坡度可以正确计算为负值。

---

### ✅ Bug 2 (致命) - analyze_and_train() 方法不存在
**状态**: **已修复**

**原问题**: `main()` 调用不存在的方法

**修复位置**: `core/predictor.py:1076`
```python
# 修复后代码
if not predictor.train_from_files(training_files):
    print("训练失败!")
    return
```
直接调用 `train_from_files()` 而不是不存在的 `analyze_and_train()`。

---

### ⚠️ Bug 3 (显著) - 速度用瞬时值而非段平均速度
**状态**: **仍存在**

**位置**: `core/predictor.py:482-483`

**问题代码**:
```python
time_diff = timestamps[i] - timestamps[i-1]
speed = (seg_dist / 1000) / (time_diff / 3600) if time_diff > 0 else 5
```

**问题分析**:
- 只使用最后一步的时间差计算速度
- 该段跨越多个 GPS 点，应该用累计时间
- 导致速度不准确，特别是当该段内速度变化较大时

**修复方案**:
```python
# 需要添加
current_seg_time = 0  # 在循环外初始化

# 循环内
current_seg_time += time_diff

# 分段时
speed = (current_seg_distance / 1000) / (current_seg_time / 3600) if current_seg_time > 0 else 5
current_seg_time = 0  # 重置
```

---

### ⚠️ Bug 4 (显著) - avg_grade = 最后一个点的瞬时坡度
**状态**: **仍存在**

**位置**: `core/predictor.py:468`

**问题代码**:
```python
avg_grade = seg_grade  # 只取最后一个 GPS 点的坡度
```

**问题分析**:
- 只用段内最后一个点的瞬时坡度
- 未考虑段内所有点的平均坡度
- 当段内坡度变化大时，代表性差

**修复方案**:
```python
# 记录段开始索引
seg_start_idx = i  # 在分段开始时记录

# 分段时
avg_grade = np.mean(grades[seg_start_idx : i + 1])
```

---

### ⚠️ Bug 5 (显著) - GPX 段的 rolling_grade_500m 直接复制 avg_grade
**状态**: **仍存在**

**位置**: `core/predictor.py:869`

**问题代码**:
```python
rolling_grade_500m=avg_grade,  # 完全没有滑动窗口
```

**问题分析**:
- GPX 的 `_create_segments()` 直接复制 avg_grade
- FIT 数据有正确的滚动窗口计算（第 470-480 行）
- 导致 LightGBM 的两个坡度特征对 GPX 段完全相同
- 浪费了一个关键特征

**影响**:
- 模型训练时，GPX 和 FIT 数据的特征不一致
- GPX 数据的 rolling_grade_500m 特征无区分度
- 降低模型预测准确性

**修复方案**:
```python
# 参考 FIT 数据的实现（第 470-480 行）
# 需要在 GPX 处理中也实现基于距离的滚动窗口
```

---

### ⚠️ Bug 6 (轻微) - 滤波后 GPX 保存时丢弃重采样点
**状态**: **仍存在**

**位置**: `core/gpx_filter.py:331`

**问题代码**:
```python
if i < len(raw_points):
    # 找到对应的原始点
    raw_idx = np.argmin(np.abs(raw_distances - dist))
    point = raw_points[raw_idx]
```

**问题分析**:
- 20 米重采样后，点数通常多于原始点
- `if i < len(raw_points)` 导致循环提前退出
- 超出原始点数的重采样数据被丢弃

**修复方案**:
```python
# 移除条件，使用插值获取经纬度
raw_idx = np.argmin(np.abs(raw_distances - dist))
point = raw_points[raw_idx]
```

---

### ⚠️ Bug 7 (轻微) - FIT 坡度计算未过滤 GPS 漂移
**状态**: **仍存在**

**位置**: `core/predictor.py:434`

**问题代码**:
```python
if dist_m > 0:  # 应该是 > 0.5
    grade = (ele_m / dist_m) * 100
```

**问题分析**:
- 使用 `> 0` 而非 `> 0.5`
- GPS 静止漂移产生 0.1m 虚假位移时，会产生极端坡度
- 与 `ElevationFilter.calculate_grade()` 中的 `min_distance_m` 设计不一致

**修复方案**:
```python
min_distance = FilterConfig.FIT.get('min_distance_m', 0.5)
if dist_m > min_distance:
    grade = (ele_m / dist_m) * 100
else:
    grade = 0
```

---

## 优先级建议

| 优先级 | Bug | 理由 |
|--------|-----|------|
| P0 | Bug 5 | 导致模型特征失效，严重影响预测准确性 |
| P1 | Bug 3 | 速度特征不准，直接影响模型训练 |
| P1 | Bug 4 | 坡度特征代表性差 |
| P2 | Bug 7 | 可能产生异常坡度值 |
| P2 | Bug 6 | 仅影响滤波 GPX 导出 |

---

## 测试建议

建议为修复的 Bug 添加单元测试：

1. **Bug 3 测试**: 创建一段有明显速度变化的数据，验证段平均速度
2. **Bug 4 测试**: 创建坡度变化段，验证 avg_grade 是否为均值
3. **Bug 5 测试**: 验证 GPX 和 FIT 的 rolling_grade_500m 计算一致性
4. **Bug 7 测试**: 模拟 GPS 漂移（小距离位移），验证坡度不超过 max_grade_pct

---

## 代码质量观察

**正面**:
- Bug 1 和 Bug 2 已被修复
- 代码结构清晰，易于理解
- 使用了 dataclass (SegmentFeatures) 提高可读性

**需要改进**:
- 缺少单元测试覆盖
- GPX 和 FIT 的特征提取逻辑不一致
- 需要添加输入验证（如 min_distance 检查）
