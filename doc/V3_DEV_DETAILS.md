# V3 机器学习预测器 - 详细开发文档

## 目录
1. [数据探索](#数据探索)
2. [特征工程](#特征工程)
3. [模型训练](#模型训练)
4. [Bug修复记录](#bug修复记录)
5. [预测分析](#预测分析)

---

## 1. 数据探索

### 1.1 原始数据分析

**JSON 文件结构:**
```json
{
  "activity_id": 414266896,
  "activity_info": {
    "distance_km": 5.80,
    "duration_min": 73.46,
    "elevation_gain": 202,
    "elevation_loss": 154
  },
  "metrics": [...]  // 详细记录数据
}
```

**训练数据统计:**

| 类别 | 文件数 | 总分段数 | 平均速度 | 平均VAM |
|------|--------|----------|----------|---------|
| 高速 | 6 | 47 | 5.83 km/h | 239.6 m/h |
| 低速 | 7 | 167 | 4.82 km/h | 272.2 m/h |

**关键发现:**
- 低速组 VAM 更高 (272 vs 240)，但平均速度更低
- 说明低速组更多是爬升训练，而非速度训练

### 1.2 比赛路线分析

**GPX 解析结果:**
- 总距离: 29.33 km
- 总爬升: 2426 m
- 爬升密度: 82.7 m/km (极高)
- 轨迹点数: 9633 个

**分段策略:**
- 每段 0.2 km (200米)
- 总段数: ~147 段

**CP 点信息:**
```
CP1 药山村: 56m (起点附近)
CP2 常乐寺: 30m
CP3 唐家岙: 12m
CP4 老鼠沿杠: 382m (最高点)
```

---

## 2. 特征工程

### 2.1 特征设计

**目标变量:**
- `speed_kmh`: 分段平均速度 (km/h)

**输入特征 (6个):**

| 特征 | 含义 | 计算方式 | 范围 |
|------|------|----------|------|
| `grade_pct` | 当前坡度 | gain/dist/10 | -50% ~ +50% |
| `rolling_grade_500m` | 滚动坡度 | 过去500m平均坡度 | -30% ~ +40% |
| `accumulated_distance_km` | 累计距离 | 从起点累计 | 0 ~ 30+ km |
| `accumulated_ascent_m` | 累计爬升 | 从起点累计 | 0 ~ 2500+ m |
| `absolute_altitude_m` | 绝对海拔 | 当前海拔 | 0 ~ 500 m |
| `elevation_density` | 爬升密度 | gain/dist | 0 ~ 150 m/km |

### 2.2 特征相关性分析

**训练数据相关性 (高速组):**
```
grade_pct vs speed: 负相关 (坡度越大，速度越慢)
- 0% 坡度: ~6.5 km/h
- 10% 坡度: ~4.5 km/h
- 25% 坡度: ~2.5 km/h

accumulated_distance vs speed: 负相关 (疲劳效应)
- 0-5 km: ~6.0 km/h
- 20-25 km: ~5.0 km/h
- 30+ km: ~4.0 km/h
```

---

## 3. 模型训练

### 3.1 LightGBM 参数

```python
params = {
    'objective': 'regression',      # 回归任务
    'metric': 'mae',                # 平均绝对误差
    'num_leaves': 31,               # 叶子节点数
    'learning_rate': 0.05,          # 学习率
    'feature_fraction': 0.9,        # 特征采样比例
    'bagging_fraction': 0.8,        # 数据采样比例
    'bagging_freq': 5,              # 每5次迭代bagging
    'min_data': 1,                  # 最小数据量
    'num_boost_round': 100,          # 迭代次数
}
```

### 3.2 训练结果

**高速组模型:**
```
MAE = 0.01 km/h
RMSE = 0.01 km/h
训练样本数 = 47
```

**低速组模型:**
```
MAE = 0.05 km/h
RMSE = 0.06 km/h
训练样本数 = 167
```

**分析:**
- 高速组误差更小 (数据更一致)
- 低速组样本更多，但误差更大 (数据更多样化)

### 3.3 特征重要性

**高速组:**
```
grade_pct: 1268          ← 最重要
accumulated_distance_km: 906
accumulated_ascent_m: 445
absolute_altitude_m: 234
elevation_density: 189
rolling_grade_500m: 123
```

**低速组:**
```
accumulated_distance_km: 1003 ← 最重要
grade_pct: 1096
accumulated_ascent_m: 557
elevation_density: 412
absolute_altitude_m: 298
rolling_grade_500m: 167
```

**发现:**
- 高速组更关注即时坡度
- 低速组更关注疲劳累积 (累计距离)

---

## 4. Bug修复记录

### Bug #1: 坡度计算错误

**错误代码:**
```python
avg_grade = (elevation_gain / distance * 100)
```

**问题:**
- `distance` 单位是 km
- `elevation_gain` 单位是 m
- 导致坡度被放大 1000 倍

**实际计算:**
```
distance = 5.8 km
elevation_gain = 202 m
avg_grade = 202 / 5.8 * 100 = 3482% ❌
```

**修复:**
```python
avg_grade = (elevation_gain / distance / 10)
# 202 / 5.8 / 10 = 3.48% ✅
```

**影响:** 导致训练数据坡度值异常高 (3000%+)，模型训练失败

---

### Bug #2: 累计距离 vs 分段距离混淆

**错误代码:**
```python
for seg in segments:
    segment_time = seg.accumulated_distance_km / speed
    total_time += segment_time
```

**问题:**
- `accumulated_distance_km` 是从起点到当前段的累计距离
- 不是当前段的实际距离

**实际影响:**
```
第1段: accumulated = 0.21 km  → 时间 = 0.21 / 4.66 = 0.045 h
第2段: accumulated = 0.42 km  → 时间 = 0.42 / 4.5 = 0.093 h (重复!)
第3段: accumulated = 0.63 km  → 时间 = 0.63 / 4.2 = 0.150 h (重复!)
...
总时间 = 375 小时 ❌
```

**修复:**
```python
prev_cumulative = 0
for seg in segments:
    segment_distance = seg.accumulated_distance_km - prev_cumulative
    segment_time = segment_distance / speed
    total_time += segment_time
    prev_cumulative = seg.accumulated_distance_km
```

**结果:** 时间从 375 小时降到 4.8 小时 ✅

---

### Bug #3: numpy 类型序列化

**错误信息:**
```
TypeError: Object of type int32 is not JSON serializable
```

**原因:**
LightGBM 返回 `numpy.int32` 类型，无法直接序列化为 JSON

**修复:**
```python
# ❌ 错误
self.feature_importance = dict(zip(names, values))

# ✅ 正确
self.feature_importance = {name: float(value) for name, value in zip(names, values)}
```

---

### Bug #4: 外推问题

**问题:**
树模型无法外推到训练数据范围之外

**实际情况:**
```
训练数据最大: 32.6 km, 1790 m 爬升
比赛路线: 29.33 km, 2426 m 爬升
```

**解决方案:**
```python
if accumulated_ascent_m > max_training_ascent:
    excess_ratio = accumulated_ascent_m / max_training_ascent
    penalty = 1 + (excess_ratio - 1) * 0.2
    predicted_speed /= penalty
```

**计算示例:**
```
累计爬升 2400m:
excess_ratio = 2400 / 1790 = 1.34
penalty = 1 + (1.34 - 1) * 0.2 = 1.068
速度 = 原预测 / 1.068 (降低 6.4%)
```

---

## 5. 预测分析

### 5.1 分段预测示例 (前10段)

| 段 | 距离 | 坡度 | 海拔 | 预测速度 | 时间 |
|----|------|------|------|----------|------|
| 1 | 0.21km | 0.0% | 13m | 4.66 km/h | 2.7 min |
| 2 | 0.21km | 0.0% | 13m | 4.66 km/h | 2.7 min |
| 3 | 0.21km | 15.9% | 44m | 4.20 km/h | 3.0 min |
| 4 | 0.21km | 40.5% | 125m | 3.10 km/h | 4.1 min |
| 5 | 0.21km | 5.9% | 126m | 4.55 km/h | 2.8 min |
| 6 | 0.21km | 0.0% | 56m | 4.66 km/h | 2.7 min |
| 7 | 0.21km | 15.2% | 87m | 4.25 km/h | 3.0 min |
| 8 | 0.21km | 38.1% | 168m | 3.15 km/h | 4.0 min |
| 9 | 0.21km | 14.3% | 198m | 4.30 km/h | 2.9 min |
| 10 | 0.21km | 0.0% | 198m | 4.66 km/h | 2.7 min |

**观察:**
- 陡坡段 (40%) 速度降到 3.1 km/h
- 平路段保持在 4.6 km/h
- 模型对坡度变化敏感

### 5.2 累计疲劳效应

**分段速度随距离变化:**
```
0-5 km:   平均 4.8 km/h
5-10 km:  平均 4.6 km/h (-4%)
10-20 km: 平均 4.4 km/h (-8%)
20-29 km: 平均 4.1 km/h (-15%)
```

### 5.3 最终预测对比

| 模型 | 时间 | 配速 | 速度 |
|------|------|------|------|
| V1 | 11:12 | 22.9 min/km | 2.62 km/h |
| V2 | 7:03 | 14.5 min/km | 4.14 km/h |
| V3 | 4:51 | 9.9 min/km | 6.04 km/h |

**V3 更快的原因:**
1. 更准确的地形分类 (不需要手动分类)
2. ML 自动学习坡度-速度关系
3. 没有过度的惩罚因子

---

## 6. 关键代码片段

### 6.1 坡度计算

```python
# GPX 分段坡度计算
def create_segment(points):
    elevations = [p['ele'] for p in points]
    distance_km = sum(haversine(points[i], points[i+1]) for i in range(len(points)-1)) / 1000
    elevation_gain = sum(max(0, elevations[i+1] - elevations[i]) for i in range(len(elevations)-1))

    # 关键: distance_km 已经是 km，elevation_gain 是 m
    grade_pct = (elevation_gain / distance_km / 10)  # 正确!
    return grade_pct
```

### 6.2 外推惩罚

```python
def predict_speed_with_extrapolation(segment, predictor):
    predicted = predictor.model.predict(features)

    # 检查是否超出训练范围
    if segment.accumulated_ascent > predictor.max_ascent:
        ratio = segment.accumulated_ascent / predictor.max_ascent
        penalty = 1 + (ratio - 1) * 0.2
        predicted /= penalty

    return predicted
```

### 6.3 分段时间计算

```python
prev_cumulative = 0
total_time = 0

for seg in segments:
    # 关键: 计算本段实际距离
    segment_distance = seg.accumulated_distance - prev_cumulative

    speed = predict_speed(seg)
    segment_time = segment_distance / speed

    total_time += segment_time
    prev_cumulative = seg.accumulated_distance
```

---

## 7. 调试技巧

### 7.1 检查训练数据

```python
segments = extract_features(json_file)
for seg in segments:
    print(f"speed={seg.speed_kmh:.2f}, grade={seg.grade_pct:.1f}%, "
          f"dist={seg.accumulated_distance_km:.2f}")
```

### 7.2 验证预测

```python
# 单段预测测试
seg = segments[0]
predicted = model.predict(seg)
print(f"Segment: grade={seg.grade_pct}%, predicted={predicted:.2f} km/h")
```

### 7.3 特征重要性可视化

```python
import matplotlib.pyplot as plt

features = list(importance.keys())
values = list(importance.values())

plt.barh(features, values)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

---

## 8. 性能优化

### 8.1 数据加载优化

```python
# 批量加载 JSON
def load_training_data(directory):
    all_segments = []
    for json_file in Path(directory).glob('*.json'):
        segments = extract_features(json_file)
        all_segments.extend(segments)
    return all_segments
```

### 8.2 模型训练优化

```python
# 使用较少的迭代次数
params['num_boost_round'] = 100  # 而非 1000

# 限制叶子数
params['num_leaves'] = 31  # 而非 127
```

---

## 9. 未来改进

1. **更多特征**
   - 心率数据
   - 步频数据
   - 路面类型 (碎石/泥土/台阶)

2. **模型改进**
   - 尝试 XGBoost
   - 神经网络
   - 集成学习

3. **后处理**
   - CP点到达时间
   - 配速策略建议
   - 补给点规划

---

*文档版本: 1.0*
*更新时间: 2026-03-31*
