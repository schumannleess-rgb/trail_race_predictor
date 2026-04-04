# Trail Race Predictor V1.2 参数配置参考手册

> 版本: V1.2
> 目的: 集中管理所有可配置参数，防止参数被意外修改

---

## 目录

1. [滤波参数](#1-滤波参数)
2. [机器学习参数](#2-机器学习参数)
3. [物理约束参数](#3-物理约束参数)
4. [显示参数](#4-显示参数)
5. [验证参数](#5-验证参数)

---

## 1. 滤波参数

### 1.1 GPX 赛道滤波

**配置位置**: `core/utils.py:16-23`

```python
GPX = {
    'resample_spacing_m': 20,      # 20米重采样
    'window_size': 7,              # 覆盖 140 米地形
    'poly_order': 2,               # 保留山体起伏曲线
    'max_grade_pct': 45.0,         # 严格截断 ±45%
    'resample_required': True,     # 必须重采样
    'min_distance_m': 0.5          # 最小距离阈值
}
```

**参数说明**:

| 参数 | 当前值 | 允许范围 | 说明 | 修改影响 |
|------|--------|----------|------|----------|
| `resample_spacing_m` | 20 | 10-50 | 重采样间距（米） | 影响 GPX 处理精度和速度 |
| `window_size` | 7 | 5-11 | SG 滤波窗口大小 | 影响噪声去除效果 |
| `poly_order` | 2 | 2-3 | 多项式阶数 | 影响曲线保真度 |
| `max_grade_pct` | 45.0 | 30-60 | 坡度截断值 | 影响极端坡度处理 |

**重要**: ⚠️ 这些参数直接影响预测精度，修改前必须充分测试

---

### 1.2 FIT 训练记录滤波

**配置位置**: `core/utils.py:25-33`

```python
FIT = {
    'resample_spacing_m': None,    # 不重采样
    'window_size': 7,              # 覆盖 7-10 秒运动
    'poly_order': 2,               # 保留加速/减速趋势
    'max_grade_pct': 50.0,         # 宽松截断 ±50%
    'resample_required': False,    # 不需要重采样
    'min_distance_m': 0.5          # 最小距离阈值
}
```

**参数说明**:

| 参数 | 当前值 | 允许范围 | 说明 | 修改影响 |
|------|--------|----------|------|----------|
| `resample_spacing_m` | None | None | 不重采样 | 保持原始数据密度 |
| `window_size` | 7 | 5-15 | SG 滤波窗口大小 | 影响噪声去除效果 |
| `poly_order` | 2 | 2-3 | 多项式阶数 | 影响曲线保真度 |
| `max_grade_pct` | 50.0 | 40-70 | 坡度截断值 | 比 GPX 更宽松 |

---

### 1.3 分段长度

**配置位置**: `core/predictor.py:265`, `core/predictor.py:473`

```python
segment_length_m = 200  # 200米分段
```

**参数说明**:

| 参数 | 当前值 | 允许范围 | 说明 | 修改影响 |
|------|--------|----------|------|----------|
| `segment_length_m` | 200 | 100-500 | 分段长度（米） | 影响特征粒度和训练数据量 |

**重要**: ⚠️ 修改此参数需要重新评估所有模型性能

---

## 2. 机器学习参数

### 2.1 LightGBM 模型参数

**配置位置**: `core/predictor.py:123-133`

```python
params = {
    'objective': 'regression',     # 回归任务
    'metric': 'mae',               # 平均绝对误差
    'num_leaves': 31,              # 叶子节点数
    'learning_rate': 0.05,         # 学习率
    'feature_fraction': 0.9,       # 特征采样比例
    'bagging_fraction': 0.8,       # 数据采样比例
    'bagging_freq': 5,             # 采样频率
    'verbose': -1,                 # 禁用日志
    'min_data': 1,                 # 最小叶子数据
}
num_boost_round = 100              # 迭代次数
```

**参数说明**:

| 参数 | 当前值 | 说明 | 修改影响 |
|------|--------|------|----------|
| `objective` | 'regression' | 任务类型 | 不应修改 |
| `metric` | 'mae' | 评估指标 | 可改为 'rmse' |
| `num_leaves` | 31 | 叶子节点数 | 影响模型复杂度 |
| `learning_rate` | 0.05 | 学习率 | 影响收敛速度 |
| `feature_fraction` | 0.9 | 特征采样 | 防止过拟合 |
| `bagging_fraction` | 0.8 | 数据采样 | 防止过拟合 |
| `bagging_freq` | 5 | 采样频率 | 影响采样效果 |
| `min_data` | 1 | 最小数据 | 小数据集必需 |
| `num_boost_round` | 100 | 迭代次数 | 影响训练时间 |

---

### 2.2 训练特征列表

**配置位置**: `core/predictor.py:92-100`, `core/predictor.py:110-117`

```python
# 特征顺序（重要）
features = [
    'grade_pct',              # 当前坡度
    'rolling_grade_500m',     # 过去500米平均坡度
    'accumulated_distance_km', # 累计距离
    'accumulated_ascent_m',    # 累计爬升
    'absolute_altitude_m',     # 绝对海拔
    'elevation_density'        # 爬升密度
]
```

**特征说明**:

| 特征 | 类型 | 单位 | 说明 | 重要性 |
|------|------|------|------|--------|
| `grade_pct` | float | % | 当前坡度 | 最重要 (~35-40%) |
| `rolling_grade_500m` | float | % | 过去500米平均坡度 | 中等 (~5-8%) |
| `accumulated_distance_km` | float | km | 累计距离 | 重要 (~30-35%) |
| `accumulated_ascent_m` | float | m | 累计爬升 | 重要 (~15-20%) |
| `absolute_altitude_m` | float | m | 绝对海拔 | 较低 (~2-5%) |
| `elevation_density` | float | m/km | 爬升密度 | 中等 (~5-10%) |

**重要**: ⚠️ 特征顺序和数量直接影响模型，不可随意修改

---

## 3. 物理约束参数

### 3.1 VAM 限制

**配置位置**: `core/predictor.py:231-238`

```python
# VAM 限制参数
VAM_THRESHOLD = 15              # 触发坡度 (%)
VAM_MAX = 1000                  # 最大 VAM (m/h)
```

**参数说明**:

| 参数 | 当前值 | 说明 | 修改影响 |
|------|--------|------|----------|
| `VAM_THRESHOLD` | 15 | 触发限制的坡度阈值 | 影响何时应用限制 |
| `VAM_MAX` | 1000 | 最大垂直上升速度 | 影响陡坡预测 |

**VAM 等级参考**:
| VAM (m/h) | 水平 |
|-----------|------|
| 400-600 | 业余水平 |
| 600-800 | 进阶水平 |
| 800-1000 | 精英水平 |
| 1000-1200 | 职业水平 |
| >1200 | 世界级 |

---

### 3.2 外推惩罚参数

**配置位置**: `core/predictor.py:219-227`

```python
# 距离外推惩罚
DISTANCE_PENALTY_FACTOR = 0.3   # 每超出10%慢3%

# 爬升外推惩罚
ASCENT_PENALTY_FACTOR = 0.2     # 每超出10%慢2%
```

**参数说明**:

| 参数 | 当前值 | 说明 | 修改影响 |
|------|--------|------|----------|
| `DISTANCE_PENALTY_FACTOR` | 0.3 | 距离外推惩罚系数 | 影响长距离预测 |
| `ASCENT_PENALTY_FACTOR` | 0.2 | 爬升外推惩罚系数 | 影响高爬升预测 |

**计算公式**:
```python
# 距离外推
penalty = 1 + (excess_ratio - 1) * DISTANCE_PENALTY_FACTOR
predicted_speed /= penalty

# 爬升外推
penalty = 1 + (excess_ratio - 1) * ASCENT_PENALTY_FACTOR
predicted_speed /= penalty
```

---

### 3.3 速度限制参数

**配置位置**: `core/predictor.py:241-242`

```python
# 速度限制
MIN_SPEED = 1.0                 # 最小速度
MAX_SPEED_FACTOR = 1.1          # 最大速度 = P90 * 1.1
DEFAULT_MAX_SPEED = 15.0        # 默认最大速度
```

---

## 4. 显示参数

### 4.1 分段表显示规则

**配置位置**: `app.py:582-727`, `app_splits.py:5-157`

```python
# 分段间隔（无CP点时）
SPLIT_INTERVAL_KM = 5            # 5km间隔

# CP点匹配距离
CP_MATCH_DISTANCE_M = 500        # 500米内匹配

# 显示模式判断
USE_CP_MODE = len(cp_points) > 0  # 有CP点用CP模式
```

**显示规则**:

| 场景 | 显示方式 | 说明 |
|------|----------|------|
| 有 CP 点 | 按 CP 点分段 | 每个CP点显示一行 |
| 无 CP 点 | 按 5km 分段 | 每5km显示一行 |
| 起点 | 永远显示 | 第一行固定为起点 |
| 终点 | 永远显示 | 最后一行固定为终点 |

---

### 4.2 难度评级参数

**配置位置**: `app.py:542-551`, `core/report_generator.py:352-363`

```python
# 难度等级阈值
DIFFICULTY_EXTREME = 100         # 极难
DIFFICULTY_HARD = 70             # 困难
DIFFICULTY_MODERATE = 40         # 中等
# < 40 为轻松
```

**难度评级表**:

| 爬升密度 (m/km) | 等级 | 图标 |
|-----------------|------|------|
| > 100 | 极难 | ⚠️ |
| 70-100 | 困难 | 🔥 |
| 40-70 | 中等 | 🏃 |
| < 40 | 轻松 | ✅ |

---

### 4.3 地形类型参数

**配置位置**: `core/predictor.py:934-947`

```python
# 坡度分类阈值
GRADE_STEEP_CLIMB = 15          # 陡上坡
GRADE_MODERATE_CLIMB = 8        # 中上坡
GRADE_GENTLE_CLIMB = 3          # 缓上坡
GRADE_GENTLE_DESCENT = -3       # 缓下坡
GRADE_MODERATE_DESCENT = -8     # 中下坡
GRADE_STEEP_DESCENT = -15       # 陡下坡
```

**地形类型表**:

| 坡度范围 (%) | 类型 |
|-------------|------|
| > 15 | 陡上坡 |
| 8-15 | 中上坡 |
| 3-8 | 缓上坡 |
| -3 到 3 | 平地 |
| -8 到 -3 | 缓下坡 |
| -15 到 -8 | 中下坡 |
| < -15 | 陡下坡 |

---

### 4.4 努力程度参数

**配置位置**: `app.py:142-158`

```python
# 努力程度滑块
EFFORT_MIN = 0.8                 # 最小值
EFFORT_MAX = 1.2                 # 最大值
EFFORT_DEFAULT = 1.0             # 默认值
EFFORT_STEP = 0.01               # 步长

# 显示阈值
EFFORT_CONSERVATIVE = 0.95       # 保守阈值
EFFORT_AGGRESSIVE = 1.05         # 比赛状态阈值
```

**努力程度说明**:

| 值范围 | 策略 | 说明 |
|--------|------|------|
| 0.8-0.95 | 保守 | 适合长距离或恢复期 |
| 0.95-1.05 | 平均 | P50 能力水平 |
| 1.05-1.2 | 比赛 | 接近 P90 能力 |

---

## 5. 验证参数

### 5.1 文件验证参数

**配置位置**: `data/data_validator.py:22-32`

```python
# 坐标范围
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)

# 中国范围（可选）
CHINA_BOUNDS = {
    'min_lat': 18,
    'max_lat': 54,
    'min_lon': 73,
    'max_lon': 135
}
```

---

### 5.2 数据质量参数

**配置位置**: `data/data_validator.py:110-111`, `data/data_validator.py:189-190`

```python
# 最小点数警告
GPX_MIN_POINTS_WARNING = 50     # GPX 最少点数警告
FIT_MIN_POINTS_WARNING = 60     # FIT 最少点数警告
```

---

### 5.3 文件数量参数

**配置位置**: `app.py:166-169`, `core/predictor.py:666-668`

```python
# 文件数量限制
MIN_FIT_FILES = 3                # 最少文件数（警告）
RECOMMENDED_FIT_FILES = 15       # 推荐文件数
OPTIMAL_FIT_FILES = 20           # 最佳文件数
MAX_AUTO_SELECT_FILES = 20       # 自动选取最大数
```

---

## 6. 报告参数

### 6.1 HTML 报告样式

**配置位置**: `core/report_generator.py:88-313`

```python
# CSS 样式（部分）
HEADER_GRADIENT = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
BODY_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
METRIC_COLOR = "#FF4B4B"          # 指标颜色
```

---

### 6.2 TXT 报告格式

**配置位置**: `core/report_generator.py:764-961`

```python
# 表格格式
TABLE_WIDTH = 90                 # 表格总宽度
COLUMN_WIDTHS = {                # 列宽配置
    'position': 12,
    'segment_dist': 10,
    'distance': 8,
    'ascent': 8,
    'descent': 8,
    'time': 10,
    'cumulative_time': 10,
    'speed': 12
}
```

---

## 7. 性能参数

### 7.1 数据处理性能

```python
# 最大文件处理数
MAX_FILES_TO_PROCESS = 20        # 最多处理20个文件

# 缓存设置
USE_CACHE = True                 # 使用缓存
CACHE_TTL = 3600                 # 缓存有效期（秒）
```

---

## 8. 调试参数

### 8.1 日志级别

```python
# 日志配置
LOG_LEVEL = 'INFO'               # 日志级别
SHOW_TRAINING_PROGRESS = True    # 显示训练进度
SHOW_FEATURE_IMPORTANCE = True   # 显示特征重要性
```

---

## 参数修改流程

1. **确认影响范围**
   - 查阅本文档了解参数影响
   - 评估修改对系统的影响

2. **在测试环境验证**
   - 修改参数值
   - 运行完整测试
   - 对比结果差异

3. **文档更新**
   - 更新本文档中的参数值
   - 记录修改原因和日期

4. **团队确认**
   - 获得团队成员确认
   - 在确认清单上签字

5. **正式发布**
   - 合并代码
   - 更新版本号

---

## 变更记录

| 日期 | 参数 | 旧值 | 新值 | 修改人 | 确认人 |
|------|------|------|------|--------|--------|
| | | | | | |

---

*本文档是 Trail Race Predictor V1.2 的参数配置参考，所有参数修改必须经过确认流程。*
