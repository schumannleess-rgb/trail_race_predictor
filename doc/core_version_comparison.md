# Core 版本对比分析

## 1. 随机噪声注入差异

| 版本 | 位置 | 噪声代码 |
|------|------|----------|
| **core/** | `_train_fallback()` (line 577-580) | `np.random.normal(0, 2)` 和 `np.random.normal(0, 0.5)` |
| **core_rebuild/** | 无 | 纯ML，无噪声 |

**用途**: core 的噪声只在 **fallback 模式**（当LightGBM不可用时）使用，用于生成虚拟训练数据。

**影响**: 正常运行 LightGBM 时不受影响，但如果模型训练失败使用 fallback，每次预测结果会略有不同。

---

## 2. Fallback 对比

两者都有 fallback，但位置不同：

| 方面 | core/ | core_rebuild/ |
|------|-------|---------------|
| **fallback存在** | ✅ 有 | ✅ 有 |
| **fallback位置** | `predict_speed()` 内部 (line 250-260) | `_predict_fallback()` 方法 (line 195-202) |
| **fallback逻辑** | 坡度>5: 爬坡速度, <-5: 下坡速度, 其他: 平路速度 | **完全相同** |
| **训练数据生成** | `np.random.normal()` 噪声 | 无噪声，固定随机种子 |

### core/ 的 fallback 训练（独有噪声）

```python
# core/predictor.py line 577-580
seg_grade = avg_grade + np.random.normal(0, 2)  # 坡度加噪声
speed_kmh = avg_speed + np.random.normal(0, 0.5)  # 速度加噪声
```

这个噪声只在 **LightGBM 训练失败** 时使用 fallback 模式。

### core_rebuild/ 的 fallback 训练

```python
# core_rebuild/predictor/model.py - 无额外噪声注入
# 直接使用固定参数
```

---

## 3. VAM 阈值验证差异

| 版本 | VAM 阈值 | 代码 |
|------|----------|------|
| **core/** | 1000 m/h | `if vam > 1000: speed /= vam / 1000` |
| **core_rebuild/** | 1000 m/h | `if vam > 1000: speed /= vam / 1000` |

**公式**: `VAM = 速度(km/h) × 10 × 坡度(%)`

**示例**:
- 速度 8 km/h，坡度 15% → VAM = 1200 m/h → 超限，应用惩罚 1.2x
- 速度 6 km/h，坡度 20% → VAM = 1200 m/h → 超限，应用惩罚 1.2x

**结论**: VAM 逻辑完全相同，都是 1000 m/h 阈值。

---

## 4. 其他共同逻辑

### 最大速度限制
两者完全相同：
```python
max_speed = self.p90_speed * 1.1 if self.p90_speed > 0 else 15.0
return max(1.0, min(max_speed, predicted_speed))
```

### 外推惩罚
两者完全相同：
```python
# 距离超出
if segment.accumulated_distance_km > self.max_training_distance:
    ratio = segment.accumulated_distance_km / self.max_training_distance
    speed /= 1 + (ratio - 1) * 0.3

# 爬升超出
if segment.accumulated_ascent_m > self.max_training_ascent:
    ratio = segment.accumulated_ascent_m / self.max_training_ascent
    speed /= 1 + (ratio - 1) * 0.2
```

### LightGBM 参数
两者都使用 `seed: 42`，保证训练可复现。

---

## 5. 预测结果差异

测试结果：
- **core/** → 6:44:54 (405 min)
- **core_rebuild/** → 6:34:32 (394 min)
- **差异**: 约 11 分钟

### 差异可能来源

1. **随机噪声** - core 在 fallback 训练时有 `np.random.normal()` 噪声注入
2. **代码结构** - core_rebuild 代码更模块化、清晰
3. **运行参数** - 传入的 effort_factor 可能不同

---

## 6. 结论与建议

### 功能上
- 两者核心预测逻辑**完全相同**
- VAM 阈值、最大速度限制、外推惩罚逻辑一致
- 差异主要在代码组织结构和 fallback 训练方式

### 效果上
- core/ 版本在 fallback 模式下有随机波动
- core_rebuild/ 版本更稳定，代码更清晰

### 建议
- **生产环境使用 core_rebuild/** - 代码更清晰稳定
- 如需使用 core/，确保 LightGBM 正常加载避免触发 fallback

---

---

## 7. 代码审查更正 (2026-04-04)

### 7.1 噪声位置更正

通过代码审查发现，原文档对噪声位置的描述存在偏差：

| 项目 | 原文档描述 | 实际位置 | 实际情况 |
|------|-----------|----------|----------|
| **噪声所在函数** | `_train_fallback()` (line 577-580) | `_extract_from_summary()` | `_train_fallback()` 本身**不含**随机噪声 |
| **噪声实际行号** | line 577-580 | `core copy/predictor.py` line 562, 565 | 位于 JSON 摘要数据提取时 |
| **触发条件** | fallback 模式（LightGBM 不可用） | `extract_from_json()` 处理无 metrics 的摘要数据时 | **正常 LightGBM 训练也会触发** |

### 7.2 实际噪声代码位置

```python
# core copy/predictor.py - FeatureExtractor._extract_from_summary() (line 559-565)
for i in range(num_segments):
    seg_distance = distance / num_segments
    seg_elevation = elevation_gain / num_segments
    seg_grade = avg_grade + np.random.normal(0, 2)  # 添加一些变化

    segments.append(SegmentFeatures(
        speed_kmh=avg_speed + np.random.normal(0, 0.5),  # 速度加噪声
        ...
    ))
```

**触发场景**：当 JSON 文件没有详细 `metrics` 字段，只有汇总数据（`activity_info`）时，会调用 `_extract_from_summary()` 生成虚拟分段数据进行训练。

### 7.3 审查确认的相同逻辑

| 逻辑 | `core copy/` | `core_rebuild/` | 状态 |
|------|--------------|-----------------|------|
| VAM 阈值 1000 m/h | line 237 | line 189 | ✅ 确认一致 |
| 最大速度 `p90_speed * 1.1` | line 242 | line 192 | ✅ 确认一致 |
| 距离外推惩罚 `(ratio-1) * 0.3` | line 222 | line 180 | ✅ 确认一致 |
| 爬升外推惩罚 `(ratio-1) * 0.2` | line 227 | line 183 | ✅ 确认一致 |
| Fallback 坡度分界 (>5, <-5) | line 248-253 | line 196-201 | ✅ 确认一致 |

### 7.4 审查结论

1. **原文档功能结论正确**：VAM、最大速度、外推惩罚等核心逻辑完全一致
2. **原文档位置信息需更正**：噪声不在 `_train_fallback()`，而在 `_extract_from_summary()`
3. **影响范围比原文档描述更广**：噪声在正常 JSON 训练时也会触发，不只是 fallback 模式
4. **代码结构差异属实**：`core_rebuild/` 更模块化，`core copy/` 更扁平

### 7.5 建议更新原文档

- 第 1 节表格中 `core/` 位置应改为：`FeatureExtractor._extract_from_summary()` (line 562, 565)
- 第 1 节"用途"应更新为：噪声在处理无详细 metrics 的 JSON 文件时触发，用于生成虚拟训练分段
- 第 2 节标题应更正为：`JSON 摘要数据提取差异` 而非 `Fallback 对比`

---

---

## 8. 再次审查更正 (2026-04-04 下午)

### 8.1 `core_rebuild/` 同样有噪声 — diff doc 错误

经再次验证，`core_rebuild/` 在 `_extract_from_summary()` 中**也有相同噪声**：

```python
# core_rebuild/predictor/extractor.py line 219-222
return [
    SegmentFeatures(
        speed_kmh=avg_speed + float(np.random.normal(0, 0.5)),   # ← 相同噪声
        grade_pct=avg_grade + float(np.random.normal(0, 2)),    # ← 相同噪声
```

**结论**：
- `core copy/` 和 `core_rebuild/` 的 `_extract_from_summary()` 都有噪声
- Diff doc 说 `core_rebuild/` 无噪声是**错误的**
- Diff doc 说噪声在 `_train_fallback()` 也是**错误的**

### 8.2 唯一真实差异：Rest Ratio

| 功能 | `core copy/` | `core_rebuild/` |
|------|-------------|-----------------|
| Rest Ratio | ❌ 没有 | ✅ 有 |
| 计算方式 | N/A | 从 FIT 的 `total_elapsed_time / total_timer_time` 计算 |
| 预测时应用 | `total_time = moving_time` | `total_time = moving_time / (1 - rest_ratio)` |

`core_rebuild/` 的 rest ratio 处理是**用户主动添加的功能**，用于更准确地预测包含休息时间的实际完赛时间。

### 8.3 共同确认的逻辑

| 项目 | 代码 | 状态 |
|------|------|------|
| VAM 阈值 | `if vam > 1000: speed /= vam / 1000` | ✅ 完全相同 |
| 最大速度 | `p90_speed * 1.1` | ✅ 完全相同 |
| 外推惩罚 | `(ratio-1) * 0.3` / `(ratio-1) * 0.2` | ✅ 完全相同 |
| Fallback 逻辑 | 坡度 >5/<-5 分组均值 | ✅ 完全相同 |
| FilterConfig | GPX/FIT 双配置 | ✅ 完全相同 |
| SegmentFeatures | 相同 dataclass | ✅ 完全相同 |

---

## 9. 结论

### 重构目标达成

`core_rebuild/` 的主要目标是**模块化重构**：
- 单文件 → 拆分 `predictor/`, `model.py`, `features.py`, `extractor.py`, `gpx_parser.py`, `cli.py`
- 无 CLI → 有 CLI (`python -m predictor.cli`)
- 添加 rest ratio 功能

### 功能验证完成

- 核心预测逻辑与 `core copy/` 完全一致
- VAM、最大速度、外推惩罚经验证相同
- 唯一额外功能：rest ratio（属于功能增强，不是 bug）

### 唯一未解决问题

**噪声未设置随机种子** — 两次运行同一数据可能产生略微不同的虚拟分段（影响仅限无 metrics 的 JSON 文件）。

---

**状态：Rebuild 完成，功能验证完成。**

*最后更新: 2026-04-04 下午*
