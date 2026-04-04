#!/usr/bin/env python
"""分析3个文件和7个文件训练结果差异 - 简化版"""
import sys
import random
import numpy as np
from pathlib import Path
import json

# Set random seed
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '.')
print("=" * 70)
print("分析：3个文件 vs 7个文件 训练结果差异")
print("=" * 70)

from core_rebuild.predictor import MLRacePredictor

records_dir = Path("temp/records")
example_dir = Path("example")

# 定义两组文件
group_3 = [
    records_dir / "玉环100KM_431586142.fit",
    records_dir / "Chiang Mai 越野跑_437031103.fit",
    records_dir / "神农架林区 越野跑_414516317.fit",
]

group_7 = list(records_dir.glob("*.fit"))
gpx_file = example_dir / "2025黄岩九峰大师赛最终版.gpx"

print(f"\n【文件列表】")
print(f"3个文件: {[f.name for f in group_3]}")
print(f"7个文件: {[f.name for f in group_7]}")

# =============================================================================
# 训练模型对比
# =============================================================================
print("\n" + "=" * 70)
print("【训练模型】")
print("=" * 70)

# 训练3文件模型
print("\n--- 训练3文件模型 ---")
predictor_3 = MLRacePredictor()
success_3 = predictor_3.train_from_files(group_3)
print(f"训练成功: {success_3}")

# 训练7文件模型
print("\n--- 训练7文件模型 ---")
predictor_7 = MLRacePredictor()
success_7 = predictor_7.train_from_files(group_7)
print(f"训练成功: {success_7}")

# =============================================================================
# 模型统计对比
# =============================================================================
print("\n" + "=" * 70)
print("【模型统计对比】")
print("=" * 70)

stats_3 = predictor_3.training_stats
stats_7 = predictor_7.training_stats

print(f"\n{'指标':<25} {'3文件':<15} {'7文件':<15} {'差异':<15}")
print("-" * 70)

for key in ['file_count', 'segment_count', 'avg_speed', 'p50_speed', 'p90_speed', 'effort_range', 'avg_rest_ratio', 'calibrated_max_grade_pct']:
    v3 = stats_3.get(key, 0)
    v7 = stats_7.get(key, 0)
    if isinstance(v3, (int, float)):
        diff = v7 - v3
        diff_str = f"{diff:+.2f}" if isinstance(v3, float) else f"{diff:+d}"
    else:
        diff_str = "N/A"
    print(f"{key:<25} {v3:<15} {v7:<15} {diff_str:<15}")

# =============================================================================
# 特征重要性对比
# =============================================================================
print("\n" + "=" * 70)
print("【特征重要性对比】")
print("=" * 70)

fi_3 = predictor_3.all_feature_importance
fi_7 = predictor_7.all_feature_importance

print(f"\n{'特征':<25} {'3文件':<15} {'7文件':<15} {'变化':<15}")
print("-" * 70)

all_features = set(fi_3.keys()) | set(fi_7.keys())
for feat in sorted(all_features, key=lambda x: fi_7.get(x, 0), reverse=True):
    v3 = fi_3.get(feat, 0)
    v7 = fi_7.get(feat, 0)
    diff = v7 - v3
    print(f"{feat:<25} {v3:<15.0f} {v7:<15.0f} {diff:+15.0f}")

# =============================================================================
# 预测对比
# =============================================================================
print("\n" + "=" * 70)
print("【预测结果对比】")
print("=" * 70)

result_3 = predictor_3.predict_race(str(gpx_file), effort_factor=1.0)
result_7 = predictor_7.predict_race(str(gpx_file), effort_factor=1.0)

print(f"\n{'指标':<25} {'3文件':<20} {'7文件':<20} {'差异':<20}")
print("-" * 85)

keys = ['total_distance_km', 'predicted_time_hm', 'predicted_time_min',
        'predicted_pace_min_km', 'predicted_speed_kmh', 'predicted_moving_time_min']

for key in keys:
    v3 = result_3.get(key, 0)
    v7 = result_7.get(key, 0)
    if isinstance(v3, float):
        diff = v7 - v3
        print(f"{key:<25} {str(v3):<20} {str(v7):<20} {diff:+20.2f}")
    else:
        print(f"{key:<25} {str(v3):<20} {str(v7):<20} {'-':<20}")

# =============================================================================
# 速度边界对比
# =============================================================================
print("\n" + "=" * 70)
print("【速度边界对比】")
print("=" * 70)

print(f"\n{'指标':<25} {'3文件':<15} {'7文件':<15} {'变化率':<15}")
print("-" * 70)

p50_3 = stats_3.get('p50_speed', 0)
p50_7 = stats_7.get('p50_speed', 0)
p90_3 = stats_3.get('p90_speed', 0)
p90_7 = stats_7.get('p90_speed', 0)
avg_3 = stats_3.get('avg_speed', 0)
avg_7 = stats_7.get('avg_speed', 0)

print(f"{'P50 (平均能力)':<25} {p50_3:<15.2f} {p50_7:<15.2f} {(p50_7-p50_3)/p50_3*100:+10.1f}%")
print(f"{'P90 (极限能力)':<25} {p90_3:<15.2f} {p90_7:<15.2f} {(p90_7-p90_3)/p90_3*100:+10.1f}%")
print(f"{'平均速度':<25} {avg_3:<15.2f} {avg_7:<15.2f} {(avg_7-avg_3)/avg_3*100:+10.1f}%")

# =============================================================================
# 分段预测对比
# =============================================================================
print("\n" + "=" * 70)
print("【分段预测对比】")
print("=" * 70)

segs_3 = result_3.get('segment_predictions', [])
segs_7 = result_7.get('segment_predictions', [])

# 比较前10段的速度预测
print(f"\n前10段速度预测对比:")
print(f"{'段':<5} {'3文件速度':<12} {'7文件速度':<12} {'差异':<10} {'差异率':<10}")
print("-" * 50)

for i in range(min(10, len(segs_3), len(segs_7))):
    s3 = segs_3[i].get('predicted_speed_kmh', 0)
    s7 = segs_7[i].get('predicted_speed_kmh', 0)
    diff = s7 - s3
    diff_pct = diff / s3 * 100 if s3 > 0 else 0
    print(f"{i+1:<5} {s3:<12.2f} {s7:<12.2f} {diff:+10.2f} {diff_pct:+10.1f}%")

# =============================================================================
# 结论
# =============================================================================
print("\n" + "=" * 70)
print("【分析结论】")
print("=" * 70)

print(f"""
1. 数据规模差异:
   - 3文件: {stats_3['segment_count']}段, 7文件: {stats_7['segment_count']}段
   - 7文件增加了 {stats_7['segment_count'] - stats_3['segment_count']} 段 (+{(stats_7['segment_count']-stats_3['segment_count'])/stats_3['segment_count']*100:.0f}%)

2. 速度边界变化:
   - P50 (平均能力): {p50_3:.2f} -> {p50_7:.2f} km/h ({(p50_7-p50_3)/p50_3*100:+.1f}%)
   - P90 (极限能力): {p90_3:.2f} -> {p90_7:.2f} km/h ({(p90_7-p90_3)/p90_3*100:+.1f}%)
   - 平均速度: {avg_3:.2f} -> {avg_7:.2f} km/h ({(avg_7-avg_3)/avg_3*100:+.1f}%)

3. 特征重要性变化:
   - grade_pct (坡度): {fi_3.get('grade_pct',0):.0f} -> {fi_7.get('grade_pct',0):.0f}
   - accumulated_distance_km (累计距离): {fi_3.get('accumulated_distance_km',0):.0f} -> {fi_7.get('accumulated_distance_km',0):.0f}
   - accumulated_ascent_m (累计爬升): {fi_3.get('accumulated_ascent_m',0):.0f} -> {fi_7.get('accumulated_ascent_m',0):.0f}

4. 预测结果变化:
   - 完赛时间: {result_3['predicted_time_hm']} -> {result_7['predicted_time_hm']}
   - 平均配速: {result_3['predicted_pace_min_km']:.1f} -> {result_7['predicted_pace_min_km']:.1f} min/km (+{(result_7['predicted_pace_min_km']-result_3['predicted_pace_min_km'])/result_3['predicted_pace_min_km']*100:.1f}%)
   - 移动时间: {result_3['predicted_moving_time_min']} -> {result_7['predicted_moving_time_min']} min

5. 核心原因:
   - 7文件包含了更多短距离训练数据(16-50km)
   - 这些数据的平均速度更慢,拉低了P50/P90能力边界
   - 模型预测更保守,假设参赛者能力较低
""")

# 保存结果到JSON
output = {
    'group_3_stats': stats_3,
    'group_7_stats': stats_7,
    'group_3_fi': fi_3,
    'group_7_fi': fi_7,
    'result_3': result_3,
    'result_7': result_7,
}

with open('reports/analysis_diff.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n分析结果已保存到: reports/analysis_diff.json")
print("\n=== 分析完成 ===")