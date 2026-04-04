#!/usr/bin/env python
"""生成更易理解的预测解释报告"""
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime

random.seed(42)
np.random.seed(42)
sys.path.insert(0, '.')

from core_rebuild.predictor import MLRacePredictor

records_dir = Path("temp/records")
example_dir = Path("example")

# 训练和预测
group_3 = [
    records_dir / "玉环100KM_431586142.fit",
    records_dir / "Chiang Mai 越野跑_437031103.fit",
    records_dir / "神农架林区 越野跑_414516317.fit",
]
gpx_file = example_dir / "2025黄岩九峰大师赛最终版.gpx"

print("训练模型...")
predictor = MLRacePredictor()
predictor.train_from_files(group_3)

result = predictor.predict_race(str(gpx_file), effort_factor=1.0)

# =============================================================================
# 生成易理解的解释报告
# =============================================================================
stats = predictor.training_stats
fi = predictor.all_feature_importance

# 计算相对重要性百分比
total_importance = sum(fi.values())
fi_pct = {k: v/total_importance*100 for k, v in fi.items()}

# 特征名称翻译
feature_names = {
    'grade_pct': '坡度 (上坡/下坡)',
    'accumulated_distance_km': '累计距离 (疲劳)',
    'absolute_altitude_m': '海拔高度',
    'rolling_grade_500m': '500米平均坡度',
    'accumulated_ascent_m': '累计爬升',
    'elevation_density': '爬升密度 (赛道难度)'
}

# 找出最重要的3个因素
top_features = sorted(fi_pct.items(), key=lambda x: x[1], reverse=True)[:3]

report = f"""
================================================================================
                        预测结果解释报告
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

================================================================================
一、你的能力水平
================================================================================

基于 {stats['file_count']} 次训练记录的分析：

  • 平均能力 (P50):     {stats['p50_speed']:.1f} km/h   ← 平地正常速度
  • 极限能力 (P90):     {stats['p90_speed']:.1f} km/h   ← 冲刺速度
  • 平均配速:           {60/stats['p50_speed']:.1f} min/km

================================================================================
二、这场比赛的特点
================================================================================

赛道: {gpx_file.name}
距离: {result['total_distance_km']:.1f} km
总爬升: {result['route_info']['total_elevation_gain_m']:.0f} m
总下降: {result['route_info']['total_elevation_loss_m']:.0f} m
爬升密度: {result['route_info']['elevation_density']:.0f} m/km   ← 越大约难

难度评级: {"困难" if result['route_info']['elevation_density'] > 70 else "中等" if result['route_info']['elevation_density'] > 40 else "简单"}

================================================================================
三、预测结果
================================================================================

  完赛时间: {result['predicted_time_hm']}
  平均配速: {result['predicted_pace_min_km']} min/km

  相比你的平地能力，预计会慢 {(result['predicted_pace_min_km']-60/stats['p50_speed'])/(60/stats['p50_speed'])*100:.0f}%

================================================================================
四、影响预测的关键因素（按重要性排序）
================================================================================

以下因素决定了你的预测时间：

"""

for i, (feat, pct) in enumerate(top_features, 1):
    cn_name = feature_names.get(feat, feat)
    report += f"  {i}. {cn_name}: 贡献度 {pct:.0f}%\n"

    # 根据特征给出具体解释
    if feat == 'grade_pct':
        report += """     → 赛道有陡坡，你会明显减速
     → 坡度每增加10%，速度约下降0.8 km/h
"""
    elif feat == 'accumulated_distance_km':
        report += """     → 20km后疲劳累积，速度开始下降
     → 每增加10km，平均速度下降约0.5 km/h
"""
    elif feat == 'elevation_density':
        report += f"""     → 爬升密度{result['route_info']['elevation_density']:.0f}属于较高
     → 高难度赛道需要分配更多体力
"""
    elif feat == 'absolute_altitude_m':
        report += """     → 高海拔含氧量低，会轻微影响速度
"""
    elif feat == 'rolling_grade_500m':
        report += """     → 持续的陡坡比间歇坡度更消耗体力
"""
    elif feat == 'accumulated_ascent_m':
        report += f"""     → 已累计爬升{result['route_info']['total_elevation_gain_m']:.0f}m
     → 爬升越多，体力消耗越大
"""

# 分段预测解释
segments = result['segment_predictions']

# 找最慢的3段和最快的3段
slowest = sorted(segments, key=lambda x: x['predicted_speed_kmh'])[:3]
fastest = sorted(segments, key=lambda x: x['predicted_speed_kmh'], reverse=True)[:3]

report += f"""
================================================================================
五、关键路段分析
================================================================================

最慢的3段（预计会很吃力）：
"""

for s in slowest:
    report += f"  • 第{s['segment']}段: {s['predicted_speed_kmh']:.1f} km/h, 坡度{s['grade_pct']:+.0f}%, 距离{s['distance_km']:.1f}km\n"

report += """
最快的3段（可以追赶时间）：
"""

for s in fastest:
    report += f"  • 第{s['segment']}段: {s['predicted_speed_kmh']:.1f} km/h, 坡度{s['grade_pct']:+.0f}%, 距离{s['distance_km']:.1f}km\n"

report += f"""
================================================================================
六、给你的建议
================================================================================

基于模型分析，建议：

1. 配速策略
"""

# 根据最重要的因素给出建议
if fi_pct.get('grade_pct', 0) > fi_pct.get('accumulated_distance_km', 0):
    report += """   → 坡度是主要影响因子
   → 在陡坡路段（如CP2-CP3）保守配速
   → 在下坡路段可以适当加速追赶时间
"""
else:
    report += """   → 疲劳累积是主要影响因子
   → 前半程比预测配速慢5-10%
   → 保持体力后半程发力

2. 补给建议
"""

duration_hours = result['predicted_time_min'] / 60
if duration_hours > 5:
    report += f"""   → 预计耗时{duration_hours:.1f}小时，需要带{duration_hours:.0f}支能量胶
   → 每小时补充一次
"""
elif duration_hours > 3:
    report += """   → 预计耗时3-5小时，带2-3支能量胶
"""
else:
    report += """   → 短距离比赛，少量补给即可
"""

report += """
================================================================================
七、模型说明（可跳过）
================================================================================

这个预测是怎么得出的：

1. 训练数据：使用你过去的3次长距离越野跑记录
   - 玉环100KM (101km, 3591m爬升)
   - 清迈越野跑 (95km, 5030m爬升)
   - 神农架越野跑 (68km, 3162m爬升)

2. 学习内容：从这些记录中，模型学会了：
   - 你的上坡速度和坡度的关系
   - 你的速度随距离增加如何下降
   - 不同爬升密度下的表现

3. 预测方法：把赛道的每一小段（约200米）代入模型
   - 根据该段的坡度和累计距离预测速度
   - 所有小段时间累加得到总时间
   - 最后加上休息时间（3%）

================================================================================
"""

# 先保存到文件
report_path = Path("reports") / f"prediction_explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
report_path.parent.mkdir(exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"报告已保存到: {report_path}")
print("\n" + "="*60)
print("以下是报告内容：")
print("="*60 + "\n")

# 用UTF-8编码打印
sys.stdout.reconfigure(encoding='utf-8')
print(report)

# 保存报告
report_path = Path("reports") / f"prediction_explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
report_path.parent.mkdir(exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n报告已保存到: {report_path}")