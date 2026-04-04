#!/usr/bin/env python
"""测试可解释性模块"""
import sys
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)
sys.path.insert(0, '.')

print("=" * 70)
print("测试可解释性模块")
print("=" * 70)

# 导入模块
from core_rebuild.predictor import MLRacePredictor
from core_rebuild import explainer

records_dir = Path("temp/records")
example_dir = Path("example")

# 训练模型
group_3 = [
    records_dir / "玉环100KM_431586142.fit",
    records_dir / "Chiang Mai 越野跑_437031103.fit",
    records_dir / "神农架林区 越野跑_414516317.fit",
]
gpx_file = example_dir / "2025黄岩九峰大师赛最终版.gpx"

print("\n【1】训练模型...")
predictor = MLRacePredictor()
predictor.train_from_files(group_3)

print("\n【2】解析GPX...")
segments, route_info = predictor.parse_gpx_route(str(gpx_file))

print("\n【3】创建解释器...")
explainer_obj = explainer.ModelExplainer(predictor._model)

print("\n【4】解释前10个segment...")
for i, seg in enumerate(segments[:10]):
    exp = explainer_obj.explain_prediction(seg)
    explainer.print_prediction_explanation(exp)
    print()

print("\n【5】生成汇总报告...")
print(explainer_obj.generate_summary_report(segments[:30], None))

print("\n【6】测试简化公式...")
p50 = predictor.training_stats['p50_speed']
print(f"P50速度: {p50:.2f} km/h")

for i, seg in enumerate(segments[:5]):
    formula_speed = explainer.simple_speed_formula(seg, p50)
    model_speed = predictor._model.predict_speed(seg, effort_factor=1.0)
    print(f"段{i+1}: 公式={formula_speed:.2f} km/h, 模型={model_speed:.2f} km/h, 差异={abs(formula_speed-model_speed):.2f}")

print("\n=== 测试完成 ===")