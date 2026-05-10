"""
Trail Race Predictor - CLI Entry Point

用法:
    python -m predictor.cli --gpx maps/race.gpx --records records/
    python -m predictor.cli --gpx maps/race.gpx --records records/ --effort 0.9 1.0 1.1
"""

import argparse
import json
from pathlib import Path

from .predictor import MLRacePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="越野赛成绩预测器 V1.2 - LightGBM 统一建模版")
    parser.add_argument('--gpx',     required=True,  help='GPX 路线文件路径')
    parser.add_argument('--records', required=True,  help='训练记录目录 (FIT / JSON)')
    parser.add_argument('--effort',  nargs='+', type=float, default=[0.9, 1.0, 1.1],
                        help='努力程度系数列表 (默认: 0.9 1.0 1.1)')
    parser.add_argument('--output',  default=None,   help='输出 JSON 文件路径 (可选)')
    args = parser.parse_args()

    gpx_path     = Path(args.gpx)
    records_dir  = Path(args.records)

    print("=" * 70)
    print("越野赛成绩预测器 V1.2 - LightGBM 统一建模版")
    print("=" * 70)

    predictor = MLRacePredictor()

    # --- Step 1: Collect training files ---
    print("\n[Step 1/2] 训练统一 ML 模型...")
    fit_files  = list(records_dir.glob('*.fit')) + list(records_dir.glob('*.FIT'))
    json_files = list(records_dir.glob('*.json'))
    training_files = [str(f) for f in fit_files + json_files]

    if not training_files:
        print("  Error: No training files found in", records_dir)
        return

    if not predictor.train_from_files(training_files):
        print("训练失败!")
        return

    # --- Step 2: Predict ---
    print("\n[Step 2/2] 预测比赛成绩:")
    print("=" * 70)

    results = {}
    for ef in args.effort:
        try:
            result = predictor.predict_race(str(gpx_path), effort_factor=ef)
            results[ef] = result
            print(f"\n【努力程度: {ef}x】")
            print(f"  预测时间:  {result['predicted_time_hm']} ({result['predicted_time_min']} 分钟)")
            print(f"  预测配速:  {result['predicted_pace_min_km']} min/km")
            print(f"  平均速度:  {result['predicted_speed_kmh']} km/h")
        except Exception as e:
            print(f"\n【努力程度: {ef}x】错误: {e}")

    # --- Save output ---
    output_path = args.output or str(gpx_path.parent / 'prediction_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': {
                'version':              'V1.2',
                'algorithm':            'LightGBM Unified Model',
                'features':             ['grade_pct', 'rolling_grade_500m',
                                         'accumulated_distance_km', 'accumulated_ascent_m',
                                         'absolute_altitude_m', 'elevation_density'],
                'includes_effort_factor': True,
                'supports_fit_files':   True,
            },
            'training_stats':    predictor.training_stats,
            'feature_importance': predictor.all_feature_importance,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
