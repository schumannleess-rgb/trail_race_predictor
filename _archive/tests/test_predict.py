"""
测试脚本：使用 FIT 训练数据预测 GPX 路线
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    output_lines = []
    
    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)
    
    log("=" * 70)
    log("越野赛成绩预测器 - 测试脚本")
    log("=" * 70)
    log(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 路径配置
    base_dir = Path(__file__).parent
    records_dir = base_dir / 'temp' / 'records'
    routes_dir = base_dir / 'temp' / 'routes'
    output_dir = base_dir / 'temp' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 立即写入日志文件
    log_file = output_dir / 'test_log.txt'
    
    def save_log():
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    
    # 收集训练文件
    log("\n" + "=" * 70)
    log("[步骤 1] 收集训练数据")
    log("=" * 70)
    save_log()
    
    fit_files = list(records_dir.glob('*.fit')) + list(records_dir.glob('*.FIT'))
    fit_files = list(set(fit_files))
    
    log(f"\n训练文件目录: {records_dir}")
    log(f"发现 FIT 文件: {len(fit_files)} 个")
    
    training_files = []
    for f in sorted(fit_files):
        size_kb = f.stat().st_size / 1024
        log(f"  - {f.name} ({size_kb:.1f} KB)")
        training_files.append(str(f))
    save_log()
    
    if not training_files:
        log("\n错误: 没有找到训练文件!")
        save_log()
        return
    
    # 收集预测路线
    log("\n" + "=" * 70)
    log("[步骤 2] 收集预测路线")
    log("=" * 70)
    save_log()
    
    gpx_files = list(routes_dir.glob('*.gpx')) + list(routes_dir.glob('*.GPX'))
    gpx_files = list(set(gpx_files))
    
    log(f"\n路线文件目录: {routes_dir}")
    log(f"发现 GPX 文件: {len(gpx_files)} 个")
    
    route_files = []
    for f in sorted(gpx_files):
        size_kb = f.stat().st_size / 1024
        log(f"  - {f.name} ({size_kb:.1f} KB)")
        route_files.append(str(f))
    save_log()
    
    if not route_files:
        log("\n错误: 没有找到路线文件!")
        save_log()
        return
    
    # 导入预测器
    log("\n" + "=" * 70)
    log("[步骤 3] 导入预测器模块")
    log("=" * 70)
    save_log()
    
    try:
        from core.predictor import MLRacePredictor
        log("导入成功!")
    except Exception as e:
        log(f"导入失败: {e}")
        import traceback
        log(traceback.format_exc())
        save_log()
        return
    save_log()
    
    # 训练模型
    log("\n" + "=" * 70)
    log("[步骤 4] 训练 ML 模型")
    log("=" * 70)
    save_log()
    
    predictor = MLRacePredictor()
    
    log(f"\n开始训练，使用 {len(training_files)} 个文件...")
    try:
        train_success = predictor.train_from_files(training_files)
    except Exception as e:
        log(f"训练失败: {e}")
        import traceback
        log(traceback.format_exc())
        save_log()
        return
    save_log()
    
    if not train_success:
        log("\n训练失败!")
        save_log()
        return
    
    log("\n训练完成!")
    
    # 输出训练统计
    log("\n" + "-" * 50)
    log("训练统计信息:")
    log("-" * 50)
    for key, value in predictor.training_stats.items():
        if isinstance(value, float):
            log(f"  {key}: {value:.2f}")
        else:
            log(f"  {key}: {value}")
    
    # 输出特征重要性
    log("\n" + "-" * 50)
    log("特征重要性:")
    log("-" * 50)
    for feature, importance in sorted(predictor.all_feature_importance.items(), key=lambda x: -x[1]):
        log(f"  {feature}: {importance:.1f}")
    save_log()
    
    # 预测每条路线
    log("\n" + "=" * 70)
    log("[步骤 5] 预测比赛成绩")
    log("=" * 70)
    save_log()
    
    all_results = []
    
    for gpx_path in route_files:
        log(f"\n{'=' * 70}")
        log(f"预测路线: {Path(gpx_path).name}")
        log("=" * 70)
        save_log()
        
        effort_levels = [
            (0.85, "保守策略"),
            (0.90, "稳健策略"),
            (1.00, "平均水平 (P50)"),
            (1.05, "积极策略"),
            (1.10, "比赛状态 (接近P90)"),
        ]
        
        route_results = {
            'route_file': Path(gpx_path).name,
            'predictions': []
        }
        
        for effort_factor, effort_name in effort_levels:
            log(f"\n【{effort_name}】 effort_factor = {effort_factor}")
            log("-" * 50)
            save_log()
            
            try:
                result = predictor.predict_race(gpx_path, effort_factor)
                
                log(f"  移动时间: {result['predicted_moving_time_min']:.0f} 分钟")
                log(f"  总时间:   {result['predicted_time_hm']} ({result['predicted_time_min']:.0f} 分钟)")
                log(f"  平均配速: {result['predicted_pace_min_km']:.1f} min/km")
                log(f"  平均速度: {result['predicted_speed_kmh']:.2f} km/h")
                log(f"  休息比例: {result['rest_ratio_used']:.1%}")
                
                route_info = result['route_info']
                log(f"\n  路线信息:")
                log(f"    总距离: {route_info['total_distance_km']:.2f} km")
                log(f"    总爬升: {route_info['total_elevation_gain_m']:.0f} m")
                log(f"    总下降: {route_info['total_elevation_loss_m']:.0f} m")
                log(f"    爬升密度: {route_info['elevation_density']:.1f} m/km")
                log(f"    分段数: {route_info['segment_count']}")
                log(f"    CP点数: {route_info['checkpoint_count']}")
                
                prediction_data = {
                    'effort_factor': effort_factor,
                    'effort_name': effort_name,
                    'predicted_moving_time_min': result['predicted_moving_time_min'],
                    'predicted_time_min': result['predicted_time_min'],
                    'predicted_time_hm': result['predicted_time_hm'],
                    'predicted_pace_min_km': result['predicted_pace_min_km'],
                    'predicted_speed_kmh': result['predicted_speed_kmh'],
                    'rest_ratio_used': result['rest_ratio_used'],
                }
                route_results['predictions'].append(prediction_data)
                
            except Exception as e:
                log(f"  预测失败: {e}")
                import traceback
                log(traceback.format_exc())
            save_log()
        
        # 输出分段详情
        log("\n" + "-" * 50)
        log("分段预测详情 (effort_factor=1.0):")
        log("-" * 50)
        save_log()
        
        try:
            result = predictor.predict_race(gpx_path, 1.0)
            segments = result['segment_predictions']
            
            log(f"\n{'段':>3} {'距离':>6} {'坡度':>6} {'海拔':>6} {'速度':>6} {'时间':>6} {'累计':>8} {'地形':>6}")
            log(f"{'号':>3} {'(km)':>6} {'(%)':>6} {'(m)':>6} {'(km/h)':>6} {'(min)':>6} {'(min)':>8} {'类型':>6}")
            log("-" * 60)
            
            for seg in segments[:20]:
                log(f"{seg['segment']:>3} "
                      f"{seg['distance_km']:>6.2f} "
                      f"{seg['grade_pct']:>6.1f} "
                      f"{seg['altitude_m']:>6.0f} "
                      f"{seg['predicted_speed_kmh']:>6.2f} "
                      f"{seg['segment_time_min']:>6.1f} "
                      f"{seg['cumulative_time_min']:>8.1f} "
                      f"{seg['grade_type']:>6}")
            
            if len(segments) > 20:
                log(f"... 省略中间 {len(segments) - 20} 段 ...")
                for seg in segments[-5:]:
                    log(f"{seg['segment']:>3} "
                          f"{seg['distance_km']:>6.2f} "
                          f"{seg['grade_pct']:>6.1f} "
                          f"{seg['altitude_m']:>6.0f} "
                          f"{seg['predicted_speed_kmh']:>6.2f} "
                          f"{seg['segment_time_min']:>6.1f} "
                          f"{seg['cumulative_time_min']:>8.1f} "
                          f"{seg['grade_type']:>6}")
            
            route_results['segments'] = segments
            route_results['route_info'] = route_info
            
        except Exception as e:
            log(f"  获取分段详情失败: {e}")
        save_log()
        
        all_results.append(route_results)
    
    # 保存完整结果
    log("\n" + "=" * 70)
    log("[步骤 6] 保存结果")
    log("=" * 70)
    
    output_data = {
        'test_time': datetime.now().isoformat(),
        'training_files': [Path(f).name for f in training_files],
        'training_stats': predictor.training_stats,
        'feature_importance': predictor.all_feature_importance,
        'routes': all_results
    }
    
    output_file = output_dir / 'prediction_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    log(f"\n结果已保存: {output_file}")
    
    # 输出汇总
    log("\n" + "=" * 70)
    log("预测结果汇总")
    log("=" * 70)
    
    for route in all_results:
        log(f"\n路线: {route['route_file']}")
        log("-" * 50)
        for pred in route['predictions']:
            log(f"  {pred['effort_name']}: {pred['predicted_time_hm']} "
                  f"(配速 {pred['predicted_pace_min_km']:.1f} min/km)")
    
    log("\n" + "=" * 70)
    log("测试完成!")
    log("=" * 70)
    
    # 保存日志
    save_log()
    log(f"\n日志已保存: {log_file}")


if __name__ == '__main__':
    main()
