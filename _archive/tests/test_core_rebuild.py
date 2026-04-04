#!/usr/bin/env python
"""Test script for core_rebuild.predictor module"""
import sys
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

sys.path.insert(0, '.')
print("Starting...", flush=True)

from core_rebuild.predictor import MLRacePredictor
from pathlib import Path
from datetime import datetime
import json

records_dir = Path("temp/records")
example_dir = Path("example")

# 使用records目录下所有FIT文件
fit_files = list(records_dir.glob("*.fit"))
print(f"Found {len(fit_files)} FIT files in records directory", flush=True)

gpx_file = example_dir / "2025黄岩九峰大师赛最终版.gpx"
print(f"GPX: {gpx_file.exists()}", flush=True)

if not fit_files:
    print("ERROR: No FIT files")
    sys.exit(1)
if not gpx_file.exists():
    print("ERROR: GPX not found")
    sys.exit(1)

print("\n=== Training ===", flush=True)
predictor = MLRacePredictor()
success = predictor.train_from_files(fit_files)
print(f"Training: {success}", flush=True)

if not success:
    print("ERROR: Training failed")
    sys.exit(1)

print("\n=== Predicting ===", flush=True)
result = predictor.predict_race(gpx_file, effort_factor=1.0)

print(f"\nTotal Distance: {result['total_distance_km']} km")
print(f"Predicted Time: {result['predicted_time_hm']} ({result['predicted_time_hours']} hours)")
print(f"Predicted Pace: {result['predicted_pace_min_km']} min/km")
print(f"Model: {result['model_type']}")

# =============================================================================
# Generate Report
# =============================================================================
print("\n=== Generating Report ===", flush=True)

# 创建输出目录
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

# 获取文件名
gpx_name = gpx_file.name
fit_names = [f.name for f in fit_files]

# 生成HTML报告
html_report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

def format_time(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    s = int((minutes % 1) * 60)
    return f"{h}:{m:02d}:{s:02d}"

# 生成完整HTML报告
def generate_html_report(result, gpx_name, fit_names):
    route_info = result.get('route_info', {})
    segments = result.get('segment_predictions', [])
    training_stats = result.get('training_stats', {})
    feature_importance = result.get('feature_importance', {})

    total_dist = result['total_distance_km']
    total_time_min = result['predicted_time_min']
    total_time_hm = result['predicted_time_hm']
    pace_min_km = result['predicted_pace_min_km']
    speed_kmh = result.get('predicted_speed_kmh', 0)
    effort = result.get('effort_factor', 1.0)

    # 计算难度
    ascent = route_info.get('total_elevation_gain_m', 0)
    descent = route_info.get('total_elevation_loss_m', 0)
    elevation_density = ascent / total_dist if total_dist > 0 else 0

    if elevation_density > 100:
        difficulty = "极难"
    elif elevation_density > 70:
        difficulty = "困难"
    elif elevation_density > 40:
        difficulty = "中等"
    else:
        difficulty = "轻松"

    # 确定努力程度
    effort_levels = {0.8: "保守", 0.9: "稳健", 1.0: "标准", 1.1: "积极", 1.2: "激进"}
    effort_level = effort_levels.get(effort, f"自定义({effort})")

    # 生成CP点或5km分段
    cp_points = []
    seen_cps = set()
    for seg in segments:
        cp = seg.get('cp_name', '')
        if cp and cp not in seen_cps:
            cp_points.append({
                'name': cp,
                'end_km': seg['distance_km'] + (cp_points[-1]['end_km'] if cp_points else 0),
                'cumulative_time_min': seg['cumulative_time_min']
            })
            seen_cps.add(cp)

    use_cp_mode = len(cp_points) > 0

    # 构建显示行
    display_rows = []
    display_rows.append({'name': '起点', 'end_km': 0, 'cumulative_time_min': 0})

    if use_cp_mode:
        for cp in cp_points:
            display_rows.append({'name': cp['name'], 'end_km': cp['end_km'], 'cumulative_time_min': cp['cumulative_time_min']})
    else:
        # 计算累计距离用于5km里程碑
        accumulated_distances = []
        cum_dist = 0
        for seg in segments:
            cum_dist += seg['distance_km']
            accumulated_distances.append(cum_dist)

        for mark_km in range(5, int(total_dist) + 1, 5):
            # 找到第一个累计距离 >= mark_km 的段
            for j, acc_dist in enumerate(accumulated_distances):
                if acc_dist >= mark_km:
                    display_rows.append({
                        'name': f'{mark_km}km',
                        'end_km': accumulated_distances[j],
                        'cumulative_time_min': segments[j]['cumulative_time_min']
                    })
                    break

    if segments:
        last_end = sum(s['distance_km'] for s in segments)
    else:
        last_end = 0
    display_rows.append({'name': '终点', 'end_km': last_end, 'cumulative_time_min': total_time_min})

    # 计算每段数据 - 使用累计数据直接计算
    table_rows = []
    for i, curr in enumerate(display_rows):
        prev = display_rows[i - 1] if i > 0 else None
        seg_dist = curr['end_km'] - (prev['end_km'] if prev else 0)

        if i == 0:
            table_rows.append(f'''<tr>
                <td>🏁 起点</td>
                <td>-</td>
                <td>{curr['end_km']:.1f}</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>''')
        else:
            # 找到prev和curr之间的segments，计算累计数据
            prev_time = prev['cumulative_time_min']
            curr_time = curr['cumulative_time_min']
            seg_time = curr_time - prev_time

            # 计算该区间的爬升和下降 - 找到prev和curr位置之间的所有segment
            seg_ascent = 0
            seg_descent = 0

            # 找到prev_km到curr_km之间的所有segments
            # 使用累计距离来判断
            running_total = 0
            for seg in segments:
                seg_start = running_total
                seg_end = running_total + seg['distance_km']

                # 如果这个segment跨越了prev和curr之间的边界
                if not (seg_end <= prev['end_km'] or seg_start >= curr['end_km']):
                    # 计算重叠部分
                    overlap_start = max(seg_start, prev['end_km'])
                    overlap_end = min(seg_end, curr['end_km'])
                    if overlap_end > overlap_start:
                        ratio = (overlap_end - overlap_start) / seg['distance_km']
                        seg_ascent += seg.get('ascent_m', 0) * ratio
                        seg_descent += seg.get('descent_m', 0) * ratio

                running_total = seg_end

            avg_speed = seg_dist / (seg_time / 60) if seg_time > 0 else 0

            name_display = '🏃 终点' if curr['name'] == '终点' else '📍 ' + curr['name']
            table_rows.append(f'''<tr>
                <td>{name_display}</td>
                <td>{seg_dist:.1f}</td>
                <td>{curr['end_km']:.1f}</td>
                <td>{seg_ascent:.0f}</td>
                <td>{seg_descent:.0f}</td>
                <td>{format_time(seg_time)}</td>
                <td>{format_time(curr['cumulative_time_min'])}</td>
                <td>{avg_speed:.1f} km/h</td>
            </tr>''')

    mode_note = "按CP点分段" if use_cp_mode else "按5km间隔分段"
    cp_count = f" (共{len(cp_points)}个CP点)" if use_cp_mode else ""

    # 生成特征重要性
    fi_html = ""
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        fi_html += f'<div style="margin:5px 0;"><span>{k}</span><div style="background:#e0e0e0;height:10px;width:{v*10}px;display:inline-block;"></div><span>{v:.0f}</span></div>'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>越野赛预测报告 - Trail Race Prediction</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
        .file-info {{ background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px 20px; margin-top: 15px; font-size: 13px; }}
        .file-info p {{ margin: 4px 0; opacity: 0.9; }}
        .section {{ padding: 30px; border-bottom: 1px solid #eee; }}
        .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 20px; color: #333; display: flex; align-items: center; }}
        .section-title::before {{ content: ''; width: 4px; height: 20px; background: #FF4B4B; margin-right: 10px; border-radius: 2px; }}
        .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .column {{ padding: 20px; background: #f8f9fa; border-radius: 12px; }}
        .column-title {{ font-size: 16px; font-weight: bold; margin-bottom: 15px; color: #333; }}
        .big-metric {{ text-align: center; padding: 20px 0; }}
        .metric-value-large {{ font-size: 48px; font-weight: bold; color: #FF4B4B; }}
        .metric-label-large {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .mini-metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 15px; }}
        .mini-metric {{ text-align: center; padding: 10px; background: white; border-radius: 8px; }}
        .mini-label {{ display: block; font-size: 11px; color: #666; margin-bottom: 5px; }}
        .mini-value {{ display: block; font-size: 14px; font-weight: bold; color: #333; }}
        .route-mini-info {{ display: flex; flex-direction: column; gap: 10px; }}
        .info-row {{ display: flex; justify-content: space-between; padding: 10px 15px; background: white; border-radius: 8px; }}
        .info-label {{ color: #666; font-size: 14px; }}
        .info-value {{ font-weight: bold; color: #333; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px; text-align: center; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #333; }}
        tr:hover {{ background: #f8f9fa; }}
        .difficulty-easy {{ color: #28a745; }}
        .difficulty-moderate {{ color: #ffc107; }}
        .difficulty-hard {{ color: #fd7e14; }}
        .difficulty-extreme {{ color: #dc3545; font-weight: bold; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; background: #f8f9fa; }}
        @media (max-width: 600px) {{ .two-column {{ grid-template-columns: 1fr; }} .mini-metrics {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏔️ 越野赛预测报告</h1>
            <p class="subtitle">Trail Race Performance Prediction</p>
            <p class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        <div class="file-info">
            <p><strong>📁 赛道:</strong> {gpx_name}</p>
            <p><strong>📁 训练记录:</strong> {len(fit_names)} 个文件</p>
            <p style="opacity:0.7;font-size:11px;font-style:italic;">{", ".join(fit_names[:5])}{"..." if len(fit_names) > 5 else ""}</p>
        </div>
        <div class="section">
            <div class="two-column">
                <div class="column">
                    <h3 class="column-title">📊 预测结果</h3>
                    <div class="big-metric">
                        <div class="metric-value-large">{total_time_hm}</div>
                        <div class="metric-label-large">预测完赛时间</div>
                    </div>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <span class="mini-label">平均配速</span>
                            <span class="mini-value">{pace_min_km:.1f} min/km</span>
                        </div>
                        <div class="mini-metric">
                            <span class="mini-label">平均速度</span>
                            <span class="mini-value">{speed_kmh:.2f} km/h</span>
                        </div>
                        <div class="mini-metric">
                            <span class="mini-label">努力程度</span>
                            <span class="mini-value">{effort_level}</span>
                        </div>
                    </div>
                </div>
                <div class="column">
                    <h3 class="column-title">🏔️ 赛道分析</h3>
                    <div class="route-mini-info">
                        <div class="info-row">
                            <span class="info-label">总距离</span>
                            <span class="info-value">{total_dist:.2f} km</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">总爬升</span>
                            <span class="info-value">{ascent:.0f} m</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">总下降</span>
                            <span class="info-value">{descent:.0f} m</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">爬升密度</span>
                            <span class="info-value">{elevation_density:.1f} m/km</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">难度等级</span>
                            <span class="info-value">{difficulty}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="section">
            <h2 class="section-title">分段配速</h2>
            <p style="margin-bottom:15px;font-size:12px;color:#666;">📍 {mode_note}{cp_count}</p>
            <table>
                <thead>
                    <tr>
                        <th>位置</th>
                        <th>本段距离(km)</th>
                        <th>距离(km)</th>
                        <th>本段爬升(m)</th>
                        <th>本段下降(m)</th>
                        <th>本段时间</th>
                        <th>累计时间</th>
                        <th>平均速度</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        <div class="section">
            <h2 class="section-title">模型统计</h2>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div style="padding:10px;background:white;border-radius:8px;">
                    <div style="font-size:12px;color:#666;">训练文件数</div>
                    <div style="font-size:18px;font-weight:bold;">{training_stats.get('file_count', 0)}</div>
                </div>
                <div style="padding:10px;background:white;border-radius:8px;">
                    <div style="font-size:12px;color:#666;">训练段数</div>
                    <div style="font-size:18px;font-weight:bold;">{training_stats.get('segment_count', 0)}</div>
                </div>
                <div style="padding:10px;background:white;border-radius:8px;">
                    <div style="font-size:12px;color:#666;">平均速度</div>
                    <div style="font-size:18px;font-weight:bold;">{training_stats.get('avg_speed', 0):.1f} km/h</div>
                </div>
                <div style="padding:10px;background:white;border-radius:8px;">
                    <div style="font-size:12px;color:#666;">休息比例</div>
                    <div style="font-size:18px;font-weight:bold;">{training_stats.get('avg_rest_ratio', 0)*100:.1f}%</div>
                </div>
            </div>
            <h3 style="margin-top:20px;font-size:14px;color:#333;">特征重要性</h3>
            <div style="margin-top:10px;">{fi_html or "无"}</div>
        </div>
        <div class="footer">
            <p>Generated by Trail Race Predictor V1.2 | LightGBM Machine Learning</p>
            <p>预测基于你的历史训练数据，仅供参考</p>
        </div>
    </div>
</body>
</html>'''

    return html

# 生成并保存HTML报告
html_content = generate_html_report(result, gpx_name, fit_names)
with open(html_report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML Report: {html_report_path}")

# 生成TXT报告
txt_report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def generate_txt_report(result, gpx_name, fit_names):
    route_info = result.get('route_info', {})
    segments = result.get('segment_predictions', [])
    training_stats = result.get('training_stats', {})

    total_dist = result['total_distance_km']
    total_time_min = result['predicted_time_min']
    total_time_hm = result['predicted_time_hm']
    pace_min_km = result['predicted_pace_min_km']
    speed_kmh = result.get('predicted_speed_kmh', 0)
    effort = result.get('effort_factor', 1.0)

    ascent = route_info.get('total_elevation_gain_m', 0)
    descent = route_info.get('total_elevation_loss_m', 0)
    elevation_density = ascent / total_dist if total_dist > 0 else 0

    if elevation_density > 100:
        difficulty = "极难"
    elif elevation_density > 70:
        difficulty = "困难"
    elif elevation_density > 40:
        difficulty = "中等"
    else:
        difficulty = "轻松"

    effort_levels = {0.8: "保守", 0.9: "稳健", 1.0: "标准", 1.1: "积极", 1.2: "激进"}
    effort_level = effort_levels.get(effort, f"自定义({effort})")

    lines = []
    lines.append("=" * 70)
    lines.append("越野赛预测报告 - Trail Race Prediction Report")
    lines.append("=" * 70)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"赛道: {gpx_name}")
    lines.append(f"训练记录: {len(fit_names)} 个文件")
    lines.append(", ".join(fit_names))
    lines.append("")

    lines.append("【预测结果】")
    lines.append(f"  完赛时间: {total_time_hm}")
    lines.append(f"  平均配速: {pace_min_km:.1f} min/km")
    lines.append(f"  平均速度: {speed_kmh:.2f} km/h")
    lines.append(f"  努力程度: {effort_level}")
    lines.append("")

    lines.append("【赛道信息】")
    lines.append(f"  总距离: {total_dist:.2f} km")
    lines.append(f"  总爬升: {ascent:.0f} m")
    lines.append(f"  总下降: {descent:.0f} m")
    lines.append(f"  爬升密度: {elevation_density:.1f} m/km")
    lines.append(f"  难度等级: {difficulty}")
    lines.append("")

    lines.append("【分段配速】")
    lines.append("-" * 90)
    lines.append(f"{'位置':^12} {'本段距离':^10} {'距离km':^8} {'爬升m':^8} {'下降m':^8} {'本段时间':^10} {'累计时间':^10} {'平均速度':^12}")
    lines.append("-" * 90)

    # 构建显示行
    display_rows = []
    display_rows.append({'name': '起点', 'end_km': 0, 'cumulative_time_min': 0})

    # 计算累计距离用于5km里程碑
    accumulated_distances = []
    cum_dist = 0
    for seg in segments:
        cum_dist += seg['distance_km']
        accumulated_distances.append(cum_dist)

    for mark_km in range(5, int(total_dist) + 1, 5):
        # 找到第一个累计距离 >= mark_km 的段
        for j, acc_dist in enumerate(accumulated_distances):
            if acc_dist >= mark_km:
                display_rows.append({
                    'name': f'{mark_km}km',
                    'end_km': accumulated_distances[j],
                    'cumulative_time_min': segments[j]['cumulative_time_min']
                })
                break

    last_end = sum(s['distance_km'] for s in segments) if segments else 0
    display_rows.append({'name': '终点', 'end_km': last_end, 'cumulative_time_min': total_time_min})

    for i, curr in enumerate(display_rows):
        prev = display_rows[i - 1] if i > 0 else None
        seg_dist = curr['end_km'] - (prev['end_km'] if prev else 0)

        if i == 0:
            lines.append(f"{'🏁 起点':^12} {'-':^10} {curr['end_km']:^8.1f} {'-':^8} {'-':^8} {'-':^10} {'-':^10} {'-':^12}")
        else:
            # 计算累计时间差
            prev_time = prev['cumulative_time_min']
            curr_time = curr['cumulative_time_min']
            seg_time = curr_time - prev_time

            # 计算该区间的爬升和下降
            seg_ascent = 0
            seg_descent = 0

            # 使用累计距离来判断
            running_total = 0
            for seg in segments:
                seg_start = running_total
                seg_end = running_total + seg['distance_km']

                # 如果这个segment跨越了prev和curr之间的边界
                if not (seg_end <= prev['end_km'] or seg_start >= curr['end_km']):
                    # 计算重叠部分
                    overlap_start = max(seg_start, prev['end_km'])
                    overlap_end = min(seg_end, curr['end_km'])
                    if overlap_end > overlap_start:
                        ratio = (overlap_end - overlap_start) / seg['distance_km']
                        seg_ascent += seg.get('ascent_m', 0) * ratio
                        seg_descent += seg.get('descent_m', 0) * ratio

                running_total = seg_end

            avg_speed = seg_dist / (seg_time / 60) if seg_time > 0 else 0

            name_display = '🏃 终点' if curr['name'] == '终点' else '📍 ' + curr['name']
            lines.append(f"{name_display:^12} {seg_dist:^10.1f} {curr['end_km']:^8.1f} {seg_ascent:^8.0f} {seg_descent:^8.0f} {format_time(seg_time):^10} {format_time(curr['cumulative_time_min']):^10} {avg_speed:^10.1f} km/h")

    lines.append("-" * 90)
    lines.append(f"\n📍 按5km间隔分段 (GPX无CP点)")
    lines.append("")

    lines.append("【模型统计】")
    lines.append(f"  训练文件数: {training_stats.get('file_count', 0)}")
    lines.append(f"  训练段数: {training_stats.get('segment_count', 0)}")
    lines.append(f"  平均速度: {training_stats.get('avg_speed', 0):.1f} km/h")
    lines.append(f"  P50速度: {training_stats.get('p50_speed', 0):.2f} km/h")
    lines.append(f"  P90速度: {training_stats.get('p90_speed', 0):.2f} km/h")
    lines.append(f"  休息比例: {training_stats.get('avg_rest_ratio', 0)*100:.1f}%")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Generated by Trail Race Predictor V1.2 | LightGBM Machine Learning")
    lines.append("预测基于你的历史训练数据，仅供参考")
    lines.append("=" * 70)

    return "\n".join(lines)

txt_content = generate_txt_report(result, gpx_name, fit_names)
with open(txt_report_path, 'w', encoding='utf-8') as f:
    f.write(txt_content)

print(f"TXT Report: {txt_report_path}")

# 保存JSON结果
json_report_path = report_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(json_report_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"JSON Result: {json_report_path}")

print("\n=== Done ===")