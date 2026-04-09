"""
Trail Race Predictor V1.2 - Report Generator

报告生成器 - 为预测结果生成 HTML 和 TXT 格式的报告
"""

import sys
from pathlib import Path
from typing import List, Optional

# 导入类型定义
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import PredictionResult, SegmentPrediction


class ReportGenerator:
    """预测报告生成器"""

    def __init__(
        self,
        prediction_result: PredictionResult = None,
        gpx_name: str = None,
        fit_names: List[str] = None
    ):
        """
        Args:
            prediction_result: 预测结果对象
            gpx_name: GPX 文件名
            fit_names: FIT 文件名列表
        """
        self.result = prediction_result
        self.gpx_name = gpx_name or "未知赛道"
        self.fit_names = fit_names or []

    def generate_html_report(self, output_path: str) -> str:
        """
        生成 HTML 报告

        Args:
            output_path: 输出文件路径

        Returns:
            生成的文件路径
        """
        if not self.result:
            raise ValueError("没有预测结果可供生成报告")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html_content()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def generate_txt_report(self) -> str:
        """
        生成 TXT 格式的报告

        Returns:
            TXT 格式的报告内容
        """
        if not self.result:
            return ""

        lines = []

        # 标题
        lines.append("=" * 70)
        lines.append("  越野赛预测报告 - Trail Race Prediction Report")
        lines.append("=" * 70)
        lines.append("")

        # 预测结果
        lines.append("【预测结果】")
        lines.append(f"  完赛时间: {self.result.total_time_hm}")
        lines.append(f"  平均配速: {self.result.pace_min_km:.1f} min/km")
        lines.append(f"  平均速度: {self.result.speed_kmh:.2f} km/h")
        lines.append(f"  努力程度: {self.result.effort_level}")
        lines.append("")

        # 赛道信息
        lines.append("【赛道信息】")
        lines.append(f"  赛道: {self.gpx_name}")
        lines.append(f"  总距离: {self.result.total_distance_km:.2f} km")
        lines.append(f"  总爬升: {self.result.total_ascent_m:.0f} m")
        lines.append(f"  总下降: {self.result.total_descent_m:.0f} m")
        lines.append(f"  爬升密度: {self.result.elevation_density:.1f} m/km")

        if self.result.elevation_density > 100:
            difficulty = "极难"
        elif self.result.elevation_density > 70:
            difficulty = "困难"
        elif self.result.elevation_density > 40:
            difficulty = "中等"
        else:
            difficulty = "轻松"
        lines.append(f"  难度等级: {difficulty}")
        lines.append("")

        # 训练记录
        if self.fit_names:
            lines.append("【训练记录】")
            for name in self.fit_names[:10]:
                lines.append(f"  - {name}")
            if len(self.fit_names) > 10:
                lines.append(f"  ... 等 {len(self.fit_names)} 个文件")
            lines.append("")

        # 分段配速表
        lines.append("【分段配速】")
        lines.append("-" * 90)
        lines.append(f"{'位置':^12} {'本段距离':^10} {'距离km':^8} {'爬升m':^8} {'下降m':^8} {'本段时间':^10} {'累计时间':^10} {'平均速度':^12}")
        lines.append("-" * 90)

        segments = self.result.segments
        total_distance = self.result.total_distance_km
        last_seg = segments[-1] if segments else None

        # 收集 CP 点（去重）
        cp_points = []
        seen_cps = set()
        for seg in segments:
            cp_name = getattr(seg, "cp_name", "")
            if cp_name and cp_name not in seen_cps:
                cp_points.append({
                    "name": cp_name,
                    "end_km": seg.end_km,
                    "cumulative_time_min": seg.cumulative_time_min,
                })
                seen_cps.add(cp_name)

        use_cp = len(cp_points) > 0

        # 构建显示行
        display_rows = []

        # 起点
        display_rows.append({
            "name": "起点",
            "end_km": 0,
            "cumulative_time_min": 0
        })

        # 中间行：CP 点或 5km 间隔
        if use_cp:
            for cp in cp_points:
                display_rows.append({
                    "name": cp["name"],
                    "end_km": cp["end_km"],
                    "cumulative_time_min": cp["cumulative_time_min"]
                })
        else:
            for mark_km in range(5, int(total_distance) + 1, 5):
                closest_seg = min(segments, key=lambda s: abs(s.end_km - mark_km))
                display_rows.append({
                    "name": f"{mark_km}km",
                    "end_km": closest_seg.end_km,
                    "cumulative_time_min": closest_seg.cumulative_time_min
                })

        # 终点
        display_rows.append({
            "name": "终点",
            "end_km": last_seg.end_km if last_seg else 0,
            "cumulative_time_min": self.result.total_time_min
        })

        # 生成表格行
        for i, curr in enumerate(display_rows):
            prev = display_rows[i - 1] if i > 0 else None

            segment_dist = curr["end_km"] - (prev["end_km"] if prev else 0)

            segment_ascent = 0
            segment_descent = 0
            segment_time = 0

            if prev:
                prev_km = prev["end_km"]
                curr_km = curr["end_km"]
                for seg in segments:
                    if prev_km < seg.end_km <= curr_km:
                        segment_ascent += getattr(seg, "ascent_m", 0)
                        segment_descent += getattr(seg, "descent_m", 0)
                        segment_time += seg.predicted_time_min

            avg_speed = segment_dist / (segment_time / 60) if segment_time > 0 else 0

            if i == 0:
                name_display = "🏁 起点"
                lines.append(
                    f"{name_display:^12} "
                    f"{'-':^10} "
                    f"{curr['end_km']:^8.1f} "
                    f"{'-':^8} "
                    f"{'-':^8} "
                    f"{'-':^10} "
                    f"{'-':^10} "
                    f"{'-':^12}"
                )
            else:
                name_display = curr["name"]
                if name_display == "终点":
                    name_display = "🏃 终点"
                else:
                    name_display = "📍 " + name_display

                time_str = self._format_time(segment_time)
                cum_time_str = self._format_time(curr["cumulative_time_min"])
                speed_str = f"{avg_speed:.1f} km/h" if avg_speed > 0 else "-"

                lines.append(
                    f"{name_display:^12} "
                    f"{segment_dist:^10.1f} "
                    f"{curr['end_km']:^8.1f} "
                    f"{segment_ascent:^8.0f} "
                    f"{segment_descent:^8.0f} "
                    f"{time_str:^10} "
                    f"{cum_time_str:^10} "
                    f"{speed_str:^12}"
                )

        lines.append("-" * 90)

        if use_cp:
            lines.append(f"\n📍 按 CP 点分段 (共 {len(cp_points)} 个 CP 点)")
        else:
            lines.append("\n📍 按 5km 间隔分段 (GPX 无 CP 点数据)")
        lines.append("")

        # 战术建议
        advice_list = self._generate_advice_content()
        lines.append("【战术建议】")

        for advice in advice_list:
            lines.append(f"\n  {advice['title']}")
            for item in advice["items"]:
                lines.append(f"    • {item}")

        # 难度预警
        warnings = self._generate_warnings()
        if warnings:
            lines.append("\n【难度预警】")
            for w in warnings:
                lines.append(f"  ⚠️ {w}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  Generated by Trail Race Predictor V1.2 | LightGBM Machine Learning")
        lines.append("  预测基于你的历史训练数据，仅供参考")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _format_time(self, minutes: float) -> str:
        """格式化时间"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours}:{mins:02d}:{secs:02d}"

    # ------------------------------------------------------------------
    # HTML 报告辅助方法
    # ------------------------------------------------------------------

    def _get_css_styles(self) -> str:
        """获取 CSS 样式"""
        return """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.8; font-size: 14px; }
        .section { padding: 30px; border-bottom: 1px solid #eee; }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
            display: flex;
            align-items: center;
        }
        .section-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #FF4B4B;
            margin-right: 10px;
            border-radius: 2px;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .column {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }
        .column-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .big-metric { text-align: center; padding: 20px 0; }
        .metric-value-large { font-size: 48px; font-weight: bold; color: #FF4B4B; }
        .metric-label-large { font-size: 14px; color: #666; margin-top: 5px; }
        .mini-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        .mini-metric {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
        }
        .mini-label { display: block; font-size: 11px; color: #666; margin-bottom: 5px; }
        .mini-value { display: block; font-size: 14px; font-weight: bold; color: #333; }
        .route-mini-info {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 15px;
            background: white;
            border-radius: 8px;
        }
        .info-label { color: #666; font-size: 14px; }
        .info-value { font-weight: bold; color: #333; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px; text-align: center; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #333; }
        tr:hover { background: #f8f9fa; }
        .difficulty-easy { color: #28a745; }
        .difficulty-moderate { color: #ffc107; }
        .difficulty-hard { color: #fd7e14; }
        .difficulty-extreme { color: #dc3545; font-weight: bold; }
        .advice-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .advice-box h4 { color: #1565c0; margin-bottom: 10px; }
        .advice-box ul { list-style: none; padding-left: 0; }
        .advice-box li { padding: 8px 0; border-bottom: 1px dashed rgba(0,0,0,0.1); }
        .advice-box li:last-child { border-bottom: none; }
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 8px 8px 0;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            background: #f8f9fa;
        }
        @media (max-width: 600px) {
            .two-column { grid-template-columns: 1fr; }
            .mini-metrics { grid-template-columns: 1fr; }
        }
    </style>"""

    def _generate_header(self) -> str:
        """生成报告头部"""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""
        <div class="header">
            <h1>🏔️ 越野赛预测报告</h1>
            <p class="subtitle">Trail Race Performance Prediction</p>
            <p class="subtitle">生成时间: {now}</p>
        </div>"""

    def _generate_summary_and_route(self) -> str:
        """生成预测内容和赛道分析的横向排布"""
        result = self.result

        if result.elevation_density > 100:
            difficulty = "极难"
            diff_class = "difficulty-extreme"
        elif result.elevation_density > 70:
            difficulty = "困难"
            diff_class = "difficulty-hard"
        elif result.elevation_density > 40:
            difficulty = "中等"
            diff_class = "difficulty-moderate"
        else:
            difficulty = "轻松"
            diff_class = "difficulty-easy"

        return f"""
        <div class="section">
            <div class="two-column">
                <div class="column">
                    <h3 class="column-title">📊 预测结果</h3>
                    <div class="big-metric">
                        <div class="metric-value-large">{result.total_time_hm}</div>
                        <div class="metric-label-large">预测完赛时间</div>
                    </div>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <span class="mini-label">平均配速</span>
                            <span class="mini-value">{result.pace_min_km:.1f} min/km</span>
                        </div>
                        <div class="mini-metric">
                            <span class="mini-label">平均速度</span>
                            <span class="mini-value">{result.speed_kmh:.2f} km/h</span>
                        </div>
                        <div class="mini-metric">
                            <span class="mini-label">努力程度</span>
                            <span class="mini-value">{result.effort_level}</span>
                        </div>
                    </div>
                </div>
                <div class="column">
                    <h3 class="column-title">🏔️ 赛道分析</h3>
                    <div class="route-mini-info">
                        <div class="info-row">
                            <span class="info-label">总距离</span>
                            <span class="info-value">{result.total_distance_km:.2f} km</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">总爬升</span>
                            <span class="info-value">{result.total_ascent_m:.0f} m</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">总下降</span>
                            <span class="info-value">{result.total_descent_m:.0f} m</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">爬升密度</span>
                            <span class="info-value">{result.elevation_density:.1f} m/km</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">难度等级</span>
                            <span class="info-value {diff_class}">{difficulty}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>"""

    def _generate_split_table(self) -> str:
        """生成分段配速表"""
        segments = self.result.segments
        if not segments:
            return '<div class="section"><h2 class="section-title">分段配速</h2><p>暂无分段数据</p></div>'

        total_distance = self.result.total_distance_km
        last_seg = segments[-1]

        # 收集 CP 点
        cp_points = []
        seen_cps = set()
        for seg in segments:
            cp_name = getattr(seg, "cp_name", "")
            if cp_name and cp_name not in seen_cps:
                cp_points.append({
                    "name": cp_name,
                    "end_km": seg.end_km,
                    "altitude_m": seg.altitude_m,
                    "cumulative_time_min": seg.cumulative_time_min,
                })
                seen_cps.add(cp_name)

        use_cp_mode = len(cp_points) > 0

        display_rows = [{
            "name": "起点",
            "end_km": 0,
            "altitude_m": segments[0].altitude_m if segments else 0,
            "cumulative_time_min": 0,
        }]

        if use_cp_mode:
            for cp in cp_points:
                display_rows.append({
                    "name": cp["name"],
                    "end_km": cp["end_km"],
                    "altitude_m": cp["altitude_m"],
                    "cumulative_time_min": cp["cumulative_time_min"],
                })
        else:
            for mark_km in range(5, int(total_distance) + 1, 5):
                closest_seg = min(segments, key=lambda s: abs(s.end_km - mark_km))
                display_rows.append({
                    "name": f"{mark_km}km",
                    "end_km": closest_seg.end_km,
                    "altitude_m": closest_seg.altitude_m,
                    "cumulative_time_min": closest_seg.cumulative_time_min,
                })

        display_rows.append({
            "name": "终点",
            "end_km": last_seg.end_km,
            "altitude_m": last_seg.altitude_m,
            "cumulative_time_min": last_seg.cumulative_time_min,
        })

        rows = []
        for i, curr in enumerate(display_rows):
            prev = display_rows[i - 1] if i > 0 else None

            segment_dist = curr["end_km"] - (prev["end_km"] if prev else 0)
            segment_ascent = 0
            segment_descent = 0
            segment_time = 0

            if prev:
                prev_km = prev["end_km"]
                curr_km = curr["end_km"]
                for seg in segments:
                    if prev_km < seg.end_km <= curr_km:
                        segment_ascent += getattr(seg, "ascent_m", 0)
                        segment_descent += getattr(seg, "descent_m", 0)
                        segment_time += seg.predicted_time_min

            avg_speed = segment_dist / (segment_time / 60) if segment_time > 0 else 0

            if i == 0:
                rows.append(f"""
                <tr>
                    <td>🏁 {curr["name"]}</td>
                    <td>-</td>
                    <td>{curr["end_km"]:.1f}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>""")
            else:
                name_display = curr["name"]
                if name_display == "终点":
                    name_display = "🏃 终点"
                else:
                    name_display = "📍 " + name_display

                rows.append(f"""
                <tr>
                    <td>{name_display}</td>
                    <td>{segment_dist:.1f}</td>
                    <td>{curr["end_km"]:.1f}</td>
                    <td>{segment_ascent:.0f}</td>
                    <td>{segment_descent:.0f}</td>
                    <td>{self._format_time(segment_time)}</td>
                    <td>{self._format_time(curr["cumulative_time_min"])}</td>
                    <td>{avg_speed:.1f} km/h</td>
                </tr>""")

        mode_note = "按 CP 点分段" if use_cp_mode else "按 5km 间隔分段 (GPX 无 CP 点)"
        cp_count_note = f" (共 {len(cp_points)} 个 CP 点)" if use_cp_mode else ""

        return f"""
        <div class="section">
            <h2 class="section-title">分段配速</h2>
            <p style="color:#666;font-size:13px;margin-bottom:12px;">📍 {mode_note}{cp_count_note}</p>
            <table>
                <thead>
                    <tr>
                        <th>位置</th>
                        <th>本段距离(km)</th>
                        <th>累计距离(km)</th>
                        <th>本段爬升(m)</th>
                        <th>本段下降(m)</th>
                        <th>本段时间</th>
                        <th>累计时间</th>
                        <th>平均速度</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>"""

    def _generate_tactical_advice(self) -> str:
        """生成战术建议"""
        advice_list = self._generate_advice_content()
        warnings = self._generate_warnings()

        advice_html = ""
        for advice in advice_list:
            advice_html += f"""
            <div class="advice-box">
                <h4>{advice["title"]}</h4>
                <ul>
                    {"".join(f"<li>{item}</li>" for item in advice["items"])}
                </ul>
            </div>"""

        warnings_html = ""
        if warnings:
            warnings_html = f"""
            <div class="warning-box">
                <strong>⚠️ 难度预警</strong>
                <ul>
                    {"".join(f"<li>{w}</li>" for w in warnings)}
                </ul>
            </div>"""

        return f"""
        <div class="section">
            <h2 class="section-title">战术建议</h2>
            {advice_html}
            {warnings_html}
        </div>"""

    def _generate_advice_content(self):
        """生成建议内容"""
        advice = []
        result = self.result

        # 配速策略
        pace_advice = {
            "title": "🎯 配速策略",
            "items": []
        }

        importance = result.feature_importance or {}

        grade_importance = importance.get("grade_pct", 0)
        dist_importance = importance.get("accumulated_distance_km", 0)

        if grade_importance > dist_importance:
            pace_advice["items"].append("坡度对你的速度影响最大，建议在陡坡段保守配速")
        else:
            pace_advice["items"].append("疲劳累积对你的影响更大，建议前半程控制速度")

        if result.elevation_density > 80:
            pace_advice["items"].append("赛道爬升密度极高，前 10km 建议配速比预测慢 10%")
        elif result.elevation_density > 50:
            pace_advice["items"].append("赛道有一定难度，建议均匀分配体力")

        if result.segments:
            climb_segments = [s for s in result.segments if s.grade_pct > 10]
            descent_segments = [s for s in result.segments if s.grade_pct < -10]

            if len(climb_segments) > len(descent_segments):
                pace_advice["items"].append("赛道以爬升为主，建议在下坡段追赶时间")
            elif len(descent_segments) > len(climb_segments):
                pace_advice["items"].append("赛道以下坡为主，注意控制下坡速度避免受伤")

        advice.append(pace_advice)

        # 补给建议
        supply_advice = {
            "title": "🏃 补给建议",
            "items": []
        }

        duration_hours = result.total_time_min / 60

        if duration_hours > 6:
            supply_advice["items"].append(f"预计耗时 {duration_hours:.1f} 小时，建议携带能量胶 {int(duration_hours)} 支")
            supply_advice["items"].append("每 45-60 分钟补充一次能量，每小时饮水 500-750ml")
        elif duration_hours > 3:
            supply_advice["items"].append(f"预计耗时 {duration_hours:.1f} 小时，建议携带能量胶 {int(duration_hours)} 支")
            supply_advice["items"].append("每 45 分钟补充一次能量")
        else:
            supply_advice["items"].append("短距离比赛，少量补给即可")

        advice.append(supply_advice)

        return advice

    def _generate_warnings(self):
        """生成难度预警"""
        warnings = []

        if self.result.segments:
            hard_segments = sorted(
                self.result.segments,
                key=lambda s: abs(s.grade_pct),
                reverse=True,
            )[:3]

            for seg in hard_segments:
                if abs(seg.grade_pct) > 25:
                    warnings.append(
                        f"第 {seg.segment_id} 段 ({seg.start_km:.1f}-{seg.end_km:.1f}km) "
                        f"坡度 {seg.grade_pct:+.1f}%，预计速度仅 {seg.predicted_speed_kmh:.1f} km/h"
                    )

        if self.result.warnings:
            warnings.extend(self.result.warnings)

        return warnings

    def _generate_footer(self) -> str:
        """生成页脚"""
        return f"""
        <div class="footer">
            <p>Generated by Trail Race Predictor V1.2 | LightGBM Machine Learning</p>
            <p>预测基于你的历史训练数据，仅供参考</p>
        </div>"""

    def _generate_html_content(self) -> str:
        """生成完整的 HTML 内容"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>越野赛预测报告 - Trail Race Prediction</title>
    {self._get_css_styles()}
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_summary_and_route()}
        {self._generate_split_table()}
        {self._generate_tactical_advice()}
        {self._generate_footer()}
    </div>
</body>
</html>"""
