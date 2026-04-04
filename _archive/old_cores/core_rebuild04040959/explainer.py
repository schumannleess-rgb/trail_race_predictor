"""
Trail Race Predictor - 模型可解释性模块

独立模块，提供预测解释功能，不需要修改现有脚本
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# 特征元数据定义
FEATURE_METADATA = {
    'grade_pct': {
        'name_cn': '坡度',
        'name_en': 'Grade',
        'unit': '%',
        'range': (-45, 80),
        'description': '当前路段坡度，正值上坡，负值下坡',
        'impact': {'上坡': '减速', '下坡': '加速', '平地': '无影响'}
    },
    'rolling_grade_500m': {
        'name_cn': '500m平均坡度',
        'name_en': 'Rolling Grade',
        'unit': '%',
        'range': (-45, 80),
        'description': '过去500m的平均坡度，识别持续陡坡'
    },
    'accumulated_distance_km': {
        'name_cn': '累计距离',
        'name_en': 'Accumulated Distance',
        'unit': 'km',
        'range': (0, 200),
        'description': '已完成距离，反映疲劳累积'
    },
    'accumulated_ascent_m': {
        'name_cn': '累计爬升',
        'name_en': 'Accumulated Ascent',
        'unit': 'm',
        'range': (0, 10000),
        'description': '累计爬升高度，体力消耗指标'
    },
    'absolute_altitude_m': {
        'name_cn': '海拔',
        'name_en': 'Altitude',
        'unit': 'm',
        'range': (0, 5000),
        'description': '当前海拔，高海拔氧气稀薄'
    },
    'elevation_density': {
        'name_cn': '爬升密度',
        'name_en': 'Elevation Density',
        'unit': 'm/km',
        'range': (0, 200),
        'description': '累计爬升/累计距离，赛道难度指标'
    }
}


@dataclass
class PredictionExplanation:
    """单次预测的解释结果"""
    predicted_speed: float
    base_speed: float  # P50
    confidence: str  # 高/中/低
    confidence_reason: str
    contributions: Dict[str, float]  # 各因素贡献
    warnings: List[str]  # 警告信息
    rules_applied: List[str]  # 应用的决策规则


class ModelExplainer:
    """模型可解释性分析器"""

    def __init__(self, model, training_segments: List[Any] = None):
        """
        初始化解释器

        Args:
            model: LightGBMPredictor 或类似模型对象
            training_segments: 训练数据中的segments，用于相似路段查找
        """
        self.model = model
        self.training_segments = training_segments or []

        # 从模型获取能力边界
        self.p50_speed = getattr(model, 'p50_speed', 6.0)
        self.p90_speed = getattr(model, 'p90_speed', 9.0)
        self.max_training_distance = getattr(model, 'max_training_distance', 100)
        self.max_training_ascent = getattr(model, 'max_training_ascent', 5000)
        self.feature_importance = getattr(model, 'feature_importance', {})

    def explain_prediction(self, segment) -> PredictionExplanation:
        """
        解释单次速度预测

        Args:
            segment: SegmentFeatures 对象

        Returns:
            PredictionExplanation 解释结果
        """
        # 获取原始预测
        predicted_speed = self.model.predict_speed(segment, effort_factor=1.0)

        # 计算各特征贡献
        contributions = self._calculate_contributions(segment)

        # 判断置信度
        confidence, reason = self._calculate_confidence(segment)

        # 生成决策规则
        rules = self._generate_decision_rules(segment)

        # 生成警告
        warnings = self._generate_warnings(segment)

        return PredictionExplanation(
            predicted_speed=predicted_speed,
            base_speed=self.p50_speed,
            confidence=confidence,
            confidence_reason=reason,
            contributions=contributions,
            warnings=warnings,
            rules_applied=rules
        )

    def _calculate_contributions(self, segment) -> Dict[str, float]:
        """计算各特征对预测的贡献"""
        contributions = {}

        # 坡度贡献 (基于特征重要性估算)
        grade_importance = self.feature_importance.get('grade_pct', 500)
        grade_contrib = -abs(segment.grade_pct) * 0.05 * (grade_importance / 500)
        contributions['坡度调节'] = round(grade_contrib, 2)

        # 距离疲劳贡献
        dist_importance = self.feature_importance.get('accumulated_distance_km', 500)
        dist_contrib = -segment.accumulated_distance_km * 0.01 * (dist_importance / 500)
        contributions['距离疲劳'] = round(dist_contrib, 2)

        # 爬升贡献
        ascent_importance = self.feature_importance.get('accumulated_ascent_m', 400)
        ascent_contrib = -segment.accumulated_ascent_m * 0.0005 * (ascent_importance / 400)
        contributions['爬升消耗'] = round(ascent_contrib, 2)

        # 海拔贡献
        if segment.absolute_altitude_m > 500:
            alt_importance = self.feature_importance.get('absolute_altitude_m', 500)
            alt_contrib = -(segment.absolute_altitude_m - 500) / 500 * 0.2 * (alt_importance / 500)
            contributions['海拔影响'] = round(alt_contrib, 2)
        else:
            contributions['海拔影响'] = 0

        # 爬升密度贡献
        elev_importance = self.feature_importance.get('elevation_density', 300)
        elev_contrib = -segment.elevation_density * 0.01 * (elev_importance / 300)
        contributions['难度惩罚'] = round(elev_contrib, 2)

        return contributions

    def _calculate_confidence(self, segment) -> tuple:
        """判断预测置信度"""
        warnings = []

        # 检查坡度范围
        if abs(segment.grade_pct) > 50:
            return '低', f'坡度{segment.grade_pct:.0f}%超出训练范围'

        # 检查累计距离
        if segment.accumulated_distance_km > self.max_training_distance:
            return '低', f'累计距离{segment.accumulated_distance_km:.1f}km超出训练范围({self.max_training_distance:.0f}km)'

        # 检查累计爬升
        if segment.accumulated_ascent_m > self.max_training_ascent:
            return '中', f'累计爬升{segment.accumulated_ascent_m:.0f}m接近训练上限'

        # 检查极端坡度
        if abs(segment.grade_pct) > 30:
            return '中', f'坡度{segment.grade_pct:.0f}%较极端'

        return '高', '所有特征在训练数据范围内'

    def _generate_decision_rules(self, segment) -> List[str]:
        """生成应用的决策规则"""
        rules = []

        # 坡度规则
        if segment.grade_pct > 20:
            rules.append(f"[!] 陡上坡({segment.grade_pct:.0f}%) -> 强制降速")
        elif segment.grade_pct > 10:
            rules.append(f"[^] 中上坡({segment.grade_pct:.0f}%) -> 适度降速")
        elif segment.grade_pct < -20:
            rules.append(f"[v] 陡下坡({segment.grade_pct:.0f}%) -> 可加速")

        # 距离规则
        if segment.accumulated_distance_km > 20:
            rules.append(f"[T] 长距离({segment.accumulated_distance_km:.0f}km) -> 疲劳降速")
        if segment.accumulated_distance_km > 40:
            rules.append(f"[!] 极长距离 -> 严重疲劳惩罚")

        # 爬升密度规则
        if segment.elevation_density > 80:
            rules.append(f"[M] 高爬升密度({segment.elevation_density:.0f}) -> 难度惩罚")
        if segment.elevation_density > 120:
            rules.append(f"[!] 极高难度赛道 -> 最大惩罚")

        # 海拔规则
        if segment.absolute_altitude_m > 2000:
            rules.append(f"[*] 高海拔({segment.absolute_altitude_m:.0f}m) -> 缺氧降速")

        return rules

    def _generate_warnings(self, segment) -> List[str]:
        """生成警告信息"""
        warnings = []

        if segment.grade_pct > 35:
            warnings.append(f'坡度{segment.grade_pct:.0f}%非常陡，注意安全')

        if segment.accumulated_distance_km > self.max_training_distance * 0.8:
            warnings.append('接近训练数据的最长距离，预测可能不准')

        if segment.elevation_density > 100:
            warnings.append('爬升密度极高，赛道非常艰难')

        return warnings

    def explain_all_segments(self, segments) -> List[PredictionExplanation]:
        """解释所有segments的预测"""
        return [self.explain_prediction(seg) for seg in segments]

    def generate_summary_report(self, segments, predictions) -> str:
        """生成汇总解释报告"""
        total_dist = segments[-1].accumulated_distance_km if segments else 0
        total_ascent = segments[-1].accumulated_ascent_m if segments else 0

        # 统计置信度
        confidences = [self.explain_prediction(seg).confidence for seg in segments]
        high_count = confidences.count('高')
        mid_count = confidences.count('中')
        low_count = confidences.count('低')

        # 统计规则应用
        all_rules = []
        for seg in segments:
            all_rules.extend(self.explain_prediction(seg).rules_applied)

        # 去重统计
        rule_counts = {}
        for rule in all_rules:
            key = rule.split('>')[0].strip()
            rule_counts[key] = rule_counts.get(key, 0) + 1

        report = f"""
+==============================================================+
|                   预测可解释性汇总报告                        |
+==============================================================+
| 赛道概况:
|   * 总距离: {total_dist:.1f} km
|   * 总爬升: {total_ascent:.0f} m
|   * 平均爬升密度: {total_ascent/total_dist:.1f} m/km
+==============================================================+
| 预测置信度统计:
|   * 高: {high_count}段 ({high_count/len(segments)*100:.0f}%)
|   * 中: {mid_count}段 ({mid_count/len(segments)*100:.0f}%)
|   * 低: {low_count}段 ({low_count/len(segments)*100:.0f}%)
+==============================================================+
| 主要决策因素 (出现次数):
"""

        for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"\n|   * {rule}: {count}次"

        report += f"""
+==============================================================+
| 模型能力:
|   * P50 (平均能力): {self.p50_speed:.2f} km/h
|   * P90 (极限能力): {self.p90_speed:.2f} km/h
+==============================================================+
"""

        return report


def simple_speed_formula(segment, p50_speed: float) -> float:
    """
    简化速度预测公式 - 不依赖模型

    公式: 速度 = P50 - 坡度惩罚 - 距离疲劳 - 爬升惩罚 - 海拔惩罚

    Args:
        segment: SegmentFeatures对象
        p50_speed: P50速度 (km/h)

    Returns:
        预测速度 (km/h)
    """
    # 坡度惩罚: 每1%坡度减0.08 km/h
    grade_penalty = abs(segment.grade_pct) * 0.08

    # 距离疲劳: 每公里减0.015 km/h
    fatigue_penalty = segment.accumulated_distance_km * 0.015

    # 爬升惩罚: 每100m减0.05 km/h
    ascent_penalty = segment.accumulated_ascent_m * 0.0005

    # 海拔惩罚: 超过500m后每500m减0.2 km/h
    altitude_penalty = max(0, (segment.absolute_altitude_m - 500) / 500) * 0.2 if segment.absolute_altitude_m > 500 else 0

    # 难度惩罚
    difficulty_penalty = segment.elevation_density * 0.01

    # 计算最终速度 (不低于1.5 km/h)
    predicted = p50_speed - grade_penalty - fatigue_penalty - ascent_penalty - altitude_penalty - difficulty_penalty

    return max(1.5, predicted)


def print_prediction_explanation(explanation: PredictionExplanation) -> None:
    """打印单次预测解释"""
    print(f"""
+==============================================================+
|                  单次速度预测解释                             |
+==============================================================+
| 预测速度: {explanation.predicted_speed:.2f} km/h
| 基础速度: {explanation.base_speed:.2f} km/h (P50)
| 置信度: {explanation.confidence} - {explanation.confidence_reason}
+==============================================================+
| 速度调节因素:
""")

    for key, value in explanation.contributions.items():
        sign = '+' if value > 0 else ''
        print(f"|   * {key}: {sign}{value:.2f} km/h")

    print("+==============================================================+")

    if explanation.rules_applied:
        print("| 决策规则:")
        for rule in explanation.rules_applied:
            print(f"|   {rule}")

    if explanation.warnings:
        print("| 警告:")
        for w in explanation.warnings:
            print(f"|   ! {w}")

    print("+==============================================================+")


if __name__ == '__main__':
    # 测试代码
    print("Model Explainer Module Loaded")
    print(f"Available features: {list(FEATURE_METADATA.keys())}")