"""
Trail Race Predictor - Orchestrator

MLRacePredictor: 训练模型 + 预测比赛成绩的主入口
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List

from .features import SegmentFeatures
from .model import LightGBMPredictor
from .extractor import FeatureExtractor
from .gpx_parser import GPXRouteParser
from ..utils import FilterConfig


# 地形类型和难度分类阈值（集中管理，方便调整）
_GRADE_TYPE = [
    (15,  '陡上坡'),
    (8,   '中上坡'),
    (3,   '缓上坡'),
    (-3,  '平地'),
    (-8,  '缓下坡'),
    (-15, '中下坡'),
]

_DIFFICULTY = [
    (30, 'extreme'),
    (20, 'hard'),
    (10, 'moderate'),
]


def _grade_type(grade: float) -> str:
    for threshold, label in _GRADE_TYPE:
        if grade > threshold:
            return label
    return '陡下坡'


def _difficulty(grade: float) -> str:
    for threshold, label in _DIFFICULTY:
        if abs(grade) > threshold:
            return label
    return 'easy'


class MLRacePredictor(GPXRouteParser):
    """基于机器学习的越野赛成绩预测器

    使用方式::

        predictor = MLRacePredictor()
        predictor.train_from_files(["run1.fit", "run2.fit"])
        result = predictor.predict_race("race.gpx", effort_factor=1.0)
    """

    def __init__(self):
        self._model: LightGBMPredictor | None = None
        self.training_stats: Dict = {}
        self.all_feature_importance: Dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_files(self, file_paths: List[str]) -> bool:
        """从 FIT / JSON 文件列表训练统一 ML 模型

        自动选取最大的 20 个文件，并先对 FIT 文件做 P99 坡度校准。
        """
        if not file_paths:
            print("  Error: No training files provided")
            return False

        valid = [p for p in (Path(f) for f in file_paths)
                 if p.is_file() and p.suffix.lower() in ('.fit', '.json')]
        if not valid:
            print("  Error: No valid FIT/JSON files found")
            return False

        # 选取最大的 20 个文件
        valid.sort(key=lambda p: p.stat().st_size, reverse=True)
        selected = valid[:20]
        print(f"\n  Received {len(file_paths)} files, using top {len(selected)} largest files")

        # P99 坡度校准
        fit_paths = [str(p) for p in selected if p.suffix.lower() == '.fit']
        calibrated = FilterConfig.calibrate_from_fit_files(fit_paths)
        original   = FilterConfig.FIT.copy()
        FilterConfig.FIT = calibrated
        print(f"    Calibrated max_grade_pct: {calibrated['max_grade_pct']:.0f}% "
              f"(default was {original['max_grade_pct']:.0f}%)")

        all_segments: List[SegmentFeatures] = []
        all_rest_ratios: List[float] = []
        file_count = 0

        try:
            for path in selected:
                try:
                    if path.suffix.lower() == '.fit':
                        segs, rest = FeatureExtractor.extract_from_fit(path)
                    else:
                        segs, rest = FeatureExtractor.extract_from_json(path)

                    if segs:
                        all_segments.extend(segs)
                        all_rest_ratios.append(rest)
                        file_count += 1
                        print(f"    {path.name}: {len(segs)} segments, rest_ratio={rest:.1%}")
                except Exception as e:
                    print(f"    Warning: Failed to process {path.name}: {e}")
        finally:
            FilterConfig.FIT = original  # always restore

        if len(all_segments) < 5:
            print(f"  Error: Only {len(all_segments)} segments extracted, need at least 5")
            return False

        print(f"\n  Total: {file_count} files, {len(all_segments)} segments")

        self._model = LightGBMPredictor()
        if not self._model.train(all_segments):
            return False

        self.all_feature_importance = self._model.get_feature_importance()

        speeds         = [s.speed_kmh for s in all_segments]
        avg_rest_ratio = max(0.03, statistics.mean(all_rest_ratios) if all_rest_ratios else 0.08)

        self.training_stats = {
            'file_count':              file_count,
            'segment_count':           len(all_segments),
            'avg_speed':               round(statistics.mean(speeds), 2),
            'p50_speed':               round(self._model.p50_speed, 2),
            'p90_speed':               round(self._model.p90_speed, 2),
            'effort_range':            round(self._model.p90_speed / self._model.p50_speed, 2)
                                        if self._model.p50_speed > 0 else 1.0,
            'avg_rest_ratio':          round(avg_rest_ratio, 3),
            'calibrated_max_grade_pct': calibrated['max_grade_pct'],
        }

        print(f"\n  Training Stats:")
        for k, v in self.training_stats.items():
            print(f"    {k}: {v}")

        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_race(self, gpx_path: str, effort_factor: float = 1.0) -> Dict:
        """预测比赛成绩

        Args:
            gpx_path: GPX 路线文件路径
            effort_factor: 努力程度系数
                1.0 = 平均水平 (P50)
                1.1-1.2 = 比赛状态 (接近 P90)
                0.8-0.9 = 保守策略

        Returns:
            包含预测时间、配速、分段详情的字典
        """
        if self._model is None:
            raise RuntimeError("No trained model. Call train_from_files() first.")

        segments, route_info = self.parse_gpx_route(gpx_path)

        segment_predictions = []
        prev_cumulative_km  = 0.0
        total_time_h        = 0.0

        for i, seg in enumerate(segments):
            seg_dist_km = seg.accumulated_distance_km - prev_cumulative_km
            speed       = self._model.predict_speed(seg, effort_factor)
            seg_time_h  = seg_dist_km / speed if speed > 0 else 0.0
            total_time_h += seg_time_h

            segment_predictions.append({
                'segment':              i + 1,
                'distance_km':          round(seg_dist_km, 2),
                'grade_pct':            round(seg.grade_pct, 1),
                'altitude_m':           round(seg.absolute_altitude_m),
                'predicted_speed_kmh':  round(speed, 2),
                'segment_time_min':     round(seg_time_h * 60, 1),
                'cumulative_time_min':  round(total_time_h * 60, 1),
                'ascent_m':             round(seg.segment_ascent_m),
                'descent_m':            round(seg.segment_descent_m),
                'cp_name':              seg.cp_name,
                'grade_type':           _grade_type(seg.grade_pct),
                'difficulty':           _difficulty(seg.grade_pct),
            })
            prev_cumulative_km = seg.accumulated_distance_km

        moving_time_min = total_time_h * 60
        avg_rest_ratio  = self.training_stats.get('avg_rest_ratio', 0.08)
        total_time_min  = moving_time_min / (1 - avg_rest_ratio)
        total_dist_km   = route_info['total_distance_km']

        return {
            'effort_factor':              effort_factor,
            'predicted_moving_time_min':  round(moving_time_min),
            'predicted_time_min':         round(total_time_min),
            'predicted_time_hours':       round(total_time_min / 60, 2),
            'predicted_time_hm':          self._format_time(total_time_min),
            'predicted_pace_min_km':      round(total_time_min / total_dist_km, 1)
                                          if total_dist_km > 0 else 0,
            'predicted_speed_kmh':        round(total_dist_km / (total_time_min / 60), 2)
                                          if total_time_min > 0 else 0,
            'rest_ratio_used':            avg_rest_ratio,
            'total_distance_km':          total_dist_km,
            'route_info':                 route_info,
            'training_stats':             self.training_stats,
            'feature_importance':         self.all_feature_importance,
            'segment_predictions':        segment_predictions,
            'model_type':                 'LightGBM Unified Model with Effort Factor',
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _format_time(minutes: float) -> str:
        h   = int(minutes // 60)
        m   = int(minutes % 60)
        s   = int((minutes % 1) * 60)
        return f"{h}:{m:02d}:{s:02d}"
