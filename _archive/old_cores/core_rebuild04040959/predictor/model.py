"""
Trail Race Predictor - ML Model

LightGBM 预测器，包含训练、回退模型和速度预测
"""

import statistics
import numpy as np
from typing import List, Dict

from .features import SegmentFeatures


class LightGBMPredictor:
    """LightGBM 速度预测器"""

    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.feature_importance: Dict[str, float] = {}

        # 外推检测边界
        self.max_training_distance = 0.0
        self.max_training_ascent = 0.0

        # 能力边界
        self.p50_speed = 0.0   # 平均能力
        self.p90_speed = 0.0   # 极限能力

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, segments: List[SegmentFeatures]) -> bool:
        """训练 LightGBM 模型，数据不足时回退到简化模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  LightGBM not available, using fallback model")
            return self._train_fallback(segments)

        if len(segments) < 10:
            print("  Not enough data for ML training")
            return self._train_fallback(segments)

        X, y = self._build_matrices(segments)

        self.max_training_distance = max(s.accumulated_distance_km for s in segments)
        self.max_training_ascent = max(s.accumulated_ascent_m for s in segments)

        self.feature_names = [
            'grade_pct', 'rolling_grade_500m',
            'accumulated_distance_km', 'accumulated_ascent_m',
            'absolute_altitude_m', 'elevation_density',
        ]

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data': 1,
            'random_state': 42,
        }

        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params, train_data, num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        self.feature_importance = dict(zip(
            self.feature_names,
            [float(v) for v in self.model.feature_importance()],
        ))

        preds = self.model.predict(X)
        mae = np.mean(np.abs(preds - y))
        rmse = np.sqrt(np.mean((preds - y) ** 2))

        self._compute_capability_bounds(X, y)

        print(f"  Model trained: MAE={mae:.2f} km/h, RMSE={rmse:.2f} km/h")
        print(f"  Capability: P50={self.p50_speed:.2f} km/h, P90={self.p90_speed:.2f} km/h")
        self.is_trained = True
        return True

    def _train_fallback(self, segments: List[SegmentFeatures]) -> bool:
        """回退到按坡度分桶的简化均值模型"""
        self.is_trained = False
        self.feature_importance = {}

        speeds = [s.speed_kmh for s in segments]
        self.baseline_speed = statistics.mean(speeds) if speeds else 5.0

        climb_speeds   = [s.speed_kmh for s in segments if s.grade_pct > 5]
        flat_speeds    = [s.speed_kmh for s in segments if -5 <= s.grade_pct <= 5]
        descent_speeds = [s.speed_kmh for s in segments if s.grade_pct < -5]

        self.climb_speed   = statistics.mean(climb_speeds)   if climb_speeds   else self.baseline_speed * 0.6
        self.flat_speed    = statistics.mean(flat_speeds)    if flat_speeds    else self.baseline_speed
        self.descent_speed = statistics.mean(descent_speeds) if descent_speeds else self.baseline_speed * 1.1

        # Set p50/p90 from available data so callers don't get zeros
        self.p50_speed = np.percentile(speeds, 50) if speeds else self.baseline_speed
        self.p90_speed = np.percentile(speeds, 90) if speeds else self.baseline_speed

        print(f"  Fallback model: climb={self.climb_speed:.2f}, "
              f"flat={self.flat_speed:.2f}, descent={self.descent_speed:.2f} km/h")
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_speed(self, segment: SegmentFeatures, effort_factor: float = 1.0) -> float:
        """
        预测分段速度

        Args:
            segment: 分段特征
            effort_factor: 努力程度系数
                1.0 = 平均水平 (P50)
                1.1-1.2 = 比赛状态 (接近 P90)
                0.8-0.9 = 保守策略
        """
        if self.is_trained and self.model:
            return self._predict_lgbm(segment, effort_factor)
        return self._predict_fallback(segment, effort_factor)

    def get_feature_importance(self) -> Dict[str, float]:
        """返回特征重要性字典"""
        return self.feature_importance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrices(segments: List[SegmentFeatures]):
        X, y = [], []
        for seg in segments:
            X.append([
                seg.grade_pct,
                seg.rolling_grade_500m,
                seg.accumulated_distance_km,
                seg.accumulated_ascent_m,
                seg.absolute_altitude_m,
                seg.elevation_density,
            ])
            y.append(seg.speed_kmh)
        return np.array(X), np.array(y)

    def _compute_capability_bounds(self, X: np.ndarray, y: np.ndarray):
        """从平路段计算 P50/P90 能力边界"""
        flat_speeds = [y[i] for i in range(len(y)) if -5 <= X[i][0] <= 5]
        src = flat_speeds if flat_speeds else list(y)
        self.p50_speed = float(np.percentile(src, 50))
        self.p90_speed = float(np.percentile(src, 90))

    def _predict_lgbm(self, segment: SegmentFeatures, effort_factor: float) -> float:
        features = [[
            segment.grade_pct,
            segment.rolling_grade_500m,
            segment.accumulated_distance_km,
            segment.accumulated_ascent_m,
            segment.absolute_altitude_m,
            segment.elevation_density,
        ]]
        speed = float(self.model.predict(features)[0]) * effort_factor

        # 外推惩罚
        if segment.accumulated_distance_km > self.max_training_distance and self.max_training_distance > 0:
            ratio = segment.accumulated_distance_km / self.max_training_distance
            speed /= 1 + (ratio - 1) * 0.3

        if segment.accumulated_ascent_m > self.max_training_ascent and self.max_training_ascent > 0:
            ratio = segment.accumulated_ascent_m / self.max_training_ascent
            speed /= 1 + (ratio - 1) * 0.2

        # VAM 验证: 陡坡 (>15%) 垂直上升速度不超过 1000 m/h
        if segment.grade_pct > 15:
            vam = speed * 10 * segment.grade_pct
            if vam > 1000:
                speed /= vam / 1000

        max_speed = self.p90_speed * 1.1 if self.p90_speed > 0 else 15.0
        return float(max(1.0, min(max_speed, speed)))

    def _predict_fallback(self, segment: SegmentFeatures, effort_factor: float) -> float:
        if segment.grade_pct > 5:
            base = self.climb_speed
        elif segment.grade_pct < -5:
            base = self.descent_speed
        else:
            base = self.flat_speed
        return base * effort_factor
