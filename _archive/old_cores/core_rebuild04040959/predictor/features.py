"""
Trail Race Predictor - Segment Features

分段特征数据结构及速度过滤常量
"""

from dataclasses import dataclass, field


# Speed filtering thresholds for rest detection
MOVING_THRESHOLD_KMH = 1.5    # Below this = rest/stop
GPS_ERROR_THRESHOLD_KMH = 18  # Above this = GPS artifact


@dataclass
class SegmentFeatures:
    """分段特征 — 用于训练和预测"""

    # 目标变量
    speed_kmh: float             # 目标: 平均速度 km/h

    # 地形特征
    grade_pct: float             # 当前坡度 %
    rolling_grade_500m: float    # 过去 500 米平均坡度

    # 疲劳/消耗特征
    accumulated_distance_km: float  # 累计距离
    accumulated_ascent_m: float     # 累计爬升

    # 生理/环境特征
    absolute_altitude_m: float   # 绝对海拔

    # 辅助特征
    elevation_density: float     # 爬升密度 m/km
    is_climbing: bool            # 是否爬升
    is_descending: bool          # 是否下降

    # 显示用字段 (可选)
    segment_ascent_m: float = 0  # 本段爬升
    segment_descent_m: float = 0 # 本段下降
    cp_name: str = ""            # 最近的 CP 点名称
