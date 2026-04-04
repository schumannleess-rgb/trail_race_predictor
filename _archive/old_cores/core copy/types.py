"""
Trail Race Predictor - Type Definitions

数据类型定义，用于类型提示和数据结构规范化
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class EffortLevel(Enum):
    """努力程度枚举"""
    HIGH = "high"      # 高速/比赛强度
    MEDIUM = "medium"  # 中速/训练强度
    LOW = "low"        # 低速/恢复强度


class DifficultyLevel(Enum):
    """难度等级枚举"""
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    segment_count: int = 0
    file_count: int = 0

    def __bool__(self) -> bool:
        return self.success


@dataclass
class SegmentPrediction:
    """单段预测结果"""
    segment_id: int
    start_km: float
    end_km: float
    distance_km: float
    grade_pct: float
    altitude_m: float
    predicted_speed_kmh: float
    predicted_time_min: float
    cumulative_time_min: float
    difficulty_level: str = "moderate"  # 'easy', 'moderate', 'hard', 'extreme'
    grade_type: str = "平地"  # 地形类型
    ascent_m: float = 0  # 本段爬升
    descent_m: float = 0  # 本段下降
    cp_name: str = ""  # 最近的CP点名称

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'segment_id': self.segment_id,
            'start_km': round(self.start_km, 2),
            'end_km': round(self.end_km, 2),
            'distance_km': round(self.distance_km, 3),
            'grade_pct': round(self.grade_pct, 1),
            'altitude_m': round(self.altitude_m, 1),
            'predicted_speed_kmh': round(self.predicted_speed_kmh, 2),
            'predicted_time_min': round(self.predicted_time_min, 2),
            'cumulative_time_min': round(self.cumulative_time_min, 2),
            'difficulty_level': self.difficulty_level,
            'grade_type': self.grade_type
        }


@dataclass
class PredictionResult:
    """完整预测结果"""
    # 基本信息
    total_time_min: float
    total_time_hm: str
    pace_min_km: float
    speed_kmh: float

    # 赛道信息
    total_distance_km: float
    total_ascent_m: float
    total_descent_m: float
    elevation_density: float

    # 详细数据
    segments: List[SegmentPrediction]

    # 模型信息
    feature_importance: Dict[str, float]
    model_confidence: float
    effort_level: str

    # 元数据
    training_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'summary': {
                'total_time_min': round(self.total_time_min, 1),
                'total_time_hm': self.total_time_hm,
                'pace_min_km': round(self.pace_min_km, 1),
                'speed_kmh': round(self.speed_kmh, 2),
                'total_distance_km': round(self.total_distance_km, 2),
                'total_ascent_m': round(self.total_ascent_m, 1),
                'total_descent_m': round(self.total_descent_m, 1),
                'elevation_density': round(self.elevation_density, 1)
            },
            'effort_level': self.effort_level,
            'model_confidence': round(self.model_confidence, 2),
            'feature_importance': {k: round(v, 1) for k, v in self.feature_importance.items()},
            'segments': [s.to_dict() for s in self.segments],
            'training_stats': self.training_stats,
            'warnings': self.warnings,
            'error': self.error
        }

    def to_json(self, indent: int = 2) -> str:
        """导出为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class PerformanceAnalysis:
    """复盘分析结果"""
    # 时间对比
    predicted_time_min: float
    actual_time_min: float
    time_difference_min: float
    performance_ratio: float  # actual/predicted

    # 分段对比
    segment_comparison: List[Dict] = field(default_factory=list)

    # 分析结论
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    # 百分位排名 (与历史数据对比)
    percentile_rankings: Dict[str, float] = field(default_factory=dict)

    # 整体评价
    overall_rating: str = "average"  # 'excellent', 'good', 'average', 'below_average'
    summary: str = ""

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'time_comparison': {
                'predicted_time_min': round(self.predicted_time_min, 1),
                'actual_time_min': round(self.actual_time_min, 1),
                'time_difference_min': round(self.time_difference_min, 1),
                'performance_ratio': round(self.performance_ratio, 3)
            },
            'segment_comparison': self.segment_comparison,
            'analysis': {
                'strengths': self.strengths,
                'weaknesses': self.weaknesses,
                'overall_rating': self.overall_rating,
                'summary': self.summary
            },
            'percentile_rankings': {k: round(v, 1) for k, v in self.percentile_rankings.items()}
        }


@dataclass
class RouteInfo:
    """路线信息"""
    name: str
    total_distance_km: float
    total_ascent_m: float
    total_descent_m: float
    elevation_density: float
    segment_count: int
    checkpoint_count: int
    coordinate_bounds: Dict[str, float]  # {min_lat, max_lat, min_lon, max_lon}

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'total_distance_km': round(self.total_distance_km, 2),
            'total_ascent_m': round(self.total_ascent_m, 1),
            'total_descent_m': round(self.total_descent_m, 1),
            'elevation_density': round(self.elevation_density, 1),
            'segment_count': self.segment_count,
            'checkpoint_count': self.checkpoint_count,
            'coordinate_bounds': self.coordinate_bounds
        }


@dataclass
class FileInfo:
    """文件信息"""
    path: str
    size_bytes: int
    file_type: str  # 'gpx', 'fit', 'json'
    activity_date: Optional[str] = None
    distance_km: Optional[float] = None
    duration_min: Optional[float] = None
    avg_speed_kmh: Optional[float] = None
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'path': self.path,
            'size_bytes': self.size_bytes,
            'file_type': self.file_type,
            'activity_date': self.activity_date,
            'distance_km': round(self.distance_km, 2) if self.distance_km else None,
            'duration_min': round(self.duration_min, 1) if self.duration_min else None,
            'avg_speed_kmh': round(self.avg_speed_kmh, 2) if self.avg_speed_kmh else None,
            'is_valid': self.is_valid,
            'error_message': self.error_message
        }
