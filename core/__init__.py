"""
Trail Race Predictor - Core Module

核心算法模块，包含预测器、滤波器和工具函数
"""

from .predictor import MLRacePredictor, LightGBMPredictor, SegmentFeatures
from .utils import FilterConfig, ElevationFilter, GradeAnalyzer, apply_fit_filter, apply_gpx_filter
from .types import (
    EffortLevel,
    ValidationResult,
    TrainingResult,
    SegmentPrediction,
    PredictionResult,
    PerformanceAnalysis
)

__all__ = [
    # 预测器
    'MLRacePredictor',
    'LightGBMPredictor',
    'SegmentFeatures',
    # 滤波器
    'FilterConfig',
    'ElevationFilter',
    'GradeAnalyzer',
    'apply_fit_filter',
    'apply_gpx_filter',
    # 类型
    'EffortLevel',
    'ValidationResult',
    'TrainingResult',
    'SegmentPrediction',
    'PredictionResult',
    'PerformanceAnalysis',
]
