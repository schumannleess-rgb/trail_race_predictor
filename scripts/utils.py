"""
越野赛数据处理工具集

提供统一的滤波算法，用于 GPX 赛道和 FIT 训练记录
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict


class FilterConfig:
    """滤波配置"""

    # GPX 赛道配置 (过滤 DEM 噪声)
    GPX = {
        'resample_spacing_m': 20,      # 20米重采样
        'window_size': 7,               # 覆盖 140 米地形
        'poly_order': 2,                # 保留山体起伏曲线
        'max_grade_pct': 45.0,           # 严格截断 ±45%
        'resample_required': True,      # 必须重采样
        'min_distance_m': 0.5           # 最小距离阈值 (防止重采样点间距过小)
    }

    # FIT 训练记录配置 (过滤传感器噪声)
    FIT = {
        'resample_spacing_m': None,     # 不重采样 (保持 1s/点)
        'window_size': 7,               # 覆盖 7-10 秒运动 (更好模拟坡度持续性)
        'poly_order': 2,                # 保留加速/减速趋势
        'max_grade_pct': 50.0,           # 宽松截断 ±50%
        'resample_required': False,     # 不需要重采样
        'min_distance_m': 0.5           # 最小距离阈值 (防止GPS漂移导致坡度爆炸)
    }


class ElevationFilter:
    """海拔滤波器 - Savitzky-Golay 滤波"""

    @staticmethod
    def smooth(elevations: List[float] or np.ndarray,
              config: dict) -> np.ndarray:
        """
        对海拔数据进行 SG 滤波

        Args:
            elevations: 海拔数组 (米)
            config: 滤波配置 (FilterConfig.GPX 或 FilterConfig.FIT)

        Returns:
            滤波后的海拔数组
        """
        elevations = np.array(elevations)

        if len(elevations) < config['window_size']:
            # 数据点太少，无法滤波
            return elevations

        window_size = config['window_size']
        poly_order = config['poly_order']

        # window_size 必须是奇数
        if window_size % 2 == 0:
            window_size += 1

        # 应用 Savitzky-Golay 滤波器
        smoothed = savgol_filter(elevations, window_size, polyorder=poly_order)

        return smoothed

    @staticmethod
    def clip_grade(grade_pct: float, max_grade: float = 45.0) -> float:
        """
        坡度截断

        Args:
            grade_pct: 原始坡度 (%)
            max_grade: 最大坡度限制 (%)

        Returns:
            截断后的坡度
        """
        return np.clip(grade_pct, -max_grade, max_grade)

    @staticmethod
    def calculate_grade(elevations: np.ndarray,
                       distances_m: np.ndarray,
                       config: dict = None) -> np.ndarray:
        """
        计算坡度并截断

        Args:
            elevations: 海拔数组 (米)
            distances_m: 距离数组 (米)
            config: 滤波配置

        Returns:
            坡度数组 (%)
        """
        if config is None:
            config = FilterConfig.GPX

        # 获取最小距离阈值 (防止GPS漂移导致坡度爆炸)
        min_distance = config.get('min_distance_m', 0.5)

        grades = []
        for i in range(len(elevations) - 1):
            dist_m = distances_m[i + 1] - distances_m[i]
            ele_m = elevations[i + 1] - elevations[i]

            # epsilon判断: 如果距离过小，坡度设为0
            if dist_m > min_distance:
                grade = (ele_m / dist_m) * 100
                grade = ElevationFilter.clip_grade(grade, config['max_grade_pct'])
            else:
                grade = 0

            grades.append(grade)

        # 最后一个点使用前一个点的坡度
        grades.append(grades[-1] if grades else 0)

        return np.array(grades)


class GradeAnalyzer:
    """坡度分析器"""

    @staticmethod
    def analyze_distribution(grades: np.ndarray,
                             spacing_m: float) -> Dict:
        """
        分析坡度分布

        Args:
            grades: 坡度数组 (%)
            spacing_m: 采样间距 (米)

        Returns:
            坡度分布统计
        """
        grade_ranges = {
            'extreme_descent': np.sum(grades < -20),
            'descent': np.sum((grades >= -20) & (grades < -10)),
            'gentle_descent': np.sum((grades >= -10) & (grades < -5)),
            'flat': np.sum((grades >= -5) & (grades < 5)),
            'gentle_climb': np.sum((grades >= 5) & (grades < 10)),
            'steep_climb': np.sum((grades >= 10) & (grades < 20)),
            'extreme_climb': np.sum(grades >= 20),
        }

        # 转换为距离和百分比
        distribution = {}
        total_points = len(grades)

        for range_name, count in grade_ranges.items():
            distance_km = count * spacing_m / 1000
            percentage = count / total_points * 100
            distribution[range_name] = {
                'distance_km': round(distance_km, 2),
                'percentage': round(percentage, 1)
            }

        return distribution

    @staticmethod
    def calculate_climbing_loss(original_gain_m: float,
                                 filtered_gain_m: float) -> Tuple[float, float]:
        """
        计算爬升损耗

        Args:
            original_gain_m: 原始爬升 (米)
            filtered_gain_m: 滤波后爬升 (米)

        Returns:
            (loss_m, loss_percentage)
        """
        loss_m = original_gain_m - filtered_gain_m
        loss_pct = (loss_m / original_gain_m * 100) if original_gain_m > 0 else 0

        return loss_m, loss_pct


def apply_fit_filter(elevations: List[float],
                      timestamps: List[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    对 FIT 训练记录应用滤波

    Args:
        elevations: 海拔数组 (米)
        timestamps: 时间戳数组 (秒)，可选

    Returns:
        (滤波后海拔, 滤波信息)
    """
    original_elevations = np.array(elevations)

    # 应用 FIT 配置的滤波
    smoothed_elevations = ElevationFilter.smooth(
        original_elevations,
        FilterConfig.FIT
    )

    # 计算统计信息
    noise_level = np.std(original_elevations - smoothed_elevations)
    max_deviation = np.max(np.abs(original_elevations - smoothed_elevations))

    info = {
        'points': len(elevations),
        'noise_std_m': round(noise_level, 2),
        'max_deviation_m': round(max_deviation, 2),
        'config': FilterConfig.FIT
    }

    return smoothed_elevations, info


def apply_gpx_filter(elevations: List[float],
                      distances_m: List[float],
                      original_gain_m: float = None) -> Tuple[np.ndarray, Dict]:
    """
    对 GPX 赛道应用滤波

    Args:
        elevations: 海拔数组 (米)
        distances_m: 距离数组 (米)
        original_gain_m: 原始爬升 (米)，用于计算损耗

    Returns:
        (滤波后海拔, 滤波信息)
    """
    original_elevations = np.array(elevations)

    # 应用 GPX 配置的滤波
    smoothed_elevations = ElevationFilter.smooth(
        original_elevations,
        FilterConfig.GPX
    )

    # 计算坡度
    grades = ElevationFilter.calculate_grade(
        smoothed_elevations,
        np.array(distances_m),
        FilterConfig.GPX
    )

    # 计算滤波后爬升
    filtered_gain_m = np.sum(np.maximum(
        grades[:-1] * (distances_m[1:] - distances_m[:-1]) / 100,
        0
    ))

    # 计算统计信息
    info = {
        'points': len(elevations),
        'original_gain_m': round(original_gain_m, 0) if original_gain_m else 0,
        'filtered_gain_m': round(filtered_gain_m, 0),
        'climbing_loss_m': round(original_gain_m - filtered_gain_m, 0) if original_gain_m else 0,
        'grade_range': f"{np.min(grades):.1f}% ~ {np.max(grades):.1f}%",
        'config': FilterConfig.GPX
    }

    return smoothed_elevations, info


# 导出配置供其他模块使用
__all__ = [
    'FilterConfig',
    'ElevationFilter',
    'GradeAnalyzer',
    'apply_fit_filter',
    'apply_gpx_filter'
]
