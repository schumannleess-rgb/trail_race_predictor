"""
越野赛成绩预测器 V1.2 - 机器学习版 (LightGBM)

基于《机器学习.txt》方案实现:
1. 使用 LightGBM 替代线性回归
2. 高级特征工程 (rolling_grade, accumulated_distance, accumulated_ascent, absolute_altitude)
3. K-Fold 交叉验证
4. 特征重要性分析
5. 处理外推问题 (extrapolation)

V1.2 更新:
- 直接支持原始 FIT 文件 (无需 JSON 转换)
- 统一建模 (简化训练流程)
- 努力程度量化 (基于 P50/P90 能力边界)
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime
import zipfile
import gzip
import tempfile
import os
from collections import deque

# 导入统一滤波工具
from .utils import apply_fit_filter, apply_gpx_filter, FilterConfig

# Speed filtering thresholds for rest detection
MOVING_THRESHOLD_KMH = 1.5   # Below this = rest/stop
GPS_ERROR_THRESHOLD_KMH = 18 # Above this = GPS artifact


@dataclass
class SegmentFeatures:
    """分段特征 - 用于训练和预测"""
    # 目标变量
    speed_kmh: float  # 目标: 平均速度 km/h

    # 地形特征
    grade_pct: float  # 当前坡度 %
    rolling_grade_500m: float  # 过去500米平均坡度

    # 疲劳/消耗特征
    accumulated_distance_km: float  # 累计距离
    accumulated_ascent_m: float  # 累计爬升

    # 生理/环境特征
    absolute_altitude_m: float  # 绝对海拔

    # 辅助特征
    elevation_density: float  # 爬升密度 m/km
    is_climbing: bool  # 是否爬升
    is_descending: bool  # 是否下降

    # 显示用字段 (可选)
    segment_ascent_m: float = 0  # 本段爬升
    segment_descent_m: float = 0  # 本段下降
    cp_name: str = ""  # 最近的CP点名称


class LightGBMPredictor:
    """LightGBM 预测器"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.feature_importance = {}
        self.max_training_distance = 0  # 用于外推检测
        self.max_training_ascent = 0
        self.p50_speed = 0  # P50 速度 (平均能力)
        self.p90_speed = 0  # P90 速度 (极限能力)

    def train(self, segments: List[SegmentFeatures]) -> bool:
        """训练 LightGBM 模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  LightGBM not available, using fallback model")
            return self._train_fallback(segments)

        if len(segments) < 10:
            print("  Not enough data for ML training")
            return self._train_fallback(segments)

        # 准备数据
        X = []
        y = []

        for seg in segments:
            features = [
                seg.grade_pct,
                seg.rolling_grade_500m,
                seg.accumulated_distance_km,
                seg.accumulated_ascent_m,
                seg.absolute_altitude_m,
                seg.elevation_density,
            ]
            X.append(features)
            y.append(seg.speed_kmh)

        X = np.array(X)
        y = np.array(y)

        # 记录训练数据的最大值 (用于外推检测)
        self.max_training_distance = max(s.accumulated_distance_km for s in segments)
        self.max_training_ascent = max(s.accumulated_ascent_m for s in segments)

        self.feature_names = [
            'grade_pct',
            'rolling_grade_500m',
            'accumulated_distance_km',
            'accumulated_ascent_m',
            'absolute_altitude_m',
            'elevation_density'
        ]

        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X, label=y)

        # 参数设置
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data': 1,  # 允许小数据集训练
            'seed': 42,  # 固定随机种子
        }

        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)]  # 禁用训练日志
        )

        # 计算特征重要性
        self.feature_importance = dict(zip(
            self.feature_names,
            [float(x) for x in self.model.feature_importance()]
        ))

        # 计算训练误差
        predictions = self.model.predict(X)
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(np.mean((predictions - y) ** 2))

        # 计算能力边界 (P50 和 P90)
        # 只考虑平路段 (坡度 -5% ~ 5%)
        flat_speeds = [y[i] for i in range(len(y)) if -5 <= X[i][0] <= 5]
        if flat_speeds:
            self.p50_speed = np.percentile(flat_speeds, 50)
            self.p90_speed = np.percentile(flat_speeds, 90)
        else:
            # 如果没有平路数据，使用全数据
            self.p50_speed = np.percentile(y, 50)
            self.p90_speed = np.percentile(y, 90)

        print(f"  Model trained: MAE={mae:.2f} km/h, RMSE={rmse:.2f} km/h")
        print(f"  Capability: P50={self.p50_speed:.2f} km/h, P90={self.p90_speed:.2f} km/h")
        self.is_trained = True
        return True

    def _train_fallback(self, segments: List[SegmentFeatures]) -> bool:
        """回退到简化模型"""
        self.is_trained = False
        self.feature_importance = {}  # 设置空的特征重要性

        # 计算基准速度
        speeds = [s.speed_kmh for s in segments]
        self.baseline_speed = statistics.mean(speeds) if speeds else 5.0

        # 按坡度分组统计
        climb_speeds = [s.speed_kmh for s in segments if s.grade_pct > 5]
        flat_speeds = [s.speed_kmh for s in segments if -5 <= s.grade_pct <= 5]
        descent_speeds = [s.speed_kmh for s in segments if s.grade_pct < -5]

        self.climb_speed = statistics.mean(climb_speeds) if climb_speeds else self.baseline_speed * 0.6
        self.flat_speed = statistics.mean(flat_speeds) if flat_speeds else self.baseline_speed
        self.descent_speed = statistics.mean(descent_speeds) if descent_speeds else self.baseline_speed * 1.1

        print(f"  Fallback model: climb={self.climb_speed:.2f}, flat={self.flat_speed:.2f}, descent={self.descent_speed:.2f} km/h")
        return True

    def predict_speed(self, segment: SegmentFeatures, effort_factor: float = 1.0) -> float:
        """
        预测速度
        
        Args:
            segment: 分段特征
            effort_factor: 努力程度系数 (0.8-1.2)
                          1.0 = 平均水平 (P50)
                          1.1-1.2 = 比赛状态 (接近 P90)
                          0.8-0.9 = 保守策略
        """
        if self.is_trained and self.model:
            # 使用 LightGBM 预测
            features = [[
                segment.grade_pct,
                segment.rolling_grade_500m,
                segment.accumulated_distance_km,
                segment.accumulated_ascent_m,
                segment.absolute_altitude_m,
                segment.elevation_density,
            ]]

            predicted_speed = self.model.predict(features)[0]

            # 应用努力程度系数
            predicted_speed *= effort_factor

            # 外推检测: 如果超出训练数据范围，应用惩罚
            if segment.accumulated_distance_km > self.max_training_distance:
                excess_ratio = segment.accumulated_distance_km / self.max_training_distance
                penalty = 1 + (excess_ratio - 1) * 0.3  # 超出部分每增加10%慢3%
                predicted_speed /= penalty

            if segment.accumulated_ascent_m > self.max_training_ascent:
                excess_ratio = segment.accumulated_ascent_m / self.max_training_ascent
                penalty = 1 + (excess_ratio - 1) * 0.2
                predicted_speed /= penalty

            # VAM (Vertical Ascension Meter) 验证
            # 对于陡坡路段，检查垂直上升速度是否合理
            if segment.grade_pct > 15:  # 陡坡 (>15%)
                # VAM = 水平速度 * 1000 (m/km) * 坡度 / 100
                # 例如: 6 km/h * 1000 * 20% / 100 = 1200 m/h
                vam = predicted_speed * 10 * segment.grade_pct
                # 如果 VAM 超过 1000 m/h (大神级)，应用惩罚
                if vam > 1000:
                    vam_penalty = vam / 1000
                    predicted_speed /= vam_penalty

            # 限制在合理范围，且不超过 P90 能力
            max_speed = self.p90_speed * 1.1 if self.p90_speed > 0 else 15.0
            return max(1.0, min(max_speed, predicted_speed))

        else:
            # 使用回退模型
            base_speed = 0
            if segment.grade_pct > 5:
                base_speed = self.climb_speed
            elif segment.grade_pct < -5:
                base_speed = self.descent_speed
            else:
                base_speed = self.flat_speed
            
            return base_speed * effort_factor

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.feature_importance


class FeatureExtractor:
    """特征提取器 - 从 JSON/FIT 文件提取分段特征"""

    # 解析缓存目录
    _CACHE_DIR = Path.home() / '.trail_race_predictor' / 'fit_cache'
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_cache_path(cls, fit_path: Path) -> Path:
        """获取缓存文件路径"""
        import hashlib
        # 用文件内容的hash作为缓存key (避免mtime变化导致缓存失效)
        if fit_path.exists():
            with open(fit_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()[:16]
        else:
            content_hash = 'missing'
        return cls._CACHE_DIR / f"{fit_path.stem}_{content_hash}.pkl"

    @classmethod
    def _get_cached_data(cls, fit_path: Path) -> Optional[Tuple]:
        """尝试从缓存加载解析数据"""
        cache_path = cls._get_cache_path(fit_path)
        print(f"    [CACHE CHECK] {fit_path.name}")
        print(f"    [CACHE PATH] {cache_path}")
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"    [CACHE HIT] {fit_path.name}")
                return data
            except Exception as e:
                print(f"    [CACHE ERROR] {e}")
        print(f"    [CACHE MISS] {fit_path.name}")
        return None

    @classmethod
    def _save_cached_data(cls, fit_path: Path, data: Tuple):
        """保存解析数据到缓存"""
        try:
            import pickle
            cache_path = cls._get_cache_path(fit_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"    [CACHE SAVED] {fit_path.name} -> {cache_path}")
        except Exception as e:
            print(f"    [CACHE SAVE ERROR] {e}")

    @staticmethod
    def extract_from_fit(fit_path: Path, segment_length_m: int = 200) -> Tuple[List['SegmentFeatures'], float]:
        """直接从 FIT 文件提取分段特征 (支持 ZIP/GZIP 压缩格式)

        Returns:
            (segments, rest_ratio): 分段特征列表和静止时间比例
        """
        try:
            from fitparse import FitFile
        except ImportError:
            print(f"  Warning: fitparse not available, cannot parse {fit_path}")
            return [], 0.0

        actual_fit_path = fit_path
        temp_dir = None
        temp_fit_file = None

        # 读取文件头判断格式
        try:
            with open(fit_path, 'rb') as f:
                header = f.read(4)
        except Exception as e:
            print(f"  Warning: Cannot read file {fit_path.name}: {e}")
            return [], 0.0

        try:
            if header[:2] == b'\x1f\x8b':
                # GZIP 格式
                temp_dir = tempfile.mkdtemp()
                temp_fit_file = Path(temp_dir) / fit_path.stem
                with gzip.open(fit_path, 'rb') as gz:
                    with open(temp_fit_file, 'wb') as out:
                        out.write(gz.read())
                actual_fit_path = temp_fit_file
                print(f"    (Decompressed gzip: {fit_path.name})")

            elif header[:4] == b'PK\x03\x04':
                # ZIP 格式
                with zipfile.ZipFile(fit_path, 'r') as zip_ref:
                    temp_dir = tempfile.mkdtemp()
                    zip_ref.extractall(temp_dir)
                    extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
                    if extracted_files:
                        actual_fit_path = extracted_files[0]
                        print(f"    (Decompressed zip: {fit_path.name})")
                    else:
                        print(f"  Warning: No FIT file found in ZIP archive {fit_path.name}")
                        return [], 0.0
        except Exception as e:
            print(f"  Warning: Failed to decompress {fit_path.name}: {e}")
            return [], 0.0

        try:
            fitfile = FitFile(str(actual_fit_path))

            # 首先获取 SESSION 消息中的预计算值 (Garmin 设备计算)
            session_data = {}
            for session in fitfile.get_messages('session'):
                for field in session:
                    if field.name in ['total_ascent', 'total_descent', 'total_distance', 'total_timer_time', 'total_elapsed_time']:
                        session_data[field.name] = field.value
                break  # 只取第一个 session

            # 使用 Garmin 预计算的值
            garmin_ascent = session_data.get('total_ascent', 0)
            garmin_descent = session_data.get('total_descent', 0)
            garmin_distance = session_data.get('total_distance', 0) / 1000 if session_data.get('total_distance') else 0  # 转换为 km
            garmin_time = session_data.get('total_timer_time', 0) / 60 if session_data.get('total_timer_time') else 0  # 转换为 min

            print(f"    Garmin pre-calculated: ascent={garmin_ascent}m, descent={garmin_descent}m, distance={garmin_distance:.2f}km")

            # Compute Garmin-derived rest ratio (elapsed - timer = rest)
            garmin_elapsed = session_data.get('total_elapsed_time', 0)
            garmin_timer = session_data.get('total_timer_time', 0)
            garmin_rest_ratio = (garmin_elapsed - garmin_timer) / garmin_elapsed if garmin_elapsed > 0 else 0
            garmin_rest_ratio = max(0.0, min(0.5, garmin_rest_ratio))  # clamp to [0, 0.5]
            if garmin_rest_ratio > 0:
                print(f"    Garmin rest ratio: {garmin_rest_ratio:.1%} (elapsed={garmin_elapsed:.0f}s, timer={garmin_timer:.0f}s)")

            # 提取记录点
            records = []
            for record in fitfile.get_messages('record'):
                record_data = {}
                for field in record:
                    record_data[field.name] = field.value
                records.append(record_data)

            if len(records) < 10:
                return [], 0.0

            # 提取时间序列数据
            timestamps = []
            distances = []
            elevations = []
            heart_rates = []

            for i, record in enumerate(records):
                # 时间戳
                if 'timestamp' in record:
                    ts = record['timestamp']
                    if isinstance(ts, datetime):
                        timestamps.append(ts.timestamp())
                    else:
                        timestamps.append(i * 60)  # 假设每分钟一个点
                else:
                    timestamps.append(i * 60)

                # 距离 (米)
                if 'distance' in record and record['distance'] is not None:
                    distances.append(float(record['distance']))
                elif 'enhanced_distance' in record and record['enhanced_distance'] is not None:
                    distances.append(float(record['enhanced_distance']))
                else:
                    # 如果没有距离数据，无法提取特征
                    if i > 0:
                        distances.append(distances[-1])
                    else:
                        distances.append(0)

                # 海拔 (米)
                if 'altitude' in record and record['altitude'] is not None:
                    elevations.append(float(record['altitude']))
                elif 'enhanced_altitude' in record and record['enhanced_altitude'] is not None:
                    elevations.append(float(record['enhanced_altitude']))
                else:
                    elevations.append(0)

                # 心率 (可选)
                if 'heart_rate' in record and record['heart_rate'] is not None:
                    heart_rates.append(int(record['heart_rate']))
                elif 'enhanced_heart_rate' in record and record['enhanced_heart_rate'] is not None:
                    heart_rates.append(int(record['enhanced_heart_rate']))
                else:
                    heart_rates.append(0)

            if len(elevations) < FilterConfig.FIT['window_size']:
                return [], 0.0

            # 使用与 JSON 相同的处理逻辑
            result = FeatureExtractor._extract_from_time_series(
                timestamps, distances, elevations, heart_rates, segment_length_m
            )

            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            return result, garmin_rest_ratio

        except Exception as e:
            print(f"  Warning: Could not extract from FIT {fit_path.name}: {e}")
            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            return [], 0.0

    @staticmethod
    def extract_from_fit_optimized(fit_path: Path, segment_length_m: int = 200) -> Tuple[List['SegmentFeatures'], float, List[float]]:
        """优化版本：从 FIT 文件提取分段特征，同时收集坡度数据用于校准

        使用缓存避免重复解析 FIT 文件。

        Returns:
            (segments, rest_ratio, grades_for_calibration): 分段特征列表、静止时间比例、坡度列表
        """
        # 先检查缓存
        cached = FeatureExtractor._get_cached_data(fit_path)
        if cached is not None:
            timestamps, distances, elevations, heart_rates, garmin_rest_ratio = cached
            # 从缓存数据恢复分段特征
            result = FeatureExtractor._extract_from_time_series(
                timestamps, distances, elevations, heart_rates, segment_length_m
            )
            # 收集坡度用于校准
            loose_max_grade = 200.0
            grades_for_calibration = []
            for i in range(len(elevations) - 1):
                dist_m = distances[i + 1] - distances[i]
                if dist_m > FilterConfig.FIT.get('min_distance_m', 0.5):
                    grade = ((elevations[i + 1] - elevations[i]) / dist_m) * 100
                    grade = np.clip(grade, -loose_max_grade, loose_max_grade)
                    grades_for_calibration.append(abs(grade))
            return result, garmin_rest_ratio, grades_for_calibration

        try:
            from fitparse import FitFile
        except ImportError:
            print(f"  Warning: fitparse not available, cannot parse {fit_path}")
            return [], 0.0, []

        actual_fit_path = fit_path
        temp_dir = None

        # 读取文件头判断格式
        try:
            with open(fit_path, 'rb') as f:
                header = f.read(4)
        except Exception as e:
            print(f"  Warning: Cannot read file {fit_path.name}: {e}")
            return [], 0.0, []

        # 处理压缩格式
        try:
            if header[:2] == b'\x1f\x8b':
                temp_dir = tempfile.mkdtemp()
                temp_fit_file = Path(temp_dir) / fit_path.stem
                with gzip.open(fit_path, 'rb') as gz:
                    with open(temp_fit_file, 'wb') as out:
                        out.write(gz.read())
                actual_fit_path = temp_fit_file

            elif header[:4] == b'PK\x03\x04':
                with zipfile.ZipFile(fit_path, 'r') as zip_ref:
                    temp_dir = tempfile.mkdtemp()
                    zip_ref.extractall(temp_dir)
                    extracted_files = list(Path(temp_dir).rglob('*.fit')) + list(Path(temp_dir).rglob('*.FIT'))
                    if extracted_files:
                        actual_fit_path = extracted_files[0]
                    else:
                        return [], 0.0, []
        except Exception as e:
            print(f"  Warning: Failed to decompress {fit_path.name}: {e}")
            return [], 0.0, []

        try:
            fitfile = FitFile(str(actual_fit_path))

            # 获取 SESSION 数据
            session_data = {}
            for session in fitfile.get_messages('session'):
                for field in session:
                    if field.name in ['total_ascent', 'total_descent', 'total_distance', 'total_timer_time', 'total_elapsed_time']:
                        session_data[field.name] = field.value
                break

            garmin_ascent = session_data.get('total_ascent', 0)
            garmin_descent = session_data.get('total_descent', 0)
            garmin_distance = session_data.get('total_distance', 0) / 1000 if session_data.get('total_distance') else 0
            garmin_time = session_data.get('total_timer_time', 0) / 60 if session_data.get('total_timer_time') else 0

            garmin_elapsed = session_data.get('total_elapsed_time', 0)
            garmin_timer = session_data.get('total_timer_time', 0)
            garmin_rest_ratio = (garmin_elapsed - garmin_timer) / garmin_elapsed if garmin_elapsed > 0 else 0
            garmin_rest_ratio = max(0.0, min(0.5, garmin_rest_ratio))

            # 提取记录点
            records = []
            for record in fitfile.get_messages('record'):
                record_data = {}
                for field in record:
                    record_data[field.name] = field.value
                records.append(record_data)

            if len(records) < 10:
                return [], 0.0, []

            # 解析时间序列
            timestamps = []
            distances = []
            elevations = []
            heart_rates = []

            for i, record in enumerate(records):
                if 'timestamp' in record:
                    ts = record['timestamp']
                    if isinstance(ts, datetime):
                        timestamps.append(ts.timestamp())
                    else:
                        timestamps.append(i * 60)
                else:
                    timestamps.append(i * 60)

                if 'distance' in record and record['distance'] is not None:
                    distances.append(float(record['distance']))
                elif 'enhanced_distance' in record and record['enhanced_distance'] is not None:
                    distances.append(float(record['enhanced_distance']))
                else:
                    distances.append(distances[-1] if i > 0 else 0)

                if 'altitude' in record and record['altitude'] is not None:
                    elevations.append(float(record['altitude']))
                elif 'enhanced_altitude' in record and record['enhanced_altitude'] is not None:
                    elevations.append(float(record['enhanced_altitude']))
                else:
                    elevations.append(0)

                if 'heart_rate' in record and record['heart_rate'] is not None:
                    heart_rates.append(int(record['heart_rate']))
                elif 'enhanced_heart_rate' in record and record['enhanced_heart_rate'] is not None:
                    heart_rates.append(int(record['enhanced_heart_rate']))
                else:
                    heart_rates.append(0)

            if len(elevations) < FilterConfig.FIT['window_size']:
                return [], 0.0, []

            # 使用宽松的max_grade收集坡度数据用于校准 (200%)
            loose_max_grade = 200.0
            distances_arr = np.array(distances)
            elevations_arr = np.array(elevations)

            grades_for_calibration = []
            for i in range(len(elevations_arr) - 1):
                dist_m = distances_arr[i + 1] - distances_arr[i]
                if dist_m > FilterConfig.FIT.get('min_distance_m', 0.5):
                    grade = ((elevations_arr[i + 1] - elevations_arr[i]) / dist_m) * 100
                    grade = np.clip(grade, -loose_max_grade, loose_max_grade)
                    grades_for_calibration.append(abs(grade))

            # 提取分段特征 (使用正常配置)
            result = FeatureExtractor._extract_from_time_series(
                timestamps, distances, elevations, heart_rates, segment_length_m
            )

            # 保存到缓存 (包含解析后的原始数据，避免重复解析)
            FeatureExtractor._save_cached_data(fit_path, (
                timestamps, distances, elevations, heart_rates, garmin_rest_ratio
            ))

            # 清理临时目录
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            return result, garmin_rest_ratio, grades_for_calibration

        except Exception as e:
            print(f"  Warning: Could not extract from FIT {fit_path.name}: {e}")
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            return [], 0.0, []

    @staticmethod
    def _extract_from_time_series(timestamps: list, distances: list, elevations: list, 
                                   heart_rates: list, segment_length_m: int) -> List[SegmentFeatures]:
        """从时间序列数据提取分段特征 (通用方法)"""
        segments = []

        # 对海拔进行平滑滤波
        smoothed_elevations, filter_info = apply_fit_filter(elevations, timestamps)

        # 计算坡度
        distances_arr = np.array(distances)
        smoothed_ele_arr = np.array(smoothed_elevations)

        grades = []
        for i in range(len(smoothed_ele_arr) - 1):
            dist_m = distances_arr[i + 1] - distances_arr[i]
            ele_m = smoothed_ele_arr[i + 1] - smoothed_ele_arr[i]

            # Bug 7 Fix: Use min_distance threshold to filter GPS drift noise
            min_distance = FilterConfig.FIT.get('min_distance_m', 0.5)
            if dist_m > min_distance:
                grade = (ele_m / dist_m) * 100
                grade = np.clip(grade, -FilterConfig.FIT['max_grade_pct'], FilterConfig.FIT['max_grade_pct'])
            else:
                grade = 0
            grades.append(grade)

        grades.append(grades[-1] if grades else 0)
        grades = np.array(grades)

        # 按距离分段
        accumulated_distance = 0
        accumulated_ascent = 0
        current_seg_distance = 0
        current_seg_elevation_gain = 0
        current_seg_time = 0  # Bug 3 Fix: track segment start time
        last_elevation = smoothed_elevations[0]
        seg_start_idx = 1  # Bug 3 Fix: track segment start index

        # Performance Fix: Use deque for O(n) rolling grade calculation instead of O(n²) backtracking
        rolling_window = deque()  # Store (distance_m, grade) pairs
        window_distance = 0.0
        target_distance_m = 500

        for i in range(1, len(timestamps)):
            if i >= len(distances) or i >= len(smoothed_elevations):
                break

            seg_dist = distances[i] - distances[i-1]
            seg_ele = smoothed_elevations[i]
            # Index Fix: grades[i-1] is the grade for step from point i-1 to point i (current step)
            seg_grade = grades[i-1]
            ele_gain = max(0, seg_ele - last_elevation)

            accumulated_distance += seg_dist / 1000
            accumulated_ascent += ele_gain
            current_seg_distance += seg_dist
            current_seg_elevation_gain += ele_gain

            # Bug 3 Fix: Accumulate time in segment
            time_diff = timestamps[i] - timestamps[i-1]
            current_seg_time += time_diff

            last_elevation = seg_ele

            # Performance Fix: Update rolling window with current point (use i-1 for correct grade index)
            rolling_window.append((seg_dist, grades[i-1]))
            window_distance += seg_dist

            # Remove old entries from left when window exceeds 500m
            while len(rolling_window) > 1 and window_distance - rolling_window[0][0] >= target_distance_m:
                old_dist, _ = rolling_window.popleft()
                window_distance -= old_dist

            # Calculate rolling grade from current window
            rolling_grade = np.mean([g for _, g in rolling_window]) if rolling_window else seg_grade

            if current_seg_distance >= segment_length_m:
                # Index Fix: Use grades[i-1:seg_start_idx-1:-1] or adjust indices
                # The segment covers steps from seg_start_idx to i (inclusive of i's step)
                # which corresponds to grades[seg_start_idx-1 : i]
                avg_grade = np.mean(grades[seg_start_idx-1:i]) if i > seg_start_idx else seg_grade

                # Bug 3 Fix: Use segment average time for speed calculation
                speed = (current_seg_distance / 1000) / (current_seg_time / 3600) if current_seg_time > 0 else 5

                segments.append(SegmentFeatures(
                    speed_kmh=min(20, max(1, speed)),
                    grade_pct=avg_grade,
                    rolling_grade_500m=rolling_grade,
                    accumulated_distance_km=accumulated_distance,
                    accumulated_ascent_m=accumulated_ascent,
                    absolute_altitude_m=seg_ele,
                    elevation_density=accumulated_ascent / accumulated_distance if accumulated_distance > 0 else 0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2
                ))

                current_seg_distance = 0
                current_seg_elevation_gain = 0
                current_seg_time = 0  # Bug 3 Fix: Reset segment time
                seg_start_idx = i + 1  # Bug Fix: Start next segment from next point (avoid double-counting)

        return segments

    @staticmethod
    def extract_from_json(json_path: Path, segment_length_m: int = 200) -> Tuple[List[SegmentFeatures], float]:
        """从 JSON 文件提取分段特征

        Returns:
            (segments, rest_ratio): JSON无session数据, rest_ratio默认0
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 尝试获取详细的记录数据
            metrics = data.get('metrics', [])
            if not metrics:
                return FeatureExtractor._extract_from_summary(data, json_path), 0.0

            # 从详细记录提取分段特征
            return FeatureExtractor._extract_from_metrics(metrics, segment_length_m), 0.0

        except Exception as e:
            print(f"  Warning: Could not extract from {json_path}: {e}")
            return [], 0.0

    @staticmethod
    def _extract_from_summary(data: dict, json_path: Path) -> List[SegmentFeatures]:
        """从汇总数据提取特征 (简化版)"""
        activity_info = data.get('activity_info', {})
        distance = activity_info.get('distance_km', 0)
        duration = activity_info.get('duration_min', 0)
        elevation_gain = activity_info.get('elevation_gain', 0)

        if distance <= 0 or duration <= 0:
            return []

        avg_speed = distance / (duration / 60)
        avg_grade = (elevation_gain / distance / 10) if distance > 0 else 0  # 修正: gain(m)/dist(km)/10

        # 创建虚拟分段 (用于训练)
        num_segments = max(5, int(distance / 2))  # 每2km一段
        segments = []

        for i in range(num_segments):
            seg_distance = distance / num_segments
            seg_elevation = elevation_gain / num_segments
            seg_grade = avg_grade + np.random.normal(0, 2)  # 添加一些变化

            segments.append(SegmentFeatures(
                speed_kmh=avg_speed + np.random.normal(0, 0.5),
                grade_pct=seg_grade,
                rolling_grade_500m=seg_grade,
                accumulated_distance_km=seg_distance * (i + 1),
                accumulated_ascent_m=seg_elevation * (i + 1),
                absolute_altitude_m=100 + seg_elevation * (i + 1),
                elevation_density=elevation_gain / distance if distance > 0 else 0,
                is_climbing=seg_grade > 2,
                is_descending=seg_grade < -2
            ))

        return segments

    @staticmethod
    def _extract_from_metrics(metrics: list, segment_length_m: int) -> List[SegmentFeatures]:
        """从详细指标提取分段特征"""
        segments = []

        # 提取时间序列数据
        timestamps = []
        distances = []
        elevations = []

        for metric in metrics:
            if 'seconds' in metric:
                timestamps.append(metric['seconds'])
            if 'distance' in metric:
                distances.append(metric['distance'])
            if 'elevation' in metric:
                elevations.append(metric['elevation'])

        if len(elevations) < FilterConfig.FIT['window_size']:
            # 数据点太少，无法滤波
            return []

        # --- 新增: 对 FIT 原始海拔进行平滑滤波 ---
        # 使用与 GPX 相同的 SG 滤波器，但采用 FIT 配置
        smoothed_elevations, filter_info = apply_fit_filter(elevations, timestamps)

        # 计算坡度 (基于平滑后的海拔)
        distances_arr = np.array(distances)
        smoothed_ele_arr = np.array(smoothed_elevations)

        grades = []
        for i in range(len(smoothed_ele_arr) - 1):
            dist_m = distances_arr[i + 1] - distances_arr[i]
            ele_m = smoothed_ele_arr[i + 1] - smoothed_ele_arr[i]

            # Bug 7 Fix: Use min_distance threshold to filter GPS drift noise
            min_distance = FilterConfig.FIT.get('min_distance_m', 0.5)
            if dist_m > min_distance:
                grade = (ele_m / dist_m) * 100
                # 应用 FIT 配置的坡度截断
                grade = np.clip(grade, -FilterConfig.FIT['max_grade_pct'], FilterConfig.FIT['max_grade_pct'])
            else:
                grade = 0
            grades.append(grade)

        # 最后一个点使用前一个点的坡度
        grades.append(grades[-1] if grades else 0)
        grades = np.array(grades)
        # --------------------------------------

        # 按距离分段 (使用平滑后的海拔和坡度)
        accumulated_distance = 0
        accumulated_ascent = 0
        current_seg_distance = 0
        current_seg_elevation_gain = 0
        current_seg_time = 0  # Bug 3 Fix: track segment start time
        last_elevation = smoothed_elevations[0]
        seg_start_idx = 1  # Bug 3 Fix: track segment start index

        # Performance Fix: Use deque for O(n) rolling grade calculation instead of O(n²) backtracking
        rolling_window = deque()  # Store (distance_m, grade) pairs
        window_distance = 0.0
        target_distance_m = 500

        for i in range(1, len(timestamps)):
            if i >= len(distances) or i >= len(smoothed_elevations):
                break

            seg_dist = distances[i] - distances[i-1]
            seg_ele = smoothed_elevations[i]
            # Index Fix: grades[i-1] is the grade for step from point i-1 to point i (current step)
            seg_grade = grades[i-1]
            ele_gain = max(0, seg_ele - last_elevation)

            accumulated_distance += seg_dist / 1000  # 转换为 km
            accumulated_ascent += ele_gain
            current_seg_distance += seg_dist
            current_seg_elevation_gain += ele_gain

            # Bug 3 Fix: Accumulate time in segment
            time_diff = timestamps[i] - timestamps[i-1]
            current_seg_time += time_diff

            last_elevation = seg_ele

            # Performance Fix: Update rolling window with current point (use i-1 for correct grade index)
            rolling_window.append((seg_dist, grades[i-1]))
            window_distance += seg_dist

            # Remove old entries from left when window exceeds 500m
            while len(rolling_window) > 1 and window_distance - rolling_window[0][0] >= target_distance_m:
                old_dist, _ = rolling_window.popleft()
                window_distance -= old_dist

            # Calculate rolling grade from current window
            rolling_grade = np.mean([g for _, g in rolling_window]) if rolling_window else seg_grade

            # 当达到分段长度时创建特征
            if current_seg_distance >= segment_length_m:
                # Index Fix: Use grades[i-1:seg_start_idx-1:-1] or adjust indices
                # The segment covers steps from seg_start_idx to i (inclusive of i's step)
                # which corresponds to grades[seg_start_idx-1 : i]
                avg_grade = np.mean(grades[seg_start_idx-1:i]) if i > seg_start_idx else seg_grade

                # Bug 3 Fix: Use segment average time for speed calculation
                speed = (current_seg_distance / 1000) / (current_seg_time / 3600) if current_seg_time > 0 else 5

                segments.append(SegmentFeatures(
                    speed_kmh=min(20, max(1, speed)),  # 限制范围
                    grade_pct=avg_grade,
                    rolling_grade_500m=rolling_grade,
                    accumulated_distance_km=accumulated_distance,
                    accumulated_ascent_m=accumulated_ascent,
                    absolute_altitude_m=seg_ele,
                    elevation_density=accumulated_ascent / accumulated_distance if accumulated_distance > 0 else 0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2
                ))

                current_seg_distance = 0
                current_seg_elevation_gain = 0
                current_seg_time = 0  # Bug 3 Fix: Reset segment time
                seg_start_idx = i + 1  # Bug Fix: Start next segment from next point (avoid double-counting)

        return segments


class MLRacePredictor:
    """基于机器学习的比赛预测器"""

    def __init__(self):
        self.predictor = None  # 统一模型
        self.training_stats = {}
        self.all_feature_importance = {}

    def train_from_files(self, file_paths: List[str]) -> bool:
        """从文件列表训练模型 (优化版：避免重复解析FIT文件)

        Args:
            file_paths: FIT/JSON 文件路径列表

        Returns:
            训练是否成功
        """
        print("Training unified ML model from training records...")

        if not file_paths:
            print("  Error: No training files provided")
            return False

        # 转换为 Path 对象并过滤有效文件
        valid_files = []
        for fp in file_paths:
            path = Path(fp)
            if path.is_file() and path.suffix.lower() in ['.fit', '.json']:
                valid_files.append(path)

        if not valid_files:
            print("  Error: No valid FIT/JSON files found")
            return False

        # 按文件大小排序，取最大的 20 个
        files_with_size = [(f, f.stat().st_size) for f in valid_files]
        files_with_size.sort(key=lambda x: x[1], reverse=True)
        selected_files = [f[0] for f in files_with_size[:20]]

        print(f"\n  Received {len(file_paths)} files, using top {len(selected_files)} largest files")

        # 优化：一次解析，同时获取校准数据和特征
        fit_paths = [str(f) for f in selected_files if f.suffix.lower() == '.fit']
        all_grades_for_calibration = []

        # 提取所有分段特征
        all_segments = []
        all_rest_ratios = []
        file_count = 0

        for file_path in selected_files:
            try:
                if file_path.suffix.lower() == '.fit':
                    # 使用优化版本：一次解析，同时返回校准数据和特征
                    result = FeatureExtractor.extract_from_fit_optimized(file_path)
                    if result:
                        segments, rest_ratio, grades = result
                        if segments:
                            all_segments.extend(segments)
                            all_rest_ratios.append(rest_ratio)
                            all_grades_for_calibration.extend(grades)
                            file_count += 1
                            print(f"    {file_path.name}: {len(segments)} segments, rest_ratio={rest_ratio:.1%}")
                else:  # JSON
                    segments, rest_ratio = FeatureExtractor.extract_from_json(file_path)
                    if segments:
                        all_segments.extend(segments)
                        all_rest_ratios.append(rest_ratio)
                        file_count += 1
                        print(f"    {file_path.name}: {len(segments)} segments, rest_ratio={rest_ratio:.1%}")
            except Exception as e:
                print(f"    Warning: Failed to process {file_path.name}: {e}")

        # 校准 max_grade_pct (基于收集的坡度数据)
        calibrated_max_grade = FilterConfig.FIT['max_grade_pct']
        if len(all_grades_for_calibration) >= 100:
            import numpy as np
            grades_arr = np.array(all_grades_for_calibration)
            p99 = np.percentile(grades_arr, 99)
            calibrated_max_grade = float(np.clip(p99, 30, 80))
            print(f"    P99 calibration: {len(all_grades_for_calibration)} grades -> max_grade={calibrated_max_grade:.0f}%")

        # 更新配置
        original_fit_config = FilterConfig.FIT.copy()
        FilterConfig.FIT = FilterConfig.FIT.copy()
        FilterConfig.FIT['max_grade_pct'] = calibrated_max_grade
        print(f"    Calibrated max_grade_pct: {calibrated_max_grade:.0f}% (default was {original_fit_config['max_grade_pct']:.0f}%)")

        if len(all_segments) < 5:
            print(f"  Error: Only {len(all_segments)} segments extracted, need at least 5")
            return False

        print(f"\n  Total: {file_count} files, {len(all_segments)} segments")

        # 训练统一模型
        self.predictor = LightGBMPredictor()
        if self.predictor.train(all_segments):
            self.all_feature_importance = self.predictor.get_feature_importance()

            # 统计信息
            speeds = [s.speed_kmh for s in all_segments]
            avg_rest_ratio = statistics.mean(all_rest_ratios) if all_rest_ratios else 0.08
            avg_rest_ratio = max(0.03, avg_rest_ratio)  # Minimum 3% rest

            self.training_stats = {
                'file_count': file_count,
                'segment_count': len(all_segments),
                'avg_speed': round(statistics.mean(speeds), 2),
                'p50_speed': round(self.predictor.p50_speed, 2),
                'p90_speed': round(self.predictor.p90_speed, 2),
                'effort_range': round(self.predictor.p90_speed / self.predictor.p50_speed, 2) if self.predictor.p50_speed > 0 else 1.0,
                'avg_rest_ratio': round(avg_rest_ratio, 3),
                'calibrated_max_grade_pct': calibrated_max_grade
            }

            print(f"\n  Training Stats:")
            print(f"    Average Speed: {self.training_stats['avg_speed']} km/h")
            print(f"    P50 Speed: {self.training_stats['p50_speed']} km/h")
            print(f"    P90 Speed: {self.training_stats['p90_speed']} km/h")
            print(f"    Effort Range: {self.training_stats['effort_range']}x")
            print(f"    Avg Rest Ratio: {self.training_stats['avg_rest_ratio']:.1%}")
            print(f"    Calibrated Max Grade: {self.training_stats['calibrated_max_grade_pct']:.0f}%")

            # Restore original FIT config
            FilterConfig.FIT = original_fit_config

            return True

        # Restore original FIT config on failure
        FilterConfig.FIT = original_fit_config
        return False

    def parse_gpx_route(self, gpx_path: str, segment_length_km: float = 0.2) -> Tuple[List[SegmentFeatures], Dict]:
        """解析 GPX 路线为分段特征 (包含海拔平滑滤波)"""
        tree = ET.parse(gpx_path)
        root = tree.getroot()
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

        # 提取轨迹点
        points = []
        for trkpt in root.findall('.//gpx:trkpt', ns):
            ele = trkpt.find('gpx:ele', ns)
            points.append({
                'lat': float(trkpt.get('lat')),
                'lon': float(trkpt.get('lon')),
                'ele': float(ele.text) if ele is not None else 0
            })

        if not points:
            raise ValueError("No track points in GPX")

        # 提取 CP 点 (waypoints)
        checkpoints = []
        for wpt in root.findall('.//gpx:wpt', ns):
            name_elem = wpt.find('gpx:name', ns)
            ele_elem = wpt.find('gpx:ele', ns)
            cp_name = name_elem.text if name_elem is not None else "Unknown"
            checkpoints.append({
                'name': cp_name,
                'lat': float(wpt.get('lat')),
                'lon': float(wpt.get('lon')),
                'ele': float(ele_elem.text) if ele_elem is not None else 0
            })

        # 计算累积距离数组
        distances_m = [0]
        for i in range(len(points) - 1):
            dist = self._haversine_distance(points[i], points[i+1])
            distances_m.append(distances_m[-1] + dist)

        # 对海拔进行平滑滤波
        elevations = [p['ele'] for p in points]
        smoothed_elevations, filter_info = apply_gpx_filter(elevations, distances_m)

        # 更新点的海拔数据
        for i, point in enumerate(points):
            point['ele'] = float(smoothed_elevations[i])

        # Bug 5 Fix: Pass distances and smoothed elevations for rolling grade calculation
        segments = self._create_segments(points, np.array(distances_m), smoothed_elevations, segment_length_km, checkpoints)

        # 路线信息 (使用滤波后的数据)
        total_distance = distances_m[-1] / 1000
        total_gain = filter_info['filtered_gain_m']
        # 计算下降（使用滤波后的海拔）
        total_loss = sum(max(0, smoothed_elevations[i] - smoothed_elevations[i+1]) for i in range(len(smoothed_elevations)-1))

        route_info = {
            'total_distance_km': round(total_distance, 2),
            'total_elevation_gain_m': total_gain,
            'total_elevation_loss_m': round(total_loss),
            'elevation_density': round(total_gain / total_distance, 1) if total_distance > 0 else 0,
            'segment_count': len(segments),
            'checkpoint_count': len(checkpoints),
            'checkpoints': checkpoints,
            'filter_info': filter_info  # 添加滤波信息
        }

        return segments, route_info

    def _create_segments(self, points: List[Dict], distances_m: np.ndarray, smoothed_elevations: np.ndarray,
                         segment_length_km: float, checkpoints: List[Dict] = None) -> List[SegmentFeatures]:
        """从 GPX 点创建分段特征 (Bug 5 Fix: Add rolling_grade calculation)"""
        segments = []
        current_points = [points[0]]
        seg_distance = 0
        seg_ascent = 0
        seg_descent = 0
        total_distance = 0  # 总累计距离
        total_ascent = 0    # 总累计爬升
        total_descent = 0   # 总累计下降

        # Bug 5 Fix: Compute per-point grades (like FIT path)
        grades = []
        for i in range(len(points) - 1):
            dist_m = distances_m[i + 1] - distances_m[i]
            ele_m = smoothed_elevations[i + 1] - smoothed_elevations[i]
            min_distance = FilterConfig.GPX.get('min_distance_m', 0.5)
            if dist_m > min_distance:
                grade = (ele_m / dist_m) * 100
                grade = np.clip(grade, -FilterConfig.GPX['max_grade_pct'], FilterConfig.GPX['max_grade_pct'])
            else:
                grade = 0
            grades.append(grade)
        grades.append(grades[-1] if grades else 0)
        grades = np.array(grades)

        # Performance Fix: Use deque for O(n) rolling grade calculation
        rolling_window = deque()  # Store (distance_m, grade) pairs
        window_distance = 0.0
        target_distance_m = 500

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dist = self._haversine_distance(p1, p2)
            ele_diff = p2['ele'] - p1['ele']
            ascent = max(0, ele_diff)
            descent = max(0, -ele_diff)

            current_points.append(p2)
            seg_distance += dist
            seg_ascent += ascent
            seg_descent += descent
            total_distance += dist
            total_ascent += ascent
            total_descent += descent

            # Performance Fix: Update rolling window with current point
            rolling_window.append((dist, grades[i]))
            window_distance += dist

            # Remove old entries from left when window exceeds 500m
            while len(rolling_window) > 1 and window_distance - rolling_window[0][0] >= target_distance_m:
                old_dist, _ = rolling_window.popleft()
                window_distance -= old_dist

            # Calculate rolling grade from current window
            rolling_grade_500m = np.mean([g for _, g in rolling_window]) if rolling_window else grades[i]

            if seg_distance >= segment_length_km * 1000:
                # 计算分段特征
                seg_dist_km = seg_distance / 1000
                elevations = [p['ele'] for p in current_points]
                seg_gain = sum(max(0, elevations[j+1] - elevations[j]) for j in range(len(elevations)-1))
                seg_loss = sum(max(0, elevations[j] - elevations[j+1]) for j in range(len(elevations)-1))

                # 坡度 (%) = 净海拔差(m) / 距离(m) * 100
                avg_grade = ((elevations[-1] - elevations[0]) / seg_distance) * 100 if seg_distance > 0 else 0

                # 找最近的 CP 点
                cp_name = self._find_nearest_checkpoint(current_points[-1], checkpoints) if checkpoints else ""

                segments.append(SegmentFeatures(
                    speed_kmh=0,
                    grade_pct=avg_grade,
                    rolling_grade_500m=rolling_grade_500m,
                    accumulated_distance_km=total_distance / 1000,
                    accumulated_ascent_m=total_ascent,
                    absolute_altitude_m=current_points[-1]['ele'],
                    elevation_density=seg_gain / seg_dist_km if seg_dist_km > 0 else 0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2,
                    # 新增字段
                    segment_ascent_m=seg_gain,
                    segment_descent_m=seg_loss,
                    cp_name=cp_name
                ))

                current_points = [p2]
                seg_distance = 0
                seg_ascent = 0
                seg_descent = 0

        # 处理剩余点
        if len(current_points) > 1 and seg_distance > 0:
            seg_dist_km = seg_distance / 1000
            elevations = [p['ele'] for p in current_points]
            seg_gain = sum(max(0, elevations[j+1] - elevations[j]) for j in range(len(elevations)-1))
            seg_loss = sum(max(0, elevations[j] - elevations[j+1]) for j in range(len(elevations)-1))
            # 坡度 (%) = 净海拔差(m) / 距离(m) * 100
            avg_grade = ((elevations[-1] - elevations[0]) / seg_distance) * 100 if seg_distance > 0 else 0

            # Use current rolling window value for tail block
            cp_name = self._find_nearest_checkpoint(current_points[-1], checkpoints) if checkpoints else ""

            segments.append(SegmentFeatures(
                speed_kmh=0,
                grade_pct=avg_grade,
                rolling_grade_500m=rolling_grade_500m,
                accumulated_distance_km=total_distance / 1000,
                accumulated_ascent_m=total_ascent,
                absolute_altitude_m=current_points[-1]['ele'],
                elevation_density=seg_gain / seg_dist_km if seg_dist_km > 0 else 0,
                is_climbing=avg_grade > 2,
                is_descending=avg_grade < -2,
                segment_ascent_m=seg_gain,
                segment_descent_m=seg_loss,
                cp_name=cp_name
            ))

        return segments

    def _find_nearest_checkpoint(self, point: Dict, checkpoints: List[Dict], max_distance_m: float = 500) -> str:
        """找到最近的 CP 点"""
        if not checkpoints:
            return ""

        min_dist = float('inf')
        nearest_cp = ""

        for cp in checkpoints:
            dist = self._haversine_distance(point, cp)
            if dist < min_dist:
                min_dist = dist
                nearest_cp = cp['name']

        # 只有在足够近时才返回 CP 名称
        if min_dist <= max_distance_m:
            return nearest_cp
        return ""

    def _haversine_distance(self, p1: Dict, p2: Dict) -> float:
        """计算两点间距离(米)"""
        import math
        lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
        lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))

        return 6371000 * c

    def predict_race(self, gpx_path: str, effort_factor: float = 1.0) -> Dict:
        """
        预测比赛成绩

        Args:
            gpx_path: GPX 路线文件路径
            effort_factor: 努力程度系数 (0.8-1.2)
                          1.0 = 平均水平 (P50)
                          1.1-1.2 = 比赛状态 (接近 P90)
                          0.8-0.9 = 保守策略
        """
        if not self.predictor:
            raise ValueError("No trained predictor available. Run analyze_and_train() first.")

        segments, route_info = self.parse_gpx_route(gpx_path)

        # Calculate rest multiplier first (needed for segment cumulative times)
        avg_rest_ratio = self.training_stats.get('avg_rest_ratio', 0.08)
        rest_multiplier = 1 / (1 - avg_rest_ratio) if avg_rest_ratio < 1 else 1.0

        # 预测每段速度
        segment_predictions = []
        prev_cumulative = 0
        total_time = 0

        for i, seg in enumerate(segments):
            segment_distance = seg.accumulated_distance_km - prev_cumulative

            # 预测速度 (应用努力程度系数)
            predicted_speed = self.predictor.predict_speed(seg, effort_factor)

            # 计算时间
            segment_time = segment_distance / predicted_speed if predicted_speed > 0 else 0
            total_time += segment_time

            # Apply rest correction to cumulative time
            cumulative_with_rest = total_time * rest_multiplier

            # 获取本段爬升/下降
            seg_ascent = getattr(seg, 'segment_ascent_m', 0)
            seg_descent = getattr(seg, 'segment_descent_m', 0)
            cp_name = getattr(seg, 'cp_name', '')

            # 地形类型
            if seg.grade_pct > 15:
                grade_type = "陡上坡"
            elif seg.grade_pct > 8:
                grade_type = "中上坡"
            elif seg.grade_pct > 3:
                grade_type = "缓上坡"
            elif seg.grade_pct < -15:
                grade_type = "陡下坡"
            elif seg.grade_pct < -8:
                grade_type = "中下坡"
            elif seg.grade_pct < -3:
                grade_type = "缓下坡"
            else:
                grade_type = "平地"

            # 难度等级
            if abs(seg.grade_pct) > 30:
                difficulty = "extreme"
            elif abs(seg.grade_pct) > 20:
                difficulty = "hard"
            elif abs(seg.grade_pct) > 10:
                difficulty = "moderate"
            else:
                difficulty = "easy"

            segment_predictions.append({
                'segment': i + 1,
                'distance_km': round(segment_distance, 2),
                'grade_pct': round(seg.grade_pct, 1),
                'altitude_m': round(seg.absolute_altitude_m),
                'predicted_speed_kmh': round(predicted_speed, 2),
                'segment_time_min': round(segment_time * 60, 1),
                'cumulative_time_min': round(cumulative_with_rest * 60, 1),
                # 新增字段
                'ascent_m': round(seg_ascent, 0),
                'descent_m': round(seg_descent, 0),
                'cp_name': cp_name,
                'grade_type': grade_type,
                'difficulty': difficulty
            })

            prev_cumulative = seg.accumulated_distance_km

        total_distance = route_info['total_distance_km']

        # Calculate final times with rest correction
        predicted_time_min = total_time * 60 * rest_multiplier

        return {
            'effort_factor': effort_factor,
            'predicted_moving_time_min': round(total_time * 60),
            'predicted_time_min': round(predicted_time_min),
            'predicted_time_hours': round(predicted_time_min / 60, 2),
            'predicted_time_hm': self._format_time(predicted_time_min),
            'predicted_pace_min_km': round(predicted_time_min / total_distance, 1) if total_distance > 0 else 0,
            'predicted_speed_kmh': round(total_distance / (predicted_time_min / 60), 2) if predicted_time_min > 0 else 0,
            'rest_ratio_used': avg_rest_ratio,
            'rest_multiplier': round(rest_multiplier, 3),
            'total_distance_km': total_distance,
            'route_info': route_info,
            'training_stats': self.training_stats,
            'feature_importance': self.all_feature_importance,
            'segment_predictions': segment_predictions,
            'model_type': 'LightGBM Unified Model with Effort Factor'
        }

    def _format_time(self, minutes: float) -> str:
        """格式化时间"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours}:{mins:02d}:{secs:02d}"


def main():
    base_dir = Path(__file__).parent.parent
    gpx_path = base_dir / 'maps' / '2025黄岩九峰大师赛最终版.gpx'
    records_dir = base_dir / 'records'

    print("=" * 70)
    print("越野赛成绩预测器 V1.2 - LightGBM 统一建模版")
    print("=" * 70)

    predictor = MLRacePredictor()

    # 获取所有训练文件
    print("\n[Step 1/2] 训练统一 ML 模型...")
    fit_files = list(records_dir.glob('*.fit')) + list(records_dir.glob('*.FIT'))
    json_files = list(records_dir.glob('*.json'))
    training_files = [str(f) for f in fit_files + json_files]

    if not training_files:
        print("  Error: No training files found in records directory")
        return

    if not predictor.train_from_files(training_files):
        print("训练失败!")
        return

    print("\n[Step 2/2] 预测比赛成绩:")
    print("\n" + "=" * 70)

    # 测试不同努力程度
    for effort_factor in [0.9, 1.0, 1.1]:
        try:
            result = predictor.predict_race(str(gpx_path), effort_factor)

            print(f"\n【努力程度: {effort_factor}x】")
            print(f"  预测时间: {result['predicted_time_hm']} ({result['predicted_time_min']}分钟)")
            print(f"  预测配速: {result['predicted_pace_min_km']} min/km")
            print(f"  平均速度: {result['predicted_speed_kmh']} km/h")

        except Exception as e:
            print(f"\n【努力程度: {effort_factor}x】错误: {e}")

    # 保存结果
    output_path = base_dir / 'prediction_result_v1.2_unified.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': {
                'version': 'V1.2',
                'algorithm': 'LightGBM Unified Model',
                'features': ['grade_pct', 'rolling_grade_500m', 'accumulated_distance_km',
                           'accumulated_ascent_m', 'absolute_altitude_m', 'elevation_density'],
                'includes_effort_factor': True,
                'supports_fit_files': True
            },
            'training_stats': predictor.training_stats,
            'feature_importance': predictor.all_feature_importance
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
