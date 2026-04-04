"""
Trail Race Predictor - Feature Extractor

从 FIT / JSON 训练记录中提取 SegmentFeatures 列表
"""

import json
import gzip
import zipfile
import tempfile
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from .features import SegmentFeatures
from ..utils import apply_fit_filter, FilterConfig


class FeatureExtractor:
    """从训练文件中提取分段特征"""

    # ------------------------------------------------------------------
    # Cache (mirrors 1.2.1_old FeatureExtractor cache)
    # ------------------------------------------------------------------

    _CACHE_DIR = Path.home() / '.trail_race_predictor' / 'fit_cache'
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_cache_path(cls, fit_path: Path) -> Path:
        """获取缓存文件路径"""
        import hashlib
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

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_fit(
        fit_path: Path,
        segment_length_m: int = 200,
    ) -> Tuple[List[SegmentFeatures], float]:
        """从 FIT 文件提取特征（支持 GZIP / ZIP 压缩）

        使用内容hash缓存避免重复解析。

        Returns:
            (segments, rest_ratio)
        """
        # 1. Cache check — use content hash of original file
        cached = FeatureExtractor._get_cached_data(fit_path)
        if cached is not None:
            timestamps, distances, elevations, heart_rates, garmin_rest_ratio = cached
            if len(elevations) >= FilterConfig.FIT['window_size']:
                segments = FeatureExtractor._extract_from_time_series(
                    timestamps, distances, elevations, heart_rates, segment_length_m
                )
                return segments, garmin_rest_ratio
            # cached data too small, fall through to re-parse

        # 2. Load / decompress
        try:
            from fitparse import FitFile
        except ImportError:
            print(f"  Warning: fitparse not available, cannot parse {fit_path}")
            return [], 0.0

        actual_path, temp_dir = FeatureExtractor._decompress_fit(fit_path)
        if actual_path is None:
            return [], 0.0

        try:
            fitfile = FitFile(str(actual_path))

            # Session-level Garmin pre-computed values
            session_data = {}
            for session in fitfile.get_messages('session'):
                for field in session:
                    if field.name in ('total_ascent', 'total_descent', 'total_distance',
                                      'total_timer_time', 'total_elapsed_time'):
                        session_data[field.name] = field.value
                break

            garmin_distance_km = (session_data.get('total_distance', 0) or 0) / 1000
            print(f"    Garmin pre-calculated: "
                  f"ascent={session_data.get('total_ascent', 0)}m, "
                  f"descent={session_data.get('total_descent', 0)}m, "
                  f"distance={garmin_distance_km:.2f}km")

            elapsed = session_data.get('total_elapsed_time', 0) or 0
            timer   = session_data.get('total_timer_time', 0) or 0
            rest_ratio = float(np.clip((elapsed - timer) / elapsed, 0.0, 0.5)) if elapsed > 0 else 0.0
            if rest_ratio > 0:
                print(f"    Garmin rest ratio: {rest_ratio:.1%} "
                      f"(elapsed={elapsed:.0f}s, timer={timer:.0f}s)")

            records = list(fitfile.get_messages('record'))
            if len(records) < 10:
                return [], 0.0

            timestamps, distances, elevations, heart_rates = \
                FeatureExtractor._parse_records(records)

            if len(elevations) < FilterConfig.FIT['window_size']:
                return [], 0.0

            # 3. Save to cache before extracting segments
            FeatureExtractor._save_cached_data(fit_path, (
                timestamps, distances, elevations, heart_rates, rest_ratio
            ))

            segments = FeatureExtractor._extract_from_time_series(
                timestamps, distances, elevations, heart_rates, segment_length_m
            )
            return segments, rest_ratio

        except Exception as e:
            print(f"  Warning: Could not extract from FIT {fit_path.name}: {e}")
            return [], 0.0

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def extract_from_json(
        json_path: Path,
        segment_length_m: int = 200,
    ) -> Tuple[List[SegmentFeatures], float]:
        """从 JSON 文件提取特征

        Returns:
            (segments, rest_ratio)  — JSON 没有 session 数据，rest_ratio 默认 0
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metrics = data.get('metrics', [])
            if not metrics:
                return FeatureExtractor._extract_from_summary(data, json_path), 0.0

            return FeatureExtractor._extract_from_metrics(metrics, segment_length_m), 0.0

        except Exception as e:
            print(f"  Warning: Could not extract from {json_path}: {e}")
            return [], 0.0

    # ------------------------------------------------------------------
    # Core time-series segmentation (shared by FIT and JSON paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_from_time_series(
        timestamps: list,
        distances: list,
        elevations: list,
        heart_rates: list,
        segment_length_m: int,
    ) -> List[SegmentFeatures]:
        """通用: 从时间序列中按距离切段，提取 SegmentFeatures"""
        smoothed_elevations, _ = apply_fit_filter(elevations, timestamps)
        distances_arr = np.array(distances)
        smoothed_arr  = np.array(smoothed_elevations)

        grades = FeatureExtractor._compute_grades(distances_arr, smoothed_arr)

        segments: List[SegmentFeatures] = []
        accumulated_distance = 0.0
        accumulated_ascent   = 0.0
        current_seg_distance = 0.0
        current_seg_time     = 0.0
        last_elevation = float(smoothed_elevations[0])
        seg_start_idx  = 1

        rolling_window = deque()   # (distance_m, grade)
        window_distance = 0.0
        TARGET_WINDOW_M = 500.0

        for i in range(1, len(timestamps)):
            if i >= len(distances) or i >= len(smoothed_elevations):
                break

            seg_dist  = distances[i] - distances[i - 1]
            seg_ele   = float(smoothed_elevations[i])
            seg_grade = grades[i - 1]
            ele_gain  = max(0.0, seg_ele - last_elevation)

            accumulated_distance += seg_dist / 1000.0
            accumulated_ascent   += ele_gain
            current_seg_distance += seg_dist
            current_seg_time     += timestamps[i] - timestamps[i - 1]
            last_elevation = seg_ele

            rolling_window.append((seg_dist, seg_grade))
            window_distance += seg_dist
            while len(rolling_window) > 1 and window_distance - rolling_window[0][0] >= TARGET_WINDOW_M:
                old_dist, _ = rolling_window.popleft()
                window_distance -= old_dist
            rolling_grade = float(np.mean([g for _, g in rolling_window])) if rolling_window else seg_grade

            if current_seg_distance >= segment_length_m:
                avg_grade = float(np.mean(grades[seg_start_idx - 1:i])) if i > seg_start_idx else seg_grade
                speed = (current_seg_distance / 1000.0) / (current_seg_time / 3600.0) \
                        if current_seg_time > 0 else 5.0

                segments.append(SegmentFeatures(
                    speed_kmh=min(20.0, max(1.0, speed)),
                    grade_pct=avg_grade,
                    rolling_grade_500m=rolling_grade,
                    accumulated_distance_km=accumulated_distance,
                    accumulated_ascent_m=accumulated_ascent,
                    absolute_altitude_m=seg_ele,
                    elevation_density=accumulated_ascent / accumulated_distance
                                      if accumulated_distance > 0 else 0.0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2,
                ))

                current_seg_distance = 0.0
                current_seg_time     = 0.0
                seg_start_idx        = i + 1

        return segments

    # ------------------------------------------------------------------
    # JSON-specific paths
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_from_summary(data: dict, json_path: Path) -> List[SegmentFeatures]:
        """从汇总字段生成虚拟分段（无详细 metrics 时的兜底）"""
        info     = data.get('activity_info', {})
        distance = info.get('distance_km', 0)
        duration = info.get('duration_min', 0)
        gain     = info.get('elevation_gain', 0)

        if distance <= 0 or duration <= 0:
            return []

        avg_speed = distance / (duration / 60.0)
        avg_grade = (gain / distance / 10.0) if distance > 0 else 0.0
        n_segs    = max(5, int(distance / 2))

        return [
            SegmentFeatures(
                speed_kmh=avg_speed + float(np.random.normal(0, 0.5)),
                grade_pct=avg_grade + float(np.random.normal(0, 2)),
                rolling_grade_500m=avg_grade,
                accumulated_distance_km=(distance / n_segs) * (i + 1),
                accumulated_ascent_m=(gain / n_segs) * (i + 1),
                absolute_altitude_m=100.0 + (gain / n_segs) * (i + 1),
                elevation_density=gain / distance if distance > 0 else 0.0,
                is_climbing=avg_grade > 2,
                is_descending=avg_grade < -2,
            )
            for i in range(n_segs)
        ]

    @staticmethod
    def _extract_from_metrics(metrics: list, segment_length_m: int) -> List[SegmentFeatures]:
        """从 JSON metrics 列表提取时间序列再切段"""
        timestamps, distances, elevations = [], [], []
        for m in metrics:
            if 'seconds'   in m: timestamps.append(m['seconds'])
            if 'distance'  in m: distances.append(m['distance'])
            if 'elevation' in m: elevations.append(m['elevation'])

        if len(elevations) < FilterConfig.FIT['window_size']:
            return []

        heart_rates = [0] * len(timestamps)
        return FeatureExtractor._extract_from_time_series(
            timestamps, distances, elevations, heart_rates, segment_length_m
        )

    # ------------------------------------------------------------------
    # FIT helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decompress_fit(fit_path: Path):
        """解压 GZIP / ZIP 格式的 FIT 文件，返回 (actual_path, temp_dir)"""
        try:
            with open(fit_path, 'rb') as f:
                header = f.read(4)
        except Exception as e:
            print(f"  Warning: Cannot read {fit_path.name}: {e}")
            return None, None

        temp_dir = None
        try:
            if header[:2] == b'\x1f\x8b':
                temp_dir = tempfile.mkdtemp()
                out_path = Path(temp_dir) / fit_path.stem
                with gzip.open(fit_path, 'rb') as gz:
                    out_path.write_bytes(gz.read())
                print(f"    (Decompressed gzip: {fit_path.name})")
                return out_path, temp_dir

            if header[:4] == b'PK\x03\x04':
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(fit_path, 'r') as zf:
                    zf.extractall(temp_dir)
                extracted = list(Path(temp_dir).rglob('*.fit')) + \
                            list(Path(temp_dir).rglob('*.FIT'))
                if extracted:
                    print(f"    (Decompressed zip: {fit_path.name})")
                    return extracted[0], temp_dir
                print(f"  Warning: No FIT file in ZIP {fit_path.name}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None, None

        except Exception as e:
            print(f"  Warning: Failed to decompress {fit_path.name}: {e}")
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None

        return fit_path, None  # uncompressed

    @staticmethod
    def _parse_records(records: list):
        """从 FIT record 消息列表中提取时间序列数组"""
        timestamps, distances, elevations, heart_rates = [], [], [], []

        for i, rec in enumerate(records):
            data = {f.name: f.value for f in rec}

            ts = data.get('timestamp')
            timestamps.append(ts.timestamp() if isinstance(ts, datetime) else i * 60)

            dist = data.get('distance') or data.get('enhanced_distance')
            distances.append(float(dist) if dist is not None else (distances[-1] if distances else 0.0))

            alt = data.get('altitude') or data.get('enhanced_altitude')
            elevations.append(float(alt) if alt is not None else 0.0)

            hr = data.get('heart_rate') or data.get('enhanced_heart_rate')
            heart_rates.append(int(hr) if hr is not None else 0)

        return timestamps, distances, elevations, heart_rates

    # ------------------------------------------------------------------
    # Grade computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_grades(distances_arr: np.ndarray, elevations_arr: np.ndarray) -> np.ndarray:
        """按 FIT 配置计算并截断坡度数组"""
        min_dist  = FilterConfig.FIT.get('min_distance_m', 0.5)
        max_grade = FilterConfig.FIT['max_grade_pct']
        grades = []
        for i in range(len(elevations_arr) - 1):
            d = distances_arr[i + 1] - distances_arr[i]
            e = elevations_arr[i + 1] - elevations_arr[i]
            g = float(np.clip((e / d) * 100, -max_grade, max_grade)) if d > min_dist else 0.0
            grades.append(g)
        grades.append(grades[-1] if grades else 0.0)
        return np.array(grades)
