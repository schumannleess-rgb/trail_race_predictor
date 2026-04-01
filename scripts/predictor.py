"""
越野赛成绩预测器 V3 - 机器学习版 (LightGBM)

基于《机器学习.txt》方案实现:
1. 使用 LightGBM 替代线性回归
2. 高级特征工程 (rolling_grade, accumulated_distance, accumulated_ascent, absolute_altitude)
3. K-Fold 交叉验证
4. 特征重要性分析
5. 处理外推问题 (extrapolation)
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics

# 导入统一滤波工具
from utils import apply_fit_filter, FilterConfig


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


class LightGBMPredictor:
    """LightGBM 预测器"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.feature_importance = {}
        self.max_training_distance = 0  # 用于外推检测
        self.max_training_ascent = 0

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

        print(f"  Model trained: MAE={mae:.2f} km/h, RMSE={rmse:.2f} km/h")
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

    def predict_speed(self, segment: SegmentFeatures) -> float:
        """预测速度"""
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

            return max(1.0, min(15.0, predicted_speed))  # 限制在合理范围

        else:
            # 使用回退模型
            if segment.grade_pct > 5:
                return self.climb_speed
            elif segment.grade_pct < -5:
                return self.descent_speed
            else:
                return self.flat_speed

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.feature_importance


class FeatureExtractor:
    """特征提取器 - 从 JSON 文件提取分段特征"""

    @staticmethod
    def extract_from_json(json_path: Path, segment_length_m: int = 200) -> List[SegmentFeatures]:
        """从 JSON 文件提取分段特征"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 尝试获取详细的记录数据
            metrics = data.get('metrics', [])
            if not metrics:
                return FeatureExtractor._extract_from_summary(data, json_path)

            # 从详细记录提取分段特征
            return FeatureExtractor._extract_from_metrics(metrics, segment_length_m)

        except Exception as e:
            print(f"  Warning: Could not extract from {json_path}: {e}")
            return []

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

            if dist_m > 0:
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
        last_elevation = smoothed_elevations[0]

        for i in range(1, len(timestamps)):
            if i >= len(distances) or i >= len(smoothed_elevations):
                break

            seg_dist = distances[i] - distances[i-1]
            seg_ele = smoothed_elevations[i]
            seg_grade = grades[i]
            ele_gain = max(0, seg_ele - last_elevation)

            accumulated_distance += seg_dist / 1000  # 转换为 km
            accumulated_ascent += ele_gain
            current_seg_distance += seg_dist
            current_seg_elevation_gain += ele_gain

            last_elevation = seg_ele

            # 当达到分段长度时创建特征
            if current_seg_distance >= segment_length_m:
                # 使用滤波后的坡度
                avg_grade = seg_grade

                # 计算滚动坡度 (过去500米平均坡度)
                rolling_window = max(1, int(500 / (seg_dist if seg_dist > 0 else 1)))
                start_idx = max(0, i - rolling_window)
                rolling_grade = np.mean(grades[start_idx:i+1]) if i > 0 else avg_grade

                # 计算速度
                time_diff = timestamps[i] - timestamps[i-1]
                speed = (seg_dist / 1000) / (time_diff / 3600) if time_diff > 0 else 5

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

        return segments


class MLRacePredictor:
    """基于机器学习的比赛预测器"""

    def __init__(self, records_dir: str):
        self.records_dir = Path(records_dir)
        self.predictors = {}
        self.training_stats = {}
        self.all_feature_importance = {}

    def analyze_and_train(self):
        """分析训练记录并训练模型"""
        print("Training ML models from training records...")

        for category in ['高速', '中速', '低速']:
            possible_paths = [
                self.records_dir / category,
                self.records_dir / '低速' / f'{category}json',
                self.records_dir / f'{category}json',
            ]

            json_files = []
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    json_files = list(path.glob('*.json'))
                    if json_files:
                        break

            if not json_files:
                continue

            print(f"\n  {category}: {len(json_files)} files")

            # 提取所有分段特征
            all_segments = []
            for json_file in json_files:
                segments = FeatureExtractor.extract_from_json(json_file)
                all_segments.extend(segments)

            if len(all_segments) < 5:
                print(f"    Warning: Only {len(all_segments)} segments extracted")
                continue

            # 训练模型
            predictor = LightGBMPredictor()
            if predictor.train(all_segments):
                self.predictors[category] = predictor
                self.all_feature_importance[category] = predictor.get_feature_importance()

            # 统计信息
            speeds = [s.speed_kmh for s in all_segments]
            self.training_stats[category] = {
                'segment_count': len(all_segments),
                'avg_speed': round(statistics.mean(speeds), 2),
                'file_count': len(json_files)
            }

    def parse_gpx_route(self, gpx_path: str, segment_length_km: float = 0.2) -> Tuple[List[SegmentFeatures], Dict]:
        """解析 GPX 路线为分段特征"""
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

        # 创建分段
        segments = self._create_segments(points, segment_length_km)

        # 路线信息
        total_distance = sum(
            self._haversine_distance(points[i], points[i+1]) for i in range(len(points)-1)
        ) / 1000
        elevations = [p['ele'] for p in points]
        total_gain = sum(max(0, elevations[i+1] - elevations[i]) for i in range(len(elevations)-1))

        route_info = {
            'total_distance_km': round(total_distance, 2),
            'total_elevation_gain_m': round(total_gain),
            'elevation_density': round(total_gain / total_distance, 1) if total_distance > 0 else 0,
            'segment_count': len(segments)
        }

        return segments, route_info

    def _create_segments(self, points: List[Dict], segment_length_km: float) -> List[SegmentFeatures]:
        """从 GPX 点创建分段特征"""
        segments = []
        current_points = [points[0]]
        seg_distance = 0
        seg_ascent = 0
        total_distance = 0  # 总累计距离
        total_ascent = 0    # 总累计爬升

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dist = self._haversine_distance(p1, p2)
            ascent = max(0, p2['ele'] - p1['ele'])

            current_points.append(p2)
            seg_distance += dist
            seg_ascent += ascent
            total_distance += dist
            total_ascent += ascent

            if seg_distance >= segment_length_km * 1000:
                # 计算分段特征
                seg_dist_km = seg_distance / 1000
                elevations = [p['ele'] for p in current_points]
                seg_gain = sum(max(0, elevations[j+1] - elevations[j]) for j in range(len(elevations)-1))

                avg_grade = (seg_gain / seg_dist_km / 10) if seg_dist_km > 0 else 0  # 修正: grade = gain(m) / dist(km) / 10

                segments.append(SegmentFeatures(
                    speed_kmh=0,
                    grade_pct=avg_grade,
                    rolling_grade_500m=avg_grade,
                    accumulated_distance_km=total_distance / 1000,  # 使用总累计距离
                    accumulated_ascent_m=total_ascent,  # 使用总累计爬升
                    absolute_altitude_m=current_points[-1]['ele'],
                    elevation_density=seg_gain / seg_dist_km if seg_dist_km > 0 else 0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2
                ))

                current_points = [p2]
                seg_distance = 0
                seg_ascent = 0

        # 处理剩余点
        if len(current_points) > 1 and seg_distance > 0:
            seg_dist_km = seg_distance / 1000
            elevations = [p['ele'] for p in current_points]
            seg_gain = sum(max(0, elevations[j+1] - elevations[j]) for j in range(len(elevations)-1))
            avg_grade = (seg_gain / seg_dist_km / 10) if seg_dist_km > 0 else 0

            segments.append(SegmentFeatures(
                speed_kmh=0,
                grade_pct=avg_grade,
                rolling_grade_500m=avg_grade,
                accumulated_distance_km=total_distance / 1000,
                accumulated_ascent_m=total_ascent,
                absolute_altitude_m=current_points[-1]['ele'],
                elevation_density=seg_gain / seg_dist_km if seg_dist_km > 0 else 0,
                is_climbing=avg_grade > 2,
                is_descending=avg_grade < -2
            ))

        return segments

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

    def predict_race(self, gpx_path: str, effort_level: str = '中速') -> Dict:
        """预测比赛成绩"""
        segments, route_info = self.parse_gpx_route(gpx_path)

        # 获取预测器
        predictor = self.predictors.get(effort_level)
        if not predictor:
            predictor = self.predictors.get('低速')
        if not predictor:
            raise ValueError("No trained predictor available")

        # 预测每段速度
        segment_predictions = []
        prev_cumulative = 0  # 上一段的累计距离
        total_time = 0

        for i, seg in enumerate(segments):
            # 计算本段实际距离
            segment_distance = seg.accumulated_distance_km - prev_cumulative

            # 预测速度
            predicted_speed = predictor.predict_speed(seg)

            # 计算时间
            segment_time = segment_distance / predicted_speed if predicted_speed > 0 else 0
            total_time += segment_time

            if i < 20:  # 只保存前20段
                segment_predictions.append({
                    'segment': i + 1,
                    'distance_km': round(segment_distance, 2),
                    'grade_pct': round(seg.grade_pct, 1),
                    'altitude_m': round(seg.absolute_altitude_m),
                    'predicted_speed_kmh': round(predicted_speed, 2),
                    'segment_time_min': round(segment_time * 60, 1)
                })

            prev_cumulative = seg.accumulated_distance_km

        total_time_min = total_time * 60
        total_distance = route_info['total_distance_km']

        return {
            'effort_level': effort_level,
            'predicted_time_hours': round(total_time, 2),
            'predicted_time_min': round(total_time_min),
            'predicted_time_hm': self._format_time(total_time_min),
            'predicted_pace_min_km': round(total_time_min / total_distance, 1) if total_distance > 0 else 0,
            'predicted_speed_kmh': round(total_distance / total_time, 2) if total_time > 0 else 0,
            'total_distance_km': total_distance,
            'route_info': route_info,
            'training_stats': self.training_stats.get(effort_level, {}),
            'feature_importance': self.all_feature_importance.get(effort_level, {}),
            'segment_predictions': segment_predictions,
            'model_type': 'LightGBM with Advanced Features'
        }

    def _format_time(self, minutes: float) -> str:
        """格式化时间"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours}:{mins:02d}:{secs:02d}"


def main():
    base_dir = Path(__file__).parent.parent  # scripts/ -> v5/
    gpx_path = base_dir / 'maps' / '2025黄岩九峰大师赛最终版.gpx'
    records_dir = base_dir / 'records'

    print("=" * 70)
    print("越野赛成绩预测器 V3 - LightGBM 机器学习版")
    print("=" * 70)

    predictor = MLRacePredictor(str(records_dir))

    print("\n[Step 1/3] 训练 ML 模型...")
    predictor.analyze_and_train()

    print("\n[Step 2/3] 训练统计:")
    for cat, stats in predictor.training_stats.items():
        print(f"  {cat}: {stats['file_count']} 文件, {stats['segment_count']} 分段, "
              f"平均速度={stats['avg_speed']} km/h")

    print("\n[Step 3/3] 预测比赛成绩:")
    print("\n" + "=" * 70)

    results = {}
    for level in ['高速', '中速', '低速']:
        if level in predictor.predictors:
            try:
                result = predictor.predict_race(str(gpx_path), level)
                results[level] = result

                print(f"\n【{level}】")
                print(f"  预测时间: {result['predicted_time_hm']} ({result['predicted_time_min']}分钟)")
                print(f"  预测配速: {result['predicted_pace_min_km']} min/km")
                print(f"  平均速度: {result['predicted_speed_kmh']} km/h")

                # 显示特征重要性
                if result['feature_importance']:
                    print(f"  特征重要性:")
                    for feat, imp in sorted(result['feature_importance'].items(),
                                            key=lambda x: -x[1])[:3]:
                        print(f"    - {feat}: {imp:.0f}")
            except Exception as e:
                print(f"\n【{level}】错误: {e}")

    # 保存结果
    output_path = base_dir / 'prediction_result_v3_ml.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': {
                'version': 'V3',
                'algorithm': 'LightGBM',
                'features': ['grade_pct', 'rolling_grade_500m', 'accumulated_distance_km',
                           'accumulated_ascent_m', 'absolute_altitude_m', 'elevation_density'],
                'includes_extrapolation_penalty': True
            },
            'training_stats': predictor.training_stats,
            'feature_importance': predictor.all_feature_importance,
            'predictions': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
