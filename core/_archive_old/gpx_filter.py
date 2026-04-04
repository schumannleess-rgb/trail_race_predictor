"""
GPX 赛道滤波器

解决 GPX 文件的三大致命问题:
1. DEM 数据陷阱 (悬崖与桥梁效应)
2. 采样间距极度不均匀
3. 爬升虚高 (海岸线悖论)

处理流程:
1. 等距重采样 (每 50 米一个点)
2. Savitzky-Golay 滤波 (去除 DEM 噪点)
3. 坡度物理截断 (限制在 [-45, 45]%)

使用方法:
    python gpx_filter.py input.gpx output.gpx
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json


class GPXFilter:
    """GPX 赛道滤波器"""

    def __init__(self, gpx_path: str):
        self.gpx_path = gpx_path
        self.ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        self.raw_data = None
        self.filtered_data = None

    def parse_gpx(self) -> Dict:
        """解析原始 GPX 文件"""
        tree = ET.parse(self.gpx_path)
        root = tree.getroot()

        # 提取轨迹点
        points = []
        for trkpt in root.findall('.//gpx:trkpt', self.ns):
            ele = trkpt.find('gpx:ele', self.ns)
            points.append({
                'lat': float(trkpt.get('lat')),
                'lon': float(trkpt.get('lon')),
                'ele': float(ele.text) if ele is not None else 0
            })

        # 提取航点 (CP点)
        waypoints = []
        for wpt in root.findall('.//gpx:wpt', self.ns):
            name = wpt.find('gpx:name', self.ns)
            ele = wpt.find('gpx:ele', self.ns)
            waypoints.append({
                'lat': float(wpt.get('lat')),
                'lon': float(wpt.get('lon')),
                'ele': float(ele.text) if ele is not None else 0,
                'name': name.text if name is not None else 'Unknown'
            })

        # 计算累计距离
        distances = [0]
        elevations = [points[0]['ele']]
        total_distance = 0

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dist = self._haversine_distance(p1, p2)
            total_distance += dist
            distances.append(total_distance)
            elevations.append(p2['ele'])

        self.raw_data = {
            'points': points,
            'waypoints': waypoints,
            'distances': np.array(distances),
            'elevations': np.array(elevations),
            'total_distance_km': total_distance / 1000
        }

        return self.raw_data

    def resample(self, spacing_m: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        等距重采样

        Args:
            spacing_m: 采样间距 (米)，默认 50 米

        Returns:
            (resampled_distances, resampled_elevations)
        """
        distances = self.raw_data['distances']
        elevations = self.raw_data['elevations']

        # 创建重采样距离点
        max_dist = distances[-1]
        resampled_distances = np.arange(0, max_dist, spacing_m)

        # 样条插值
        # 使用 'cubic' 样条可以获得平滑的插值结果
        interp_ele = interp1d(distances, elevations, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
        resampled_elevations = interp_ele(resampled_distances)

        print(f"  重采样: {len(distances)} 点 → {len(resampled_distances)} 点")
        print(f"  采样间距: {spacing_m} 米")

        return resampled_distances, resampled_elevations

    def smooth_elevation(self, elevations: np.ndarray,
                        method: str = 'savgol',
                        window_size: int = 7,
                        poly_order: int = 2) -> np.ndarray:
        """
        海拔滤波

        Args:
            elevations: 海拔数组 (米)
            method: 滤波方法 ('savgol' 或 'moving_average')
            window_size: 窗口大小 (点数)，建议 5-11
            poly_order: 多项式阶数，建议 2-3

        Returns:
            滤波后的海拔数组
        """
        if method == 'savgol':
            # Savitzky-Golay 滤波器
            # window_size 必须是奇数
            if window_size % 2 == 0:
                window_size += 1

            # polyorder=2 保留海拔变化的曲率
            smoothed = savgol_filter(elevations, window_size, polyorder=poly_order)

            print(f"  Savitzky-Golay 滤波 (窗口={window_size}, 阶数={poly_order})")

        elif method == 'moving_average':
            # 滑动平均
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(elevations, kernel, mode='same')

            print(f"  滑动平均滤波 (窗口={window_size})")

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return smoothed

    def calculate_grade(self, distances: np.ndarray,
                        elevations: np.ndarray,
                        max_grade: float = 45.0) -> np.ndarray:
        """
        计算坡度并截断

        Args:
            distances: 距离数组 (米)
            elevations: 海拔数组 (米)
            max_grade: 最大坡度限制 (%)

        Returns:
            坡度数组 (%)
        """
        grades = []

        for i in range(len(elevations) - 1):
            dist_m = distances[i + 1] - distances[i]
            ele_m = elevations[i + 1] - elevations[i]

            if dist_m > 0:
                # 计算坡度 (%)
                grade = (ele_m / dist_m) * 100

                # 物理截断
                grade = np.clip(grade, -max_grade, max_grade)
            else:
                grade = 0

            grades.append(grade)

        # 最后一个点使用前一个点的坡度
        grades.append(grades[-1] if grades else 0)

        return np.array(grades)

    def process(self, spacing_m: float = 20,
                smoothing_method: str = 'savgol',
                window_size: int = 7,
                poly_order: int = 2,
                max_grade: float = 45.0) -> Dict:
        """
        完整处理流程

        Returns:
            处理后的数据字典
        """
        print("=" * 60)
        print("GPX 滤波处理")
        print("=" * 60)

        # 步骤 1: 解析 GPX
        print("\n[步骤 1/4] 解析 GPX 文件...")
        self.parse_gpx()
        print(f"  原始点数: {len(self.raw_data['points'])}")
        print(f"  总距离: {self.raw_data['total_distance_km']:.2f} km")
        print(f"  CP 点数: {len(self.raw_data['waypoints'])}")

        # 计算原始统计信息
        raw_dists = self.raw_data['distances'][1:] - self.raw_data['distances'][:-1]
        raw_grades = self.calculate_grade(
            self.raw_data['distances'],
            self.raw_data['elevations'],
            max_grade=100  # 不截断，看原始数据
        )[:-1]  # 去掉最后一个点
        raw_gain = np.sum(np.maximum(raw_grades * raw_dists / 100, 0))

        print(f"  原始累计爬升: {raw_gain:.0f} m")
        print(f"  原始坡度范围: {np.min(raw_grades):.1f}% ~ {np.max(raw_grades):.1f}%")

        # 步骤 2: 等距重采样
        print("\n[步骤 2/4] 等距重采样...")
        resampled_dist, resampled_ele = self.resample(spacing_m)

        # 步骤 3: 滤波
        print("\n[步骤 3/4] 海拔滤波...")
        smoothed_ele = self.smooth_elevation(resampled_ele, smoothing_method, window_size)

        # 步骤 4: 计算坡度并截断
        print("\n[步骤 4/4] 计算坡度并截断...")
        grades = self.calculate_grade(resampled_dist, smoothed_ele, max_grade)

        # 计算滤波后的统计信息
        filtered_dists = resampled_dist[1:] - resampled_dist[:-1]
        filtered_grades = grades[:-1]  # 去掉最后一个点
        filtered_gain = np.sum(np.maximum(filtered_grades * filtered_dists / 100, 0))

        print(f"  滤波后累计爬升: {filtered_gain:.0f} m")
        print(f"  滤波后坡度范围: {np.min(grades):.1f}% ~ {np.max(grades):.1f}%")
        print(f"  爬升减少: {raw_gain - filtered_gain:.0f} m "
              f"({(raw_gain - filtered_gain) / raw_gain * 100:.1f}%)")

        # 统计坡度分布
        grade_ranges = {
            '下坡 (< -10%)': np.sum(grades < -10),
            '缓下 (-10% ~ -5%)': np.sum((grades >= -10) & (grades < -5)),
            '平路 (-5% ~ 5%)': np.sum((grades >= -5) & (grades < 5)),
            '缓上 (5% ~ 15%)': np.sum((grades >= 5) & (grades < 15)),
            '陡上 (15% ~ 30%)': np.sum((grades >= 15) & (grades < 30)),
            '极陡 (> 30%)': np.sum(grades >= 30),
        }

        print("\n坡度分布:")
        for range_name, count in grade_ranges.items():
            pct = count / len(grades) * 100
            dist_km = count * spacing_m / 1000
            print(f"  {range_name}: {dist_km:.2f} km ({pct:.1f}%)")

        self.filtered_data = {
            'distances_m': resampled_dist,
            'elevations_m': smoothed_ele,
            'grades_pct': grades,
            'spacing_m': spacing_m,
            'total_distance_km': resampled_dist[-1] / 1000,
            'total_elevation_gain_m': filtered_gain,
            'waypoints': self.raw_data['waypoints'],
            'grade_distribution': grade_ranges
        }

        print("\n" + "=" * 60)
        print("滤波完成!")
        print("=" * 60)

        return self.filtered_data

    def save_filtered_gpx(self, output_path: str,
                          include_waypoints: bool = True):
        """
        保存滤波后的 GPX 文件

        Args:
            output_path: 输出文件路径
            include_waypoints: 是否包含 CP 点
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Run process() first.")

        # 创建 GPX 根元素
        root = ET.Element('gpx', {
            'version': '1.1',
            'creator': 'GPX Filter - Trail Race Predictor',
            'xmlns': 'http://www.topografix.com/GPX/1/1'
        })

        # 添加元数据
        metadata = ET.SubElement(root, 'metadata')
        name = ET.SubElement(metadata, 'name')
        name.text = f"Filtered - {Path(self.gpx_path).stem}"

        # 添加 CP 点
        if include_waypoints:
            for wp in self.filtered_data['waypoints']:
                wpt = ET.SubElement(root, 'wpt', {
                    'lat': str(wp['lat']),
                    'lon': str(wp['lon'])
                })
                ET.SubElement(wpt, 'ele').text = str(wp['ele'])
                ET.SubElement(wpt, 'name').text = wp['name']

        # 添加轨迹 - 使用滤波后的数据
        trk = ET.SubElement(root, 'trk')
        ET.SubElement(trk, 'name').text = 'Filtered Track'

        trkseg = ET.SubElement(trk, 'trkseg')

        # 使用原始点但更新海拔
        raw_points = self.raw_data['points']
        raw_distances = self.raw_data['distances']
        filtered_distances = self.filtered_data['distances_m']
        filtered_elevations = self.filtered_data['elevations_m']

        # 创建插值函数: 原始距离 → 滤波后海拔
        ele_interp = interp1d(filtered_distances, filtered_elevations,
                             kind='linear', bounds_error=False,
                             fill_value='extrapolate')

        # 生成新的轨迹点 (使用重采样后的数据)
        for i, dist in enumerate(filtered_distances):
            # Bug 6 Fix: Remove unnecessary guard condition - argmin works for any index
            # 找到对应的原始点
            raw_idx = np.argmin(np.abs(raw_distances - dist))
            point = raw_points[raw_idx]

            trkpt = ET.SubElement(trkseg, 'trkpt', {
                'lat': str(point['lat']),
                'lon': str(point['lon'])
            })
            ET.SubElement(trkpt, 'ele').text = str(filtered_elevations[i])

        # 写入文件
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)

        print(f"\n滤波后 GPX 已保存: {output_path}")

    def save_filtered_json(self, output_path: str):
        """
        保存滤波后的 JSON 文件 (用于预测)

        Args:
            output_path: 输出 JSON 文件路径
        """
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Run process() first.")

        # 准备分段数据 (每 200 米一段，与预测器一致)
        segment_length_m = 200
        segments = []

        distances = self.filtered_data['distances_m']
        elevations = self.filtered_data['elevations_m']
        grades = self.filtered_data['grades_pct']

        num_segments = int(len(distances) * self.filtered_data['spacing_m'] / segment_length_m)

        for i in range(num_segments):
            start_idx = int(i * segment_length_m / self.filtered_data['spacing_m'])
            end_idx = int((i + 1) * segment_length_m / self.filtered_data['spacing_m'])
            end_idx = min(end_idx, len(distances) - 1)

            seg_dist = distances[end_idx] - distances[start_idx]
            seg_ele_start = elevations[start_idx]
            seg_ele_end = elevations[end_idx]
            seg_grade = grades[start_idx]  # 使用起始点坡度

            segments.append({
                'segment': i + 1,
                'distance_m': seg_dist,
                'elevation_gain_m': max(0, seg_ele_end - seg_ele_start),
                'grade_pct': seg_grade,
                'elevation_m': seg_ele_start
            })

        output_data = {
            'metadata': {
                'source': self.gpx_path,
                'filter_params': {
                    'spacing_m': self.filtered_data['spacing_m'],
                    'smoothing_method': 'savgol',
                    'max_grade': 45.0
                }
            },
            'route_info': {
                'total_distance_km': self.filtered_data['total_distance_km'],
                'total_elevation_gain_m': self.filtered_data['total_elevation_gain_m'],
                'num_segments': len(segments)
            },
            'waypoints': self.filtered_data['waypoints'],
            'segments': segments
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"滤波后数据已保存: {output_path}")

    def _haversine_distance(self, p1: Dict, p2: Dict) -> float:
        """计算两点间距离 (米)"""
        import math
        lat1, lon1 = math.radians(p1['lat']), math.radians(p1['lon'])
        lat2, lon2 = math.radians(p2['lat']), math.radians(p2['lon'])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))

        return 6371000 * c


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python gpx_filter.py <input.gpx> [output.gpx]")
        print("\n示例:")
        print("  python gpx_filter.py maps/2025黄岩九峰大师赛最终版.gpx")
        print("  python gpx_filter.py input.gpx output.gpx")
        return

    input_gpx = sys.argv[1]

    if len(sys.argv) >= 3:
        output_gpx = sys.argv[2]
    else:
        # 自动生成输出文件名
        input_path = Path(input_gpx)
        output_gpx = input_path.parent / f"{input_path.stem}_filtered.gpx"

    # 创建滤波器
    filter_processor = GPXFilter(input_gpx)

    # 执行滤波 (优化参数)
    filtered_data = filter_processor.process(
        spacing_m=20,              # 20 米重采样 (更精细)
        smoothing_method='savgol',  # Savitzky-Golay 滤波
        window_size=7,              # 窗口大小 (7点 = 140米)
        poly_order=2,               # 多项式阶数
        max_grade=45.0              # 坡度限制 ±45%
    )

    # 保存结果
    filter_processor.save_filtered_gpx(str(output_gpx))

    # 保存 JSON (用于预测)
    json_path = Path(str(output_gpx).replace('.gpx', '_filtered.json'))
    filter_processor.save_filtered_json(str(json_path))

    print(f"\n[OK] Complete!")
    print(f"  Filtered GPX: {output_gpx}")
    print(f"  JSON for prediction: {json_path}")


if __name__ == '__main__':
    main()
