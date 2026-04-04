"""
Trail Race Predictor - GPX Route Parser

解析 GPX 赛道为 SegmentFeatures 列表，供 MLRacePredictor 使用
"""

import math
import xml.etree.ElementTree as ET
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .features import SegmentFeatures
from ..utils import apply_gpx_filter, FilterConfig


class GPXRouteParser:
    """解析 GPX 赛道文件并生成分段特征"""

    _NS = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    def parse_gpx_route(
        self,
        gpx_path: str,
        segment_length_km: float = 0.2,
    ) -> Tuple[List[SegmentFeatures], Dict]:
        """解析 GPX 路线为分段特征（含海拔平滑滤波）

        Returns:
            (segments, route_info)
        """
        root = ET.parse(gpx_path).getroot()

        points      = self._extract_track_points(root)
        checkpoints = self._extract_waypoints(root)

        if not points:
            raise ValueError("No track points found in GPX file")

        distances_m = self._cumulative_distances(points)

        elevations = [p['ele'] for p in points]
        smoothed, filter_info = apply_gpx_filter(elevations, distances_m)

        for i, pt in enumerate(points):
            pt['ele'] = float(smoothed[i])

        segments = self._create_segments(
            points, np.array(distances_m), smoothed, segment_length_km, checkpoints
        )

        total_dist_km = distances_m[-1] / 1000.0
        total_gain    = filter_info['filtered_gain_m']
        total_loss    = sum(
            max(0.0, smoothed[i] - smoothed[i + 1])
            for i in range(len(smoothed) - 1)
        )

        route_info = {
            'total_distance_km':      round(total_dist_km, 2),
            'total_elevation_gain_m': total_gain,
            'total_elevation_loss_m': round(total_loss),
            'elevation_density':      round(total_gain / total_dist_km, 1) if total_dist_km > 0 else 0,
            'segment_count':          len(segments),
            'checkpoint_count':       len(checkpoints),
            'checkpoints':            checkpoints,
            'filter_info':            filter_info,
        }

        return segments, route_info

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_track_points(self, root: ET.Element) -> List[Dict]:
        points = []
        for trkpt in root.findall('.//gpx:trkpt', self._NS):
            ele = trkpt.find('gpx:ele', self._NS)
            points.append({
                'lat': float(trkpt.get('lat')),
                'lon': float(trkpt.get('lon')),
                'ele': float(ele.text) if ele is not None else 0.0,
            })
        return points

    def _extract_waypoints(self, root: ET.Element) -> List[Dict]:
        wpts = []
        for wpt in root.findall('.//gpx:wpt', self._NS):
            name = wpt.find('gpx:name', self._NS)
            ele  = wpt.find('gpx:ele',  self._NS)
            wpts.append({
                'name': name.text if name is not None else 'Unknown',
                'lat':  float(wpt.get('lat')),
                'lon':  float(wpt.get('lon')),
                'ele':  float(ele.text) if ele is not None else 0.0,
            })
        return wpts

    def _cumulative_distances(self, points: List[Dict]) -> List[float]:
        dists = [0.0]
        for i in range(len(points) - 1):
            dists.append(dists[-1] + self._haversine(points[i], points[i + 1]))
        return dists

    # ------------------------------------------------------------------
    # Segment creation
    # ------------------------------------------------------------------

    def _create_segments(
        self,
        points: List[Dict],
        distances_m: np.ndarray,
        smoothed_elevations: np.ndarray,
        segment_length_km: float,
        checkpoints: List[Dict],
    ) -> List[SegmentFeatures]:
        """按 segment_length_km 步长将 GPX 轨迹切成 SegmentFeatures"""
        segments: List[SegmentFeatures] = []

        grades = self._compute_grades(distances_m, smoothed_elevations)

        # Rolling 500-m grade window
        rolling_window = deque()
        window_dist    = 0.0
        TARGET_M       = 500.0

        current_points: List[Dict] = [points[0]]
        seg_dist = seg_ascent = seg_descent = 0.0
        total_dist = total_ascent = total_descent = 0.0

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dist     = self._haversine(p1, p2)
            ele_diff = p2['ele'] - p1['ele']

            current_points.append(p2)
            seg_dist    += dist
            seg_ascent  += max(0.0,  ele_diff)
            seg_descent += max(0.0, -ele_diff)
            total_dist   += dist
            total_ascent += max(0.0,  ele_diff)
            total_descent += max(0.0, -ele_diff)

            rolling_window.append((dist, grades[i]))
            window_dist += dist
            while len(rolling_window) > 1 and window_dist - rolling_window[0][0] >= TARGET_M:
                old, _ = rolling_window.popleft()
                window_dist -= old
            rolling_grade = float(np.mean([g for _, g in rolling_window])) if rolling_window else grades[i]

            if seg_dist >= segment_length_km * 1000:
                eles = [p['ele'] for p in current_points]
                gain = sum(max(0.0, eles[j + 1] - eles[j]) for j in range(len(eles) - 1))
                loss = sum(max(0.0, eles[j] - eles[j + 1]) for j in range(len(eles) - 1))
                avg_grade = ((eles[-1] - eles[0]) / seg_dist) * 100 if seg_dist > 0 else 0.0
                seg_dist_km = seg_dist / 1000.0
                cp_name = self._nearest_checkpoint(current_points[-1], checkpoints)

                segments.append(SegmentFeatures(
                    speed_kmh=0.0,
                    grade_pct=avg_grade,
                    rolling_grade_500m=rolling_grade,
                    accumulated_distance_km=total_dist / 1000.0,
                    accumulated_ascent_m=total_ascent,
                    absolute_altitude_m=p2['ele'],
                    elevation_density=gain / seg_dist_km if seg_dist_km > 0 else 0.0,
                    is_climbing=avg_grade > 2,
                    is_descending=avg_grade < -2,
                    segment_ascent_m=gain,
                    segment_descent_m=loss,
                    cp_name=cp_name,
                ))

                current_points = [p2]
                seg_dist = seg_ascent = seg_descent = 0.0

        # Tail block
        if len(current_points) > 1 and seg_dist > 0:
            eles        = [p['ele'] for p in current_points]
            seg_dist_km = seg_dist / 1000.0
            gain = sum(max(0.0, eles[j + 1] - eles[j]) for j in range(len(eles) - 1))
            loss = sum(max(0.0, eles[j] - eles[j + 1]) for j in range(len(eles) - 1))
            avg_grade = ((eles[-1] - eles[0]) / seg_dist) * 100 if seg_dist > 0 else 0.0
            cp_name = self._nearest_checkpoint(current_points[-1], checkpoints)

            segments.append(SegmentFeatures(
                speed_kmh=0.0,
                grade_pct=avg_grade,
                rolling_grade_500m=rolling_grade,   # last computed value
                accumulated_distance_km=total_dist / 1000.0,
                accumulated_ascent_m=total_ascent,
                absolute_altitude_m=current_points[-1]['ele'],
                elevation_density=gain / seg_dist_km if seg_dist_km > 0 else 0.0,
                is_climbing=avg_grade > 2,
                is_descending=avg_grade < -2,
                segment_ascent_m=gain,
                segment_descent_m=loss,
                cp_name=cp_name,
            ))

        return segments

    def _compute_grades(self, distances_m: np.ndarray, elevations: np.ndarray) -> np.ndarray:
        min_dist  = FilterConfig.GPX.get('min_distance_m', 0.5)
        max_grade = FilterConfig.GPX['max_grade_pct']
        grades = []
        for i in range(len(elevations) - 1):
            d = float(distances_m[i + 1] - distances_m[i])
            e = float(elevations[i + 1] - elevations[i])
            g = float(np.clip((e / d) * 100, -max_grade, max_grade)) if d > min_dist else 0.0
            grades.append(g)
        grades.append(grades[-1] if grades else 0.0)
        return np.array(grades)

    # ------------------------------------------------------------------
    # Geo helpers
    # ------------------------------------------------------------------

    def _nearest_checkpoint(
        self,
        point: Dict,
        checkpoints: List[Dict],
        max_dist_m: float = 500.0,
    ) -> str:
        if not checkpoints:
            return ""
        best_name = ""
        best_dist = float('inf')
        for cp in checkpoints:
            d = self._haversine(point, cp)
            if d < best_dist:
                best_dist, best_name = d, cp['name']
        return best_name if best_dist <= max_dist_m else ""

    @staticmethod
    def _haversine(p1: Dict, p2: Dict) -> float:
        lat1 = math.radians(p1['lat']); lon1 = math.radians(p1['lon'])
        lat2 = math.radians(p2['lat']); lon2 = math.radians(p2['lon'])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6_371_000 * 2 * math.asin(math.sqrt(a))
