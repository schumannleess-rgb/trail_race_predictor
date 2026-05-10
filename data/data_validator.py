"""
Trail Race Predictor - Data Validator

数据验证器 - 防呆设计，确保数据质量
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math

# 导入类型定义
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import ValidationResult, FileInfo


class DataValidator:
    """数据验证器 - 防呆设计"""

    # 地球坐标范围
    LAT_RANGE = (-90, 90)
    LON_RANGE = (-180, 180)

    # 中国大陆大致范围 (用于检测是否为国内数据)
    CHINA_BOUNDS = {
        'min_lat': 18,
        'max_lat': 54,
        'min_lon': 73,
        'max_lon': 135
    }

    @staticmethod
    def validate_gpx(file_path: str) -> ValidationResult:
        """
        验证GPX文件

        检查项:
        - 文件格式正确
        - 包含轨迹点
        - 包含海拔数据
        - 坐标范围合理

        Args:
            file_path: GPX文件路径

        Returns:
            ValidationResult: 验证结果
        """
        warnings = []

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 命名空间
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

            # 检查轨迹点
            trkpts = root.findall('.//gpx:trkpt', ns)
            if not trkpts:
                # 尝试无命名空间解析
                trkpts = [e for e in root.iter() if e.tag.endswith('trkpt')]

            if not trkpts:
                return ValidationResult(
                    valid=False,
                    error="GPX文件中未找到轨迹点 (trkpt)"
                )

            # 检查海拔数据
            has_elevation = False
            coords = []

            for trkpt in trkpts[:10]:  # 检查前10个点
                lat = float(trkpt.get('lat', 0))
                lon = float(trkpt.get('lon', 0))
                coords.append((lat, lon))

                # 检查海拔
                ele = trkpt.find('gpx:ele', ns)
                if ele is not None and ele.text:
                    has_elevation = True

                # 备用海拔检查
                if not has_elevation:
                    for child in trkpt:
                        if child.tag.endswith('ele') and child.text:
                            has_elevation = True
                            break

            if not has_elevation:
                warnings.append("GPX文件缺少海拔数据，预测精度可能降低")

            # 检查坐标范围
            for lat, lon in coords:
                if not (DataValidator.LAT_RANGE[0] <= lat <= DataValidator.LAT_RANGE[1]):
                    return ValidationResult(
                        valid=False,
                        error=f"纬度超出范围: {lat}"
                    )
                if not (DataValidator.LON_RANGE[0] <= lon <= DataValidator.LON_RANGE[1]):
                    return ValidationResult(
                        valid=False,
                        error=f"经度超出范围: {lon}"
                    )

            # 轨迹点数量警告
            if len(trkpts) < 50:
                warnings.append(f"轨迹点较少 ({len(trkpts)}个)，可能影响预测精度")

            return ValidationResult(
                valid=True,
                warnings=warnings
            )

        except ET.ParseError as e:
            return ValidationResult(
                valid=False,
                error=f"GPX文件格式错误: {str(e)}"
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"验证失败: {str(e)}"
            )

    @staticmethod
    def validate_fit(file_path: str) -> ValidationResult:
        """
        验证FIT文件

        检查项:
        - 文件格式正确
        - 包含GPS坐标 (非跑步机)
        - 包含海拔数据
        - 时间范围合理

        Args:
            file_path: FIT文件路径

        Returns:
            ValidationResult: 验证结果
        """
        warnings = []

        try:
            # 检查文件头
            with open(file_path, 'rb') as f:
                header = f.read(12)

                # FIT文件魔数检查
                if len(header) < 12 or header[8:12] != b'.FIT':
                    # 有些FIT文件可能没有标准头部，尝试继续
                    warnings.append("FIT文件头格式非标准")

            # 尝试解析FIT文件
            try:
                from fitparse import FitFile
                fit = FitFile(file_path)

                # 检查是否有GPS记录
                has_gps = False
                has_elevation = False
                record_count = 0

                for record in fit.get_messages('record'):
                    record_count += 1
                    for field in record:
                        if field.name == 'position_lat' and field.value is not None:
                            has_gps = True
                        if field.name == 'altitude' and field.value is not None:
                            has_elevation = True

                    # 只检查前100条记录
                    if record_count >= 100:
                        break

                if not has_gps:
                    return ValidationResult(
                        valid=False,
                        error="检测到室内/跑步机数据 (无GPS坐标)，无法用于山地建模"
                    )

                if not has_elevation:
                    warnings.append("FIT文件缺少海拔数据，预测精度可能降低")

                if record_count < 60:
                    warnings.append(f"记录点较少 ({record_count}个)，可能影响模型训练")

            except ImportError:
                warnings.append("fitparse未安装，无法深度验证FIT文件")

            return ValidationResult(
                valid=True,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"验证失败: {str(e)}"
            )

    @staticmethod
    def validate_json(file_path: str) -> ValidationResult:
        """
        验证JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            ValidationResult: 验证结果
        """
        import json

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查必要字段
            warnings = []

            if 'metrics' not in data and 'activity_info' not in data:
                warnings.append("JSON文件缺少标准字段，可能无法正确解析")

            return ValidationResult(
                valid=True,
                warnings=warnings
            )

        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error=f"JSON格式错误: {str(e)}"
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"验证失败: {str(e)}"
            )

    @staticmethod
    def check_coordinate_alignment(gpx_coords: Tuple[float, float],
                                   fit_coords_list: List[Tuple[float, float]],
                                   threshold_km: float = 500) -> ValidationResult:
        """
        检查赛道和训练记录的坐标对齐

        Args:
            gpx_coords: 赛道中心坐标 (lat, lon)
            fit_coords_list: 训练记录坐标列表 [(lat, lon), ...]
            threshold_km: 警告阈值 (公里)

        Returns:
            ValidationResult: 是否对齐、距离偏差、警告消息
        """
        warnings = []

        if not fit_coords_list:
            return ValidationResult(
                valid=True,
                warnings=["无训练记录坐标可供对比"]
            )

        # 计算训练记录的中心点
        avg_lat = sum(c[0] for c in fit_coords_list) / len(fit_coords_list)
        avg_lon = sum(c[1] for c in fit_coords_list) / len(fit_coords_list)

        # 计算两点距离 (使用Haversine公式)
        distance_km = DataValidator._haversine_distance(
            gpx_coords[0], gpx_coords[1],
            avg_lat, avg_lon
        )

        if distance_km > threshold_km:
            warnings.append(
                f"赛道与训练记录位置偏差较大 ({distance_km:.0f}km)，"
                f"地形差异可能影响预测精度"
            )

        # 检查是否在国内
        gpx_in_china = DataValidator._is_in_china(gpx_coords)
        fit_in_china = DataValidator._is_in_china((avg_lat, avg_lon))

        if gpx_in_china != fit_in_china:
            warnings.append(
                "赛道与训练记录位于不同地区 (国内/国外)，"
                "气候和地形差异可能影响预测"
            )

        return ValidationResult(
            valid=True,
            warnings=warnings
        )

    @staticmethod
    def detect_unit_system(data: dict) -> str:
        """
        自动检测单位系统

        通过分析速度值范围判断是公制还是英制

        Args:
            data: 包含速度数据的字典

        Returns:
            'metric' (公制) 或 'imperial' (英制)
        """
        # 获取速度值
        speeds = []

        if 'metrics' in data:
            for m in data['metrics'][:100]:
                if 'speed' in m and m['speed']:
                    speeds.append(m['speed'])

        if not speeds:
            return 'metric'  # 默认公制

        avg_speed = sum(speeds) / len(speeds)

        # 如果平均速度 > 30，很可能是英制 (mph)
        # 跑步速度通常 5-15 km/h 或 3-9 mph
        if avg_speed > 30:
            return 'imperial'

        return 'metric'

    @staticmethod
    def filter_by_time(files: List[FileInfo],
                       years: int = 2,
                       reference_date: str = None) -> List[FileInfo]:
        """
        按时间过滤文件

        Args:
            files: 文件信息列表
            years: 只保留最近N年的数据
            reference_date: 参考日期 (默认今天)，格式 'YYYY-MM-DD'

        Returns:
            过滤后的文件列表
        """
        from datetime import datetime, timedelta

        if reference_date:
            ref = datetime.strptime(reference_date, '%Y-%m-%d')
        else:
            ref = datetime.now()

        cutoff = ref - timedelta(days=years * 365)

        filtered = []
        for f in files:
            if f.activity_date:
                try:
                    act_date = datetime.strptime(f.activity_date, '%Y-%m-%d')
                    if act_date >= cutoff:
                        filtered.append(f)
                except ValueError:
                    # 日期格式错误，保留文件
                    filtered.append(f)
            else:
                # 无日期信息，保留文件
                filtered.append(f)

        return filtered

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """计算两点间距离 (公里)"""
        R = 6371  # 地球半径 (公里)

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    @staticmethod
    def _is_in_china(coords: Tuple[float, float]) -> bool:
        """检查坐标是否在中国范围内"""
        lat, lon = coords
        bounds = DataValidator.CHINA_BOUNDS
        return (bounds['min_lat'] <= lat <= bounds['max_lat'] and
                bounds['min_lon'] <= lon <= bounds['max_lon'])


def validate_file(file_path: str) -> ValidationResult:
    """
    自动检测文件类型并验证

    Args:
        file_path: 文件路径

    Returns:
        ValidationResult: 验证结果
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.gpx':
        return DataValidator.validate_gpx(file_path)
    elif suffix == '.fit':
        return DataValidator.validate_fit(file_path)
    elif suffix == '.json':
        return DataValidator.validate_json(file_path)
    else:
        return ValidationResult(
            valid=False,
            error=f"不支持的文件格式: {suffix}"
        )
