"""
Trail Race Predictor - File Handler

文件处理器 - 处理上传文件的保存、筛选和分类
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

# 导入类型定义
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.types import FileInfo, ValidationResult
from data.data_validator import DataValidator, validate_file


class FileHandler:
    """文件上传处理器"""

    def __init__(self, temp_dir: str = './temp'):
        """
        初始化文件处理器

        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def clear_subdir(self, subdir: str = None):
        """
        清理子目录中的所有文件

        Args:
            subdir: 子目录名称
        """
        if subdir:
            target_dir = self.temp_dir / subdir
        else:
            target_dir = self.temp_dir

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file, subdir: str = None) -> str:
        """
        保存上传的文件到临时目录

        Args:
            uploaded_file: Streamlit上传的文件对象
            subdir: 子目录名称

        Returns:
            保存后的文件路径
        """
        if subdir:
            save_dir = self.temp_dir / subdir
        else:
            save_dir = self.temp_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        file_path = save_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)

    def save_file_from_path(self, source_path: str, subdir: str = None) -> str:
        """
        从路径复制文件到临时目录

        Args:
            source_path: 源文件路径
            subdir: 子目录名称

        Returns:
            保存后的文件路径
        """
        if subdir:
            save_dir = self.temp_dir / subdir
        else:
            save_dir = self.temp_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        source = Path(source_path)
        dest = save_dir / source.name
        shutil.copy2(source, dest)

        return str(dest)

    def get_file_info(self, file_path: str) -> FileInfo:
        """
        获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            FileInfo: 文件信息对象
        """
        path = Path(file_path)

        if not path.exists():
            return FileInfo(
                path=file_path,
                size_bytes=0,
                file_type='unknown',
                is_valid=False,
                error_message="文件不存在"
            )

        # 基本信息
        size = path.stat().st_size
        suffix = path.suffix.lower()

        # 文件类型
        type_map = {
            '.gpx': 'gpx',
            '.fit': 'fit',
            '.json': 'json',
            '.tcx': 'tcx'
        }
        file_type = type_map.get(suffix, 'unknown')

        # 验证文件
        validation = validate_file(file_path)

        # 尝试提取活动信息
        activity_date = None
        distance_km = None
        duration_min = None
        avg_speed_kmh = None

        try:
            if file_type == 'json':
                info = self._extract_json_info(file_path)
                activity_date = info.get('date')
                distance_km = info.get('distance_km')
                duration_min = info.get('duration_min')
                avg_speed_kmh = info.get('avg_speed_kmh')

            elif file_type == 'fit':
                info = self._extract_fit_info(file_path)
                activity_date = info.get('date')
                distance_km = info.get('distance_km')
                duration_min = info.get('duration_min')
                avg_speed_kmh = info.get('avg_speed_kmh')

            elif file_type == 'gpx':
                info = self._extract_gpx_info(file_path)
                distance_km = info.get('distance_km')
                # GPX通常没有时间信息

        except Exception as e:
            pass  # 提取失败不影响基本信息

        return FileInfo(
            path=file_path,
            size_bytes=size,
            file_type=file_type,
            activity_date=activity_date,
            distance_km=distance_km,
            duration_min=duration_min,
            avg_speed_kmh=avg_speed_kmh,
            is_valid=validation.valid,
            error_message=validation.error
        )

    def auto_select_best_files(self, file_paths: List[str],
                               max_count: int = 20) -> List[str]:
        """
        智能筛选最优文件

        选择逻辑:
        1. 过滤无效文件
        2. 按文件大小排序 (越大=数据越完整)
        3. 返回最大的N个

        Args:
            file_paths: 文件路径列表
            max_count: 最大返回数量

        Returns:
            筛选后的文件路径列表
        """
        # 获取所有文件信息
        file_infos = []
        for path in file_paths:
            info = self.get_file_info(path)
            if info.is_valid:
                file_infos.append(info)

        # 按文件大小降序排序
        file_infos.sort(key=lambda x: x.size_bytes, reverse=True)

        # 返回前N个
        return [f.path for f in file_infos[:max_count]]

    def classify_by_speed(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        自动按速度分类

        分类逻辑:
        - 计算每个文件的平均速度
        - 前1/3 -> 高速
        - 中1/3 -> 中速
        - 后1/3 -> 低速

        Args:
            file_paths: 文件路径列表

        Returns:
            分类结果 {'high': [...], 'medium': [...], 'low': [...]}
        """
        # 获取有速度信息的文件
        files_with_speed = []
        files_without_speed = []

        for path in file_paths:
            info = self.get_file_info(path)
            if info.avg_speed_kmh is not None:
                files_with_speed.append((path, info.avg_speed_kmh))
            else:
                files_without_speed.append(path)

        # 按速度排序
        files_with_speed.sort(key=lambda x: x[1], reverse=True)

        # 分类
        n = len(files_with_speed)
        high_end = n // 3
        low_start = 2 * n // 3

        result = {
            'high': [f[0] for f in files_with_speed[:high_end]],
            'medium': [f[0] for f in files_with_speed[high_end:low_start]],
            'low': [f[0] for f in files_with_speed[low_start:]]
        }

        # 无速度信息的文件归入中速
        if files_without_speed:
            result['medium'].extend(files_without_speed)

        return result

    def cleanup_temp(self, older_than_hours: int = 24):
        """
        清理临时文件

        Args:
            older_than_hours: 清理N小时前的文件
        """
        import time

        cutoff = time.time() - (older_than_hours * 3600)

        for item in self.temp_dir.iterdir():
            if item.is_file():
                if item.stat().st_mtime < cutoff:
                    item.unlink()
            elif item.is_dir():
                # 清理子目录
                try:
                    shutil.rmtree(item)
                except (OSError, PermissionError) as e:
                    # 忽略无法删除的目录（可能被其他进程占用）
                    pass

    def _extract_json_info(self, file_path: str) -> Dict:
        """从JSON文件提取活动信息"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {}

        # 尝试从文件名提取日期
        path = Path(file_path)
        name = path.stem
        if '_' in name:
            date_part = name.split('_')[-1]
            if len(date_part) == 8 and date_part.isdigit():
                result['date'] = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

        # 从数据中提取
        if 'activity_info' in data:
            info = data['activity_info']
            result['distance_km'] = info.get('distance_km')
            result['duration_min'] = info.get('duration_min')
            if result['distance_km'] and result['duration_min']:
                result['avg_speed_kmh'] = result['distance_km'] / (result['duration_min'] / 60)

        return result

    def _extract_fit_info(self, file_path: str) -> Dict:
        """从FIT文件提取活动信息"""
        result = {}

        try:
            from fitparse import FitFile
            fit = FitFile(file_path)

            # 获取会话信息
            for session in fit.get_messages('session'):
                for field in session:
                    if field.name == 'total_distance' and field.value:
                        result['distance_km'] = field.value / 1000
                    elif field.name == 'total_timer_time' and field.value:
                        result['duration_min'] = field.value / 60
                    elif field.name == 'enhanced_avg_speed' and field.value:
                        result['avg_speed_kmh'] = field.value * 3.6
                    elif field.name == 'start_time' and field.value:
                        result['date'] = str(field.value)[:10]

                break  # 只取第一个session

            # 计算平均速度
            if result.get('distance_km') and result.get('duration_min'):
                if 'avg_speed_kmh' not in result:
                    result['avg_speed_kmh'] = result['distance_km'] / (result['duration_min'] / 60)

        except ImportError:
            pass
        except Exception as e:
            pass

        return result

    def _extract_gpx_info(self, file_path: str) -> Dict:
        """从GPX文件提取信息"""
        import xml.etree.ElementTree as ET
        import math

        result = {}

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

            points = []
            for trkpt in root.findall('.//gpx:trkpt', ns):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                points.append((lat, lon))

            if len(points) > 1:
                # 计算总距离
                total_dist = 0
                for i in range(len(points) - 1):
                    lat1, lon1 = points[i]
                    lat2, lon2 = points[i + 1]

                    # Haversine公式
                    R = 6371  # km
                    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    total_dist += R * 2 * math.asin(math.sqrt(a))

                result['distance_km'] = total_dist

        except Exception as e:
            pass

        return result
