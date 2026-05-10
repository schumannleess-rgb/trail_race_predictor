"""
Trail Race Predictor - Data Module

数据处理模块，包含文件处理、数据验证和坐标检查
"""

from .file_handler import FileHandler
from .data_validator import DataValidator

__all__ = [
    'FileHandler',
    'DataValidator',
]
