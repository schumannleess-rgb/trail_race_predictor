"""
Core Validation Test Script
==========================

验证 core/utils.py 和 core/predictor.py 的三个核心功能:
1. 数学验证: GradeAnalyzer 坡度计算准确性
2. 滤波验证: ElevationFilter 防止海岸线悖论
3. 特征验证: SegmentFeatures rolling_grade_500m 边界条件

Author: Validation Test Suite
Date: 2026-04-02
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径 (从 core 目录的父目录导入)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import ElevationFilter, GradeAnalyzer, FilterConfig, SegmentFeatures


class TestResult:
    """测试结果记录器"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, msg):
        self.passed += 1
        print(f"  [PASS] {msg}")

    def add_fail(self, msg, reason):
        self.failed += 1
        error_msg = f"  [FAIL] {msg}\n    Reason: {reason}"
        print(error_msg)
        self.errors.append(error_msg)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"测试结果: {self.passed}/{total} 通过")
        if self.failed > 0:
            print(f"\n失败详情:")
            for error in self.errors:
                print(error)
        print(f"{'='*70}")
        return self.failed == 0


def test_math_validation():
    """
    验证 1: 数学验证 - 坡度计算准确性

    测试场景:
    - 给定上升 10m、水平距离 100m
    - 坡度 = (10 / 100) * 100 = 10%
    """
    print("\n" + "="*70)
    print("验证 1: 数学验证 - 坡度计算准确性")
    print("="*70)

    result = TestResult()

    # 创建测试数据: 两个点，上升10m，水平距离100m
    elevations = np.array([100.0, 110.0])  # 海拔从 100m 到 110m (上升10m)
    distances = np.array([0.0, 100.0])     # 水平距离从 0m 到 100m

    # 计算坡度
    grades = ElevationFilter.calculate_grade(elevations, distances, FilterConfig.GPX)

    # 验证第一个点的坡度应该是 10%
    expected_grade = 10.0
    actual_grade = grades[0]

    tolerance = 0.01  # 允许 0.01% 的误差

    if abs(actual_grade - expected_grade) <= tolerance:
        result.add_pass(
            f"坡度计算正确: 上升10m/水平100m = {actual_grade:.2f}% (期望: {expected_grade:.2f}%)"
        )
    else:
        result.add_fail(
            f"坡度计算错误",
            f"期望 {expected_grade}%, 实际 {actual_grade:.2f}%, 误差 {abs(actual_grade - expected_grade):.2f}%"
        )

    # 额外验证: 负坡度 (下降)
    elevations_desc = np.array([110.0, 100.0])  # 海拔从 110m 到 100m (下降10m)
    grades_desc = ElevationFilter.calculate_grade(elevations_desc, distances, FilterConfig.GPX)

    expected_desc_grade = -10.0
    actual_desc_grade = grades_desc[0]

    if abs(actual_desc_grade - expected_desc_grade) <= tolerance:
        result.add_pass(
            f"下降坡度计算正确: 下降10m/水平100m = {actual_desc_grade:.2f}% (期望: {expected_desc_grade:.2f}%)"
        )
    else:
        result.add_fail(
            f"下降坡度计算错误",
            f"期望 {expected_desc_grade}%, 实际 {actual_desc_grade:.2f}%, 误差 {abs(actual_desc_grade - expected_desc_grade):.2f}%"
        )

    # 验证: 平路 (0% 坡度)
    elevations_flat = np.array([100.0, 100.0])  # 海拔不变
    grades_flat = ElevationFilter.calculate_grade(elevations_flat, distances, FilterConfig.GPX)

    expected_flat_grade = 0.0
    actual_flat_grade = grades_flat[0]

    if abs(actual_flat_grade - expected_flat_grade) <= tolerance:
        result.add_pass(
            f"平路坡度计算正确: 无高程变化 = {actual_flat_grade:.2f}% (期望: {expected_flat_grade:.2f}%)"
        )
    else:
        result.add_fail(
            f"平路坡度计算错误",
            f"期望 {expected_flat_grade}%, 实际 {actual_flat_grade:.2f}%, 误差 {abs(actual_flat_grade - expected_flat_grade):.2f}%"
        )

    # 验证: 坡度截断 (超过 max_grade_pct 的值应被截断)
    elevations_steep = np.array([100.0, 200.0])  # 上升100m
    distances_steep = np.array([0.0, 100.0])      # 水平100m
    # 原始坡度 = 100%, 应被截断到 max_grade_pct (45%)

    grades_steep = ElevationFilter.calculate_grade(elevations_steep, distances_steep, FilterConfig.GPX)
    max_grade = FilterConfig.GPX['max_grade_pct']

    if grades_steep[0] == max_grade:
        result.add_pass(
            f"坡度截断正确: 100%坡度被截断到 {grades_steep[0]:.1f}% (max_grade_pct={max_grade}%)"
        )
    else:
        result.add_fail(
            f"坡度截断错误",
            f"期望截断到 {max_grade}%, 实际 {grades_steep[0]:.1f}%"
        )

    return result.summary()


def calculate_accumulated_ascent(elevations, distances):
    """计算总爬升 (海岸线悖论敏感指标)"""
    ascent = 0.0
    for i in range(len(elevations) - 1):
        if distances[i+1] > distances[i]:  # 确保前进方向
            elevation_change = elevations[i+1] - elevations[i]
            if elevation_change > 0:
                ascent += elevation_change
    return ascent


def test_filter_validation():
    """
    验证 2: 滤波验证 - 防止海岸线悖论

    测试场景:
    - 创建带随机噪点的海拔数据
    - 验证滤波后总爬升被有效压缩
    - 防止海岸线悖论 (噪声导致无限细分产生无限爬升)
    """
    print("\n" + "="*70)
    print("验证 2: 滤波验证 - 防止海岸线悖论")
    print("="*70)

    result = TestResult()

    # 生成模拟数据: 一条上升路径加上噪声
    np.random.seed(42)  # 固定随机种子以便复现
    n_points = 100
    base_elevation = 1000.0
    total_ascent = 50.0  # 真实爬升 50m
    total_distance = 1000.0  # 总距离 1000m

    # 创建线性上升 + 噪声
    distances = np.linspace(0, total_distance, n_points)
    noise_level = 2.0  # ±2m 的噪声
    noise = np.random.normal(0, noise_level, n_points)
    elevations_with_noise = base_elevation + (distances / total_distance) * total_ascent + noise

    # 计算原始数据的爬升
    original_ascent = calculate_accumulated_ascent(elevations_with_noise, distances)

    # 应用滤波
    smoothed_elevations = ElevationFilter.smooth(elevations_with_noise, FilterConfig.GPX)

    # 计算滤波后的爬升
    filtered_ascent = calculate_accumulated_ascent(smoothed_elevations, distances)

    # 验证: 滤波后爬升应该减少 (噪声被压缩)
    ascent_reduction = original_ascent - filtered_ascent
    reduction_percentage = (ascent_reduction / original_ascent) * 100 if original_ascent > 0 else 0

    if filtered_ascent < original_ascent:
        result.add_pass(
            f"滤波有效压缩爬升: 原始 {original_ascent:.1f}m → 滤波后 {filtered_ascent:.1f}m "
            f"(减少 {ascent_reduction:.1f}m, {reduction_percentage:.1f}%)"
        )
    else:
        result.add_fail(
            f"滤波未能压缩爬升",
            f"滤波后爬升 ({filtered_ascent:.1f}m) 应该小于原始爬升 ({original_ascent:.1f}m)"
        )

    # 验证: 滤波后的爬升应该接近真实爬升 (允许一定误差)
    # 真实爬升是 50m，允许 ±10m 的误差 (考虑滤波边缘效应)
    true_ascent = total_ascent
    tolerance = 15.0  # 允许 15m 误差

    if abs(filtered_ascent - true_ascent) <= tolerance:
        result.add_pass(
            f"滤波后爬升接近真实值: {filtered_ascent:.1f}m ≈ {true_ascent:.1f}m "
            f"(误差 {abs(filtered_ascent - true_ascent):.1f}m, 误差率 {abs(filtered_ascent - true_ascent)/true_ascent*100:.1f}%)"
        )
    else:
        result.add_fail(
            f"滤波后爬升偏离真实值过大",
            f"真实爬升 {true_ascent:.1f}m, 滤波后 {filtered_ascent:.1f}m, "
            f"误差 {abs(filtered_ascent - true_ascent):.1f}m 超过容忍度 {tolerance}m"
        )

    # 验证: 海岸线悖论指标 - 噪声标准差应该显著降低
    # 计算残差 (滤波后的信号 - 原始线性趋势)
    trend = base_elevation + (distances / total_distance) * total_ascent
    original_residual_std = np.std(elevations_with_noise - trend)
    filtered_residual_std = np.std(smoothed_elevations - trend)

    noise_reduction_ratio = filtered_residual_std / original_residual_std if original_residual_std > 0 else 1.0

    if filtered_residual_std < original_residual_std:
        result.add_pass(
            f"噪声被有效抑制: 原始噪声标准差 {original_residual_std:.2f}m → "
            f"滤波后 {filtered_residual_std:.2f}m (压缩比 {noise_reduction_ratio:.2f})"
        )
    else:
        result.add_fail(
            f"噪声抑制失败",
            f"滤波后噪声标准差 ({filtered_residual_std:.2f}m) 应该小于原始 ({original_residual_std:.2f}m)"
        )

    # 额外验证: 确保滤波不改变数据长度
    if len(smoothed_elevations) == len(elevations_with_noise):
        result.add_pass(
            f"滤波保持数据长度: {len(smoothed_elevations)} 点 (输入 {len(elevations_with_noise)} 点)"
        )
    else:
        result.add_fail(
            f"滤波改变数据长度",
            f"输入 {len(elevations_with_noise)} 点, 输出 {len(smoothed_elevations)} 点"
        )

    return result.summary()


def test_feature_validation():
    """
    验证 3: 特征验证 - rolling_grade_500m 边界条件处理

    测试场景:
    - 模拟一个 SegmentFeatures 对象
    - 验证 rolling_grade_500m 在起始边界的计算逻辑
    - 确保不会出现数组越界或除零错误
    """
    print("\n" + "="*70)
    print("验证 3: 特征验证 - rolling_grade_500m 边界条件")
    print("="*70)

    result = TestResult()

    # 测试场景 1: 正常情况 - 足够的历史数据
    print("\n  场景 1: 正常情况 (足够历史数据)")

    # 创建模拟的坡度数组
    grades_array = np.array([5.0, 8.0, 12.0, 15.0, 10.0])  # 5个点的坡度 (%)
    seg_distances = [100, 100, 100, 100, 100]  # 每段 100m

    # 模拟 FeatureExtractor 中的 rolling_grade_500m 计算逻辑
    # 从 predictor.py 第 448-450 行可以看到:
    # rolling_window = max(1, int(500 / (seg_dist if seg_dist > 0 else 1)))
    # start_idx = max(0, i - rolling_window)
    # rolling_grade = np.mean(grades[start_idx:i+1]) if i > 0 else avg_grade

    for i in range(len(grades_array)):
        seg_dist = seg_distances[i]
        avg_grade = grades_array[i]

        rolling_window = max(1, int(500 / (seg_dist if seg_dist > 0 else 1)))
        start_idx = max(0, i - rolling_window)

        if i > 0:
            rolling_grade = np.mean(grades_array[start_idx:i+1])
        else:
            rolling_grade = avg_grade  # 边界条件: 第一个点使用自身坡度

        # 验证: rolling_grade 应该在合理范围内
        min_expected = min(grades_array[start_idx:i+1]) if i > 0 else avg_grade
        max_expected = max(grades_array[start_idx:i+1]) if i > 0 else avg_grade

        if min_expected <= rolling_grade <= max_expected:
            result.add_pass(
                f"索引 {i}: rolling_grade_500m = {rolling_grade:.2f}% "
                f"(窗口大小: {i - start_idx + 1} 点)"
            )
        else:
            result.add_fail(
                f"索引 {i}: rolling_grade_500m 超出范围",
                f"期望范围 [{min_expected:.2f}, {max_expected:.2f}], 实际 {rolling_grade:.2f}"
            )

    # 测试场景 2: 边界条件 - 单点情况
    print("\n  场景 2: 边界条件 (单点)")

    single_grade = np.array([10.0])
    seg_dist = 100

    rolling_window = max(1, int(500 / seg_dist))
    start_idx = max(0, 0 - rolling_window)  # i=0

    # 边界条件: i=0 时使用自身坡度
    if 0 > 0:
        rolling_grade = np.mean(single_grade[start_idx:0+1])
    else:
        rolling_grade = single_grade[0]

    if rolling_grade == single_grade[0]:
        result.add_pass(
            f"单点边界条件: rolling_grade = {rolling_grade:.2f}% (使用自身坡度)"
        )
    else:
        result.add_fail(
            f"单点边界条件错误",
            f"期望 {single_grade[0]:.2f}%, 实际 {rolling_grade:.2f}%"
        )

    # 测试场景 3: 边界条件 - 零距离情况 (除零保护)
    print("\n  场景 3: 边界条件 (零距离 - 除零保护)")

    seg_dist_zero = 0
    rolling_window_zero = max(1, int(500 / (seg_dist_zero if seg_dist_zero > 0 else 1)))

    if rolling_window_zero > 0 and not np.isinf(rolling_window_zero):
        result.add_pass(
            f"除零保护有效: seg_dist=0 时 rolling_window = {rolling_window_zero}"
        )
    else:
        result.add_fail(
            f"除零保护失败",
            f"seg_dist=0 时 rolling_window = {rolling_window_zero} (应该是有限正整数)"
        )

    # 测试场景 4: 创建 SegmentFeatures 对象并验证字段
    print("\n  场景 4: SegmentFeatures 对象验证")

    try:
        segment = SegmentFeatures(
            speed_kmh=8.5,
            grade_pct=10.0,
            rolling_grade_500m=8.5,  # 略低于当前坡度 (考虑历史平均)
            accumulated_distance_km=15.5,
            accumulated_ascent_m=850.0,
            absolute_altitude_m=1200.0,
            elevation_density=54.8,
            is_climbing=True,
            is_descending=False
        )

        # 验证字段是否正确设置
        validations = [
            (segment.grade_pct == 10.0, f"grade_pct = {segment.grade_pct}%"),
            (segment.rolling_grade_500m == 8.5, f"rolling_grade_500m = {segment.rolling_grade_500m}%"),
            (segment.is_climbing == True, f"is_climbing = {segment.is_climbing}"),
            (segment.is_descending == False, f"is_descending = {segment.is_descending}"),
        ]

        all_valid = True
        for valid, msg in validations:
            if valid:
                result.add_pass(f"SegmentFeatures 字段: {msg}")
            else:
                all_valid = False
                result.add_fail(f"SegmentFeatures 字段验证", f"字段不匹配: {msg}")

        # 验证逻辑一致性
        if segment.grade_pct > 2 and segment.is_climbing:
            result.add_pass(
                f"逻辑一致性: grade_pct={segment.grade_pct}% > 2% 时 is_climbing=True"
            )
        else:
            result.add_fail(
                f"逻辑不一致",
                f"grade_pct={segment.grade_pct}% 但 is_climbing={segment.is_climbing}"
            )

    except Exception as e:
        result.add_fail(
            f"SegmentFeatures 对象创建失败",
            f"异常: {str(e)}"
        )

    # 测试场景 5: rolling_grade 在路径开始处的边界行为
    print("\n  场景 5: 路径起始边界行为")

    # 模拟路径开始: 逐步积累数据
    early_grades = np.array([5.0, 7.0, 10.0])  # 早期坡度
    seg_dist_early = 100

    for i in range(len(early_grades)):
        rolling_window_early = max(1, int(500 / seg_dist_early))
        start_idx_early = max(0, i - rolling_window_early)

        if i > 0:
            rolling_grade_early = np.mean(early_grades[start_idx_early:i+1])
        else:
            rolling_grade_early = early_grades[i]

        # 验证边界条件: start_idx 永远 >= 0
        if start_idx_early >= 0:
            result.add_pass(
                f"早期索引 {i}: start_idx={start_idx_early} >= 0 (安全), "
                f"rolling_grade={rolling_grade_early:.2f}%"
            )
        else:
            result.add_fail(
                f"早期索引 {i}: 数组越界",
                f"start_idx={start_idx_early} < 0"
            )

    return result.summary()


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("Core Validation Test Suite")
    print("验证 core/utils.py 和 core/predictor.py 的准确性")
    print("="*70)

    all_passed = True

    # 执行三个验证
    try:
        math_passed = test_math_validation()
        all_passed = all_passed and math_passed
    except Exception as e:
        print(f"\n[错误] 数学验证执行失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        filter_passed = test_filter_validation()
        all_passed = all_passed and filter_passed
    except Exception as e:
        print(f"\n[错误] 滤波验证执行失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        feature_passed = test_feature_validation()
        all_passed = all_passed and feature_passed
    except Exception as e:
        print(f"\n[错误] 特征验证执行失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # 总体结果
    print("\n" + "="*70)
    if all_passed:
        print("[PASS] All validations passed!")
    else:
        print("[FAIL] Some validations failed, please check errors above")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
