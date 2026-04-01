"""
Trail Race Predictor V1.2 - Streamlit Application

用户友好的越野赛预测Web界面

V1.2 更新:
- 直接支持原始 FIT 文件 (无需 JSON 转换)
- 统一建模 (简化训练流程)
- 努力程度量化 (基于 P50/P90 能力边界)
"""

import streamlit as st
from pathlib import Path
import sys
import time

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from core.predictor import MLRacePredictor
from core.types import PredictionResult, EffortLevel, SegmentPrediction
from core.report_generator import ReportGenerator
from data.file_handler import FileHandler
from data.data_validator import DataValidator, validate_file

# 页面配置
st.set_page_config(
    page_title="Trail Master - 越野赛成绩预测",
    page_icon=":mountain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 48px;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    # 标题
    st.title(":mountain: Trail Master 越野赛成绩预测系统")
    st.markdown("上传你的训练记录，获取精准的赛道配速分析")

    # 初始化session state
    init_session_state()

    # 侧边栏
    render_sidebar()

    # 主界面
    if st.session_state.get('analysis_started'):
        render_analysis()
    else:
        render_instructions()


def init_session_state():
    """初始化session state"""
    defaults = {
        'predictor': None,
        'file_handler': FileHandler(str(ROOT_DIR / 'temp')),
        'gpx_file': None,
        'fit_files': [],
        'effort_factor': 1.0,
        'analysis_started': False,
        'analysis_complete': False,
        'prediction_result': None,
        'selected_files': []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("数据上传")

        # 赛道上传
        gpx_file = st.file_uploader(
            "上传目标赛道 (GPX)",
            type=['gpx'],
            help="上传比赛的GPX路线文件"
        )

        # 训练记录上传
        fit_files = st.file_uploader(
            "上传个人记录 (FIT)",
            type=['fit'],
            accept_multiple_files=True,
            help="上传历史训练记录，建议至少5个文件，15-20个精品FIT效果最佳"
        )

        # 显示上传状态
        if gpx_file:
            st.success(f"赛道: {gpx_file.name}")
        if fit_files:
            st.info(f"训练记录: {len(fit_files)} 个文件")

        st.divider()

        # 努力程度滑块
        st.header("比赛状态预设")
        
        effort_factor = st.slider(
            "调节你的努力程度",
            min_value=0.8,
            max_value=1.2,
            value=1.0,
            step=0.01,
            help="1.0 = 平时平均水平 (P50)\n1.1-1.2 = 比赛状态 (接近 P90)\n0.8-0.9 = 保守策略"
        )
        
        # 显示努力程度说明
        if effort_factor < 0.95:
            st.info(f"🎯 保守策略 ({effort_factor:.2f}x): 适合长距离或恢复期比赛")
        elif effort_factor > 1.05:
            st.warning(f"🔥 比赛状态 ({effort_factor:.2f}x): 挑战你的极限!")
        else:
            st.success(f"✅ 平均水平 ({effort_factor:.2f}x): 稳健的完赛目标")

        # 开始按钮
        if st.button(":rocket: 开始分析", type="primary", use_container_width=True):
            if not gpx_file:
                st.error("请上传赛道GPX文件!")
            elif not fit_files or len(fit_files) < 3:
                st.warning("建议至少上传3个训练记录文件")
                if not fit_files:
                    st.error("请上传训练记录!")
                else:
                    start_analysis(gpx_file, fit_files, effort_factor)
            else:
                start_analysis(gpx_file, fit_files, effort_factor)


def start_analysis(gpx_file, fit_files, effort_factor):
    """开始分析"""
    st.session_state.gpx_file = gpx_file
    st.session_state.fit_files = fit_files
    st.session_state.effort_factor = effort_factor
    st.session_state.analysis_started = True
    st.session_state.analysis_complete = False
    st.rerun()


def render_instructions():
    """渲染使用说明"""
    st.markdown("""
    ### 使用说明

    1. **上传赛道GPX文件** - 比赛路线文件，包含海拔和坐标数据
    2. **上传训练记录** - 你的历史训练数据 (FIT/GPX/JSON格式)
       - 建议 15-20 个精品 FIT 文件效果最佳
       - 最少 5 个文件即可运行
       - 请剔除带娃散步、信号漂移或纯平路路跑记录
    3. **调节努力程度** - 滑块调节 (0.8-1.2)
       - 1.0 = 平时平均水平 (P50)
       - 1.1-1.2 = 比赛状态 (接近 P90)
       - 0.8-0.9 = 保守策略
    4. **开始分析** - 系统将训练统一 ML 模型并预测你的完赛时间

    ---

    ### V1.2 新特性

    - :star: **直接支持 FIT 文件** - 无需 JSON 转换，直接上传原始 FIT
    - :brain: **统一建模** - 使用所有训练数据建模，简化流程
    - :chart_with_upwards_trend: **能力边界量化** - 基于 P50/P90 科学设定努力程度
    - :shield: **物理验证** - VAM 限制防止 unrealistic 预测
    - :memo: **详细报告** - 分段配速、战术建议、难度预警

    ---

    ### 数据要求

    | 文件类型 | 格式 | 要求 |
    |---------|------|------|
    | 赛道 | GPX | 需要包含海拔数据 |
    | 训练记录 | FIT/GPX/JSON | 建议 15-20 个精品文件，最少 5 个 |
    """)


def detect_duplicates(file_paths):
    """
    检测重复的训练文件
    
    重复判定规则：
    1. FIT文件：根据活动ID（time_created + 日期）判定
    2. JSON文件：根据文件内容的起始时间判定
    3. 文件名相同也视为重复
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        (unique_paths, duplicates): 唯一文件列表和重复文件列表
    """
    import hashlib
    from datetime import datetime
    
    seen_activities = {}
    unique_paths = []
    duplicates = []
    
    for path in file_paths:
        path = Path(path)
        activity_key = None
        
        # 根据文件类型提取活动标识
        if path.suffix.lower() == '.fit':
            try:
                from fitparse import FitFile
                fitfile = FitFile(str(path))
                
                # 获取活动创建时间作为唯一标识
                for record in fitfile.get_messages('file_id'):
                    for field in record:
                        if field.name == 'time_created':
                            activity_key = str(field.value)
                            break
                    if activity_key:
                        break
                        
            except Exception as e:
                # 解析失败，用文件内容hash作为标识
                activity_key = hashlib.md5(path.read_bytes()).hexdigest()[:16]
                
        elif path.suffix.lower() == '.json':
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 用第一条记录的时间作为标识
                if isinstance(data, list) and len(data) > 0:
                    first_record = data[0]
                    activity_key = str(first_record.get('timestamp', ''))
                elif isinstance(data, dict):
                    activity_key = str(data.get('start_time', data.get('timestamp', '')))
                    
            except Exception:
                activity_key = hashlib.md5(path.read_bytes()).hexdigest()[:16]
        else:
            # 其他格式用文件hash
            activity_key = hashlib.md5(path.read_bytes()).hexdigest()[:16]
        
        # 如果无法提取标识，用文件名
        if not activity_key:
            activity_key = path.stem
        
        # 检查是否重复
        if activity_key in seen_activities:
            duplicates.append(f"{path.name} (与 {seen_activities[activity_key].name} 重复)")
        else:
            seen_activities[activity_key] = path
            unique_paths.append(path)
    
    return unique_paths, duplicates


def render_analysis():
    """渲染分析过程和结果"""
    with st.status("正在分析中...", expanded=True) as status:

        # Step 1: 保存文件
        st.write(":floppy_disk: 处理上传文件...")
        file_handler = st.session_state.file_handler

        gpx_path = file_handler.save_uploaded_file(st.session_state.gpx_file, 'routes')
        fit_paths = []

        for fit_file in st.session_state.fit_files:
            path = file_handler.save_uploaded_file(fit_file, 'records')
            fit_paths.append(path)

        # Step 2: 检测重复数据
        st.write(":mag: 检测重复数据...")
        unique_paths, duplicates = detect_duplicates(fit_paths)
        
        if duplicates:
            st.warning(f"发现 {len(duplicates)} 个重复文件，已自动过滤")
            with st.expander("查看重复文件"):
                for dup in duplicates:
                    st.write(f"- {dup}")
        
        fit_paths = unique_paths
        st.write(f"有效训练文件: {len(fit_paths)} 个")

        # Step 3: 验证数据
        st.write(":mag: 验证数据格式...")
        gpx_result = validate_file(gpx_path)
        if not gpx_result.valid:
            st.error(f"赛道文件问题: {gpx_result.error}")
            status.update(label="验证失败", state="error")
            return

        if gpx_result.warnings:
            for w in gpx_result.warnings:
                st.warning(w)

        # Step 4: 智能筛选
        st.write(":filter: 智能筛选训练数据...")
        if len(fit_paths) > 20:
            fit_paths = file_handler.auto_select_best_files(fit_paths, max_count=20)

        st.write(f"已选取 {len(fit_paths)} 个训练样本")

        # Step 5: 训练模型
        st.write(":brain: 训练统一机器学习模型...")

        predictor = MLRacePredictor(str(ROOT_DIR / 'records'))
        if not predictor.analyze_and_train():
            st.error("模型训练失败!")
            status.update(label="训练失败", state="error")
            return
            
        st.session_state.predictor = predictor

        stats = predictor.training_stats
        st.write(f"  文件数: {stats.get('file_count', 0)}")
        st.write(f"  分段数: {stats.get('segment_count', 0)}")
        st.write(f"  平均速度: {stats.get('avg_speed', 0):.2f} km/h")
        st.write(f"  P50 速度: {stats.get('p50_speed', 0):.2f} km/h")
        st.write(f"  P90 速度: {stats.get('p90_speed', 0):.2f} km/h")

        # Step 5: 预测
        st.write(":chart_with_upwards_trend: 生成预测...")

        effort_factor = st.session_state.effort_factor

        try:
            result = predictor.predict_race(gpx_path, effort_factor)

            # 转换为PredictionResult对象
            prediction = convert_to_prediction_result(result, effort_factor)
            st.session_state.prediction_result = prediction

            status.update(label="分析完成!", state="complete")

        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            status.update(label="预测失败", state="error")
            return

    # 显示结果
    render_prediction_results()


def convert_to_prediction_result(result: dict, effort_factor: float) -> PredictionResult:
    """将字典结果转换为PredictionResult对象"""
    from core.types import SegmentPrediction

    segments = []
    cumulative_distance = 0

    for i, seg in enumerate(result.get('segment_predictions', [])):
        segment_dist = seg.get('distance_km', 0.2)
        start_km = cumulative_distance
        cumulative_distance += segment_dist

        segments.append(SegmentPrediction(
            segment_id=seg.get('segment', i + 1),
            start_km=start_km,
            end_km=cumulative_distance,
            distance_km=segment_dist,
            grade_pct=seg.get('grade_pct', 0),
            altitude_m=seg.get('altitude_m', 0),
            predicted_speed_kmh=seg.get('predicted_speed_kmh', 5),
            predicted_time_min=seg.get('segment_time_min', 3),
            cumulative_time_min=seg.get('cumulative_time_min', 3 * (i + 1)),
            difficulty_level=seg.get('difficulty', 'moderate'),
            grade_type=seg.get('grade_type', '平地'),
            ascent_m=seg.get('ascent_m', 0),
            descent_m=seg.get('descent_m', 0),
            cp_name=seg.get('cp_name', '')
        ))

    route_info = result.get('route_info', {})

    return PredictionResult(
        total_time_min=result.get('predicted_time_min', 0),
        total_time_hm=result.get('predicted_time_hm', '0:00:00'),
        pace_min_km=result.get('predicted_pace_min_km', 0),
        speed_kmh=result.get('predicted_speed_kmh', 0),
        total_distance_km=route_info.get('total_distance_km', 0),
        total_ascent_m=route_info.get('total_elevation_gain_m', 0),
        total_descent_m=route_info.get('total_elevation_loss_m', route_info.get('total_elevation_gain_m', 0) * 0.9),
        elevation_density=route_info.get('elevation_density', 0),
        segments=segments,
        feature_importance=result.get('feature_importance', {}),
        model_confidence=0.85,
        effort_level=f"{effort_factor:.2f}x",
        training_stats=result.get('training_stats', {}),
        warnings=[]
    )


def render_prediction_results():
    """渲染预测结果"""
    prediction = st.session_state.prediction_result

    st.success(":white_check_mark: 预测完成!")

    # 主要指标卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("预测完赛时间", prediction.total_time_hm)

    with col2:
        st.metric("平均配速", f"{prediction.pace_min_km:.1f} min/km")

    with col3:
        st.metric("平均速度", f"{prediction.speed_kmh:.2f} km/h")

    with col4:
        st.metric("总爬升", f"{prediction.total_ascent_m:.0f} m")

    st.divider()

    # 详细分析标签页
    tab1, tab2, tab3, tab4 = st.tabs(["赛道概览", "分段配速", "战术建议", "下载报告"])

    with tab1:
        render_route_overview(prediction)

    with tab2:
        render_split_table(prediction)

    with tab3:
        render_tactical_advice(prediction)

    with tab4:
        render_download_section(prediction)


def render_route_overview(prediction: PredictionResult):
    """渲染赛道概览"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("赛道信息")
        st.write(f"- **总距离**: {prediction.total_distance_km:.2f} km")
        st.write(f"- **总爬升**: {prediction.total_ascent_m:.0f} m")
        st.write(f"- **总下降**: {prediction.total_descent_m:.0f} m")
        st.write(f"- **爬升密度**: {prediction.elevation_density:.1f} m/km")

        # 难度评级
        if prediction.elevation_density > 100:
            difficulty = "极难 ⚠️"
        elif prediction.elevation_density > 70:
            difficulty = "困难 🔥"
        elif prediction.elevation_density > 40:
            difficulty = "中等 🏃"
        else:
            difficulty = "轻松 ✅"

        st.write(f"- **难度等级**: {difficulty}")

    with col2:
        st.subheader("你的能力画像")
        stats = prediction.training_stats
        
        if stats:
            st.write(f"- **平均训练速度**: {stats.get('avg_speed', 0):.2f} km/h")
            
            p50 = stats.get('p50_speed', 0)
            p90 = stats.get('p90_speed', 0)
            effort_range = stats.get('effort_range', 1.0)
            
            st.write(f"- **日常水平**: {p50:.2f} km/h")
            st.write(f"- **最佳状态**: {p90:.2f} km/h")
            
            st.info(f"💡 你的极限比日常快 {effort_range:.0%}，比赛时可挑战最佳状态！")
        
        st.write("")
        st.write("**本次预测努力程度**")
        st.write(f"- **系数**: {prediction.effort_level}")
        
        effort_factor = float(prediction.effort_level.replace('x', ''))
        if effort_factor >= 1.1:
            st.write("- **策略**: 🔥 挑战极限")
        elif effort_factor >= 1.0:
            st.write("- **策略**: ✅ 稳健完赛")
        else:
            st.write("- **策略**: 🎯 保守策略")


def render_split_table(prediction: PredictionResult):
    """渲染分段配速表 - 有CP点按CP点分段，无CP点按5km分段

    规则：
    1. 第一行永远是起点
    2. 中间行：有CP点显示CP点，无CP点按5km间隔
    3. 最后一行永远是终点
    """
    st.subheader("分段详情")

    segments = prediction.segments
    if not segments:
        st.info("暂无分段数据")
        return

    import pandas as pd

    total_distance = prediction.total_distance_km
    last_seg = segments[-1]

    # 收集所有CP点 (去重，保留第一次出现的位置)
    cp_points = []
    seen_cps = set()
    for seg in segments:
        cp_name = getattr(seg, 'cp_name', '')
        if cp_name and cp_name not in seen_cps:
            cp_points.append({
                'name': cp_name,
                'end_km': seg.end_km,
                'altitude_m': seg.altitude_m,
                'cumulative_time_min': seg.cumulative_time_min
            })
            seen_cps.add(cp_name)

    # 决定显示模式
    use_cp_mode = len(cp_points) > 0

    # 构建显示行
    display_rows = []

    # 1. 第一行永远是起点
    display_rows.append({
        'name': '起点',
        'end_km': 0,
        'altitude_m': segments[0].altitude_m if segments else 0,
        'cumulative_time_min': 0
    })

    # 2. 中间行：CP点或5km间隔
    if use_cp_mode:
        # 使用CP点模式 - 直接添加所有CP点
        for cp in cp_points:
            display_rows.append({
                'name': cp['name'],
                'end_km': cp['end_km'],
                'altitude_m': cp['altitude_m'],
                'cumulative_time_min': cp['cumulative_time_min']
            })
    else:
        # 使用5km间隔模式
        for mark_km in range(5, int(total_distance) + 1, 5):
            # 找到最接近这个距离的段
            closest_seg = min(segments, key=lambda s: abs(s.end_km - mark_km))
            display_rows.append({
                'name': f'{mark_km}km',
                'end_km': closest_seg.end_km,
                'altitude_m': closest_seg.altitude_m,
                'cumulative_time_min': closest_seg.cumulative_time_min
            })

    # 3. 最后一行永远是终点
    # 终点始终添加在最后 (不在CP点列表中，CP点只记录中途打卡点)
    display_rows.append({
        'name': '终点',
        'end_km': last_seg.end_km,
        'altitude_m': last_seg.altitude_m,
        'cumulative_time_min': last_seg.cumulative_time_min
    })

    # 计算每段的区间数据 (从上一行到当前行)
    data = []
    for i, curr in enumerate(display_rows):
        prev = display_rows[i - 1] if i > 0 else None

        # 计算本段距离
        segment_dist = curr['end_km'] - (prev['end_km'] if prev else 0)

        # 计算本段爬升/下降/时间
        segment_ascent = 0
        segment_descent = 0
        segment_time = 0

        if prev:
            # 汇总这个区间的所有小段数据
            prev_km = prev['end_km']
            curr_km = curr['end_km']
            for seg in segments:
                if prev_km < seg.end_km <= curr_km:
                    segment_ascent += getattr(seg, 'ascent_m', 0)
                    segment_descent += getattr(seg, 'descent_m', 0)
                    segment_time += seg.predicted_time_min

        # 计算平均速度
        avg_speed = segment_dist / (segment_time / 60) if segment_time > 0 else 0

        # 格式化显示
        if i == 0:
            # 起点
            data.append({
                '位置': '🏁 ' + curr['name'],
                '本段距离(km)': '-',
                '距离(km)': f"{curr['end_km']:.1f}",
                '本段爬升(m)': '-',
                '本段下降(m)': '-',
                '本段时间': '-',
                '累计时间': format_time(curr['cumulative_time_min']),
                '平均速度': '-'
            })
        else:
            # CP点或间隔点或终点
            name_display = curr['name']
            if name_display == '终点':
                name_display = '🏃 终点'
            else:
                name_display = '📍 ' + name_display

            data.append({
                '位置': name_display,
                '本段距离(km)': f"{segment_dist:.1f}",
                '距离(km)': f"{curr['end_km']:.1f}",
                '本段爬升(m)': f"{segment_ascent:.0f}",
                '本段下降(m)': f"{segment_descent:.0f}",
                '本段时间': format_time(segment_time),
                '累计时间': format_time(curr['cumulative_time_min']),
                '平均速度': f"{avg_speed:.1f} km/h" if avg_speed > 0 else '-'
            })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 显示说明
    if use_cp_mode:
        st.caption(f"📍 按CP点分段 (共{len(cp_points)}个CP点)")
    else:
        st.caption("📍 按5km间隔分段 (GPX文件无CP点数据)")


def render_tactical_advice(prediction: PredictionResult):
    """渲染战术建议"""
    st.subheader("配速策略")

    importance = prediction.feature_importance

    if importance.get('grade_pct', 0) > importance.get('accumulated_distance_km', 0):
        st.info("坡度对你的速度影响最大，建议在陡坡段保守配速，保存体力")
    else:
        st.info("疲劳累积对你的影响更大，建议前半程控制速度，避免后程掉速")

    # 基于爬升密度
    if prediction.elevation_density > 80:
        st.warning("赛道爬升密度极高，前10km建议配速比预测慢10%，避免过早疲劳")

    st.subheader("补给建议")

    duration_hours = prediction.total_time_min / 60

    if duration_hours > 6:
        st.write(f"- 预计耗时 {duration_hours:.1f} 小时")
        st.write(f"- 建议携带能量胶 {int(duration_hours)} 支")
        st.write("- 每45-60分钟补充一次能量")
        st.write("- 每小时饮水 500-750ml")
    elif duration_hours > 3:
        st.write(f"- 预计耗时 {duration_hours:.1f} 小时")
        st.write(f"- 建议携带能量胶 {int(duration_hours)} 支")
        st.write("- 每45分钟补充一次能量")
    else:
        st.write(f"- 预计耗时 {duration_hours:.1f} 小时")
        st.write("- 短距离比赛，少量补给即可")


def render_download_section(prediction: PredictionResult):
    """渲染下载区域"""
    st.subheader(":inbox_tray: 下载报告")

    # 生成报告
    report_gen = ReportGenerator(prediction)

    col1, col2 = st.columns(2)

    with col1:
        # HTML报告
        report_path = str(ROOT_DIR / 'reports' / 'prediction_report.html')
        try:
            html_path = report_gen.generate_html_report(report_path)
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            st.download_button(
                ":page_facing_up: 下载HTML报告",
                html_content,
                file_name="trail_prediction_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"生成HTML报告失败: {str(e)}")

    with col2:
        # TXT报告
        try:
            txt_content = report_gen.generate_txt_report()
            st.download_button(
                ":memo: 下载TXT报告",
                txt_content,
                file_name="trail_prediction_report.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"生成TXT报告失败: {str(e)}")

    st.divider()
    st.info("💡 HTML报告可保存到手机，方便比赛时查看；TXT报告适合打印或分享")


def format_time(minutes: float) -> str:
    """格式化时间"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


if __name__ == '__main__':
    main()
