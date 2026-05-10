"""
分段详情表格渲染模块
"""

def render_split_table_v2(prediction, segments, format_time_fn):
    """渲染分段配速表 - 有CP点按CP点分段，无CP点按5km分段

    规则：
    1. 第一行永远是起点
    2. 中间行：有CP点显示CP点，无CP点按5km间隔
    3. 最后一行永远是终点
    """
    import pandas as pd

    st_subheader = __import__('streamlit').subheader
    st_info = __import__('streamlit').info
    st_dataframe = __import__('streamlit').dataframe
    st_caption = __import__('streamlit').caption

    st_subheader("分段详情")

    if not segments:
        st_info("暂无分段数据")
        return

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
    # 检查终点是否已经在列表中
    if not display_rows or display_rows[-1]['end_km'] < last_seg.end_km - 0.5:
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
                '距离(km)': f"{curr['end_km']:.1f}",
                '海拔(m)': f"{curr['altitude_m']:.0f}",
                '累计时间': format_time_fn(curr['cumulative_time_min']),
                '本段距离(km)': '-',
                '本段爬升(m)': '-',
                '本段下降(m)': '-',
                '本段时间': '-',
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
                '距离(km)': f"{curr['end_km']:.1f}",
                '海拔(m)': f"{curr['altitude_m']:.0f}",
                '累计时间': format_time_fn(curr['cumulative_time_min']),
                '本段距离(km)': f"{segment_dist:.1f}",
                '本段爬升(m)': f"{segment_ascent:.0f}",
                '本段下降(m)': f"{segment_descent:.0f}",
                '本段时间': format_time_fn(segment_time),
                '平均速度': f"{avg_speed:.1f} km/h" if avg_speed > 0 else '-'
            })

    df = pd.DataFrame(data)
    st_dataframe(df, use_container_width=True, hide_index=True)

    # 显示说明
    if use_cp_mode:
        st_caption(f"📍 按CP点分段 (共{len(cp_points)}个CP点)")
    else:
        st_caption("📍 按5km间隔分段 (GPX文件无CP点数据)")
