"""Garmin 活动下载页面 — 登录 → 拉取 → 筛选 → 选择 → 下载 FIT 到本地。"""

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from login.garmin_login import garmin_login
from activities.selector import fetch_run_activities, filter_activities
from garminconnect import GarminConnectAuthenticationError, GarminConnectConnectionError

MAX_SELECT = 15
TOKENSTORE = str(ROOT_DIR / "tokens")

TYPE_LABEL = {"running": "跑步", "trail_running": "越野跑", "trail_running_v2": "越野跑v2"}
LABEL_TYPE = {v: k for k, v in TYPE_LABEL.items()}


def main():
    st.set_page_config(page_title="Garmin 活动下载", page_icon=":runner:", layout="wide")

    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            min-width: 480px !important;
            max-width: 480px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -480px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("活动下载")

    _init_state()
    _render_sidebar()
    _render_main()


def _init_state():
    for k, v in {
        "garmin": None,
        "user": "",
        "all_activities": [],
        "filtered": [],
        "selected": [],
        "activity_selector_persist": [],
        "auto_login": True,
        "downloaded_files": [],  # [(filename, bytes), ...]
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ──────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        _render_login()
        if st.session_state.garmin:
            st.divider()
            _render_filters()
            st.divider()
            _render_download()


def _render_login():
    st.header("🔐 登录")

    if st.session_state.garmin:
        st.success(f"已登录: {st.session_state.user}")
        if st.button("退出登录", key="logout_btn"):
            st.session_state.garmin = None
            st.session_state.user = ""
            st.session_state.auto_login = False
            st.rerun()
        return

    # 尝试 token 恢复（仅当用户未主动登出时）
    if not st.session_state.get("auto_login", True):
        st.info("已退出登录，请重新输入账号密码。")
    else:
        try:
            garmin = garmin_login(tokenstore=TOKENSTORE)
            st.session_state.garmin = garmin
            st.session_state.user = garmin.display_name
            st.success("Token 恢复成功，自动登录!")
            st.rerun()
        except GarminConnectAuthenticationError as e:
            st.info(f"Token 已过期或无效，请重新登录: {e}")
        except GarminConnectConnectionError as e:
            st.warning(f"网络问题，无法自动恢复登录: {e}")
        except Exception as e:
            st.warning(f"自动登录失败: {type(e).__name__}: {e}")

    # 首次登录
    with st.form("login"):
        email = st.text_input("Garmin 邮箱")
        password = st.text_input("密码", type="password")
        if st.form_submit_button("登录"):
            if not email or not password:
                st.error("请输入邮箱和密码。")
                return
            try:
                garmin = garmin_login(email=email, password=password, tokenstore=TOKENSTORE)
                st.session_state.garmin = garmin
                st.session_state.user = garmin.display_name
                st.session_state.auto_login = True
                st.success("登录成功!")
                st.rerun()
            except GarminConnectAuthenticationError as e:
                st.error(f"账号或密码错误: {e}")
            except GarminConnectConnectionError as e:
                st.error(f"网络连接失败: {e}")
            except Exception as e:
                st.error(f"登录异常: {type(e).__name__}: {e}")


def _render_filters():
    st.header("📋 筛选")

    if not st.session_state.all_activities:
        with st.spinner("正在拉取活动列表..."):
            st.session_state.all_activities = fetch_run_activities(st.session_state.garmin)

    activities = st.session_state.all_activities
    if not activities:
        st.warning("未找到跑步/越野跑活动。")
        return

    st.caption(f"共 {len(activities)} 条活动")

    # 类型
    type_opts = sorted({a["type"] for a in activities})
    labels = [TYPE_LABEL.get(t, t) for t in type_opts]
    picked = st.multiselect("类型", labels, default=labels)
    picked_types = {LABEL_TYPE.get(l, l) for l in picked}

    # 距离、爬升
    dist = st.slider("距离 (km)", 0, 500, (0, 500))
    elev = st.slider("爬升 (m)", 0, 20000, (0, 20000))

    # 年份、月份
    years = sorted({a["year"] for a in activities if a["year"]})
    year = st.selectbox("年份", ["全部"] + years)
    month = st.selectbox("月份", ["全部"] + list(range(1, 13)))

    # 筛选
    filtered = filter_activities(
        activities,
        dist_min=dist[0], dist_max=dist[1],
        elev_min=elev[0], elev_max=elev[1],
        year=year if year != "全部" else None,
        month=month if month != "全部" else None,
    )
    if picked_types != set(type_opts):
        filtered = [a for a in filtered if a["type"] in picked_types]

    st.session_state.filtered = filtered
    st.caption(f"筛选后: {len(filtered)} 条")


def _render_download():
    st.header("📥 下载")
    sel = st.session_state.selected
    if not sel:
        st.caption("请先在主区域选择活动。")
        return

    st.write(f"已选 {len(sel)} 条")

    if st.button("下载 FIT 文件", type="primary", use_container_width=True):
        _do_download(sel)

    # 显示已下载文件的下载按钮
    downloaded = st.session_state.get("downloaded_files", [])
    if downloaded:
        st.divider()
        st.subheader(f"已下载 {len(downloaded)} 个文件，点击保存到本地：")

        for fname, data in downloaded:
            st.download_button(
                label=f"📄 {fname}",
                data=data,
                file_name=fname,
                mime="application/octet-stream",
                key=f"dl_{fname}",
            )

        # 打包全部下载
        if len(downloaded) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in downloaded:
                    zf.writestr(fname, data)
            zip_buf.seek(0)
            st.download_button(
                label=f"📦 全部下载 (ZIP, {len(downloaded)} 个文件)",
                data=zip_buf,
                file_name="garmin_activities.zip",
                mime="application/zip",
                key="dl_all_zip",
            )


def _do_download(activities):
    garmin = st.session_state.garmin

    progress = st.progress(0)
    status = st.empty()
    downloaded = []
    fail = 0

    for i, a in enumerate(activities, 1):
        status.text(f"[{i}/{len(activities)}] {a['name'][:40]}...")
        try:
            fname = re.sub(r'[<>:"/\\|?*]', '', a["name"])
            fname = re.sub(r'\s+', ' ', fname).strip()[:80]
            fname = f"{fname or 'Unnamed'}_{a['id']}.fit"

            data = garmin.download_activity(a["id"], dl_fmt=garmin.ActivityDownloadFormat.ORIGINAL)
            downloaded.append((fname, data))
            time.sleep(0.5)
        except Exception:
            fail += 1
        progress.progress(i / len(activities))

    status.empty()
    progress.empty()

    st.session_state.downloaded_files = downloaded

    if fail == 0:
        st.success(f"完成: {len(downloaded)} 个文件下载成功")
        st.balloons()
    else:
        st.warning(f"完成: {len(downloaded)} 成功, {fail} 失败")

    st.rerun()


# ── Main Area ────────────────────────────────────────────

def _render_main():
    if not st.session_state.garmin:
        st.info("请先在侧边栏登录 Garmin 账号。")
        return

    filtered = st.session_state.filtered
    if not filtered:
        return

    selected_ids = {a["id"] for a in st.session_state.selected}

    st.title("活动下载")

    # 筛选结果表格（可勾选）
    st.subheader("筛选结果")
    event = st.dataframe(
        filtered,
        column_config={
            "id": st.column_config.NumberColumn("ID"),
            "date": st.column_config.TextColumn("日期"),
            "type": st.column_config.TextColumn("类型"),
            "distance_km": st.column_config.NumberColumn("距离(km)", format="%.1f"),
            "elevation_m": st.column_config.NumberColumn("爬升(m)", format="%.0f"),
            "name": st.column_config.TextColumn("名称"),
        },
        column_order=["date", "type", "distance_km", "elevation_m", "name"],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="activity_table",
    )

    # 将勾选项添加到已选
    selected_rows = event.selection.rows if event else []
    if selected_rows:
        st.caption(f"已勾选 {len(selected_rows)} 条")
        if st.button("将勾选项添加到已选", key="add_from_table"):
            for idx in selected_rows:
                a = filtered[idx]
                if a["id"] not in selected_ids:
                    st.session_state.selected.append(a)
                    selected_ids.add(a["id"])
            st.rerun()

    # 已选列表
    if st.session_state.selected:
        st.subheader(f"已选活动 ({len(st.session_state.selected)}/{MAX_SELECT})")
        for i, a in enumerate(st.session_state.selected):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"**{a['date']}** | {TYPE_LABEL.get(a['type'], a['type'])} | {a['distance_km']:.1f}km | {a['elevation_m']:.0f}m | {a['name']}")
            with c2:
                if st.button("删除", key=f"del_{a['id']}"):
                    st.session_state.selected.pop(i)
                    st.rerun()


if __name__ == "__main__":
    main()
