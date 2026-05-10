# 越野赛时间预测器

**当前版本：v1.2.4**

---

## 版本历史

| 版本 | 状态 | 说明 |
|------|------|------|
| v1.2.2 | ✅ 稳定 | 当前 GitHub 发布版（模块化结构：core / data / scripts / ui） |
| v1.2.4 | ✅ 已发布 | 单应用整合 + Garmin Connect 活动下载 + GPX 数据处理优化 |

---

## v1.2.4 更新内容

### 1. 单应用整合
- 删除 `pages/` 目录，统一在 `app.py` 内通过 `st.radio` 切换"越野赛预测"和"活动下载"两个视图
- 状态持久化：解决 tab 切换后 widget 状态丢失问题（GPX/FIT 文件、活动选择）

### 2. 活动下载页面优化
- 删除重复的 multiselect 选择器，统一使用表格行选择
- 简化页面标题和布局
- 登录错误分类提示（Token 过期 / 网络 / 其他）
- 新增"退出登录"按钮，登出后不再自动重连

### 3. GPX 数据处理优化
- **滤波器升级**：GPX 海拔滤波从 Savitzky-Golay (`window=7`) 改为 **高斯滤波 (`σ=3.0`)**，避免 SG 滤波在起伏地形上的过冲放大虚假爬升
- **爬升计算修正**：总爬升/总下降直接从平滑后的海拔差累加，不再通过"坡度→截断→反推"的绕弯子方式
- **数据一致性**：`route_info` 的总爬升/总下降从 `segments` 汇总，确保和分段详情严格一致
- **Split Points**：新增观测点区间预计算，有 CP 点按 CP 点精确分段，无 CP 点按 5km 间隔分段。距离/爬升/下降直接从原始轨迹点累加，不再被 0.2km 小段边界量化

### 4. 预测结果显示优化
- `render_split_table()` 直接显示预计算的 `split_points`，逻辑大幅简化
- 观测点位置精确（不再绑定到 0.2km 的整数倍边界）

---

## 项目结构

```
trail_race_predictor_v1.2.4/
├── login/                  ← Garmin Connect 登录模块（token 持久化）
│   ├── garmin_login.py
│   └── README.md
├── activities/             ← 活动获取模块
│   ├── selector.py         ← 拉取列表、筛选
│   └── downloader.py       ← 下载 FIT 文件
├── core/                   ← 核心预测逻辑
│   ├── predictor/          ← 预测器子模块
│   │   ├── predictor.py    ← MLRacePredictor 主入口
│   │   ├── gpx_parser.py   ← GPX 解析 + Split Points 计算
│   │   ├── model.py        ← LightGBM 模型
│   │   ├── extractor.py    ← 特征提取
│   │   └── features.py     ← SegmentFeatures 定义
│   ├── types.py            ← 数据类型定义
│   └── utils.py            ← 滤波算法 + 配置
├── data/                   ← 数据校验与文件处理
├── scripts/                ← CLI 脚本
├── reports/                ← 报告生成
├── app.py                  ← Streamlit Web 主入口（预测 + 活动下载）
├── app_activities.py       ← 活动下载视图
├── requirements.txt
└── README.md               ← 本文件
```

---

## 快速开始

### 启动 Web 界面

```bash
streamlit run app.py
```

界面包含两个功能：
- **🏔️ 越野赛预测**：上传 GPX 赛道 + FIT 训练记录 → 预测完赛时间
- **🏃 活动下载**：登录 Garmin Connect → 筛选活动 → 批量下载 FIT 文件

### CLI 方式下载活动

```bash
# 交互式：登录 → 拉取全部跑步/越野跑 → 筛选 → 选择 → 下载
python fetch_activities.py

# 按条件筛选
python fetch_activities.py --type trail_running --dist-min 10 --dist-max 50 --year 2025

# 指定输出目录
python fetch_activities.py --output records/fit --max-select 10
```

---

## 依赖

```bash
pip install -r requirements.txt
```

额外依赖（登录模块）：
- `python-garminconnect-master/`（已包含在项目中）
- `curl_cffi`（可选，提升连接稳定性）

---

## 开发说明

- **开发基线**：基于 v1.2.2 核心代码
- **主要改动**：
  - GPX 滤波器从 Savitzky-Golay 改为高斯滤波（`core/utils.py`）
  - 爬升/下降计算改为直接累加（`core/predictor/gpx_parser.py`, `core/utils.py`）
  - 新增 `split_points` 预计算机制（`core/predictor/gpx_parser.py`, `core/predictor/predictor.py`）
  - 单应用整合（`app.py`, `app_activities.py`）
