# Trail Master 小程序改造 — 工作量分析

**版本**：v1.0
**日期**：2026-05-19
**基于**：Streamlit v1.2.5 → 微信小程序

---

## 一、现有项目概况

### 代码规模

| 模块 | 文件 | 行数 | 职责 |
|------|------|------|------|
| 前端 UI | `app.py` | 818 | Streamlit 页面、侧边栏、分析流程 |
| 活动下载 | `app_activities.py` | 282 | Garmin 活动列表、筛选、下载 |
| ML 核心 | `core/predictor/model.py` | 202 | LightGBM 训练 + 预测 |
| 特征提取 | `core/predictor/extractor.py` | 402 | FIT/JSON → SegmentFeatures |
| GPX 解析 | `core/predictor/gpx_parser.py` | 329 | GPX → 赛道分段 |
| 预测编排 | `core/predictor/predictor.py` | 255 | 训练 + 预测主入口 |
| 数据结构 | `core/predictor/features.py` | 41 | SegmentFeatures dataclass |
| 滤波工具 | `core/utils.py` | 413 | SG/高斯滤波、坡度截断 |
| 报告生成 | `reports/report_generator.py` | 755 | HTML/TXT 报告 |
| Garmin 登录 | `login/garmin_login.py` | 123 | Token 持久化登录 |
| 活动筛选 | `activities/selector.py` | ~150 | Garmin 活动拉取 + 筛选 |
| 数据验证 | `data/data_validator.py` | ~100 | 文件格式校验 |
| 文件管理 | `data/file_handler.py` | ~100 | 上传文件管理 |
| **合计** | **13 个核心文件** | **~4,000** | |

**依赖库**：lightgbm, numpy, scipy, fitparse, streamlit, garminconnect (vendored)

### 已有小程序设计资产

| 文件 | 内容 | 完成度 |
|------|------|--------|
| `docs/miniprogram-design.md` | 完整 UI 设计文档（4 Tab、数据流、交互细节） | 100% |
| `docs/miniprogram-prototype.html` | 1346 行可交互 HTML 原型（含五行主题） | 100% |
| `docs/theme-system.md` | 7 套五行配色方案 + SVG 图标规范 | 100% |

---

## 二、改造核心难点

### 2.1 架构变化（最大难点）

```
当前: Streamlit (Python 全栈，浏览器运行)
       ↓
目标: 小程序 (WXML/WXSS/JS 前端) + 后端 API (Python)
```

**关键约束**：

| 约束 | 说明 | 影响 |
|------|------|------|
| LightGBM 不能在小程序运行 | 需要 numpy、C 库 | **必须后端部署** |
| FIT 文件解析依赖 fitparse | Python 库，~402 行逻辑 | **必须后端处理** |
| Garmin API 需要后端中转 | 小程序不能直接调 Garmin | **必须后端代理** |
| scipy 滤波不可用 | Savitzky-Golay / Gaussian | **后端处理或 JS 重写** |
| 微信本地存储 10MB 限制 | GPX+FIT 文件存储 | 需要分批/按需 |

**结论**：这不是简单的前端迁移，而是**前后端分离重构**。

### 2.2 模块迁移难度评估

| 模块 | 难度 | 说明 |
|------|------|------|
| GPX 解析 | ⭐⭐ 中 | XML 解析 WXML 不支持，需 JS 重写 haversine + 分段逻辑（~300 行 JS） |
| FIT 解析 | ⭐⭐⭐ 高 | fitparse 是 Python 专用，必须后端处理，小程序只传文件 |
| LightGBM 训练 | ⭐⭐⭐ 高 | 依赖 numpy/lightgbm，**必须后端**，小程序只调 API |
| 海拔滤波 | ⭐⭐⭐ 高 | scipy 依赖，需后端或纯 JS 重写（~200 行 JS） |
| Garmin 登录 | ⭐⭐ 中 | 后端代理，小程序存 token |
| 活动列表/筛选 | ⭐⭐ 中 | 后端 API 返回，小程序展示 |
| UI 页面 | ⭐ 低 | 原型已有，WXML/WXSS 实现 |
| 五行主题 | ⭐ 低 | CSS 变量已定义，WXSS 直接用 |
| 本地存储管理 | ⭐ 低 | wx.setStorageSync，设计文档已明确 |
| 报告生成 | ⭐ 低 | 后端生成或小程序内渲染 |

---

## 三、需要新建的模块

### 3.1 后端 API 服务

```
trail-master-api/
├── main.py                    # FastAPI/Flask 入口
├── routes/
│   ├── predict.py             # POST /api/predict — 上传 GPX+FIT → 预测结果
│   ├── garmin.py              # POST /api/garmin/login — Garmin 登录代理
│   │                         # GET  /api/garmin/activities — 活动列表
│   │                         # GET  /api/garmin/download/:id — 下载 FIT
│   └── health.py              # GET /api/health
├── core/                      # 迁移现有 Python 核心（几乎原样搬）
│   ├── predictor/
│   ├── utils.py
│   └── types.py
├── login/
│   └── garmin_login.py        # 原样搬
├── activities/
│   └── selector.py            # 原样搬
├── reports/
│   └── report_generator.py    # 原样搬
└── requirements.txt           # 同现有
```

**后端工作量**：核心 Python 代码几乎原样迁移，新增 API 路由层（~300-500 行）。

### 3.2 小程序前端

```
trail-master-mini/
├── app.js                     # 入口 + 五行主题初始化
├── app.json                   # 路由、Tab Bar 配置
├── app.wxss                   # 全局样式 + 主题变量
├── utils/
│   ├── api.js                 # 后端 API 调用封装
│   ├── storage.js             # 本地存储管理（10MB 限制）
│   ├── gpx-parser.js          # GPX 解析（可选，用于轻量预览）
│   └── theme.js               # 五行主题切换
├── pages/
│   ├── data/                  # Tab 1: 数据管理
│   │   ├── data.js
│   │   ├── data.wxml
│   │   ├── data.wxss
│   │   └── data.json
│   ├── predict/               # Tab 2: 预测主页 + 结果
│   │   ├── predict.js
│   │   ├── predict.wxml
│   │   ├── predict.wxss
│   │   └── predict.json
│   ├── settings/              # Tab 3: 预测设置
│   │   ├── settings.js
│   │   ├── settings.wxml
│   │   ├── settings.wxss
│   │   └── settings.json
│   ├── profile/               # Tab 4: 我的
│   │   ├── profile.js
│   │   ├── profile.wxml
│   │   ├── profile.wxss
│   │   └── profile.json
│   └── report/                # 预测报告详情页
│       ├── report.js
│       ├── report.wxml
│       ├── report.wxss
│       └── report.json
└── components/
    ├── file-card/             # 文件卡片组件
    ├── loading/               # 加载动画组件
    └── split-table/           # 分段配速表格组件
```

---

## 四、工作量估算

### 4.1 分项估算

| 工作项 | 预估工时 | 说明 |
|--------|---------|------|
| **后端 API 服务搭建** | 2-3 天 | FastAPI 框架 + 路由 + 文件上传 |
| **核心代码迁移** | 1-2 天 | Python 核心几乎原样搬，适配 API 接口 |
| **小程序项目初始化** | 0.5 天 | 项目结构、Tab Bar、全局样式、主题 |
| **数据 Tab（文件管理）** | 1-2 天 | 列表、添加、删除、存储管理 |
| **设置 Tab（配置选择）** | 1 天 | GPX/FIT 选择器、能力系数滑块 |
| **预测 Tab（主页 + 结果）** | 2-3 天 | 预测流程、报告 4 子 Tab、历史列表 |
| **我的 Tab（Garmin）** | 2-3 天 | 登录、活动列表、筛选、下载 |
| **五行主题系统** | 0.5 天 | 已有完整 CSS 变量定义 |
| **报告详情页** | 1-2 天 | 分段表格、战术建议渲染 |
| **分享功能** | 0.5 天 | wx.shareAppMessage |
| **联调 + 测试** | 2-3 天 | 前后端联调、Garmin API 测试 |
| **合计** | **13-20 天** | |

### 4.2 按角色分工

| 角色 | 工作内容 | 工时 |
|------|---------|------|
| 后端开发 | API 服务 + 核心迁移 + Garmin 代理 | 5-7 天 |
| 前端开发 | 小程序 4 Tab + 组件 + 主题 | 6-9 天 |
| 联调测试 | 前后端对接 + 真机测试 | 2-3 天 |

---

## 五、复杂度评估

### 总体评级：**中高复杂度**

| 维度 | 评级 | 说明 |
|------|------|------|
| 前端 UI | ⭐⭐ 低-中 | 原型完整，WXML 实现直白 |
| 后端 API | ⭐⭐ 中 | 核心代码迁移为主，新增路由层 |
| ML 集成 | ⭐⭐⭐ 高 | LightGBM 部署、模型训练流程 |
| 文件处理 | ⭐⭐⭐ 高 | FIT 解析、GPX 解析、10MB 存储限制 |
| 第三方集成 | ⭐⭐⭐ 高 | Garmin Connect 代理、微信登录 |
| 主题系统 | ⭐ 低 | 已有完整定义 |

### 主要风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| LightGBM 后端部署 | 训练耗时 1-5 分钟，需异步处理 | WebSocket/SSE 进度推送 |
| FIT 文件大小 | 单个 50-500KB，20 个 = 10MB | 分批上传，后端缓存 |
| Garmin API 稳定性 | Token 30 小时过期 | 后端统一管理 token |
| 微信审核 | Garmin 登录可能触发审核 | 账号密码走后端，不暴露 |

---

## 六、Token 消耗估算

### AI 辅助开发（Claude Code）

| 阶段 | 预估 Token | 说明 |
|------|-----------|------|
| 后端 API 搭建 | 80K-120K | FastAPI 路由 + 文件处理 |
| 核心代码迁移 | 30K-50K | 已有代码，适配为主 |
| 小程序前端 | 150K-250K | 4 Tab + 组件 + 主题 + 样式 |
| 联调修复 | 50K-100K | API 对接、bug 修复 |
| **合计** | **310K-520K** | |

**换算**：
- Opus: ~$5-9（按 $15/M input + $75/M output 估算）
- Sonnet: ~$1.5-3（按 $3/M input + $15/M output 估算）

### 纯人工开发（无 AI 辅助）

- 前端熟练度要求：微信小程序开发经验
- 后端熟练度要求：Python API + ML 部署经验
- 预估总工时：**15-25 人天**

---

## 七、推荐实施路径

### Phase 1：后端先行（3-5 天）

1. 搭建 FastAPI 服务
2. 迁移 `core/` 模块（predictor、utils、types）
3. 实现 `/api/predict` 接口（上传 GPX+FIT → 返回预测结果）
4. 实现 `/api/garmin/*` 接口（登录、活动列表、下载）
5. 本地测试通过

### Phase 2：小程序前端（5-8 天）

1. 项目初始化 + Tab Bar + 全局样式
2. 数据 Tab（文件管理）
3. 设置 Tab（配置选择）
4. 预测 Tab（主页 + 流程 + 结果）
5. 我的 Tab（Garmin 集成）
6. 五行主题

### Phase 3：联调 + 打磨（3-5 天）

1. 前后端联调
2. 真机测试
3. 分享功能
4. 性能优化（文件上传、加载速度）
5. 提交审核

---

## 八、与网页版功能对照

| 功能 | 网页版 (v1.2.5) | 小程序版 | 迁移方式 |
|------|-----------------|---------|---------|
| GPX 解析 | `gpx_parser.py` (329行) | 后端 API | Python 原样迁移 |
| FIT 解析 | `extractor.py` (402行) | 后端 API | Python 原样迁移 |
| 海拔滤波 | `utils.py` (413行) | 后端 API | Python 原样迁移 |
| LightGBM 训练 | `model.py` (202行) | 后端 API | Python 原样迁移 |
| 预测编排 | `predictor.py` (255行) | 后端 API | Python 原样迁移 |
| 报告生成 | `report_generator.py` (755行) | 后端/前端 | HTML 报告保留，前端渲染简化版 |
| Garmin 登录 | `garmin_login.py` (123行) | 后端代理 | Python 原样迁移 |
| 活动筛选 | `selector.py` (~150行) | 后端 API | Python 原样迁移 |
| Streamlit UI | `app.py` (818行) | 小程序 WXML | **全部重写** |
| 活动下载页 | `app_activities.py` (282行) | 小程序 WXML | **全部重写** |
| 五行主题 | 无 | CSS 变量 | **新增** |
| 本地文件存储 | 无（临时文件） | wx.setStorage | **新增** |
| 分享功能 | 无 | wx.shareAppMessage | **新增** |

**关键发现**：后端 Python 核心代码（~2,500 行）几乎可以原样迁移到 API 服务，真正的重写工作在前端 UI（~1,100 行 Streamlit → WXML/WXSS）。

---

*文档由 Claude 分析生成，基于 v1.2.5 源码 + docs 设计文档，2026-05-19*
