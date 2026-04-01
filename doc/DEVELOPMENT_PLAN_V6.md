# Trail Race Predictor V6 - 开发规划文档

**版本**: V6.0
**规划日期**: 2026-04-01
**目标**: 将硬核ML模型转化为用户友好的越野赛预测工具

---

## 一、项目目标

将现有的 LightGBM 越野赛预测器从"命令行工具"升级为"傻瓜式Web应用"，让普通跑友无需Python环境即可使用。

### 核心价值主张
1. **零门槛**: 无需配置Python环境，拖拽上传即可
2. **智能化**: 自动数据处理，智能容错
3. **专业性**: 生成精美报告，提供战术建议
4. **稳定性**: 健壮的错误处理，降级运行能力

---

## 二、目标架构

```
/TrailRacePredictor/
├── app.py                      # Streamlit 主入口
├── core/                       # 核心算法层 (重构自 scripts/)
│   ├── __init__.py
│   ├── predictor.py           # ML预测器 (保留原有逻辑)
│   ├── gpx_filter.py          # GPX滤波器
│   ├── utils.py               # 工具函数
│   ├── feature_extractor.py   # 特征提取器 (新增)
│   └── report_generator.py     # 报告生成器 (新增)
├── ui/                         # UI组件层 (新增)
│   ├── __init__.py
│   ├── sidebar.py             # 侧边栏组件
│   ├── results_display.py     # 结果展示组件
│   └── charts.py               # 图表组件
├── data/                       # 数据处理层 (新增)
│   ├── __init__.py
│   ├── file_handler.py         # 文件上传处理
│   ├── data_validator.py       # 数据验证
│   └── coordinate_checker.py   # 坐标对齐检查
├── demo/                       # 内置演示数据
│   ├── demo_route.gpx          # 黄岩九峰演示路线
│   └── demo_records/           # 演示训练记录
├── reports/                    # 生成的报告
├── temp/                       # 临时文件目录
├── requirements.txt            # 依赖包
├── run.bat                     # Windows一键启动脚本
├── run.sh                      # Linux/Mac一键启动脚本
├── build_exe.py                # 打包脚本
└── README.md                   # 用户文档
```

---

## 三、技术选型

| 组件 | 技术方案 | 理由 |
|------|---------|------|
| **Web框架** | Streamlit | Python原生，代码量少，自动处理文件上传 |
| **数据处理** | Pandas | 报告生成，数据展示 |
| **图表** | Plotly | 交互式图表，支持Streamlit集成 |
| **报告格式** | HTML + Markdown | 可保存可分享 |
| **打包工具** | PyInstaller | 生成独立EXE |

### 依赖包清单 (requirements.txt)
```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.20.0
scipy>=1.7.0
lightgbm>=3.3.0
gpxpy>=1.6.0  # GPX解析
fitparse>=1.2.0  # FIT解析 (可选)
```

---

## 四、核心模块设计

### 4.1 核心算法层 (core/) - 保持稳定性

**设计原则**: 最小化修改现有代码，仅添加接口层

```python
# core/predictor.py - 保持原有类，添加统一接口
class TrailRacePredictor:
    """
    统一接口的预测器类
    封装现有的 MLRacePredictor，添加错误处理
    """
    def __init__(self):
        self._internal_predictor = MLRacePredictor()
        self._is_trained = False
        self._training_stats = {}

    def train_from_files(self, file_paths: List[str],
                         category: str = 'auto') -> TrainingResult:
        """
        从文件列表训练模型

        Args:
            file_paths: FIT/GPX文件路径列表
            category: 'high'/'medium'/'low'/'auto'

        Returns:
            TrainingResult: 包含成功状态、统计信息、错误消息
        """
        try:
            # 自动分类逻辑
            if category == 'auto':
                category = self._auto_classify(file_paths)

            # 调用原有训练逻辑
            self._internal_predictor.analyze_and_train()

            return TrainingResult(success=True, stats=self._training_stats)
        except Exception as e:
            return TrainingResult(success=False, error=str(e))

    def predict(self, gpx_path: str, effort_level: str) -> PredictionResult:
        """
        预测比赛成绩

        Args:
            gpx_path: GPX路线文件
            effort_level: 'high'/'medium'/'low'

        Returns:
            PredictionResult: 包含预测时间、分段数据、图表数据
        """
        pass

    def analyze_performance(self, gpx_path: str, record_path: str) -> PerformanceAnalysis:
        """
        复盘功能: 对比预测vs实际

        Args:
            gpx_path: 赛道GPX
            record_path: 实际比赛记录

        Returns:
            PerformanceAnalysis: 包含各段对比、优势/劣势分析
        """
        pass
```

### 4.2 数据验证层 (data/)

```python
# data/data_validator.py
class DataValidator:
    """数据验证器 - 防呆设计"""

    @staticmethod
    def validate_gpx(file_path: str) -> ValidationResult:
        """
        验证GPX文件

        检查项:
        - 文件格式正确
        - 包含轨迹点
        - 包含海拔数据
        - 坐标范围合理 (地球范围内)
        """
        pass

    @staticmethod
    def validate_fit(file_path: str) -> ValidationResult:
        """
        验证FIT文件

        检查项:
        - 文件格式正确
        - 包含GPS坐标 (非跑步机)
        - 包含海拔数据
        - 时间范围合理
        """
        pass

    @staticmethod
    def check_coordinate_alignment(gpx_coords: Tuple, fit_coords: List) -> AlignmentResult:
        """
        检查赛道和训练记录的坐标对齐

        Args:
            gpx_coords: 赛道中心坐标 (lat, lon)
            fit_coords: 训练记录坐标列表

        Returns:
            AlignmentResult: 是否对齐、距离偏差、警告消息
        """
        # 计算赛道中心点
        # 计算训练记录分布范围
        # 如果偏差>500km，返回警告
        pass

    @staticmethod
    def detect_unit_system(data: dict) -> str:
        """
        自动检测单位系统

        Returns:
            'metric' (公制) 或 'imperial' (英制)
        """
        # 检查速度值范围
        # 如果速度>20，可能是英制
        pass

    @staticmethod
    def filter_by_time(files: List, years: int = 2) -> List:
        """
        按时间过滤文件

        Args:
            files: 文件列表
            years: 只保留最近N年的数据

        Returns:
            过滤后的文件列表
        """
        pass
```

### 4.3 文件处理层 (data/)

```python
# data/file_handler.py
class FileHandler:
    """文件上传处理器"""

    def __init__(self, temp_dir: str = './temp'):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def save_uploaded_file(self, uploaded_file) -> str:
        """保存上传的文件到临时目录"""
        pass

    def auto_select_best_files(self, files: List[str], max_count: int = 20) -> List[str]:
        """
        智能筛选最优文件

        选择逻辑:
        1. 按文件大小排序 (越大=数据越完整)
        2. 按时间过滤 (最近2年优先)
        3. 返回最大的N个
        """
        pass

    def classify_by_speed(self, files: List[str]) -> Dict[str, List]:
        """
        自动按速度分类

        分类逻辑:
        - 计算每个文件的平均速度
        - 前1/3 -> 高速
        - 中1/3 -> 中速
        - 后1/3 -> 低速
        """
        pass

    def cleanup_temp(self):
        """清理临时文件"""
        pass
```

### 4.4 报告生成器 (core/)

```python
# core/report_generator.py
class ReportGenerator:
    """精美报告生成器"""

    def __init__(self, prediction_result: PredictionResult):
        self.result = prediction_result

    def generate_html_report(self, output_path: str) -> str:
        """
        生成HTML报告

        报告结构:
        1. 概览区: 大字显示预测时间
        2. 赛道分析: 距离、爬升、难度等级
        3. 分段配速表: 每5km/CP点的详细数据
        4. 海拔曲线图: Plotly交互图
        5. 战术建议: AI生成的建议
        6. 难度预警: 最难3段标红
        """
        pass

    def _generate_elevation_chart(self) -> str:
        """生成海拔曲线图 (Plotly HTML)"""
        pass

    def _generate_tactical_advice(self) -> List[str]:
        """
        生成战术建议

        基于特征重要性分析:
        - 如果下坡表现好 -> "下坡是你追赶时间的窗口"
        - 如果爬坡弱 -> "建议在爬坡段保守配速"
        - 如果有极端陡坡 -> "模型已应用VAM限制"
        """
        pass

    def _generate_difficulty_warnings(self) -> List[Dict]:
        """
        识别最难路段

        识别逻辑:
        - 坡度>30%的路段
        - 累计爬升最密集的路段
        - 返回前3个最难路段
        """
        pass

    def generate_summary_card(self) -> str:
        """生成摘要卡片 (用于界面显示)"""
        pass
```

### 4.5 Streamlit 界面 (app.py)

```python
# app.py
import streamlit as st
from core.predictor import TrailRacePredictor
from data.file_handler import FileHandler
from data.data_validator import DataValidator
from core.report_generator import ReportGenerator

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
</style>
""", unsafe_allow_html=True)

def main():
    # 标题
    st.title(":mountain: Trail Master 越野赛成绩预测系统")
    st.markdown("上传你的训练记录，获取精准的赛道配速分析")

    # 初始化
    if 'predictor' not in st.session_state:
        st.session_state.predictor = TrailRacePredictor()
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = FileHandler()

    # 侧边栏
    render_sidebar()

    # 主界面
    if st.session_state.get('files_uploaded'):
        render_main_analysis()
    else:
        render_instructions()

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header(":folder: 数据上传")

        # 赛道上传
        gpx_file = st.file_uploader(
            "上传目标赛道 (GPX)",
            type=['gpx'],
            help="上传比赛的GPX路线文件"
        )

        # 训练记录上传
        fit_files = st.file_uploader(
            "上传个人记录 (FIT/GPX)",
            type=['fit', 'gpx'],
            accept_multiple_files=True,
            help="上传历史训练记录，建议至少5个文件"
        )

        # 高级选项
        with st.expander(":gear: 高级选项"):
            effort_level = st.select_slider(
                "预测努力程度",
                options=['保守', '中等', '激进'],
                value='中等'
            )

            time_filter = st.checkbox(
                "只使用最近2年的数据",
                value=True
            )

            auto_classify = st.checkbox(
                "自动分类训练记录",
                value=True,
                help="根据速度自动分为高中低三档"
            )

        # 开始按钮
        if st.button(":rocket: 开始分析", type="primary", use_container_width=True):
            if not gpx_file:
                st.error("请上传赛道GPX文件!")
            elif not fit_files or len(fit_files) < 3:
                st.error("请至少上传3个训练记录文件!")
            else:
                st.session_state.files_uploaded = True
                st.session_state.gpx_file = gpx_file
                st.session_state.fit_files = fit_files
                st.session_state.effort_level = effort_level
                st.rerun()

def render_main_analysis():
    """渲染分析主界面"""
    # 进度条
    with st.status("正在分析中...", expanded=True) as status:

        # Step 1: 保存文件
        st.write(":floppy_disk: 处理上传文件...")
        file_handler = st.session_state.file_handler
        gpx_path = file_handler.save_uploaded_file(st.session_state.gpx_file)
        fit_paths = [file_handler.save_uploaded_file(f) for f in st.session_state.fit_files]

        # Step 2: 验证数据
        st.write(":mag: 验证数据格式...")
        validator = DataValidator()

        gpx_result = validator.validate_gpx(gpx_path)
        if not gpx_result.valid:
            st.error(f"赛道文件问题: {gpx_result.error}")
            return

        # 检查是否为跑步机数据
        for fit_path in fit_paths:
            fit_result = validator.validate_fit(fit_path)
            if not fit_result.valid:
                st.warning(f"跳过文件 {fit_path}: {fit_result.error}")

        # Step 3: 智能筛选
        st.write(":filter: 智能筛选训练数据...")
        selected_files = file_handler.auto_select_best_files(fit_paths, max_count=20)
        st.write(f"已选取 {len(selected_files)} 个高质量样本")

        # Step 4: 训练模型
        st.write(":brain: 训练机器学习模型...")
        predictor = st.session_state.predictor
        train_result = predictor.train_from_files(selected_files)

        if not train_result.success:
            st.error(f"训练失败: {train_result.error}")
            return

        # Step 5: 预测
        st.write(":chart_with_upwards_trend: 生成预测...")
        effort_map = {'保守': 'low', '中等': 'medium', '激进': 'high'}
        prediction = predictor.predict(gpx_path, effort_map[st.session_state.effort_level])

        status.update(label="分析完成!", state="complete")

    # 显示结果
    render_prediction_results(prediction)

def render_prediction_results(prediction):
    """渲染预测结果"""
    st.success(":white_check_mark: 预测完成!")

    # 主要指标卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-label">预测完赛时间</div>
        <div class="big-metric">{prediction.time_hm}</div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("平均配速", f"{prediction.pace_min_km:.1f} min/km")

    with col3:
        st.metric("平均速度", f"{prediction.speed_kmh:.2f} km/h")

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

def render_download_section(prediction):
    """渲染下载区域"""
    st.subheader(":inbox_tray: 下载报告")

    # 生成报告
    report_gen = ReportGenerator(prediction)

    col1, col2 = st.columns(2)

    with col1:
        # HTML报告
        html_path = report_gen.generate_html_report("./reports/prediction_report.html")
        with open(html_path, 'rb') as f:
            st.download_button(
                ":page_facing_up: 下载HTML报告",
                f,
                file_name="trail_prediction_report.html",
                mime="text/html"
            )

    with col2:
        # JSON数据
        json_data = prediction.to_json()
        st.download_button(
            ":card_file_box: 下载JSON数据",
            json_data,
            file_name="prediction_data.json",
            mime="application/json"
        )

if __name__ == '__main__':
    main()
```

---

## 五、数据结构定义

```python
# core/types.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class EffortLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = None

@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    stats: Optional[Dict] = None
    error: Optional[str] = None

@dataclass
class SegmentPrediction:
    """单段预测"""
    segment_id: int
    start_km: float
    end_km: float
    distance_km: float
    grade_pct: float
    altitude_m: float
    predicted_speed_kmh: float
    predicted_time_min: float
    cumulative_time_min: float
    difficulty_level: str  # 'easy', 'moderate', 'hard', 'extreme'

@dataclass
class PredictionResult:
    """预测结果"""
    total_time_min: float
    total_time_hm: str
    pace_min_km: float
    speed_kmh: float
    total_distance_km: float
    total_ascent_m: float
    segments: List[SegmentPrediction]
    feature_importance: Dict[str, float]
    model_confidence: float
    warnings: List[str]

    def to_json(self) -> str:
        """导出为JSON"""
        pass

@dataclass
class PerformanceAnalysis:
    """复盘分析结果"""
    predicted_time_min: float
    actual_time_min: float
    time_difference_min: float
    segment_comparison: List[Dict]
    strengths: List[str]
    weaknesses: List[str]
    percentile_rankings: Dict[str, float]
```

---

## 六、开发阶段规划

### Phase 1: 架构重组 (稳定性优先)
**目标**: 重构目录结构，确保现有功能不受影响

**任务**:
1. 创建 `core/` 目录，移动现有脚本
2. 添加类型定义文件 `core/types.py`
3. 创建 `TrailRacePredictor` 封装类
4. 编写单元测试验证功能不变

**验收标准**:
- 现有 `predictor.py` 功能100%保留
- 所有单元测试通过
- 可以从 `core/` 导入并运行

**风险**: 低 - 只是移动和封装，不修改核心逻辑

---

### Phase 2: 数据验证层
**目标**: 添加防呆设计和错误处理

**任务**:
1. 实现 `DataValidator` 类
2. 添加GPX/FIT格式验证
3. 实现坐标对齐检查
4. 实现单位自动检测
5. 实现时间过滤功能

**验收标准**:
- 跑步机数据能被正确拒绝
- 缺失海拔数据时能降级运行
- 坐标偏差过大时显示警告

**风险**: 低 - 独立模块，不影响核心逻辑

---

### Phase 3: Streamlit 界面
**目标**: 创建用户友好的Web界面

**任务**:
1. 创建 `app.py` 主入口
2. 实现文件上传组件
3. 实现进度显示
4. 实现结果展示界面
5. 添加Demo数据

**验收标准**:
- 拖拽上传正常工作
- 进度条正确显示
- 结果美观易读

**风险**: 中 - 需要测试多种浏览器

---

### Phase 4: 报告生成器
**目标**: 生成精美的HTML报告

**任务**:
1. 实现 `ReportGenerator` 类
2. 创建海拔曲线图 (Plotly)
3. 实现战术建议生成
4. 实现难度预警
5. 设计HTML模板

**验收标准**:
- HTML报告可保存到手机
- 图表交互正常
- 战术建议有实际价值

**风险**: 低 - 独立模块

---

### Phase 5: 复盘功能
**目标**: 实现比赛复盘分析

**任务**:
1. 实现 `analyze_performance()` 方法
2. 对比预测vs实际
3. 生成优势/劣势分析
4. 计算百分位排名

**验收标准**:
- 能识别用户强项/弱项
- 分析结果有意义

**风险**: 中 - 需要更多测试数据

---

### Phase 6: 打包发布
**目标**: 生成可分发的EXE

**任务**:
1. 编写 `build_exe.py`
2. 配置PyInstaller
3. 测试EXE运行
4. 编写用户文档
5. 创建启动脚本

**验收标准**:
- EXE在无Python环境运行
- 所有功能正常
- 文档清晰易懂

**风险**: 中 - 打包可能遇到依赖问题

---

## 七、风险控制

### 7.1 核心代码保护策略

1. **封装而非修改**: 所有现有代码通过封装类调用，不直接修改
2. **降级运行**: 遇到错误时回退到简化模型
3. **版本锁定**: requirements.txt 锁定具体版本号
4. **完整备份**: V5代码已备份到 `trail_race_predictor_v5_backup/`

### 7.2 错误处理策略

```python
# 所有公共方法都应有try-except包装
def safe_predict(self, *args, **kwargs):
    try:
        return self._internal_predict(*args, **kwargs)
    except LightGBMError as e:
        # 降级到简化模型
        return self._fallback_predict(*args, **kwargs)
    except Exception as e:
        # 返回错误结果而非抛出异常
        return PredictionResult(error=str(e))
```

### 7.3 测试策略

1. **单元测试**: 每个模块都有对应测试
2. **集成测试**: 端到端流程测试
3. **回归测试**: 确保V5功能不变
4. **Demo数据**: 使用黄岩九峰数据作为标准测试

---

## 八、文件清单

### 新增文件
| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit主入口 |
| `core/types.py` | 类型定义 |
| `core/feature_extractor.py` | 特征提取器 |
| `core/report_generator.py` | 报告生成器 |
| `data/__init__.py` | 数据模块初始化 |
| `data/file_handler.py` | 文件处理器 |
| `data/data_validator.py` | 数据验证器 |
| `data/coordinate_checker.py` | 坐标检查器 |
| `ui/__init__.py` | UI模块初始化 |
| `ui/sidebar.py` | 侧边栏组件 |
| `ui/results_display.py` | 结果展示 |
| `ui/charts.py` | 图表组件 |
| `demo/demo_route.gpx` | 演示路线 |
| `run.bat` | Windows启动脚本 |
| `run.sh` | Linux启动脚本 |
| `build_exe.py` | 打包脚本 |

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `core/predictor.py` | 添加封装接口，保留原有代码 |
| `requirements.txt` | 添加新依赖 |
| `README.md` | 更新使用说明 |

---

## 九、时间线建议

| 阶段 | 预估工作量 | 优先级 |
|------|-----------|--------|
| Phase 1: 架构重组 | 2小时 | P0 - 必须首先完成 |
| Phase 2: 数据验证 | 3小时 | P1 - 核心功能 |
| Phase 3: Streamlit界面 | 4小时 | P1 - 核心功能 |
| Phase 4: 报告生成 | 3小时 | P2 - 增值功能 |
| Phase 5: 复盘功能 | 2小时 | P3 - 可选功能 |
| Phase 6: 打包发布 | 2小时 | P2 - 发布必需 |

**总计**: 约16小时

---

## 十、下一步行动

请确认此规划后，我将按Phase顺序开始实施。首先从Phase 1开始：
1. 创建 `core/` 目录结构
2. 封装现有 `predictor.py`
3. 编写单元测试
4. 验证功能100%保留

---

*规划版本: 1.0*
*更新时间: 2026-04-01*
