# 五行主题系统

## 概述

小程序采用"五行"理论设计了 7 套轻盈透明的配色方案，每周一自动切换，无需用户手动操作。

## 主题切换逻辑

- **自动切换**：根据星期几自动设置当日主题，用户无感知
- **JS 实现**：`new Date().getDay()` 获取星期几，映射到对应主题 class
- **原型预览**：数据页顶部的 7 个色点仅用于原型预览，正式版移除

```
周一 → 木（森林溪流）
周二 → 火（赤陶暖色）
周三 → 土（沙丘禅意）
周四 → 金（深海色系）
周五 → 水（自然四色）
周六 → 火（大地赤陶）
周日 → 水（大地色系）
```

## 7 套配色详情

### 周一 木 — Forest Stream 森林溪流

| 变量 | 值 | 用途 |
|------|-----|------|
| --primary | #6BCB77 | 主色（按钮、图标、高亮） |
| --primary-light | #95D9A0 | 浅色变体 |
| --primary-bg | rgba(107,203,119,0.08) | 背景填充 |
| --success | #3AAF4E | 成功状态 |
| --warning | #5CC8C8 | 警告状态 |
| --bg | #F7FBF7 | 页面背景 |
| --border | rgba(107,203,119,0.15) | 边框 |

### 周二 火 — Terracotta Warm 赤陶暖色

| 变量 | 值 |
|------|-----|
| --primary | #E8A87C |
| --primary-light | #F0C4A4 |
| --success | #BA8635 |
| --warning | #AE431E |
| --bg | #FDF9F6 |

### 周三 土 — Sandy Zen 沙丘禅意

| 变量 | 值 |
|------|-----|
| --primary | #C8B9A0 |
| --primary-light | #DDD2C0 |
| --success | #A89880 |
| --warning | #6B5E4F |
| --bg | #FAF8F5 |

### 周四 金 — Deep Sea 深海色系

| 变量 | 值 |
|------|-----|
| --primary | #4A9BAF |
| --primary-light | #7BB8C8 |
| --success | #5CC8C8 |
| --warning | #2E5A7C |
| --bg | #F5F9FB |

### 周五 水 — Palette Colori 自然四色

| 变量 | 值 |
|------|-----|
| --primary | #7FB5B0 |
| --primary-light | #A5CCC8 |
| --success | #8B9A6B |
| --warning | #C67B5C |
| --bg | #F6FAF9 |

### 周六 火 — Terra Cotta 大地赤陶

| 变量 | 值 |
|------|-----|
| --primary | #C47A3C |
| --primary-light | #D9A06A |
| --success | #8B3A1A |
| --warning | #B8652A |
| --bg | #FDF8F4 |

### 周日 水 — Earth Colour 大地色系

| 变量 | 值 |
|------|-----|
| --primary | #5A7A9A |
| --primary-light | #8298B0 |
| --success | #3A4A3A |
| --warning | #B8860B |
| --bg | #F7F8FA |

## SVG 图标规范

所有图标使用 SVG 线性描边，自动跟随主题变色：

- **主题色图标**：`stroke: var(--primary)` — 数据、预测、定位、跑者、山峰等
- **中性色图标**：`stroke: #666666` — 设置齿轮、Garmin 手表（不抢主题色）
- **描边参数**：1.5px，`stroke-linecap: round`，`stroke-linejoin: round`

## 配色来源

7 个色系提取自小红书自然配色帖子，原始数据见 `xhs_natural_palettes.md`。

## 正式版注意事项

1. 移除数据页顶部的 7 个色点（主题切换预览 UI）
2. 保留 `setTheme()` 函数和星期自动切换逻辑
3. 所有 SVG 图标的 `class="icon-primary"` 确保跟随 `var(--primary)` 变色
4. 滑块（range input）已自定义样式，跟随主题色
