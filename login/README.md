# Garmin Connect Login Module

Garmin Connect 中国区登录模块，支持 token 自动持久化和刷新。

## 快速使用

```python
from login.garmin_login import garmin_login

# 第一次需要账号密码，之后自动从 token 恢复
garmin = garmin_login(email="your@email.com", password="your_password")

# 不传参数即可自动恢复登录态（token 存在项目 tokens/ 目录下）
garmin = garmin_login()
```

## 集成到项目

1. 复制 `login/` 目录到你的项目
2. 确保父目录下有 `python-garminconnect-master/`（依赖库）
3. 调用：

```python
from login.garmin_login import garmin_login

garmin = garmin_login(email="...", password="...")
```

## API

### `garmin_login(email, password, tokenstore, is_cn)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `email` | `str \| None` | `None` | Garmin 账号，首次登录必填 |
| `password` | `str \| None` | `None` | 密码，首次登录必填 |
| `tokenstore` | `str` | `项目根目录/tokens/` | token 存储目录 |
| `is_cn` | `bool` | `True` | 是否使用中国区 (garmin.cn) |

**返回**: 已认证的 `Garmin` 对象

**行为**:
- 优先尝试 token 恢复（无需密码）
- token 失效则用密码登录
- 登录成功后自动保存 token

### `garmin_login_interactive(tokenstore, is_cn)`

交互式登录，token 失效时从环境变量或 stdin 读取账号密码。

环境变量: `GARMIN_EMAIL`, `GARMIN_PASSWORD`（或 `EMAIL`, `PASSWORD`）

## Token 生命周期

| Token | 有效期 | 说明 |
|-------|--------|------|
| access_token | ~30 小时 | 自动刷新 |
| refresh_token | ~30 天 | 用于续期 access_token |

只要 30 天内调用过一次 `garmin_login()`，登录态就一直有效。

## 常用 API 调用

```python
from datetime import date

today = date.today().isoformat()

# 用户摘要
summary = garmin.get_user_summary(today)

# 心率
hr = garmin.get_heart_rates(today)

# 活动记录
activities = garmin.get_activities(0, 10)

# 睡眠
sleep = garmin.get_sleep_data(today)

# 步数
steps = garmin.get_steps_data(today)
```

## 依赖

- Python 3.10+
- `python-garminconnect-master/`（已包含在项目中）
- `curl_cffi`（可选，提升连接稳定性）
