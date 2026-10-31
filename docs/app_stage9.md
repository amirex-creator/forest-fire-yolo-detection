# 阶段九：Streamlit 网页系统（app.py）

本文件记录阶段九的实际实现与实测结果。**所有数字都来自本机真实运行**，没有推测或占位数据。

---

## 一、做了什么

`app.py` 从阶段二的骨架（2,946 B，只打印免责声明）改写为可运行的网页应用（**21,530 B**，sha256 前缀 `fdbc0ebc57088cb1`）。

四个页面：

| 页面 | 内容 |
| --- | --- |
| 项目介绍 | 项目/模型/数据一句话说明、可用权重数、test 集 mAP50 与 mAP50-95（无评价结果时显示"未评价"）、适用范围与限制、免责声明 |
| 图片检测 | 上传 JPG/JPEG/PNG（≤20 MB）→ 显示原图 → "开始检测" → 显示渲染结果图 + 检测框数/smoke 数/fire 数/推理耗时 + 逐框表格 + 耗时口径 + 下载 JSON/CSV |
| 实验结果 | 读 `outputs/reports/metrics.json`：总体指标、分类别表、推理耗时、误报漏检计数、图表（只显示真实存在的文件）、实验完成情况表；**没有 metrics.json 时显示"尚未完成实验"** |
| 模型说明 | 当前权重（名称/路径/大小/修改时间/SHA256）、模型与训练信息、置信度含义、系统边界、免责声明 |

侧栏：页面导航 + 系统状态（可用权重数、评价结果是否存在）+ 免责声明；在"图片检测"页还提供权重选择、conf 阈值滑块（0.05–0.95，默认 0.25）、IoU 滑块（0.10–0.90，默认 0.45）。

---

## 二、如何满足阶段九的硬性约束

| 约束 | 实现 |
| --- | --- |
| 复用 `src/inference.py`，不在 app.py 里重写推理 | app.py 只调用 `SmokeDetector.predict_image()`、`build_payload()`、`write_json_report()`、`write_csv_report()`、`describe_weights()`、`resolve_weights_path()`、`_list_trained_weights()`；**没有任何 YOLO 调用或检测框解析代码** |
| 模型只加载一次 | `@st.cache_resource` 缓存 `SmokeDetector`，以权重路径为 key；切换页面/重复上传不会重新加载。阈值变化时直接改缓存对象的 `conf`/`iou` 属性，**不重建模型** |
| 支持 jpg/jpeg/png + 格式与大小校验 | `ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}`、`MAX_UPLOAD_MB = 20`；`validate_upload()` 依次检查后缀、空文件、大小，错误走 `st.warning` 友好提示 |
| 原图 → 开始检测 → 结果图 + 类别/置信度/框数 + 保存 | 左右两栏布局；结果表列 `类别/置信度/x1/y1/x2/y2`；四列指标卡（框数、smoke、fire、推理耗时）；`st.download_button` 提供 JSON 与 CSV 下载，同时落盘到 `outputs/predictions/web_<时间戳>/` |
| 实验结果页只读真实文件、缺失显示"尚未完成实验" | `load_metrics()` 读不到或解析失败都返回 `None` → 页面显示警告 + 复现命令；图表用 `existing_figures()` 逐个 `is_file()` 校验后才展示 |
| 模型说明页写模型版本/权重来源 + 免责声明 | 见页面四；免责声明用 `st.error` 醒目展示 |
| 权重缺失不崩溃 | 无权重时"图片检测"页给出训练提示并 `return`；加载异常被 `try/except` 捕获后 `st.error` + 建议；`resolve_weights_safe()` 把 `FileNotFoundError` 转成文案 |
| 全部路径用 pathlib | 所有路径来自 `src/paths.py`（`PROJECT_ROOT` / `PREDICTIONS_DIR` / `REPORTS_DIR` / `MODELS_DIR`）与 `pathlib.Path`，**源码内没有任何盘符路径**（已用正则扫描确认） |

---

## 三、启动命令

```powershell
cd "E:\DeepSeek Harness\fs_smoke"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端提示的地址（默认 http://localhost:8501）。

本次自检用的是带参数版本（不打开浏览器、指定端口、关闭文件监视）：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8517 --server.address 127.0.0.1 --server.fileWatcherType none --browser.gatherUsageStats false
```

---

## 四、真实运行记录

### 4.1 服务启动（真实）

```
2026-10-28 21:22:08.991 Uvicorn server started on 127.0.0.1:8517
  URL: http://127.0.0.1:8517
```

HTTP 探测（`requests`，真实返回）：

| 请求 | 状态码 | 响应长度 |
| --- | --- | --- |
| `GET /_stcore/health` | **200** | 2 B（`ok`） |
| `GET /` | **200** | 6,467 B（Streamlit 前端 HTML） |

### 4.2 四页面渲染（`streamlit.testing.v1.AppTest`，进程内真实执行）

| 页面 | 异常 | 说明 |
| --- | --- | --- |
| 项目介绍（首屏） | 无 | 标题 `🔥 森林烟雾目标检测系统` 正常渲染 |
| 图片检测 | 无 | 无上传文件时只显示上传提示，不报错 |
| 实验结果 | 无 | 读到 `metrics.json`；error 组件 1 个（免责声明） |
| 模型说明 | 无 | 权重信息表 + 免责声明（error 1 个）+ 边界提示（info 1 个） |

> 说明：Streamlit 的 `error` 组件在本项目里被用作"醒目提示块"，所以"error 数=1"指的是免责声明，不是运行错误。

### 4.3 上传校验（伪造上传对象，真实调用 `validate_upload`）

| 输入 | 返回 |
| --- | --- |
| `a.gif`（GIF 内容） | `不支持的格式 .gif，仅支持 JPG / JPEG / PNG` |
| `a.jpg`（0 字节） | `上传的文件是空的，请换一张图片` |
| `big.jpg`（21 MB） | `文件过大（21.0 MB），上限 20 MB` |
| `AoF04049.jpg`（真实图 63,929 B） | 通过，另存为 `outputs/predictions/uploads/20261028_212034_AoF04049.jpg` |

### 4.4 检测链路（复用 app.py 的函数，真实推理）

- 权重自动选择：`batch1_e10_full_20261028_130129_best.pt`（列表里 6 个权重，**`yolov8n.pt` 被正确排除**）
- 结果：**3 个框** —— `fire 0.741`、`fire 0.612`、`smoke 0.335`（与阶段七命令行推理结果一致）
- 耗时：读图 77.3 ms / 纯预测 66.4 ms / 画图写盘 3.8 ms
- 结果图：`outputs/predictions/stage9_selftest/20261028_212034_AoF04049__pred.jpg`（83,983 B）
- 结构化输出：`result.json` 3,328 B、`result.csv` 738 B

### 4.5 权重缺失提示（模拟 `models/` 无训练权重）

`resolve_weights_safe(None)` 返回 `(None, 提示文案)`，文案内容为：

```
在 E:\DeepSeek Harness\fs_smoke\models 下没有找到本项目训练出的权重（*_best.pt / *_last.pt）。
下一步：
  1) 先训练一个模型（分批训练，示例）：…
```

界面据此显示 `st.error` + 训练命令，**不抛异常、不崩溃**。

### 4.6 其它

- `metrics.json` 读取成功，页面可用图表 **10 张**（混淆矩阵、PR 曲线、分类别柱状图、误报/漏检样本图等）
- 首次自检时发现 Streamlit 1.66 已弃用 `use_container_width`，全部 9 处改为 `width="stretch"`，改后复跑无弃用告警

---

## 五、产物路径

| 内容 | 路径 |
| --- | --- |
| 网页代码 | `app.py` |
| 上传的原图副本 | `outputs/predictions/uploads/<时间戳>_<原名>` |
| 每次检测的结果目录 | `outputs/predictions/web_<时间戳>/`（结果图、`result.json`、`result.csv`） |
| 自检产物 | `outputs/predictions/stage9_selftest/` |
| 实验结果数据源 | `outputs/reports/metrics.json` + `outputs/figures/eval_*/` |

---

## 六、关键实现决定与理由

1. **上传文件先落盘再推理**：`src/inference.py` 的接口是"按路径推理"（保证命令行与网页共用同一套读图逻辑，且不内嵌 `cv2.imread`）。所以网页把上传内容存到 `outputs/predictions/uploads/`，再交给 `predict_image()`。好处是保留原始输入，事后可复现。
2. **阈值改属性而不是重建检测器**：`st.cache_resource` 缓存的是模型本身；如果 conf 进了缓存 key，用户每拖一次滑块就会重新加载 6.5 MB 权重（约 2 秒 + 显存抖动）。因此 conf/iou 作为可写属性在每次检测前赋值。
3. **权重白名单复用推理模块的判断**：`available_weights()` 直接调 `_list_trained_weights()`，避免网页端重新实现"哪些 pt 是训练产物"，也避免把 COCO 权重当烟雾模型用。
4. **图表逐个校验存在性**：`metrics.json` 里记录的是生成时的路径，用户可能删除或移动图片；`existing_figures()` 用 `is_file()` 过滤，缺失就不显示，而不是让页面报错。
5. **`error` 组件用于免责声明**：Streamlit 里 `st.error` 视觉效果最强，符合"必须醒目展示'仅供辅助分析'"的要求。
6. **不做多页目录（`pages/`）**：四页用侧栏 radio 切换，单文件便于课程展示与讲解，也避免 `pages/` 目录与 Streamlit 版本兼容问题。

---

## 七、未验证与限制（重要）

1. **没有做人工浏览器点击测试**：本次验证是"服务能启动 + HTTP 200"与"`AppTest` 四页面无异常 + 上传/推理函数真实跑通"，**没有在真实浏览器里逐个点击按钮**，也没有截图。因此"按钮点击后布局是否美观"未验证。
2. **多用户/并发未测**：`st.cache_resource` 在多会话共享同一模型实例，本项目按单机单人演示设计。
3. **视频流、批量上传未实现**（阶段九未要求）。
4. **上传大小上限 20 MB 是本项目自定**，不是 Streamlit 默认值。
5. **图表中的中文由 matplotlib 渲染**，网页里的中文由浏览器渲染，两者字体不同属正常现象。
6. **依赖 Streamlit 1.66.1**：若换版本，`width="stretch"` 等参数可能需要调整。

---

## 八、复现

```powershell
cd "E:\DeepSeek Harness\fs_smoke"
.\.venv\Scripts\python.exe -m streamlit run app.py
# 另开一个终端：
# 打开 http://localhost:8501 → 侧栏切到"图片检测" → 上传一张 jpg → 开始检测
```
