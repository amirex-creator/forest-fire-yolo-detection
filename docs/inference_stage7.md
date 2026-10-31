# 阶段七：模型推理模块（`src/inference.py` + `predict.py`）

> 本文件记录**已实际跑通**的推理能力与真实结果。
> 所有数字都来自 2026-10-28 在本机 RTX 3060 Laptop 上的真实执行，未做任何修饰。

---

## 一、本阶段做了什么

| 交付物 | 状态 |
| --- | --- |
| `src/inference.py`（推理核心，25,153 B） | ✅ 已实现并被真实调用 |
| `predict.py`（命令行入口，7,376 B） | ✅ 已实现并被真实调用 |
| `tests/test_inference.py`（16 项测试） | ✅ **16 passed** |
| 单图 / 批量推理 | ✅ 实跑 |
| JSON / CSV 结构化结果 | ✅ 实跑并核对 |
| 无目标图片处理 | ✅ 实跑（不报错、正常写空结果） |
| 权重缺失友好提示 | ✅ 由测试覆盖 |

权重用的是第一批训练的产物：
`models/batch1_e10_full_20261028_130129_best.pt`（6,479,146 B，类别 `{0: smoke, 1: fire}`）。

## 二、能力清单

1. **单张图片**推理，输出检测框、类别、置信度；
2. **目录批量**推理（可选 `--recursive`、`--limit`）；
3. `--conf` 置信度阈值、`--iou` NMS 阈值可调，**非法值在加载模型之前就报错**；
4. 保存渲染后的图片（带框、带类别与置信度）；
5. 输出 **JSON**（完整结构）与 **CSV**（每个框一行，无目标图也写一行空检测）；
6. **没有检测到目标时不报错**，正常返回空列表并在摘要里标 `[无目标]`；
7. 耗时拆成四段：读图 / 纯预测 / 画图写盘 / 端到端；
8. 权重自动选择：只认本项目训练产物（`*_best.pt`、`*_last.pt`），**拒绝把 `yolov8n.pt` 这类 COCO 权重当烟雾模型**；
9. 记录权重路径、大小、修改时间、sha256、类别表、加载耗时、预热耗时；
10. 退出码区分场景：`0` 成功、`1` 未预期异常、`2` 参数错误、`3` 权重问题、`4` 目录里没有图片。

## 三、命令行用法

```powershell
# 1) 单张图片（自动挑最新的 *_best.pt）
.\.venv\Scripts\python.exe predict.py --source datasets\forest_smoke\images\test\AoF06155.jpg

# 2) 指定权重与阈值
.\.venv\Scripts\python.exe predict.py --source path\to\img.jpg --weights models\batch1_e10_full_20261028_130129_best.pt --conf 0.4

# 3) 目录批量（只看前 30 张，递归子目录）
.\.venv\Scripts\python.exe predict.py --source datasets\forest_smoke\images\test --limit 30 --recursive

# 4) 只要结构化结果，不保存渲染图
.\.venv\Scripts\python.exe predict.py --source path\to\images --no-save-image --json out\res.json --csv out\res.csv
```

默认输出目录：`outputs/predictions/predict_<YYYYmmdd_HHMMSS>/`。

## 四、真实运行记录

### 4.1 单张正样本：`AoF06155.jpg`

- 该图人工标注为 **2 个 smoke**；模型给出 **1 个 smoke**（漏检 1 个）。
- 结果：`outputs/predictions/stage7_demo/`，退出码 0。
- 耗时（已预热）：端到端 146.51 ms，其中纯预测 59.1 ms。

### 4.2 单张负样本：`AoF00003.jpg`

- 该图标签为 0 字节（无目标）；模型给出 **0 个框**，摘要显示 `[无目标]`，退出码 0。
- 说明"没有检测到目标"是一条正常路径，不是异常。

### 4.3 批量 30 张（`--limit 30`，test 集按文件名排序的前 30 张）

- 真实情况：这 30 张的标签**全部为空**（0 个 smoke、0 个 fire）。
- 模型输出：有目标 3 张 / 无目标 27 张，共 3 个框。
- **这 3 个框全是误报**（`AoF00115`、`AoF00116`、`AoF00117`，各 1 个 smoke）。
  → 在 conf=0.25 下，负样本上的误报率 **3/30 = 10%**。这是真实结果，写论文时要如实反映。
- 耗时（每张平均）：纯预测 10.97 ms / 读图 12.21 ms / 画图写盘 4.24 ms / 端到端 27.47 ms。

### 4.4 批量 12 张正样本（从 test 集挑前 12 张含 smoke 的图）

| 文件 | 标注 | 预测 |
| --- | --- | --- |
| AoF04049 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04050 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04051 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04052 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04053 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04054 | smoke 1 / fire 3 | smoke 1 / fire 2 |
| AoF04055 | smoke 1 / fire 3 | smoke 1 / fire 2 |
| AoF04440 | smoke 1 / fire 1 | smoke 1 / fire 2 |
| AoF04441 | smoke 1 / fire 1 | smoke 1 / fire 2 |
| AoF04469 | smoke 1 / fire 3 | smoke 1 / fire 5 |
| AoF04489 | smoke 1 / fire 2 | smoke 1 / fire 2 |
| AoF04562 | smoke 1 / fire 1 | smoke 1 / fire 1 |
| **合计** | smoke 12 / fire 24 | smoke **12** / fire **26** |

- 烟雾框 12/12 全中；火焰框 26 个（多 2 个，属于同一片火焰被拆成多个框）。
- 目视检查渲染图（`AoF04049__pred.jpg`）：框贴合真实火焰与烟羽，标签文字 `smoke 0.34` / `fire 0.74` 正常，中文未乱码。
- 耗时（每张平均）：纯预测 23.27 ms / 读图 14.68 ms / 画图写盘 4.17 ms / 端到端 42.41 ms。

> 单张跑时 146 ms、批量跑时 27–42 ms，差异来自进程启动后的一次性开销（显存分配、算子加载）。
> **批量数字才接近稳态**，论文里报告推理速度应说明是哪种测法。

## 五、输出结构

- `results.json`：`generated_at` / `source` / `model`（权重路径、大小、sha256、类别、加载与预热耗时）/ `summary`（张数、框数、各段耗时、失败数）/ `images`（每张图的框列表与耗时）/ `disclaimer`。
- `results.csv`（UTF-8 BOM，Excel 可直接打开）：16 列
  `image_path, image_name, num_boxes, read_ms, predict_ms, render_ms, total_ms,
   class_id, class_name, confidence, x1, y1, x2, y2, w_norm, h_norm`
  ——**无目标的图片也会有一行**，`num_boxes=0`，检测相关列为空，避免"结果里少了图"的误会。
- 渲染图：`<原文件名>__pred.jpg`。

## 六、关键实现决定（附理由）

1. **不直接用 `cv2.imread` 读图**（用户第 4 条约束）：统一走 `src/image_io.safe_imread`，
   它用 `np.fromfile` + `cv2.imdecode`，能处理中文路径与带空格路径，且对后缀/存在性做前置校验。
2. **加载后先做一次空白预热**：第一版把首次推理的 CUDA 初始化算进了单图耗时，
   实测被夸大到 **3,420 ms/张**；预热后降到 146 ms，纯预测 59 ms。
3. **耗时拆成四段**：否则"推理慢"到底是读图慢、预测慢还是存图慢，无法判断。
4. **默认权重只认训练产物**：`_list_trained_weights()` 明确排除 `yolov8n.pt` 等 COCO 权重。
   用 COCO 权重跑出来的 "person/car" 不是烟雾检测，这条约束能防止误用。
5. **无目标也写 CSV 行**：下游统计（比如按图片算误报率）需要知道"这张图被处理过且没检出"。

## 七、未验证与限制（已完成的都标了 ✅，没做的照实写）

1. **未验证**：还没有在 Streamlit 网页里调用（阶段九）；
2. **未验证**：没有跑完整个 test 集（3,229 张）的指标统计，那是阶段八的事；
3. **未验证**：没有对比 CPU 推理速度（目前只测了 GPU）；
4. conf=0.25 在 30 张负样本上误报 3 张，说明阈值可能偏低，**需要在阶段八用 PR 曲线定阈值**，不能凭感觉；
5. 只测了 `.jpg`；`.png` / `.bmp` 的读取由单元测试覆盖，但没有真实样本跑过；
6. 推理速度随图片分辨率变化（本批图片多为 1280×720 到 1920×1080），论文报告时应给出图片尺寸范围。

> 模型识别结果仅供辅助分析，需人工核实；本系统不做森林火灾预警，不能仅凭检测结果确认真实火灾。
