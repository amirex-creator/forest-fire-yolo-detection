# 基于 YOLO 的森林火灾烟雾与明火检测

> 《人工智能基础》课程项目 —— 目标检测 + 实验 + 网页演示
>
> ⚠️ **重要声明**：本系统检测图像中的 `smoke` 和 `fire`，属于辅助分析工具。
> 检测到疑似目标**不等于确认发生森林火灾**，不能作为火灾预警依据，所有识别结果都必须由人工核实。

## 提交包说明

此文件夹是课程提交版。已包含最终 40 轮模型、10 轮基线模型、项目代码、实验报告、误检案例、系统截图、PPTX 和逐页讲稿。提交或汇报时使用 `outputs/deliverables/` 中带 `v3` 的文件。

## 最简单的查看方式

不需要安装 Python、VS Code 或任何依赖，直接双击根目录中的 `start.bat`，或直接打开 `index.html`。浏览器会离线展示项目简介、真实指标、实验对比、误检案例、图表、系统截图、PPT 和讲稿。中文入口 `双击查看项目.bat`、`查看项目.html` 也可以使用。

这个离线页面适合老师查看提交成果；它不运行模型。若要运行交互式图片/视频检测系统，再按后面的环境安装说明操作。`http://localhost:8501` 只对启动 Streamlit 的那台电脑有效，不能作为发给其他电脑的通用网址。

Windows 上运行交互式系统时，优先双击根目录的 `启动检测系统.bat`。它会自动寻找真正安装了 Streamlit 的 Python 环境；如果 8501 已被其他程序占用，会自动改用 8502，并打开正确的网址。需要公网演示时，等网页打开后再运行 `start_cloudflare_tunnel.bat`。

完整数据集约 3 GB，未放入提交包；数据来源、许可说明和准备步骤见 `data/数据准备说明.md` 与 `data/dataset_info.md`。运行图片检测不需要下载完整数据集，可以直接上传图片；如需本地样例或重新训练，再按说明准备 `datasets/forest_smoke/`。

若要手动运行交互式系统，在本文件夹打开 PowerShell 后启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

然后访问 `http://localhost:8501`。首次运行前需按本 README 的依赖说明建立 `.venv` 并安装依赖。

---

## 一、当前状态（务必先读）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段一 | 环境检查 | ✅ 已完成（`docs/environment_report.md`） |
| — | 依赖版本兼容性核实 | ✅ 已完成（`docs/compatibility_final.md`） |
| 阶段二 | 项目目录与基础文件 | ✅ 已完成（本文件所属阶段） |
| 阶段三 | 数据集准备与检查 | ✅ 已完成（21,527 图 / 26,531 有效框 / 按组 70-15-15 划分 / 校验 0 阻塞错误） |
| 阶段四 | 安装 YOLO 环境 | ✅ 已完成（venv 4.75 GiB；V1–V4 全 PASS，**CUDA 可用**） |
| 阶段五 | 数据检查工具 | ✅ 已实现（`src/dataset_utils.py`，阶段三已实际全量跑过） |
| 阶段六 | 模型训练 | ✅ 基线 10/30 轮与困难负样本累计 40 轮均已完成；40 轮最终 val mAP50 **0.7118** / mAP50-95 **0.4024** |
| 阶段七 | 推理 | ✅ 已完成（`src/inference.py` + `predict.py`；单图/批量/JSON/CSV 均实跑；16 项测试通过） |
| 阶段八 | 评价与实验 | ✅ 四组模型统一 test 比较；40 轮模型 mAP50 **0.6909** / mAP50-95 **0.3738** |
| 阶段九 | Streamlit 网页 | ✅ 图片、视频、实验对比、误检案例和模型数据工作台已实现，使用新界面 |
| 阶段十 | 自动化测试 | ✅ 核心数据、推理、评价测试已通过；最终结果已重新生成 |
| 阶段十一 | 论文辅助材料 | ✅ 真实指标、训练报告、案例图和讲稿素材已具备 |
| 阶段十二 | 最终说明 | ✅ PPTX、逐页讲稿、系统截图和交付清单已完成 |

最终权重、评价报告和系统产物均已写入本项目目录；所有指标都来自实际运行，不使用占位数字。

---

## 二、项目简介

在森林、野外和复杂背景图像中定位疑似烟雾与明火区域，输出：

- 检测框位置（像素坐标 + 归一化坐标）；
- 类别（`smoke` / `fire`）；
- 置信度；
- 可视化结果图与结构化结果（JSON / CSV）。

技术路线：YOLOv8n（Ultralytics）+ PyTorch + OpenCV，预训练权重迁移学习，
Web 演示使用 Streamlit。默认可以本地运行；启用 `FOREST_SMOKE_PUBLIC_MODE=1` 后，图片在运行 Streamlit 的服务器处理，不会转发到其他推理服务。

## 三、技术栈与版本

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.11.9 | 已实测存在；streamlit 1.66 要求 ≥3.11，不能降级 |
| PyTorch | 2.13.0 (cu126) | 官方历史版本页 + vision 官方矩阵背书 |
| torchvision | 0.28.0 | 与 2.13.0 官方配对 |
| ultralytics | 8.4.190 | CI 稳定矩阵就是用 2.13.0 / 0.28.0 |
| opencv-python | 4.14.0.94 | 锁 4.x（5.x 无兼容证据） |
| numpy | 由依赖自动决定 | 注意：本机现存 numpy 2.4.6 偏新，需在独立环境内重装 |

版本选择的完整依据与排除理由见 **`docs/compatibility_final.md`**。

## 四、项目目录说明

```
fs_smoke/
├── app.py                     # Streamlit 网页（阶段九）
├── train.py                   # 训练入口（阶段六）
├── predict.py                 # 推理命令行（阶段七）
├── evaluate.py                # 评价入口（阶段八）
├── requirements.txt           # 依赖清单（含两步安装说明）
├── README.md                  # 本文件
├── .gitignore                 # 排除数据/权重/输出
├── configs/
│   ├── forest_smoke.yaml      # Ultralytics 数据集配置（nc=2, smoke/fire）
│   └── train.yaml             # 训练超参
├── src/
│   ├── paths.py               # 路径统一入口（唯一允许定义目录的地方）
│   ├── environment.py         # 环境信息采集（只读，纯标准库）
│   ├── image_io.py            # 中文/空格路径安全的图片读写（P2 备用防护）
│   ├── dataset_utils.py       # 数据检查（阶段五）
│   ├── inference.py           # 推理封装（阶段七）
│   └── evaluation.py          # 指标与实验记录（阶段八）
├── data/                      # 数据来源与准备说明（完整数据集不随包提供）
├── datasets/forest_smoke/     # 按 data/说明下载并整理后生成
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset_info.md
├── models/                    # 训练得到的 *_best.pt / *_last.pt
├── outputs/
│   ├── predictions/           # 检测结果图
│   ├── figures/               # 图表
│   ├── reports/               # 指标与统计报告
│   └── train_runs/            # 训练实验目录
├── docs/                      # 环境/兼容性/数据集/实验/论文文档
└── tests/                     # 结构、接口与推理测试
```

**路径约定**：代码里**不写死任何盘符或绝对路径**。所有路径都由
`src/paths.py` 通过 `Path(__file__).resolve().parents[1]` 反推项目根目录，
把整个项目复制到别的位置也能直接运行。

## 五、环境安装步骤

> 项目已在独立 CUDA 环境中完成训练和推理。下面的命令用于在另一台机器复现。

```powershell
# 在项目根目录打开 PowerShell，再运行以下命令

# 1) 创建独立虚拟环境（不动系统 Python）
python -m venv .venv

# 2) 激活
.\.venv\Scripts\Activate.ps1
#    若提示"禁止运行脚本"，先执行（仅当前会话生效）：
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3) 升级 pip 与 setuptools（torch 2.13.0 要求 setuptools>=77.0.3）
python -m pip install --upgrade pip setuptools wheel

# 4) 安装 PyTorch（独立一步，走官方 cu126 索引，约 2.42 GiB）
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# 5) 安装其余依赖（走 PyPI 官方源；本机清华源 SSL 失败，禁用）
python -m pip install ultralytics==8.4.190 "opencv-python==4.14.0.94" pandas matplotlib streamlit pyyaml tqdm pytest psutil

# 6) 验证环境（本仓库自带，只读采集，会输出真实结果）
python -m src.environment
```

> 💡 下载慢时可用阿里云镜像（本机实测可访问）：
> `-i https://mirrors.aliyun.com/pypi/simple/`
> 但 **PyTorch 那一步必须用官方 cu126 索引**，镜像可能没有 `+cu126` 轮子。

## 六、数据集准备方法

完整数据集未放入提交包。来源、许可边界和准备步骤见 **`data/数据准备说明.md`**、**`data/dataset_info.md`** 与 **`docs/dataset_guide.md`**。

| 项目 | 实际情况 |
| --- | --- |
| 数据集 | D-Fire（Kaggle 镜像 `sayedgamal99/smoke-fire-detection-yolo`），许可 CC0 |
| 规模 | 21,527 张图；26,531 个有效框（smoke 11,846 / fire 14,685）；9,838 张空标签负样本 |
| 类别 | **0 = smoke，1 = fire**（双类别检测） |
| 划分 | 15,069 / 3,229 / 3,229（70.00% / 15.00% / 15.00%，seed=42，按组划分） |
| 校验 | 图片不可读 0、缺标签 0、类别非法 0；26 个缺陷框仅告警不阻塞 |
| 原项目实测报告 | `outputs/reports/dataset_summary.{json,md}`、`split_manifest.json`（用于说明原训练集统计；重新准备数据后可再生成） |

要点：

1. 只接受**带矩形框标注**的检测数据集，只有分类标签的不能用；
2. 标签格式 `class cx cy w h`（归一化），类别编号与 `configs/forest_smoke.yaml` 必须一致；
3. 无目标图片用**空标签文件**，不得编造框；
4. 划分按"组"进行（dHash 近似重复聚类），同组只进一个集合。
### 6.1 数据检查与可视化（阶段五，已可用）

```powershell
# 完整检查：扫描 + 标签校验 + 统计 + 画图 + 写报告（首次约 3 分钟，有缓存时约 1 分钟）
python src/dataset_utils.py --check

# 只重画图：读取已有 JSON 报告，不重新扫描（几秒钟）
python src/dataset_utils.py --plots-only

# 每个集合抽 12 张有框图 + 3 张无目标负样本，换随机种子
python src/dataset_utils.py --plots-only --samples 12 --samples-negative 3 --seed 7
```

产物：

| 路径 | 内容 |
| --- | --- |
| `outputs/reports/dataset_summary.json` | 机器可读的完整统计、异常清单、去重结果 |
| `outputs/reports/dataset_summary.md` | 人读报告（末节列出可视化产物） |
| `outputs/figures/dataset_inspect/class_distribution.png` | 类别标签数量分布柱状图 |
| `outputs/figures/dataset_inspect/samples_train.png` | train 抽样样本（画出真实标注框） |
| `outputs/figures/dataset_inspect/samples_val.png` | val 抽样样本 |
| `outputs/figures/dataset_inspect/samples_test.png` | test 抽样样本 |

> 绘图**只读取**图片与标签，不修改任何数据集文件；抽样同时包含"无目标"负样本，
> 避免只看成功案例造成"所有图都有烟"的错觉。退出码 0 = 无阻塞错误，3 = 有阻塞性标签错误。


## 七、模型训练命令（阶段六，分批训练已跑通）

> 训练采用**分批递进**：每批 10 轮，逐批从上一批最佳权重继续，并保存完整报告。
> 目前已完成 10/30 轮基线，以及困难负样本累计 40 轮微调。40 轮批次耗时约 37.1 分钟，
> 使用 RTX 3060 Laptop 6 GB、batch=8、imgsz=640、AMP；训练峰值分配显存约 0.94 GB，未发生训练 OOM。

```powershell
# 只做数据校验、不训练
.\.venv\Scripts\python.exe train.py --check-data-only --deep-check

# 小规模试跑（验证流程与真实耗时/显存）
.\.venv\Scripts\python.exe train.py --smoke-test --trial-size 3000

# 分批正式训练（每批 10 轮，显存不足时立即停止）
.\.venv\Scripts\python.exe train.py --full --epochs 10 --workers 1 --batch 8 --imgsz 640 --device 0 --no-downgrade --name batch1_e10

# 续训（仅在你确认要追加轮次时）：用上一批 last.pt 起步
.\.venv\Scripts\python.exe train.py --full --epochs 10 --workers 1 --batch 8 --imgsz 640 --device 0 --no-downgrade --weights models/batch1_e10_full_<时间戳>_last.pt --name batch2_e10

# 继续困难负样本模型（示例：从 30 轮最佳权重追加 10 轮）
.\.venv\Scripts\python.exe train.py --full --config configs/train_hardneg.yaml --data-config configs/forest_smoke_hard_negatives_full.yaml --epochs 10 --workers 1 --batch 8 --imgsz 640 --device 0 --no-downgrade --weights models/hardneg_finetune_cont20_full_20261029_024922_best.pt --name hardneg_finetune_cont30 --epoch-budget 40
```

参数集中在 `configs/train.yaml`，不要在多处重复定义。

批次台账在 `outputs/reports/training_ledger.json`；每批训练前会自动对全部图片做
sha256 快照，训练后列出**真被改写**的图片（Ultralytics 会就地修复损坏的 JPEG）。

训练产物（真实存在）：

| 路径 | 内容 |
| --- | --- |
| `outputs/train_runs/<run>/train.log` | 完整训练日志 |
| `outputs/train_runs/<run>/batch8/` | Ultralytics 原始输出（results.csv、weights/、混淆矩阵） |
| `outputs/figures/<run>/loss_curves.png` | loss 与 mAP 曲线（本项目自绘） |
| `outputs/reports/train_<run>.json` / `.md` | 参数、资源峰值、每轮指标、数据集改动 |
| `models/<run>_best.pt` / `_last.pt` | 该批权重（约 6.5 MB） |

> 训练报告会记录越界标注、数据快照和图片是否被改写。40 轮批次扫描 21,527 张图片，
> 训练前后内容变化为 0；Ultralytics 在训练日志中忽略了 3 张越界 train 标注和 1 张越界 val 标注。

## 八、模型推理命令（阶段七已完成并实跑）

```powershell
# 单张图片（自动挑最新的 *_best.pt）
.\.venv\Scripts\python.exe predict.py --source datasets\forest_smoke\images\test\AoF06155.jpg

# 指定权重与置信度阈值
.\.venv\Scripts\python.exe predict.py --source path\to\img.jpg --conf 0.4

# 目录批量（前 30 张，递归子目录）
.\.venv\Scripts\python.exe predict.py --source datasets\forest_smoke\images\test --limit 30 --recursive

# 只要结构化结果，不保存渲染图
.\.venv\Scripts\python.exe predict.py --source path\to\images --no-save-image --json out\res.json --csv out\res.csv
```

输出在 `outputs/predictions/predict_<时间戳>/`：渲染图 + `results.json` + `results.csv`（无目标图片也有一行）。

退出码：`0` 成功、`1` 未预期异常、`2` 参数错误、`3` 权重问题、`4` 目录无图片。

> 默认权重只接受本项目训练产物（`*_best.pt` / `*_last.pt`）；
> `yolov8n.pt` 是 COCO 权重，不含 smoke/fire，会被明确拒绝。
> 最终 40 轮模型的部署阈值由 val 网格搜索得到 `conf=0.30`，并原样用于 test。
> 详见 `outputs/reports/calibrated_test_results_40epoch.csv`。

## 九、模型评价命令（阶段八已完成并实跑）

```powershell
# 完整评价：test 集一次性最终评价（指标 + 图表 + 错误案例）
.\.venv\Scripts\python.exe evaluate.py

# 只算指标，不收集错误案例（更快）
.\.venv\Scripts\python.exe evaluate.py --no-error-cases

# 指定最终 40 轮权重与验证集标定阈值
.\.venv\Scripts\python.exe evaluate.py --weights models\hardneg_finetune_cont30_full_20261029_034435_best.pt --conf 0.30

# 只重写 metrics.md（不重跑模型）
.\.venv\Scripts\python.exe evaluate.py --rebuild
```

最终报告：`outputs/reports/evaluation_hardneg_40epoch_cases/metrics.{json,md}`；
图表：`outputs/figures/evaluation_hardneg_40epoch_cases/`；误报和漏检图在
`outputs/predictions/error_cases_20261029_042905/`。

**最终 40 轮模型的真实结果（test 集 3,229 张 / 3,878 框）**：

| 指标 | 总体 | smoke | fire |
| --- | --- | --- | --- |
| Precision | 0.7027 | 0.7671 | 0.6382 |
| Recall | 0.6346 | 0.7103 | 0.5589 |
| mAP50 | **0.6909** | 0.7651 | 0.6166 |
| mAP50-95 | **0.3738** | 0.4561 | 0.2915 |

conf=0.30 / IoU=0.5 下：TP 2,296 / FP 943 / FN 1,582（误报图 654 张、漏检图 841 张）。

统一实验对比见 `outputs/reports/experiment_comparison.{json,csv,md}`：40 轮模型在验证集选定
`conf=0.30`，test 空标签图误报率为 **89/1645 = 5.41%**，test 阈值化 F1 为 **0.6452**。

> 实验二（imgsz 512 vs 640）与实验三（复杂场景分组）**未完成**，报告里明确标注，没有编造对比数字。
> 详见 `docs/evaluation_stage8.md` 与 `docs/experiment_plan.md`。

## 十、网页启动命令（阶段九已完成并实跑）

```powershell
# 在项目根目录打开 PowerShell，再运行：
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端提示的地址（默认 http://localhost:8501）。

也可以直接双击 `启动检测系统.bat`。它会自动检查依赖和端口，并在服务就绪后打开浏览器。不要双击 `app.py`，那只会用文本编辑器打开源代码，不会启动网页。

四个页面（侧栏切换）：

| 页面 | 说明 |
| --- | --- |
| 项目介绍 | 项目与模型概况、数据规模、最终 test 指标、适用范围、免责声明 |
| 图片检测 | 上传 JPG/JPEG/PNG（≤20 MB）→ 检测 → 结果图、框数、类别、置信度、JSON/CSV 下载 |
| 视频检测 | 上传 MP4/AVI/MOV/MKV（≤120 MB）→ 逐帧检测 → 输出视频、框数、耗时和 FPS |
| 实验结果 | 读取四组模型的真实对比、图表和误检案例 |
| 模型说明 | 权重 SHA256、训练配置、阈值含义、系统边界和免责声明 |

- 只列出本项目训练产物（`*_best.pt` / `*_last.pt`），`yolov8n.pt` 不会被当作烟雾模型；
- 模型用 `st.cache_resource` 缓存，**整个进程只加载一次**，调阈值不会重新加载权重；
- 默认本地模式在本机处理；公网模式在当前 Streamlit 服务器处理，不转发到其他推理服务；
- 本地模式产物在 `outputs/predictions/web_<时间戳>/`，公网模式产物在 `outputs/predictions/public/`，并按保留期清理。

实测记录见 `docs/app_stage9.md`；系统默认使用最新的 40 轮最佳权重，检测结果只供课程演示。

### 课程交付物

- 汇报 PPTX：`outputs/deliverables/森林火灾烟雾与明火检测_课程汇报_20261029_v3.pptx`
- 逐页讲稿：`outputs/deliverables/逐页讲稿_森林火灾检测_20261029_v3.md`
- 系统截图：`outputs/screenshots/system_image_detection.png`
- PPT 校验收据：`outputs/.presentation_build/validation_v3.json`（14 页、字体、版式和图表结构校验通过）
- 旧版本 PPT/讲稿保留在同一目录，便于回溯；提交或汇报时使用 `v3` 文件。

## 十一、常见报错解决方案

| 报错现象 | 原因 | 处理办法 |
| --- | --- | --- |
| `python3` 打不开或跳转应用商店 | `python3` 指向 WindowsApps 占位程序 | 用 `python` 或完整路径 `D:\python\python.exe` |
| `Activate.ps1 无法加载，禁止运行脚本` | PowerShell 执行策略 | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| 装 torch 时自动装成 CPU 版 | 用了默认 PyPI 源 | 必须加 `--index-url https://download.pytorch.org/whl/cu126` |
| 清华源 `SSL: UNEXPECTED_EOF_WHILE_READING` | 本机到清华源握手失败 | 换阿里云源或官方源 |
| `CUDA out of memory` | 6 GB 显存不够 | 降 `batch`（8→4→2）、降 `imgsz`（640→512）、保持 `amp: true` |
| 训练中途电脑卡死 | 可用内存不足（检查时仅 2.62 GB） | 关闭 QQ/浏览器等；**不要**把 `cache` 设为 true |
| 训练卡在 `Scanning` 不动 | Windows 下 `workers>0` 多进程问题 | 把 `configs/train.yaml` 的 `workers` 改为 `0` |
| `FileNotFoundError: 权重文件不存在` | 尚未训练或路径写错 | 先跑阶段六训练，或用 `--weights` 指定已有权重 |
| `torch.cuda.is_available()` 为 True 但一跑就崩 | CUDA 运行时/driver 不匹配 | 按 `docs/compatibility_final.md` 第十二节排查表逐项核对 |
| `ModuleNotFoundError: No module named 'src'` | 不在项目根目录执行 | `cd` 到项目根目录再运行 |

## 十二、数据与模型的来源、许可

- **数据集**：D-Fire，来自 Kaggle 镜像 `sayedgamal99/smoke-fire-detection-yolo`
  （<https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo>），
  平台标注许可 **CC0: Public Domain**；其中 `PublicDataset` 前缀图片源自
  HPWREN / FESB / FireNet，原始条款未逐一核实，本项目仅用于课程学习。
  压缩包 SHA256 `65d81b60…5c6d83`。**不含纯森林专属数据**：该数据集含城市、
  工业等非森林场景，论文中不得表述为"森林专用数据集"。
- **预训练权重**：使用 Ultralytics YOLOv8n 预训练权重（`yolov8n.pt`），遵循其 AGPL-3.0 许可；
  本项目仅用于课程学习。
- **训练权重**：保存在 `models/`，最终模型为
  `hardneg_finetune_cont30_full_20261029_034435_best.pt`。
- **第三方库**：PyTorch（BSD 风格）、Ultralytics（AGPL-3.0）、OpenCV（Apache-2.0）、
  Streamlit（Apache-2.0），具体许可以其官方仓库为准。

## 十三、项目的局限性

1. **单帧检测**：区分 `smoke` 与 `fire`，但视频系统逐帧处理，不做时序建模；
2. **数据域限制**：D-Fire 同时包含林地、城市、工业和夜间画面，不是纯森林数据集；
3. **硬件约束**：6 GB 显存只能跑轻量模型（YOLOv8n/s）与小尺寸输入；
4. **易混淆场景**：云雾、晨雾、扬尘、逆光可能与烟雾混淆，误报难以完全避免；
5. **不可作为安全依据**：模型输出仅供辅助分析，必须人工核实，
   **不得用于真实火灾预警或安全决策**；
6. **无跨域验证**：若测试集与训练集来自同一批视频，指标会偏高，
   论文中必须说明适用范围。

## 十四、免责声明

本项目为《人工智能基础》课程作业，用于学习目标检测的完整流程。
系统输出的所有结果都是**模型推断**，可能存在漏检与误检；
**模型识别结果仅供辅助分析，需人工核实**，不构成任何火灾预警或安全建议。

## 十五、GitHub 与 Cloudflare 公网访问

GitHub 只保存代码、部署文件、报告和经过确认的演示模型；它不会替代 Python 推理服务器。Cloudflare Pages 不能直接运行当前的 PyTorch、Ultralytics 和 Streamlit 应用。本项目提供两种公网方式：

1. **临时演示**：运行 `run_streamlit_public.bat`，再运行 `start_cloudflare_tunnel.bat`，把 Cloudflare 输出的 `trycloudflare.com` 地址发给访问者。运行电脑必须保持开机。
2. **长期部署**：使用 `Dockerfile` 将服务部署到云服务器，再把 Cloudflare Tunnel 指向服务器的 `127.0.0.1:8501`。完整步骤见 `docs/deployment_github_cloudflare.md`。

公网模式默认使用 CPU，环境变量如下：

```text
FOREST_SMOKE_PUBLIC_MODE=1
FOREST_SMOKE_DEVICE=cpu
FOREST_SMOKE_MODEL=models/hardneg_finetune_cont30_full_20261029_034435_best.pt
```

公网模式的上传和结果目录为 `outputs/predictions/public/`，默认保留 24 小时后清理。不要上传个人隐私或敏感图片，也不要把 Cloudflare 凭证提交到 GitHub。GitHub 仓库已经公开并核对完整；当前仍没有可用的公网检测网址，因为本机网络禁止 `cloudflared` 连接 Cloudflare 边缘。配置文件和启动脚本已经准备好，具体错误和处理方式见 `docs/deployment_github_cloudflare.md`。
