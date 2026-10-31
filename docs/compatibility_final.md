# 兼容性最终确认文档（阶段一收口）

- **文档性质：** 汇总确认版。整合 `docs/environment_report.md`（阶段一环境检查报告）与 `docs/compatibility_check.md`（安装前兼容性核实报告），作为后续阶段唯一的版本/环境依据。
- **生成时间：** 2026-10-27 01:15（Asia/Shanghai）
- **当前执行状态：** 仅完成文档整理。**未创建虚拟环境、未执行任何 pip 安装、未下载任何数据集或模型权重。**
- **CUDA 状态：** 【未验证】。必须留到阶段四安装后，用第七节 V1–V4 命令实测判定，在此之前不得宣称"CUDA 可用"。
- **原始报告处理：** 本文件不改动 `environment_report.md` 与 `compatibility_check.md`。二者内容为历史记录，如有冲突，以**本文件**为准。

> 说明：`environment_report.md` 第 125–130 行有一处"本节推荐已作废"的补记，是上一轮（本次约束下达之前）就更正的，属于原始报告的历史痕迹，本轮未再改动。

---

## 一、本机硬件与软件基线

### 1.1 操作系统与运行时

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Windows 10 Home China，版本 25H2，Build 26200.9711 |
| 架构 / 长路径 | 64 位（AMD64）；`LongPathsEnabled = 1` |
| 系统区域 / 代码页 | zh-CN / GB2312（936） |
| PowerShell | 5.1.26100.9711 |
| 计算机名 / 用户 | 云落 / 13613 |

### 1.2 计算与存储资源

| 项目 | 实测值 | 对项目的影响 |
| --- | --- | --- |
| CPU | 12th Gen Intel Core i7-12700H，20 逻辑处理器 | 数据预处理与 CPU 推理充足 |
| 物理内存 | 15.63 GB 总计，检查时可用 **2.62 GB（占用 83%）** | 训练时必须先释放内存，`workers` 不能大 |
| 独立显卡 | **NVIDIA GeForce RTX 3060 Laptop GPU**，6144 MiB，Ampere，compute capability 8.6 | 本地训练可行 |
| 显存占用 | 检查时占用 64 MiB，空闲 5933 MiB | 空闲充足 |
| 集成显卡 | Intel Iris Xe Graphics（驱动 32.0.101.6790） | 需确认 python 走独显 |
| NVIDIA 驱动 | **566.07**，`nvidia-smi` 报告支持 CUDA 12.7 | 只支持 CUDA 12.x 通道 |
| CUDA Toolkit | **未安装** | 本项目**不需要**安装 |
| 磁盘 | C: 130 GB 可用 / D: 89 GB / **E: 530 GB（项目盘）** / F: 353 GB | 空间充足 |

### 1.3 Python 环境

| 项目 | 实测值 |
| --- | --- |
| 解释器 | `D:\python\python.exe`，**Python 3.11.9**（MSC v.1938 64-bit） |
| pip | 24.0 |
| venv | 可用（`python -m venv`） |
| Conda | **未安装**（本项目不需要） |
| `python3` 命令 | 指向 WindowsApps 占位程序，**禁止使用** |
| pip 镜像配置 | 无（`pip config list` 为空） |

**已安装第三方包（仅 39 项）**：`numpy 2.4.6`、`Pillow 13.0.0`、`requests 2.34.2`、`lxml 6.1.3`、`setuptools 65.5.0` 等。
**项目核心依赖全部缺失**：`torch`、`torchvision`、`ultralytics`、`opencv-python`、`pandas`、`matplotlib`、`streamlit`、`PyYAML`、`tqdm`、`pytest`、`psutil`。

### 1.4 网络连通性（实测）

| 目标 | 结果 | 用途 |
| --- | --- | --- |
| `https://pypi.org/simple/` | 200 | 安装非 torch 依赖 ✅ |
| `https://mirrors.aliyun.com/pypi/simple/` | 200 | 备用镜像 ✅ |
| `https://pypi.tuna.tsinghua.edu.cn/simple/` | **SSL 失败**（`UNEXPECTED_EOF_WHILE_READING`） | **禁止使用清华源** ❌ |
| `https://download.pytorch.org/whl/cu126` | 200 | 安装 torch ✅ |
| `https://download.pytorch.org/whl/cu130` | 200（源可访问，但轮子需要 580+ 驱动） | 本机**不可用** ❌ |
| `https://github.com/` | 200 | 下载 YOLO 权重 ✅ |
| `https://api.github.com/` | 403（未认证限流） | 不影响 Release 直接下载 |
| `yolov8n.pt`（GitHub Release） | 200，6,549,796 字节，已读到 zip 文件头 | 权重可下载 ✅ |

> 注：本机 `curl` 因 Windows schannel 凭据问题全部失败，属沙箱限制，**不代表网络不通**。统一用 `python -m pip` / `requests` / 浏览器。

---

## 二、版本决策：排除项与选定项

### 2.1 被排除的组合：`torch 2.14.1 + torchvision 0.29.1`

| 排除理由 | 具体证据 | 官方来源 |
| --- | --- | --- |
| PyTorch 官方安装页**没有 2.14 条目** | 抓取该页全文 348,259 字节：字符串 `2.14` 出现 **0 次**，`2.13` 出现 8 次；页面最顶部最新版本条目即 v2.13.0 | <https://pytorch.org/get-started/previous-versions/> |
| torchvision 官方兼容矩阵**最新一行是 2.13 / 0.28** | 对照表只列到 `2.13 → 0.28`，**不存在 2.14 / 0.29 这一行** | <https://raw.githubusercontent.com/pytorch/vision/main/README.md> |
| Ultralytics 官方 CI **未测试**该组合 | CI 版本矩阵最后一条稳定条目为 `torch: "2.13.0" / torchvision: "0.28.0"`，之后无更新条目 | <https://raw.githubusercontent.com/ultralytics/ultralytics/main/.github/workflows/ci.yml> |
| 轮子虽存在，但不构成支持依据 | `torch-2.14.1+cu126-cp311-cp311-win_amd64.whl` 实际存在（2,602,753,280 字节，HTTP 206 可下载），但"能下载" ≠ "官方发布" | download.pytorch.org 索引实测 |

**结论：** 2.14.1 / 0.29.1 属于"文件存在但无官方发布与测试背书"的组合。
补充说明（保持客观）：二者并非绝对不兼容——`torchvision 0.29.1` 的 PyPI 元数据为 `torch>=2.14.0`，理论上自洽。但课程项目无必要承担无背书风险，故排除。

### 2.2 选定组合（**锁定，不再变动**）

| 组件 | 锁定版本 | 官方依据（逐项） | 依据强度 |
| --- | --- | --- | --- |
| Python | **3.11.9**（`D:\python\python.exe`） | torch 2.13.0 `requires_python >= 3.10`；torchvision 矩阵标注 Python `>=3.10, <=3.14`；streamlit 1.66.1 要求 `>=3.11` → 3.11.9 是满足全部约束的既有解释器 | 官方元数据 |
| PyTorch | **torch==2.13.0** | 官方安装页 v2.13.0 段落明确标注 "Linux and Windows"，并给出 cu126 安装命令原文 | 官方安装页 |
| Torchvision | **torchvision==0.28.0** | torchvision 官方矩阵：`2.13 ↔ 0.28`，Python `>=3.10, <=3.14` | 官方矩阵 |
| 安装索引 | **`https://download.pytorch.org/whl/cu126`** | 官方安装页原文命令即使用该索引；cu126 = CUDA 12.6 运行时 | 官方安装页 |
| CUDA 运行时通道 | **cu126** | NVIDIA 官方：CUDA 12.x 最低驱动 **>= 525**；CUDA 13.x 需 **>= 580**。本机 566.07 → 12.x ✅ / 13.x ❌ | NVIDIA 官方 |
| Ultralytics | **ultralytics==8.4.190** | PyPI 最新稳定版；依赖约束 `torch>=1.8.0`（Windows 排除 `2.4.0`）与 2.13.0 相容；其 CI 矩阵含 2.13.0+0.28.0 | PyPI 元数据 + CI |
| OpenCV | **opencv-python==4.14.0.94** | Ultralytics 要求 `>=4.7.0, !=4.13.0.90`；PyPI 最新为 5.0.0.93（5.x 大版本跃迁，无证据表明已被 8.4.190 覆盖），故锁 4.x 末版 | PyPI 元数据 + 保守原则 |

**轮子真实存在性（实测，非推测）**——对每个文件发 `Range: bytes=0-1023` 请求：

| 文件 | HTTP | 总大小 |
| --- | --- | --- |
| `torch-2.13.0+cu126-cp311-cp311-win_amd64.whl` | 206 | 2,594,548,547 字节（≈ 2.42 GiB） |
| `torchvision-0.28.0+cu126-cp311-cp311-win_amd64.whl` | 206 | 8,520,320 字节（≈ 8.1 MiB） |

cp311 / win_amd64 标签表明与"Python 3.11 + 64 位 Windows"精确匹配，**无需本地编译**。

### 2.3 必须写明的边界：Windows + cu126 组合【未在 Ultralytics CI 覆盖】

准确表述（不夸大、不模糊）：

- ✅ **已验证（有官方依据）**：Ultralytics 官方 CI 的版本矩阵中存在 `torch 2.13.0 + torchvision 0.28.0` 这组**版本对**（作业运行在 `ubuntu-latest`）；
- ❌ **未覆盖**：CI 的 `windows-latest` 作业使用的是 `torch: [latest]`，**没有任何一个官方 CI 作业同时覆盖 "Windows + torch 2.13.0 + cu126"**；
- ⚠️ **【未验证】**：`sm_86` 内核是否包含在该 cu126 轮子中（按 PyTorch 官方对 Ampere 的支持应当包含，但**本机未实测，不作断言**）；
- ⚠️ **【未验证】**：Ultralytics 8.4.190 在本机 Windows + cu126 环境下的实际推理表现。

**因此：本机安装后必须完成第七节 V1–V4 全部验证，才算"环境就绪"。在 V3 通过之前，禁止在任何文档或汇报中声称"CUDA 可用"。**

---

## 三、Ultralytics 依赖校验结果

`ultralytics 8.4.190` 核心依赖（`pyproject.toml` 与 PyPI 元数据两处一致）：

```toml
requires-python = ">=3.8"
"numpy>=1.23.0; platform_system != 'Darwin'"
"matplotlib>=3.3.0"
"opencv-python>=4.7.0,!=4.13.0.90"
"pillow>=7.1.2"
"pyyaml>=5.3.1"
"requests>=2.23.0"
"torch>=1.8.0"
"torch>=1.8.0,!=2.4.0; sys_platform == 'win32'"   # Windows 上 2.4.0 有 CPU 报错，官方排除
"torchvision>=0.9.0"
"psutil>=5.8.0"
"polars>=0.20.0"
"nvidia-ml-py>=12.0.0"
"ultralytics-thop>=2.2.3"
"ultralytics-platform>=0.1.45; python_version >= '3.11'"
```

逐项校验：

| 约束 | 锁定值 | 结论 |
| --- | --- | --- |
| `requires-python >= 3.8` | 3.11.9 | ✅ 通过 |
| `torch >= 1.8.0` 且 `!= 2.4.0`（win） | 2.13.0 | ✅ 通过，且不属于被排除版本 |
| `torchvision >= 0.9.0` | 0.28.0 | ✅ 通过 |
| `opencv-python >= 4.7.0, != 4.13.0.90` | 4.14.0.94 | ✅ 通过 |
| `numpy >= 1.23.0` | pip 自动解析（当前最新 2.4.6） | ✅ 通过（下限满足） |
| `matplotlib >= 3.3.0` | 最新 3.11.2（要求 Python >=3.11） | ✅ 通过 |
| `pyyaml >= 5.3.1` / `psutil >= 5.8.0` | 未钉版本，pip 解析 | ✅ 通过 |
| `ultralytics-platform >= 0.1.45`（py>=3.11） | 自动安装 | ⚠️ 正常现象：会额外引入一个依赖包，非报错 |

**校验结论：** 锁定组合与 Ultralytics 声明的全部约束**无冲突**；且 `torch 2.13.0 + torchvision 0.28.0` 出现在其官方 CI 矩阵中。
但"约束兼容 + CI 测过版本对" ≠ "本机实测通过"，后者由阶段四负责。

**已知需观察项（不是错误，先记录）**：
1. `pandas 3.1.0` 属大版本（3.x），仅用于结果导出/绘图；若 `results.to_df()` 或画图报错，再降级到 2.x（届时按真实报错定版本，不预先瞎猜）。
2. pip 解析出的 numpy 版本可能较新（如 2.4.x）；若出现 numpy ABI 警告，再钉版本。

---

## 四、训练资源约束与保守参数

> 依据：显存 6144 MiB（空闲 5933 MiB）、物理内存 15.63 GB（检查时可用 2.62 GB）。

### 4.1 保守训练参数表（阶段六试跑起点）

| 参数 | 取值 | 理由 |
| --- | --- | --- |
| `model` | **`yolov8n.pt`**（必要时 `yolov8s.pt`） | 6 GB 显存的安全选择；n ≈ 3.2M 参数，s ≈ 11M |
| `imgsz` | **640** | YOLO 默认值，先不上调 |
| `batch` | **8**（OOM → 4 → 2） | 640×8 在 6 GB 上通常有余量，但不保证 |
| `workers` | **2**（最多 4） | Windows 多进程开销大，且当前可用内存仅 2.62 GB |
| `amp` | **True** | 混合精度，省显存且提速，Ampere 支持良好 |
| `cache` | **False** | 缓存会把数据集读入内存，严禁开启 |
| `device` | `0`（GPU）；异常时回退 `cpu` | 以 V2/V3 验证结果为准 |
| `epochs`（试跑） | **1–3** | 仅验证流程与测真实耗时，不直接跑完整长训练 |
| `patience` | 20–50（正式训练时） | 早停，避免无谓耗时 |
| `seed` | 固定值（阶段二写入配置） | 保证可复现 |

### 4.2 内存释放建议（本机必做）

检查时内存占用 83%、可用仅 2.62 GB，主要占用来自 QQ（`QQNT`，同时占用少量显存）。

1. 训练前关闭 **QQ / 微信 / 浏览器多标签**（浏览器常占 2–4 GB）；
2. 训练期间不开视频、不同时开多个大型项目；
3. 用任务管理器确认"可用内存"回到 **6 GB 以上**再开始正式训练；
4. 训练脚本中打印 `psutil.virtual_memory().available` 并写入日志（真实数字是论文材料）;
5. `workers` 不要超过 4——Windows 下每个 worker 都会复制进程内存。

### 4.3 OOM / 崩溃降级流程（严格按顺序）

| 步骤 | 操作 | 目的 |
| --- | --- | --- |
| 1 | `batch` 8 → 4 → 2 | 显存占用线性下降 |
| 2 | `imgsz` 640 → 512 | 显存占用约按面积平方下降 |
| 3 | 模型 `yolov8s` → `yolov8n` | 参数量与激活值减小 |
| 4 | `workers` 2 → 0 | 单进程，牺牲速度换稳定（针对内存不足/卡死） |
| 5 | 关闭 AMP（**最后手段**） | 通常反而更费显存，仅用于排查异常 |

每次降级都要记录：改了哪个参数、是否解决、真实耗时变化——这是实验二的素材。

### 4.4 磁盘预算

| 项目 | 预估 |
| --- | --- |
| torch cu126 轮子 | 2.42 GiB |
| 解压后 torch + 依赖 | 约 5–7 GB |
| venv 整体 | 约 7–9 GB |
| pip 缓存（可 `pip cache purge` 清理） | 约 3 GB |
| E: 盘可用 | **530 GB** ✅ |

---

## 五、中文 + 空格路径：风险与两套方案

### 5.1 风险说明（事实与未验证项分开写）

当前项目路径：`E:\DeepSeek Harness\森林防火yolo`（1 个空格 + 中文目录名）。

| 风险点 | 说明 | 严重度 |
| --- | --- | --- |
| `cv2.imread` / `cv2.imwrite` | Windows 版 OpenCV 通过 ANSI（C 字符串）API 打开文件，非 ASCII 路径可能**直接返回 `None` 而不抛异常**，表现为"图片读不到"的静默失败 | 高 |
| Ultralytics 内部读图 | 其数据集加载默认走 OpenCV，若不做转码，训练时可能报 `No images found` 或漏图【未实测】 | 高 |
| `torch.load` / 权重路径 | 新版 torch 走 `pathlib`/`os`（宽字符 API），通常安全 | 低 |
| pip 安装路径 | 对 Unicode 路径处理正常 | 低 |
| Streamlit 上传 | 上传文件在内存中处理，不落盘到中文路径时风险低 | 低 |
| PowerShell 命令 | 正常，`cd` 时建议加引号 | 低 |

### 5.2 方案 P1（推荐）：迁移到纯英文路径 `E:\fs_smoke`

**一次性根治**，且避免后续"数据集/权重已生成后再搬会破坏配置"的问题。

**操作必须由您手动完成**（我不会自行移动、重命名或删除本机任何文件）。建议步骤：

```powershell
# 1) 新建纯英文目录（当前项目里只有 docs\ 两份报告 + 本文件，迁移量极小）
New-Item -ItemType Directory -Path "E:\fs_smoke" -Force

# 2) 复制（不是移动）现有 docs 目录到新位置
Copy-Item -Path "E:\DeepSeek Harness\森林防火yolo\docs" -Destination "E:\fs_smoke\docs" -Recurse -Force

# 3) 校验复制结果（应看到 3 个 .md 文件）
Get-ChildItem "E:\fs_smoke\docs"

# 4) 确认无误后，由您自行决定是否删除旧目录（我不代为删除）
```

**迁移后的影响：**
- 后续所有阶段的工作目录改为 `E:\fs_smoke`；
- 本项目**尚无** `.venv`、数据集、权重，因此不存在"环境失效"问题——现在迁移成本最低；
- 需要在会话中告知我切换工作目录，我才会在新路径下继续。

### 5.3 方案 P2（保留现路径）：配套路径安全代码

保留 `E:\DeepSeek Harness\森林防火yolo`，但代码层强制使用 `pathlib` + 二进制读写绕过 ANSI API。以下片段将放入 `src/dataset_utils.py`（阶段五）与 `src/inference.py`（阶段七），此处先行确认写法。

```python
"""安全读写工具：用于兼容中文/空格路径（Windows OpenCV ANSI 限制）。

用法：
    from src.image_io import safe_imread, safe_imwrite
    img = safe_imread(image_path)
    safe_imwrite(out_path, img)

核心原理：
    cv2.imread / cv2.imwrite 在 Windows 上通过 ANSI(C 字符串) API 打开文件，
    非 ASCII 路径可能静默失败（返回 None / 写入失败）。
    np.fromfile / ndarray.tofile 走 Python 文件对象（宽字符 API），可正确处理中文路径；
    cv2.imdecode / cv2.imencode 只在内存中完成编解码，不接触文件系统。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 允许的图片后缀（与项目验收要求一致：JPG / JPEG / PNG）
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def safe_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片，兼容中文/空格路径。失败时抛出明确异常，绝不静默返回 None。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片不存在或不是文件：{p}")

    if p.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{p.suffix}（允许：{sorted(ALLOWED_IMAGE_SUFFIXES)}）")

    buffer = np.fromfile(str(p), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"图片文件为空：{p}")

    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"图片解码失败（文件可能已损坏）：{p}")
    return image


def safe_imwrite(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    """写出图片，兼容中文/空格路径。自动创建父目录，失败时抛异常。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的输出格式：{suffix}（允许：{sorted(ALLOWED_IMAGE_SUFFIXES)}）")

    p.parent.mkdir(parents=True, exist_ok=True)

    param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if suffix in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(suffix, image, param)
    if not ok:
        raise IOError(f"图片编码失败：{p}（shape={getattr(image, 'shape', None)}）")

    buffer.tofile(str(p))
    return p


def as_ascii_safe_source(path: str | Path) -> np.ndarray | str:
    """为 Ultralytics 准备输入：能读出就用内存数组，避免它内部 cv2.imread 踩中文路径。

    返回 ndarray 时，Ultralytics 的 YOLO.predict(source=...) 可直接接受（BGR 数组）。
    """
    p = Path(path)
    if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
        return safe_imread(p)  # 返回 BGR ndarray
    return str(p)
```

**配套编码规范（全项目强制）：**

| 规则 | 做法 |
| --- | --- |
| 路径处理 | 一律 `pathlib.Path`；**禁止**写死 `E:\...` 绝对盘符；根目录用 `Path(__file__).resolve().parents[n]` 或配置文件推导 |
| 读图 | 一律 `safe_imread()`；**禁止**直接 `cv2.imread` |
| 写图 | 一律 `safe_imwrite()`；**禁止**直接 `cv2.imwrite` |
| 传给 Ultralytics 的图片源 | 优先传 ndarray（内存），而非文件路径 |
| 目录名 | 项目内子目录保持纯 ASCII（`datasets/forest_plume/images/train` 等本就为英文）✅ |
| 训练数据配置 | `configs/forest_smoke.yaml` 中路径使用 `pathlib` 生成的字符串，并在训练前用脚本自检"首张图可解码" |

**P2 的残留风险（必须承认）：** Ultralytics 训练阶段读取数据集时由其内部实现决定是否走 OpenCV，该行为**尚未实测**。若阶段六试跑出现 `No images found` 或数据集图片数少于实际，则说明 P2 不足，届时必须改用 P1。这一点会在阶段六首日实测并如实报告。

### 5.4 不推荐的第三条路：目录联接（junction）

`mklink /J E:\fs_smoke "E:\DeepSeek Harness\森林防火yolo"` 看似能"不搬文件就得到英文路径"，但 **`pathlib.Path.resolve()` 会把 junction 解析回真实的中文路径**，导致我们精心做的路径推导又回到原点，反而增加隐蔽性故障。**不采用。**

### 5.5 两方案对照

| 维度 | P1（`E:\fs_smoke`） | P2（保留中文路径） |
| --- | --- | --- |
| 彻底性 | 根治，OpenCV/Ultralytics 都不再受路径影响 | 仅覆盖我们自己的代码，第三方库内部行为仍不可控 |
| 迁移成本 | 现在几乎为零（仅 3 个 md 文件） | 零，但要长期维护 `safe_*` 规范 |
| 后续风险 | 极低 | 训练阶段存在【未实测】风险 |
| 交给他人在别的电脑运行 | 最省心 | 需一并解释路径规范 |
| 是否需我操作文件 | **否**，您手动迁移 | 否 |

---

## 六、安装命令（**仅文本，尚未执行**）

> 前置：路径方案已确定。若选 P1，请把第 0 步的 `cd` 换成 `E:\fs_smoke`。
> 环境策略：venv 建在**项目目录内**，不动系统 Python `D:\python`（阶段一已实测该目录在本会话沙箱下不可写）。

```powershell
# ---- 0. 进入项目目录 ----
cd "E:\DeepSeek Harness\森林防火yolo"     # P1 方案改为: cd "E:\fs_smoke"

# ---- 1. 用现有 3.11.9 创建独立虚拟环境（无需管理员权限）----
& "D:\python\python.exe" -m venv .venv

# ---- 2. 激活虚拟环境 ----
.\.venv\Scripts\Activate.ps1
# 若提示"禁止运行脚本"，先仅对当前用户放行一次，然后重试上面这行：
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# ---- 3. 确认已处于 .venv（输出必须指向 .venv\Scripts\python.exe）----
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip setuptools wheel

# ---- 4. 第一步安装：PyTorch（官方原文命令，cu126 索引）----
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# ---- 5. 第二步安装：其余依赖（走 PyPI 官方源；禁止清华源）----
python -m pip install ultralytics==8.4.190 "opencv-python==4.14.0.94" pandas matplotlib streamlit pyyaml tqdm pytest psutil
```

**三条硬性注意：**

1. **第 4 步与第 5 步必须分开执行，禁止合并成一条 pip 命令。** 第 4 步带 `--index-url` 指向 PyTorch 私有索引；若合并，pip 会去 PyTorch 索引里查找 `ultralytics` 而失败。
2. 第 5 步**不要**加 `--index-url`，也**不要**混用 `--extra-index-url`（易造成依赖混淆）。若需加速，用 `-i https://mirrors.aliyun.com/pypi/simple/`。
3. 全程**不使用清华镜像**（本机实测 SSL 握手失败）。

预计下载量：torch 约 2.42 GiB + 其余约 400 MB。**耗时未实测**，以实际网络为准。

---

## 七、安装后验证命令（V1–V4，阶段四执行）

```powershell
# ---- V1：版本对齐 ----
python -c "import torch, torchvision, sys; print('python', sys.version); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_build', torch.version.cuda); print('cudnn', torch.backends.cudnn.version())"

# ---- V2：CUDA 可见性 ----
python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

# ---- V3：真实 GPU 矩阵运算（唯一可信的"CUDA 可用"证据）----
python -c "import torch; a=torch.randn(1024,1024,device='cuda'); b=torch.randn(1024,1024,device='cuda'); print('matmul_ok', float((a@b).sum())); torch.cuda.synchronize()"

# ---- V4：显存与 Ultralytics ----
python -c "import torch; print('vram_total_GB', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)); import ultralytics; print('ultralytics', ultralytics.__version__)"
```

### 7.1 判定标准（预期值仅供核对，最终以实测输出为准）

| 编号 | 通过条件 | 预期值（待实测核对） |
| --- | --- | --- |
| V1 | `torch.__version__` **必须含 `+cu126`** | `2.13.0+cu126`；`torchvision 0.28.0`；`cuda_build 12.6` |
| V2 | `cuda_available True` 且 `device_count 1` | `name` 含 `RTX 3060 Laptop`；`capability (8, 6)` |
| V3 | 无异常抛出，输出有限浮点数 | `matmul_ok` 为有限数值（非 `nan`/`inf`） |
| V4 | 显存 ≈ 6 GB，Ultralytics 版本为 8.4.190 | `vram_total_GB` 约 5.9–6.0；`ultralytics 8.4.190` |

**判定规则：**
- **V3 通过前，不得声称"CUDA 可用"**；
- 若 V2 为 `False` 而 V3 能跑，视为异常（通常是环境变量或索引指错），按失败处理并排查；
- 四项输出**逐条原样记录**到环境验证记录（阶段四产出），不得只写"验证通过"。

---

## 八、故障排查表

| 现象 | 最可能原因 | 处理 |
| --- | --- | --- |
| `cuda_available` 为 `False` | 装成了 CPU 版轮子 | `pip show torch` 看版本号是否带 `+cu126`；不带则重装第 4 步 |
| 同上 | 第 5 步覆盖了 torch | 安装日志出现 `Downloading torch-2.13.0 (CPU)` 即混装，卸载后分步重装 |
| 同上 | 笔记本默认用集显 | NVIDIA 控制面板 → 管理 3D 设置 → 将 `python.exe` 设为"高性能 NVIDIA 处理器" |
| 同上 | 驱动异常 | `nvidia-smi` 是否正常输出；必要时重装/更新驱动（需用户确认） |
| `OSError: [WinError 126] 找不到指定的模块` | 缺 VC++ 运行库 | 已确认 `vcruntime140.dll`、`msvcp140.dll` 存在；仍报错再装 VC++ 2015-2022 x64（需用户确认） |
| `ERROR: Could not find a version that satisfies torch==2.13.0` | 第 4 步漏了 `--index-url` | 必须带 `https://download.pytorch.org/whl/cu126` |
| 安装解析超时 / 卡住 | 网络波动 | 重试；第 5 步可改 `-i https://mirrors.aliyun.com/pypi/simple/` |
| `ModuleNotFoundError: No module named 'cv2'` | 装进了系统 Python | 确认提示符有 `(.venv)`；`sys.executable` 应指向 `.venv\Scripts\python.exe` |
| 训练 `CUDA out of memory` | batch/imgsz 过大 | 按 4.3 降级流程 |
| 训练报 `No images found` / 图片数为 0 | 中文路径导致 OpenCV 静默失败 | 改用 P1 英文路径；或确认 `safe_imread` 已接管读图 |
| Streamlit 启动报 `requires_python>=3.11` 冲突 | 用了别的 Python 解释器 | 必须在 `.venv`（3.11.9）内启动 |
| `pip install` 提示 `setuptools` 版本过低 | venv 自带 65.5.0，而 torch 要求 `>=77.0.3` | 第 3 步已升级；若跳过则补跑该步 |

---

## 九、需要您确认的两项

### ① 路径方案选择

| 选项 | 内容 | 说明 |
| --- | --- | --- |
| **P1（推荐）** | 迁移到 `E:\fs_smoke` | 由**您手动**执行 5.2 的复制命令；我**不会**移动或删除任何文件。迁移后请告知我切换工作目录 |
| **P2** | 保留 `E:\DeepSeek Harness\森林防火yolo` | 全项目强制 `pathlib` + `safe_imread`/`safe_imwrite`（5.3 已给出完整实现）；Ultralytics 训练阶段仍有【未实测】风险，若试跑失败则改 P1 |

### ② 是否确认采用选定依赖版本组合

| 组件 | 锁定版本 |
| --- | --- |
| Python | 3.11.9（现有 `D:\python\python.exe`） |
| torch | **2.13.0**（cu126 轮子） |
| torchvision | **0.28.0** |
| ultralytics | **8.4.190** |
| opencv-python | **4.14.0.94** |
| 其余（pandas / matplotlib / streamlit / pyyaml / tqdm / pytest / psutil） | 由 pip 解析最新兼容版 |
| 索引 | PyTorch: `https://download.pytorch.org/whl/cu126`；其余: PyPI 官方源 |

**请回复：** `P1` 或 `P2`，以及"版本组合确认 / 需要调整"。

---

## 十、确认之后的推进边界（明确的下一步）

收到上述确认后，我只会启动**【阶段二：建立项目目录和基础文件】**，内容严格限定为：

- 创建目录结构（`configs/`、`src/`、`datasets/forest_smoke/{images,labels}/{train,val,test}`、`models/`、`outputs/{predictions,figures,reports}`、`docs/`、`tests/`）；
- 生成项目骨架文件：`requirements.txt`、`README.md`、`.gitignore`、`configs/forest_smoke.yaml`，以及 `src/` 与 `tests/` 下的**占位/基础**模块（不含已完成业务逻辑的 train/predict 实现，按计划书阶段顺序推进）；
- 所有路径均用 `pathlib`，不写死绝对盘符；
- `.gitignore` 排除 `datasets/`、`models/*.pt`、`outputs/`、`.venv/` 等大文件。

**阶段二依然不会执行任何安装、不会下载数据集或权重。** 安装动作必须等到阶段四，且需要您再次明确同意。

---

## 附录 A：官方依据链接清单（可复核）

| 用途 | 链接 |
| --- | --- |
| PyTorch 历史版本安装命令（含 v2.13.0 + cu126 原文） | <https://pytorch.org/get-started/previous-versions/> |
| torchvision 官方版本对照矩阵 | <https://raw.githubusercontent.com/pytorch/vision/main/README.md> |
| Ultralytics CI 版本矩阵 | <https://raw.githubusercontent.com/ultralytics/ultralytics/main/.github/workflows/ci.yml> |
| Ultralytics 依赖声明 | <https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml> |
| NVIDIA CUDA 驱动版本兼容表 | <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html> |
| torch 元数据（requires_python / setuptools） | <https://pypi.org/pypi/torch/2.13.0/json> |
| ultralytics 元数据 | <https://pypi.org/pypi/ultralytics/8.4.190/json> |
| PyTorch cu126 轮子索引 | <https://download.pytorch.org/whl/cu126/torch/> |

## 附录 B：本文件与原始报告的关系

| 文件 | 角色 | 本次是否改动 |
| --- | --- | --- |
| `docs/environment_report.md` | 阶段一环境检查的原始记录（硬件、Python、网络、磁盘实测） | **未改动**（第 125–130 行的"作废"补记是上一轮写入的） |
| `docs/compatibility_check.md` | 安装前兼容性核实的证据链（官方原文摘录、轮子存在性验证） | **未改动** |
| `docs/compatibility_final.md` | 本文件：两者的汇总确认版，后续阶段的唯一依据 | 新建 |

## 附录 C：当前状态自检

| 检查项 | 状态 |
| --- | --- |
| 是否创建 `.venv` | ❌ 未创建 |
| 是否有任何 pip 安装 | ❌ 无 |
| 是否下载数据集 | ❌ 无 |
| 是否下载模型权重（含 `yolov8n.pt`） | ❌ 无 |
| 是否已宣称 CUDA 可用 | ❌ 未宣称，标注【未验证】 |
| 是否移动/重命名/删除本机文件 | ❌ 无 |
| 已生成的文件 | `docs/environment_report.md`、`docs/compatibility_check.md`、`docs/compatibility_final.md` |
