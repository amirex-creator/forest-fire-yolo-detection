# 兼容性核实报告（安装前，未执行任何安装）

- **核实时间：** 2026-10-27 00:47–01:05（Asia/Shanghai）
- **核实范围：** PyTorch / Torchvision / Ultralytics / CUDA / OpenCV 在本机（Windows 10 + Python 3.11.9 + RTX 3060 Laptop 6GB + 驱动 566.07）上的版本兼容性
- **本轮未做的事：** 未创建虚拟环境、未安装任何包、未下载任何模型权重或数据集
- **重要更正：** 阶段一报告中"推荐 torch 2.14.1 + torchvision 0.29.1"的**结论已作废**，理由见第一节。以本报告为准。

---

## 一、核实结论（先看这里）

| 项目 | 结论 | 依据等级 |
| --- | --- | --- |
| 推荐 PyTorch | **torch 2.13.0** | 官方安装页 + 官方兼容矩阵 + Ultralytics CI，三重一致 |
| 推荐 Torchvision | **torchvision 0.28.0** | 同上（官方矩阵明确 2.13 ↔ 0.28） |
| 安装源 | **https://download.pytorch.org/whl/cu126** | 官方安装页原文给出该命令 |
| Python | 3.11.9 ✅（要求 >=3.10，且矩阵覆盖 3.10–3.14） | 官方矩阵 |
| NVIDIA 驱动 | 566.07 ✅（CUDA 12.x 需要 >= 525） | NVIDIA 官方兼容性文档 |
| CUDA 13.x 轮子 | ❌ 不可用（需要驱动 >= 580） | NVIDIA 官方兼容性文档 |
| Ultralytics | 8.4.190，与 torch 2.13.0 组合**在官方 CI 中被测试** | Ultralytics CI 配置 |
| OpenCV | 建议 `opencv-python==4.14.0.94`，**不用 5.0.0.93** | 理由见第六节 |
| 本地训练 | 可行，但需保守配置（见第七节） | 显存 6 GB / 内存 15.6 GB |

**未验证的部分（不假装已验证）：** CUDA 是否真正可用、`sm_86` 内核是否包含在轮子中、Ultralytics 在本机 Windows + cu126 组合下的实跑、OpenCV 在中文路径下的读写行为。这四项只能在阶段四安装后用真实代码验证。

---

## 二、为什么放弃 torch 2.14.1 + torchvision 0.29.1

上一版推荐是**仅凭 PyPI 索引里的版本号推出来的**，缺少官方依据。逐条核实后发现三个问题：

1. **PyTorch 官方安装页没有 2.14 这个版本。**
   抓取 `https://pytorch.org/get-started/previous-versions/` 全文（348,259 字节）后统计：字符串 `2.14` 出现 **0 次**，`2.13` 出现 8 次。该页面最顶部的最新版本条目就是 v2.13.0。
   也就是说：2.14.1 的轮子确实躺在下载服务器上（能下），但 PyTorch **没有为它发布官方安装命令**。

2. **torchvision 官方兼容矩阵的最新一行是 2.13 / 0.28。**
   `pytorch/vision` 仓库 README 的对照表：`2.13 → 0.28`、`2.12 → 0.27`、`2.11 → 0.26`……**没有 2.14 / 0.29 这一行**。

3. **Ultralytics 官方 CI 测到的最后一组稳定组合就是 2.13.0 + 0.28.0。**
   `.github/workflows/ci.yml` 的版本矩阵里逐条列出 `torch: "2.13.0" / torchvision: "0.28.0"`，之后就没有更新的稳定条目了。

**结论：** 2.14.1 / 0.29.1 属于"文件存在但无官方发布与测试背书"的组合。对课程项目来说没有必要冒这个险，因此改用 **2.13.0 / 0.28.0**。

> 补充：2.14.1 与 0.29.1 之间并非绝对不兼容——torchvision 0.29.1 的 PyPI 元数据写的是 `torch>=2.14.0`，理论上自洽。但"理论上自洽"不等于"官方支持"，本项目采用前者已足够新（Ampere 架构、Python 3.11 全部覆盖）。

---

## 三、官方依据原文（可复核）

### 3.1 PyTorch 官方安装页（v2.13.0 段落原文）

页面结构与抓取到的原文（HTML 去标签后）：

```
v2.13.0
Wheel
OSX                     pip install torch==2.13.0 torchvision==0.28.0
Linux and Windows
  # ROCM 7.1 (Linux only)
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.1
  # ROCM 7.2 (Linux only)
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/rocm7.2
  # CUDA 12.6
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
  # CUDA 13.0
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
  # CUDA 13.2
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
  # CPU only
    pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

来源：<https://pytorch.org/get-started/previous-versions/>
注意官方把 v2.13.0 标注为 **"Linux and Windows"**，即 Windows 在官方支持范围内。

### 3.2 torchvision 官方版本对照表（原文节选）

| `torch` | `torchvision` | Python |
| --- | --- | --- |
| `main` / `nightly` | `main` / `nightly` | `>=3.11`, `<=3.14` |
| **`2.13`** | **`0.28`** | **`>=3.10`, `<=3.14`** |
| `2.12` | `0.27` | `>=3.10`, `<=3.14` |
| `2.11` | `0.26` | `>=3.10`, `<=3.14` |
| `2.10` | `0.25` | `>=3.10`, `<=3.14` |

来源：<https://raw.githubusercontent.com/pytorch/vision/main/README.md>
Python 3.11.9 落在 `>=3.10, <=3.14` 内 ✅

### 3.3 Ultralytics CI 中的版本矩阵（原文节选）

```yaml
- os: ubuntu-latest
  torch: "2.12.0"
  torchvision: "0.27.0"
- os: ubuntu-latest
  torch: "2.13.0"        # ← 矩阵中最新的一组稳定组合
  torchvision: "0.28.0"
```

来源：<https://raw.githubusercontent.com/ultralytics/ultralytics/main/.github/workflows/ci.yml>
另外 ROCm 的 GPU 作业也使用 `torch[device-all]==2.13.0+rocm10.0.0` / `torchvision[device-all]==0.28.0+rocm10.0.0`。

**注意（诚实的边界）：** 这些 CI 作业跑在 `ubuntu-latest` 上；`windows-latest` 作业用的是 `torch: [latest]`。所以准确表述是"**Ultralytics 官方测试过 2.13.0+0.28.0 这组版本**"，而不是"官方测试过 Windows + cu126 这个具体组合"。

### 3.4 NVIDIA 官方驱动兼容表

| CUDA Toolkit | 最低驱动版本 | 备注 |
| --- | --- | --- |
| CUDA 13.x | **>= 580** | 本机 566.07 **不满足** |
| CUDA 12.x | **>= 525** | 本机 566.07 **满足** ✅ |
| CUDA 11.x | >= 450 | — |

来源：<https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>

本机 `nvidia-smi` 显示 566.07 / 驱动支持 CUDA 12.7。因此：
- **cu126 可用**（12.6 ≤ 12.7，属于同一 major 版本内的 minor compatibility）；
- **cu130 / cu132 不可用**（需要 580+ 驱动）。这正是我们排除 CUDA 13.x 通道的硬依据。

### 3.5 轮子真实存在性（不是"看名字像"，而是实际发请求验证）

对 `https://download.pytorch.org/whl/cu126/` 索引做了精确匹配，并对命中文件发 `Range: bytes=0-1023` 请求：

| 文件 | HTTP | 总大小 |
| --- | --- | --- |
| `torch-2.13.0+cu126-cp311-cp311-win_amd64.whl` | 206 | 2,594,548,547 字节（≈ 2.42 GiB） |
| `torchvision-0.28.0+cu126-cp311-cp311-win_amd64.whl` | 206 | 8,520,320 字节（≈ 8.1 MiB） |
| `torch-2.14.1+cu126-cp311-cp311-win_amd64.whl` | 206 | 2,602,753,280 字节（存在但不推荐） |
| `torchvision-0.29.1+cu126-cp311-cp311-win_amd64.whl` | 206 | 6,111,881 字节（存在但不推荐） |

`206 Partial Content` + 正确的 `Content-Range` 说明文件不仅存在、而且可断点下载。cp311 / win_amd64 标签表明**与 Python 3.11 + 64 位 Windows 精确匹配**，不需要编译。

---

## 四、Ultralytics 依赖兼容性核实

Ultralytics 8.4.190 的核心依赖（来自 `pyproject.toml` 与 PyPI 元数据，两处一致）：

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

逐项核对：

| 约束 | 本机/所选版本 | 结论 |
| --- | --- | --- |
| `requires-python >= 3.8` | 3.11.9 | ✅ |
| `torch >= 1.8.0` 且 `!= 2.4.0` (win) | 2.13.0 | ✅ 满足，且不是被排除版本 |
| `torchvision >= 0.9.0` | 0.28.0 | ✅ |
| `opencv-python >= 4.7.0, != 4.13.0.90` | 4.14.0.94 | ✅ |
| `numpy >= 1.23.0` | 由 pip 解析（最新 2.4.6） | ✅ 满足下限 |
| `ultralytics-platform >= 0.1.45` (py>=3.11) | 会自动安装 | ⚠️ 会额外引入一个依赖包，属正常现象 |

**结论：Ultralytics 的声明约束与 torch 2.13.0 / torchvision 0.28.0 完全兼容；并且这组版本出现在其官方 CI 矩阵中。** 但"约束兼容 + CI 测过该版本对" ≠ "已在你本机实测"，最终仍以阶段四的真实推理结果为准。

---

## 五、Python 3.11.9 相关核实

| 检查项 | 结果 |
| --- | --- |
| torch 2.13.0 `requires_python` | `>=3.10` → 3.11.9 ✅ |
| cp311 win_amd64 轮子 | 存在（见 3.5）✅ |
| torchvision 0.28.0 对应 Python 范围 | `>=3.10, <=3.14` ✅ |
| torch 2.13.0 依赖 `setuptools>=77.0.3` | 新建 venv 自带 65.5.0，pip 会自动升级到满足版本（安装时可见） |
| streamlit 1.66.1 `requires_python` | `>=3.11` ✅（3.11.9 正好达标，**不要再降 Python 版本**） |
| pandas 3.1.0 / matplotlib 3.11.2 `requires_python` | 均 `>=3.11` ✅ |

---

## 六、OpenCV 版本选择：为什么不直接装最新

- PyPI 上 `opencv-python` 最新是 **5.0.0.93**（5.x 是一次大版本跃迁）。
- Ultralytics 只在声明里要求 `>=4.7.0, !=4.13.0.90`，其 CI 与生态长期基于 4.x。
- OpenCV 5.x 的 API 与行为变化**没有证据**表明已被 Ultralytics 8.4.190 覆盖。

因此建议显式钉住 **4.x 的最后一个版本 `opencv-python==4.14.0.94`**（已确认存在 `cp37-abi3-win_amd64` 轮子，abi3 兼容 Python 3.11）。
这是"保守选择"，不是"5.0 一定不能用"；若后续需要 5.x 特性，再单独升级并回归测试。

同理，`pandas 3.1.0` 也是一次大版本（3.x）。它只在"结果导出表格"用到，风险较低，**先按最新装，若 `results.to_df()` 或画图报错，再降到 2.x**（具体版本届时按报错再定，不提前瞎猜）。

---

## 七、6 GB 显存 + 剩余 2.6 GB 内存的保守训练配置

> 依据：显存 6144 MiB（当前空闲 5933 MiB）、物理内存 15.63 GB（检查时仅剩 2.62 GB）。

### 7.1 推荐起始配置（阶段六小规模试跑用）

| 参数 | 建议值 | 理由 |
| --- | --- | --- |
| 模型 | `yolov8n.pt`（必要时 `yolov8s.pt`） | 6 GB 显存的安全选择；n 参数量约 3.2M，s 约 11M |
| `imgsz` | **640** | YOLO 默认值；先别提高 |
| `batch` | **8**（OOM 则 4，再不行 2） | 640×batch8 在 6 GB 上通常有余量，但不保证 |
| `workers` | **2**（最多 4） | Windows 上 DataLoader 多进程开销大，且当前空闲内存少 |
| `amp` | **True** | 混合精度，省显存又提速，Ampere 支持良好 |
| `cache` | **False** | 缓存会把整份数据集读进内存，现在只剩 2.6 GB，**务必别开** |
| `device` | `0`（GPU）；失败回退 `cpu` | 先确认 `torch.cuda.is_available()` |
| `epochs`（试跑） | **1–3** | 只为验证流程与测真实耗时 |
| `patience` | 20–50（正式训练时） | 早停，避免无谓耗时 |

### 7.2 显存/内存不足时的降级顺序

1. `batch` 8 → 4 → 2
2. `imgsz` 640 → 512
3. 换更小模型（`yolov8n` 已是最小，可考虑 nano 的更小输入）
4. `workers` 2 → 0（单进程，牺牲速度换稳定）
5. 关闭 AMP（**最后手段**，通常反而更费显存）

### 7.3 训练前的内存释放建议（本机情况特殊，必须做）

检查时内存占用 83%、只剩 2.62 GB，主要占用来自 QQ（`QQNT`，同时也占着 GPU 的一小部分显存）。建议：

1. 关闭 **QQ / 微信 / 浏览器多标签**（浏览器往往占用 2–4 GB）；
2. 训练期间不要开视频、不要开大型 IDE 的多个项目；
3. 用任务管理器确认"可用内存"回到 **6 GB 以上**再开始正式训练；
4. 训练脚本里打印 `psutil.virtual_memory().available`，把真实数字记录下来（这是论文材料，不是装饰）；
5. 不要设置太大的 `workers`，Windows 下每个 worker 都会复制一份进程内存。

### 7.4 磁盘预算

| 项目 | 预估 |
| --- | --- |
| torch cu126 轮子 | 2.42 GiB（下载） |
| 解压后 torch + 依赖 | 约 5–7 GB |
| venv 整体 | 约 7–9 GB |
| pip 缓存（可 `pip cache purge` 清理） | 约 3 GB |
| E: 盘可用 | **530 GB** ✅ 完全够 |

---

## 八、中文与空格路径的风险评估

当前项目路径：`E:\DeepSeek Harness\森林防火yolo`（含 1 个空格 + 中文目录名）。

### 8.1 已知风险点

| 风险 | 说明 | 严重度 |
| --- | --- | --- |
| `cv2.imread` / `cv2.imwrite` | Windows 版 OpenCV 用 ANSI(C 字符串) 打开文件，非 ASCII 路径可能**直接返回 None 而不报错**，表现为"图片读不到/静默失败" | 高 |
| Ultralytics 内部读图 | 其数据集加载默认走 OpenCV，若同样不做转码，训练时可能报 `No images found` 或漏图 | 高（**未实测**） |
| `torch.load` / 权重路径 | 新版本 torch 通过 `pathlib`/`os` 处理，通常安全 | 低 |
| pip 安装路径 | pip 对 Unicode 路径处理正常，历史上有过偶发报错 | 低 |
| Streamlit | 上传文件走内存，不落盘到中文路径时风险低 | 低 |
| 命令行 | PowerShell 对中文路径一般正常，`cd` 时建议加引号 | 低 |

### 8.2 处置建议（两条路，二选一）

**方案 P1（推荐，彻底解决）：改用纯英文短路径**
例如 `E:\fs_smoke`。项目一开始就迁移，成本几乎为零；等数据集和权重都生成后再搬会很麻烦。
**我不会自行移动或删除您已有的任何文件**——本次目录里只有我生成的 `docs/environment_report.md`。是否迁移由您决定。

**方案 P2（保留现路径）：代码层面统一做路径安全处理**
- 全部用 `pathlib.Path`，不写死盘符；
- 读图统一走 `np.fromfile(str(p), dtype=np.uint8)` + `cv2.imdecode(...)`；
- 写图统一走 `cv2.imencode(...)` + `tofile`（不要用 `cv2.imwrite`）；
- 给 Ultralytics 传路径时先 `str(Path(...).resolve())`，并在训练前用脚本验证"它能读到第一张图"；
- 数据集与输出目录尽量保持 **ASCII 子目录名**（如 `datasets/forest_smoke/images/train`，这部分本来就是英文 ✅）。

**我的建议：** 如果这套代码最终要交给老师/同学在别的电脑跑，P1 更省心。若您希望保留 `森林防火yolo` 这个名字，P2 也能做，但必须按上面的写法执行，且需要在阶段五/六实测验证。

---

## 九、最终建议的安装命令（**尚未执行，等您确认**）

> 说明：venv 必须建在项目目录内。阶段一已实测：本会话沙箱只允许写工作区，`D:\python` 写入被拒绝，因此不动系统 Python。

```powershell
# 0) 进入项目目录（若采用 P1，请改成新路径）
cd "E:\DeepSeek Harness\森林防火yolo"

# 1) 用 3.11.9 建独立虚拟环境（不动系统 Python，无需管理员权限）
& "D:\python\python.exe" -m venv .venv

# 2) 激活
.\.venv\Scripts\Activate.ps1
#    若提示"禁止运行脚本"，先只对当前用户放行一次：
#    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3) 升级 pip（venv 自带的 pip 24.0 较旧；torch 要求 setuptools>=77）
python -m pip install --upgrade pip setuptools wheel

# 4) 安装 GPU 版 PyTorch（官方原文命令，cu126 通道）
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# 5) 安装其余依赖（回官方 PyPI；不要用清华源，本机实测 SSL 失败）
python -m pip install ultralytics==8.4.190 "opencv-python==4.14.0.94" pandas matplotlib streamlit pyyaml tqdm pytest psutil
```

**两点必须注意：**

1. **第 4 步和第 5 步必须分开执行**。第 4 步带 `--index-url` 指向 PyTorch 私有索引，如果和第 5 步混在一起，pip 会去 PyTorch 索引里找 ultralytics 而找不到。
2. 第 5 步**不要**加 `--index-url`，也不要混用 `--extra-index-url`（容易造成依赖混淆）；如需加速可用 `-i https://mirrors.aliyun.com/pypi/simple/`。

预计下载量：torch 约 2.42 GiB + 其余约 400 MB。耗时取决于带宽，**未实测**。

---

## 十、安装后必须执行的验证命令（阶段四用）

逐条执行，输出要**如实记录**（这是论文与环境报告的真实材料）：

```powershell
# V1 版本对齐
python -c "import torch, torchvision, sys; print('python', sys.version); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_build', torch.version.cuda); print('cudnn', torch.backends.cudnn.version())"

# V2 CUDA 是否真的可用（这是唯一可信的判定）
python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

# V3 真实矩阵运算（比 is_available 更硬的证据；能跑通说明内核真的加载了）
python -c "import torch; a=torch.randn(1024,1024,device='cuda'); b=torch.randn(1024,1024,device='cuda'); c=(a@b).sum().item(); torch.cuda.synchronize(); print('matmul_ok', c)"

# V4 显存与 Ultralytics
python -c "import torch; print('vram_total_GB', round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)); import ultralytics; print('ultralytics', ultralytics.__version__)"
```

V3 通过前，**都不要说"CUDA 可用"**。若 V2 为 False 但 V3 能跑，说明是环境变量/索引问题，仍按失败处理并排查。

---

## 十一、失败排查步骤（按顺序，不要跳）

| 现象 | 最可能原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版轮子 | `pip show torch` 看版本号是否带 `+cu126`；没有就重装第 4 步 |
| 同上 | 第 5 步把 torch 覆盖了 | `pip install` 输出里若出现 `Downloading torch-2.13.0 (CPU)` 说明混装了，卸载重来 |
| 同上 | 驱动/显卡不可见 | `nvidia-smi` 是否正常；笔记本是否走集显（在 NVIDIA 控制面板里把 python.exe 设为独显） |
| `OSError: [WinError 126] 找不到指定的模块` | 缺 VC++ 运行库 | 已确认 `vcruntime140.dll`、`msvcp140.dll` 存在（阶段一查过），若仍报错再装 VC++ 2015-2022 x64 |
| 安装时解析超时 / 卡住 | 网络波动 | 重试；或第 5 步改用阿里云 `-i` |
| `ERROR: Could not find a version that satisfies torch==2.13.0` | 用错了源（把第 4 步加到 PyPI 源） | 必须带 `--index-url https://download.pytorch.org/whl/cu126` |
| `ModuleNotFoundError: No module named 'cv2'` | 装到了系统 Python | 确认提示符前缀有 `(.venv)`，`python -c "import sys; print(sys.executable)"` 应指向 `.venv\Scripts\python.exe` |
| `CUDA out of memory`（后续训练） | batch/imgsz 太大 | 按 7.2 降级 |
| 训练时 `No images found` / 图片数为 0 | 中文路径导致 OpenCV 静默失败 | 见第八节：迁移到 ASCII 路径，或改用 `imdecode` 读图 |
| Streamlit 启动报 `requires_python>=3.11` 冲突 | 用了别的 Python | 必须用 `.venv` 内的 3.11.9 |

---

## 十二、备选/回退方案

| 场景 | 方案 |
| --- | --- |
| torch 2.13.0 在本机出现未知问题 | 回退 **torch 2.12.0 + torchvision 0.27.0**（同属官方矩阵与 CI 矩阵，cu126 轮子存在） |
| 必须用 2.14.1/0.29.1 | 可用，但需自行承担无官方发布背书的风险，并在报告中注明 |
| 显存实在不够 | 换 CPU 训练（`device=cpu`）会慢很多但可行；或降到 `imgsz=512` |
| CUDA 始终不可用 | 用 CPU 版轮子 `--index-url https://download.pytorch.org/whl/cpu` 完成流程开发，训练另行考虑云 GPU |

---

## 附录：本轮核实用到的原始命令

```bash
# 索引精确匹配 + 断点请求验证轮子真实可下载
requests.get("https://download.pytorch.org/whl/cu126/torch/")
requests.get(url, headers={"Range": "bytes=0-1023"})   # 期望 206 + Content-Range

# 官方安装页统计（2.14 出现 0 次，2.13 出现 8 次）
requests.get("https://pytorch.org/get-started/previous-versions/")

# 官方兼容矩阵 / CI 矩阵
requests.get("https://raw.githubusercontent.com/pytorch/vision/main/README.md")
requests.get("https://raw.githubusercontent.com/ultralytics/ultralytics/main/.github/workflows/ci.yml")
requests.get("https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml")

# 依赖元数据
requests.get("https://pypi.org/pypi/torch/2.13.0/json")
requests.get("https://pypi.org/pypi/ultralytics/8.4.190/json")
```

NVIDIA 官方驱动兼容表：<https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
