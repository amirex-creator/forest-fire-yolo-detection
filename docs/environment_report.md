# 环境检查报告（阶段一）

- **检查时间：** 2026-10-27 00:35–00:41（Asia/Shanghai）
- **检查方式：** 仅执行只读检测命令，未安装任何依赖、未下载数据集或模型权重
- **项目工作目录：** `E:\DeepSeek Harness\森林防火yolo`（检查时为空目录）
- **检查执行环境：** DSH 沙箱 `workspace-write`（仅允许写入上述工作区，WMI/CIM 查询被沙箱拒绝，因此改用注册表与 Win32 API 读取硬件信息）

> 说明：本报告所有数值均来自实际命令输出，未做任何推测性填充。凡属"推断/未验证"的内容均已明确标注。

---

## 一、操作系统与 Python 版本

| 项目 | 实际情况 |
| --- | --- |
| 操作系统 | Windows 10 Home China，版本 25H2，Build 26200.9711 |
| 系统架构 | 64 位（AMD64），`Is64BitOS = True` |
| 计算机名 / 用户 | 云落 / 13613 |
| 系统区域 | zh-CN（代码页 GB2312 / 936） |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H，20 个逻辑处理器 |
| 物理内存 | 15.63 GB 总计，检查时可用 2.62 GB（占用率 83%） |
| PowerShell | 5.1.26100.9711 |
| 长路径支持 | `LongPathsEnabled = 1`（已开启） |

## 二、Python 环境

| 项目 | 实际情况 |
| --- | --- |
| Python 解释器 | `D:\python\python.exe` |
| Python 版本 | **3.11.9**（MSC v.1938 64-bit） |
| pip 版本 | 24.0（`D:\python\Lib\site-packages\pip`） |
| `py` 启动器 | 仅注册 3.11：`py -0p` → `-V:3.11 * D:\python\python.exe` |
| `python3` 命令 | 指向 `C:\Users\13613\AppData\Local\Microsoft\WindowsApps\python3`（Microsoft Store 占位程序，**不要使用**） |
| 虚拟环境 | `python -m venv` 可用（已测试 `--help` 返回正常） |
| Conda | **未安装**（`conda` 命令不存在，无 Miniconda/Anaconda 目录） |
| 其他 Python 环境 | 未发现 venv / virtualenv / pyenv |
| pip 镜像配置 | **无**：`pip config list` 为空，`%APPDATA%\pip\pip.ini`、`%ProgramData%\pip\pip.ini`、`D:\python\pip.ini` 均不存在 |

### 已安装的第三方包（共 39 个条目，多数为依赖）

实际存在的：`numpy 2.4.6`、`Pillow 13.0.0`、`requests 2.34.2`、`lxml 6.1.3`、`certifi`、`charset_normalizer`、`idna`、`urllib3`、`six`、`packaging`、`pyparsing`、`narwhals`、`threadpoolctl`、`typing_extensions`、`tzdata`、`setuptools 65.5.0`、`pip 24.0`。

### 本项目所缺的核心依赖（全部未安装）

`torch`、`torchvision`、`ultralytics`、`opencv-python`、`pandas`、`matplotlib`、`streamlit`、`PyYAML`、`tqdm`、`pytest`、`psutil`。

**结论：这是一个"干净"的 Python 环境，项目依赖需要从零安装。**

## 三、GPU 与 CUDA

| 项目 | 实际情况 |
| --- | --- |
| 独立显卡 | **NVIDIA GeForce RTX 3060 Laptop GPU** |
| 显存 | 6144 MiB（6 GB），检查时占用 64 MiB，空闲 5933 MiB |
| 驱动版本 | 566.07 |
| 驱动支持的最高 CUDA 版本 | **CUDA 12.7** |
| 当前显存占用进程 | 仅 QQ（QQNT），占用极小 |
| 集成显卡 | Intel(R) Iris(R) Xe Graphics（驱动 32.0.101.6790） |
| CUDA Toolkit | **未安装**（无 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`） |
| `nvidia-smi` | 可用：`C:\Windows\System32\nvidia-smi.exe` |
| GPU 架构 | Ampere（RTX 3060，compute capability 8.6，PyTorch 官方轮子完整支持） |

### 关于 CUDA 可用性的准确表述

- **驱动层面**：驱动可支持到 CUDA 12.7，因此安装基于 **CUDA 12.x** 的 PyTorch 预编译轮子不需要单独安装 CUDA Toolkit。
- **PyTorch 层面**：`torch.cuda.is_available()` **尚未验证**，因为 PyTorch 目前未安装。此项必须在阶段四安装后用真实代码测试，不能提前宣称"CUDA 可用"。

## 四、磁盘空间

| 盘符 | 总容量 | 已用 | 可用 | 说明 |
| --- | --- | --- | --- | --- |
| C: | 356 GB | 226 GB | **130 GB** | 系统盘，pip 缓存与临时文件默认位置 |
| D: | 99 GB | 11 GB | 89 GB | 当前 Python 安装盘 |
| E: | 532 GB | 1.6 GB | **530 GB** | **项目所在盘，空间充足** |
| F: | 400 GB | 48 GB | 353 GB | — |

**结论：磁盘空间完全满足需求。** 预估需要：PyTorch + CUDA 轮子约 2.5–3 GB，项目虚拟环境约 4–5 GB，数据集（需下载后再定，通常 1–10 GB 量级），权重与训练输出若干 GB。E: 盘 530 GB 余量充裕。

## 五、网络与下载源连通性

测试方式：Python `requests`（走系统证书），并对每个源做了实际 HTTP 请求。

| 目标 | 结果 |
| --- | --- |
| `https://pypi.org/simple/` | **200**，索引 7.6 MB |
| `https://mirrors.aliyun.com/pypi/simple/` | **200**，索引 7.1 MB |
| `https://pypi.tuna.tsinghua.edu.cn/simple/` | **失败**（SSL `UNEXPECTED_EOF_WHILE_READING`） |
| `https://download.pytorch.org/whl/cu126` | **200** |
| `https://download.pytorch.org/whl/cu121` / `cu124` / `cu128` / `cu129` / `cu130` | 均 **200** |
| `https://github.com/` | **200** |
| `https://api.github.com/` | 403（未认证限流，不影响直接下载 Release 资源） |
| Ultralytics 权重 `yolov8n.pt`（GitHub Release） | **200**，Content-Length = 6,549,796 字节，已读到 PK 文件头（zip），说明可正常下载 |
| 实际 pip 下载测试 | 成功下载 `six-1.17.0-py2.py3-none-any.whl`（11 KB）到临时目录，随后已清理 |

**结论：PyPI 官方源、阿里云镜像、PyTorch 官方轮子源、GitHub Release（模型权重）均可访问，安装与权重下载路径可行。**

需要注意的两点：
1. 清华源在本机 SSL 握手失败，**不要配置清华镜像**；如需镜像请用阿里云。
2. `curl` 在本沙箱内因 Windows 凭据（schannel）限制全部失败，属沙箱限制，**不代表网络不通**；后续请用 `python -m pip` 或浏览器下载。

## 六、候选版本（来自 PyPI / PyTorch 官方索引的真实查询结果）

| 包 | 最新版本 | 备注 |
| --- | --- | --- |
| ultralytics | 8.4.190 | 要求 `torch>=1.8.0`（Windows 上排除 `torch==2.4.0`）、`opencv-python>=4.7.0`、`numpy>=1.23.0` |
| torch | 2.14.1 | 要求 Python >= 3.10 |
| torchvision | 0.29.1 | 要求 `torch>=2.14.0`（与 torch 2.14.1 匹配） |
| streamlit | 1.66.1 | 要求 Python >= 3.11 |
| opencv-python | 5.0.0.93 | — |
| pandas | 3.1.0 | 要求 Python >= 3.11 |
| matplotlib | 3.11.2 | 要求 Python >= 3.11 |
| pytest | 9.1.1 | — |

CUDA 轮子可用性（cp311 / win_amd64 实际存在）：

| 通道 | 可用 torch 版本（近期） |
| --- | --- |
| cu126 | 2.11.0 / 2.12.0 / 2.12.1 / 2.13.0 / 2.14.0 / **2.14.1** |
| cu130 | 2.11.0 – 2.14.1（需 580+ 驱动，**本机 566.07 不满足，不可用**） |
| cu128 | 至 2.11.0 |
| cu121 / cu124 | 较旧版本 |

**推荐通道为 `cu126`**（CUDA 12.6 运行时，兼容 566.07 驱动）。

> **⚠️ 本节推荐已作废（2026-10-27 补记）**
> 上表"最新版本"仅来自 PyPI 索引，**不构成兼容性依据**。经官方资料核实后，最终选定
> **`torch 2.13.0 + torchvision 0.28.0`（cu126）**：
> PyTorch 官方安装页给出该组合的 cu126 命令（且页面无 2.14 条目）、torchvision 官方矩阵写明 `2.13 ↔ 0.28`、
> Ultralytics 官方 CI 矩阵亦测试该组合。
> 完整证据链见 **`docs/compatibility_check.md`**，该文档为准。

## 七、已发现的问题

| 编号 | 问题 | 影响 | 处理建议 |
| --- | --- | --- | --- |
| P1 | 系统 Python（`D:\python`）**没有** torch / ultralytics / opencv 等依赖 | 无法直接推理或训练 | 必须建独立虚拟环境并安装依赖 |
| P2 | `D:\python` 在本次会话沙箱下**不可写**（写入被拒绝） | 不能直接往全局 Python 装包 | 这是 DSH `workspace-write` 沙箱策略，不代表系统权限不足。方案：虚拟环境建在**项目目录内**，完全绕开该限制 |
| P3 | 全局 Python 里 `numpy 2.4.6` 版本很新 | 若直接在全局安装，可能与其他包的 numpy 约束冲突 | 虚拟环境隔离，由 pip 自行解析 numpy 版本 |
| P4 | **无 Conda** | 无法用 conda 管理 CUDA | 用 `venv` + 官方 CUDA 轮子即可，无需 Conda |
| P5 | 项目路径含**中文和空格**：`E:\DeepSeek Harness\森林防火yolo` | `cv2.imread` 等基于 C 字符串的 API 在非 ASCII 路径下可能读不到图片 | 代码中统一用 `pathlib` + `np.fromfile/imdecode` 或 PIL 读取；若后续仍出问题，再考虑改建到纯 ASCII 路径（如 `E:\fs_smoke`） |
| P6 | 检查时系统内存占用 83%，可用仅 2.62 GB | 训练时可能与其他程序争抢内存，导致 dataloader 卡顿或崩溃 | 训练前关闭 QQ、浏览器等占用大的程序；Windows 上 `workers` 建议 2–4 |
| P7 | 显存仅 6 GB | 大批次会 OOM | 首轮用 `imgsz=640, batch=8`（或 4）小规模试跑，按实测再上调；开启 AMP |
| P8 | 未安装 CUDA Toolkit | 一般**不需要**（官方轮子自带运行时）；仅做 TensorRT/自定义算子编译才需要 | 本项目不需要安装 |
| P9 | 本机 `curl` 因 schannel 凭据问题不可用 | 不能用 curl 测网/下载 | 统一用 `python -m pip`、`requests` 或浏览器 |
| P10 | 清华 PyPI 镜像 SSL 失败 | 若配了清华源会安装失败 | 不配置清华源；用官方源或阿里云 |
| P11 | 沙箱禁用了 WMI/CIM 查询 | `Get-CimInstance` 全部报"拒绝访问" | 硬件信息已通过注册表与 Win32 API 获得，不影响后续开发 |

## 八、本地训练可行性判断

**结论：本地训练可行，推荐直接在 Windows 本地完成，无需云端 GPU。**

依据：
- GPU：RTX 3060 Laptop 6 GB，Ampere 架构，PyTorch 官方支持完善；
- CPU：i7-12700H 20 线程，数据预处理与 CPU 推理充足；
- 内存 15.6 GB：满足 YOLO 小模型训练（建议 workers 2–4）；
- 磁盘：E: 盘 530 GB 可用；
- 网络：依赖与权重均可下载。

限制与对策：
- **6 GB 显存是主要约束** → 使用 YOLOv8n / YOLOv8s 量级模型，`imgsz=640`，`batch=8` 起步，开启 AMP；必要时降到 `imgsz=512` 或 `batch=4`。
- **训练耗时未实测** → 阶段六必须先做 1–3 epoch 小规模试跑，记录真实耗时后再决定正式训练的轮数，不预设虚假数字。
- 若后续数据集很大或需要更大模型（如 YOLOv8m/l），再评估云端 GPU 方案。

## 九、推荐的安装方案（待您确认后执行）

**方案 A（推荐）：项目内独立 venv + pip 官方 CUDA 轮子**

1. 在项目目录内创建虚拟环境（不动系统 Python，不需要管理员权限）：
   ```
   E:\DeepSeek Harness\森林防火yolo\.venv
   ```
2. 升级 pip，然后用 PyTorch 官方索引安装 GPU 版 PyTorch（版本组合以 `docs/compatibility_check.md` 为准）：
   ```
   python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
   ```
3. 再安装其余依赖（走 PyPI 官方源或阿里云）：
   ```
   python -m pip install ultralytics==8.4.190 "opencv-python==4.14.0.94" pandas matplotlib streamlit pyyaml tqdm pytest psutil
   ```
4. 安装后立即验证（阶段四）：
   - `torch.__version__`、`torch.version.cuda`
   - `torch.cuda.is_available()`、`torch.cuda.get_device_name(0)`
   - `ultralytics` 导入与版本
   - 用 `yolov8n.pt` 对一张图片做真实推理，确认输出框与耗时

**为什么不用方案 B（全局安装到 D:\python）**：会污染系统 Python、与现有 numpy 2.4.6 冲突，且本次会话沙箱不允许写入该目录。**方案 C（Conda）**需要额外下载安装 Miniconda，本机无此必要。

## 十、下一步需要您完成/确认的事项

1. **确认项目根目录**：当前目录 `E:\DeepSeek Harness\森林防火yolo` 为空，建议**直接作为项目根目录**（不再嵌套 `forest_smoke_project/` 子目录，避免路径过长、层级混乱）。若您希望改成纯英文路径（如 `E:\fs_smoke`）以规避 P5，请现在告知。
2. **确认安装方案 A**（项目内 venv + cu126 轮子）。
3. 确认数据集来源方向（阶段三再细化）：您是否有已下载的森林烟雾数据集？若无，我会先检索公开数据集并核实其标注格式与许可，**不会伪造数据**。
4. 训练前请关闭占用内存较多的程序（QQ、浏览器等）。

确认后我再进入阶段二（建立项目目录与基础文件），不会提前安装依赖或下载数据集。

---

## 附录：本报告使用的关键检查命令

```powershell
# 系统与硬件（WMI 被沙箱拒绝，改用注册表/API）
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' | Select ProductName,DisplayVersion,CurrentBuild,UBR
Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' | Select ProcessorNameString
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000' | Select DriverDesc,DriverVersion
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
```

```bash
python --version ; py -0p ; pip --version ; conda --version
ls /d/python/Lib/site-packages
df -h
python -m pip download --no-deps -d "$TEMP/dsh_piptest" six   # 真实下载验证
```

```python
# 内存信息（回避被拒绝的 WMI）
ctypes.windll.kernel32.GlobalMemoryStatusEx(byref(m))
```
