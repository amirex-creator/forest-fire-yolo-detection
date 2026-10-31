# 阶段四环境验证记录

> 本文件所有数字与输出均来自本机**实际执行**，不是预期值。
> 生成时间：2026-10-27 23:51（V1–V4 首轮）、2026-10-28 00:05（复跑 + 推理）
> 结论先行：**V1–V4 全部 PASS，CUDA 可用（依据 V2 + V3 的 GPU 矩阵乘法）。**

---

## 一、结论摘要

| 编号 | 检查项 | 实测结果 | 判定 |
| --- | --- | --- | --- |
| V1 | 版本对齐 | `torch 2.13.0+cu126`、`torchvision 0.28.0+cu126`、`cuda_build 12.6`、`cudnn 91002` | ✅ PASS |
| V2 | CUDA 可见性 | `cuda_available True`、`device_count 1`、`NVIDIA GeForce RTX 3060 Laptop GPU`、`capability (8, 6)` | ✅ PASS |
| V3 | 真实 GPU 矩阵运算 | 1024×1024 `cuda` matmul 求和 = `-29385.65234375`（首轮 `-5245.2216796875`），有限值 | ✅ PASS |
| V4 | 显存与 Ultralytics | `vram_total_GB 6.0`、`ultralytics 8.4.190`、`opencv 4.14.0` | ✅ PASS |

**CUDA 结论：可用。** 依据不是"装上了 CUDA 版 torch"，而是 V3 的矩阵乘法**真的在 GPU 上算出了有限结果**，
并配合 V2 的设备枚举。按 `docs/compatibility_final.md` 第七节的规则，V3 通过前不允许宣称 CUDA 可用——现在已通过。

## 二、环境信息

| 项目 | 值 |
| --- | --- |
| 项目根 | `E:\DeepSeek Harness\fs_smoke` |
| 虚拟环境 | `E:\DeepSeek Harness\fs_smoke\.venv`（**独立 venv，未改动系统 Python**） |
| venv 解释器 | `.venv\Scripts\python.exe` |
| Python | 3.11.9 (MSC v.1938 64-bit) |
| 创建时间 | 2026-10-27 23:01:39（`venv python.exe` 落盘时间） |
| venv 体积 | **4.75 GiB / 34,602 个文件** |
| 平台 | Windows-10-10.0.26200-SP0 |
| GPU / 驱动 | RTX 3060 Laptop 6,144 MiB / 566.07 |
| 系统 Python（未改动） | `D:\python\python.exe`，无任何新包 |

## 三、安装过程（严格两步，未使用清华源）

### 第 1 步：PyTorch（官方 cu126 索引）

```powershell
.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 `
    --index-url https://download.pytorch.org/whl/cu126
```

实际输出（节选，完整见 `outputs/reports/stage4_install_log.txt`）：

```text
Downloading torch-2.13.0%2Bcu126-cp311-cp311-win_amd64.whl (2594.5 MB)
Successfully installed MarkupSafe-3.0.3 filelock-3.32.3 fsspec-2026.7.0 jinja2-3.1.6
mpmath-1.3.0 networkx-3.6.1 numpy-2.4.6 pillow-12.3.0 sympy-1.14.0 torch-2.13.0+cu126
torchvision-0.28.0+cu126 typing-extensions-4.16.0
```

- 下载的 torch 轮子 **2,594.5 MB**（与兼容性报告核实的 2,594,548,547 字节一致）；
- 从 23:01 开始到这个阶段结束，安装总耗时约 **41 分钟**（大部分时间在下载 2.59 GB）。

### 第 2 步：其余依赖（阿里云镜像）

```powershell
.venv\Scripts\python.exe -m pip install ultralytics==8.4.190 "opencv-python==4.14.0.94" `
    pandas matplotlib streamlit pyyaml tqdm psutil pytest `
    -i https://mirrors.aliyun.com/pypi/simple/
```

实际输出（节选）：

```text
Successfully installed altair-6.3.0 ... matplotlib-3.11.2 ... numpy-2.4.6
opencv-python-4.14.0.94 pandas-3.1.0 ... polars-2.1.0 ... pytest-9.1.1 ...
streamlit-1.66.1 ... ultralytics-8.4.190 ultralytics-platform-0.1.99
ultralytics-thop-2.2.3 ...
```

两步退出码均为 **0**，没有回退到 PyPI 官方源。

## 四、V1–V4 原始输出（原样照抄）

### V1 版本对齐

```text
python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
executable E:\DeepSeek Harness\fs_smoke\.venv\Scripts\python.exe
torch 2.13.0+cu126
torchvision 0.28.0+cu126
cuda_build 12.6
cudnn 91002
```

### V2 CUDA 可见性

```text
cuda_available True
device_count 1
name NVIDIA GeForce RTX 3060 Laptop GPU
capability (8, 6)
```

### V3 真实 GPU 矩阵运算

```text
matmul_ok -29385.65234375        # 复跑值；首轮为 -5245.2216796875，随机矩阵不同故不同
```

### V4 显存与 Ultralytics

```text
vram_total_GB 6.0
device_name NVIDIA GeForce RTX 3060 Laptop GPU
torch_cudnn_enabled True
ultralytics 8.4.190
opencv 4.14.0
```

> 首次运行时 Ultralytics 自动创建了配置文件（提示 `Creating new Ultralytics Settings v0.0.8 file`），
> 属于正常行为，未修改项目文件。

## 五、预训练权重与单张图片推理

| 项目 | 实测值 |
| --- | --- |
| 权重文件 | `models\yolov8n.pt` |
| 字节数 | **6,549,796** |
| SHA256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| 文件头 | `b'PK\x03\x04'`（.pt 是 zip 容器） |
| 类别数 | 80（COCO） |
| 模型加载耗时 | 0.045 秒 |
| 输入图片 | `datasets\forest_smoke\images\test\AoF04049.jpg`（63,929 字节） |
| 参数 | imgsz 640 / conf 0.25 / iou 0.45 / device 0 |
| 推理总耗时 | 578.2 ms（含首次 CUDA 预热） |
| 预处理 / 推理 / 后处理 | 48.1 / 6.4 / 1.0 ms |
| 检测框数量 | **0** |
| 结果图 | `outputs\predictions\stage4_pretrained\AoF04049__pred.jpg` |
| 结构化结果 | `outputs\reports\stage4_inference_result.json` |

> ⚠️ **必须说清楚**：`yolov8n.pt` 是 COCO 80 类权重，**没有 smoke / fire 类别**。
> 它对这张烟雾图给出 0 个框是**预期结果**，不能解读为"这张图没有烟"，
> 更不能当作森林火灾预警。这一步只证明：权重能加载、图片能进网络、GPU 能推理、结果能落盘。

## 六、测试套件

```text
$ .venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
....................                                                     [100%]
20 passed in 224.13s (0:03:44)
```

- `tests/test_dataset.py` 12 项 + `tests/test_inference.py` 8 项 = **20 项全部通过**；
- 阶段七的推理模块仍是骨架，测试里"未实现必须抛 NotImplementedError"的断言依然成立，
  所以这 20 项通过**不代表推理功能已完成**。

## 七、磁盘占用变化

| 项目 | 变化 |
| --- | --- |
| E: 可用空间 | 529.9 GiB → **516.2 GiB**（减少约 13.7 GiB） |
| 其中 venv | 4.75 GiB |
| 其中 pip 临时下载缓存 | 安装后已由 pip 自动清理（`%TEMP%\pip-unpack-*`） |
| 其中数据集 | 3.05 GiB 压缩包 + 3.12 GiB 解压 + 约 3.1 GiB 项目副本（阶段三已完成） |

## 八、未验证 / 局限（如实列出）

1. **未跑任何训练**，因此"训练能不能跑通、显存够不够、AMP 是否稳定"**全部未验证**；
2. V3 只证明单次矩阵乘法可用，**不等于**长时间训练的稳定性（6 GB 显存仍是主要约束）；
3. Windows + cu126 组合**不在 Ultralytics 官方 CI 覆盖范围内**（见 `docs/compatibility_final.md`），
   阶段六训练时需重点观察是否出现 Windows 特有报错；
4. `src/inference.py`、`src/evaluation.py` 仍是骨架，**推理/评价功能未实现**；
5. Streamlit 已装但**未启动验证**（计划阶段九）。

## 九、复现方式

```powershell
cd "E:\DeepSeek Harness\fs_smoke"
.venv\Scripts\python.exe src\env_check.py
```

该脚本会重新生成 `outputs\reports\stage4_env_verification.json`、
`stage4_verification.md`、`stage4_inference_result.json`。
