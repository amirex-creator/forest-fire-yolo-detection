# 数据集说明（forest_smoke）—— 双类别 smoke / fire

> **状态：数据已下载、已划分、已校验。全部数字来自 `src/dataset_utils.py --check`
> 对磁盘文件的实际扫描（2026-10-27 02:13，扫描耗时 306.4 秒），不是估算，也不是模板。**
>
> 完整统计见 `outputs/reports/dataset_summary.json` 与 `outputs/reports/dataset_summary.md`；
> 划分清单见 `outputs/reports/split_manifest.json`。

---

## 一、数据集身份（可溯源）

| 字段 | 值 |
| --- | --- |
| 名称 | D-Fire（Kaggle 镜像 `sayedgamal99/smoke-fire-detection-yolo`） |
| 来源 URL | <https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo> |
| 许可（平台字段） | **CC0: Public Domain** |
| 许可备注 | 原始 D-Fire 中文件名以 `PublicDataset` 开头的图片来自 HPWREN / FESB / FireNet，原始条款未逐一核实；仅作课程研究使用 |
| 版本 | v1（Kaggle `lastUpdated` 2025-01-27） |
| 下载时间 | 2026-10-27 01:51:04 (+08:00) |
| 压缩包 | `datasets/_downloads/smoke-fire-detection-yolo.zip`，**3,049,605,157 字节** |
| 压缩包 SHA256 | `65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83` |
| 解压结果 | `datasets/_downloads/extracted/`，**43,055 个文件 / 3,118,334,483 字节** |
| 原始 YOLO 标注格式 | 每行 `class_id cx cy w h`，坐标已归一化 |
| 类别 | **0 = smoke（烟雾）**，**1 = fire（火焰）**（已抽查确认，不是猜测） |
| 场景构成 | 野外/林区监控画面 + 网络收集图像，**含城市、工业等非森林场景，不是纯森林数据集** |

## 二、目录约定（本项目的整理结果）

```
datasets/forest_smoke/
├── images/{train,val,test}/   JPEG 图片
├── labels/{train,val,test}/   与图片同名的 .txt
└── dataset_info.md            本文件
```

- 图片与标签**只复制、不移动、不改写**；原始解压目录保持原样（`datasets/_downloads/extracted/`）。
- 标签内容与原始完全一致（双类别，未合并、未丢弃任何框）。
- `.gitignore` 已排除 `datasets/forest_smoke/images/**`、`labels/**`、`datasets/_downloads/**`。

## 三、真实统计（扫描结果）

| 集合 | 图片数 | 有标签 | 缺标签 | 空标签(负样本) | 烟雾框 | 火焰框 | 框合计 | 组数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15,069 | 15,069 | 0 | 6,491 | 8,569 | 10,709 | 19,278 | 5,942 |
| val | 3,229 | 3,229 | 0 | 1,702 | 1,591 | 1,789 | 3,380 | 1,275 |
| test | 3,229 | 3,229 | 0 | 1,645 | 1,686 | 2,187 | 3,873 | 1,276 |
| **合计** | **21,527** | — | — | **9,838** | **11,846** | **14,685** | **26,531** | — |

类别占比：smoke 44.65%，fire 55.35%（按有效框计）。

> 说明：26,531 是**通过校验的框**数量。源数据另有 26 个缺陷框（见第六节），
> 因此原始标签里的框总数为 26,557（= 11,846+14,685+26）。

## 四、划分规则（已执行，可复现）

| 项目 | 值 |
| --- | --- |
| 比例 | 训练 70.00% / 验证 15.00% / 测试 15.00%（实测 15,069 / 3,229 / 3,229） |
| 随机种子 | **42** |
| 划分单位 | **group（组）**，一组只进一个集合 |
| 分组方式 | 感知哈希近似重复聚类（dHash ≤ 8，union-find），共 **8,493 组**；其中 4 个超大簇按文件名顺序切成 ≤200 张的块 |
| 组大小 | 最大 200，中位数 1，单图组 7,910 个 |
| 泄漏自检 | 同组跨集合 **0** 个 |
| 复现命令 | `python src/prepare_dataset.py --raw datasets/_downloads/extracted/data --group-by phash --max-group-size 200` |

**必须如实说明的三点限制**：

1. **原始视频/事件 ID 已丢失**：这个 Kaggle 镜像把文件名重编成了
   `AoF00000…`、`WEB00000…`、`PublicDataset00000…` 三套连续编号，前缀只有 3 类，
   无法据此还原"哪几张来自同一个视频"。
2. 因此分组只能用**感知哈希近似聚类**作为"同事件"的代理 —— 这是**近似分组**，
   不是严格的视频级分组。
3. 实测仍有 **19 组跨集合近似重复**（占 21,527 张的约 0.09%），
   这是近似分组无法完全消除的残留泄漏，**测试指标可能因此略微偏乐观**。

## 五、标签质量检查结果（真实）

| 检查项 | 结果 |
| --- | --- |
| 图片无法打开 | 0 |
| 缺少标签文件 | 0 |
| 字段数 ≠ 5 | 0 |
| 类别编号非法 | 0 |
| 中心坐标越界 | 0 |
| 宽或高 ≤ 0（退化框） | 18 |
| 宽或高 > 1（几何越界） | 8 |
| 精确重复图片组 | 0 |
| 感知哈希近似重复组 | 388 |
| 跨集合精确重复组 | 0 |
| 跨集合近似重复组 | 19 |
| 跨集合同组泄漏 | 0 |

## 六、已知缺陷与处理方式

- **退化框 18 个**（`w=0.0` 或 `h=0.0`）：无法作为真值，已逐条记录文件名与行号
  （`outputs/reports/dataset_summary.md` 第六节）。
- **几何越界框 8 个**（`w` 或 `h` 略大于 1，最大 1.056）：Ultralytics 会按图像边界裁剪或忽略。

**处理原则：不修改原始标签文件。** 本项目把这 26 个框记为**警告**而非阻塞错误，
训练流程会在阶段六的标签加载处跳过非法框（Ultralytics 自身也会告警），
数据集整体仍可用于训练。若你希望先做一份"清洗副本"，可在阶段五用
`src/dataset_utils.py` 的报告生成 `datasets/forest_smoke_clean`，**不动当前目录**。

## 七、复现与核对命令

```powershell
# 1) 重新划分（会重新计算感知哈希，约 5 分钟；不改动原始解压目录）
python src/prepare_dataset.py --raw datasets/_downloads/extracted/data --group-by phash --max-group-size 200

# 2) 重新校验并生成报告（约 5 分钟；本机有哈希缓存时更快）
python src/dataset_utils.py --check --report outputs/reports/dataset_summary.json --markdown outputs/reports/dataset_summary.md

# 3) 核对压缩包完整性（PowerShell）
Get-FileHash datasets/_downloads/smoke-fire-detection-yolo.zip -Algorithm SHA256
```

## 八、可视化检查产物（阶段五）

数据检查工具除了写 JSON/Markdown 报告，还会用 matplotlib 生成两类图，输出到
`outputs/figures/dataset_inspect/`：

| 文件 | 内容 | 本机实际大小 |
| --- | --- | --- |
| `class_distribution.png` | 左：各集合 smoke/fire 框数分组柱状图；右：全数据集合计 + 负样本数 | 74,328 B |
| `samples_train.png` | train 抽样 10 张（8 张有框 + 2 张负样本），画出真实标注框 | 1,736,799 B |
| `samples_val.png` | val 抽样 10 张 | 1,264,904 B |
| `samples_test.png` | test 抽样 10 张 | 1,591,581 B |

生成命令（两个都可用）：

```powershell
python src/dataset_utils.py --check      # 完整重扫 + 重画
python src/dataset_utils.py --plots-only # 只重画（读已有 JSON 报告，几秒）
```

抽样规则：每个集合随机取 8 张"至少有一个合法框"的图片 + 2 张空标签负样本，
随机种子 `seed=42`（与数据划分同种子，可复现）。

**抽样时观察到的一个现象（如实记录，不作结论）**：
val 集合的 `WEB00349.jpg` 是空标签文件（0 字节，按负样本处理），但画面中有淡色烟羽状物体。
仅凭肉眼无法判断这是**源数据集漏标**，还是**云雾/霾**——两者在单帧图像上都可能呈现类似形态。
这类样本留待阶段八（错误案例分析）复查，**现在不修改任何标签**。

> 绘图过程只读取图片与标签，`images/` 与 `labels/` 下的文件未被改动（文件时间戳仍为划分时间）。
