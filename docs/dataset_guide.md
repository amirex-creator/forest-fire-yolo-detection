# 数据集准备指南（阶段三，已完成实测）

> **本文件的每个数字都来自本机真实执行，不是推测。**
> 数据集已下载、已按"组"划分、已用 `src/dataset_utils.py` 全量校验。
> 统计文件：`outputs/reports/dataset_summary.json` / `dataset_summary.md`；
> 划分清单：`outputs/reports/split_manifest.json`。
>
> 结果摘要：**21,527 张图片**、**26,531 个有效框**（smoke 11,846 / fire 14,685）、
> **9,838 张空标签负样本**、划分 **15,069 / 3,229 / 3,229**（70.00% / 15.00% / 15.00%，seed=42）。

---

## 一、先分清"检测数据集"和"分类数据集"

| 判断依据 | 检测数据集（可用于 YOLO） | 分类数据集（不能直接用于 YOLO） |
| --- | --- | --- |
| 标注文件 | 每张图有 `labels/*.txt` 或 COCO/VOC 的 **bbox** | 只有图片所在文件夹名，如 `smoke/`、`nofire/` |
| 标注内容 | 矩形框坐标 | 整图标签（有烟 / 无烟） |
| 能否训练检测模型 | 能 | 不能。强行当检测用就是造假 |

本项目选用的 D-Fire 属于**带框检测数据集**：`data/{train,val,test}/{images,labels}`，
每张图有同名 `.txt`，5 字段归一化坐标 —— 已实地抽查确认。

## 二、候选数据源核实结果（2026-10-27 实测）

| 候选 | 类型 / 格式 | 许可 | 关键事实 | 采纳 |
| --- | --- | --- | --- | --- |
| **D-Fire 的 Kaggle 镜像** `sayedgamal99/smoke-fire-detection-yolo` | 检测 / YOLO | **CC0: Public Domain**（平台字段） | 压缩包 **3,049,605,157 B**；`nc=2`；文件名重编为 3 套连续编号 | ✅ **已采用** |
| D-Fire 原版 `github.com/gaiasd/DFireDataset` | 检测 / YOLO | CC0 1.0（`LICENSE` 实测），`PublicDataset` 前缀图仍受原提供方约束 | 21,527 图；smoke 11,865 / fire 14,692 框（与本次实测完全一致） | 同源，未单独下载 |
| HF `keremberke/smoke-object-detection` | 检测 / **COCO** | CC BY 4.0 | 21,578 图，单类 `smoke`；与 D-Fire 规模接近，疑似同源 | 未采用（需转换 + 去重） |
| HF `fireviewer/fire-smoke-detection-corpus-v1` | 检测 / Parquet | `other`（fasdd 源 CC-BY-SA-4.0、pyro-sdis 源 Apache-2.0） | 87,568 行，自带 `split_group`；体积大，建议流式读取 | 未采用（体量/许可传染性） |
| FASDD（国家对地观测数据中心） | 未核实 | **未核实** | 官方站点本机 SSL 不可达（`UNEXPECTED_EOF_WHILE_READING`） | 未采用 |

**关于"森林"的诚实说明**：D-Fire 含 HPWREN 野外摄像头时序帧与网络收集的野外火情图像，
**但不是纯森林数据集**，其中包含城市、工业等场景。论文中必须表述为
"野外/林区场景为主的公开火灾烟雾数据集"，**不得写成"森林专用数据集"**。

## 三、实际执行的流程（可复现）

阶段三在**本机真实执行**了下载 → 解压 → 分组划分 → 全量校验四步。
下载与解压用的是一次性辅助脚本（放在项目外，未纳入版本管理）；
下面给出等价的、可自己重跑的命令。

```powershell
# ① 下载（Kaggle 匿名直链，302 跳转到 storage.googleapis.com；实测无需登录）
#    实测结果：3,049,605,157 B，耗时 311.5 秒，平均 9.34 MiB/s
#    SHA256 = 65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83
#    浏览器方式：打开数据集页面 -> Download，把 zip 放到 datasets\_downloads
# ② 解压（只解压，不改动压缩包）
Expand-Archive -LiteralPath datasets\_downloads\smoke-fire-detection-yolo.zip `
    -DestinationPath datasets\_downloads\extracted
#    实测结果：43,055 个文件 / 3,118,334,483 字节

# ③ 分组 + 按组划分 + 复制（21,527 图 + 21,527 标签，实测 313.9 秒）
python src/prepare_dataset.py --raw datasets/_downloads/extracted/data `
    --group-by phash --max-group-size 200 --seed 42

# ④ 全量校验并生成报告（实测 306.4 秒）
python src/dataset_utils.py --check `
    --report outputs/reports/dataset_summary.json `
    --markdown outputs/reports/dataset_summary.md
```

`prepare_dataset.py` 的行为：**只 `copy` 不 `move`**，原始解压目录一个字节都不改
（解压目录里唯一的额外文件是解压工具写的 `.extracted_ok` 标记，154 字节）；
目标目录已存在同名文件时默认加短哈希前缀而不是覆盖。

## 四、划分规则与"同视频不跨集合"的落实

| 项目 | 规定 / 实际 |
| --- | --- |
| 比例 | 70% / 15% / 15%（实测 70.00% / 15.00% / 15.00%） |
| 随机种子 | **42** |
| 划分单位 | **组**，一组整体只进一个集合 |
| 分组方式 | dHash ≤ 8 的近似重复聚类 + union-find（**8,493 组**） |
| 超大簇处理 | 4 个超大簇按文件名顺序切成 ≤200 张的块，避免一组上 8,000 张导致划分失衡 |
| 泄漏自检 | 脚本内置断言：同组跨集合立即终止且不复制任何文件 → 实测 **0** |
| 复现性 | 种子、分组方式、组↔集合归属全部写入 `outputs/reports/split_manifest.json` |

### 为什么只能用"近似分组"（重要，必须写进论文）

该 Kaggle 镜像把原始文件名重编为 `AoF00000…`/`WEB00000…`/`PublicDataset00000…`
三套连续编号，**原始视频/摄像头 ID 已被抹掉**，前缀只有 3 类，
按前缀分组根本无法做 70/15/15。因此本项目的分组是"画面近似 = 可能同源"的**代理分组**：

- 好处：几乎把所有近似重复帧都收进同一组，跨集合精确重复 **0 组**；
- 局限：仍残留 **19 组跨集合近似重复**（约 0.09%），测试指标可能略偏乐观；
- 规范：论文的实验设置章节必须写明这一点，不能宣称"严格按视频划分"。

## 五、下载后完成的校验（全部为真实结果）

| 检查项 | 结果 |
| --- | --- |
| 图片总数 / 标签总数 | 21,527 / 21,527（一一配对，缺标签 0） |
| 图片无法打开 | 0 |
| 字段数 ≠ 5 / 类别非法 / 中心坐标越界 | 0 / 0 / 0 |
| 退化框（宽或高 ≤ 0） | 18（警告级，已逐条列出） |
| 几何越界框（宽或高 > 1） | 8（警告级，Ultralytics 会裁剪/忽略） |
| 空标签文件（负样本） | 9,838（合法，未补造任何框） |
| 精确重复组 / 跨集合精确重复 | 0 / 0 |
| 感知哈希近似重复组 / 跨集合近似重复 | 388 / 19 |
| 同组跨集合泄漏 | 0 |

## 六、类别与配置的一致性

- 数据集实际类别：`0 = smoke`、`1 = fire`（由 D-Fire README 统计数字与本次全量扫描
  smoke 11,865 / fire 14,692 **完全吻合**反推确认，已抽查标签文件）；
- `configs/forest_smoke.yaml`：`nc: 2`、`names: [smoke, fire]`；
- 两者不一致会在训练时直接报类别数错误，因此每次改动数据都要同步检查。

## 七、缺陷处理原则

1. **不修改原始文件**（压缩包与 `datasets/_downloads/extracted/` 保持原样）；
2. 26 个缺陷框按**警告**记录，不阻塞训练；`dataset_summary.md` 第六节逐条给出文件名与行号；
3. 空标签文件是**合法负样本**，不删除、不补框；
4. 如需"清洗副本"，另建目录（如 `datasets/forest_smoke_clean`）生成，**不动当前数据集**。

## 八、下载包完整性复核

```powershell
Get-FileHash datasets\_downloads\smoke-fire-detection-yolo.zip -Algorithm SHA256
# 期望：65D81B602991261B47DDC71281E99108362A30336A448E1F8D21352C825C6D83
```

若哈希不一致：说明文件在下载或存储中损坏，**必须重新下载**，
不得继续用它划分或训练。

## 九、数据获取阶段的结论

- 数据来源、许可、格式、类别映射均已核实到可复核的程度；
- 划分可复现（种子 42 + 分组方式已固定）；
- 残留风险（近似分组、19 组跨集合近似重复、26 个缺陷框）已量化并记录，
  将作为论文"研究局限性"的实证依据；
- **后续不再需要联网下载数据**；下一步是阶段四（安装依赖并做一次预训练权重推理验证）。

## 十、可视化检查（阶段五，已实现）

工具 `src/dataset_utils.py` 不只输出数字，还会画图，便于人工复核标注质量：

```powershell
# 完整检查 + 画图
python src/dataset_utils.py --check

# 只重画（读已有报告，几秒；改抽样数量或种子时用这个）
python src/dataset_utils.py --plots-only --samples 12 --samples-negative 3 --seed 7
```

| 图 | 看什么 |
| --- | --- |
| `class_distribution.png` | smoke 与 fire 的框数是否悬殊过大；train/val/test 的类别比例是否接近 |
| `samples_train/val/test.png` | 框是否贴合目标；有没有"框住整张图"或"框住无关区域"；负样本是否真的没有烟 |

**看图时必须警惕两件事**：

1. 抽样图好看 ≠ 数据集没问题。样本图只是抽样，真实质量以 `dataset_summary.json` 的
   全量扫描结果为准；
2. 负样本（空标签）里可能出现"看起来像烟"的云雾或霾。这是单帧图像检测的固有歧义，
   不要凭肉眼断言"源数据集标错了"，应在阶段八的错误分析中复查后再下结论。

> 绘图只读数据，不改文件；本项目的原始标签至今未做任何人工修改。
