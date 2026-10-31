# 阶段五工作报告：数据检查工具与可视化

> 本文档记录**实际执行**的结果。所有数字与文件大小都来自 2026-10-28 的真实运行，
> 未完成的项在第六节明确列出。

## 一、本阶段完成的工作

`src/dataset_utils.py` 从"只做校验和统计"扩展为"校验 + 统计 + **绘图**"三合一工具：

1. 新增**类别标签数量分布柱状图**（`class_distribution.png`）；
2. 新增**带标注框的样本可视化**（train/val/test 各一张，`samples_*.png`）；
3. 图片统一输出到 `outputs/figures/dataset_inspect/`；
4. 命令行入口补全：新增 `--plots-only`、`--no-plots`、`--figures-dir`、`--samples`、
   `--samples-negative`、`--seed`，并把完整调用示例写进模块 docstring 与 `--help` 的 epilog；
5. 绘图**全部用 matplotlib**（Agg 后端，无显示器也能跑），中文字体自动探测
   （本机命中 `Microsoft YaHei`，故图为中文）；
6. 绘图**只读取**图片与标签，没有修改任何数据集文件。

## 二、工具能力清单

**校验项（全量扫描，不做抽样）**

| 检查 | 说明 |
| --- | --- |
| 目录结构 | `images/{train,val,test}` 与 `labels/{train,val,test}` 是否存在 |
| 图片可读性 | 逐张打开，坏图单独记录 |
| 标签配对 | 每张图是否有同名 `.txt` |
| 标签格式 | 5 字段、类别编号合法、坐标在 `[0,1]`、`w/h > 0`、无 NaN/Inf |
| 缺陷分级 | `error` 阻塞训练 / `warning` 仅提示（退化框、越界框） |
| 重复检测 | 精确 SHA256 + 感知哈希（dHash，汉明距离 ≤ 8）双路去重 |
| 泄漏检测 | 跨集合重复、同组跨集合泄漏 |

**统计项**：每集合图片数、框数、**按类别的框数（smoke / fire 分列）**、空标签数、
目标框相对面积分布、组数、近似重复组数。

**绘图项（本阶段新增）**

| 图 | 内容 |
| --- | --- |
| `class_distribution.png` | 左：各集合 smoke/fire 框数分组柱状图；右：全数据集合计 + 负样本数 |
| `samples_train.png` / `samples_val.png` / `samples_test.png` | 每集合 8 张有框图 + 2 张无目标负样本，画出**真实**标注框（红=smoke，橙=fire） |

## 三、命令行用法

```powershell
# 完整检查：扫描 + 校验 + 统计 + 画图 + 写 JSON/Markdown 报告
.\.venv\Scripts\python.exe src\dataset_utils.py --check

# 只重画图（读已有 JSON 报告，不重新扫描，几秒钟）
.\.venv\Scripts\python.exe src\dataset_utils.py --plots-only

# 改抽样数量与种子（每个集合 12 张有框 + 3 张负样本，seed=7）
.\.venv\Scripts\python.exe src\dataset_utils.py --plots-only --samples 12 --samples-negative 3 --seed 7

# 图表输出到别的目录
.\.venv\Scripts\python.exe src\dataset_utils.py --plots-only --figures-dir outputs/figures/my_check

# 只检查不画图（更快）
.\.venv\Scripts\python.exe src\dataset_utils.py --check --no-plots
```

退出码：`0` = 无阻塞性问题；`1` = 路径/参数错误；`3` = 存在阻塞性标签错误。

## 四、本次真实运行记录

| 项目 | 实际结果 |
| --- | --- |
| 完整检查命令 | `python src/dataset_utils.py --check` |
| 退出码 | **0** |
| 扫描耗时 | 首次 **173.11 秒**（无哈希缓存）；最终复跑 **52.58 秒**（命中 dHash 缓存，数字均取自 JSON 的 `scan_seconds`） |
| 报告生成时间 | 最终复跑 2026-10-28T00:36:53+08:00 |
| 结论 | `usable_for_training = true`，阻塞性问题 **0**，警告 **26**（`zero_size` 18 + `wh_gt_one` 8） |
| 图片 / 框 | **21,527 张 / 26,531 个**（smoke 11,846 / fire 14,685） |
| 测试 | `pytest tests/test_dataset.py -q` → **16 passed in 15.56s**（最终复跑；首次 82.18 s） |

**生成的图（真实文件大小）**

| 文件 | 大小 |
| --- | --- |
| `outputs/figures/dataset_inspect/class_distribution.png` | 74,328 B |
| `outputs/figures/dataset_inspect/samples_train.png` | 1,736,799 B |
| `outputs/figures/dataset_inspect/samples_val.png` | 1,264,904 B |
| `outputs/figures/dataset_inspect/samples_test.png` | 1,591,581 B |

**数据集未被改动**：`images/**` 与 `labels/**` 的文件修改时间仍为划分时的
2026-10-27 01:51，绘图与检查都只读取。

## 五、看图时发现的一个现象（如实记录，未下结论）

val 集合的 `WEB00349.jpg` 是**空标签**（`labels/val/WEB00349.txt` = 0 字节，按负样本处理），
但画面里有淡色烟羽状物体。仅凭单帧图像无法判断这是：

- **源数据集漏标**（该有 smoke 框但没有），还是
- **云雾/霾**（形态相似但没有真实烟源）。

**处理方式**：不修改任何标签；把这类样本留给阶段八的错误案例分析复查。
这条记录也写进了 `datasets/forest_smoke/dataset_info.md` 第八节，避免论文里
把"负样本"当成"确定无烟"来论证。

## 六、未完成 / 局限（不得当成已完成）

1. 样本图是**抽样**（每集合 10 张），只能用于人工抽查，不能当作全量质量证据；
2. 没有做**交互式**可视化（如逐张浏览、按类别筛选），本项目只产出静态 PNG；
3. 没有对"近似重复组 388 组"逐组人工复核，只在报告里给出统计与示例；
4. 图表**未用于训练**，只是检查材料；训练相关图表属阶段六；
5. 阶段五的绘图不涉及模型，因此**不能**据此得出任何检测精度结论。

## 七、复现方式

```powershell
# 1) 只重画图（几秒，最快验证）
.\.venv\Scripts\python.exe src\dataset_utils.py --plots-only

# 2) 完整检查（首次约 3 分钟；有哈希缓存时约 1 分钟）
.\.venv\Scripts\python.exe src\dataset_utils.py --check

# 3) 跑测试（16 项，约 1.5 分钟）
.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -q
```
