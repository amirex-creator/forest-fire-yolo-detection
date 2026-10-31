# 数据集统计报告（真实扫描结果）

> 本文件由 `src/dataset_utils.py --check` 扫描 `datasets/forest_smoke` 实际文件生成，
> 表中每个数字都可在磁盘上复核。**不是模板，也不是估算。**

- 生成时间：2026-10-28T00:36:53+08:00
- 数据根目录：`E:\DeepSeek Harness\fs_smoke\datasets\forest_smoke`
- 扫描耗时：52.6 秒
- 数据集名称：D-Fire（Kaggle 镜像 sayedgamal99/smoke-fire-detection-yolo）
- 来源 URL：https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
- 许可：CC0: Public Domain
- 许可备注：PublicDataset 前缀图片来自 HPWREN/FESB/FireNet，原始条款未逐一核实
- 版本 / 提交：v1 (Kaggle lastUpdated 2025-01-27)
- 下载日期：2026-10-27T02:02:45+08:00
- 原始包字节数：3049605157
- 原始包 SHA256：`65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83`
- 场景构成说明：D-Fire 为野外/林区监控与公开火灾图像合集，包含城市、工业等非森林场景；不是纯森林数据集。

## 一、数据划分配置

| 项目 | 值 |
| --- | --- |
| 训练 / 验证 / 测试比例 | 0.7 / 0.15 / 0.15 |
| 随机种子 | 42 |
| 划分单位 | group |
| 分组键来源 | 感知哈希近似重复聚类（dHash≤8，union-find），其中 4 个超大簇按文件名顺序切成 ≤200 张的块 |
| 分组是否近似 | True |
| 泄漏风险说明 | 分组基于感知哈希近似聚类，属于近似分组，指标可能偏乐观 |

## 二、各集合数量统计

| 集合 | 图片数 | 有标签 | 缺标签 | 空标签(负样本) | 烟雾框 | 火焰框 | 框合计 | 组数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15069 | 15069 | 0 | 6491 | 8569 | 10709 | 19278 | 5942 |
| val | 3229 | 3229 | 0 | 1702 | 1591 | 1789 | 3380 | 1275 |
| test | 3229 | 3229 | 0 | 1645 | 1686 | 2187 | 3873 | 1276 |
| **合计** | **21527** | — | — | **9838** | **11846** | **14685** | **26531** | — |

## 三、类别分布

| 类别编号 | 类别名 | 框数 | 占全部框比例 |
| --- | --- | --- | --- |
| 0 | smoke | 11846 | 44.65% |
| 1 | fire | 14685 | 55.35% |

实测出现的类别编号：[0, 1]

## 四、质量检查结果

| 检查项 | 结果 |
| --- | --- |
| 扫描图片文件总数 | 21527 |
| 图片无法打开 | 0 |
| 标签行格式错误（合计） | 26 |
| 字段数 ≠ 5 | 0 |
| 类别编号非法 | 0 |
| 坐标越界 | 0 |
| 退化框（宽或高 ≤ 0） | 18 |
| 几何越界框（宽或高 > 1） | 8 |
| 宽或高非法（合计） | 26 |
| 缺少标签文件 | 0 |
| 精确重复图片组 | 0 |
| 感知哈希近似重复组 | 388 |
| 跨集合精确重复组 | 0 |
| 跨集合近似重复组 | 19 |
| 跨集合同组泄漏 | 0 |

## 五、目标框相对面积分布

| 相对面积（w×h） | 框数 |
| --- | --- |
| < 0.1% | 2595 |
| 0.1%–1% | 9243 |
| 1%–10% | 7225 |
| 10%–50% | 5529 |
| > 50% | 1939 |

> 相对面积 = w*h（标签已归一化），仅用于观察小目标占比。

## 六、异常清单

共 26 条（下表最多展示 60 条，完整清单见 JSON 报告）。

| # | 集合 | 文件 | 标签行 | 问题码 | 说明 | 级别 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | train | train/AoF05470.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 2 | train | train/AoF06348.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 3 | train | train/AoF06439.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 4 | train | train/AoF06456.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 5 | train | train/AoF08348.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 6 | train | train/WEB04243.jpg | 5 | zero_size | 第 5 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 7 | train | train/WEB04742.jpg | 5 | zero_size | 第 5 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 8 | train | train/WEB07998.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 9 | train | train/WEB08540.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 10 | train | train/WEB10669.jpg | 5 | zero_size | 第 5 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 11 | train | train/WEB10769.jpg | 2 | wh_gt_one | 第 2 行 w=1.0296875 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 12 | train | train/WEB11090.jpg | 1 | wh_gt_one | 第 1 行 h=1.0027777777777778 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 13 | train | train/WEB11598.jpg | 1 | wh_gt_one | 第 1 行 w=1.0359375 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 14 | train | train/WEB11600.jpg | 1 | wh_gt_one | 第 1 行 w=1.0562500000000001 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 15 | val | val/AoF07743.jpg | 3 | zero_size | 第 3 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 16 | val | val/AoF07774.jpg | 3 | zero_size | 第 3 行 h=0.0 不满足 0 < h（退化框，无法作为真值） | warning |
| 17 | val | val/PublicDataset00056.jpg | 5 | zero_size | 第 5 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 18 | val | val/WEB05882.jpg | 7 | zero_size | 第 7 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 19 | val | val/WEB10770.jpg | 2 | wh_gt_one | 第 2 行 w=1.0078125 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 20 | val | val/WEB10775.jpg | 3 | wh_gt_one | 第 3 行 w=1.015625 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 21 | val | val/WEB10821.jpg | 3 | wh_gt_one | 第 3 行 w=1.0093750000000001 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |
| 22 | test | test/AoF06155.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 23 | test | test/WEB04527.jpg | 2 | zero_size | 第 2 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 24 | test | test/WEB04849.jpg | 5 | zero_size | 第 5 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 25 | test | test/WEB06900.jpg | 8 | zero_size | 第 8 行 w=0.0 不满足 0 < w（退化框，无法作为真值） | warning |
| 26 | test | test/WEB11606.jpg | 1 | wh_gt_one | 第 1 行 h=1.0027777777777778 超过 1（源数据几何越界，Ultralytics 会按图像边界裁剪或忽略该框） | warning |

## 七、可用性结论

| 项目 | 结论 |
| --- | --- |
| 是否可用于训练 | **是**（阻塞性问题 0 项，警告 26 项） |
| 图片总数 | 21527 |
| 目标框总数 | 26531 |
| 说明 | 由 src/prepare_dataset.py 整理，src/dataset_utils.py 校验。 |

## 八、复现方式

```powershell
.\\.venv\\Scripts\\python.exe src\\dataset_utils.py --check
```

## 九、可视化产物

由 `src/dataset_utils.py` 用 matplotlib 生成，生成时间 2026-10-28 00:36:59。
抽样规则：每个集合取 8 张有框图片 + 2 张无目标负样本，随机种子 seed=42（可复现）。

| 文件 | 内容 |
| --- | --- |
| `outputs/figures/dataset_inspect/class_distribution.png` | 类别标签数量分布柱状图（左：按集合分组的 smoke/fire 框数；右：全数据集合计与负样本数） |
| `outputs/figures/dataset_inspect/samples_train.png` | train 集合抽样样本（画出真实标注框，含负样本） |
| `outputs/figures/dataset_inspect/samples_val.png` | val 集合抽样样本（画出真实标注框，含负样本） |
| `outputs/figures/dataset_inspect/samples_test.png` | test 集合抽样样本（画出真实标注框，含负样本） |

> 绘图过程只**读取**图片与标签，未修改任何数据集文件。
