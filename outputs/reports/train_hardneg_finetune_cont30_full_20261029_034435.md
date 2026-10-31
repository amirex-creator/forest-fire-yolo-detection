# 训练运行报告：hardneg_finetune_cont30_full_20261029_034435

- 模式：**full**（正式训练）
- 状态：**成功**
- 开始：2026-10-29T03:44:35+08:00
- 结束：2026-10-29T04:21:39+08:00
- 总耗时：**2223.8 秒**（约 37.1 分钟）
- 设备：`0`（NVIDIA GeForce RTX 3060 Laptop GPU）；选择原因：GPU 矩阵乘法通过
- 实际批大小：**8**
- 批次轮数：本次 10 轮；此前累计 30 轮；累计上限 40 轮
- 起始权重：`models\hardneg_finetune_cont20_full_20261029_024922_best.pt`

## 一、运行参数（最终生效值）

| 参数 | 值 |
| --- | --- |
| model | models\hardneg_finetune_cont20_full_20261029_024922_best.pt |
| epochs | 10 |
| imgsz | 640 |
| batch | 8 |
| workers | 1 |
| patience | 10 |
| seed | 0 |
| deterministic | True |
| amp | True |
| cache | False |
| fraction | 1.0 |
| mode | full |
| device | 0 |

## 二、数据

| 集合 | 图片数 |
| --- | --- |
| train | 15217 |
| val | 3229 |
| test | 3229 |

类别：`['smoke', 'fire']`；阻塞性问题 **0**，警告 **0**。

## 三、硬件占用（训练期间采样）

| 指标 | 值 |
| --- | --- |
| 采样次数（间隔 2.0 s） | 1082 |
| 进程内存峰值 | 2683.5 MB |
| 系统可用内存最低点 | 2617.6 MB |
| CUDA 已分配显存峰值 | 940.5 MB |
| CUDA 保留显存峰值 | 1484.0 MB |

## 四、逐轮指标（真实，来自 results.csv）

| epoch | box_loss | cls_loss | dfl_loss | val_box_loss | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.3153 | 1.0588 | 1.2698 | 1.1828 | 0.6500 | 0.5989 | 0.6464 | 0.3490 |
| 2 | 1.3609 | 1.1301 | 1.2958 | 1.2133 | 0.6875 | 0.5930 | 0.6492 | 0.3419 |
| 3 | 1.3437 | 1.1067 | 1.2862 | 1.1660 | 0.6758 | 0.6216 | 0.6799 | 0.3721 |
| 4 | 1.3284 | 1.0811 | 1.2760 | 1.1739 | 0.7088 | 0.6154 | 0.6747 | 0.3684 |
| 5 | 1.3168 | 1.0575 | 1.2729 | 1.1461 | 0.7016 | 0.6260 | 0.6838 | 0.3794 |
| 6 | 1.3066 | 1.0423 | 1.2675 | 1.1367 | 0.6999 | 0.6219 | 0.6840 | 0.3796 |
| 7 | 1.2969 | 1.0171 | 1.2602 | 1.1364 | 0.7046 | 0.6376 | 0.6946 | 0.3896 |
| 8 | 1.2853 | 1.0076 | 1.2556 | 1.1136 | 0.7172 | 0.6421 | 0.7062 | 0.4017 |
| 9 | 1.2952 | 0.9342 | 1.2540 | 1.1454 | 0.7196 | 0.6414 | 0.7061 | 0.3942 |
| 10 | 1.2861 | 0.9080 | 1.2552 | 1.1111 | 0.7284 | 0.6446 | 0.7118 | 0.4024 |

## 四之三、分类别指标（val，用 best.pt 重新验证一次）

| 类别 | 实例数 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| smoke | 1591 | 0.7735 | 0.7108 | 0.7671 | 0.4559 |
| fire | 1789 | 0.683 | 0.5819 | 0.6565 | 0.3496 |
| **all** | - | 0.7283 | 0.6464 | 0.7118 | 0.4027 |

- 验证速度：预处理 0.19 ms / 推理 3.197 ms / 后处理 1.11 ms（每张）

## 四之四、过拟合判断（只看可观测趋势）

- 判断：**暂无明显过拟合迹象**
- 依据：训练与验证曲线方向一致。
- 最近 3 轮 train/box_loss：[1.28529, 1.29523, 1.28615]
- 最近 3 轮 val/box_loss：[1.11361, 1.14535, 1.11111]
- 最近 3 轮 val mAP50：[0.70618, 0.70611, 0.71178]
- val−train 的 box_loss 差距：[-0.1717, -0.1499, -0.175]

> 3 轮窗口只是启发式判断，不等于统计意义上的过拟合结论。

## 五、产物

- 运行目录：`outputs/train_runs/hardneg_finetune_cont30_full_20261029_034435`
- 权重 best.pt：`models/hardneg_finetune_cont30_full_20261029_034435_best.pt`（6,480,170 B）
- 权重 last.pt：`models/hardneg_finetune_cont30_full_20261029_034435_last.pt`（6,480,170 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/results.png`（294,835 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/results.csv`（1,427 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/confusion_matrix.png`（104,757 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/confusion_matrix_normalized.png`（102,530 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/labels.jpg`（127,402 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/train_batch0.jpg`（444,861 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/val_batch0_labels.jpg`（295,935 B）
- 图：`outputs/figures/hardneg_finetune_cont30_full_20261029_034435/val_batch0_pred.jpg`（298,932 B）

## 五之二、训练期间数据集文件的变化

- 扫描图片 21527 张，训练期间被修改 **0** 张

## 五之三、训练前图片哈希快照与对比

- 快照文件：`outputs/train_runs/hardneg_finetune_cont30_full_20261029_034435/image_snapshot.json`
- 快照图片数：21527 张 / 3,125,783,486 字节（计算耗时 13.7 s）
- 训练后对比：内容被改变的图片 **0** 张；缺失 0 张；新增 0 张

## 六、说明与限制

- 模型的识别结果仅供辅助分析，需人工核实；本系统不做森林火灾预警，也不能仅凭检测结果确认真实火灾。
- 数据划分是近似分组（原始视频 ID 已被抹除），跨集合仍存在少量近似重复（阶段五统计 19 组）。

