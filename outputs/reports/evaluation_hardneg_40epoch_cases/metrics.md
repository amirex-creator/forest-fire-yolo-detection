# 测试集评价报告（阶段八）

- 生成时间：2026-10-29T04:29:43+08:00
- 权重：`hardneg_finetune_cont30_full_20261029_034435_best.pt`（6,480,170 B，sha256 `6c959cfd4ef8309d…`）
- 数据：D-Fire（Kaggle 镜像 sayedgamal99/smoke-fire-detection-yolo）；**test 集**
- 测试集规模：**3,229 张图片 / 3,878 个标注框**（空标签图片 1,645 张）
- 各类实例数：smoke 1,688，fire 2,190
- 评价参数：imgsz=640，batch=8，device=0，mAP 用 conf=0.001 / iou=0.7

## 一、总体指标（test 集，一次性最终评价）

| 指标 | 值 |
| --- | --- |
| Precision | **0.7027** |
| Recall | **0.6346** |
| mAP50 | **0.6909** |
| mAP50-95 | **0.3738** |
| F1（由 P/R 计算） | 0.6669 |

## 二、分类别指标

| 类别 | 实例数 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| smoke | 1,688 | 0.7671 | 0.7103 | 0.7651 | 0.4561 |
| fire | 2,190 | 0.6382 | 0.5589 | 0.6166 | 0.2915 |

## 三、推理耗时（验证流程内，每张图片）

| 阶段 | 毫秒/张 |
| --- | --- |
| preprocess | 0.166 |
| inference | 2.248 |
| loss | 0.001 |
| postprocess | 0.669 |

> 验证总耗时 32.2 秒（含模型加载与全部 3,229 张图片）。

## 四、错误案例分析

在 **conf=0.3、IoU=0.5** 的固定阈值下逐图匹配（扫描 3,229 张）：

| 项目 | 数量 |
| --- | --- |
| 命中（TP） | 2,296 |
| 误报（FP） | 943 |
| 漏检（FN） | 1,582 |
| 含误报的图片数 | 654 |
| 含漏检的图片数 | 841 |

**漏检按类别**：fire 1,039，smoke 543

**误报按类别**：fire 579，smoke 364

- 漏检样本图（12 张）：见 `figures/error_cases_missed.png` 与 `E:\forest_smoke\outputs\predictions\error_cases_20261029_042905\missed`
- 误报样本图（12 张）：见 `figures/error_cases_false_alarm.png` 与 `E:\forest_smoke\outputs\predictions\error_cases_20261029_042905\false_alarm`

### 漏检样例（按标注框面积从大到小）

| 图片 | 类别 | 面积占比 | 该图预测框数 |
| --- | --- | --- | --- |
| WEB03881.jpg | fire | 0.9581 | 0 |
| WEB05732.jpg | fire | 0.8954 | 3 |
| WEB11571.jpg | smoke | 0.8782 | 0 |
| WEB08423.jpg | smoke | 0.8760 | 0 |
| WEB07003.jpg | smoke | 0.8008 | 3 |
| WEB10723.jpg | fire | 0.7410 | 3 |
| WEB04562.jpg | smoke | 0.7361 | 1 |
| WEB10613.jpg | smoke | 0.7198 | 4 |

### 误报样例（按置信度从高到低）

| 图片 | 类别 | 置信度 | 与最近标注框 IoU |
| --- | --- | --- | --- |
| WEB05808.jpg | smoke | 0.833 | 0.011 |
| WEB07755.jpg | smoke | 0.832 | 0.477 |
| WEB05732.jpg | smoke | 0.823 | 0.948 |
| WEB06129.jpg | smoke | 0.817 | 0.316 |
| WEB09359.jpg | smoke | 0.816 | 0.470 |
| WEB05247.jpg | smoke | 0.808 | 0.239 |
| WEB07350.jpg | smoke | 0.807 | 0.342 |
| WEB05672.jpg | fire | 0.806 | 0.463 |

## 五、实验完成情况（如实标注）

| 实验 | 状态 | 说明 |
| --- | --- | --- |
| 实验一：基础模型（10 epoch，yolov8n） | ✅ 已完成 | 上表全部指标即本实验在 test 集的结果 |
| 实验二：参数对比（输入尺寸 512 vs 640） | ⬜ **未完成** | 需要重新训练，受本机 6 GB 显存与时间限制，尚未执行；本文件不含任何对比数字 |
| 实验三：复杂场景分组分析 | ⬜ **未完成** | 数据集没有“清晰/远距离/云雾/复杂背景”这类场景标签，需要人工分组；本文件只做通用误报/漏检统计 |

## 六、图表

- `outputs\figures\evaluation_hardneg_40epoch_cases\BoxF1_curve.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\BoxP_curve.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\BoxPR_curve.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\BoxR_curve.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\confusion_matrix.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\confusion_matrix_normalized.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\pr_curve.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\metrics_by_class.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\error_cases_missed.png`
- `outputs\figures\evaluation_hardneg_40epoch_cases\error_cases_false_alarm.png`

## 七、局限与偏差（必须写进论文）

- 数据划分是按感知哈希近似分组，不是真实视频/事件分组，指标可能偏乐观。
- 跨集合仍存在少量近似重复图片（阶段五统计：19 组），会抬高测试集指标。
- 数据集不是纯森林场景（含城市、工业、夜间监控画面），不能宣称“森林专用模型”。
- test 集只在本次评价中使用一次；本文件中所有指标都来自这一次运行。
- 错误案例分析使用的是 conf=0.25 的固定阈值，mAP 使用的是 conf=0.001 积分，两者不可混用。
- test 集标注共 3,878 条，其中含极少量退化/越界框（全数据集共 26 条），本次未逐条剔除。

## 八、声明

> 模型识别结果仅供辅助分析，需人工核实；本系统不做森林火灾预警，不能仅凭检测结果确认真实火灾。
