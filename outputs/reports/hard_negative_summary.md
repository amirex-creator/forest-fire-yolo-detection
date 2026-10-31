# 困难负样本整理报告

- 基线权重：`models\batch3_complete_30_full_20261029_003055_best.pt`
- train 集空标签候选：6491 张；本次扫描：6491 张（seed=42）
- 触发预测并复制：148
- 置信度阈值：0.25
- 训练清单：`datasets/hard_negatives_full/train_manifest.txt`

- 改进实验配置：`configs/forest_smoke_hard_negatives_full.yaml`

这些图片来自 train 集，标签保持为空；val/test 未参与构建。
