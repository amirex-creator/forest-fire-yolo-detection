# 验证集阈值标定与测试集结果

- 生成时间：2026-10-29T04:24:50+08:00
- 方法：Maximum micro F1 over the supplied confidence grid on val; tie-break by precision then higher confidence. Predictions use the recorded NMS IoU. The selected confidence threshold is applied unchanged to test.
- IoU 匹配阈值：0.5
- 预测 NMS IoU：0.7

| 模型 | val 选中置信度 | val F1 | test P | test R | test F1 | 空标签测试图误报率 |
|---|---:|---:|---:|---:|---:|---:|
| `hardneg_finetune_cont30_full_20261029_034435_best.pt` | 0.30 | 0.6787 | 0.7089 | 0.5921 | 0.6452 | 89/1645 (5.4%) |

Empty-label image false-alarm rate is measured only on labeled negatives in this dataset and is not a real-world wildfire alarm rate.
