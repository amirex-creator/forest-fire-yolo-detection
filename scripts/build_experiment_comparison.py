"""Build presentation-ready reports from completed evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def weight_name(metrics: dict[str, Any]) -> str:
    weights = metrics.get("weights") or {}
    if isinstance(weights, str):
        value = weights
    elif isinstance(weights, dict):
        value = weights.get("name") or weights.get("path")
    else:
        value = None
    if not value:
        raise ValueError("评价报告缺少权重路径")
    return Path(value).name


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine real evaluation reports and calibrated test results")
    parser.add_argument("--metrics-10", default="outputs/reports/metrics_10epoch_test.json")
    parser.add_argument("--metrics-30", default="outputs/reports/evaluation_30epoch/metrics.json")
    parser.add_argument("--metrics-hardneg-first", default="outputs/reports/evaluation_hardneg/metrics.json")
    parser.add_argument("--metrics-hardneg-final", default="outputs/reports/evaluation_hardneg_40epoch_cases/metrics.json")
    parser.add_argument("--calibration", default="outputs/reports/calibrated_test_results_merged.json")
    parser.add_argument("--train-final", default="outputs/reports/train_hardneg_finetune_cont30_full_20261029_034435.json")
    parser.add_argument("--hard-negative", default="outputs/reports/hard_negative_summary.json")
    parser.add_argument("--dataset", default="outputs/reports/dataset_summary.json")
    parser.add_argument("--final-finetune-epochs", type=int, default=40,
                        help="累计困难负样本微调轮数，不含 30 轮基线")
    parser.add_argument("--out", default="outputs/reports/experiment_comparison.json")
    args = parser.parse_args()

    calibration = read_json(ROOT / args.calibration)
    calibrated = {weight_name(item): item for item in calibration.get("models", [])}
    inputs = [
        ("10 轮基线", 10, Path(args.metrics_10)),
        ("30 轮基线", 30, Path(args.metrics_30)),
        ("困难负样本首轮微调", 10, Path(args.metrics_hardneg_first)),
        ("困难负样本微调", args.final_finetune_epochs, Path(args.metrics_hardneg_final)),
    ]

    models: list[dict[str, Any]] = []
    for label, epoch_count, raw_path in inputs:
        metrics = read_json(ROOT / raw_path)
        filename = weight_name(metrics)
        tuned = calibrated.get(filename)
        if tuned is None:
            raise ValueError(f"验证集阈值标定报告缺少模型：{filename}")
        overall = metrics.get("overall") or {}
        for key in ("precision", "recall", "map50", "map50_95"):
            if key not in overall:
                raise ValueError(f"{raw_path} 缺少 overall.{key}")
        dataset = metrics.get("dataset") or {}
        calibrated_classes = tuned["test"].get("per_class") or {}
        per_class = []
        for item in metrics.get("per_class") or []:
            name = item["class_name"]
            counts = calibrated_classes.get(name)
            if counts is None:
                raise ValueError(f"阈值化测试报告缺少类别统计：{filename} / {name}")
            tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
            per_class.append({
                **item,
                "instances": tp + fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "threshold_tp": tp,
                "threshold_fp": fp,
                "threshold_fn": fn,
            })
        model = {
            "label": label,
            "model": filename,
            "epochs": epoch_count,
            "test_images": dataset.get("images"),
            "test_objects": dataset.get("objects"),
            "overall": overall,
            "per_class": per_class,
            "evaluation_config": metrics.get("config") or {},
            "validation_threshold": tuned["validation"]["selected_threshold"],
            "validation_at_threshold": tuned["validation"]["selected_metrics"],
            "threshold_test": tuned["test"],
            "evaluation_report": str(raw_path).replace("\\", "/"),
            "figures": metrics.get("figures") or [],
        }
        if dataset.get("images") != tuned["test"].get("images"):
            raise ValueError(f"评价与阈值测试的图像数量不一致：{filename}")
        models.append(model)

    dataset_summary = read_json(ROOT / args.dataset)
    hard_negative = read_json(ROOT / args.hard_negative)
    training_path = ROOT / args.train_final
    training = read_json(training_path) if training_path.is_file() else None
    if training and training.get("status") != "success":
        raise ValueError(f"最终训练报告未成功完成：{training.get('status')}")

    protocol = (
        "mAP 指标来自 Ultralytics 对同一 test 划分的评价；固定阈值 P/R/F1 和空标签图误报率使用验证集网格搜索选出的阈值，"
        "并原样应用到 test。分类别 P/R 也由该固定阈值下的匹配计数计算。空标签图误报率 = 至少出现一个预测框的空标签 test 图片数 / 空标签 test 图片总数。"
    )
    limitations = [
        "原始视频或事件编号不可用，数据按感知哈希近似分组；仍存在少量近似重复图跨集合。",
        "数据集包含林地、野外、城市与工业等画面，不是纯森林数据集。",
        "多个检查点在同一 test 划分上进行比较，因此该划分不再是未触碰的盲测集。",
        "空标签图误报率仅描述本数据集的标注负样本，不代表现实场景中的火灾报警率。",
    ]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "D-Fire Kaggle 镜像；train/val/test = 70/15/15，按感知哈希近似分组",
        "protocol": protocol,
        "protocol_note": protocol,
        "models": models,
        "limitations": limitations,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, payload)

    csv_path = out.with_suffix(".csv")
    fields = [
        "model", "epochs", "test_images", "test_objects", "threshold", "precision", "recall", "f1",
        "map50", "map50_95", "negative_images", "negative_images_flagged", "negative_image_false_alarm_rate",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            test = model["threshold_test"]
            writer.writerow({
                "model": model["label"],
                "epochs": model["epochs"],
                "test_images": model["test_images"],
                "test_objects": model["test_objects"],
                "threshold": model["validation_threshold"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "map50": model["overall"]["map50"],
                "map50_95": model["overall"]["map50_95"],
                "negative_images": test["negative_images"],
                "negative_images_flagged": test["negative_images_flagged"],
                "negative_image_false_alarm_rate": test["negative_image_false_alarm_rate"],
            })

    md = [
        "# 森林火灾检测模型对比", "", f"- 生成时间：{payload['generated_at']}", f"- 评价协议：{protocol}", "",
        "| 模型 | 训练轮数 | test 图像 | 阈值 | Precision | Recall | F1 | mAP50 | mAP50-95 | 空标签图误报率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        test = model["threshold_test"]
        md.append(
            f"| {model['label']} | {model['epochs']} | {model['test_images']:,} | {model['validation_threshold']:.2f} | "
            f"{test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | "
            f"{model['overall']['map50']:.4f} | {model['overall']['map50_95']:.4f} | "
            f"{test['negative_images_flagged']}/{test['negative_images']} ({test['negative_image_false_alarm_rate']:.1%}) |"
        )
    md.extend(["", "## 数据限制", "", *[f"- {item}" for item in limitations], ""])
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    selected = models[-1]
    selected_payload = {
        "generated_at": payload["generated_at"],
        "selected_model": selected,
        "dataset": dataset_summary,
        "hard_negative": {
            "candidates": hard_negative.get("candidate_pool"),
            "scanned": hard_negative.get("scanned"),
            "mined_images": hard_negative.get("selected"),
            "confidence_floor": hard_negative.get("min_conf"),
            "source_counts": hard_negative.get("selected_source_prefix_counts"),
            "summary_report": args.hard_negative.replace("\\", "/"),
        },
        "training": {
            "run_name": training.get("run_name") if training else None,
            "status": training.get("status") if training else None,
            "continued_epochs": len(training.get("epochs", [])) if training else None,
            "cumulative_hard_negative_epochs": args.final_finetune_epochs,
            "settings": training.get("settings") if training else None,
            "device": training.get("device") if training else None,
            "cuda_peak": training.get("cuda_peak") if training else None,
            "report": args.train_final.replace("\\", "/"),
        },
        "limitations": limitations,
    }
    write_json(out.with_name("selected_model_summary.json"), selected_payload)

    print(f"[文件] {out}")
    print(f"[文件] {csv_path}")
    print(f"[文件] {out.with_suffix('.md')}")
    print(f"[文件] {out.with_name('selected_model_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
