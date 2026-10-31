"""Choose confidence thresholds on validation data, then report test errors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src import evaluation, paths


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_paths(split: str) -> tuple[list[Path], Path]:
    image_dir, label_dir = paths.split_dirs(split)
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), label_dir


def infer_split(model: YOLO, split: str, args: argparse.Namespace, infer_conf: float) -> list[dict]:
    images, label_dir = image_paths(split)
    records: list[dict] = []
    for index, image_path in enumerate(images, start=1):
        result = model.predict(
            source=str(image_path),
            conf=infer_conf,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        label_path = label_dir / f"{image_path.stem}.txt"
        gt = evaluation.read_yolo_labels(label_path, width, height)
        preds: list[tuple[int, float, tuple[float, float, float, float]]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            preds = [
                (int(classes[i]), float(scores[i]), tuple(float(v) for v in xyxy[i]))
                for i in range(len(xyxy))
            ]
        empty_label = label_path.is_file() and not label_path.read_text(encoding="utf-8", errors="replace").strip()
        records.append({"image": image_path.name, "gt": gt, "preds": preds, "empty": empty_label})
        if index % 500 == 0:
            print(f"  {split}: {index}/{len(images)}", flush=True)
    return records


def score(records: list[dict], conf: float, iou: float, include_examples: bool = False) -> dict:
    counts = {"tp": 0, "fp": 0, "fn": 0, "negative_images": 0, "negative_images_flagged": 0, "negative_fp_boxes": 0}
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("smoke", "fire")}
    negative_false_alarm_examples: list[dict] = []
    for row in records:
        preds = [p for p in row["preds"] if p[1] >= conf]
        matched, unmatched_pred, unmatched_gt = evaluation.match_boxes(preds, row["gt"], iou)
        counts["tp"] += len(matched)
        counts["fp"] += len(unmatched_pred)
        counts["fn"] += len(unmatched_gt)
        if row["empty"]:
            counts["negative_images"] += 1
            counts["negative_fp_boxes"] += len(unmatched_pred)
            counts["negative_images_flagged"] += int(bool(unmatched_pred))
            if include_examples and unmatched_pred:
                detections = sorted(
                    (
                        {
                            "class_name": "smoke" if preds[index][0] == 0 else "fire",
                            "confidence": round(preds[index][1], 6),
                            "box_xyxy": [round(value, 1) for value in preds[index][2]],
                        }
                        for index in unmatched_pred
                    ),
                    key=lambda item: item["confidence"],
                    reverse=True,
                )
                negative_false_alarm_examples.append({
                    "image": row["image"],
                    "max_confidence": detections[0]["confidence"],
                    "detections": detections,
                })
        for index in matched:
            per_class["smoke" if preds[index][0] == 0 else "fire"]["tp"] += 1
        for index in unmatched_pred:
            per_class["smoke" if preds[index][0] == 0 else "fire"]["fp"] += 1
        for index in unmatched_gt:
            per_class["smoke" if row["gt"][index][0] == 0 else "fire"]["fn"] += 1
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    negative_false_alarm_examples.sort(key=lambda item: item["max_confidence"], reverse=True)
    result = {
        **counts,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
        "negative_image_false_alarm_rate": round(
            counts["negative_images_flagged"] / max(1, counts["negative_images"]), 6
        ),
        "per_class": per_class,
    }
    if include_examples:
        result["negative_false_alarm_examples"] = negative_false_alarm_examples
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate on val and evaluate the fixed threshold on test")
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--infer-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--out", default="outputs/reports/calibrated_test_results.json")
    args = parser.parse_args()
    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise SystemExit("置信度候选值必须位于 (0, 1)")
    if not 0 < args.iou < 1 or not 0 < args.nms_iou < 1:
        raise SystemExit("匹配 IoU 与 NMS IoU 必须位于 (0, 1)")

    models = []
    for raw in args.weights:
        weight = Path(raw)
        if not weight.is_absolute():
            weight = ROOT / weight
        weight = weight.resolve()
        if not weight.is_file():
            raise SystemExit(f"权重不存在：{weight}")
        print(f"[校准] {weight.name}：在 val 集选择阈值", flush=True)
        model = YOLO(str(weight))
        started = time.perf_counter()
        val_records = infer_split(model, "val", args, args.infer_conf)
        sweep = [{"threshold": t, **score(val_records, t, args.iou)} for t in thresholds]
        # Prefer higher precision and then the higher threshold when F1 is tied.
        best = max(sweep, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
        best = {"threshold": best["threshold"], **score(val_records, best["threshold"], args.iou, include_examples=True)}
        print(f"[测试] val 选中 conf={best['threshold']:.2f}, F1={best['f1']:.4f}; 测试集评估开始", flush=True)
        test_records = infer_split(model, "test", args, args.infer_conf)
        test_metrics = score(test_records, best["threshold"], args.iou, include_examples=True)
        models.append({
            "weights": str(weight.relative_to(ROOT)).replace("\\", "/"),
            "weights_sha256": sha256(weight),
            "validation": {
                "images": len(val_records),
                "objects": sum(len(row["gt"]) for row in val_records),
                "iou_threshold": args.iou,
                "nms_iou": args.nms_iou,
                "infer_conf_floor": args.infer_conf,
                "threshold_sweep": sweep,
                "selected_threshold": best["threshold"],
                "selected_metrics": {k: v for k, v in best.items() if k != "threshold"},
            },
            "test": {
                "images": len(test_records),
                "objects": sum(len(row["gt"]) for row in test_records),
                "iou_threshold": args.iou,
                "nms_iou": args.nms_iou,
                "confidence_threshold_from_validation": best["threshold"],
                **test_metrics,
            },
            "seconds": round(time.perf_counter() - started, 2),
        })

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Maximum micro F1 over the supplied confidence grid on val; tie-break by precision then higher confidence. Predictions use the recorded NMS IoU. The selected confidence threshold is applied unchanged to test.",
        "models": models,
        "disclaimer": "Empty-label image false-alarm rate is measured only on labeled negatives in this dataset and is not a real-world wildfire alarm rate.",
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["model", "threshold", "val_precision", "val_recall", "val_f1", "test_precision", "test_recall", "test_f1", "negative_images", "negative_images_flagged", "negative_image_false_alarm_rate", "negative_fp_boxes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in models:
            val = item["validation"]["selected_metrics"]
            test = item["test"]
            writer.writerow({
                "model": Path(item["weights"]).name,
                "threshold": item["validation"]["selected_threshold"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_f1": val["f1"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
                "negative_images": test["negative_images"],
                "negative_images_flagged": test["negative_images_flagged"],
                "negative_image_false_alarm_rate": test["negative_image_false_alarm_rate"],
                "negative_fp_boxes": test["negative_fp_boxes"],
            })
    md = [
        "# 验证集阈值标定与测试集结果",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 方法：{payload['method']}",
        f"- IoU 匹配阈值：{args.iou}",
        f"- 预测 NMS IoU：{args.nms_iou}",
        "",
        "| 模型 | val 选中置信度 | val F1 | test P | test R | test F1 | 空标签测试图误报率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        val = item["validation"]["selected_metrics"]
        test = item["test"]
        md.append(
            f"| `{Path(item['weights']).name}` | {item['validation']['selected_threshold']:.2f} | "
            f"{val['f1']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | "
            f"{test['f1']:.4f} | {test['negative_images_flagged']}/{test['negative_images']} "
            f"({test['negative_image_false_alarm_rate']:.1%}) |"
        )
    md.extend(["", payload["disclaimer"], ""])
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"[文件] {out}", flush=True)
    print(f"[文件] {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
