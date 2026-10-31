"""Collect representative error cases from completed test artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "smoke", 1: "fire"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolved(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build representative false-positive and miss examples")
    parser.add_argument("--metrics", default="outputs/reports/evaluation_hardneg_40epoch_cases/metrics.json")
    parser.add_argument("--calibration", default="outputs/reports/calibrated_test_results_merged.json")
    parser.add_argument("--out", default="outputs/reports/case_review.json")
    args = parser.parse_args()

    metrics = read_json(resolved(args.metrics))
    calibration = read_json(resolved(args.calibration))
    weight = Path(metrics["weights"]["path"]).name
    calibrated = next((item for item in calibration["models"] if Path(item["weights"]).name == weight), None)
    if calibrated is None:
        raise ValueError(f"阈值标定报告缺少模型：{weight}")

    threshold = float(calibrated["validation"]["selected_threshold"])
    eval_threshold = float(metrics["config"]["error_conf"])
    if abs(threshold - eval_threshold) > 1e-9:
        raise ValueError(f"案例评价阈值 {eval_threshold:.3f} 与验证集选定阈值 {threshold:.3f} 不一致")

    negatives = calibrated["test"].get("negative_false_alarm_examples", [])
    if not negatives:
        raise ValueError("该模型没有空标签图误报，无法生成实际误检案例")
    fp = max(negatives, key=lambda row: row["max_confidence"])
    fp_source = ROOT / "datasets" / "forest_smoke" / "images" / "test" / fp["image"]
    fp_label = ROOT / "datasets" / "forest_smoke" / "labels" / "test" / f"{fp_source.stem}.txt"
    if not fp_source.is_file() or not fp_label.is_file() or fp_label.read_text(encoding="utf-8").strip():
        raise ValueError(f"误检样本不是有效的空标签测试图：{fp_source.name}")

    output_dir = ROOT / "outputs" / "figures" / "case_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_image = cv2.imread(str(fp_source))
    if fp_image is None:
        raise ValueError(f"无法读取误检源图片：{fp_source}")
    for detection in fp["detections"]:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["box_xyxy"])
        color = (52, 91, 200) if detection["class_name"] == "fire" else (41, 138, 128)
        cv2.rectangle(fp_image, (x1, y1), (x2, y2), color, max(2, min(fp_image.shape[:2]) // 250))
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(fp_image, label, (max(0, x1), max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, min(fp_image.shape[:2]) / 1000), color, 2, cv2.LINE_AA)
    fp_output = output_dir / "false_alarm_hardneg_final.jpg"
    if not cv2.imwrite(str(fp_output), fp_image):
        raise OSError(f"无法写入误检案例图：{fp_output}")
    top_detection = max(fp["detections"], key=lambda row: row["confidence"])

    saved_fn = (metrics.get("error_cases") or {}).get("saved_fn") or []
    missed_examples = (metrics.get("error_case_details") or {}).get("missed_examples") or []
    missed_by_name = {row["image"]: row for row in missed_examples}
    fn_output = next((resolved(path) for path in saved_fn if resolved(path).is_file()), None)
    if fn_output is None:
        raise ValueError("评价报告中没有可用的漏检案例图片")
    fn_name = fn_output.name.split("__missed")[0] + ".jpg"
    fn_record = missed_by_name.get(fn_name)
    fn_source = ROOT / "datasets" / "forest_smoke" / "images" / "test" / fn_name
    fn_label = ROOT / "datasets" / "forest_smoke" / "labels" / "test" / f"{Path(fn_name).stem}.txt"
    if fn_record is None and fn_label.is_file():
        for line in fn_label.read_text(encoding="utf-8", errors="replace").splitlines():
            columns = line.split()
            if len(columns) == 5 and columns[0].isdigit():
                class_id = int(columns[0])
                fn_record = {"image": fn_name, "class_id": class_id, "class_name": CLASS_NAMES.get(class_id, str(class_id))}
                break
    if fn_record is None:
        raise ValueError(f"无法关联漏检样例的真实类别：{fn_name}")

    cases = [
        {
            "error_type": "false_positive",
            "category": "室内高亮结构误报",
            "image": fp["image"],
            "class_name": top_detection["class_name"],
            "confidence": top_detection["confidence"],
            "validation_threshold": threshold,
            "observation": f"测试集该图的标签为空；画面是室内大厅，模型在左上方明亮的天花板/墙面区域输出 {top_detection['class_name']} 框，置信度 {top_detection['confidence']:.3f}。",
            "hypothesis": "大面积高亮、低纹理区域可能与模型学到的烟雾外观产生相似响应；单个案例不能确认因果，需要更多室内和高亮背景负样本验证。",
            "source_image": str(fp_source.relative_to(ROOT)).replace("\\", "/"),
            "annotated_image": str(fp_output.relative_to(ROOT)).replace("\\", "/"),
        },
        {
            "error_type": "false_negative",
            "category": "低照度小目标漏检",
            "image": fn_name,
            "class_name": fn_record["class_name"],
            "confidence": None,
            "validation_threshold": threshold,
            "observation": f"测试集夜间画面中包含 {fn_record['class_name']} 标注；较大的火焰可见，但中间偏小的火点在置信度阈值 {threshold:.3f}、匹配 IoU 0.50 下未被正确匹配。",
            "hypothesis": "低照度、强亮度对比和小目标尺寸可能共同降低定位稳定性；这里是基于画面的分析假设，不是单样本因果证明。",
            "source_image": str(fn_source.relative_to(ROOT)).replace("\\", "/"),
            "annotated_image": str(fn_output.relative_to(ROOT)).replace("\\", "/"),
        },
    ]
    out = resolved(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": weight,
        "test_report": args.metrics.replace("\\", "/"),
        "confidence_threshold": threshold,
        "match_iou": metrics["error_cases"]["iou_threshold"],
        "cases": cases,
        "note": "图片和预测取自本项目实际 test 评估；hypothesis 字段需要在浏览图片后核实或修订。",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[文件] {out}")
    for case in cases:
        print(f"[{case['error_type']}] {case['image']} -> {case['annotated_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
