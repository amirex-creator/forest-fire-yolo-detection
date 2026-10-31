"""Measure image-level false alarms on the same empty-label test subset."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets" / "forest_smoke" / "images" / "test"
LABEL_DIR = ROOT / "datasets" / "forest_smoke" / "labels" / "test"
OUT_DIR = ROOT / "outputs" / "reports"


def empty_label_images() -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    return [
        image for image in images
        if (LABEL_DIR / f"{image.stem}.txt").is_file()
        and not (LABEL_DIR / f"{image.stem}.txt").read_text(encoding="utf-8").strip()
    ]


def evaluate(weights: Path, images: list[Path], conf: float, imgsz: int, device: str, batch: int) -> dict:
    model = YOLO(str(weights))
    total = 0
    flagged: list[dict] = []
    boxes_by_class: dict[str, int] = {}
    # Predict one image at a time. Passing a large list of sources to the
    # Ultralytics stream predictor can make its warmup infer an oversized
    # tensor on some Windows/CUDA combinations, causing an avoidable OOM.
    for image in images:
        result = model.predict(
            source=str(image),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        total += 1
        boxes = result.boxes
        count = int(len(boxes)) if boxes is not None else 0
        if not count:
            continue
        names = result.names
        classes: dict[str, int] = {}
        for class_id in boxes.cls.detach().cpu().tolist():
            name = str(names.get(int(class_id), int(class_id)))
            classes[name] = classes.get(name, 0) + 1
            boxes_by_class[name] = boxes_by_class.get(name, 0) + 1
        flagged.append({
            "image": Path(result.path).name,
            "boxes": count,
            "classes": classes,
            "max_confidence": round(float(boxes.conf.max().item()), 6),
        })
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights": str(weights.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "split": "test",
        "subset": "images whose ground-truth label files are empty",
        "images": total,
        "images_with_false_alarm": len(flagged),
        "image_false_alarm_rate": round(len(flagged) / total, 6) if total else None,
        "false_positive_boxes": sum(row["boxes"] for row in flagged),
        "boxes_by_class": boxes_by_class,
        "thresholds": {"conf": conf, "imgsz": imgsz, "device": device, "batch": batch},
        "flagged_images": flagged,
        "disclaimer": "Image-level false alarms on labeled negatives; this does not measure all real-world backgrounds.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    images = empty_label_images()
    if not images:
        raise SystemExit("No empty-label images found in test split")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for raw_weight in args.weights:
        weight = Path(raw_weight)
        if not weight.is_absolute():
            weight = ROOT / weight
        result = evaluate(weight.resolve(), images, args.conf, args.imgsz, args.device, args.batch)
        out = OUT_DIR / f"negative_false_alarms_{weight.stem}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_rows.append({
            "model": weight.name,
            "negative_images": result["images"],
            "images_with_false_alarm": result["images_with_false_alarm"],
            "image_false_alarm_rate": result["image_false_alarm_rate"],
            "false_positive_boxes": result["false_positive_boxes"],
            "conf": args.conf,
        })
        print(f"{weight.name}: {result['images_with_false_alarm']}/{result['images']} images, rate={result['image_false_alarm_rate']}")
    comparison = OUT_DIR / "negative_false_alarm_comparison.csv"
    with comparison.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
