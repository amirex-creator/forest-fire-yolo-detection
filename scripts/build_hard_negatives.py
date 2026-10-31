"""Build a training-only hard-negative manifest from empty-label images.

The script never reads the validation or test split. It runs the baseline model
on empty-label training images, copies images that trigger a prediction into a
separate folder, and writes empty YOLO labels plus a manifest for a second run.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES = ROOT / "datasets" / "forest_smoke" / "images" / "train"
TRAIN_LABELS = ROOT / "datasets" / "forest_smoke" / "labels" / "train"
REPORT_DIR = ROOT / "outputs" / "reports"


def source_family(image_name: str) -> str:
    match = re.match(r"[A-Za-z]+", Path(image_name).stem)
    return match.group(0) if match else Path(image_name).stem


def image_paths() -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        p for p in TRAIN_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
        and (TRAIN_LABELS / f"{p.stem}.txt").is_file()
        and (TRAIN_LABELS / f"{p.stem}.txt").read_text(encoding="utf-8").strip() == ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select baseline false-positive training negatives")
    parser.add_argument("--weights", required=True, help="baseline *_best.pt")
    parser.add_argument("--limit", type=int, default=0, help="maximum candidates to scan; 0 scans all")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--max-selected", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default="datasets/hard_negatives_full")
    args = parser.parse_args()

    candidates = image_paths()
    if not candidates:
        raise SystemExit("没有找到 train 集空标签图片")
    random.Random(args.seed).shuffle(candidates)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    model = YOLO(str((ROOT / args.weights).resolve() if not Path(args.weights).is_absolute() else args.weights))
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    scanned = 0
    for image in candidates:
        scanned += 1
        result = model.predict(
            source=str(image), conf=float(args.min_conf), imgsz=640,
            device=args.device, verbose=False,
        )[0]
        count = int(len(result.boxes)) if result.boxes is not None else 0
        if count <= 0:
            continue
        target_name = f"hn_{len(selected):05d}{image.suffix.lower()}"
        shutil.copy2(image, out_images / target_name)
        (out_labels / f"{Path(target_name).stem}.txt").write_text("", encoding="utf-8")
        selected.append({
            "source": str(image.relative_to(ROOT)).replace("\\", "/"),
            "copy": str((out_images / target_name).relative_to(ROOT)).replace("\\", "/"),
            "predicted_boxes": count,
            "max_conf": round(float(result.boxes.conf.max().item()), 6),
        })
        if len(selected) >= args.max_selected:
            break

    # A text manifest lets Ultralytics mix the original train images with the
    # selected hard negatives without changing the source dataset in place.
    manifest = out_root / "train_manifest.txt"
    originals = sorted(TRAIN_IMAGES.glob("*"))
    lines = [str(p.resolve()).replace("\\", "/") for p in originals if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    lines.extend(str((out_images / Path(item["copy"]).name).resolve()).replace("\\", "/") for item in selected)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hardneg_yaml = ROOT / "configs" / f"forest_smoke_{out_root.name}.yaml"
    hardneg_yaml.write_text(
        "\n".join(
            [
                f"path: {(ROOT / 'datasets' / 'forest_smoke').as_posix()}",
                f"train: {manifest.resolve().as_posix()}",
                "val: images/val",
                "test: images/test",
                "nc: 2",
                "names: [smoke, fire]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_weights": str(Path(args.weights)),
        "source_split": "train only, empty-label images",
        "candidate_limit": args.limit,
        "candidate_pool": len(image_paths()),
        "scanned": scanned,
        "selected": len(selected),
        "min_conf": args.min_conf,
        "inference_batch": 1,
        "seed": args.seed,
        "selected_source_prefix_counts": {
            family: sum(source_family(row["source"]) == family for row in selected)
            for family in sorted({source_family(row["source"]) for row in selected})
        },
        "manifest": str(manifest.relative_to(ROOT)).replace("\\", "/"),
        "data_config": str(hardneg_yaml.relative_to(ROOT)).replace("\\", "/"),
        "examples": selected[:50],
        "note": "Selected images retain empty YOLO labels and are never taken from val/test.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "hard_negative_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "hard_negative_summary.md").write_text(
        "# 困难负样本整理报告\n\n"
        f"- 基线权重：`{args.weights}`\n"
        f"- train 集空标签候选：{report['candidate_pool']} 张；本次扫描：{scanned} 张（seed={args.seed}）\n"
        f"- 触发预测并复制：{len(selected)}\n"
        f"- 置信度阈值：{args.min_conf}\n"
        f"- 训练清单：`{manifest.relative_to(ROOT).as_posix()}`\n\n"
        f"- 改进实验配置：`{hardneg_yaml.relative_to(ROOT).as_posix()}`\n\n"
        "这些图片来自 train 集，标签保持为空；val/test 未参与构建。\n",
        encoding="utf-8",
    )
    print(json.dumps({"scanned": scanned, "selected": len(selected), "manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
