"""模型评价与实验对比（阶段八）。

这个模块负责三件事，且**只使用 test 集做一次性最终评价**：

1. 用 Ultralytics 的 ``model.val`` 在 test 集上算 Precision / Recall / mAP50 / mAP50-95，
   并分 smoke、fire 两类给出同样四项指标；
2. 复制/绘制混淆矩阵与 PR 曲线；
3. 按固定阈值（默认 conf=0.25、IoU=0.5）跑一遍逐图匹配，收集**漏检**与**误报**样本，
   保存带标注的样本图，做案例分析。

所有数字都来自真实运行；没跑的实验在报告里标“未完成”，不填任何推测值。

命令行示例
----------
    # 1) 完整评价（test 集 + 错误案例 + 图表），权重自动取最新 *_best.pt
    python evaluate.py

    # 2) 指定权重与阈值
    python evaluate.py --weights models/batch1_e10_full_20261028_130129_best.pt --conf 0.25

    # 3) 只看指标，不收集错误案例（更快）
    python evaluate.py --no-error-cases

    # 4) 只用现有指标重绘报告（不重跑模型）
    python -c "from src import evaluation; evaluation.rebuild_reports()"

设计约束
--------
- 不内嵌 ``cv2.imread``：读图统一走 :mod:`src.image_io`。
- 路径全部来自 :mod:`src.paths`，不写死盘符。
- mAP 的置信度阈值固定为 Ultralytics 默认的 0.001（mAP 需要低阈值积分）；
  错误案例分析用的是**另一个**阈值（默认 0.25），两者在报告里分开写清楚。
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src import paths
from src.image_io import safe_imread, safe_imwrite

#: 评价指标的最小集合（课程论文要求）
REQUIRED_METRICS: tuple[str, ...] = ("precision", "recall", "map50", "map50_95")

#: 预训练 COCO 权重，不能当作本项目模型
_COCO_WEIGHTS: tuple[str, ...] = tuple(
    f"yolo{v}{s}.pt" for v in ("v8", "v9", "v10", "11") for s in ("n", "s", "m", "l", "x")
)
_TRAINED_SUFFIXES: tuple[str, ...] = ("_best.pt", "_last.pt")

DISCLAIMER = (
    "模型识别结果仅供辅助分析，需人工核实；本系统不做森林火灾预警，"
    "不能仅凭检测结果确认真实火灾。"
)

SCHEMA_VERSION = "1.0"
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
#: mAP 计算用的置信度阈值（Ultralytics 默认，低阈值才能积分出完整 PR 曲线）
MAP_CONF = 0.001
#: mAP 计算用的 NMS IoU
MAP_IOU = 0.7
#: 错误案例分析用的置信度阈值（部署时的典型阈值）
ERROR_CONF = 0.25
#: 判定“命中”的 IoU 阈值
MATCH_IOU = 0.5


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassMetrics:
    """单个类别的四项指标。"""

    class_id: int
    class_name: str
    instances: int
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(slots=True)
class MatchStats:
    """一次逐图匹配的统计（用于错误案例）。"""

    iou_threshold: float
    conf_threshold: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    fn_per_class: dict[str, int] = field(default_factory=dict)
    fp_per_class: dict[str, int] = field(default_factory=dict)
    images_with_fp: int = 0
    images_with_fn: int = 0
    images_scanned: int = 0
    saved_fn: list[str] = field(default_factory=list)
    saved_fp: list[str] = field(default_factory=list)
    seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "conf_threshold": self.conf_threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "fn_per_class": dict(sorted(self.fn_per_class.items())),
            "fp_per_class": dict(sorted(self.fp_per_class.items())),
            "images_with_fp": self.images_with_fp,
            "images_with_fn": self.images_with_fn,
            "images_scanned": self.images_scanned,
            "saved_fn": list(self.saved_fn),
            "saved_fp": list(self.saved_fp),
            "seconds": self.seconds,
        }


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def list_trained_weights(models_dir: Path | None = None) -> list[Path]:
    """列出本项目训练产物（排除 COCO 预训练权重），按修改时间倒序。"""
    root = Path(models_dir) if models_dir else paths.MODELS_DIR
    if not root.is_dir():
        return []
    found = [
        p
        for p in root.glob("*.pt")
        if p.name not in _COCO_WEIGHTS and p.name.endswith(_TRAINED_SUFFIXES)
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_weights(weights: str | Path | None = None) -> Path:
    """解析权重路径；显式路径不存在时抛 ``FileNotFoundError``。"""
    if weights is not None:
        p = Path(weights)
        if not p.is_absolute():
            candidate = (Path.cwd() / p).resolve()
            if candidate.is_file():
                return candidate
            candidate = (paths.PROJECT_ROOT / p).resolve()
            if candidate.is_file():
                return candidate
        elif p.is_file():
            return p.resolve()
        available = list_trained_weights()
        hint = "\n".join(f"  - {w.name}" for w in available[:5]) or "  （models 目录下暂无训练产物）"
        raise FileNotFoundError(
            f"权重文件不存在：{weights}\n可用的本项目权重：\n{hint}\n"
            "请先跑 train.py 训练，或用 --weights 指定已有的 *_best.pt / *_last.pt。"
        )

    available = list_trained_weights()
    if not available:
        raise FileNotFoundError(
            "没有找到本项目训练出的权重（models/*_best.pt 或 *_last.pt）。\n"
            "请先执行：python train.py --full --epochs 10\n"
            "注意：yolov8n.pt 是 COCO 预训练权重，不含 smoke/fire 类别，不能用来做本项目的评价。"
        )
    return available[0]


def weights_info(weights: Path) -> dict[str, Any]:
    path = Path(weights)
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(path),
    }


# ---------------------------------------------------------------------------
# 标注读取与几何工具
# ---------------------------------------------------------------------------


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO 标签，返回像素坐标 ``[(cls, x1, y1, x2, y2), ...]``。

    非法行直接抛 ``ValueError``（评价阶段宁可报错，也不要静默跳过）。
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return boxes
    for lineno, raw in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path.name} 第 {lineno} 行字段数不是 5：{line!r}")
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:])
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        x2 = (cx + w / 2.0) * width
        y2 = (cy + h / 2.0) * height
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    preds: Sequence[tuple[int, float, tuple[float, float, float, float]]],
    gts: Sequence[tuple[int, float, float, float, float]],
    iou_threshold: float = MATCH_IOU,
) -> tuple[list[int], list[int], list[int]]:
    """按“同类 + IoU≥阈值”做贪心匹配。

    ``preds`` 为 ``[(cls, conf, xyxy), ...]``；``gts`` 为 ``[(cls, x1, y1, x2, y2), ...]``。
    返回 ``(matched_pred_indices, unmatched_pred_indices, unmatched_gt_indices)``。
    """
    used_gt: set[int] = set()
    matched_pred: list[int] = []
    unmatched_pred: list[int] = []

    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    for pi in order:
        pcls, _conf, pbox = preds[pi]
        best_gi, best_iou = -1, 0.0
        for gi, gt in enumerate(gts):
            if gi in used_gt or gt[0] != pcls:
                continue
            iou = box_iou(pbox, gt[1:])
            if iou > best_iou:
                best_gi, best_iou = gi, iou
        if best_gi >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gi)
            matched_pred.append(pi)
        else:
            unmatched_pred.append(pi)

    unmatched_gt = [gi for gi in range(len(gts)) if gi not in used_gt]
    return matched_pred, unmatched_pred, unmatched_gt


# ---------------------------------------------------------------------------
# 主评价：调用 Ultralytics val
# ---------------------------------------------------------------------------


def _import_yolo():
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才触发
        raise RuntimeError(
            "需要 ultralytics：请先安装依赖\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return YOLO


def count_test_objects(split: str = "test") -> dict[str, Any]:
    """统计某个划分的图片数与各类实例数（不依赖模型，纯读标签）。"""
    img_dir, lbl_dir = paths.split_dirs(split)
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    per_class: dict[str, int] = {"smoke": 0, "fire": 0}
    objects = 0
    empty = 0
    for img in images:
        label = lbl_dir / f"{img.stem}.txt"
        if not label.is_file() or label.stat().st_size == 0:
            empty += 1
            continue
        for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            cls = int(float(line.split()[0]))
            objects += 1
            name = "smoke" if cls == 0 else "fire" if cls == 1 else f"class_{cls}"
            per_class[name] = per_class.get(name, 0) + 1
    return {"split": split, "images": len(images), "objects": objects, "empty_labels": empty, "per_class": per_class}


def _instance_counts_from_labels(split: str) -> dict[str, int]:
    """直接读标签文件统计每类实例数（不依赖模型）。"""
    try:
        return dict(count_test_objects(split)["per_class"])
    except Exception:  # noqa: BLE001 - 统计失败不应让评价崩掉
        return {}


def run_validation(
    weights: Path,
    split: str = "test",
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str | int | None = None,
    project: Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """在指定划分上跑 ``model.val``，返回指标与产物路径。

    注意：``workers=0`` 是必须的 —— 该沙箱禁止匿名管道，Ultralytics 的 val 默认用多进程。
    """
    YOLO = _import_yolo()
    out_project = Path(project) if project else paths.OUTPUTS_DIR / "eval_runs"
    out_project.mkdir(parents=True, exist_ok=True)
    run_name = name or f"val_{split}_{_stamp()}"

    # Ultralytics resolves ``train``/``val`` relative to ``path``.  The checked-in
    # config is intentionally project-relative for portability, so write a small
    # run-local config with an absolute dataset root before calling ``model.val``.
    runtime_yaml = out_project / f"{run_name}_data.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {paths.DATA_ROOT.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 2",
                "names:",
                "  0: smoke",
                "  1: fire",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = YOLO(str(weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(runtime_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=MAP_CONF,
        iou=MAP_IOU,
        device=device,
        workers=0,
        plots=True,
        project=str(out_project),
        name=run_name,
        verbose=False,
        exist_ok=True,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    box = metrics.box
    names = {int(k): str(v) for k, v in dict(getattr(metrics, "names", {}) or {}).items()}
    class_metrics = _per_class_metrics(
        box, names, instance_by_name=_instance_counts_from_labels(split)
    )

    save_dir = Path(getattr(metrics, "save_dir", out_project / run_name))
    figures = [p for p in sorted(save_dir.glob("*.png"))] if save_dir.is_dir() else []

    return {
        "value": metrics,
        "box": box,
        "names": names,
        "per_class": class_metrics,
        "speed": {k: _round(v, 3) for k, v in dict(getattr(metrics, "speed", {}) or {}).items()},
        "run_dir": str(save_dir),
        "val_figures": [str(p) for p in figures],
        "seconds": elapsed,
    }


def _per_class_metrics(
    box: Any, names: dict[int, str], instance_by_name: dict[str, int] | None = None
) -> list[ClassMetrics]:
    """从 ``Metric`` 对象抽出分类别指标。

    ``instances`` 优先用**读标签**得到的实例数：Ultralytics 的 ``nt_per_class`` 挂在
    ``DetMetrics`` 上而不是 ``Metric`` 上，直接取错对象会得到 0。
    """
    out: list[ClassMetrics] = []
    raw_index = getattr(box, "ap_class_index", None)
    indices = [int(x) for x in raw_index] if raw_index is not None else []
    if not indices:
        indices = sorted(names)
    p_arr = np.atleast_1d(np.asarray(getattr(box, "p", []), dtype=float))
    r_arr = np.atleast_1d(np.asarray(getattr(box, "r", []), dtype=float))
    ap50_arr = np.atleast_1d(np.asarray(getattr(box, "ap50", []), dtype=float))
    ap_arr = np.atleast_1d(np.asarray(getattr(box, "ap", []), dtype=float))
    nt = getattr(box, "nt_per_class", None)
    nt_arr = np.atleast_1d(np.asarray(nt, dtype=float)) if nt is not None else None
    counts = instance_by_name or {}

    for slot, cls_id in enumerate(indices):
        name = names.get(cls_id, f"class_{cls_id}")
        if name in counts:
            instances = int(counts[name])
        elif nt_arr is not None and cls_id < len(nt_arr):
            instances = int(nt_arr[cls_id])
        else:
            instances = 0
        out.append(
            ClassMetrics(
                class_id=cls_id,
                class_name=name,
                instances=instances,
                precision=_round(p_arr[slot]) or 0.0,
                recall=_round(r_arr[slot]) or 0.0,
                map50=_round(ap50_arr[slot]) or 0.0,
                map50_95=_round(ap_arr[slot]) or 0.0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 错误案例收集
# ---------------------------------------------------------------------------


def _draw_box(canvas: np.ndarray, xyxy: Sequence[float], color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    import cv2  # noqa: PLC0415

    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    if label:
        ty = y1 - 6 if y1 > 18 else y1 + 18
        cv2.putText(canvas, label, (max(0, x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def collect_error_cases(
    weights: Path,
    split: str = "test",
    conf: float = ERROR_CONF,
    iou_threshold: float = MATCH_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str | int | None = None,
    max_samples: int = 12,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """逐图匹配，收集漏检（FN）与误报（FP），保存代表性样本图。"""
    YOLO = _import_yolo()
    img_dir, lbl_dir = paths.split_dirs(split)
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if limit:
        images = images[:limit]
    out_root = Path(out_dir) if out_dir else paths.PREDICTIONS_DIR / f"error_cases_{_stamp()}"
    fn_dir = out_root / "missed"
    fp_dir = out_root / "false_alarm"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    stats = MatchStats(iou_threshold=iou_threshold, conf_threshold=conf)
    fn_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for img in images:
        image = safe_imread(img)
        h, w = image.shape[:2]
        gts = read_yolo_labels(lbl_dir / f"{img.stem}.txt", w, h)

        results = model.predict(
            source=image, conf=conf, iou=MAP_IOU, imgsz=imgsz, device=device, verbose=False
        )
        r0 = results[0]
        preds: list[tuple[int, float, tuple[float, float, float, float]]] = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(xyxy)):
                preds.append((int(clss[k]), float(confs[k]), tuple(float(v) for v in xyxy[k])))

        matched, unmatched_pred, unmatched_gt = match_boxes(preds, gts, iou_threshold)
        stats.images_scanned += 1
        stats.true_positives += len(matched)
        stats.false_positives += len(unmatched_pred)
        stats.false_negatives += len(unmatched_gt)
        if unmatched_pred:
            stats.images_with_fp += 1
        if unmatched_gt:
            stats.images_with_fn += 1

        name_by_cls = {0: "smoke", 1: "fire"}
        for gi in unmatched_gt:
            cls = gts[gi][0]
            cname = name_by_cls.get(cls, f"class_{cls}")
            stats.fn_per_class[cname] = stats.fn_per_class.get(cname, 0) + 1
            fn_rows.append(
                {
                    "image": img.name,
                    "class_id": cls,
                    "class_name": cname,
                    "gt_box_xyxy": [round(v, 1) for v in gts[gi][1:]],
                    "area_norm": round(
                        (gts[gi][3] - gts[gi][1]) * (gts[gi][4] - gts[gi][2]) / max(1.0, w * h), 6
                    ),
                    "predictions_in_image": len(preds),
                }
            )
        for pi in unmatched_pred:
            cls, score, pbox = preds[pi]
            cname = name_by_cls.get(cls, f"class_{cls}")
            stats.fp_per_class[cname] = stats.fp_per_class.get(cname, 0) + 1
            best_iou = max((box_iou(pbox, gts[g][1:]) for g in range(len(gts))), default=0.0)
            fp_rows.append(
                {
                    "image": img.name,
                    "class_id": cls,
                    "class_name": cname,
                    "confidence": round(score, 4),
                    "pred_box_xyxy": [round(v, 1) for v in pbox],
                    "best_iou_with_gt": round(best_iou, 4),
                    "gt_count_in_image": len(gts),
                }
            )

        # 保存样本图（每类最多 max_samples 张，按出现顺序）
        if unmatched_gt and len(stats.saved_fn) < max_samples:
            canvas = image.copy()
            for gt in gts:
                _draw_box(canvas, gt[1:], (0, 200, 0), f"GT {name_by_cls.get(gt[0], 'obj')}", 1)
            for gi in unmatched_gt:
                gt = gts[gi]
                _draw_box(canvas, gt[1:], (0, 0, 255), f"MISSED {name_by_cls.get(gt[0], 'obj')}", 3)
            target = fn_dir / f"{img.stem}__missed.jpg"
            safe_imwrite(target, canvas)
            stats.saved_fn.append(str(target))

        if unmatched_pred and len(stats.saved_fp) < max_samples:
            canvas = image.copy()
            for gt in gts:
                _draw_box(canvas, gt[1:], (0, 200, 0), "", 1)
            for pi in unmatched_pred:
                cls, score, pbox = preds[pi]
                _draw_box(canvas, pbox, (0, 165, 255), f"FP {name_by_cls.get(cls, 'obj')} {score:.2f}", 3)
            target = fp_dir / f"{img.stem}__false_alarm.jpg"
            safe_imwrite(target, canvas)
            stats.saved_fp.append(str(target))

    stats.seconds = round(time.perf_counter() - t0, 2)

    def _rank_fp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: r["confidence"], reverse=True)

    def _rank_fn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: r["area_norm"], reverse=True)

    payload = {
        "stats": stats.to_dict(),
        "out_dir": str(out_root),
        "missed_examples": _rank_fn(fn_rows)[:20],
        "false_alarm_examples": _rank_fp(fp_rows)[:20],
        "totals": {
            "missed_rows": len(fn_rows),
            "false_alarm_rows": len(fp_rows),
        },
    }
    (out_root / "error_cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------


def _setup_matplotlib() -> tuple[Any, bool]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import font_manager  # noqa: PLC0415

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True
    return plt, False


def _t(cjk: bool, zh: str, en: str) -> str:
    return zh if cjk else en


def plot_pr_curve(box: Any, figures_dir: Path | None = None, names: dict[int, str] | None = None) -> Path | None:
    """画 PR 曲线：能取到分类别曲线就画两条，否则退化为一条平均曲线。"""
    try:
        curves = box.curves_results
        px, py = curves[0][0], curves[0][1]
    except Exception:  # noqa: BLE001 - 曲线不可用时不阻断整个评价
        return None
    if px is None or py is None or len(px) == 0:
        return None

    plt, cjk = _setup_matplotlib()
    out_dir = Path(figures_dir) if figures_dir else paths.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pr_curve.png"

    arr = np.asarray(py, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    if arr.ndim == 2 and arr.shape[0] >= 1:
        labels = names or {}
        for row in range(arr.shape[0]):
            label = labels.get(row, f"class {row}")
            ax.plot(np.asarray(px), arr[row], linewidth=1.8, label=str(label))
    else:
        ax.plot(np.asarray(px), arr, linewidth=1.8, label="all")
    ax.set_xlabel(_t(cjk, "召回率 Recall", "Recall"))
    ax.set_ylabel(_t(cjk, "精确率 Precision", "Precision"))
    ax.set_title(_t(cjk, "测试集 PR 曲线（IoU=0.5）", "PR curve on test split (IoU=0.5)"))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metrics_bar(per_class: Sequence[ClassMetrics], figures_dir: Path | None = None) -> Path | None:
    """画“全类 vs smoke vs fire”的四项指标柱状图。"""
    if not per_class:
        return None
    plt, cjk = _setup_matplotlib()
    out_dir = Path(figures_dir) if figures_dir else paths.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_by_class.png"

    labels = [c.class_name for c in per_class]
    metrics = [
        ("Precision", [c.precision for c in per_class]),
        ("Recall", [c.recall for c in per_class]),
        ("mAP50", [c.map50 for c in per_class]),
        ("mAP50-95", [c.map50_95 for c in per_class]),
    ]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    for i, (name, values) in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(_t(cjk, "指标值", "score"))
    ax.set_title(_t(cjk, "测试集分类别指标", "Per-class metrics on test split"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_error_montage(
    image_paths: Sequence[str | Path],
    out_path: Path,
    title: str = "",
    max_items: int = 6,
    cols: int = 3,
) -> Path | None:
    """把若干错误样本图拼成一张总览图（便于论文插图）。"""
    picked = [Path(p) for p in image_paths[:max_items]]
    if not picked:
        return None
    plt, cjk = _setup_matplotlib()
    rows = math.ceil(len(picked) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), dpi=120)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.axis("off")
    for ax, path in zip(axes_arr, picked):
        img = safe_imread(path, flags=1)
        ax.imshow(img[:, :, ::-1])
        ax.set_title(path.stem.replace("__missed", "").replace("__false_alarm", ""), fontsize=7)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def evaluate(
    weights: str | Path | None = None,
    split: str = "test",
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str | int | None = None,
    error_conf: float = ERROR_CONF,
    with_error_cases: bool = True,
    max_error_samples: int = 12,
    error_limit: int | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """完整评价：val 指标 → 分类别 → 图表 → 错误案例 → 返回结构化结果。"""
    wpath = resolve_weights(weights)
    started = _now()
    t_all = time.perf_counter()

    dataset_info = count_test_objects(split)
    val = run_validation(wpath, split=split, imgsz=imgsz, batch=batch, device=device)
    box = val["box"]

    figs_dir = Path(figures_dir) if figures_dir else paths.FIGURES_DIR / f"eval_{_stamp()}"
    figs_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for png in map(Path, val["val_figures"]):
        if png.is_file():
            target = figs_dir / png.name
            shutil.copy2(png, target)
            copied.append(str(target))

    pr_path = plot_pr_curve(box, figures_dir=figs_dir, names=val["names"])
    if pr_path:
        copied.append(str(pr_path))
    bar_path = plot_metrics_bar(val["per_class"], figures_dir=figs_dir)
    if bar_path:
        copied.append(str(bar_path))

    error_payload: dict[str, Any] | None = None
    montages: list[str] = []
    if with_error_cases:
        error_payload = collect_error_cases(
            wpath,
            split=split,
            conf=error_conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
            max_samples=max_error_samples,
            out_dir=paths.PREDICTIONS_DIR / f"error_cases_{_stamp()}",
            limit=error_limit,
        )
        stats = error_payload["stats"]
        fn_montage = build_error_montage(
            stats["saved_fn"], figs_dir / "error_cases_missed.png", title="漏检样本（FN）"
        )
        fp_montage = build_error_montage(
            stats["saved_fp"], figs_dir / "error_cases_false_alarm.png", title="误报样本（FP）"
        )
        montages = [str(p) for p in (fn_montage, fp_montage) if p]

    overall_p, overall_r = box.mp, box.mr
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "阶段八：模型评价与实验对比",
        "generated_at": _now(),
        "started_at": started,
        "seconds_total": round(time.perf_counter() - t_all, 2),
        "weights": weights_info(wpath),
        "dataset": {
            "name": "D-Fire（Kaggle 镜像 sayedgamal99/smoke-fire-detection-yolo）",
            "split": split,
            "images": dataset_info["images"],
            "objects": dataset_info["objects"],
            "empty_label_images": dataset_info["empty_labels"],
            "objects_per_class": dataset_info["per_class"],
            "split_method": "按近似分组 70/15/15（seed=42）；原始视频 ID 已被抹除，分组为近似分组",
        },
        "config": {
            "imgsz": imgsz,
            "batch": batch,
            "device": str(device) if device is not None else "auto",
            "map_conf": MAP_CONF,
            "map_iou": MAP_IOU,
            "error_conf": error_conf,
            "match_iou": MATCH_IOU,
            "workers": 0,
        },
        "overall": {
            "precision": _round(overall_p),
            "recall": _round(overall_r),
            "map50": _round(box.map50),
            "map50_95": _round(box.map),
            "f1": _round(
                2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
            ),
        },
        "per_class": [
            {
                "class_id": c.class_id,
                "class_name": c.class_name,
                "instances": c.instances,
                "precision": c.precision,
                "recall": c.recall,
                "map50": c.map50,
                "map50_95": c.map50_95,
            }
            for c in val["per_class"]
        ],
        "speed_ms_per_image": val["speed"],
        "val_seconds": val["seconds"],
        "error_cases": error_payload["stats"] if error_payload else None,
        "error_case_details": (
            {
                "dir": error_payload["out_dir"],
                "missed_examples": error_payload["missed_examples"],
                "false_alarm_examples": error_payload["false_alarm_examples"],
            }
            if error_payload
            else None
        ),
        "figures": copied + montages,
        "limitations": [
            "数据划分是按感知哈希近似分组，不是真实视频/事件分组，指标可能偏乐观。",
            "跨集合仍存在少量近似重复图片（阶段五统计：19 组），会抬高测试集指标。",
            "数据集不是纯森林场景（含城市、工业、夜间监控画面），不能宣称“森林专用模型”。",
            "test 集只在本次评价中使用一次；本文件中所有指标都来自这一次运行。",
            "错误案例分析使用的是 conf=0.25 的固定阈值，mAP 使用的是 conf=0.001 积分，两者不可混用。",
            "test 集标注共 3,878 条，其中含极少量退化/越界框（全数据集共 26 条），本次未逐条剔除。",
        ],
        "disclaimer": DISCLAIMER,
    }
    return result


def save_metrics(result: dict[str, Any], out_dir: Path | None = None) -> tuple[Path, Path]:
    """把评价结果写成 ``metrics.json`` 与 ``metrics.md``。"""
    root = Path(out_dir) if out_dir else paths.REPORTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "metrics.json"
    md_path = root / "metrics.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    """生成人读的 ``metrics.md``。"""
    w = result["weights"]
    ds = result["dataset"]
    cfg = result["config"]
    ov = result["overall"]
    err = result.get("error_cases")

    lines: list[str] = []
    lines.append("# 测试集评价报告（阶段八）")
    lines.append("")
    lines.append(f"- 生成时间：{result['generated_at']}")
    lines.append(f"- 权重：`{w['name']}`（{w['bytes']:,} B，sha256 `{w['sha256'][:16]}…`）")
    lines.append(f"- 数据：{ds['name']}；**{ds['split']} 集**")
    lines.append(f"- 测试集规模：**{ds['images']:,} 张图片 / {ds['objects']:,} 个标注框**（空标签图片 {ds['empty_label_images']:,} 张）")
    lines.append(f"- 各类实例数：" + "，".join(f"{k} {v:,}" for k, v in ds["objects_per_class"].items()))
    lines.append(f"- 评价参数：imgsz={cfg['imgsz']}，batch={cfg['batch']}，device={cfg['device']}，"
                 f"mAP 用 conf={cfg['map_conf']} / iou={cfg['map_iou']}")
    lines.append("")
    lines.append("## 一、总体指标（test 集，一次性最终评价）")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| Precision | **{_fmt(ov['precision'])}** |")
    lines.append(f"| Recall | **{_fmt(ov['recall'])}** |")
    lines.append(f"| mAP50 | **{_fmt(ov['map50'])}** |")
    lines.append(f"| mAP50-95 | **{_fmt(ov['map50_95'])}** |")
    lines.append(f"| F1（由 P/R 计算） | {_fmt(ov['f1'])} |")
    lines.append("")
    lines.append("## 二、分类别指标")
    lines.append("")
    lines.append("| 类别 | 实例数 | Precision | Recall | mAP50 | mAP50-95 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for c in result["per_class"]:
        lines.append(
            f"| {c['class_name']} | {c['instances']:,} | {_fmt(c['precision'])} | "
            f"{_fmt(c['recall'])} | {_fmt(c['map50'])} | {_fmt(c['map50_95'])} |"
        )
    lines.append("")
    lines.append("## 三、推理耗时（验证流程内，每张图片）")
    lines.append("")
    lines.append("| 阶段 | 毫秒/张 |")
    lines.append("| --- | --- |")
    for k, v in result["speed_ms_per_image"].items():
        lines.append(f"| {k} | {_fmt(v, 3)} |")
    lines.append("")
    lines.append(f"> 验证总耗时 {_fmt(result['val_seconds'], 1)} 秒（含模型加载与全部 {ds['images']:,} 张图片）。")
    lines.append("")
    lines.append("## 四、错误案例分析")
    lines.append("")
    if err:
        lines.append(
            f"在 **conf={err['conf_threshold']}、IoU={err['iou_threshold']}** 的固定阈值下逐图匹配"
            f"（扫描 {err['images_scanned']:,} 张）："
        )
        lines.append("")
        lines.append("| 项目 | 数量 |")
        lines.append("| --- | --- |")
        lines.append(f"| 命中（TP） | {err['true_positives']:,} |")
        lines.append(f"| 误报（FP） | {err['false_positives']:,} |")
        lines.append(f"| 漏检（FN） | {err['false_negatives']:,} |")
        lines.append(f"| 含误报的图片数 | {err['images_with_fp']:,} |")
        lines.append(f"| 含漏检的图片数 | {err['images_with_fn']:,} |")
        lines.append("")
        lines.append("**漏检按类别**：" + "，".join(f"{k} {v:,}" for k, v in (err["fn_per_class"] or {"—": 0}).items()))
        lines.append("")
        lines.append("**误报按类别**：" + "，".join(f"{k} {v:,}" for k, v in (err["fp_per_class"] or {"—": 0}).items()))
        lines.append("")
        if err.get("saved_fn"):
            lines.append(f"- 漏检样本图（{len(err['saved_fn'])} 张）：见 `figures/error_cases_missed.png` 与 `{Path(err['saved_fn'][0]).parent}`")
        if err.get("saved_fp"):
            lines.append(f"- 误报样本图（{len(err['saved_fp'])} 张）：见 `figures/error_cases_false_alarm.png` 与 `{Path(err['saved_fp'][0]).parent}`")
        lines.append("")
    else:
        lines.append("本次运行使用 `--no-error-cases`，未收集错误案例（**未完成**，不是“没有错误”）。")
        lines.append("")

    det = result.get("error_case_details") or {}
    if det.get("missed_examples"):
        lines.append("### 漏检样例（按标注框面积从大到小）")
        lines.append("")
        lines.append("| 图片 | 类别 | 面积占比 | 该图预测框数 |")
        lines.append("| --- | --- | --- | --- |")
        for row in det["missed_examples"][:8]:
            lines.append(
                f"| {row['image']} | {row['class_name']} | {row['area_norm']:.4f} | {row['predictions_in_image']} |"
            )
        lines.append("")
    if det.get("false_alarm_examples"):
        lines.append("### 误报样例（按置信度从高到低）")
        lines.append("")
        lines.append("| 图片 | 类别 | 置信度 | 与最近标注框 IoU |")
        lines.append("| --- | --- | --- | --- |")
        for row in det["false_alarm_examples"][:8]:
            lines.append(
                f"| {row['image']} | {row['class_name']} | {row['confidence']:.3f} | {row['best_iou_with_gt']:.3f} |"
            )
        lines.append("")

    lines.append("## 五、实验完成情况（如实标注）")
    lines.append("")
    lines.append("| 实验 | 状态 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 实验一：基础模型（10 epoch，yolov8n） | ✅ 已完成 | 上表全部指标即本实验在 test 集的结果 |")
    lines.append("| 实验二：参数对比（输入尺寸 512 vs 640） | ⬜ **未完成** | 需要重新训练，受本机 6 GB 显存与时间限制，尚未执行；本文件不含任何对比数字 |")
    lines.append("| 实验三：复杂场景分组分析 | ⬜ **未完成** | 数据集没有“清晰/远距离/云雾/复杂背景”这类场景标签，需要人工分组；本文件只做通用误报/漏检统计 |")
    lines.append("")
    lines.append("## 六、图表")
    lines.append("")
    for fig in result.get("figures", []):
        lines.append(f"- `{fig}`")
    lines.append("")
    lines.append("## 七、局限与偏差（必须写进论文）")
    lines.append("")
    for item in result.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 八、声明")
    lines.append("")
    lines.append(f"> {result.get('disclaimer', DISCLAIMER)}")
    lines.append("")
    return "\n".join(lines)


def rebuild_reports(reports_dir: Path | None = None) -> tuple[Path, Path]:
    """读取已有的 ``metrics.json``，仅重写 ``metrics.md``（不重跑模型）。"""
    root = Path(reports_dir) if reports_dir else paths.REPORTS_DIR
    json_path = root / "metrics.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到 {json_path}，请先跑一次 evaluate.py")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    md_path = root / "metrics.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    """模块自检入口（打印关键路径与可用权重，不跑评价）。"""
    print("[评价模块] 阶段八已实现")
    print(f"   权重目录 : {paths.MODELS_DIR}")
    print(f"   报告目录 : {paths.REPORTS_DIR}")
    print(f"   图表目录 : {paths.FIGURES_DIR}")
    available = list_trained_weights()
    print(f"   可用权重 : {len(available)} 个")
    for w in available[:5]:
        print(f"     - {w.name} ({w.stat().st_size:,} B)")
    print(f"   mAP 阈值 : conf={MAP_CONF} iou={MAP_IOU}；错误案例阈值 conf={ERROR_CONF}，匹配 IoU={MATCH_IOU}")
    print(f"   {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
