"""推理封装（阶段七实现）：加载训练好的 YOLO 权重，对图片做 smoke/fire 检测。

设计要点
--------
1. **权重来源必须可追溯**：``resolve_weights_path`` 只接受显式路径或 ``models/`` 下
   **本项目训练出来的** 权重（``*_best.pt`` / ``*_last.pt``）；自动挑选时**故意排除**
   ``yolov8n.pt`` 这类 COCO 预训练权重，因为它没有 smoke/fire 类别，
   拿它出图就是伪造检测结果。
2. **图片读写走项目自己的兼容层**（``src.image_io.safe_imread`` / ``safe_imwrite``），
   不使用 ``cv2.imread``/``cv2.imwrite``，避免中文或空格路径下静默失败。
3. **模型只加载一次**：``SmokeDetector`` 实例化一次可反复推理，供 ``predict.py``
   与 Streamlit 页面共用，不在页面里重复写模型代码。
4. **无目标图片是正常情况**：返回空的 ``detections``，照样保存图片、照样出现在
   JSON/CSV 里，并计入"无目标图片数"，不能报错退出。
5. 所有时间、阈值、权重哈希都写进结构化结果，作为论文"推理耗时"一节的真实数据来源。

典型用法::

    from src.inference import SmokeDetector, write_json_report, write_csv_report

    detector = SmokeDetector(weights=None, conf=0.25)      # 自动挑选最新 *_best.pt
    result = detector.predict_image("demo.jpg")
    print(result.num_boxes, result.inference_ms)
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from src import paths
from src.image_io import ALLOWED_IMAGE_SUFFIXES, as_predict_source, safe_imwrite

#: 本项目的数据集类别（与 configs/forest_smoke.yaml 保持一致）
CLASS_NAMES: dict[int, str] = {0: "smoke", 1: "fire"}
SMOKE_CLASS_NAME = "smoke"

#: 默认推理参数（与训练时的 imgsz 一致）
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

#: COCO 预训练权重的文件名前缀，自动挑选权重时必须排除
_COCO_WEIGHT_PATTERNS: tuple[str, ...] = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")

#: 本项目训练产出的权重文件名后缀
_TRAINED_WEIGHT_SUFFIXES: tuple[str, ...] = ("_best.pt", "_last.pt")

DISCLAIMER = "模型识别结果仅供辅助分析，需人工核实；本系统不做森林火灾预警，不能仅凭检测结果确认真实火灾。"


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """单个检测框。"""

    class_id: int
    class_name: str
    confidence: float
    #: 左上/右下角像素坐标 (x1, y1, x2, y2)
    box_xyxy: tuple[float, float, float, float]
    #: 归一化中心点宽高 (cx, cy, w, h)，与 YOLO 标签格式一致，便于人工核对
    box_xywh_norm: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ImageResult:
    """一张图片的推理结果。"""

    image_path: str
    detections: list[Detection] = field(default_factory=list)
    #: 端到端耗时（读图 + 推理 + 后处理），单位毫秒
    inference_ms: float | None = None
    image_name: str = ""
    image_width: int | None = None
    image_height: int | None = None
    #: 读图与解码耗时（毫秒）
    read_ms: float | None = None
    #: 纯预测耗时（毫秒，不含读图、画图、写盘）
    predict_ms: float | None = None
    #: 画框与写盘耗时（毫秒）
    render_ms: float | None = None
    #: 端到端总耗时（毫秒）
    total_ms: float | None = None
    #: 保存下来的可视化图片（未保存时为 None）
    saved_image: str | None = None
    #: Ultralytics 自报的分阶段耗时（preprocess / inference / postprocess）
    speed_ms: dict[str, float] = field(default_factory=dict)
    #: 该图片处理失败时的错误信息（此时 detections 为空）
    error: str | None = None

    @property
    def num_boxes(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "image_name": self.image_name,
            "num_boxes": self.num_boxes,
            "read_ms": self.read_ms,
            "predict_ms": self.predict_ms,
            "render_ms": self.render_ms,
            "total_ms": self.total_ms,
            "inference_ms": self.inference_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "saved_image": self.saved_image,
            "speed_ms": self.speed_ms,
            "error": self.error,
            "detections": [d.to_dict() for d in self.detections],
        }


# --------------------------------------------------------------------------
# 权重解析
# --------------------------------------------------------------------------
def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_trained_weights() -> list[Path]:
    """列出 ``models/`` 下本项目训练出来的权重（排除 COCO 预训练权重）。"""
    if not paths.MODELS_DIR.is_dir():
        return []
    found: list[Path] = []
    for item in paths.MODELS_DIR.glob("*.pt"):
        name = item.name.lower()
        if name in _COCO_WEIGHT_PATTERNS:
            continue
        if name.endswith(_TRAINED_WEIGHT_SUFFIXES):
            found.append(item)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_weights_path(weights: str | Path | None = None) -> Path:
    """解析并校验权重路径。

    Parameters
    ----------
    weights:
        显式权重路径（可为相对项目根的路径）；``None`` 时自动挑选
        ``models/`` 下最新的 ``*_best.pt``，其次最新的 ``*_last.pt``。

    Raises
    ------
    FileNotFoundError
        文件不存在，或没有任何本项目训练出的权重（消息里给出下一步该怎么做）。
    """
    if weights is not None:
        candidate = Path(weights)
        if not candidate.is_absolute():
            candidate = paths.PROJECT_ROOT / candidate
        if not candidate.is_file():
            hint = ""
            trained = _list_trained_weights()
            if trained:
                hint = "\n可用的本项目权重：\n" + "\n".join(f"  - {p}" for p in trained[:5])
            raise FileNotFoundError(
                f"权重文件不存在: {candidate}\n"
                f"请先用 train.py 训练，或用 --weights 指定已有权重。{hint}"
            )
        return candidate

    trained = _list_trained_weights()
    if not trained:
        raise FileNotFoundError(
            f"在 {paths.MODELS_DIR} 下没有找到本项目训练出的权重（*_best.pt / *_last.pt）。\n"
            "下一步：\n"
            "  1) 先训练一个模型（分批训练，示例）：\n"
            "     .\\.venv\\Scripts\\python.exe train.py --full --epochs 10 --workers 1 --batch 8 --no-downgrade --name batch1_e10\n"
            "  2) 或者用 --weights 显式指定已有的 .pt 权重路径。\n"
            "注意：yolov8n.pt 是 COCO 预训练权重，不含 smoke/fire 类别，不能当作烟雾模型使用。"
        )
    # 优先 best，其次 last，同类按修改时间取最新
    best = [p for p in trained if p.name.lower().endswith("_best.pt")]
    return (best or trained)[0]


def describe_weights(weights_path: Path, compute_sha256: bool = True) -> dict[str, Any]:
    """收集权重文件的可追溯信息。"""
    stat = weights_path.stat()
    info: dict[str, Any] = {
        "path": str(weights_path),
        "name": weights_path.name,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }
    if compute_sha256:
        info["sha256"] = _sha256_of(weights_path)
    return info


# --------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------
class SmokeDetector:
    """烟雾检测器封装：模型只加载一次，供脚本与 Streamlit 复用。"""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | int | None = None,
        verbose: bool = False,
        warmup: bool = True,
    ) -> None:
        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError(f"conf 必须在 0–1 之间，当前: {conf}")
        if not 0.0 <= float(iou) <= 1.0:
            raise ValueError(f"iou 必须在 0–1 之间，当前: {iou}")
        if int(imgsz) < 32:
            raise ValueError(f"imgsz 至少为 32，当前: {imgsz}")

        self.weights_path: Path = resolve_weights_path(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = None if device is None else str(device)
        self.verbose = bool(verbose)

        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 依赖缺失分支
            raise RuntimeError(
                "缺少依赖 ultralytics。请在项目根目录执行：\n"
                "  .\\.venv\\Scripts\\Activate.ps1\n"
                "  python -m pip install \"ultralytics==8.4.190\"\n"
                "（完整步骤见 docs/compatibility_final.md 第六节）"
            ) from exc

        self.load_seconds: float = 0.0
        t0 = time.perf_counter()
        self.model = YOLO(str(self.weights_path))
        self.load_seconds = round(time.perf_counter() - t0, 4)

        names = getattr(self.model, "names", {}) or {}
        self.class_names: dict[int, str] = {int(k): str(v) for k, v in dict(names).items()}
        if not self.class_names:
            self.class_names = dict(CLASS_NAMES)

        #: 首次推理会包含 CUDA 上下文初始化等一次性开销（实测可达 3 秒），
        #: 先跑一次空白图把它排除，否则单张图片的"耗时"会被严重夸大。
        self.warmup_seconds: float | None = None
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """用一张空白图做一次推理，排除首次调用的初始化开销。"""
        import numpy as np  # noqa: PLC0415

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self.model.predict(
            source=dummy,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        self.warmup_seconds = round(time.perf_counter() - t0, 3)

    # -- 元信息 ---------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def describe(self, compute_sha256: bool = True) -> dict[str, Any]:
        """返回本次推理使用的模型与参数信息（写进结果 JSON，便于复现）。"""
        return {
            "weights": describe_weights(self.weights_path, compute_sha256=compute_sha256),
            "classes": {str(k): v for k, v in sorted(self.class_names.items())},
            "params": {
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
                "device": self.device or "auto",
            },
            "load_seconds": self.load_seconds,
            "warmup_seconds": self.warmup_seconds,
            "timing_note": "inference_ms 为单张端到端耗时（读图+推理+后处理），已排除模型加载与首次预热开销",
        }

    def check_class_names(self) -> list[str]:
        """返回与预期类别不一致的警告文本（空列表表示一致）。"""
        warnings: list[str] = []
        if set(self.class_names.values()) != set(CLASS_NAMES.values()):
            warnings.append(
                f"权重里的类别 {sorted(self.class_names.values())} 与项目约定 "
                f"{sorted(CLASS_NAMES.values())} 不一致，结果类别名以权重为准。"
            )
        return warnings

    # -- 推理 -----------------------------------------------------------
    def predict_image(self, image_path: str | Path, save_image: Path | None = None) -> ImageResult:
        """对单张图片推理。

        Parameters
        ----------
        image_path:
            待检测图片。
        save_image:
            可视化结果图的输出路径；``None`` 表示不保存。

        Raises
        ------
        FileNotFoundError / ValueError
            图片不存在、格式不支持或解码失败（消息里会指明具体文件）。
        """
        p = Path(image_path)
        result = ImageResult(image_path=str(p), image_name=p.name)
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        source = as_predict_source(p)  # 走 safe_imread，不碰 cv2.imread
        result.image_height, result.image_width = int(source.shape[0]), int(source.shape[1])
        result.read_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        t1 = time.perf_counter()
        predictions = self.model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=self.verbose,
        )
        result.predict_ms = round((time.perf_counter() - t1) * 1000.0, 2)
        result.inference_ms = result.predict_ms
        if len(predictions) != 1:
            raise RuntimeError(f"单图推理返回了 {len(predictions)} 个结果，预期 1 个（{p}）")
        pred = predictions[0]

        height, width = result.image_height or 1, result.image_width or 1
        boxes = getattr(pred, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, classes):
                cid = int(cls_id)
                w = max(float(x2) - float(x1), 0.0)
                h = max(float(y2) - float(y1), 0.0)
                result.detections.append(
                    Detection(
                        class_id=cid,
                        class_name=self.class_names.get(cid, f"class{cid}"),
                        confidence=round(float(score), 6),
                        box_xyxy=(round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)),
                        box_xywh_norm=(
                            round((float(x1) + float(x2)) / 2.0 / width, 6),
                            round((float(y1) + float(y2)) / 2.0 / height, 6),
                            round(w / width, 6),
                            round(h / height, 6),
                        ),
                    )
                )

        speed = getattr(pred, "speed", None) or {}
        result.speed_ms = {str(k): round(float(v), 3) for k, v in dict(speed).items()}

        if save_image is not None:
            t2 = time.perf_counter()
            annotated = pred.plot()  # BGR ndarray
            result.saved_image = str(safe_imwrite(save_image, annotated))
            result.render_ms = round((time.perf_counter() - t2) * 1000.0, 2)

        result.total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return result

    def predict_dir(
        self,
        image_dir: str | Path,
        recursive: bool = False,
        limit: int | None = None,
        save_dir: Path | None = None,
        on_error: str = "record",
    ) -> list[ImageResult]:
        """对目录下的图片批量推理。

        Parameters
        ----------
        image_dir:
            图片目录。
        recursive:
            是否递归子目录。
        limit:
            最多处理多少张（``None`` 表示全部）。
        save_dir:
            可视化结果图输出目录；``None`` 表示不保存。
        on_error:
            ``"record"``（默认）把单图错误记录到结果里继续；``"raise"`` 直接抛出。
        """
        directory = Path(image_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"图片目录不存在或不是目录: {directory}")

        images = collect_images(directory, recursive=recursive)
        if limit is not None:
            images = images[: max(int(limit), 0)]
        if not images:
            raise FileNotFoundError(f"目录里没有可识别的图片（支持 {sorted(ALLOWED_IMAGE_SUFFIXES)}）: {directory}")

        results: list[ImageResult] = []
        for index, image in enumerate(images, start=1):
            target = None if save_dir is None else Path(save_dir) / f"{image.stem}__pred.jpg"
            try:
                result = self.predict_image(image, save_image=target)
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                result = ImageResult(image_path=str(image), image_name=image.name, error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            if self.verbose:
                print(f"[{index}/{len(images)}] {image.name} -> {result.num_boxes} 个目标")
        return results


# --------------------------------------------------------------------------
# 目录扫描
# --------------------------------------------------------------------------
def collect_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """收集目录下的图片（按文件名排序，保证结果可复现）。"""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"图片目录不存在或不是目录: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES]
    return sorted(images, key=lambda p: str(p).lower())


# --------------------------------------------------------------------------
# 结果汇总与输出
# --------------------------------------------------------------------------
def summarize(results: Sequence[ImageResult]) -> dict[str, Any]:
    """汇总一批推理结果（无目标图片也计入）。"""

    def _mean(values: Sequence[float | None]) -> float | None:
        picked = [v for v in values if v is not None]
        return round(sum(picked) / len(picked), 2) if picked else None

    total_boxes = sum(r.num_boxes for r in results)
    per_class: dict[str, int] = {}
    for r in results:
        for name, count in r.class_counts().items():
            per_class[name] = per_class.get(name, 0) + count
    predict_times = [r.predict_ms for r in results]
    return {
        "num_images": len(results),
        "num_images_with_detections": sum(1 for r in results if r.num_boxes > 0),
        "num_images_without_detections": sum(1 for r in results if r.num_boxes == 0),
        "num_images_failed": sum(1 for r in results if r.error),
        "num_boxes": total_boxes,
        "boxes_per_class": dict(sorted(per_class.items())),
        "predict_ms_mean": _mean(predict_times),
        "predict_ms_total": round(sum(v for v in predict_times if v is not None), 2) if any(predict_times) else None,
        "read_ms_mean": _mean([r.read_ms for r in results]),
        "render_ms_mean": _mean([r.render_ms for r in results]),
        "total_ms_mean": _mean([r.total_ms for r in results]),
        "inference_ms_mean": _mean([r.inference_ms for r in results]),
    }


def build_payload(
    detector: SmokeDetector,
    results: Sequence[ImageResult],
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装写入 JSON 的完整结果。"""
    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "model": detector.describe(),
        "summary": summarize(results),
        "images": [r.to_dict() for r in results],
        "disclaimer": DISCLAIMER,
    }
    if extra:
        payload.update(extra)
    return payload


CSV_COLUMNS: tuple[str, ...] = (
    "image_path",
    "image_name",
    "num_boxes",
    "read_ms",
    "predict_ms",
    "render_ms",
    "total_ms",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "w_norm",
    "h_norm",
)


def write_csv_report(path: str | Path, results: Sequence[ImageResult]) -> Path:
    """写出 CSV（每个检测框一行；无目标的图片写一行空检测，便于统计）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for result in results:
            base = {
                "image_path": result.image_path,
                "image_name": result.image_name,
                "num_boxes": result.num_boxes,
                "read_ms": result.read_ms,
                "predict_ms": result.predict_ms,
                "render_ms": result.render_ms,
                "total_ms": result.total_ms,
            }
            if not result.detections:
                writer.writerow(base)
                continue
            for det in result.detections:
                row = dict(base)
                x1, y1, x2, y2 = det.box_xyxy
                norm = det.box_xywh_norm or (None, None, None, None)
                row.update(
                    {
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "w_norm": norm[2],
                        "h_norm": norm[3],
                    }
                )
                writer.writerow(row)
    return target


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """写出 JSON（UTF-8，不转义中文）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def format_summary_text(results: Sequence[ImageResult]) -> str:
    """把汇总结果渲染成终端可读的中文摘要。"""
    s = summarize(results)
    lines = [
        f"图片 {s['num_images']} 张：有目标 {s['num_images_with_detections']} 张 / "
        f"无目标 {s['num_images_without_detections']} 张 / 处理失败 {s['num_images_failed']} 张",
        f"检测框合计 {s['num_boxes']} 个" + (f"（{', '.join(f'{k} {v}' for k, v in s['boxes_per_class'].items())}）" if s["boxes_per_class"] else ""),
        f"耗时（每张平均）：纯预测 {s['predict_ms_mean']} ms，读图 {s['read_ms_mean']} ms，"
        f"画图写盘 {s['render_ms_mean']} ms，端到端 {s['total_ms_mean']} ms",
    ]
    return "\n".join(lines)


def iter_detections(results: Iterable[ImageResult]) -> Iterable[tuple[ImageResult, Detection]]:
    """遍历 (图片结果, 检测框) 组合，便于统计与作图。"""
    for result in results:
        for det in result.detections:
            yield result, det


def main(argv: list[str] | None = None) -> int:
    """模块自检入口：不加载模型，只打印路径与可用权重。"""
    print("src.inference 自检（不加载模型）")
    print(f"  项目根      : {paths.PROJECT_ROOT}")
    print(f"  权重目录    : {paths.MODELS_DIR}")
    print(f"  结果图目录  : {paths.PREDICTIONS_DIR}")
    trained = _list_trained_weights()
    print(f"  可用本项目权重: {len(trained)} 个")
    for item in trained[:5]:
        print(f"    - {item.name} ({item.stat().st_size:,} B)")
    try:
        print(f"  自动选择的权重: {resolve_weights_path(None)}")
    except FileNotFoundError as exc:
        print(f"  [提示] {exc}")
    print(f"  类别约定    : {CLASS_NAMES}")
    print("  命令行入口请用 predict.py（见 README 第八节）")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
