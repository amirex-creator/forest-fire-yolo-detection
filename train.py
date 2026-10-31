"""YOLO 森林烟雾检测 —— 训练入口（阶段六）。

功能
----
用 ``yolov8n.pt`` 预训练权重做**迁移学习**（单阶段 YOLOv8 检测头微调），
把日志、权重、训练曲线全部保存到项目内，并把真实运行信息写成报告。

设计要点
--------
1. 参数集中来自 ``configs/train.yaml``，命令行可覆盖；数据集信息读
   ``configs/forest_smoke.yaml``（``nc=2`` / ``names=[smoke, fire]``）。
2. 所有路径经 :mod:`src.paths` 用 ``pathlib`` 解析，代码里不写盘符路径。
3. 训练前先校验数据集（目录 / 配对 / 标签抽样；``--deep-check`` 用阶段五工具全量扫描）。
4. 设备选择：优先 GPU(0)，做一次真实矩阵乘法确认可用；失败则回退 CPU 并在报告中注明。
5. 显存不足（OOM）时按 8 -> 4 -> 2 自动降级重试；全部失败给出明确报错与建议。
6. 每次运行使用**新的**时间戳目录，绝不删除或覆盖历史实验输出。
7. 训练过程采样内存 / 显存占用，训练结束把峰值写进报告。
8. 训练结束把 ``best.pt`` / ``last.pt`` 复制到 ``models/``，把曲线复制到
   ``outputs/figures/<run>/``，并生成 ``outputs/reports/train_<run>.json/.md``。

用法示例
--------
::

    # 1) 只做数据与环境校验，不训练（最快，用来确认参数对不对）
    .\\.venv\\Scripts\\python.exe train.py --smoke-test --dry-run

    # 2) 小规模试跑：2 轮 + 10% 数据 + 全量数据校验（阶段六第一步）
    .\\.venv\\Scripts\\python.exe train.py --smoke-test --deep-check

    # 3) 试跑通过后，正式训练 50 轮（读 configs/train.yaml 的 epochs）
    .\\.venv\\Scripts\\python.exe train.py --full

    # 4) 临时改参数（不改配置文件）
    .\\.venv\\Scripts\\python.exe train.py --full --epochs 100 --batch 4 --device cpu

    # 5) 只校验数据集，不训练
    .\\.venv\\Scripts\\python.exe train.py --check-data-only

退出码
------
``0`` 成功 / ``1`` 参数或配置错误 / ``2`` 数据集校验未通过 /
``3`` 依赖缺失 / ``4`` 显存不足（降级后仍失败）/ ``5`` 训练过程异常。

免责声明：训练得到的模型只做图像层面的烟雾 / 火焰识别辅助分析，
不能作为森林火灾预警依据，任何结果都需人工核实。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLANNED_STAGE = "阶段六"
DISCLAIMER = (
    "模型的识别结果仅供辅助分析，需人工核实；"
    "本系统不做森林火灾预警，也不能仅凭检测结果确认真实火灾。"
)

# OOM 自动降级顺序：优先用配置里的 batch，失败后依次尝试更小的批大小
OOM_FALLBACK_BATCHES: tuple[int, ...] = (4, 2)

# 训练结束后需要复制的曲线文件（存在才复制，不报错）
CURVE_FILES: tuple[str, ...] = (
    "results.png",
    "results.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "PR_curve.png",
    "train_batch0.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _rel(path: Path) -> str:
    """把绝对路径转成相对项目根的字符串（跨盘时退化为绝对路径）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except (ValueError, OSError):
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # PyYAML 是 ultralytics 的依赖，装好环境后必然存在

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容不是字典结构: {path}")
    return data


class Tee:
    """把 stdout 同时写到控制台与日志文件（不依赖 subprocess 管道）。"""

    def __init__(self, stream: Any, log_path: Path) -> None:
        self.stream = stream
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = log_path.open("a", encoding="utf-8", newline="")

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 资源监控
# ---------------------------------------------------------------------------
@dataclass
class ResourceSample:
    t: float
    rss_mb: float | None
    ram_available_mb: float | None
    vram_allocated_mb: float | None
    vram_reserved_mb: float | None


@dataclass
class ResourceMonitor:
    """后台线程，每 2 秒采一次内存 / 显存，只记录峰值。"""

    interval: float = 2.0
    samples: list[ResourceSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def _snapshot(self) -> ResourceSample:
        rss_mb = ram_avail_mb = alloc_mb = resv_mb = None
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            ram_avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:  # psutil 缺失或受限，忽略
            pass
        try:
            import torch

            if torch.cuda.is_available():
                alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                resv_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass
        return ResourceSample(time.time(), rss_mb, ram_avail_mb, alloc_mb, resv_mb)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._snapshot())
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples.append(self._snapshot())
        self._thread = threading.Thread(target=self._run, name="res-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.samples.append(self._snapshot())

    def summary(self) -> dict[str, Any]:
        def peak(values: list[float | None]) -> float | None:
            real = [v for v in values if v is not None]
            return round(max(real), 1) if real else None

        def low(values: list[float | None]) -> float | None:
            real = [v for v in values if v is not None]
            return round(min(real), 1) if real else None

        return {
            "samples": len(self.samples),
            "interval_s": self.interval,
            "process_rss_peak_mb": peak([s.rss_mb for s in self.samples]),
            "system_ram_available_low_mb": low([s.ram_available_mb for s in self.samples]),
            "cuda_allocated_peak_mb": peak([s.vram_allocated_mb for s in self.samples]),
            "cuda_reserved_peak_mb": peak([s.vram_reserved_mb for s in self.samples]),
        }



# ---------------------------------------------------------------------------
# 图片哈希快照（训练前后对比，追踪被 Ultralytics 就地改写/修复的图片）
# ---------------------------------------------------------------------------
SNAPSHOT_SCHEMA = 1
TOTAL_EPOCH_BUDGET_DEFAULT = 30
LEDGER_NAME = "training_ledger.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_suffixes() -> frozenset[str]:
    from src.dataset_utils import IMAGE_SUFFIXES

    return IMAGE_SUFFIXES


def snapshot_images(out_path: Path) -> dict[str, Any]:
    """对 train/val/test 的图片做 sha256 快照。**只读取**，不改动任何图片。"""
    from src import paths

    suffixes = _image_suffixes()
    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    t0 = time.time()
    for split in paths.SPLITS:
        image_dir = paths.IMAGES_DIR / split
        if not image_dir.is_dir():
            continue
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            stat = path.stat()
            total_bytes += stat.st_size
            files[f"{split}/{path.name}"] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "created_at": _now_local(),
        "images": len(files),
        "total_bytes": total_bytes,
        "seconds": round(time.time() - t0, 1),
        "files": files,
    }
    _write_json(out_path, payload)
    return payload


def diff_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """对比快照与当前磁盘状态。

    先用 size/mtime 快速筛，只对"看起来变了"的文件重算 sha256 确认，
    避免对 21,527 张图全部重算哈希。
    """
    from src import paths

    suffixes = _image_suffixes()
    before = snapshot.get("files") or {}
    changed: list[dict[str, Any]] = []
    missing: list[str] = []
    for key, meta in before.items():
        split, name = key.split("/", 1)
        path = paths.IMAGES_DIR / split / name
        if not path.is_file():
            missing.append(key)
            continue
        stat = path.stat()
        if stat.st_size == meta.get("bytes") and stat.st_mtime_ns == meta.get("mtime_ns"):
            continue
        digest = _sha256_file(path)
        if digest == meta.get("sha256"):
            continue  # 只是被 touch 过，内容没变
        changed.append(
            {
                "file": key,
                "before_bytes": meta.get("bytes"),
                "after_bytes": stat.st_size,
                "after_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    added: list[str] = []
    for split in paths.SPLITS:
        image_dir = paths.IMAGES_DIR / split
        if not image_dir.is_dir():
            continue
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in suffixes:
                key = f"{split}/{path.name}"
                if key not in before:
                    added.append(key)
    return {
        "checked": len(before),
        "changed_count": len(changed),
        "changed": changed,
        "missing": missing,
        "added": added,
    }


# ---------------------------------------------------------------------------
# 训练批次台账（累计轮数上限，防止无意中超训）
# ---------------------------------------------------------------------------
def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    return entries if isinstance(entries, list) else []


def cumulative_epochs(entries: list[dict[str, Any]]) -> int:
    total = 0
    for item in entries:
        if item.get("status") == "success":
            total += int(item.get("epochs") or 0)
        else:
            total += int(item.get("completed_epochs") or 0)
    return total


def append_ledger(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    entries = read_ledger(path)
    entries.append(entry)
    _write_json(path, {"updated_at": _now_local(), "entries": entries})
    return entries


# ---------------------------------------------------------------------------
# 分类别验证（smoke / fire 各自的 P、R、mAP）
# ---------------------------------------------------------------------------
def _count_instances(split: str) -> dict[str, int]:
    from src import paths
    from src.dataset_utils import CLASS_NAMES, parse_yolo_label_line

    counts = {name: 0 for name in CLASS_NAMES.values()}
    label_dir = paths.LABELS_DIR / split
    if not label_dir.is_dir():
        return counts
    for path in label_dir.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                class_id, _ = parse_yolo_label_line(line, line_no)
            except Exception:
                continue
            name = CLASS_NAMES.get(class_id)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def per_class_validation(
    weights: Path,
    data_yaml_path: Path,
    *,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """用训练出的 best.pt 在 val/test 上跑一次 Ultralytics 验证，取到分类别指标。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
        workers=0,  # 验证不需要多进程；workers>0 会开命名管道，在受限环境里会被拒绝
    )
    box = metrics.box
    names = getattr(metrics, "names", {}) or {}
    raw_index = getattr(box, "ap_class_index", None)
    ap_index = [int(x) for x in raw_index] if raw_index is not None else []
    instances = _count_instances(split)
    per_class: dict[str, Any] = {}
    for pos, class_id in enumerate(ap_index):
        cid = int(class_id)
        name = names.get(cid) if isinstance(names, dict) else None
        name = name or str(cid)
        per_class[name] = {
            "instances": instances.get(name),
            "precision": round(float(box.p[pos]), 4) if pos < len(box.p) else None,
            "recall": round(float(box.r[pos]), 4) if pos < len(box.r) else None,
            "mAP50": round(float(box.ap50[pos]), 4) if pos < len(box.ap50) else None,
            "mAP50_95": round(float(box.ap[pos]), 4) if pos < len(box.ap) else None,
        }
    return {
        "split": split,
        "weights": _rel(weights),
        "aggregate": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "instances_by_class": instances,
        },
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 3) for k, v in (getattr(metrics, "speed", {}) or {}).items()},
    }


# ---------------------------------------------------------------------------
# 过拟合判断（只陈述可观测趋势，不下结论）
# ---------------------------------------------------------------------------
def overfitting_assessment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 3:
        return {
            "verdict": "轮数少于 3，无法判断趋势",
            "detail": f"本次只有 {len(rows)} 轮，看不出 loss 的走势是否分化。",
        }
    window = rows[-3:]
    train_box = [r.get("train/box_loss") for r in window]
    val_box = [r.get("val/box_loss") for r in window]
    train_cls = [r.get("train/cls_loss") for r in window]
    val_cls = [r.get("val/cls_loss") for r in window]
    map50 = [r.get("metrics/mAP50(B)") for r in window]
    if any(v is None for v in train_box + val_box + map50):
        return {"verdict": "数据缺失，无法判断", "detail": "results.csv 中缺少部分字段。"}

    train_down = train_box[-1] < train_box[0]
    val_up = val_box[-1] > val_box[0]
    gap = [round(v - t, 4) for t, v in zip(train_box, val_box)]
    gap_widening = gap[-1] > gap[0]
    map_last = map50[-1]
    map_prev = map50[-2] if len(map50) >= 2 else map_last
    map_rising = map_last > map_prev

    if train_down and val_up and gap_widening and not map_rising:
        verdict = "有过拟合迹象"
        detail = "训练 loss 下降而验证 loss 上升、两者差距扩大，且 mAP50 没有继续提升。"
    elif train_down and val_up and gap_widening:
        verdict = "存在轻微过拟合风险"
        detail = "训练 loss 下降、验证 loss 上升，但 mAP50 仍在提升，需要下一批继续观察。"
    elif train_down and not val_up:
        verdict = "暂无明显过拟合迹象"
        detail = "训练与验证 loss 同步下降，验证指标仍在提升。"
    else:
        verdict = "暂无明显过拟合迹象"
        detail = "训练与验证曲线方向一致。"

    return {
        "verdict": verdict,
        "detail": detail,
        "window_epochs": [r.get("epoch") for r in window],
        "train_box_loss": train_box,
        "val_box_loss": val_box,
        "train_cls_loss": train_cls,
        "val_cls_loss": val_cls,
        "val_minus_train_box_gap": gap,
        "mAP50_window": map50,
    }


# ---------------------------------------------------------------------------
# 参数装配
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from src import paths

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="YOLOv8 森林烟雾/火焰检测训练入口（迁移学习）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python train.py --smoke-test --dry-run      只校验，不训练\n"
            "  python train.py --check-data-only           只校验数据集\n"
            "  python train.py --smoke-test --deep-check   小规模试跑(2轮/10%数据)\n"
            "  python train.py --full                      正式训练(读 train.yaml)\n"
            "  python train.py --full --epochs 100 --batch 4\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="小规模试跑（用 train.yaml 的 smoke_test 段）")
    mode.add_argument("--full", action="store_true", help="正式训练（用 train.yaml 的正式参数）")
    parser.add_argument("--config", type=Path, default=paths.TRAIN_YAML, help="训练参数 yaml")
    parser.add_argument("--data-config", type=Path, default=paths.DATASET_YAML, help="数据集 yaml")
    parser.add_argument("--dry-run", action="store_true", help="只校验与打印参数，不训练")
    parser.add_argument("--check-data-only", action="store_true", help="只做数据集校验后退出")
    parser.add_argument("--deep-check", action="store_true", help="用阶段五工具全量扫描标签（约 1-2 分钟）")
    parser.add_argument("--skip-data-check", action="store_true", help="跳过训练前校验（不推荐）")
    parser.add_argument("--epochs", type=int, help="覆盖轮数")
    parser.add_argument("--batch", type=int, help="覆盖批大小")
    parser.add_argument("--imgsz", type=int, help="覆盖输入尺寸")
    parser.add_argument("--workers", type=int, help="覆盖 dataloader 进程数")
    parser.add_argument("--device", help="覆盖设备：0 或 cpu")
    parser.add_argument("--seed", type=int, help="覆盖随机种子")
    parser.add_argument("--fraction", type=float, help="覆盖训练数据比例（0-1）")
    parser.add_argument("--trial-size", type=int, default=None,
                        help="试跑用训练图片数（分层抽样，0=不限制；仅 smoke 模式生效）")
    parser.add_argument("--name", default=None, help="实验名前缀（默认读 train.yaml 的 name）")
    parser.add_argument("--no-plots", action="store_true", help="关闭 Ultralytics 的绘图（更省内存）")
    parser.add_argument("--weights", type=Path, default=None,
                        help="覆盖起始权重（续训时指向上一批的 last.pt；默认用 train.yaml 的 model）")
    parser.add_argument("--no-downgrade", action="store_true",
                        help="OOM 时立即终止，不自动把 batch 降到 4/2")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="跳过训练前的图片哈希快照（默认会做）")
    parser.add_argument("--no-per-class-val", action="store_true",
                        help="跳过训练后的分类别验证（默认会做）")
    parser.add_argument("--epoch-budget", type=int, default=TOTAL_EPOCH_BUDGET_DEFAULT,
                        help=f"累计训练轮数上限（默认 {TOTAL_EPOCH_BUDGET_DEFAULT}）")
    parser.add_argument("--allow-exceed-budget", action="store_true",
                        help="允许超过累计轮数上限（默认禁止）")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="训练批次台账路径（默认 outputs/reports/training_ledger.json）")
    return parser


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    """合并 yaml 与命令行覆盖，返回最终参数字典。"""
    cfg = _load_yaml(args.config)
    smoke_cfg = cfg.get("smoke_test") or {}

    settings: dict[str, Any] = {
        "model": cfg.get("model", "yolov8n.pt"),
        "pretrained": bool(cfg.get("pretrained", True)),
        "data_config": args.data_config,
        "epochs": int(cfg.get("epochs", 50)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "batch": int(cfg.get("batch", 8)),
        "workers": int(cfg.get("workers", 2)),
        "patience": int(cfg.get("patience", 30)),
        "seed": int(cfg.get("seed", 0)),
        "deterministic": bool(cfg.get("deterministic", True)),
        "amp": bool(cfg.get("amp", True)),
        "cache": bool(cfg.get("cache", False)),
        "device": str(cfg.get("device", 0)),
        "project": cfg.get("project", "outputs/train_runs"),
        "name": cfg.get("name", "baseline"),
        "fraction": 1.0,
        "plots": True,
        "max_images": 0,
        "mode": "full",
        "no_downgrade": bool(args.no_downgrade),
        "snapshot": not bool(args.no_snapshot),
        "per_class_val": not bool(args.no_per_class_val),
        "epoch_budget": int(args.epoch_budget),
        "allow_exceed_budget": bool(args.allow_exceed_budget),
    }
    for key in ("lr0", "lrf", "warmup_epochs", "close_mosaic"):
        if cfg.get(key) is not None:
            settings[key] = cfg[key]

    if args.smoke_test:
        settings["mode"] = "smoke"
        settings["max_images"] = int(smoke_cfg.get("max_images", 0) or 0)
        for key in ("epochs", "imgsz", "batch", "fraction"):
            if key in smoke_cfg and smoke_cfg[key] is not None:
                settings[key] = type(settings[key])(smoke_cfg[key])

    for key, value in (
        ("epochs", args.epochs),
        ("batch", args.batch),
        ("imgsz", args.imgsz),
        ("workers", args.workers),
        ("seed", args.seed),
        ("fraction", args.fraction),
    ):
        if value is not None:
            settings[key] = value
    if args.trial_size is not None:
        settings["max_images"] = int(args.trial_size)
    if args.device is not None:
        settings["device"] = str(args.device)
    if args.weights is not None:
        settings["model"] = str(args.weights)
    if args.name:
        settings["name"] = args.name
    if args.no_plots:
        settings["plots"] = False
    return settings


# ---------------------------------------------------------------------------
# 数据集校验
# ---------------------------------------------------------------------------
@dataclass
class DataCheckResult:
    ok: bool
    split_images: dict[str, int]
    split_labels: dict[str, int]
    class_names: list[str]
    blocking: int
    warnings: int
    notes: list[str]


def validate_dataset(data_yaml: dict[str, Any], *, deep: bool) -> DataCheckResult:
    """校验数据集是否可用于训练。只读。

    顺序：目录存在 -> 图片数非零 -> 图片/标签配对 -> 标签抽样解析 ->
    （deep 时）调用阶段五 ``check_dataset`` 全量扫描并检查阻塞性问题。
    """
    from src import paths
    from src.dataset_utils import CLASS_NAMES, check_dataset, parse_yolo_label_line

    notes: list[str] = []
    names = data_yaml.get("names") or []
    class_names = [str(n) for n in names] if isinstance(names, list) else [str(names)]
    if len(class_names) != int(data_yaml.get("nc", 0)):
        notes.append(f"数据集 yaml 的 nc={data_yaml.get('nc')} 与 names 数量 {len(class_names)} 不一致")
        return DataCheckResult(False, {}, {}, class_names, 1, 0, notes)
    if class_names != [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]:
        notes.append(f"数据集类别顺序 {class_names} 与代码内 {list(CLASS_NAMES.values())} 不一致")

    split_images: dict[str, int] = {}
    split_labels: dict[str, int] = {}
    blocking = 0
    for split in paths.SPLITS:
        image_dir, label_dir = paths.split_dirs(split)
        if not image_dir.is_dir():
            notes.append(f"缺少图片目录: {_rel(image_dir)}")
            blocking += 1
            continue
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        labels = [p for p in label_dir.iterdir() if p.suffix.lower() == ".txt"] if label_dir.is_dir() else []
        split_images[split] = len(images)
        split_labels[split] = len(labels)
        if not images:
            notes.append(f"{split} 集合没有任何图片: {_rel(image_dir)}")
            blocking += 1
        if split in {"train", "val"} and not images:
            notes.append(f"{split} 集合为空，无法训练/验证")

    train_spec = data_yaml.get("train")
    if train_spec:
        manifest_path = Path(str(train_spec))
        if not manifest_path.is_absolute():
            dataset_root = Path(str(data_yaml.get("path") or paths.DATA_ROOT))
            if not dataset_root.is_absolute():
                dataset_root = paths.PROJECT_ROOT / dataset_root
            manifest_path = dataset_root / manifest_path
        if manifest_path.is_file() and manifest_path.suffix.lower() == ".txt":
            manifest_images = [
                Path(line.strip())
                for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]

            def manifest_label_path(image_path: Path) -> Path:
                parts = list(image_path.parts)
                image_index = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
                if image_index is not None:
                    parts[image_index] = "labels"
                    return Path(*parts).with_suffix(".txt")
                return image_path.with_suffix(".txt")

            missing_images = [p for p in manifest_images if not p.is_file()]
            missing_manifest_labels = [p for p in manifest_images if p.is_file() and not manifest_label_path(p).is_file()]
            split_images["train"] = len(manifest_images)
            split_labels["train"] = len(manifest_images) - len(missing_manifest_labels)
            notes.append(
                f"train 清单 {manifest_path.name}：{len(manifest_images)} 张图片，"
                f"缺图片 {len(missing_images)}，缺标签 {len(missing_manifest_labels)}"
            )
            if missing_images:
                notes.append(f"train 清单图片不存在，例如：{missing_images[0]}")
                blocking += 1
            if missing_manifest_labels:
                notes.append(f"train 清单缺少标签，例如：{missing_manifest_labels[0]}")
                blocking += 1

    # 标签文件名与图片名配对抽查（每集合最多 200 张），并抽样解析 20 行
    for split in paths.SPLITS:
        image_dir, label_dir = paths.split_dirs(split)
        if not image_dir.is_dir():
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        missing = [p.name for p in images[:200] if not (label_dir / f"{p.stem}.txt").exists()]
        if missing:
            notes.append(f"{split} 有 {len(missing)} 张图缺标签（示例: {missing[0]}）")
            blocking += 1
        for label in sorted(label_dir.glob("*.txt"))[:5]:
            try:
                for line_no, line in enumerate(
                    label.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    parse_yolo_label_line(line, line_no)
            except Exception as exc:  # LabelFormatError 或读文件失败
                notes.append(f"{split} 标签解析失败 {label.name}: {exc}")
                blocking += 1
            break

    warnings = 0
    if deep:
        print("[校验] 全量扫描数据集（阶段五工具，compute_phash=False）...", flush=True)
        t0 = time.time()
        report = check_dataset(compute_phash=False)
        errors = report.errors()
        warns = report.warnings()
        blocking += len(errors)
        warnings = len(warns)
        notes.append(f"全量扫描耗时 {time.time() - t0:.1f}s，阻塞错误 {len(errors)}，警告 {len(warns)}")
        for issue in errors[:10]:
            notes.append(f"错误[{issue.code}] {issue.split}/{issue.image}:{issue.label_line} {issue.message}")
        for split_name, stats in report.splits.items():
            split_images[split_name] = stats.images
            split_labels[split_name] = stats.images
        if errors:
            notes.append("存在阻塞性标签错误，按项目规则不得开始训练")

    ok = blocking == 0
    return DataCheckResult(ok, split_images, split_labels, class_names, blocking, warnings, notes)


def build_trial_train_list(
    image_dir: Path,
    label_dir: Path,
    n_images: int,
    seed: int,
    out_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """为小规模试跑生成**分层抽样**的训练图片清单。

    为什么需要它：Ultralytics 的 ``fraction`` 是``im_files[:count]``，
    取的是**按文件名排序后的前 N 张**，不是随机抽样。本项目文件名已被重编号，
    前 1507 张里 1502 张是空标签负样本 —— 直接用它试跑会得到"模型只学会预测空白"
    的假象（第一次试跑就是这样，实测 mAP=0 且 box_loss≈0.07）。

    这里改为按"有框/无框"两层按比例随机抽样，种子固定，结果可复现。
    """
    import random

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise ValueError(f"训练集没有图片: {image_dir}")
    labeled: list[Path] = []
    unlabeled: list[Path] = []
    for path in images:
        label_path = label_dir / f"{path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            labeled.append(path)
        else:
            unlabeled.append(path)
    n_images = min(int(n_images), len(images))
    ratio = len(labeled) / len(images)
    n_labeled = min(len(labeled), max(1, round(n_images * ratio)))
    n_unlabeled = min(len(unlabeled), n_images - n_labeled)
    rng = random.Random(seed)
    picked = rng.sample(labeled, n_labeled) + rng.sample(unlabeled, n_unlabeled)
    picked.sort(key=lambda p: p.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p.resolve().as_posix() for p in picked) + "\n", encoding="utf-8")
    info = {
        "requested": n_images,
        "selected": len(picked),
        "with_boxes": n_labeled,
        "empty_labels": n_unlabeled,
        "full_train_images": len(images),
        "full_train_empty_ratio": round(len(unlabeled) / len(images), 4),
        "seed": seed,
        "list_file": _rel(out_path),
    }
    return out_path, info


def scan_modified_images(since_epoch: float) -> dict[str, Any]:
    """找出训练期间被修改过的数据集图片（Ultralytics 会就地"修复"损坏的 JPEG）。"""
    from src import paths

    changed: list[dict[str, Any]] = []
    total = 0
    for split in paths.SPLITS:
        image_dir = paths.IMAGES_DIR / split
        if not image_dir.is_dir():
            continue
        for entry in os.scandir(image_dir):
            if not entry.is_file() or Path(entry.name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            total += 1
            try:
                st = entry.stat()
            except OSError:
                continue
            if st.st_mtime >= since_epoch:
                changed.append({"split": split, "name": entry.name, "bytes": st.st_size,
                                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return {"scanned_images": total, "modified_count": len(changed), "modified": changed[:50]}


def write_runtime_data_yaml(data_yaml: dict[str, Any], path: Path, train_source: str | None = None) -> Path:
    """生成给 Ultralytics 用的运行时数据 yaml（绝对路径，避免它按自身默认目录解析）。

    yaml 里仍写 ``path: <项目内 datasets/forest_smoke 的绝对路径>``，
    该绝对路径由 ``src/paths.py`` 在运行时算出，**不是**写死在源码或配置里。
    """
    from src import paths

    meta = {
        "path": paths.DATA_ROOT.resolve().as_posix(),
        "train": train_source or data_yaml.get("train", "images/train"),
        "val": data_yaml.get("val", "images/val"),
        "test": data_yaml.get("test", "images/test"),
        "nc": int(data_yaml.get("nc", 2)),
        "names": data_yaml.get("names", ["smoke", "fire"]),
    }
    lines = [
        "# 由 train.py 在运行时生成，仅供本次训练使用（不要手工编辑）",
        f"# 生成时间: {_now_local()}",
        "",
    ]
    for key in ("path", "train", "val", "test", "nc"):
        lines.append(f"{key}: {json.dumps(meta[key]) if isinstance(meta[key], str) else meta[key]}")
    names = meta["names"]
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 设备选择
# ---------------------------------------------------------------------------
def select_device(preferred: str) -> tuple[str, dict[str, Any]]:
    """优先 GPU，做一次真实矩阵乘法确认；失败回退 cpu 并说明原因。"""
    info: dict[str, Any] = {"preferred": preferred, "selected": "cpu", "reason": ""}
    try:
        import torch
    except Exception as exc:
        info["reason"] = f"torch 不可用（{exc}）"
        return "cpu", info

    info["torch"] = torch.__version__
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    info["cuda_available"] = bool(torch.cuda.is_available())

    if preferred == "cpu":
        info["reason"] = "命令行/配置指定 cpu"
        return "cpu", info
    if not torch.cuda.is_available():
        info["reason"] = "torch.cuda.is_available() 为 False，回退 CPU"
        return "cpu", info

    try:
        device_index = int(preferred)
    except (TypeError, ValueError):
        info["reason"] = f"无法解析 device={preferred!r}，回退 CPU"
        return "cpu", info
    if device_index >= torch.cuda.device_count():
        info["reason"] = f"device {device_index} 不存在（可见 GPU 数 {torch.cuda.device_count()}），回退 CPU"
        return "cpu", info

    try:
        a = torch.randn(512, 512, device=f"cuda:{device_index}")
        b = a @ a
        torch.cuda.synchronize()
        if not bool(torch.isfinite(b).all()):
            raise RuntimeError("矩阵乘法结果含非有限值")
        info["gpu_name"] = torch.cuda.get_device_name(device_index)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(device_index))
        info["vram_total_mb"] = round(
            torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024), 1
        )
        info["matmul_check"] = "PASS"
        info["selected"] = str(device_index)
        info["reason"] = "GPU 矩阵乘法通过"
        return str(device_index), info
    except Exception as exc:
        info["reason"] = f"GPU 自检失败（{type(exc).__name__}: {exc}），回退 CPU"
        return "cpu", info


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    tokens = (
        "out of memory",
        "not enough memory",
        "can't allocate memory",
        "bad_alloc",
        "cublas_status_alloc_failed",
    )
    return any(token in text for token in tokens)


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    """读取 Ultralytics 的 results.csv（列名带空格，需 strip）。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None:
                        continue
                    clean = key.strip()
                    try:
                        row[clean] = float(value) if value not in (None, "") else None
                    except (TypeError, ValueError):
                        row[clean] = value
                rows.append(row)
    except OSError:
        return rows
    return rows


def plot_loss_curves(rows: list[dict[str, Any]], out_png: Path) -> Path | None:
    """用 results.csv 画 loss 与 mAP 曲线（训练流程是否正常的直接证据）。"""
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            try:
                matplotlib.rcParams["font.sans-serif"] = [font]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
    except Exception:
        return None

    epochs = [int(r.get("epoch") or (i + 1)) for i, r in enumerate(rows)]

    def series(*candidates: str) -> list[float | None]:
        for name in candidates:
            if rows and name in rows[0]:
                return [r.get(name) for r in rows]
        return [None] * len(rows)

    loss_train = series("train/box_loss")
    loss_cls = series("train/cls_loss")
    loss_dfl = series("train/dfl_loss")
    val_box = series("val/box_loss")
    map50 = series("metrics/mAP50(B)", "metrics/mAP50")
    map5095 = series("metrics/mAP50-95(B)", "metrics/mAP50-95")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    ax = axes[0]
    have_loss = False
    for values, label, style in (
        (loss_train, "train box_loss", "-"),
        (loss_cls, "train cls_loss", "-"),
        (loss_dfl, "train dfl_loss", "--"),
        (val_box, "val box_loss", "-."),
    ):
        if any(v is not None for v in values):
            ax.plot(epochs, values, style, marker="o", ms=3, label=label)
            have_loss = True
    ax.set_title("损失曲线（来自 results.csv）" if have_loss else "损失曲线（results.csv 无此列）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    if have_loss:
        ax.legend(fontsize=8)

    ax2 = axes[1]
    have_metric = any(v is not None for v in map50) or any(v is not None for v in map5095)
    if any(v is not None for v in map50):
        ax2.plot(epochs, map50, "-o", ms=3, label="mAP50")
    if any(v is not None for v in map5095):
        ax2.plot(epochs, map5095, "-o", ms=3, label="mAP50-95")
    ax2.set_title("验证指标曲线" if have_metric else "验证指标（results.csv 无此列）")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)
    if have_metric:
        ax2.legend(fontsize=8)

    fig.suptitle("YOLOv8 森林烟雾/火焰检测 —— 训练曲线", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def summarize_epochs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "lr/pg0",
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {k: row.get(k) for k in keep if k in row}
        if "epoch" in item and isinstance(item["epoch"], float):
            item["epoch"] = int(item["epoch"])
        out.append(item)
    return out


def run_training(
    settings: dict[str, Any],
    data_yaml_path: Path,
    run_root: Path,
    device: str,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """调用 Ultralytics 训练；OOM 时按 batch 降级重试。"""
    from ultralytics import YOLO

    if settings.get("no_downgrade"):
        batches = [int(settings["batch"])]  # 用户要求：OOM 立即终止，不降级重试
    else:
        batches = [int(settings["batch"])] + [b for b in OOM_FALLBACK_BATCHES if b < int(settings["batch"])]
    attempts: list[dict[str, Any]] = []
    last_error: str | None = None

    for idx, batch in enumerate(batches, start=1):
        attempt_dir = run_root / f"batch{batch}"
        params = {
            "model": settings["model"],
            "data": str(data_yaml_path),
            "epochs": int(settings["epochs"]),
            "imgsz": int(settings["imgsz"]),
            "batch": batch,
            "workers": int(settings["workers"]),
            "patience": int(settings["patience"]),
            "seed": int(settings["seed"]),
            "deterministic": bool(settings["deterministic"]),
            "amp": bool(settings["amp"]),
            "cache": bool(settings["cache"]),
            "device": device,
            "project": str(run_root),
            "name": attempt_dir.name,
            "exist_ok": True,
            "pretrained": bool(settings["pretrained"]),
            "plots": bool(settings["plots"]),
            "fraction": float(settings["fraction"]),
            "verbose": True,
        }
        for key in ("lr0", "lrf", "warmup_epochs", "close_mosaic"):
            if key in settings:
                params[key] = settings[key]
        print(f"\n[训练] 第 {idx}/{len(batches)} 次尝试：batch={batch} device={device} "
              f"epochs={params['epochs']} imgsz={params['imgsz']} fraction={params['fraction']}", flush=True)
        t0 = time.time()
        try:
            model = YOLO(settings["model"])
            model.train(**params)
            elapsed = time.time() - t0
            attempts.append({"batch": batch, "status": "success", "seconds": round(elapsed, 1)})
            print(f"[训练] 成功：batch={batch}，耗时 {elapsed:.1f}s", flush=True)
            return {
                "status": "success",
                "batch_used": batch,
                "attempts": attempts,
                "attempt_dir": str(attempt_dir),
                "seconds": round(elapsed, 1),
                "params": params,
                "model": model,
            }
        except BaseException as exc:  # noqa: BLE001 - 需要区分 OOM 与其他错误
            elapsed = time.time() - t0
            oom = _is_oom(exc)
            attempts.append(
                {
                    "batch": batch,
                    "status": "oom" if oom else "error",
                    "seconds": round(elapsed, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if oom:
                print(f"[显存] batch={batch} 显存不足：{exc}", flush=True)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                last_error = f"{type(exc).__name__}: {exc}"
                if settings.get("no_downgrade"):
                    print("[显存] 已启用 --no-downgrade：按要求立即终止，不再重试。", flush=True)
                    return {
                        "status": "oom_aborted",
                        "batch_used": None,
                        "attempts": attempts,
                        "attempt_dir": None,
                        "seconds": None,
                        "params": None,
                        "error": last_error,
                    }
                continue
            raise

    return {
        "status": "oom_exhausted",
        "batch_used": None,
        "attempts": attempts,
        "attempt_dir": None,
        "seconds": None,
        "params": None,
        "error": last_error,
    }


def collect_latest_csv(attempt_dir: Path) -> Path | None:
    csv_path = attempt_dir / "results.csv"
    return csv_path if csv_path.exists() else None


def copy_artifacts(attempt_dir: Path, run_name: str) -> dict[str, Any]:
    """把权重复制到 models/，曲线复制到 outputs/figures/<run>/（只复制，不移动）。"""
    from src import paths

    copied: dict[str, Any] = {"weights": {}, "figures": []}

    for weight in ("best.pt", "last.pt"):
        src = attempt_dir / "weights" / weight
        if src.exists():
            dst = paths.MODELS_DIR / f"{run_name}_{weight}"
            paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied["weights"][weight] = {"path": _rel(dst), "bytes": dst.stat().st_size}

    fig_dir = paths.FIGURES_DIR / run_name
    for name in CURVE_FILES:
        src = attempt_dir / name
        if src.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
            dst = fig_dir / name
            shutil.copy2(src, dst)
            copied["figures"].append({"name": name, "path": _rel(dst), "bytes": dst.stat().st_size})
    return copied


def write_markdown_report(report: dict[str, Any], path: Path) -> Path:
    lines: list[str] = []
    add = lines.append

    status_cn = {"success": "成功", "oom_exhausted": "显存不足（降级后仍失败）", "error": "失败"}
    add(f"# 训练运行报告：{report['run_name']}")
    add("")
    add(f"- 模式：**{report['settings']['mode']}**"
        + ("（小规模试跑，仅验证流程）" if report["settings"]["mode"] == "smoke" else "（正式训练）"))
    add(f"- 状态：**{status_cn.get(report['status'], report['status'])}**")
    add(f"- 开始：{report['started_at']}")
    add(f"- 结束：{report['finished_at']}")
    if report.get("seconds") is not None:
        add(f"- 总耗时：**{report['seconds']} 秒**（约 {report['seconds'] / 60:.1f} 分钟）")
    add(f"- 设备：`{report['device']['selected']}`"
        + (f"（{report['device'].get('gpu_name', '')}）" if report["device"].get("gpu_name") else "")
        + f"；选择原因：{report['device']['reason']}")
    if report.get("batch_used"):
        add(f"- 实际批大小：**{report['batch_used']}**")
    led = report.get("epochs_budget") or {}
    if led:
        add(f"- 批次轮数：本次 {led.get('this_batch')} 轮；此前累计 {led.get('completed_before')} 轮；"
            f"累计上限 {led.get('budget')} 轮")
    if report.get("base_weights"):
        add(f"- 起始权重：`{report['base_weights']}`")
    add("")

    add("## 一、运行参数（最终生效值）")
    add("")
    add("| 参数 | 值 |")
    add("| --- | --- |")
    for key in ("model", "epochs", "imgsz", "batch", "workers", "patience", "seed",
                "deterministic", "amp", "cache", "fraction", "mode"):
        add(f"| {key} | {report['settings'].get(key)} |")
    add(f"| device | {report['device']['selected']} |")
    add("")

    add("## 二、数据")
    add("")
    add("| 集合 | 图片数 |")
    add("| --- | --- |")
    for split, count in report["data_check"]["split_images"].items():
        add(f"| {split} | {count} |")
    add("")
    add(f"类别：`{report['data_check']['class_names']}`；阻塞性问题 **{report['data_check']['blocking']}**，"
        f"警告 **{report['data_check']['warnings']}**。")
    add("")

    add("## 三、硬件占用（训练期间采样）")
    add("")
    mon = report["resources"]
    add("| 指标 | 值 |")
    add("| --- | --- |")
    add(f"| 采样次数（间隔 {mon['interval_s']} s） | {mon['samples']} |")
    add(f"| 进程内存峰值 | {mon['process_rss_peak_mb']} MB |")
    add(f"| 系统可用内存最低点 | {mon['system_ram_available_low_mb']} MB |")
    add(f"| CUDA 已分配显存峰值 | {mon['cuda_allocated_peak_mb']} MB |")
    add(f"| CUDA 保留显存峰值 | {mon['cuda_reserved_peak_mb']} MB |")
    add("")

    add("## 四、逐轮指标（真实，来自 results.csv）")
    add("")
    epochs = report.get("epochs") or []
    if epochs:
        add("| epoch | box_loss | cls_loss | dfl_loss | val_box_loss | P | R | mAP50 | mAP50-95 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in epochs:
            def fmt(key: str, digits: int = 4) -> str:
                value = row.get(key)
                return "—" if value is None else f"{value:.{digits}f}"

            add(
                f"| {row.get('epoch', '—')} | {fmt('train/box_loss')} | {fmt('train/cls_loss')} | "
                f"{fmt('train/dfl_loss')} | {fmt('val/box_loss')} | {fmt('metrics/precision(B)')} | "
                f"{fmt('metrics/recall(B)')} | {fmt('metrics/mAP50(B)')} | {fmt('metrics/mAP50-95(B)')} |"
            )
    else:
        add("（没有 results.csv，说明本次没有完成任何训练轮次）")
    add("")

    trial = report.get("trial_split")
    if trial:
        add("## 四之二、试跑抽样（不是直接取前 N 张）")
        add("")
        add(f"- 抽中 {trial['selected']} 张 / 全量 {trial['full_train_images']} 张")
        add(f"- 有框 {trial['with_boxes']} 张，空标签负样本 {trial['empty_labels']} 张"
            f"（全量空标签比例 {trial['full_train_empty_ratio']}）")
        add(f"- 抽样种子 {trial['seed']}；清单 `{trial['list_file']}`")
        add("")
        add("> Ultralytics 的 `fraction` 取的是按文件名排序后的前 N 张。本项目文件名已重编号，"
            "前 1507 张里有 1502 张是空标签负样本，直接用它试跑会得到“只学会预测空白”的假象。"
            "因此试跑改为按“有框/无框”比例分层抽样，种子固定。")
        add("")

    per_class = report.get("per_class_val")
    if per_class:
        add("## 四之三、分类别指标（val，用 best.pt 重新验证一次）")
        add("")
        if per_class.get("error"):
            add(f"- 分类别验证失败：{per_class['error']}")
        else:
            add("| 类别 | 实例数 | P | R | mAP50 | mAP50-95 |")
            add("| --- | --- | --- | --- | --- | --- |")
            for name, item in (per_class.get("per_class") or {}).items():
                add(f"| {name} | {item['instances']} | {item['precision']} | {item['recall']} "
                    f"| {item['mAP50']} | {item['mAP50_95']} |")
            agg = per_class.get("aggregate") or {}
            add(f"| **all** | - | {agg.get('precision')} | {agg.get('recall')} "
                f"| {agg.get('mAP50')} | {agg.get('mAP50_95')} |")
            speed = per_class.get("speed_ms") or {}
            if speed:
                add("")
                add(f"- 验证速度：预处理 {speed.get('preprocess')} ms / 推理 {speed.get('inference')} ms "
                    f"/ 后处理 {speed.get('postprocess')} ms（每张）")
        add("")

    overfit = report.get("overfitting")
    if overfit:
        add("## 四之四、过拟合判断（只看可观测趋势）")
        add("")
        add(f"- 判断：**{overfit.get('verdict')}**")
        add(f"- 依据：{overfit.get('detail')}")
        if overfit.get("train_box_loss"):
            add(f"- 最近 3 轮 train/box_loss：{overfit.get('train_box_loss')}")
            add(f"- 最近 3 轮 val/box_loss：{overfit.get('val_box_loss')}")
            add(f"- 最近 3 轮 val mAP50：{overfit.get('mAP50_window')}")
            add(f"- val−train 的 box_loss 差距：{overfit.get('val_minus_train_box_gap')}")
        add("")
        add("> 3 轮窗口只是启发式判断，不等于统计意义上的过拟合结论。")
        add("")

    add("## 五、产物")
    add("")
    add(f"- 运行目录：`{report['run_dir']}`")
    for name, item in (report.get("artifacts") or {}).get("weights", {}).items():
        add(f"- 权重 {name}：`{item['path']}`（{item['bytes']:,} B）")
    for fig in (report.get("artifacts") or {}).get("figures", []):
        add(f"- 图：`{fig['path']}`（{fig['bytes']:,} B）")
    add("")

    changes = report.get("dataset_changes") or {}
    add("## 五之二、训练期间数据集文件的变化")
    add("")
    add(f"- 扫描图片 {changes.get('scanned_images')} 张，训练期间被修改 **{changes.get('modified_count')}** 张")
    if changes.get("modified_count"):
        add("")
        add("Ultralytics 会对“损坏的 JPEG”就地修复并保存，因此这些文件的字节与修改时间会变化：")
        add("")
        add("| 集合 | 文件 | 大小 | 修改时间 |")
        add("| --- | --- | --- | --- |")
        for item in changes.get("modified", [])[:20]:
            add(f"| {item['split']} | {item['name']} | {item['bytes']:,} B | {item['mtime']} |")
        add("")
        add("> 这些图片**原文件已被改写**（内容修复），标签未动。若要完全留档，"
            "需在训练前对 images 目录做一次哈希快照。")
    add("")

    snap = report.get("snapshot") or {}
    diff = report.get("snapshot_diff") or {}
    if snap.get("path"):
        add("## 五之三、训练前图片哈希快照与对比")
        add("")
        add(f"- 快照文件：`{snap.get('path')}`")
        add(f"- 快照图片数：{snap.get('images')} 张 / {snap.get('total_bytes'):,} 字节"
            f"（计算耗时 {snap.get('seconds')} s）")
        add(f"- 训练后对比：内容被改变的图片 **{diff.get('changed_count', 0)}** 张；"
            f"缺失 {len(diff.get('missing') or [])} 张；新增 {len(diff.get('added') or [])} 张")
        changed = diff.get("changed") or []
        if changed:
            add("")
            add("| 文件 | 训练前字节 | 训练后字节 | 训练后修改时间 |")
            add("| --- | --- | --- | --- |")
            for item in changed[:20]:
                add(f"| {item['file']} | {item['before_bytes']:,} B | {item['after_bytes']:,} B "
                    f"| {item['after_mtime']} |")
            add("")
            add("> 这些图片是被 Ultralytics 的“corrupt JPEG restored and saved”**就地覆盖**的，"
                "原始字节已不可恢复；标签文件未被改动。")
        add("")

    add("## 六、说明与限制")
    add("")
    add(f"- {DISCLAIMER}")
    if report["settings"]["mode"] == "smoke":
        add("- 本次是**小规模试跑**：轮数少、只用了部分数据，指标**不能**当作模型效果结论。")
    add("- 数据划分是近似分组（原始视频 ID 已被抹除），跨集合仍存在少量近似重复（阶段五统计 19 组）。")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    from src import paths

    try:
        settings = resolve_settings(args)
    except Exception as exc:
        print(f"[错误] 读取训练配置失败：{exc}")
        return 1

    try:
        data_yaml = _load_yaml(Path(args.data_config))
    except Exception as exc:
        print(f"[错误] 读取数据集配置失败：{exc}")
        return 1

    print("=" * 78)
    print("YOLOv8 森林烟雾 / 火焰检测 —— 训练入口（阶段六）")
    print("=" * 78)
    print(f"项目根      : {paths.PROJECT_ROOT}")
    print(f"训练配置    : {_rel(Path(args.config))}")
    print(f"数据配置    : {_rel(Path(args.data_config))}")
    print(f"运行模式    : {settings['mode']}"
          + ("（小规模试跑）" if settings["mode"] == "smoke" else "（正式训练）"))
    print(f"epochs/imgsz/batch/workers : {settings['epochs']} / {settings['imgsz']} / "
          f"{settings['batch']} / {settings['workers']}")
    print(f"fraction    : {settings['fraction']}")
    print(f"amp/cache   : {settings['amp']} / {settings['cache']}")
    print(f"预训练权重  : {settings['model']}")
    print(f"{DISCLAIMER}")

    # ---- 数据集校验 ----
    if not args.skip_data_check:
        print("\n[校验] 检查数据集目录、图片/标签配对与标签格式 ...", flush=True)
        try:
            data_check = validate_dataset(data_yaml, deep=bool(args.deep_check and not args.dry_run))
        except Exception as exc:
            print(f"[错误] 数据集校验过程异常：{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 2
        for note in data_check.notes:
            print(f"       - {note}")
        if not data_check.ok:
            print("\n[错误] 数据集校验未通过，已停止（不会开始训练）。")
            print("       处理建议：先运行 src/dataset_utils.py --check 查看完整报告。")
            return 2
        print(f"[校验] 通过：图片数 {data_check.split_images}")
    else:
        data_check = None
        print("\n[校验] 已被 --skip-data-check 跳过（不推荐）")

    if args.check_data_only:
        print("\n[完成] 只做数据校验，已按要求退出（未训练）。")
        return 0

    # ---- 设备 ----
    device, device_info = select_device(settings["device"])
    print(f"\n[设备] 选择 {device} —— {device_info['reason']}")
    if device_info.get("gpu_name"):
        print(f"       {device_info['gpu_name']}，显存 {device_info.get('vram_total_mb')} MB")

    if args.dry_run:
        print("\n[完成] --dry-run：只校验与打印参数，未训练、未写任何权重。")
        return 0

    # ---- 累计轮数预算（防止无意中超训；见约束"累计上限 30 epoch"）----
    ledger_path = Path(args.ledger) if args.ledger else paths.REPORTS_DIR / LEDGER_NAME
    ledger_entries = read_ledger(ledger_path)
    used_epochs = cumulative_epochs(ledger_entries)
    budget = int(settings.get("epoch_budget") or TOTAL_EPOCH_BUDGET_DEFAULT)
    planned_epochs = int(settings["epochs"])
    if used_epochs + planned_epochs > budget and not settings.get("allow_exceed_budget"):
        print(f"\n[停止] 累计轮数会超过上限：已完成 {used_epochs} 轮 + 本批 {planned_epochs} 轮 "
              f"> 上限 {budget} 轮。")
        print("       如确需继续，请显式加 --allow-exceed-budget（不建议）。")
        return 4
    print(f"\n[批次] 已完成 {used_epochs} 轮，本批 {planned_epochs} 轮，"
          f"累计上限 {budget} 轮 → 完成后 {used_epochs + planned_epochs} 轮")

    # ---- 目录与日志 ----
    try:
        paths.ensure_project_dirs(create=True)
    except OSError as exc:
        print(f"[错误] 无法创建项目目录：{exc}")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{settings['name']}_{settings['mode']}_{stamp}"
    run_root = paths.TRAIN_RUNS_DIR / run_name
    run_root.mkdir(parents=True, exist_ok=False)  # 绝不覆盖已有实验目录

    log_path = run_root / "train.log"
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee  # type: ignore[assignment]
    started_at = _now_local()
    t_start = time.time()
    monitor = ResourceMonitor()

    try:
        print(f"\n[输出] 运行目录：{_rel(run_root)}")

        trial_info: dict[str, Any] | None = None
        train_source: str | None = None
        max_images = int(settings.get("max_images") or 0)
        if settings["mode"] == "smoke" and max_images > 0:
            trial_list, trial_info = build_trial_train_list(
                paths.IMAGES_DIR / "train",
                paths.LABELS_DIR / "train",
                max_images,
                int(settings["seed"]),
                run_root / "trial_train.txt",
            )
            train_source = trial_list.resolve().as_posix()
            settings["fraction"] = 1.0  # 抽样已由清单完成，关掉 fraction 的“取前 N 张”行为
            print(f"[试跑] 分层抽样 {trial_info['selected']} 张（有框 {trial_info['with_boxes']} + "
                  f"负样本 {trial_info['empty_labels']}；全量 {trial_info['full_train_images']} 张，"
                  f"全量空标签比例 {trial_info['full_train_empty_ratio']}）")

        data_yaml_path = write_runtime_data_yaml(data_yaml, run_root / "data_runtime.yaml", train_source)
        print(f"[输出] 运行时数据配置：{_rel(data_yaml_path)}")

        # 环境快照（真实采集，不做任何推测）
        env_snapshot: dict[str, Any] = {
            "captured_at": _now_local(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        }
        try:
            from src import environment

            env_snapshot["environment"] = environment.collect()
        except Exception as exc:
            env_snapshot["environment_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(run_root / "environment.json", env_snapshot)

        run_params = {
            "run_name": run_name,
            "started_at": started_at,
            "mode": settings["mode"],
            "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()},
            "device": device_info,
            "data_config": _rel(Path(args.data_config)),
            "data_config_sha256": _sha256(Path(args.data_config)),
            "train_config": _rel(Path(args.config)),
            "train_config_sha256": _sha256(Path(args.config)),
            "command_line": [sys.executable, "train.py", *(argv if argv is not None else sys.argv[1:])],
            "dataset_meta": data_yaml.get("dataset_meta"),
            "split_meta": data_yaml.get("split_meta"),
        }
        _write_json(run_root / "run_params.json", run_params)
        print(f"[输出] 环境快照与运行参数已写入：environment.json / run_params.json")

        snapshot_payload: dict[str, Any] | None = None
        if settings.get("snapshot"):
            print("\n[快照] 计算图片 sha256（只读操作，约 1-2 分钟）...", flush=True)
            snapshot_payload = snapshot_images(run_root / "image_snapshot.json")
            print(f"[快照] {snapshot_payload['images']:,} 张 / "
                  f"{snapshot_payload['total_bytes']:,} 字节，耗时 {snapshot_payload['seconds']} s")
            print(f"       已写入 {_rel(run_root / 'image_snapshot.json')}")

        monitor.start()
        try:
            result = run_training(settings, data_yaml_path, run_root, device, monitor)
        finally:
            monitor.stop()

        artifacts: dict[str, Any] = {}
        epochs_rows: list[dict[str, Any]] = []
        if result["status"] == "success":
            attempt_dir = Path(result["attempt_dir"])
            csv_path = collect_latest_csv(attempt_dir)
            if csv_path:
                rows = parse_results_csv(csv_path)
                epochs_rows = summarize_epochs(rows)
                plot_loss_curves(rows, paths.FIGURES_DIR / run_name / "loss_curves.png")
            artifacts = copy_artifacts(attempt_dir, run_name)

        per_class: dict[str, Any] | None = None
        if result["status"] == "success" and settings.get("per_class_val"):
            best_rel = ((artifacts.get("weights") or {}).get("best.pt") or {}).get("path")
            best_path = (paths.PROJECT_ROOT / best_rel) if best_rel else None
            if best_path and best_path.is_file():
                print("\n[评价] 用 best.pt 在 val 集做一次分类别验证 ...", flush=True)
                try:
                    per_class = per_class_validation(
                        best_path,
                        data_yaml_path,
                        split="val",
                        imgsz=int(settings["imgsz"]),
                        batch=int(settings["batch"]),
                        device=device,
                    )
                    for name, item in (per_class.get("per_class") or {}).items():
                        print(f"       {name}: P={item['precision']} R={item['recall']} "
                              f"mAP50={item['mAP50']} mAP50-95={item['mAP50_95']} "
                              f"（{item['instances']} 个实例）")
                except Exception as exc:
                    per_class = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"[警告] 分类别验证失败（不影响训练结果）：{per_class['error']}")
            else:
                print("[警告] 找不到 best.pt，跳过分类别验证")

        dataset_changes = scan_modified_images(t_start)
        snapshot_diff = diff_snapshot(snapshot_payload) if snapshot_payload else None
        overfit = overfitting_assessment(epochs_rows)

        try:
            import torch

            cuda_summary = {
                "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                if torch.cuda.is_available()
                else None,
                "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
                if torch.cuda.is_available()
                else None,
            }
        except Exception:
            cuda_summary = {"max_memory_allocated_mb": None, "max_memory_reserved_mb": None}

        report: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": _rel(run_root),
            "status": result["status"],
            "started_at": started_at,
            "finished_at": _now_local(),
            "seconds": round(time.time() - t_start, 1),
            "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in settings.items()},
            "device": device_info,
            "batch_used": result.get("batch_used"),
            "attempts": result["attempts"],
            "data_check": {
                "split_images": data_check.split_images if data_check else {},
                "split_labels": data_check.split_labels if data_check else {},
                "class_names": data_check.class_names if data_check else [],
                "blocking": data_check.blocking if data_check else None,
                "warnings": data_check.warnings if data_check else None,
                "notes": data_check.notes if data_check else [],
            },
            "resources": monitor.summary(),
            "trial_split": trial_info,
            "dataset_changes": dataset_changes,
            "snapshot": {
                "path": _rel(run_root / "image_snapshot.json") if snapshot_payload else None,
                "images": snapshot_payload.get("images") if snapshot_payload else None,
                "total_bytes": snapshot_payload.get("total_bytes") if snapshot_payload else None,
                "seconds": snapshot_payload.get("seconds") if snapshot_payload else None,
            },
            "snapshot_diff": snapshot_diff,
            "per_class_val": per_class,
            "overfitting": overfit,
            "epochs_budget": {
                "completed_before": used_epochs,
                "this_batch": planned_epochs if result["status"] == "success" else 0,
                "budget": budget,
            },
            "base_weights": settings["model"],
            "cuda_peak": cuda_summary,
            "epochs": epochs_rows,
            "artifacts": artifacts,
            "disclaimer": DISCLAIMER,
        }
        json_path = paths.REPORTS_DIR / f"train_{run_name}.json"
        md_path = paths.REPORTS_DIR / f"train_{run_name}.md"
        _write_json(json_path, report)
        write_markdown_report(report, md_path)

        last_row = epochs_rows[-1] if epochs_rows else {}
        ledger_entries = append_ledger(
            ledger_path,
            {
                "run_name": run_name,
                "status": result["status"],
                "mode": settings["mode"],
                "epochs": planned_epochs if result["status"] == "success" else 0,
                "base_weights": settings["model"],
                "batch_used": result.get("batch_used"),
                "started_at": started_at,
                "finished_at": report["finished_at"],
                "mAP50": last_row.get("metrics/mAP50(B)"),
                "mAP50_95": last_row.get("metrics/mAP50-95(B)"),
                "report": _rel(json_path),
            },
        )
        print(f"\n[批次] 本次 {planned_epochs if result['status'] == 'success' else 0} 轮已记入台账；"
              f"累计 {cumulative_epochs(ledger_entries)} / {budget} 轮")

        print("\n" + "=" * 78)
        if result["status"] == "success":
            print(f"[完成] 训练成功：batch={result['batch_used']}，"
                  f"总耗时 {report['seconds']} 秒")
            if epochs_rows:
                last = epochs_rows[-1]
                print(f"       最后一轮 mAP50={last.get('metrics/mAP50(B)')} "
                      f"mAP50-95={last.get('metrics/mAP50-95(B)')}")
            print(f"       报告：{_rel(json_path)}")
            print(f"             {_rel(md_path)}")
            print(f"       权重：{_rel(paths.MODELS_DIR / (run_name + '_best.pt'))}")
            print("[提醒] 请先看损失曲线与验证指标是否正常，再决定是否执行正式训练。")
            return 0
        if result["status"] == "oom_aborted":
            print(f"[显存不足] 按要求立即终止（--no-downgrade）：{result.get('error')}")
            print("       处理建议：降低 --batch 或 --imgsz 后重跑；不要在同一显存状态下强行继续。")
            print(f"       报告：{_rel(md_path)}")
            return 4
        if result["status"] == "oom_exhausted":
            print("[显存不足] 依次尝试的批大小全部失败：")
            for attempt in result["attempts"]:
                print(f"       - batch={attempt['batch']}：{attempt.get('error')}")
            print("       处理建议：")
            print("       1) 降低输入尺寸：--imgsz 512")
            print("       2) 关闭 GUI/浏览器等占用显存的程序后重试")
            print("       3) 仍失败则用 CPU 小规模验证流程：--device cpu --epochs 1")
            return 4
        print(f"[失败] 训练过程抛出异常：{result.get('error')}")
        print(f"       报告：{_rel(md_path)}")
        return 5
    except KeyboardInterrupt:
        print("\n[中断] 用户中断训练；已产生的日志与权重不会被删除。")
        return 130
    except Exception as exc:
        print(f"\n[失败] 训练入口异常：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 5
    finally:
        sys.stdout = tee.stream  # type: ignore[assignment]
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
