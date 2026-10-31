r"""数据集检查与统计工具（双类别：0=smoke 烟雾，1=fire 火焰）。

设计原则
--------
1. **只读**：本模块绝不删除、移动、修改原始图片与标签文件；发现问题只记录。
2. **不猜**：所有统计数字都来自真实扫描；无法判断的填 None，不写 0 冒充。
3. **可复现**：随机种子、分组规则、扫描时间全部写进报告。

本模块同时负责**阶段五要求的绘图产物**：
    - 类别标签数量分布柱状图；
    - 每个集合（train/val/test）带标注框的样本可视化图。
    图片统一输出到 ``outputs/figures/dataset_inspect/``，绘图只用 matplotlib，
    且只**读取**原始图片与标签，不改动任何数据集文件。

命令行用法
----------
完整检查（扫描 + 统计 + 校验 + 绘图 + 写 JSON/Markdown 报告，最常用）::

    .venv\Scripts\python.exe src\dataset_utils.py --check

只要检查、不要画图（更快）::

    .venv\Scripts\python.exe src\dataset_utils.py --check --no-plots

跳过感知哈希去重（大集合上更快，但拿不到近似重复统计）::

    .venv\Scripts\python.exe src\dataset_utils.py --check --no-phash

指定报告输出位置::

    .venv\Scripts\python.exe src\dataset_utils.py --check ^
        --report outputs/reports/dataset_summary.json ^
        --markdown outputs/reports/dataset_summary.md

**只重画图**（读取已有 JSON 报告，不重新扫描，几秒钟完成）::

    .venv\Scripts\python.exe src\dataset_utils.py --plots-only

改变抽样数量与随机种子（每个集合抽 12 张有框 + 3 张无目标负样本）::

    .venv\Scripts\python.exe src\dataset_utils.py --plots-only --samples 12 --samples-negative 3 --seed 7

把图输出到别的目录::

    .venv\Scripts\python.exe src\dataset_utils.py --plots-only --figures-dir outputs/figures/my_check

检查另一个数据集目录::

    .venv\Scripts\python.exe src\dataset_utils.py --check --root datasets/forest_smoke

退出码：0 = 无阻塞性问题；1 = 路径/参数错误；3 = 存在阻塞性标签错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # 作为包导入（python -m src.dataset_utils）
    from .paths import DATA_ROOT, FIGURES_DIR, IMAGES_DIR, LABELS_DIR, REPORTS_DIR, SPLITS
except ImportError:  # 作为脚本运行（python src/dataset_utils.py）
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.paths import DATA_ROOT, FIGURES_DIR, IMAGES_DIR, LABELS_DIR, REPORTS_DIR, SPLITS

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 类别编号 -> 名称。与 configs/forest_smoke.yaml 的 names 顺序必须一致。
CLASS_NAMES: dict[int, str] = {0: "smoke", 1: "fire"}

#: 合法的类别编号集合。出现其它编号即为标签错误。
VALID_CLASS_IDS: frozenset[int] = frozenset(CLASS_NAMES)

#: YOLO 检测标签每行的字段数：class cx cy w h
YOLO_FIELD_COUNT: int = 5

#: 允许的图片后缀
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

#: 感知哈希（dHash）汉明距离阈值：<= 该值视为近似重复
PHASH_MAX_DISTANCE: int = 8

#: 报告 schema 版本
SCHEMA_VERSION: str = "1.1"

#: 阶段五绘图产物目录：outputs/figures/dataset_inspect
INSPECT_FIGURES_DIR: Path = FIGURES_DIR / "dataset_inspect"

#: 类别颜色（与报告文字对应：0=烟雾 红，1=火焰 橙）
CLASS_COLORS: dict[int, str] = {0: "#d62728", 1: "#ff7f0e"}

#: 每个集合默认抽样：多少张"有框"图片、多少张"无目标"负样本
DEFAULT_SAMPLES_PER_SPLIT: int = 8
DEFAULT_NEGATIVE_SAMPLES_PER_SPLIT: int = 2

#: 抽样随机种子（与数据划分的 seed 一致，便于复现）
DEFAULT_SAMPLE_SEED: int = 42


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetIssue:
    """一条数据问题记录。severity: error（影响训练）/ warning（需注意）/ info。"""

    split: str
    image: str
    label_line: int | None
    code: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "image": self.image,
            "label_line": self.label_line,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class SplitStats:
    """单个划分（train/val/test）的统计结果。"""

    name: str
    images: int = 0
    images_with_labels: int = 0
    images_without_labels: int = 0
    images_unreadable: int = 0
    empty_label_files: int = 0
    boxes: int = 0
    boxes_per_class: dict[int, int] = field(default_factory=dict)
    images_with_smoke: int = 0
    images_with_fire: int = 0
    groups: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "images_with_labels": self.images_with_labels,
            "images_without_labels": self.images_without_labels,
            "images_unreadable": self.images_unreadable,
            "empty_label_files": self.empty_label_files,
            "boxes": self.boxes,
            "boxes_per_class": {
                CLASS_NAMES.get(k, str(k)): v for k, v in sorted(self.boxes_per_class.items())
            },
            "images_with_smoke": self.images_with_smoke,
            "images_with_fire": self.images_with_fire,
            "groups": self.groups,
        }


@dataclass
class DatasetReport:
    """整份检查报告。"""

    root: str
    generated_at: str
    splits: dict[str, SplitStats]
    checks: dict[str, int | None]
    issues: list[DatasetIssue]
    box_area_histogram: dict[str, int]
    duplicates_exact: list[list[str]]
    duplicates_perceptual: list[list[str]]
    dataset_meta: dict[str, Any]
    split_config: dict[str, Any]
    scan_seconds: float

    @property
    def total_images(self) -> int:
        return sum(s.images for s in self.splits.values())

    @property
    def total_boxes(self) -> int:
        return sum(s.boxes for s in self.splits.values())

    @property
    def total_boxes_per_class(self) -> dict[int, int]:
        acc: Counter[int] = Counter()
        for s in self.splits.values():
            for cid, n in s.boxes_per_class.items():
                acc[cid] += n
        return dict(sorted(acc.items()))

    def errors(self) -> list[DatasetIssue]:
        """阻塞性问题：标签内容错到无法使用。"""
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[DatasetIssue]:
        """非阻塞警告：源数据有缺陷，但训练流程可绕过（不外改原始标签）。"""
        return [i for i in self.issues if i.severity == "warning"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "_template": False,
            "_status": "filled_from_real_scan",
            "_notice": "本文件由 src/dataset_utils.py 扫描真实数据生成；所有数字来自实际文件。",
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "generated_by": "src/dataset_utils.py --check",
            "scan_seconds": round(self.scan_seconds, 2),
            "root": self.root,
            "dataset": self.dataset_meta,
            "classes": {
                "nc": len(CLASS_NAMES),
                "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
                "class_id_map": {str(i): n for i, n in sorted(CLASS_NAMES.items())},
                "source_class_ids_seen": self.checks.get("source_class_ids_seen"),
                "conversion_note": self.dataset_meta.get("conversion_note"),
            },
            "split_config": self.split_config,
            "splits": {name: s.as_dict() for name, s in self.splits.items()},
            "totals": {
                "images": self.total_images,
                "boxes": self.total_boxes,
                "boxes_per_class": {
                    CLASS_NAMES.get(k, str(k)): v
                    for k, v in self.total_boxes_per_class.items()
                },
                "empty_label_files": sum(s.empty_label_files for s in self.splits.values()),
                "images_missing_label_file": sum(
                    s.images_without_labels for s in self.splits.values()
                ),
            },
            "checks": self.checks,
            "box_size_note": self.dataset_meta.get("box_size_note"),
            "box_area_histogram": self.box_area_histogram,
            "duplicates": {
                "exact_groups": len(self.duplicates_exact),
                "exact_images": sum(len(g) for g in self.duplicates_exact),
                "perceptual_groups": len(self.duplicates_perceptual),
                "perceptual_images": sum(len(g) for g in self.duplicates_perceptual),
                "exact_examples": self.duplicates_exact[:5],
                "perceptual_examples": self.duplicates_perceptual[:5],
            },
            "issues": [i.as_dict() for i in self.issues[:2000]],
            "issue_codes": dict(Counter(i.code for i in self.issues)),
            "conclusion": {
                "usable_for_training": not self.errors(),
                "blocking_issues": len(self.errors()),
                "notes": self.dataset_meta.get("conclusion_notes"),
            },
        }


class LabelFormatError(ValueError):
    """标签行格式错误，带具体位置信息。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# 标签解析
# --------------------------------------------------------------------------


#: 问题码 -> 严重级别。
#: error 表示"标签内容错到无法使用"，warning 表示"源数据有缺陷但训练可绕过"。
SEVERITY_BY_CODE: dict[str, str] = {
    "zero_size": "warning",   # 退化框：无法作为真值，但不影响其它框
    "wh_gt_one": "warning",   # 越界框：Ultralytics 会裁剪/忽略
}


def parse_yolo_label_line(line: str, line_no: int) -> tuple[int, tuple[float, float, float, float]]:
    """解析一行 YOLO 标签。

    返回 ``(class_id, (cx, cy, w, h))``；格式非法时抛 :class:`LabelFormatError`。
    只做格式与取值范围检查，不做任何"修正"。
    """
    raw = line.strip()
    if not raw:
        raise LabelFormatError("empty_line", f"第 {line_no} 行为空行（空行应删除或整文件为空）")

    parts = raw.split()
    if len(parts) != YOLO_FIELD_COUNT:
        raise LabelFormatError(
            "bad_field_count",
            f"第 {line_no} 行字段数 {len(parts)} != {YOLO_FIELD_COUNT}: {raw!r}",
        )

    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise LabelFormatError("bad_class_id", f"第 {line_no} 行类别非整数: {parts[0]!r}") from exc

    if class_id not in VALID_CLASS_IDS:
        raise LabelFormatError(
            "illegal_class_id",
            f"第 {line_no} 行类别 {class_id} 不在合法集合 {sorted(VALID_CLASS_IDS)} 内",
        )

    try:
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError as exc:
        raise LabelFormatError("bad_float", f"第 {line_no} 行坐标无法解析为浮点数: {raw!r}") from exc

    for name, value in (("cx", cx), ("cy", cy)):
        if not (0.0 <= value <= 1.0):
            raise LabelFormatError(
                "coord_out_of_range", f"第 {line_no} 行 {name}={value} 不在 [0,1] 内"
            )
    for name, value in (("w", w), ("h", h)):
        if value <= 0.0:
            raise LabelFormatError(
                "zero_size", f"第 {line_no} 行 {name}={value} 不满足 0 < {name}（退化框，无法作为真值）"
            )
        if value > 1.0:
            raise LabelFormatError(
                "wh_gt_one",
                f"第 {line_no} 行 {name}={value} 超过 1（源数据几何越界，"
                f"Ultralytics 会按图像边界裁剪或忽略该框）",
            )
    if not math.isfinite(cx + cy + w + h):
        raise LabelFormatError("non_finite", f"第 {line_no} 行含 NaN/Inf")

    return class_id, (cx, cy, w, h)


# --------------------------------------------------------------------------
# 感知哈希（仅用 Pillow，不引入额外依赖）
# --------------------------------------------------------------------------


def dhash64(path: Path, size: int = 8) -> int | None:
    """计算图片的 64 位差值哈希；Pillow 缺失或图片损坏时返回 None。"""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
            # 用 tobytes() 而不是 getdata()：Pillow 14 起 getdata 已废弃（2027-10-15 移除），
            # 两者在 "L" 模式下都是逐行、从左到右的 0–255 像素值，结果完全一致。
            pixels = small.tobytes()
    except Exception:  # noqa: BLE001 - 坏图不应中断扫描
        return None

    bits = 0
    idx = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            if left > right:
                bits |= 1 << idx
            idx += 1
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_near_duplicates(
    hashes: dict[str, int], max_distance: int = PHASH_MAX_DISTANCE
) -> list[list[str]]:
    """用 4 段 16 位桶做候选筛选，再算汉明距离；避免 O(n^2) 全比较。"""
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, h in hashes.items():
        for seg in range(4):
            buckets[(seg, (h >> (seg * 16)) & 0xFFFF)].append(name)

    seen_pairs: set[tuple[str, str]] = set()
    parent: dict[str, str] = {name: name for name in hashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for names in buckets.values():
        if len(names) < 2 or len(names) > 400:  # 超大桶多为空白图，跳过以免过慢
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                key = (a, b) if a < b else (b, a)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if _hamming(hashes[a], hashes[b]) <= max_distance:
                    union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for name in hashes:
        groups[find(name)].append(name)
    return [sorted(g) for g in groups.values() if len(g) > 1]


# --------------------------------------------------------------------------
# 主检查流程
# --------------------------------------------------------------------------


def _iter_images(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _label_path_for(image: Path, split: str) -> Path:
    return LABELS_DIR / split / f"{image.stem}.txt"


def check_dataset(
    data_root: Path | None = None,
    *,
    compute_phash: bool = True,
    max_phash_images: int = 30000,
    dataset_meta: dict[str, Any] | None = None,
    split_config: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> DatasetReport:
    """扫描 ``datasets/forest_smoke``，返回真实统计结果。只读。"""
    t0 = time.time()
    root = Path(data_root) if data_root else DATA_ROOT

    issues: list[DatasetIssue] = []
    splits: dict[str, SplitStats] = {}
    area_hist: Counter[str] = Counter()
    seen_class_ids: set[int] = set()
    exact_hash_map: dict[str, list[str]] = defaultdict(list)
    phash_map: dict[str, int] = {}

    counters = {
        "images_unreadable": 0,
        "labels_malformed_lines": 0,
        "labels_bad_field_count": 0,
        "labels_illegal_class_id": 0,
        "labels_coord_out_of_range": 0,
        "labels_zero_size": 0,
        "labels_wh_gt_one": 0,
        "labels_nonpositive_wh": 0,
        "images_missing_label_file": 0,
        "labels_empty_file": 0,
        "files_total": 0,
    }

    for split in SPLITS:
        stats = SplitStats(name=split, boxes_per_class={cid: 0 for cid in CLASS_NAMES})
        image_dir = IMAGES_DIR / split
        if not image_dir.is_dir():
            issues.append(
                DatasetIssue(split, "", None, "missing_split_dir",
                             f"图片目录不存在: {image_dir}", "error")
            )
            splits[split] = stats
            continue

        for image in _iter_images(image_dir):
            counters["files_total"] += 1
            stats.images += 1
            rel = f"{split}/{image.name}"

            # 1) 可读性
            try:
                from PIL import Image
                with Image.open(image) as im:
                    im.verify()
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001
                stats.images_unreadable += 1
                counters["images_unreadable"] += 1
                issues.append(
                    DatasetIssue(split, rel, None, "image_unreadable",
                                 f"图片无法打开: {type(exc).__name__}: {exc}", "error")
                )
                continue

            # 2) 标签
            label = _label_path_for(image, split)
            if not label.is_file():
                stats.images_without_labels += 1
                counters["images_missing_label_file"] += 1
                issues.append(
                    DatasetIssue(split, rel, None, "missing_label_file",
                                 f"缺少标签文件: {label.name}", "error")
                )
                continue

            stats.images_with_labels += 1
            text = label.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if not text.strip():
                stats.empty_label_files += 1
                counters["labels_empty_file"] += 1
                continue

            has_smoke = has_fire = False
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    class_id, (cx, cy, w, h) = parse_yolo_label_line(line, line_no)
                except LabelFormatError as exc:
                    counters["labels_malformed_lines"] += 1
                    if exc.code == "bad_field_count":
                        counters["labels_bad_field_count"] += 1
                    elif exc.code == "illegal_class_id":
                        counters["labels_illegal_class_id"] += 1
                    elif exc.code == "coord_out_of_range":
                        counters["labels_coord_out_of_range"] += 1
                    elif exc.code == "zero_size":
                        counters["labels_zero_size"] += 1
                        counters["labels_nonpositive_wh"] += 1
                    elif exc.code == "wh_gt_one":
                        counters["labels_wh_gt_one"] += 1
                        counters["labels_nonpositive_wh"] += 1
                    issues.append(
                        DatasetIssue(
                            split, rel, line_no, exc.code, exc.message,
                            SEVERITY_BY_CODE.get(exc.code, "error"),
                        )
                    )
                    continue

                seen_class_ids.add(class_id)
                stats.boxes += 1
                stats.boxes_per_class[class_id] = stats.boxes_per_class.get(class_id, 0) + 1
                if class_id == 0:
                    has_smoke = True
                elif class_id == 1:
                    has_fire = True

                area = w * h
                if area < 0.001:
                    area_hist["< 0.1%"] += 1
                elif area < 0.01:
                    area_hist["0.1%–1%"] += 1
                elif area < 0.1:
                    area_hist["1%–10%"] += 1
                elif area < 0.5:
                    area_hist["10%–50%"] += 1
                else:
                    area_hist["> 50%"] += 1

            if has_smoke:
                stats.images_with_smoke += 1
            if has_fire:
                stats.images_with_fire += 1

            # 3) 精确重复（文件级 sha256）
            digest = _sha256_file(image)
            exact_hash_map[digest].append(rel)

            # 4) 感知哈希
            if compute_phash and len(phash_map) < max_phash_images:
                h = dhash64(image)
                if h is not None:
                    phash_map[rel] = h

        splits[split] = stats

    # 跨集合：同一文件内容出现在多个划分
    dup_exact = [sorted(v) for v in exact_hash_map.values() if len(v) > 1]
    cross_split = [
        g for g in dup_exact if len({p.split("/")[0] for p in g}) > 1
    ]
    for group in cross_split:
        issues.append(
            DatasetIssue("all", group[0], None, "duplicate_across_splits",
                         "同一图片出现在多个划分: " + ", ".join(group), "error")
        )

    dup_perceptual = find_near_duplicates(phash_map) if phash_map else []
    cross_perceptual = [
        g for g in dup_perceptual if len({p.split("/")[0] for p in g}) > 1
    ]

    # 5) 分组泄漏检查（若提供了 manifest）
    leakage_groups = None
    if split_manifest:
        group_to_splits: dict[str, set[str]] = defaultdict(set)
        for entry in split_manifest.get("images", []):
            group_to_splits[str(entry.get("group"))].add(str(entry.get("split")))
        leaked = {g: s for g, s in group_to_splits.items() if len(s) > 1}
        leakage_groups = len(leaked)
        for g, s in list(leaked.items())[:50]:
            issues.append(
                DatasetIssue("all", g, None, "group_leakage",
                             f"同一事件组 {g} 出现在多个划分: {sorted(s)}", "error")
            )
        for split, stats in splits.items():
            stats.groups = len({e["group"] for e in split_manifest.get("images", [])
                                if e.get("split") == split})

    checks: dict[str, int | None] = {
        "files_total": counters["files_total"],
        "images_unreadable": counters["images_unreadable"],
        "labels_malformed_lines": counters["labels_malformed_lines"],
        "labels_bad_field_count": counters["labels_bad_field_count"],
        "labels_illegal_class_id": counters["labels_illegal_class_id"],
        "labels_coord_out_of_range": counters["labels_coord_out_of_range"],
        "labels_zero_size": counters["labels_zero_size"],
        "labels_wh_gt_one": counters["labels_wh_gt_one"],
        "labels_nonpositive_wh": counters["labels_nonpositive_wh"],
        "images_missing_label_file": counters["images_missing_label_file"],
        "duplicate_exact": len(dup_exact),
        "duplicate_perceptual": len(dup_perceptual),
        "duplicate_across_splits": len(cross_split),
        "duplicate_perceptual_across_splits": len(cross_perceptual),
        "split_leakage_groups": leakage_groups,
        "source_class_ids_seen": sorted(seen_class_ids),
    }

    meta = dict(dataset_meta or {})
    meta.setdefault(
        "box_size_note",
        "相对面积 = w*h（标签已归一化），仅用于观察小目标占比。",
    )

    return DatasetReport(
        root=str(root),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        splits=splits,
        checks=checks,
        issues=issues,
        box_area_histogram={
            k: area_hist.get(k, 0)
            for k in ("< 0.1%", "0.1%–1%", "1%–10%", "10%–50%", "> 50%")
        },
        duplicates_exact=dup_exact,
        duplicates_perceptual=dup_perceptual,
        dataset_meta=meta,
        split_config=dict(split_config or {}),
        scan_seconds=time.time() - t0,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 报告输出
# --------------------------------------------------------------------------


def write_json_report(report: DatasetReport, path: Path | None = None) -> Path:
    """把报告写成 JSON（真实数字）。"""
    out = Path(path) if path else REPORTS_DIR / "dataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def write_markdown_report(report: DatasetReport, path: Path | None = None) -> Path:
    """把报告写成 Markdown，数字全部取自真实扫描结果。"""
    out = Path(path) if path else REPORTS_DIR / "dataset_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = report.dataset_meta
    total_by_class = report.total_boxes_per_class
    lines: list[str] = []
    add = lines.append

    add("# 数据集统计报告（真实扫描结果）")
    add("")
    add("> 本文件由 `src/dataset_utils.py --check` 扫描 `datasets/forest_smoke` 实际文件生成，")
    add("> 表中每个数字都可在磁盘上复核。**不是模板，也不是估算。**")
    add("")
    add(f"- 生成时间：{report.generated_at}")
    add(f"- 数据根目录：`{report.root}`")
    add(f"- 扫描耗时：{report.scan_seconds:.1f} 秒")
    add(f"- 数据集名称：{meta.get('name')}")
    add(f"- 来源 URL：{meta.get('source_url')}")
    add(f"- 许可：{meta.get('license')}")
    add(f"- 许可备注：{meta.get('license_notes')}")
    add(f"- 版本 / 提交：{meta.get('version_or_commit')}")
    add(f"- 下载日期：{meta.get('acquired_at')}")
    add(f"- 原始包字节数：{meta.get('archive_bytes')}")
    add(f"- 原始包 SHA256：`{meta.get('archive_sha256')}`")
    add(f"- 场景构成说明：{meta.get('scene_composition_note')}")
    add("")

    add("## 一、数据划分配置")
    add("")
    add("| 项目 | 值 |")
    add("| --- | --- |")
    sc = report.split_config
    ratio = sc.get("ratio", {})
    add(f"| 训练 / 验证 / 测试比例 | {ratio.get('train')} / {ratio.get('val')} / {ratio.get('test')} |")
    add(f"| 随机种子 | {sc.get('seed')} |")
    add(f"| 划分单位 | {sc.get('unit')} |")
    add(f"| 分组键来源 | {sc.get('group_key_source')} |")
    add(f"| 分组是否近似 | {sc.get('grouping_approximate')} |")
    add(f"| 泄漏风险说明 | {sc.get('leakage_risk_note')} |")
    add("")

    add("## 二、各集合数量统计")
    add("")
    add("| 集合 | 图片数 | 有标签 | 缺标签 | 空标签(负样本) | 烟雾框 | 火焰框 | 框合计 | 组数 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in SPLITS:
        s = report.splits.get(name)
        if s is None:
            continue
        d = s.as_dict()
        add(
            f"| {name} | {d['images']} | {d['images_with_labels']} | "
            f"{d['images_without_labels']} | {d['empty_label_files']} | "
            f"{d['boxes_per_class'].get('smoke', 0)} | {d['boxes_per_class'].get('fire', 0)} | "
            f"{d['boxes']} | {d['groups'] if d['groups'] is not None else '—'} |"
        )
    add(
        f"| **合计** | **{report.total_images}** | — | — | "
        f"**{sum(s.empty_label_files for s in report.splits.values())}** | "
        f"**{total_by_class.get(0, 0)}** | **{total_by_class.get(1, 0)}** | "
        f"**{report.total_boxes}** | — |"
    )
    add("")

    add("## 三、类别分布")
    add("")
    add("| 类别编号 | 类别名 | 框数 | 占全部框比例 |")
    add("| --- | --- | --- | --- |")
    for cid, cname in sorted(CLASS_NAMES.items()):
        n = total_by_class.get(cid, 0)
        pct = f"{n / report.total_boxes * 100:.2f}%" if report.total_boxes else "—"
        add(f"| {cid} | {cname} | {n} | {pct} |")
    add("")
    add(f"实测出现的类别编号：{report.checks.get('source_class_ids_seen')}")
    add("")

    add("## 四、质量检查结果")
    add("")
    add("| 检查项 | 结果 |")
    add("| --- | --- |")
    labels = {
        "files_total": "扫描图片文件总数",
        "images_unreadable": "图片无法打开",
        "labels_malformed_lines": "标签行格式错误（合计）",
        "labels_bad_field_count": "字段数 ≠ 5",
        "labels_illegal_class_id": "类别编号非法",
        "labels_coord_out_of_range": "坐标越界",
        "labels_zero_size": "退化框（宽或高 ≤ 0）",
        "labels_wh_gt_one": "几何越界框（宽或高 > 1）",
        "labels_nonpositive_wh": "宽或高非法（合计）",
        "images_missing_label_file": "缺少标签文件",
        "duplicate_exact": "精确重复图片组",
        "duplicate_perceptual": "感知哈希近似重复组",
        "duplicate_across_splits": "跨集合精确重复组",
        "duplicate_perceptual_across_splits": "跨集合近似重复组",
        "split_leakage_groups": "跨集合同组泄漏",
    }
    for key, desc in labels.items():
        add(f"| {desc} | {report.checks.get(key)} |")
    add("")

    add("## 五、目标框相对面积分布")
    add("")
    add("| 相对面积（w×h） | 框数 |")
    add("| --- | --- |")
    for k, v in report.box_area_histogram.items():
        add(f"| {k} | {v} |")
    add("")
    add(f"> {meta.get('box_size_note')}")
    add("")

    add("## 六、异常清单")
    add("")
    if not report.issues:
        add("**未发现任何异常。**")
    else:
        add(f"共 {len(report.issues)} 条（下表最多展示 60 条，完整清单见 JSON 报告）。")
        add("")
        add("| # | 集合 | 文件 | 标签行 | 问题码 | 说明 | 级别 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for i, issue in enumerate(report.issues[:60], start=1):
            msg = issue.message.replace("|", "\\|")
            add(
                f"| {i} | {issue.split} | {issue.image} | {issue.label_line or '—'} | "
                f"{issue.code} | {msg} | {issue.severity} |"
            )
        if len(report.issues) > 60:
            add("")
            add(f"（其余 {len(report.issues) - 60} 条见 JSON）")
    add("")

    add("## 七、可用性结论")
    add("")
    usable = "是" if not report.errors() else "否"
    add(f"| 项目 | 结论 |")
    add(f"| --- | --- |")
    n_err = len(report.errors())
    n_warn = len(report.warnings())
    add(f"| 是否可用于训练 | **{usable}**（阻塞性问题 {n_err} 项，警告 {n_warn} 项） |")
    add(f"| 图片总数 | {report.total_images} |")
    add(f"| 目标框总数 | {report.total_boxes} |")
    add(f"| 说明 | {meta.get('conclusion_notes')} |")
    add("")
    add("## 八、复现方式")
    add("")
    add("```powershell")
    add(".\\\\.venv\\\\Scripts\\\\python.exe src\\\\dataset_utils.py --check")
    add("```")
    add("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# 可视化（阶段五：类别分布柱状图 + 带框样本可视化）
# --------------------------------------------------------------------------


def _setup_matplotlib() -> tuple[Any, bool]:
    """初始化 matplotlib（Agg 后端，无显示器也能画），并尽量启用中文字体。

    返回 ``(plt, cjk_available)``。找不到中文字体时返回 False，
    调用方改用英文标签，避免图上出现方块乱码。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    chosen = [name for name in preferred if name in available]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, bool(chosen)


def _t(cjk: bool, zh: str, en: str) -> str:
    """中文可用就用中文，否则退回英文，避免乱码。"""
    return zh if cjk else en


def plot_class_distribution(
    summary: dict[str, Any], figures_dir: Path | None = None
) -> Path:
    """绘制类别标签数量分布柱状图。

    左图：每个集合（train/val/test）分别统计 smoke / fire 的框数（分组柱状图）；
    右图：全数据集合计的 smoke / fire 框数 + 空标签负样本图片数。

    参数
    ----
    summary : :meth:`DatasetReport.as_dict` 的返回值，或读取到的 ``dataset_summary.json``
    figures_dir : 输出目录，默认 ``outputs/figures/dataset_inspect``
    """
    plt, cjk = _setup_matplotlib()
    out_dir = Path(figures_dir) if figures_dir else INSPECT_FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = summary.get("splits") or {}
    names = [s for s in splits] or list(SPLITS)
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    width = 0.8 / max(len(class_names), 1)
    x_base = list(range(len(names)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=150)

    for k, cname in enumerate(class_names):
        vals = [
            int(((splits.get(n) or {}).get("boxes_per_class") or {}).get(cname, 0))
            for n in names
        ]
        xs = [x - 0.4 + width / 2.0 + k * width for x in x_base]
        bars = ax0.bar(
            xs, vals, width=width, label=cname, color=CLASS_COLORS[k], edgecolor="white"
        )
        for bar, v in zip(bars, vals):
            ax0.annotate(
                f"{v:,}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    ax0.set_xticks(x_base)
    ax0.set_xticklabels(names)
    ax0.set_xlabel(_t(cjk, "数据集划分", "split"))
    ax0.set_ylabel(_t(cjk, "目标框数量", "box count"))
    ax0.set_title(_t(cjk, "各集合的类别框数", "boxes per class by split"))
    ax0.legend()
    ax0.grid(axis="y", alpha=0.3)

    totals: dict[str, Any] = summary.get("totals") or {}
    per_class = totals.get("boxes_per_class") or {}
    values = [int(per_class.get(c, 0)) for c in class_names]
    negatives = int(totals.get("empty_label_files") or 0)
    bar_labels = list(class_names) + [_t(cjk, "负样本(空标签)", "negatives")]
    bar_values = values + [negatives]
    colors = [CLASS_COLORS[i] for i in sorted(CLASS_NAMES)] + ["#7f7f7f"]
    bars = ax1.bar(bar_labels, bar_values, color=colors, edgecolor="white")
    for bar, v in zip(bars, bar_values):
        ax1.annotate(
            f"{v:,}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
    ax1.set_ylabel(_t(cjk, "数量", "count"))
    ax1.set_title(_t(cjk, "全数据集合计", "dataset totals"))
    ax1.grid(axis="y", alpha=0.3)

    images = totals.get("images")
    boxes = totals.get("boxes")
    suptitle = (
        f"数据集类别分布（图片 {images:,} 张 / 框 {boxes:,} 个 / 负样本 {negatives:,} 张）"
        if cjk
        else f"class distribution (images {images:,} / boxes {boxes:,} / negatives {negatives:,})"
    )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = out_dir / "class_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读取一个标签文件里的**合法**框；非法行直接跳过（本函数只用于画图）。"""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return boxes
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return boxes
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            class_id, (cx, cy, w, h) = parse_yolo_label_line(line, line_no)
        except LabelFormatError:
            continue
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def plot_split_samples(
    root: Path,
    split: str,
    figures_dir: Path | None = None,
    *,
    n_positive: int = DEFAULT_SAMPLES_PER_SPLIT,
    n_negative: int = DEFAULT_NEGATIVE_SAMPLES_PER_SPLIT,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Path | None:
    """抽样若干张图片并画出真实标注框，保存为一张网格图。

    抽样规则：每个集合抽 ``n_positive`` 张"至少有一个合法框"的图片，
    再抽 ``n_negative`` 张"空标签/无框"的负样本 —— 负样本也要展示，
    否则会给人"所有图都有烟"的错觉。
    """
    from matplotlib.patches import Rectangle
    from PIL import Image

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir():
        return None

    positive: list[Path] = []
    negative: list[Path] = []
    for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        if _read_boxes(label_dir / f"{img.stem}.txt"):
            positive.append(img)
        else:
            negative.append(img)

    rng = random.Random(seed)
    picked = rng.sample(positive, min(max(n_positive, 0), len(positive)))
    picked += rng.sample(negative, min(max(n_negative, 0), len(negative)))
    if not picked:
        return None

    plt, cjk = _setup_matplotlib()
    out_dir = Path(figures_dir) if figures_dir else INSPECT_FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = math.ceil(len(picked) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), dpi=110, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, img_path in zip(axes.ravel(), picked):
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                width_px, height_px = rgb.size
                ax.imshow(rgb)
        except Exception:  # noqa: BLE001
            ax.set_title(
                f"{img_path.name}\n{_t(cjk, '无法读取', 'unreadable')}", fontsize=8
            )
            continue

        boxes = _read_boxes(label_dir / f"{img_path.stem}.txt")
        for class_id, cx, cy, w, h in boxes:
            x1 = max(0.0, (cx - w / 2.0) * width_px)
            y1 = max(0.0, (cy - h / 2.0) * height_px)
            box_w = min(width_px - x1, w * width_px)
            box_h = min(height_px - y1, h * height_px)
            if box_w <= 0 or box_h <= 0:
                continue
            color = CLASS_COLORS.get(class_id, "#1f77b4")
            ax.add_patch(
                Rectangle((x1, y1), box_w, box_h, fill=False, edgecolor=color, linewidth=2)
            )
            ax.text(
                x1 + 2,
                y1 + 2,
                CLASS_NAMES.get(class_id, str(class_id)),
                color="white",
                fontsize=7,
                va="top",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
            )

        counts = Counter(b[0] for b in boxes)
        if counts:
            desc = ", ".join(
                f"{CLASS_NAMES.get(cid, cid)} x{n}" for cid, n in sorted(counts.items())
            )
        else:
            desc = _t(cjk, "无目标（负样本）", "no object (negative)")
        ax.set_title(f"{img_path.name}\n{desc}", fontsize=8)

    suptitle = (
        f"{split} 抽样 {len(picked)} 张"
        f"（有框 {min(n_positive, len(positive))} 张 + 负样本 {min(n_negative, len(negative))} 张，seed={seed}）"
        if cjk
        else f"{split}: {len(picked)} samples (seed={seed})"
    )
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = out_dir / f"samples_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_figures(
    root: Path,
    summary: dict[str, Any],
    figures_dir: Path | None = None,
    *,
    n_positive: int = DEFAULT_SAMPLES_PER_SPLIT,
    n_negative: int = DEFAULT_NEGATIVE_SAMPLES_PER_SPLIT,
    seed: int = DEFAULT_SAMPLE_SEED,
    splits: Iterable[str] = SPLITS,
) -> list[Path]:
    """生成阶段五要求的全部图：1 张类别分布图 + 每个集合 1 张样本图。"""
    produced: list[Path] = [plot_class_distribution(summary, figures_dir)]
    for split in splits:
        path = plot_split_samples(
            root,
            split,
            figures_dir,
            n_positive=n_positive,
            n_negative=n_negative,
            seed=seed,
        )
        if path is not None:
            produced.append(path)
    return produced


FIGURES_SECTION_TITLE = "## 九、可视化产物"


def append_figures_section(
    markdown_path: Path,
    figures: Iterable[Path],
    *,
    n_positive: int,
    n_negative: int,
    seed: int,
) -> Path:
    """把「可视化产物」一节写进 Markdown 报告（幂等：重跑会替换旧章节）。"""
    md = Path(markdown_path)
    if not md.is_file():
        return md
    text = md.read_text(encoding="utf-8")
    marker = FIGURES_SECTION_TITLE
    cut = text.find(marker)
    if cut != -1:
        text = text[:cut].rstrip() + "\n"

    rows = []
    for fig_path in figures:
        try:
            rel = fig_path.relative_to(md.parents[2])
        except (ValueError, IndexError):
            rel = fig_path
        if fig_path.name == "class_distribution.png":
            desc = "类别标签数量分布柱状图（左：按集合分组的 smoke/fire 框数；右：全数据集合计与负样本数）"
        elif fig_path.stem.startswith("samples_"):
            split = fig_path.stem.replace("samples_", "")
            desc = f"{split} 集合抽样样本（画出真实标注框，含负样本）"
        else:
            desc = "图表"
        rows.append(f"| `outputs/figures/dataset_inspect/{fig_path.name}` | {desc} |")

    lines = [
        "",
        marker,
        "",
        f"由 `src/dataset_utils.py` 用 matplotlib 生成，生成时间 {datetime.now():%Y-%m-%d %H:%M:%S}。",
        f"抽样规则：每个集合取 {n_positive} 张有框图片 + {n_negative} 张无目标负样本，"
        f"随机种子 seed={seed}（可复现）。",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        *rows,
        "",
        "> 绘图过程只**读取**图片与标签，未修改任何数据集文件。",
        "",
    ]
    md.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    return md


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。用法见模块文档字符串顶部的示例。"""
    parser = argparse.ArgumentParser(
        description="森林烟雾/火焰检测数据集检查工具（只读）：校验标签、统计分布、绘图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python src/dataset_utils.py --check\n"
            "  python src/dataset_utils.py --check --no-plots\n"
            "  python src/dataset_utils.py --plots-only\n"
            "  python src/dataset_utils.py --plots-only --samples 12 --samples-negative 3 --seed 7\n"
            "  python src/dataset_utils.py --check --figures-dir outputs/figures/my_check\n"
        ),
    )
    parser.add_argument("--check", action="store_true", help="执行完整检查并输出报告")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="只重画图：读取已有 JSON 报告，不重新扫描数据集",
    )
    parser.add_argument("--root", default=None, help="数据集根目录（默认 datasets/forest_smoke）")
    parser.add_argument("--report", default=None, help="JSON 报告输出路径")
    parser.add_argument("--markdown", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--no-phash", action="store_true", help="跳过感知哈希去重（更快）")
    parser.add_argument("--max-phash", type=int, default=30000, help="最多对多少张图算感知哈希")
    parser.add_argument("--no-plots", action="store_true", help="不生成图表（--check 时有效）")
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="图表输出目录（默认 outputs/figures/dataset_inspect）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPLIT,
        help=f"每个集合抽样多少张有框图片（默认 {DEFAULT_SAMPLES_PER_SPLIT}）",
    )
    parser.add_argument(
        "--samples-negative",
        type=int,
        default=DEFAULT_NEGATIVE_SAMPLES_PER_SPLIT,
        help=f"每个集合额外抽样多少张无目标负样本（默认 {DEFAULT_NEGATIVE_SAMPLES_PER_SPLIT}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"抽样随机种子（默认 {DEFAULT_SAMPLE_SEED}）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.check and not args.plots_only:
        parser.print_help()
        return 2
    if args.samples < 0 or args.samples_negative < 0:
        print("[错误] --samples / --samples-negative 不能为负数")
        return 1

    root = Path(args.root) if args.root else DATA_ROOT
    figures_dir = Path(args.figures_dir) if args.figures_dir else INSPECT_FIGURES_DIR
    markdown_path = Path(args.markdown) if args.markdown else REPORTS_DIR / "dataset_summary.md"
    plot_kwargs = {
        "n_positive": args.samples,
        "n_negative": args.samples_negative,
        "seed": args.seed,
    }

    # ---- 模式一：只重画图（不扫描，几秒钟） ----
    if args.plots_only:
        json_path = Path(args.report) if args.report else REPORTS_DIR / "dataset_summary.json"
        summary = _load_json_if_exists(json_path)
        if summary is None:
            print(f"[错误] 找不到可用的 JSON 报告：{json_path}")
            print("       请先运行: python src/dataset_utils.py --check")
            return 1
        print(f"[1/2] 读取已有报告 {json_path}")
        print(f"[2/2] 生成图表 -> {figures_dir}")
        produced = generate_figures(root, summary, figures_dir, **plot_kwargs)
        for path in produced:
            print(f"      {path}")
        append_figures_section(markdown_path, produced, **plot_kwargs)
        print(f"      已在 {markdown_path} 追加「可视化产物」一节")
        return 0

    # ---- 模式二：完整检查 + 绘图 ----
    if not root.is_dir():
        print(f"[错误] 数据集根目录不存在: {root}")
        return 1
    if not any((root / "images" / s).is_dir() for s in SPLITS):
        print(f"[错误] {root} 下没有 images/train|val|test，先准备数据集")
        return 1

    manifest = _load_json_if_exists(REPORTS_DIR / "split_manifest.json")
    meta = (manifest or {}).get("dataset_meta") or {}
    split_config = (manifest or {}).get("split_config") or {}

    print(f"[1/3] 扫描 {root} ...")
    report = check_dataset(
        root,
        compute_phash=not args.no_phash,
        max_phash_images=args.max_phash,
        dataset_meta=meta,
        split_config=split_config,
        split_manifest=manifest,
    )

    print("[2/3] 写入 JSON 报告 ...")
    json_path = write_json_report(report, Path(args.report) if args.report else None)
    print(f"      {json_path}")

    print("[3/3] 写入 Markdown 报告 ...")
    md_path = write_markdown_report(report, Path(args.markdown) if args.markdown else None)
    print(f"      {md_path}")

    if args.no_plots:
        print("[跳过] 未生成图表（--no-plots）")
    else:
        print(f"[+] 生成图表 -> {figures_dir}")
        produced = generate_figures(root, report.as_dict(), figures_dir, **plot_kwargs)
        for path in produced:
            print(f"      {path}")
        append_figures_section(md_path, produced, **plot_kwargs)
        print(f"      已在 {md_path} 追加「可视化产物」一节")

    print("-" * 60)
    for name in SPLITS:
        s = report.splits.get(name)
        if s:
            print(f"  {name:5s} 图片 {s.images:6d}  框 {s.boxes:6d}  "
                  f"(smoke {s.boxes_per_class.get(0, 0)}, fire {s.boxes_per_class.get(1, 0)})  "
                  f"空标签 {s.empty_label_files}")
    print(f"  合计  图片 {report.total_images}  框 {report.total_boxes}  "
          f"(smoke {report.total_boxes_per_class.get(0, 0)}, "
          f"fire {report.total_boxes_per_class.get(1, 0)})")
    print(f"  阻塞性错误 {len(report.errors())} 项")
    print("-" * 60)
    return 0 if not report.errors() else 3


if __name__ == "__main__":
    raise SystemExit(main())
