"""把原始下载的数据集整理成项目的双类别 YOLO 结构（只复制，不改原始文件）。

流程
----
1. 扫描原始目录，把每张图片和它的 ``.txt`` 标签配成一对；
2. 按"视频 / 事件"给每张图分配 group（分组键来源可配置）；
3. 以 **group 为单位** 按 70/15/15 划分（seed=42），保证同一 group 只进一个集合；
4. 复制到 ``datasets/forest_smoke/{images,labels}/{train,val,test}/``；
5. 写出 ``outputs/reports/split_manifest.json``（划分清单，供校验脚本复查泄漏）。

分组模式
--------
``auto``         先试文件名前缀，再退回目录名，最后退回"顺序切块"（并标记为近似）
``pattern``      用正则提取 group（第 1 个捕获组），例如 ``--pattern "^(.+?)_\\d+$"``
``directory``    用图片所在目录名当 group
``stem-prefix``  去掉文件名末尾数字当 group（``cam01_0007`` -> ``cam01_``）
``sequence``     按自然排序每 N 张切一块（近似事件，N 由 ``--block`` 指定）
``phash``        用感知哈希近似重复聚类当 group（最慢，最贴近"同事件"）
``none``         每张图一个 group（**不做分组**，必须自行声明泄漏风险）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from .dataset_utils import (
        CLASS_NAMES,
        IMAGE_SUFFIXES,
        dhash64,
        find_near_duplicates,
        parse_yolo_label_line,
    )
    from .paths import DATA_ROOT, REPORTS_DIR, SPLITS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset_utils import (  # type: ignore[no-redef]
        CLASS_NAMES,
        IMAGE_SUFFIXES,
        dhash64,
        find_near_duplicates,
        parse_yolo_label_line,
    )
    from src.paths import DATA_ROOT, REPORTS_DIR, SPLITS  # type: ignore[no-redef]

#: 划分比例（任务书规定 70/15/15）
SPLIT_RATIO: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

#: 感知哈希缓存（只用于加速重复运行，不参与结果；可用环境变量覆盖路径）
DHASH_CACHE_PATH = Path(
    os.environ.get("DSH_DHASH_CACHE")
    or str(Path(tempfile.gettempdir()) / "fs_smoke_dhash_cache.json")
)


def _load_hash_cache() -> dict[str, int]:
    """读感知哈希缓存；文件不存在或损坏时返回空字典（不影响正确性）。"""
    try:
        raw = json.loads(DHASH_CACHE_PATH.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_hash_cache(cache: dict[str, int]) -> None:
    try:
        DHASH_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _cached_dhash(path: Path, cache: dict[str, int]) -> int | None:
    """带缓存的 dHash：键为绝对路径 + 文件大小，避免同路径换文件后读到旧值。"""
    try:
        key = f"{path.resolve()}|{path.stat().st_size}"
    except OSError:
        key = str(path.resolve())
    if key in cache:
        return cache[key]
    h = dhash64(path)
    if h is not None:
        cache[key] = h
    return h

#: 固定随机种子
SEED: int = 42


@dataclass
class Pair:
    """一张图片 + 它的标签文件（标签可能为空文件 = 负样本）。"""

    image: Path
    label: Path | None
    group: str = ""
    split: str = ""

    @property
    def name(self) -> str:
        return self.image.name


# --------------------------------------------------------------------------
# 1. 发现图片-标签对
# --------------------------------------------------------------------------


def _label_candidates(image: Path) -> list[Path]:
    """按常见命名约定推断标签路径（不猜内容，只找文件）。"""
    cands: list[Path] = []
    parts = list(image.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        replaced = parts.copy()
        replaced[idx] = "labels"
        cands.append(Path(*replaced).with_suffix(".txt"))
    cands.append(image.with_suffix(".txt"))
    cands.append(image.parent.parent / "labels" / f"{image.stem}.txt")
    cands.append(image.parent / "labels" / f"{image.stem}.txt")
    return cands


def find_pairs(raw_root: Path) -> list[Pair]:
    """递归扫描原始目录，返回图片-标签对列表（保持稳定排序以便复现）。"""
    pairs: list[Pair] = []
    images = sorted(
        p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for img in images:
        label = next((c for c in _label_candidates(img) if c.is_file()), None)
        pairs.append(Pair(image=img, label=label))
    return pairs


# --------------------------------------------------------------------------
# 2. 分组
# --------------------------------------------------------------------------


def _natural_key(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def assign_groups(pairs: list[Pair], mode: str = "auto", *,
                  pattern: str | None = None, block: int = 30,
                  max_group_size: int = 200) -> tuple[str, bool]:
    """给每对图片分配 group。返回 (实际使用的模式描述, 是否为近似分组)。"""
    if not pairs:
        return "empty", True

    if mode == "pattern":
        if not pattern:
            raise ValueError("mode=pattern 时必须提供 --pattern")
        rx = re.compile(pattern)
        missing = 0
        for p in pairs:
            m = rx.search(p.image.stem)
            if m:
                p.group = m.group(1)
            else:
                p.group = f"__nomatch__/{p.image.stem}"
                missing += 1
        desc = f"文件名正则 {pattern!r}（未匹配 {missing} 张，各自成组）"
        return desc, False

    if mode == "directory":
        for p in pairs:
            p.group = p.image.parent.name
        return "所在目录名", False

    if mode == "stem-prefix":
        for p in pairs:
            p.group = re.sub(r"\d+$", "", p.image.stem) or p.image.stem
        return "文件名去掉末尾数字", False

    if mode == "sequence":
        ordered = sorted(pairs, key=lambda p: _natural_key(p.image.stem))
        for i, p in enumerate(ordered):
            p.group = f"seq_{i // max(block, 1):05d}"
        return f"按文件名自然序每 {block} 张切一块（近似事件）", True

    if mode == "phash":
        hashes: dict[str, int] = {}
        cache = _load_hash_cache()
        before = len(cache)
        for p in pairs:
            h = _cached_dhash(p.image, cache)
            if h is not None:
                hashes[str(p.image)] = h
        if len(cache) != before:
            _save_hash_cache(cache)
        groups = find_near_duplicates(hashes)
        # 超大簇按文件名顺序切成若干块：dHash 的 union-find 会“链式合并”，
        # 可能把成千上万张只是画面相似的图并成一组；若不分块，这一整组会
        # 全部落进同一个集合，导致划分严重失衡。
        chunks: list[list[str]] = []
        capped = 0
        for members in groups:
            if len(members) <= max_group_size:
                chunks.append(members)
                continue
            capped += 1
            ordered = sorted(members, key=lambda s: _natural_key(Path(s).stem))
            for i in range(0, len(ordered), max_group_size):
                chunks.append(ordered[i:i + max_group_size])
        owner: dict[str, str] = {}
        for gi, members in enumerate(chunks):
            gid = f"ph_{gi:05d}"
            for m in members:
                owner[m] = gid
        for p in pairs:
            p.group = owner.get(str(p.image), f"solo_{p.image.stem}")
        note = f"，其中 {capped} 个超大簇按文件名顺序切成 ≤{max_group_size} 张的块"
        return f"感知哈希近似重复聚类（dHash≤8，union-find）{note}", True

    if mode == "none":
        for p in pairs:
            p.group = p.image.stem
        return "未分组（每张图独立）", True

    if mode != "auto":
        raise ValueError(f"未知分组模式: {mode}")

    # auto：先看文件名前缀是否真的产生有意义的组
    stems = [p.image.stem for p in pairs]
    prefixed = [re.sub(r"\d+$", "", s) for s in stems]
    distinct = len(set(prefixed))
    if 1 < distinct < len(pairs) * 0.9:
        biggest = Counter(prefixed).most_common(1)[0][1]
        if biggest < len(pairs) * 0.5:
            for p, pre in zip(pairs, prefixed):
                p.group = pre or p.image.stem
            return f"自动选择：文件名去掉末尾数字（{distinct} 组）", False

    # 再看目录名
    dirs = [p.image.parent.name for p in pairs]
    if 1 < len(set(dirs)) <= len(pairs):
        biggest = Counter(dirs).most_common(1)[0][1]
        if biggest < len(pairs) * 0.9:
            for p, d in zip(pairs, dirs):
                p.group = d
            return f"自动选择：所在目录名（{len(set(dirs))} 组）", False

    return "自动回退：未找到可靠分组线索，每张图独立成组", True


# --------------------------------------------------------------------------
# 3. 按组划分
# --------------------------------------------------------------------------


def split_by_group(
    pairs: list[Pair],
    *,
    seed: int = SEED,
    ratio: dict[str, float] | None = None,
) -> dict[str, int]:
    """以 group 为单位划分，返回每个集合分到的组数与图片数统计。"""
    ratio = ratio or SPLIT_RATIO
    groups: dict[str, int] = Counter(p.group for p in pairs)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    # 先随机打乱（保证同尺寸组的归属不由文件名决定），再按组大小降序，
    # 用 Best-Fit Decreasing 分配到"放进去后超出目标比例最少"的集合。
    keys.sort(key=lambda k: -groups[k])

    total = len(pairs)
    targets = {s: total * ratio[s] for s in ("train", "val", "test")}
    assigned: dict[str, list[str]] = {s: [] for s in ("train", "val", "test")}
    counts = {s: 0 for s in ("train", "val", "test")}

    for key in keys:
        n = groups[key]

        def excess(split: str, _n: int = n) -> tuple[int, float]:
            """越小越好：先看是否会超出目标（0 优先），再看超出比例。"""
            over = (counts[split] + _n) - targets[split]
            return (1 if over > 0 else 0, over / max(targets[split], 1e-9))

        choice = min(("train", "val", "test"), key=excess)
        assigned[choice].append(key)
        counts[choice] += n

    group_to_split = {g: s for s, gs in assigned.items() for g in gs}
    for p in pairs:
        p.split = group_to_split[p.group]

    sizes = sorted(groups.values(), reverse=True)
    return {
        "images": {s: counts[s] for s in counts},
        "groups": {s: len(assigned[s]) for s in assigned},
        "total_groups": len(keys),
        "group_size": {
            "max": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "singletons": sum(1 for v in sizes if v == 1),
        },
        "actual_ratio": {
            s: (counts[s] / total if total else 0.0) for s in counts
        },
    }


# --------------------------------------------------------------------------
# 4. 落盘（复制，不移动、不改原文件）
# --------------------------------------------------------------------------


def materialize(
    pairs: list[Pair],
    data_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """把图片与标签复制到目标结构；已存在同名文件时加短哈希前缀避免覆盖。"""
    stats = {"copied_images": 0, "copied_labels": 0, "renamed": 0, "skipped": 0}
    for split in SPLITS:
        (data_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for p in pairs:
        dst_img = data_root / "images" / p.split / p.name
        if dst_img.exists() and not overwrite:
            tag = hashlib.sha1(str(p.image).encode()).hexdigest()[:8]
            dst_img = dst_img.with_name(f"{p.image.stem}__{tag}{p.image.suffix}")
            stats["renamed"] += 1
        elif dst_img.exists():
            stats["skipped"] += 1
            continue
        shutil.copy2(p.image, dst_img)
        stats["copied_images"] += 1

        if p.label is not None:
            dst_lbl = data_root / "labels" / p.split / f"{dst_img.stem}.txt"
            shutil.copy2(p.label, dst_lbl)
            stats["copied_labels"] += 1
    return stats


def write_manifest(
    pairs: list[Pair],
    path: Path,
    *,
    dataset_meta: dict,
    split_config: dict,
    group_mode: str,
    approximate: bool,
    split_stats: dict,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for p in sorted(pairs, key=lambda x: (x.split, x.name)):
        class_counts: dict[str, int] = {}
        if p.label is not None and p.label.is_file():
            for line_no, line in enumerate(
                p.label.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    cid, _ = parse_yolo_label_line(line, line_no)
                except Exception:  # noqa: BLE001 - 明细错误交给检查脚本报告
                    continue
                name = CLASS_NAMES.get(cid, str(cid))
                class_counts[name] = class_counts.get(name, 0) + 1
        images.append(
            {
                "split": p.split,
                "group": p.group,
                "image": p.name,
                "source": str(p.image),
                "label_found": p.label is not None,
                "boxes": class_counts,
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "src/prepare_dataset.py",
        "dataset_meta": dataset_meta,
        "split_config": split_config,
        "group_mode": group_mode,
        "grouping_approximate": approximate,
        "split_stats": split_stats,
        "counts": {
            "images": len(images),
            "groups": len({i["group"] for i in images}),
            "per_split": {s: sum(1 for i in images if i["split"] == s) for s in SPLITS},
        },
        "images": images,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build(raw_root: Path, data_root: Path, *, mode: str, pattern: str | None,
          block: int, seed: int, dry_run: bool, overwrite: bool,
          dataset_meta: dict, max_group_size: int = 200) -> int:
    t0 = time.time()
    print(f"[1/5] 扫描原始目录: {raw_root}")
    pairs = find_pairs(raw_root)
    print(f"      找到 {len(pairs)} 张图片")
    if not pairs:
        print("[错误] 没有找到任何图片，检查 --raw 是否指对")
        return 1
    missing = sum(1 for p in pairs if p.label is None)
    print(f"      其中 {missing} 张没有找到标签文件（检查阶段会报错）")

    print("[2/5] 分组 ...")
    desc, approximate = assign_groups(
        pairs, mode, pattern=pattern, block=block, max_group_size=max_group_size)
    n_groups = len({p.group for p in pairs})
    print(f"      模式: {desc}")
    print(f"      组数: {n_groups}（近似分组={approximate}）")

    print(f"[3/5] 按组划分（seed={seed}, 70/15/15）...")
    split_stats = split_by_group(pairs, seed=seed)
    for s in ("train", "val", "test"):
        share = split_stats["actual_ratio"][s] * 100
        print(f"      {s:5s} 图片 {split_stats['images'][s]:6d}  组 "
              f"{split_stats['groups'][s]:5d}  占比 {share:5.2f}%")
    gs = split_stats["group_size"]
    print(f"      组大小: 最大 {gs['max']}，中位数 {gs['median']}，"
          f"单图组 {gs['singletons']} 个")

    # 泄漏自检：同组必须只在一个集合
    g2s: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        g2s[p.group].add(p.split)
    leaked = {g: s for g, s in g2s.items() if len(s) > 1}
    print(f"      泄漏自检: {len(leaked)} 个组跨集合（必须为 0）")
    if leaked:
        print("[错误] 分组划分泄漏，终止，不复制任何文件")
        return 1

    if dry_run:
        print("[dry-run] 不复制文件、不写清单，结束")
        return 0

    print(f"[4/5] 复制到 {data_root} ...")
    copy_stats = materialize(pairs, data_root, overwrite=overwrite)
    print(f"      {copy_stats}")

    print("[5/5] 写划分清单 ...")
    split_config = {
        "ratio": SPLIT_RATIO,
        "seed": seed,
        "unit": "group",
        "group_key_source": desc,
        "grouping_approximate": approximate,
        "leakage_risk_note": (
            "分组基于感知哈希近似聚类，属于近似分组，指标可能偏乐观"
            if approximate else "按文件名/目录中的事件标识分组，同一组只进入一个集合"
        ),
    }
    manifest_path = write_manifest(
        pairs, REPORTS_DIR / "split_manifest.json",
        dataset_meta=dataset_meta, split_config=split_config,
        group_mode=desc, approximate=approximate, split_stats=split_stats,
    )
    print(f"      {manifest_path}")
    print(f"完成，用时 {time.time() - t0:.1f}s")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="整理原始数据集为双类别 YOLO 结构（只复制不改原件）")
    ap.add_argument("--raw", required=True, help="原始解压目录")
    ap.add_argument("--out", default=str(DATA_ROOT), help="输出数据集根目录")
    ap.add_argument("--group-by", default="auto",
                    choices=["auto", "pattern", "directory", "stem-prefix",
                             "sequence", "phash", "none"])
    ap.add_argument("--pattern", default=None, help="分组正则（第 1 个捕获组为 group）")
    ap.add_argument("--block", type=int, default=30, help="sequence 模式每块图片数")
    ap.add_argument("--max-group-size", type=int, default=200,
                    help="phash 模式超大簇的最大张数（超过则按文件名顺序切块）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖同名目标文件（默认不覆盖）")
    ap.add_argument("--dataset-name", default="")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--license-notes", default="")
    ap.add_argument("--version", default="")
    ap.add_argument("--archive-bytes", default="")
    ap.add_argument("--archive-sha256", default="")
    args = ap.parse_args(list(argv) if argv is not None else None)

    raw = Path(args.raw)
    if not raw.is_dir():
        print(f"[错误] 原始目录不存在: {raw}")
        return 1

    meta = {
        "name": args.dataset_name,
        "source_url": args.source_url,
        "license": args.license,
        "license_notes": args.license_notes,
        "version_or_commit": args.version,
        "acquired_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "archive_bytes": int(args.archive_bytes) if args.archive_bytes.isdigit() else args.archive_bytes,
        "archive_sha256": args.archive_sha256,
        "scene_composition_note": (
            "D-Fire 为野外/林区监控与公开火灾图像合集，"
            "包含城市、工业等非森林场景；不是纯森林数据集。"
        ),
        "conversion_note": "原始 YOLO 标签原样复制（双类别 smoke/fire），未合并、未丢弃任何标注框。",
        "conclusion_notes": "由 src/prepare_dataset.py 整理，src/dataset_utils.py 校验。",
    }
    return build(
        raw, Path(args.out), mode=args.group_by, pattern=args.pattern,
        block=args.block, seed=args.seed, dry_run=args.dry_run,
        overwrite=args.overwrite, dataset_meta=meta,
        max_group_size=args.max_group_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
