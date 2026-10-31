"""项目路径统一入口。

设计要点
--------
1. 项目根目录由本文件位置反推：``Path(__file__).resolve().parents[1]``。
   因此**代码里不出现任何盘符或绝对路径**，整个项目目录搬到别处也能直接运行。
   本项目约定的部署位置是 ``E:\\DeepSeek Harness\\fs_smoke``，但该字符串只写在
   文档里，代码中不硬编码。
2. 任何模块需要访问磁盘时，一律从本模块取路径；不要在业务代码里拼接字符串路径。
3. 目录按需创建（:func:`ensure_project_dirs`），不依赖 Git 是否保留空目录。

典型用法::

    from src import paths
    paths.ensure_project_dirs()
    img_dir, lbl_dir = paths.split_dirs("train")
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# 项目根目录
# --------------------------------------------------------------------------
# src/paths.py -> parents[0] = src, parents[1] = 项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 一级目录
# --------------------------------------------------------------------------
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
SRC_DIR: Path = PROJECT_ROOT / "src"

# --------------------------------------------------------------------------
# 输出子目录
# --------------------------------------------------------------------------
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
TRAIN_RUNS_DIR: Path = OUTPUTS_DIR / "train_runs"

# --------------------------------------------------------------------------
# 数据集目录（单类别 smoke）
# --------------------------------------------------------------------------
DATA_ROOT: Path = DATASETS_DIR / "forest_smoke"
IMAGES_DIR: Path = DATA_ROOT / "images"
LABELS_DIR: Path = DATA_ROOT / "labels"

#: 数据集划分名称，顺序固定，供全项目统一引用
SPLITS: tuple[str, ...] = ("train", "val", "test")

# --------------------------------------------------------------------------
# 配置文件
# --------------------------------------------------------------------------
DATASET_YAML: Path = CONFIGS_DIR / "forest_smoke.yaml"
TRAIN_YAML: Path = CONFIGS_DIR / "train.yaml"

#: 需要保证存在的全部目录（不含各划分的 images/labels 子目录，见 _all_dirs()）
_BASE_DIRS: tuple[Path, ...] = (
    CONFIGS_DIR,
    DATASETS_DIR,
    DATA_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    TRAIN_RUNS_DIR,
    DOCS_DIR,
    TESTS_DIR,
)


def _validate_split(split: str) -> str:
    """校验划分名称，非法时抛出带提示的 ValueError。"""
    if split not in SPLITS:
        raise ValueError(f"未知的数据集划分: {split!r}（只允许 {list(SPLITS)}）")
    return split


def split_dirs(split: str) -> tuple[Path, Path]:
    """返回某个划分的 (图片目录, 标签目录)。

    >>> img, lbl = split_dirs("train")
    >>> img.name, lbl.name
    ('train', 'train')
    """
    _validate_split(split)
    return IMAGES_DIR / split, LABELS_DIR / split


def all_dirs() -> tuple[Path, ...]:
    """返回项目应存在的全部目录（含各划分的 images/labels 子目录）。"""
    dirs: list[Path] = list(_BASE_DIRS)
    for split in SPLITS:
        dirs.extend(split_dirs(split))
    return tuple(dirs)


def ensure_project_dirs(create: bool = True) -> dict[str, Path]:
    """确保目录结构存在，返回"名称 -> 路径"的清单。

    Parameters
    ----------
    create:
        True 时实际创建缺失目录；False 时只检查并返回（用于测试/预演）。
    """
    manifest: dict[str, Path] = {}
    for d in all_dirs():
        if create:
            d.mkdir(parents=True, exist_ok=True)
        manifest[str(d.relative_to(PROJECT_ROOT)).replace("\\", "/")] = d
    return manifest


def project_root() -> Path:
    """返回项目根目录（绝对路径）。"""
    return PROJECT_ROOT


def describe() -> dict[str, str]:
    """返回关键路径的自述信息，便于日志记录与排错。"""
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(DATA_ROOT),
        "dataset_yaml": str(DATASET_YAML),
        "train_yaml": str(TRAIN_YAML),
        "predictions_dir": str(PREDICTIONS_DIR),
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
    }


if __name__ == "__main__":  # pragma: no cover - 手工检查用
    for key, value in describe().items():
        print(f"{key:16s} {value}")
