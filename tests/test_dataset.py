"""数据 / 项目结构测试 —— 不依赖 torch、cv2、pytest。

运行方式（两种都可以）：

    # 1) pytest（阶段四装好 pytest 后）
    python -m pytest tests/test_dataset.py -v

    # 2) 不装 pytest 也能跑（本文件只用了 Python 内置 assert）
    python -m tests.test_dataset

说明：断言只针对**当前真实存在的文件与真实行为**，不做任何"假装通过"。
"""

from __future__ import annotations

import sys
from pathlib import Path

from src import dataset_utils, paths

#: 项目根目录下必须存在的一级条目（计划书阶段二要求的目录结构）
EXPECTED_ENTRIES: tuple[str, ...] = (
    "app.py",
    "train.py",
    "predict.py",
    "evaluate.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "configs",
    "src",
    "datasets",
    "models",
    "outputs",
    "docs",
    "tests",
)


def test_project_root_exists() -> None:
    """项目根目录存在，且是绝对路径。"""
    assert paths.PROJECT_ROOT.is_dir(), f"项目根不存在: {paths.PROJECT_ROOT}"
    assert paths.PROJECT_ROOT.is_absolute()


def test_expected_entries_exist() -> None:
    """计划书要求的一级文件/目录全部存在。"""
    missing = [name for name in EXPECTED_ENTRIES if not (paths.PROJECT_ROOT / name).exists()]
    assert not missing, f"项目根缺少以下条目: {missing}"


def test_required_dirs_exist() -> None:
    """目录结构完整（含 train/val/test 的 images 与 labels 子目录）。"""
    missing = [str(d.relative_to(paths.PROJECT_ROOT)) for d in paths.all_dirs() if not d.is_dir()]
    assert not missing, f"缺少目录: {missing}"


def test_split_dirs_returns_image_and_label_paths() -> None:
    """split_dirs 对三个合法划分都返回正确命名的一对目录。"""
    for split in paths.SPLITS:
        img_dir, lbl_dir = paths.split_dirs(split)
        assert img_dir.name == split and lbl_dir.name == split
        assert img_dir.parent == paths.IMAGES_DIR
        assert lbl_dir.parent == paths.LABELS_DIR


def test_split_dirs_rejects_unknown_split() -> None:
    """非法划分名必须报错（不静默返回错误路径）。"""
    try:
        paths.split_dirs("valid")
    except ValueError as exc:
        assert "valid" in str(exc)
    else:
        raise AssertionError("split_dirs('valid') 应当抛出 ValueError")


def test_dataset_yaml_declares_two_classes() -> None:
    """数据集配置必须是双类别 smoke/fire，且不含盘符（禁止写死绝对路径）。"""
    text = paths.DATASET_YAML.read_text(encoding="utf-8")
    assert "nc: 2" in text, "configs/forest_smoke.yaml 必须声明 nc: 2"
    assert "names:" in text and "smoke" in text and "fire" in text, \
        "configs/forest_smoke.yaml 必须声明 names: [smoke, fire]"
    assert "E:\\" not in text and "E:/" not in text, "数据集配置里不允许出现硬编码盘符路径"


def test_train_yaml_contains_managed_params() -> None:
    """训练参数集中在 configs/train.yaml（含试跑段），且不写死盘符。"""
    text = paths.TRAIN_YAML.read_text(encoding="utf-8")
    for key in ("model:", "data:", "epochs:", "imgsz:", "batch:", "workers:", "device:", "seed:", "smoke_test:"):
        assert key in text, f"configs/train.yaml 缺少关键参数: {key}"
    assert "E:\\" not in text and "E:/" not in text, "训练配置里不允许出现硬编码盘符路径"


def test_dataset_info_document_is_honest() -> None:
    """数据说明文档必须如实描述当前状态（含双类别与数据是否已就位）。"""
    info = paths.DATA_ROOT / "dataset_info.md"
    assert info.is_file(), f"缺少数据说明文档: {info}"
    text = info.read_text(encoding="utf-8")
    assert ("smoke" in text and "fire" in text), "dataset_info.md 必须说明双类别 smoke/fire"
    assert ("尚未下载" in text) or ("已下载" in text), \
        "dataset_info.md 必须明确写出数据当前是否已下载"


def test_parse_yolo_label_accepts_valid_lines() -> None:
    """合法标签（两个类别、归一化坐标）必须解析成功。"""
    cid, box = dataset_utils.parse_yolo_label_line("0 0.5 0.5 0.2 0.2", 1)
    assert cid == 0 and box == (0.5, 0.5, 0.2, 0.2)
    cid, box = dataset_utils.parse_yolo_label_line("1 0.1 0.9 0.05 0.05", 2)
    assert cid == 1 and box == (0.1, 0.9, 0.05, 0.05)


def test_parse_yolo_label_rejects_invalid_lines() -> None:
    """非法标签必须抛出 LabelFormatError，并带具体错误码（不静默修正）。"""
    bad_cases = {
        "0 1.2 0.5 0.2 0.2": "coord_out_of_range",
        "0 0.5 0.5 0.0 0.2": "zero_size",
        "9 0.5 0.5 0.2 0.2": "illegal_class_id",
        "0 0.5 0.5 0.2": "bad_field_count",
        "x 0.5 0.5 0.2 0.2": "bad_class_id",
    }
    for line, expected_code in bad_cases.items():
        try:
            dataset_utils.parse_yolo_label_line(line, 1)
        except dataset_utils.LabelFormatError as exc:
            assert exc.code == expected_code, f"{line!r} 期望 {expected_code}，实际 {exc.code}"
        else:
            raise AssertionError(f"{line!r} 应当被拒绝")


def test_class_names_are_two_classes() -> None:
    """类别常量必须与配置一致：0=smoke, 1=fire。"""
    assert dataset_utils.CLASS_NAMES == {0: "smoke", 1: "fire"}
    assert dataset_utils.VALID_CLASS_IDS == frozenset({0, 1})


def test_check_dataset_returns_real_numbers_or_empty() -> None:
    """check_dataset 必须返回可解释的结构；没有数据时统计为 0，不得编造。"""
    report = dataset_utils.check_dataset(compute_phash=False)
    for split in paths.SPLITS:
        assert split in report.splits, f"报告缺少 {split} 分节"
        stats = report.splits[split]
        # 统计值必须是真实的非负整数
        assert stats.images >= 0 and stats.boxes >= 0
        assert all(v >= 0 for v in stats.boxes_per_class.values())
    # 没有图片时，总框数必须为 0（这是"没有数据"的真实结果，不是占位符）
    if report.total_images == 0:
        assert report.total_boxes == 0



# --------------------------------------------------------------------------
# 阶段五新增：绘图相关（matplotlib 只读数据，写图到临时目录）
# --------------------------------------------------------------------------


def test_plot_constants_match_config() -> None:
    """绘图常量必须与双类别配置一致，且输出目录指向 outputs/figures/dataset_inspect。"""
    assert dataset_utils.CLASS_COLORS.keys() == {0, 1}
    assert dataset_utils.INSPECT_FIGURES_DIR.name == "dataset_inspect"
    assert dataset_utils.INSPECT_FIGURES_DIR.parent.name == "figures"
    assert dataset_utils.INSPECT_FIGURES_DIR.parent.parent.name == "outputs"


def test_plot_class_distribution_writes_png() -> None:
    """给定已知统计值必须画出 PNG（不读真实数据集，避免测试依赖数据规模）。"""
    import tempfile

    # 测试用的假统计值：只用于验证绘图函数，不代表任何真实数据
    summary = {
        "splits": {
            "train": {"boxes_per_class": {"smoke": 6, "fire": 4}},
            "val": {"boxes_per_class": {"smoke": 3, "fire": 2}},
            "test": {"boxes_per_class": {"smoke": 1, "fire": 1}},
        },
        "totals": {
            "images": 40,
            "boxes": 17,
            "boxes_per_class": {"smoke": 10, "fire": 7},
            "empty_label_files": 5,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = dataset_utils.plot_class_distribution(summary, Path(tmp))
        assert out.is_file(), "类别分布图未生成"
        assert out.name == "class_distribution.png"
        data = out.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "输出不是合法的 PNG"
        assert out.stat().st_size > 3000, "PNG 太小，可能是空图"


def test_plot_split_samples_handles_missing_split() -> None:
    """没有对应集合目录时必须返回 None，而不是抛异常或画空图。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        (root / "images" / "train").mkdir(parents=True)
        out = dataset_utils.plot_split_samples(root, "train", Path(tmp) / "figs")
        assert out is None, "空目录不应生成样本图"
        # 不存在的集合也不能崩
        assert dataset_utils.plot_split_samples(root, "val", Path(tmp) / "figs") is None


def test_append_figures_section_is_idempotent() -> None:
    """重复追加"可视化产物"章节只能有一个，不能越跑越长。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "dataset_summary.md"
        md.write_text("# 数据集报告\n\n## 一、正文\n\n内容\n", encoding="utf-8")
        fake = Path(tmp) / "figs" / "class_distribution.png"
        fake.parent.mkdir()
        fake.write_bytes(b"x")
        for _ in range(3):
            dataset_utils.append_figures_section(
                md, [fake], n_positive=4, n_negative=1, seed=42
            )
        text = md.read_text(encoding="utf-8")
        assert text.count(dataset_utils.FIGURES_SECTION_TITLE) == 1, "章节被重复追加"
        assert "no object" not in text  # 只是确认没写乱码式英文占位
        assert "seed=42" in text


def _run_all() -> int:
    """不依赖 pytest 的最小执行器。"""
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    failed: list[tuple[str, str]] = []
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - 测试执行器需要捕获一切
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failed.append((name, str(exc)))
    print(f"\n合计 {len(tests)} 项，通过 {len(tests) - len(failed)} 项，失败 {len(failed)} 项")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
