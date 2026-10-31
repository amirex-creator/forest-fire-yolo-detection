"""阶段八自测：评价模块与命令行入口。

只使用**内置 assert**，不依赖 pytest（与 tests/test_dataset.py、tests/test_inference.py 保持一致）。
既可以用 pytest 跑，也可以直接 ``python tests/test_evaluation.py`` 跑。

原则：不联网、不训练；凡是需要真实权重的测试，缺少权重时打印“跳过”，
绝不假装通过。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import evaluation, paths  # noqa: E402


# ---------------------------------------------------------------------------
# 几何与匹配
# ---------------------------------------------------------------------------


def test_box_iou_basic():
    a = (0, 0, 10, 10)
    assert evaluation.box_iou(a, a) == 1.0
    assert evaluation.box_iou(a, (0, 0, 10, 20)) == 0.5
    assert evaluation.box_iou(a, (20, 20, 30, 30)) == 0.0


def test_match_boxes_perfect():
    preds = [(0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    gts = [(0, 0.0, 0.0, 10.0, 10.0)]
    matched, unmatched_pred, unmatched_gt = evaluation.match_boxes(preds, gts, 0.5)
    assert matched == [0] and unmatched_pred == [] and unmatched_gt == []


def test_match_boxes_class_mismatch_counts_as_fp_and_fn():
    # 预测成 fire、标注是 smoke：既不是 TP，也要同时算 1 个 FP 和 1 个 FN
    preds = [(1, 0.9, (0.0, 0.0, 10.0, 10.0))]
    gts = [(0, 0.0, 0.0, 10.0, 10.0)]
    matched, unmatched_pred, unmatched_gt = evaluation.match_boxes(preds, gts, 0.5)
    assert matched == []
    assert unmatched_pred == [0]
    assert unmatched_gt == [0]


def test_match_boxes_low_iou_is_not_matched():
    preds = [(0, 0.9, (0.0, 0.0, 10.0, 10.0))]
    gts = [(0, 8.0, 8.0, 18.0, 18.0)]  # IoU ≈ 0.04
    matched, unmatched_pred, unmatched_gt = evaluation.match_boxes(preds, gts, 0.5)
    assert matched == []
    assert unmatched_pred == [0] and unmatched_gt == [0]


def test_match_boxes_greedy_prefers_highest_confidence():
    # 两个预测框抢同一个标注框：置信度高的应拿到 TP，另一个算 FP
    preds = [(0, 0.3, (0.0, 0.0, 10.0, 10.0)), (0, 0.8, (0.0, 0.0, 10.0, 10.0))]
    gts = [(0, 0.0, 0.0, 10.0, 10.0)]
    matched, unmatched_pred, unmatched_gt = evaluation.match_boxes(preds, gts, 0.5)
    assert matched == [1]
    assert unmatched_pred == [0]
    assert unmatched_gt == []


def test_match_boxes_empty_inputs():
    assert evaluation.match_boxes([], []) == ([], [], [])
    matched, unmatched_pred, unmatched_gt = evaluation.match_boxes([], [(0, 0, 0, 5, 5)], 0.5)
    assert matched == [] and unmatched_pred == [] and unmatched_gt == [0]


# ---------------------------------------------------------------------------
# 标签解析
# ---------------------------------------------------------------------------


def test_read_yolo_labels_pixels():
    with tempfile.TemporaryDirectory() as tmp:
        label = Path(tmp) / "a.txt"
        label.write_text("0 0.5 0.5 0.5 0.5\n1 0.5 0.25 1.0 0.5\n", encoding="utf-8")
        boxes = evaluation.read_yolo_labels(label, 200, 100)
        assert len(boxes) == 2
        cls, x1, y1, x2, y2 = boxes[0]
        assert cls == 0
        assert (round(x1), round(y1), round(x2), round(y2)) == (50, 25, 150, 75)
        assert boxes[1][0] == 1


def test_read_yolo_labels_missing_file_returns_empty():
    boxes = evaluation.read_yolo_labels(Path("不存在的标签.txt"), 100, 100)
    assert boxes == []


def test_read_yolo_labels_bad_field_count_raises():
    with tempfile.TemporaryDirectory() as tmp:
        label = Path(tmp) / "b.txt"
        label.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
        try:
            evaluation.read_yolo_labels(label, 100, 100)
        except ValueError as exc:
            assert "字段数不是 5" in str(exc)
        else:
            raise AssertionError("字段数不足时应抛 ValueError")


# ---------------------------------------------------------------------------
# 权重解析
# ---------------------------------------------------------------------------


def test_resolve_weights_missing_raises_with_hint():
    try:
        evaluation.resolve_weights("models/绝对不存在的权重.pt")
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "权重" in msg
        assert "train.py" in msg or "--weights" in msg
    else:
        raise AssertionError("权重不存在时应抛 FileNotFoundError")


def test_default_weights_never_coco():
    available = evaluation.list_trained_weights()
    if not available:
        print("[跳过] models 下暂无训练产物，无法验证默认权重选择")
        return
    for w in available:
        assert w.name.endswith(("_best.pt", "_last.pt")), f"{w.name} 不应出现在候选里"
        assert w.name not in evaluation._COCO_WEIGHTS
    default = evaluation.resolve_weights(None)
    assert default.name.endswith(("_best.pt", "_last.pt"))
    assert default.name != "yolov8n.pt"


def test_weights_info_has_sha256():
    available = evaluation.list_trained_weights()
    if not available:
        print("[跳过] 无训练产物")
        return
    info = evaluation.weights_info(available[0])
    assert info["bytes"] > 0
    assert len(info["sha256"]) == 64
    assert info["name"].endswith(".pt")


# ---------------------------------------------------------------------------
# 真实数据集统计（不跑模型，只读标签）
# ---------------------------------------------------------------------------


def test_count_test_objects_matches_split():
    img_dir, _ = paths.split_dirs("test")
    if not img_dir.is_dir() or not any(img_dir.glob("*.jpg")):
        print("[跳过] test 集为空，无法验证统计")
        return
    info = evaluation.count_test_objects("test")
    assert info["images"] > 0
    assert info["objects"] >= 0
    assert set(info["per_class"]) >= {"smoke", "fire"}
    # 图片数应与目录里的实际文件数一致
    actual = sum(1 for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    assert info["images"] == actual


# ---------------------------------------------------------------------------
# 报告渲染与落盘（用合成数据，不需要模型）
# ---------------------------------------------------------------------------


def _fake_result() -> dict:
    return {
        "schema_version": evaluation.SCHEMA_VERSION,
        "stage": "阶段八：模型评价与实验对比",
        "generated_at": "2026-10-28T21:00:00+08:00",
        "started_at": "2026-10-28T20:59:00+08:00",
        "seconds_total": 1.0,
        "weights": {"path": "models/x_best.pt", "name": "x_best.pt", "bytes": 123, "mtime": "2026-10-28", "sha256": "a" * 64},
        "dataset": {
            "name": "D-Fire",
            "split": "test",
            "images": 10,
            "objects": 20,
            "empty_label_images": 3,
            "objects_per_class": {"smoke": 8, "fire": 12},
            "split_method": "近似分组",
        },
        "config": {
            "imgsz": 640,
            "batch": 8,
            "device": "0",
            "map_conf": 0.001,
            "map_iou": 0.7,
            "error_conf": 0.25,
            "match_iou": 0.5,
            "workers": 0,
        },
        "overall": {"precision": 0.5, "recall": 0.4, "map50": 0.45, "map50_95": 0.3, "f1": 0.4444},
        "per_class": [
            {"class_id": 0, "class_name": "smoke", "instances": 8, "precision": 0.5, "recall": 0.4, "map50": 0.45, "map50_95": 0.3},
            {"class_id": 1, "class_name": "fire", "instances": 12, "precision": 0.6, "recall": 0.5, "map50": 0.55, "map50_95": 0.35},
        ],
        "speed_ms_per_image": {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
        "val_seconds": 12.3,
        "error_cases": {
            "iou_threshold": 0.5,
            "conf_threshold": 0.25,
            "true_positives": 5,
            "false_positives": 2,
            "false_negatives": 3,
            "fn_per_class": {"smoke": 2, "fire": 1},
            "fp_per_class": {"fire": 2},
            "images_with_fp": 2,
            "images_with_fn": 3,
            "images_scanned": 10,
            "saved_fn": ["outputs/predictions/x/missed/a__missed.jpg"],
            "saved_fp": ["outputs/predictions/x/false_alarm/b__false_alarm.jpg"],
            "seconds": 5.0,
        },
        "error_case_details": {"dir": "outputs/predictions/x", "missed_examples": [], "false_alarm_examples": []},
        "figures": ["outputs/figures/eval_x/confusion_matrix.png"],
        "limitations": ["近似分组", "非纯森林场景"],
        "disclaimer": evaluation.DISCLAIMER,
    }


def test_render_markdown_contains_required_sections():
    md = evaluation.render_markdown(_fake_result())
    for needle in ("# 测试集评价报告", "Precision", "Recall", "mAP50", "mAP50-95",
                   "分类别指标", "错误案例分析", "实验一", "实验二", "实验三"):
        assert needle in md, f"markdown 缺少 {needle}"
    # 未完成的实验必须明确写“未完成”，不能留空
    assert "**未完成**" in md
    # 免责声明必须在
    assert "需人工核实" in md


def test_render_markdown_without_error_cases_is_honest():
    data = _fake_result()
    data["error_cases"] = None
    data["error_case_details"] = None
    md = evaluation.render_markdown(data)
    assert "未收集错误案例" in md
    assert "不是“没有错误”" in md


def test_save_metrics_writes_both_files():
    with tempfile.TemporaryDirectory() as tmp:
        json_path, md_path = evaluation.save_metrics(_fake_result(), Path(tmp))
        assert json_path.is_file() and md_path.is_file()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["overall"]["map50"] == 0.45
        assert "测试集评价报告" in md_path.read_text(encoding="utf-8")


def test_rebuild_reports_requires_existing_json():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            evaluation.rebuild_reports(Path(tmp))
        except FileNotFoundError as exc:
            assert "metrics.json" in str(exc)
        else:
            raise AssertionError("缺少 metrics.json 时应抛 FileNotFoundError")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def test_cli_parser_defaults():
    import evaluate as evaluate_cli

    args = evaluate_cli.build_parser().parse_args([])
    assert args.split == "test"
    assert args.imgsz == 640
    assert args.batch == 8
    assert args.conf == 0.25
    assert args.no_error_cases is False
    assert args.rebuild is False
    helptext = evaluate_cli.build_parser().format_help()
    assert "evaluate.py" in helptext
    assert "test 集只用于最终评价" in helptext


def test_cli_bad_conf_returns_usage_error():
    import evaluate as evaluate_cli

    if not any(paths.split_dirs("test")[0].glob("*.jpg")):
        print("[跳过] test 集为空，跳过 CLI 参数校验测试")
        return
    args = evaluate_cli.build_parser().parse_args(["--conf", "1.5"])
    assert evaluate_cli.run(args) == evaluate_cli.EXIT_USAGE


def test_cli_missing_weights_returns_weights_error():
    import evaluate as evaluate_cli

    if not any(paths.split_dirs("test")[0].glob("*.jpg")):
        print("[跳过] test 集为空，跳过 CLI 权重测试")
        return
    args = evaluate_cli.build_parser().parse_args(["--weights", "models/不存在.pt"])
    assert evaluate_cli.run(args) == evaluate_cli.EXIT_WEIGHTS


# ---------------------------------------------------------------------------
# 真实评价（有 GPU/权重才跑，否则跳过）
# ---------------------------------------------------------------------------


def test_real_evaluation_on_small_subset():
    """真跑一次极小规模评价：验证 val → 分类别 → 图表全链路。"""
    available = evaluation.list_trained_weights()
    if not available:
        print("[跳过] 无训练产物，跳过真实评价测试")
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("[跳过] 未安装 ultralytics，跳过真实评价测试")
        return

    with tempfile.TemporaryDirectory() as tmp:
        result = evaluation.evaluate(
            weights=available[0],
            split="val",
            imgsz=320,
            batch=4,
            device=0,
            with_error_cases=False,
            figures_dir=Path(tmp),
        )
        for key in evaluation.REQUIRED_METRICS:
            assert result["overall"][key] is not None, f"缺少指标 {key}"
            assert 0.0 <= result["overall"][key] <= 1.0
        assert result["dataset"]["images"] > 0
        assert len(result["per_class"]) == 2
        names = {c["class_name"] for c in result["per_class"]}
        assert names == {"smoke", "fire"}


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {name}")
    print(f"\n合计 {len(tests)} 项：通过 {passed}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
