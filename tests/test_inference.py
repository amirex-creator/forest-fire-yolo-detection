"""推理相关测试（阶段七）：真实模型推理 + 结果输出 + 错误路径。

覆盖计划书阶段十的第 1、2、4、5、6 项：

1. 权重不存在时给出明确提示，权重存在时能解析出真实路径；
2. 图片格式不正确时给出明确提示；
3. 无检测目标时输出空结果而不报错；
4. 正常图片能完成推理（真实加载 best.pt，耗时数秒）；
5. JSON / CSV 结构化结果可正确写出与读回；
6. ``predict.py`` 的 CLI 参数与错误退出码。

运行方式：

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_inference.py -q
    .\\.venv\\Scripts\\python.exe -m tests.test_inference     # 不装 pytest 也能跑
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import predict
from src import image_io, inference, paths


# --------------------------------------------------------------------------
# 基础工具与权重解析
# --------------------------------------------------------------------------
def test_allowed_suffixes() -> None:
    """允许的图片后缀与项目约定一致。"""
    assert ".jpg" in image_io.ALLOWED_IMAGE_SUFFIXES
    assert ".jpeg" in image_io.ALLOWED_IMAGE_SUFFIXES
    assert ".png" in image_io.ALLOWED_IMAGE_SUFFIXES
    assert ".txt" not in image_io.ALLOWED_IMAGE_SUFFIXES


def test_safe_imread_missing_file_raises() -> None:
    """文件不存在时抛 FileNotFoundError，而不是返回 None。"""
    ghost = paths.PROJECT_ROOT / "outputs" / "__not_exists__" / "ghost.jpg"
    try:
        image_io.safe_imread(ghost)
    except FileNotFoundError as exc:
        assert "ghost.jpg" in str(exc)
    else:
        raise AssertionError("safe_imread 对不存在的文件应当抛 FileNotFoundError")


def test_safe_imread_rejects_bad_suffix() -> None:
    """后缀不受支持时抛 ValueError。"""
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "note.txt"
        bad.write_text("not an image", encoding="utf-8")
        try:
            image_io.safe_imread(bad)
        except ValueError as exc:
            assert ".txt" in str(exc) or "格式" in str(exc)
        else:
            raise AssertionError("safe_imread 对 .txt 应当抛 ValueError")


def test_safe_imread_fake_png_reports_clearly() -> None:
    """伪造的 .png 必须报错，不能静默返回 None。"""
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake.png"
        fake.write_bytes(b"this is definitely not a png")
        try:
            image_io.safe_imread(fake)
        except RuntimeError as exc:
            assert "opencv-python" in str(exc)
        except ValueError as exc:
            assert "解码" in str(exc) or "损坏" in str(exc)
        else:
            raise AssertionError("伪造的 png 不应被成功读取")


def test_safe_imwrite_rejects_bad_suffix() -> None:
    """写图后缀不受支持时抛 ValueError。"""
    try:
        image_io.safe_imwrite(paths.PROJECT_ROOT / "outputs" / "x.xyz", object())
    except ValueError as exc:
        assert ".xyz" in str(exc) or "格式" in str(exc)
    else:
        raise AssertionError("safe_imwrite 对 .xyz 应当抛 ValueError")


def test_resolve_weights_explicit_missing_is_explicit() -> None:
    """显式指定不存在的权重时，必须是带引导的 FileNotFoundError。"""
    ghost = paths.PROJECT_ROOT / "models" / "__no_such_weight__.pt"
    try:
        inference.resolve_weights_path(ghost)
    except FileNotFoundError as exc:
        text = str(exc)
        assert "权重" in text
        assert "train.py" in text or "--weights" in text
    else:
        raise AssertionError("不存在的权重路径不应解析成功")


def test_resolve_weights_default_skips_coco_weights() -> None:
    """默认解析必须给出本项目训练出的权重，绝不能是 yolov8n.pt 这类 COCO 权重。"""
    try:
        path = inference.resolve_weights_path(None)
    except FileNotFoundError as exc:
        # 尚未训练时允许报错，但必须说清楚下一步
        assert "train.py" in str(exc)
        return
    assert path.is_file()
    assert path.name.lower().endswith(("_best.pt", "_last.pt")), f"不应挑到 {path.name}"
    assert path.name.lower() not in {n for n in ("yolov8n.pt", "yolov8s.pt")}


def test_detector_validates_thresholds() -> None:
    """非法阈值必须在加载模型前就被拒绝。"""
    for kwargs in ({"conf": 1.5}, {"conf": -0.1}, {"iou": 2.0}, {"imgsz": 8}):
        try:
            inference.SmokeDetector(**kwargs)
        except ValueError as exc:
            assert str(exc)
        else:
            raise AssertionError(f"非法参数 {kwargs} 应当抛 ValueError")


# --------------------------------------------------------------------------
# 数据结构与结果输出（不需要模型）
# --------------------------------------------------------------------------
def test_image_result_handles_empty_detections() -> None:
    """无目标时数据结构正常，且 JSON 里是空列表。"""
    result = inference.ImageResult(image_path="demo.jpg", detections=[], inference_ms=12.5)
    payload = result.to_dict()
    assert payload["num_boxes"] == 0
    assert payload["detections"] == []
    assert payload["inference_ms"] == 12.5


def test_summarize_counts_empty_and_failed_images() -> None:
    """汇总要把"无目标"和"失败"的图片单独计数，不能混为一起。"""
    with_box = inference.ImageResult(
        image_path="a.jpg",
        image_name="a.jpg",
        detections=[
            inference.Detection(0, "smoke", 0.9, (1.0, 2.0, 30.0, 40.0), (0.1, 0.2, 0.3, 0.4)),
            inference.Detection(1, "fire", 0.5, (5.0, 6.0, 20.0, 25.0)),
        ],
        inference_ms=10.0,
    )
    empty = inference.ImageResult(image_path="b.jpg", image_name="b.jpg", detections=[], inference_ms=8.0)
    failed = inference.ImageResult(image_path="c.jpg", image_name="c.jpg", error="ValueError: bad")
    summary = inference.summarize([with_box, empty, failed])
    assert summary["num_images"] == 3
    assert summary["num_images_with_detections"] == 1
    assert summary["num_images_without_detections"] == 2  # 包含失败的那张（它也没有框）
    assert summary["num_images_failed"] == 1
    assert summary["num_boxes"] == 2
    assert summary["boxes_per_class"] == {"fire": 1, "smoke": 1}
    assert summary["inference_ms_mean"] == 9.0


def test_write_json_and_csv_roundtrip() -> None:
    """JSON/CSV 必须能写出再读回，且行数与检测框数一致。"""
    results = [
        inference.ImageResult(
            image_path="x/a.jpg",
            image_name="a.jpg",
            detections=[inference.Detection(0, "smoke", 0.88, (1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4))],
            inference_ms=7.5,
        ),
        inference.ImageResult(image_path="x/b.jpg", image_name="b.jpg", detections=[], inference_ms=6.5),
    ]
    payload = {"generated_at": "test", "summary": inference.summarize(results), "images": [r.to_dict() for r in results]}
    with tempfile.TemporaryDirectory() as tmp:
        json_path = inference.write_json_report(Path(tmp) / "r.json", payload)
        csv_path = inference.write_csv_report(Path(tmp) / "r.csv", results)

        reloaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert reloaded["summary"]["num_images"] == 2
        assert reloaded["images"][1]["detections"] == []

        lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
        assert lines[0].split(",")[:3] == ["image_path", "image_name", "num_boxes"]
        assert len(lines) == 3, f"表头 + 1 个检测行 + 1 个空检测行，实际 {len(lines)} 行"


def test_collect_images_sorted_and_filtered() -> None:
    """目录扫描只收图片后缀，并按文件名排序（保证结果可复现）。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("b.jpg", "a.PNG", "c.txt", "d.jpeg"):
            (root / name).write_bytes(b"x")
        images = inference.collect_images(root)
        assert [p.name for p in images] == ["a.PNG", "b.jpg", "d.jpeg"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_parser_defaults_and_examples() -> None:
    """CLI 默认参数与帮助文本必须包含真实调用示例。"""
    parser = predict.build_parser()
    args = parser.parse_args(["--source", "demo.jpg"])
    assert args.conf == 0.25 and args.iou == 0.45 and args.imgsz == 640
    assert args.recursive is False and args.no_save_image is False
    assert "predict.py --source demo.jpg" in parser.epilog


def test_cli_missing_source_returns_usage_error() -> None:
    """输入路径不存在时退出码为 2，并提示路径问题。"""
    code = predict.main(["--source", "definitely_not_here_123.jpg"])
    assert code == predict.EXIT_USAGE


def test_cli_bad_weights_returns_weights_error() -> None:
    """权重缺失时退出码为 3，不能抛堆栈崩掉。"""
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "fake.jpg"
        img.write_bytes(b"\xff\xd8\xff\xd9")
        code = predict.main(["--source", str(img), "--weights", str(Path(tmp) / "nope.pt"), "--no-save-image"])
    assert code == predict.EXIT_WEIGHTS


# --------------------------------------------------------------------------
# 真实推理（需要 ultralytics + torch + 训练好的权重）
# --------------------------------------------------------------------------
def test_real_inference_on_dataset_image() -> None:
    """真实加载 best.pt 跑一张测试集图片：不报错、有耗时、结果结构完整。"""
    try:
        detector = inference.SmokeDetector(conf=0.25)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  [跳过] 无可用权重或依赖：{exc}")
        return

    candidates = inference.collect_images(paths.IMAGES_DIR / "test")
    assert candidates, "测试集里没有图片"
    result = detector.predict_image(candidates[0])

    assert result.image_width and result.image_width > 0
    assert result.image_height and result.image_height > 0
    assert result.inference_ms is not None and result.inference_ms > 0
    assert isinstance(result.detections, list)
    for det in result.detections:
        assert 0.0 <= det.confidence <= 1.0
        x1, y1, x2, y2 = det.box_xyxy
        assert x2 >= x1 and y2 >= y1
        assert det.class_name in set(detector.class_names.values())


def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    failed: list[tuple[str, str]] = []
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failed.append((name, str(exc)))
    print(f"\n合计 {len(tests)} 项，通过 {len(tests) - len(failed)} 项，失败 {len(failed)} 项")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
