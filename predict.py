"""YOLO 森林烟雾检测 —— 推理命令行入口（阶段七）。

功能
----
1. 单张图片推理：``python predict.py --source demo.jpg``
2. 目录批量推理：``python predict.py --source some_dir``（``--recursive`` 递归子目录）
3. 可调置信度 / IoU / 输入尺寸：``--conf 0.4 --iou 0.5 --imgsz 640``
4. 保存带框的可视化结果图到 ``outputs/predictions/<run>/``
5. 输出结构化结果 JSON + CSV（**无目标的图片也会出现在结果里**，不会报错退出）
6. 记录权重路径与 sha256、输入路径、阈值、设备、每张图的推理耗时

退出码
------
0 全部处理完成（包含"没检测到目标"的正常情况）
1 未预期的异常
2 参数错误 / 输入路径不存在
3 权重缺失或依赖缺失（会给出下一步该做什么）
4 输入目录里没有可识别的图片
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src import paths

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_WEIGHTS = 3
EXIT_NO_IMAGES = 4


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（单独成函数，便于测试）。"""
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="用训练好的 YOLO 权重检测图片中的 smoke / fire（结果仅供辅助分析，需人工核实）",
        epilog=(
            "示例：\n"
            "  .\\.venv\\Scripts\\python.exe predict.py --source demo.jpg\n"
            "  .\\.venv\\Scripts\\python.exe predict.py --source datasets\\forest_smoke\\images\\test --limit 50\n"
            "  .\\.venv\\Scripts\\python.exe predict.py --source some_dir --recursive --conf 0.4\n"
            "  .\\.venv\\Scripts\\python.exe predict.py --source demo.jpg --weights models/xxx_best.pt --no-save-image\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="图片文件或图片目录")
    parser.add_argument("--weights", default=None, help="权重路径；默认自动挑 models/ 下最新的 *_best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 0–1（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值 0–1（默认 0.45）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸（默认 640，与训练一致）")
    parser.add_argument("--device", default=None, help="推理设备：0 / cpu（默认自动）")
    parser.add_argument("--out", default=None, help="结果输出目录（默认 outputs/predictions/predict_<时间戳>）")
    parser.add_argument("--json", dest="json_path", default=None, help="JSON 结果路径（默认 <out>/results.json）")
    parser.add_argument("--csv", dest="csv_path", default=None, help="CSV 结果路径（默认 <out>/results.csv）")
    parser.add_argument("--no-save-image", action="store_true", help="只算结果，不保存可视化图片")
    parser.add_argument("--recursive", action="store_true", help="目录模式下递归子目录")
    parser.add_argument("--limit", type=int, default=None, help="目录模式最多处理多少张")
    parser.add_argument("--quiet", action="store_true", help="只输出最终摘要")
    return parser


def _resolve_out_dir(raw: str | None) -> Path:
    if raw:
        out = Path(raw)
        return out if out.is_absolute() else paths.PROJECT_ROOT / out
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return paths.PREDICTIONS_DIR / f"predict_{stamp}"


def run(args: argparse.Namespace) -> int:
    """执行一次推理任务，返回退出码。"""
    from src import inference  # 延迟导入：缺 ultralytics 也能打印参数帮助

    source = Path(args.source)
    if not source.is_absolute():
        # 先按当前工作目录解析；找不到再按项目根解析，命令行更宽容
        source = source if source.exists() else paths.PROJECT_ROOT / args.source
    if not source.exists():
        print(f"[错误] 输入路径不存在: {args.source}")
        print("       请检查路径，或用 --source 指定图片文件/图片目录。")
        return EXIT_USAGE

    try:
        detector = inference.SmokeDetector(
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=not args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        return EXIT_WEIGHTS
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return EXIT_WEIGHTS
    except ValueError as exc:
        print(f"[错误] {exc}")
        return EXIT_USAGE

    out_dir = _resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = None if args.no_save_image else out_dir

    print(f"[模型] {detector.weights_path}")
    print(f"       类别 {detector.class_names}；conf={detector.conf} iou={detector.iou} imgsz={detector.imgsz}")
    print(f"       加载耗时 {detector.load_seconds} s；输出目录 {out_dir}")

    try:
        if source.is_file():
            target = None if save_dir is None else save_dir / f"{source.stem}__pred.jpg"
            results = [detector.predict_image(source, save_image=target)]
        else:
            results = detector.predict_dir(
                source,
                recursive=args.recursive,
                limit=args.limit,
                save_dir=save_dir,
            )
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        return EXIT_NO_IMAGES
    except NotADirectoryError as exc:
        print(f"[错误] {exc}")
        return EXIT_USAGE
    except ValueError as exc:
        print(f"[错误] {exc}")
        return EXIT_USAGE

    payload = inference.build_payload(detector, results, source=str(source), extra={"output_dir": str(out_dir)})
    json_path = Path(args.json_path) if args.json_path else out_dir / "results.json"
    csv_path = Path(args.csv_path) if args.csv_path else out_dir / "results.csv"
    inference.write_json_report(json_path, payload)
    inference.write_csv_report(csv_path, results)

    print()
    print(inference.format_summary_text(results))
    if not args.quiet:
        for result in results:
            flag = "  [无目标]" if result.num_boxes == 0 and not result.error else ""
            error = f"  [失败] {result.error}" if result.error else ""
            print(f"  - {result.image_name}: {result.num_boxes} 个框{flag}{error}")
    print()
    print(f"[文件] JSON: {json_path}")
    print(f"[文件] CSV : {csv_path}")
    if save_dir is not None:
        print(f"[文件] 可视化图片: {save_dir}")
    print()
    print(inference.DISCLAIMER)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消。")
        return 130
    except Exception:  # noqa: BLE001
        print("[错误] 发生未预期的异常，以下是完整堆栈：")
        traceback.print_exc()
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
