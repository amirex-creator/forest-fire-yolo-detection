"""YOLO 森林烟雾检测 —— 评价入口（阶段八）。

在 **独立 test 集** 上做一次性最终评价：Precision / Recall / mAP50 / mAP50-95
（总体与 smoke、fire 分类别），并收集误报、漏检样本做案例分析。

用法示例
--------
    # 1) 默认：自动选最新的 *_best.pt，跑 test 集 + 错误案例
    .\\.venv\\Scripts\\python.exe evaluate.py

    # 2) 指定权重与错误案例阈值
    .\\.venv\\Scripts\\python.exe evaluate.py ^
        --weights models\\batch1_e10_full_20261028_130129_best.pt --conf 0.4

    # 3) 只算指标，不收集错误案例（更快）
    .\\.venv\\Scripts\\python.exe evaluate.py --no-error-cases

    # 4) 只重写 metrics.md（不重跑模型）
    .\\.venv\\Scripts\\python.exe evaluate.py --rebuild

退出码
------
- 0 成功；1 未预期异常；2 参数错误；3 权重缺失；4 数据集缺失。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src import paths

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_WEIGHTS = 3
EXIT_DATASET = 4

PLANNED_STAGE = "阶段八"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="在 test 集上做一次性最终评价，输出 metrics.json / metrics.md 与图表",
        epilog=(
            "示例：\n"
            "  python evaluate.py\n"
            "  python evaluate.py --weights models/batch1_e10_full_20261028_130129_best.pt --conf 0.4\n"
            "  python evaluate.py --no-error-cases\n"
            "  python evaluate.py --rebuild\n"
            "\n注意：test 集只用于最终评价，不要用它反复调参；调参请用 val 集。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=str, default=None, help="权重路径；默认取 models 下最新的 *_best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="评价用划分，默认 test")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=8, help="批大小，默认 8")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 0 或 cpu；默认自动")
    parser.add_argument("--conf", type=float, default=0.25, help="错误案例分析用的置信度阈值，默认 0.25")
    parser.add_argument("--max-error-samples", type=int, default=12, help="每类错误最多保存多少张样本图")
    parser.add_argument("--error-limit", type=int, default=None, help="错误案例只扫描前 N 张图（调试用）")
    parser.add_argument("--no-error-cases", action="store_true", help="不收集错误案例（只算指标）")
    parser.add_argument("--rebuild", action="store_true", help="只用现有 metrics.json 重写 metrics.md，不重跑模型")
    parser.add_argument("--out", type=str, default=None, help="报告输出目录，默认 outputs/reports")
    parser.add_argument("--figures-dir", type=str, default=None, help="图表输出目录，默认 outputs/figures/eval_<时间戳>")
    parser.add_argument("--quiet", action="store_true", help="少打印")
    return parser


def _count_images(split: str) -> int:
    img_dir, _ = paths.split_dirs(split)
    if not img_dir.is_dir():
        return 0
    return sum(1 for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def run(args: argparse.Namespace) -> int:
    from src import evaluation

    out_dir = Path(args.out) if args.out else paths.REPORTS_DIR

    if args.rebuild:
        try:
            json_path, md_path = evaluation.rebuild_reports(out_dir)
        except FileNotFoundError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return EXIT_DATASET
        print(f"[重建] {json_path}")
        print(f"[重建] {md_path}")
        return EXIT_OK

    if _count_images(args.split) == 0:
        print(f"[错误] {args.split} 集没有图片，无法评价。数据目录：{paths.DATA_ROOT}", file=sys.stderr)
        return EXIT_DATASET

    try:
        weights = evaluation.resolve_weights(args.weights)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_WEIGHTS

    if not (0.0 < args.conf < 1.0):
        print(f"[错误] --conf 必须在 (0,1) 之间，当前 {args.conf}", file=sys.stderr)
        return EXIT_USAGE

    print("=" * 72)
    print(f"[评价] 权重  : {weights}")
    print(f"[评价] 划分  : {args.split}（{_count_images(args.split):,} 张图片）")
    print(f"[评价] 参数  : imgsz={args.imgsz} batch={args.batch} device={args.device or 'auto'}")
    print(f"[评价] 注意  : test 集只做一次性最终评价，不要用它调参")
    print("=" * 72)

    try:
        result = evaluation.evaluate(
            weights=weights,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            error_conf=args.conf,
            with_error_cases=not args.no_error_cases,
            max_error_samples=args.max_error_samples,
            error_limit=args.error_limit,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_WEIGHTS
    except Exception:  # noqa: BLE001 - 真实异常要完整打印，便于定位
        traceback.print_exc()
        print("[错误] 评价过程出现未预期异常，见上面的调用栈", file=sys.stderr)
        return EXIT_UNEXPECTED

    json_path, md_path = evaluation.save_metrics(result, out_dir)

    if not args.quiet:
        ov = result["overall"]
        print("")
        print("[结果] 总体指标（test 集）")
        print(f"        Precision {ov['precision']:.4f}   Recall {ov['recall']:.4f}")
        print(f"        mAP50     {ov['map50']:.4f}   mAP50-95 {ov['map50_95']:.4f}   F1 {ov['f1']:.4f}")
        print("")
        print("[结果] 分类别")
        for c in result["per_class"]:
            print(
                f"        {c['class_name']:6s} 实例 {c['instances']:>5,}  P {c['precision']:.4f}  "
                f"R {c['recall']:.4f}  mAP50 {c['map50']:.4f}  mAP50-95 {c['map50_95']:.4f}"
            )
        speed = result["speed_ms_per_image"]
        print("")
        print(
            f"[耗时] 预处理 {speed.get('preprocess')} / 推理 {speed.get('inference')} / "
            f"后处理 {speed.get('postprocess')} ms 每张"
        )
        err = result.get("error_cases")
        if err:
            print("")
            print(
                f"[错误] conf={err['conf_threshold']} IoU={err['iou_threshold']}："
                f"TP {err['true_positives']:,} / FP {err['false_positives']:,} / FN {err['false_negatives']:,}"
            )
            print(f"        误报图 {err['images_with_fp']:,} 张，漏检图 {err['images_with_fn']:,} 张")
        print("")
        print(f"[文件] JSON : {json_path}")
        print(f"[文件] Markdown : {md_path}")
        for fig in result.get("figures", [])[:12]:
            print(f"[图]   {fig}")
        print("")
        print(f"> {result['disclaimer']}")

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
