"""阶段四：环境验证与预训练权重推理自检（必须在项目 venv 内运行）。

运行：
    .venv\\Scripts\\python.exe src\\env_check.py

它做两件事：
  A) V1–V4 四项验证（版本对齐 / CUDA 可见性 / 真实 GPU 矩阵运算 / 显存与 Ultralytics）
  B) 用 COCO 预训练权重 yolov8n.pt 对一张测试集图片做单张推理最小测试

⚠️ 重要声明（不得省略）：yolov8n.pt 是 COCO 80 类权重，**不含 smoke / fire**。
   B 部分只证明"权重能加载、图片能过网络、能输出结果"，不是森林烟雾检测结果，
   也不能当作任何火灾预警结论。

输出：
  outputs/reports/stage4_env_verification.json   结构化验证结果
  outputs/reports/stage4_inference_result.json   结构化推理结果
  outputs/reports/stage4_verification.md         人类可读的完整记录
  outputs/predictions/stage4_pretrained/*.jpg    标注后的结果图
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "stage4_pretrained"
WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"
TEST_IMAGES = PROJECT_ROOT / "datasets" / "forest_smoke" / "images" / "test"
TEST_LABELS = PROJECT_ROOT / "datasets" / "forest_smoke" / "labels" / "test"

DISCLAIMER = (
    "COCO 预训练权重不含 smoke/fire 类别；本结果仅证明推理链路可运行，"
    "不是森林烟雾检测结果，也不是火灾预警结论。"
)

LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def banner(title: str) -> None:
    say()
    say("=" * 68)
    say(title)
    say("=" * 68)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v1() -> dict:
    banner("V1 版本对齐")
    import torch
    import torchvision

    say("[原始输出]")
    say(f"python {sys.version.replace(chr(10), ' ')}")
    say(f"executable {sys.executable}")
    say(f"torch {torch.__version__}")
    say(f"torchvision {torchvision.__version__}")
    say(f"cuda_build {torch.version.cuda}")
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001
        cudnn = f"<读取失败: {exc!r}>"
    say(f"cudnn {cudnn}")

    ok = (
        torch.__version__.startswith("2.13.0")
        and "+cu126" in torch.__version__
        and torchvision.__version__.startswith("0.28.0")
    )
    say(
        f"[判定] V1 {'PASS' if ok else 'FAIL'} —— torch={torch.__version__}"
        f"（需 2.13.0+cu126）, torchvision={torchvision.__version__}（需 0.28.0）"
    )
    return {
        "id": "V1",
        "name": "版本对齐",
        "pass": ok,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": cudnn,
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def v2() -> dict:
    banner("V2 CUDA 可见性")
    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if available else "N/A"
    cap = torch.cuda.get_device_capability(0) if available else "N/A"
    say("[原始输出]")
    say(f"cuda_available {available}")
    say(f"device_count {count}")
    say(f"name {name}")
    say(f"capability {cap}")
    ok = bool(available) and count >= 1
    say(f"[判定] V2 {'PASS' if ok else 'FAIL'} —— available={available}, count={count}")
    return {
        "id": "V2",
        "name": "CUDA 可见性",
        "pass": ok,
        "cuda_available": bool(available),
        "device_count": count,
        "device_name": name,
        "capability": list(cap) if isinstance(cap, tuple) else cap,
    }


def v3(cuda_ok: bool) -> dict:
    banner("V3 真实 GPU 矩阵运算（CUDA 可用的唯一可信证据）")
    import torch

    if not cuda_ok:
        say("[判定] V3 FAIL —— V2 未通过，按失败处理")
        return {"id": "V3", "name": "GPU 矩阵运算", "pass": False, "reason": "V2 failed"}
    say("[原始输出]")
    ok = False
    value = None
    error = None
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        value = float((a @ b).sum())
        torch.cuda.synchronize()
        say(f"matmul_ok {value}")
        ok = math.isfinite(value)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        say(f"matmul 异常 {error}")
    say(
        f"[判定] V3 {'PASS' if ok else 'FAIL'} —— "
        f"1024x1024 cuda matmul 求和={value}（有限值={ok}）"
    )
    return {
        "id": "V3",
        "name": "GPU 矩阵运算",
        "pass": ok,
        "matmul_sum": value,
        "error": error,
        "shape": [1024, 1024],
    }


def v4(cuda_ok: bool) -> dict:
    banner("V4 显存与 Ultralytics")
    import torch

    vram_gb = None
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1024**3, 2)
        say("[原始输出]")
        say(f"vram_total_GB {vram_gb}")
        say(f"device_name {props.name}")
        say(f"torch_cudnn_enabled {torch.backends.cudnn.enabled}")
    else:
        say("[原始输出]")
        say("vram_total_GB N/A（CUDA 不可用）")

    import ultralytics

    say(f"ultralytics {ultralytics.__version__}")
    import cv2

    say(f"opencv {cv2.__version__}")
    ok = ultralytics.__version__ == "8.4.190" and (vram_gb is None or 5.5 <= vram_gb <= 6.5)
    say(
        f"[判定] V4 {'PASS' if ok else 'FAIL'} —— vram_total_GB={vram_gb}, "
        f"ultralytics={ultralytics.__version__}（需 8.4.190）, opencv={cv2.__version__}"
    )
    return {
        "id": "V4",
        "name": "显存与 Ultralytics",
        "pass": ok,
        "vram_total_gb": vram_gb,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
    }


def pick_image() -> Path:
    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(f"测试集图片目录不存在：{TEST_IMAGES}")
    candidates = sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() == ".jpg")
    if not candidates:
        raise FileNotFoundError(f"测试集目录里没有 jpg 图片：{TEST_IMAGES}")
    for img in candidates:
        label = TEST_LABELS / f"{img.stem}.txt"
        if label.is_file() and label.read_text(encoding="utf-8").strip():
            return img
    return candidates[0]


def part_b_inference(cuda_ok: bool) -> dict:
    banner("B. 预训练权重单张图片推理最小测试")
    say(f"声明: {DISCLAIMER}")
    say(f"权重路径: {WEIGHTS}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"权重不存在：{WEIGHTS}\n请先下载 yolov8n.pt 到 models/ 目录，或指定已有权重。"
        )
    w_bytes = WEIGHTS.stat().st_size
    w_sha = sha256_of(WEIGHTS)
    say(f"[权重] 字节数 = {w_bytes}")
    say(f"[权重] SHA256 = {w_sha}")

    import torch
    from ultralytics import YOLO

    device = 0 if cuda_ok else "cpu"
    say(f"[设备] 使用 device={device}")
    image = pick_image()
    say(f"[输入] 图片 = {image}")
    say(f"[输入] 字节 = {image.stat().st_size}")

    t_load = time.perf_counter()
    model = YOLO(str(WEIGHTS))
    load_seconds = time.perf_counter() - t_load
    say(f"[模型] 加载耗时 = {load_seconds:.3f} 秒；类别数 = {len(model.names)}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = model.predict(
        source=str(image), conf=0.25, iou=0.45, imgsz=640, device=device, verbose=False
    )
    total_seconds = time.perf_counter() - t0
    res = results[0]
    saved = PRED_DIR / f"{image.stem}__pred.jpg"
    res.save(filename=str(saved))

    boxes = res.boxes
    n = 0 if boxes is None else len(boxes)
    dets: list[dict] = []
    if boxes is not None and n:
        for box in boxes:
            cls_id = int(box.cls[0])
            dets.append(
                {
                    "class_id": cls_id,
                    "class_name": res.names.get(cls_id, str(cls_id)),
                    "confidence": round(float(box.conf[0]), 6),
                    "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                }
            )
    speed = getattr(res, "speed", {}) or {}
    say("[推理结果]")
    say(f"  权重            : {WEIGHTS.name}")
    say(f"  输入图片        : {image.name}")
    say("  imgsz / conf/iou: 640 / 0.25 / 0.45")
    say(f"  device          : {device}")
    say(f"  推理总耗时      : {total_seconds * 1000:.1f} ms")
    say(
        "  预处理/推理/后处理: "
        f"{float(speed.get('preprocess', 0.0)):.1f} / "
        f"{float(speed.get('inference', 0.0)):.1f} / "
        f"{float(speed.get('postprocess', 0.0)):.1f} ms"
    )
    say(f"  检测框数量      : {n}")
    for d in dets:
        say(
            f"    - {d['class_name']} (id={d['class_id']}) conf={d['confidence']:.4f} "
            f"xyxy={d['xyxy']}"
        )
    say(f"  结果图          : {saved}")
    say("  [说明] 框数为 0 或类别与烟雾无关都属于预期：COCO 权重不认识森林烟雾。")

    return {
        "_disclaimer": DISCLAIMER,
        "weights": {"path": str(WEIGHTS), "bytes": w_bytes, "sha256": w_sha,
                    "classes": len(model.names)},
        "input": {"path": str(image), "bytes": image.stat().st_size},
        "params": {"imgsz": 640, "conf": 0.25, "iou": 0.45, "device": str(device)},
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timing": {
            "model_load_seconds": round(load_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 3),
            "inference_ms": round(float(speed.get("inference", 0.0)), 3),
            "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 3),
        },
        "detections": dets,
        "detection_count": n,
        "saved_image": str(saved),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    say("阶段四环境验证记录")
    say(f"生成时间: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    say(f"平台: {platform.platform()}")
    say(f"解释器: {sys.executable}")
    say(f"项目根: {PROJECT_ROOT}")

    checks = [v1(), v2()]
    cuda_ok = bool(checks[1]["pass"])
    checks.append(v3(cuda_ok))
    checks.append(v4(cuda_ok))
    gpu_ok = cuda_ok and bool(checks[2]["pass"])

    banner("汇总")
    all_pass = all(c["pass"] for c in checks)
    for c in checks:
        say(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['id']} {c['name']}")
    say(f"CUDA 结论: {'可用（V2+V3 均通过）' if gpu_ok else '不可用或不完全可用'}")
    say(f"V1–V4 总体: {'全部通过' if all_pass else '存在未通过项'}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "checks": checks,
        "cuda_usable": gpu_ok,
        "all_pass": all_pass,
    }
    (REPORTS_DIR / "stage4_env_verification.json").write_text(
        json.dumps(env_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    infer_error = None
    infer_payload: dict = {}
    try:
        infer_payload = part_b_inference(gpu_ok)
    except Exception:  # noqa: BLE001
        infer_error = traceback.format_exc()
        say("[错误] 推理部分失败：")
        say(infer_error)

    if infer_payload:
        (REPORTS_DIR / "stage4_inference_result.json").write_text(
            json.dumps(infer_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    header = [
        "# 阶段四环境验证记录（自动生成）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 平台：{platform.platform()}",
        f"- 解释器：`{sys.executable}`",
        f"- CUDA 结论：**{'可用' if gpu_ok else '不可用'}**（依据 V2 + V3）",
        f"- V1–V4：**{'全部通过' if all_pass else '存在未通过项'}**",
        "",
        "```text",
    ]
    md = "\n".join(header + LINES + ["```", ""])
    (REPORTS_DIR / "stage4_verification.md").write_text(md, encoding="utf-8")

    say()
    say(f"[记录] {REPORTS_DIR / 'stage4_env_verification.json'}")
    say(f"[记录] {REPORTS_DIR / 'stage4_verification.md'}")
    if infer_payload:
        say(f"[记录] {REPORTS_DIR / 'stage4_inference_result.json'}")

    if not all_pass:
        return 1
    if infer_error is not None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
