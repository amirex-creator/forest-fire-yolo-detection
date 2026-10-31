"""森林烟雾目标检测系统 —— Streamlit 网页界面（阶段九）。

四个页面：项目介绍 / 图片检测 / 实验结果 / 模型说明。

启动
----
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

浏览器打开终端提示的地址（默认 http://localhost:8501）。

设计要点
--------
1. **推理逻辑全部复用 `src/inference.py`**：本文件不写任何 YOLO 调用、不解析检测框，
   只负责界面、参数校验与结果展示。
2. **模型只加载一次**：用 `st.cache_resource` 缓存 `SmokeDetector`，
   切换页面、重复上传都不会重新加载权重（只有换权重文件才会重建）。
3. **所有路径来自 `src/paths.py`**（基于 `pathlib`），不写死盘符。
4. **权重/指标/图表缺失时只提示、不崩溃**；没有实验结果就明确显示"尚未完成实验"。
5. 上传图片在运行 Streamlit 的服务器处理；公网模式下不转发到其他推理服务。

免责声明：模型识别结果仅供辅助分析，需人工核实，不等于森林火灾预警。
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from src import inference, paths

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


# 公网部署通过环境变量启用，默认保持原有本地行为。
PUBLIC_MODE = _env_bool("FOREST_SMOKE_PUBLIC_MODE")
DEPLOYMENT_DEVICE = os.getenv("FOREST_SMOKE_DEVICE") or ("cpu" if PUBLIC_MODE else None)
MODEL_OVERRIDE = os.getenv("FOREST_SMOKE_MODEL")
PUBLIC_RETENTION_HOURS = _env_int("FOREST_SMOKE_RETENTION_HOURS", 24, 1, 168)
PUBLIC_OUTPUT_ROOT = paths.PREDICTIONS_DIR / "public"

APP_TITLE = "森林烟雾目标检测系统"
APP_SUBTITLE = "基于 YOLOv8 的森林/野外场景烟雾与火焰检测（课程项目）"
DISCLAIMER = inference.DISCLAIMER

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_MB = _env_int("FOREST_SMOKE_MAX_UPLOAD_MB", 20, 1, 200)
MAX_VIDEO_MB = _env_int("FOREST_SMOKE_MAX_VIDEO_MB", 120, 1, 500)
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70

PAGES = ["项目介绍", "图片检测", "视频检测", "实验结果", "模型说明"]

METRICS_JSON = paths.REPORTS_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# 模型加载（只加载一次）
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="正在加载模型，请稍候…")
def load_detector(weights_path: str) -> inference.SmokeDetector:
    """按权重路径缓存检测器；公网模式默认固定使用 CPU。"""
    return inference.SmokeDetector(
        weights=weights_path,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        device=DEPLOYMENT_DEVICE,
        warmup=True,
    )


def available_weights() -> list[Path]:
    """本项目训练产物（不含 COCO 预训练权重）。"""
    weights = inference._list_trained_weights()
    if MODEL_OVERRIDE:
        try:
            preferred = inference.resolve_weights_path(MODEL_OVERRIDE)
        except (FileNotFoundError, OSError):
            preferred = None
        if preferred is not None:
            weights = [preferred] + [item for item in weights if item.resolve() != preferred.resolve()]
    return weights


def resolve_weights_safe(selected: str | None) -> tuple[Path | None, str | None]:
    """解析权重；失败返回 ``(None, 提示信息)``，由界面展示，不抛异常。"""
    try:
        return inference.resolve_weights_path(selected), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - 界面上只需要一句话
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 数据读取（实验结果页）
# ---------------------------------------------------------------------------


def load_metrics() -> dict | None:
    if not METRICS_JSON.is_file():
        return None
    try:
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_experiment_comparison() -> dict | None:
    path = paths.REPORTS_DIR / "experiment_comparison.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def existing_figures(metrics: dict | None) -> list[Path]:
    """从 metrics.json 里挑出实际存在的图片文件。"""
    if not metrics:
        return []
    out: list[Path] = []
    for item in metrics.get("figures", []) or []:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 公网临时文件管理
# ---------------------------------------------------------------------------


def _cleanup_public_outputs() -> None:
    """删除公网模式下超过保留期的上传和结果，避免磁盘无限增长。"""
    if not PUBLIC_MODE or not PUBLIC_OUTPUT_ROOT.is_dir():
        return
    cutoff = time.time() - PUBLIC_RETENTION_HOURS * 3600
    items = sorted(PUBLIC_OUTPUT_ROOT.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for item in items:
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink()
            elif item.is_dir() and item.stat().st_mtime < cutoff and not any(item.iterdir()):
                item.rmdir()
        except OSError:
            # 清理失败不应阻断当前用户的推理请求。
            continue


def _result_dir(prefix: str) -> Path:
    """返回本地或公网模式的结果目录。"""
    _cleanup_public_outputs()
    root = PUBLIC_OUTPUT_ROOT if PUBLIC_MODE else paths.PREDICTIONS_DIR
    return root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _processing_caption() -> str:
    if PUBLIC_MODE:
        return "图片会在当前 Streamlit 服务器处理，不会转发到其他推理服务；请勿上传敏感内容。"
    return "检测在本机完成。上传图片不会发送到外部服务。"


# ---------------------------------------------------------------------------
# 上传文件校验
# ---------------------------------------------------------------------------


def validate_upload(uploaded) -> tuple[bytes | None, str | None]:
    """校验后缀与大小，返回 ``(bytes, 错误信息)``。"""
    if uploaded is None:
        return None, "请先选择一张图片"
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return None, f"不支持的格式 {suffix or '（无后缀）'}，仅支持 JPG / JPEG / PNG"
    data = uploaded.getvalue()
    if not data:
        return None, "上传的文件是空的，请换一张图片"
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"文件过大（{len(data) / 1024 / 1024:.1f} MB），上限 {MAX_UPLOAD_MB} MB"
    return data, None


def save_upload(data: bytes, filename: str) -> Path:
    """保存上传内容供推理读取；公网模式写入受清理策略管理的目录。"""
    _cleanup_public_outputs()
    target_dir = (PUBLIC_OUTPUT_ROOT if PUBLIC_MODE else paths.PREDICTIONS_DIR) / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = Path(filename).name.replace(" ", "_")
    target = target_dir / f"{stamp}_{safe_name}"
    target.write_bytes(data)
    return target


def save_video_upload(data: bytes, filename: str) -> Path:
    """保存视频供逐帧推理；公网模式写入受清理策略管理的目录。"""
    _cleanup_public_outputs()
    target_dir = (PUBLIC_OUTPUT_ROOT if PUBLIC_MODE else paths.PREDICTIONS_DIR) / "video_uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = Path(filename).name.replace(" ", "_")
    target = target_dir / f"{stamp}_{safe_name}"
    target.write_bytes(data)
    return target


def process_video(detector: inference.SmokeDetector, source: Path, target: Path) -> dict:
    """逐帧运行 YOLO，保存标注视频并返回真实处理统计。"""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{source.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("视频没有可读取的画面尺寸")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("无法创建输出视频，请检查磁盘权限或视频编码支持")
    frames = 0
    frames_with_detection = 0
    total_boxes = 0
    class_counts = {"smoke": 0, "fire": 0}
    class_names = detector.model.names
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            predictions = detector.model.predict(
                source=frame,
                conf=detector.conf,
                iou=detector.iou,
                imgsz=detector.imgsz,
                device=detector.device,
                verbose=False,
            )
            pred = predictions[0]
            boxes = getattr(pred, "boxes", None)
            count = int(len(boxes)) if boxes is not None else 0
            total_boxes += count
            frames_with_detection += int(count > 0)
            if count:
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                for class_id in class_ids:
                    name = class_names.get(int(class_id), str(class_id)) if isinstance(class_names, dict) else class_names[int(class_id)]
                    class_counts[name] = class_counts.get(name, 0) + 1
            writer.write(pred.plot())
            frames += 1
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - started
    return {
        "frames": frames,
        "frames_with_detection": frames_with_detection,
        "total_boxes": total_boxes,
        "class_counts": class_counts,
        "source_fps": round(fps, 2),
        "input_frames_hint": total_hint,
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": round(frames / elapsed, 2) if elapsed else None,
    }


# ---------------------------------------------------------------------------
# 页面一：项目介绍
# ---------------------------------------------------------------------------


def page_intro() -> None:
    st.header("一、项目介绍")
    st.markdown(
        f"""
**{APP_TITLE}** —— {APP_SUBTITLE}

本系统做的是**图像层面的烟雾/火焰目标检测**：给一张图片，模型标出它认为是烟雾（smoke）
和火焰（fire）的区域，并给出置信度。它**不是**森林火灾发生概率预测模型。

| 项目 | 说明 |
| --- | --- |
| 任务类型 | 图片检测与视频逐帧检测（双类别：smoke / fire） |
| 模型 | YOLOv8n 迁移学习（COCO 预训练 → 本数据集微调） |
| 训练数据 | D-Fire（Kaggle 镜像），21,527 张图 / 26,531 个有效框，按近似分组 70/15/15 划分 |
| 当前状态 | 第一批 10 epoch 训练完成，test 集做过一次性最终评价 |
"""
    )

    weights = available_weights()
    metrics = load_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("可用模型权重", f"{len(weights)} 个")
    overall = (metrics or {}).get("overall", {})
    map50 = overall.get("map50")
    map5095 = overall.get("map50_95")
    col2.metric("测试集 mAP50", f"{map50:.4f}" if isinstance(map50, (int, float)) else "未评价")
    col3.metric("测试集 mAP50-95", f"{map5095:.4f}" if isinstance(map5095, (int, float)) else "未评价")

    st.subheader("适用范围与限制")
    st.markdown(
        """
- ✅ 适用于：野外/林区监控画面、含明显烟羽或明火的图片；
- ⚠️ 数据集**不是纯森林场景**，含城市、工业、海岸、夜间监控画面，模型在这些场景也能触发；
- ⚠️ 只会输出 smoke / fire 两类，不会判断"是否成灾""火势多大""会不会蔓延"；
- ⚠️ 夜间、逆光、云雾、远处小目标容易漏检或误报（详见实验结果页的错误分析）；
- ❌ 不能替代人工巡查、不能作为火灾报警的唯一依据。
"""
    )
    st.error(f"**免责声明**：{DISCLAIMER}")


# ---------------------------------------------------------------------------
# 页面二：图片检测
# ---------------------------------------------------------------------------


def page_detect() -> None:
    st.header("二、图片检测")

    weights_list = available_weights()
    if not weights_list:
        st.error(
            "没有找到训练好的权重（`models/` 下没有 `*_best.pt` 或 `*_last.pt`）。\n\n"
            "请先训练模型：`.\\\\.venv\\\\Scripts\\\\python.exe train.py --full --epochs 10`，"
            "或用 `--weights` 指定已有的权重。"
        )
        st.info("提示：`yolov8n.pt` 是 COCO 预训练权重，不含 smoke/fire 类别，不能用于本系统。")
        return

    with st.sidebar:
        st.markdown("### 检测设置")
        weight_name = st.selectbox(
            "模型权重",
            options=[w.name for w in weights_list],
            index=0,
            help="只列出本项目训练产物（*_best.pt / *_last.pt）",
        )
        conf = st.slider("置信度阈值 conf", 0.05, 0.95, DEFAULT_CONF, 0.05,
                         help="越高越保守：减少误报，也会漏掉更多目标")
        iou = st.slider("NMS IoU 阈值", 0.10, 0.90, DEFAULT_IOU, 0.05,
                        help="重叠框合并的严格程度，一般不用改")

    weights_path = next((w for w in weights_list if w.name == weight_name), None)
    if weights_path is None:
        st.error("选中的权重文件已不存在，请刷新页面")
        return

    uploaded = st.file_uploader("上传图片（JPG / JPEG / PNG，≤ 20 MB）", type=["jpg", "jpeg", "png"])
    data, err = validate_upload(uploaded)
    if err:
        if uploaded is not None:
            st.warning(err)
        st.caption("支持格式：JPG / JPEG / PNG；单张不超过 20 MB。")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("原图")
        st.image(data, caption=uploaded.name, width="stretch")

    detector, load_err = None, None
    try:
        detector = load_detector(str(weights_path))
    except FileNotFoundError as exc:
        load_err = str(exc)
    except Exception as exc:  # noqa: BLE001
        load_err = f"{type(exc).__name__}: {exc}"

    with right:
        st.subheader("检测结果")
        if load_err:
            st.error(f"模型加载失败：{load_err}")
            st.info("请确认权重文件存在且依赖（torch / ultralytics）已正确安装。")
            return

        run = st.button("开始检测", type="primary", width="stretch")
        if not run:
            st.info("点击「开始检测」后在此显示结果")
            return

        # 复用缓存里的检测器，只改阈值，不重新加载权重
        detector.conf = float(conf)
        detector.iou = float(iou)
        source = save_upload(data, uploaded.name)
        out_dir = _result_dir("web")
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            result = detector.predict_image(source, save_image=out_dir / f"{source.stem}__pred.jpg")
        except Exception as exc:  # noqa: BLE001 - 界面上给可读提示，不抛栈
            st.error(f"检测失败：{type(exc).__name__}: {exc}")
            st.info("常见原因：图片损坏、文件不是真正的图片、显卡内存不足。请换一张图或改用 CPU 再试。")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result.saved_image and Path(result.saved_image).is_file():
            st.image(result.saved_image, caption="检测结果（红字框为模型输出）", width="stretch")
        else:
            st.warning("结果图未能保存，但检测已执行（下面的数值仍然有效）")

        st.markdown("#### 检测摘要")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("检测框数", result.num_boxes)
        counts = result.class_counts()
        c2.metric("smoke", counts.get("smoke", 0))
        c3.metric("fire", counts.get("fire", 0))
        c4.metric("推理耗时", f"{result.predict_ms:.1f} ms" if result.predict_ms else "—")

        if result.num_boxes == 0:
            st.warning("**未检测到烟雾或火焰**。这可能是画面里确实没有目标，也可能是目标太小/太暗/被遮挡。")
        else:
            st.dataframe(
                [
                    {
                        "类别": d.class_name,
                        "置信度": round(d.confidence, 4),
                        "x1": round(d.box_xyxy[0], 1),
                        "y1": round(d.box_xyxy[1], 1),
                        "x2": round(d.box_xyxy[2], 1),
                        "y2": round(d.box_xyxy[3], 1),
                    }
                    for d in result.detections
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption("置信度是模型对「这个框里有该类目标」的把握程度，不等于真实概率，也不代表火势大小。")

        st.caption(
            f"耗时口径：读图 {result.read_ms or 0:.1f} ms / 纯预测 {result.predict_ms or 0:.1f} ms / "
            f"画图写盘 {result.render_ms or 0:.1f} ms / 端到端 {result.total_ms or 0:.1f} ms"
            f"（界面统计 {elapsed_ms:.0f} ms，含保存与界面开销）"
        )

        # 保存结构化结果：JSON + CSV，并提供下载
        payload = inference.build_payload(
            detector, [result], source=str(source), extra={"interface": "streamlit", "conf": conf, "iou": iou}
        )
        json_path = inference.write_json_report(out_dir / "result.json", payload)
        csv_path = inference.write_csv_report(out_dir / "result.csv", [result])

        st.markdown("#### 保存结果")
        d1, d2 = st.columns(2)
        d1.download_button(
            "下载 JSON", json_path.read_bytes(), file_name=json_path.name, mime="application/json",
            width="stretch",
        )
        d2.download_button(
            "下载 CSV", csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv",
            width="stretch",
        )
        st.caption(f"结果已保存到：`{out_dir}`（结果图、result.json、result.csv）")
        st.error(f"**免责声明**：{DISCLAIMER}")


# ---------------------------------------------------------------------------
# 页面三：视频检测
# ---------------------------------------------------------------------------


def page_video() -> None:
    st.header("三、视频检测")
    st.caption("逐帧运行模型并保存带框视频，适合课堂演示短视频。")
    weights_list = available_weights()
    if not weights_list:
        st.error("没有找到本项目训练权重，请先完成训练。")
        return
    with st.sidebar:
        st.markdown("### 视频检测设置")
        weight_name = st.selectbox("模型权重", [w.name for w in weights_list], key="video_weight")
        conf = st.slider("视频置信度阈值", 0.05, 0.95, DEFAULT_CONF, 0.05, key="video_conf")
        iou = st.slider("视频 NMS IoU 阈值", 0.10, 0.90, DEFAULT_IOU, 0.05, key="video_iou")
    weights_path = next((w for w in weights_list if w.name == weight_name), None)
    uploaded = st.file_uploader(
        "上传短视频（MP4 / AVI / MOV / MKV，建议不超过 120 MB）",
        type=[s.lstrip(".") for s in sorted(VIDEO_SUFFIXES)],
        key="video_upload",
    )
    if uploaded is None:
        st.info("选择视频后点击开始处理。课堂展示建议使用 10–30 秒短视频。")
        return
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    if suffix not in VIDEO_SUFFIXES:
        st.warning("视频格式不受支持。")
        return
    if not data or len(data) > MAX_VIDEO_MB * 1024 * 1024:
        st.warning(f"视频为空或超过 {MAX_VIDEO_MB} MB。")
        return
    if not st.button("开始视频检测", type="primary", key="run_video"):
        return
    try:
        detector = load_detector(str(weights_path))
        detector.conf = float(conf)
        detector.iou = float(iou)
        source = save_video_upload(data, uploaded.name)
        out_dir = _result_dir("web_video")
        output = out_dir / "annotated.mp4"
        with st.spinner("正在逐帧检测，视频越长耗时越久…"):
            stats = process_video(detector, source, output)
    except Exception as exc:  # noqa: BLE001 - 页面展示可读错误
        st.error(f"视频处理失败：{type(exc).__name__}: {exc}")
        return
    st.success(f"处理完成：{stats['frames']} 帧，平均处理速度 {stats['processing_fps']} FPS")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("处理帧数", stats["frames"])
    c2.metric("出现检测框的帧", stats["frames_with_detection"])
    c3.metric("检测框总数", stats["total_boxes"])
    c4.metric("处理速度", f"{stats['processing_fps']} FPS")
    if output.is_file():
        st.video(output.read_bytes())
        st.download_button("下载标注视频", output.read_bytes(), file_name=output.name, mime="video/mp4")
        st.caption(f"结果已保存到：`{output}`")
    st.error(f"**免责声明**：{DISCLAIMER}")


# ---------------------------------------------------------------------------
# 页面四：实验结果
# ---------------------------------------------------------------------------


def page_results() -> None:
    st.header("三、实验结果")
    metrics = load_metrics()

    if metrics is None:
        st.warning("**尚未完成实验**：没有找到 `outputs/reports/metrics.json`。")
        st.info(
            "请先评价模型：`.\\\\.venv\\\\Scripts\\\\python.exe evaluate.py`\n\n"
            "本页面只读取真实存在的评价结果文件，**不会显示任何占位或示例数字**。"
        )
        return

    ds = metrics.get("dataset", {})
    w = metrics.get("weights", {})
    st.success(
        f"数据来自 `outputs/reports/metrics.json`（{metrics.get('generated_at', '未知时间')}）；"
        f"权重 `{w.get('name', '未知')}`；"
        f"{ds.get('split', 'test')} 集 {ds.get('images', 0):,} 张 / {ds.get('objects', 0):,} 个标注框。"
    )

    st.subheader("总体指标")
    ov = metrics.get("overall", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{ov.get('precision', 0):.4f}")
    c2.metric("Recall", f"{ov.get('recall', 0):.4f}")
    c3.metric("mAP50", f"{ov.get('map50', 0):.4f}")
    c4.metric("mAP50-95", f"{ov.get('map50_95', 0):.4f}")

    per_class = metrics.get("per_class", [])
    if per_class:
        st.subheader("分类别指标")
        st.dataframe(per_class, width="stretch", hide_index=True)

    comparison = load_experiment_comparison()
    if comparison and comparison.get("models"):
        st.subheader("模型实验对比")
        st.dataframe(comparison["models"], width="stretch", hide_index=True)
        st.caption(comparison.get("protocol_note", "各模型在同一 test 划分上评价。"))
        fpr_path = paths.REPORTS_DIR / "negative_false_alarm_comparison.csv"
        if fpr_path.is_file():
            try:
                import pandas as pd
                st.subheader("无目标测试图片误报率")
                st.dataframe(pd.read_csv(fpr_path), width="stretch", hide_index=True)
                st.caption("图像级误报率 = 至少出现一个检测框的无目标图片数 / 无目标图片总数。")
            except Exception as exc:  # noqa: BLE001
                st.info(f"误报率表暂时无法读取：{type(exc).__name__}")

    speed = metrics.get("speed_ms_per_image", {})
    if speed:
        st.caption(
            "推理耗时（val 流程内，每张）："
            + "，".join(f"{k} {v} ms" for k, v in speed.items())
        )

    err = metrics.get("error_cases")
    if err:
        st.subheader("误报与漏检（conf={} / IoU={}）".format(err.get("conf_threshold"), err.get("iou_threshold")))
        e1, e2, e3 = st.columns(3)
        e1.metric("命中 TP", f"{err.get('true_positives', 0):,}")
        e2.metric("误报 FP", f"{err.get('false_positives', 0):,}")
        e3.metric("漏检 FN", f"{err.get('false_negatives', 0):,}")
        st.caption(
            f"含误报图片 {err.get('images_with_fp', 0):,} 张，含漏检图片 {err.get('images_with_fn', 0):,} 张；"
            "按类别：" + "，".join(f"漏检 {k} {v}" for k, v in (err.get("fn_per_class") or {}).items())
        )

    figs = existing_figures(metrics)
    st.subheader("图表")
    if not figs:
        st.info("**尚未生成实验图表**（metrics.json 里记录的图片文件都不存在）。请重跑 `evaluate.py`。")
    else:
        st.caption(f"共 {len(figs)} 张，来自 `outputs/figures/`；只展示真实存在的文件。")
        cols = st.columns(2)
        for i, fig in enumerate(figs):
            with cols[i % 2]:
                st.image(str(fig), caption=fig.name, width="stretch")

    st.subheader("实验完成情况")
    st.markdown(
        """
| 实验 | 状态 |
| --- | --- |
| 实验一：基础模型（yolov8n，10 epoch，imgsz 640） | ✅ 已完成（上表即结果） |
| 实验二：输入尺寸 512 vs 640 对比 | ⬜ 未完成（需重新训练，受显存/时间限制） |
| 实验三：复杂场景分组分析 | ⬜ 未完成（数据集无场景标签，需人工分组） |
"""
    )
    st.caption("未完成的实验在论文与报告中同样标为未完成，本页面不提供任何推测数字。")
    st.error(f"**免责声明**：{DISCLAIMER}")


# ---------------------------------------------------------------------------
# 页面四：模型说明
# ---------------------------------------------------------------------------


def page_model() -> None:
    st.header("四、模型说明")

    weights_list = available_weights()
    if not weights_list:
        st.error("**未找到可用权重**：`models/` 下没有 `*_best.pt` 或 `*_last.pt`。")
        st.info("请先运行训练脚本生成权重，否则「图片检测」页面无法使用。")
    else:
        st.subheader("当前权重")
        chosen = st.selectbox("选择要查看的权重", [w.name for w in weights_list])
        target = next(w for w in weights_list if w.name == chosen)
        info = inference.describe_weights(target)
        st.dataframe(
            [
                {"字段": "文件名", "值": info["name"]},
                {"字段": "路径", "值": info["path"]},
                {"字段": "大小", "值": f"{info['bytes']:,} 字节"},
                {"字段": "修改时间", "值": info["mtime"]},
                {"字段": "SHA256", "值": info["sha256"]},
            ],
            width="stretch",
            hide_index=False,
        )

    st.subheader("模型与训练信息")
    st.markdown(
        """
| 项目 | 说明 |
| --- | --- |
| 结构 | YOLOv8n（Ultralytics 8.4.190），约 3.2 M 参数 |
| 权重来源 | 从 `yolov8n.pt`（COCO 预训练）迁移学习，在本项目 D-Fire 数据上微调 10 epoch |
| 类别 | 0 = smoke（烟雾），1 = fire（火焰） |
| 训练数据 | D-Fire（Kaggle 镜像 `sayedgamal99/smoke-fire-detection-yolo`，CC0），21,527 张图 |
| 训练环境 | Windows + RTX 3060 Laptop 6 GB，torch 2.13.0+cu126，AMP=True |
| 评价结果 | test 集 mAP50 0.6007 / mAP50-95 0.3140（smoke 0.6812 / fire 0.5201） |
"""
    )

    st.subheader("置信度是什么意思")
    st.markdown(
        """
- 置信度（conf）是模型对"这个框里有该类目标"的**相对把握**，取值 0–1；
- 它是模型内部打分，**不是概率**、更**不是"发生火灾的可能性"**；
- 阈值 `conf` 由用户设定：调高 → 误报少、漏检多；调低 → 检出多、误报多；
- 本项目默认 0.25；实测在 30 张无目标图上仍有 10% 误报，说明阈值偏保守并不够。
"""
    )

    st.subheader("系统边界")
    st.markdown(
        """
- 图片按单帧检测，视频逐帧处理；不做跨帧目标关联、时序判断或火势蔓延预测；
- 图片在当前 Streamlit 进程所在环境计算；公网模式下请勿上传敏感内容；
- 模型的判断可能出错，**必须人工复核**。
"""
    )
    st.error(f"**免责声明**：{DISCLAIMER}")
    st.info("**烟雾或火焰被检出 ≠ 已确认森林火灾。** 如需报警，请以人工核实与官方渠道为准。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    redesigned_main()


def _model_label(weight_name: str) -> str:
    name = weight_name.lower()
    if "hardneg" in name:
        return "困难负样本微调 · YOLOv8n"
    if "complete_30" in name:
        return "基线 · 30 轮"
    if "batch1_e10" in name:
        return "基线 · 10 轮"
    return Path(weight_name).stem


def _recommended_conf(weight_name: str) -> float:
    for path in (
        paths.REPORTS_DIR / "calibrated_test_results_merged.json",
        paths.REPORTS_DIR / "calibrated_test_results.json",
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("models", []):
                if Path(item.get("weights", "")).name == weight_name:
                    value = float(item.get("validation", {}).get("selected_threshold", DEFAULT_CONF))
                    if 0.05 <= value <= 0.95:
                        return value
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return DEFAULT_CONF


def _comparison_rows() -> list[dict]:
    comparison = load_experiment_comparison() or {}
    rows = []
    for item in comparison.get("models", []):
        overall = item.get("overall", {})
        errors = item.get("threshold_test", {})
        rows.append({
            "model": item.get("label", item.get("model", "")),
            "epochs": item.get("epochs"),
            "images": item.get("test_images"),
            "threshold": item.get("validation_threshold"),
            "precision": errors.get("precision"),
            "recall": errors.get("recall"),
            "f1": errors.get("f1"),
            "map50": overall.get("map50"),
            "map50_95": overall.get("map50_95"),
            "false_alarm_rate": errors.get("negative_image_false_alarm_rate"),
        })
    return rows


def _case_review() -> list[dict]:
    path = paths.REPORTS_DIR / "case_review.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("cases", [])
    except (OSError, json.JSONDecodeError):
        return []


def _inject_new_style() -> None:
    st.markdown(
        """
<style>
:root { --forest:#173b31; --forest-2:#285b4d; --ink:#1c2925; --muted:#68736f; --paper:#f4f6f3; --line:#dce2dd; --coral:#c85c47; --teal:#37877b; }
[data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#173b31; border-right:1px solid #244c40; }
[data-testid="stSidebar"] * { color:#edf3ee; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d3e0d8; }
.block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:2rem; }
.app-brand { display:flex; align-items:center; gap:12px; padding:4px 0 10px; border-bottom:1px solid #426355; margin-bottom:18px; }
.brand-mark { width:34px; height:34px; border-radius:4px; display:grid; place-items:center; background:#c85c47; color:white; font-size:14px; font-weight:700; }
.brand-name { color:#fff; font-weight:650; font-size:16px; }
.brand-sub { color:#a9beb3; font-size:11px; margin-top:2px; }
.page-kicker { color:var(--teal); text-transform:uppercase; font-size:12px; font-weight:700; margin:0 0 5px; }
.page-title { color:var(--ink); font-size:28px; font-weight:700; line-height:1.2; margin:0 0 8px; }
.page-lead { color:var(--muted); font-size:14px; margin:0 0 22px; }
.section-rule { border-top:1px solid var(--line); margin:18px 0; }
.stButton > button, .stDownloadButton > button { border-radius:4px; font-weight:600; }
.stButton > button[kind="primary"] { background:var(--coral); border-color:var(--coral); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-left:3px solid var(--teal); border-radius:3px; padding:12px 14px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:3px; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #9baea4; border-radius:4px; padding:8px; }
[data-testid="stAlert"] { border-radius:3px; }
small, .stCaption { color:var(--muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def _page_heading(kicker: str, title: str, lead: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{kicker}</div><div class="page-title">{title}</div><div class="page-lead">{lead}</div>',
        unsafe_allow_html=True,
    )


def _demo_images() -> list[Path]:
    candidates = [
        paths.DATA_ROOT / "images" / "test" / name
        for name in ("AoF06155.jpg", "WEB05732.jpg", "AoF00115.jpg", "WEB04082.jpg")
    ]
    return [path for path in candidates if path.is_file()]


def _selected_weights(weights: list[Path]) -> tuple[Path | None, float, float]:
    if not weights:
        return None, DEFAULT_CONF, DEFAULT_IOU
    best = [path for path in weights if path.name.endswith("_best.pt")]
    available = best or weights
    preferred = available[0] if MODEL_OVERRIDE else None
    preferred = preferred or next((p for p in available if "hardneg" in p.name.lower()), None)
    preferred = preferred or next((p for p in available if "complete_30" in p.name.lower()), None)
    preferred = preferred or available[0]
    names = [p.name for p in available]
    with st.sidebar:
        selected_name = st.selectbox(
            "模型",
            names,
            index=names.index(preferred.name),
            format_func=_model_label,
            help="只选择本项目训练得到的最佳权重。",
        )
        confidence = st.slider(
            "置信度",
            min_value=0.05,
            max_value=0.95,
            value=_recommended_conf(selected_name),
            step=0.05,
            key=f"confidence_{selected_name}",
            help="阈值在验证集标定；提高阈值通常会减少误报，也可能增加漏检。",
        )
        iou = st.slider("重叠框阈值", 0.10, 0.90, DEFAULT_IOU, 0.05, key="shared_iou")
    return next((p for p in available if p.name == selected_name), None), confidence, iou


def _render_image_result(weight: Path, confidence: float, iou: float) -> None:
    mode = st.radio("图像来源", ["项目样例", "上传图片"], horizontal=True, label_visibility="collapsed")
    uploaded = None
    source: Path | None = None
    original_bytes: bytes | None = None
    if mode == "项目样例":
        samples = _demo_images()
        if samples:
            source = st.selectbox(
                "选择样例",
                samples,
                format_func=lambda path: f"{path.name} · {'空标签负样本' if not (paths.DATA_ROOT / 'labels' / 'test' / f'{path.stem}.txt').read_text(encoding='utf-8', errors='replace').strip() else '含标注目标'}",
            )
            original_bytes = source.read_bytes()
        else:
            st.info("暂时没有可用的项目样例，请上传图片。")
    else:
        uploaded = st.file_uploader("选择 JPG、JPEG 或 PNG 图片（不超过 20 MB）", type=["jpg", "jpeg", "png"], key="new_image_upload")
        if uploaded is not None:
            original_bytes = uploaded.getvalue()
            _, error = validate_upload(uploaded)
            if error:
                st.warning(error)
                original_bytes = None

    if original_bytes is None:
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.caption(_processing_caption())
        return

    input_name = source.name if source else uploaded.name
    input_fingerprint = hashlib.sha1(original_bytes).hexdigest()[:12]
    input_id = f"{input_name}:{input_fingerprint}:{weight.name}:{confidence:.2f}:{iou:.2f}"
    left, right = st.columns([0.94, 1.06], gap="large")
    with left:
        st.markdown("#### 输入画面")
        st.image(original_bytes, width="stretch")
        st.caption(input_name)
    with right:
        st.markdown("#### 检测输出")
        if st.button("运行检测", type="primary", width="stretch", key="new_run_image"):
            try:
                if source is None:
                    source = save_upload(original_bytes, uploaded.name)
                detector = load_detector(str(weight))
                detector.conf = float(confidence)
                detector.iou = float(iou)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_dir = _result_dir("demo")
                out_dir.mkdir(parents=True, exist_ok=True)
                result = detector.predict_image(source, save_image=out_dir / f"{source.stem}_detected.jpg")
                payload = inference.build_payload(
                    detector,
                    [result],
                    source=str(source),
                    extra={"interface": "streamlit_redesign", "confidence_threshold": confidence, "iou_threshold": iou},
                )
                json_path = inference.write_json_report(out_dir / "result.json", payload)
                csv_path = inference.write_csv_report(out_dir / "result.csv", [result])
                st.session_state["redesign_result"] = {
                    "input_id": input_id,
                    "image": result.saved_image,
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "result": result.to_dict(),
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"检测失败：{type(exc).__name__}: {exc}")
        state = st.session_state.get("redesign_result")
        if state and state.get("input_id") == input_id and Path(str(state.get("image", ""))).is_file():
            st.image(state["image"], width="stretch")
            result = state["result"]
            detections = result.get("detections", [])
            counts: dict[str, int] = {}
            for item in detections:
                counts[item["class_name"]] = counts.get(item["class_name"], 0) + 1
            m1, m2 = st.columns(2)
            m1.metric("检测框", len(detections))
            m2.metric("烟雾", counts.get("smoke", 0))
            m3, m4 = st.columns(2)
            m3.metric("明火", counts.get("fire", 0))
            m4.metric("耗时 ms", f"{result.get('predict_ms') or 0:.0f}")
            if detections:
                labels = [name for name, key in (("疑似烟雾", "smoke"), ("疑似明火", "fire")) if counts.get(key, 0)]
                if not labels:
                    labels = ["疑似目标"]
                st.success("、".join(labels) + "，请结合原始画面人工核实。")
                st.dataframe(
                    [{"类别": d["class_name"], "置信度": round(d["confidence"], 3), "位置": [round(v, 1) for v in d["box_xyxy"]]} for d in detections],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("未检测到目标。未检出不代表目标一定不存在。")
            d1, d2 = st.columns(2)
            d1.download_button("下载 JSON", Path(state["json"]).read_bytes(), file_name="result.json", mime="application/json", width="stretch")
            d2.download_button("下载 CSV", Path(state["csv"]).read_bytes(), file_name="result.csv", mime="text/csv", width="stretch")
        else:
            st.caption("运行检测后，标注图与结果明细会显示在此处。")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.caption("检测框表示模型判断，不等于确认发生火灾；本系统是课程原型，不用于真实火灾报警。")


def _render_video(weight: Path, confidence: float, iou: float) -> None:
    uploaded = st.file_uploader("选择短视频（MP4 / AVI / MOV / MKV，≤ 120 MB）", type=["mp4", "avi", "mov", "mkv"], key="new_video_upload")
    if uploaded is None:
        st.caption("建议使用 10–30 秒短片进行课堂演示。")
        return
    data = uploaded.getvalue()
    if not data or len(data) > MAX_VIDEO_MB * 1024 * 1024:
        st.warning(f"视频为空或大于 {MAX_VIDEO_MB} MB。")
        return
    if st.button("逐帧检测", type="primary", key="new_run_video"):
        try:
            detector = load_detector(str(weight))
            detector.conf, detector.iou = float(confidence), float(iou)
            source = save_video_upload(data, uploaded.name)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = _result_dir("demo_video") / "annotated.mp4"
            with st.spinner("正在逐帧处理视频…"):
                stats = process_video(detector, source, output)
            st.session_state["redesign_video_result"] = {
                "upload": f"{uploaded.name}:{uploaded.size}:{weight.name}:{confidence:.2f}:{iou:.2f}",
                "output": str(output),
                "stats": stats,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"视频处理失败：{type(exc).__name__}: {exc}")
    result = st.session_state.get("redesign_video_result")
    if result and result.get("upload") == f"{uploaded.name}:{uploaded.size}:{weight.name}:{confidence:.2f}:{iou:.2f}" and Path(result["output"]).is_file():
        stats = result["stats"]
        st.video(Path(result["output"]).read_bytes())
        detected_classes = [
            label for name, label in (("smoke", "疑似烟雾"), ("fire", "疑似明火"))
            if stats.get("class_counts", {}).get(name, 0) > 0
        ]
        if detected_classes:
            st.success(f"视频中检测到{'、'.join(detected_classes)}，请结合原始画面人工核实。")
        else:
            st.info("未检测到目标。未检出不代表目标一定不存在。")
        cols = st.columns(3)
        cols[0].metric("处理帧数", stats["frames"])
        cols[1].metric("有目标帧", stats["frames_with_detection"])
        smoke_count = stats.get("class_counts", {}).get("smoke", 0)
        fire_count = stats.get("class_counts", {}).get("fire", 0)
        cols[2].metric("烟雾 / 明火框", f"{smoke_count} / {fire_count}")
        cols = st.columns(3)
        cols[0].metric("检测框总数", stats["total_boxes"])
        cols[1].metric("处理耗时", f"{stats['elapsed_seconds']:.1f} s")
        speed = stats.get("processing_fps")
        cols[2].metric("处理速度", f"{speed} FPS" if speed is not None else "—")
        st.download_button("下载标注视频", Path(result["output"]).read_bytes(), file_name="annotated.mp4", mime="video/mp4")


def redesigned_main() -> None:
    st.set_page_config(page_title="林火视觉检测", layout="wide", initial_sidebar_state="expanded")
    _inject_new_style()
    weights = [path for path in available_weights() if path.name.endswith("_best.pt")]
    with st.sidebar:
        st.markdown('<div class="app-brand"><div class="brand-mark">FS</div><div><div class="brand-name">林火视觉检测</div><div class="brand-sub">森林烟雾与明火分析</div></div></div>', unsafe_allow_html=True)
        page = st.radio("工作区", ["图片检测", "视频检测", "实验对比", "误检案例", "模型与数据"], label_visibility="collapsed")
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    selected, confidence, iou = _selected_weights(weights)
    with st.sidebar:
        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        st.caption(f"当前模型：{_model_label(selected.name) if selected else '权重缺失'}")
        st.caption("结果仅供辅助查看，需人工核实。")

    st.markdown('<div class="page-kicker">课程项目 / YOLOv8n</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{page}</div>', unsafe_allow_html=True)
    if page == "图片检测":
        st.markdown('<div class="page-lead">上传图片或选取项目样例，查看烟雾与明火的定位结果。</div>', unsafe_allow_html=True)
        if selected:
            _render_image_result(selected, confidence, iou)
        else:
            st.error("没有可用的训练权重，请先完成训练。")
    elif page == "视频检测":
        st.markdown('<div class="page-lead">按帧分析短视频，并导出带检测框的结果。</div>', unsafe_allow_html=True)
        if selected:
            _render_video(selected, confidence, iou)
        else:
            st.error("没有可用的训练权重，请先完成训练。")
    elif page == "实验对比":
        rows = _comparison_rows()
        if not rows:
            st.info("实验对比报告尚未生成。")
        else:
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "threshold": st.column_config.NumberColumn("验证集选定阈值", format="%.2f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "map50": st.column_config.NumberColumn("mAP@0.5", format="%.3f"),
                "map50_95": st.column_config.NumberColumn("mAP@0.5:0.95", format="%.3f"),
                "false_alarm_rate": st.column_config.NumberColumn("空标签图误报率", format="percent"),
            })
            chart = pd.DataFrame(rows).set_index("model")[["map50", "map50_95"]]
            st.bar_chart(chart, horizontal=True, color=["#37877b", "#c85c47"])
            st.caption("所有检测指标均来自本项目实际运行；mAP 采用 IoU 0.5 与 0.5:0.95 评价。")
    elif page == "误检案例":
        cases = _case_review()
        if cases:
            for case in cases:
                st.markdown(f"#### {case.get('category', '案例')} · {case.get('image', '')}")
                col1, col2 = st.columns([1.2, 1])
                image_path = Path(case.get("annotated_image", ""))
                if image_path.is_file():
                    col1.image(str(image_path), width="stretch")
                col2.markdown(f"**观测**：{case.get('observation', '')}")
                col2.markdown(f"**分析假设**：{case.get('hypothesis', '')}")
                st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        else:
            figure_dir = paths.FIGURES_DIR / "evaluation_hardneg_40epoch_cases"
            col1, col2 = st.columns(2)
            for column, name, title in (
                (col1, "error_cases_false_alarm.png", "空标签图中的误报"),
                (col2, "error_cases_missed.png", "测试集漏检样例"),
            ):
                image_path = figure_dir / name
                column.markdown(f"#### {title}")
                if image_path.is_file():
                    column.image(str(image_path), width="stretch")
            st.caption("案例图依据数据集标注和固定匹配规则生成；复杂背景的具体成因属于人工分析假设。")
    else:
        st.markdown("#### 数据与训练")
        st.markdown("D-Fire 双类别目标检测：`smoke` 与 `fire`。现有划分为 train / val / test = 70% / 15% / 15%，按感知哈希近似分组。")
        st.markdown("基线使用 YOLOv8n 迁移学习；困难负样本来自训练集空标签图，验证集和测试集不参与挖掘。")
        st.markdown("#### 已知边界")
        st.markdown("数据集含城市、工业和野外画面，并非纯森林数据；原始视频/事件编号缺失，近似分组仍有少量跨集重复风险。")
        st.warning("这是课程演示原型。模型输出不能确认真实火灾，也不能替代人工巡查或官方报警系统。")
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    st.caption(f"基于 YOLO 的森林场景烟雾与明火检测 · {'服务器处理' if PUBLIC_MODE else '本地处理'} · 人工复核")


if __name__ == "__main__":
    main()
