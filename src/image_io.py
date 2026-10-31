"""中文 / 空格路径安全的图片读写工具（P2 备用防护）。

为什么需要它
------------
Windows 版 OpenCV 的 ``cv2.imread`` / ``cv2.imwrite`` 通过 ANSI(C 字符串) API
打开文件，路径含中文时可能**静默失败**：`imread` 返回 `None`、`imwrite` 返回
`False`，都不抛异常。这类故障极难排查（表现为"图片读不到""训练集图片数为 0"）。

本项目已采用 P1 方案（项目根目录为纯英文路径），但按项目约束仍保留本模块作为
第二道防线：一旦项目被复制到含中文的目录，只要业务代码统一走 ``safe_imread`` /
``safe_imwrite``，就不会踩这个坑。

原理
----
``numpy.fromfile`` / ``ndarray.tofile`` 走 Python 的文件对象（Windows 下为宽字符
API），能正确打开任意 Unicode 路径；``cv2.imdecode`` / ``cv2.imencode`` 只在内存
中完成编解码，不接触文件系统。

约定
----
- 失败一律抛出明确异常（FileNotFoundError / ValueError / RuntimeError），
  **绝不返回 None 让调用方去猜**；
- 本模块对 cv2 / numpy 采用函数内延迟导入，未安装依赖时模块仍可导入，
  只在真正调用时报出可操作的错误信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: 项目允许读写的图片后缀（上传/推理/数据检查共用同一份定义）
ALLOWED_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

_MISSING_CV2_MESSAGE = (
    "缺少依赖 opencv-python（或 numpy）。\n"
    "本项目约定在项目内虚拟环境中安装，请在项目根目录执行：\n"
    "  .\\.venv\\Scripts\\Activate.ps1\n"
    "  python -m pip install \"opencv-python==4.14.0.94\"\n"
    "（完整安装步骤见 docs/compatibility_final.md 第六节）"
)


def _require_cv2() -> tuple[Any, Any]:
    """延迟导入 cv2 与 numpy，缺失时给出可操作的提示。"""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖未安装时的分支
        raise RuntimeError(f"{_MISSING_CV2_MESSAGE}\n原始错误: {exc}") from exc
    return cv2, np


def _validate_suffix(path: Path, kind: str = "图片") -> str:
    """校验后缀是否受支持，返回小写后缀。"""
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(
            f"不支持的{kind}格式: {path.name!r}（后缀 {suffix or '无'}）；"
            f"允许: {sorted(ALLOWED_IMAGE_SUFFIXES)}"
        )
    return suffix


def safe_imread(path: str | Path, flags: int | None = None) -> Any:
    """读取图片，兼容中文/空格路径。

    Parameters
    ----------
    path:
        图片路径（str 或 pathlib.Path）。
    flags:
        OpenCV 读取标志，默认 ``cv2.IMREAD_COLOR``（返回 BGR 三通道）。

    Returns
    -------
    numpy.ndarray
        BGR 图像数组。

    Raises
    ------
    FileNotFoundError
        文件不存在或不是普通文件。
    ValueError
        后缀不受支持、文件为空、或解码失败。
    RuntimeError
        未安装 opencv-python / numpy。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片不存在或不是普通文件: {p}")
    _validate_suffix(p)

    cv2, np = _require_cv2()
    read_flag = cv2.IMREAD_COLOR if flags is None else flags

    buffer = np.fromfile(str(p), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"图片文件为空（0 字节）: {p}")

    image = cv2.imdecode(buffer, read_flag)
    if image is None:
        raise ValueError(f"图片解码失败（文件可能已损坏或不是真实图片）: {p}")
    return image


def safe_imwrite(path: str | Path, image: Any, quality: int = 95) -> Path:
    """写出图片，兼容中文/空格路径。

    Parameters
    ----------
    path:
        目标路径，后缀决定编码格式（.jpg/.jpeg/.png/.bmp）。
    image:
        numpy 数组（BGR）。
    quality:
        JPEG 质量 1–100，仅对 .jpg/.jpeg 生效。

    Returns
    -------
    pathlib.Path
        实际写入的路径。

    Raises
    ------
    ValueError
        后缀不受支持、图像为空数组、或编码失败。
    RuntimeError
        未安装 opencv-python / numpy。
    """
    p = Path(path)
    suffix = _validate_suffix(p, kind="输出图片")
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"JPEG 质量必须在 1–100 之间: {quality}")

    cv2, np = _require_cv2()
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"待写入的图像为空或类型不正确（path={p}）")

    p.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if suffix in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"图片编码失败（shape={getattr(image, 'shape', None)}, dtype={getattr(image, 'dtype', None)}）: {p}")

    buffer.tofile(str(p))
    return p


def as_predict_source(path: str | Path) -> Any:
    """为 Ultralytics 准备输入源，绕开其内部的 ``cv2.imread``。

    返回内存中的 BGR 数组（``YOLO.predict(source=...)`` 可直接接受），
    从而不受工作目录是否含中文影响。
    """
    p = Path(path)
    if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
        return safe_imread(p)
    return str(p)


__all__ = ["ALLOWED_IMAGE_SUFFIXES", "safe_imread", "safe_imwrite", "as_predict_source"]
