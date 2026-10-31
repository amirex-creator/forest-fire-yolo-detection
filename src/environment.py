"""环境信息采集（只读，不安装、不下载任何东西）。

用途
----
1. 阶段二：验证项目骨架可正常导入与运行；
2. 阶段四：安装依赖后，用它一次性输出"真实"的环境记录，作为验证证据；
3. 全程只用标准库 —— 即使 torch / cv2 尚未安装也能运行，
   缺失的依赖会明确显示为 "未安装"，不会伪造版本号。

命令行用法::

    python -m src.environment                # 打印文本报告
    python -m src.environment --json         # 打印 JSON
    python -m src.environment --output outputs/reports/env.json   # 写入文件

注意：本模块**不判断 CUDA 是否可用**。`torch.cuda.is_available()` 仅供记录，
真正判定必须执行 GPU 矩阵运算（阶段四 V3），因为 is_available 为 True 也可能
在加载内核时失败。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from src import paths

#: 需要记录版本的依赖（包名: import 名仅用于提示，这里用发行包名查询）
TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "numpy",
    "pandas",
    "matplotlib",
    "streamlit",
    "PyYAML",
    "tqdm",
    "pytest",
    "psutil",
    "Pillow",
    "requests",
)


# --------------------------------------------------------------------------
# 各分项采集
# --------------------------------------------------------------------------
def python_info() -> dict[str, Any]:
    """Python 解释器信息。"""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "bits": 64 if sys.maxsize > 2**32 else 32,
        "is_venv": sys.prefix != sys.base_prefix,
        "venv_path": sys.prefix if sys.prefix != sys.base_prefix else None,
    }


def os_info() -> dict[str, Any]:
    """操作系统信息。"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def memory_info() -> dict[str, Any]:
    """物理内存信息。

    Windows 上使用 Win32 API ``GlobalMemoryStatusEx``（沙箱禁用了 WMI，
    因此不用 Get-CimInstance）。失败时返回 ``{"available": False, "error": ...}``，
    绝不编造数值。
    """
    if platform.system() == "Windows":
        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                raise OSError("GlobalMemoryStatusEx 返回失败")
            total = stat.ullTotalPhys / 1024**3
            available = stat.ullAvailPhys / 1024**3
            return {
                "available": True,
                "total_gb": round(total, 2),
                "available_gb": round(available, 2),
                "used_percent": round(stat.dwMemoryLoad, 1),
            }
        except Exception as exc:  # pragma: no cover - 平台相关
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:  # 非 Windows 兜底
        page = __import__("os").sysconf("SC_PAGE_SIZE")
        total = page * __import__("os").sysconf("SC_PHYS_PAGES") / 1024**3
        return {"available": True, "total_gb": round(total, 2)}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def disk_info(path: Path | None = None) -> dict[str, Any]:
    """项目所在磁盘的可用空间。"""
    target = Path(path) if path is not None else paths.PROJECT_ROOT
    # 项目目录若尚不存在，向上找到存在的祖先
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
        return {
            "available": True,
            "path": str(probe),
            "total_gb": round(usage.total / 1024**3, 1),
            "free_gb": round(usage.free / 1024**3, 1),
            "used_gb": round(usage.used / 1024**3, 1),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "path": str(probe), "error": f"{type(exc).__name__}: {exc}"}


def cpu_info() -> dict[str, Any]:
    """CPU 信息。"""
    return {
        "processor": platform.processor() or platform.machine(),
        "logical_cores": __import__("os").cpu_count(),
        "machine": platform.machine(),
    }


def package_versions() -> dict[str, str | None]:
    """已安装依赖的版本；未安装的记为 None（不猜版本）。"""
    result: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
        except Exception:  # pragma: no cover - 元数据损坏等
            result[name] = None
    return result


def torch_cuda_info() -> dict[str, Any]:
    """在 torch 可用时记录 CUDA 相关字段。

    这里只是"记录"，**不作为 CUDA 可用的证据**。真正判定见阶段四 V3
    （必须实际执行 GPU 矩阵运算）。
    """
    try:
        import torch  # noqa: PLC0415 - 延迟导入，未安装时不应影响本模块
    except Exception as exc:
        return {"torch_importable": False, "reason": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "note": "cuda_is_available 仅供参考；CUDA 是否真正可用必须由 GPU 矩阵运算实测判定",
    }
    if info["cuda_is_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "capability": torch.cuda.get_device_capability(0),
                    "vram_total_gb": round(props.total_memory / 1024**3, 2),
                }
            )
        except Exception as exc:  # pragma: no cover
            info["device_error"] = f"{type(exc).__name__}: {exc}"
    return info


def gpu_info_smi() -> dict[str, Any]:
    """通过 nvidia-smi 读取显卡信息（驱动层面，不依赖 torch）。

    实现说明：输出重定向到**临时文件**而不是管道。部分受限制的沙箱环境
    禁止创建匿名管道（``subprocess.PIPE`` 会抛 WinError 5），改用文件重定向
    后在该类环境下同样可用，且不需要额外权限。
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "reason": "未找到 nvidia-smi（可能没有 NVIDIA 驱动）"}

    query = "name,memory.total,memory.free,driver_version"
    tmpdir = Path(tempfile.mkdtemp(prefix="fs_smoke_smi_"))
    out_path = tmpdir / "stdout.txt"
    err_path = tmpdir / "stderr.txt"
    try:
        with out_path.open("w", encoding="utf-8", errors="replace") as fout, err_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as ferr:
            proc = subprocess.run(  # noqa: S603 - 固定参数，无 shell
                [exe, f"--query-gpu={query}", "--format=csv,noheader"],
                stdout=fout,
                stderr=ferr,
                timeout=20,
                check=False,
            )
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
        returncode = proc.returncode
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if returncode != 0:
        return {
            "available": False,
            "reason": f"nvidia-smi 退出码 {returncode}",
            "stderr": (stderr or "").strip()[:300],
        }

    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {"name": parts[0], "memory_total": parts[1], "memory_free": parts[2], "driver": parts[3]}
            )
    return {"available": bool(gpus), "gpus": gpus}


# --------------------------------------------------------------------------
# 汇总与输出
# --------------------------------------------------------------------------
def collect() -> dict[str, Any]:
    """采集全部环境信息。"""
    return {
        "collected_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": paths.describe(),
        "python": python_info(),
        "os": os_info(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "disk": disk_info(),
        "gpu_smi": gpu_info_smi(),
        "packages": package_versions(),
        "torch_cuda": torch_cuda_info(),
    }


def _fmt(value: Any) -> str:
    return "未安装" if value is None else str(value)


def format_text(report: dict[str, Any]) -> str:
    """把采集结果渲染成便于阅读的中文文本。"""
    lines: list[str] = []
    add = lines.append

    add("=" * 68)
    add("森林烟雾检测项目 —— 环境信息报告")
    add(f"采集时间: {report['collected_at']}")
    add("=" * 68)

    py = report["python"]
    add("\n[Python]")
    add(f"  版本        : {py['version']} ({py['bits']} bit, {py['implementation']})")
    add(f"  解释器      : {py['executable']}")
    add(f"  虚拟环境    : {'是 -> ' + str(py['venv_path']) if py['is_venv'] else '否（系统 Python）'}")
    add(f"  编译器      : {py['compiler']}")

    os_ = report["os"]
    add("\n[操作系统]")
    add(f"  {os_['system']} {os_['release']} ({os_['machine']})")
    add(f"  {os_['platform']}")

    cpu = report["cpu"]
    add("\n[CPU]")
    add(f"  {cpu['processor']} / {cpu['logical_cores']} 逻辑核心")

    mem = report["memory"]
    add("\n[内存]")
    if mem.get("available"):
        add(f"  总计 {mem['total_gb']} GB，可用 {mem.get('available_gb', '?')} GB，占用 {mem.get('used_percent', '?')}%")
    else:
        add(f"  读取失败: {mem.get('error')}")

    disk = report["disk"]
    add("\n[磁盘]")
    if disk.get("available"):
        add(f"  {disk['path']} 总计 {disk['total_gb']} GB，可用 {disk['free_gb']} GB")
    else:
        add(f"  读取失败: {disk.get('error')}")

    add("\n[GPU - nvidia-smi]")
    smi = report["gpu_smi"]
    if smi.get("available"):
        for gpu in smi["gpus"]:
            add(f"  {gpu['name']} | 显存 {gpu['memory_total']} (空闲 {gpu['memory_free']}) | 驱动 {gpu['driver']}")
    else:
        add(f"  不可用: {smi.get('reason')}")

    add("\n[Python 依赖]")
    for name, ver in report["packages"].items():
        mark = "OK  " if ver else "缺失"
        add(f"  [{mark}] {name:16s} {_fmt(ver)}")

    add("\n[torch / CUDA]")
    tc = report["torch_cuda"]
    if not tc.get("torch_importable"):
        add(f"  torch 未安装或无法导入: {tc.get('reason')}")
    else:
        add(f"  torch {tc.get('torch_version')} (CUDA build {tc.get('cuda_build')}, cuDNN {tc.get('cudnn_version')})")
        add(f"  cuda_is_available = {tc.get('cuda_is_available')}, device_count = {tc.get('device_count')}")
        if tc.get("device_name"):
            add(f"  设备: {tc['device_name']}, capability={tc.get('capability')}, 显存 {tc.get('vram_total_gb')} GB")
        add(f"  ※ {tc.get('note')}")

    add("\n" + "=" * 68)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.environment",
        description="采集并输出本机环境信息（只读，不安装任何依赖）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--output", type=Path, default=None, help="同时把结果写入指定文件（相对项目根的路径或绝对路径）")
    args = parser.parse_args(argv)

    report = collect()
    text = json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_text(report)
    print(text)

    if args.output is not None:
        out = args.output if args.output.is_absolute() else paths.PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, ensure_ascii=False, indent=2) if args.json else text
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"\n已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
