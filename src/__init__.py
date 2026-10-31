"""森林场景烟雾目标检测系统 —— 源码包。

模块规划（按计划书阶段顺序实现）：

    paths.py          项目路径统一入口（本阶段已实现，禁止硬编码绝对路径）
    environment.py    环境信息采集（本阶段已实现，只读，不安装任何东西）
    image_io.py       中文/空格路径安全的图片读写（本阶段已实现，作为 P2 备用防护）
    dataset_utils.py  数据集检查工具（阶段五实现）
    inference.py      推理封装（阶段七实现）
    evaluation.py     评价与指标（阶段八实现）

设计约定：
    1. 所有磁盘路径从 `src.paths` 取，禁止在业务代码里写带盘符的绝对路径；
    2. 依赖第三方库（cv2 / torch / ultralytics）的模块采用【函数内延迟导入】，
       这样在尚未安装依赖的阶段，模块依然可以被导入、被静态检查；
    3. 未实现的函数显式抛 NotImplementedError 并注明所属阶段，
       绝不返回假数据或伪造结果。
"""

from __future__ import annotations

__version__ = "0.1.0"
__stage__ = "阶段二：项目骨架（尚未安装依赖，未训练任何模型）"

__all__ = ["__version__", "__stage__"]
