"""测试包入口。

存在的目的：让 ``python -m tests.test_dataset`` / ``python -m tests.test_inference``
可以直接执行（不依赖 pytest）。阶段四装好 pytest 后，``pytest`` 仍可正常收集这些文件。
"""
