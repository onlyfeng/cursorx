#!/usr/bin/env python3
"""兼容入口：self_iterate.py -> run_iterate.py

此文件仅作为向后兼容入口，保留旧脚本名称的调用方式。
所有功能已迁移至 scripts/run_iterate.py。

用法:
    # 以下两种调用方式等效：
    python scripts/self_iterate.py "任务描述"
    python scripts/run_iterate.py "任务描述"

注意:
    建议使用 python scripts/run_iterate.py 作为主要入口。
    此兼容入口将在未来版本中移除。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 发出弃用警告（可选，当前仅记录日志）
# warnings.warn(
#     "self_iterate.py 已弃用，请使用 run_iterate.py",
#     DeprecationWarning,
#     stacklevel=2,
# )

# 导入并调用 run_iterate 的 main 函数
from scripts.run_iterate import main

if __name__ == "__main__":
    main()
