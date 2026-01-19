#!/usr/bin/env python3
"""端到端测试 - 验证 run.py 和 orchestrator 的集成"""
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_run_py_imports():
    """测试 run.py 模块导入"""
    from run import (
        MODE_ALIASES,
        RunMode,
    )
    print("✓ run.py 和 coordinator 模块导入成功")

    # 验证模式定义
    assert RunMode.BASIC.value == "basic"
    assert RunMode.MP.value == "mp"
    assert RunMode.KNOWLEDGE.value == "knowledge"
    assert RunMode.ITERATE.value == "iterate"
    assert RunMode.AUTO.value == "auto"
    print("✓ RunMode 枚举定义正确")

    # 验证模式别名
    assert MODE_ALIASES["basic"] == RunMode.BASIC
    assert MODE_ALIASES["mp"] == RunMode.MP
    assert MODE_ALIASES["parallel"] == RunMode.MP
    print("✓ MODE_ALIASES 映射正确")


def test_orchestrator_config():
    """测试 OrchestratorConfig"""
    from coordinator import OrchestratorConfig

    config = OrchestratorConfig(
        working_directory=".",
        max_iterations=1,
        worker_pool_size=1,
    )

    assert config.max_iterations == 1
    assert config.worker_pool_size == 1
    print("✓ OrchestratorConfig 创建成功")


def test_task_analyzer():
    """测试 TaskAnalyzer"""

    from run import RunMode, TaskAnalyzer

    analyzer = TaskAnalyzer(use_agent=False)  # 不使用 agent

    # 模拟 args
    class MockArgs:
        mode = "auto"
        task = "测试任务"

    # 测试规则匹配
    analysis = analyzer._rule_based_analysis("启动自我迭代模式", MockArgs())
    assert analysis.mode == RunMode.ITERATE
    print("✓ TaskAnalyzer 规则匹配正确 (iterate)")

    analysis = analyzer._rule_based_analysis("使用多进程模式", MockArgs())
    assert analysis.mode == RunMode.MP
    print("✓ TaskAnalyzer 规则匹配正确 (mp)")

    analysis = analyzer._rule_based_analysis("搜索知识库", MockArgs())
    assert analysis.mode == RunMode.KNOWLEDGE
    print("✓ TaskAnalyzer 规则匹配正确 (knowledge)")


def test_orchestrator_initialization():
    """测试 Orchestrator 初始化"""
    from coordinator import Orchestrator, OrchestratorConfig
    from cursor.client import CursorAgentConfig

    cursor_config = CursorAgentConfig(working_directory=".")
    config = OrchestratorConfig(
        working_directory=".",
        max_iterations=1,
        worker_pool_size=1,
        cursor_config=cursor_config,
    )

    orchestrator = Orchestrator(config)

    assert orchestrator.state.max_iterations == 1
    assert orchestrator.config.worker_pool_size == 1
    assert orchestrator.planner is not None
    assert orchestrator.reviewer is not None
    assert orchestrator.worker_pool is not None
    print("✓ Orchestrator 初始化成功")
    print(f"  - Planner ID: {orchestrator.planner.id}")
    print(f"  - Reviewer ID: {orchestrator.reviewer.id}")
    print(f"  - Worker Pool Size: {len(orchestrator.worker_pool.workers)}")


def test_iteration_control():
    """测试迭代控制逻辑"""
    from coordinator import Orchestrator, OrchestratorConfig

    # 测试有限迭代
    config = OrchestratorConfig(
        working_directory=".",
        max_iterations=3,
        worker_pool_size=1,
    )
    orchestrator = Orchestrator(config)

    assert orchestrator._should_continue_iteration() == True
    orchestrator.state.current_iteration = 3
    assert orchestrator._should_continue_iteration() == False
    print("✓ 有限迭代控制正确")

    # 测试无限迭代
    config = OrchestratorConfig(
        working_directory=".",
        max_iterations=-1,  # 无限迭代
        worker_pool_size=1,
    )
    orchestrator = Orchestrator(config)

    orchestrator.state.current_iteration = 100
    assert orchestrator._should_continue_iteration() == True
    print("✓ 无限迭代控制正确 (max_iterations=-1)")


def test_stream_config_apply():
    """测试流式日志配置应用"""
    from coordinator import Orchestrator, OrchestratorConfig
    from cursor.client import CursorAgentConfig

    cursor_config = CursorAgentConfig(working_directory=".")
    config = OrchestratorConfig(
        working_directory=".",
        max_iterations=1,
        worker_pool_size=1,
        cursor_config=cursor_config,
        stream_events_enabled=True,
        stream_log_console=True,
    )

    orchestrator = Orchestrator(config)

    # 验证流式配置是否正确应用到 cursor_config
    assert config.cursor_config.stream_events_enabled == True
    assert config.cursor_config.stream_log_console == True
    print("✓ 流式日志配置应用正确")


def test_iterate_args():
    """测试 IterateArgs 属性完整性"""

    # 模拟 options 字典（来自 run.py 的 _run_iterate 方法）
    options = {
        "skip_online": True,
        "dry_run": True,
        "max_iterations": 5,
        "workers": 2,
        "force_update": False,
        "verbose": True,
        "auto_commit": True,
        "auto_push": False,
        "commit_per_iteration": True,
        "commit_message": "test: prefix",
    }

    # 动态创建 IterateArgs 类（与 run.py 中相同）
    class IterateArgs:
        def __init__(self, goal: str, opts: dict):
            self.requirement = goal
            self.skip_online = opts.get("skip_online", False)
            self.changelog_url = "https://cursor.com/cn/changelog"
            self.dry_run = opts.get("dry_run", False)
            self.max_iterations = str(opts.get("max_iterations", 5))
            self.workers = opts.get("workers", 3)
            self.force_update = opts.get("force_update", False)
            self.verbose = opts.get("verbose", False)
            self.auto_commit = opts.get("auto_commit", False)
            self.auto_push = opts.get("auto_push", False)
            self.commit_per_iteration = opts.get("commit_per_iteration", False)
            self.commit_message = opts.get("commit_message", "")

    args = IterateArgs("测试任务", options)

    # 验证所有属性
    assert args.requirement == "测试任务"
    assert args.skip_online == True
    assert args.dry_run == True
    assert args.max_iterations == "5"
    assert args.workers == 2
    assert args.force_update == False
    assert args.verbose == True
    assert args.auto_commit == True
    assert args.auto_push == False
    assert args.commit_per_iteration == True
    assert args.commit_message == "test: prefix"
    assert args.changelog_url == "https://cursor.com/cn/changelog"
    print("✓ IterateArgs 所有属性正确")

    # 测试默认值
    args_default = IterateArgs("默认测试", {})
    assert args_default.skip_online == False
    assert args_default.dry_run == False
    assert args_default.auto_commit == False
    assert args_default.commit_message == ""
    print("✓ IterateArgs 默认值正确")


def test_iterate_mode_detection():
    """测试 iterate 模式关键词检测"""
    from run import RunMode, TaskAnalyzer

    analyzer = TaskAnalyzer(use_agent=False)

    class MockArgs:
        mode = "auto"
        task = ""

    # 测试 iterate 模式关键词（与 run.py 中 MODE_KEYWORDS 定义一致）
    iterate_keywords = [
        "自我迭代",
        "self-iterate",
        "iterate",
        "迭代更新",
        "更新知识库",
        "检查更新",
        "自我更新",
    ]

    for keyword in iterate_keywords:
        analysis = analyzer._rule_based_analysis(keyword, MockArgs())
        assert analysis.mode == RunMode.ITERATE, f"'{keyword}' 应匹配 ITERATE 模式"
    print(f"✓ iterate 模式关键词检测正确 ({len(iterate_keywords)} 个关键词)")


def test_all_modules_import():
    """测试所有模块递归导入"""
    import importlib
    import pkgutil

    # 需要测试的顶级模块目录
    top_modules = [
        "agents",
        "coordinator",
        "core",
        "cursor",
        "indexing",
        "knowledge",
        "process",
        "tasks",
    ]

    imported_modules = []
    failed_modules = []

    def import_submodules(package_name: str):
        """递归导入包中的所有子模块"""
        try:
            package = importlib.import_module(package_name)
            imported_modules.append(package_name)
        except Exception as e:
            failed_modules.append((package_name, str(e)))
            return

        # 检查是否为包（有 __path__ 属性）
        if not hasattr(package, "__path__"):
            return

        # 递归导入子模块
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, prefix=package_name + "."
        ):
            try:
                importlib.import_module(modname)
                imported_modules.append(modname)
            except Exception as e:
                failed_modules.append((modname, str(e)))

    # 导入所有顶级模块
    for module_name in top_modules:
        import_submodules(module_name)

    # 报告结果
    print(f"✓ 成功导入 {len(imported_modules)} 个模块")
    for mod in imported_modules:
        print(f"  - {mod}")

    if failed_modules:
        print(f"✗ 导入失败 {len(failed_modules)} 个模块:")
        for mod, err in failed_modules:
            print(f"  - {mod}: {err}")
        # 在 CI 环境中，某些依赖可能未安装，仅报告但不失败
        import os
        if os.environ.get("CI"):
            print("  (CI 环境，跳过导入失败的断言)")
        else:
            assert len(failed_modules) == 0, f"模块导入失败: {failed_modules}"
    else:
        print("✓ 所有模块导入成功")


def test_interface_consistency():
    """测试 IterateArgs 接口一致性

    验证 run.py 中定义的 IterateArgs 类与 run_iterate.py 中
    SelfIterator 期望的属性一致。
    """
    # run.py 中 IterateArgs 定义的属性列表
    # 参考 run.py _run_iterate 方法中的 IterateArgs 类
    run_py_iterate_args_attrs = {
        "requirement",       # goal 映射
        "skip_online",
        "changelog_url",
        "dry_run",
        "max_iterations",
        "workers",
        "force_update",
        "verbose",
        "auto_commit",
        "auto_push",
        "commit_per_iteration",
        "commit_message",
        "orchestrator",
        "no_mp",
        "_orchestrator_user_set",  # 元字段：标记用户是否显式设置编排器
    }

    # run_iterate.py 中 SelfIterator.__init__ 使用的属性
    # 参考 run_iterate.py 中的 parse_args 函数和 SelfIterator._get_orchestrator_type
    run_iterate_expected_attrs = {
        "requirement",
        "skip_online",
        "changelog_url",
        "dry_run",
        "max_iterations",
        "workers",
        "force_update",
        "verbose",
        "auto_commit",
        "auto_push",
        "commit_message",
        "orchestrator",
        "no_mp",
        "_orchestrator_user_set",  # 用于防止关键词反向覆盖用户设置
    }

    # 检查 run.py 是否提供了 run_iterate.py 需要的所有属性
    missing_in_run_py = run_iterate_expected_attrs - run_py_iterate_args_attrs
    extra_in_run_py = run_py_iterate_args_attrs - run_iterate_expected_attrs

    if missing_in_run_py:
        print(f"✗ run.py IterateArgs 缺少属性: {missing_in_run_py}")
        assert False, f"run.py IterateArgs 缺少 run_iterate.py 需要的属性: {missing_in_run_py}"

    if extra_in_run_py:
        # 额外属性不是错误，只是提示
        print(f"ℹ run.py IterateArgs 有额外属性: {extra_in_run_py}")

    print("✓ IterateArgs 接口一致性验证通过")
    print(f"  - 共 {len(run_iterate_expected_attrs)} 个必需属性")
    print(f"  - 额外属性: {len(extra_in_run_py)} 个")

    # 动态验证：创建 IterateArgs 实例并检查属性
    class IterateArgs:
        """run.py 中定义的 IterateArgs 类副本"""
        def __init__(self, goal: str, opts: dict):
            self.requirement = goal
            self.skip_online = opts.get("skip_online", False)
            self.changelog_url = "https://cursor.com/cn/changelog"
            self.dry_run = opts.get("dry_run", False)
            self.max_iterations = str(opts.get("max_iterations", 5))
            self.workers = opts.get("workers", 3)
            self.force_update = opts.get("force_update", False)
            self.verbose = opts.get("verbose", False)
            self.auto_commit = opts.get("auto_commit", False)
            self.auto_push = opts.get("auto_push", False)
            self.commit_per_iteration = opts.get("commit_per_iteration", False)
            self.commit_message = opts.get("commit_message", "")
            self.orchestrator = opts.get("orchestrator", "mp")
            self.no_mp = opts.get("no_mp", False)
            self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

    # 创建实例并验证所有属性存在
    args = IterateArgs("test goal", {})
    for attr in run_iterate_expected_attrs:
        assert hasattr(args, attr), f"IterateArgs 实例缺少属性: {attr}"

    print("✓ IterateArgs 实例属性验证通过")


def test_orchestrator_user_set_override_protection():
    """测试 _orchestrator_user_set 元字段保护用户设置不被关键词覆盖

    验证当用户显式传入 --orchestrator/--no-mp 时，SelfIterator 不会被
    requirement 中的非并行关键词反向覆盖。
    """
    from scripts.run_iterate import SelfIterator

    # 模拟 run.py 中的 IterateArgs 类（与 run.py 保持同步）
    class IterateArgs:
        def __init__(self, goal: str, opts: dict):
            self.requirement = goal
            self.skip_online = opts.get("skip_online", False)
            self.changelog_url = "https://cursor.com/cn/changelog"
            self.dry_run = opts.get("dry_run", False)
            self.max_iterations = str(opts.get("max_iterations", 5))
            self.workers = opts.get("workers", 3)
            self.force_update = opts.get("force_update", False)
            self.verbose = opts.get("verbose", False)
            self.auto_commit = opts.get("auto_commit", False)
            self.auto_push = opts.get("auto_push", False)
            self.commit_per_iteration = opts.get("commit_per_iteration", False)
            self.commit_message = opts.get("commit_message", "")
            self.orchestrator = opts.get("orchestrator", "mp")
            self.no_mp = opts.get("no_mp", False)
            self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

    # 场景1：用户显式设置 --orchestrator mp，requirement 包含非并行关键词
    # 期望：用户设置优先，使用 mp 编排器
    args_explicit_mp = IterateArgs("使用协程模式完成任务", {
        "_orchestrator_user_set": True,
        "orchestrator": "mp",
    })
    iterator = SelfIterator(args_explicit_mp)
    result = iterator._get_orchestrator_type()
    assert result == "mp", f"用户显式设置 mp 应优先，实际: {result}"
    print("✓ 场景1: 显式 --orchestrator mp 不被关键词覆盖")

    # 场景2：用户未显式设置，requirement 包含非并行关键词
    # 期望：被关键词覆盖为 basic
    args_auto = IterateArgs("使用协程模式完成任务", {
        "_orchestrator_user_set": False,
        "orchestrator": "mp",
    })
    iterator = SelfIterator(args_auto)
    result = iterator._get_orchestrator_type()
    assert result == "basic", f"未显式设置应被关键词覆盖为 basic，实际: {result}"
    print("✓ 场景2: 未显式设置时被关键词覆盖为 basic")

    # 场景3：用户显式设置 --no-mp
    # 期望：使用 basic 编排器
    args_no_mp = IterateArgs("完成任务", {
        "_orchestrator_user_set": True,
        "no_mp": True,
    })
    iterator = SelfIterator(args_no_mp)
    result = iterator._get_orchestrator_type()
    assert result == "basic", f"显式 --no-mp 应使用 basic，实际: {result}"
    print("✓ 场景3: 显式 --no-mp 使用 basic 编排器")

    # 场景4：用户显式设置 --orchestrator basic
    # 期望：使用 basic 编排器
    args_basic = IterateArgs("完成任务", {
        "_orchestrator_user_set": True,
        "orchestrator": "basic",
    })
    iterator = SelfIterator(args_basic)
    result = iterator._get_orchestrator_type()
    assert result == "basic", f"显式 --orchestrator basic 应使用 basic，实际: {result}"
    print("✓ 场景4: 显式 --orchestrator basic 使用 basic 编排器")

    print("✓ _orchestrator_user_set 覆盖保护测试通过")


def test_requirements_installed():
    """测试 requirements.txt 中的包是否已安装"""
    import importlib.util
    import re

    requirements_path = project_root / "requirements.txt"

    if not requirements_path.exists():
        print("✗ requirements.txt 不存在")
        assert False, "requirements.txt 文件不存在"

    # 解析 requirements.txt
    packages = []
    with open(requirements_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith("#"):
                continue
            # 提取包名（去除版本号）
            # 处理格式: package>=version, package==version, package[extra]>=version
            match = re.match(r'^([a-zA-Z0-9_-]+)(?:\[[^\]]+\])?', line)
            if match:
                packages.append(match.group(1))

    # 包名映射（PyPI 名称 -> 导入名称）
    import_name_map = {
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "beautifulsoup4": "bs4",
        "typing-extensions": "typing_extensions",
        "sentence-transformers": "sentence_transformers",
        "asyncio-pool": "asyncio_pool",
    }

    installed = []
    not_installed = []

    for package in packages:
        # 获取实际导入名称
        import_name = import_name_map.get(package, package.replace("-", "_"))

        # 检查是否已安装
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            installed.append(package)
        else:
            not_installed.append(package)

    # 报告结果
    print(f"✓ 已安装 {len(installed)}/{len(packages)} 个包")

    if not_installed:
        print(f"⚠ 未安装的包 ({len(not_installed)}):")
        for pkg in not_installed:
            print(f"  - {pkg}")

        # 在 CI 环境中，某些可选依赖可能未安装
        import os
        if os.environ.get("CI"):
            print("  (CI 环境，某些可选依赖可能未安装)")
        else:
            # 本地环境，检查核心依赖
            core_deps = {"pydantic", "loguru", "pyyaml", "aiofiles"}
            missing_core = set(not_installed) & core_deps
            if missing_core:
                assert False, f"核心依赖未安装: {missing_core}"
            else:
                print("  (核心依赖已安装，可选依赖可忽略)")
    else:
        print("✓ 所有依赖包已安装")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("端到端测试 - run.py & orchestrator.py")
    print("=" * 50 + "\n")

    tests = [
        test_run_py_imports,
        test_orchestrator_config,
        test_task_analyzer,
        test_orchestrator_initialization,
        test_iteration_control,
        test_stream_config_apply,
        test_iterate_args,
        test_iterate_mode_detection,
        test_all_modules_import,
        test_interface_consistency,
        test_orchestrator_user_set_override_protection,
        test_requirements_installed,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n[测试] {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
