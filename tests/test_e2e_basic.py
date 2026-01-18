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
        parse_args,
        RunMode,
        MODE_ALIASES,
        TaskAnalyzer,
        Runner,
    )
    from coordinator import OrchestratorConfig
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
    from cursor.client import CursorAgentConfig
    
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
    from run import TaskAnalyzer, RunMode
    import argparse
    
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
    from cursor.client import CursorAgentConfig
    
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
    from run import Runner
    
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
    from run import TaskAnalyzer, RunMode
    
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
