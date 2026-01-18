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
