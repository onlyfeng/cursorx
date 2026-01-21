"""pytest 测试配置

全局 pytest 配置，包含共享 fixtures 和标记定义。

E2E 测试相关配置请参见 conftest_e2e.py。

平台兼容性说明：
- skip_on_windows: 跳过 Windows 平台
- skip_on_macos: 跳过 macOS 平台
- skip_on_linux: 跳过 Linux 平台
- requires_fork: 需要 fork 启动方式的测试
- requires_spawn: 需要 spawn 启动方式的测试
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 E2E 测试配置
from tests.conftest_e2e import (
    # pytest 配置
    pytest_configure,
    # Mock 类
    MockAgentExecutor,
    MockAgentResult,
    MockKnowledgeManager,
    MockTaskQueue,
    # Fixtures
    mock_executor,
    mock_knowledge_manager,
    mock_task_queue,
    orchestrator_factory,
    sample_tasks,
    temp_workspace,
    # 断言助手
    assert_executor_called_with,
    assert_iteration_failed,
    assert_iteration_success,
    assert_no_file_modified,
    assert_task_completed,
    assert_task_failed,
    assert_tasks_in_order,
    # 辅助函数
    create_test_document,
    create_test_task,
    wait_for_condition,
)


# ==================== 平台特定 Fixtures 和标记 ====================

@pytest.fixture
def skip_on_windows():
    """跳过 Windows 平台的 fixture

    用于需要 Unix 特性的测试。

    Example:
        def test_unix_feature(skip_on_windows):
            # 此测试在 Windows 上跳过
            ...
    """
    if sys.platform == "win32":
        pytest.skip("Windows 暂不支持此功能")


@pytest.fixture
def skip_on_macos():
    """跳过 macOS 平台的 fixture

    Example:
        def test_linux_specific(skip_on_macos):
            # 此测试在 macOS 上跳过
            ...
    """
    if sys.platform == "darwin":
        pytest.skip("macOS 暂不支持此功能")


@pytest.fixture
def skip_on_linux():
    """跳过 Linux 平台的 fixture

    Example:
        def test_non_linux(skip_on_linux):
            # 此测试在 Linux 上跳过
            ...
    """
    if sys.platform.startswith("linux"):
        pytest.skip("Linux 暂不支持此功能")


@pytest.fixture
def requires_fork():
    """需要 fork 启动方式的 fixture

    macOS 和 Windows 默认使用 spawn，此 fixture 会跳过测试。

    Example:
        def test_fork_specific(requires_fork):
            # 此测试仅在 fork 启动方式下运行
            ...
    """
    method = mp.get_start_method(allow_none=True)
    if method is None:
        # 未设置时检查平台默认值
        if sys.platform != "linux" and not sys.platform.startswith("linux"):
            pytest.skip("需要 fork 启动方式（仅 Linux 默认支持）")
    elif method != "fork":
        pytest.skip(f"需要 fork 启动方式，当前为 {method}")


@pytest.fixture
def requires_spawn():
    """需要 spawn 启动方式的 fixture

    Linux 默认使用 fork，此 fixture 会跳过测试。

    Example:
        def test_spawn_specific(requires_spawn):
            # 此测试仅在 spawn 启动方式下运行
            ...
    """
    method = mp.get_start_method(allow_none=True)
    if method is None:
        # 未设置时检查平台默认值
        if sys.platform.startswith("linux"):
            pytest.skip("需要 spawn 启动方式（Linux 默认使用 fork）")
    elif method != "spawn":
        pytest.skip(f"需要 spawn 启动方式，当前为 {method}")


@pytest.fixture
def current_platform() -> str:
    """返回当前平台名称

    Returns:
        平台名称: 'linux', 'macos', 'windows', 'unknown'
    """
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "win32":
        return "windows"
    return "unknown"


@pytest.fixture
def mp_start_method() -> str:
    """返回当前多进程启动方式

    Returns:
        启动方式: 'fork', 'spawn', 'forkserver'
    """
    method = mp.get_start_method(allow_none=True)
    if method is None:
        # 返回平台默认值
        if sys.platform.startswith("linux"):
            return "fork"
        return "spawn"
    return method


# ==================== 通用 Fixtures ====================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录 fixture

    Returns:
        项目根目录路径
    """
    return PROJECT_ROOT


@pytest.fixture
def env_with_api_key(monkeypatch):
    """设置带有 API Key 的环境变量 fixture

    用于需要 API Key 的测试场景。

    Args:
        monkeypatch: pytest 的 monkeypatch fixture
    """
    # 使用测试用的假 API Key
    monkeypatch.setenv("CURSOR_API_KEY", "test-api-key-12345")
    yield
    # monkeypatch 会自动清理


@pytest.fixture
def clean_env(monkeypatch):
    """清理环境变量 fixture

    移除可能影响测试的环境变量。

    Args:
        monkeypatch: pytest 的 monkeypatch fixture
    """
    # 移除可能存在的 API Key
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
    yield


@pytest.fixture(scope="session")
def event_loop_policy():
    """事件循环策略 fixture

    确保测试使用正确的事件循环策略。
    """
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# ==================== 导出 ====================

__all__ = [
    # pytest 配置
    "pytest_configure",
    # Mock 类
    "MockAgentExecutor",
    "MockAgentResult",
    "MockKnowledgeManager",
    "MockTaskQueue",
    # Fixtures
    "mock_executor",
    "mock_knowledge_manager",
    "mock_task_queue",
    "orchestrator_factory",
    "sample_tasks",
    "temp_workspace",
    "project_root",
    "env_with_api_key",
    "clean_env",
    "event_loop_policy",
    # 平台特定 Fixtures
    "skip_on_windows",
    "skip_on_macos",
    "skip_on_linux",
    "requires_fork",
    "requires_spawn",
    "current_platform",
    "mp_start_method",
    # 断言助手
    "assert_executor_called_with",
    "assert_iteration_failed",
    "assert_iteration_success",
    "assert_no_file_modified",
    "assert_task_completed",
    "assert_task_failed",
    "assert_tasks_in_order",
    # 辅助函数
    "create_test_document",
    "create_test_task",
    "wait_for_condition",
]
