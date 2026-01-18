"""pytest 测试配置

全局 pytest 配置，包含共享 fixtures 和标记定义。

E2E 测试相关配置请参见 conftest_e2e.py。
"""
from __future__ import annotations

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
