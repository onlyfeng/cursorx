"""E2E 测试配置和 Fixtures

提供端到端测试所需的 Mock 对象、fixtures 和断言助手。

包含:
- MockAgentExecutor: 模拟 Agent 执行器
- MockTaskQueue: 模拟任务队列
- MockKnowledgeManager: 模拟知识库管理器
- temp_workspace: 临时工作空间 fixture
- orchestrator_factory: Orchestrator 工厂 fixture
- 断言助手函数
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from cursor.executor import AgentResult, ExecutionMode
from knowledge.manager import KnowledgeManager
from knowledge.models import Document
from knowledge.storage import SearchResult
from tasks.queue import TaskQueue
from tasks.task import Task, TaskPriority, TaskStatus, TaskType


# ==================== pytest 标记定义 ====================

def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line(
        "markers", "e2e: 标记为端到端测试（可能较慢且需要完整环境）"
    )
    config.addinivalue_line(
        "markers", "slow: 标记为慢速测试（执行时间较长）"
    )
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )


# ==================== Mock 执行器 ====================

class MockAgentResult(BaseModel):
    """Mock Agent 执行结果"""
    success: bool = True
    output: str = ""
    error: Optional[str] = None
    exit_code: int = 0
    duration: float = 0.1
    executor_type: str = "mock"
    files_modified: list[str] = []
    session_id: Optional[str] = None


class ExecutionTrace(BaseModel):
    """执行追踪记录

    记录单次执行的详细信息，用于断言和验证。
    """
    execution_id: str
    prompt: str
    context: Optional[dict[str, Any]] = None
    working_directory: Optional[str] = None
    timeout: Optional[int] = None

    # 状态追踪
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed

    # 执行结果
    success: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None
    files_modified: list[str] = Field(default_factory=list)

    # 执行时长（秒）
    duration: float = 0.0


class MockAgentExecutor:
    """模拟 Agent 执行器

    用于测试时模拟 Cursor Agent 的行为，避免实际调用 CLI。

    用法:
        # 配置成功响应
        executor = MockAgentExecutor()
        executor.configure_response(success=True, output="任务完成")

        # 配置多个响应（按顺序返回）
        executor.configure_responses([
            {"success": True, "output": "第一次执行"},
            {"success": True, "output": "第二次执行"},
        ])

        # 配置特定 prompt 的响应
        executor.configure_response_for_prompt(
            prompt_contains="分析代码",
            response={"success": True, "output": "代码分析完成"},
        )

        # 获取执行追踪信息
        traces = executor.execution_traces
        for trace in traces:
            print(f"执行 {trace.execution_id}: {trace.status}")
    """

    def __init__(
        self,
        default_success: bool = True,
        default_output: str = "Mock execution completed",
        default_delay: float = 0.0,
    ):
        """初始化 Mock 执行器

        Args:
            default_success: 默认执行成功
            default_output: 默认输出内容
            default_delay: 默认执行延迟（秒）
        """
        self._default_success = default_success
        self._default_output = default_output
        self._default_delay = default_delay

        # 响应配置
        self._responses: list[dict[str, Any]] = []
        self._response_index = 0
        self._prompt_responses: dict[str, dict[str, Any]] = {}

        # 执行记录（向后兼容）
        self._execution_history: list[dict[str, Any]] = []
        self._available = True

        # 增强的执行追踪
        self._execution_traces: list[ExecutionTrace] = []
        self._execution_counter = 0

    @property
    def executor_type(self) -> str:
        return "mock"

    @property
    def execution_history(self) -> list[dict[str, Any]]:
        """获取执行历史（向后兼容）"""
        return self._execution_history

    @property
    def execution_count(self) -> int:
        """获取执行次数"""
        return len(self._execution_history)

    @property
    def execution_traces(self) -> list[ExecutionTrace]:
        """获取执行追踪记录列表"""
        return self._execution_traces

    def get_trace_by_id(self, execution_id: str) -> Optional[ExecutionTrace]:
        """根据执行 ID 获取追踪记录"""
        for trace in self._execution_traces:
            if trace.execution_id == execution_id:
                return trace
        return None

    def get_traces_by_status(self, status: str) -> list[ExecutionTrace]:
        """获取指定状态的追踪记录"""
        return [t for t in self._execution_traces if t.status == status]

    def get_successful_traces(self) -> list[ExecutionTrace]:
        """获取成功的执行追踪"""
        return [t for t in self._execution_traces if t.success is True]

    def get_failed_traces(self) -> list[ExecutionTrace]:
        """获取失败的执行追踪"""
        return [t for t in self._execution_traces if t.success is False]

    def configure_response(
        self,
        success: bool = True,
        output: str = "",
        error: Optional[str] = None,
        files_modified: Optional[list[str]] = None,
        duration: float = 0.1,
    ) -> None:
        """配置下一次执行的响应

        Args:
            success: 是否成功
            output: 输出内容
            error: 错误信息
            files_modified: 修改的文件列表
            duration: 执行时长
        """
        self._responses.append({
            "success": success,
            "output": output,
            "error": error,
            "files_modified": files_modified or [],
            "duration": duration,
        })

    def configure_responses(self, responses: list[dict[str, Any]]) -> None:
        """配置多个响应（按顺序返回）

        Args:
            responses: 响应配置列表
        """
        self._responses.extend(responses)

    def configure_response_for_prompt(
        self,
        prompt_contains: str,
        response: dict[str, Any],
    ) -> None:
        """配置特定 prompt 的响应

        当 prompt 包含指定内容时返回对应响应。

        Args:
            prompt_contains: prompt 包含的内容
            response: 响应配置
        """
        self._prompt_responses[prompt_contains] = response

    def set_available(self, available: bool) -> None:
        """设置执行器可用状态

        Args:
            available: 是否可用
        """
        self._available = available

    def reset(self) -> None:
        """重置执行器状态"""
        self._responses.clear()
        self._response_index = 0
        self._prompt_responses.clear()
        self._execution_history.clear()
        self._execution_traces.clear()
        self._execution_counter = 0

    def _get_response(self, prompt: str) -> dict[str, Any]:
        """获取响应配置

        优先级:
        1. prompt 匹配的响应
        2. 预配置的响应队列
        3. 默认响应
        """
        # 检查 prompt 匹配
        for pattern, response in self._prompt_responses.items():
            if pattern in prompt:
                return response

        # 检查响应队列
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response

        # 返回默认响应
        return {
            "success": self._default_success,
            "output": self._default_output,
            "error": None,
            "files_modified": [],
            "duration": 0.1,
        }

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """模拟执行任务

        Args:
            prompt: 指令
            context: 上下文
            working_directory: 工作目录
            timeout: 超时时间

        Returns:
            模拟的执行结果
        """
        # 生成执行 ID
        self._execution_counter += 1
        execution_id = f"exec-{self._execution_counter:04d}"
        started_at = datetime.now()

        # 创建执行追踪记录
        trace = ExecutionTrace(
            execution_id=execution_id,
            prompt=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
            started_at=started_at,
            status="in_progress",
        )
        self._execution_traces.append(trace)

        # 记录执行历史（向后兼容）
        self._execution_history.append({
            "prompt": prompt,
            "context": context,
            "working_directory": working_directory,
            "timeout": timeout,
            "timestamp": started_at.isoformat(),
            "execution_id": execution_id,
        })

        # 模拟延迟
        if self._default_delay > 0:
            await asyncio.sleep(self._default_delay)

        # 获取响应配置
        response = self._get_response(prompt)
        completed_at = datetime.now()
        success = response.get("success", True)

        # 更新追踪记录
        trace.completed_at = completed_at
        trace.status = "completed" if success else "failed"
        trace.success = success
        trace.output = response.get("output", "")
        trace.error = response.get("error")
        trace.files_modified = response.get("files_modified", [])
        trace.duration = (completed_at - started_at).total_seconds()

        return AgentResult(
            success=success,
            output=response.get("output", ""),
            error=response.get("error"),
            exit_code=0 if success else 1,
            duration=response.get("duration", 0.1),
            started_at=started_at,
            completed_at=completed_at,
            executor_type="mock",
            files_modified=response.get("files_modified", []),
        )

    async def check_available(self) -> bool:
        """检查执行器是否可用"""
        return self._available

    def check_available_sync(self) -> bool:
        """同步版本：检查执行器是否可用"""
        return self._available


# ==================== Mock 任务队列 ====================

class MockTaskQueue:
    """模拟任务队列

    用于测试时模拟任务队列行为，支持预设任务列表。

    用法:
        # 创建带预设任务的队列
        tasks = [
            Task(title="任务1", description="描述1", instruction="指令1"),
            Task(title="任务2", description="描述2", instruction="指令2"),
        ]
        queue = MockTaskQueue(preset_tasks=tasks)

        # 或动态添加任务
        queue = MockTaskQueue()
        await queue.enqueue(task)
    """

    def __init__(
        self,
        preset_tasks: Optional[list[Task]] = None,
    ):
        """初始化 Mock 任务队列

        Args:
            preset_tasks: 预设的任务列表
        """
        self._tasks: dict[str, Task] = {}
        self._queues: dict[int, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

        # 添加预设任务
        if preset_tasks:
            for task in preset_tasks:
                self._tasks[task.id] = task

    async def enqueue(self, task: Task) -> None:
        """入队任务"""
        async with self._lock:
            iteration_id = task.iteration_id
            if iteration_id not in self._queues:
                self._queues[iteration_id] = asyncio.Queue()

            self._tasks[task.id] = task
            task.status = TaskStatus.QUEUED
            await self._queues[iteration_id].put(task.id)

    async def dequeue(
        self,
        iteration_id: int,
        timeout: Optional[float] = None,
    ) -> Optional[Task]:
        """出队任务"""
        if iteration_id not in self._queues:
            return None

        queue = self._queues[iteration_id]
        try:
            if timeout:
                task_id = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                task_id = await asyncio.wait_for(queue.get(), timeout=0.1)

            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.ASSIGNED
                return task
            return None
        except asyncio.TimeoutError:
            return None

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)

    def update_task(self, task: Task) -> None:
        """更新任务"""
        self._tasks[task.id] = task

    def get_tasks_by_iteration(self, iteration_id: int) -> list[Task]:
        """获取指定迭代的所有任务"""
        return [t for t in self._tasks.values() if t.iteration_id == iteration_id]

    def get_pending_count(self, iteration_id: int) -> int:
        """获取待处理任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id
            and t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
        )

    def get_in_progress_count(self, iteration_id: int) -> int:
        """获取执行中任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id
            and t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        )

    def get_completed_count(self, iteration_id: int) -> int:
        """获取已完成任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status == TaskStatus.COMPLETED
        )

    def get_failed_count(self, iteration_id: int) -> int:
        """获取失败任务数量"""
        return sum(
            1 for t in self._tasks.values()
            if t.iteration_id == iteration_id and t.status == TaskStatus.FAILED
        )

    def is_iteration_complete(self, iteration_id: int) -> bool:
        """检查迭代是否完成"""
        tasks = self.get_tasks_by_iteration(iteration_id)
        if not tasks:
            return True
        return all(t.is_terminal() for t in tasks)

    def get_statistics(self, iteration_id: int) -> dict:
        """获取迭代统计信息"""
        tasks = self.get_tasks_by_iteration(iteration_id)
        return {
            "total": len(tasks),
            "pending": self.get_pending_count(iteration_id),
            "in_progress": self.get_in_progress_count(iteration_id),
            "completed": self.get_completed_count(iteration_id),
            "failed": self.get_failed_count(iteration_id),
        }

    async def clear_iteration(self, iteration_id: int) -> None:
        """清除指定迭代的队列"""
        if iteration_id in self._queues:
            while not self._queues[iteration_id].empty():
                try:
                    self._queues[iteration_id].get_nowait()
                except asyncio.QueueEmpty:
                    break

    def add_preset_tasks(self, tasks: list[Task], iteration_id: int = 1) -> None:
        """添加预设任务到指定迭代

        Args:
            tasks: 任务列表
            iteration_id: 迭代 ID
        """
        for task in tasks:
            task.iteration_id = iteration_id
            self._tasks[task.id] = task


# ==================== Mock 知识库管理器 ====================

class MockKnowledgeManager:
    """模拟知识库管理器

    用于测试时模拟知识库行为，支持预设文档和搜索结果。

    用法:
        # 创建带预设文档的管理器
        manager = MockKnowledgeManager()
        manager.add_document(Document(
            url="https://example.com",
            title="示例文档",
            content="文档内容...",
        ))

        # 配置搜索结果
        manager.configure_search_results([
            SearchResult(doc_id="doc-1", url="...", title="...", score=0.9),
        ])
    """

    def __init__(
        self,
        name: str = "mock",
    ):
        """初始化 Mock 知识库管理器

        Args:
            name: 知识库名称
        """
        self._name = name
        self._documents: dict[str, Document] = {}
        self._search_results: list[SearchResult] = []
        self._initialized = False
        self._url_to_doc_id: dict[str, str] = {}

    @property
    def name(self) -> str:
        """知识库名称"""
        return self._name

    async def initialize(self) -> None:
        """初始化管理器"""
        self._initialized = True

    def add_document(self, doc: Document) -> None:
        """添加文档

        Args:
            doc: 文档对象
        """
        self._documents[doc.id] = doc
        self._url_to_doc_id[doc.url] = doc.id

    async def add_url(
        self,
        url: str,
        metadata: Optional[dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Optional[Document]:
        """模拟添加 URL

        返回预设的文档或创建一个简单的 Mock 文档。
        """
        if url in self._url_to_doc_id:
            return self._documents.get(self._url_to_doc_id[url])

        doc = Document(
            url=url,
            title=f"Mock Document: {url}",
            content=f"Mock content for {url}",
            metadata=metadata or {},
        )
        self.add_document(doc)
        return doc

    async def add_urls(
        self,
        urls: list[str],
        metadata: Optional[dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> list[Document]:
        """模拟批量添加 URL"""
        docs = []
        for url in urls:
            doc = await self.add_url(url, metadata, force_refresh)
            if doc:
                docs.append(doc)
        return docs

    def configure_search_results(self, results: list[SearchResult]) -> None:
        """配置搜索结果

        Args:
            results: 预设的搜索结果列表
        """
        self._search_results = results

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """搜索知识库

        返回预设的搜索结果。
        """
        results = self._search_results[:max_results]
        return [r for r in results if r.score >= min_score]

    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        search_mode: str = "hybrid",
    ) -> list[SearchResult]:
        """异步搜索知识库"""
        return self.search(query, max_results, min_score)

    def list(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Document]:
        """列出所有文档"""
        docs = list(self._documents.values())
        if offset:
            docs = docs[offset:]
        if limit:
            docs = docs[:limit]
        return docs

    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取指定文档"""
        return self._documents.get(doc_id)

    def get_document_by_url(self, url: str) -> Optional[Document]:
        """根据 URL 获取文档"""
        doc_id = self._url_to_doc_id.get(url)
        if doc_id:
            return self._documents.get(doc_id)
        return None

    def remove(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id in self._documents:
            doc = self._documents.pop(doc_id)
            if doc.url in self._url_to_doc_id:
                del self._url_to_doc_id[doc.url]
            return True
        return False

    async def remove_async(self, doc_id: str) -> bool:
        """异步删除文档"""
        return self.remove(doc_id)

    def clear(self) -> int:
        """清空知识库"""
        count = len(self._documents)
        self._documents.clear()
        self._url_to_doc_id.clear()
        return count

    async def clear_async(self) -> int:
        """异步清空知识库"""
        return self.clear()

    @property
    def vector_search_enabled(self) -> bool:
        """向量搜索是否已启用"""
        return False

    def __len__(self) -> int:
        """返回文档数量"""
        return len(self._documents)


# ==================== Fixtures ====================

@pytest.fixture
def temp_workspace(tmp_path: Path):
    """临时工作空间 fixture

    创建一个临时目录作为工作空间，包含 git 仓库初始化。

    Yields:
        工作空间路径
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 初始化 git 仓库
    try:
        subprocess.run(
            ["git", "init"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )

        # 创建初始提交
        readme = workspace / "README.md"
        readme.write_text("# Test Workspace\n")
        subprocess.run(
            ["git", "add", "."],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # git 不可用时跳过 git 初始化
        pass

    yield workspace

    # 清理
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def mock_executor():
    """Mock Agent 执行器 fixture

    Returns:
        MockAgentExecutor 实例
    """
    return MockAgentExecutor()


@pytest.fixture
def mock_task_queue():
    """Mock 任务队列 fixture

    Returns:
        MockTaskQueue 实例
    """
    return MockTaskQueue()


@pytest.fixture
def mock_knowledge_manager():
    """Mock 知识库管理器 fixture

    Returns:
        MockKnowledgeManager 实例
    """
    return MockKnowledgeManager()


@pytest.fixture
def sample_tasks() -> list[Task]:
    """示例任务列表 fixture

    Returns:
        包含多个示例任务的列表
    """
    return [
        Task(
            title="分析代码结构",
            description="分析项目的代码结构和模块组织",
            instruction="使用 ls 和 tree 命令分析项目结构，总结主要模块",
            type=TaskType.ANALYZE,
            priority=TaskPriority.HIGH,
        ),
        Task(
            title="实现功能 A",
            description="实现用户认证功能",
            instruction="在 auth/ 目录下实现用户登录和注册功能",
            type=TaskType.IMPLEMENT,
            priority=TaskPriority.NORMAL,
        ),
        Task(
            title="编写测试",
            description="为认证模块编写单元测试",
            instruction="在 tests/ 目录下为 auth 模块编写测试用例",
            type=TaskType.TEST,
            priority=TaskPriority.NORMAL,
        ),
    ]


@pytest.fixture
def orchestrator_factory(
    temp_workspace: Path,
    mock_executor: MockAgentExecutor,
    mock_knowledge_manager: MockKnowledgeManager,
):
    """Orchestrator 工厂 fixture

    提供一个工厂函数，用于创建不同配置的 Orchestrator 实例。

    用法:
        def test_example(orchestrator_factory):
            # 使用默认配置创建
            orchestrator = orchestrator_factory()

            # 自定义配置
            orchestrator = orchestrator_factory(
                max_iterations=5,
                worker_pool_size=2,
            )
    """
    created_orchestrators: list[Orchestrator] = []

    def _factory(
        max_iterations: int = 3,
        worker_pool_size: int = 1,
        strict_review: bool = False,
        enable_auto_commit: bool = False,
        **kwargs,
    ) -> Orchestrator:
        """创建 Orchestrator 实例

        Args:
            max_iterations: 最大迭代次数
            worker_pool_size: Worker 池大小
            strict_review: 严格评审模式
            enable_auto_commit: 是否启用自动提交
            **kwargs: 其他配置参数

        Returns:
            配置好的 Orchestrator 实例
        """
        config = OrchestratorConfig(
            working_directory=str(temp_workspace),
            max_iterations=max_iterations,
            worker_pool_size=worker_pool_size,
            strict_review=strict_review,
            enable_auto_commit=enable_auto_commit,
            execution_mode=ExecutionMode.CLI,
            **kwargs,
        )

        # 使用 mock 替换实际的执行器
        with patch('agents.worker.WorkerAgent._create_executor') as mock_create:
            mock_create.return_value = mock_executor
            orchestrator = Orchestrator(
                config=config,
                knowledge_manager=mock_knowledge_manager,
            )
            created_orchestrators.append(orchestrator)
            return orchestrator

    yield _factory

    # 清理
    for orchestrator in created_orchestrators:
        if hasattr(orchestrator, 'worker_pool'):
            orchestrator.worker_pool.shutdown()


# ==================== 断言助手 ====================

def assert_iteration_success(
    result: dict[str, Any],
    expected_iterations: Optional[int] = None,
    min_tasks_completed: int = 0,
) -> None:
    """断言迭代执行成功

    Args:
        result: Orchestrator.run() 的返回结果
        expected_iterations: 预期的迭代次数（None 表示不检查）
        min_tasks_completed: 最少完成的任务数

    Raises:
        AssertionError: 如果断言失败
    """
    assert result.get("success"), f"迭代应该成功，但失败了: {result.get('error')}"

    if expected_iterations is not None:
        actual = result.get("iterations_completed", 0)
        assert actual == expected_iterations, (
            f"预期 {expected_iterations} 次迭代，实际 {actual} 次"
        )

    tasks_completed = result.get("total_tasks_completed", 0)
    assert tasks_completed >= min_tasks_completed, (
        f"预期至少完成 {min_tasks_completed} 个任务，实际完成 {tasks_completed} 个"
    )


def assert_iteration_failed(
    result: dict[str, Any],
    expected_error_contains: Optional[str] = None,
) -> None:
    """断言迭代执行失败

    Args:
        result: Orchestrator.run() 的返回结果
        expected_error_contains: 预期错误信息包含的内容

    Raises:
        AssertionError: 如果断言失败
    """
    assert not result.get("success"), "迭代应该失败，但成功了"

    if expected_error_contains:
        error = result.get("error", "")
        assert expected_error_contains in error, (
            f"错误信息应包含 '{expected_error_contains}'，实际: '{error}'"
        )


def assert_task_completed(
    task: Task,
    expected_result_contains: Optional[str] = None,
) -> None:
    """断言任务完成

    Args:
        task: 任务对象
        expected_result_contains: 预期结果包含的内容

    Raises:
        AssertionError: 如果断言失败
    """
    assert task.status == TaskStatus.COMPLETED, (
        f"任务应为完成状态，实际: {task.status.value}"
    )
    assert task.completed_at is not None, "任务完成时间应该被设置"

    if expected_result_contains and task.result:
        result_str = str(task.result)
        assert expected_result_contains in result_str, (
            f"任务结果应包含 '{expected_result_contains}'，实际: '{result_str}'"
        )


def assert_task_failed(
    task: Task,
    expected_error_contains: Optional[str] = None,
) -> None:
    """断言任务失败

    Args:
        task: 任务对象
        expected_error_contains: 预期错误信息包含的内容

    Raises:
        AssertionError: 如果断言失败
    """
    assert task.status == TaskStatus.FAILED, (
        f"任务应为失败状态，实际: {task.status.value}"
    )

    if expected_error_contains:
        error = task.error or ""
        assert expected_error_contains in error, (
            f"错误信息应包含 '{expected_error_contains}'，实际: '{error}'"
        )


def assert_tasks_in_order(
    tasks: list[Task],
    expected_order: list[str],
) -> None:
    """断言任务按预期顺序执行

    Args:
        tasks: 任务列表
        expected_order: 预期的任务标题顺序

    Raises:
        AssertionError: 如果断言失败
    """
    completed_tasks = [
        t for t in tasks
        if t.status == TaskStatus.COMPLETED and t.completed_at
    ]
    completed_tasks.sort(key=lambda t: t.completed_at)

    actual_order = [t.title for t in completed_tasks]
    assert actual_order == expected_order, (
        f"任务顺序不匹配\n预期: {expected_order}\n实际: {actual_order}"
    )


def assert_executor_called_with(
    executor: MockAgentExecutor,
    prompt_contains: str,
    times: int = 1,
) -> None:
    """断言执行器被调用且 prompt 包含指定内容

    Args:
        executor: Mock 执行器
        prompt_contains: prompt 应包含的内容
        times: 期望的调用次数

    Raises:
        AssertionError: 如果断言失败
    """
    matching_calls = [
        call for call in executor.execution_history
        if prompt_contains in call.get("prompt", "")
    ]

    assert len(matching_calls) == times, (
        f"预期包含 '{prompt_contains}' 的调用 {times} 次，"
        f"实际 {len(matching_calls)} 次"
    )


def assert_no_file_modified(executor: MockAgentExecutor) -> None:
    """断言没有文件被修改

    Args:
        executor: Mock 执行器

    Raises:
        AssertionError: 如果有文件被修改
    """
    for trace in executor.execution_traces:
        assert len(trace.files_modified) == 0, (
            f"执行 {trace.execution_id} 不应修改文件，"
            f"但修改了: {trace.files_modified}"
        )


def assert_execution_trace_count(
    executor: MockAgentExecutor,
    expected_count: int,
) -> None:
    """断言执行追踪记录数量

    Args:
        executor: Mock 执行器
        expected_count: 期望的追踪记录数量

    Raises:
        AssertionError: 如果数量不匹配
    """
    actual_count = len(executor.execution_traces)
    assert actual_count == expected_count, (
        f"预期执行追踪记录 {expected_count} 条，实际 {actual_count} 条"
    )


def assert_all_executions_completed(executor: MockAgentExecutor) -> None:
    """断言所有执行都已完成（成功或失败）

    Args:
        executor: Mock 执行器

    Raises:
        AssertionError: 如果有未完成的执行
    """
    for trace in executor.execution_traces:
        assert trace.status in ("completed", "failed"), (
            f"执行 {trace.execution_id} 状态应为 completed 或 failed，"
            f"实际: {trace.status}"
        )
        assert trace.completed_at is not None, (
            f"执行 {trace.execution_id} 的完成时间未设置"
        )


def assert_execution_status_transitions(
    executor: MockAgentExecutor,
    expected_final_statuses: Optional[list[str]] = None,
) -> None:
    """断言执行状态变更正确

    验证每个执行追踪都经历了正确的状态变更：
    - 开始时设置 started_at
    - 完成时设置 completed_at
    - 最终状态为 completed 或 failed

    Args:
        executor: Mock 执行器
        expected_final_statuses: 期望的最终状态列表（按顺序）

    Raises:
        AssertionError: 如果状态变更不正确
    """
    traces = executor.execution_traces

    for i, trace in enumerate(traces):
        # 验证开始时间已设置
        assert trace.started_at is not None, (
            f"执行 {trace.execution_id} 的开始时间未设置"
        )

        # 验证最终状态合法
        assert trace.status in ("in_progress", "completed", "failed"), (
            f"执行 {trace.execution_id} 状态不合法: {trace.status}"
        )

        # 如果已完成，验证完成时间和时长
        if trace.status in ("completed", "failed"):
            assert trace.completed_at is not None, (
                f"执行 {trace.execution_id} 已完成但未设置完成时间"
            )
            assert trace.completed_at >= trace.started_at, (
                f"执行 {trace.execution_id} 完成时间早于开始时间"
            )
            assert trace.duration >= 0, (
                f"执行 {trace.execution_id} 时长为负数: {trace.duration}"
            )

        # 验证期望的最终状态
        if expected_final_statuses and i < len(expected_final_statuses):
            expected = expected_final_statuses[i]
            assert trace.status == expected, (
                f"执行 {trace.execution_id} 状态应为 {expected}，"
                f"实际: {trace.status}"
            )


def assert_execution_success_rate(
    executor: MockAgentExecutor,
    min_success_rate: float = 1.0,
) -> None:
    """断言执行成功率

    Args:
        executor: Mock 执行器
        min_success_rate: 最低成功率（0.0-1.0）

    Raises:
        AssertionError: 如果成功率低于预期
    """
    traces = executor.execution_traces
    if not traces:
        return  # 没有执行记录时不做断言

    success_count = len(executor.get_successful_traces())
    actual_rate = success_count / len(traces)

    assert actual_rate >= min_success_rate, (
        f"执行成功率应至少为 {min_success_rate:.1%}，"
        f"实际: {actual_rate:.1%} ({success_count}/{len(traces)})"
    )


def assert_execution_durations(
    executor: MockAgentExecutor,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> None:
    """断言执行时长在预期范围内

    Args:
        executor: Mock 执行器
        max_duration: 最大执行时长（秒）
        min_duration: 最小执行时长（秒）

    Raises:
        AssertionError: 如果时长超出范围
    """
    for trace in executor.execution_traces:
        if trace.status not in ("completed", "failed"):
            continue

        if max_duration is not None:
            assert trace.duration <= max_duration, (
                f"执行 {trace.execution_id} 时长 {trace.duration:.3f}s "
                f"超过最大限制 {max_duration}s"
            )

        if min_duration is not None:
            assert trace.duration >= min_duration, (
                f"执行 {trace.execution_id} 时长 {trace.duration:.3f}s "
                f"低于最小限制 {min_duration}s"
            )


def assert_execution_prompts_contain(
    executor: MockAgentExecutor,
    patterns: list[str],
) -> None:
    """断言所有执行的 prompt 中包含指定模式

    Args:
        executor: Mock 执行器
        patterns: 每个执行的 prompt 应包含的模式列表

    Raises:
        AssertionError: 如果 prompt 不包含预期模式
    """
    traces = executor.execution_traces

    assert len(traces) >= len(patterns), (
        f"执行追踪记录 ({len(traces)}) 少于预期模式数 ({len(patterns)})"
    )

    for i, pattern in enumerate(patterns):
        assert pattern in traces[i].prompt, (
            f"执行 {traces[i].execution_id} 的 prompt 应包含 '{pattern}'，"
            f"实际 prompt: '{traces[i].prompt[:100]}...'"
        )


# ==================== 辅助工具函数 ====================

def create_test_task(
    title: str = "测试任务",
    task_type: TaskType = TaskType.CUSTOM,
    priority: TaskPriority = TaskPriority.NORMAL,
    iteration_id: int = 1,
    **kwargs,
) -> Task:
    """创建测试任务

    Args:
        title: 任务标题
        task_type: 任务类型
        priority: 优先级
        iteration_id: 迭代 ID
        **kwargs: 其他任务属性

    Returns:
        Task 实例
    """
    return Task(
        title=title,
        description=kwargs.pop("description", f"测试任务描述: {title}"),
        instruction=kwargs.pop("instruction", f"执行: {title}"),
        type=task_type,
        priority=priority,
        iteration_id=iteration_id,
        **kwargs,
    )


def create_test_document(
    url: str = "https://example.com/test",
    title: str = "测试文档",
    content: str = "测试内容",
    **kwargs,
) -> Document:
    """创建测试文档

    Args:
        url: 文档 URL
        title: 文档标题
        content: 文档内容
        **kwargs: 其他文档属性

    Returns:
        Document 实例
    """
    return Document(
        url=url,
        title=title,
        content=content,
        **kwargs,
    )


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
) -> bool:
    """等待条件满足

    Args:
        condition: 条件函数
        timeout: 超时时间（秒）
        interval: 检查间隔（秒）

    Returns:
        条件是否在超时前满足
    """
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    return False
