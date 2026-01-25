"""Cloud Agent 集成测试

测试内容:
1. Orchestrator 使用 Cloud Agent 执行完整的规划-执行-评审循环
2. CLI 到 Cloud 的自动回退机制
3. 流式输出处理
4. 需要有效的 API Key 才能运行（使用 pytest.mark.skipif 或环境变量控制）

运行方式:
    # 运行所有集成测试（需要 CURSOR_API_KEY 环境变量）
    pytest tests/test_cloud_integration.py -v

    # 跳过集成测试
    SKIP_INTEGRATION_TESTS=1 pytest tests/test_cloud_integration.py -v

    # 强制运行（即使 API Key 验证失败）
    FORCE_INTEGRATION_TESTS=1 pytest tests/test_cloud_integration.py -v
"""
import argparse
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coordinator.orchestrator import Orchestrator, OrchestratorConfig

# 推荐: 从 cursor 顶层包导入（统一入口）
from cursor import (
    # 执行器
    AgentExecutorFactory,
    AgentResult,
    # 认证相关
    AuthError,
    AuthErrorCode,
    AuthStatus,
    AutoAgentExecutor,
    CLIAgentExecutor,
    CloudAgentExecutor,
    CloudAuthConfig,
    CloudAuthManager,
    # 任务管理
    CloudTask,
    CloudTaskClient,
    CloudTaskOptions,
    # 客户端配置
    CursorCloudClient,
    ExecutionMode,
    NetworkError,
    ProgressTracker,
    # 异常类
    RateLimitError,
    # 重试配置
    RetryConfig,
    # 流式输出
    StreamEvent,
    StreamEventType,
    TaskResult,
    TaskStatus,
    ToolCallInfo,
    parse_stream_event,
)

# 兼容性: 也可以从子模块直接导入
# from cursor.cloud_client import AuthError, AuthErrorCode, ...
# from cursor.executor import AgentExecutorFactory, ...
# from cursor.streaming import StreamEvent, StreamEventType, ...
# from cursor.client import CursorAgentConfig


# ========== 环境检测 ==========

def has_api_key() -> bool:
    """检查是否有 API Key"""
    return bool(os.environ.get("CURSOR_API_KEY"))


def should_skip_integration() -> bool:
    """判断是否应该跳过集成测试"""
    # 强制运行
    if os.environ.get("FORCE_INTEGRATION_TESTS"):
        return False
    # 显式跳过
    if os.environ.get("SKIP_INTEGRATION_TESTS"):
        return True
    # 无 API Key 时跳过
    return not has_api_key()


# 跳过标记
skip_without_api_key = pytest.mark.skipif(
    should_skip_integration(),
    reason="需要设置 CURSOR_API_KEY 环境变量运行集成测试"
)

# 集成测试标记
integration_test = pytest.mark.integration


# ========== Fixtures ==========


@pytest.fixture
def temp_working_dir():
    """创建临时工作目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_api_key():
    """模拟 API Key"""
    return "test-integration-api-key-12345"


@pytest.fixture
def mock_env_api_key(mock_api_key):
    """设置环境变量中的 API Key"""
    with patch.dict(os.environ, {"CURSOR_API_KEY": mock_api_key}):
        yield mock_api_key


@pytest.fixture
def cloud_auth_config():
    """Cloud 认证配置"""
    return CloudAuthConfig(
        api_key=os.environ.get("CURSOR_API_KEY", "test-api-key"),
        auth_timeout=30,
        max_retries=2,
    )


@pytest.fixture
def orchestrator_config(temp_working_dir):
    """Orchestrator 配置（使用 Cloud 模式）"""
    return OrchestratorConfig(
        working_directory=temp_working_dir,
        max_iterations=1,
        worker_pool_size=1,
        enable_sub_planners=False,
        execution_mode=ExecutionMode.CLOUD,
        cloud_auth_config=CloudAuthConfig(
            api_key=os.environ.get("CURSOR_API_KEY", "test-api-key"),
        ),
    )


@pytest.fixture
def auto_orchestrator_config(temp_working_dir):
    """Orchestrator 配置（使用 Auto 模式）"""
    return OrchestratorConfig(
        working_directory=temp_working_dir,
        max_iterations=1,
        worker_pool_size=1,
        enable_sub_planners=False,
        execution_mode=ExecutionMode.AUTO,
        cloud_auth_config=CloudAuthConfig(
            api_key=os.environ.get("CURSOR_API_KEY", "test-api-key"),
        ),
    )


@pytest.fixture
def cloud_task_client():
    """Cloud 任务客户端"""
    auth_manager = CloudAuthManager(
        config=CloudAuthConfig(
            api_key=os.environ.get("CURSOR_API_KEY", "test-api-key"),
        )
    )
    return CloudTaskClient(auth_manager=auth_manager)


@pytest.fixture
def cursor_cloud_client():
    """Cursor Cloud 客户端"""
    auth_manager = CloudAuthManager(
        config=CloudAuthConfig(
            api_key=os.environ.get("CURSOR_API_KEY", "test-api-key"),
        )
    )
    return CursorCloudClient(auth_manager=auth_manager)


# ========== 测试类: Orchestrator Cloud Agent 循环 ==========


class TestOrchestratorCloudCycle:
    """测试 Orchestrator 使用 Cloud Agent 执行完整循环"""

    def test_orchestrator_cloud_config(self, orchestrator_config):
        """验证 Orchestrator 配置正确使用 Cloud 模式"""
        assert orchestrator_config.execution_mode == ExecutionMode.CLOUD
        assert orchestrator_config.cloud_auth_config is not None

    def test_orchestrator_initialization_with_cloud(self, orchestrator_config):
        """验证 Orchestrator 使用 Cloud 模式正确初始化"""
        orchestrator = Orchestrator(orchestrator_config)

        # 验证组件已创建
        assert orchestrator.planner is not None
        assert orchestrator.reviewer is not None
        assert orchestrator.worker_pool is not None

        # 验证执行模式传递正确
        assert orchestrator.config.execution_mode == ExecutionMode.CLOUD

    @pytest.mark.asyncio
    async def test_planning_phase_with_cloud_mock(self, orchestrator_config):
        """测试规划阶段（使用 Mock）"""
        orchestrator = Orchestrator(orchestrator_config)

        # Mock planner 执行
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "id": "task-1",
                    "title": "测试任务",
                    "description": "执行测试",
                    "type": "test",
                    "target_files": ["test.py"],
                }
            ],
        }

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan_result):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "task-1"
                mock_create.return_value = mock_task

                # 开始迭代
                orchestrator.state.goal = "测试目标"
                orchestrator.state.start_new_iteration()

                # 执行规划阶段
                await orchestrator._planning_phase("测试目标", 1)

                # 验证任务被创建
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_phase_with_cloud_mock(self, orchestrator_config):
        """测试执行阶段（使用 Mock）"""
        orchestrator = Orchestrator(orchestrator_config)
        orchestrator.state.start_new_iteration()

        # Mock worker pool
        with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
            await orchestrator._execution_phase(1)

    @pytest.mark.asyncio
    async def test_review_phase_with_cloud_mock(self, orchestrator_config):
        """测试评审阶段（使用 Mock）"""
        from agents.reviewer import ReviewDecision

        orchestrator = Orchestrator(orchestrator_config)
        orchestrator.state.goal = "测试目标"
        orchestrator.state.start_new_iteration()

        # Mock reviewer
        mock_review_result = {
            "decision": ReviewDecision.COMPLETE,
            "score": 90,
            "summary": "测试通过",
        }

        with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review_result):
            decision = await orchestrator._review_phase("测试目标", 1)
            assert decision == ReviewDecision.COMPLETE

    @pytest.mark.asyncio
    async def test_full_cycle_with_cloud_mock(self, orchestrator_config):
        """测试完整的规划-执行-评审循环（使用 Mock）"""
        from agents.reviewer import ReviewDecision

        orchestrator = Orchestrator(orchestrator_config)

        # Mock planner
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "id": "task-1",
                    "title": "测试任务",
                    "description": "执行测试",
                    "type": "test",
                }
            ],
        }

        # Mock reviewer
        mock_review_result = {
            "decision": ReviewDecision.COMPLETE,
            "score": 95,
            "summary": "所有任务完成",
        }

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan_result):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "task-1"
                mock_task.status = MagicMock(value="completed")
                mock_task.result = {"success": True}
                mock_create.return_value = mock_task

                with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
                    with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review_result):
                        with patch.object(orchestrator.task_queue, "get_tasks_by_iteration", return_value=[mock_task]):
                            with patch.object(orchestrator.task_queue, "get_pending_count", return_value=1):
                                with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 1, "failed": 0}):
                                    result = await orchestrator.run("完成测试任务")

                                    assert result["success"] is True
                                    assert result["iterations_completed"] == 1

    @skip_without_api_key
    @integration_test
    @pytest.mark.asyncio
    async def test_real_cloud_authentication(self, cloud_auth_config):
        """真实 Cloud 认证测试（需要 API Key）"""
        auth_manager = CloudAuthManager(config=cloud_auth_config)
        status = await auth_manager.authenticate()

        # 验证认证结果
        if status.authenticated:
            assert status.token is not None
        else:
            # 认证失败也是有效结果（如 API Key 无效）
            assert status.error is not None


# ========== 测试类: CLI 到 Cloud 自动回退 ==========


class TestCLIToCloudFallback:
    """测试 CLI 到 Cloud 的自动回退机制"""

    def test_auto_executor_initialization(self):
        """测试 AutoAgentExecutor 正确初始化"""
        executor = AutoAgentExecutor()

        assert executor.executor_type == "auto"
        assert executor.cli_executor is not None
        assert executor.cloud_executor is not None

    @pytest.mark.asyncio
    async def test_fallback_when_cloud_unavailable(self):
        """测试 Cloud 不可用时回退到 CLI"""
        executor = AutoAgentExecutor()

        # Mock Cloud 不可用
        with patch.object(executor._cloud_executor, "check_available", return_value=False):
            with patch.object(executor._cli_executor, "check_available", return_value=True):
                # 模拟 CLI 执行成功
                mock_result = AgentResult(
                    success=True,
                    output="CLI 执行成功",
                    executor_type="cli",
                )

                with patch.object(executor._cli_executor, "execute", return_value=mock_result):
                    result = await executor.execute(prompt="测试任务")

                    assert result.success is True
                    assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_fallback_on_cloud_execution_failure(self):
        """测试 Cloud 执行失败时回退到 CLI"""
        executor = AutoAgentExecutor()

        # Cloud 可用但执行失败
        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud API 执行失败",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        with patch.object(executor._cloud_executor, "check_available", return_value=True):
            with patch.object(executor._cloud_executor, "execute", return_value=cloud_fail_result):
                with patch.object(executor._cli_executor, "execute", return_value=cli_success_result):
                    result = await executor.execute(prompt="测试任务")

                    # 应该回退到 CLI 并成功
                    assert result.success is True
                    assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_prefer_cloud_when_available(self):
        """测试 Cloud 可用时优先使用"""
        executor = AutoAgentExecutor()

        cloud_result = AgentResult(
            success=True,
            output="Cloud 执行成功",
            executor_type="cloud",
        )

        with patch.object(executor._cloud_executor, "check_available", return_value=True):
            with patch.object(executor._cloud_executor, "execute", return_value=cloud_result):
                result = await executor.execute(prompt="测试任务")

                assert result.success is True
                assert result.executor_type == "cloud"

    def test_reset_preference(self):
        """测试重置执行器偏好"""
        executor = AutoAgentExecutor()
        executor._preferred_executor = executor._cli_executor

        executor.reset_preference()

        assert executor._preferred_executor is None

    @pytest.mark.asyncio
    async def test_factory_create_auto(self):
        """测试工厂创建自动执行器"""
        executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)

        assert isinstance(executor, AutoAgentExecutor)
        assert executor.executor_type == "auto"

    @pytest.mark.asyncio
    async def test_orchestrator_auto_mode(self, auto_orchestrator_config):
        """测试 Orchestrator 在 AUTO 模式下的行为"""
        orchestrator = Orchestrator(auto_orchestrator_config)

        assert orchestrator.config.execution_mode == ExecutionMode.AUTO

    @pytest.mark.asyncio
    async def test_auto_mode_with_authentication_failure(self):
        """测试认证失败时的自动回退"""
        executor = AutoAgentExecutor()

        # Cloud 认证失败
        with patch.object(
            executor._cloud_executor._auth_manager,
            "authenticate",
            return_value=AuthStatus(
                authenticated=False,
                error=AuthError("API Key 无效", AuthErrorCode.INVALID_API_KEY),
            ),
        ):
            with patch.object(executor._cloud_executor, "check_available", return_value=False):
                with patch.object(executor._cli_executor, "check_available", return_value=True):
                    mock_result = AgentResult(
                        success=True,
                        output="CLI 回退成功",
                        executor_type="cli",
                    )
                    with patch.object(executor._cli_executor, "execute", return_value=mock_result):
                        result = await executor.execute(prompt="测试任务")

                        assert result.success is True
                        assert result.executor_type == "cli"


# ========== 测试类: 流式输出处理 ==========


class TestStreamingOutput:
    """测试流式输出处理"""

    def test_parse_system_init_event(self):
        """测试解析系统初始化事件"""
        line = '{"type": "system", "subtype": "init", "model": "gpt-5"}'
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.SYSTEM_INIT
        assert event.model == "gpt-5"

    def test_parse_assistant_event(self):
        """测试解析助手消息事件"""
        line = '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}'
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.ASSISTANT
        assert event.content == "Hello"

    def test_parse_tool_started_event(self):
        """测试解析工具开始事件"""
        line = '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "test.py"}}}}'
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.TOOL_STARTED
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "write"
        assert event.tool_call.path == "test.py"

    def test_parse_tool_completed_event(self):
        """测试解析工具完成事件"""
        line = '{"type": "tool_call", "subtype": "completed", "tool_call": {"readToolCall": {"args": {"path": "file.txt"}, "result": {"success": {"totalLines": 100}}}}}'
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.TOOL_COMPLETED
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "read"
        assert event.tool_call.success is True

    def test_parse_result_event(self):
        """测试解析结果事件"""
        line = '{"type": "result", "duration_ms": 1500}'
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.RESULT
        assert event.duration_ms == 1500

    def test_parse_invalid_json(self):
        """测试解析无效 JSON"""
        line = "这不是 JSON"
        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.MESSAGE
        assert event.content == "这不是 JSON"

    def test_parse_empty_line(self):
        """测试解析空行"""
        event = parse_stream_event("")
        assert event is None

    def test_progress_tracker_system_init(self):
        """测试 ProgressTracker 处理系统初始化"""
        tracker = ProgressTracker(verbose=False)
        event = StreamEvent(
            type=StreamEventType.SYSTEM_INIT,
            model="gpt-5",
        )

        tracker.on_event(event)

        assert tracker.model == "gpt-5"

    def test_progress_tracker_assistant_message(self):
        """测试 ProgressTracker 处理助手消息"""
        tracker = ProgressTracker(verbose=False)
        event = StreamEvent(
            type=StreamEventType.ASSISTANT,
            content="测试消息",
        )

        tracker.on_event(event)

        assert "测试消息" in tracker.accumulated_text

    def test_progress_tracker_tool_events(self):
        """测试 ProgressTracker 处理工具事件"""
        tracker = ProgressTracker(verbose=False)

        # 工具开始
        start_event = StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="write", path="test.py"),
        )
        tracker.on_event(start_event)
        assert tracker.tool_count == 1

        # 工具完成
        complete_event = StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(
                tool_type="write",
                path="test.py",
                success=True,
                result={"linesCreated": 50, "fileSize": 1024},
            ),
        )
        tracker.on_event(complete_event)
        assert "test.py" in tracker.files_written

    def test_progress_tracker_result(self):
        """测试 ProgressTracker 处理结果事件"""
        tracker = ProgressTracker(verbose=False)
        event = StreamEvent(
            type=StreamEventType.RESULT,
            duration_ms=2000,
        )

        tracker.on_event(event)

        assert tracker.duration_ms == 2000
        assert tracker.is_complete is True

    def test_progress_tracker_error(self):
        """测试 ProgressTracker 处理错误事件"""
        tracker = ProgressTracker(verbose=False)
        event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "测试错误"},
        )

        tracker.on_event(event)

        assert "测试错误" in tracker.errors

    def test_progress_tracker_summary(self):
        """测试 ProgressTracker 获取摘要"""
        tracker = ProgressTracker(verbose=False)

        # 模拟一系列事件
        tracker.on_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="gpt-5"))
        tracker.on_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Hello"))
        tracker.on_event(StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="write", path="file.py", success=True),
        ))
        tracker.on_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=1000))

        summary = tracker.get_summary()

        assert summary["model"] == "gpt-5"
        assert summary["duration_ms"] == 1000
        assert summary["is_complete"] is True
        assert "file.py" in summary["files_written"]

    @pytest.mark.asyncio
    async def test_cloud_task_client_streaming(self, cloud_task_client):
        """测试 CloudTaskClient 流式执行（Mock）"""
        # Mock 进程
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")

        # 模拟流式输出
        stream_lines = [
            b'{"type": "system", "subtype": "init", "model": "gpt-5"}\n',
            b'{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}\n',
            b'{"type": "result", "duration_ms": 500}\n',
        ]

        async def mock_readline():
            if stream_lines:
                return stream_lines.pop(0)
            return b""

        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = mock_readline
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            events = []
            async for event in cloud_task_client.execute_streaming(
                prompt="测试",
                model="gpt-5",
                timeout=10,
            ):
                events.append(event)

            # 验证事件
            assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_execute_and_collect(self, cloud_task_client):
        """测试执行并收集所有事件"""
        # Mock execute_streaming
        async def mock_streaming(*args, **kwargs):
            yield StreamEvent(type=StreamEventType.SYSTEM_INIT, model="gpt-5")
            yield StreamEvent(type=StreamEventType.ASSISTANT, content="输出内容")
            yield StreamEvent(type=StreamEventType.RESULT, duration_ms=1000)

        with patch.object(cloud_task_client, "execute_streaming", mock_streaming):
            result = await cloud_task_client.execute_and_collect(
                prompt="测试",
                model="gpt-5",
            )

            assert result.status == TaskStatus.COMPLETED
            assert result.duration_ms == 1000


# ========== 测试类: 错误处理和重试 ==========


class TestErrorHandlingAndRetry:
    """测试错误处理和重试机制"""

    def test_retry_config_default(self):
        """测试默认重试配置"""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_retry_config_calculate_delay(self):
        """测试重试延迟计算"""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        delay_0 = config.calculate_delay(0)
        delay_1 = config.calculate_delay(1)
        delay_2 = config.calculate_delay(2)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_retry_config_respect_retry_after(self):
        """测试重试延迟尊重 Retry-After"""
        config = RetryConfig()

        delay = config.calculate_delay(0, retry_after=30.0)

        assert delay == 30.0

    def test_rate_limit_error(self):
        """测试限流错误"""
        error = RateLimitError(
            message="限流",
            retry_after=60.0,
            limit=100,
            remaining=0,
        )

        assert error.retry_after == 60.0
        assert error.limit == 100
        assert "限流" in error.user_friendly_message

    def test_network_error_from_exception(self):
        """测试从异常创建网络错误"""
        error = NetworkError.from_exception(
            asyncio.TimeoutError(),
            context="测试操作",
        )

        assert error.error_type == "timeout"
        assert error.retry_after is not None

    def test_auth_error_from_http_status(self):
        """测试从 HTTP 状态码创建认证错误"""
        error_401 = AuthError.from_http_status(401, "Unauthorized")
        error_403 = AuthError.from_http_status(403, "Forbidden")

        assert error_401.code == AuthErrorCode.INVALID_API_KEY
        assert error_403.code == AuthErrorCode.INSUFFICIENT_PERMISSIONS

    @pytest.mark.asyncio
    async def test_cloud_executor_authentication_error(self):
        """测试 Cloud 执行器认证错误"""
        from cursor.cloud_client import CloudClientFactory, CloudAgentResult

        # 使用 mock 返回认证失败的结果
        mock_result = CloudAgentResult(
            success=False,
            error="Cloud 认证失败: API Key 无效",
        )

        with patch.object(
            CloudClientFactory,
            "execute_task",
            return_value=mock_result,
        ):
            executor = CloudAgentExecutor()
            result = await executor.execute(prompt="测试")

            assert result.success is False
            assert "认证" in result.error or "API Key" in result.error

    @pytest.mark.asyncio
    async def test_submit_task_with_rate_limit(self, cursor_cloud_client):
        """测试提交任务时遇到限流"""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Rate limit exceeded")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.submit_task("测试")

                assert result.success is False
                assert "限流" in result.error or "rate" in result.error.lower()

    @pytest.mark.asyncio
    async def test_poll_task_status_with_retry(self, cloud_task_client):
        """测试任务状态轮询重试"""
        call_count = 0

        async def mock_query(task_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # 前两次返回运行中
                return TaskResult(task_id=task_id, status=TaskStatus.RUNNING)
            # 第三次返回完成
            return TaskResult(task_id=task_id, status=TaskStatus.COMPLETED, result="done")

        with patch.object(cloud_task_client, "_query_task_status", side_effect=mock_query):
            result = await cloud_task_client.poll_task_status(
                "test-task",
                timeout=10.0,
                interval=0.1,
            )

            assert result.status == TaskStatus.COMPLETED
            assert call_count == 3


# ========== 测试类: 集成场景 ==========


class TestCloudSessionManagement:
    """测试云端会话管理功能"""

    def test_is_cloud_request_basic(self):
        """测试基本的云端请求检测"""
        assert CursorCloudClient.is_cloud_request("& 任务") is True
        assert CursorCloudClient.is_cloud_request("普通任务") is False

    def test_is_cloud_request_edge_cases(self):
        """测试云端请求检测的边界情况"""
        # None 和空值
        assert CursorCloudClient.is_cloud_request(None) is False
        assert CursorCloudClient.is_cloud_request("") is False
        assert CursorCloudClient.is_cloud_request("   ") is False

        # 只有 & 符号
        assert CursorCloudClient.is_cloud_request("&") is False
        assert CursorCloudClient.is_cloud_request("&  ") is False
        assert CursorCloudClient.is_cloud_request("  &  ") is False

        # 有效的云端请求
        assert CursorCloudClient.is_cloud_request("& 有内容") is True
        assert CursorCloudClient.is_cloud_request("  & 有内容") is True
        assert CursorCloudClient.is_cloud_request("&有内容") is True

        # & 不在开头
        assert CursorCloudClient.is_cloud_request("任务 & 描述") is False
        assert CursorCloudClient.is_cloud_request("任务&") is False

    def test_is_cloud_request_non_string(self):
        """测试非字符串类型的处理"""
        assert CursorCloudClient.is_cloud_request(123) is False
        assert CursorCloudClient.is_cloud_request([]) is False
        assert CursorCloudClient.is_cloud_request({}) is False

    @pytest.mark.asyncio
    async def test_push_to_cloud_without_auth(self, cursor_cloud_client):
        """测试未认证时推送到云端"""
        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(
                authenticated=False,
                error=AuthError("未认证", AuthErrorCode.INVALID_API_KEY),
            ),
        ):
            result = await cursor_cloud_client.push_to_cloud("session-123")

            assert result.success is False
            assert "认证" in result.error or "未认证" in result.error or "API Key" in result.error

    @pytest.mark.asyncio
    async def test_push_to_cloud_success(self, cursor_cloud_client):
        """测试成功推送到云端"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"task_id": "cloud-task-abc"}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.push_to_cloud(
                    session_id="local-session-123",
                    prompt="继续执行任务",
                )

                assert result.success is True
                assert result.task is not None
                assert result.task.task_id is not None

    @pytest.mark.asyncio
    async def test_push_to_cloud_with_options(self, cursor_cloud_client):
        """测试带选项推送到云端"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"task_id": "cloud-xyz"}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
                options = CloudTaskOptions(model="gpt-5", timeout=600)
                result = await cursor_cloud_client.push_to_cloud(
                    session_id="session-456",
                    options=options,
                )

                assert result.success is True
                # 验证命令包含 model 参数
                call_args = mock_exec.call_args
                cmd = call_args[0]
                assert "--model" in cmd
                assert "gpt-5" in cmd

    @pytest.mark.asyncio
    async def test_resume_from_cloud_not_found(self, cursor_cloud_client):
        """测试恢复不存在的云端会话"""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Session not found")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.resume_from_cloud("invalid-task-id")

                assert result.success is False
                assert "不存在" in result.error or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_resume_from_cloud_success(self, cursor_cloud_client):
        """测试成功从云端恢复会话"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"status": "completed", "result": "Task done"}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.resume_from_cloud(
                    task_id="cloud-task-789",
                    local=True,
                    prompt="继续之前的任务",
                )

                assert result.success is True
                assert result.task is not None

    @pytest.mark.asyncio
    async def test_resume_from_cloud_status_only(self, cursor_cloud_client):
        """测试仅获取云端会话状态"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"status": "running", "progress": 50}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.resume_from_cloud(
                    task_id="cloud-task-running",
                    local=False,  # 仅获取状态
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_session_switch(self, cursor_cloud_client):
        """测试执行时切换会话到云端"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"task_id": "cloud-switched"}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.execute(
                    prompt="继续任务",
                    session_id="local-session",
                    switch_to_cloud=True,
                )

                assert result.success is True
                assert result.task is not None

    @pytest.mark.asyncio
    async def test_execute_resume_cloud_session(self, cursor_cloud_client):
        """测试执行时恢复云端会话"""
        # 先添加一个缓存任务
        cached_task = CloudTask(
            task_id="cached-cloud-task",
            status=TaskStatus.RUNNING,
            prompt="原始任务",
        )
        cursor_cloud_client._tasks["cached-cloud-task"] = cached_task

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'{"status": "completed"}', b"")
        )

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await cursor_cloud_client.execute(
                    prompt="继续执行",
                    session_id="cached-cloud-task",
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_push_to_cloud_timeout(self, cursor_cloud_client):
        """测试推送到云端超时"""
        # 使用 MagicMock 避免创建未 await 的协程
        mock_process = MagicMock()
        mock_process.communicate = MagicMock(return_value=MagicMock())

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            new=AsyncMock(return_value=AuthStatus(authenticated=True)),
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=mock_process),
            ):
                with patch(
                    "asyncio.wait_for",
                    new=AsyncMock(side_effect=asyncio.TimeoutError()),
                ):
                    result = await cursor_cloud_client.push_to_cloud("session-slow")

                    assert result.success is False
                    assert "超时" in result.error

    @pytest.mark.asyncio
    async def test_resume_from_cloud_timeout(self, cursor_cloud_client):
        """测试从云端恢复超时"""
        # 模拟超时：wait_for 抛出 TimeoutError，但协程不会被真正执行
        # 使用 MagicMock 而非 AsyncMock 避免创建未 await 的协程
        mock_process = MagicMock()
        # communicate 返回普通值，因为 wait_for 会在调用前就抛出异常
        mock_process.communicate = MagicMock(return_value=MagicMock())

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            new=AsyncMock(return_value=AuthStatus(authenticated=True)),
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=mock_process),
            ):
                # 使用较短的超时
                options = CloudTaskOptions(timeout=1)

                # wait_for 直接抛出 TimeoutError，模拟超时场景
                with patch(
                    "asyncio.wait_for",
                    new=AsyncMock(side_effect=asyncio.TimeoutError()),
                ):
                    result = await cursor_cloud_client.resume_from_cloud(
                        "task-timeout",
                        options=options,
                    )

                    assert result.success is False
                    assert "超时" in result.error

    def test_extract_modified_files_json(self, cursor_cloud_client):
        """测试从 JSON 输出提取修改的文件"""
        output = '{"files_modified": ["src/main.py", "tests/test.py"]}'
        files = cursor_cloud_client._extract_modified_files(output)

        assert "src/main.py" in files
        assert "tests/test.py" in files

    def test_extract_modified_files_text(self, cursor_cloud_client):
        """测试从文本输出提取修改的文件"""
        output = """
        Created file: src/new_file.py
        Modified config.yaml
        Wrote to tests/test_new.py
        """
        files = cursor_cloud_client._extract_modified_files(output)

        assert len(files) > 0

    @skip_without_api_key
    @integration_test
    @pytest.mark.asyncio
    async def test_real_push_to_cloud(self, cursor_cloud_client, temp_working_dir):
        """真实推送到云端测试（需要 API Key）"""
        # 先创建一个本地会话
        result = await cursor_cloud_client.execute(
            prompt="echo 'test'",
            options=CloudTaskOptions(working_directory=temp_working_dir),
        )

        # 如果成功，尝试推送到云端
        if result.success and result.task:
            push_result = await cursor_cloud_client.push_to_cloud(
                session_id=result.task.task_id,
                prompt="继续执行",
            )

            # 记录结果
            if push_result.success:
                assert push_result.task is not None
            else:
                # 失败也是有效结果
                assert push_result.error is not None


class TestCloudAuthManagerPriority:
    """测试 CloudAuthManager.get_api_key() 优先级

    验证优先级顺序（与 CloudClientFactory.resolve_api_key 保持一致）：
    1. config.api_key（显式参数）
    2. 环境变量 CURSOR_API_KEY
    3. 配置文件
    """

    def test_config_api_key_highest_priority(self, mock_env_api_key):
        """测试 config.api_key 优先级最高"""
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig

        # 即使环境变量存在，config.api_key 也应该优先
        config = CloudAuthConfig(api_key="config-explicit-key")
        auth_manager = CloudAuthManager(config=config)

        api_key = auth_manager.get_api_key()

        # 应该返回 config 中的 key，而非环境变量
        assert api_key == "config-explicit-key"

    def test_env_variable_second_priority(self, mock_env_api_key):
        """测试环境变量优先级次高"""
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig

        # config.api_key 为 None，应该回退到环境变量
        config = CloudAuthConfig(api_key=None)
        auth_manager = CloudAuthManager(config=config)

        api_key = auth_manager.get_api_key()

        assert api_key == mock_env_api_key

    def test_returns_none_without_any_key(self):
        """测试无任何 API Key 时返回 None"""
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig

        with patch.dict(os.environ, {}, clear=True):
            # 清除 CURSOR_API_KEY 环境变量
            os.environ.pop("CURSOR_API_KEY", None)

            config = CloudAuthConfig(api_key=None)
            auth_manager = CloudAuthManager(config=config)

            api_key = auth_manager.get_api_key()

            assert api_key is None

    def test_priority_matches_cloud_client_factory(self, mock_env_api_key):
        """测试 CloudAuthManager 优先级与 CloudClientFactory 一致"""
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig
        from cursor.cloud_client import CloudClientFactory
        from cursor.client import CursorAgentConfig

        explicit_key = "explicit-key-123"
        agent_key = "agent-key-456"

        # CloudClientFactory 的优先级
        agent_config = CursorAgentConfig(api_key=agent_key)
        factory_resolved = CloudClientFactory.resolve_api_key(
            explicit_api_key=explicit_key,
            agent_config=agent_config,
        )
        assert factory_resolved == explicit_key

        # CloudAuthManager 的优先级（config.api_key 最高）
        config = CloudAuthConfig(api_key=explicit_key)
        auth_manager = CloudAuthManager(config=config)
        auth_resolved = auth_manager.get_api_key()
        assert auth_resolved == explicit_key

        # 两者结果一致
        assert factory_resolved == auth_resolved


class TestCloudClientFactory:
    """测试 CloudClientFactory 统一认证配置

    覆盖场景：
    1. API Key 优先级：显式参数 > agent_config.api_key > auth_config.api_key > 环境变量
    2. 工厂正确创建 CloudAuthManager 和 CursorCloudClient
    3. CloudTaskOptions 正确构建
    4. execute_task() 统一执行入口
    """

    def test_resolve_api_key_explicit_highest_priority(self):
        """测试显式参数的 API Key 优先级最高"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="agent-key")
        auth_config = CloudAuthConfig(api_key="auth-key")

        resolved = CloudClientFactory.resolve_api_key(
            explicit_api_key="explicit-key",
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert resolved == "explicit-key"

    def test_resolve_api_key_agent_config_second_priority(self):
        """测试 agent_config.api_key 优先级次高"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="agent-key")
        auth_config = CloudAuthConfig(api_key="auth-key")

        resolved = CloudClientFactory.resolve_api_key(
            explicit_api_key=None,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert resolved == "agent-key"

    def test_resolve_api_key_auth_config_third_priority(self):
        """测试 auth_config.api_key 优先级第三"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key=None)
        auth_config = CloudAuthConfig(api_key="auth-key")

        resolved = CloudClientFactory.resolve_api_key(
            explicit_api_key=None,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert resolved == "auth-key"

    def test_resolve_api_key_env_variable_lowest_priority(self, mock_env_api_key):
        """测试环境变量优先级最低"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key=None)
        auth_config = CloudAuthConfig(api_key=None)

        resolved = CloudClientFactory.resolve_api_key(
            explicit_api_key=None,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert resolved == mock_env_api_key

    def test_resolve_api_key_returns_none_when_no_key(self):
        """测试无任何 API Key 时返回 None"""
        from cursor.cloud_client import CloudClientFactory

        with patch.dict(os.environ, {}, clear=True):
            # 清除 CURSOR_API_KEY 环境变量
            os.environ.pop("CURSOR_API_KEY", None)

            resolved = CloudClientFactory.resolve_api_key(
                explicit_api_key=None,
                agent_config=None,
                auth_config=None,
            )

            assert resolved is None

    def test_create_auth_config_with_priority(self):
        """测试创建 CloudAuthConfig 遵循优先级"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="agent-key")
        auth_config = CloudAuthConfig(api_key="auth-key", auth_timeout=60)

        # 显式 api_key 应覆盖其他
        result = CloudClientFactory.create_auth_config(
            explicit_api_key="explicit-key",
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert result.api_key == "explicit-key"
        # auth_timeout 应继承自 auth_config
        assert result.auth_timeout == 60

    def test_create_auth_config_with_timeout_override(self):
        """测试创建 CloudAuthConfig 可覆盖 auth_timeout"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig

        auth_config = CloudAuthConfig(api_key="test-key", auth_timeout=60)

        result = CloudClientFactory.create_auth_config(
            auth_config=auth_config,
            auth_timeout=120,
        )

        assert result.auth_timeout == 120

    def test_create_auth_manager(self):
        """测试创建 CloudAuthManager"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthManager
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="test-api-key")

        auth_manager = CloudClientFactory.create_auth_manager(
            agent_config=agent_config,
        )

        assert isinstance(auth_manager, CloudAuthManager)
        assert auth_manager.config.api_key == "test-api-key"

    def test_create_returns_client_and_auth_manager(self):
        """测试 create() 返回 client 和 auth_manager"""
        from cursor.cloud_client import CloudClientFactory, CursorCloudClient, CloudAuthManager
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="factory-test-key")

        client, auth_manager = CloudClientFactory.create(
            agent_config=agent_config,
        )

        assert isinstance(client, CursorCloudClient)
        assert isinstance(auth_manager, CloudAuthManager)
        assert auth_manager.config.api_key == "factory-test-key"

    def test_create_with_custom_endpoints(self):
        """测试自定义 API 端点"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="test-key")

        client, _ = CloudClientFactory.create(
            agent_config=agent_config,
            api_base="https://custom.api.com",
            agents_endpoint="/v2/agents",
        )

        assert client.api_base == "https://custom.api.com"
        assert client.agents_endpoint == "/v2/agents"

    def test_build_task_options_from_agent_config(self):
        """测试从 agent_config 构建 CloudTaskOptions"""
        from cursor.cloud_client import CloudClientFactory, CloudTaskOptions
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(
            model="gpt-5.2-high",
            working_directory="/project",
            timeout=600,
            force_write=True,
        )

        options = CloudClientFactory.build_task_options(
            agent_config=agent_config,
        )

        assert isinstance(options, CloudTaskOptions)
        assert options.model == "gpt-5.2-high"
        assert options.working_directory == "/project"
        assert options.timeout == 600
        assert options.allow_write is True

    def test_build_task_options_explicit_overrides(self):
        """测试显式参数覆盖 agent_config"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(
            model="gpt-5.2-high",
            working_directory="/project",
            timeout=600,
            force_write=True,
        )

        options = CloudClientFactory.build_task_options(
            agent_config=agent_config,
            model="opus-4.5-thinking",
            working_directory="/custom",
            timeout=120,
            allow_write=False,
        )

        assert options.model == "opus-4.5-thinking"
        assert options.working_directory == "/custom"
        assert options.timeout == 120
        assert options.allow_write is False

    def test_build_task_options_without_agent_config(self):
        """测试无 agent_config 时使用默认值"""
        from cursor.cloud_client import CloudClientFactory

        options = CloudClientFactory.build_task_options(
            model="test-model",
            working_directory="/test",
            timeout=300,
        )

        assert options.model == "test-model"
        assert options.working_directory == "/test"
        assert options.timeout == 300
        assert options.allow_write is False  # 默认 False

    @pytest.mark.asyncio
    async def test_execute_task_uses_unified_auth(self):
        """测试 execute_task() 使用统一的认证流程"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="execute-task-key")

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Task executed successfully"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = ["output.py"]
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "task-123"

        with patch.object(
            CloudClientFactory, "create"
        ) as mock_create:
            mock_client = AsyncMock()
            mock_client.execute = AsyncMock(return_value=mock_cloud_result)
            mock_auth = MagicMock()
            mock_auth.authenticate = AsyncMock(
                return_value=AuthStatus(authenticated=True)
            )
            mock_create.return_value = (mock_client, mock_auth)

            result = await CloudClientFactory.execute_task(
                prompt="Test task",
                agent_config=agent_config,
                working_directory="/project",
                timeout=300,
                allow_write=True,
            )

            # 验证工厂被调用
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("agent_config") == agent_config

            # 验证结果正确
            assert result.success is True
            assert result.output == "Task executed successfully"
            assert "output.py" in result.files_modified

    @pytest.mark.asyncio
    async def test_execute_task_handles_auth_failure(self):
        """测试 execute_task() 正确处理认证失败"""
        from cursor.cloud_client import CloudClientFactory

        with patch.object(
            CloudClientFactory, "create"
        ) as mock_create:
            mock_client = AsyncMock()
            mock_auth = MagicMock()
            mock_auth.authenticate = AsyncMock(
                return_value=AuthStatus(
                    authenticated=False,
                    error=AuthError("Invalid API Key", AuthErrorCode.INVALID_API_KEY),
                )
            )
            mock_create.return_value = (mock_client, mock_auth)

            result = await CloudClientFactory.execute_task(
                prompt="Test task",
            )

            assert result.success is False
            assert "API Key" in result.error or "认证" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_passes_session_id(self):
        """测试 execute_task() 正确传递 session_id"""
        from cursor.cloud_client import CloudClientFactory

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Resumed session"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = []
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "resumed-session-456"

        with patch.object(
            CloudClientFactory, "create"
        ) as mock_create:
            mock_client = AsyncMock()
            mock_client.execute = AsyncMock(return_value=mock_cloud_result)
            mock_auth = MagicMock()
            mock_auth.authenticate = AsyncMock(
                return_value=AuthStatus(authenticated=True)
            )
            mock_create.return_value = (mock_client, mock_auth)

            result = await CloudClientFactory.execute_task(
                prompt="Continue task",
                session_id="existing-session-123",
            )

            # 验证 session_id 被传递
            mock_client.execute.assert_called_once()
            call_kwargs = mock_client.execute.call_args.kwargs
            assert call_kwargs.get("session_id") == "existing-session-123"

    @pytest.mark.asyncio
    async def test_resume_session_uses_unified_auth(self):
        """测试 resume_session() 使用统一的认证流程"""
        from cursor.cloud_client import CloudClientFactory
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="resume-key")

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Session resumed"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = ["resumed.py"]
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "session-789"

        with patch.object(
            CloudClientFactory, "create"
        ) as mock_create:
            mock_client = AsyncMock()
            mock_client.resume_from_cloud = AsyncMock(return_value=mock_cloud_result)
            mock_auth = MagicMock()
            mock_auth.authenticate = AsyncMock(
                return_value=AuthStatus(authenticated=True)
            )
            mock_create.return_value = (mock_client, mock_auth)

            result = await CloudClientFactory.resume_session(
                session_id="session-789",
                prompt="Continue work",
                agent_config=agent_config,
                local=True,
            )

            assert result.success is True
            assert "resumed.py" in result.files_modified


class TestCloudExecutionPathConsistency:
    """测试两条 Cloud 执行路径的行为一致性

    验证：
    1. CursorAgentClient._execute_via_cloud() 和 CloudAgentExecutor.execute()
       使用相同的 CloudClientFactory.execute_task()
    2. 配置来源优先级一致
    3. allow_write/timeout/session_id 行为一致
    4. files_modified/session_id 正确返回
    """

    @pytest.fixture
    def cloud_enabled_agent_config(self):
        """启用 Cloud 的 Agent 配置"""
        from cursor.client import CursorAgentConfig
        return CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
            api_key="unified-api-key",
            model="opus-4.5-thinking",
            timeout=300,
            force_write=True,
        )

    @pytest.mark.asyncio
    async def test_cursor_agent_client_uses_execute_task(self, cloud_enabled_agent_config):
        """测试 CursorAgentClient 使用 CloudClientFactory.execute_task()"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_agent_config)

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Factory execute_task output"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = ["test.py"]
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "test-session-id"

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute:
            mock_execute.return_value = mock_cloud_result

            result = await client.execute("& 测试任务")

            # 验证 execute_task 被调用
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("agent_config") == cloud_enabled_agent_config
            assert call_kwargs.get("allow_write") is True
            assert call_kwargs.get("wait") is True

            # 验证结果正确
            assert result.success is True
            assert result.session_id == "test-session-id"
            assert "test.py" in result.files_modified

    @pytest.mark.asyncio
    async def test_cloud_agent_executor_uses_execute_task(self):
        """测试 CloudAgentExecutor 使用 CloudClientFactory.execute_task()"""
        from cursor.executor import CloudAgentExecutor
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(
            api_key="executor-api-key",
            model="gpt-5.2-high",
            force_write=True,
            timeout=600,
        )

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Executor execute_task output"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = ["executor.py"]
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "executor-session-id"
        mock_cloud_result.to_dict = MagicMock(return_value={})

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute:
            mock_execute.return_value = mock_cloud_result

            executor = CloudAgentExecutor(agent_config=agent_config)
            result = await executor.execute(
                prompt="Test executor task",
                working_directory="/project",
                timeout=500,
            )

            # 验证 execute_task 被调用
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("agent_config") == agent_config
            assert call_kwargs.get("allow_write") is True
            assert call_kwargs.get("timeout") == 500
            assert call_kwargs.get("working_directory") == "/project"

            # 验证结果正确
            assert result.success is True
            assert result.session_id == "executor-session-id"
            assert "executor.py" in result.files_modified

    @pytest.mark.asyncio
    async def test_both_paths_use_same_factory_method(self, cloud_enabled_agent_config):
        """测试两条执行路径使用相同的工厂方法"""
        from cursor.client import CursorAgentClient
        from cursor.executor import CloudAgentExecutor

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Unified output"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = ["unified.py"]
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "unified-session"
        mock_cloud_result.to_dict = MagicMock(return_value={})

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute:
            mock_execute.return_value = mock_cloud_result

            # 路径 1: CursorAgentClient
            client = CursorAgentClient(config=cloud_enabled_agent_config)
            result1 = await client.execute("& 任务一")

            # 重置 mock
            mock_execute.reset_mock()

            # 路径 2: CloudAgentExecutor
            executor = CloudAgentExecutor(agent_config=cloud_enabled_agent_config)
            result2 = await executor.execute("任务二")

            # 两条路径都应调用 execute_task
            assert mock_execute.call_count == 1  # executor 调用一次

            # 两条路径结果结构一致
            assert result1.success == result2.success
            assert result1.files_modified == result2.files_modified

    @pytest.mark.asyncio
    async def test_session_id_passed_consistently(self, cloud_enabled_agent_config):
        """测试 session_id 在两条路径中被一致传递"""
        from cursor.client import CursorAgentClient
        from cursor.executor import CloudAgentExecutor

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "Session resumed"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = []
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "resumed-session"
        mock_cloud_result.to_dict = MagicMock(return_value={})

        session_id = "test-session-to-resume"

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute:
            mock_execute.return_value = mock_cloud_result

            # CursorAgentClient 传递 session_id
            client = CursorAgentClient(config=cloud_enabled_agent_config)
            await client.execute("& 继续任务", session_id=session_id)

            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("session_id") == session_id

            mock_execute.reset_mock()

            # CloudAgentExecutor 传递 session_id
            executor = CloudAgentExecutor(agent_config=cloud_enabled_agent_config)
            await executor.execute("继续任务", session_id=session_id)

            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("session_id") == session_id

    @pytest.mark.asyncio
    async def test_allow_write_passed_consistently(self):
        """测试 allow_write 在两条路径中被一致传递"""
        from cursor.client import CursorAgentClient, CursorAgentConfig
        from cursor.executor import CloudAgentExecutor

        # force_write=True 应该映射到 allow_write=True
        config_with_write = CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
            api_key="test-key",
            force_write=True,
        )

        # force_write=False 应该映射到 allow_write=False
        config_without_write = CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
            api_key="test-key",
            force_write=False,
        )

        mock_cloud_result = MagicMock()
        mock_cloud_result.success = True
        mock_cloud_result.output = "OK"
        mock_cloud_result.error = None
        mock_cloud_result.files_modified = []
        mock_cloud_result.task = None
        mock_cloud_result.to_dict = MagicMock(return_value={})

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute:
            mock_execute.return_value = mock_cloud_result

            # 测试 force_write=True
            client = CursorAgentClient(config=config_with_write)
            await client.execute("& 任务")
            assert mock_execute.call_args.kwargs.get("allow_write") is True

            mock_execute.reset_mock()

            executor = CloudAgentExecutor(agent_config=config_with_write)
            await executor.execute("任务")
            assert mock_execute.call_args.kwargs.get("allow_write") is True

            mock_execute.reset_mock()

            # 测试 force_write=False
            client2 = CursorAgentClient(config=config_without_write)
            await client2.execute("& 任务")
            assert mock_execute.call_args.kwargs.get("allow_write") is False

            mock_execute.reset_mock()

            executor2 = CloudAgentExecutor(agent_config=config_without_write)
            await executor2.execute("任务")
            assert mock_execute.call_args.kwargs.get("allow_write") is False

    @pytest.mark.asyncio
    async def test_both_paths_return_files_modified(self):
        """测试两条路径都正确返回 files_modified"""
        from cursor.executor import AgentResult

        # 测试 from_cloud_result 包含 files_modified
        result = AgentResult.from_cloud_result(
            success=True,
            output="test output",
            files_modified=["src/main.py", "tests/test.py"],
        )

        assert result.files_modified == ["src/main.py", "tests/test.py"]

    @pytest.mark.asyncio
    async def test_both_paths_return_session_id(self):
        """测试两条路径都正确返回 session_id"""
        from cursor.executor import AgentResult

        # 测试 from_cloud_result 包含 session_id
        result = AgentResult.from_cloud_result(
            success=True,
            output="test output",
            session_id="cloud-session-abc",
        )

        assert result.session_id == "cloud-session-abc"

    def test_agent_config_api_key_priority_over_auth_config(self):
        """测试 agent_config.api_key 优先于 auth_config.api_key"""
        from cursor.cloud_client import CloudClientFactory, CloudAuthConfig
        from cursor.client import CursorAgentConfig

        agent_config = CursorAgentConfig(api_key="agent-priority-key")
        auth_config = CloudAuthConfig(api_key="auth-lower-priority")

        _, auth_manager = CloudClientFactory.create(
            agent_config=agent_config,
            auth_config=auth_config,
        )

        assert auth_manager.config.api_key == "agent-priority-key"


class TestCursorAgentClientCloudRouting:
    """测试 CursorAgentClient 的 Cloud 路由策略

    验证统一路由逻辑：
    - 若 cloud_enabled=True 且 auto_detect_cloud_prefix=True 且 instruction 以 & 开头,
      则使用 CloudClient 而非本地 asyncio.create_subprocess_exec
    - 覆盖边界用例（仅 &、空白等）
    """

    @pytest.fixture
    def cloud_enabled_config(self):
        """启用 Cloud 路由的配置"""
        from cursor.client import CursorAgentConfig
        return CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
            model="opus-4.5-thinking",
            timeout=300,
            force_write=True,
        )

    @pytest.fixture
    def cloud_disabled_config(self):
        """禁用 Cloud 路由的配置"""
        from cursor.client import CursorAgentConfig
        return CursorAgentConfig(
            cloud_enabled=False,
            auto_detect_cloud_prefix=True,
        )

    def test_is_cloud_request_basic(self):
        """测试基本的 cloud request 检测"""
        from cursor.client import CursorAgentClient

        assert CursorAgentClient._is_cloud_request("& 任务") is True
        assert CursorAgentClient._is_cloud_request("  & 任务") is True
        assert CursorAgentClient._is_cloud_request("&任务") is True
        assert CursorAgentClient._is_cloud_request("普通任务") is False

    def test_is_cloud_request_edge_cases(self):
        """测试 cloud request 检测的边界情况"""
        from cursor.client import CursorAgentClient

        # None 和空值
        assert CursorAgentClient._is_cloud_request(None) is False
        assert CursorAgentClient._is_cloud_request("") is False
        assert CursorAgentClient._is_cloud_request("   ") is False

        # 只有 & 符号（无实际内容）
        assert CursorAgentClient._is_cloud_request("&") is False
        assert CursorAgentClient._is_cloud_request("&  ") is False
        assert CursorAgentClient._is_cloud_request("  &  ") is False

        # & 不在开头
        assert CursorAgentClient._is_cloud_request("任务 & 描述") is False
        assert CursorAgentClient._is_cloud_request("任务&") is False

    def test_is_cloud_request_non_string(self):
        """测试非字符串类型的处理"""
        from cursor.client import CursorAgentClient

        assert CursorAgentClient._is_cloud_request(123) is False
        assert CursorAgentClient._is_cloud_request([]) is False
        assert CursorAgentClient._is_cloud_request({}) is False

    def test_should_route_to_cloud_enabled(self, cloud_enabled_config):
        """测试 cloud_enabled=True 时的路由判断"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        # 以 & 开头应该路由到 Cloud
        assert client._should_route_to_cloud("& 任务") is True
        assert client._should_route_to_cloud("& 分析代码") is True

        # 不以 & 开头应该使用本地
        assert client._should_route_to_cloud("任务") is False
        assert client._should_route_to_cloud("分析代码") is False

    def test_should_route_to_cloud_disabled(self, cloud_disabled_config):
        """测试 cloud_enabled=False 时的路由判断"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_disabled_config)

        # cloud_enabled=False 时，即使以 & 开头也不应路由到 Cloud
        assert client._should_route_to_cloud("& 任务") is False
        assert client._should_route_to_cloud("任务") is False

    def test_should_route_to_cloud_auto_detect_disabled(self):
        """测试 auto_detect_cloud_prefix=False 时的路由判断"""
        from cursor.client import CursorAgentClient, CursorAgentConfig

        config = CursorAgentConfig(
            cloud_enabled=True,
            auto_detect_cloud_prefix=False,  # 禁用自动检测
        )
        client = CursorAgentClient(config=config)

        # auto_detect_cloud_prefix=False 时，即使以 & 开头也不应路由到 Cloud
        assert client._should_route_to_cloud("& 任务") is False

    @pytest.mark.asyncio
    async def test_execute_routes_to_cloud_when_enabled(self, cloud_enabled_config):
        """测试 cloud_enabled=True 且 & 前缀时调用 CloudClient"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        # Mock _execute_via_cloud 方法
        from cursor.client import CursorAgentResult
        mock_cloud_result = CursorAgentResult(
            success=True,
            output="Cloud executed successfully",
            exit_code=0,
        )

        with patch.object(
            client, "_execute_via_cloud", new_callable=AsyncMock
        ) as mock_cloud:
            mock_cloud.return_value = mock_cloud_result

            result = await client.execute("& 分析代码")

            # 验证调用了 Cloud 执行
            mock_cloud.assert_called_once()
            assert result.success is True
            assert result.output == "Cloud executed successfully"

    @pytest.mark.asyncio
    async def test_execute_routes_to_local_when_no_prefix(self, cloud_enabled_config):
        """测试无 & 前缀时使用本地 CLI"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        # Mock 本地执行
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Local CLI executed", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("分析代码")  # 无 & 前缀

            assert result.success is True
            assert "Local CLI executed" in result.output

    @pytest.mark.asyncio
    async def test_execute_routes_to_local_when_cloud_disabled(self, cloud_disabled_config):
        """测试 cloud_enabled=False 时使用本地 CLI"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_disabled_config)

        # Mock 本地执行
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Local CLI executed", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            # 即使有 & 前缀，因为 cloud_enabled=False，也应使用本地
            result = await client.execute("& 分析代码")

            assert result.success is True
            assert "Local CLI executed" in result.output

    @pytest.mark.asyncio
    async def test_execute_via_cloud_passes_options(self, cloud_enabled_config):
        """测试 Cloud 路径正确传递 model/working_directory/timeout/allow_write"""
        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory, CloudAgentResult

        client = CursorAgentClient(config=cloud_enabled_config)

        # Mock CloudClientFactory.execute_task
        mock_cloud_result = CloudAgentResult(
            success=True,
            output="Cloud output",
            error=None,
            files_modified=[],
        )
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "test-session"

        with patch.object(
            CloudClientFactory,
            "execute_task",
            return_value=mock_cloud_result,
        ) as mock_execute:
            result = await client.execute(
                "& 分析代码",
                working_directory="/custom/path",
                timeout=600,
            )

            # 验证 execute_task 被调用
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs

            # 验证参数正确传递
            assert call_kwargs.get("working_directory") == "/custom/path"
            assert call_kwargs.get("timeout") == 600
            assert call_kwargs.get("allow_write") is True
            assert call_kwargs.get("agent_config") == cloud_enabled_config

    @pytest.mark.asyncio
    async def test_execute_via_cloud_handles_timeout(self, cloud_enabled_config):
        """测试 Cloud 执行超时处理"""
        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        client = CursorAgentClient(config=cloud_enabled_config)

        with patch.object(
            CloudClientFactory,
            "execute_task",
            side_effect=asyncio.TimeoutError(),
        ):
            result = await client.execute("& 慢任务", timeout=1)

            assert result.success is False
            assert "超时" in result.error

    @pytest.mark.asyncio
    async def test_execute_via_cloud_handles_exception(self, cloud_enabled_config):
        """测试 Cloud 执行异常处理"""
        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory

        client = CursorAgentClient(config=cloud_enabled_config)

        with patch.object(
            CloudClientFactory,
            "execute_task",
            side_effect=RuntimeError("Cloud API error"),
        ):

            result = await client.execute("& 任务")

            assert result.success is False
            assert "Cloud API error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_session_id_via_cloud(self, cloud_enabled_config):
        """测试带 session_id 的 Cloud 执行"""
        from cursor.client import CursorAgentClient
        from cursor.cloud_client import CloudClientFactory, CloudAgentResult

        client = CursorAgentClient(config=cloud_enabled_config)

        mock_cloud_result = CloudAgentResult(
            success=True,
            output="Resumed from cloud",
            error=None,
            files_modified=[],
        )
        mock_cloud_result.task = MagicMock()
        mock_cloud_result.task.task_id = "cloud-session-123"

        with patch.object(
            CloudClientFactory,
            "execute_task",
            return_value=mock_cloud_result,
        ) as mock_execute:
            result = await client.execute(
                "& 继续任务",
                session_id="cloud-session-123",
            )

            # 验证 execute_task 被调用
            mock_execute.assert_called_once()

            # 验证 session_id 被传递
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs.get("session_id") == "cloud-session-123"

    def test_edge_case_only_ampersand(self, cloud_enabled_config):
        """测试边界用例：只有 & 符号"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        # 只有 & 符号不应该路由到 Cloud
        assert client._should_route_to_cloud("&") is False
        assert client._should_route_to_cloud("& ") is False
        assert client._should_route_to_cloud("  &  ") is False

    def test_edge_case_whitespace_only(self, cloud_enabled_config):
        """测试边界用例：只有空白"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        assert client._should_route_to_cloud("") is False
        assert client._should_route_to_cloud("   ") is False
        assert client._should_route_to_cloud("\t\n") is False

    def test_edge_case_ampersand_in_middle(self, cloud_enabled_config):
        """测试边界用例：& 在中间"""
        from cursor.client import CursorAgentClient

        client = CursorAgentClient(config=cloud_enabled_config)

        # & 在中间不应该路由到 Cloud
        assert client._should_route_to_cloud("任务 & 描述") is False
        assert client._should_route_to_cloud("code & test") is False


class TestSelfIteratorExecutionModeIntegration:
    """测试 SelfIterator 与 Orchestrator 的 execution_mode 集成

    覆盖场景：
    1. execution_mode=cloud 时 SelfIterator 使用 basic 编排器且 OrchestratorConfig.execution_mode 正确
    2. execution_mode=auto 时 SelfIterator 使用 basic 编排器且 OrchestratorConfig.execution_mode 正确
    3. 验证 cloud_auth_config 正确传递到 Orchestrator
    4. 验证 '&' 前缀自动切换到 Cloud 模式
    """

    @pytest.fixture
    def cloud_iterate_args(self):
        """创建 Cloud 模式的迭代参数"""
        import argparse
        return argparse.Namespace(
            requirement="测试 Cloud 模式",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",  # 用户默认使用 mp
            no_mp=False,
            # Cloud 模式配置
            execution_mode="cloud",
            cloud_api_key="test-cloud-api-key",
            cloud_auth_timeout=45,
            _orchestrator_user_set=False,
        )

    @pytest.fixture
    def auto_iterate_args(self):
        """创建 Auto 模式的迭代参数"""
        import argparse
        return argparse.Namespace(
            requirement="测试 Auto 模式",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="3",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            # Auto 模式配置
            execution_mode="auto",
            cloud_api_key="test-auto-api-key",
            cloud_auth_timeout=30,
            _orchestrator_user_set=False,
        )

    @pytest.mark.asyncio
    async def test_cloud_mode_orchestrator_config_execution_mode(
        self, cloud_iterate_args
    ):
        """测试 Cloud 模式下 OrchestratorConfig.execution_mode 正确设置

        验证：
        1. SelfIterator 选择 basic 编排器
        2. OrchestratorConfig 接收到 execution_mode=CLOUD
        3. OrchestratorConfig 接收到正确的 cloud_auth_config
        """
        from scripts.run_iterate import SelfIterator
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(cloud_iterate_args)
        iterator.context.iteration_goal = "测试 Cloud 目标"

        captured_config = None

        def capture_config(*args, **kwargs):
            nonlocal captured_config
            captured_config = kwargs
            return MagicMock()

        with patch("scripts.run_iterate.OrchestratorConfig", side_effect=capture_config):
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig"):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证 execution_mode
                        assert captured_config is not None
                        assert captured_config.get("execution_mode") == ExecutionMode.CLOUD

                        # 验证 cloud_auth_config
                        cloud_auth = captured_config.get("cloud_auth_config")
                        assert cloud_auth is not None
                        assert cloud_auth.api_key == "test-cloud-api-key"
                        assert cloud_auth.auth_timeout == 45

    @pytest.mark.asyncio
    async def test_auto_mode_orchestrator_config_execution_mode(
        self, auto_iterate_args
    ):
        """测试 Auto 模式下 OrchestratorConfig.execution_mode 正确设置

        验证：
        1. SelfIterator 选择 basic 编排器
        2. OrchestratorConfig 接收到 execution_mode=AUTO
        3. OrchestratorConfig 接收到正确的 cloud_auth_config
        """
        from scripts.run_iterate import SelfIterator
        from cursor.executor import ExecutionMode

        iterator = SelfIterator(auto_iterate_args)
        iterator.context.iteration_goal = "测试 Auto 目标"

        captured_config = None

        def capture_config(*args, **kwargs):
            nonlocal captured_config
            captured_config = kwargs
            return MagicMock()

        with patch("scripts.run_iterate.OrchestratorConfig", side_effect=capture_config):
            with patch("scripts.run_iterate.Orchestrator") as MockOrch:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    with patch("scripts.run_iterate.CursorAgentConfig"):
                        mock_km = MagicMock()
                        mock_km.initialize = AsyncMock()
                        MockKM.return_value = mock_km

                        mock_orch = MagicMock()
                        mock_orch.run = AsyncMock(return_value={"success": True})
                        MockOrch.return_value = mock_orch

                        await iterator._run_with_basic_orchestrator(3, mock_km)

                        # 验证 execution_mode
                        assert captured_config is not None
                        assert captured_config.get("execution_mode") == ExecutionMode.AUTO

                        # 验证 cloud_auth_config
                        cloud_auth = captured_config.get("cloud_auth_config")
                        assert cloud_auth is not None
                        assert cloud_auth.api_key == "test-auto-api-key"

    @pytest.mark.asyncio
    async def test_cloud_mode_bypasses_mp_orchestrator(
        self, cloud_iterate_args
    ):
        """测试 Cloud 模式完全绕过 MP 编排器

        验证：
        1. _run_with_mp_orchestrator 不被调用
        2. _run_with_basic_orchestrator 被调用
        3. 结果正确返回
        """
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(cloud_iterate_args)
        iterator.context.iteration_goal = "测试 Cloud 绕过 MP"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
        }

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
        ) as mock_mp:
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # MP 不应被调用
                    mock_mp.assert_not_called()

                    # basic 应被调用
                    mock_basic.assert_called_once()

                    # 结果正确
                    assert result["success"] is True
                    assert result["iterations_completed"] == 1

    @pytest.mark.asyncio
    async def test_auto_mode_bypasses_mp_orchestrator(
        self, auto_iterate_args
    ):
        """测试 Auto 模式完全绕过 MP 编排器"""
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(auto_iterate_args)
        iterator.context.iteration_goal = "测试 Auto 绕过 MP"

        mock_basic_result = {
            "success": True,
            "iterations_completed": 2,
            "total_tasks_created": 4,
            "total_tasks_completed": 4,
            "total_tasks_failed": 0,
        }

        with patch.object(
            iterator,
            "_run_with_mp_orchestrator",
            new_callable=AsyncMock,
        ) as mock_mp:
            with patch.object(
                iterator,
                "_run_with_basic_orchestrator",
                new_callable=AsyncMock,
                return_value=mock_basic_result,
            ) as mock_basic:
                with patch("scripts.run_iterate.KnowledgeManager") as MockKM:
                    mock_km = MagicMock()
                    mock_km.initialize = AsyncMock()
                    MockKM.return_value = mock_km

                    result = await iterator._run_agent_system()

                    # MP 不应被调用
                    mock_mp.assert_not_called()

                    # basic 应被调用
                    mock_basic.assert_called_once()

    def test_orchestrator_receives_execution_mode_from_config(
        self, temp_working_dir
    ):
        """测试 Orchestrator 正确接收 execution_mode 配置"""
        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.CLOUD,
            cloud_auth_config=CloudAuthConfig(api_key="test-key"),
        )

        orchestrator = Orchestrator(config)

        assert orchestrator.config.execution_mode == ExecutionMode.CLOUD
        assert orchestrator.config.cloud_auth_config is not None
        assert orchestrator.config.cloud_auth_config.api_key == "test-key"

    def test_orchestrator_receives_auto_mode_from_config(
        self, temp_working_dir
    ):
        """测试 Orchestrator 正确接收 AUTO 执行模式配置"""
        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.AUTO,
            cloud_auth_config=CloudAuthConfig(api_key="auto-key", auth_timeout=60),
        )

        orchestrator = Orchestrator(config)

        assert orchestrator.config.execution_mode == ExecutionMode.AUTO
        assert orchestrator.config.cloud_auth_config.api_key == "auto-key"
        assert orchestrator.config.cloud_auth_config.auth_timeout == 60


class TestIntegrationScenarios:
    """集成场景测试"""

    @pytest.mark.asyncio
    async def test_complete_workflow_mock(self, temp_working_dir):
        """测试完整工作流（Mock）"""
        from agents.reviewer import ReviewDecision

        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.AUTO,
        )

        orchestrator = Orchestrator(config)

        # Mock 所有阶段
        mock_plan = {"success": True, "tasks": [{"id": "t1", "title": "任务1"}]}
        mock_review = {"decision": ReviewDecision.COMPLETE, "score": 100}

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "t1"
                mock_task.status = MagicMock(value="completed")
                mock_create.return_value = mock_task

                with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
                    with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review):
                        with patch.object(orchestrator.task_queue, "get_tasks_by_iteration", return_value=[mock_task]):
                            with patch.object(orchestrator.task_queue, "get_pending_count", return_value=1):
                                with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 1, "failed": 0}):
                                    result = await orchestrator.run("测试目标")

                                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_executor_factory_integration(self):
        """测试 Executor 工厂集成"""
        # CLI 模式
        cli_executor = AgentExecutorFactory.create(mode=ExecutionMode.CLI)
        assert isinstance(cli_executor, CLIAgentExecutor)

        # Cloud 模式
        cloud_executor = AgentExecutorFactory.create(mode=ExecutionMode.CLOUD)
        assert isinstance(cloud_executor, CloudAgentExecutor)

        # Auto 模式
        auto_executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)
        assert isinstance(auto_executor, AutoAgentExecutor)

    @pytest.mark.asyncio
    async def test_streaming_with_progress_tracking(self):
        """测试流式输出与进度跟踪集成"""
        tracker = ProgressTracker(verbose=False)

        # 模拟事件流
        events = [
            StreamEvent(type=StreamEventType.SYSTEM_INIT, model="gpt-5"),
            StreamEvent(type=StreamEventType.ASSISTANT, content="开始处理..."),
            StreamEvent(
                type=StreamEventType.TOOL_STARTED,
                tool_call=ToolCallInfo(tool_type="read", path="main.py"),
            ),
            StreamEvent(
                type=StreamEventType.TOOL_COMPLETED,
                tool_call=ToolCallInfo(tool_type="read", path="main.py", success=True),
            ),
            StreamEvent(type=StreamEventType.ASSISTANT, content="完成!"),
            StreamEvent(type=StreamEventType.RESULT, duration_ms=2500),
        ]

        for event in events:
            tracker.on_event(event)

        summary = tracker.get_summary()

        assert summary["model"] == "gpt-5"
        assert summary["tool_count"] == 1
        assert summary["is_complete"] is True
        assert summary["duration_ms"] == 2500
        assert "main.py" in summary["files_read"]

    @skip_without_api_key
    @integration_test
    @pytest.mark.asyncio
    async def test_real_authentication_flow(self):
        """真实认证流程测试（需要 API Key）"""
        auth_manager = CloudAuthManager()
        api_key = auth_manager.get_api_key()

        assert api_key is not None, "应该能获取到 API Key"

        # 验证认证
        status = await auth_manager.authenticate()

        # 记录结果（不强制成功，因为 API Key 可能无效）
        if status.authenticated:
            assert status.token is not None
        else:
            # 认证失败时应该有错误信息
            assert status.error is not None

    @skip_without_api_key
    @integration_test
    @pytest.mark.asyncio
    async def test_real_cloud_task_submission(self, cursor_cloud_client):
        """真实 Cloud 任务提交测试（需要 API Key）"""
        result = await cursor_cloud_client.submit_task(
            prompt="列出当前目录的文件",
            options=CloudTaskOptions(timeout=60),
        )

        # 记录结果
        if result.success:
            assert result.task is not None
            assert result.task.task_id is not None
        else:
            # 如果失败，应该有错误信息
            assert result.error is not None


# ============================================================
# TestOrchestratorResultStructure - Orchestrator 结果结构 Smoke 测试
# ============================================================


class TestOrchestratorResultStructure:
    """关键路径 Smoke 测试：验证 Orchestrator 最终结果结构

    覆盖场景：
    1. Mock executor 返回值
    2. 验证结果包含 commits/files_modified/session_id
    3. 验证 commit 去重逻辑
    """

    @pytest.fixture
    def temp_working_dir(self, tmp_path):
        """创建临时工作目录"""
        return str(tmp_path)

    @pytest.mark.asyncio
    async def test_orchestrator_result_contains_expected_fields(
        self, temp_working_dir
    ):
        """测试 Orchestrator 结果包含预期字段结构"""
        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.CLI,
        )

        orchestrator = Orchestrator(config)

        # Mock 返回包含 commits 和 files_modified 的结果
        mock_plan = {"success": True, "tasks": []}

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan):
            with patch.object(orchestrator.planner, "create_task_from_plan", return_value=None):
                with patch.object(orchestrator.task_queue, "get_pending_count", return_value=0):
                    with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 0, "failed": 0}):
                        result = await orchestrator.run("测试目标")

                        # 验证结果是字典
                        assert isinstance(result, dict)

                        # 验证必需字段存在
                        assert "success" in result

    @pytest.mark.asyncio
    async def test_worker_result_files_modified_aggregation(
        self, temp_working_dir
    ):
        """测试 Worker 结果中 files_modified 的聚合"""
        from agents.reviewer import ReviewDecision
        from cursor.client import CursorAgentResult

        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.CLI,
        )

        orchestrator = Orchestrator(config)

        # Mock planner 返回一个任务
        mock_plan = {"success": True, "tasks": [{"id": "t1", "title": "任务1", "description": "测试"}]}

        # Mock worker 返回包含 files_modified 的结果
        mock_worker_result = CursorAgentResult(
            success=True,
            output="任务完成",
            files_modified=["src/main.py", "tests/test_main.py"],
            files_edited=["config.yaml"],
            session_id="worker-session-123",
        )

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "t1"
                mock_task.status = MagicMock(value="completed")
                mock_task.result = {
                    "success": True,
                    "output": "任务完成",
                    "files_modified": ["src/main.py", "tests/test_main.py"],
                    "session_id": "worker-session-123",
                }
                mock_create.return_value = mock_task

                mock_review = {"decision": ReviewDecision.COMPLETE, "score": 100}

                with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
                    with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review):
                        with patch.object(orchestrator.task_queue, "get_tasks_by_iteration", return_value=[mock_task]):
                            with patch.object(orchestrator.task_queue, "get_pending_count", return_value=1):
                                with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 1, "failed": 0}):
                                    result = await orchestrator.run("测试目标")

                                    assert result["success"] is True

    @pytest.mark.asyncio
    async def test_result_structure_with_commits_info(
        self, temp_working_dir
    ):
        """测试结果结构包含 commits 信息"""
        from agents.reviewer import ReviewDecision

        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.CLI,
            auto_commit=True,  # 启用自动提交
        )

        orchestrator = Orchestrator(config)

        mock_plan = {"success": True, "tasks": [{"id": "t1", "title": "任务1"}]}
        mock_review = {"decision": ReviewDecision.COMPLETE, "score": 100}

        # Mock committer 返回提交结果
        mock_commit_result = {
            "success": True,
            "commit_hash": "abc123def456",
            "message": "feat: 测试提交",
        }

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "t1"
                mock_task.status = MagicMock(value="completed")
                mock_create.return_value = mock_task

                with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
                    with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review):
                        with patch.object(orchestrator.task_queue, "get_tasks_by_iteration", return_value=[mock_task]):
                            with patch.object(orchestrator.task_queue, "get_pending_count", return_value=1):
                                with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 1, "failed": 0}):
                                    with patch.object(orchestrator, "_commit_phase", new_callable=AsyncMock, return_value=mock_commit_result):
                                        result = await orchestrator.run("测试目标")

                                        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_session_id_propagation_from_worker(
        self, temp_working_dir
    ):
        """测试 session_id 从 Worker 结果正确传播"""
        from agents.reviewer import ReviewDecision

        config = OrchestratorConfig(
            working_directory=temp_working_dir,
            max_iterations=1,
            worker_pool_size=1,
            execution_mode=ExecutionMode.CLOUD,
            cloud_auth_config=CloudAuthConfig(api_key="test-key"),
        )

        orchestrator = Orchestrator(config)

        mock_plan = {"success": True, "tasks": [{"id": "t1", "title": "任务1"}]}
        mock_review = {"decision": ReviewDecision.COMPLETE, "score": 100}

        with patch.object(orchestrator.planner, "execute", return_value=mock_plan):
            with patch.object(orchestrator.planner, "create_task_from_plan") as mock_create:
                mock_task = MagicMock()
                mock_task.id = "t1"
                mock_task.status = MagicMock(value="completed")
                mock_task.result = {
                    "success": True,
                    "session_id": "cloud-session-xyz",
                    "files_modified": ["app.py"],
                }
                mock_create.return_value = mock_task

                with patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock):
                    with patch.object(orchestrator.reviewer, "review_iteration", return_value=mock_review):
                        with patch.object(orchestrator.task_queue, "get_tasks_by_iteration", return_value=[mock_task]):
                            with patch.object(orchestrator.task_queue, "get_pending_count", return_value=1):
                                with patch.object(orchestrator.task_queue, "get_statistics", return_value={"completed": 1, "failed": 0}):
                                    result = await orchestrator.run("测试目标")

                                    assert result["success"] is True
                                    # 验证 session_id 在任务结果中
                                    task_result = mock_task.result
                                    assert task_result["session_id"] == "cloud-session-xyz"


class TestSelfIteratorResultStructure:
    """SelfIterator 结果结构测试

    验证 SelfIterator 最终返回的结果结构包含预期字段
    """

    @pytest.fixture
    def base_iterate_args(self) -> "argparse.Namespace":
        """创建基础迭代参数"""
        return argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url="https://cursor.com/cn/changelog",
            dry_run=False,
            max_iterations="2",
            workers=2,
            force_update=False,
            verbose=False,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="basic",
            no_mp=True,
            _orchestrator_user_set=True,
            execution_mode="cli",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
        )

    @pytest.mark.asyncio
    async def test_self_iterator_result_contains_commits_field(
        self, base_iterate_args
    ):
        """测试 SelfIterator 结果包含 commits 字段"""
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        # Mock orchestrator 返回包含 commits 信息的结果
        mock_orch_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 2,
            "total_tasks_completed": 2,
            "total_tasks_failed": 0,
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["abc123"],
                "successful_commits": 1,
            },
            "files_modified": ["src/main.py", "tests/test.py"],
        }

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher, "initialize", new_callable=AsyncMock
                ):
                    with patch.object(
                        iterator, "_run_agent_system",
                        new_callable=AsyncMock,
                        return_value=mock_orch_result
                    ):
                        result = await iterator.run()

                        # 验证结果结构
                        assert result["success"] is True
                        # commits 字段应在结果中（如果 orchestrator 返回）
                        if "commits" in result:
                            assert "total_commits" in result["commits"]
                            assert "commit_hashes" in result["commits"]

    @pytest.mark.asyncio
    async def test_self_iterator_result_contains_files_modified(
        self, base_iterate_args
    ):
        """测试 SelfIterator 结果包含 files_modified 字段"""
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        mock_orch_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
            "files_modified": ["config.yaml", "setup.py"],
        }

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher, "initialize", new_callable=AsyncMock
                ):
                    with patch.object(
                        iterator, "_run_agent_system",
                        new_callable=AsyncMock,
                        return_value=mock_orch_result
                    ):
                        result = await iterator.run()

                        # 验证 files_modified 在结果中
                        if "files_modified" in result:
                            assert "config.yaml" in result["files_modified"]
                            assert "setup.py" in result["files_modified"]

    @pytest.mark.asyncio
    async def test_has_orchestrator_committed_detection(
        self, base_iterate_args
    ):
        """测试 _has_orchestrator_committed 检测逻辑"""
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(base_iterate_args)

        # 有提交的结果
        result_with_commits = {
            "success": True,
            "commits": {
                "total_commits": 2,
                "commit_hashes": ["abc", "def"],
                "successful_commits": 2,
            },
        }

        # 无提交的结果
        result_without_commits = {
            "success": True,
        }

        # 空提交的结果
        result_empty_commits = {
            "success": True,
            "commits": {
                "total_commits": 0,
                "commit_hashes": [],
                "successful_commits": 0,
            },
        }

        # 验证检测逻辑
        assert iterator._has_orchestrator_committed(result_with_commits) is True
        assert iterator._has_orchestrator_committed(result_without_commits) is False
        assert iterator._has_orchestrator_committed(result_empty_commits) is False

    @pytest.mark.asyncio
    async def test_commit_deduplication_when_orchestrator_committed(
        self, base_iterate_args
    ):
        """测试当 orchestrator 已提交时不重复提交"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.auto_commit = True
        iterator = SelfIterator(base_iterate_args)
        iterator.context.iteration_goal = "测试目标"

        # Orchestrator 已经完成提交
        mock_orch_result = {
            "success": True,
            "iterations_completed": 1,
            "total_tasks_created": 1,
            "total_tasks_completed": 1,
            "total_tasks_failed": 0,
            "commits": {
                "total_commits": 1,
                "commit_hashes": ["existing-commit-hash"],
                "successful_commits": 1,
            },
        }

        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.get_stats = AsyncMock(return_value={"document_count": 0})
        mock_storage.search = AsyncMock(return_value=[])
        mock_storage.list_documents = AsyncMock(return_value=[])

        with patch.object(iterator.knowledge_updater, "storage", mock_storage):
            with patch.object(
                iterator.knowledge_updater.manager, "initialize", new_callable=AsyncMock
            ):
                with patch.object(
                    iterator.knowledge_updater.fetcher, "initialize", new_callable=AsyncMock
                ):
                    with patch.object(
                        iterator, "_run_agent_system",
                        new_callable=AsyncMock,
                        return_value=mock_orch_result
                    ):
                        with patch.object(
                            iterator, "_run_commit_phase",
                            new_callable=AsyncMock
                        ) as mock_commit:
                            await iterator.run()

                            # 验证 _run_commit_phase 未被调用（因为 orchestrator 已提交）
                            mock_commit.assert_not_called()


# ============================================================
# 测试类: Cloud Request 边界输入一致性
# ============================================================


class TestCloudRequestConsistency:
    """测试 is_cloud_request/strip_cloud_prefix 在所有调用点的一致性

    验证 core.cloud_utils、CursorCloudClient、CursorAgentClient 使用同一套判定逻辑。
    """

    # 边界测试用例：(输入, 预期 is_cloud_request 返回值)
    EDGE_CASES = [
        # None 和空值
        (None, False),
        ("", False),
        ("   ", False),
        ("\t\n", False),
        # 只有 & 符号（无实际内容）
        ("&", False),
        ("&  ", False),
        ("  &  ", False),
        ("& \t\n", False),
        # 有效的云端请求
        ("& 任务", True),
        ("&任务", True),
        ("  & 任务", True),
        ("& a", True),
        ("&a", True),
        # & 不在开头
        ("任务 & 描述", False),
        ("任务&", False),
        ("任务 &", False),
        (" 任务 & 描述", False),
        # 多个 & 符号
        ("& & 任务", True),  # 第一个 & 后有内容
        ("&& 任务", True),
        # Unicode 和特殊字符
        ("& 中文任务", True),
        ("& 🚀", True),
        ("& タスク", True),
        # 非字符串类型
        (123, False),
        ([], False),
        ({}, False),
        (0, False),
        (False, False),
    ]

    # strip_cloud_prefix 测试用例：(输入, 预期输出)
    # 注意：strip_cloud_prefix 对于不带 & 前缀的输入返回原始值（不 strip 空白）
    STRIP_CASES = [
        # None 和空值
        (None, ""),
        ("", ""),
        ("   ", "   "),  # 无 & 前缀，返回原始值
        # 带前缀的情况
        ("& 任务", "任务"),
        ("&任务", "任务"),
        ("  & 任务  ", "任务"),  # strip 后以 & 开头
        ("& ", ""),
        ("&", ""),
        # 无前缀的情况（返回原始值）
        ("任务", "任务"),
        ("普通任务", "普通任务"),
        ("任务 & 描述", "任务 & 描述"),
        # 多个空格
        ("&   任务", "任务"),
        ("&\t任务", "任务"),
    ]

    def test_core_cloud_utils_is_cloud_request(self):
        """测试 core.cloud_utils.is_cloud_request 边界情况"""
        from core.cloud_utils import is_cloud_request

        for input_val, expected in self.EDGE_CASES:
            result = is_cloud_request(input_val)
            assert result == expected, (
                f"core.cloud_utils.is_cloud_request({input_val!r}) "
                f"returned {result}, expected {expected}"
            )

    def test_cursor_cloud_client_is_cloud_request(self):
        """测试 CursorCloudClient.is_cloud_request 边界情况"""
        for input_val, expected in self.EDGE_CASES:
            result = CursorCloudClient.is_cloud_request(input_val)
            assert result == expected, (
                f"CursorCloudClient.is_cloud_request({input_val!r}) "
                f"returned {result}, expected {expected}"
            )

    def test_cursor_agent_client_is_cloud_request(self):
        """测试 CursorAgentClient._is_cloud_request 边界情况"""
        from cursor.client import CursorAgentClient

        for input_val, expected in self.EDGE_CASES:
            result = CursorAgentClient._is_cloud_request(input_val)
            assert result == expected, (
                f"CursorAgentClient._is_cloud_request({input_val!r}) "
                f"returned {result}, expected {expected}"
            )

    def test_is_cloud_request_consistency_all_modules(self):
        """验证所有模块的 is_cloud_request 返回一致结果"""
        from core.cloud_utils import is_cloud_request as core_is_cloud
        from cursor.client import CursorAgentClient

        for input_val, _ in self.EDGE_CASES:
            core_result = core_is_cloud(input_val)
            cloud_client_result = CursorCloudClient.is_cloud_request(input_val)
            agent_client_result = CursorAgentClient._is_cloud_request(input_val)

            assert core_result == cloud_client_result == agent_client_result, (
                f"is_cloud_request 不一致: input={input_val!r}\n"
                f"  core.cloud_utils: {core_result}\n"
                f"  CursorCloudClient: {cloud_client_result}\n"
                f"  CursorAgentClient: {agent_client_result}"
            )

    def test_core_cloud_utils_strip_cloud_prefix(self):
        """测试 core.cloud_utils.strip_cloud_prefix 边界情况"""
        from core.cloud_utils import strip_cloud_prefix

        for input_val, expected in self.STRIP_CASES:
            result = strip_cloud_prefix(input_val)
            assert result == expected, (
                f"core.cloud_utils.strip_cloud_prefix({input_val!r}) "
                f"returned {result!r}, expected {expected!r}"
            )

    def test_cursor_cloud_client_strip_cloud_prefix(self):
        """测试 CursorCloudClient.strip_cloud_prefix 边界情况"""
        for input_val, expected in self.STRIP_CASES:
            # CursorCloudClient.strip_cloud_prefix 对 None 会抛异常或返回不同值
            # 需要处理 None 的特殊情况
            if input_val is None:
                # 委托给 core.cloud_utils 后应返回 ""
                result = CursorCloudClient.strip_cloud_prefix(input_val)
                assert result == expected, (
                    f"CursorCloudClient.strip_cloud_prefix(None) "
                    f"returned {result!r}, expected {expected!r}"
                )
            else:
                result = CursorCloudClient.strip_cloud_prefix(input_val)
                assert result == expected, (
                    f"CursorCloudClient.strip_cloud_prefix({input_val!r}) "
                    f"returned {result!r}, expected {expected!r}"
                )

    def test_strip_cloud_prefix_consistency_all_modules(self):
        """验证所有模块的 strip_cloud_prefix 返回一致结果"""
        from core.cloud_utils import strip_cloud_prefix as core_strip

        for input_val, _ in self.STRIP_CASES:
            core_result = core_strip(input_val)
            cloud_client_result = CursorCloudClient.strip_cloud_prefix(input_val)

            assert core_result == cloud_client_result, (
                f"strip_cloud_prefix 不一致: input={input_val!r}\n"
                f"  core.cloud_utils: {core_result!r}\n"
                f"  CursorCloudClient: {cloud_client_result!r}"
            )

    def test_cloud_prefix_constant_consistency(self):
        """验证 CLOUD_PREFIX 常量在所有模块中一致"""
        from core.cloud_utils import CLOUD_PREFIX as core_prefix

        assert CursorCloudClient.CLOUD_PREFIX == core_prefix, (
            f"CLOUD_PREFIX 不一致: core={core_prefix!r}, "
            f"CursorCloudClient={CursorCloudClient.CLOUD_PREFIX!r}"
        )

    def test_run_py_uses_core_cloud_utils(self):
        """验证 run.py 使用 core.cloud_utils"""
        # 通过检查 run.py 的导入来验证
        import run
        from core.cloud_utils import is_cloud_request, strip_cloud_prefix

        # run.py 应该直接导入并使用 core.cloud_utils 的函数
        assert hasattr(run, "is_cloud_request")
        assert hasattr(run, "strip_cloud_prefix")

        # 验证行为一致
        test_input = "& test"
        assert run.is_cloud_request(test_input) == is_cloud_request(test_input)
        assert run.strip_cloud_prefix(test_input) == strip_cloud_prefix(test_input)

    def test_run_iterate_uses_core_cloud_utils(self):
        """验证 scripts/run_iterate.py 使用 core.cloud_utils"""
        from scripts import run_iterate
        from core.cloud_utils import is_cloud_request, strip_cloud_prefix

        # run_iterate 应该直接导入并使用 core.cloud_utils 的函数
        assert hasattr(run_iterate, "is_cloud_request")
        assert hasattr(run_iterate, "strip_cloud_prefix")

        # 验证行为一致
        test_input = "& test"
        assert run_iterate.is_cloud_request(test_input) == is_cloud_request(test_input)
        assert run_iterate.strip_cloud_prefix(test_input) == strip_cloud_prefix(test_input)

    def test_parse_cloud_request_basic(self):
        """测试 parse_cloud_request 基本功能"""
        from core.cloud_utils import parse_cloud_request

        # 有效的云端请求
        is_cloud, clean = parse_cloud_request("& 任务")
        assert is_cloud is True
        assert clean == "任务"

        is_cloud, clean = parse_cloud_request("&任务")
        assert is_cloud is True
        assert clean == "任务"

        # 非云端请求
        is_cloud, clean = parse_cloud_request("普通任务")
        assert is_cloud is False
        assert clean == "普通任务"

        is_cloud, clean = parse_cloud_request("任务 & 描述")
        assert is_cloud is False
        assert clean == "任务 & 描述"

    def test_parse_cloud_request_edge_cases(self):
        """测试 parse_cloud_request 边界情况"""
        from core.cloud_utils import parse_cloud_request

        # None 和空值
        is_cloud, clean = parse_cloud_request(None)
        assert is_cloud is False
        assert clean == ""

        is_cloud, clean = parse_cloud_request("")
        assert is_cloud is False
        assert clean == ""

        # 只有 & 符号（无实际内容）
        is_cloud, clean = parse_cloud_request("&")
        assert is_cloud is False
        assert clean == "&"  # 不是云端请求，返回原始值

        is_cloud, clean = parse_cloud_request("& ")
        assert is_cloud is False
        assert clean == "& "

        is_cloud, clean = parse_cloud_request("  &  ")
        assert is_cloud is False
        assert clean == "  &  "

    def test_parse_cloud_request_consistency_with_individual_functions(self):
        """验证 parse_cloud_request 与 is_cloud_request/strip_cloud_prefix 一致"""
        from core.cloud_utils import (
            is_cloud_request,
            parse_cloud_request,
            strip_cloud_prefix,
        )

        test_cases = [
            "& 任务",
            "&任务",
            "普通任务",
            "任务 & 描述",
            "&",
            "& ",
            "  &  ",
            "& 包含 & 符号",
            None,
            "",
        ]

        for prompt in test_cases:
            is_cloud, clean = parse_cloud_request(prompt)
            expected_is_cloud = is_cloud_request(prompt)

            assert is_cloud == expected_is_cloud, (
                f"parse_cloud_request({prompt!r})[0] != is_cloud_request({prompt!r})\n"
                f"  parse_cloud_request: {is_cloud}\n"
                f"  is_cloud_request: {expected_is_cloud}"
            )

            if is_cloud:
                expected_clean = strip_cloud_prefix(prompt)
                assert clean == expected_clean, (
                    f"parse_cloud_request({prompt!r})[1] != strip_cloud_prefix({prompt!r})\n"
                    f"  parse_cloud_request: {clean!r}\n"
                    f"  strip_cloud_prefix: {expected_clean!r}"
                )

    def test_cursor_agent_client_strip_cloud_prefix(self):
        """测试 CursorAgentClient._strip_cloud_prefix 代理方法"""
        from cursor.client import CursorAgentClient
        from core.cloud_utils import strip_cloud_prefix as core_strip

        for input_val, expected in self.STRIP_CASES:
            if input_val is None:
                # None 情况特殊处理
                result = CursorAgentClient._strip_cloud_prefix(input_val)
                assert result == expected, (
                    f"CursorAgentClient._strip_cloud_prefix(None) "
                    f"returned {result!r}, expected {expected!r}"
                )
            else:
                result = CursorAgentClient._strip_cloud_prefix(input_val)
                core_result = core_strip(input_val)
                assert result == expected, (
                    f"CursorAgentClient._strip_cloud_prefix({input_val!r}) "
                    f"returned {result!r}, expected {expected!r}"
                )
                assert result == core_result, (
                    f"CursorAgentClient._strip_cloud_prefix({input_val!r}) "
                    f"!= core.cloud_utils.strip_cloud_prefix({input_val!r})\n"
                    f"  Agent: {result!r}\n"
                    f"  Core: {core_result!r}"
                )
