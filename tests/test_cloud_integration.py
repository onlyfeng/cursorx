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
        executor = CloudAgentExecutor()

        with patch.object(
            executor._auth_manager,
            "authenticate",
            return_value=AuthStatus(
                authenticated=False,
                error=AuthError("API Key 无效", AuthErrorCode.INVALID_API_KEY),
            ),
        ):
            result = await executor.execute(prompt="测试")

            assert result.success is False
            assert "认证" in result.error

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
        async def slow_communicate():
            await asyncio.sleep(100)
            return (b"", b"")

        mock_process = AsyncMock()
        mock_process.communicate = slow_communicate

        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    result = await cursor_cloud_client.push_to_cloud("session-slow")

                    assert result.success is False
                    assert "超时" in result.error

    @pytest.mark.asyncio
    async def test_resume_from_cloud_timeout(self, cursor_cloud_client):
        """测试从云端恢复超时"""
        with patch.object(
            cursor_cloud_client.auth_manager,
            "authenticate",
            return_value=AuthStatus(authenticated=True),
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError()):
                # 使用较短的超时
                options = CloudTaskOptions(timeout=1)

                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
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
