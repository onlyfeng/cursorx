"""执行模式端到端测试

测试 cursor/executor.py 中各种执行模式的功能，包括：
- CLI 模式执行工作流
- Cloud 模式执行工作流（使用 CursorCloudClient）
- Auto 模式自动选择和回退
- Auto 模式 Cloud 冷却策略
- ExecutorFactory 工厂创建

使用 Mock 替代真实 Cursor CLI/Cloud 调用
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cursor.client import CursorAgentConfig, CursorAgentResult
from cursor.cloud_client import AuthStatus, CloudAuthConfig
from cursor.cloud.client import CloudAgentResult, CursorCloudClient
from cursor.cloud.task import CloudTask, CloudTaskOptions, TaskStatus
from cursor.executor import (
    AgentExecutorFactory,
    AgentResult,
    AskAgentExecutor,
    AutoAgentExecutor,
    CLIAgentExecutor,
    CloudAgentExecutor,
    ExecutionMode,
    PlanAgentExecutor,
    execute_agent,
)


# ==================== TestCLIExecutionMode ====================


class TestCLIExecutionMode:
    """CLI 执行模式测试"""

    @pytest.fixture
    def cli_config(self) -> CursorAgentConfig:
        """创建 CLI 配置"""
        return CursorAgentConfig(
            model="opus-4.5-thinking",
            timeout=300,
            force_write=True,
        )

    @pytest.fixture
    def mock_cli_result(self) -> CursorAgentResult:
        """创建 Mock CLI 执行结果"""
        return CursorAgentResult(
            success=True,
            output="任务执行完成",
            error=None,
            exit_code=0,
            duration=1.5,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            files_modified=["src/main.py", "tests/test_main.py"],
        )

    @pytest.mark.asyncio
    async def test_cli_mode_workflow(
        self,
        cli_config: CursorAgentConfig,
        mock_cli_result: CursorAgentResult,
    ) -> None:
        """测试 CLI 模式完整工作流"""
        executor = CLIAgentExecutor(config=cli_config)

        with patch.object(
            executor._client, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_cli_result

            result = await executor.execute(
                prompt="分析代码结构",
                context={"files": ["main.py"]},
                working_directory="/tmp/workspace",
                timeout=60,
            )

            # 验证执行调用
            mock_execute.assert_called_once_with(
                instruction="分析代码结构",
                context={"files": ["main.py"]},
                working_directory="/tmp/workspace",
                timeout=60,
            )

            # 验证结果转换
            assert result.success is True
            assert result.output == "任务执行完成"
            assert result.executor_type == "cli"
            assert result.files_modified == ["src/main.py", "tests/test_main.py"]
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_cli_mode_configuration(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 CLI 配置传递"""
        # 测试默认配置
        executor_default = CLIAgentExecutor()
        assert executor_default.executor_type == "cli"
        assert executor_default.config is not None

        # 测试自定义配置
        executor_custom = CLIAgentExecutor(config=cli_config)
        assert executor_custom.config.model == "opus-4.5-thinking"
        assert executor_custom.config.timeout == 300
        assert executor_custom.config.force_write is True

        # 测试 mode 参数覆盖
        executor_plan = CLIAgentExecutor(config=cli_config, mode="plan")
        assert executor_plan.cli_mode == "plan"

        executor_ask = CLIAgentExecutor(config=cli_config, mode="ask")
        assert executor_ask.cli_mode == "ask"

    @pytest.mark.asyncio
    async def test_cli_mode_availability_check(self) -> None:
        """测试 CLI 可用性检查"""
        executor = CLIAgentExecutor()

        with patch.object(
            executor._client, "check_agent_available", return_value=True
        ):
            result = await executor.check_available()
            assert result is True

        # 重新创建执行器测试不可用情况
        executor2 = CLIAgentExecutor()
        with patch.object(
            executor2._client, "check_agent_available", return_value=False
        ):
            result = await executor2.check_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_cli_mode_with_retry(
        self,
        cli_config: CursorAgentConfig,
        mock_cli_result: CursorAgentResult,
    ) -> None:
        """测试 CLI 模式带重试执行"""
        executor = CLIAgentExecutor(config=cli_config)

        with patch.object(
            executor._client, "execute_with_retry", new_callable=AsyncMock
        ) as mock_retry:
            mock_retry.return_value = mock_cli_result

            result = await executor.execute_with_retry(
                prompt="重构代码",
                context={"target": "module.py"},
                max_retries=3,
            )

            mock_retry.assert_called_once()
            assert result.success is True
            assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_cli_mode_error_handling(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 CLI 模式错误处理"""
        executor = CLIAgentExecutor(config=cli_config)

        error_result = CursorAgentResult(
            success=False,
            output="",
            error="命令执行失败: timeout",
            exit_code=1,
            duration=30.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            files_modified=[],
        )

        with patch.object(
            executor._client, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = error_result

            result = await executor.execute(prompt="失败的任务")

            assert result.success is False
            assert result.error == "命令执行失败: timeout"
            assert result.exit_code == 1


# ==================== TestCloudExecutionMode ====================


class TestCloudExecutionMode:
    """Cloud 执行模式测试

    使用 Mock CursorCloudClient 验证:
    - 成功执行路径
    - 失败执行路径
    - 错误处理
    - 后台任务提交
    - 会话恢复
    """

    @pytest.fixture
    def cloud_auth_config(self) -> CloudAuthConfig:
        """创建 Cloud 认证配置"""
        return CloudAuthConfig(
            api_key="test-api-key-12345",
            base_url="https://api.cursor.com",
        )

    @pytest.fixture
    def agent_config(self) -> CursorAgentConfig:
        """创建 Agent 配置"""
        return CursorAgentConfig(
            model="gpt-5.2-high",
            timeout=600,
        )

    @pytest.fixture
    def mock_cloud_client(self) -> MagicMock:
        """创建 Mock CursorCloudClient"""
        client = MagicMock(spec=CursorCloudClient)
        client.execute = AsyncMock()
        client.submit_task = AsyncMock()
        client.wait_for_completion = AsyncMock()
        client.resume_from_cloud = AsyncMock()
        client.is_cloud_request = CursorCloudClient.is_cloud_request
        return client

    @pytest.fixture
    def mock_cloud_success_result(self) -> CloudAgentResult:
        """创建成功的 Cloud 执行结果"""
        task = CloudTask(
            task_id="task-12345",
            status=TaskStatus.COMPLETED,
            prompt="测试任务",
            output="任务执行成功",
        )
        return CloudAgentResult(
            success=True,
            task=task,
            output="任务执行成功，已完成代码分析。",
            duration=2.5,
            files_modified=["src/main.py", "tests/test_main.py"],
        )

    @pytest.fixture
    def mock_cloud_failure_result(self) -> CloudAgentResult:
        """创建失败的 Cloud 执行结果"""
        task = CloudTask(
            task_id="task-failed-123",
            status=TaskStatus.FAILED,
            prompt="失败任务",
            error="任务执行失败: 资源不足",
        )
        return CloudAgentResult(
            success=False,
            task=task,
            output="",
            error="任务执行失败: 资源不足",
        )

    @pytest.mark.asyncio
    async def test_cloud_mode_success_workflow(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """测试 Cloud 模式成功执行工作流"""
        # 使用注入的 mock client
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
            cloud_client=mock_cloud_client,
        )

        # Mock 认证成功
        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        # 配置 mock client 返回成功结果
        mock_cloud_client.execute.return_value = mock_cloud_success_result

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(
                prompt="分析代码结构",
                context={"files": ["app.py"]},
                working_directory="/tmp/workspace",
            )

            # 验证 Cloud Client 被正确调用
            mock_cloud_client.execute.assert_called_once()
            call_args = mock_cloud_client.execute.call_args

            # 验证执行结果
            assert result.success is True
            assert result.executor_type == "cloud"
            assert "任务执行成功" in result.output
            assert result.session_id == "task-12345"

    @pytest.mark.asyncio
    async def test_cloud_mode_failure_workflow(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_failure_result: CloudAgentResult,
    ) -> None:
        """测试 Cloud 模式失败执行工作流"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        # 配置 mock client 返回失败结果
        mock_cloud_client.execute.return_value = mock_cloud_failure_result

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(prompt="失败的任务")

            # 验证失败处理
            assert result.success is False
            assert result.executor_type == "cloud"
            assert "资源不足" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cloud_authentication(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """测试 Cloud 认证流程"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 测试认证成功
        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        mock_cloud_client.execute.return_value = mock_cloud_success_result

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(prompt="测试认证")
            assert result.executor_type == "cloud"
            assert result.success is True

        # 测试认证失败
        executor2 = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )
        mock_auth_failed = AuthStatus(
            authenticated=False,
            error="Invalid API key",
        )

        with patch.object(
            executor2._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_failed,
        ):
            result = await executor2.execute(prompt="测试认证失败")
            assert result.success is False
            assert "认证失败" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cloud_background_task_submission(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试 Cloud 后台任务提交"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        # 创建后台提交成功的结果
        background_task = CloudTask(
            task_id="bg-task-001",
            status=TaskStatus.QUEUED,
            prompt="后台任务",
        )
        mock_submit_result = CloudAgentResult(
            success=True,
            task=background_task,
            output="任务已提交",
        )

        mock_cloud_client.submit_task.return_value = mock_submit_result

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.submit_background_task(
                prompt="执行长时间任务",
                options=CloudTaskOptions(timeout=600),
            )

            # 验证后台任务提交
            mock_cloud_client.submit_task.assert_called_once()
            assert result.success is True
            assert result.session_id == "bg-task-001"

    @pytest.mark.asyncio
    async def test_cloud_wait_for_task(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """测试等待后台任务完成"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_cloud_client.wait_for_completion.return_value = mock_cloud_success_result

        result = await executor.wait_for_task(
            task_id="task-12345",
            timeout=300,
        )

        mock_cloud_client.wait_for_completion.assert_called_once_with(
            task_id="task-12345",
            timeout=300,
        )
        assert result.success is True
        assert result.session_id == "task-12345"

    @pytest.mark.asyncio
    async def test_cloud_session_resume(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """测试云端会话恢复"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_cloud_client.resume_from_cloud.return_value = mock_cloud_success_result

        result = await executor.resume_session(
            session_id="session-abc123",
            prompt="继续之前的任务",
        )

        mock_cloud_client.resume_from_cloud.assert_called_once_with(
            task_id="session-abc123",
            local=True,
            prompt="继续之前的任务",
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_cloud_availability_check_success(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试 Cloud 可用性检查 - 成功"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_success = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_success,
        ):
            result = await executor.check_available()
            # 认证成功，Cloud 可用
            assert result is True

    @pytest.mark.asyncio
    async def test_cloud_availability_check_failure(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试 Cloud 可用性检查 - 失败"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_failed = AuthStatus(
            authenticated=False,
            error="No API key",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_failed,
        ):
            result = await executor.check_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_cloud_timeout_handling(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试 Cloud 超时处理"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        # 模拟 Cloud Client 执行超时
        mock_cloud_client.execute.side_effect = asyncio.TimeoutError()

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(
                prompt="长时间任务",
                timeout=1,
            )

            assert result.success is False
            assert "超时" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cloud_error_handling(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试 Cloud 错误处理"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        # 模拟 Cloud Client 抛出异常
        mock_cloud_client.execute.side_effect = RuntimeError("网络连接失败")

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(prompt="测试异常")

            assert result.success is False
            assert "网络连接失败" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cloud_reset_availability_cache(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
    ) -> None:
        """测试重置可用性缓存"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 设置缓存
        executor._available = True
        executor._auth_status = AuthStatus(authenticated=True)

        # 重置缓存
        executor.reset_availability_cache()

        assert executor._available is None
        assert executor._auth_status is None

    @pytest.mark.asyncio
    async def test_cloud_executor_normal_prompt_uses_cloud_path(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """测试 CloudAgentExecutor.execute(prompt='普通任务') 必须走云端提交路径

        验证普通任务（不带 & 前缀）也会通过 CloudClientFactory.execute_task 提交到云端。
        这是 Cloud 执行模式的核心保证。
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        # 使用 patch 验证 CloudClientFactory.execute_task 被调用
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.return_value = mock_cloud_success_result

            with patch.object(
                executor._auth_manager,
                "authenticate",
                new_callable=AsyncMock,
                return_value=mock_auth_status,
            ):
                # 执行普通任务（无 & 前缀）
                result = await executor.execute(
                    prompt="普通任务",
                    working_directory="/tmp/workspace",
                )

                # 验证 CloudClientFactory.execute_task 被调用
                mock_execute_task.assert_called_once()
                call_kwargs = mock_execute_task.call_args.kwargs

                # 验证任务被提交到云端
                assert call_kwargs.get("prompt") == "普通任务"
                assert call_kwargs.get("wait") is True

                # 验证结果
                assert result.success is True
                assert result.executor_type == "cloud"

    @pytest.mark.asyncio
    async def test_cloud_executor_verifies_cloud_client_submit_or_execute(
        self,
        cloud_auth_config: CloudAuthConfig,
        mock_cloud_client: MagicMock,
        mock_cloud_success_result: CloudAgentResult,
    ) -> None:
        """验证 CloudAgentExecutor 通过 CursorCloudClient 提交任务

        此测试通过 mock CursorCloudClient.execute 或 submit_task 验证调用链。
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        # 使用 CloudClientFactory.execute_task 作为入口点
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_factory_execute:
            mock_factory_execute.return_value = mock_cloud_success_result

            with patch.object(
                executor._auth_manager,
                "authenticate",
                new_callable=AsyncMock,
                return_value=mock_auth_status,
            ):
                result = await executor.execute(prompt="测试云端路径")

                # 验证工厂方法被调用
                mock_factory_execute.assert_called_once()

                # 验证 prompt 被正确包装并提交
                call_kwargs = mock_factory_execute.call_args.kwargs
                assert "测试云端路径" in call_kwargs.get("prompt", "")


# ==================== TestAutoExecutionMode ====================


class TestAutoExecutionMode:
    """自动执行模式测试"""

    @pytest.fixture
    def cli_config(self) -> CursorAgentConfig:
        """创建 CLI 配置"""
        return CursorAgentConfig(
            model="opus-4.5-thinking",
            timeout=300,
        )

    @pytest.fixture
    def cloud_auth_config(self) -> CloudAuthConfig:
        """创建 Cloud 认证配置"""
        return CloudAuthConfig(
            api_key="test-api-key",
        )

    @pytest.mark.asyncio
    async def test_auto_mode_fallback(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 失败后回退到 CLI"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # Mock Cloud 不可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # Mock CLI 可用
            with patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                # Mock CLI 执行成功
                cli_result = AgentResult(
                    success=True,
                    output="CLI 执行完成",
                    executor_type="cli",
                )

                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_result,
                ):
                    result = await executor.execute(prompt="测试回退")

                    # 应该回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"
                    assert result.output == "CLI 执行完成"

    @pytest.mark.asyncio
    async def test_auto_mode_preference(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试执行器偏好"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 初始时没有偏好
        assert executor._preferred_executor is None

        # Mock Cloud 不可用，CLI 可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                cli_result = AgentResult(
                    success=True,
                    output="执行成功",
                    executor_type="cli",
                )

                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_result,
                ):
                    # 第一次执行，选择执行器
                    await executor.execute(prompt="第一次执行")
                    assert executor._preferred_executor is not None

                    # 第二次执行，使用已选择的执行器
                    await executor.execute(prompt="第二次执行")

    @pytest.mark.asyncio
    async def test_auto_mode_reset(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试偏好重置"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 设置偏好
        executor._preferred_executor = executor._cli_executor

        # 验证偏好已设置
        assert executor._preferred_executor is not None

        # 重置偏好
        executor.reset_preference()

        # 验证偏好已清除
        assert executor._preferred_executor is None

    @pytest.mark.asyncio
    async def test_auto_mode_cloud_then_fallback(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 执行失败后回退到 CLI"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # Mock Cloud 可用但执行失败
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            cloud_fail_result = AgentResult(
                success=False,
                output="",
                error="Cloud 执行失败",
                executor_type="cloud",
            )

            cli_success_result = AgentResult(
                success=True,
                output="CLI 回退成功",
                executor_type="cli",
            )

            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ):
                    result = await executor.execute(prompt="测试 Cloud 失败回退")

                    # 应该回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_auto_mode_cloud_error_fallback_to_cli(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 执行异常后回退到 CLI（使用 mock CursorCloudClient）"""
        # 创建 mock cloud client
        mock_cloud_client = MagicMock(spec=CursorCloudClient)
        mock_cloud_client.execute = AsyncMock()

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 注入 mock cloud client 到 cloud executor
        executor._cloud_executor._cloud_client = mock_cloud_client

        # Mock Cloud 可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Cloud 执行返回网络错误
            cloud_error_result = CloudAgentResult(
                success=False,
                error="网络连接失败",
            )
            mock_cloud_client.execute.return_value = cloud_error_result

            cli_success_result = AgentResult(
                success=True,
                output="CLI 成功执行",
                executor_type="cli",
            )

            with patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ):
                result = await executor.execute(prompt="测试网络错误回退")

                # 应该回退到 CLI
                assert result.success is True
                assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_auto_mode_cloud_timeout_fallback(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 超时后回退到 CLI"""
        mock_cloud_client = MagicMock(spec=CursorCloudClient)
        mock_cloud_client.execute = AsyncMock()

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        executor._cloud_executor._cloud_client = mock_cloud_client

        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Cloud 执行返回超时错误
            cloud_timeout_result = CloudAgentResult(
                success=False,
                error="执行超时",
            )
            mock_cloud_client.execute.return_value = cloud_timeout_result

            cli_success_result = AgentResult(
                success=True,
                output="CLI 执行完成",
                executor_type="cli",
            )

            with patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ):
                result = await executor.execute(prompt="测试超时回退")

                assert result.success is True
                assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_auto_mode_availability(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Auto 模式可用性检查"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 测试两者都可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await executor.check_available()
                assert result is True

        # 测试只有 CLI 可用
        executor2 = AutoAgentExecutor(cli_config=cli_config)
        with patch.object(
            executor2._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch.object(
                executor2._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await executor2.check_available()
                assert result is True

        # 测试两者都不可用
        executor3 = AutoAgentExecutor(cli_config=cli_config)
        with patch.object(
            executor3._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch.object(
                executor3._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=False,
            ):
                result = await executor3.check_available()
                assert result is False

    @pytest.mark.asyncio
    async def test_auto_executor_cloud_available_uses_cloud_path(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 AutoAgentExecutor 在 Cloud 可用时优先使用云端路径

        验证当 Cloud 可用时，AutoAgentExecutor 会将任务路由到 CloudAgentExecutor。
        """
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        cloud_success_result = AgentResult(
            success=True,
            output="Cloud 执行成功",
            executor_type="cloud",
            session_id="cloud-session-123",
        )

        # Mock Cloud 可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Mock Cloud 执行成功
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ) as mock_cloud_execute:
                result = await executor.execute(prompt="测试任务")

                # 验证 Cloud 执行器被调用
                mock_cloud_execute.assert_called_once()
                call_kwargs = mock_cloud_execute.call_args.kwargs
                assert call_kwargs.get("prompt") == "测试任务"

                # 验证结果来自 Cloud
                assert result.success is True
                assert result.executor_type == "cloud"

    @pytest.mark.asyncio
    async def test_auto_executor_cloud_unavailable_falls_back_to_cli(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 AutoAgentExecutor 在 Cloud 不可用时回退到 CLI

        验证 Cloud 不可用时，任务会被路由到 CLIAgentExecutor。
        """
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        # Mock Cloud 不可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # Mock CLI 可用并执行成功
            with patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ) as mock_cli_execute:
                    result = await executor.execute(prompt="测试回退任务")

                    # 验证 CLI 执行器被调用
                    mock_cli_execute.assert_called_once()

                    # 验证结果来自 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"

    @pytest.mark.asyncio
    async def test_auto_executor_verifies_cloud_submission_path(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 AutoAgentExecutor 通过 Cloud 执行器提交到云端

        验证 AutoAgentExecutor 使用的 CloudAgentExecutor 最终会调用
        CloudClientFactory.execute_task 提交任务到云端。
        """
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # Mock Cloud 可用
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Mock CloudClientFactory.execute_task
            with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output = "任务执行完成"
                mock_result.error = None
                mock_result.files_modified = ["test.py"]
                mock_result.task = MagicMock()
                mock_result.task.task_id = "auto-cloud-session"
                mock_result.to_dict = MagicMock(return_value={})
                mock_execute_task.return_value = mock_result

                result = await executor.execute(prompt="验证云端提交路径")

                # 验证 CloudClientFactory.execute_task 被调用
                mock_execute_task.assert_called_once()
                call_kwargs = mock_execute_task.call_args.kwargs
                assert "验证云端提交路径" in call_kwargs.get("prompt", "")

                # 验证结果
                assert result.success is True
                assert result.executor_type == "cloud"


# ==================== TestAutoExecutorCooldown ====================


class TestAutoExecutorCooldown:
    """Auto 执行器 Cloud 冷却策略测试

    测试 Cloud 失败后的冷却策略：
    - 冷却期内不尝试 Cloud
    - 冷却期结束后重新尝试 Cloud
    - 默认行为不变（不引入频繁切换导致的抖动）
    """

    @pytest.fixture
    def cli_config(self) -> CursorAgentConfig:
        """创建 CLI 配置"""
        return CursorAgentConfig(
            model="opus-4.5-thinking",
            timeout=300,
        )

    @pytest.fixture
    def cloud_auth_config(self) -> CloudAuthConfig:
        """创建 Cloud 认证配置"""
        return CloudAuthConfig(
            api_key="test-api-key",
        )

    def test_cooldown_default_values(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试冷却策略默认值"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 默认启用冷却
        assert executor._enable_cooldown is True
        # 默认冷却时间 300 秒
        assert executor.cloud_cooldown_seconds == 300
        # 初始状态不在冷却期
        assert executor.is_cloud_in_cooldown is False
        assert executor.cloud_cooldown_remaining is None
        # 初始失败计数为 0
        assert executor._cloud_failure_count == 0

    def test_cooldown_custom_values(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试自定义冷却配置"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=60,  # 自定义 60 秒
            enable_cooldown=True,
        )

        assert executor.cloud_cooldown_seconds == 60
        assert executor._enable_cooldown is True

    def test_cooldown_disabled(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试禁用冷却策略"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            enable_cooldown=False,
        )

        assert executor._enable_cooldown is False
        # 禁用冷却时，is_cloud_in_cooldown 始终为 False
        assert executor.is_cloud_in_cooldown is False

        # 即使手动设置冷却时间，禁用冷却时也不生效
        executor._cloud_cooldown_until = datetime.now() + timedelta(hours=1)
        assert executor.is_cloud_in_cooldown is False

    def test_start_cooldown(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试启动冷却"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=60,
        )

        # 启动冷却
        executor._start_cloud_cooldown()

        # 验证冷却状态
        assert executor.is_cloud_in_cooldown is True
        assert executor._cloud_failure_count == 1
        assert executor._cloud_cooldown_until is not None
        assert executor.cloud_cooldown_remaining is not None
        assert 0 < executor.cloud_cooldown_remaining <= 60

    def test_reset_cooldown(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试重置冷却"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 先启动冷却
        executor._start_cloud_cooldown()
        assert executor.is_cloud_in_cooldown is True

        # 重置冷却
        executor._reset_cloud_cooldown()

        # 验证冷却已重置
        assert executor.is_cloud_in_cooldown is False
        assert executor._cloud_failure_count == 0
        assert executor._cloud_cooldown_until is None

    def test_cooldown_expiry(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试冷却过期检测"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=60,
        )

        # 设置已过期的冷却时间
        executor._cloud_cooldown_until = datetime.now() - timedelta(seconds=1)

        # 检查过期应返回 True 并重置冷却
        with patch.object(
            executor._cloud_executor,
            "reset_availability_cache",
        ) as mock_reset:
            expired = executor._check_cooldown_expired()
            assert expired is True
            assert executor._cloud_cooldown_until is None
            mock_reset.assert_called_once()

    def test_manual_reset_cooldown(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试手动重置冷却"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 设置冷却和偏好
        executor._start_cloud_cooldown()
        executor._preferred_executor = executor._cli_executor

        # 手动重置
        with patch.object(
            executor._cloud_executor,
            "reset_availability_cache",
        ) as mock_reset:
            executor.reset_cooldown()

            # 验证所有状态已重置
            assert executor._cloud_cooldown_until is None
            assert executor._cloud_failure_count == 0
            assert executor._preferred_executor is None
            mock_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_cloud_failure_triggers_cooldown(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 失败后触发冷却"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=60,
        )

        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud 执行失败",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        # Mock Cloud 可用但执行失败
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ):
                    result = await executor.execute(prompt="测试冷却触发")

                    # 验证回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"

                    # 验证冷却已启动
                    assert executor.is_cloud_in_cooldown is True
                    assert executor._cloud_failure_count == 1
                    assert executor._preferred_executor == executor._cli_executor

    @pytest.mark.asyncio
    async def test_cooldown_prevents_cloud_retry(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试冷却期内不重试 Cloud"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=300,
        )

        # 手动设置冷却状态
        executor._cloud_cooldown_until = datetime.now() + timedelta(seconds=300)
        executor._preferred_executor = executor._cli_executor

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        # Mock Cloud check_available（不应被调用）
        cloud_check_mock = AsyncMock(return_value=True)
        cli_execute_mock = AsyncMock(return_value=cli_success_result)

        with patch.object(
            executor._cloud_executor,
            "check_available",
            cloud_check_mock,
        ):
            with patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    cli_execute_mock,
                ):
                    result = await executor.execute(prompt="冷却期内执行")

                    # 验证使用 CLI（不应尝试 Cloud）
                    assert result.success is True
                    assert result.executor_type == "cli"

                    # Cloud check_available 不应被调用（因为在冷却期）
                    cloud_check_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_expiry_retries_cloud(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试冷却过期后重新尝试 Cloud"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=60,
        )

        # 设置已过期的冷却
        executor._cloud_cooldown_until = datetime.now() - timedelta(seconds=1)
        executor._preferred_executor = executor._cli_executor

        cloud_success_result = AgentResult(
            success=True,
            output="Cloud 执行成功",
            executor_type="cloud",
        )

        # Mock Cloud 可用并执行成功
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "reset_availability_cache",
            ):
                with patch.object(
                    executor._cloud_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cloud_success_result,
                ):
                    result = await executor.execute(prompt="冷却过期后执行")

                    # 验证切换回 Cloud
                    assert result.success is True
                    assert result.executor_type == "cloud"
                    assert executor._preferred_executor == executor._cloud_executor

    @pytest.mark.asyncio
    async def test_cloud_success_resets_cooldown(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 成功后重置冷却状态"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 设置一些冷却状态（模拟之前的失败）
        executor._cloud_failure_count = 2

        cloud_success_result = AgentResult(
            success=True,
            output="Cloud 执行成功",
            executor_type="cloud",
        )

        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ):
                result = await executor.execute(prompt="测试成功重置")

                # 验证 Cloud 成功
                assert result.success is True
                assert result.executor_type == "cloud"

                # 验证冷却状态已重置
                assert executor._cloud_failure_count == 0
                assert executor._cloud_cooldown_until is None

    @pytest.mark.asyncio
    async def test_multiple_failures_accumulate_count(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试多次失败累计失败计数"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cloud_cooldown_seconds=1,  # 短冷却便于测试
        )

        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud 执行失败",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        # 第一次失败
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ):
                    await executor.execute(prompt="第一次失败")
                    assert executor._cloud_failure_count == 1

        # 等待冷却过期
        executor._cloud_cooldown_until = datetime.now() - timedelta(seconds=1)

        # 第二次失败
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "reset_availability_cache",
            ):
                with patch.object(
                    executor._cloud_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cloud_fail_result,
                ):
                    with patch.object(
                        executor._cli_executor,
                        "execute",
                        new_callable=AsyncMock,
                        return_value=cli_success_result,
                    ):
                        await executor.execute(prompt="第二次失败")
                        assert executor._cloud_failure_count == 2

    @pytest.mark.asyncio
    async def test_cooldown_disabled_no_delay(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试禁用冷却时 Cloud 失败后立即重试"""
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            enable_cooldown=False,  # 禁用冷却
        )

        cloud_fail_result = AgentResult(
            success=False,
            error="Cloud 执行失败",
            executor_type="cloud",
        )

        cli_success_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        cloud_success_result = AgentResult(
            success=True,
            output="Cloud 执行成功",
            executor_type="cloud",
        )

        # 第一次执行：Cloud 失败，回退到 CLI
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ):
                with patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ):
                    result = await executor.execute(prompt="禁用冷却-失败")
                    assert result.executor_type == "cli"

        # 重置偏好以便第二次重新选择
        executor.reset_preference()

        # 第二次执行：应该立即尝试 Cloud（因为冷却被禁用）
        with patch.object(
            executor._cloud_executor,
            "check_available",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_cloud_check:
            with patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ):
                result = await executor.execute(prompt="禁用冷却-重试")

                # 验证 Cloud 被检查和使用
                mock_cloud_check.assert_called()
                assert result.executor_type == "cloud"

    def test_default_behavior_unchanged(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试默认行为不变（向后兼容性）

        验证启用冷却策略后，对于正常使用场景，行为与之前一致：
        - 首次执行优先 Cloud
        - Cloud 成功继续使用 Cloud
        - Cloud 不可用时回退到 CLI
        """
        # 默认参数创建执行器
        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )

        # 验证默认配置
        assert executor._enable_cooldown is True
        assert executor.cloud_cooldown_seconds == 300
        assert executor._preferred_executor is None
        assert executor.is_cloud_in_cooldown is False

        # 验证属性访问器
        assert executor.executor_type == "auto"
        assert executor.cli_executor is not None
        assert executor.cloud_executor is not None


# ==================== TestExecutorFactory ====================


class TestExecutorFactory:
    """执行器工厂测试"""

    @pytest.fixture
    def cli_config(self) -> CursorAgentConfig:
        """创建 CLI 配置"""
        return CursorAgentConfig(
            model="opus-4.5-thinking",
            timeout=300,
        )

    @pytest.fixture
    def cloud_auth_config(self) -> CloudAuthConfig:
        """创建 Cloud 认证配置"""
        return CloudAuthConfig(
            api_key="test-api-key",
        )

    def test_factory_create_all_modes(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试工厂创建所有执行模式"""
        # 测试 CLI 模式
        cli_executor = AgentExecutorFactory.create(
            mode=ExecutionMode.CLI,
            cli_config=cli_config,
        )
        assert isinstance(cli_executor, CLIAgentExecutor)
        assert cli_executor.executor_type == "cli"

        # 测试 Cloud 模式
        cloud_executor = AgentExecutorFactory.create(
            mode=ExecutionMode.CLOUD,
            cloud_auth_config=cloud_auth_config,
        )
        assert isinstance(cloud_executor, CloudAgentExecutor)
        assert cloud_executor.executor_type == "cloud"

        # 测试 Auto 模式
        auto_executor = AgentExecutorFactory.create(
            mode=ExecutionMode.AUTO,
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
        )
        assert isinstance(auto_executor, AutoAgentExecutor)
        assert auto_executor.executor_type == "auto"

        # 测试 Plan 模式
        plan_executor = AgentExecutorFactory.create(
            mode=ExecutionMode.PLAN,
            cli_config=cli_config,
        )
        assert isinstance(plan_executor, PlanAgentExecutor)
        assert plan_executor.executor_type == "plan"

        # 测试 Ask 模式
        ask_executor = AgentExecutorFactory.create(
            mode=ExecutionMode.ASK,
            cli_config=cli_config,
        )
        assert isinstance(ask_executor, AskAgentExecutor)
        assert ask_executor.executor_type == "ask"

    def test_mode_switching(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试模式切换"""
        # 创建不同模式的执行器
        executors = {}

        for mode in ExecutionMode:
            executor = AgentExecutorFactory.create(
                mode=mode,
                cli_config=cli_config,
                cloud_auth_config=cloud_auth_config,
            )
            executors[mode] = executor

        # 验证每个模式都创建了正确类型的执行器
        assert executors[ExecutionMode.CLI].executor_type == "cli"
        assert executors[ExecutionMode.CLOUD].executor_type == "cloud"
        assert executors[ExecutionMode.AUTO].executor_type == "auto"
        assert executors[ExecutionMode.PLAN].executor_type == "plan"
        assert executors[ExecutionMode.ASK].executor_type == "ask"

    def test_factory_create_from_config(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试从配置创建执行器"""
        # 默认应创建 AUTO 模式
        executor = AgentExecutorFactory.create_from_config(config=cli_config)
        assert executor.executor_type == "auto"

    @pytest.mark.asyncio
    async def test_factory_create_best_available(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试创建最佳可用执行器"""
        # Mock Cloud 不可用，CLI 可用
        with patch(
            "cursor.executor.CloudAgentExecutor.check_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch(
                "cursor.executor.CLIAgentExecutor.check_available",
                new_callable=AsyncMock,
                return_value=True,
            ):
                executor = await AgentExecutorFactory.create_best_available(
                    cli_config=cli_config,
                    cloud_auth_config=cloud_auth_config,
                )
                # 应该返回 CLI 执行器
                assert executor.executor_type == "cli"

    def test_factory_invalid_mode(self) -> None:
        """测试无效模式处理"""
        with pytest.raises(ValueError, match="未知的执行模式"):
            AgentExecutorFactory.create(mode="invalid_mode")


# ==================== TestPlanAndAskExecutors ====================


class TestPlanAndAskExecutors:
    """Plan 和 Ask 执行器测试"""

    @pytest.fixture
    def cli_config(self) -> CursorAgentConfig:
        """创建 CLI 配置"""
        return CursorAgentConfig(
            model="opus-4.5-thinking",
            force_write=True,  # 这个应该被覆盖
        )

    @pytest.mark.asyncio
    async def test_plan_executor_readonly(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 Plan 执行器只读模式"""
        executor = PlanAgentExecutor(config=cli_config)

        # 验证 force_write 被强制设为 False
        assert executor.config.force_write is False
        assert executor.config.mode == "plan"
        assert executor.executor_type == "plan"

    @pytest.mark.asyncio
    async def test_ask_executor_readonly(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 Ask 执行器只读模式"""
        executor = AskAgentExecutor(config=cli_config)

        # 验证 force_write 被强制设为 False
        assert executor.config.force_write is False
        assert executor.config.mode == "ask"
        assert executor.executor_type == "ask"

    @pytest.mark.asyncio
    async def test_plan_executor_execution(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 Plan 执行器执行"""
        executor = PlanAgentExecutor(config=cli_config)

        mock_result = CursorAgentResult(
            success=True,
            output="任务规划完成",
            exit_code=0,
            duration=1.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            files_modified=[],
        )

        with patch.object(
            executor._cli_executor._client,
            "execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await executor.execute(prompt="规划任务分解")

            assert result.success is True
            assert result.executor_type == "plan"
            assert result.output == "任务规划完成"

    @pytest.mark.asyncio
    async def test_ask_executor_execution(
        self,
        cli_config: CursorAgentConfig,
    ) -> None:
        """测试 Ask 执行器执行"""
        executor = AskAgentExecutor(config=cli_config)

        mock_result = CursorAgentResult(
            success=True,
            output="这是代码解释...",
            exit_code=0,
            duration=0.5,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            files_modified=[],
        )

        with patch.object(
            executor._cli_executor._client,
            "execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await executor.execute(prompt="解释这段代码")

            assert result.success is True
            assert result.executor_type == "ask"
            assert result.output == "这是代码解释..."


# ==================== TestConvenienceFunctions ====================


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.mark.asyncio
    async def test_execute_agent_function(self) -> None:
        """测试 execute_agent 便捷函数"""
        mock_result = AgentResult(
            success=True,
            output="执行完成",
            executor_type="cli",
        )

        with patch(
            "cursor.executor.AgentExecutorFactory.create"
        ) as mock_factory:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_factory.return_value = mock_executor

            result = await execute_agent(
                prompt="测试任务",
                context={"key": "value"},
                mode=ExecutionMode.CLI,
            )

            # 验证工厂调用
            mock_factory.assert_called_once_with(
                mode=ExecutionMode.CLI,
                cli_config=None,
            )

            # 验证执行调用
            mock_executor.execute.assert_called_once_with(
                prompt="测试任务",
                context={"key": "value"},
            )

            # 验证结果
            assert result.success is True
            assert result.output == "执行完成"


# ==================== TestAgentResult ====================


class TestAgentResult:
    """AgentResult 模型测试"""

    def test_from_cli_result(self) -> None:
        """测试从 CLI 结果转换"""
        cli_result = CursorAgentResult(
            success=True,
            output="执行成功",
            error=None,
            exit_code=0,
            duration=2.5,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            files_modified=["file1.py", "file2.py"],
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.success is True
        assert agent_result.output == "执行成功"
        assert agent_result.executor_type == "cli"
        assert agent_result.files_modified == ["file1.py", "file2.py"]
        assert agent_result.duration == 2.5

    def test_from_cloud_result(self) -> None:
        """测试从 Cloud 结果转换"""
        agent_result = AgentResult.from_cloud_result(
            success=True,
            output="Cloud 执行完成",
            session_id="session-123",
            duration=5.0,
            raw_result={"status": "completed"},
        )

        assert agent_result.success is True
        assert agent_result.output == "Cloud 执行完成"
        assert agent_result.executor_type == "cloud"
        assert agent_result.session_id == "session-123"
        assert agent_result.raw_result == {"status": "completed"}

    def test_result_with_error(self) -> None:
        """测试带错误的结果"""
        result = AgentResult(
            success=False,
            output="",
            error="任务执行失败",
            exit_code=1,
            executor_type="cli",
        )

        assert result.success is False
        assert result.error == "任务执行失败"
        assert result.exit_code == 1
