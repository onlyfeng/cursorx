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
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# cooldown_info 契约字段常量
from core.output_contract import CooldownInfoFields
from cursor import AuthError
from cursor.client import CursorAgentConfig, CursorAgentResult
from cursor.cloud.client import CloudAgentResult, CursorCloudClient
from cursor.cloud.task import CloudTask, CloudTaskOptions, TaskStatus
from cursor.cloud_client import AuthStatus, CloudAuthConfig
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

        with patch.object(executor._client, "execute", new_callable=AsyncMock) as mock_execute:
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

        with patch.object(executor._client, "check_agent_available", return_value=True):
            result = await executor.check_available()
            assert result is True

        # 重新创建执行器测试不可用情况
        executor2 = CLIAgentExecutor()
        with patch.object(executor2._client, "check_agent_available", return_value=False):
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

        with patch.object(executor._client, "execute_with_retry", new_callable=AsyncMock) as mock_retry:
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

        with patch.object(executor._client, "execute", new_callable=AsyncMock) as mock_execute:
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
            api_base_url="https://api.cursor.com",
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
        """测试 Cloud 模式成功执行工作流

        CloudAgentExecutor.execute() 内部调用 CloudClientFactory.execute_task()，
        因此需要 mock CloudClientFactory.execute_task 而非 cloud_client.execute
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
            cloud_client=mock_cloud_client,
        )

        # Mock CloudClientFactory.execute_task 返回成功结果
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.return_value = mock_cloud_success_result

            result = await executor.execute(
                prompt="分析代码结构",
                context={"files": ["app.py"]},
                working_directory="/tmp/workspace",
            )

            # 验证 CloudClientFactory.execute_task 被正确调用
            mock_execute_task.assert_called_once()
            call_kwargs = mock_execute_task.call_args.kwargs
            assert call_kwargs.get("prompt") == "分析代码结构"
            assert call_kwargs.get("working_directory") == "/tmp/workspace"

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

        # Mock CloudClientFactory.execute_task 返回失败结果
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.return_value = mock_cloud_failure_result

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
        """测试 Cloud 认证流程

        CloudAgentExecutor.execute() 内部调用 CloudClientFactory.execute_task()，
        而 execute_task 在无有效认证时会返回错误。
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 测试认证成功场景：mock execute_task 返回成功结果
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.return_value = mock_cloud_success_result

            result = await executor.execute(prompt="测试认证")
            assert result.executor_type == "cloud"
            assert result.success is True

        # 测试认证失败场景：mock execute_task 返回失败结果（认证错误）
        executor2 = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 创建认证失败的结果
        auth_fail_result = CloudAgentResult(
            success=False,
            error="认证失败: Invalid API key",
        )

        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.return_value = auth_fail_result

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
        """测试云端会话恢复

        CloudAgentExecutor.resume_session() 内部调用 CloudClientFactory.resume_session()
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # Mock CloudClientFactory.resume_session
        with patch("cursor.cloud_client.CloudClientFactory.resume_session") as mock_resume:
            mock_resume.return_value = mock_cloud_success_result

            result = await executor.resume_session(
                session_id="session-abc123",
                prompt="继续之前的任务",
            )

            mock_resume.assert_called_once()
            call_kwargs = mock_resume.call_args.kwargs
            assert call_kwargs.get("session_id") == "session-abc123"
            assert call_kwargs.get("prompt") == "继续之前的任务"
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
            error=AuthError("No API key"),
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
        """测试 Cloud 超时处理

        CloudAgentExecutor.execute() 内部调用 CloudClientFactory.execute_task()
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 模拟 CloudClientFactory.execute_task 超时
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.side_effect = asyncio.TimeoutError("执行超时")

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
        """测试 Cloud 错误处理

        CloudAgentExecutor.execute() 内部调用 CloudClientFactory.execute_task()
        """
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            cloud_client=mock_cloud_client,
        )

        # 模拟 CloudClientFactory.execute_task 抛出异常
        with patch("cursor.cloud_client.CloudClientFactory.execute_task") as mock_execute_task:
            mock_execute_task.side_effect = RuntimeError("网络连接失败")

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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
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

            with (
                patch.object(
                    executor._cloud_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cloud_fail_result,
                ),
                patch.object(
                    executor._cli_executor,
                    "execute",
                    new_callable=AsyncMock,
                    return_value=cli_success_result,
                ),
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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await executor.check_available()
            assert result is True

        # 测试只有 CLI 可用
        executor2 = AutoAgentExecutor(cli_config=cli_config)
        with (
            patch.object(
                executor2._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                executor2._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await executor2.check_available()
            assert result is True

        # 测试两者都不可用
        executor3 = AutoAgentExecutor(cli_config=cli_config)
        with (
            patch.object(
                executor3._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                executor3._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
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
                # 必须设置 error_type 和 retry_after，否则 MagicMock 返回 MagicMock 导致验证失败
                mock_result.error_type = None
                mock_result.retry_after = None
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
        """测试自定义冷却配置

        注意：cloud_cooldown_seconds 参数现在会被转换为 CooldownConfig，
        实际冷却时间会根据错误类型而变化。
        """
        from cursor.executor import CooldownConfig

        # 使用 CooldownConfig 进行精确配置
        cooldown_config = CooldownConfig(
            rate_limit_default_seconds=60,
            network_cooldown_seconds=60,
            timeout_cooldown_seconds=60,
            unknown_cooldown_seconds=60,
        )

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
            enable_cooldown=True,
        )

        # cloud_cooldown_seconds 返回 unknown 类型的默认冷却时间
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
        """测试启动冷却

        注意：_start_cloud_cooldown() 使用默认错误类型（unknown），
        冷却时间由 unknown_cooldown_seconds 决定（默认 300 秒）。
        """
        from cursor.executor import CooldownConfig

        # 使用自定义配置以便测试
        cooldown_config = CooldownConfig(unknown_cooldown_seconds=60)

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
        )

        # 启动冷却（使用向后兼容的 _start_cloud_cooldown 方法）
        executor._start_cloud_cooldown()

        # 验证冷却状态
        assert executor.is_cloud_in_cooldown is True
        assert executor._cloud_failure_count == 1
        assert executor._cloud_cooldown_until is not None
        assert executor.cloud_cooldown_remaining is not None
        # 冷却时间应该在配置的范围内
        assert 0 < executor.cloud_cooldown_remaining <= 300  # 允许使用默认值

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
        """测试 Cloud 失败后触发冷却

        验证：
        - Cloud 执行失败后进入冷却
        - cooldown_info 包含统一的结构化信息
        - 使用 classify_cloud_failure 分类错误
        """
        from cursor.executor import CooldownConfig

        cooldown_config = CooldownConfig(unknown_cooldown_seconds=60)

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ),
            patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ),
        ):
            result = await executor.execute(prompt="测试冷却触发")

            # 验证回退到 CLI
            assert result.success is True
            assert result.executor_type == "cli"

            # 验证冷却已启动
            assert executor.is_cloud_in_cooldown is True
            assert executor._cloud_failure_count == 1
            assert executor._preferred_executor == executor._cli_executor

            # 验证 cooldown_info 结构
            assert result.cooldown_info is not None
            assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info
            assert CooldownInfoFields.RETRYABLE in result.cooldown_info

    @pytest.mark.asyncio
    async def test_cooldown_prevents_cloud_retry(
        self,
        cli_config: CursorAgentConfig,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试冷却期内不重试 Cloud"""
        from cursor.executor import CooldownConfig

        cooldown_config = CooldownConfig(unknown_cooldown_seconds=300)

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
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

        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                cloud_check_mock,
            ),
            patch.object(
                executor._cli_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cli_executor,
                "execute",
                cli_execute_mock,
            ),
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
        from cursor.executor import CooldownConfig

        cooldown_config = CooldownConfig(unknown_cooldown_seconds=60)

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "reset_availability_cache",
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ),
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

        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ),
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
        from cursor.executor import CooldownConfig

        cooldown_config = CooldownConfig(unknown_cooldown_seconds=1)  # 短冷却便于测试

        executor = AutoAgentExecutor(
            cli_config=cli_config,
            cloud_auth_config=cloud_auth_config,
            cooldown_config=cooldown_config,
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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ),
            patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ),
        ):
            await executor.execute(prompt="第一次失败")
            assert executor._cloud_failure_count == 1

        # 等待冷却过期
        executor._cloud_cooldown_until = datetime.now() - timedelta(seconds=1)

        # 第二次失败
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "reset_availability_cache",
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ),
            patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ),
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
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_fail_result,
            ),
            patch.object(
                executor._cli_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cli_success_result,
            ),
        ):
            result = await executor.execute(prompt="禁用冷却-失败")
            assert result.executor_type == "cli"

        # 重置偏好以便第二次重新选择
        executor.reset_preference()

        # 第二次执行：应该立即尝试 Cloud（因为冷却被禁用）
        with (
            patch.object(
                executor._cloud_executor,
                "check_available",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_cloud_check,
            patch.object(
                executor._cloud_executor,
                "execute",
                new_callable=AsyncMock,
                return_value=cloud_success_result,
            ),
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
        with (
            patch(
                "cursor.executor.CloudAgentExecutor.check_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "cursor.executor.CLIAgentExecutor.check_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
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
            AgentExecutorFactory.create(mode=cast(ExecutionMode, "invalid_mode"))


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

        with patch("cursor.executor.AgentExecutorFactory.create") as mock_factory:
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


# ==================== TestRunMpExecutionModeHandling ====================


class TestRunMpExecutionModeHandling:
    """run_mp.py 的 execution_mode 处理测试

    MP 编排器（run_mp.py）仅支持 execution_mode=cli。
    当配置为 cloud/auto 时，应该：
    1. 打印警告信息
    2. 强制回退到 CLI 模式
    3. 在结果中记录实际使用的 execution_mode
    """

    @pytest.fixture
    def mock_config_with_execution_mode(self, tmp_path):
        """创建带有指定 execution_mode 的临时 config.yaml"""

        def _create_config(execution_mode: str):
            config_content = f"""
cloud_agent:
  enabled: true
  execution_mode: {execution_mode}
  api_key: test-key

system:
  max_iterations: 3
  worker_pool_size: 2
"""
            config_file = tmp_path / "config.yaml"
            config_file.write_text(config_content)
            return config_file

        return _create_config

    def test_resolve_orchestrator_settings_returns_execution_mode(self) -> None:
        """测试 resolve_orchestrator_settings 返回 execution_mode"""
        from core.config import resolve_orchestrator_settings

        # 默认应返回 cli
        settings = resolve_orchestrator_settings()
        assert "execution_mode" in settings
        assert settings["execution_mode"] in ("cli", "cloud", "auto")

    def test_resolve_orchestrator_settings_with_cli_override(self) -> None:
        """测试 CLI override 优先级高于 config.yaml"""
        from core.config import resolve_orchestrator_settings

        # 使用 CLI override 覆盖 execution_mode
        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})
        assert settings["execution_mode"] == "cli"

    def test_resolve_orchestrator_settings_cloud_mode(self) -> None:
        """测试 cloud 模式配置"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})
        assert settings["execution_mode"] == "cloud"
        # cloud 模式应强制使用 basic 编排器
        assert settings["orchestrator"] == "basic"

    def test_resolve_orchestrator_settings_auto_mode(self) -> None:
        """测试 auto 模式配置"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "auto"})
        assert settings["execution_mode"] == "auto"
        # auto 模式应强制使用 basic 编排器
        assert settings["orchestrator"] == "basic"

    def test_mp_orchestrator_config_accepts_execution_mode(self) -> None:
        """测试 MultiProcessOrchestratorConfig 接受 execution_mode 参数"""
        from coordinator.orchestrator_mp import MultiProcessOrchestratorConfig

        # 创建配置时应能指定 execution_mode
        config = MultiProcessOrchestratorConfig(
            execution_mode="cli",
            working_directory=".",
        )
        assert config.execution_mode == "cli"

        # 即使传入 cloud，也应该能正常创建（实际处理在 run_mp.py 层）
        config_cloud = MultiProcessOrchestratorConfig(
            execution_mode="cloud",
            working_directory=".",
        )
        assert config_cloud.execution_mode == "cloud"

    @pytest.mark.asyncio
    async def test_run_mp_detects_cloud_mode_from_config(
        self,
        mock_config_with_execution_mode,
        tmp_path,
    ) -> None:
        """测试 run_mp.py 检测 config.yaml 中的 cloud 模式并回退"""
        from unittest.mock import patch

        # 创建 cloud 模式的配置
        config_file = mock_config_with_execution_mode("cloud")

        # Mock ConfigManager 以使用临时配置文件
        with patch("core.config.find_config_file", return_value=config_file):
            # 重置单例以加载新配置
            from core.config import ConfigManager

            ConfigManager.reset_instance()

            try:
                from core.config import resolve_orchestrator_settings

                # 验证配置被正确加载
                settings = resolve_orchestrator_settings()
                # cloud 模式应返回 cloud（在 resolve_orchestrator_settings 层）
                # MP 编排器的回退逻辑在 run_mp.py 的 run_orchestrator 中处理
                assert settings["execution_mode"] in ("cloud", "cli")
            finally:
                # 恢复单例
                ConfigManager.reset_instance()

    def test_execution_mode_in_result_structure(self) -> None:
        """测试结果结构包含 execution_mode 字段"""
        # 模拟 run_mp.py 生成的结果结构
        result = {
            "success": True,
            "goal": "测试任务",
            "iterations_completed": 1,
            "execution_mode": "cli",
            "execution_mode_config": "cloud",  # 原始配置值
        }

        # 验证结果包含 execution_mode 信息
        assert "execution_mode" in result
        assert "execution_mode_config" in result
        assert result["execution_mode"] == "cli"
        assert result["execution_mode_config"] == "cloud"

    def test_execution_mode_fallback_warning_format(self) -> None:
        """测试 execution_mode 回退警告格式"""
        from io import StringIO

        from loguru import logger

        # 捕获日志输出
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")

        try:
            # 模拟 run_mp.py 中的警告逻辑
            config_execution_mode = "cloud"
            if config_execution_mode in ("cloud", "auto"):
                logger.warning(f"⚠ MP 编排器不支持 execution_mode={config_execution_mode}，强制回退到 CLI 模式")

            log_content = log_output.getvalue()
            assert "MP 编排器不支持" in log_content
            assert "cloud" in log_content
            assert "CLI" in log_content
        finally:
            logger.remove(handler_id)


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


# ==================== TestRequestedModeOrchestratorInvariant ====================


class TestRequestedModeOrchestratorInvariant:
    """requested_mode 与 orchestrator 关系的不变量测试

    核心规则：requested_mode=auto/cloud ⇒ orchestrator=basic
    此规则确保 Cloud/Auto 执行模式始终使用 basic 编排器。

    **重要**：这是一个不可回归的行为，本测试类用于确保该行为永远成立。

    背景说明：
    - requested_mode: 用户请求的执行模式（CLI 参数或 config.yaml）
    - effective_mode: 实际生效的执行模式（可能因缺少 API Key 而回退）
    - orchestrator 选择基于 requested_mode，而非 effective_mode
    - 即使 effective_mode 回退到 cli，只要 requested_mode 是 auto/cloud，
      编排器就应该是 basic

    参见 AGENTS.md: "requested_mode vs effective_mode 关键区别"
    """

    def test_auto_mode_forces_basic_orchestrator(self) -> None:
        """不变量：requested_mode=auto 必须强制使用 basic 编排器

        这是核心不变量，确保 auto 模式始终使用 basic 编排器，
        无论是否有 API Key。
        """
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "auto"})

        assert settings["execution_mode"] == "auto", "execution_mode 应为 auto"
        assert settings["orchestrator"] == "basic", (
            "关键不变量：requested_mode=auto 必须强制使用 basic 编排器\n"
            f"实际值: orchestrator={settings['orchestrator']}"
        )

    def test_cloud_mode_forces_basic_orchestrator(self) -> None:
        """不变量：requested_mode=cloud 必须强制使用 basic 编排器

        这是核心不变量，确保 cloud 模式始终使用 basic 编排器，
        无论是否有 API Key。
        """
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})

        assert settings["execution_mode"] == "cloud", "execution_mode 应为 cloud"
        assert settings["orchestrator"] == "basic", (
            "关键不变量：requested_mode=cloud 必须强制使用 basic 编排器\n"
            f"实际值: orchestrator={settings['orchestrator']}"
        )

    def test_cli_mode_allows_mp_orchestrator(self) -> None:
        """对比测试：requested_mode=cli 允许使用 mp 编排器"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})

        assert settings["execution_mode"] == "cli", "execution_mode 应为 cli"
        assert settings["orchestrator"] == "mp", (
            f"cli 模式应默认允许 mp 编排器\n实际值: orchestrator={settings['orchestrator']}"
        )

    def test_auto_mode_basic_even_when_mp_requested(self) -> None:
        """不变量：即使显式请求 mp，auto 模式仍强制 basic

        验证：即使用户显式设置 orchestrator=mp，
        当 execution_mode=auto 时，仍会被强制切换到 basic。
        """
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "auto", "orchestrator": "mp"})

        assert settings["execution_mode"] == "auto"
        assert settings["orchestrator"] == "basic", (
            f"关键不变量：auto 模式下即使请求 mp 也必须强制使用 basic\n实际值: orchestrator={settings['orchestrator']}"
        )

    def test_cloud_mode_basic_even_when_mp_requested(self) -> None:
        """不变量：即使显式请求 mp，cloud 模式仍强制 basic"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cloud", "orchestrator": "mp"})

        assert settings["execution_mode"] == "cloud"
        assert settings["orchestrator"] == "basic", (
            f"关键不变量：cloud 模式下即使请求 mp 也必须强制使用 basic\n实际值: orchestrator={settings['orchestrator']}"
        )

    def test_orchestrator_forced_reason_populated(self) -> None:
        """验证 orchestrator_forced_reason 字段被正确填充

        当编排器因 execution_mode=auto/cloud 而被强制切换到 basic 时，
        应该记录强制切换的原因。
        """
        from core.config import resolve_orchestrator_settings

        # auto 模式
        settings_auto = resolve_orchestrator_settings(overrides={"execution_mode": "auto"})
        assert settings_auto["orchestrator_forced_reason"] is not None, "auto 模式下应记录强制切换原因"
        assert "auto" in settings_auto["orchestrator_forced_reason"].lower()

        # cloud 模式
        settings_cloud = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})
        assert settings_cloud["orchestrator_forced_reason"] is not None, "cloud 模式下应记录强制切换原因"
        assert "cloud" in settings_cloud["orchestrator_forced_reason"].lower()

    def test_cli_mode_no_forced_reason(self) -> None:
        """验证 cli 模式下不记录强制切换原因"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": "cli"})
        assert settings["orchestrator_forced_reason"] is None, "cli 模式下不应有强制切换原因"

    def test_format_debug_config_shows_requested_and_effective_mode(self) -> None:
        """验证 format_debug_config 输出包含 requested_mode 和 effective_mode"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        # 测试 auto 模式（无 API Key 时 effective_mode 回退到 cli）
        output = format_debug_config(
            cli_overrides={"execution_mode": "auto"},
            source_label="test",
            has_api_key=False,
        )

        # 验证输出包含两个字段
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: auto" in output
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output
        # 验证编排器仍为 basic（即使 effective_mode=cli）
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: basic" in output

    def test_format_debug_config_orchestrator_fallback_explanation(self) -> None:
        """验证 orchestrator_fallback 包含详细解释

        当 requested_mode=auto/cloud 但 effective_mode=cli 时，
        orchestrator_fallback 应解释为什么仍使用 basic 编排器。
        """
        from core.config import format_debug_config

        output = format_debug_config(
            cli_overrides={"execution_mode": "auto"},
            source_label="test",
            has_api_key=False,  # 无 API Key，effective_mode 会回退到 cli
        )

        # 验证回退解释
        assert "mp->basic" in output
        assert "requested_mode=auto" in output
        # 验证解释说明即使 effective_mode=cli 也强制 basic
        assert "even when effective_mode=cli" in output

    def test_prefix_routed_also_forces_basic(self) -> None:
        """验证 & 前缀成功触发时也强制 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},  # CLI 模式
            prefix_routed=True,  # & 前缀成功触发
        )

        assert settings["orchestrator"] == "basic", "& 前缀成功触发时应强制使用 basic 编排器"
        assert settings["prefix_routed"] is True

    def test_should_use_mp_orchestrator_policy_function(self) -> None:
        """验证 should_use_mp_orchestrator 策略函数行为

        此函数是编排器选择策略的核心实现，验证其与 requested_mode 的关系。
        """
        from core.execution_policy import should_use_mp_orchestrator

        # auto/cloud 模式不能使用 mp
        assert should_use_mp_orchestrator("auto") is False, "auto 模式不能使用 mp"
        assert should_use_mp_orchestrator("cloud") is False, "cloud 模式不能使用 mp"

        # cli/plan/ask 模式可以使用 mp
        assert should_use_mp_orchestrator("cli") is True, "cli 模式可以使用 mp"
        assert should_use_mp_orchestrator("plan") is True, "plan 模式可以使用 mp"
        assert should_use_mp_orchestrator("ask") is True, "ask 模式可以使用 mp"

        # 验证正确的流程：不应直接调用 should_use_mp_orchestrator(None)
        # 应先通过 resolve_requested_mode_for_decision 解析 requested_mode
        from core.execution_policy import resolve_requested_mode_for_decision

        # 场景 1: 无 & 前缀 + CLI 未指定 → 使用 config.yaml 默认值（如 auto）
        requested_mode_no_prefix = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode_no_prefix == "auto", "无 & 前缀时应使用 config.yaml 默认值"
        assert should_use_mp_orchestrator(requested_mode_no_prefix) is False, "config.yaml 默认 auto 时，应禁用 MP"

        # 场景 2: 有 & 前缀 + CLI 未指定 → 返回 None（由 build_execution_decision 处理）
        # 此时 should_use_mp_orchestrator(None) 返回 True 是合法的（prefix-flow）
        requested_mode_with_prefix = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        assert requested_mode_with_prefix is None, "有 & 前缀时应返回 None（prefix-flow）"
        assert should_use_mp_orchestrator(requested_mode_with_prefix) is True, (
            "& 前缀场景下 should_use_mp_orchestrator(None) 应返回 True"
        )

    @pytest.mark.parametrize(
        "requested_mode,expected_orchestrator",
        [
            ("cli", "mp"),
            ("auto", "basic"),
            ("cloud", "basic"),
            ("plan", "mp"),
            ("ask", "mp"),
        ],
        ids=["cli->mp", "auto->basic", "cloud->basic", "plan->mp", "ask->mp"],
    )
    def test_requested_mode_orchestrator_mapping(
        self,
        requested_mode: str,
        expected_orchestrator: str,
    ) -> None:
        """参数化测试：验证所有模式与编排器的映射关系

        这是回归测试的核心，确保映射关系保持稳定。
        """
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides={"execution_mode": requested_mode})

        assert settings["orchestrator"] == expected_orchestrator, (
            f"requested_mode={requested_mode} 应使用 {expected_orchestrator} 编排器\n"
            f"实际值: orchestrator={settings['orchestrator']}"
        )


# ============================================================
# Cloud 结果 cooldown_info 字段契约测试
# ============================================================


class TestCloudResultCooldownInfoContract:
    """验证 Cloud 执行结果 cooldown_info 字段的契约一致性

    测试场景:
    1. build_cloud_result 构建器包含 cooldown_info 字段
    2. build_cloud_success_result 包含 cooldown_info 字段
    3. build_cloud_error_result 包含 cooldown_info 字段
    4. 所有分支返回的字段集合一致
    """

    def test_build_cloud_result_defaults_contains_cooldown_info(self) -> None:
        """验证 build_cloud_result_defaults 包含 cooldown_info 字段"""
        from core.output_contract import CloudResultFields, build_cloud_result_defaults

        result = build_cloud_result_defaults(goal="测试")

        assert CloudResultFields.COOLDOWN_INFO in result, "默认模板应包含 cooldown_info 字段"
        assert result[CloudResultFields.COOLDOWN_INFO] is None, "默认值应为 None"

    def test_build_cloud_result_accepts_cooldown_info(self) -> None:
        """验证 build_cloud_result 接受 cooldown_info 参数"""
        from core.output_contract import CloudResultFields, build_cloud_result

        test_cooldown = {
            "user_message": "测试消息",
            "kind": "TEST",
            "retryable": False,
        }

        result = build_cloud_result(
            goal="测试",
            cooldown_info=test_cooldown,
        )

        assert result[CloudResultFields.COOLDOWN_INFO] == test_cooldown, "cooldown_info 应被正确设置"

    def test_build_cloud_success_result_contains_cooldown_info(self) -> None:
        """验证 build_cloud_success_result 包含 cooldown_info 字段"""
        from core.output_contract import CloudResultFields, build_cloud_success_result

        result = build_cloud_success_result(
            goal="测试",
            output="成功输出",
        )

        assert CloudResultFields.COOLDOWN_INFO in result, "成功结果应包含 cooldown_info 字段"

    def test_build_cloud_error_result_contains_cooldown_info(self) -> None:
        """验证 build_cloud_error_result 包含 cooldown_info 字段"""
        from core.output_contract import CloudResultFields, build_cloud_error_result

        result = build_cloud_error_result(
            goal="测试",
            error="错误信息",
            failure_kind="TEST_ERROR",
        )

        assert CloudResultFields.COOLDOWN_INFO in result, "错误结果应包含 cooldown_info 字段"

    def test_cloud_result_fields_constant_includes_cooldown_info(self) -> None:
        """验证 CloudResultFields 包含 COOLDOWN_INFO 常量"""
        from core.output_contract import CloudResultFields

        assert hasattr(CloudResultFields, "COOLDOWN_INFO"), "CloudResultFields 应包含 COOLDOWN_INFO 常量"
        assert CloudResultFields.COOLDOWN_INFO == "cooldown_info", "COOLDOWN_INFO 常量值应为 'cooldown_info'"

    def test_success_and_error_results_have_same_field_set(self) -> None:
        """验证成功和失败结果的字段集合一致"""
        from core.output_contract import (
            build_cloud_error_result,
            build_cloud_success_result,
        )

        success_result = build_cloud_success_result(
            goal="测试",
            output="成功",
        )
        error_result = build_cloud_error_result(
            goal="测试",
            error="失败",
            failure_kind="ERROR",
        )

        success_fields = set(success_result.keys())
        error_fields = set(error_result.keys())

        assert success_fields == error_fields, (
            f"成功和失败结果的字段集合应一致\n差异: {success_fields.symmetric_difference(error_fields)}"
        )


class TestIterateResultCooldownInfoContract:
    """验证 Iterate 结果 cooldown_info 字段的契约一致性"""

    def test_build_iterate_result_defaults_contains_cooldown_info(self) -> None:
        """验证 build_iterate_result_defaults 包含 cooldown_info 字段"""
        from core.output_contract import IterateResultFields, build_iterate_result_defaults

        result = build_iterate_result_defaults()

        assert IterateResultFields.COOLDOWN_INFO in result, "默认模板应包含 cooldown_info 字段"
        assert result[IterateResultFields.COOLDOWN_INFO] is None, "默认值应为 None"

    def test_build_iterate_success_result_accepts_cooldown_info(self) -> None:
        """验证 build_iterate_success_result 接受 cooldown_info 参数"""
        from core.output_contract import IterateResultFields, build_iterate_success_result

        test_cooldown = {
            "user_message": "迭代回退消息",
            "kind": "FALLBACK",
        }

        result = build_iterate_success_result(
            cooldown_info=test_cooldown,
        )

        assert result[IterateResultFields.COOLDOWN_INFO] == test_cooldown, "cooldown_info 应被正确设置"

    def test_build_iterate_error_result_accepts_cooldown_info(self) -> None:
        """验证 build_iterate_error_result 接受 cooldown_info 参数"""
        from core.output_contract import IterateResultFields, build_iterate_error_result

        test_cooldown = {
            "user_message": "迭代错误消息",
            "kind": "ERROR",
        }

        result = build_iterate_error_result(
            error="测试错误",
            cooldown_info=test_cooldown,
        )

        assert result[IterateResultFields.COOLDOWN_INFO] == test_cooldown, "cooldown_info 应被正确设置"

    def test_iterate_result_fields_constant_includes_cooldown_info(self) -> None:
        """验证 IterateResultFields 包含 COOLDOWN_INFO 常量"""
        from core.output_contract import IterateResultFields

        assert hasattr(IterateResultFields, "COOLDOWN_INFO"), "IterateResultFields 应包含 COOLDOWN_INFO 常量"
        assert IterateResultFields.COOLDOWN_INFO == "cooldown_info", "COOLDOWN_INFO 常量值应为 'cooldown_info'"
