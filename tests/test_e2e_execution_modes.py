"""执行模式端到端测试

测试 cursor/executor.py 中各种执行模式的功能，包括：
- CLI 模式执行工作流
- Cloud 模式执行工作流
- Auto 模式自动选择和回退
- ExecutorFactory 工厂创建

使用 Mock 替代真实 Cursor CLI/Cloud 调用
"""
import asyncio
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cursor.client import CursorAgentConfig, CursorAgentResult
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
    """Cloud 执行模式测试"""

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

    @pytest.mark.asyncio
    async def test_cloud_mode_workflow(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
    ) -> None:
        """测试 Cloud 模式完整工作流"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
        )

        # Mock 认证成功
        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            result = await executor.execute(
                prompt="分析代码",
                context={"files": ["app.py"]},
            )

            # 当前 Cloud API 未实现，应返回失败
            assert result.executor_type == "cloud"
            # Cloud API 尚未实现，预期返回失败
            assert result.success is False
            assert "Cloud API 尚未实现" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cloud_authentication(
        self,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 认证流程"""
        executor = CloudAgentExecutor(auth_config=cloud_auth_config)

        # 测试认证成功
        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            # 执行会触发认证
            result = await executor.execute(prompt="测试认证")
            assert result.executor_type == "cloud"

        # 测试认证失败
        executor2 = CloudAgentExecutor(auth_config=cloud_auth_config)
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
    async def test_cloud_task_submission(
        self,
        cloud_auth_config: CloudAuthConfig,
        agent_config: CursorAgentConfig,
    ) -> None:
        """测试 Cloud 任务提交"""
        executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=agent_config,
        )

        # Mock 认证和任务执行
        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            # 测试任务提交（当前会返回未实现）
            result = await executor.execute(
                prompt="执行云端任务",
                context={"task_id": "task-001"},
                timeout=120,
            )

            assert result.executor_type == "cloud"
            assert result.raw_result is not None
            assert result.raw_result.get("status") == "not_implemented"

    @pytest.mark.asyncio
    async def test_cloud_availability_check(
        self,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 可用性检查"""
        executor = CloudAgentExecutor(auth_config=cloud_auth_config)

        # Mock 认证失败（Cloud 不可用）
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
            # 当前实现 Cloud API 总是返回不可用
            assert result is False

    @pytest.mark.asyncio
    async def test_cloud_timeout_handling(
        self,
        cloud_auth_config: CloudAuthConfig,
    ) -> None:
        """测试 Cloud 超时处理"""
        executor = CloudAgentExecutor(auth_config=cloud_auth_config)

        mock_auth_status = AuthStatus(
            authenticated=True,
            user_id="user-123",
        )

        with patch.object(
            executor._auth_manager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            with patch.object(
                executor,
                "_execute_via_api",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError(),
            ):
                result = await executor.execute(
                    prompt="长时间任务",
                    timeout=1,
                )

                assert result.success is False
                assert "超时" in (result.error or "")


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
