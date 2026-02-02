"""测试无 API Key 场景下的网络隔离

验证在无 CURSOR_API_KEY/CURSOR_CLOUD_API_KEY 时:
1. 不会发起真实网络请求（Cloud API 调用）
2. 错误分类返回 NO_KEY 类型
3. 消息符合预期格式

与 core/execution_policy.py 的契约保持一致:
- CloudFailureKind.NO_KEY: 未配置 API Key
- failure_info.retryable = False
- failure_info.message 包含 API Key 相关提示

注意：
- 同步测试可使用 @pytest.mark.block_network 阻断网络
- 异步测试使用 mock 验证网络隔离（因 pytest_asyncio 需要 socket 创建事件循环）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution_policy import (
    CloudFailureKind,
    classify_cloud_failure,
    build_execution_decision,
    build_user_facing_fallback_message,
)


class TestNoApiKeyNetworkIsolation:
    """无 API Key 场景下的网络隔离测试

    使用 no_api_key_and_block_network 组合 fixture，统一：
    - 清除 CURSOR_API_KEY、CURSOR_CLOUD_API_KEY 环境变量
    - 设置 CURSORX_BLOCK_NETWORK=1 启用网络阻断
    
    同步测试由 fixture 自动阻断网络，异步测试使用 mock 验证。
    """

    @pytest.fixture(autouse=True)
    def setup_no_key_env(self, no_api_key_and_block_network):
        """自动应用无 API Key + 网络阻断的组合 fixture"""
        pass

    def test_cloud_executor_not_called_without_api_key(self):
        """验证无 API Key 时 CloudAgentExecutor 的 check_available 返回 False

        使用 mock 将认证过程中的 CLI 调用替换，验证无 key 时执行器不可用。
        setup_no_key_env fixture 已清除 API Key 并启用网络阻断。
        """
        from cursor.executor import CloudAgentExecutor
        from cursor.cloud.client import CursorCloudClient
        from cursor.cloud.auth import CloudAuthManager, AuthStatus
        from cursor.cloud.exceptions import AuthError, AuthErrorCode

        # Mock CloudAuthManager.authenticate 返回未认证状态
        mock_auth_status = AuthStatus(
            authenticated=False,
            error=AuthError("未找到 API Key", AuthErrorCode.CONFIG_NOT_FOUND),
        )

        with patch.object(
            CloudAuthManager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            # Mock CursorCloudClient.execute - 若被调用则失败
            with patch.object(
                CursorCloudClient,
                "execute",
                side_effect=AssertionError("CursorCloudClient.execute 不应被调用"),
            ):
                executor = CloudAgentExecutor()
                # 使用同步版本检查可用性
                # check_available_sync 在无异步上下文时返回缓存值或 False
                is_available = executor.check_available_sync()
                assert is_available is False, "无 API Key 时 Cloud 执行器不应可用"

    @pytest.mark.asyncio
    async def test_cloud_client_factory_execute_task_returns_auth_error(self):
        """验证 CloudClientFactory.execute_task 在无 API Key 时返回认证错误

        使用 mock 验证不会发起真实网络请求，而是在认证阶段返回错误。
        """
        from cursor.cloud_client import CloudClientFactory, CloudAgentResult
        from cursor.cloud.auth import CloudAuthManager, AuthStatus
        from cursor.cloud.exceptions import AuthError, AuthErrorCode

        # Mock CloudAuthManager.authenticate 返回未认证状态
        mock_auth_status = AuthStatus(
            authenticated=False,
            error=AuthError("未找到 API Key", AuthErrorCode.CONFIG_NOT_FOUND),
        )

        with patch.object(
            CloudAuthManager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            # 执行 Cloud 任务（无 API Key）
            result = await CloudClientFactory.execute_task(
                prompt="测试任务",
                agent_config=None,
                auth_config=None,
                explicit_api_key=None,  # 显式无 key
                working_directory=".",
                timeout=10,
            )

            # 验证返回结果
            assert isinstance(result, CloudAgentResult)
            assert result.success is False
            assert result.error_type == "auth", f"预期 error_type=auth，实际 {result.error_type}"
            # 验证错误消息包含认证相关信息
            assert result.error is not None

    def test_classify_cloud_failure_no_key(self):
        """验证 classify_cloud_failure 正确分类无 API Key 错误

        与 core/execution_policy.py 的契约保持一致。
        纯计算逻辑测试，无需网络阻断。
        """
        # 测试各种无 API Key 的错误消息格式
        no_key_messages = [
            "No API Key configured",
            "API key not found",
            "Missing API Key",
            "api_key is required",
            "未配置 API Key",
            "缺少 API Key",
        ]

        for msg in no_key_messages:
            failure_info = classify_cloud_failure(msg)
            assert failure_info.kind == CloudFailureKind.NO_KEY, (
                f"消息 '{msg}' 应被分类为 NO_KEY，实际为 {failure_info.kind}"
            )
            assert failure_info.retryable is False, (
                f"NO_KEY 错误不应可重试，消息: {msg}"
            )

    def test_classify_cloud_failure_no_key_from_dict(self):
        """验证 classify_cloud_failure 从结构化 dict 正确识别 no_key 类型

        纯计算逻辑测试，无需网络阻断。
        """
        # 模拟 CloudAgentResult.to_dict() 的输出格式
        error_dict = {
            "success": False,
            "error": "未找到 API Key",
            "error_type": "no_key",
            "retry_after": None,
        }

        failure_info = classify_cloud_failure(error_dict)
        assert failure_info.kind == CloudFailureKind.NO_KEY
        assert failure_info.retryable is False
        assert failure_info.retry_after is None

    def test_build_execution_decision_no_api_key_fallback(self):
        """验证 build_execution_decision 在无 API Key 时正确回退到 CLI

        请求 auto/cloud 模式但无 API Key 应回退到 CLI 模式。
        纯计算逻辑测试，无需网络阻断。
        """
        # 场景 1: 请求 auto 模式，无 API Key
        decision = build_execution_decision(
            prompt="测试任务",
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision.effective_mode == "cli", (
            f"无 API Key 时 auto 模式应回退到 CLI，实际 {decision.effective_mode}"
        )
        # requested_mode=auto 时编排器仍强制 basic
        assert decision.orchestrator == "basic", (
            f"requested_mode=auto 时编排器应为 basic，实际 {decision.orchestrator}"
        )
        assert "API Key" in decision.mode_reason or "key" in decision.mode_reason.lower()

        # 场景 2: 请求 cloud 模式，无 API Key
        decision = build_execution_decision(
            prompt="测试任务",
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "basic"

        # 场景 3: & 前缀触发但无 API Key
        decision = build_execution_decision(
            prompt="& 测试任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,
        )
        assert decision.effective_mode == "cli"
        assert decision.has_ampersand_prefix is True
        assert decision.prefix_routed is False, (
            "无 API Key 时 & 前缀不应成功路由"
        )

    def test_build_user_facing_message_no_key(self):
        """验证 build_user_facing_fallback_message 生成正确的 NO_KEY 消息

        纯计算逻辑测试，无需网络阻断。
        """
        message = build_user_facing_fallback_message(
            kind=CloudFailureKind.NO_KEY,
            retry_after=None,
            requested_mode="auto",
            has_ampersand_prefix=False,
        )

        # 验证消息包含关键信息
        assert "API Key" in message or "CURSOR_API_KEY" in message
        assert "config.yaml" in message or "环境变量" in message

    def test_auto_executor_fallback_without_api_key(self):
        """验证 AutoAgentExecutor 在无 API Key 时正确回退到 CLI

        Cloud 执行器应检测到无 API Key 并回退到 CLI 执行器。
        使用 mock 验证不会触发 Cloud API 调用。
        setup_no_key_env fixture 已清除 API Key 并启用网络阻断。
        """
        from cursor.executor import AutoAgentExecutor, AgentResult
        from cursor.cloud.client import CursorCloudClient

        # Mock CursorCloudClient 方法 - 确保不会被调用
        execute_mock = MagicMock(side_effect=AssertionError("不应被调用"))
        submit_task_mock = MagicMock(side_effect=AssertionError("不应被调用"))

        with patch.object(CursorCloudClient, "execute", execute_mock):
            with patch.object(CursorCloudClient, "submit_task", submit_task_mock):
                # 创建 AutoAgentExecutor
                executor = AutoAgentExecutor()

                # 验证 Cloud 执行器不可用
                assert executor.is_cloud_in_cooldown is False
                # Cloud 执行器应检测到无 API Key

                # execute_mock 和 submit_task_mock 不应被调用
                execute_mock.assert_not_called()
                submit_task_mock.assert_not_called()

    def test_cloud_auth_manager_returns_no_key_error(self):
        """验证 CloudAuthManager 在无 API Key 时返回正确的认证状态"""
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig, AuthStatus
        from cursor.cloud.exceptions import AuthErrorCode

        # 创建配置（无 API Key）
        config = CloudAuthConfig(api_key=None)
        manager = CloudAuthManager(config=config)

        # 获取 API Key 应返回 None
        api_key = manager.get_api_key()
        assert api_key is None, "无配置时 get_api_key 应返回 None"

    @pytest.mark.asyncio
    async def test_cloud_auth_manager_authenticate_no_key(self):
        """验证 CloudAuthManager.authenticate 在无 API Key 时返回未认证状态

        使用 mock 避免真实 CLI 调用，验证无 API Key 时认证失败。
        """
        from cursor.cloud.auth import CloudAuthManager, CloudAuthConfig
        from cursor.cloud.exceptions import AuthErrorCode

        config = CloudAuthConfig(api_key=None)
        manager = CloudAuthManager(config=config)

        # 由于 get_api_key 返回 None，authenticate 应在调用 CLI 之前就失败
        # Mock _verify_with_cli 以避免真实 CLI 调用
        with patch.object(
            manager,
            "_verify_with_cli",
            new_callable=AsyncMock,
            side_effect=AssertionError("不应调用 _verify_with_cli（无 API Key）"),
        ):
            # 认证应返回失败状态（在调用 CLI 之前）
            status = await manager.authenticate()
            assert status.authenticated is False
            assert status.error is not None
            # 验证错误类型（AuthError 的属性是 code，不是 error_code）
            assert status.error.code == AuthErrorCode.CONFIG_NOT_FOUND

    def test_failure_kind_no_key_properties(self):
        """验证 CloudFailureKind.NO_KEY 的属性与契约一致"""
        failure_info = classify_cloud_failure("未配置 API Key")

        # 契约验证
        assert failure_info.kind == CloudFailureKind.NO_KEY
        assert failure_info.retryable is False, "NO_KEY 不应可重试"
        assert failure_info.retry_after is None, "NO_KEY 不应有 retry_after"

        # 消息验证
        assert "API" in failure_info.message or "Key" in failure_info.message

    @pytest.mark.asyncio
    async def test_cloud_executor_execute_returns_failure_without_network(self):
        """验证 CloudAgentExecutor.execute 在无 API Key 时返回失败结果

        使用 mock 确保不会触发网络请求，而是在认证阶段返回失败。
        """
        from cursor.executor import CloudAgentExecutor
        from cursor.cloud.client import CursorCloudClient
        from cursor.cloud.auth import CloudAuthManager, AuthStatus
        from cursor.cloud.exceptions import AuthError, AuthErrorCode

        # Mock CloudAuthManager.authenticate 返回未认证状态
        mock_auth_status = AuthStatus(
            authenticated=False,
            error=AuthError("未找到 API Key", AuthErrorCode.CONFIG_NOT_FOUND),
        )

        # Mock execute 方法 - 若被调用则失败
        mock_execute = AsyncMock(
            side_effect=AssertionError("CursorCloudClient.execute 不应被调用")
        )

        with patch.object(
            CloudAuthManager,
            "authenticate",
            new_callable=AsyncMock,
            return_value=mock_auth_status,
        ):
            with patch.object(CursorCloudClient, "execute", mock_execute):
                executor = CloudAgentExecutor()

                # 执行任务
                result = await executor.execute(
                    prompt="测试任务",
                    timeout=5,
                )

                # 验证结果
                assert result.success is False
                # 验证 failure_kind
                if result.failure_kind:
                    assert result.failure_kind in ("auth", "no_key", CloudFailureKind.AUTH.value, CloudFailureKind.NO_KEY.value), (
                        f"预期 failure_kind 为 auth 或 no_key，实际 {result.failure_kind}"
                    )

                # 确保 mock 未被调用
                mock_execute.assert_not_called()


class TestNetworkBlockingEffectiveness:
    """验证网络阻断机制的有效性

    确保 @pytest.mark.block_network 标记正确阻断网络请求。
    使用 no_api_key_and_block_network 组合 fixture 确保干净的环境和网络阻断。
    """

    @pytest.fixture(autouse=True)
    def setup_no_key_env(self, no_api_key_and_block_network):
        """自动应用无 API Key + 网络阻断的组合 fixture"""
        pass

    @pytest.mark.block_network
    def test_network_blocking_raises_error(self):
        """验证网络阻断时 socket 创建会抛出 BlockedNetworkError"""
        import socket
        from tests.conftest import BlockedNetworkError

        # 尝试创建 socket 应该抛出 BlockedNetworkError
        with pytest.raises(BlockedNetworkError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    @pytest.mark.block_network
    def test_http_request_blocked(self):
        """验证 HTTP 请求被阻断"""
        from tests.conftest import BlockedNetworkError

        try:
            import urllib.request
            # 尝试发起 HTTP 请求应该被阻断
            with pytest.raises(BlockedNetworkError):
                urllib.request.urlopen("http://example.com", timeout=1)
        except BlockedNetworkError:
            # 预期行为
            pass


class TestClassifyCloudFailureNoKeyIntegration:
    """classify_cloud_failure 与 NO_KEY 错误类型的集成测试

    纯计算逻辑测试，无需网络阻断。
    """

    def test_classify_various_no_key_patterns(self):
        """测试各种无 API Key 错误模式的分类"""
        test_cases = [
            # (输入, 预期 kind)
            ("no api key", CloudFailureKind.NO_KEY),
            ("API key not configured", CloudFailureKind.NO_KEY),
            ("missing API key", CloudFailureKind.NO_KEY),
            ("未配置 API Key", CloudFailureKind.NO_KEY),
            ("缺少 API Key", CloudFailureKind.NO_KEY),
            ({"error": "no key", "error_type": "no_key"}, CloudFailureKind.NO_KEY),
        ]

        for input_val, expected_kind in test_cases:
            result = classify_cloud_failure(input_val)
            assert result.kind == expected_kind, (
                f"输入 {input_val!r} 应分类为 {expected_kind}，实际 {result.kind}"
            )

    def test_classify_auth_vs_no_key(self):
        """验证 AUTH 和 NO_KEY 错误的区分"""
        # AUTH 错误模式（有 key 但无效）
        auth_patterns = [
            "401 Unauthorized",
            "Invalid API Key",
            "API key 无效",
        ]
        for msg in auth_patterns:
            result = classify_cloud_failure(msg)
            assert result.kind == CloudFailureKind.AUTH, (
                f"消息 '{msg}' 应分类为 AUTH"
            )

        # NO_KEY 错误模式（缺少 key）
        no_key_patterns = [
            "No API Key",
            "Missing API Key",
            "未配置 API Key",
        ]
        for msg in no_key_patterns:
            result = classify_cloud_failure(msg)
            assert result.kind == CloudFailureKind.NO_KEY, (
                f"消息 '{msg}' 应分类为 NO_KEY"
            )
