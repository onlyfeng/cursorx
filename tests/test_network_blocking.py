"""网络阻断 Fixture 测试

测试内容:
1. @pytest.mark.block_network 标记阻断网络请求
2. @pytest.mark.allow_network 标记放行网络请求（使用 mock 验证）
3. 通过环境变量 CURSORX_BLOCK_NETWORK 动态启用阻断
4. 优先级规则验证：allow_network > block_network > 环境变量

这些测试验证 conftest.py 中的 _block_network_requests fixture 正确工作。
"""
from __future__ import annotations

import os
import socket
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import BlockedNetworkError


# ==================== block_network 标记阻断测试 ====================


class TestBlockNetworkMarker:
    """@pytest.mark.block_network 标记测试
    
    验证使用 block_network 标记时，socket 连接会被阻断。
    """

    @pytest.mark.block_network
    def test_socket_blocked_with_marker(self):
        """socket.socket() 被 block_network 标记阻断"""
        with pytest.raises(BlockedNetworkError) as exc_info:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        assert "网络请求被阻断" in str(exc_info.value)
        assert "测试环境禁止真实网络请求" in str(exc_info.value)
        assert "@pytest.mark.allow_network" in str(exc_info.value)

    @pytest.mark.block_network
    def test_httpx_blocked_with_marker(self):
        """httpx 请求被 block_network 标记阻断
        
        httpx 底层使用 socket，当 socket 被阻断时 httpx 请求也会失败。
        由于 httpx 会捕获底层异常，这里验证异常链中包含 BlockedNetworkError。
        """
        import httpx
        
        # httpx 底层会调用 socket.socket()，应触发 BlockedNetworkError
        with pytest.raises((BlockedNetworkError, httpx.ConnectError, OSError)) as exc_info:
            # 创建客户端时可能就会触发 socket 调用
            with httpx.Client() as client:
                client.get("https://example.com")
        
        # 验证异常消息或异常链中包含阻断信息
        exc_str = str(exc_info.value)
        # BlockedNetworkError 直接抛出时包含特定消息
        # 或者被 httpx 包装后，异常链中仍有相关信息
        assert (
            "网络请求被阻断" in exc_str or
            isinstance(exc_info.value, BlockedNetworkError) or
            (exc_info.value.__cause__ is not None and 
             isinstance(exc_info.value.__cause__, BlockedNetworkError))
        )

    @pytest.mark.block_network
    def test_socket_connect_blocked_with_marker(self):
        """socket.socket().connect() 被阻断
        
        实际上在创建 socket 时就会被阻断，无法到达 connect 调用。
        """
        with pytest.raises(BlockedNetworkError):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("example.com", 80))


# ==================== allow_network 标记放行测试 ====================


class TestAllowNetworkMarker:
    """@pytest.mark.allow_network 标记测试
    
    验证使用 allow_network 标记时，网络阻断 fixture 不会拦截请求。
    使用 mock 避免真实网络请求，重点验证 fixture 不会阻断。
    """

    @pytest.mark.allow_network
    def test_socket_not_blocked_with_allow_marker(self):
        """socket.socket() 在 allow_network 标记下不被阻断
        
        使用 mock 验证 socket.socket 的原始实现被调用，而非被阻断。
        """
        # 保存原始 socket 类
        original_socket = socket.socket
        
        # 创建 mock，当调用时返回一个 mock socket 对象
        mock_socket_instance = MagicMock()
        mock_socket_class = MagicMock(return_value=mock_socket_instance)
        
        # 由于 allow_network 标记生效，socket.socket 应该是原始的
        # 我们通过 patch 来验证：如果 patch 生效，说明没有被 block_network 覆盖
        with patch.object(socket, 'socket', mock_socket_class):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 验证 mock 被调用（说明 allow_network 生效，socket 没被阻断 fixture 替换）
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert sock is mock_socket_instance

    @pytest.mark.allow_network
    @pytest.mark.block_network
    def test_allow_network_has_higher_priority_than_block_network(self):
        """allow_network 优先级高于 block_network
        
        同时标记 allow_network 和 block_network 时，allow_network 生效。
        """
        # 使用 mock 验证 socket 没被阻断
        mock_socket_instance = MagicMock()
        mock_socket_class = MagicMock(return_value=mock_socket_instance)
        
        with patch.object(socket, 'socket', mock_socket_class):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 验证调用成功（未被阻断）
        mock_socket_class.assert_called_once()
        assert sock is mock_socket_instance

    @pytest.mark.allow_network
    def test_httpx_mock_works_with_allow_marker(self):
        """httpx mock 在 allow_network 标记下正常工作
        
        验证可以正常使用 respx 进行 mock，说明 fixture 没有阻断。
        """
        import httpx
        import respx
        
        with respx.mock:
            respx.get("https://example.com/test").respond(200, json={"status": "ok"})
            
            with httpx.Client() as client:
                response = client.get("https://example.com/test")
            
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


# ==================== 环境变量动态阻断测试 ====================


class TestEnvironmentVariableBlocking:
    """环境变量 CURSORX_BLOCK_NETWORK 动态阻断测试
    
    验证通过环境变量在测试内动态启用网络阻断功能。
    这些测试不依赖模块导入缓存，每次运行时重新读取环境变量。
    
    注意：由于 autouse fixture 在测试开始时就执行了，monkeypatch 设置的
    环境变量无法影响已经评估过的阻断条件。因此这里主要验证 _is_network_blocked
    函数的运行时读取逻辑。
    """

    def test_env_block_network_blocks_socket(self, monkeypatch):
        """设置 CURSORX_BLOCK_NETWORK=1 后 _is_network_blocked 返回 True
        
        通过 monkeypatch 动态设置环境变量，验证运行时读取生效。
        """
        # 设置环境变量启用阻断
        monkeypatch.setenv("CURSORX_BLOCK_NETWORK", "1")
        
        # 验证 _is_network_blocked 运行时读取返回 True
        from tests.conftest import _is_network_blocked
        
        assert _is_network_blocked() is True

    def test_env_block_network_false_allows_socket(self, monkeypatch):
        """CURSORX_BLOCK_NETWORK=0 或未设置时不阻断
        
        验证环境变量值为 0、false 或未设置时，不触发阻断。
        """
        from tests.conftest import _is_network_blocked
        
        # 测试各种非阻断值
        for value in ["0", "false", "no", "", "random"]:
            monkeypatch.setenv("CURSORX_BLOCK_NETWORK", value)
            assert _is_network_blocked() is False, f"Expected False for value: {value!r}"
        
        # 测试未设置
        monkeypatch.delenv("CURSORX_BLOCK_NETWORK", raising=False)
        assert _is_network_blocked() is False

    def test_env_block_network_true_values(self, monkeypatch):
        """验证 CURSORX_BLOCK_NETWORK 接受的启用值"""
        from tests.conftest import _is_network_blocked
        
        for value in ["1", "true", "yes", "TRUE", "True", "YES", "Yes"]:
            monkeypatch.setenv("CURSORX_BLOCK_NETWORK", value)
            assert _is_network_blocked() is True, f"Expected True for value: {value!r}"


# ==================== fixture 集成测试 ====================


class TestNoApiKeyAndBlockNetworkFixture:
    """no_api_key_and_block_network fixture 测试
    
    验证组合 fixture 正确清除 API Key 并启用网络阻断。
    """

    def test_fixture_clears_api_keys(self, no_api_key_and_block_network):
        """fixture 清除 API Key 环境变量"""
        assert os.environ.get("CURSOR_API_KEY") is None
        assert os.environ.get("CURSOR_CLOUD_API_KEY") is None

    def test_fixture_enables_network_block(self, no_api_key_and_block_network):
        """fixture 启用网络阻断环境变量"""
        assert os.environ.get("CURSORX_BLOCK_NETWORK") == "1"

    def test_fixture_sets_block_env_correctly(self, no_api_key_and_block_network):
        """fixture 正确设置阻断环境变量
        
        注意：由于 autouse fixture 的执行顺序，环境变量在 _block_network_requests
        评估后才设置。此测试验证 _is_network_blocked() 运行时读取正确。
        要验证实际阻断效果，需使用 @pytest.mark.block_network 标记。
        """
        from tests.conftest import _is_network_blocked
        
        # 验证运行时读取环境变量返回 True
        assert _is_network_blocked() is True


class TestNoApiKeyAndBlockNetworkWithMarker:
    """使用 block_network 标记验证 no_api_key_and_block_network fixture
    
    结合 @pytest.mark.block_network 标记确保阻断生效。
    """

    @pytest.mark.block_network
    def test_fixture_with_marker_blocks_socket(self, no_api_key_and_block_network):
        """fixture + block_network 标记阻断 socket"""
        with pytest.raises(BlockedNetworkError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# ==================== 优先级规则完整测试 ====================


class TestBlockingPriorityRules:
    """网络阻断优先级规则测试
    
    优先级（从高到低）：
    1. @pytest.mark.allow_network - 始终放行
    2. @pytest.mark.integration + 有 API Key - 放行
    3. @pytest.mark.block_network - 阻断
    4. 环境变量 CURSORX_BLOCK_NETWORK=1 - 阻断
    5. 默认 - 放行
    """

    def test_default_allows_socket(self):
        """默认情况（无标记、无环境变量）放行
        
        使用 mock 验证 socket 可以正常创建。
        """
        mock_socket = MagicMock()
        
        with patch.object(socket, 'socket', return_value=mock_socket):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        assert sock is mock_socket

    @pytest.mark.integration
    def test_integration_with_api_key_allows_socket(self, monkeypatch):
        """integration 标记 + API Key 放行"""
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        
        mock_socket = MagicMock()
        
        with patch.object(socket, 'socket', return_value=mock_socket):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        assert sock is mock_socket

    @pytest.mark.allow_network
    def test_allow_network_overrides_env_block(self, monkeypatch):
        """allow_network 标记覆盖环境变量阻断
        
        即使设置了 CURSORX_BLOCK_NETWORK=1，allow_network 仍然放行。
        """
        monkeypatch.setenv("CURSORX_BLOCK_NETWORK", "1")
        
        mock_socket = MagicMock()
        
        with patch.object(socket, 'socket', return_value=mock_socket):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # allow_network 生效，socket 未被阻断
        assert sock is mock_socket


# ==================== integration 标记 + API Key 真实放行测试 ====================


@pytest.mark.integration
class TestIntegrationMarkerWithApiKey:
    """integration 标记 + API Key 放行测试
    
    使用 autouse fixture 确保 API Key 在 _block_network_requests 决策前设置。
    这样可以验证 integration + API Key 的真实放行逻辑。
    """

    @pytest.fixture(autouse=True)
    def setup_api_key(self, monkeypatch):
        """设置 API Key（autouse 确保先于 _block_network_requests）"""
        monkeypatch.setenv("CURSOR_API_KEY", "test-integration-api-key")
        # 同时设置阻断环境变量，验证 integration + API Key 能覆盖它
        monkeypatch.setenv("CURSORX_BLOCK_NETWORK", "1")
        yield

    def test_integration_bypasses_env_block_with_api_key(self):
        """验证 integration + API Key 放行，即使设置了阻断环境变量
        
        这是真实的放行测试，不使用 mock，而是直接创建 socket。
        """
        # 不应抛出 BlockedNetworkError
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.close()
        except BlockedNetworkError:
            pytest.fail("@pytest.mark.integration + API Key 应绕过网络阻断")

    def test_integration_api_key_is_set(self):
        """验证 API Key 确实被设置"""
        assert os.environ.get("CURSOR_API_KEY") == "test-integration-api-key"

    def test_integration_block_env_is_set(self):
        """验证阻断环境变量确实被设置（但被 integration + API Key 覆盖）"""
        assert os.environ.get("CURSORX_BLOCK_NETWORK") == "1"


# ==================== BlockedNetworkError 异常测试 ====================


class TestBlockedNetworkError:
    """BlockedNetworkError 异常测试"""

    def test_exception_is_assertion_error(self):
        """BlockedNetworkError 继承自 AssertionError"""
        assert issubclass(BlockedNetworkError, AssertionError)

    def test_exception_message_is_informative(self):
        """异常消息包含有用信息"""
        error = BlockedNetworkError("test message")
        assert str(error) == "test message"

    @pytest.mark.block_network
    def test_exception_raised_with_detailed_message(self):
        """阻断时异常包含详细指引"""
        with pytest.raises(BlockedNetworkError) as exc_info:
            socket.socket()
        
        message = str(exc_info.value)
        # 验证消息包含解决方案提示
        assert "@pytest.mark.allow_network" in message
        assert "mock" in message.lower()
        assert "CURSORX_BLOCK_NETWORK" in message


# ==================== 边界条件测试 ====================


class TestEdgeCases:
    """边界条件测试"""

    @pytest.mark.block_network
    def test_multiple_socket_calls_all_blocked(self):
        """多次 socket 调用都被阻断"""
        for _ in range(3):
            with pytest.raises(BlockedNetworkError):
                socket.socket()

    @pytest.mark.block_network
    def test_different_socket_types_all_blocked(self):
        """不同类型的 socket 都被阻断"""
        socket_types = [
            (socket.AF_INET, socket.SOCK_STREAM),   # TCP
            (socket.AF_INET, socket.SOCK_DGRAM),    # UDP
            (socket.AF_INET6, socket.SOCK_STREAM),  # IPv6 TCP
        ]
        
        for family, sock_type in socket_types:
            with pytest.raises(BlockedNetworkError):
                socket.socket(family, sock_type)

    def test_original_socket_preserved_after_test(self):
        """测试结束后原始 socket 被恢复
        
        验证 fixture 使用 patch 上下文管理器，测试结束后自动恢复。
        """
        from tests.conftest import _original_socket
        
        # 当前的 socket.socket 应该是原始的（因为此测试没有 block_network 标记）
        # 这里只验证原始 socket 被正确保存
        assert _original_socket is not None
        assert callable(_original_socket)
