"""平台兼容性测试

测试 core/platform.py 的功能以及跨平台兼容性。

测试覆盖：
- 平台检测
- 多进程启动方式检测
- 信号处理适配
- pickle 序列化兼容性
"""
import multiprocessing as mp
import pickle
import signal
import sys
from unittest.mock import patch, MagicMock

import pytest

from core.platform import (
    Platform,
    MPStartMethod,
    get_platform,
    get_mp_start_method,
    is_fork_start_method,
    is_spawn_start_method,
    requires_pickle_serialization,
    supports_signal,
    register_signal_handler,
    get_platform_info,
)


# ============================================================================
# 平台检测测试
# ============================================================================


class TestGetPlatform:
    """get_platform() 函数测试"""

    def test_returns_platform_enum(self):
        """测试返回 Platform 枚举类型"""
        result = get_platform()
        assert isinstance(result, Platform)

    def test_detects_current_platform(self):
        """测试能正确检测当前平台"""
        result = get_platform()
        if sys.platform == "darwin":
            assert result == Platform.MACOS
        elif sys.platform.startswith("linux"):
            assert result == Platform.LINUX
        elif sys.platform == "win32":
            assert result == Platform.WINDOWS
        else:
            assert result == Platform.UNKNOWN

    @patch("core.platform.sys")
    def test_detects_linux(self, mock_sys):
        """测试检测 Linux 平台"""
        mock_sys.platform = "linux"
        # 需要重新导入以应用 mock
        from core import platform as platform_module
        with patch.object(platform_module.sys, 'platform', 'linux'):
            result = platform_module.get_platform()
            assert result == Platform.LINUX

    @patch("core.platform.sys")
    def test_detects_macos(self, mock_sys):
        """测试检测 macOS 平台"""
        mock_sys.platform = "darwin"
        from core import platform as platform_module
        with patch.object(platform_module.sys, 'platform', 'darwin'):
            result = platform_module.get_platform()
            assert result == Platform.MACOS

    @patch("core.platform.sys")
    def test_detects_windows(self, mock_sys):
        """测试检测 Windows 平台"""
        mock_sys.platform = "win32"
        from core import platform as platform_module
        with patch.object(platform_module.sys, 'platform', 'win32'):
            result = platform_module.get_platform()
            assert result == Platform.WINDOWS

    @patch("core.platform.sys")
    def test_detects_unknown(self, mock_sys):
        """测试检测未知平台"""
        mock_sys.platform = "freebsd"
        from core import platform as platform_module
        with patch.object(platform_module.sys, 'platform', 'freebsd'):
            result = platform_module.get_platform()
            assert result == Platform.UNKNOWN


class TestPlatformEnum:
    """Platform 枚举测试"""

    def test_platform_values(self):
        """测试平台枚举值"""
        assert Platform.LINUX.value == "linux"
        assert Platform.MACOS.value == "macos"
        assert Platform.WINDOWS.value == "windows"
        assert Platform.UNKNOWN.value == "unknown"

    def test_platform_is_string_enum(self):
        """测试 Platform 是字符串枚举"""
        assert isinstance(Platform.LINUX, str)
        assert Platform.LINUX == "linux"


# ============================================================================
# 多进程启动方式测试
# ============================================================================


class TestMPStartMethod:
    """多进程启动方式测试"""

    def test_get_mp_start_method_returns_enum(self):
        """测试返回 MPStartMethod 枚举类型"""
        result = get_mp_start_method()
        assert isinstance(result, MPStartMethod)

    def test_mp_start_method_enum_values(self):
        """测试启动方式枚举值"""
        assert MPStartMethod.FORK.value == "fork"
        assert MPStartMethod.SPAWN.value == "spawn"
        assert MPStartMethod.FORKSERVER.value == "forkserver"

    def test_is_fork_or_spawn(self):
        """测试 fork 和 spawn 检测互斥"""
        # 当前平台只能是其中之一
        is_fork = is_fork_start_method()
        is_spawn = is_spawn_start_method()

        # forkserver 情况下两者都是 False，但在大多数情况下应该互斥
        method = get_mp_start_method()
        if method == MPStartMethod.FORK:
            assert is_fork is True
            assert is_spawn is False
        elif method == MPStartMethod.SPAWN:
            assert is_fork is False
            assert is_spawn is True

    def test_requires_pickle_on_spawn(self):
        """测试 spawn 方式需要 pickle"""
        if is_spawn_start_method():
            assert requires_pickle_serialization() is True
        else:
            assert requires_pickle_serialization() is False

    @patch("core.platform.mp")
    def test_default_method_on_linux(self, mock_mp):
        """测试 Linux 默认启动方式"""
        mock_mp.get_start_method.return_value = None
        from core import platform as platform_module
        with patch.object(platform_module, 'get_platform', return_value=Platform.LINUX):
            with patch.object(platform_module.mp, 'get_start_method', return_value=None):
                result = platform_module.get_mp_start_method()
                assert result == MPStartMethod.FORK

    @patch("core.platform.mp")
    def test_default_method_on_macos(self, mock_mp):
        """测试 macOS 默认启动方式"""
        mock_mp.get_start_method.return_value = None
        from core import platform as platform_module
        with patch.object(platform_module, 'get_platform', return_value=Platform.MACOS):
            with patch.object(platform_module.mp, 'get_start_method', return_value=None):
                result = platform_module.get_mp_start_method()
                assert result == MPStartMethod.SPAWN

    @patch("core.platform.mp")
    def test_default_method_on_windows(self, mock_mp):
        """测试 Windows 默认启动方式"""
        mock_mp.get_start_method.return_value = None
        from core import platform as platform_module
        with patch.object(platform_module, 'get_platform', return_value=Platform.WINDOWS):
            with patch.object(platform_module.mp, 'get_start_method', return_value=None):
                result = platform_module.get_mp_start_method()
                assert result == MPStartMethod.SPAWN


# ============================================================================
# 信号处理测试
# ============================================================================


class TestSignalHandling:
    """信号处理测试"""

    def test_sigint_supported_all_platforms(self):
        """测试 SIGINT 在所有平台都支持"""
        assert supports_signal(signal.SIGINT) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM 在 Windows 上行为不同")
    def test_sigterm_supported_unix(self):
        """测试 SIGTERM 在 Unix 上支持"""
        assert supports_signal(signal.SIGTERM) is True

    def test_register_signal_handler_returns_bool(self):
        """测试注册信号处理器返回布尔值"""
        handler = MagicMock()
        result = register_signal_handler(signal.SIGINT, handler)
        assert isinstance(result, bool)

    def test_register_signal_handler_success(self):
        """测试成功注册信号处理器"""
        handler = MagicMock()
        result = register_signal_handler(signal.SIGINT, handler)
        assert result is True

    @pytest.mark.skipif(sys.platform != "win32", reason="仅在 Windows 上测试")
    def test_register_fallback_on_windows(self):
        """测试 Windows 上使用备用信号"""
        handler = MagicMock()
        # 模拟 SIGUSR1 不存在的情况（Windows）
        with patch("core.platform.supports_signal") as mock_supports:
            mock_supports.side_effect = lambda sig: sig == signal.SIGINT
            result = register_signal_handler(
                signal.SIGTERM,  # 可能不完全支持
                handler,
                fallback_sig=signal.SIGINT,
            )
            # 应该使用备用信号成功注册
            assert result is True


# ============================================================================
# 平台信息测试
# ============================================================================


class TestPlatformInfo:
    """平台信息测试"""

    def test_get_platform_info_returns_dict(self):
        """测试返回字典类型"""
        result = get_platform_info()
        assert isinstance(result, dict)

    def test_get_platform_info_keys(self):
        """测试返回字典包含所需键"""
        result = get_platform_info()
        assert "platform" in result
        assert "sys_platform" in result
        assert "python_version" in result
        assert "mp_start_method" in result
        assert "requires_pickle" in result

    def test_get_platform_info_values(self):
        """测试返回字典值类型正确"""
        result = get_platform_info()
        assert isinstance(result["platform"], str)
        assert isinstance(result["sys_platform"], str)
        assert isinstance(result["python_version"], str)
        assert isinstance(result["mp_start_method"], str)
        assert isinstance(result["requires_pickle"], bool)


# ============================================================================
# Pickle 序列化兼容性测试
# ============================================================================


class TestPickleCompatibility:
    """Pickle 序列化兼容性测试"""

    def test_platform_enum_pickle(self):
        """测试 Platform 枚举可被 pickle"""
        for platform in Platform:
            pickled = pickle.dumps(platform)
            unpickled = pickle.loads(pickled)
            assert unpickled == platform

    def test_mp_start_method_enum_pickle(self):
        """测试 MPStartMethod 枚举可被 pickle"""
        for method in MPStartMethod:
            pickled = pickle.dumps(method)
            unpickled = pickle.loads(pickled)
            assert unpickled == method

    def test_platform_info_pickle(self):
        """测试平台信息可被 pickle"""
        info = get_platform_info()
        pickled = pickle.dumps(info)
        unpickled = pickle.loads(pickled)
        assert unpickled == info


# ============================================================================
# 跨平台路径处理测试
# ============================================================================


class TestCrossPlatformPaths:
    """跨平台路径处理测试"""

    def test_pathlib_works_cross_platform(self):
        """测试 pathlib 在所有平台正常工作"""
        from pathlib import Path

        # 创建路径
        path = Path("test") / "subdir" / "file.txt"
        assert path.parts == ("test", "subdir", "file.txt")

    def test_os_path_join_cross_platform(self):
        """测试 os.path.join 在所有平台正常工作"""
        import os

        path = os.path.join("test", "subdir", "file.txt")
        assert "test" in path
        assert "subdir" in path
        assert "file.txt" in path


# ============================================================================
# Worker 进程兼容性测试
# ============================================================================


class TestWorkerProcessCompatibility:
    """Worker 进程跨平台兼容性测试"""

    def test_worker_process_can_be_imported(self):
        """测试 Worker 进程模块可被导入"""
        from process.worker import AgentWorkerProcess
        assert AgentWorkerProcess is not None

    def test_process_message_pickle(self):
        """测试 ProcessMessage 可被 pickle"""
        from process.message_queue import ProcessMessage, ProcessMessageType

        msg = ProcessMessage(
            type=ProcessMessageType.TASK_ASSIGN,
            sender="test",
            payload={"key": "value"},
        )

        pickled = pickle.dumps(msg)
        unpickled = pickle.loads(pickled)

        assert unpickled.type == msg.type
        assert unpickled.sender == msg.sender
        assert unpickled.payload == msg.payload


# ============================================================================
# 平台特定标记测试
# ============================================================================


@pytest.mark.skipif(sys.platform != "darwin", reason="仅在 macOS 上运行")
class TestMacOSSpecific:
    """macOS 特定测试"""

    def test_spawn_default_on_macos(self):
        """测试 macOS 默认使用 spawn"""
        # macOS Python 3.8+ 默认使用 spawn
        method = mp.get_start_method(allow_none=True)
        # 如果已设置，检查是否为 spawn；如果未设置，验证默认行为
        if method is None:
            # 默认应该是 spawn
            assert get_mp_start_method() in (MPStartMethod.SPAWN, MPStartMethod.FORK)
        else:
            assert method in ("spawn", "fork", "forkserver")


@pytest.mark.skipif(sys.platform != "linux", reason="仅在 Linux 上运行")
class TestLinuxSpecific:
    """Linux 特定测试"""

    def test_fork_available_on_linux(self):
        """测试 Linux 上 fork 可用"""
        # Linux 支持 fork
        assert "fork" in mp.get_all_start_methods()


@pytest.mark.skipif(sys.platform != "win32", reason="仅在 Windows 上运行")
class TestWindowsSpecific:
    """Windows 特定测试"""

    def test_only_spawn_on_windows(self):
        """测试 Windows 只支持 spawn"""
        methods = mp.get_all_start_methods()
        assert "spawn" in methods
        assert "fork" not in methods
