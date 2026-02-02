"""平台检测和自适应逻辑

提供跨平台兼容性支持，包括：
- 平台检测 (Linux/macOS/Windows)
- 多进程启动方式检测
- 信号处理适配
"""

import multiprocessing as mp
import signal
import sys
from enum import Enum
from typing import Callable, Optional


class Platform(str, Enum):
    """平台类型枚举"""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class MPStartMethod(str, Enum):
    """多进程启动方式枚举"""

    FORK = "fork"
    SPAWN = "spawn"
    FORKSERVER = "forkserver"


def get_platform() -> Platform:
    """获取当前运行平台

    Returns:
        Platform: 平台类型枚举值
    """
    if sys.platform == "darwin":
        return Platform.MACOS
    elif sys.platform.startswith("linux"):
        return Platform.LINUX
    elif sys.platform == "win32":
        return Platform.WINDOWS
    return Platform.UNKNOWN


def get_mp_start_method() -> MPStartMethod:
    """获取当前多进程启动方式

    Returns:
        MPStartMethod: 启动方式枚举值

    Notes:
        - Linux 默认使用 fork
        - macOS (Python 3.8+) 默认使用 spawn
        - Windows 只支持 spawn
    """
    method = mp.get_start_method(allow_none=True)
    if method is None:
        # 未设置时返回平台默认值
        platform = get_platform()
        if platform == Platform.LINUX:
            return MPStartMethod.FORK
        else:
            return MPStartMethod.SPAWN

    return MPStartMethod(method)


def is_fork_start_method() -> bool:
    """检查是否使用 fork 启动方式

    Returns:
        bool: 是否使用 fork
    """
    return get_mp_start_method() == MPStartMethod.FORK


def is_spawn_start_method() -> bool:
    """检查是否使用 spawn 启动方式

    spawn 方式需要所有对象可被 pickle 序列化。

    Returns:
        bool: 是否使用 spawn
    """
    return get_mp_start_method() == MPStartMethod.SPAWN


def requires_pickle_serialization() -> bool:
    """检查是否需要 pickle 序列化

    macOS 和 Windows 使用 spawn 方式，需要序列化所有对象。

    Returns:
        bool: 是否需要 pickle 序列化
    """
    return not is_fork_start_method()


def supports_signal(sig: signal.Signals) -> bool:
    """检查当前平台是否支持指定信号

    Args:
        sig: 信号类型

    Returns:
        bool: 是否支持该信号

    Notes:
        - Windows 不支持 SIGTERM、SIGKILL 等 Unix 信号
        - Windows 仅支持 SIGINT、SIGBREAK、SIGABRT、SIGFPE、SIGILL、SIGSEGV、SIGTERM（有限支持）
    """
    platform = get_platform()

    if platform == Platform.WINDOWS:
        # Windows 支持的信号列表（有限）
        windows_supported = {
            signal.SIGINT,
            signal.SIGABRT,
            signal.SIGFPE,
            signal.SIGILL,
            signal.SIGSEGV,
        }
        # SIGTERM 在 Windows 上可用但行为不同
        if hasattr(signal, "SIGTERM"):
            windows_supported.add(signal.SIGTERM)
        if hasattr(signal, "SIGBREAK"):
            windows_supported.add(signal.SIGBREAK)
        return sig in windows_supported

    # Unix 系统支持所有标准信号
    return True


def register_signal_handler(
    sig: signal.Signals,
    handler: Callable,
    fallback_sig: Optional[signal.Signals] = None,
) -> bool:
    """跨平台注册信号处理器

    Args:
        sig: 主信号类型
        handler: 信号处理函数
        fallback_sig: 备用信号（当主信号不支持时使用）

    Returns:
        bool: 是否成功注册

    Example:
        ```python
        def shutdown_handler(signum, frame):
            print("Shutting down...")

        # 在 Unix 上使用 SIGTERM，在 Windows 上回退到 SIGINT
        register_signal_handler(
            signal.SIGTERM,
            shutdown_handler,
            fallback_sig=signal.SIGINT
        )
        ```
    """
    if supports_signal(sig):
        try:
            signal.signal(sig, handler)
            return True
        except (OSError, ValueError):
            pass

    # 尝试使用备用信号
    if fallback_sig is not None and supports_signal(fallback_sig):
        try:
            signal.signal(fallback_sig, handler)
            return True
        except (OSError, ValueError):
            pass

    return False


def get_platform_info() -> dict:
    """获取平台信息摘要

    Returns:
        dict: 包含平台相关信息的字典
    """
    return {
        "platform": get_platform().value,
        "sys_platform": sys.platform,
        "python_version": sys.version,
        "mp_start_method": get_mp_start_method().value,
        "requires_pickle": requires_pickle_serialization(),
    }


# ==================== 导出 ====================

__all__ = [
    # 枚举类型
    "Platform",
    "MPStartMethod",
    # 平台检测
    "get_platform",
    "get_mp_start_method",
    "is_fork_start_method",
    "is_spawn_start_method",
    "requires_pickle_serialization",
    # 信号处理
    "supports_signal",
    "register_signal_handler",
    # 信息获取
    "get_platform_info",
]
