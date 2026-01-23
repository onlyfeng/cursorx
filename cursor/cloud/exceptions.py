"""异常类和错误处理工具

包含:
- CloudAgentError: 错误基类
- RateLimitError: 限流错误（HTTP 429）
- NetworkError: 网络错误
- TaskError: 任务执行错误
- AuthError: 认证错误
- handle_http_error: HTTP 错误处理工具
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

if TYPE_CHECKING:
    from .task import TaskStatus

# 检查 httpx 可用性
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ========== 异常层次结构 ==========

class CloudAgentError(Exception):
    """Cloud Agent 错误基类

    所有 Cloud Agent 相关错误的基类，提供统一的错误处理接口。

    Attributes:
        message: 错误消息
        details: 额外的错误详情
        timestamp: 错误发生时间
        retry_after: 建议重试等待时间（秒）
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        self.retry_after = retry_after

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message}"

    @property
    def user_friendly_message(self) -> str:
        """返回用户友好的错误消息"""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "retry_after": self.retry_after,
        }


class RateLimitError(CloudAgentError):
    """限流错误（HTTP 429）

    当请求频率超过限制时抛出，包含 Retry-After 信息。

    Attributes:
        retry_after: 需要等待的秒数（从 Retry-After 头解析）
        limit: 请求限制
        remaining: 剩余请求数
        reset_time: 限制重置时间
    """

    def __init__(
        self,
        message: str = "请求过于频繁，已被限流",
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details, retry_after)
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time

        logger.warning(
            f"限流错误: {message}, retry_after={retry_after}s, "
            f"limit={limit}, remaining={remaining}"
        )

    @property
    def user_friendly_message(self) -> str:
        """返回用户友好的错误消息"""
        if self.retry_after:
            return (
                f"请求过于频繁，已被限流。\n"
                f"请在 {self.retry_after:.1f} 秒后重试。"
            )
        return "请求过于频繁，已被限流。请稍后重试。"

    @classmethod
    def from_response_headers(
        cls,
        status_code: int,
        headers: dict[str, str],
        body: Optional[str] = None,
    ) -> "RateLimitError":
        """从 HTTP 响应头解析限流错误

        Args:
            status_code: HTTP 状态码
            headers: 响应头
            body: 响应体

        Returns:
            RateLimitError 实例
        """
        # 解析 Retry-After 头
        retry_after = None
        retry_after_raw = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after_raw:
            try:
                # Retry-After 可以是秒数或 HTTP 日期
                retry_after = float(retry_after_raw)
            except ValueError:
                # 尝试解析为 HTTP 日期
                try:
                    from email.utils import parsedate_to_datetime
                    retry_date = parsedate_to_datetime(retry_after_raw)
                    retry_after = (retry_date - datetime.now()).total_seconds()
                    if retry_after < 0:
                        retry_after = 60.0  # 默认 60 秒
                except Exception:
                    retry_after = 60.0  # 解析失败，默认 60 秒

        # 解析其他限流相关头
        limit = None
        remaining = None
        reset_time = None

        # X-RateLimit-Limit
        limit_raw = headers.get("X-RateLimit-Limit") or headers.get("x-ratelimit-limit")
        if limit_raw:
            try:
                limit = int(limit_raw)
            except ValueError:
                pass

        # X-RateLimit-Remaining
        remaining_raw = headers.get("X-RateLimit-Remaining") or headers.get("x-ratelimit-remaining")
        if remaining_raw:
            try:
                remaining = int(remaining_raw)
            except ValueError:
                pass

        # X-RateLimit-Reset
        reset_raw = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
        if reset_raw:
            try:
                reset_timestamp = int(reset_raw)
                reset_time = datetime.fromtimestamp(reset_timestamp)
            except ValueError:
                pass

        # 构建详细信息
        details = {
            "status_code": status_code,
            "headers": dict(headers),
        }
        if body:
            details["body"] = body[:500]  # 截断响应体

        message = body if body else "请求过于频繁，已被限流"

        return cls(
            message=message,
            retry_after=retry_after,
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            details=details,
        )


class NetworkError(CloudAgentError):
    """网络错误

    包括连接超时、DNS 解析失败、SSL 错误等网络层面的问题。

    Attributes:
        error_type: 网络错误类型（timeout, dns, ssl, connection, etc.）
        original_error: 原始异常
    """

    def __init__(
        self,
        message: str,
        error_type: str = "unknown",
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, details, retry_after)
        self.error_type = error_type
        self.original_error = original_error

        # 记录网络错误日志
        logger.error(
            f"网络错误 [{error_type}]: {message}",
            exc_info=original_error,
        )

    @property
    def user_friendly_message(self) -> str:
        """返回用户友好的错误消息"""
        messages = {
            "timeout": "请求超时。请检查网络连接或稍后重试。",
            "dns": "DNS 解析失败。请检查网络配置。",
            "ssl": "SSL/TLS 连接错误。请检查证书配置。",
            "connection": "无法连接到服务器。请检查网络连接。",
            "connection_refused": "连接被拒绝。服务器可能暂时不可用。",
            "connection_reset": "连接被重置。请稍后重试。",
        }
        return messages.get(self.error_type, f"网络连接失败: {self.message}")

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        context: str = "",
    ) -> "NetworkError":
        """从异常创建 NetworkError

        自动检测异常类型并设置对应的 error_type。

        Args:
            error: 原始异常
            context: 上下文描述

        Returns:
            NetworkError 实例
        """
        error_str = str(error).lower()
        error_type = "unknown"
        message = str(error)
        retry_after = None

        # 检测异常类型
        if isinstance(error, asyncio.TimeoutError):
            error_type = "timeout"
            message = f"请求超时{': ' + context if context else ''}"
            retry_after = 5.0  # 超时后建议等待 5 秒
        elif "timeout" in error_str:
            error_type = "timeout"
            retry_after = 5.0
        elif "dns" in error_str or "name resolution" in error_str:
            error_type = "dns"
        elif "ssl" in error_str or "certificate" in error_str:
            error_type = "ssl"
        elif "connection refused" in error_str:
            error_type = "connection_refused"
            retry_after = 10.0
        elif "connection reset" in error_str:
            error_type = "connection_reset"
            retry_after = 5.0
        elif "connection" in error_str:
            error_type = "connection"
            retry_after = 5.0

        # httpx 特定异常
        if HTTPX_AVAILABLE:
            if isinstance(error, httpx.ConnectError):
                error_type = "connection"
                retry_after = 5.0
            elif isinstance(error, httpx.TimeoutException):
                error_type = "timeout"
                retry_after = 5.0
            elif isinstance(error, httpx.RequestError):
                error_type = "client_error"

        return cls(
            message=f"{context}: {message}" if context else message,
            error_type=error_type,
            original_error=error,
            details={"error_class": type(error).__name__},
            retry_after=retry_after,
        )


class TaskError(CloudAgentError):
    """任务执行错误

    任务执行过程中的错误，包括任务失败、取消、超时等。

    Attributes:
        task_id: 任务 ID
        task_status: 任务状态
        exit_code: 进程退出码（如果适用）
    """

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        task_status: Optional["TaskStatus"] = None,
        exit_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.task_id = task_id
        self.task_status = task_status
        self.exit_code = exit_code

        logger.error(
            f"任务错误 [task_id={task_id}, status={task_status}]: {message}"
        )

    @property
    def user_friendly_message(self) -> str:
        """返回用户友好的错误消息"""
        # 延迟导入避免循环依赖
        from .task import TaskStatus

        if self.task_status == TaskStatus.TIMEOUT:
            return f"任务执行超时: {self.message}"
        elif self.task_status == TaskStatus.CANCELLED:
            return f"任务已取消: {self.message}"
        elif self.task_status == TaskStatus.FAILED:
            return f"任务执行失败: {self.message}"
        return f"任务错误: {self.message}"


class AuthErrorCode(Enum):
    """认证错误代码"""
    INVALID_API_KEY = "invalid_api_key"
    EXPIRED_TOKEN = "expired_token"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    CONFIG_NOT_FOUND = "config_not_found"
    OAUTH_FAILED = "oauth_failed"
    REFRESH_FAILED = "refresh_failed"
    UNKNOWN = "unknown"


class AuthError(CloudAgentError):
    """认证错误异常

    继承自 CloudAgentError，提供认证相关的错误处理。

    Attributes:
        code: 认证错误代码
    """

    def __init__(
        self,
        message: str,
        code: AuthErrorCode = AuthErrorCode.UNKNOWN,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.code = code

        # 记录认证错误日志
        if code in (AuthErrorCode.INVALID_API_KEY, AuthErrorCode.EXPIRED_TOKEN):
            logger.warning(f"认证错误 [{code.value}]: {message}")
        else:
            logger.error(f"认证错误 [{code.value}]: {message}")

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    @property
    def user_friendly_message(self) -> str:
        """返回用户友好的错误消息"""
        messages = {
            AuthErrorCode.INVALID_API_KEY: (
                "API Key 无效。请检查您的 CURSOR_API_KEY 环境变量或配置文件中的 API Key。\n"
                "获取 API Key: https://cursor.com/settings/api-keys"
            ),
            AuthErrorCode.EXPIRED_TOKEN: (
                "Token 已过期。正在尝试刷新...\n"
                "如果问题持续，请重新登录: agent login"
            ),
            AuthErrorCode.NETWORK_ERROR: (
                "网络连接失败。请检查您的网络连接后重试。"
            ),
            AuthErrorCode.RATE_LIMITED: (
                "请求过于频繁，已被限流。请稍后重试。"
            ),
            AuthErrorCode.INSUFFICIENT_PERMISSIONS: (
                "权限不足。请确认您的账户有权访问此 API。"
            ),
            AuthErrorCode.CONFIG_NOT_FOUND: (
                "未找到配置文件。请设置 CURSOR_API_KEY 环境变量或创建配置文件。\n"
                "设置方法: export CURSOR_API_KEY=your_api_key_here"
            ),
            AuthErrorCode.OAUTH_FAILED: (
                "OAuth 认证失败。请重试或使用 API Key 方式认证。"
            ),
            AuthErrorCode.REFRESH_FAILED: (
                "Token 刷新失败。请重新登录: agent login"
            ),
            AuthErrorCode.UNKNOWN: (
                "发生未知认证错误。请检查日志获取更多信息。"
            ),
        }
        return messages.get(self.code, str(self))

    @classmethod
    def from_http_status(
        cls,
        status_code: int,
        body: Optional[str] = None,
    ) -> "AuthError":
        """从 HTTP 状态码创建认证错误

        Args:
            status_code: HTTP 状态码
            body: 响应体

        Returns:
            AuthError 实例
        """
        if status_code == 401:
            return cls(
                message=body or "认证失败：API Key 无效或已过期",
                code=AuthErrorCode.INVALID_API_KEY,
                details={"status_code": 401, "hint": "请检查 CURSOR_API_KEY 环境变量"},
            )
        elif status_code == 403:
            return cls(
                message=body or "权限不足：无权访问此资源",
                code=AuthErrorCode.INSUFFICIENT_PERMISSIONS,
                details={"status_code": 403, "hint": "请确认账户权限"},
            )
        else:
            return cls(
                message=body or f"认证失败 (HTTP {status_code})",
                code=AuthErrorCode.UNKNOWN,
                details={"status_code": status_code},
            )


# ========== HTTP 错误处理工具 ==========

def handle_http_error(
    status_code: int,
    headers: Optional[dict[str, str]] = None,
    body: Optional[str] = None,
    context: str = "",
) -> CloudAgentError:
    """处理 HTTP 错误响应，返回对应的异常

    根据 HTTP 状态码创建适当的异常类型：
    - 401/403: AuthError
    - 429: RateLimitError
    - 5xx: NetworkError (服务器错误)
    - 其他: CloudAgentError

    Args:
        status_code: HTTP 状态码
        headers: 响应头
        body: 响应体
        context: 上下文描述

    Returns:
        对应的异常实例
    """
    headers = headers or {}

    # 401/403 认证错误
    if status_code in (401, 403):
        error = AuthError.from_http_status(status_code, body)
        logger.warning(f"认证错误 [{context}]: {error.user_friendly_message}")
        return error

    # 429 限流错误
    if status_code == 429:
        error = RateLimitError.from_response_headers(status_code, headers, body)
        return error

    # 5xx 服务器错误
    if 500 <= status_code < 600:
        error = NetworkError(
            message=body or f"服务器错误 (HTTP {status_code})",
            error_type="server_error",
            details={"status_code": status_code, "headers": headers},
            retry_after=5.0,  # 服务器错误建议 5 秒后重试
        )
        return error

    # 其他错误
    return CloudAgentError(
        message=body or f"请求失败 (HTTP {status_code})",
        details={"status_code": status_code, "headers": headers, "context": context},
    )
