"""Cursor Cloud 模块

提供 Cloud Agent 相关功能的统一入口，包括:
- 认证管理 (CloudAuthManager)
- 任务管理 (CloudTaskClient)
- Cloud Agent 客户端 (CursorCloudClient)
- 异常层次结构 (CloudAgentError, RateLimitError, etc.)
- 重试策略 (RetryConfig, with_retry)

用法:
    from cursor.cloud import CloudAuthManager, CursorCloudClient
    
    # 认证管理
    auth = CloudAuthManager()
    status = await auth.authenticate()
    
    # Cloud Agent
    client = CursorCloudClient()
    result = await client.execute("& 实现功能")
"""

# ========== 异常类和错误处理工具 ==========
from .exceptions import (
    CloudAgentError,
    RateLimitError,
    NetworkError,
    TaskError,
    AuthError,
    AuthErrorCode,
    handle_http_error,
)

# ========== 重试工具 ==========
from .retry import RetryConfig, with_retry, retry_async

# ========== 认证管理 ==========
from .auth import (
    AuthToken,
    AuthStatus,
    CloudAuthConfig,
    CloudAuthManager,
    get_api_key,
    verify_auth,
    verify_auth_sync,
    require_auth,
)

# ========== 任务管理 ==========
from .task import (
    TaskStatus,
    TaskResult,
    CloudTaskClient,
    CloudTaskOptions,
    CloudTask,
)

# ========== Cloud Agent 客户端 ==========
from .client import (
    CursorCloudClient,
    CloudAgentResult,
)

# ========== 别名（统一命名风格）==========
CloudAgentClient = CursorCloudClient
CloudAgentConfig = CloudAuthConfig


__all__ = [
    # 任务状态
    "TaskStatus",
    "TaskResult",
    # 异常层次结构
    "CloudAgentError",
    "RateLimitError",
    "NetworkError",
    "TaskError",
    "AuthError",
    "AuthErrorCode",
    # 重试工具
    "RetryConfig",
    "with_retry",
    "retry_async",
    "handle_http_error",
    # 认证管理
    "CloudAuthManager",
    "CloudAuthConfig",
    "AuthToken",
    "AuthStatus",
    "get_api_key",
    "verify_auth",
    "verify_auth_sync",
    "require_auth",
    # 任务管理客户端
    "CloudTaskClient",
    "CloudTaskOptions",
    "CloudTask",
    # Cloud Agent
    "CursorCloudClient",
    "CloudAgentResult",
    # 别名
    "CloudAgentClient",
    "CloudAgentConfig",
]
