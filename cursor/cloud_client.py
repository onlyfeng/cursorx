"""Cursor Cloud 客户端入口模块

统一入口，重新导出所有 cloud 子模块的公共 API:
- 异常类和错误处理工具 (cloud.exceptions)
- 认证管理 (cloud.auth)
- 任务管理 (cloud.task)
- Cloud Agent 客户端 (cloud.client)
"""

# 导入异常类和错误处理工具
from .cloud.exceptions import (
    CloudAgentError,
    RateLimitError,
    NetworkError,
    TaskError,
    AuthErrorCode,
    AuthError,
    handle_http_error,
)
from .cloud.retry import (
    RetryConfig,
    with_retry,
    retry_async,
)

# 导入认证模块
from .cloud.auth import (
    AuthToken,
    AuthStatus,
    CloudAuthConfig,
    CloudAuthManager,
    get_api_key,
    verify_auth,
    verify_auth_sync,
    require_auth,
)

# 导入任务管理模块
from .cloud.task import (
    TaskStatus,
    TaskResult,
    CloudTaskClient,
    CloudTaskOptions,
    CloudTask,
)

# 导入 Cloud Agent 客户端
from .cloud.client import (
    CursorCloudClient,
    CloudAgentResult,
)

# 导入流式事件类型
from .streaming import (
    StreamEvent,
    StreamEventType,
    ToolCallInfo,
    parse_stream_event,
)

# 别名（统一命名风格）
CloudAgentClient = CursorCloudClient
CloudAgentConfig = CloudAuthConfig


__all__ = [
    # 异常类
    "CloudAgentError",
    "RateLimitError",
    "NetworkError",
    "TaskError",
    "AuthErrorCode",
    "AuthError",
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
    # 任务管理
    "TaskStatus",
    "TaskResult",
    "CloudTaskClient",
    # Cloud Agent
    "CursorCloudClient",
    "CloudTaskOptions",
    "CloudTask",
    "CloudAgentResult",
    # 流式事件
    "StreamEvent",
    "StreamEventType",
    "ToolCallInfo",
    "parse_stream_event",
    # 别名
    "CloudAgentClient",
    "CloudAgentConfig",
]
