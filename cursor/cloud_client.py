"""Cursor Cloud 客户端入口模块

统一入口，重新导出所有 cloud 子模块的公共 API:
- 异常类和错误处理工具 (cloud.exceptions)
- 认证管理 (cloud.auth)
- 任务管理 (cloud.task)
- Cloud Agent 客户端 (cloud.client)
"""

# 导入异常类和错误处理工具
# 导入认证模块
from .cloud.auth import (
    AuthStatus,
    AuthToken,
    CloudAuthConfig,
    CloudAuthManager,
    get_api_key,
    require_auth,
    verify_auth,
    verify_auth_sync,
)

# 导入 Cloud Agent 客户端
from .cloud.client import (
    CloudAgentResult,
    CursorCloudClient,
)
from .cloud.exceptions import (
    AuthError,
    AuthErrorCode,
    CloudAgentError,
    NetworkError,
    RateLimitError,
    TaskError,
    handle_http_error,
)
from .cloud.retry import (
    RetryConfig,
    retry_async,
    with_retry,
)

# 导入任务管理模块
from .cloud.task import (
    CloudTask,
    CloudTaskClient,
    CloudTaskOptions,
    TaskResult,
    TaskStatus,
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
