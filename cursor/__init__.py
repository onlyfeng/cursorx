"""Cursor Agent 集成层"""
from .client import CursorAgentClient, CursorAgentConfig, CursorAgentPool, CursorAgentResult, ModelPresets
from .cloud_client import (
    # 异常层次结构
    CloudAgentError,
    RateLimitError,
    NetworkError,
    TaskError,
    AuthError,
    AuthErrorCode,
    # 重试工具
    RetryConfig,
    with_retry,
    retry_async,
    handle_http_error,
    # 认证管理
    CloudAuthManager,
    CloudAuthConfig,
    AuthToken,
    AuthStatus,
    get_api_key,
    verify_auth,
    verify_auth_sync,
    require_auth,
    # 任务管理
    CloudTaskClient,
    TaskStatus,
    TaskResult,
    # Cloud Agent（& 前缀推送到云端）
    CursorCloudClient,
    CloudTask,
    CloudTaskOptions,
    CloudAgentResult,
)
from .executor import (
    AgentExecutor,
    AgentResult,
    ExecutionMode,
    CLIAgentExecutor,
    CloudAgentExecutor,
    AutoAgentExecutor,
    PlanAgentExecutor,
    AskAgentExecutor,
    AgentExecutorFactory,
    execute_agent,
    execute_agent_sync,
)
from .mcp import MCPManager, MCPServer, MCPTool, ensure_mcp_servers_enabled
from .streaming import (
    StreamingClient,
    StreamEvent,
    StreamEventType,
    ProgressTracker,
    ToolCallInfo,
    StreamEventLogger,
    parse_stream_event,
    # 差异相关
    DiffInfo,
    format_diff,
    format_inline_diff,
    format_colored_diff,
    get_diff_stats,
    parse_diff_event,
)
from .network import (
    EgressIPConfig,
    EgressIPManager,
    FirewallFormat,
    fetch_egress_ip_ranges,
    is_allowed_ip,
    export_firewall_rules,
    get_manager as get_network_manager,
    IPTABLES, NGINX, APACHE, UFW, CLOUDFLARE, JSON,
)

# 别名（兼容性，统一命名风格）
CloudAgentClient = CursorCloudClient  # Cloud Agent 客户端别名
CloudAgentConfig = CloudAuthConfig    # Cloud Agent 配置别名

__all__ = [
    # 客户端
    "CursorAgentClient",
    "CursorAgentConfig",
    "CursorAgentPool",
    "CursorAgentResult",
    "ModelPresets",
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
    # 任务管理
    "CloudTaskClient",
    "TaskStatus",
    "TaskResult",
    # Cloud Agent（& 前缀推送到云端）
    "CursorCloudClient",
    "CloudTask",
    "CloudTaskOptions",
    "CloudAgentResult",
    # Cloud Agent 别名（兼容性）
    "CloudAgentClient",
    "CloudAgentConfig",
    # 执行器
    "AgentExecutor",
    "AgentResult",
    "ExecutionMode",
    "CLIAgentExecutor",
    "CloudAgentExecutor",
    "AutoAgentExecutor",
    "PlanAgentExecutor",
    "AskAgentExecutor",
    "AgentExecutorFactory",
    "execute_agent",
    "execute_agent_sync",
    # MCP
    "MCPManager",
    "MCPServer",
    "MCPTool",
    "ensure_mcp_servers_enabled",
    # 流式输出
    "StreamingClient",
    "StreamEvent",
    "StreamEventType",
    "ProgressTracker",
    "ToolCallInfo",
    "StreamEventLogger",
    "parse_stream_event",
    # 差异相关
    "DiffInfo",
    "format_diff",
    "format_inline_diff",
    "format_colored_diff",
    "get_diff_stats",
    "parse_diff_event",
    # 网络/IP 管理
    "EgressIPConfig",
    "EgressIPManager",
    "FirewallFormat",
    "fetch_egress_ip_ranges",
    "is_allowed_ip",
    "export_firewall_rules",
    "get_network_manager",
    "IPTABLES",
    "NGINX",
    "APACHE",
    "UFW",
    "CLOUDFLARE",
    "JSON",
]
