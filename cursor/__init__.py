"""Cursor Agent 集成层"""
from .client import CursorAgentClient, CursorAgentConfig, CursorAgentPool, ModelPresets
from .mcp import MCPManager, MCPServer, MCPTool, ensure_mcp_servers_enabled
from .streaming import StreamingClient, StreamEvent, StreamEventType, ProgressTracker, ToolCallInfo

__all__ = [
    # 客户端
    "CursorAgentClient",
    "CursorAgentConfig",
    "CursorAgentPool",
    "ModelPresets",
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
]
