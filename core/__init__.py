"""核心抽象层"""
from .base import AgentRole, AgentStatus, BaseAgent
from .cloud_utils import CLOUD_PREFIX, is_cloud_request, strip_cloud_prefix
from .knowledge import (
    CURSOR_KEYWORDS,
    FALLBACK_CHARS_PER_DOC,
    KnowledgeDoc,
    MAX_CHARS_PER_DOC,
    MAX_CLI_ASK_CHARS_PER_DOC,
    MAX_CLI_ASK_DOCS,
    MAX_KNOWLEDGE_DOCS,
    MAX_TOTAL_KNOWLEDGE_CHARS,
    MIN_DOCS_ON_FALLBACK,
    is_cursor_related,
    truncate_knowledge_docs,
)
from .message import Message, MessageType
from .state import (
    AgentState,
    CommitContext,
    CommitPolicy,
    IterationState,
    IterationStatus,
    SystemState,
)

__all__ = [
    # Agent 基础
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    # 消息
    "Message",
    "MessageType",
    # 状态
    "AgentState",
    "CommitContext",
    "CommitPolicy",
    "IterationState",
    "IterationStatus",
    "SystemState",
    # Cloud 工具
    "CLOUD_PREFIX",
    "is_cloud_request",
    "strip_cloud_prefix",
    # 知识库共享模块
    "CURSOR_KEYWORDS",
    "KnowledgeDoc",
    "is_cursor_related",
    "truncate_knowledge_docs",
    "MAX_KNOWLEDGE_DOCS",
    "MAX_CHARS_PER_DOC",
    "MAX_TOTAL_KNOWLEDGE_CHARS",
    "MAX_CLI_ASK_DOCS",
    "MAX_CLI_ASK_CHARS_PER_DOC",
    "FALLBACK_CHARS_PER_DOC",
    "MIN_DOCS_ON_FALLBACK",
]
