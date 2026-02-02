"""知识库共享模块

统一定义知识库相关的数据结构和工具函数，供协程版与多进程版共享。

使用方式:
    # 协程版 (agents/worker.py)
    from core.knowledge import CURSOR_KEYWORDS, is_cursor_related, KnowledgeDoc

    # 多进程版 (coordinator/orchestrator_mp.py, agents/worker_process.py)
    from core.knowledge import CURSOR_KEYWORDS, is_cursor_related, KnowledgeDoc
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ============================================================================
# Cursor 关键词检测
# ============================================================================

# Cursor 相关关键词，用于自动检测是否需要知识库上下文
CURSOR_KEYWORDS: list[str] = [
    "cursor",
    "agent",
    "cli",
    "mcp",
    "hook",
    "subagent",
    "skill",
    "stream-json",
    "output-format",
    "cursor-agent",
    "--force",
    "--print",
    "cursor.com",
    "cursor api",
    "cursor 命令",
    "cursor 工具",
]


def is_cursor_related(text: str) -> bool:
    """检测文本是否与 Cursor 相关

    通过关键词匹配判断文本是否涉及 Cursor 相关主题，
    用于决定是否自动搜索知识库补充上下文。

    Args:
        text: 要检测的文本

    Returns:
        是否与 Cursor 相关

    Example:
        >>> is_cursor_related("如何使用 cursor agent cli")
        True
        >>> is_cursor_related("普通的 Python 代码问题")
        False
    """
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in CURSOR_KEYWORDS)


# ============================================================================
# 知识库文档数据结构
# ============================================================================


class KnowledgeDoc(BaseModel):
    """知识库文档数据结构

    统一的知识库文档格式，用于协程版与多进程版的知识库上下文传递。

    字段说明:
        必填字段:
            - title: 文档标题
            - url: 文档来源 URL
            - content: 文档内容（可能已截断）
            - score: 相关度分数 (0.0-1.0)
            - source: 来源类型 ("cursor-docs", "cli-ask")

        可选字段:
            - context_used: CLI ask 模式使用的上下文文档 URL 列表
            - query: CLI ask 查询的原始问题
            - truncated: 内容是否已被截断

    来源类型说明:
        - "cursor-docs": 来自 Cursor 官方文档的静态内容
        - "cli-ask": 通过 CLI --mode=ask 查询获得的智能回答

    使用示例:
        # 创建普通文档
        doc = KnowledgeDoc(
            title="MCP 服务器配置",
            url="https://cursor.com/docs/mcp",
            content="MCP 配置说明...",
            score=0.85,
            source="cursor-docs",
        )

        # 创建 CLI ask 结果
        ask_doc = KnowledgeDoc(
            title="CLI Ask 查询结果",
            url="cli-ask://knowledge-query",
            content="根据知识库，MCP 的配置方法是...",
            score=1.0,
            source="cli-ask",
            query="如何配置 MCP 服务器",
            context_used=["https://cursor.com/docs/mcp"],
        )

        # 转换为字典
        doc_dict = doc.model_dump()
    """

    # 必填字段
    title: str = Field(..., description="文档标题")
    url: str = Field(..., description="文档来源 URL")
    content: str = Field(..., description="文档内容（可能已截断）")
    score: float = Field(..., ge=0.0, le=1.0, description="相关度分数 (0.0-1.0)")
    source: str = Field(..., description="来源类型: 'cursor-docs' | 'cli-ask'")

    # 可选字段
    context_used: Optional[list[str]] = Field(
        default=None,
        description="CLI ask 模式使用的上下文文档 URL 列表",
    )
    query: Optional[str] = Field(
        default=None,
        description="CLI ask 查询的原始问题",
    )
    truncated: Optional[bool] = Field(
        default=None,
        description="内容是否已被截断",
    )

    def is_cli_ask_result(self) -> bool:
        """判断是否为 CLI ask 模式的查询结果"""
        return self.source == "cli-ask"

    def is_truncated(self) -> bool:
        """判断内容是否已被截断"""
        return self.truncated is True


# ============================================================================
# Payload 上限常量 - 用于控制传递给 Worker 的知识库文档大小
# ============================================================================

# 普通知识库文档限制
MAX_KNOWLEDGE_DOCS: int = 3  # 最大文档数量
MAX_CHARS_PER_DOC: int = 1200  # 单文档最大字符数
MAX_TOTAL_KNOWLEDGE_CHARS: int = 3000  # 知识库总字符上限

# CLI Ask 模式结果限制（优先级更高，内容更精准）
MAX_CLI_ASK_DOCS: int = 2  # CLI ask 结果最大数量
MAX_CLI_ASK_CHARS_PER_DOC: int = 1500  # CLI ask 单个结果最大字符

# 降级策略参数
FALLBACK_CHARS_PER_DOC: int = 600  # 超限时降级到的单文档字符数
MIN_DOCS_ON_FALLBACK: int = 2  # 降级后最少保留的文档数


def truncate_knowledge_docs(
    docs: list[dict],
    max_docs: int = MAX_KNOWLEDGE_DOCS,
    max_chars_per_doc: int = MAX_CHARS_PER_DOC,
    max_total_chars: int = MAX_TOTAL_KNOWLEDGE_CHARS,
) -> tuple[list[dict], int]:
    """对知识库文档进行截断和降级处理

    降级策略（按优先级执行）：
        1. 限制文档数量（不超过 max_docs）
        2. 截断每个文档内容（不超过 max_chars_per_doc）
        3. 若总字符数仍超限：
           a. 先将每文档截断长度降级到 FALLBACK_CHARS_PER_DOC
           b. 若仍超限，减少文档数量（最少保留 MIN_DOCS_ON_FALLBACK）

    Args:
        docs: 原始知识库文档列表，每个文档包含 content 字段
        max_docs: 最大文档数量
        max_chars_per_doc: 单文档最大字符数
        max_total_chars: 总字符数上限

    Returns:
        tuple: (处理后的文档列表, 实际总字符数)
    """
    if not docs:
        return [], 0

    # Step 1: 限制文档数量
    limited_docs = docs[:max_docs]

    # Step 2: 截断每个文档内容
    truncated_docs = []
    for doc in limited_docs:
        doc_copy = doc.copy()
        content = doc_copy.get("content", "")
        if len(content) > max_chars_per_doc:
            doc_copy["content"] = content[:max_chars_per_doc] + "..."
            doc_copy["truncated"] = True
        truncated_docs.append(doc_copy)

    # Step 3: 计算总字符数
    total_chars = sum(len(d.get("content", "")) for d in truncated_docs)

    # Step 4: 降级策略 - 若总字符数超限
    if total_chars > max_total_chars:
        # 4a: 先降低单文档截断长度
        for doc in truncated_docs:
            content = doc.get("content", "")
            if len(content) > FALLBACK_CHARS_PER_DOC:
                doc["content"] = content[:FALLBACK_CHARS_PER_DOC] + "..."
                doc["truncated"] = True

        total_chars = sum(len(d.get("content", "")) for d in truncated_docs)

        # 4b: 若仍超限，减少文档数量
        while total_chars > max_total_chars and len(truncated_docs) > MIN_DOCS_ON_FALLBACK:
            truncated_docs.pop()
            total_chars = sum(len(d.get("content", "")) for d in truncated_docs)

    return truncated_docs, total_chars
