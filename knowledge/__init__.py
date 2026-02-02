"""知识库模块

提供文档管理、分块、向量化和检索功能
"""

from .doc_sources import (
    DEFAULT_ALLOWED_DOC_URL_PREFIXES,
    DEFAULT_DOC_SOURCE_FILES,
    is_valid_doc_url,
    load_core_docs,
    load_core_docs_with_fallback,
    parse_url_list_file,
)
from .doc_url_strategy import (
    VALID_EXECUTION_MODES,
    VALID_EXTERNAL_LINK_MODES,
    DocURLStrategyConfig,
    # 外链白名单校验
    ExternalLinkAllowlist,
    FetchPolicyResult,
    apply_fetch_policy,
    deduplicate_urls,
    # 外链判定与策略
    derive_primary_domains,
    extract_domain,
    filter_urls_by_keywords,
    is_allowed_doc_url,
    is_external_link,
    # 校验函数（契约级公共 API）
    is_full_url_prefix,
    is_path_prefix,
    is_valid_execution_mode,
    is_valid_external_link_mode,
    normalize_url,
    parse_llms_txt_urls,
    # 测试辅助（重置 deprecated 警告状态）
    reset_deprecated_func_warnings,
    select_urls_to_fetch,
    validate_execution_mode,
    validate_external_link_allowlist,
    validate_external_link_mode,
    validate_fetch_policy_path_prefixes,
    # [DEPRECATED] 兼容别名，将在 v2.0 移除
    # 新名: validate_fetch_policy_path_prefixes
    validate_fetch_policy_prefixes,
    validate_url_strategy_prefixes,
)
from .fetcher import (
    DEFAULT_URL_POLICY,
    ContentFormat,
    FetchConfig,
    FetchMethod,
    FetchResult,
    UrlPolicy,
    UrlPolicyError,
    UrlRejectionReason,
    WebFetcher,
    fetch_url,
    fetch_urls,
    sanitize_url_for_log,
)
from .manager import AskResult, KnowledgeManager
from .models import (
    Document,
    # 数据模型
    DocumentChunk,
    FetchPriority,
    # 枚举类型
    FetchStatus,
    FetchTask,
    KnowledgeBase,
    KnowledgeBaseStats,
)
from .parser import (
    ChunkSplitter,
    # 内容清洗
    CleanedContent,
    ContentCleaner,
    ContentCleanMode,
    # 解析器
    HTMLParser,
    MarkdownConverter,
    # 数据结构
    ParsedContent,
    # 清洗函数
    clean_content_unified,
    compute_content_fingerprint,
)
from .semantic_search import (
    HybridSearchConfig,
    KnowledgeSemanticSearch,
)
from .storage import (
    IndexEntry,
    KnowledgeStorage,
    ReadOnlyStorageError,
    SearchResult,
    StorageConfig,
)
from .vector import (
    KnowledgeVectorConfig,
    KnowledgeVectorStore,
    VectorSearchResult,
)
from .vector_adapter import (
    DocumentChunkAdapter,
    DocumentChunkConfig,
    DocumentTextChunker,
)
from .vector_store import (
    KnowledgeVectorStore as ChromaVectorStore,
)
from .vector_store import (
    VectorSearchResult as ChromaVectorSearchResult,
)

__all__ = [
    # 枚举类型
    "FetchStatus",
    "FetchPriority",
    "FetchMethod",
    "ContentFormat",
    "ContentCleanMode",
    # 数据模型
    "DocumentChunk",
    "Document",
    "KnowledgeBase",
    "KnowledgeBaseStats",
    "FetchTask",
    "CleanedContent",
    # 解析器
    "ParsedContent",
    "HTMLParser",
    "ContentCleaner",
    "MarkdownConverter",
    "ChunkSplitter",
    # 清洗函数
    "clean_content_unified",
    "compute_content_fingerprint",
    # 存储管理
    "IndexEntry",
    "StorageConfig",
    "SearchResult",
    "KnowledgeStorage",
    "ReadOnlyStorageError",
    # 获取器
    "WebFetcher",
    "FetchConfig",
    "FetchResult",
    "fetch_url",
    "fetch_urls",
    # URL 安全策略
    "UrlPolicy",
    "UrlPolicyError",
    "UrlRejectionReason",
    "DEFAULT_URL_POLICY",
    "sanitize_url_for_log",
    # 向量搜索
    "KnowledgeVectorConfig",
    "KnowledgeVectorStore",
    "VectorSearchResult",
    "ChromaVectorStore",
    "ChromaVectorSearchResult",
    # 向量适配器
    "DocumentChunkConfig",
    "DocumentChunkAdapter",
    "DocumentTextChunker",
    # 语义搜索
    "KnowledgeSemanticSearch",
    "HybridSearchConfig",
    # 知识库管理器
    "KnowledgeManager",
    "AskResult",
    # 便捷函数
    "semantic_search",
    # 文档源加载 (doc_sources)
    "DEFAULT_DOC_SOURCE_FILES",
    "DEFAULT_ALLOWED_DOC_URL_PREFIXES",
    "parse_url_list_file",
    "is_valid_doc_url",
    "load_core_docs",
    "load_core_docs_with_fallback",
    # URL 策略 (doc_url_strategy)
    "DocURLStrategyConfig",
    "normalize_url",
    "is_allowed_doc_url",
    "deduplicate_urls",
    "parse_llms_txt_urls",
    "select_urls_to_fetch",
    "filter_urls_by_keywords",
    "extract_domain",
    # 外链判定与策略
    "derive_primary_domains",
    "is_external_link",
    "FetchPolicyResult",
    "apply_fetch_policy",
    # 校验函数（契约级公共 API）
    "is_full_url_prefix",
    "is_path_prefix",
    "is_valid_execution_mode",
    "is_valid_external_link_mode",
    "validate_execution_mode",
    "validate_external_link_mode",
    "validate_fetch_policy_path_prefixes",
    "validate_url_strategy_prefixes",
    "VALID_EXECUTION_MODES",
    "VALID_EXTERNAL_LINK_MODES",
    # 外链白名单校验
    "ExternalLinkAllowlist",
    "validate_external_link_allowlist",
    # 测试辅助
    "reset_deprecated_func_warnings",
    # [DEPRECATED] 兼容别名，将在 v2.0 移除
    # 新名: validate_fetch_policy_path_prefixes
    "validate_fetch_policy_prefixes",
]


async def semantic_search(
    query: str,
    knowledge_base: KnowledgeManager,
    max_results: int = 10,
    min_score: float = 0.3,
    search_mode: str = "hybrid",
) -> list[SearchResult]:
    """模块级别的语义搜索便捷函数

    提供简单的接口来执行语义搜索，无需直接操作底层组件。

    Args:
        query: 搜索查询文本
        knowledge_base: KnowledgeManager 实例
        max_results: 最大返回结果数，默认 10
        min_score: 最小相似度分数，默认 0.3
        search_mode: 搜索模式，可选 'keyword', 'semantic', 'hybrid'，默认 'hybrid'

    Returns:
        按相似度降序排列的搜索结果列表

    Example:
        ```python
        from knowledge import KnowledgeManager, semantic_search

        manager = KnowledgeManager()
        await manager.initialize()
        await manager.add_url("https://example.com")

        results = await semantic_search("查询内容", manager)
        for result in results:
            print(f"{result.title}: {result.score}")
        ```
    """
    return await knowledge_base.search(
        query=query,
        max_results=max_results,
        min_score=min_score,
        search_mode=search_mode,  # type: ignore
    )
