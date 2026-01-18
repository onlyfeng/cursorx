"""知识库模块

提供文档管理、分块、向量化和检索功能
"""
from .models import (
    # 枚举类型
    FetchStatus,
    FetchPriority,
    # 数据模型
    DocumentChunk,
    Document,
    KnowledgeBase,
    KnowledgeBaseStats,
    FetchTask,
)
from .parser import (
    # 数据结构
    ParsedContent,
    # 解析器
    HTMLParser,
    ContentCleaner,
    MarkdownConverter,
    ChunkSplitter,
)
from .storage import (
    IndexEntry,
    StorageConfig,
    SearchResult,
    KnowledgeStorage,
)
from .fetcher import (
    WebFetcher,
    FetchConfig,
    FetchResult,
    FetchMethod,
    ContentFormat,
    fetch_url,
    fetch_urls,
)
from .vector import (
    KnowledgeVectorConfig,
    KnowledgeVectorStore,
    VectorSearchResult,
)
from .vector_store import (
    KnowledgeVectorStore as ChromaVectorStore,
    VectorSearchResult as ChromaVectorSearchResult,
)
from .vector_adapter import (
    DocumentChunkConfig,
    DocumentChunkAdapter,
    DocumentTextChunker,
)
from .semantic_search import (
    KnowledgeSemanticSearch,
    HybridSearchConfig,
)
from .manager import KnowledgeManager, AskResult

__all__ = [
    # 枚举类型
    "FetchStatus",
    "FetchPriority",
    "FetchMethod",
    "ContentFormat",
    # 数据模型
    "DocumentChunk",
    "Document",
    "KnowledgeBase",
    "KnowledgeBaseStats",
    "FetchTask",
    # 解析器
    "ParsedContent",
    "HTMLParser",
    "ContentCleaner",
    "MarkdownConverter",
    "ChunkSplitter",
    # 存储管理
    "IndexEntry",
    "StorageConfig",
    "SearchResult",
    "KnowledgeStorage",
    # 获取器
    "WebFetcher",
    "FetchConfig",
    "FetchResult",
    "fetch_url",
    "fetch_urls",
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
