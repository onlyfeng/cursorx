"""代码索引模块

提供代码库的语义索引和检索功能
"""
from .base import (
    ChunkType,
    CodeChunk,
    CodeChunker,
    EmbeddingModel,
    SearchResult,
    VectorStore,
)
from .chunker import (
    EXTENSION_LANGUAGE_MAP,
    ChunkContext,
    SemanticCodeChunker,
    chunk_file,
    chunk_text,
)
from .config import (
    ChunkConfig,
    ChunkStrategy,
    EmbeddingConfig,
    EmbeddingProvider,
    IndexConfig,
    VectorStoreConfig,
    VectorStoreType,
)
from .embedding import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    EmbeddingCache,
    SentenceTransformerEmbedding,
    create_embedding_model,
    get_available_models,
)
from .indexer import (
    CodebaseIndexer,
    FileState,
    IndexProgress,
    IndexStateManager,
    ProgressCallback,
)
from .search import (
    SearchOptions,
    SearchResultWithContext,
    SearchStats,
    SemanticSearch,
    create_semantic_search,
)
from .vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    ChromaVectorStore,
    create_vector_store,
)


# CLI 入口点（延迟导入避免循环依赖）
def run_cli():
    """运行命令行接口"""
    from .cli import run
    run()

__all__ = [
    # 基类
    "ChunkType",
    "CodeChunk",
    "SearchResult",
    "EmbeddingModel",
    "VectorStore",
    "CodeChunker",
    # 配置
    "EmbeddingProvider",
    "VectorStoreType",
    "ChunkStrategy",
    "EmbeddingConfig",
    "ChunkConfig",
    "VectorStoreConfig",
    "IndexConfig",
    # 嵌入模型实现
    "SentenceTransformerEmbedding",
    "EmbeddingCache",
    "create_embedding_model",
    "get_available_models",
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    # 分块器实现
    "SemanticCodeChunker",
    "ChunkContext",
    "EXTENSION_LANGUAGE_MAP",
    "chunk_file",
    "chunk_text",
    # 向量存储实现
    "ChromaVectorStore",
    "create_vector_store",
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_COLLECTION_NAME",
    # 索引器
    "FileState",
    "IndexProgress",
    "ProgressCallback",
    "IndexStateManager",
    "CodebaseIndexer",
    # 语义搜索
    "SemanticSearch",
    "SearchOptions",
    "SearchResultWithContext",
    "SearchStats",
    "create_semantic_search",
    # CLI
    "run_cli",
]
