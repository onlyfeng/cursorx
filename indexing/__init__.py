"""代码索引模块

提供代码库的语义索引和检索功能
"""
from .base import (
    ChunkType,
    CodeChunk,
    SearchResult,
    EmbeddingModel,
    VectorStore,
    CodeChunker,
)
from .config import (
    EmbeddingProvider,
    VectorStoreType,
    ChunkStrategy,
    EmbeddingConfig,
    ChunkConfig,
    VectorStoreConfig,
    IndexConfig,
)
from .embedding import (
    SentenceTransformerEmbedding,
    EmbeddingCache,
    create_embedding_model,
    get_available_models,
    MODEL_CONFIGS,
    DEFAULT_MODEL,
)
from .chunker import (
    SemanticCodeChunker,
    ChunkContext,
    EXTENSION_LANGUAGE_MAP,
    chunk_file,
    chunk_text,
)
from .vector_store import (
    ChromaVectorStore,
    create_vector_store,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
)
from .indexer import (
    FileState,
    IndexProgress,
    ProgressCallback,
    IndexStateManager,
    CodebaseIndexer,
)
from .search import (
    SemanticSearch,
    SearchOptions,
    SearchResultWithContext,
    SearchStats,
    create_semantic_search,
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
