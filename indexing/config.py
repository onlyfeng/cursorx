"""索引模块配置定义"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingProvider(str, Enum):
    """Embedding 模型提供商"""

    OPENAI = "openai"  # OpenAI embeddings
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # 本地模型
    HUGGINGFACE = "huggingface"  # HuggingFace API
    CUSTOM = "custom"  # 自定义


class VectorStoreType(str, Enum):
    """向量存储类型"""

    MEMORY = "memory"  # 内存存储
    FAISS = "faiss"  # FAISS
    CHROMADB = "chromadb"  # ChromaDB
    QDRANT = "qdrant"  # Qdrant
    CUSTOM = "custom"  # 自定义


class ChunkStrategy(str, Enum):
    """代码分块策略"""

    FIXED_SIZE = "fixed_size"  # 固定大小分块
    AST_BASED = "ast_based"  # 基于 AST 的分块
    SEMANTIC = "semantic"  # 语义分块
    HYBRID = "hybrid"  # 混合策略


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置

    配置用于生成向量嵌入的模型参数

    Note:
        默认值与 core.config.IndexingConfig 保持同步，确保一致性。
        优先使用本地 sentence-transformers 模型，与 config.yaml 中的配置对齐。
    """

    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"  # 模型名称（与 core.config.IndexingConfig.model 对齐）
    dimension: int = 384  # 向量维度（all-MiniLM-L6-v2 输出维度）
    batch_size: int = 100  # 批量处理大小

    # API 配置（用于远程服务）
    api_key: Optional[str] = None  # API 密钥
    api_base: Optional[str] = None  # API 基础 URL

    # 本地模型配置
    model_path: Optional[str] = None  # 本地模型路径
    device: str = "cpu"  # 运行设备（cpu/cuda）

    # 性能配置
    max_retries: int = 3  # 最大重试次数
    timeout: int = 30  # 超时时间（秒）

    model_config = ConfigDict(use_enum_values=True)


class ChunkConfig(BaseModel):
    """代码分块配置

    配置代码分块的策略和参数

    Note:
        默认值与 core.config.IndexingConfig 保持同步，确保一致性。
        chunk_size 和 chunk_overlap 与 config.yaml 中的 indexing.* 配置对齐。
    """

    strategy: ChunkStrategy = ChunkStrategy.AST_BASED

    # 分块大小控制（默认值与 core.config.IndexingConfig 对齐）
    chunk_size: int = 500  # 目标分块大小（字符数）
    chunk_overlap: int = 50  # 分块重叠大小
    min_chunk_size: int = 100  # 最小分块大小
    max_chunk_size: int = 3000  # 最大分块大小

    # AST 分块配置
    split_functions: bool = True  # 是否按函数分块
    split_classes: bool = True  # 是否按类分块
    include_imports: bool = True  # 是否在分块中包含导入语句
    include_docstrings: bool = True  # 是否包含文档字符串

    # 语言特定配置
    supported_languages: list[str] = Field(default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"])

    model_config = ConfigDict(use_enum_values=True)


class VectorStoreConfig(BaseModel):
    """向量存储配置

    配置向量数据库的连接和存储参数

    Note:
        默认值与 core.config.IndexingConfig 保持同步，确保一致性。
        默认使用 ChromaDB 持久化存储，路径与 config.yaml 中的 indexing.persist_path 对齐。
    """

    store_type: VectorStoreType = VectorStoreType.CHROMADB

    # 存储路径（用于持久化存储，与 core.config.IndexingConfig.persist_path 对齐）
    persist_directory: str = ".cursor/vector_index/"  # 持久化目录
    collection_name: str = "code_index"  # 集合名称

    # 连接配置（用于远程存储）
    host: Optional[str] = None  # 服务器地址
    port: Optional[int] = None  # 端口
    api_key: Optional[str] = None  # API 密钥

    # 索引配置
    metric: str = "cosine"  # 相似度度量（cosine/euclidean/dot）
    ef_construction: int = 200  # HNSW 构建参数
    ef_search: int = 50  # HNSW 搜索参数
    m: int = 16  # HNSW 连接数

    model_config = ConfigDict(use_enum_values=True)


class IndexConfig(BaseModel):
    """索引总配置

    整合所有索引相关的配置

    Note:
        默认值与 core.config.IndexingConfig 和 config.yaml 保持同步，确保一致性。
        如需修改默认值，请同时更新 core/config.py 中的 IndexingConfig 和 config.yaml。
    """

    # 子配置
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    # 索引元数据
    index_name: str = "default"  # 索引名称
    description: str = ""  # 索引描述

    # 文件过滤（与 core.config.IndexingConfig 对齐）
    include_patterns: list[str] = Field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"])
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
        ]
    )

    # 并发配置
    max_workers: int = 4  # 最大并发数

    # 增量索引
    incremental: bool = True  # 是否启用增量索引

    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        return self.embedding.dimension

    model_config = ConfigDict(use_enum_values=True)


def normalize_indexing_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """标准化索引配置键名

    将 config.yaml 中的新键名映射到内部使用的旧键名，同时保持向后兼容

    键名映射规则:
        - model → embedding_model (优先使用 model，回退到 embedding_model)
        - persist_path → persist_dir (优先使用 persist_path，回退到 persist_dir)

    Args:
        raw_config: 原始的 indexing 配置字典

    Returns:
        标准化后的配置字典
    """
    if not raw_config:
        return {}

    normalized = dict(raw_config)

    # model / embedding_model 映射: 优先使用 model，回退到 embedding_model
    if "model" in normalized:
        normalized["embedding_model"] = normalized.pop("model")
    # 如果两者都存在，model 优先

    # persist_path / persist_dir 映射: 优先使用 persist_path，回退到 persist_dir
    if "persist_path" in normalized:
        normalized["persist_dir"] = normalized.pop("persist_path")
    # 如果两者都存在，persist_path 优先

    return normalized


def extract_search_options(raw_config: dict[str, Any]) -> dict[str, Any]:
    """从配置中提取搜索选项

    提取 indexing.search 子配置项

    Args:
        raw_config: 原始的 indexing 配置字典

    Returns:
        搜索选项字典，包含 top_k, min_score, include_context, context_lines 等
        如果原始配置中没有 search 字段则返回空字典
    """
    if not raw_config:
        return {}

    # 检查 search 字段是否存在（包括空字典的情况）
    if "search" not in raw_config:
        return {}

    search_config = raw_config.get("search") or {}

    return {
        "top_k": search_config.get("top_k", 10),
        "min_score": search_config.get("min_score", 0.3),
        "include_context": search_config.get("include_context", True),
        "context_lines": search_config.get("context_lines", 3),
    }
