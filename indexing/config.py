"""索引模块配置定义"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class EmbeddingProvider(str, Enum):
    """Embedding 模型提供商"""
    OPENAI = "openai"                    # OpenAI embeddings
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # 本地模型
    HUGGINGFACE = "huggingface"          # HuggingFace API
    CUSTOM = "custom"                    # 自定义


class VectorStoreType(str, Enum):
    """向量存储类型"""
    MEMORY = "memory"                    # 内存存储
    FAISS = "faiss"                      # FAISS
    CHROMADB = "chromadb"                # ChromaDB
    QDRANT = "qdrant"                    # Qdrant
    CUSTOM = "custom"                    # 自定义


class ChunkStrategy(str, Enum):
    """代码分块策略"""
    FIXED_SIZE = "fixed_size"            # 固定大小分块
    AST_BASED = "ast_based"              # 基于 AST 的分块
    SEMANTIC = "semantic"                # 语义分块
    HYBRID = "hybrid"                    # 混合策略


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置
    
    配置用于生成向量嵌入的模型参数
    """
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model_name: str = "text-embedding-3-small"  # 模型名称
    dimension: int = 1536                       # 向量维度
    batch_size: int = 100                       # 批量处理大小
    
    # API 配置（用于远程服务）
    api_key: Optional[str] = None               # API 密钥
    api_base: Optional[str] = None              # API 基础 URL
    
    # 本地模型配置
    model_path: Optional[str] = None            # 本地模型路径
    device: str = "cpu"                         # 运行设备（cpu/cuda）
    
    # 性能配置
    max_retries: int = 3                        # 最大重试次数
    timeout: int = 30                           # 超时时间（秒）

    model_config = ConfigDict(use_enum_values=True)


class ChunkConfig(BaseModel):
    """代码分块配置
    
    配置代码分块的策略和参数
    """
    strategy: ChunkStrategy = ChunkStrategy.AST_BASED
    
    # 分块大小控制
    chunk_size: int = 1500                      # 目标分块大小（字符数）
    chunk_overlap: int = 200                    # 分块重叠大小
    min_chunk_size: int = 100                   # 最小分块大小
    max_chunk_size: int = 3000                  # 最大分块大小
    
    # AST 分块配置
    split_functions: bool = True                # 是否按函数分块
    split_classes: bool = True                  # 是否按类分块
    include_imports: bool = True                # 是否在分块中包含导入语句
    include_docstrings: bool = True             # 是否包含文档字符串
    
    # 语言特定配置
    supported_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"]
    )

    model_config = ConfigDict(use_enum_values=True)


class VectorStoreConfig(BaseModel):
    """向量存储配置
    
    配置向量数据库的连接和存储参数
    """
    store_type: VectorStoreType = VectorStoreType.MEMORY
    
    # 存储路径（用于持久化存储）
    persist_directory: Optional[str] = None     # 持久化目录
    collection_name: str = "code_index"         # 集合名称
    
    # 连接配置（用于远程存储）
    host: Optional[str] = None                  # 服务器地址
    port: Optional[int] = None                  # 端口
    api_key: Optional[str] = None               # API 密钥
    
    # 索引配置
    metric: str = "cosine"                      # 相似度度量（cosine/euclidean/dot）
    ef_construction: int = 200                  # HNSW 构建参数
    ef_search: int = 50                         # HNSW 搜索参数
    m: int = 16                                 # HNSW 连接数

    model_config = ConfigDict(use_enum_values=True)


class IndexConfig(BaseModel):
    """索引总配置
    
    整合所有索引相关的配置
    """
    # 子配置
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # 索引元数据
    index_name: str = "default"                 # 索引名称
    description: str = ""                       # 索引描述
    
    # 文件过滤
    include_patterns: list[str] = Field(
        default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"]
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**"
        ]
    )
    
    # 并发配置
    max_workers: int = 4                        # 最大并发数
    
    # 增量索引
    incremental: bool = True                    # 是否启用增量索引
    
    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        return self.embedding.dimension

    model_config = ConfigDict(use_enum_values=True)
