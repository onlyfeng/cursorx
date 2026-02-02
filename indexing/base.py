"""索引模块抽象基类定义"""

import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """代码分块类型"""

    FUNCTION = "function"  # 函数
    CLASS = "class"  # 类
    METHOD = "method"  # 方法
    MODULE = "module"  # 模块级代码
    IMPORT = "import"  # 导入语句
    COMMENT = "comment"  # 注释块
    UNKNOWN = "unknown"  # 未知类型


class CodeChunk(BaseModel):
    """代码分块

    表示一个代码片段及其元数据
    """

    chunk_id: str = Field(default_factory=lambda: f"chunk-{uuid.uuid4().hex[:8]}")
    content: str  # 代码内容

    # 位置信息
    file_path: str  # 文件路径
    start_line: int = 0  # 起始行号
    end_line: int = 0  # 结束行号

    # 类型信息
    chunk_type: ChunkType = ChunkType.UNKNOWN  # 分块类型
    language: str = "unknown"  # 编程语言

    # 语义信息
    name: Optional[str] = None  # 名称（函数名/类名等）
    parent_name: Optional[str] = None  # 父级名称（类名等）
    signature: Optional[str] = None  # 签名
    docstring: Optional[str] = None  # 文档字符串

    # 向量嵌入
    embedding: Optional[list[float]] = None  # 向量嵌入

    # 元数据
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_embedding(self) -> bool:
        """是否已生成向量嵌入"""
        return self.embedding is not None and len(self.embedding) > 0

    def get_display_name(self) -> str:
        """获取显示名称"""
        if self.name:
            if self.parent_name:
                return f"{self.parent_name}.{self.name}"
            return self.name
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class SearchResult(BaseModel):
    """搜索结果"""

    chunk: CodeChunk  # 匹配的代码分块
    score: float  # 相似度分数
    rank: int = 0  # 排名
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingModel(ABC):
    """Embedding 模型抽象基类

    定义生成文本向量嵌入的接口
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度

        Returns:
            向量维度
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称

        Returns:
            模型名称
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            向量嵌入
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入

        Args:
            texts: 输入文本列表

        Returns:
            向量嵌入列表
        """
        pass

    async def embed_chunk(self, chunk: CodeChunk) -> CodeChunk:
        """为代码分块生成向量嵌入

        Args:
            chunk: 代码分块

        Returns:
            带有向量嵌入的代码分块
        """
        embedding = await self.embed_text(chunk.content)
        chunk.embedding = embedding
        return chunk

    async def embed_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """批量为代码分块生成向量嵌入

        Args:
            chunks: 代码分块列表

        Returns:
            带有向量嵌入的代码分块列表
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_batch(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks


class VectorStore(ABC):
    """向量存储抽象基类

    定义向量数据库的操作接口
    """

    @abstractmethod
    async def add(self, chunks: list[CodeChunk]) -> list[str]:
        """添加代码分块到存储

        Args:
            chunks: 代码分块列表（需要已包含向量嵌入）

        Returns:
            添加的分块 ID 列表
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """搜索相似的代码分块

        Args:
            query_embedding: 查询向量
            top_k: 返回的最大结果数
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """删除代码分块

        Args:
            chunk_ids: 要删除的分块 ID 列表

        Returns:
            实际删除的数量
        """
        pass

    @abstractmethod
    async def persist(self) -> None:
        """持久化存储

        将内存中的数据保存到持久化存储
        """
        pass

    @abstractmethod
    async def load(self) -> None:
        """加载存储

        从持久化存储加载数据
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空存储"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """获取存储的分块数量

        Returns:
            分块数量
        """
        pass

    async def search_by_text(
        self,
        query_text: str,
        embedding_model: EmbeddingModel,
        top_k: int = 10,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """通过文本搜索相似的代码分块

        Args:
            query_text: 查询文本
            embedding_model: Embedding 模型
            top_k: 返回的最大结果数
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        query_embedding = await embedding_model.embed_text(query_text)
        return await self.search(query_embedding, top_k, filter_dict)


class CodeChunker(ABC):
    """代码分块器抽象基类

    定义将代码文件分割成语义块的接口
    """

    @abstractmethod
    async def chunk_file(self, file_path: str) -> list[CodeChunk]:
        """分块单个代码文件

        Args:
            file_path: 文件路径

        Returns:
            代码分块列表
        """
        pass

    @abstractmethod
    async def chunk_text(
        self, text: str, file_path: str = "<unknown>", language: Optional[str] = None
    ) -> list[CodeChunk]:
        """分块代码文本

        Args:
            text: 代码文本
            file_path: 文件路径（用于元数据）
            language: 编程语言（可选，会自动检测）

        Returns:
            代码分块列表
        """
        pass

    @abstractmethod
    def detect_language(self, file_path: str) -> str:
        """检测文件的编程语言

        Args:
            file_path: 文件路径

        Returns:
            编程语言标识符
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """获取支持的编程语言列表

        Returns:
            编程语言标识符列表
        """
        pass
