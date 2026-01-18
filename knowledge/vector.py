"""向量存储和语义搜索

提供文档的向量化存储和语义搜索功能。

特性：
- 文档向量化和索引
- 语义相似度搜索
- 混合搜索（关键词 + 语义）
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from .models import Document, DocumentChunk


@dataclass
class KnowledgeVectorConfig:
    """向量搜索配置

    配置向量存储和语义搜索的参数。
    """
    # 是否启用向量搜索
    enabled: bool = True

    # 向量模型配置
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # 存储配置
    vector_storage_path: str = ".cursor/knowledge/vectors"

    # 索引配置
    index_type: str = "flat"  # flat, hnsw, ivf
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product

    # 搜索配置
    default_top_k: int = 10
    min_similarity_score: float = 0.5

    # 分块配置（用于向量化）
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 混合搜索权重
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "vector_storage_path": self.vector_storage_path,
            "index_type": self.index_type,
            "similarity_metric": self.similarity_metric,
            "default_top_k": self.default_top_k,
            "min_similarity_score": self.min_similarity_score,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeVectorConfig":
        """从字典创建"""
        return cls(
            enabled=data.get("enabled", True),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            embedding_dimension=data.get("embedding_dimension", 1536),
            vector_storage_path=data.get("vector_storage_path", ".cursor/knowledge/vectors"),
            index_type=data.get("index_type", "flat"),
            similarity_metric=data.get("similarity_metric", "cosine"),
            default_top_k=data.get("default_top_k", 10),
            min_similarity_score=data.get("min_similarity_score", 0.5),
            chunk_size=data.get("chunk_size", 512),
            chunk_overlap=data.get("chunk_overlap", 50),
            semantic_weight=data.get("semantic_weight", 0.7),
            keyword_weight=data.get("keyword_weight", 0.3),
        )


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    doc_id: str
    chunk_id: str
    url: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeVectorStore:
    """向量存储管理器

    管理文档的向量化存储和索引。

    使用示例:
    ```python
    config = KnowledgeVectorConfig()
    store = KnowledgeVectorStore(config)
    await store.initialize()

    # 索引文档
    await store.index_document(doc)

    # 更新文档
    await store.update_document(doc)

    # 删除文档
    await store.delete_document(doc_id)
    ```
    """

    def __init__(self, config: Optional[KnowledgeVectorConfig] = None):
        """初始化向量存储

        Args:
            config: 向量配置
        """
        self.config = config or KnowledgeVectorConfig()
        self._initialized = False
        self._lock = asyncio.Lock()

        # 向量索引（内存中）
        # 结构: {doc_id: {chunk_id: vector}}
        self._vectors: dict[str, dict[str, list[float]]] = {}

        # 文档元数据
        # 结构: {doc_id: {chunk_id: chunk_info}}
        self._chunk_info: dict[str, dict[str, dict[str, Any]]] = {}

    async def initialize(self) -> None:
        """初始化向量存储"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # TODO: 加载持久化的向量索引
            self._initialized = True
            logger.info("KnowledgeVectorStore 初始化完成")

    async def index_document(self, doc: Document) -> bool:
        """索引文档

        为文档生成向量并添加到索引中。

        Args:
            doc: 要索引的文档

        Returns:
            是否索引成功
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                # 如果文档已有分块，使用现有分块
                chunks = doc.chunks if doc.chunks else self._create_chunks(doc)

                # 为每个分块生成向量
                self._vectors[doc.id] = {}
                self._chunk_info[doc.id] = {}

                for chunk in chunks:
                    # TODO: 调用嵌入模型生成向量
                    # 目前使用占位向量
                    vector = await self._generate_embedding(chunk.content)

                    self._vectors[doc.id][chunk.chunk_id] = vector
                    self._chunk_info[doc.id][chunk.chunk_id] = {
                        "content": chunk.content,
                        "url": doc.url,
                        "title": doc.title,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "metadata": chunk.metadata,
                    }

                logger.debug(f"文档索引成功: {doc.id}, 分块数: {len(chunks)}")
                return True

            except Exception as e:
                logger.error(f"索引文档失败 {doc.id}: {e}")
                return False

    async def update_document(self, doc: Document) -> bool:
        """更新文档索引

        删除旧索引并重新索引文档。

        Args:
            doc: 要更新的文档

        Returns:
            是否更新成功
        """
        # 先删除旧索引
        await self.delete_document(doc.id)
        # 重新索引
        return await self.index_document(doc)

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档索引

        Args:
            doc_id: 文档 ID

        Returns:
            是否删除成功
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                if doc_id in self._vectors:
                    del self._vectors[doc_id]
                if doc_id in self._chunk_info:
                    del self._chunk_info[doc_id]

                logger.debug(f"文档索引已删除: {doc_id}")
                return True

            except Exception as e:
                logger.error(f"删除文档索引失败 {doc_id}: {e}")
                return False

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        """向量相似度搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            min_score: 最小相似度分数

        Returns:
            搜索结果列表
        """
        if not self._initialized:
            await self.initialize()

        top_k = top_k or self.config.default_top_k
        min_score = min_score if min_score is not None else self.config.min_similarity_score

        # 生成查询向量
        query_vector = await self._generate_embedding(query)

        results: list[VectorSearchResult] = []

        # 遍历所有向量计算相似度
        for doc_id, chunks in self._vectors.items():
            for chunk_id, vector in chunks.items():
                score = self._compute_similarity(query_vector, vector)

                if score >= min_score:
                    chunk_info = self._chunk_info[doc_id][chunk_id]
                    results.append(VectorSearchResult(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        url=chunk_info["url"],
                        title=chunk_info["title"],
                        content=chunk_info["content"],
                        score=score,
                        metadata=chunk_info.get("metadata", {}),
                    ))

        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]

    async def rebuild_all(self, documents: list[Document]) -> int:
        """重建所有文档索引

        Args:
            documents: 文档列表

        Returns:
            成功索引的文档数量
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            # 清空现有索引
            self._vectors.clear()
            self._chunk_info.clear()

        # 重新索引所有文档
        success_count = 0
        for doc in documents:
            if await self.index_document(doc):
                success_count += 1

        logger.info(f"向量索引重建完成: {success_count}/{len(documents)} 成功")
        return success_count

    def _create_chunks(self, doc: Document) -> list[DocumentChunk]:
        """为文档创建分块"""
        chunks: list[DocumentChunk] = []
        content = doc.content
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # 尝试在句子边界处分割
            if end < len(content):
                # 查找最近的句子结束符
                for sep in ['\n\n', '。', '.', '\n', ' ']:
                    last_sep = content.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc.id}_chunk_{chunk_index}",
                    doc_id=doc.id,
                    content=chunk_content,
                    start_index=start,
                    end_index=end,
                ))
                chunk_index += 1

            start = end - overlap if end < len(content) else len(content)

        return chunks

    async def _generate_embedding(self, text: str) -> list[float]:
        """生成文本向量

        TODO: 集成实际的嵌入模型（如 OpenAI text-embedding-3-small）
        当前返回基于文本哈希的模拟向量
        """
        import hashlib

        # 模拟向量生成（实际应调用嵌入 API）
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # 生成固定维度的伪向量
        vector = []
        for i in range(0, min(len(text_hash), self.config.embedding_dimension * 2), 2):
            value = int(text_hash[i:i+2], 16) / 255.0 - 0.5
            vector.append(value)

        # 填充到目标维度
        while len(vector) < self.config.embedding_dimension:
            vector.append(0.0)

        return vector[:self.config.embedding_dimension]

    def _compute_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """计算向量相似度"""
        if self.config.similarity_metric == "cosine":
            return self._cosine_similarity(vec1, vec2)
        elif self.config.similarity_metric == "euclidean":
            return self._euclidean_similarity(vec1, vec2)
        elif self.config.similarity_metric == "dot_product":
            return self._dot_product(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """余弦相似度"""
        import math

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _euclidean_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """欧氏距离相似度（转换为 0-1 范围）"""
        import math

        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        return 1.0 / (1.0 + distance)

    def _dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        """点积"""
        return sum(a * b for a, b in zip(vec1, vec2))

    @property
    def document_count(self) -> int:
        """已索引的文档数量"""
        return len(self._vectors)

    @property
    def chunk_count(self) -> int:
        """已索引的分块数量"""
        return sum(len(chunks) for chunks in self._vectors.values())


class KnowledgeSemanticSearch:
    """语义搜索引擎

    提供基于向量的语义搜索和混合搜索功能。

    使用示例:
    ```python
    config = KnowledgeVectorConfig()
    vector_store = KnowledgeVectorStore(config)
    search = KnowledgeSemanticSearch(vector_store, config)

    # 语义搜索
    results = await search.semantic_search("查询内容")

    # 混合搜索
    results = await search.hybrid_search("查询内容", keyword_results=[...])
    ```
    """

    def __init__(
        self,
        vector_store: KnowledgeVectorStore,
        config: Optional[KnowledgeVectorConfig] = None,
    ):
        """初始化语义搜索

        Args:
            vector_store: 向量存储
            config: 向量配置
        """
        self.vector_store = vector_store
        self.config = config or KnowledgeVectorConfig()

    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        """纯语义搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            min_score: 最小相似度分数

        Returns:
            搜索结果列表
        """
        return await self.vector_store.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    async def hybrid_search(
        self,
        query: str,
        keyword_results: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[VectorSearchResult]:
        """混合搜索（关键词 + 语义）

        结合关键词搜索和语义搜索的结果。

        Args:
            query: 搜索查询
            keyword_results: 关键词搜索结果列表
                             格式: [{"doc_id": str, "score": float, ...}, ...]
            top_k: 返回结果数量

        Returns:
            合并后的搜索结果列表
        """
        top_k = top_k or self.config.default_top_k

        # 获取语义搜索结果
        semantic_results = await self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # 获取更多结果用于合并
        )

        # 构建文档分数映射
        doc_scores: dict[str, dict[str, Any]] = {}

        # 添加关键词搜索结果
        for kr in keyword_results:
            doc_id = kr.get("doc_id", "")
            if doc_id:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                        "result": kr,
                    }
                doc_scores[doc_id]["keyword_score"] = kr.get("score", 0.0)

        # 添加语义搜索结果
        for sr in semantic_results:
            if sr.doc_id not in doc_scores:
                doc_scores[sr.doc_id] = {
                    "keyword_score": 0.0,
                    "semantic_score": 0.0,
                    "result": sr,
                }
            doc_scores[sr.doc_id]["semantic_score"] = max(
                doc_scores[sr.doc_id]["semantic_score"],
                sr.score
            )
            # 如果语义结果更详细，更新结果
            if isinstance(doc_scores[sr.doc_id]["result"], dict):
                doc_scores[sr.doc_id]["result"] = sr

        # 计算混合分数
        results: list[VectorSearchResult] = []
        for doc_id, scores in doc_scores.items():
            keyword_weight = self.config.keyword_weight
            semantic_weight = self.config.semantic_weight

            # 归一化分数
            keyword_normalized = min(scores["keyword_score"] / 10.0, 1.0)  # 假设关键词分数上限为 10
            semantic_normalized = scores["semantic_score"]

            hybrid_score = (
                keyword_weight * keyword_normalized +
                semantic_weight * semantic_normalized
            )

            result = scores["result"]
            if isinstance(result, VectorSearchResult):
                result.score = hybrid_score
                results.append(result)
            else:
                # 从关键词结果创建 VectorSearchResult
                results.append(VectorSearchResult(
                    doc_id=doc_id,
                    chunk_id="",
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    content=result.get("snippet", ""),
                    score=hybrid_score,
                    metadata={},
                ))

        # 按混合分数排序
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]
