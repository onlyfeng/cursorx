"""知识库向量存储实现

基于 ChromaDB 和 SentenceTransformers 的向量存储实现，
提供文档的向量化索引和语义搜索功能。

特性：
- 使用 ChromaDB 进行向量持久化存储
- 使用 SentenceTransformers 生成高质量嵌入
- 支持批量索引和增量更新
- 独立的知识库集合命名空间
- 增量索引：基于内容哈希避免重复索引
- 索引状态跟踪：记录索引进度和统计信息
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger

from .models import Document, DocumentChunk
from .vector import KnowledgeVectorConfig

if TYPE_CHECKING:
    from indexing.embedding import EmbeddingCache as EmbeddingCacheType
    from indexing.embedding import SentenceTransformerEmbedding as SentenceTransformerEmbeddingType
    from indexing.vector_store import ChromaVectorStore as ChromaVectorStoreType

# 为单元测试提供可 patch 的模块级依赖占位符
# 测试会 patch `knowledge.vector_store.ChromaVectorStore`
ChromaVectorStore: type["ChromaVectorStoreType"] | None = None
SentenceTransformerEmbedding: type["SentenceTransformerEmbeddingType"] | None = None
EmbeddingCache: type["EmbeddingCacheType"] | None = None


# 类型别名：进度回调函数
ProgressCallback = Callable[[int, int, Optional[str]], None]


# 知识库专用集合名称
KNOWLEDGE_COLLECTION_NAME = "knowledge_docs"

# 嵌入缓存持久化目录
EMBEDDING_CACHE_DIR = ".cursor/knowledge/embedding_cache"


@dataclass
class IndexingStats:
    """索引状态统计

    记录索引操作的统计信息和状态
    """

    last_index_time: Optional[datetime] = None  # 最后索引时间
    indexed_doc_count: int = 0  # 已索引文档数
    indexed_chunk_count: int = 0  # 已索引分块数
    pending_queue_size: int = 0  # 待索引队列大小
    total_index_operations: int = 0  # 总索引操作次数
    failed_index_count: int = 0  # 失败索引次数
    skipped_count: int = 0  # 跳过（无需重索引）次数

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "last_index_time": self.last_index_time.isoformat() if self.last_index_time else None,
            "indexed_doc_count": self.indexed_doc_count,
            "indexed_chunk_count": self.indexed_chunk_count,
            "pending_queue_size": self.pending_queue_size,
            "total_index_operations": self.total_index_operations,
            "failed_index_count": self.failed_index_count,
            "skipped_count": self.skipped_count,
        }


@dataclass
class VectorSearchResult:
    """向量搜索结果

    包含文档分块的搜索结果信息
    """

    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeVectorStore:
    """知识库向量存储

    基于 ChromaDB 和 SentenceTransformers 的向量存储实现，
    用于知识库文档的向量化索引和语义搜索。

    使用示例:
    ```python
    config = KnowledgeVectorConfig()
    store = KnowledgeVectorStore(config)
    await store.initialize()

    # 索引单个文档
    await store.index_document(doc)

    # 批量索引
    await store.index_documents([doc1, doc2, doc3])

    # 语义搜索
    results = await store.search("查询内容", top_k=10)

    # 更新文档
    await store.update_document(doc)

    # 删除文档
    await store.delete_document(doc_id)

    # 获取统计信息
    stats = await store.get_stats()
    ```
    """

    def __init__(self, config: KnowledgeVectorConfig):
        """初始化知识库向量存储

        Args:
            config: 向量配置
        """
        self.config = config
        self._initialized = False
        self._lock = asyncio.Lock()

        # ChromaDB 向量存储实例
        self._vector_store: "ChromaVectorStoreType | None" = None

        # SentenceTransformer 嵌入模型实例
        self._embedding_model: "SentenceTransformerEmbeddingType | None" = None

        # EmbeddingCache 实例
        self._embedding_cache: "EmbeddingCacheType | None" = None

        # 文档 ID 到分块 ID 的映射
        self._doc_chunk_mapping: dict[str, list[str]] = {}

        # 文档 ID 到内容哈希的映射（用于增量索引）
        self._doc_content_hash: dict[str, str] = {}

        # 待索引文档队列
        self._pending_queue: list[Document] = []

        # 索引状态统计
        self._indexing_stats = IndexingStats()

    async def initialize(self) -> None:
        """初始化向量存储

        创建 ChromaVectorStore 和 SentenceTransformerEmbedding 实例，
        配置 EmbeddingCache 持久化目录
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                # 延迟导入，避免启动时加载
                global ChromaVectorStore, SentenceTransformerEmbedding, EmbeddingCache
                if ChromaVectorStore is None:
                    from indexing.vector_store import ChromaVectorStore as _ChromaVectorStore

                    ChromaVectorStore = _ChromaVectorStore
                if SentenceTransformerEmbedding is None or EmbeddingCache is None:
                    from indexing.embedding import (
                        EmbeddingCache as _EmbeddingCache,
                    )
                    from indexing.embedding import (
                        SentenceTransformerEmbedding as _SentenceTransformerEmbedding,
                    )

                    SentenceTransformerEmbedding = _SentenceTransformerEmbedding
                    EmbeddingCache = _EmbeddingCache

                # 确定嵌入模型名称
                # 如果配置使用 OpenAI 模型，切换到本地模型
                model_name = self.config.embedding_model
                if model_name.startswith("text-embedding"):
                    model_name = "all-MiniLM-L6-v2"  # 使用默认本地模型
                    logger.info(f"切换到本地嵌入模型: {model_name}")

                # 创建嵌入缓存，使用专用持久化目录
                # 配置路径: .cursor/knowledge/embedding_cache
                cache_dir = Path(EMBEDDING_CACHE_DIR)
                cache_dir.mkdir(parents=True, exist_ok=True)

                self._embedding_cache = EmbeddingCache(
                    max_size=10000,
                    cache_dir=str(cache_dir),
                )
                logger.info(f"EmbeddingCache 已初始化，缓存目录: {cache_dir}")

                # 创建嵌入模型，使用持久化缓存
                self._embedding_model = SentenceTransformerEmbedding(
                    model_name=model_name,
                    batch_size=32,
                    cache=self._embedding_cache,
                )

                # 更新配置中的维度（根据实际模型）
                embedding_dimension = self._embedding_model.dimension

                # 创建 ChromaDB 向量存储
                # 使用知识库专用的集合名称
                self._vector_store = ChromaVectorStore(
                    persist_directory=self.config.vector_storage_path,
                    collection_name=KNOWLEDGE_COLLECTION_NAME,
                    embedding_dimension=embedding_dimension,
                    metric=self.config.similarity_metric,
                )

                self._initialized = True
                logger.info(
                    f"KnowledgeVectorStore 初始化完成: "
                    f"collection={KNOWLEDGE_COLLECTION_NAME}, "
                    f"model={model_name}, "
                    f"dimension={embedding_dimension}"
                )

            except ImportError as e:
                logger.error(f"初始化失败，缺少依赖: {e}")
                raise ImportError("请安装必要依赖: pip install chromadb sentence-transformers") from e
            except Exception as e:
                logger.error(f"初始化向量存储失败: {e}")
                raise

    async def index_document(self, doc: Document, force: bool = False) -> bool:
        """索引文档

        为文档生成分块和嵌入，存储到向量数据库。
        支持增量索引：如果文档内容未变化则跳过。

        Args:
            doc: 要索引的文档
            force: 是否强制重新索引（忽略内容哈希检查）

        Returns:
            是否索引成功
        """
        if not self._initialized:
            await self.initialize()

        assert self._embedding_model is not None
        assert self._vector_store is not None

        async with self._lock:
            try:
                # 增量索引检查：如果内容未变化，跳过索引
                if not force and not self.should_reindex(doc):
                    logger.debug(f"文档 {doc.id} 内容未变化，跳过索引")
                    self._indexing_stats.skipped_count += 1
                    return True

                # 创建或使用现有分块
                chunks = doc.chunks if doc.chunks else self._create_chunks(doc)

                if not chunks:
                    logger.warning(f"文档 {doc.id} 没有可索引的内容")
                    return False

                # 准备分块数据
                chunk_contents = [chunk.content for chunk in chunks]
                chunk_ids = [chunk.chunk_id for chunk in chunks]

                # 生成嵌入向量
                embeddings = await self._embedding_model.embed_batch(chunk_contents)

                # 计算并存储内容哈希
                content_hash = self._compute_content_hash(doc)

                # 转换为 CodeChunk 格式（ChromaVectorStore 需要）
                from indexing.base import ChunkType, CodeChunk

                code_chunks = []
                for i, chunk in enumerate(chunks):
                    code_chunk = CodeChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        file_path=doc.url,  # 使用 URL 作为路径
                        start_line=chunk.start_index,
                        end_line=chunk.end_index,
                        chunk_type=ChunkType.UNKNOWN,
                        language="text",
                        embedding=embeddings[i],
                        metadata={
                            "doc_id": doc.id,
                            "title": doc.title,
                            "url": doc.url,
                            "content_hash": content_hash,  # 存储内容哈希
                            **chunk.metadata,
                        },
                    )
                    code_chunks.append(code_chunk)

                # 存储到向量数据库
                await self._vector_store.upsert(code_chunks)

                # 记录文档到分块的映射
                self._doc_chunk_mapping[doc.id] = chunk_ids

                # 记录内容哈希
                self._doc_content_hash[doc.id] = content_hash

                # 更新索引统计
                self._indexing_stats.last_index_time = datetime.now()
                self._indexing_stats.indexed_doc_count = len(self._doc_chunk_mapping)
                self._indexing_stats.indexed_chunk_count += len(chunks)
                self._indexing_stats.total_index_operations += 1

                logger.debug(f"文档索引成功: {doc.id}, 分块数: {len(chunks)}")
                return True

            except Exception as e:
                logger.error(f"索引文档失败 {doc.id}: {e}")
                self._indexing_stats.failed_index_count += 1
                return False

    async def index_documents(self, docs: list[Document]) -> int:
        """批量索引文档（顺序处理）

        Args:
            docs: 文档列表

        Returns:
            成功索引的文档数量
        """
        if not self._initialized:
            await self.initialize()

        success_count = 0
        for doc in docs:
            if await self.index_document(doc):
                success_count += 1

        logger.info(f"批量索引完成: {success_count}/{len(docs)} 成功")
        return success_count

    async def batch_index_documents(
        self,
        docs: list[Document],
        concurrency: int = 4,
        progress_callback: Optional[ProgressCallback] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """批量并发索引文档

        使用 asyncio.gather 实现并发嵌入生成，支持进度回调。

        Args:
            docs: 文档列表
            concurrency: 并发数量，控制同时处理的文档数
            progress_callback: 进度回调函数 (current, total, doc_id)
            force: 是否强制重新索引（忽略内容哈希检查）

        Returns:
            包含索引结果的字典:
            - success_count: 成功数量
            - failed_count: 失败数量
            - skipped_count: 跳过数量
            - duration_ms: 耗时（毫秒）
            - results: 每个文档的处理结果
        """
        if not self._initialized:
            await self.initialize()

        if not docs:
            return {
                "success_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "duration_ms": 0,
                "results": [],
            }

        start_time = datetime.now()
        total = len(docs)
        results: list[dict[str, Any]] = []

        # 更新待索引队列
        self._pending_queue = list(docs)
        self._indexing_stats.pending_queue_size = len(self._pending_queue)

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        async def process_doc(doc: Document, index: int) -> dict[str, Any]:
            """处理单个文档"""
            nonlocal completed

            async with semaphore:
                try:
                    # 增量索引检查
                    if not force and not self.should_reindex(doc):
                        result = {
                            "doc_id": doc.id,
                            "status": "skipped",
                            "reason": "content_unchanged",
                        }
                        self._indexing_stats.skipped_count += 1
                    else:
                        # 执行索引
                        success = await self._index_document_internal(doc)
                        result = {
                            "doc_id": doc.id,
                            "status": "success" if success else "failed",
                        }

                    # 从待索引队列移除
                    if doc in self._pending_queue:
                        self._pending_queue.remove(doc)
                        self._indexing_stats.pending_queue_size = len(self._pending_queue)

                    completed += 1

                    # 调用进度回调
                    if progress_callback:
                        progress_callback(completed, total, doc.id)

                    return result

                except Exception as e:
                    logger.error(f"处理文档 {doc.id} 时发生错误: {e}")
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, doc.id)
                    return {
                        "doc_id": doc.id,
                        "status": "failed",
                        "error": str(e),
                    }

        # 并发处理所有文档
        tasks = [process_doc(doc, i) for i, doc in enumerate(docs)]
        results = await asyncio.gather(*tasks)

        # 统计结果
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        logger.info(
            f"批量并发索引完成: 成功={success_count}, 失败={failed_count}, 跳过={skipped_count}, 耗时={duration_ms}ms"
        )

        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "duration_ms": duration_ms,
            "results": results,
        }

    async def _index_document_internal(self, doc: Document) -> bool:
        """内部索引方法（不加锁，供并发调用）

        Args:
            doc: 要索引的文档

        Returns:
            是否索引成功
        """
        assert self._embedding_model is not None
        assert self._vector_store is not None
        try:
            # 创建或使用现有分块
            chunks = doc.chunks if doc.chunks else self._create_chunks(doc)

            if not chunks:
                logger.warning(f"文档 {doc.id} 没有可索引的内容")
                return False

            # 准备分块数据
            chunk_contents = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]

            # 生成嵌入向量
            embeddings = await self._embedding_model.embed_batch(chunk_contents)

            # 计算并存储内容哈希
            content_hash = self._compute_content_hash(doc)

            # 转换为 CodeChunk 格式
            from indexing.base import ChunkType, CodeChunk

            code_chunks = []
            for i, chunk in enumerate(chunks):
                code_chunk = CodeChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    file_path=doc.url,
                    start_line=chunk.start_index,
                    end_line=chunk.end_index,
                    chunk_type=ChunkType.UNKNOWN,
                    language="text",
                    embedding=embeddings[i],
                    metadata={
                        "doc_id": doc.id,
                        "title": doc.title,
                        "url": doc.url,
                        "content_hash": content_hash,
                        **chunk.metadata,
                    },
                )
                code_chunks.append(code_chunk)

            # 存储到向量数据库（使用锁保护）
            async with self._lock:
                await self._vector_store.upsert(code_chunks)

                # 记录映射和哈希
                self._doc_chunk_mapping[doc.id] = chunk_ids
                self._doc_content_hash[doc.id] = content_hash

                # 更新索引统计
                self._indexing_stats.last_index_time = datetime.now()
                self._indexing_stats.indexed_doc_count = len(self._doc_chunk_mapping)
                self._indexing_stats.indexed_chunk_count += len(chunks)
                self._indexing_stats.total_index_operations += 1

            logger.debug(f"文档索引成功: {doc.id}, 分块数: {len(chunks)}")
            return True

        except Exception as e:
            logger.error(f"索引文档失败 {doc.id}: {e}")
            self._indexing_stats.failed_index_count += 1
            return False

    async def update_document(self, doc: Document) -> bool:
        """更新已索引的文档

        删除旧的分块向量并重新索引。

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
        """删除文档的所有分块向量

        Args:
            doc_id: 文档 ID

        Returns:
            是否删除成功
        """
        if not self._initialized:
            await self.initialize()

        assert self._vector_store is not None

        async with self._lock:
            try:
                # 获取该文档的所有分块 ID
                chunk_ids = self._doc_chunk_mapping.get(doc_id, [])

                if chunk_ids:
                    # 删除分块向量
                    await self._vector_store.delete(chunk_ids)
                    # 清除映射
                    del self._doc_chunk_mapping[doc_id]
                    logger.debug(f"文档索引已删除: {doc_id}, 删除 {len(chunk_ids)} 个分块")
                else:
                    # 尝试按 doc_id 元数据删除
                    await self._vector_store.delete_by_filter({"meta_doc_id": doc_id})
                    logger.debug(f"文档索引已删除（按元数据）: {doc_id}")

                # 清除内容哈希
                if doc_id in self._doc_content_hash:
                    del self._doc_content_hash[doc_id]

                return True

            except Exception as e:
                logger.error(f"删除文档索引失败 {doc_id}: {e}")
                return False

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """向量语义搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            min_score: 最小相似度分数（0-1）

        Returns:
            搜索结果列表，按相似度降序排列
        """
        if not self._initialized:
            await self.initialize()

        assert self._embedding_model is not None
        assert self._vector_store is not None

        try:
            # 生成查询向量
            query_embedding = await self._embedding_model.embed_text(query)

            # 执行向量搜索
            search_results = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=min_score if min_score > 0 else None,
            )

            # 转换为 VectorSearchResult
            results: list[VectorSearchResult] = []
            for sr in search_results:
                # 从元数据中提取文档信息
                doc_id = sr.chunk.metadata.get("doc_id", "")

                results.append(
                    VectorSearchResult(
                        doc_id=doc_id,
                        chunk_id=sr.chunk.chunk_id,
                        content=sr.chunk.content,
                        score=sr.score,
                        metadata={
                            "title": sr.chunk.metadata.get("title", ""),
                            "url": sr.chunk.file_path,
                            "rank": sr.rank,
                            **{k: v for k, v in sr.chunk.metadata.items() if k not in ("doc_id", "title")},
                        },
                    )
                )

            # 二次过滤：即使底层存储没有应用阈值，也确保按 min_score 过滤
            if min_score > 0:
                results = [r for r in results if r.score >= min_score]

            return results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    async def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息

        Returns:
            统计信息字典，包含：
            - 基本配置信息
            - 向量存储状态
            - 嵌入模型信息
            - 索引状态统计
        """
        if not self._initialized:
            await self.initialize()

        assert self._vector_store is not None
        assert self._embedding_model is not None

        try:
            # 获取向量存储统计
            store_stats = self._vector_store.get_stats()

            # 获取嵌入模型信息
            model_info = self._embedding_model.get_model_info()

            # 获取索引状态统计
            indexing_stats = self.get_indexing_stats()

            return {
                "initialized": self._initialized,
                "collection_name": KNOWLEDGE_COLLECTION_NAME,
                "document_count": len(self._doc_chunk_mapping),
                "chunk_count": store_stats.get("current_collection_count", 0),
                "embedding_model": model_info.get("model_name", "unknown"),
                "embedding_dimension": model_info.get("dimension", 0),
                "persist_directory": store_stats.get("persist_directory", ""),
                "is_persistent": store_stats.get("is_persistent", False),
                "metric": store_stats.get("metric", "cosine"),
                "cache_stats": model_info.get("cache_stats", {}),
                # 索引状态统计
                "indexing_stats": indexing_stats.to_dict(),
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "initialized": self._initialized,
                "error": str(e),
            }

    def _compute_content_hash(self, doc: Document) -> str:
        """计算文档内容哈希

        用于增量索引，检测内容是否发生变化。

        Args:
            doc: 文档对象

        Returns:
            内容的 MD5 哈希值
        """
        # 组合文档 ID、标题和内容计算哈希
        content_to_hash = f"{doc.id}:{doc.title or ''}:{doc.content or ''}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()

    def should_reindex(self, doc: Document) -> bool:
        """判断文档是否需要重新索引

        通过比较内容哈希来判断文档内容是否发生变化。

        Args:
            doc: 文档对象

        Returns:
            True 表示需要重新索引，False 表示可以跳过
        """
        # 如果文档从未索引过，需要索引
        if doc.id not in self._doc_content_hash:
            return True

        # 计算当前内容哈希
        current_hash = self._compute_content_hash(doc)
        stored_hash = self._doc_content_hash.get(doc.id)

        # 如果哈希值不同，需要重新索引
        return current_hash != stored_hash

    def get_content_hash(self, doc_id: str) -> Optional[str]:
        """获取已存储的文档内容哈希

        Args:
            doc_id: 文档 ID

        Returns:
            内容哈希值，不存在则返回 None
        """
        return self._doc_content_hash.get(doc_id)

    def get_indexing_stats(self) -> IndexingStats:
        """获取索引状态统计

        Returns:
            索引状态统计对象
        """
        # 更新当前队列大小
        self._indexing_stats.pending_queue_size = len(self._pending_queue)
        self._indexing_stats.indexed_doc_count = len(self._doc_chunk_mapping)
        return self._indexing_stats

    def add_to_pending_queue(self, docs: list[Document]) -> None:
        """添加文档到待索引队列

        Args:
            docs: 要添加的文档列表
        """
        for doc in docs:
            if doc not in self._pending_queue:
                self._pending_queue.append(doc)
        self._indexing_stats.pending_queue_size = len(self._pending_queue)

    def get_pending_queue(self) -> list[Document]:
        """获取待索引队列

        Returns:
            待索引文档列表
        """
        return list(self._pending_queue)

    def clear_pending_queue(self) -> None:
        """清空待索引队列"""
        self._pending_queue.clear()
        self._indexing_stats.pending_queue_size = 0

    async def close(self) -> None:
        """关闭向量存储

        保存嵌入缓存到磁盘
        """
        if self._embedding_cache is not None:
            try:
                self._embedding_cache.save_to_disk()
                logger.info("嵌入缓存已保存到磁盘")
            except Exception as e:
                logger.error(f"保存嵌入缓存失败: {e}")

        self._initialized = False
        logger.info("KnowledgeVectorStore 已关闭")

    def _create_chunks(self, doc: Document) -> list[DocumentChunk]:
        """为文档创建分块

        使用配置的分块大小和重叠参数进行智能分块。

        Args:
            doc: 文档对象

        Returns:
            分块列表
        """
        chunks: list[DocumentChunk] = []
        content = doc.content

        if not content or not content.strip():
            return chunks

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # 尝试在自然边界处分割
            if end < len(content):
                # 查找最近的段落或句子结束符
                for sep in ["\n\n", "。", ".", "！", "!", "？", "?", "\n", " "]:
                    last_sep = content.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk = DocumentChunk(
                    chunk_id=f"{doc.id}_chunk_{chunk_index}",
                    content=chunk_content,
                    source_doc=doc.id,
                    start_index=start,
                    end_index=end,
                    metadata={
                        "chunk_index": chunk_index,
                        "doc_title": doc.title,
                        "doc_url": doc.url,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # 计算下一个起始位置（考虑重叠）
            start = end - overlap if end < len(content) else len(content)

        return chunks

    @property
    def document_count(self) -> int:
        """已索引的文档数量"""
        return len(self._doc_chunk_mapping)

    @property
    def chunk_count(self) -> int:
        """已索引的分块数量"""
        return sum(len(chunks) for chunks in self._doc_chunk_mapping.values())

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    async def rebuild_all(
        self,
        documents: list[Document],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> int:
        """重建所有文档索引

        清空现有索引并重新索引所有文档。

        Args:
            documents: 文档列表
            progress_callback: 进度回调函数

        Returns:
            成功索引的文档数量
        """
        if not self._initialized:
            await self.initialize()

        assert self._vector_store is not None

        # 清空现有索引
        await self._vector_store.clear()
        self._doc_chunk_mapping.clear()
        self._doc_content_hash.clear()

        # 重置索引统计
        self._indexing_stats = IndexingStats()

        # 使用并发批量索引
        result = await self.batch_index_documents(
            documents,
            progress_callback=progress_callback,
            force=True,  # 重建时强制索引
        )
        success_count = result["success_count"]

        logger.info(f"向量索引重建完成: {success_count}/{len(documents)} 成功")
        return success_count
