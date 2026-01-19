"""知识库存储管理

提供文档的持久化存储、索引管理和检索功能。

存储结构:
    .cursor/knowledge/
    ├── index.json           # 知识库索引（文档列表、元数据摘要）
    ├── docs/
    │   └── {doc_id}.md      # 文档内容（Markdown 格式）
    └── metadata/
        └── {doc_id}.json    # 文档元数据（完整元数据）

特性:
- 增量更新：仅更新变化的文档
- 去重机制：基于 URL 和内容哈希去重
- 文档检索：支持按 ID、URL、标题搜索
"""
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .models import Document
from .vector import KnowledgeVectorConfig
from .vector_store import KnowledgeVectorStore, VectorSearchResult


@dataclass
class IndexEntry:
    """索引条目

    存储在 index.json 中的文档摘要信息
    """
    doc_id: str
    url: str
    title: str
    content_hash: str                       # 内容哈希，用于去重和变更检测
    chunk_count: int = 0
    content_size: int = 0                   # 内容字符数
    created_at: str = ""                    # ISO 格式时间戳
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "content_size": self.content_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        """从字典创建"""
        return cls(
            doc_id=data.get("doc_id", ""),
            url=data.get("url", ""),
            title=data.get("title", ""),
            content_hash=data.get("content_hash", ""),
            chunk_count=data.get("chunk_count", 0),
            content_size=data.get("content_size", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def from_document(cls, doc: Document) -> "IndexEntry":
        """从文档创建索引条目"""
        content_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()[:16]
        return cls(
            doc_id=doc.id,
            url=doc.url,
            title=doc.title,
            content_hash=content_hash,
            chunk_count=len(doc.chunks),
            content_size=len(doc.content),
            created_at=doc.created_at.isoformat(),
            updated_at=doc.updated_at.isoformat(),
        )


@dataclass
class StorageConfig:
    """存储配置"""
    # 存储根目录（相对于工作目录）
    storage_root: str = ".cursor/knowledge"

    # 子目录名称
    docs_dir: str = "docs"
    metadata_dir: str = "metadata"
    vectors_dir: str = "vectors"              # 向量索引子目录

    # 索引文件名
    index_file: str = "index.json"

    # 是否自动创建目录
    auto_create_dirs: bool = True

    # 是否在保存时备份
    backup_on_save: bool = False

    # 是否启用向量索引
    enable_vector_index: bool = False


@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    url: str
    title: str
    score: float = 1.0                      # 匹配分数
    snippet: str = ""                       # 内容摘要
    match_type: str = "exact"               # 匹配类型: exact, partial, fuzzy


class KnowledgeStorage:
    """知识库存储管理器

    管理文档的持久化存储、索引和检索。

    使用示例:
    ```python
    storage = KnowledgeStorage()
    await storage.initialize()

    # 保存文档
    doc = Document(url="https://example.com", title="Example", content="...")
    await storage.save_document(doc)

    # 加载文档
    doc = await storage.load_document("doc-xxxx")

    # 搜索文档
    results = await storage.search("example")
    ```
    """

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        workspace_root: Optional[str] = None,
    ):
        """初始化存储管理器

        Args:
            config: 存储配置
            workspace_root: 工作区根目录（默认当前目录）
        """
        self.config = config or StorageConfig()
        self.workspace_root = Path(workspace_root or os.getcwd())

        # 计算存储路径
        self.storage_path = self.workspace_root / self.config.storage_root
        self.docs_path = self.storage_path / self.config.docs_dir
        self.metadata_path = self.storage_path / self.config.metadata_dir
        self.index_path = self.storage_path / self.config.index_file
        self.vector_index_path = self.storage_path / self.config.vectors_dir

        # 索引缓存
        self._index: dict[str, IndexEntry] = {}
        self._url_to_id: dict[str, str] = {}    # URL -> doc_id 映射
        self._initialized = False
        self._lock = asyncio.Lock()

        # 向量存储（延迟初始化）
        self._vector_store: Optional[KnowledgeVectorStore] = None

    async def initialize(self) -> None:
        """初始化存储，创建必要的目录结构并加载索引"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # 创建目录结构
            if self.config.auto_create_dirs:
                self._ensure_directories()

            # 加载索引
            await self._load_index()

            # 初始化向量存储（如果启用）
            if self.config.enable_vector_index:
                await self._init_vector_store()

            self._initialized = True
            logger.info(f"知识库存储初始化完成，路径: {self.storage_path}, 文档数: {len(self._index)}")

    def _ensure_directories(self) -> None:
        """确保存储目录存在"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        if self.config.enable_vector_index:
            self.vector_index_path.mkdir(parents=True, exist_ok=True)

    async def _init_vector_store(self) -> None:
        """初始化向量存储"""
        try:
            vector_config = KnowledgeVectorConfig(
                vector_storage_path=str(self.vector_index_path),
            )
            self._vector_store = KnowledgeVectorStore(vector_config)
            await self._vector_store.initialize()
            logger.info(f"向量存储初始化完成: {self.vector_index_path}")
        except Exception as e:
            logger.warning(f"向量存储初始化失败，将禁用语义搜索: {e}")
            self._vector_store = None

    async def _load_index(self) -> None:
        """加载索引文件"""
        self._index = {}
        self._url_to_id = {}

        if not self.index_path.exists():
            logger.debug("索引文件不存在，将创建新索引")
            return

        try:
            content = await asyncio.to_thread(self.index_path.read_text, encoding="utf-8")
            data = json.loads(content)

            entries = data.get("documents", [])
            for entry_data in entries:
                entry = IndexEntry.from_dict(entry_data)
                self._index[entry.doc_id] = entry
                if entry.url:
                    self._url_to_id[entry.url] = entry.doc_id

            logger.debug(f"加载索引成功，共 {len(self._index)} 个文档")

        except json.JSONDecodeError as e:
            logger.error(f"索引文件格式错误: {e}")
        except Exception as e:
            logger.error(f"加载索引失败: {e}")

    async def _save_index(self) -> None:
        """保存索引文件"""
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "document_count": len(self._index),
            "documents": [entry.to_dict() for entry in self._index.values()],
        }

        content = json.dumps(data, ensure_ascii=False, indent=2)

        try:
            await asyncio.to_thread(self.index_path.write_text, content, encoding="utf-8")
            logger.debug(f"索引保存成功，共 {len(self._index)} 个文档")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    async def save_document(
        self,
        doc: Document,
        force: bool = False,
    ) -> tuple[bool, str]:
        """保存文档

        Args:
            doc: 要保存的文档
            force: 是否强制更新（忽略内容哈希检查）

        Returns:
            (是否保存成功, 操作说明)
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            # 计算内容哈希
            content_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()[:16]

            # 检查去重
            existing_entry = self._index.get(doc.id) or self._get_entry_by_url(doc.url)

            if existing_entry:
                # 文档已存在，检查是否需要更新
                if not force and existing_entry.content_hash == content_hash:
                    return False, f"文档内容未变化，跳过更新: {doc.id}"

                # 更新现有文档
                doc.id = existing_entry.doc_id  # 使用已有 ID
                doc.created_at = datetime.fromisoformat(existing_entry.created_at)
                doc.updated_at = datetime.now()
                action = "更新"
            else:
                # 新文档
                action = "创建"

            try:
                # 保存文档内容 (Markdown)
                await self._save_doc_content(doc)

                # 保存元数据 (JSON)
                await self._save_doc_metadata(doc)

                # 更新索引
                entry = IndexEntry.from_document(doc)
                self._index[doc.id] = entry
                if doc.url:
                    self._url_to_id[doc.url] = doc.id

                # 保存索引
                await self._save_index()

                # 同步更新向量索引
                if self._vector_store is not None:
                    try:
                        if action == "更新":
                            await self._vector_store.update_document(doc)
                        else:
                            await self._vector_store.index_document(doc)
                    except Exception as ve:
                        logger.warning(f"向量索引更新失败: {ve}")

                logger.info(f"文档{action}成功: {doc.id} ({doc.title})")
                return True, f"文档{action}成功: {doc.id}"

            except Exception as e:
                logger.error(f"保存文档失败: {e}")
                return False, f"保存文档失败: {e}"

    async def _save_doc_content(self, doc: Document) -> None:
        """保存文档内容为 Markdown 文件"""
        file_path = self.docs_path / f"{doc.id}.md"

        # 构建 Markdown 内容
        lines = [
            f"# {doc.title}",
            "",
            f"> 来源: {doc.url}",
            f"> 创建时间: {doc.created_at.isoformat()}",
            f"> 更新时间: {doc.updated_at.isoformat()}",
            "",
            "---",
            "",
            doc.content,
        ]
        content = "\n".join(lines)

        await asyncio.to_thread(file_path.write_text, content, encoding="utf-8")

    async def _save_doc_metadata(self, doc: Document) -> None:
        """保存文档元数据为 JSON 文件"""
        file_path = self.metadata_path / f"{doc.id}.json"

        # 构建元数据
        metadata = {
            "id": doc.id,
            "url": doc.url,
            "title": doc.title,
            "content_hash": hashlib.sha256(doc.content.encode("utf-8")).hexdigest()[:16],
            "content_size": len(doc.content),
            "chunk_count": len(doc.chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "has_embedding": chunk.has_embedding(),
                    "metadata": chunk.metadata,
                }
                for chunk in doc.chunks
            ],
            "metadata": doc.metadata,
            "created_at": doc.created_at.isoformat(),
            "updated_at": doc.updated_at.isoformat(),
        }

        content = json.dumps(metadata, ensure_ascii=False, indent=2)
        await asyncio.to_thread(file_path.write_text, content, encoding="utf-8")

    def _get_entry_by_url(self, url: str) -> Optional[IndexEntry]:
        """根据 URL 获取索引条目"""
        doc_id = self._url_to_id.get(url)
        if doc_id:
            return self._index.get(doc_id)
        return None

    async def load_document(self, doc_id: str) -> Optional[Document]:
        """加载文档

        Args:
            doc_id: 文档 ID

        Returns:
            文档对象，不存在时返回 None
        """
        if not self._initialized:
            await self.initialize()

        if doc_id not in self._index:
            return None

        try:
            # 读取元数据
            metadata_path = self.metadata_path / f"{doc_id}.json"
            if not metadata_path.exists():
                logger.warning(f"元数据文件不存在: {doc_id}")
                return None

            metadata_content = await asyncio.to_thread(
                metadata_path.read_text, encoding="utf-8"
            )
            metadata = json.loads(metadata_content)

            # 读取文档内容
            doc_path = self.docs_path / f"{doc_id}.md"
            if doc_path.exists():
                raw_content = await asyncio.to_thread(
                    doc_path.read_text, encoding="utf-8"
                )
                # 解析 Markdown，提取实际内容（跳过元数据头）
                content = self._extract_content_from_markdown(raw_content)
            else:
                content = ""

            # 构建文档对象
            doc = Document(
                id=metadata.get("id", doc_id),
                url=metadata.get("url", ""),
                title=metadata.get("title", ""),
                content=content,
                metadata=metadata.get("metadata", {}),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
            )

            return doc

        except Exception as e:
            logger.error(f"加载文档失败 {doc_id}: {e}")
            return None

    def _extract_content_from_markdown(self, raw_content: str) -> str:
        """从 Markdown 文件中提取实际内容（跳过元数据头）"""
        lines = raw_content.split("\n")

        # 查找分隔线 "---" 的位置
        separator_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "---":
                separator_index = i
                break

        if separator_index >= 0:
            # 返回分隔线之后的内容
            content_lines = lines[separator_index + 1:]
            # 跳过开头的空行
            while content_lines and not content_lines[0].strip():
                content_lines.pop(0)
            return "\n".join(content_lines)

        return raw_content

    async def load_document_by_url(self, url: str) -> Optional[Document]:
        """根据 URL 加载文档

        Args:
            url: 文档来源 URL

        Returns:
            文档对象，不存在时返回 None
        """
        if not self._initialized:
            await self.initialize()

        doc_id = self._url_to_id.get(url)
        if doc_id:
            return await self.load_document(doc_id)
        return None

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档

        Args:
            doc_id: 文档 ID

        Returns:
            是否删除成功
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            if doc_id not in self._index:
                return False

            try:
                # 删除文档文件
                doc_path = self.docs_path / f"{doc_id}.md"
                if doc_path.exists():
                    await asyncio.to_thread(doc_path.unlink)

                # 删除元数据文件
                metadata_path = self.metadata_path / f"{doc_id}.json"
                if metadata_path.exists():
                    await asyncio.to_thread(metadata_path.unlink)

                # 更新索引
                entry = self._index.pop(doc_id, None)
                if entry and entry.url:
                    self._url_to_id.pop(entry.url, None)

                # 保存索引
                await self._save_index()

                # 同步删除向量索引
                if self._vector_store is not None:
                    try:
                        await self._vector_store.delete_document(doc_id)
                    except Exception as ve:
                        logger.warning(f"删除向量索引失败: {ve}")

                logger.info(f"文档删除成功: {doc_id}")
                return True

            except Exception as e:
                logger.error(f"删除文档失败 {doc_id}: {e}")
                return False

    async def search(
        self,
        query: str,
        limit: int = 10,
        search_content: bool = True,
        mode: str = "keyword",
    ) -> list[SearchResult]:
        """搜索文档

        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
            search_content: 是否搜索文档内容
            mode: 搜索模式 - "keyword"(关键词), "semantic"(语义), "hybrid"(混合)

        Returns:
            搜索结果列表
        """
        if not self._initialized:
            await self.initialize()

        if not query.strip():
            return []

        # 语义搜索模式
        if mode == "semantic":
            return await self._search_semantic_internal(query, limit)

        # 混合搜索模式：关键词 + 语义
        if mode == "hybrid":
            keyword_results = await self._search_keyword(query, limit, search_content)
            semantic_results = await self._search_semantic_internal(query, limit)
            return self._merge_search_results(keyword_results, semantic_results, limit)

        # 默认关键词搜索
        return await self._search_keyword(query, limit, search_content)

    async def _search_keyword(
        self,
        query: str,
        limit: int,
        search_content: bool,
    ) -> list[SearchResult]:
        """关键词搜索（内部方法）"""
        query_lower = query.lower()
        results: list[SearchResult] = []

        for doc_id, entry in self._index.items():
            score = 0.0
            match_type = ""
            snippet = ""

            # 标题匹配
            title_lower = entry.title.lower()
            if query_lower in title_lower:
                if title_lower == query_lower:
                    score = 1.0
                    match_type = "exact"
                else:
                    score = 0.8
                    match_type = "partial"
                snippet = entry.title

            # URL 匹配
            elif query_lower in entry.url.lower():
                score = 0.6
                match_type = "partial"
                snippet = entry.url

            # 内容匹配（需要加载文档）
            elif search_content:
                doc = await self.load_document(doc_id)
                if doc and query_lower in doc.content.lower():
                    score = 0.4
                    match_type = "partial"
                    # 提取匹配片段
                    snippet = self._extract_snippet(doc.content, query, max_length=100)

            if score > 0:
                results.append(SearchResult(
                    doc_id=doc_id,
                    url=entry.url,
                    title=entry.title,
                    score=score,
                    snippet=snippet,
                    match_type=match_type,
                ))

        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    async def _search_semantic_internal(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """语义搜索（内部方法）"""
        if self._vector_store is None:
            logger.warning("向量存储未初始化，无法进行语义搜索")
            return []

        try:
            vector_results = await self._vector_store.search(query, top_k=limit)

            results: list[SearchResult] = []
            seen_doc_ids = set()

            for vr in vector_results:
                # 去重（同一文档可能有多个分块匹配）
                if vr.doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(vr.doc_id)

                entry = self._index.get(vr.doc_id)
                if entry:
                    results.append(SearchResult(
                        doc_id=vr.doc_id,
                        url=entry.url,
                        title=entry.title,
                        score=vr.score,
                        snippet=vr.content[:100] + "..." if len(vr.content) > 100 else vr.content,
                        match_type="semantic",
                    ))

            return results

        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []

    def _merge_search_results(
        self,
        keyword_results: list[SearchResult],
        semantic_results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """合并关键词和语义搜索结果"""
        # 使用 doc_id 去重，取分数较高者
        merged: dict[str, SearchResult] = {}

        for r in keyword_results:
            merged[r.doc_id] = r

        for r in semantic_results:
            if r.doc_id not in merged or r.score > merged[r.doc_id].score:
                merged[r.doc_id] = r

        # 排序并返回
        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _extract_snippet(self, content: str, query: str, max_length: int = 100) -> str:
        """提取匹配片段"""
        query_lower = query.lower()
        content_lower = content.lower()

        index = content_lower.find(query_lower)
        if index < 0:
            return content[:max_length] + "..." if len(content) > max_length else content

        # 计算片段范围
        start = max(0, index - max_length // 2)
        end = min(len(content), index + len(query) + max_length // 2)

        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    async def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """语义搜索

        使用向量存储进行语义相似度搜索。

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            min_score: 最小相似度分数（0-1）

        Returns:
            向量搜索结果列表
        """
        if not self._initialized:
            await self.initialize()

        if self._vector_store is None:
            logger.warning("向量存储未初始化，无法进行语义搜索")
            return []

        return await self._vector_store.search(query, top_k=top_k, min_score=min_score)

    async def sync_vector_index(self) -> dict[str, Any]:
        """同步向量索引

        检查并同步文档与向量索引的一致性，确保所有文档都已索引。

        Returns:
            同步结果统计 {added: 新增数, removed: 移除数, total: 总数}
        """
        if not self._initialized:
            await self.initialize()

        if self._vector_store is None:
            logger.warning("向量存储未初始化，跳过同步")
            return {"added": 0, "removed": 0, "total": 0, "error": "向量存储未初始化"}

        try:
            added = 0
            removed = 0

            # 获取向量存储中已索引的文档 ID
            indexed_doc_ids = set(self._vector_store._doc_chunk_mapping.keys())
            storage_doc_ids = set(self._index.keys())

            # 查找需要添加的文档（在存储中但不在向量索引中）
            docs_to_add = storage_doc_ids - indexed_doc_ids
            for doc_id in docs_to_add:
                doc = await self.load_document(doc_id)
                if doc:
                    success = await self._vector_store.index_document(doc)
                    if success:
                        added += 1

            # 查找需要移除的文档（在向量索引中但不在存储中）
            docs_to_remove = indexed_doc_ids - storage_doc_ids
            for doc_id in docs_to_remove:
                success = await self._vector_store.delete_document(doc_id)
                if success:
                    removed += 1

            logger.info(f"向量索引同步完成: 新增 {added}, 移除 {removed}, 总数 {len(storage_doc_ids)}")

            return {
                "added": added,
                "removed": removed,
                "total": len(storage_doc_ids),
            }

        except Exception as e:
            logger.error(f"向量索引同步失败: {e}")
            return {"added": 0, "removed": 0, "total": 0, "error": str(e)}

    async def list_documents(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IndexEntry]:
        """列出文档

        Args:
            offset: 起始位置
            limit: 返回数量限制

        Returns:
            索引条目列表
        """
        if not self._initialized:
            await self.initialize()

        entries = list(self._index.values())

        # 按更新时间排序（最新的在前）
        entries.sort(key=lambda e: e.updated_at, reverse=True)

        return entries[offset:offset + limit]

    async def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息"""
        if not self._initialized:
            await self.initialize()

        total_size = sum(entry.content_size for entry in self._index.values())
        total_chunks = sum(entry.chunk_count for entry in self._index.values())

        stats = {
            "document_count": len(self._index),
            "total_content_size": total_size,
            "total_chunk_count": total_chunks,
            "storage_path": str(self.storage_path),
            "index_path": str(self.index_path),
            "vector_index_enabled": self.config.enable_vector_index,
            "vector_index_path": str(self.vector_index_path),
        }

        # 添加向量存储统计
        if self._vector_store is not None:
            try:
                vector_stats = await self._vector_store.get_stats()
                stats["vector_stats"] = vector_stats
            except Exception as e:
                stats["vector_stats"] = {"error": str(e)}

        return stats

    def has_document(self, doc_id: str) -> bool:
        """检查文档是否存在"""
        return doc_id in self._index

    def has_url(self, url: str) -> bool:
        """检查 URL 是否已存在"""
        return url in self._url_to_id

    def get_document_id_by_url(self, url: str) -> Optional[str]:
        """根据 URL 获取文档 ID"""
        return self._url_to_id.get(url)

    def get_content_hash_by_url(self, url: str) -> Optional[str]:
        """根据 URL 获取文档内容哈希（fingerprint）

        用于快速比较内容是否变化，无需加载完整文档。

        Args:
            url: 文档来源 URL

        Returns:
            内容哈希（16字符 SHA256 前缀），不存在时返回 None
        """
        doc_id = self._url_to_id.get(url)
        if doc_id:
            entry = self._index.get(doc_id)
            if entry:
                return entry.content_hash
        return None

    async def get_index_entry_by_url(self, url: str) -> Optional[IndexEntry]:
        """根据 URL 获取索引条目

        用于获取文档的摘要信息（不加载完整内容）。

        Args:
            url: 文档来源 URL

        Returns:
            索引条目，不存在时返回 None
        """
        if not self._initialized:
            await self.initialize()

        doc_id = self._url_to_id.get(url)
        if doc_id:
            return self._index.get(doc_id)
        return None

    async def save_documents_batch(
        self,
        docs: list[Document],
        force: bool = False,
    ) -> dict[str, tuple[bool, str]]:
        """批量保存文档

        Args:
            docs: 文档列表
            force: 是否强制更新

        Returns:
            {doc_id: (是否成功, 操作说明)}
        """
        results = {}
        for doc in docs:
            success, message = await self.save_document(doc, force=force)
            results[doc.id] = (success, message)
        return results

    async def clear_all(self) -> bool:
        """清空所有文档

        警告: 此操作不可逆！

        Returns:
            是否成功
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                # 删除所有文档文件
                for doc_id in list(self._index.keys()):
                    doc_path = self.docs_path / f"{doc_id}.md"
                    if doc_path.exists():
                        await asyncio.to_thread(doc_path.unlink)

                    metadata_path = self.metadata_path / f"{doc_id}.json"
                    if metadata_path.exists():
                        await asyncio.to_thread(metadata_path.unlink)

                # 清空索引
                self._index.clear()
                self._url_to_id.clear()

                # 保存空索引
                await self._save_index()

                # 清空向量索引
                if self._vector_store is not None:
                    try:
                        await self._vector_store.rebuild_all([])
                    except Exception as ve:
                        logger.warning(f"清空向量索引失败: {ve}")

                logger.info("知识库已清空")
                return True

            except Exception as e:
                logger.error(f"清空知识库失败: {e}")
                return False
