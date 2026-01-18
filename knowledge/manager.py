"""知识库管理器

提供知识库的高层管理接口，整合文档获取、存储和检索功能。

特性：
- 统一的 URL 添加和管理接口
- 支持本地文件添加（.txt, .md, .rst 等文本格式）
- 支持批量操作
- 支持关键词搜索、语义搜索和混合搜索
- 文档刷新和删除
- 向量索引管理
- 支持 CLI ask 模式只读查询
"""
from __future__ import annotations

import asyncio
import subprocess
import shutil
import os
from pathlib import Path
from typing import Any, Optional, Union, Literal
from datetime import datetime
from loguru import logger
from pydantic import BaseModel

from .models import Document, KnowledgeBase, KnowledgeBaseStats
from .fetcher import WebFetcher, FetchConfig, FetchResult
from .storage import SearchResult
from .vector import (
    KnowledgeVectorConfig,
    KnowledgeVectorStore,
    KnowledgeSemanticSearch,
)


class AskResult(BaseModel):
    """CLI ask 模式查询结果"""
    success: bool
    answer: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    query: str = ""
    context_used: list[str] = []  # 使用的上下文文档标题/URL


class KnowledgeManager:
    """知识库管理器
    
    提供知识库的高层管理接口，封装文档的添加、获取、搜索和删除等操作。
    
    使用示例:
    ```python
    manager = KnowledgeManager()
    await manager.initialize()
    
    # 添加单个 URL
    doc = await manager.add_url("https://example.com")
    
    # 批量添加
    docs = await manager.add_urls([
        "https://example.com/page1",
        "https://example.com/page2",
    ])
    
    # 搜索
    results = manager.search("关键词")
    
    # 列出所有文档
    all_docs = manager.list()
    
    # 刷新文档
    await manager.refresh("doc-xxx")
    
    # 删除文档
    manager.remove("doc-xxx")
    ```
    """
    
    def __init__(
        self,
        name: str = "default",
        fetch_config: Optional[FetchConfig] = None,
        vector_config: Optional[KnowledgeVectorConfig] = None,
    ):
        """初始化知识库管理器
        
        Args:
            name: 知识库名称
            fetch_config: 网页获取配置
            vector_config: 向量搜索配置（None 表示禁用向量搜索）
        """
        self._knowledge_base = KnowledgeBase(name=name)
        self._fetcher = WebFetcher(fetch_config or FetchConfig())
        self._initialized = False
        self._url_to_doc_id: dict[str, str] = {}  # URL 到文档 ID 的映射
        
        # 向量搜索相关
        self._vector_config = vector_config
        self._vector_store: Optional[KnowledgeVectorStore] = None
        self._semantic_search: Optional[KnowledgeSemanticSearch] = None
        self._use_vector_search: bool = vector_config is not None and vector_config.enabled
    
    @property
    def name(self) -> str:
        """知识库名称"""
        return self._knowledge_base.name
    
    @property
    def stats(self) -> KnowledgeBaseStats:
        """知识库统计信息"""
        return self._knowledge_base.stats
    
    async def initialize(self) -> None:
        """初始化管理器
        
        初始化网页获取器、向量存储和其他组件
        """
        if self._initialized:
            return
        
        await self._fetcher.initialize()
        
        # 初始化向量存储（如果配置启用）
        if self._use_vector_search and self._vector_config:
            self._vector_store = KnowledgeVectorStore(self._vector_config)
            await self._vector_store.initialize()
            self._semantic_search = KnowledgeSemanticSearch(
                self._vector_store,
                self._vector_config,
            )
            logger.info(f"向量搜索已启用，模型: {self._vector_config.embedding_model}")
        
        self._initialized = True
        logger.info(f"KnowledgeManager '{self.name}' 初始化完成")
    
    async def add_url(
        self,
        url: str,
        metadata: Optional[dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Optional[Document]:
        """添加单个 URL 到知识库
        
        Args:
            url: 目标 URL
            metadata: 附加元数据
            force_refresh: 如果 URL 已存在，是否强制刷新
            
        Returns:
            创建或更新的文档，失败返回 None
        """
        if not self._initialized:
            await self.initialize()
        
        # 检查 URL 是否已存在
        if url in self._url_to_doc_id and not force_refresh:
            doc_id = self._url_to_doc_id[url]
            existing_doc = self._knowledge_base.get_document(doc_id)
            if existing_doc:
                logger.info(f"URL 已存在: {url} (doc_id={doc_id})")
                return existing_doc
        
        # 验证 URL
        if not self._fetcher.check_url_valid(url):
            logger.warning(f"无效的 URL: {url}")
            return None
        
        # 获取网页内容
        logger.info(f"正在获取: {url}")
        result = await self._fetcher.fetch(url)
        
        if not result.success:
            logger.error(f"获取失败: {url} - {result.error}")
            return None
        
        # 创建文档
        doc = self._create_document_from_result(result, metadata)
        
        # 添加到知识库
        self._knowledge_base.add_document(doc)
        self._url_to_doc_id[url] = doc.id
        
        # 索引到向量存储
        if self._use_vector_search and self._vector_store:
            await self._vector_store.index_document(doc)
        
        logger.info(f"文档添加成功: {doc.id} ({url})")
        return doc
    
    async def add_urls(
        self,
        urls: list[str],
        metadata: Optional[dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> list[Document]:
        """批量添加 URL 到知识库
        
        Args:
            urls: URL 列表
            metadata: 附加元数据（应用于所有文档）
            force_refresh: 是否强制刷新已存在的 URL
            
        Returns:
            成功添加的文档列表
        """
        if not self._initialized:
            await self.initialize()
        
        # 过滤需要获取的 URL
        urls_to_fetch = []
        existing_docs = []
        
        for url in urls:
            if not self._fetcher.check_url_valid(url):
                logger.warning(f"跳过无效 URL: {url}")
                continue
            
            if url in self._url_to_doc_id and not force_refresh:
                doc_id = self._url_to_doc_id[url]
                doc = self._knowledge_base.get_document(doc_id)
                if doc:
                    existing_docs.append(doc)
                    continue
            
            urls_to_fetch.append(url)
        
        if not urls_to_fetch:
            logger.info("没有需要获取的新 URL")
            return existing_docs
        
        # 并发获取
        logger.info(f"正在获取 {len(urls_to_fetch)} 个 URL...")
        results = await self._fetcher.fetch_many(urls_to_fetch)
        
        # 处理结果
        new_docs = []
        for result in results:
            if result.success:
                doc = self._create_document_from_result(result, metadata)
                self._knowledge_base.add_document(doc)
                self._url_to_doc_id[result.url] = doc.id
                new_docs.append(doc)
                
                # 索引到向量存储
                if self._use_vector_search and self._vector_store:
                    await self._vector_store.index_document(doc)
                
                logger.info(f"文档添加成功: {doc.id} ({result.url})")
            else:
                logger.error(f"获取失败: {result.url} - {result.error}")
        
        logger.info(f"批量添加完成: {len(new_docs)}/{len(urls_to_fetch)} 成功")
        return existing_docs + new_docs
    
    # 支持的本地文件扩展名
    SUPPORTED_FILE_EXTENSIONS = {".txt", ".md", ".rst", ".text", ".markdown"}
    
    async def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[dict[str, Any]] = None,
        encoding: str = "utf-8",
    ) -> Optional[Document]:
        """添加本地文件到知识库
        
        支持 .txt, .md, .rst 等文本格式文件。
        
        Args:
            file_path: 本地文件路径
            metadata: 附加元数据
            encoding: 文件编码，默认 utf-8
            
        Returns:
            创建的文档，失败返回 None
            
        示例:
            >>> doc = await manager.add_file("./docs/readme.md")
            >>> doc = await manager.add_file("/path/to/file.txt", metadata={"author": "test"})
        """
        if not self._initialized:
            await self.initialize()
        
        path = Path(file_path)
        
        # 检查文件是否存在
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
        
        if not path.is_file():
            logger.error(f"不是有效文件: {file_path}")
            return None
        
        # 检查文件扩展名
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FILE_EXTENSIONS:
            logger.warning(
                f"不支持的文件格式: {suffix}，支持的格式: {', '.join(self.SUPPORTED_FILE_EXTENSIONS)}"
            )
            return None
        
        # 读取文件内容
        try:
            content = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            logger.error(f"文件编码错误，尝试使用 {encoding} 解码失败: {file_path}")
            return None
        except Exception as e:
            logger.error(f"读取文件失败: {file_path} - {e}")
            return None
        
        if not content.strip():
            logger.warning(f"文件内容为空: {file_path}")
            return None
        
        # 生成 file:// URL
        abs_path = path.resolve()
        file_url = f"file://{abs_path}"
        
        # 检查是否已存在
        if file_url in self._url_to_doc_id:
            doc_id = self._url_to_doc_id[file_url]
            existing_doc = self._knowledge_base.get_document(doc_id)
            if existing_doc:
                logger.info(f"文件已存在: {file_path} (doc_id={doc_id})")
                return existing_doc
        
        # 构建元数据
        doc_metadata = {
            "source_type": "file",
            "file_path": str(abs_path),
            "file_name": path.name,
            "file_extension": suffix,
            "file_size": path.stat().st_size,
            "encoding": encoding,
            **(metadata or {}),
        }
        
        # 创建文档
        doc = Document(
            url=file_url,
            title=self._extract_title_from_file(content, path),
            content=content,
            metadata=doc_metadata,
        )
        
        # 添加到知识库
        self._knowledge_base.add_document(doc)
        self._url_to_doc_id[file_url] = doc.id
        
        # 索引到向量存储
        if self._use_vector_search and self._vector_store:
            await self._vector_store.index_document(doc)
        
        logger.info(f"文件添加成功: {doc.id} ({file_path})")
        return doc
    
    def _extract_title_from_file(self, content: str, path: Path) -> str:
        """从文件内容中提取标题
        
        Args:
            content: 文件内容
            path: 文件路径
            
        Returns:
            提取的标题，失败时返回文件名
        """
        import re
        
        lines = content.strip().split('\n')
        if not lines:
            return path.stem  # 使用文件名（不含扩展名）
        
        first_line = lines[0].strip()
        
        # Markdown 标题 (# Title)
        if first_line.startswith('#'):
            title = re.sub(r'^#+\s*', '', first_line)
            if title:
                return title[:200]
        
        # RST 标题 (下划线形式)
        if len(lines) >= 2:
            second_line = lines[1].strip()
            # RST 使用 =, -, ~ 等作为标题下划线
            if second_line and all(c in '=-~^"' for c in second_line):
                if first_line:
                    return first_line[:200]
        
        # 普通文本：使用第一行非空内容
        if first_line and len(first_line) <= 200:
            return first_line
        
        # 回退：使用文件名
        return path.stem
    
    async def refresh(
        self,
        url_or_doc_id: str,
    ) -> Optional[Document]:
        """刷新指定文档
        
        重新获取 URL 内容并更新文档
        
        Args:
            url_or_doc_id: URL 或文档 ID
            
        Returns:
            更新后的文档，失败返回 None
        """
        if not self._initialized:
            await self.initialize()
        
        # 查找文档
        doc = self._find_document(url_or_doc_id)
        if not doc:
            logger.warning(f"未找到文档: {url_or_doc_id}")
            return None
        
        # 重新获取内容
        logger.info(f"正在刷新文档: {doc.id} ({doc.url})")
        result = await self._fetcher.fetch(doc.url)
        
        if not result.success:
            logger.error(f"刷新失败: {doc.url} - {result.error}")
            return None
        
        # 更新文档内容
        doc.update_content(
            content=result.content,
            title=self._extract_title(result.content),
        )
        doc.metadata["last_refresh"] = datetime.now().isoformat()
        doc.metadata["fetch_method"] = result.method_used.value
        
        # 清空旧分块（需要重新分块）
        doc.chunks = []
        
        # 更新向量索引
        if self._use_vector_search and self._vector_store:
            await self._vector_store.update_document(doc)
        
        logger.info(f"文档刷新成功: {doc.id}")
        return doc

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """搜索知识库（同步）

        说明：
        - 同步版本仅进行**关键词搜索**，用于快速检索和兼容同步调用场景（例如单元测试/简单脚本）。
        - 如需语义/混合搜索，请使用 `await search_async(...)`。
        """
        if not query.strip():
            return []
        return self._keyword_search(query, max_results, min_score)
    
    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        search_mode: Literal["keyword", "semantic", "hybrid"] = "hybrid",
    ) -> list[SearchResult]:
        """搜索知识库
        
        支持三种搜索模式:
        - keyword: 关键词搜索（文本匹配）
        - semantic: 语义搜索（向量相似度）
        - hybrid: 混合搜索（关键词 + 语义）
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            min_score: 最小匹配分数
            search_mode: 搜索模式，默认 'hybrid'
            
        Returns:
            搜索结果列表，按相关度排序
        """
        if not query.strip():
            return []
        
        # 如果向量搜索未启用，回退到关键词搜索
        if search_mode in ("semantic", "hybrid") and not self._use_vector_search:
            search_mode = "keyword"
            logger.debug("向量搜索未启用，回退到关键词搜索")
        
        if search_mode == "keyword":
            return self._keyword_search(query, max_results, min_score)
        if search_mode == "semantic":
            return await self._semantic_search(query, max_results, min_score)
        # hybrid
        return await self._hybrid_search(query, max_results, min_score)
    
    def _keyword_search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """关键词搜索（文本匹配）"""
        results: list[SearchResult] = []
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        for doc in self._knowledge_base.documents.values():
            # 计算匹配分数
            score, matched_content = self._calculate_match_score(
                doc, query_terms, query_lower
            )
            
            if score >= min_score:
                results.append(SearchResult(
                    doc_id=doc.id,
                    url=doc.url,
                    title=doc.title,
                    score=score,
                    snippet=matched_content[:200] if matched_content else "",
                    match_type="exact" if query_lower in doc.content.lower() else "partial",
                ))
        
        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:max_results]
    
    async def _semantic_search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """语义搜索（向量相似度）"""
        if not self._semantic_search or not self._vector_store:
            return []
        
        vector_results = await self._semantic_search.semantic_search(
            query=query,
            top_k=max_results,
            min_score=min_score,
        )
        
        # 转换为 SearchResult
        results: list[SearchResult] = []
        for vr in vector_results:
            results.append(SearchResult(
                doc_id=vr.doc_id,
                url=vr.url,
                title=vr.title,
                score=vr.score,
                snippet=vr.content[:200] if vr.content else "",
                match_type="semantic",
            ))
        
        return results
    
    async def _hybrid_search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """混合搜索（关键词 + 语义）"""
        if not self._semantic_search or not self._vector_store:
            return self._keyword_search(query, max_results, min_score)
        
        # 获取关键词搜索结果
        keyword_results = self._keyword_search(query, max_results * 2, min_score)
        
        # 转换为混合搜索所需格式
        keyword_dicts = [
            {
                "doc_id": r.doc_id,
                "url": r.url,
                "title": r.title,
                "score": r.score,
                "snippet": r.snippet,
            }
            for r in keyword_results
        ]
        
        # 执行混合搜索
        vector_results = await self._semantic_search.hybrid_search(
            query=query,
            keyword_results=keyword_dicts,
            top_k=max_results,
        )
        
        # 转换为 SearchResult
        results: list[SearchResult] = []
        for vr in vector_results:
            results.append(SearchResult(
                doc_id=vr.doc_id,
                url=vr.url,
                title=vr.title,
                score=vr.score,
                snippet=vr.content[:200] if vr.content else "",
                match_type="hybrid",
            ))
        
        return results
    
    def list(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Document]:
        """列出所有文档
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            文档列表
        """
        docs = list(self._knowledge_base.documents.values())
        
        # 按更新时间排序（最新的在前）
        docs.sort(key=lambda d: d.updated_at, reverse=True)
        
        # 应用分页
        if offset:
            docs = docs[offset:]
        if limit:
            docs = docs[:limit]
        
        return docs
    
    def remove(self, doc_id: str) -> bool:
        """删除文档（同步）

        说明：
        - 同步版本会从内存知识库中移除文档并更新映射。
        - 若启用了向量索引且当前存在事件循环，则会**异步触发**向量删除任务（不阻塞）。
        - 如需确保向量索引删除完成，请使用 `await remove_async(...)`。
        """
        doc = self._knowledge_base.get_document(doc_id)
        if not doc:
            logger.warning(f"未找到文档: {doc_id}")
            return False

        # 尝试异步删除向量索引（不阻塞）
        if self._use_vector_search and self._vector_store:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._vector_store.delete_document(doc_id))
            except RuntimeError:
                # 没有运行中的事件循环，跳过向量删除（由调用方选择使用 remove_async）
                pass

        if doc.url in self._url_to_doc_id:
            del self._url_to_doc_id[doc.url]
        self._knowledge_base.remove_document(doc_id)
        logger.info(f"文档已删除: {doc_id}")
        return True

    async def remove_async(self, doc_id: str) -> bool:
        """删除文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            是否删除成功
        """
        doc = self._knowledge_base.get_document(doc_id)
        if not doc:
            logger.warning(f"未找到文档: {doc_id}")
            return False

        if self._use_vector_search and self._vector_store:
            await self._vector_store.delete_document(doc_id)

        if doc.url in self._url_to_doc_id:
            del self._url_to_doc_id[doc.url]
        self._knowledge_base.remove_document(doc_id)
        logger.info(f"文档已删除: {doc_id}")
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取指定文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            文档对象，不存在返回 None
        """
        return self._knowledge_base.get_document(doc_id)
    
    def get_document_by_url(self, url: str) -> Optional[Document]:
        """根据 URL 获取文档
        
        Args:
            url: 文档 URL
            
        Returns:
            文档对象，不存在返回 None
        """
        return self._knowledge_base.get_document_by_url(url)
    
    def _find_document(self, url_or_doc_id: str) -> Optional[Document]:
        """查找文档（支持 URL 或文档 ID）"""
        # 尝试作为文档 ID 查找
        doc = self._knowledge_base.get_document(url_or_doc_id)
        if doc:
            return doc
        
        # 尝试作为 URL 查找
        return self._knowledge_base.get_document_by_url(url_or_doc_id)
    
    def _create_document_from_result(
        self,
        result: FetchResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Document:
        """从获取结果创建文档"""
        doc_metadata = {
            "fetch_method": result.method_used.value,
            "fetch_duration": result.duration,
            "status_code": result.status_code,
            "content_type": result.content_type,
            **(result.metadata or {}),
            **(metadata or {}),
        }
        
        return Document(
            url=result.url,
            title=self._extract_title(result.content),
            content=result.content,
            metadata=doc_metadata,
        )
    
    def _extract_title(self, content: str) -> str:
        """从内容中提取标题"""
        import re
        
        # 尝试从 HTML title 标签提取
        match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 尝试从第一行提取
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 清理可能的标记符号
            first_line = re.sub(r'^[#\-*=]+\s*', '', first_line)
            if first_line and len(first_line) <= 200:
                return first_line
        
        return ""
    
    def _calculate_match_score(
        self,
        doc: Document,
        query_terms: list[str],
        query_lower: str,
    ) -> tuple[float, str]:
        """计算文档与查询的匹配分数
        
        Returns:
            (分数, 匹配的内容片段)
        """
        score = 0.0
        matched_content = ""
        
        # 标题匹配（权重更高）
        title_lower = doc.title.lower()
        content_lower = doc.content.lower()
        
        # 完整查询匹配
        if query_lower in title_lower:
            score += 10.0
        if query_lower in content_lower:
            score += 5.0
            # 提取匹配上下文
            idx = content_lower.find(query_lower)
            start = max(0, idx - 50)
            end = min(len(doc.content), idx + len(query_lower) + 50)
            matched_content = doc.content[start:end]
        
        # 分词匹配
        for term in query_terms:
            if term in title_lower:
                score += 3.0
            if term in content_lower:
                score += 1.0
                if not matched_content:
                    idx = content_lower.find(term)
                    start = max(0, idx - 50)
                    end = min(len(doc.content), idx + len(term) + 50)
                    matched_content = doc.content[start:end]
        
        # URL 匹配
        if query_lower in doc.url.lower():
            score += 2.0
        
        return score, matched_content
    
    async def rebuild_index(self) -> int:
        """重建所有文档的向量索引
        
        清空现有向量索引并重新索引所有文档。
        
        Returns:
            成功索引的文档数量
        """
        if not self._use_vector_search or not self._vector_store:
            logger.warning("向量搜索未启用，无法重建索引")
            return 0
        
        documents = list(self._knowledge_base.documents.values())
        if not documents:
            logger.info("知识库为空，无需重建索引")
            return 0
        
        logger.info(f"开始重建向量索引，共 {len(documents)} 个文档...")
        success_count = await self._vector_store.rebuild_all(documents)
        logger.info(f"向量索引重建完成: {success_count}/{len(documents)} 成功")
        
        return success_count
    
    def clear(self) -> int:
        """清空知识库（同步）

        返回：
            删除的文档数量

        说明：
        - 同步版本只清空内存中的文档与映射，并更新统计信息。
        - 若启用了向量索引且存在事件循环，则会异步触发逐文档删除任务（不阻塞）。
        - 如需确保向量索引清空完成，请使用 `await clear_async()`。
        """
        count = len(self._knowledge_base.documents)

        if self._use_vector_search and self._vector_store:
            try:
                loop = asyncio.get_running_loop()
                for doc_id in list(self._knowledge_base.documents.keys()):
                    loop.create_task(self._vector_store.delete_document(doc_id))
            except RuntimeError:
                pass

        self._knowledge_base.documents.clear()
        self._url_to_doc_id.clear()
        self._knowledge_base._update_stats()
        logger.info(f"知识库已清空，删除 {count} 个文档")
        return count

    async def clear_async(self) -> int:
        """清空知识库
        
        Returns:
            删除的文档数量
        """
        count = len(self._knowledge_base.documents)

        if self._use_vector_search and self._vector_store:
            for doc_id in list(self._knowledge_base.documents.keys()):
                await self._vector_store.delete_document(doc_id)

        self._knowledge_base.documents.clear()
        self._url_to_doc_id.clear()
        self._knowledge_base._update_stats()
        logger.info(f"知识库已清空，删除 {count} 个文档")
        return count
    
    @property
    def vector_search_enabled(self) -> bool:
        """向量搜索是否已启用"""
        return self._use_vector_search
    
    @property
    def indexed_document_count(self) -> int:
        """已索引的文档数量"""
        if self._vector_store:
            return self._vector_store.document_count
        return 0
    
    def __len__(self) -> int:
        """返回文档数量"""
        return len(self._knowledge_base.documents)
    
    def __repr__(self) -> str:
        return (
            f"KnowledgeManager(name='{self.name}', "
            f"documents={len(self)})"
        )
    
    # ========== CLI Ask 模式查询 ==========
    
    async def ask_with_cli(
        self,
        question: str,
        max_context_docs: int = 3,
        timeout: int = 60,
        model: Optional[str] = None,
        working_directory: Optional[str] = None,
    ) -> AskResult:
        """使用 CLI ask 模式执行只读查询
        
        通过 Cursor CLI 的 --mode=ask 模式，结合知识库上下文回答问题。
        此方法不会修改任何文件，仅用于问答查询。
        
        Args:
            question: 用户问题
            max_context_docs: 最大上下文文档数量
            timeout: 超时时间（秒）
            model: 使用的模型（默认使用 gpt-4o-mini）
            working_directory: 工作目录
            
        Returns:
            AskResult: 查询结果
            
        示例:
            >>> result = await manager.ask_with_cli("如何使用 Cursor Agent 的 MCP 功能？")
            >>> if result.success:
            ...     print(result.answer)
        """
        start_time = datetime.now()
        
        # 查找 agent CLI
        agent_path = self._find_agent_cli()
        if not agent_path:
            return AskResult(
                success=False,
                error="找不到 agent CLI，请先安装: curl https://cursor.com/install -fsS | bash",
                query=question,
            )
        
        # 从知识库搜索相关文档作为上下文
        context_docs = self.search(question, max_results=max_context_docs)
        context_used = []
        context_content = ""
        
        if context_docs:
            context_parts = ["## 参考文档\n"]
            for i, result in enumerate(context_docs, 1):
                doc = self.get_document(result.doc_id)
                if doc:
                    context_used.append(doc.title or doc.url)
                    # 限制每个文档内容长度
                    content_preview = doc.content[:1500] if doc.content else ""
                    context_parts.append(
                        f"### {i}. {doc.title or '未命名'}\n"
                        f"来源: {doc.url}\n"
                        f"```\n{content_preview}\n```\n"
                    )
            context_content = "\n".join(context_parts)
        
        # 构建完整 prompt
        prompt_parts = []
        if context_content:
            prompt_parts.append(context_content)
        prompt_parts.append(f"## 问题\n{question}")
        prompt_parts.append("\n请根据上述参考文档回答问题。如果文档中没有相关信息，请明确说明。")
        
        full_prompt = "\n".join(prompt_parts)
        
        # 构建 CLI 命令
        cmd = [
            agent_path,
            "-p", full_prompt,
            "--mode", "ask",
            "--output-format", "text",
        ]
        
        if model:
            cmd.extend(["--model", model])
        
        try:
            # 构建环境变量
            env = os.environ.copy()
            
            # 执行 CLI 命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_directory or ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if process.returncode == 0:
                logger.info(f"CLI ask 查询成功，耗时 {duration:.2f}s")
                return AskResult(
                    success=True,
                    answer=output.strip(),
                    query=question,
                    duration=duration,
                    context_used=context_used,
                )
            else:
                error_msg = error_output.strip() or f"exit_code: {process.returncode}"
                logger.warning(f"CLI ask 查询失败: {error_msg}")
                return AskResult(
                    success=False,
                    answer=output.strip(),
                    error=error_msg,
                    query=question,
                    duration=duration,
                    context_used=context_used,
                )
                
        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"CLI ask 查询超时 ({timeout}s)")
            return AskResult(
                success=False,
                error=f"查询超时 ({timeout}s)",
                query=question,
                duration=duration,
                context_used=context_used,
            )
        except FileNotFoundError:
            logger.error(f"找不到 agent CLI: {agent_path}")
            return AskResult(
                success=False,
                error=f"找不到 agent CLI: {agent_path}",
                query=question,
            )
        except Exception as e:
            logger.exception(f"CLI ask 查询异常: {e}")
            return AskResult(
                success=False,
                error=str(e),
                query=question,
            )
    
    def _find_agent_cli(self) -> Optional[str]:
        """查找 agent CLI 可执行文件"""
        # 尝试常见路径
        possible_paths = [
            shutil.which("agent"),
            "/usr/local/bin/agent",
            os.path.expanduser("~/.local/bin/agent"),
            os.path.expanduser("~/.cursor/bin/agent"),
        ]
        
        for path in possible_paths:
            if path and os.path.isfile(path):
                return path
        
        # 回退检查 PATH 中是否有 agent
        if shutil.which("agent"):
            return "agent"
        
        return None
    
    def ask_with_cli_sync(
        self,
        question: str,
        max_context_docs: int = 3,
        timeout: int = 60,
        model: Optional[str] = None,
        working_directory: Optional[str] = None,
    ) -> AskResult:
        """同步版本：使用 CLI ask 模式执行只读查询
        
        适用于同步调用场景（如测试、脚本）。
        
        Args:
            question: 用户问题
            max_context_docs: 最大上下文文档数量
            timeout: 超时时间（秒）
            model: 使用的模型
            working_directory: 工作目录
            
        Returns:
            AskResult: 查询结果
        """
        start_time = datetime.now()
        
        agent_path = self._find_agent_cli()
        if not agent_path:
            return AskResult(
                success=False,
                error="找不到 agent CLI，请先安装: curl https://cursor.com/install -fsS | bash",
                query=question,
            )
        
        # 从知识库搜索相关文档作为上下文
        context_docs = self.search(question, max_results=max_context_docs)
        context_used = []
        context_content = ""
        
        if context_docs:
            context_parts = ["## 参考文档\n"]
            for i, result in enumerate(context_docs, 1):
                doc = self.get_document(result.doc_id)
                if doc:
                    context_used.append(doc.title or doc.url)
                    content_preview = doc.content[:1500] if doc.content else ""
                    context_parts.append(
                        f"### {i}. {doc.title or '未命名'}\n"
                        f"来源: {doc.url}\n"
                        f"```\n{content_preview}\n```\n"
                    )
            context_content = "\n".join(context_parts)
        
        # 构建完整 prompt
        prompt_parts = []
        if context_content:
            prompt_parts.append(context_content)
        prompt_parts.append(f"## 问题\n{question}")
        prompt_parts.append("\n请根据上述参考文档回答问题。如果文档中没有相关信息，请明确说明。")
        
        full_prompt = "\n".join(prompt_parts)
        
        cmd = [
            agent_path,
            "-p", full_prompt,
            "--mode", "ask",
            "--output-format", "text",
        ]
        
        if model:
            cmd.extend(["--model", model])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_directory or ".",
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"CLI ask 查询成功，耗时 {duration:.2f}s")
                return AskResult(
                    success=True,
                    answer=result.stdout.strip(),
                    query=question,
                    duration=duration,
                    context_used=context_used,
                )
            else:
                error_msg = result.stderr.strip() or f"exit_code: {result.returncode}"
                logger.warning(f"CLI ask 查询失败: {error_msg}")
                return AskResult(
                    success=False,
                    answer=result.stdout.strip(),
                    error=error_msg,
                    query=question,
                    duration=duration,
                    context_used=context_used,
                )
                
        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"CLI ask 查询超时 ({timeout}s)")
            return AskResult(
                success=False,
                error=f"查询超时 ({timeout}s)",
                query=question,
                duration=duration,
                context_used=context_used,
            )
        except FileNotFoundError:
            logger.error(f"找不到 agent CLI: {agent_path}")
            return AskResult(
                success=False,
                error=f"找不到 agent CLI: {agent_path}",
                query=question,
            )
        except Exception as e:
            logger.exception(f"CLI ask 查询异常: {e}")
            return AskResult(
                success=False,
                error=str(e),
                query=question,
            )
