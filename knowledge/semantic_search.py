"""知识库语义搜索模块

提供基于向量的语义搜索、关键词搜索和混合搜索功能。

特性：
- 纯语义搜索：基于向量相似度的搜索
- 关键词搜索：基于文本匹配的搜索
- 混合搜索：结合语义和关键词的加权搜索
- 结果去重和分数归一化
"""
import asyncio
import re
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from .models import Document
from .storage import SearchResult
from .vector import KnowledgeVectorConfig
from .vector_store import KnowledgeVectorStore, VectorSearchResult


@dataclass
class HybridSearchConfig:
    """混合搜索配置"""
    semantic_weight: float = 0.7    # 语义搜索权重
    keyword_weight: float = 0.3     # 关键词搜索权重
    normalize_scores: bool = True   # 是否归一化分数
    dedup_by_doc: bool = True       # 是否按文档去重（保留最高分块）


class KnowledgeSemanticSearch:
    """知识库语义搜索引擎
    
    提供基于向量的语义搜索和混合搜索功能，支持：
    - 纯语义搜索：利用向量相似度查找语义相关内容
    - 关键词搜索：基于文本匹配的传统搜索
    - 混合搜索：加权合并语义和关键词搜索结果
    
    使用示例:
    ```python
    from knowledge.vector_store import KnowledgeVectorStore
    from knowledge.vector import KnowledgeVectorConfig
    
    config = KnowledgeVectorConfig()
    vector_store = KnowledgeVectorStore(config)
    await vector_store.initialize()
    
    search = KnowledgeSemanticSearch(vector_store, config)
    
    # 纯语义搜索
    results = await search.semantic_search("如何配置 MCP 服务器")
    
    # 关键词搜索
    results = await search.keyword_search("MCP", documents)
    
    # 混合搜索
    results = await search.hybrid_search("MCP 配置", documents)
    ```
    """
    
    def __init__(
        self,
        vector_store: KnowledgeVectorStore,
        config: Optional[KnowledgeVectorConfig] = None,
    ):
        """初始化语义搜索引擎
        
        Args:
            vector_store: 向量存储实例（基于 ChromaDB）
            config: 向量配置（可选）
        """
        self.vector_store = vector_store
        self.config = config or KnowledgeVectorConfig()
        self._lock = asyncio.Lock()
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """纯语义搜索
        
        基于向量相似度的语义搜索，查找与查询语义最相关的文档分块。
        
        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            min_score: 最小相似度分数（0-1）
            
        Returns:
            按相似度降序排列的搜索结果列表
        """
        if not query.strip():
            return []
        
        try:
            # 调用向量存储进行搜索
            vector_results = await self.vector_store.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
            )
            
            # 转换为 SearchResult 格式
            results = self._convert_vector_results(vector_results, match_type="semantic")
            
            logger.debug(f"语义搜索完成: query='{query[:50]}...', 结果数={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    async def keyword_search(
        self,
        query: str,
        documents: dict[str, Document],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """关键词搜索
        
        基于文本匹配的传统搜索，支持标题、URL和内容匹配。
        
        Args:
            query: 搜索查询文本
            documents: 文档字典 {doc_id: Document}
            top_k: 返回结果数量
            
        Returns:
            按匹配分数降序排列的搜索结果列表
        """
        if not query.strip():
            return []
        
        query_lower = query.lower()
        query_terms = self._tokenize(query)
        results: list[SearchResult] = []
        
        for doc_id, doc in documents.items():
            score = 0.0
            match_type = "keyword"
            snippet = ""
            
            title_lower = doc.title.lower()
            url_lower = doc.url.lower()
            content_lower = doc.content.lower()
            
            # 标题精确匹配（最高分）
            if query_lower == title_lower:
                score = 1.0
                snippet = doc.title
            # 标题包含查询
            elif query_lower in title_lower:
                score = 0.85
                snippet = doc.title
            # 查询包含在标题中的词
            elif self._terms_match(query_terms, title_lower):
                term_match_ratio = self._calculate_term_match_ratio(query_terms, title_lower)
                score = 0.7 * term_match_ratio
                snippet = doc.title
            # URL 匹配
            elif query_lower in url_lower:
                score = 0.5
                snippet = doc.url
            # 内容匹配
            elif query_lower in content_lower:
                score = 0.4
                snippet = self._extract_snippet(doc.content, query)
            # 内容中的词匹配
            elif self._terms_match(query_terms, content_lower):
                term_match_ratio = self._calculate_term_match_ratio(query_terms, content_lower)
                score = 0.3 * term_match_ratio
                snippet = self._extract_snippet_by_terms(doc.content, query_terms)
            
            if score > 0:
                results.append(SearchResult(
                    doc_id=doc_id,
                    url=doc.url,
                    title=doc.title,
                    score=score,
                    snippet=snippet,
                    match_type=match_type,
                ))
        
        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)
        
        logger.debug(f"关键词搜索完成: query='{query[:50]}...', 结果数={len(results[:top_k])}")
        return results[:top_k]
    
    async def hybrid_search(
        self,
        query: str,
        documents: dict[str, Document],
        top_k: int = 10,
        semantic_weight: float = 0.7,
    ) -> list[SearchResult]:
        """混合搜索
        
        结合语义搜索和关键词搜索的结果，使用加权分数合并。
        
        合并策略：
        1. 并行执行语义搜索和关键词搜索
        2. 对结果按文档 ID 去重（保留最高分的分块）
        3. 计算加权混合分数
        4. 按最终分数排序
        
        Args:
            query: 搜索查询文本
            documents: 文档字典 {doc_id: Document}
            top_k: 返回结果数量
            semantic_weight: 语义搜索权重（关键词权重 = 1 - semantic_weight）
            
        Returns:
            按混合分数降序排列的搜索结果列表
        """
        if not query.strip():
            return []
        
        keyword_weight = 1.0 - semantic_weight
        
        try:
            # 并行执行两种搜索
            semantic_task = self.semantic_search(
                query=query,
                top_k=top_k * 2,  # 获取更多结果用于合并
                min_score=0.1,   # 降低阈值以获取更多候选
            )
            keyword_task = self.keyword_search(
                query=query,
                documents=documents,
                top_k=top_k * 2,
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task,
                keyword_task,
            )
            
            # 合并结果
            merged_results = self._merge_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
            
            logger.debug(
                f"混合搜索完成: query='{query[:50]}...', "
                f"语义结果={len(semantic_results)}, "
                f"关键词结果={len(keyword_results)}, "
                f"合并后={len(merged_results[:top_k])}"
            )
            
            return merged_results[:top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            # 降级到关键词搜索
            return await self.keyword_search(query, documents, top_k)
    
    def _merge_results(
        self,
        semantic_results: list[SearchResult],
        keyword_results: list[SearchResult],
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        """合并语义和关键词搜索结果
        
        Args:
            semantic_results: 语义搜索结果
            keyword_results: 关键词搜索结果
            semantic_weight: 语义分数权重
            keyword_weight: 关键词分数权重
            
        Returns:
            合并后的结果列表
        """
        # 文档分数映射 {doc_id: {semantic_score, keyword_score, best_result}}
        doc_scores: dict[str, dict[str, Any]] = {}
        
        # 处理语义搜索结果
        for result in semantic_results:
            doc_id = result.doc_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_result": None,
                    "keyword_result": None,
                }
            # 保留该文档的最高语义分数
            if result.score > doc_scores[doc_id]["semantic_score"]:
                doc_scores[doc_id]["semantic_score"] = result.score
                doc_scores[doc_id]["semantic_result"] = result
        
        # 处理关键词搜索结果
        for result in keyword_results:
            doc_id = result.doc_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_result": None,
                    "keyword_result": None,
                }
            # 保留该文档的最高关键词分数
            if result.score > doc_scores[doc_id]["keyword_score"]:
                doc_scores[doc_id]["keyword_score"] = result.score
                doc_scores[doc_id]["keyword_result"] = result
        
        # 计算混合分数并创建结果
        merged: list[SearchResult] = []
        
        for doc_id, scores in doc_scores.items():
            # 计算加权混合分数
            hybrid_score = (
                semantic_weight * scores["semantic_score"] +
                keyword_weight * scores["keyword_score"]
            )
            
            # 选择最佳结果作为基础
            # 优先选择语义结果（通常包含更相关的分块内容）
            base_result = scores["semantic_result"] or scores["keyword_result"]
            
            if base_result:
                # 合并 snippet：如果两种搜索都有结果，可能使用关键词的 snippet（更精确）
                snippet = base_result.snippet
                if scores["keyword_result"] and scores["keyword_result"].snippet:
                    snippet = scores["keyword_result"].snippet
                
                merged.append(SearchResult(
                    doc_id=doc_id,
                    url=base_result.url,
                    title=base_result.title,
                    score=hybrid_score,
                    snippet=snippet,
                    match_type="hybrid",
                ))
        
        # 按混合分数排序
        merged.sort(key=lambda r: r.score, reverse=True)
        
        return merged
    
    def _convert_vector_results(
        self,
        vector_results: list[VectorSearchResult],
        match_type: str = "semantic",
    ) -> list[SearchResult]:
        """将向量搜索结果转换为 SearchResult 格式
        
        Args:
            vector_results: 向量搜索结果列表
            match_type: 匹配类型
            
        Returns:
            SearchResult 列表
        """
        results: list[SearchResult] = []
        
        for vr in vector_results:
            # 提取元数据
            title = vr.metadata.get("title", "")
            url = vr.metadata.get("url", vr.metadata.get("doc_url", ""))
            
            # 使用分块内容作为 snippet（截断到合理长度）
            snippet = vr.content[:200] + "..." if len(vr.content) > 200 else vr.content
            
            results.append(SearchResult(
                doc_id=vr.doc_id,
                url=url,
                title=title,
                score=vr.score,
                snippet=snippet,
                match_type=match_type,
            ))
        
        return results
    
    def _tokenize(self, text: str) -> list[str]:
        """分词
        
        简单的分词实现，支持中英文。
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
        # 移除标点符号，转小写
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())
        
        # 按空格分割，过滤空字符串和短词
        terms = [t.strip() for t in text.split() if t.strip() and len(t.strip()) > 1]
        
        return terms
    
    def _terms_match(self, terms: list[str], text: str) -> bool:
        """检查是否有词匹配
        
        Args:
            terms: 词列表
            text: 目标文本（小写）
            
        Returns:
            是否有任何词匹配
        """
        return any(term in text for term in terms)
    
    def _calculate_term_match_ratio(self, terms: list[str], text: str) -> float:
        """计算词匹配比例
        
        Args:
            terms: 词列表
            text: 目标文本（小写）
            
        Returns:
            匹配词数量占总词数的比例
        """
        if not terms:
            return 0.0
        
        matched = sum(1 for term in terms if term in text)
        return matched / len(terms)
    
    def _extract_snippet(
        self,
        content: str,
        query: str,
        max_length: int = 150,
    ) -> str:
        """提取匹配片段
        
        Args:
            content: 文档内容
            query: 搜索查询
            max_length: 片段最大长度
            
        Returns:
            包含匹配内容的片段
        """
        query_lower = query.lower()
        content_lower = content.lower()
        
        index = content_lower.find(query_lower)
        if index < 0:
            # 未找到，返回开头部分
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # 计算片段范围（以匹配位置为中心）
        start = max(0, index - max_length // 3)
        end = min(len(content), index + len(query) + max_length * 2 // 3)
        
        snippet = content[start:end]
        
        # 添加省略号
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _extract_snippet_by_terms(
        self,
        content: str,
        terms: list[str],
        max_length: int = 150,
    ) -> str:
        """根据词列表提取匹配片段
        
        Args:
            content: 文档内容
            terms: 搜索词列表
            max_length: 片段最大长度
            
        Returns:
            包含匹配内容的片段
        """
        content_lower = content.lower()
        
        # 找到第一个匹配词的位置
        first_match = len(content)
        for term in terms:
            index = content_lower.find(term)
            if 0 <= index < first_match:
                first_match = index
        
        if first_match >= len(content):
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # 计算片段范围
        start = max(0, first_match - max_length // 3)
        end = min(len(content), first_match + max_length * 2 // 3)
        
        snippet = content[start:end]
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    @property
    def is_ready(self) -> bool:
        """检查搜索引擎是否就绪"""
        return self.vector_store.is_initialized
