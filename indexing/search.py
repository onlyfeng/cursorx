"""语义搜索实现

提供基于向量嵌入的语义搜索功能，支持多种搜索模式
"""
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .base import CodeChunk, EmbeddingModel, SearchResult as BaseSearchResult, VectorStore


class SearchOptions(BaseModel):
    """搜索选项配置
    
    Attributes:
        top_k: 返回结果数量
        min_score: 最低相似度阈值 (0-1)
        file_filter: 文件路径过滤（支持 glob 模式）
        language_filter: 语言过滤列表
        chunk_types: 代码块类型过滤
        include_context: 是否包含上下文
        context_lines: 上下文行数（上下各多少行）
    """
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    file_filter: Optional[list[str]] = None
    language_filter: Optional[list[str]] = None
    chunk_types: Optional[list[str]] = None
    include_context: bool = False
    context_lines: int = Field(default=3, ge=0, le=20)


class SearchResultWithContext(BaseModel):
    """带上下文的搜索结果
    
    扩展基础搜索结果，添加上下文信息和便捷访问属性
    
    Attributes:
        content: 匹配的代码块内容
        file_path: 文件路径
        start_line: 起始行号
        end_line: 结束行号
        score: 相似度分数 (0-1)
        context: 上下文代码（包含前后行）
        context_start_line: 上下文起始行号
        context_end_line: 上下文结束行号
        chunk: 原始代码分块对象
        name: 代码块名称（函数名/类名等）
        language: 编程语言
        chunk_type: 代码块类型
        metadata: 附加元数据
    """
    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    context: Optional[str] = None
    context_start_line: Optional[int] = None
    context_end_line: Optional[int] = None
    chunk: Optional[CodeChunk] = None
    name: Optional[str] = None
    language: str = "unknown"
    chunk_type: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_base_result(
        cls,
        result: BaseSearchResult,
        context: Optional[str] = None,
        context_start_line: Optional[int] = None,
        context_end_line: Optional[int] = None,
    ) -> "SearchResultWithContext":
        """从基础搜索结果创建带上下文的结果
        
        Args:
            result: 基础搜索结果
            context: 上下文代码
            context_start_line: 上下文起始行
            context_end_line: 上下文结束行
            
        Returns:
            SearchResultWithContext 实例
        """
        chunk = result.chunk
        return cls(
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=result.score,
            context=context,
            context_start_line=context_start_line,
            context_end_line=context_end_line,
            chunk=chunk,
            name=chunk.name,
            language=chunk.language,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
            metadata={**chunk.metadata, **result.metadata},
        )
    
    def get_display_name(self) -> str:
        """获取显示名称"""
        if self.name:
            return self.name
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
    
    def get_location(self) -> str:
        """获取位置字符串"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class SearchStats:
    """搜索统计信息
    
    Attributes:
        total_results: 总结果数（过滤前）
        filtered_results: 过滤后结果数
        search_time_ms: 搜索耗时（毫秒）
        embedding_time_ms: 嵌入生成耗时（毫秒）
        filters_applied: 应用的过滤器
    """
    total_results: int = 0
    filtered_results: int = 0
    search_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    filters_applied: dict[str, Any] = field(default_factory=dict)


class SemanticSearch:
    """语义搜索引擎
    
    基于向量嵌入的代码语义搜索，支持多种搜索模式：
    - 基本语义搜索
    - 带上下文的搜索
    - 限定文件范围搜索
    - 混合搜索（语义 + 关键词）
    
    Attributes:
        embedding_model: 嵌入模型实例
        vector_store: 向量存储实例
    
    Example:
        >>> search = SemanticSearch(embedding_model, vector_store)
        >>> results = await search.search("用户认证逻辑", top_k=5)
        >>> for r in results:
        ...     print(f"{r.file_path}:{r.start_line} - {r.score:.2f}")
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        default_options: Optional[SearchOptions] = None,
    ):
        """初始化语义搜索引擎
        
        Args:
            embedding_model: 嵌入模型实例，用于将查询转换为向量
            vector_store: 向量存储实例，用于相似度搜索
            default_options: 默认搜索选项
        """
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._default_options = default_options or SearchOptions()
        
        # 用于读取文件上下文的缓存
        self._file_cache: dict[str, list[str]] = {}
        
        logger.info(
            f"语义搜索引擎已初始化: "
            f"embedding={embedding_model.model_name}, "
            f"dimension={embedding_model.dimension}"
        )
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        """嵌入模型"""
        return self._embedding_model
    
    @property
    def vector_store(self) -> VectorStore:
        """向量存储"""
        return self._vector_store
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        file_filter: Optional[list[str]] = None,
        language_filter: Optional[list[str]] = None,
    ) -> list[BaseSearchResult]:
        """基本语义搜索
        
        将查询文本转换为向量，在向量存储中搜索最相似的代码块
        
        Args:
            query: 查询文本（自然语言描述或代码片段）
            top_k: 返回结果数量，默认使用 default_options
            min_score: 最低相似度阈值，默认使用 default_options
            file_filter: 文件路径过滤列表
            language_filter: 编程语言过滤列表
            
        Returns:
            搜索结果列表，按相似度降序排列
        """
        # 合并选项
        _top_k = top_k if top_k is not None else self._default_options.top_k
        _min_score = min_score if min_score is not None else self._default_options.min_score
        _file_filter = file_filter if file_filter is not None else self._default_options.file_filter
        _language_filter = language_filter if language_filter is not None else self._default_options.language_filter
        
        # 生成查询向量
        query_embedding = await self._embedding_model.embed_text(query)
        
        # 构建过滤条件
        filter_dict = self._build_filter_dict(_file_filter, _language_filter)
        
        # 执行向量搜索
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=_top_k,
            filter_dict=filter_dict,
            threshold=_min_score if _min_score > 0 else None,
        )
        
        logger.debug(f"搜索完成: query='{query[:50]}...', results={len(results)}")
        return results
    
    async def search_with_context(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> list[SearchResultWithContext]:
        """带上下文的语义搜索
        
        返回匹配结果及其周围的上下文代码
        
        Args:
            query: 查询文本
            options: 搜索选项，包含 context_lines 等配置
            
        Returns:
            带上下文的搜索结果列表
        """
        opts = options or self._default_options
        
        # 先执行基本搜索
        base_results = await self.search(
            query=query,
            top_k=opts.top_k,
            min_score=opts.min_score,
            file_filter=opts.file_filter,
            language_filter=opts.language_filter,
        )
        
        # 为每个结果添加上下文
        results_with_context = []
        for result in base_results:
            context, ctx_start, ctx_end = await self._get_context(
                file_path=result.chunk.file_path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                context_lines=opts.context_lines,
            )
            
            result_with_ctx = SearchResultWithContext.from_base_result(
                result=result,
                context=context,
                context_start_line=ctx_start,
                context_end_line=ctx_end,
            )
            results_with_context.append(result_with_ctx)
        
        return results_with_context
    
    async def search_in_files(
        self,
        query: str,
        file_paths: list[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[BaseSearchResult]:
        """限定文件范围的语义搜索
        
        只在指定的文件列表中搜索
        
        Args:
            query: 查询文本
            file_paths: 限定的文件路径列表
            top_k: 返回结果数量
            min_score: 最低相似度阈值
            
        Returns:
            搜索结果列表
        """
        if not file_paths:
            logger.warning("文件列表为空，返回空结果")
            return []
        
        # 使用文件路径作为过滤条件
        return await self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            file_filter=file_paths,
        )
    
    async def hybrid_search(
        self,
        query: str,
        keywords: Optional[list[str]] = None,
        options: Optional[SearchOptions] = None,
        keyword_weight: float = 0.3,
    ) -> list[SearchResultWithContext]:
        """混合搜索：结合语义搜索和关键词匹配
        
        综合语义相似度和关键词匹配度计算最终分数
        
        Args:
            query: 查询文本（用于语义搜索）
            keywords: 关键词列表（用于关键词匹配）
            options: 搜索选项
            keyword_weight: 关键词匹配权重 (0-1)，语义权重 = 1 - keyword_weight
            
        Returns:
            混合排序的搜索结果列表
        """
        opts = options or self._default_options
        
        # 如果没有指定关键词，从查询中提取
        if keywords is None:
            keywords = self._extract_keywords(query)
        
        # 执行语义搜索（获取更多结果用于重排序）
        search_top_k = min(opts.top_k * 3, 50)  # 获取更多候选
        base_results = await self.search(
            query=query,
            top_k=search_top_k,
            min_score=opts.min_score,
            file_filter=opts.file_filter,
            language_filter=opts.language_filter,
        )
        
        # 计算混合分数
        scored_results: list[tuple[BaseSearchResult, float]] = []
        for result in base_results:
            semantic_score = result.score
            keyword_score = self._compute_keyword_score(result.chunk.content, keywords)
            
            # 加权平均
            hybrid_score = (
                (1 - keyword_weight) * semantic_score +
                keyword_weight * keyword_score
            )
            scored_results.append((result, hybrid_score))
        
        # 按混合分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top_k 结果并添加上下文
        final_results: list[SearchResultWithContext] = []
        for result, hybrid_score in scored_results[:opts.top_k]:
            context, ctx_start, ctx_end = await self._get_context(
                file_path=result.chunk.file_path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                context_lines=opts.context_lines,
            )
            
            result_with_ctx = SearchResultWithContext.from_base_result(
                result=result,
                context=context,
                context_start_line=ctx_start,
                context_end_line=ctx_end,
            )
            # 更新为混合分数
            result_with_ctx.score = hybrid_score
            result_with_ctx.metadata["semantic_score"] = result.score
            result_with_ctx.metadata["keyword_score"] = self._compute_keyword_score(
                result.chunk.content, keywords
            )
            final_results.append(result_with_ctx)
        
        return final_results
    
    async def search_similar(
        self,
        code_chunk: CodeChunk,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> list[BaseSearchResult]:
        """搜索与给定代码块相似的代码
        
        Args:
            code_chunk: 参考代码块
            top_k: 返回结果数量
            exclude_self: 是否排除自身
            
        Returns:
            相似代码块列表
        """
        # 确保代码块有嵌入向量
        if not code_chunk.has_embedding():
            await self._embedding_model.embed_chunk(code_chunk)
        
        # 搜索相似代码
        results = await self._vector_store.search(
            query_embedding=code_chunk.embedding,
            top_k=top_k + 1 if exclude_self else top_k,
        )
        
        # 排除自身
        if exclude_self:
            results = [
                r for r in results
                if r.chunk.chunk_id != code_chunk.chunk_id
            ][:top_k]
        
        return results
    
    async def search_by_function_name(
        self,
        function_name: str,
        exact_match: bool = False,
        top_k: int = 10,
    ) -> list[BaseSearchResult]:
        """按函数/方法名搜索
        
        Args:
            function_name: 函数名
            exact_match: 是否精确匹配
            top_k: 返回结果数量
            
        Returns:
            匹配的代码块列表
        """
        # 构建搜索查询
        if exact_match:
            query = f"def {function_name}("
        else:
            query = f"function {function_name} implementation"
        
        # 执行搜索
        results = await self.search(query=query, top_k=top_k * 2)
        
        # 如果精确匹配，过滤结果
        if exact_match:
            results = [
                r for r in results
                if r.chunk.name == function_name
            ]
        
        return results[:top_k]
    
    # ==================== 内部方法 ====================
    
    def _build_filter_dict(
        self,
        file_filter: Optional[list[str]],
        language_filter: Optional[list[str]],
    ) -> Optional[dict[str, Any]]:
        """构建向量存储的过滤条件
        
        Args:
            file_filter: 文件路径过滤
            language_filter: 语言过滤
            
        Returns:
            过滤条件字典，如果无过滤则返回 None
        """
        conditions = []
        
        if file_filter:
            if len(file_filter) == 1:
                # 单个文件，精确匹配或前缀匹配
                file_path = file_filter[0]
                if "*" in file_path:
                    # 包含通配符，使用前缀匹配
                    prefix = file_path.replace("*", "")
                    conditions.append({"file_path": {"$contains": prefix}})
                else:
                    conditions.append({"file_path": file_path})
            else:
                # 多个文件，使用 $in
                conditions.append({"file_path": {"$in": file_filter}})
        
        if language_filter:
            if len(language_filter) == 1:
                conditions.append({"language": language_filter[0]})
            else:
                conditions.append({"language": {"$in": language_filter}})
        
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    async def _get_context(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int,
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """获取代码块的上下文
        
        Args:
            file_path: 文件路径
            start_line: 代码块起始行
            end_line: 代码块结束行
            context_lines: 上下文行数
            
        Returns:
            (上下文代码, 上下文起始行, 上下文结束行)
        """
        if context_lines <= 0:
            return None, None, None
        
        try:
            # 读取文件（使用缓存）
            lines = await self._read_file_lines(file_path)
            if not lines:
                return None, None, None
            
            # 计算上下文范围
            ctx_start = max(1, start_line - context_lines)
            ctx_end = min(len(lines), end_line + context_lines)
            
            # 提取上下文（行号从 1 开始）
            context_lines_list = lines[ctx_start - 1:ctx_end]
            context = "\n".join(context_lines_list)
            
            return context, ctx_start, ctx_end
            
        except Exception as e:
            logger.warning(f"读取上下文失败: {file_path}, error={e}")
            return None, None, None
    
    async def _read_file_lines(self, file_path: str) -> list[str]:
        """读取文件行（带缓存）
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件行列表
        """
        # 检查缓存
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        try:
            path = Path(file_path)
            if not path.exists():
                return []
            
            # 异步读取文件
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: path.read_text(encoding="utf-8", errors="ignore")
            )
            lines = content.splitlines()
            
            # 缓存结果（限制缓存大小）
            if len(self._file_cache) < 100:
                self._file_cache[file_path] = lines
            
            return lines
            
        except Exception as e:
            logger.warning(f"读取文件失败: {file_path}, error={e}")
            return []
    
    def _extract_keywords(self, query: str) -> list[str]:
        """从查询中提取关键词
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取：分词 + 过滤停用词
        # 匹配英文单词、驼峰命名、下划线命名等
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[\u4e00-\u9fff]+', query)
        
        # 过滤短词和常见停用词
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'this', 'that', 'these', 'those',
        }
        
        keywords = [
            w.lower() for w in words
            if len(w) > 2 and w.lower() not in stopwords
        ]
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _compute_keyword_score(
        self,
        content: str,
        keywords: list[str],
    ) -> float:
        """计算关键词匹配分数
        
        Args:
            content: 代码内容
            keywords: 关键词列表
            
        Returns:
            匹配分数 (0-1)
        """
        if not keywords:
            return 0.0
        
        content_lower = content.lower()
        matched = sum(1 for kw in keywords if kw in content_lower)
        
        return matched / len(keywords)
    
    def clear_cache(self) -> None:
        """清空文件缓存"""
        self._file_cache.clear()
        logger.debug("文件缓存已清空")
    
    def get_stats(self) -> dict[str, Any]:
        """获取搜索引擎统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "embedding_model": self._embedding_model.model_name,
            "embedding_dimension": self._embedding_model.dimension,
            "file_cache_size": len(self._file_cache),
            "default_options": self._default_options.model_dump(),
        }


async def create_semantic_search(
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    default_options: Optional[SearchOptions] = None,
) -> SemanticSearch:
    """工厂函数：创建语义搜索引擎
    
    Args:
        embedding_model: 嵌入模型实例
        vector_store: 向量存储实例
        default_options: 默认搜索选项
        
    Returns:
        SemanticSearch 实例
    """
    return SemanticSearch(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_options=default_options,
    )
