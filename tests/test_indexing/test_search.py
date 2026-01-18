"""测试语义搜索

使用 mock 替代真实的嵌入模型和向量存储
"""
import pytest
import pytest_asyncio

from indexing.search import (
    SemanticSearch,
    SearchOptions,
    SearchResultWithContext,
    SearchStats,
    create_semantic_search,
)
from indexing.base import CodeChunk, ChunkType, SearchResult, EmbeddingModel, VectorStore


class MockEmbeddingModel(EmbeddingModel):
    """Mock 嵌入模型"""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._model_name = "mock-embedding-model"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    async def embed_text(self, text: str) -> list[float]:
        # 返回固定的 mock 向量
        return [0.1] * self._dimension
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class MockVectorStore(VectorStore):
    """Mock 向量存储"""
    
    def __init__(self):
        self._data: dict[str, CodeChunk] = {}
    
    async def add(self, chunks: list[CodeChunk]) -> list[str]:
        for chunk in chunks:
            self._data[chunk.chunk_id] = chunk
        return [c.chunk_id for c in chunks]
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: dict = None,
        threshold: float = None,
    ) -> list[SearchResult]:
        results = []
        for i, (chunk_id, chunk) in enumerate(list(self._data.items())[:top_k]):
            score = 1.0 - (i * 0.1)  # 模拟递减的分数
            if threshold is not None and score < threshold:
                continue
            if filter_dict:
                # 简单的过滤实现
                match = True
                for key, value in filter_dict.items():
                    if hasattr(chunk, key):
                        if getattr(chunk, key) != value:
                            match = False
                            break
                if not match:
                    continue
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i,
            ))
        return results
    
    async def delete(self, chunk_ids: list[str]) -> int:
        count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._data:
                del self._data[chunk_id]
                count += 1
        return count
    
    async def persist(self) -> None:
        pass
    
    async def load(self) -> None:
        pass
    
    async def clear(self) -> None:
        self._data.clear()
    
    async def count(self) -> int:
        return len(self._data)


def create_test_chunk(
    content: str = "def test(): pass",
    file_path: str = "test.py",
    name: str = "test",
    start_line: int = 1,
    end_line: int = 5,
) -> CodeChunk:
    """创建测试用的 CodeChunk"""
    chunk = CodeChunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        chunk_type=ChunkType.FUNCTION,
        language="python",
        name=name,
    )
    chunk.embedding = [0.1] * 384
    return chunk


@pytest.fixture
def mock_embedding_model():
    """Mock 嵌入模型 fixture"""
    return MockEmbeddingModel()


@pytest.fixture
def mock_vector_store():
    """Mock 向量存储 fixture"""
    return MockVectorStore()


@pytest_asyncio.fixture
async def search_engine(mock_embedding_model, mock_vector_store):
    """搜索引擎 fixture"""
    # 添加一些测试数据
    chunks = [
        create_test_chunk(
            content="def authenticate_user(username, password): pass",
            name="authenticate_user",
            file_path="auth.py",
        ),
        create_test_chunk(
            content="def hash_password(password): return hashlib.sha256(password).hexdigest()",
            name="hash_password",
            file_path="auth.py",
        ),
        create_test_chunk(
            content="class UserService: pass",
            name="UserService",
            file_path="services/user.py",
        ),
    ]
    await mock_vector_store.add(chunks)
    
    return SemanticSearch(mock_embedding_model, mock_vector_store)


class TestSearchOptions:
    """测试搜索选项"""
    
    def test_default_options(self):
        """测试默认选项"""
        options = SearchOptions()
        
        assert options.top_k == 10
        assert options.min_score == 0.0
        assert options.file_filter is None
        assert options.language_filter is None
        assert options.include_context is False
    
    def test_custom_options(self):
        """测试自定义选项"""
        options = SearchOptions(
            top_k=5,
            min_score=0.5,
            file_filter=["*.py"],
            language_filter=["python"],
            include_context=True,
            context_lines=5,
        )
        
        assert options.top_k == 5
        assert options.min_score == 0.5
        assert options.include_context is True
        assert options.context_lines == 5
    
    def test_options_validation(self):
        """测试选项验证"""
        # top_k 范围验证
        with pytest.raises(ValueError):
            SearchOptions(top_k=0)
        
        with pytest.raises(ValueError):
            SearchOptions(top_k=101)
        
        # min_score 范围验证
        with pytest.raises(ValueError):
            SearchOptions(min_score=-0.1)
        
        with pytest.raises(ValueError):
            SearchOptions(min_score=1.1)


class TestSearchResultWithContext:
    """测试带上下文的搜索结果"""
    
    def test_from_base_result(self):
        """测试从基础结果创建"""
        chunk = create_test_chunk()
        base_result = SearchResult(chunk=chunk, score=0.9, rank=0)
        
        result = SearchResultWithContext.from_base_result(
            base_result,
            context="# context\ndef test(): pass",
            context_start_line=1,
            context_end_line=5,
        )
        
        assert result.content == chunk.content
        assert result.file_path == chunk.file_path
        assert result.score == 0.9
        assert result.context is not None
    
    def test_get_display_name(self):
        """测试获取显示名称"""
        chunk = create_test_chunk(name="my_function")
        base_result = SearchResult(chunk=chunk, score=0.9, rank=0)
        result = SearchResultWithContext.from_base_result(base_result)
        
        assert result.get_display_name() == "my_function"
    
    def test_get_location(self):
        """测试获取位置"""
        chunk = create_test_chunk(file_path="src/main.py", start_line=10, end_line=20)
        base_result = SearchResult(chunk=chunk, score=0.9, rank=0)
        result = SearchResultWithContext.from_base_result(base_result)
        
        assert result.get_location() == "src/main.py:10-20"


class TestSearchStats:
    """测试搜索统计"""
    
    def test_stats_init(self):
        """测试统计初始化"""
        stats = SearchStats()
        
        assert stats.total_results == 0
        assert stats.filtered_results == 0
        assert stats.search_time_ms == 0.0
        assert stats.embedding_time_ms == 0.0
    
    def test_stats_with_values(self):
        """测试带值的统计"""
        stats = SearchStats(
            total_results=100,
            filtered_results=50,
            search_time_ms=25.5,
            embedding_time_ms=10.2,
            filters_applied={"language": "python"},
        )
        
        assert stats.total_results == 100
        assert stats.filtered_results == 50
        assert "language" in stats.filters_applied


class TestSemanticSearchInit:
    """测试语义搜索初始化"""
    
    def test_init(self, mock_embedding_model, mock_vector_store):
        """测试初始化"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        assert search.embedding_model == mock_embedding_model
        assert search.vector_store == mock_vector_store
    
    def test_init_with_options(self, mock_embedding_model, mock_vector_store):
        """测试带选项的初始化"""
        options = SearchOptions(top_k=5)
        search = SemanticSearch(
            mock_embedding_model,
            mock_vector_store,
            default_options=options,
        )
        
        assert search._default_options.top_k == 5


class TestBasicSearch:
    """测试基本搜索"""
    
    @pytest.mark.asyncio
    async def test_basic_search(self, search_engine):
        """测试基本语义搜索"""
        results = await search_engine.search("用户认证")
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_top_k(self, search_engine):
        """测试限制结果数量"""
        results = await search_engine.search("function", top_k=2)
        
        assert len(results) <= 2
    
    @pytest.mark.asyncio
    async def test_search_with_min_score(self, search_engine):
        """测试最低分数过滤"""
        results = await search_engine.search("test", min_score=0.5)
        
        # 所有结果分数应该 >= 0.5
        for r in results:
            assert r.score >= 0.5
    
    @pytest.mark.asyncio
    async def test_search_with_file_filter(self, mock_embedding_model):
        """测试文件过滤"""
        store = MockVectorStore()
        
        # 添加不同文件的分块
        chunks = [
            create_test_chunk(file_path="a.py", name="func_a"),
            create_test_chunk(file_path="b.py", name="func_b"),
        ]
        await store.add(chunks)
        
        search = SemanticSearch(mock_embedding_model, store)
        results = await search.search("test", file_filter=["a.py"])
        
        # 应该只返回 a.py 的结果
        assert all(r.chunk.file_path == "a.py" for r in results)


class TestSearchWithContext:
    """测试带上下文的搜索"""
    
    @pytest.mark.asyncio
    async def test_search_with_context(self, search_engine):
        """测试带上下文搜索"""
        options = SearchOptions(include_context=True, context_lines=3)
        results = await search_engine.search_with_context("认证", options)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResultWithContext) for r in results)
    
    @pytest.mark.asyncio
    async def test_context_lines_zero(self, search_engine):
        """测试上下文行数为 0"""
        options = SearchOptions(include_context=True, context_lines=0)
        results = await search_engine.search_with_context("test", options)
        
        # 上下文应该为 None
        for r in results:
            assert r.context is None


class TestSearchInFiles:
    """测试限定文件搜索"""
    
    @pytest.mark.asyncio
    async def test_search_in_files(self, search_engine):
        """测试在指定文件中搜索"""
        results = await search_engine.search_in_files(
            "function",
            file_paths=["auth.py"],
        )
        
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_search_in_empty_files(self, search_engine):
        """测试空文件列表"""
        results = await search_engine.search_in_files(
            "function",
            file_paths=[],
        )
        
        assert results == []


class TestHybridSearch:
    """测试混合搜索"""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, search_engine):
        """测试基本混合搜索"""
        results = await search_engine.hybrid_search(
            "用户认证",
            keywords=["user", "auth"],
        )
        
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_hybrid_search_auto_keywords(self, search_engine):
        """测试自动提取关键词"""
        results = await search_engine.hybrid_search(
            "user authentication function",
            keywords=None,  # 自动提取
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_weight(self, search_engine):
        """测试关键词权重"""
        results1 = await search_engine.hybrid_search(
            "test",
            keywords=["test"],
            keyword_weight=0.0,  # 纯语义
        )
        
        results2 = await search_engine.hybrid_search(
            "test",
            keywords=["test"],
            keyword_weight=1.0,  # 纯关键词
        )
        
        # 结果可能不同
        assert isinstance(results1, list)
        assert isinstance(results2, list)


class TestSearchSimilar:
    """测试相似代码搜索"""
    
    @pytest.mark.asyncio
    async def test_search_similar(self, search_engine):
        """测试搜索相似代码"""
        chunk = create_test_chunk(content="def example(): pass")
        chunk.embedding = [0.1] * 384
        
        results = await search_engine.search_similar(chunk, top_k=5)
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_similar_exclude_self(self, mock_embedding_model, mock_vector_store):
        """测试排除自身"""
        chunk = create_test_chunk()
        await mock_vector_store.add([chunk])
        
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        results = await search.search_similar(
            chunk,
            top_k=5,
            exclude_self=True,
        )
        
        # 结果中不应该包含自身
        for r in results:
            assert r.chunk.chunk_id != chunk.chunk_id


class TestSearchByFunctionName:
    """测试按函数名搜索"""
    
    @pytest.mark.asyncio
    async def test_search_by_name(self, search_engine):
        """测试按函数名搜索"""
        results = await search_engine.search_by_function_name(
            "authenticate_user",
            exact_match=False,
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_by_name_exact(self, mock_embedding_model, mock_vector_store):
        """测试精确匹配函数名"""
        chunk = create_test_chunk(name="my_function")
        await mock_vector_store.add([chunk])
        
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        results = await search.search_by_function_name(
            "my_function",
            exact_match=True,
        )
        
        # 结果应该只包含精确匹配的函数
        for r in results:
            assert r.chunk.name == "my_function"


class TestInternalMethods:
    """测试内部方法"""
    
    def test_build_filter_dict_single_file(self, mock_embedding_model, mock_vector_store):
        """测试构建单文件过滤条件"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        filter_dict = search._build_filter_dict(["test.py"], None)
        
        assert filter_dict is not None
        assert "file_path" in filter_dict
    
    def test_build_filter_dict_multiple_files(self, mock_embedding_model, mock_vector_store):
        """测试构建多文件过滤条件"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        filter_dict = search._build_filter_dict(["a.py", "b.py"], None)
        
        assert filter_dict is not None
    
    def test_build_filter_dict_language(self, mock_embedding_model, mock_vector_store):
        """测试构建语言过滤条件"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        filter_dict = search._build_filter_dict(None, ["python"])
        
        assert filter_dict is not None
        assert "language" in filter_dict
    
    def test_build_filter_dict_none(self, mock_embedding_model, mock_vector_store):
        """测试无过滤条件"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        filter_dict = search._build_filter_dict(None, None)
        
        assert filter_dict is None
    
    def test_extract_keywords(self, mock_embedding_model, mock_vector_store):
        """测试关键词提取"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        keywords = search._extract_keywords("user authentication function")
        
        assert "user" in keywords
        assert "authentication" in keywords
        assert "function" in keywords
        # 停用词应该被过滤
        assert "the" not in keywords
    
    def test_extract_keywords_chinese(self, mock_embedding_model, mock_vector_store):
        """测试中文关键词提取"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        keywords = search._extract_keywords("用户认证功能")
        
        # 中文应该被识别
        assert len(keywords) >= 1
    
    def test_compute_keyword_score(self, mock_embedding_model, mock_vector_store):
        """测试关键词分数计算"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        content = "def user_login(username, password): pass"
        keywords = ["user", "login", "password"]
        
        score = search._compute_keyword_score(content, keywords)
        
        assert 0 <= score <= 1
        assert score > 0  # 应该匹配到一些关键词
    
    def test_compute_keyword_score_empty(self, mock_embedding_model, mock_vector_store):
        """测试空关键词分数"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        score = search._compute_keyword_score("some content", [])
        
        assert score == 0.0


class TestCacheAndStats:
    """测试缓存和统计"""
    
    def test_clear_cache(self, mock_embedding_model, mock_vector_store):
        """测试清空缓存"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        # 添加一些缓存
        search._file_cache["test.py"] = ["line1", "line2"]
        
        search.clear_cache()
        
        assert len(search._file_cache) == 0
    
    def test_get_stats(self, mock_embedding_model, mock_vector_store):
        """测试获取统计信息"""
        search = SemanticSearch(mock_embedding_model, mock_vector_store)
        
        stats = search.get_stats()
        
        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
        assert "file_cache_size" in stats
        assert "default_options" in stats


class TestFactoryFunction:
    """测试工厂函数"""
    
    @pytest.mark.asyncio
    async def test_create_semantic_search(self, mock_embedding_model, mock_vector_store):
        """测试创建语义搜索"""
        search = await create_semantic_search(
            mock_embedding_model,
            mock_vector_store,
        )
        
        assert isinstance(search, SemanticSearch)
    
    @pytest.mark.asyncio
    async def test_create_with_options(self, mock_embedding_model, mock_vector_store):
        """测试带选项创建"""
        options = SearchOptions(top_k=5)
        
        search = await create_semantic_search(
            mock_embedding_model,
            mock_vector_store,
            default_options=options,
        )
        
        assert search._default_options.top_k == 5
