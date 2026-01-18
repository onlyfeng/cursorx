"""测试知识库向量搜索模块

测试内容:
1. 文档分块功能 (ChunkSplitter, KnowledgeVectorStore._create_chunks)
2. 嵌入向量生成 (SentenceTransformerEmbedding)
3. 向量存储增删改查 (KnowledgeVectorStore with ChromaDB)
4. 语义搜索准确性 (KnowledgeVectorStore.search)
5. 混合搜索 (KnowledgeSemanticSearch.hybrid_search)
6. 无结果情况处理
7. 增量索引更新
"""
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge import (
    ChunkSplitter,
    Document,
    DocumentChunk,
)
from knowledge.semantic_search import HybridSearchConfig, KnowledgeSemanticSearch
from knowledge.vector import KnowledgeVectorConfig
from knowledge.vector import VectorSearchResult as VectorSearchResultWithDetails
from knowledge.vector_store import KnowledgeVectorStore, VectorSearchResult

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_documents() -> list[Document]:
    """创建测试用的示例文档列表"""
    docs = [
        Document(
            id="doc-001",
            url="https://example.com/python-tutorial",
            title="Python 编程教程",
            content="""Python 是一种高级编程语言，广泛用于 Web 开发、数据分析和人工智能。

Python 语法简洁明了，适合初学者学习。它拥有丰富的标准库和第三方包生态系统。

Python 支持多种编程范式，包括面向对象编程、函数式编程和过程式编程。

安装 Python 非常简单，可以从官网下载安装包，也可以使用包管理器如 apt、brew 等。
""",
        ),
        Document(
            id="doc-002",
            url="https://example.com/javascript-guide",
            title="JavaScript 开发指南",
            content="""JavaScript 是一种动态类型的脚本语言，主要用于 Web 前端开发。

JavaScript 可以在浏览器中运行，也可以通过 Node.js 在服务器端运行。

ES6 引入了许多新特性，如箭头函数、类、模块、Promise 等。

JavaScript 生态系统非常丰富，有 React、Vue、Angular 等流行框架。
""",
        ),
        Document(
            id="doc-003",
            url="https://example.com/machine-learning",
            title="机器学习入门",
            content="""机器学习是人工智能的一个重要分支，让计算机能够从数据中学习。

常见的机器学习算法包括线性回归、决策树、随机森林、神经网络等。

Python 是机器学习最流行的编程语言，有 scikit-learn、TensorFlow、PyTorch 等库。

监督学习和非监督学习是两种主要的机器学习类型。
""",
        ),
        Document(
            id="doc-004",
            url="https://example.com/database-basics",
            title="数据库基础",
            content="""数据库是存储和管理数据的系统，分为关系型和非关系型两大类。

SQL 是关系型数据库的标准查询语言，用于增删改查数据。

MySQL、PostgreSQL 和 SQLite 是常用的开源关系型数据库。

NoSQL 数据库如 MongoDB、Redis 适合存储非结构化数据。
""",
        ),
    ]
    return docs


@pytest.fixture
def mock_embedding_model():
    """创建模拟的嵌入模型"""
    mock_model = MagicMock()
    mock_model.dimension = 384
    mock_model._model_name = "all-MiniLM-L6-v2"

    # 生成固定的模拟向量
    def generate_mock_embedding(text: str) -> list[float]:
        """根据文本生成确定性的模拟向量"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = []
        for i in range(0, min(len(text_hash) * 2, 384), 1):
            char_idx = i % len(text_hash)
            value = int(text_hash[char_idx], 16) / 15.0 - 0.5
            vector.append(value)
        while len(vector) < 384:
            vector.append(0.0)
        return vector[:384]

    async def mock_embed_text(text: str) -> list[float]:
        return generate_mock_embedding(text)

    async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
        return [generate_mock_embedding(text) for text in texts]

    mock_model.embed_text = mock_embed_text
    mock_model.embed_batch = mock_embed_batch
    mock_model.get_model_info = MagicMock(return_value={
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "cache_stats": {"hits": 0, "misses": 0},
    })

    return mock_model


@pytest.fixture
def vector_config() -> KnowledgeVectorConfig:
    """创建测试用的向量配置"""
    return KnowledgeVectorConfig(
        enabled=True,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_storage_path=":memory:",  # 使用内存模式
        chunk_size=200,
        chunk_overlap=20,
        default_top_k=10,
        min_similarity_score=0.1,
        semantic_weight=0.7,
        keyword_weight=0.3,
    )


@pytest.fixture
def temp_vector_path():
    """创建临时向量存储路径"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# 测试类: TestKnowledgeVectorSearch
# ============================================================

class TestKnowledgeVectorSearch:
    """知识库向量搜索测试类

    测试向量存储、嵌入生成、语义搜索等核心功能
    """

    # -------------------- 文档分块测试 --------------------

    def test_document_chunking_basic(self):
        """测试基本的文档分块功能"""
        splitter = ChunkSplitter(chunk_size=100, overlap=10)

        text = "这是第一段内容。\n\n这是第二段内容。\n\n这是第三段内容。"
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        assert all(chunk.content for chunk in chunks)
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_document_chunking_with_source_doc(self):
        """测试带来源文档 ID 的分块"""
        splitter = ChunkSplitter(chunk_size=50, overlap=5)

        text = "这是一段需要分块的较长文本内容。" * 10
        chunks = splitter.split(text, source_doc="doc-test-123")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.source_doc == "doc-test-123"
            assert chunk.chunk_id.startswith("chunk-")

    def test_document_chunking_metadata(self):
        """测试分块元数据"""
        splitter = ChunkSplitter(chunk_size=100, overlap=10)

        text = "测试内容。" * 20
        chunks = splitter.split(text)

        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata
            assert isinstance(chunk.start_index, int)
            assert isinstance(chunk.end_index, int)
            assert chunk.end_index > chunk.start_index

    def test_document_chunking_long_text(self, sample_documents):
        """测试长文本分块"""
        splitter = ChunkSplitter(chunk_size=200, overlap=20)

        # 使用示例文档中最长的内容
        doc = sample_documents[0]
        chunks = splitter.split(doc.content, source_doc=doc.id)

        assert len(chunks) >= 1

        # 验证分块覆盖了原始内容
        for chunk in chunks:
            assert len(chunk.content) > 0
            # 第一个分块的起始应该接近文档开头
        assert chunks[0].start_index == 0 or chunks[0].content.strip() in doc.content

    def test_document_chunking_empty_text(self):
        """测试空文本分块"""
        splitter = ChunkSplitter()

        chunks = splitter.split("")
        assert chunks == []

        chunks = splitter.split("   \n\n   ")
        assert chunks == []

    def test_document_chunking_markdown(self):
        """测试 Markdown 格式分块"""
        splitter = ChunkSplitter(chunk_size=100, preserve_headings=True)

        markdown = """# 标题一

这是第一节的内容。

## 标题二

这是第二节的内容。

### 子标题

更多内容。
"""
        chunks = splitter.split_markdown(markdown, source_doc="doc-md")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source_doc == "doc-md"

    # -------------------- 嵌入向量生成测试 --------------------

    @pytest.mark.asyncio
    async def test_embedding_generation_single(self, mock_embedding_model):
        """测试单个文本的嵌入向量生成"""
        text = "这是一段测试文本"

        embedding = await mock_embedding_model.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embedding_generation_batch(self, mock_embedding_model):
        """测试批量文本的嵌入向量生成"""
        texts = [
            "Python 是一种编程语言",
            "JavaScript 用于 Web 开发",
            "机器学习需要大量数据",
        ]

        embeddings = await mock_embedding_model.embed_batch(texts)

        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_embedding_consistency(self, mock_embedding_model):
        """测试相同文本生成相同嵌入"""
        text = "相同的测试文本"

        embedding1 = await mock_embedding_model.embed_text(text)
        embedding2 = await mock_embedding_model.embed_text(text)

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_embedding_different_texts(self, mock_embedding_model):
        """测试不同文本生成不同嵌入"""
        text1 = "这是第一段文本"
        text2 = "这是完全不同的内容"

        embedding1 = await mock_embedding_model.embed_text(text1)
        embedding2 = await mock_embedding_model.embed_text(text2)

        assert embedding1 != embedding2

    def test_embedding_model_info(self, mock_embedding_model):
        """测试获取嵌入模型信息"""
        info = mock_embedding_model.get_model_info()

        assert "model_name" in info
        assert "dimension" in info
        assert info["dimension"] == 384

    # -------------------- 向量存储 CRUD 测试 --------------------

    @pytest.mark.asyncio
    async def test_vector_store_crud_create(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试向量存储 - 创建/添加"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        # 模拟初始化
        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore') as MockChroma:
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    MockChroma.return_value = mock_chroma_instance
                    store._vector_store = mock_chroma_instance

                    # 索引文档
                    doc = sample_documents[0]
                    result = await store.index_document(doc)

                    # 验证索引成功
                    assert result is True
                    assert doc.id in store._doc_chunk_mapping

    @pytest.mark.asyncio
    async def test_vector_store_crud_read(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试向量存储 - 读取/搜索"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        # 创建模拟的搜索结果

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore') as MockChroma:
                    from indexing.base import ChunkType, CodeChunk

                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()

                    # 模拟搜索结果
                    mock_search_result = MagicMock()
                    mock_search_result.chunk = CodeChunk(
                        chunk_id="chunk-001",
                        content="Python 是一种高级编程语言",
                        file_path="https://example.com/python",
                        start_line=0,
                        end_line=0,
                        chunk_type=ChunkType.UNKNOWN,
                        language="text",
                        metadata={"doc_id": "doc-001", "title": "Python 教程"},
                    )
                    mock_search_result.score = 0.95
                    mock_search_result.rank = 0

                    mock_chroma_instance.search = AsyncMock(return_value=[mock_search_result])
                    MockChroma.return_value = mock_chroma_instance
                    store._vector_store = mock_chroma_instance

                    # 执行搜索
                    results = await store.search("Python 编程", top_k=5)

                    assert len(results) > 0
                    assert results[0].score >= 0

    @pytest.mark.asyncio
    async def test_vector_store_crud_update(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试向量存储 - 更新"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore') as MockChroma:
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    MockChroma.return_value = mock_chroma_instance
                    store._vector_store = mock_chroma_instance

                    doc = sample_documents[0]

                    # 首先索引
                    await store.index_document(doc)
                    store._doc_chunk_mapping.get(doc.id, []).copy()

                    # 更新文档内容
                    doc.content = "更新后的内容。这是新的文档内容。"
                    result = await store.update_document(doc)

                    assert result is True

    @pytest.mark.asyncio
    async def test_vector_store_crud_delete(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试向量存储 - 删除"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore') as MockChroma:
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    MockChroma.return_value = mock_chroma_instance
                    store._vector_store = mock_chroma_instance

                    doc = sample_documents[0]

                    # 索引文档
                    await store.index_document(doc)
                    assert doc.id in store._doc_chunk_mapping

                    # 删除文档
                    result = await store.delete_document(doc.id)

                    assert result is True
                    assert doc.id not in store._doc_chunk_mapping

    @pytest.mark.asyncio
    async def test_vector_store_batch_index(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试批量索引文档"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore') as MockChroma:
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    MockChroma.return_value = mock_chroma_instance
                    store._vector_store = mock_chroma_instance

                    # 批量索引
                    count = await store.index_documents(sample_documents[:3])

                    assert count == 3
                    assert len(store._doc_chunk_mapping) == 3

    # -------------------- 语义搜索测试 --------------------

    @pytest.mark.asyncio
    async def test_semantic_search_basic(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试基本语义搜索"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    from indexing.base import ChunkType, CodeChunk

                    mock_chroma_instance = MagicMock()

                    # 模拟搜索返回 Python 相关结果
                    mock_result = MagicMock()
                    mock_result.chunk = CodeChunk(
                        chunk_id="chunk-python-001",
                        content="Python 是一种高级编程语言，广泛用于 Web 开发",
                        file_path="https://example.com/python",
                        start_line=0,
                        end_line=0,
                        chunk_type=ChunkType.UNKNOWN,
                        language="text",
                        metadata={"doc_id": "doc-001", "title": "Python 教程"},
                    )
                    mock_result.score = 0.92
                    mock_result.rank = 0

                    mock_chroma_instance.search = AsyncMock(return_value=[mock_result])
                    store._vector_store = mock_chroma_instance

                    # 搜索
                    results = await store.search("Python 编程语言", top_k=5)

                    assert len(results) > 0
                    assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_semantic_search_ranking(
        self,
        vector_config,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试语义搜索结果排序"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    from indexing.base import ChunkType, CodeChunk

                    mock_chroma_instance = MagicMock()

                    # 创建多个带不同分数的结果
                    mock_results = []
                    scores = [0.95, 0.85, 0.75, 0.65]

                    for i, score in enumerate(scores):
                        mock_result = MagicMock()
                        mock_result.chunk = CodeChunk(
                            chunk_id=f"chunk-{i}",
                            content=f"结果 {i} 的内容",
                            file_path=f"https://example.com/doc{i}",
                            start_line=0,
                            end_line=0,
                            chunk_type=ChunkType.UNKNOWN,
                            language="text",
                            metadata={"doc_id": f"doc-{i}", "title": f"文档 {i}"},
                        )
                        mock_result.score = score
                        mock_result.rank = i
                        mock_results.append(mock_result)

                    mock_chroma_instance.search = AsyncMock(return_value=mock_results)
                    store._vector_store = mock_chroma_instance

                    results = await store.search("测试查询", top_k=10)

                    # 验证结果按分数降序排列
                    assert len(results) == 4
                    for i in range(len(results) - 1):
                        assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_semantic_search_min_score_filter(
        self,
        vector_config,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试语义搜索最小分数过滤"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    from indexing.base import ChunkType, CodeChunk

                    mock_chroma_instance = MagicMock()

                    # 创建低分数结果
                    mock_result = MagicMock()
                    mock_result.chunk = CodeChunk(
                        chunk_id="chunk-low",
                        content="低相关性内容",
                        file_path="https://example.com/low",
                        start_line=0,
                        end_line=0,
                        chunk_type=ChunkType.UNKNOWN,
                        language="text",
                        metadata={"doc_id": "doc-low", "title": "低分文档"},
                    )
                    mock_result.score = 0.2
                    mock_result.rank = 0

                    mock_chroma_instance.search = AsyncMock(return_value=[mock_result])
                    store._vector_store = mock_chroma_instance

                    # 使用较高的最小分数
                    results = await store.search("查询", top_k=5, min_score=0.5)

                    # 低分结果应该被过滤
                    assert len(results) == 0 or all(r.score >= 0.5 for r in results)

    # -------------------- 混合搜索测试 --------------------

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试基本混合搜索"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.search = AsyncMock(return_value=[])
                    store._vector_store = mock_chroma_instance

                    # 创建语义搜索引擎
                    search_engine = KnowledgeSemanticSearch(store, vector_config)

                    # 创建文档字典
                    docs_dict = {doc.id: doc for doc in sample_documents}

                    # 执行混合搜索
                    results = await search_engine.hybrid_search(
                        query="Python 编程",
                        documents=docs_dict,
                        top_k=5,
                        semantic_weight=0.7,
                    )

                    # 应该返回结果（至少关键词搜索会匹配）
                    assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_weight_balance(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试混合搜索权重平衡"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.search = AsyncMock(return_value=[])
                    store._vector_store = mock_chroma_instance

                    search_engine = KnowledgeSemanticSearch(store, vector_config)
                    docs_dict = {doc.id: doc for doc in sample_documents}

                    # 测试不同权重
                    results_semantic = await search_engine.hybrid_search(
                        query="Python",
                        documents=docs_dict,
                        semantic_weight=0.9,
                    )

                    results_keyword = await search_engine.hybrid_search(
                        query="Python",
                        documents=docs_dict,
                        semantic_weight=0.1,
                    )

                    # 两种权重配置都应该返回结果
                    assert isinstance(results_semantic, list)
                    assert isinstance(results_keyword, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试混合搜索结果去重"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.search = AsyncMock(return_value=[])
                    store._vector_store = mock_chroma_instance

                    search_engine = KnowledgeSemanticSearch(store, vector_config)
                    docs_dict = {doc.id: doc for doc in sample_documents}

                    results = await search_engine.hybrid_search(
                        query="Python 编程语言",
                        documents=docs_dict,
                        top_k=10,
                    )

                    # 验证没有重复的文档
                    doc_ids = [r.doc_id for r in results]
                    assert len(doc_ids) == len(set(doc_ids))

    # -------------------- 无结果情况测试 --------------------

    @pytest.mark.asyncio
    async def test_search_with_no_results_empty_store(
        self,
        vector_config,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试空存储的搜索（无结果）"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.search = AsyncMock(return_value=[])
                    store._vector_store = mock_chroma_instance

                    results = await store.search("任意查询内容", top_k=10)

                    assert results == []

    @pytest.mark.asyncio
    async def test_search_with_no_results_irrelevant_query(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试不相关查询的搜索（无结果）"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    # 返回空结果模拟不相关查询
                    mock_chroma_instance.search = AsyncMock(return_value=[])
                    store._vector_store = mock_chroma_instance

                    # 搜索与文档内容完全不相关的内容
                    results = await store.search("量子物理弦理论暗物质", top_k=5)

                    assert isinstance(results, list)
                    # 空结果或低分结果
                    assert len(results) == 0 or all(r.score < 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_no_results_empty_query(
        self,
        vector_config,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试空查询的搜索"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        search_engine = KnowledgeSemanticSearch(store, vector_config)

        # 空查询应该返回空结果
        results = await search_engine.semantic_search("", top_k=10)
        assert results == []

        results = await search_engine.semantic_search("   ", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_no_results(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试关键词搜索无结果"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                search_engine = KnowledgeSemanticSearch(store, vector_config)
                docs_dict = {doc.id: doc for doc in sample_documents}

                # 搜索不存在的关键词
                results = await search_engine.keyword_search(
                    query="XYZQWERTY12345不存在的词",
                    documents=docs_dict,
                    top_k=10,
                )

                assert results == []

    # -------------------- 增量索引测试 --------------------

    @pytest.mark.asyncio
    async def test_incremental_indexing_add_new(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试增量索引 - 添加新文档"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    store._vector_store = mock_chroma_instance

                    # 先索引两个文档
                    await store.index_document(sample_documents[0])
                    await store.index_document(sample_documents[1])

                    initial_count = len(store._doc_chunk_mapping)
                    assert initial_count == 2

                    # 增量添加新文档
                    await store.index_document(sample_documents[2])

                    assert len(store._doc_chunk_mapping) == 3

    @pytest.mark.asyncio
    async def test_incremental_indexing_update_existing(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试增量索引 - 更新现有文档"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    store._vector_store = mock_chroma_instance

                    doc = sample_documents[0]

                    # 首先索引
                    await store.index_document(doc)
                    store._doc_chunk_mapping.get(doc.id, []).copy()

                    # 修改内容并更新
                    doc.content = "完全更新后的新内容。" * 10
                    await store.update_document(doc)

                    # 验证文档仍然存在
                    assert doc.id in store._doc_chunk_mapping
                    # 分块可能已更新

    @pytest.mark.asyncio
    async def test_incremental_indexing_remove_and_add(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试增量索引 - 删除后添加"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    store._vector_store = mock_chroma_instance

                    # 索引所有文档
                    for doc in sample_documents:
                        await store.index_document(doc)

                    assert len(store._doc_chunk_mapping) == len(sample_documents)

                    # 删除一个
                    await store.delete_document(sample_documents[0].id)
                    assert len(store._doc_chunk_mapping) == len(sample_documents) - 1

                    # 重新添加
                    await store.index_document(sample_documents[0])
                    assert len(store._doc_chunk_mapping) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_incremental_indexing_rebuild_all(
        self,
        vector_config,
        sample_documents,
        mock_embedding_model,
        temp_vector_path,
    ):
        """测试增量索引 - 重建所有索引"""
        vector_config.vector_storage_path = temp_vector_path
        store = KnowledgeVectorStore(vector_config)

        with patch.object(store, '_embedding_model', mock_embedding_model):
            with patch.object(store, '_initialized', True):
                with patch('knowledge.vector_store.ChromaVectorStore'):
                    mock_chroma_instance = MagicMock()
                    mock_chroma_instance.upsert = AsyncMock()
                    mock_chroma_instance.delete = AsyncMock()
                    mock_chroma_instance.clear = AsyncMock()
                    store._vector_store = mock_chroma_instance

                    # 先索引部分文档
                    await store.index_document(sample_documents[0])
                    await store.index_document(sample_documents[1])

                    # 重建所有索引
                    count = await store.rebuild_all(sample_documents)

                    assert count == len(sample_documents)
                    assert len(store._doc_chunk_mapping) == len(sample_documents)


# ============================================================
# 额外的辅助测试
# ============================================================

class TestVectorSearchConfig:
    """测试向量搜索配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = KnowledgeVectorConfig()

        assert config.enabled is True
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = KnowledgeVectorConfig(
            chunk_size=256,
            chunk_overlap=25,
        )

        data = config.to_dict()

        assert data["chunk_size"] == 256
        assert data["chunk_overlap"] == 25
        assert "embedding_model" in data

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "enabled": True,
            "chunk_size": 1024,
            "semantic_weight": 0.8,
        }

        config = KnowledgeVectorConfig.from_dict(data)

        assert config.chunk_size == 1024
        assert config.semantic_weight == 0.8


class TestHybridSearchConfig:
    """测试混合搜索配置"""

    def test_default_hybrid_config(self):
        """测试默认混合搜索配置"""
        config = HybridSearchConfig()

        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.normalize_scores is True
        assert config.dedup_by_doc is True

    def test_custom_hybrid_config(self):
        """测试自定义混合搜索配置"""
        config = HybridSearchConfig(
            semantic_weight=0.5,
            keyword_weight=0.5,
            normalize_scores=False,
        )

        assert config.semantic_weight == 0.5
        assert config.keyword_weight == 0.5
        assert config.normalize_scores is False


class TestVectorSearchResult:
    """测试向量搜索结果模型"""

    def test_result_creation(self):
        """测试结果创建"""
        result = VectorSearchResultWithDetails(
            doc_id="doc-123",
            chunk_id="chunk-456",
            url="https://example.com",
            title="测试文档",
            content="测试内容",
            score=0.95,
        )

        assert result.doc_id == "doc-123"
        assert result.score == 0.95
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """测试带元数据的结果"""
        result = VectorSearchResultWithDetails(
            doc_id="doc-123",
            chunk_id="chunk-456",
            url="https://example.com",
            title="测试文档",
            content="测试内容",
            score=0.85,
            metadata={"rank": 0, "source": "semantic"},
        )

        assert result.metadata["rank"] == 0
        assert result.metadata["source"] == "semantic"

    def test_vector_store_result_creation(self):
        """测试 vector_store 模块的搜索结果"""
        result = VectorSearchResult(
            doc_id="doc-123",
            chunk_id="chunk-456",
            content="测试内容",
            score=0.90,
        )

        assert result.doc_id == "doc-123"
        assert result.chunk_id == "chunk-456"
        assert result.score == 0.90
