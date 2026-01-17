"""测试向量存储

使用 mock 替代真实的 ChromaDB，加快测试速度
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import tempfile
import os

from indexing.vector_store import (
    ChromaVectorStore,
    create_vector_store,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
)
from indexing.base import CodeChunk, ChunkType, SearchResult
from indexing.config import VectorStoreConfig, VectorStoreType


class MockChromaCollection:
    """Mock ChromaDB 集合"""
    
    def __init__(self, name="test", metadata=None):
        self.name = name
        self.metadata = metadata or {}
        self._data = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
        }
    
    def add(self, ids, embeddings, documents, metadatas):
        self._data["ids"].extend(ids)
        self._data["embeddings"].extend(embeddings)
        self._data["documents"].extend(documents)
        self._data["metadatas"].extend(metadatas)
    
    def query(self, query_embeddings, n_results=10, include=None, where=None):
        # 简化的查询实现：返回前 n_results 条
        n = min(n_results, len(self._data["ids"]))
        return {
            "ids": [self._data["ids"][:n]],
            "documents": [self._data["documents"][:n]],
            "metadatas": [self._data["metadatas"][:n]],
            "distances": [[0.1 * i for i in range(n)]],
        }
    
    def get(self, ids=None, include=None):
        result = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        for i, id_ in enumerate(self._data["ids"]):
            if ids is None or id_ in ids:
                result["ids"].append(id_)
                result["documents"].append(self._data["documents"][i])
                result["metadatas"].append(self._data["metadatas"][i])
                result["embeddings"].append(self._data["embeddings"][i])
        return result
    
    def delete(self, ids=None, where=None):
        if ids:
            for id_ in ids:
                if id_ in self._data["ids"]:
                    idx = self._data["ids"].index(id_)
                    for key in self._data:
                        self._data[key].pop(idx)
    
    def update(self, ids, embeddings, documents, metadatas):
        for i, id_ in enumerate(ids):
            if id_ in self._data["ids"]:
                idx = self._data["ids"].index(id_)
                self._data["embeddings"][idx] = embeddings[i]
                self._data["documents"][idx] = documents[i]
                self._data["metadatas"][idx] = metadatas[i]
    
    def upsert(self, ids, embeddings, documents, metadatas):
        for i, id_ in enumerate(ids):
            if id_ in self._data["ids"]:
                self.update([id_], [embeddings[i]], [documents[i]], [metadatas[i]])
            else:
                self.add([id_], [embeddings[i]], [documents[i]], [metadatas[i]])
    
    def count(self):
        return len(self._data["ids"])


class MockChromaClient:
    """Mock ChromaDB 客户端"""
    
    def __init__(self):
        self._collections = {}
    
    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = MockChromaCollection(name, metadata)
        return self._collections[name]
    
    def get_collection(self, name):
        return self._collections.get(name)
    
    def create_collection(self, name, metadata=None):
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")
        self._collections[name] = MockChromaCollection(name, metadata)
        return self._collections[name]
    
    def delete_collection(self, name):
        if name in self._collections:
            del self._collections[name]
    
    def list_collections(self):
        return [MagicMock(name=n) for n in self._collections.keys()]


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB"""
    with patch('indexing.vector_store.chromadb') as mock_chroma:
        mock_client = MockChromaClient()
        mock_chroma.Client.return_value = mock_client
        mock_chroma.PersistentClient.return_value = mock_client
        
        # Mock Settings
        mock_chroma.config.Settings = MagicMock
        
        yield mock_chroma, mock_client


def create_test_chunk(
    content: str = "def test(): pass",
    file_path: str = "test.py",
    embedding: list = None,
) -> CodeChunk:
    """创建测试用的 CodeChunk"""
    chunk = CodeChunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=1,
        chunk_type=ChunkType.FUNCTION,
        language="python",
        name="test",
    )
    if embedding is None:
        embedding = [0.1] * 384
    chunk.embedding = embedding
    return chunk


class TestChromaVectorStoreInit:
    """测试 ChromaVectorStore 初始化"""
    
    def test_memory_mode_init(self, mock_chromadb):
        """测试内存模式初始化"""
        mock_chroma, mock_client = mock_chromadb
        
        store = ChromaVectorStore(
            persist_directory=None,
            collection_name="test_collection",
        )
        
        assert not store.is_persistent
        assert store.collection_name == "test_collection"
    
    def test_persistent_mode_init(self, mock_chromadb):
        """测试持久化模式初始化"""
        mock_chroma, mock_client = mock_chromadb
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(
                persist_directory=tmpdir,
                collection_name="test_collection",
            )
            
            assert store.is_persistent
            assert store.persist_directory == tmpdir
    
    def test_default_values(self, mock_chromadb):
        """测试默认值"""
        mock_chroma, mock_client = mock_chromadb
        
        store = ChromaVectorStore()
        
        assert store.collection_name == DEFAULT_COLLECTION_NAME


class TestChromaVectorStoreAdd:
    """测试添加功能"""
    
    @pytest.mark.asyncio
    async def test_add_single_chunk(self, mock_chromadb):
        """测试添加单个分块"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunk = create_test_chunk()
        ids = await store.add([chunk])
        
        assert len(ids) == 1
        assert ids[0] == chunk.chunk_id
    
    @pytest.mark.asyncio
    async def test_add_multiple_chunks(self, mock_chromadb):
        """测试添加多个分块"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [
            create_test_chunk(content="def func1(): pass", embedding=[0.1] * 384),
            create_test_chunk(content="def func2(): pass", embedding=[0.2] * 384),
            create_test_chunk(content="def func3(): pass", embedding=[0.3] * 384),
        ]
        
        ids = await store.add(chunks)
        
        assert len(ids) == 3
    
    @pytest.mark.asyncio
    async def test_add_empty_list(self, mock_chromadb):
        """测试添加空列表"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        ids = await store.add([])
        
        assert ids == []
    
    @pytest.mark.asyncio
    async def test_add_without_embedding_raises(self, mock_chromadb):
        """测试添加无嵌入的分块应报错"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunk = CodeChunk(
            content="def test(): pass",
            file_path="test.py",
        )
        # 不设置 embedding
        
        with pytest.raises(ValueError, match="没有向量嵌入"):
            await store.add([chunk])


class TestChromaVectorStoreSearch:
    """测试搜索功能"""
    
    @pytest.mark.asyncio
    async def test_basic_search(self, mock_chromadb):
        """测试基本搜索"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        # 添加测试数据
        chunks = [create_test_chunk() for _ in range(3)]
        await store.add(chunks)
        
        # 搜索
        query_embedding = [0.1] * 384
        results = await store.search(query_embedding, top_k=5)
        
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, mock_chromadb):
        """测试带过滤条件的搜索"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        # 添加测试数据
        chunks = [
            create_test_chunk(file_path="a.py"),
            create_test_chunk(file_path="b.py"),
        ]
        await store.add(chunks)
        
        # 带过滤条件搜索
        query_embedding = [0.1] * 384
        results = await store.search(
            query_embedding,
            top_k=5,
            filter_dict={"file_path": "a.py"},
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(self, mock_chromadb):
        """测试带阈值的搜索"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [create_test_chunk()]
        await store.add(chunks)
        
        query_embedding = [0.1] * 384
        results = await store.search(
            query_embedding,
            top_k=5,
            threshold=0.8,
        )
        
        # 所有结果的分数应该 >= 阈值
        for r in results:
            assert r.score >= 0.8
    
    @pytest.mark.asyncio
    async def test_search_empty_store(self, mock_chromadb):
        """测试空存储搜索"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        query_embedding = [0.1] * 384
        results = await store.search(query_embedding, top_k=5)
        
        assert results == []


class TestChromaVectorStoreDelete:
    """测试删除功能"""
    
    @pytest.mark.asyncio
    async def test_delete_by_ids(self, mock_chromadb):
        """测试按 ID 删除"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [create_test_chunk() for _ in range(3)]
        ids = await store.add(chunks)
        
        # 删除第一个
        deleted = await store.delete([ids[0]])
        
        assert deleted >= 0
    
    @pytest.mark.asyncio
    async def test_delete_empty_list(self, mock_chromadb):
        """测试删除空列表"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        deleted = await store.delete([])
        
        assert deleted == 0
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, mock_chromadb):
        """测试按条件删除"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [
            create_test_chunk(file_path="a.py"),
            create_test_chunk(file_path="b.py"),
        ]
        await store.add(chunks)
        
        deleted = await store.delete_by_filter({"file_path": "a.py"})
        
        assert deleted >= 0


class TestChromaVectorStorePersistence:
    """测试持久化功能"""
    
    @pytest.mark.asyncio
    async def test_persist(self, mock_chromadb):
        """测试持久化"""
        mock_chroma, mock_client = mock_chromadb
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(persist_directory=tmpdir)
            
            chunks = [create_test_chunk()]
            await store.add(chunks)
            
            # 持久化（ChromaDB PersistentClient 自动持久化）
            await store.persist()
    
    @pytest.mark.asyncio
    async def test_load(self, mock_chromadb):
        """测试加载"""
        mock_chroma, mock_client = mock_chromadb
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(persist_directory=tmpdir)
            
            await store.load()
    
    @pytest.mark.asyncio
    async def test_clear(self, mock_chromadb):
        """测试清空"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [create_test_chunk()]
        await store.add(chunks)
        
        await store.clear()
        
        count = await store.count()
        assert count == 0


class TestChromaVectorStoreCount:
    """测试计数功能"""
    
    @pytest.mark.asyncio
    async def test_count_empty(self, mock_chromadb):
        """测试空存储计数"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        count = await store.count()
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_count_with_data(self, mock_chromadb):
        """测试有数据的计数"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [create_test_chunk() for _ in range(5)]
        await store.add(chunks)
        
        count = await store.count()
        
        assert count == 5


class TestChromaVectorStoreMetadata:
    """测试元数据处理"""
    
    def test_chunk_to_metadata(self, mock_chromadb):
        """测试 CodeChunk 转元数据"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunk = CodeChunk(
            content="def test(): pass",
            file_path="test.py",
            start_line=10,
            end_line=15,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            name="test",
            docstring="Test function",
        )
        chunk.embedding = [0.1] * 384
        
        metadata = store._chunk_to_metadata(chunk)
        
        assert metadata["file_path"] == "test.py"
        assert metadata["start_line"] == 10
        assert metadata["end_line"] == 15
        assert metadata["language"] == "python"
        assert metadata["name"] == "test"
    
    def test_metadata_to_chunk_fields(self, mock_chromadb):
        """测试元数据转 CodeChunk 字段"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        metadata = {
            "file_path": "test.py",
            "start_line": 10,
            "end_line": 15,
            "chunk_type": "function",
            "language": "python",
            "name": "test",
        }
        
        fields = store._metadata_to_chunk_fields(metadata)
        
        assert fields["file_path"] == "test.py"
        assert fields["start_line"] == 10
        assert fields["chunk_type"] == ChunkType.FUNCTION


class TestCollectionManagement:
    """测试集合管理"""
    
    def test_list_collections(self, mock_chromadb):
        """测试列出集合"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore(collection_name="test1")
        
        collections = store.list_collections()
        
        assert isinstance(collections, list)
    
    def test_switch_collection(self, mock_chromadb):
        """测试切换集合"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore(collection_name="collection1")
        
        store.switch_collection("collection2")
        
        assert store.collection_name == "collection2"
    
    def test_get_collection_info(self, mock_chromadb):
        """测试获取集合信息"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        info = store.get_collection_info()
        
        assert "name" in info
        assert "count" in info
    
    def test_get_stats(self, mock_chromadb):
        """测试获取统计信息"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        stats = store.get_stats()
        
        assert "current_collection" in stats
        assert "is_persistent" in stats
        assert "metric" in stats


class TestVectorStoreFactory:
    """测试向量存储工厂"""
    
    def test_create_chromadb_store(self, mock_chromadb):
        """测试创建 ChromaDB 存储"""
        mock_chroma, mock_client = mock_chromadb
        
        config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            collection_name="test",
        )
        
        store = create_vector_store(config)
        
        assert isinstance(store, ChromaVectorStore)
    
    def test_create_memory_store(self, mock_chromadb):
        """测试创建内存存储"""
        mock_chroma, mock_client = mock_chromadb
        
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
        )
        
        store = create_vector_store(config)
        
        assert isinstance(store, ChromaVectorStore)
        assert not store.is_persistent
    
    def test_faiss_not_implemented(self):
        """测试 FAISS 未实现"""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
        )
        
        with pytest.raises(NotImplementedError):
            create_vector_store(config)
    
    def test_qdrant_not_implemented(self):
        """测试 Qdrant 未实现"""
        config = VectorStoreConfig(
            store_type=VectorStoreType.QDRANT,
        )
        
        with pytest.raises(NotImplementedError):
            create_vector_store(config)


class TestChromaVectorStoreAdvanced:
    """测试高级功能"""
    
    @pytest.mark.asyncio
    async def test_get_by_ids(self, mock_chromadb):
        """测试按 ID 获取"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunks = [create_test_chunk() for _ in range(3)]
        ids = await store.add(chunks)
        
        retrieved = await store.get_by_ids([ids[0], ids[1]])
        
        assert len(retrieved) == 2
    
    @pytest.mark.asyncio
    async def test_update(self, mock_chromadb):
        """测试更新"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunk = create_test_chunk()
        await store.add([chunk])
        
        # 更新内容
        chunk.content = "def updated(): pass"
        updated = await store.update([chunk])
        
        assert updated == 1
    
    @pytest.mark.asyncio
    async def test_upsert(self, mock_chromadb):
        """测试 upsert"""
        mock_chroma, mock_client = mock_chromadb
        store = ChromaVectorStore()
        
        chunk = create_test_chunk()
        
        # 第一次 upsert（插入）
        count1 = await store.upsert([chunk])
        assert count1 == 1
        
        # 第二次 upsert（更新）
        chunk.content = "def updated(): pass"
        count2 = await store.upsert([chunk])
        assert count2 == 1
