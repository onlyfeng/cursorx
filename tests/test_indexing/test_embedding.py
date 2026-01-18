"""测试嵌入模型

使用 mock 替代真实的 SentenceTransformers 模型，加快测试速度
"""
from unittest.mock import MagicMock, patch

import pytest

# numpy 是可选依赖，用于测试中的 mock
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from indexing.config import EmbeddingConfig, EmbeddingProvider
from indexing.embedding import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    EmbeddingCache,
    SentenceTransformerEmbedding,
    create_embedding_model,
    get_available_models,
)


class TestEmbeddingCache:
    """测试嵌入缓存"""

    def test_cache_init(self):
        """测试缓存初始化"""
        cache = EmbeddingCache(max_size=100)
        assert len(cache) == 0
        assert cache.max_size == 100

    def test_cache_put_and_get(self):
        """测试缓存存取"""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]

        # 存入缓存
        cache.put("test text", "test-model", embedding)

        # 从缓存获取
        result = cache.get("test text", "test-model")
        assert result == embedding

        # 获取不存在的
        result = cache.get("other text", "test-model")
        assert result is None

    def test_cache_lru_eviction(self):
        """测试 LRU 缓存淘汰"""
        cache = EmbeddingCache(max_size=3)

        # 添加 3 个条目
        cache.put("text1", "model", [1.0])
        cache.put("text2", "model", [2.0])
        cache.put("text3", "model", [3.0])

        # 缓存已满，再添加一个
        cache.put("text4", "model", [4.0])

        # 最老的应该被淘汰
        assert cache.get("text1", "model") is None
        assert cache.get("text4", "model") == [4.0]

    def test_cache_batch_operations(self):
        """测试批量缓存操作"""
        cache = EmbeddingCache()

        texts = ["text1", "text2", "text3"]
        embeddings = [[1.0], [2.0], [3.0]]

        # 批量存入
        cache.put_batch(texts, "model", embeddings)

        # 批量获取
        results, miss_indices = cache.get_batch(texts, "model")

        assert len(results) == 3
        assert len(miss_indices) == 0
        assert results[0] == [1.0]
        assert results[2] == [3.0]

        # 部分命中测试
        new_texts = ["text1", "text_new", "text3"]
        results, miss_indices = cache.get_batch(new_texts, "model")

        assert miss_indices == [1]
        assert results[0] == [1.0]
        assert results[1] is None

    def test_cache_stats(self):
        """测试缓存统计信息"""
        cache = EmbeddingCache()

        cache.put("text", "model", [1.0])
        cache.get("text", "model")  # 命中
        cache.get("other", "model")  # 未命中

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """测试清空缓存"""
        cache = EmbeddingCache()
        cache.put("text", "model", [1.0])

        cache.clear()

        assert len(cache) == 0
        assert cache.get("text", "model") is None


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy 未安装")
class TestSentenceTransformerEmbedding:
    """测试 SentenceTransformer 嵌入模型（使用 mock）"""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer 模型"""
        with patch('indexing.embedding.SentenceTransformer') as mock_st, \
             patch('indexing.embedding.torch') as mock_torch:

            mock_torch.cuda.is_available.return_value = False

            # 创建 mock 模型
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            # mock encode 方法返回 numpy 数组
            def mock_encode(texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.rand(len(texts), 384)

            mock_model.encode.side_effect = mock_encode
            mock_st.return_value = mock_model

            yield mock_st

    def test_embedding_init(self, mock_sentence_transformer):
        """测试嵌入模型初始化"""
        embedding = SentenceTransformerEmbedding(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        assert embedding.model_name == "all-MiniLM-L6-v2"
        assert embedding.dimension == 384
        assert embedding.device == "cpu"

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_sentence_transformer):
        """测试单文本嵌入"""
        embedding = SentenceTransformerEmbedding()

        result = await embedding.embed_text("def hello(): pass")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_sentence_transformer):
        """测试批量嵌入"""
        embedding = SentenceTransformerEmbedding()

        texts = [
            "def func1(): pass",
            "def func2(): return 1",
            "class MyClass: pass",
        ]

        results = await embedding.embed_batch(texts)

        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self, mock_sentence_transformer):
        """测试空批量嵌入"""
        embedding = SentenceTransformerEmbedding()

        results = await embedding.embed_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_embed_with_cache(self, mock_sentence_transformer):
        """测试带缓存的嵌入"""
        cache = EmbeddingCache()
        embedding = SentenceTransformerEmbedding(cache=cache)

        text = "def cached_func(): pass"

        # 第一次调用
        result1 = await embedding.embed_text(text)

        # 第二次调用（应该命中缓存）
        result2 = await embedding.embed_text(text)

        # 结果应该相同
        assert result1 == result2

        # 缓存应该有一条记录
        assert len(cache) == 1

    def test_get_model_info(self, mock_sentence_transformer):
        """测试获取模型信息"""
        embedding = SentenceTransformerEmbedding()

        info = embedding.get_model_info()

        assert "model_name" in info
        assert "dimension" in info
        assert "device" in info
        assert info["dimension"] == 384


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy 未安装")
class TestEmbeddingFactory:
    """测试嵌入模型工厂函数"""

    @patch('indexing.embedding.SentenceTransformer')
    @patch('indexing.embedding.torch')
    def test_create_sentence_transformer_model(self, mock_torch, mock_st):
        """测试创建 SentenceTransformer 模型"""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        config = EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
        )

        model = create_embedding_model(config)

        assert model is not None
        assert isinstance(model, SentenceTransformerEmbedding)

    def test_openai_not_implemented(self):
        """测试 OpenAI 模型未实现"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
        )

        with pytest.raises(NotImplementedError):
            create_embedding_model(config)

    def test_huggingface_not_implemented(self):
        """测试 HuggingFace 模型未实现"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.HUGGINGFACE,
        )

        with pytest.raises(NotImplementedError):
            create_embedding_model(config)


class TestModelConfigs:
    """测试模型配置"""

    def test_get_available_models(self):
        """测试获取可用模型列表"""
        models = get_available_models()

        assert len(models) > 0
        assert DEFAULT_MODEL in models

    def test_model_config_structure(self):
        """测试模型配置结构"""
        for model_name, config in MODEL_CONFIGS.items():
            assert "dimension" in config
            assert "description" in config
            assert "max_seq_length" in config
            assert isinstance(config["dimension"], int)
            assert config["dimension"] > 0

    def test_default_model_exists(self):
        """测试默认模型存在"""
        assert DEFAULT_MODEL in MODEL_CONFIGS
        assert MODEL_CONFIGS[DEFAULT_MODEL]["dimension"] == 384
