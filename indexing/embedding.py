"""Embedding 模型实现

提供基于 SentenceTransformers 的本地嵌入模型实现
"""
import asyncio
import hashlib
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .base import EmbeddingModel
from .config import EmbeddingConfig, EmbeddingProvider

# 为单元测试提供可 patch 的模块级依赖占位符
# - 测试会 patch `indexing.embedding.torch` / `indexing.embedding.SentenceTransformer`
# - 运行时会在需要时延迟导入真实依赖
torch = None
SentenceTransformer = None


# 预定义的模型配置
MODEL_CONFIGS = {
    # 快速模型，适合大规模索引
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "快速轻量，适合大规模索引",
        "max_seq_length": 256,
    },
    # 高精度模型，适合需要更好语义理解的场景
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "更高精度，适合语义理解要求高的场景",
        "max_seq_length": 384,
    },
    # 代码专用模型
    "microsoft/codebert-base": {
        "dimension": 768,
        "description": "微软 CodeBERT，针对代码优化",
        "max_seq_length": 512,
    },
    "microsoft/graphcodebert-base": {
        "dimension": 768,
        "description": "微软 GraphCodeBERT，支持代码结构理解",
        "max_seq_length": 512,
    },
    # 多语言模型
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "description": "多语言支持，包括中文",
        "max_seq_length": 128,
    },
}

# 默认模型
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingCache:
    """Embedding 缓存

    使用 LRU 缓存避免重复计算，支持持久化到本地文件

    Attributes:
        max_size: 缓存最大条目数
        cache_dir: 缓存持久化目录（可选）
    """

    def __init__(
        self,
        max_size: int = 10000,
        cache_dir: Optional[str] = None,
    ):
        """初始化缓存

        Args:
            max_size: 缓存最大条目数
            cache_dir: 缓存持久化目录
        """
        self.max_size = max_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

        # 如果指定了缓存目录，尝试加载已有缓存
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _compute_key(self, text: str, model_name: str) -> str:
        """计算缓存键

        使用 MD5 哈希来生成固定长度的键

        Args:
            text: 输入文本
            model_name: 模型名称

        Returns:
            缓存键
        """
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[list[float]]:
        """获取缓存的嵌入向量

        Args:
            text: 输入文本
            model_name: 模型名称

        Returns:
            缓存的嵌入向量，不存在则返回 None
        """
        key = self._compute_key(text, model_name)
        if key in self._cache:
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, model_name: str, embedding: list[float]) -> None:
        """存储嵌入向量到缓存

        Args:
            text: 输入文本
            model_name: 模型名称
            embedding: 嵌入向量
        """
        key = self._compute_key(text, model_name)

        # 如果已存在，更新并移到末尾
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = embedding
            return

        # 如果缓存已满，删除最旧的条目
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding

    def get_batch(
        self,
        texts: list[str],
        model_name: str
    ) -> tuple[list[Optional[list[float]]], list[int]]:
        """批量获取缓存

        Args:
            texts: 输入文本列表
            model_name: 模型名称

        Returns:
            (缓存结果列表, 未命中的索引列表)
        """
        results: list[Optional[list[float]]] = []
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model_name)
            results.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return results, miss_indices

    def put_batch(
        self,
        texts: list[str],
        model_name: str,
        embeddings: list[list[float]]
    ) -> None:
        """批量存储缓存

        Args:
            texts: 输入文本列表
            model_name: 模型名称
            embeddings: 嵌入向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, model_name, embedding)

    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        assert self.cache_dir is not None
        return self.cache_dir / "embedding_cache.pkl"

    def _load_from_disk(self) -> None:
        """从磁盘加载缓存"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    self._cache = OrderedDict(data.get("cache", {}))
                    self._hits = data.get("hits", 0)
                    self._misses = data.get("misses", 0)
                logger.info(f"已从磁盘加载 {len(self._cache)} 条缓存记录")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self._cache = OrderedDict()

    def save_to_disk(self) -> None:
        """保存缓存到磁盘"""
        if self.cache_dir is None:
            logger.warning("未指定缓存目录，无法持久化")
            return

        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "cache": dict(self._cache),
                    "hits": self._hits,
                    "misses": self._misses,
                }, f)
            logger.info(f"已保存 {len(self._cache)} 条缓存记录到磁盘")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息

        Returns:
            统计信息字典
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple[str, str]) -> bool:
        text, model_name = key
        return self._compute_key(text, model_name) in self._cache


class SentenceTransformerEmbedding(EmbeddingModel):
    """基于 SentenceTransformers 的嵌入模型

    使用本地 SentenceTransformers 模型生成文本嵌入向量。
    支持 GPU/CPU 自动选择，提供可选的缓存功能。

    Attributes:
        model_name: 模型名称
        device: 运行设备
        cache: 嵌入缓存（可选）

    Example:
        >>> embedding = SentenceTransformerEmbedding()
        >>> vector = await embedding.embed_text("def hello(): pass")
        >>> len(vector)
        384
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache: Optional[EmbeddingCache] = None,
        show_progress: bool = False,
    ):
        """初始化 SentenceTransformer 嵌入模型

        Args:
            model_name: 模型名称，可选：
                - all-MiniLM-L6-v2: 快速，384维（默认）
                - all-mpnet-base-v2: 更准确，768维
                - microsoft/codebert-base: 代码专用
                - microsoft/graphcodebert-base: 代码结构理解
            device: 运行设备，None 时自动选择（优先 CUDA）
            batch_size: 批量处理大小
            cache: 嵌入缓存实例
            show_progress: 批量处理时是否显示进度条
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache = cache
        self._show_progress = show_progress

        # 延迟导入（同时兼容测试 patch）
        global torch, SentenceTransformer
        if torch is None or SentenceTransformer is None:
            try:
                import torch as _torch
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "请安装 sentence-transformers: pip install sentence-transformers"
                ) from e
            torch = _torch
            SentenceTransformer = _SentenceTransformer

        # 自动选择设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # 获取模型配置
        if model_name in MODEL_CONFIGS:
            self._dimension = MODEL_CONFIGS[model_name]["dimension"]
        else:
            # 对于自定义模型，稍后从模型中获取维度
            self._dimension = None

        # 加载模型
        logger.info(f"正在加载嵌入模型: {model_name} (device={device})")
        self._model = SentenceTransformer(model_name, device=device)

        # 如果维度未知，从模型获取
        if self._dimension is None:
            self._dimension = self._model.get_sentence_embedding_dimension()

        # 用于异步执行的线程池
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(
            f"嵌入模型已加载: {model_name}, "
            f"维度={self._dimension}, 设备={device}"
        )

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension

    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model_name

    @property
    def device(self) -> str:
        """运行设备"""
        return self._device

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """同步编码文本

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=self._show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    async def embed_text(self, text: str) -> list[float]:
        """生成单个文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            向量嵌入
        """
        # 检查缓存
        if self._cache is not None:
            cached = self._cache.get(text, self._model_name)
            if cached is not None:
                return cached

        # 在线程池中执行同步编码
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self._executor,
            self._encode_sync,
            [text]
        )
        embedding = embeddings[0]

        # 存入缓存
        if self._cache is not None:
            self._cache.put(text, self._model_name, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的向量嵌入

        Args:
            texts: 输入文本列表

        Returns:
            向量嵌入列表
        """
        if not texts:
            return []

        # 检查缓存
        if self._cache is not None:
            cached_results, miss_indices = self._cache.get_batch(
                texts, self._model_name
            )

            # 如果全部命中缓存
            if not miss_indices:
                return [r for r in cached_results if r is not None]

            # 只编码未命中的文本
            texts_to_encode = [texts[i] for i in miss_indices]
        else:
            cached_results = [None] * len(texts)
            miss_indices = list(range(len(texts)))
            texts_to_encode = texts

        # 在线程池中执行同步编码
        loop = asyncio.get_event_loop()
        new_embeddings = await loop.run_in_executor(
            self._executor,
            self._encode_sync,
            texts_to_encode
        )

        # 合并结果
        results: list[list[float]] = []
        new_idx = 0
        for i in range(len(texts)):
            if cached_results[i] is not None:
                results.append(cached_results[i])
            else:
                embedding = new_embeddings[new_idx]
                results.append(embedding)
                # 存入缓存
                if self._cache is not None:
                    self._cache.put(texts[i], self._model_name, embedding)
                new_idx += 1

        return results

    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息字典
        """
        info = {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "device": self._device,
            "batch_size": self._batch_size,
        }

        # 添加预定义模型的额外信息
        if self._model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[self._model_name]
            info.update({
                "description": config["description"],
                "max_seq_length": config["max_seq_length"],
            })

        # 添加缓存统计
        if self._cache is not None:
            info["cache_stats"] = self._cache.get_stats()

        return info

    def __del__(self):
        """清理资源"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """根据配置创建嵌入模型

    工厂函数，根据配置创建相应的嵌入模型实例

    Args:
        config: 嵌入模型配置

    Returns:
        嵌入模型实例

    Raises:
        ValueError: 不支持的提供商
    """
    if config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        # 确定模型名称
        model_name = config.model_name
        if config.model_path:
            model_name = config.model_path
        elif model_name == "text-embedding-3-small":
            # 默认配置是 OpenAI，切换到本地默认模型
            model_name = DEFAULT_MODEL

        # 创建缓存（如果指定了持久化目录）
        cache = None
        # 可以通过 metadata 或其他方式传递缓存配置

        return SentenceTransformerEmbedding(
            model_name=model_name,
            device=config.device if config.device != "cpu" else None,
            batch_size=config.batch_size,
            cache=cache,
        )

    elif config.provider == EmbeddingProvider.OPENAI:
        # TODO: 实现 OpenAI 嵌入模型
        raise NotImplementedError("OpenAI 嵌入模型尚未实现")

    elif config.provider == EmbeddingProvider.HUGGINGFACE:
        # TODO: 实现 HuggingFace API 嵌入模型
        raise NotImplementedError("HuggingFace API 嵌入模型尚未实现")

    else:
        raise ValueError(f"不支持的嵌入模型提供商: {config.provider}")


def get_available_models() -> dict[str, dict[str, Any]]:
    """获取可用的模型列表

    Returns:
        模型配置字典
    """
    return MODEL_CONFIGS.copy()
