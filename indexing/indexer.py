"""代码库索引器

提供代码库的索引、更新和管理功能
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .base import CodeChunk, CodeChunker, EmbeddingModel, VectorStore


@dataclass
class FileState:
    """文件状态信息

    用于追踪文件的修改状态，支持增量索引
    """

    file_path: str  # 文件路径
    mtime: float  # 修改时间戳
    content_hash: str  # 内容哈希
    chunk_ids: list[str] = field(default_factory=list)  # 关联的分块 ID
    indexed_at: datetime | None = None  # 索引时间

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "file_path": self.file_path,
            "mtime": self.mtime,
            "content_hash": self.content_hash,
            "chunk_ids": self.chunk_ids,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileState:
        """从字典创建"""
        indexed_at = None
        if data.get("indexed_at"):
            indexed_at = datetime.fromisoformat(data["indexed_at"])
        return cls(
            file_path=data["file_path"],
            mtime=data["mtime"],
            content_hash=data["content_hash"],
            chunk_ids=data.get("chunk_ids", []),
            indexed_at=indexed_at,
        )


@dataclass
class IndexProgress:
    """索引进度信息"""

    total_files: int = 0  # 总文件数
    processed_files: int = 0  # 已处理文件数
    total_chunks: int = 0  # 总分块数
    current_file: str = ""  # 当前处理的文件
    status: str = "pending"  # 状态: pending/indexing/completed/failed
    errors: list[str] = field(default_factory=list)  # 错误列表

    @property
    def progress_percent(self) -> float:
        """进度百分比"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100


# 进度回调函数类型
ProgressCallback = Callable[[IndexProgress], None]


class IndexStateManager:
    """索引状态管理器

    管理文件索引状态，支持持久化和增量索引
    """

    def __init__(self, state_file: Path | None = None):
        """初始化状态管理器

        Args:
            state_file: 状态持久化文件路径
        """
        self._state_file = state_file
        self._file_states: dict[str, FileState] = {}

        # 如果指定了状态文件，尝试加载
        if self._state_file and self._state_file.exists():
            self._load_state()

    def _load_state(self) -> None:
        """从文件加载状态"""
        assert self._state_file is not None
        try:
            with open(self._state_file, encoding="utf-8") as f:
                data = json.load(f)
                for file_path, state_data in data.get("files", {}).items():
                    self._file_states[file_path] = FileState.from_dict(state_data)
            logger.info(f"已加载 {len(self._file_states)} 个文件的索引状态")
        except Exception as e:
            logger.warning(f"加载索引状态失败: {e}")
            self._file_states = {}

    def save_state(self) -> None:
        """保存状态到文件"""
        if self._state_file is None:
            return

        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "files": {fp: state.to_dict() for fp, state in self._file_states.items()},
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"已保存 {len(self._file_states)} 个文件的索引状态")
        except Exception as e:
            logger.error(f"保存索引状态失败: {e}")

    def get_file_state(self, file_path: str) -> FileState | None:
        """获取文件状态"""
        return self._file_states.get(file_path)

    def set_file_state(self, state: FileState) -> None:
        """设置文件状态"""
        self._file_states[state.file_path] = state

    def remove_file_state(self, file_path: str) -> FileState | None:
        """移除文件状态"""
        return self._file_states.pop(file_path, None)

    def get_all_indexed_files(self) -> set[str]:
        """获取所有已索引的文件路径"""
        return set(self._file_states.keys())

    def is_file_changed(self, file_path: str, mtime: float, content_hash: str) -> bool:
        """检查文件是否已变更

        Args:
            file_path: 文件路径
            mtime: 当前修改时间
            content_hash: 当前内容哈希

        Returns:
            True 如果文件已变更或未索引
        """
        state = self._file_states.get(file_path)
        if state is None:
            return True

        # 先检查修改时间（快速判断）
        if state.mtime != mtime:
            # 修改时间变了，再检查内容哈希
            return state.content_hash != content_hash

        return False

    def clear(self) -> None:
        """清空所有状态"""
        self._file_states.clear()


class CodebaseIndexer:
    """代码库索引器

    提供代码库的全量索引、增量更新和文件管理功能

    Attributes:
        root_path: 代码库根目录
        embedding_model: 嵌入模型
        vector_store: 向量存储
        chunker: 代码分块器

    Example:
        >>> indexer = CodebaseIndexer(
        ...     root_path=Path("/path/to/repo"),
        ...     embedding_model=embedding,
        ...     vector_store=store,
        ...     chunker=chunker
        ... )
        >>> await indexer.index_codebase()
    """

    def __init__(
        self,
        root_path: Path | str,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: CodeChunker,
        *,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        respect_gitignore: bool = True,
        state_file: Path | str | None = None,
        max_concurrent: int = 4,
    ):
        """初始化代码库索引器

        Args:
            root_path: 代码库根目录
            embedding_model: 嵌入模型实例
            vector_store: 向量存储实例
            chunker: 代码分块器实例
            include_patterns: 包含的文件模式 (glob)，如 ["*.py", "*.js"]
            exclude_patterns: 排除的文件模式 (glob)，如 ["**/test_*.py"]
            respect_gitignore: 是否遵守 .gitignore 规则
            state_file: 索引状态持久化文件路径
            max_concurrent: 最大并发索引数
        """
        self.root_path = Path(root_path).resolve()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker

        # 文件过滤配置
        self._include_patterns = include_patterns or ["**/*.py", "**/*.js", "**/*.ts"]
        self._exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/*.min.js",
            "**/*.min.css",
        ]
        self._respect_gitignore = respect_gitignore
        self._gitignore_patterns: list[str] = []

        # 并发控制
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # 状态管理
        state_path = Path(state_file) if state_file else None
        self._state_manager = IndexStateManager(state_path)

        # 进度追踪
        self._progress = IndexProgress()
        self._progress_callback: ProgressCallback | None = None

        # 加载 .gitignore 规则
        if self._respect_gitignore:
            self._load_gitignore()

        logger.info(f"CodebaseIndexer 初始化完成: root={self.root_path}, max_concurrent={max_concurrent}")

    def _load_gitignore(self) -> None:
        """加载 .gitignore 规则"""
        gitignore_path = self.root_path / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            with open(gitignore_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith("#"):
                        continue
                    # 转换为 glob 模式
                    line = "**/" + line if not line.startswith("/") else line[1:]
                    # 目录模式
                    if line.endswith("/"):
                        line = line + "**"
                    self._gitignore_patterns.append(line)

            logger.debug(f"已加载 {len(self._gitignore_patterns)} 条 .gitignore 规则")
        except Exception as e:
            logger.warning(f"读取 .gitignore 失败: {e}")

    def _should_include_file(self, file_path: Path) -> bool:
        """判断文件是否应该被索引

        Args:
            file_path: 文件路径

        Returns:
            True 如果文件应该被索引
        """
        # 获取相对路径
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            rel_path = file_path
        rel_path_str = str(rel_path)

        # 检查排除模式
        for pattern in self._exclude_patterns:
            if fnmatch(rel_path_str, pattern):
                return False

        # 检查 .gitignore 规则
        if self._respect_gitignore:
            for pattern in self._gitignore_patterns:
                if fnmatch(rel_path_str, pattern):
                    return False

        # 检查包含模式
        return any(fnmatch(rel_path_str, pattern) for pattern in self._include_patterns)

    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件内容哈希

        Args:
            file_path: 文件路径

        Returns:
            MD5 哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _update_progress(self, **kwargs) -> None:
        """更新进度并触发回调"""
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)

        if self._progress_callback:
            try:
                self._progress_callback(self._progress)
            except Exception as e:
                logger.warning(f"进度回调执行失败: {e}")

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """设置进度回调函数

        Args:
            callback: 进度回调函数，接收 IndexProgress 参数
        """
        self._progress_callback = callback

    def collect_files(self) -> list[Path]:
        """收集需要索引的文件

        Returns:
            文件路径列表
        """
        files: list[Path] = []

        for root, dirs, filenames in os.walk(self.root_path):
            root_path = Path(root)

            # 过滤目录（原地修改以跳过子目录）
            dirs[:] = [d for d in dirs if not d.startswith(".") and not self._is_excluded_dir(root_path / d)]

            for filename in filenames:
                file_path = root_path / filename
                if self._should_include_file(file_path):
                    files.append(file_path)

        logger.info(f"收集到 {len(files)} 个待索引文件")
        return files

    def _is_excluded_dir(self, dir_path: Path) -> bool:
        """检查目录是否应该被排除"""
        try:
            rel_path = dir_path.relative_to(self.root_path)
        except ValueError:
            rel_path = dir_path
        rel_path_str = str(rel_path) + "/"

        for pattern in self._exclude_patterns:
            if fnmatch(rel_path_str, pattern):
                return True

        if self._respect_gitignore:
            for pattern in self._gitignore_patterns:
                if fnmatch(rel_path_str, pattern):
                    return True

        return False

    async def index_file(self, file_path: Path | str) -> list[CodeChunk]:
        """索引单个文件

        Args:
            file_path: 文件路径

        Returns:
            生成的代码分块列表
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.root_path / file_path

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return []

        logger.debug(f"开始索引文件: {file_path}")

        try:
            # 计算文件状态
            mtime = file_path.stat().st_mtime
            content_hash = self._compute_file_hash(file_path)

            # 分块
            chunks = await self.chunker.chunk_file(str(file_path))
            if not chunks:
                logger.debug(f"文件无有效分块: {file_path}")
                return []

            # 生成嵌入
            chunks = await self.embedding_model.embed_chunks(chunks)

            # 存储到向量库
            chunk_ids = await self.vector_store.add(chunks)

            # 更新文件状态
            state = FileState(
                file_path=str(file_path),
                mtime=mtime,
                content_hash=content_hash,
                chunk_ids=chunk_ids,
                indexed_at=datetime.now(),
            )
            self._state_manager.set_file_state(state)

            logger.debug(f"文件索引完成: {file_path}, {len(chunks)} 个分块")
            return chunks

        except Exception as e:
            logger.error(f"索引文件失败 {file_path}: {e}")
            self._progress.errors.append(f"{file_path}: {e}")
            return []

    async def _index_file_with_semaphore(self, file_path: Path) -> list[CodeChunk]:
        """带并发控制的文件索引"""
        async with self._semaphore:
            self._update_progress(current_file=str(file_path))
            chunks = await self.index_file(file_path)
            self._update_progress(
                processed_files=self._progress.processed_files + 1,
                total_chunks=self._progress.total_chunks + len(chunks),
            )
            return chunks

    async def index_directory(self, directory: Path | str, recursive: bool = True) -> list[CodeChunk]:
        """索引目录中的文件

        Args:
            directory: 目录路径
            recursive: 是否递归索引子目录

        Returns:
            所有生成的代码分块列表
        """
        directory = Path(directory)
        if not directory.is_absolute():
            directory = self.root_path / directory

        if not directory.exists() or not directory.is_dir():
            logger.warning(f"目录不存在或不是目录: {directory}")
            return []

        logger.info(f"开始索引目录: {directory}")

        # 收集文件
        files: list[Path] = []
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                root_path = Path(root)
                dirs[:] = [d for d in dirs if not self._is_excluded_dir(root_path / d)]
                for filename in filenames:
                    file_path = root_path / filename
                    if self._should_include_file(file_path):
                        files.append(file_path)
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and self._should_include_file(file_path):
                    files.append(file_path)

        # 初始化进度
        self._progress = IndexProgress(total_files=len(files), processed_files=0, total_chunks=0, status="indexing")
        self._update_progress()

        # 并发索引
        tasks = [self._index_file_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        all_chunks: list[CodeChunk] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"索引任务异常: {result}")
            elif isinstance(result, list) and result:
                all_chunks.extend(result)

        self._update_progress(status="completed")
        logger.info(f"目录索引完成: {directory}, {len(files)} 个文件, {len(all_chunks)} 个分块")

        return all_chunks

    async def index_codebase(
        self, incremental: bool = True, progress_callback: ProgressCallback | None = None
    ) -> list[CodeChunk]:
        """索引整个代码库

        Args:
            incremental: 是否增量索引（只索引变更文件）
            progress_callback: 进度回调函数

        Returns:
            所有生成的代码分块列表
        """
        logger.info(f"开始索引代码库: {self.root_path}, incremental={incremental}")

        if progress_callback:
            self.set_progress_callback(progress_callback)

        # 收集所有需要索引的文件
        all_files = self.collect_files()

        # 如果是增量索引，过滤出需要更新的文件
        files_to_index: list[Path] = []
        if incremental:
            for file_path in all_files:
                try:
                    mtime = file_path.stat().st_mtime
                    content_hash = self._compute_file_hash(file_path)
                    if self._state_manager.is_file_changed(str(file_path), mtime, content_hash):
                        files_to_index.append(file_path)
                except Exception as e:
                    logger.warning(f"检查文件变更失败 {file_path}: {e}")
                    files_to_index.append(file_path)

            logger.info(f"增量索引: {len(files_to_index)}/{len(all_files)} 个文件需要更新")
        else:
            files_to_index = all_files

        # 初始化进度
        self._progress = IndexProgress(
            total_files=len(files_to_index), processed_files=0, total_chunks=0, status="indexing"
        )
        self._update_progress()

        # 并发索引
        tasks = [self._index_file_with_semaphore(f) for f in files_to_index]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        all_chunks: list[CodeChunk] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"索引任务异常: {result}")
            elif isinstance(result, list) and result:
                all_chunks.extend(result)

        # 检测并删除已删除文件的索引
        if incremental:
            await self._cleanup_deleted_files(all_files)

        # 持久化状态和向量存储
        self._state_manager.save_state()
        await self.vector_store.persist()

        self._update_progress(status="completed")
        logger.info(f"代码库索引完成: {len(files_to_index)} 个文件, {len(all_chunks)} 个分块")

        return all_chunks

    async def _cleanup_deleted_files(self, current_files: list[Path]) -> None:
        """清理已删除文件的索引

        Args:
            current_files: 当前存在的文件列表
        """
        current_file_set = {str(f) for f in current_files}
        indexed_files = self._state_manager.get_all_indexed_files()
        deleted_files = indexed_files - current_file_set

        for file_path in deleted_files:
            await self.remove_file(file_path)

        if deleted_files:
            logger.info(f"已清理 {len(deleted_files)} 个已删除文件的索引")

    async def update_file(self, file_path: Path | str) -> list[CodeChunk]:
        """增量更新单个文件

        检测文件是否变更，如果变更则重新索引

        Args:
            file_path: 文件路径

        Returns:
            更新后的代码分块列表（如果未变更返回空列表）
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.root_path / file_path

        if not file_path.exists():
            # 文件已删除，移除索引
            await self.remove_file(str(file_path))
            return []

        # 检查文件是否变更
        try:
            mtime = file_path.stat().st_mtime
            content_hash = self._compute_file_hash(file_path)

            if not self._state_manager.is_file_changed(str(file_path), mtime, content_hash):
                logger.debug(f"文件未变更，跳过: {file_path}")
                return []
        except Exception as e:
            logger.warning(f"检查文件变更失败 {file_path}: {e}")

        # 先删除旧索引
        await self.remove_file(str(file_path))

        # 重新索引
        chunks = await self.index_file(file_path)

        # 保存状态
        self._state_manager.save_state()

        return chunks

    async def remove_file(self, file_path: str) -> bool:
        """从索引中删除文件

        Args:
            file_path: 文件路径

        Returns:
            True 如果成功删除
        """
        state = self._state_manager.get_file_state(file_path)
        if state is None:
            logger.debug(f"文件未在索引中: {file_path}")
            return False

        # 删除向量存储中的分块
        if state.chunk_ids:
            try:
                deleted_count = await self.vector_store.delete(state.chunk_ids)
                logger.debug(f"已删除 {deleted_count} 个分块: {file_path}")
            except Exception as e:
                logger.error(f"删除分块失败 {file_path}: {e}")
                return False

        # 移除状态
        self._state_manager.remove_file_state(file_path)
        logger.info(f"已从索引中删除文件: {file_path}")

        return True

    async def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息

        Returns:
            统计信息字典
        """
        chunk_count = await self.vector_store.count()
        indexed_files = self._state_manager.get_all_indexed_files()

        return {
            "root_path": str(self.root_path),
            "indexed_files": len(indexed_files),
            "total_chunks": chunk_count,
            "embedding_model": self.embedding_model.model_name,
            "include_patterns": self._include_patterns,
            "exclude_patterns": self._exclude_patterns,
        }

    async def clear(self) -> None:
        """清空所有索引"""
        await self.vector_store.clear()
        self._state_manager.clear()
        self._state_manager.save_state()
        logger.info("已清空所有索引")

    def get_progress(self) -> IndexProgress:
        """获取当前索引进度

        Returns:
            当前进度信息
        """
        return self._progress
