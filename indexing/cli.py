#!/usr/bin/env python3
"""代码索引 CLI 工具

提供命令行接口管理代码索引，支持构建、更新、搜索、状态查询等操作

使用方法:
    python -m indexing.cli build [--path PATH] [--full]
    python -m indexing.cli update [--path PATH]
    python -m indexing.cli search <query> [--top-k N]
    python -m indexing.cli status
    python -m indexing.cli clear [--confirm]
    python -m indexing.cli info
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from indexing.base import SearchResult
from indexing.chunker import SemanticCodeChunker
from indexing.config import (
    ChunkConfig,
    ChunkStrategy,
    EmbeddingConfig,
    EmbeddingProvider,
    IndexConfig,
    VectorStoreConfig,
    VectorStoreType,
    extract_search_options,
    normalize_indexing_config,
)
from indexing.embedding import (
    EmbeddingCache,
    SentenceTransformerEmbedding,
    get_available_models,
)
from indexing.indexer import CodebaseIndexer, IndexProgress
from indexing.search import SearchOptions, SearchResultWithContext, SemanticSearch
from indexing.vector_store import ChromaVectorStore

# 默认配置
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_STATE_FILE = ".cursor/index_state.json"


class ProgressBar:
    """简单的进度条显示"""

    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, progress: IndexProgress):
        """更新进度显示"""
        self.current = progress.processed_files
        self._render(progress.current_file)

    def _render(self, current_file: str = ""):
        """渲染进度条"""
        percent = 100 if self.total == 0 else int(self.current / self.total * 100)

        filled = int(self.width * self.current / max(self.total, 1))
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        # 截断文件名
        if len(current_file) > 30:
            current_file = "..." + current_file[-27:]

        sys.stdout.write(
            f"\r{self.prefix} |{bar}| {percent:3d}% ({self.current}/{self.total}) {eta_str} {current_file:<30}"
        )
        sys.stdout.flush()

    def finish(self):
        """完成进度显示"""
        elapsed = time.time() - self.start_time
        print(f"\n完成! 耗时: {elapsed:.1f}s")


def load_config(config_file: Optional[str] = None) -> IndexConfig:
    """加载配置文件

    配置查找策略:
        1. 显式传入的 config_file 路径（最高优先级）
        2. 当 config_file 为 None 时，使用与 ConfigManager 相同的查找策略：
           - 当前目录的 config.yaml
           - 项目根目录的 config.yaml（通过 .git 目录识别）
           - 模块所在目录的 config.yaml
        3. 未找到配置文件时使用默认配置

    支持新旧键名兼容:
        - indexing.model (新) → indexing.embedding_model (旧)
        - indexing.persist_path (新) → indexing.persist_dir (旧)

    回退默认值说明:
        当某些 indexing 子键未配置时，优先从 core.config.get_config().indexing
        读取作为回退默认值，确保与 config.yaml 中定义的默认值一致。
        只有当 core.config 也未提供值时，才使用 indexing 模块自身的硬编码默认值。

    Args:
        config_file: 配置文件路径，为 None 时自动查找

    Returns:
        IndexConfig 实例
    """
    # 确定要使用的配置文件路径
    resolved_config_path: Optional[Path] = None

    if config_file:
        # 显式指定的路径优先
        resolved_config_path = Path(config_file)
    else:
        # 未指定时使用与 ConfigManager 相同的查找策略
        from core.config import find_config_file

        resolved_config_path = find_config_file()
        if resolved_config_path:
            logger.debug(f"自动发现配置文件: {resolved_config_path}")

    # 获取 core.config 中的 indexing 默认值作为回退
    # 这确保与 config.yaml 中定义的默认值一致
    from core.config import get_config

    core_indexing = get_config().indexing

    # 从 core.config.IndexingConfig 获取回退默认值
    fallback_model = core_indexing.model
    fallback_persist_dir = core_indexing.persist_path
    fallback_chunk_size = core_indexing.chunk_size
    fallback_chunk_overlap = core_indexing.chunk_overlap
    fallback_include_patterns = core_indexing.include_patterns
    fallback_exclude_patterns = core_indexing.exclude_patterns

    if resolved_config_path and resolved_config_path.exists():
        try:
            import yaml

            with open(resolved_config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # 从配置文件提取索引相关配置
            raw_index_config = data.get("indexing", {}) if data else {}

            # 标准化键名映射（model → embedding_model, persist_path → persist_dir）
            index_config = normalize_indexing_config(raw_index_config)

            # 提取搜索选项（search.top_k, search.min_score 等）
            # search_options 可以在需要时使用，目前记录到日志供调试
            search_options = extract_search_options(raw_index_config)
            if search_options:
                logger.debug(f"搜索选项: {search_options}")

            # 优先使用配置文件中的值，回退到 core.config 的默认值
            # 只有当 core.config 也未提供时才使用 indexing 模块的硬编码默认值
            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name=index_config.get("embedding_model", fallback_model),
                device=index_config.get("device", "cpu"),
            )

            vector_config = VectorStoreConfig(
                store_type=VectorStoreType.CHROMADB,
                persist_directory=index_config.get("persist_dir", fallback_persist_dir),
                collection_name=index_config.get("collection_name", "code_index"),
            )

            chunk_config = ChunkConfig(
                strategy=ChunkStrategy.AST_BASED,
                chunk_size=index_config.get("chunk_size", fallback_chunk_size),
                chunk_overlap=index_config.get("chunk_overlap", fallback_chunk_overlap),
            )

            return IndexConfig(
                embedding=embedding_config,
                vector_store=vector_config,
                chunking=chunk_config,
                include_patterns=index_config.get("include_patterns", fallback_include_patterns),
                exclude_patterns=index_config.get("exclude_patterns", fallback_exclude_patterns),
                max_workers=index_config.get("max_workers", 4),
            )
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")

    # 返回默认配置 - 使用 core.config 的默认值确保一致性
    return IndexConfig(
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name=fallback_model,
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            persist_directory=fallback_persist_dir,
        ),
        chunking=ChunkConfig(
            strategy=ChunkStrategy.AST_BASED,
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_chunk_overlap,
        ),
        include_patterns=fallback_include_patterns,
        exclude_patterns=fallback_exclude_patterns,
    )


async def create_indexer(root_path: Path, config: IndexConfig, show_progress: bool = True) -> CodebaseIndexer:
    """创建索引器实例

    Args:
        root_path: 代码库根目录
        config: 索引配置
        show_progress: 是否显示进度

    Returns:
        CodebaseIndexer 实例
    """
    # 创建嵌入模型
    cache_dir = root_path / ".cursor" / "embedding_cache"
    cache = EmbeddingCache(max_size=10000, cache_dir=str(cache_dir))

    embedding_model = SentenceTransformerEmbedding(
        model_name=config.embedding.model_name,
        device=config.embedding.device if config.embedding.device != "cpu" else None,
        cache=cache,
        show_progress=show_progress,
    )

    # 创建向量存储
    persist_dir = root_path / config.vector_store.persist_directory
    vector_store = ChromaVectorStore(
        persist_directory=str(persist_dir),
        collection_name=config.vector_store.collection_name,
        embedding_dimension=embedding_model.dimension,
    )

    # 创建分块器
    chunker = SemanticCodeChunker(config.chunking)

    # 创建索引器
    state_file = root_path / DEFAULT_STATE_FILE
    indexer = CodebaseIndexer(
        root_path=root_path,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunker=chunker,
        include_patterns=config.include_patterns,
        exclude_patterns=config.exclude_patterns,
        state_file=state_file,
        max_concurrent=config.max_workers,
    )

    return indexer


async def cmd_build(args: argparse.Namespace) -> int:
    """构建索引命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"错误: 路径不存在: {root_path}")
        return 1

    config = load_config(args.config)
    incremental = not args.full

    print(f"{'全量' if args.full else '增量'}索引代码库: {root_path}")
    print(f"嵌入模型: {config.embedding.model_name}")
    print(f"索引目录: {config.vector_store.persist_directory}")
    print()

    try:
        indexer = await create_indexer(root_path, config)

        # 收集文件用于进度显示
        files = indexer.collect_files()
        if not files:
            print("没有找到需要索引的文件")
            return 0

        print(f"找到 {len(files)} 个文件")

        # 设置进度回调
        progress_bar = ProgressBar(len(files), prefix="索引进度")
        indexer.set_progress_callback(progress_bar.update)

        # 执行索引
        await indexer.index_codebase(incremental=incremental)
        progress_bar.finish()

        # 显示结果
        stats = await indexer.get_stats()
        print("\n索引统计:")
        print(f"  已索引文件: {stats['indexed_files']}")
        print(f"  代码分块数: {stats['total_chunks']}")
        print(f"  嵌入模型: {stats['embedding_model']}")

        return 0

    except Exception as e:
        logger.exception("索引失败")
        print(f"\n错误: {e}")
        return 1


async def cmd_update(args: argparse.Namespace) -> int:
    """增量更新命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"错误: 路径不存在: {root_path}")
        return 1

    config = load_config(args.config)

    print(f"增量更新索引: {root_path}")

    try:
        indexer = await create_indexer(root_path, config)

        # 执行增量索引
        start_time = time.time()
        chunks = await indexer.index_codebase(incremental=True)
        elapsed = time.time() - start_time

        # 显示结果
        progress = indexer.get_progress()
        print("\n更新完成!")
        print(f"  处理文件: {progress.processed_files}")
        print(f"  新增分块: {len(chunks)}")
        print(f"  耗时: {elapsed:.1f}s")

        if progress.errors:
            print(f"  警告: {len(progress.errors)} 个文件处理失败")

        return 0

    except Exception as e:
        logger.exception("更新失败")
        print(f"\n错误: {e}")
        return 1


async def cmd_search(args: argparse.Namespace) -> int:
    """搜索命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    config = load_config(args.config)

    try:
        # 创建嵌入模型和向量存储
        embedding_model = SentenceTransformerEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device if config.embedding.device != "cpu" else None,
        )

        persist_dir = root_path / config.vector_store.persist_directory
        if not persist_dir.exists():
            print("错误: 索引不存在，请先运行 build 命令")
            return 1

        vector_store = ChromaVectorStore(
            persist_directory=str(persist_dir),
            collection_name=config.vector_store.collection_name,
        )

        # 创建搜索引擎
        search_engine = SemanticSearch(embedding_model, vector_store)

        # 执行搜索
        query = args.query
        top_k = args.top_k

        print(f'搜索: "{query}"')
        print(f"返回前 {top_k} 个结果\n")

        start_time = time.time()

        results: Sequence[SearchResult | SearchResultWithContext]
        if args.context:
            # 带上下文的搜索
            options = SearchOptions(
                top_k=top_k,
                include_context=True,
                context_lines=args.context_lines,
            )
            results = await search_engine.search_with_context(query, options)
        else:
            # 基本搜索
            results = await search_engine.search(query, top_k=top_k)

        elapsed = time.time() - start_time

        if not results:
            print("未找到匹配的结果")
            return 0

        # 显示结果
        print(f"找到 {len(results)} 个结果 (耗时: {elapsed * 1000:.1f}ms)\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            if isinstance(result, SearchResultWithContext):
                # SearchResultWithContext
                print(f"\n[{i}] {result.file_path}:{result.start_line}-{result.end_line}")
                print(f"    分数: {result.score:.4f} | 类型: {result.chunk_type} | 语言: {result.language}")
                if result.name:
                    print(f"    名称: {result.name}")

                # 显示代码片段
                content = result.content
                if len(content) > 500:
                    content = content[:500] + "\n... (已截断)"

                print("-" * 40)
                for line in content.split("\n")[:15]:
                    print(f"    {line}")
                if content.count("\n") > 15:
                    print("    ... (更多内容)")
            elif isinstance(result, SearchResult):
                # SearchResult
                chunk = result.chunk
                print(f"\n[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
                print(f"    分数: {result.score:.4f} | 类型: {chunk.chunk_type.value}")
                if chunk.name:
                    print(f"    名称: {chunk.name}")

                content = chunk.content
                if len(content) > 300:
                    content = content[:300] + "..."
                print("-" * 40)
                for line in content.split("\n")[:10]:
                    print(f"    {line}")

        print("\n" + "=" * 80)

        if args.json:
            # JSON 输出
            output = []
            for result in results:
                if isinstance(result, SearchResultWithContext):
                    output.append(
                        {
                            "file_path": result.file_path,
                            "start_line": result.start_line,
                            "end_line": result.end_line,
                            "score": result.score,
                            "name": result.name,
                            "content": result.content[:500],
                        }
                    )
                elif isinstance(result, SearchResult):
                    output.append(
                        {
                            "file_path": result.chunk.file_path,
                            "start_line": result.chunk.start_line,
                            "end_line": result.chunk.end_line,
                            "score": result.score,
                            "name": result.chunk.name,
                            "content": result.chunk.content[:500],
                        }
                    )
            print("\nJSON 输出:")
            print(json.dumps(output, ensure_ascii=False, indent=2))

        return 0

    except Exception as e:
        logger.exception("搜索失败")
        print(f"\n错误: {e}")
        return 1


async def cmd_status(args: argparse.Namespace) -> int:
    """显示索引状态命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    config = load_config(args.config)

    state_file = root_path / DEFAULT_STATE_FILE
    persist_dir = root_path / config.vector_store.persist_directory

    print(f"索引状态: {root_path}")
    print("=" * 60)

    # 检查状态文件
    if state_file.exists():
        try:
            with open(state_file, encoding="utf-8") as f:
                state_data = json.load(f)

            updated_at = state_data.get("updated_at", "未知")
            files = state_data.get("files", {})

            print(f"状态文件: {state_file}")
            print(f"最后更新: {updated_at}")
            print(f"已索引文件数: {len(files)}")

            if files:
                # 计算统计信息
                total_chunks = sum(len(f.get("chunk_ids", [])) for f in files.values())
                languages: dict[str, int] = {}
                for fp in files:
                    ext = Path(fp).suffix
                    languages[ext] = languages.get(ext, 0) + 1

                print(f"总分块数: {total_chunks}")
                print("文件类型分布:")
                for ext, count in sorted(languages.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {ext or '(无扩展名)'}: {count}")
        except Exception as e:
            print(f"读取状态文件失败: {e}")
    else:
        print(f"状态文件不存在: {state_file}")

    print()

    # 检查向量存储
    if persist_dir.exists():
        try:
            vector_store = ChromaVectorStore(
                persist_directory=str(persist_dir),
                collection_name=config.vector_store.collection_name,
            )

            stats = vector_store.get_stats()
            print(f"向量存储: {persist_dir}")
            print(f"集合名称: {stats['current_collection']}")
            print(f"向量数量: {stats['current_collection_count']}")
            print(f"所有集合: {stats['collections']}")
            print(f"持久化模式: {'是' if stats['is_persistent'] else '否'}")
        except Exception as e:
            print(f"读取向量存储失败: {e}")
    else:
        print(f"向量存储不存在: {persist_dir}")

    return 0


async def cmd_clear(args: argparse.Namespace) -> int:
    """清除索引命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    config = load_config(args.config)

    state_file = root_path / DEFAULT_STATE_FILE
    persist_dir = root_path / config.vector_store.persist_directory
    cache_dir = root_path / ".cursor" / "embedding_cache"

    if not args.confirm:
        print("警告: 此操作将清除所有索引数据!")
        print(f"  状态文件: {state_file}")
        print(f"  向量存储: {persist_dir}")
        print(f"  嵌入缓存: {cache_dir}")
        print()

        response = input("确认清除? (输入 'yes' 确认): ")
        if response.lower() != "yes":
            print("已取消")
            return 0

    cleared = []

    # 清除状态文件
    if state_file.exists():
        state_file.unlink()
        cleared.append("状态文件")

    # 清除向量存储
    if persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
        cleared.append("向量存储")

    # 清除嵌入缓存
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        cleared.append("嵌入缓存")

    if cleared:
        print(f"已清除: {', '.join(cleared)}")
    else:
        print("没有需要清除的数据")

    return 0


async def cmd_info(args: argparse.Namespace) -> int:
    """显示索引信息命令

    Args:
        args: 命令行参数

    Returns:
        退出码
    """
    root_path = Path(args.path).resolve()
    config = load_config(args.config)

    print("索引系统信息")
    print("=" * 60)

    # 配置信息
    print("\n配置:")
    print(f"  工作目录: {root_path}")
    print(f"  配置文件: {args.config or '(默认配置)'}")

    # 嵌入模型信息
    print("\n嵌入模型:")
    print(f"  提供商: {config.embedding.provider}")
    print(f"  模型: {config.embedding.model_name}")
    print(f"  设备: {config.embedding.device}")

    # 可用模型列表
    available_models = get_available_models()
    print("\n可用的本地模型:")
    for name, info in available_models.items():
        print(f"  - {name}: {info['dimension']}维 - {info['description']}")

    # 向量存储信息
    print("\n向量存储:")
    print(f"  类型: {config.vector_store.store_type}")
    print(f"  目录: {config.vector_store.persist_directory}")
    print(f"  集合: {config.vector_store.collection_name}")
    print(f"  度量: {config.vector_store.metric}")

    # 分块配置
    print("\n分块策略:")
    print(f"  策略: {config.chunking.strategy}")
    print(f"  目标大小: {config.chunking.chunk_size} 字符")
    print(f"  重叠大小: {config.chunking.chunk_overlap} 字符")
    print(f"  最小/最大: {config.chunking.min_chunk_size}/{config.chunking.max_chunk_size}")

    # 文件过滤
    print("\n文件过滤:")
    print(f"  包含模式: {config.include_patterns}")
    print(f"  排除模式: {config.exclude_patterns[:3]}...")

    # 统计现有索引
    persist_dir = root_path / config.vector_store.persist_directory
    if persist_dir.exists():
        try:
            vector_store = ChromaVectorStore(
                persist_directory=str(persist_dir),
                collection_name=config.vector_store.collection_name,
            )
            count = vector_store._collection.count()
            print("\n现有索引:")
            print(f"  向量数量: {count}")
        except Exception:
            pass

    return 0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器

    Returns:
        ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        prog="index",
        description="代码索引管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s build                    # 增量构建索引
  %(prog)s build --full             # 全量重建索引
  %(prog)s update                   # 增量更新
  %(prog)s search "用户认证"         # 语义搜索
  %(prog)s search "login" --top-k 5 # 返回前 5 个结果
  %(prog)s status                   # 查看索引状态
  %(prog)s info                     # 显示系统信息
  %(prog)s clear --confirm          # 清除所有索引
""",
    )

    # 全局参数
    parser.add_argument(
        "-p",
        "--path",
        default=".",
        help="代码库路径 (默认: 当前目录)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # build 命令
    build_parser = subparsers.add_parser("build", help="构建代码索引")
    build_parser.add_argument(
        "--full",
        action="store_true",
        help="全量重建索引 (忽略增量)",
    )

    # update 命令
    subparsers.add_parser("update", help="增量更新索引")

    # search 命令
    search_parser = subparsers.add_parser("search", help="语义搜索")
    search_parser.add_argument(
        "query",
        help="搜索查询",
    )
    search_parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="返回结果数量 (默认: 10)",
    )
    search_parser.add_argument(
        "--context",
        action="store_true",
        help="包含上下文代码",
    )
    search_parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="上下文行数 (默认: 3)",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式",
    )

    # status 命令
    subparsers.add_parser("status", help="显示索引状态")

    # clear 命令
    clear_parser = subparsers.add_parser("clear", help="清除索引")
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认清除 (不询问)",
    )

    # info 命令
    subparsers.add_parser("info", help="显示索引系统信息")

    return parser


async def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # 分发命令
    if args.command == "build":
        return await cmd_build(args)
    elif args.command == "update":
        return await cmd_update(args)
    elif args.command == "search":
        return await cmd_search(args)
    elif args.command == "status":
        return await cmd_status(args)
    elif args.command == "clear":
        return await cmd_clear(args)
    elif args.command == "info":
        return await cmd_info(args)
    else:
        parser.print_help()
        return 0


def run():
    """同步入口点"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
    except Exception as e:
        logger.exception("程序异常")
        print(f"\n致命错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
