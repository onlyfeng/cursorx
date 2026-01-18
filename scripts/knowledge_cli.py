#!/usr/bin/env python3
"""知识库管理 CLI

提供知识库文档的管理功能，包括添加、导入、列出、搜索、删除、刷新和向量索引管理。

用法:
    python knowledge_cli.py add <url>                           # 添加单个 URL
    python knowledge_cli.py add-file <file>                     # 添加本地文件
    python knowledge_cli.py import <file>                       # 批量导入 URL
    python knowledge_cli.py list                                # 列出所有文档
    python knowledge_cli.py search <query>                      # 搜索文档
    python knowledge_cli.py search --mode semantic <query>      # 语义搜索
    python knowledge_cli.py search --mode hybrid <query>        # 混合搜索
    python knowledge_cli.py remove <doc_id>                     # 删除文档
    python knowledge_cli.py refresh [doc_id]                    # 刷新文档
    python knowledge_cli.py refresh --all                       # 刷新所有文档
    python knowledge_cli.py index build                         # 构建向量索引
    python knowledge_cli.py index rebuild                       # 重建向量索引
    python knowledge_cli.py index stats                         # 显示索引统计

示例:
    python knowledge_cli.py add https://example.com
    python knowledge_cli.py add-file ./docs/readme.md
    python knowledge_cli.py import urls.txt
    python knowledge_cli.py search "Python"
    python knowledge_cli.py search --mode semantic --top-k 5 --min-score 0.5 "机器学习"
    python knowledge_cli.py search --mode hybrid --semantic-weight 0.8 "API 配置"
    python knowledge_cli.py index build
    python knowledge_cli.py refresh doc-abc123
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge import Document, KnowledgeManager, KnowledgeStorage  # noqa: E402


class ProgressBar:
    """简单的进度条显示器"""

    def __init__(self, total: int, description: str = "", width: int = 40):
        """初始化进度条

        Args:
            total: 总任务数
            description: 任务描述
            width: 进度条宽度
        """
        self.total = max(total, 1)
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()

    def update(self, n: int = 1, message: str = "") -> None:
        """更新进度

        Args:
            n: 增加的进度数
            message: 当前任务消息
        """
        self.current += n
        self._render(message)

    def _render(self, message: str = "") -> None:
        """渲染进度条"""
        percent = min(self.current / self.total, 1.0)
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            time_str = f"ETA: {eta:.0f}s"
        else:
            time_str = "计算中..."

        status = f"\r{self.description} |{bar}| {self.current}/{self.total} ({percent*100:.1f}%) {time_str}"
        if message:
            # 截断过长的消息
            max_msg_len = 40
            if len(message) > max_msg_len:
                message = message[:max_msg_len-3] + "..."
            status += f" - {message}"

        # 清除行尾并打印
        sys.stdout.write(status + " " * 20)
        sys.stdout.flush()

    def finish(self, message: str = "完成") -> None:
        """完成进度条"""
        self.current = self.total
        elapsed = time.time() - self.start_time
        print(f"\r{self.description} |{'█' * self.width}| {self.total}/{self.total} (100.0%) 用时: {elapsed:.1f}s - {message}")


class KnowledgeCLI:
    """知识库 CLI 管理器"""

    def __init__(self, kb_name: str = "default"):
        """初始化 CLI

        Args:
            kb_name: 知识库名称
        """
        self.manager = KnowledgeManager(name=kb_name)
        self.storage = KnowledgeStorage()

    async def add(self, url: str) -> bool:
        """添加单个 URL 到知识库

        Args:
            url: 目标 URL

        Returns:
            是否成功
        """
        print(f"📥 正在添加: {url}")

        await self.manager.initialize()
        doc = await self.manager.add_url(url)

        if doc:
            # 保存到持久化存储
            await self.storage.initialize()
            success, message = await self.storage.save_document(doc)

            if success:
                print(f"✅ 添加成功: {doc.id}")
                print(f"   标题: {doc.title or '(无标题)'}")
                print(f"   内容大小: {len(doc.content)} 字符")
                return True
            else:
                print(f"⚠️ 保存失败: {message}")
                return False
        else:
            print("❌ 添加失败: 无法获取 URL 内容")
            return False

    async def add_file(self, file_path: str, encoding: str = "utf-8") -> bool:
        """添加本地文件到知识库

        支持 .txt, .md, .rst 等文本格式。

        Args:
            file_path: 本地文件路径
            encoding: 文件编码，默认 utf-8

        Returns:
            是否成功
        """
        path = Path(file_path)
        if not path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return False

        print(f"📄 正在添加文件: {file_path}")

        await self.manager.initialize()
        doc = await self.manager.add_file(file_path, encoding=encoding)

        if doc:
            # 保存到持久化存储
            await self.storage.initialize()
            success, message = await self.storage.save_document(doc)

            if success:
                print(f"✅ 添加成功: {doc.id}")
                print(f"   标题: {doc.title or '(无标题)'}")
                print(f"   内容大小: {len(doc.content)} 字符")
                print(f"   文件类型: {doc.metadata.get('file_extension', 'unknown')}")
                return True
            else:
                print(f"⚠️ 保存失败: {message}")
                return False
        else:
            print("❌ 添加失败: 无法读取文件或格式不支持")
            print("   支持的格式: .txt, .md, .rst, .text, .markdown")
            return False

    async def import_urls(self, file_path: str) -> int:
        """批量导入 URL

        Args:
            file_path: URL 文件路径（每行一个 URL）

        Returns:
            成功添加的数量
        """
        path = Path(file_path)
        if not path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return 0

        # 读取 URL 列表
        urls = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if line and not line.startswith("#"):
                    urls.append(line)

        if not urls:
            print("⚠️ 文件中没有有效的 URL")
            return 0

        print(f"📥 正在导入 {len(urls)} 个 URL...")

        await self.manager.initialize()
        await self.storage.initialize()

        docs = await self.manager.add_urls(urls)

        # 保存到持久化存储
        success_count = 0
        for doc in docs:
            success, _ = await self.storage.save_document(doc)
            if success:
                success_count += 1
                print(f"  ✅ {doc.id}: {doc.title or doc.url}")

        print(f"\n📊 导入完成: {success_count}/{len(urls)} 成功")
        return success_count

    async def list_docs(
        self,
        limit: int = 50,
        offset: int = 0,
        verbose: bool = False,
    ) -> list:
        """列出所有文档

        Args:
            limit: 返回数量限制
            offset: 起始位置
            verbose: 是否显示详细信息

        Returns:
            文档列表
        """
        await self.storage.initialize()
        entries = await self.storage.list_documents(offset=offset, limit=limit)

        if not entries:
            print("📭 知识库为空")
            return []

        print(f"📚 知识库文档列表 (共 {len(entries)} 条):\n")
        print(f"{'ID':<16} {'标题':<40} {'大小':>8}")
        print("-" * 70)

        for entry in entries:
            title = entry.title[:37] + "..." if len(entry.title) > 40 else entry.title
            size = f"{entry.content_size:,}"
            print(f"{entry.doc_id:<16} {title:<40} {size:>8}")

            if verbose:
                print(f"    URL: {entry.url}")
                print(f"    更新时间: {entry.updated_at}")
                print()

        return entries

    async def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "keyword",
        min_score: float = 0.0,
        semantic_weight: float = 0.7,
    ) -> list:
        """搜索文档

        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
            mode: 搜索模式 - "keyword"(关键词), "semantic"(语义), "hybrid"(混合)
            min_score: 最低相似度阈值 (0.0-1.0)
            semantic_weight: 混合搜索中语义权重 (0.0-1.0)

        Returns:
            搜索结果列表
        """
        mode_names = {
            "keyword": "关键词",
            "semantic": "语义",
            "hybrid": "混合",
        }
        mode_display = mode_names.get(mode, mode)
        print(f"🔍 {mode_display}搜索: {query}")

        if mode in ("semantic", "hybrid"):
            print(f"   参数: top_k={limit}, min_score={min_score}", end="")
            if mode == "hybrid":
                print(f", semantic_weight={semantic_weight}")
            else:
                print()
        print()

        await self.storage.initialize()

        # 根据模式选择搜索方法
        if mode == "semantic":
            # 使用语义搜索
            vector_results = await self.storage.search_semantic(
                query,
                top_k=limit,
                min_score=min_score,
            )
            # 转换为统一格式
            results = []
            for vr in vector_results:
                from knowledge import SearchResult
                entry = self.storage._index.get(vr.doc_id)
                if entry:
                    results.append(SearchResult(
                        doc_id=vr.doc_id,
                        url=entry.url,
                        title=entry.title,
                        score=vr.score,
                        snippet=vr.content[:100] + "..." if len(vr.content) > 100 else vr.content,
                        match_type="semantic",
                    ))
        elif mode == "hybrid":
            # 使用混合搜索
            results = await self.storage.search(query, limit=limit, mode="hybrid")
            # 过滤低于最低分数的结果
            if min_score > 0:
                results = [r for r in results if r.score >= min_score]
        else:
            # 默认关键词搜索
            results = await self.storage.search(query, limit=limit, mode="keyword")

        if not results:
            print("📭 未找到匹配的文档")
            return []

        print(f"找到 {len(results)} 条结果:\n")

        # 表头
        print(f"{'序号':<4} {'分数':>6} {'匹配类型':<10} {'标题':<40}")
        print("-" * 70)

        for i, result in enumerate(results, 1):
            # 格式化分数
            score_str = f"{result.score:.3f}" if result.score < 1 else f"{result.score:.1f}"

            # 匹配类型标记
            match_type_display = {
                "semantic": "🧠 语义",
                "keyword": "🔤 关键词",
                "hybrid": "🔀 混合",
                "exact": "✓ 精确",
                "partial": "~ 部分",
            }.get(result.match_type, result.match_type)

            # 截断标题
            title = result.title or "(无标题)"
            if len(title) > 38:
                title = title[:35] + "..."

            print(f"{i:<4} {score_str:>6} {match_type_display:<10} {title:<40}")
            print(f"     ID: {result.doc_id}")
            print(f"     URL: {result.url}")
            if result.snippet:
                # 截取并清理摘要
                snippet = result.snippet.replace("\n", " ")[:100]
                print(f"     摘要: {snippet}...")
            print()

        return results

    async def remove(self, doc_id: str) -> bool:
        """删除文档

        Args:
            doc_id: 文档 ID

        Returns:
            是否成功
        """
        print(f"🗑️ 正在删除: {doc_id}")

        await self.storage.initialize()

        # 检查文档是否存在
        if not self.storage.has_document(doc_id):
            print(f"❌ 文档不存在: {doc_id}")
            return False

        success = await self.storage.delete_document(doc_id)

        if success:
            print(f"✅ 删除成功: {doc_id}")
            return True
        else:
            print("❌ 删除失败")
            return False

    async def refresh(
        self,
        doc_id: Optional[str] = None,
        refresh_all: bool = False,
    ) -> int:
        """刷新文档（重新获取 URL 内容）

        Args:
            doc_id: 指定文档 ID
            refresh_all: 是否刷新所有文档

        Returns:
            成功刷新的数量
        """
        await self.manager.initialize()
        await self.storage.initialize()

        if refresh_all:
            # 刷新所有文档
            entries = await self.storage.list_documents(limit=1000)
            if not entries:
                print("📭 知识库为空，无需刷新")
                return 0

            print(f"🔄 正在刷新所有文档 ({len(entries)} 个)...\n")

            success_count = 0
            for entry in entries:
                print(f"  刷新: {entry.doc_id} ({entry.title or entry.url})")

                # 加载文档获取 URL
                doc = await self.storage.load_document(entry.doc_id)
                if not doc:
                    print("    ⚠️ 加载失败，跳过")
                    continue

                # 刷新
                refreshed_doc = await self.manager.refresh(doc.url)
                if refreshed_doc:
                    # 保存更新
                    success, _ = await self.storage.save_document(refreshed_doc, force=True)
                    if success:
                        success_count += 1
                        print("    ✅ 成功")
                    else:
                        print("    ⚠️ 保存失败")
                else:
                    print("    ❌ 获取失败")

            print(f"\n📊 刷新完成: {success_count}/{len(entries)} 成功")
            return success_count

        elif doc_id:
            # 刷新指定文档
            print(f"🔄 正在刷新: {doc_id}")

            doc = await self.storage.load_document(doc_id)
            if not doc:
                print(f"❌ 文档不存在: {doc_id}")
                return 0

            refreshed_doc = await self.manager.refresh(doc.url)
            if refreshed_doc:
                success, _ = await self.storage.save_document(refreshed_doc, force=True)
                if success:
                    print(f"✅ 刷新成功: {doc_id}")
                    print(f"   内容大小: {len(refreshed_doc.content)} 字符")
                    return 1
                else:
                    print("⚠️ 保存失败")
                    return 0
            else:
                print("❌ 刷新失败: 无法获取 URL 内容")
                return 0

        else:
            print("❌ 请指定文档 ID 或使用 --all 刷新所有文档")
            return 0

    async def stats(self) -> dict:
        """显示知识库统计信息

        Returns:
            统计信息字典
        """
        await self.storage.initialize()
        stats = await self.storage.get_stats()

        print("📊 知识库统计:\n")
        print(f"  文档数量: {stats['document_count']}")
        print(f"  总内容大小: {stats['total_content_size']:,} 字符")
        print(f"  分块总数: {stats['total_chunk_count']}")
        print(f"  存储路径: {stats['storage_path']}")

        return stats

    async def index_build(self) -> dict[str, Any]:
        """构建向量索引

        为所有未索引的文档构建向量索引。

        Returns:
            构建结果统计
        """
        print("🔧 正在构建向量索引...\n")

        await self.storage.initialize()

        # 获取所有文档
        entries = await self.storage.list_documents(limit=10000)
        if not entries:
            print("📭 知识库为空，无需构建索引")
            return {"indexed": 0, "failed": 0, "total": 0}

        # 检查向量存储
        if self.storage._vector_store is None:
            print("❌ 向量存储未初始化，请检查依赖")
            return {"indexed": 0, "failed": 0, "total": len(entries), "error": "向量存储未初始化"}

        # 获取已索引的文档 ID
        indexed_doc_ids = set(self.storage._vector_store._doc_chunk_mapping.keys())

        # 找出未索引的文档
        unindexed_entries = [e for e in entries if e.doc_id not in indexed_doc_ids]

        if not unindexed_entries:
            print("✅ 所有文档已完成索引，无需重新构建")
            return {"indexed": 0, "failed": 0, "total": len(entries), "already_indexed": len(indexed_doc_ids)}

        print(f"📊 发现 {len(unindexed_entries)} 个待索引文档 (已索引: {len(indexed_doc_ids)})\n")

        # 创建进度条
        progress = ProgressBar(len(unindexed_entries), "构建索引")

        indexed = 0
        failed = 0

        for entry in unindexed_entries:
            doc = await self.storage.load_document(entry.doc_id)
            if doc:
                try:
                    success = await self.storage._vector_store.index_document(doc)
                    if success:
                        indexed += 1
                        progress.update(1, f"✓ {entry.title[:30] if entry.title else entry.doc_id}")
                    else:
                        failed += 1
                        progress.update(1, f"✗ {entry.doc_id}")
                except Exception:
                    failed += 1
                    progress.update(1, "✗ 错误")
            else:
                failed += 1
                progress.update(1, "✗ 加载失败")

        progress.finish(f"索引: {indexed}, 失败: {failed}")

        print("\n📊 索引构建完成:")
        print(f"   成功索引: {indexed}")
        print(f"   索引失败: {failed}")
        print(f"   文档总数: {len(entries)}")

        return {"indexed": indexed, "failed": failed, "total": len(entries)}

    async def index_rebuild(self) -> dict[str, Any]:
        """重建向量索引

        清空现有索引并重新索引所有文档。

        Returns:
            重建结果统计
        """
        print("🔄 正在重建向量索引...\n")
        print("⚠️ 注意: 这将清空现有索引并重新构建\n")

        await self.storage.initialize()

        # 获取所有文档
        entries = await self.storage.list_documents(limit=10000)
        if not entries:
            print("📭 知识库为空，无需重建索引")
            return {"indexed": 0, "failed": 0, "total": 0}

        # 检查向量存储
        if self.storage._vector_store is None:
            print("❌ 向量存储未初始化，请检查依赖")
            return {"indexed": 0, "failed": 0, "total": len(entries), "error": "向量存储未初始化"}

        print(f"📊 准备重建 {len(entries)} 个文档的索引\n")

        # 加载所有文档
        documents: list[Document] = []
        load_progress = ProgressBar(len(entries), "加载文档")

        for entry in entries:
            doc = await self.storage.load_document(entry.doc_id)
            if doc:
                documents.append(doc)
                load_progress.update(1, f"✓ {entry.title[:30] if entry.title else entry.doc_id}")
            else:
                load_progress.update(1, f"✗ {entry.doc_id}")

        load_progress.finish(f"已加载 {len(documents)}/{len(entries)} 个文档")
        print()

        if not documents:
            print("❌ 没有可用的文档进行索引")
            return {"indexed": 0, "failed": len(entries), "total": len(entries)}

        # 重建索引
        index_progress = ProgressBar(len(documents), "重建索引")

        # 清空现有索引
        await self.storage._vector_store._vector_store.clear()
        self.storage._vector_store._doc_chunk_mapping.clear()

        indexed = 0
        failed = 0

        for doc in documents:
            try:
                success = await self.storage._vector_store.index_document(doc)
                if success:
                    indexed += 1
                    title = doc.title[:30] if doc.title else doc.id
                    index_progress.update(1, f"✓ {title}")
                else:
                    failed += 1
                    index_progress.update(1, f"✗ {doc.id}")
            except Exception:
                failed += 1
                index_progress.update(1, "✗ 错误")

        index_progress.finish(f"索引: {indexed}, 失败: {failed}")

        print("\n📊 索引重建完成:")
        print(f"   成功索引: {indexed}")
        print(f"   索引失败: {failed}")
        print(f"   文档总数: {len(entries)}")

        return {"indexed": indexed, "failed": failed, "total": len(entries)}

    async def index_stats(self) -> dict[str, Any]:
        """显示向量索引统计信息

        Returns:
            索引统计信息字典
        """
        print("📊 向量索引统计:\n")

        await self.storage.initialize()

        # 检查向量存储
        if self.storage._vector_store is None:
            print("❌ 向量存储未初始化")
            return {"error": "向量存储未初始化"}

        try:
            stats = await self.storage._vector_store.get_stats()

            print(f"  初始化状态: {'✓ 已初始化' if stats.get('initialized') else '✗ 未初始化'}")
            print(f"  集合名称: {stats.get('collection_name', 'N/A')}")
            print(f"  已索引文档数: {stats.get('document_count', 0)}")
            print(f"  已索引分块数: {stats.get('chunk_count', 0)}")
            print(f"  嵌入模型: {stats.get('embedding_model', 'N/A')}")
            print(f"  嵌入维度: {stats.get('embedding_dimension', 0)}")
            print(f"  存储路径: {stats.get('persist_directory', 'N/A')}")
            print(f"  持久化: {'✓' if stats.get('is_persistent') else '✗'}")
            print(f"  相似度度量: {stats.get('metric', 'N/A')}")

            # 显示缓存统计
            cache_stats = stats.get('cache_stats', {})
            if cache_stats:
                print("\n  嵌入缓存统计:")
                print(f"    缓存大小: {cache_stats.get('memory_cache_size', 0)}")
                print(f"    命中次数: {cache_stats.get('hits', 0)}")
                print(f"    未命中次数: {cache_stats.get('misses', 0)}")
                hit_rate = cache_stats.get('hit_rate', 0)
                print(f"    命中率: {hit_rate:.1%}")

            # 与存储文档对比
            storage_count = len(self.storage._index)
            indexed_count = stats.get('document_count', 0)
            if storage_count != indexed_count:
                print(f"\n  ⚠️ 索引不同步: 存储 {storage_count} 个文档，已索引 {indexed_count} 个")
                print("     建议运行: python knowledge_cli.py index build")
            else:
                print(f"\n  ✓ 索引同步: 所有 {storage_count} 个文档已索引")

            return stats

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {"error": str(e)}


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="knowledge_cli",
        description="知识库管理 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s add https://example.com           添加 URL
  %(prog)s add-file ./docs/readme.md         添加本地文件
  %(prog)s import urls.txt                   批量导入
  %(prog)s list                              列出文档
  %(prog)s list -v                           列出文档（详细）
  %(prog)s search "Python"                   关键词搜索
  %(prog)s search --mode semantic "机器学习"  语义搜索
  %(prog)s search --mode hybrid "API"        混合搜索
  %(prog)s remove doc-abc123                 删除文档
  %(prog)s refresh doc-abc123                刷新指定文档
  %(prog)s refresh --all                     刷新所有文档
  %(prog)s stats                             显示统计信息
  %(prog)s index build                       构建向量索引
  %(prog)s index rebuild                     重建向量索引
  %(prog)s index stats                       显示索引统计
        """,
    )

    parser.add_argument(
        "--kb-name",
        default="default",
        help="知识库名称 (默认: default)",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # add 命令
    add_parser = subparsers.add_parser("add", help="添加单个 URL")
    add_parser.add_argument("url", help="目标 URL")

    # add-file 命令
    add_file_parser = subparsers.add_parser("add-file", help="添加本地文件")
    add_file_parser.add_argument("file", help="本地文件路径（支持 .txt, .md, .rst 等）")
    add_file_parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码 (默认: utf-8)"
    )

    # import 命令
    import_parser = subparsers.add_parser("import", help="批量导入 URL")
    import_parser.add_argument("file", help="URL 文件路径（每行一个 URL）")

    # list 命令
    list_parser = subparsers.add_parser("list", help="列出所有文档")
    list_parser.add_argument("-n", "--limit", type=int, default=50, help="返回数量限制")
    list_parser.add_argument("--offset", type=int, default=0, help="起始位置")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索文档")
    search_parser.add_argument("query", help="搜索关键词")
    search_parser.add_argument(
        "-n", "--limit", "--top-k",
        type=int,
        default=10,
        dest="limit",
        help="返回结果数量 (默认: 10)"
    )
    search_parser.add_argument(
        "--mode",
        choices=["keyword", "semantic", "hybrid"],
        default="keyword",
        help="搜索模式: keyword(关键词), semantic(语义), hybrid(混合) (默认: keyword)"
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="最低相似度阈值 (0.0-1.0, 默认: 0.0)"
    )
    search_parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="混合搜索中语义权重 (0.0-1.0, 默认: 0.7)"
    )

    # remove 命令
    remove_parser = subparsers.add_parser("remove", help="删除文档")
    remove_parser.add_argument("doc_id", help="文档 ID")

    # refresh 命令
    refresh_parser = subparsers.add_parser("refresh", help="刷新文档")
    refresh_parser.add_argument("doc_id", nargs="?", help="文档 ID")
    refresh_parser.add_argument("--all", action="store_true", dest="refresh_all", help="刷新所有文档")

    # stats 命令
    subparsers.add_parser("stats", help="显示知识库统计信息")

    # index 命令组
    index_parser = subparsers.add_parser("index", help="向量索引管理")
    index_subparsers = index_parser.add_subparsers(dest="index_command", help="索引子命令")

    # index build 子命令
    index_subparsers.add_parser("build", help="为所有未索引的文档构建向量索引")

    # index rebuild 子命令
    index_subparsers.add_parser("rebuild", help="清空并重建所有文档的向量索引")

    # index stats 子命令
    index_subparsers.add_parser("stats", help="显示向量索引统计信息")

    return parser


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cli = KnowledgeCLI(kb_name=args.kb_name)

    try:
        if args.command == "add":
            success = await cli.add(args.url)
            return 0 if success else 1

        elif args.command == "add-file":
            success = await cli.add_file(args.file, encoding=args.encoding)
            return 0 if success else 1

        elif args.command == "import":
            count = await cli.import_urls(args.file)
            return 0 if count > 0 else 1

        elif args.command == "list":
            await cli.list_docs(
                limit=args.limit,
                offset=args.offset,
                verbose=args.verbose,
            )
            return 0

        elif args.command == "search":
            results = await cli.search(
                query=args.query,
                limit=args.limit,
                mode=args.mode,
                min_score=args.min_score,
                semantic_weight=args.semantic_weight,
            )
            return 0 if results else 1

        elif args.command == "remove":
            success = await cli.remove(args.doc_id)
            return 0 if success else 1

        elif args.command == "refresh":
            count = await cli.refresh(
                doc_id=args.doc_id,
                refresh_all=args.refresh_all,
            )
            return 0 if count > 0 else 1

        elif args.command == "stats":
            await cli.stats()
            return 0

        elif args.command == "index":
            # 处理 index 子命令
            if not hasattr(args, 'index_command') or not args.index_command:
                # 没有指定子命令，显示帮助
                print("用法: knowledge_cli.py index {build,rebuild,stats}")
                print("\n可用的索引子命令:")
                print("  build    为所有未索引的文档构建向量索引")
                print("  rebuild  清空并重建所有文档的向量索引")
                print("  stats    显示向量索引统计信息")
                return 1

            if args.index_command == "build":
                result = await cli.index_build()
                return 0 if result.get("indexed", 0) > 0 or result.get("already_indexed", 0) > 0 else 1

            elif args.index_command == "rebuild":
                result = await cli.index_rebuild()
                return 0 if result.get("indexed", 0) > 0 else 1

            elif args.index_command == "stats":
                await cli.index_stats()
                return 0

            else:
                print(f"未知的索引子命令: {args.index_command}")
                return 1

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ 操作已取消")
        return 130
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
