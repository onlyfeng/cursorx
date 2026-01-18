#!/usr/bin/env python3
"""Cursor CLI 知识库集成测试脚本

验证 Cursor CLI/Agent 文档的知识库集成：
1. 从网页/本地文档获取 Cursor CLI 信息
2. 更新到知识库
3. 验证知识库搜索功能
4. 模拟 Agent 从知识库获取信息回答问题

用法:
    python scripts/test_cursor_knowledge.py update          # 更新知识库
    python scripts/test_cursor_knowledge.py verify          # 验证知识库
    python scripts/test_cursor_knowledge.py query "问题"    # 查询知识库
    python scripts/test_cursor_knowledge.py test            # 完整测试
    python scripts/test_cursor_knowledge.py interactive     # 交互模式
"""
import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge import (  # noqa: E402
    Document,
    KnowledgeManager,
    KnowledgeStorage,
    SearchResult,
)

# ============================================================
# Cursor CLI 测试数据
# ============================================================

# Cursor CLI 文档 URL 列表
CURSOR_CLI_URLS = [
    "https://cursor.com/cn/docs/cli/overview",
    "https://cursor.com/cn/docs/cli/using",
    "https://cursor.com/cn/docs/cli/reference/parameters",
    "https://cursor.com/cn/docs/cli/reference/slash-commands",
    "https://cursor.com/cn/docs/cli/mcp",
]

# Agent 可能会问的问题和预期能找到的关键信息
AGENT_QUERIES = [
    {
        "question": "Cursor CLI 支持哪些命令行参数",
        "search_terms": ["CLI 参数", "命令行参数", "parameters"],
        "expected_info": ["--model", "--print", "--force", "-p", "-m"],
        "description": "查询 CLI 命令行参数",
    },
    {
        "question": "agent 命令有哪些斜杠命令",
        "search_terms": ["斜杠命令", "slash commands", "/model"],
        "expected_info": ["/model", "/new-chat", "/quit", "/rules"],
        "description": "查询斜杠命令",
    },
    {
        "question": "如何使用 MCP 服务器",
        "search_terms": ["MCP", "MCP 服务器", "mcp server"],
        "expected_info": ["mcp", "enable", "disable", "list"],
        "description": "查询 MCP 使用方法",
    },
    {
        "question": "Cursor CLI 如何进行认证",
        "search_terms": ["认证", "登录", "API Key", "authentication"],
        "expected_info": ["login", "CURSOR_API_KEY", "logout"],
        "description": "查询认证方法",
    },
    {
        "question": "如何查看可用的模型列表",
        "search_terms": ["模型", "models", "list-models"],
        "expected_info": ["models", "--list-models", "/models"],
        "description": "查询模型列表",
    },
]

# 本地文档路径（如果无法从网络获取）
LOCAL_DOCS = [
    "cursor_cli_docs.txt",
    "cursor_agent_docs.txt",
    "cursor_docs_full.txt",
    "AGENTS.md",
]


# ============================================================
# 颜色输出
# ============================================================

class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{'─' * 50}{Colors.NC}")
    print(f"{Colors.CYAN}  {text}{Colors.NC}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.NC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.NC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.NC} {text}")


# ============================================================
# 知识库管理
# ============================================================

class CursorKnowledgeManager:
    """Cursor CLI 知识库管理器"""

    def __init__(self):
        self.storage = KnowledgeStorage()
        self.manager = KnowledgeManager(name="cursor-cli")

    async def initialize(self):
        """初始化"""
        await self.storage.initialize()
        await self.manager.initialize()
        print_success("知识库初始化完成")

    async def update_from_local_docs(self) -> int:
        """从本地文档更新知识库

        Returns:
            成功添加的文档数
        """
        print_section("从本地文档更新知识库")

        success_count = 0

        for doc_file in LOCAL_DOCS:
            doc_path = project_root / doc_file

            if not doc_path.exists():
                print_warning(f"文件不存在: {doc_file}")
                continue

            print_info(f"处理文件: {doc_file}")

            try:
                content = doc_path.read_text(encoding="utf-8")

                if not content.strip():
                    print_warning(f"文件为空: {doc_file}")
                    continue

                # 根据文件类型提取标题
                title = self._extract_title(content, doc_path)

                # 创建文档
                doc = Document(
                    url=f"file://{doc_path.resolve()}",
                    title=title,
                    content=content,
                    metadata={
                        "source": "local",
                        "file_name": doc_file,
                        "category": "cursor-cli",
                    },
                )

                # 保存到存储
                success, message = await self.storage.save_document(doc, force=True)

                if success:
                    success_count += 1
                    print_success(f"添加成功: {title} ({len(content)} 字符)")
                else:
                    print_warning(f"保存失败: {message}")

            except Exception as e:
                print_error(f"处理失败 {doc_file}: {e}")

        print(f"\n本地文档更新完成: {success_count}/{len(LOCAL_DOCS)} 成功")
        return success_count

    async def update_from_urls(self, urls: Optional[list[str]] = None) -> int:
        """从 URL 更新知识库

        Args:
            urls: URL 列表，默认使用 CURSOR_CLI_URLS

        Returns:
            成功添加的文档数
        """
        print_section("从网页 URL 更新知识库")

        urls = urls or CURSOR_CLI_URLS

        success_count = 0

        for url in urls:
            print_info(f"获取: {url}")

            try:
                doc = await self.manager.add_url(url, force_refresh=True)

                if doc:
                    # 保存到持久化存储
                    success, _ = await self.storage.save_document(doc, force=True)
                    if success:
                        success_count += 1
                        print_success(f"添加成功: {doc.title or url}")
                    else:
                        print_warning(f"保存失败: {url}")
                else:
                    print_warning(f"获取失败: {url}")

            except Exception as e:
                print_error(f"处理失败 {url}: {e}")

        print(f"\nURL 更新完成: {success_count}/{len(urls)} 成功")
        return success_count

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """搜索知识库

        Args:
            query: 搜索查询
            limit: 返回结果数

        Returns:
            搜索结果列表
        """
        return await self.storage.search(query, limit=limit)

    async def query_for_agent(
        self,
        question: str,
        search_terms: list[str],
    ) -> dict:
        """模拟 Agent 查询知识库

        使用多个搜索词进行搜索，合并结果。

        Args:
            question: 原始问题
            search_terms: 搜索关键词列表

        Returns:
            查询结果字典，包含相关内容
        """
        all_results = []
        seen_doc_ids = set()

        # 使用多个搜索词搜索
        for term in search_terms:
            results = await self.search(term, limit=3)
            for r in results:
                if r.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(r.doc_id)
                    all_results.append(r)

        if not all_results:
            return {
                "question": question,
                "found": False,
                "results": [],
                "context": "",
            }

        # 加载相关文档内容
        context_parts = []
        for result in all_results[:3]:  # 最多 3 个文档
            doc = await self.storage.load_document(result.doc_id)
            if doc:
                # 提取相关段落
                relevant_content = self._extract_relevant_content(
                    doc.content, search_terms
                )
                if relevant_content:
                    context_parts.append(f"【{doc.title}】\n{relevant_content}")

        return {
            "question": question,
            "found": True,
            "results": all_results,
            "context": "\n\n".join(context_parts),
        }

    def _extract_title(self, content: str, path: Path) -> str:
        """从内容提取标题"""
        lines = content.strip().split('\n')

        # Markdown 标题
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # 第一行非空内容
        for line in lines[:3]:
            line = line.strip()
            if line and not line.startswith('#'):
                return line[:100]

        return path.stem

    def _extract_relevant_content(
        self,
        content: str,
        search_terms: list[str],
        max_length: int = 500,
    ) -> str:
        """提取与搜索词相关的内容片段"""
        content_lower = content.lower()

        # 按段落分割
        paragraphs = content.split('\n\n')

        relevant_paragraphs = []
        total_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检查是否包含任何搜索词
            para_lower = para.lower()
            if any(term.lower() in para_lower for term in search_terms):
                if total_length + len(para) <= max_length:
                    relevant_paragraphs.append(para)
                    total_length += len(para)
                else:
                    # 截断
                    remaining = max_length - total_length
                    if remaining > 50:
                        relevant_paragraphs.append(para[:remaining] + "...")
                    break

        return '\n\n'.join(relevant_paragraphs)

    async def get_stats(self) -> dict:
        """获取知识库统计"""
        return await self.storage.get_stats()

    async def list_documents(self) -> list:
        """列出所有文档"""
        return await self.storage.list_documents()


# ============================================================
# 测试功能
# ============================================================

async def test_update(manager: CursorKnowledgeManager, from_url: bool = False):
    """测试更新知识库"""
    print_header("更新知识库")

    if from_url:
        count = await manager.update_from_urls()
    else:
        count = await manager.update_from_local_docs()

    if count > 0:
        print_success(f"成功更新 {count} 个文档")
        return True
    else:
        print_error("未能更新任何文档")
        return False


async def test_verify(manager: CursorKnowledgeManager) -> bool:
    """验证知识库内容"""
    print_header("验证知识库")

    # 1. 检查文档数量
    print_section("检查文档")
    entries = await manager.list_documents()

    if not entries:
        print_error("知识库为空！请先运行 update 命令")
        return False

    print_success(f"知识库包含 {len(entries)} 个文档")

    for entry in entries[:5]:
        print(f"  - {entry.title} ({entry.content_size} 字符)")

    if len(entries) > 5:
        print(f"  ... 还有 {len(entries) - 5} 个文档")

    # 2. 检查统计信息
    stats = await manager.get_stats()
    print_info(f"总内容大小: {stats.get('total_content_size', 0):,} 字符")

    # 3. 测试基本搜索
    print_section("测试搜索功能")

    test_queries = ["CLI", "命令", "agent", "模型"]
    search_passed = True

    for query in test_queries:
        results = await manager.search(query, limit=3)
        if results:
            print_success(f"搜索 '{query}' -> 找到 {len(results)} 个结果")
        else:
            print_warning(f"搜索 '{query}' -> 无结果")
            search_passed = False

    return search_passed


async def test_agent_queries(manager: CursorKnowledgeManager) -> dict:
    """测试 Agent 查询场景"""
    print_header("Agent 查询测试")

    results = {
        "total": len(AGENT_QUERIES),
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    for test in AGENT_QUERIES:
        print_section(test["description"])
        print(f"问题: {test['question']}")
        print(f"搜索词: {', '.join(test['search_terms'])}")

        # 执行查询
        query_result = await manager.query_for_agent(
            test["question"],
            test["search_terms"],
        )

        if not query_result["found"]:
            print_error("未找到相关信息")
            results["failed"] += 1
            results["details"].append({
                "question": test["question"],
                "passed": False,
                "reason": "未找到结果",
            })
            continue

        # 检查是否包含预期信息
        context = query_result["context"].lower()
        found_info = []
        missing_info = []

        for expected in test["expected_info"]:
            if expected.lower() in context:
                found_info.append(expected)
            else:
                missing_info.append(expected)

        # 计算匹配率
        match_rate = len(found_info) / len(test["expected_info"]) if test["expected_info"] else 0

        if match_rate >= 0.5:  # 至少 50% 匹配视为通过
            print_success(f"找到预期信息: {', '.join(found_info)}")
            if missing_info:
                print_warning(f"缺少信息: {', '.join(missing_info)}")
            results["passed"] += 1
            passed = True
        else:
            print_error(f"信息不足: 仅找到 {len(found_info)}/{len(test['expected_info'])}")
            results["failed"] += 1
            passed = False

        # 显示找到的内容摘要
        if query_result["context"]:
            print(f"\n{Colors.YELLOW}相关内容摘要:{Colors.NC}")
            context_preview = query_result["context"][:300]
            if len(query_result["context"]) > 300:
                context_preview += "..."
            print(context_preview)

        results["details"].append({
            "question": test["question"],
            "passed": passed,
            "found_info": found_info,
            "missing_info": missing_info,
            "match_rate": match_rate,
        })

        print()

    # 汇总
    print_section("测试汇总")
    print(f"通过: {results['passed']}/{results['total']}")
    print(f"失败: {results['failed']}/{results['total']}")

    success_rate = results["passed"] / results["total"] if results["total"] > 0 else 0

    if success_rate >= 0.8:
        print_success(f"Agent 查询测试通过 (成功率: {success_rate:.0%})")
    elif success_rate >= 0.5:
        print_warning(f"Agent 查询测试部分通过 (成功率: {success_rate:.0%})")
    else:
        print_error(f"Agent 查询测试失败 (成功率: {success_rate:.0%})")

    return results


async def interactive_query(manager: CursorKnowledgeManager):
    """交互式查询模式"""
    print_header("交互式知识库查询")
    print_info("输入问题查询知识库，输入 'q' 退出\n")

    while True:
        try:
            query = input(f"{Colors.CYAN}问题> {Colors.NC}").strip()

            if query.lower() in ("q", "quit", "exit"):
                print_info("退出交互模式")
                break

            if not query:
                continue

            # 分析查询，提取关键词
            search_terms = query.split()

            # 执行查询
            result = await manager.query_for_agent(query, search_terms)

            if result["found"]:
                print(f"\n{Colors.GREEN}找到 {len(result['results'])} 个相关结果:{Colors.NC}\n")

                for i, r in enumerate(result["results"][:3], 1):
                    print(f"{i}. {r.title} (分数: {r.score:.2f})")

                if result["context"]:
                    print(f"\n{Colors.YELLOW}相关内容:{Colors.NC}")
                    print(result["context"][:800])
                    if len(result["context"]) > 800:
                        print("...")
            else:
                print_warning("未找到相关信息")

            print()

        except KeyboardInterrupt:
            print("\n")
            print_info("退出交互模式")
            break
        except EOFError:
            break


async def direct_query(manager: CursorKnowledgeManager, question: str):
    """直接查询"""
    print_section(f"查询: {question}")

    # 提取关键词
    search_terms = question.split()

    result = await manager.query_for_agent(question, search_terms)

    if result["found"]:
        print(f"\n找到 {len(result['results'])} 个相关结果:\n")

        for i, r in enumerate(result["results"][:5], 1):
            print(f"{Colors.GREEN}{i}. {r.title}{Colors.NC}")
            print(f"   分数: {r.score:.2f}")

        if result["context"]:
            print(f"\n{Colors.YELLOW}相关内容:{Colors.NC}\n")
            print(result["context"])
    else:
        print_warning("未找到相关信息")
        print_info("请确保已运行 'update' 命令更新知识库")


async def run_full_test():
    """运行完整测试"""
    print_header("Cursor CLI 知识库完整测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    manager = CursorKnowledgeManager()
    await manager.initialize()

    all_passed = True

    # 1. 更新知识库
    update_ok = await test_update(manager, from_url=False)
    if not update_ok:
        print_warning("知识库更新失败，尝试使用现有数据继续测试")

    # 2. 验证知识库
    verify_ok = await test_verify(manager)
    if not verify_ok:
        print_error("知识库验证失败")
        all_passed = False

    # 3. Agent 查询测试
    agent_results = await test_agent_queries(manager)
    if agent_results["passed"] < agent_results["total"] * 0.6:
        all_passed = False

    # 最终结果
    print_header("测试结果")

    if all_passed:
        print_success("所有测试通过！知识库功能正常。")
        print_info("Agent 可以从知识库获取 Cursor CLI 相关信息。")
        return 0
    else:
        print_warning("部分测试未通过，请检查知识库内容。")
        print_info("建议运行: python scripts/test_cursor_knowledge.py update")
        return 1


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cursor CLI 知识库集成测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s update              # 从本地文档更新知识库
  %(prog)s update --url        # 从网页 URL 更新知识库
  %(prog)s verify              # 验证知识库内容
  %(prog)s query "CLI 参数"    # 直接查询
  %(prog)s test                # 运行完整测试
  %(prog)s interactive         # 交互式查询模式
        """,
    )

    parser.add_argument(
        "command",
        choices=["update", "verify", "query", "test", "interactive"],
        help="要执行的命令",
    )
    parser.add_argument(
        "query_text",
        nargs="?",
        help="查询文本 (用于 query 命令)",
    )
    parser.add_argument(
        "--url",
        action="store_true",
        help="从 URL 更新 (用于 update 命令)",
    )

    args = parser.parse_args()

    async def run():
        manager = CursorKnowledgeManager()
        await manager.initialize()

        if args.command == "update":
            await test_update(manager, from_url=args.url)
        elif args.command == "verify":
            await test_verify(manager)
        elif args.command == "query":
            if not args.query_text:
                print_error("请提供查询文本")
                return 1
            await direct_query(manager, args.query_text)
        elif args.command == "test":
            return await run_full_test()
        elif args.command == "interactive":
            await interactive_query(manager)

        return 0

    try:
        exit_code = asyncio.run(run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)


if __name__ == "__main__":
    main()
