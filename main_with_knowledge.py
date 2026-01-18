#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 - 知识库增强版

集成知识库和语义搜索功能，支持：
1. 从知识库获取相关文档作为上下文
2. 使用语义搜索增强代码理解
3. 自我更新迭代能力
"""
import asyncio
import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from coordinator import Orchestrator, OrchestratorConfig
from cursor.client import CursorAgentConfig
from knowledge import KnowledgeManager, KnowledgeStorage

# 默认 Cursor 文档知识库名称
CURSOR_DOCS_KB_NAME = "cursor-docs"

# Cursor 相关关键词，用于自动检测是否需要知识库上下文
CURSOR_KEYWORDS = [
    "cursor", "agent", "cli", "mcp", "hook", "subagent", "skill",
    "stream-json", "output-format", "cursor-agent", "--force", "--print",
    "cursor.com", "cursor api", "cursor 命令", "cursor 工具",
]


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )
    logger.add(
        "logs/agent_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="规划者-执行者 多Agent系统 (知识库增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用知识库上下文执行任务
  python main_with_knowledge.py "优化 CLI 参数处理" --use-knowledge
  
  # 搜索知识库获取相关信息
  python main_with_knowledge.py "实现 stream-json 解析" --search-knowledge "stream-json output format"
  
  # 自我更新模式：根据知识库更新自身代码
  python main_with_knowledge.py "根据最新 Cursor CLI 文档更新代码" --self-update
        """,
    )
    
    parser.add_argument(
        "goal",
        type=str,
        help="要完成的目标",
    )
    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="工作目录 (默认: 当前目录)",
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="Worker 池大小 (默认: 3)",
    )
    
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=10,
        help="最大迭代次数 (默认: 10)",
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="启用严格评审模式",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    # 知识库相关参数
    parser.add_argument(
        "--use-knowledge",
        action="store_true",
        help="使用知识库上下文增强",
    )
    
    parser.add_argument(
        "--search-knowledge",
        type=str,
        metavar="QUERY",
        help="搜索知识库获取相关信息作为上下文",
    )
    
    parser.add_argument(
        "--kb-name",
        type=str,
        default="default",
        help="知识库名称 (默认: default)",
    )
    
    parser.add_argument(
        "--kb-limit",
        type=int,
        default=5,
        help="知识库搜索结果数量限制 (默认: 5)",
    )
    
    # 自我更新模式
    parser.add_argument(
        "--self-update",
        action="store_true",
        help="自我更新模式：允许系统根据知识库更新自身代码",
    )
    
    # 语义搜索
    parser.add_argument(
        "--use-semantic-search",
        action="store_true",
        help="启用语义搜索增强",
    )
    
    return parser.parse_args()


async def search_knowledge_base(
    query: str,
    kb_name: str = "default",
    limit: int = 5,
) -> list[dict]:
    """搜索知识库
    
    Args:
        query: 搜索查询
        kb_name: 知识库名称
        limit: 返回结果数量
        
    Returns:
        相关文档列表
    """
    storage = KnowledgeStorage()
    await storage.initialize()
    
    results = await storage.search(query, limit=limit)
    
    knowledge_context = []
    for result in results:
        # 加载完整文档内容
        doc = await storage.load_document(result.doc_id)
        if doc:
            knowledge_context.append({
                "title": doc.title or result.url,
                "url": doc.url,
                "content": doc.content[:2000],  # 限制内容长度
                "score": result.score,
            })
    
    return knowledge_context


async def get_all_knowledge(kb_name: str = "default", limit: int = 10) -> list[dict]:
    """获取所有知识库文档摘要
    
    Args:
        kb_name: 知识库名称
        limit: 返回数量限制
        
    Returns:
        文档摘要列表
    """
    storage = KnowledgeStorage()
    await storage.initialize()
    
    entries = await storage.list_documents(limit=limit)
    
    knowledge_context = []
    for entry in entries:
        doc = await storage.load_document(entry.doc_id)
        if doc:
            knowledge_context.append({
                "title": doc.title or entry.url,
                "url": doc.url,
                "summary": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
            })
    
    return knowledge_context


async def initialize_cursor_docs_kb() -> KnowledgeManager:
    """初始化 Cursor 文档知识库
    
    在 Agent 启动时加载 cursor-docs 知识库，用于自动补充 Cursor 相关上下文。
    
    Returns:
        初始化好的 KnowledgeManager 实例
    """
    logger.info(f"初始化 Cursor 文档知识库: {CURSOR_DOCS_KB_NAME}")
    
    manager = KnowledgeManager(name=CURSOR_DOCS_KB_NAME)
    await manager.initialize()
    
    doc_count = len(manager)
    if doc_count > 0:
        logger.info(f"Cursor 文档知识库已加载: {doc_count} 个文档")
    else:
        logger.warning("Cursor 文档知识库为空，可使用 'python -m knowledge.cli add' 添加文档")
    
    return manager


def is_cursor_related_query(query: str) -> bool:
    """检测查询是否与 Cursor 相关
    
    Args:
        query: 用户查询或目标
        
    Returns:
        是否与 Cursor 相关
    """
    query_lower = query.lower()
    return any(keyword.lower() in query_lower for keyword in CURSOR_KEYWORDS)


async def search_cursor_docs(
    manager: KnowledgeManager,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """从 Cursor 文档知识库搜索相关内容
    
    Args:
        manager: KnowledgeManager 实例
        query: 搜索查询
        limit: 返回结果数量
        
    Returns:
        相关文档列表
    """
    if len(manager) == 0:
        return []
    
    results = await manager.search_async(query, max_results=limit, search_mode="hybrid")
    
    knowledge_context = []
    for result in results:
        doc = manager.get_document(result.doc_id)
        if doc:
            knowledge_context.append({
                "title": doc.title or result.url,
                "url": doc.url,
                "content": doc.content[:2000],
                "score": result.score,
            })
    
    return knowledge_context


async def run_orchestrator(args: argparse.Namespace) -> dict:
    """运行编排器"""
    # 初始化 Cursor 文档知识库（始终加载，用于自动检测）
    cursor_docs_manager = await initialize_cursor_docs_kb()
    
    # 准备知识库上下文
    knowledge_context = None
    
    if args.search_knowledge:
        logger.info(f"搜索知识库: {args.search_knowledge}")
        knowledge_context = await search_knowledge_base(
            query=args.search_knowledge,
            kb_name=args.kb_name,
            limit=args.kb_limit,
        )
        logger.info(f"找到 {len(knowledge_context)} 个相关文档")
    
    elif args.use_knowledge:
        logger.info("加载知识库上下文...")
        knowledge_context = await get_all_knowledge(
            kb_name=args.kb_name,
            limit=args.kb_limit,
        )
        logger.info(f"加载了 {len(knowledge_context)} 个文档")
    
    # 自动检测：如果目标与 Cursor 相关，自动搜索 cursor-docs 知识库
    if not knowledge_context and is_cursor_related_query(args.goal):
        logger.info("检测到 Cursor 相关问题，自动搜索 cursor-docs 知识库...")
        cursor_context = await search_cursor_docs(
            manager=cursor_docs_manager,
            query=args.goal,
            limit=args.kb_limit,
        )
        if cursor_context:
            knowledge_context = cursor_context
            logger.info(f"从 cursor-docs 找到 {len(knowledge_context)} 个相关文档")
    
    # 如果是自我更新模式，调整工作目录和目标
    if args.self_update:
        # 确保工作目录是项目根目录
        project_root = Path(__file__).parent
        args.directory = str(project_root)
        logger.info(f"自我更新模式: 工作目录设置为 {args.directory}")
        
        # 如果没有搜索知识库，自动搜索与目标相关的内容
        if not knowledge_context:
            logger.info("自动搜索相关知识库内容...")
            knowledge_context = await search_knowledge_base(
                query=args.goal,
                kb_name=args.kb_name,
                limit=10,
            )
    
    # 创建配置
    cursor_config = CursorAgentConfig(
        working_directory=args.directory,
    )
    
    config = OrchestratorConfig(
        working_directory=args.directory,
        max_iterations=args.max_iterations,
        worker_pool_size=args.workers,
        strict_review=args.strict,
        cursor_config=cursor_config,
    )
    
    # 构建增强的目标（包含知识库上下文）
    enhanced_goal = args.goal
    if knowledge_context:
        context_text = "\n\n## 参考文档（来自知识库）\n\n"
        for i, doc in enumerate(knowledge_context, 1):
            context_text += f"### {i}. {doc['title']}\n"
            context_text += f"URL: {doc['url']}\n"
            if 'content' in doc:
                context_text += f"内容:\n```\n{doc['content'][:1000]}\n```\n\n"
            elif 'summary' in doc:
                context_text += f"摘要: {doc['summary']}\n\n"
        
        enhanced_goal = f"{args.goal}\n{context_text}"
    
    # 创建并运行编排器（传递知识库管理器）
    orchestrator = Orchestrator(config, knowledge_manager=cursor_docs_manager)
    
    # 如果启用语义搜索，初始化并注入
    if args.use_semantic_search:
        try:
            from indexing import SemanticSearch
            semantic_search = SemanticSearch()
            await semantic_search.initialize()
            # TODO: 注入到 orchestrator
            logger.info("语义搜索已启用")
        except Exception as e:
            logger.warning(f"语义搜索初始化失败: {e}")
    
    result = await orchestrator.run(enhanced_goal)
    
    return result


def print_result(result: dict) -> None:
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)
    
    print(f"\n状态: {'成功' if result.get('success') else '未完成'}")
    print(f"目标: {result.get('goal', 'N/A')[:100]}...")
    print(f"完成迭代: {result.get('iterations_completed', 0)}")
    print("\n任务统计:")
    print(f"  - 创建: {result.get('total_tasks_created', 0)}")
    print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
    print(f"  - 失败: {result.get('total_tasks_failed', 0)}")
    
    if result.get('final_score'):
        print(f"\n最终评分: {result['final_score']:.1f}")
    
    print("=" * 60)


def main() -> None:
    """主函数"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("规划者-执行者 多Agent系统 (知识库增强版) 启动")
    logger.info(f"目标: {args.goal}")
    
    if args.self_update:
        logger.warning("⚠️ 自我更新模式已启用 - 系统将修改自身代码")
    
    try:
        result = asyncio.run(run_orchestrator(args))
        print_result(result)
        
        sys.exit(0 if result.get("success") else 1)
        
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
