#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 - 知识库增强版

集成知识库和语义搜索功能，支持：
1. 从知识库获取相关文档作为上下文
2. 使用语义搜索增强代码理解
3. 自我更新迭代能力
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from coordinator import Orchestrator, OrchestratorConfig
from core.config import (
    build_cloud_auth_config,
    get_config,
    parse_max_iterations,
    resolve_orchestrator_settings,
    resolve_stream_log_config,
)
from cursor.client import CursorAgentConfig
from cursor.cloud_client import CloudAuthConfig
from cursor.executor import ExecutionMode
from knowledge import KnowledgeManager, KnowledgeStorage

# 默认 Cursor 文档知识库名称
CURSOR_DOCS_KB_NAME = "cursor-docs"

# Cursor 相关关键词，用于自动检测是否需要知识库上下文
CURSOR_KEYWORDS = [
    "cursor",
    "agent",
    "cli",
    "mcp",
    "hook",
    "subagent",
    "skill",
    "stream-json",
    "output-format",
    "cursor-agent",
    "--force",
    "--print",
    "cursor.com",
    "cursor api",
    "cursor 命令",
    "cursor 工具",
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
    """解析命令行参数

    默认值来源优先级:
    1. CLI 显式参数 (最高优先级)
    2. config.yaml (通过 core.config.resolve_orchestrator_settings())
    3. 代码默认值 (最低优先级)

    tri-state 参数设计:
    - 部分参数使用 default=None 实现三态逻辑
    - None 表示"未显式指定"，允许后续按优先级合并
    - 在 run_orchestrator 中通过 resolve_orchestrator_settings 解析最终值
    """
    # 加载配置，仅用于帮助信息显示（不作为 argparse 默认值）
    config = get_config()
    cfg_workers = config.system.worker_pool_size
    cfg_max_iterations = config.system.max_iterations
    cfg_kb_limit = config.worker.knowledge_integration.max_docs
    cfg_enable_sub_planners = config.system.enable_sub_planners
    cfg_strict_review = config.system.strict_review
    cfg_planner_model = config.models.planner
    cfg_worker_model = config.models.worker
    cfg_reviewer_model = config.models.reviewer
    cfg_planner_timeout = config.planner.timeout
    cfg_worker_timeout = config.worker.task_timeout
    cfg_reviewer_timeout = config.reviewer.timeout
    # Cloud Agent 配置默认值
    cfg_execution_mode = config.cloud_agent.execution_mode
    cfg_cloud_timeout = config.cloud_agent.timeout
    cfg_cloud_auth_timeout = config.cloud_agent.auth_timeout

    parser = argparse.ArgumentParser(
        description="规划者-执行者 多Agent系统 (知识库增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 通过统一入口运行 knowledge 模式
  python run.py --mode knowledge "优化 CLI 参数处理" --use-knowledge
  python run.py --mode knowledge "实现 stream-json 解析" --search-knowledge "stream-json output format"
  python run.py --mode knowledge "根据最新 Cursor CLI 文档更新代码" --self-update

  # 或直接运行本脚本
  python scripts/run_knowledge.py "优化 CLI 参数处理" --use-knowledge
  python scripts/run_knowledge.py "实现 stream-json 解析" --search-knowledge "stream-json output format"
  python scripts/run_knowledge.py "根据最新 Cursor CLI 文档更新代码" --self-update
        """,
    )

    parser.add_argument(
        "goal",
        type=str,
        help="要完成的目标",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="工作目录 (默认: 当前目录)",
    )

    # workers 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=f"Worker 池大小 (默认: {cfg_workers}，来自 config.yaml)",
    )

    # max_iterations 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-m",
        "--max-iterations",
        type=str,
        default=None,
        help=f"最大迭代次数 (默认: {cfg_max_iterations}，使用 MAX 或 -1 表示无限迭代直到完成或用户中断)",
    )

    # 严格评审模式 - 互斥组支持 tri-state (None=未指定, 使用 config 默认值)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict_review",
        action="store_true",
        default=None,
        help=f"启用严格评审模式 (config.yaml 默认: {cfg_strict_review})",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict_review",
        action="store_false",
        help="禁用严格评审模式",
    )

    # 子规划者 - 互斥组支持 tri-state (None=未指定, 使用 config 默认值)
    sub_planners_group = parser.add_mutually_exclusive_group()
    sub_planners_group.add_argument(
        "--sub-planners",
        dest="enable_sub_planners",
        action="store_true",
        default=None,
        help=f"启用子规划者 (config.yaml 默认: {cfg_enable_sub_planners})",
    )
    sub_planners_group.add_argument(
        "--no-sub-planners",
        dest="enable_sub_planners",
        action="store_false",
        help="禁用子规划者",
    )

    parser.add_argument(
        "-v",
        "--verbose",
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
        default=cfg_kb_limit,
        help=f"知识库搜索结果数量限制 (默认: {cfg_kb_limit}，来自 config.yaml)",
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

    # ============================================================
    # 模型配置参数 - tri-state (None=未指定，使用 config.yaml)
    # ============================================================
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help=f"Planner 模型 (默认: {cfg_planner_model}，来自 config.yaml)",
    )
    parser.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help=f"Worker 模型 (默认: {cfg_worker_model}，来自 config.yaml)",
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=None,
        help=f"Reviewer 模型 (默认: {cfg_reviewer_model}，来自 config.yaml)",
    )

    # ============================================================
    # 超时配置参数 - tri-state (None=未指定，使用 config.yaml)
    # ============================================================
    parser.add_argument(
        "--planner-timeout",
        type=float,
        default=None,
        help=f"Planner 超时（秒）(默认: {cfg_planner_timeout}，来自 config.yaml)",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help=f"Worker 超时（秒）(默认: {cfg_worker_timeout}，来自 config.yaml)",
    )
    parser.add_argument(
        "--reviewer-timeout",
        type=float,
        default=None,
        help=f"Reviewer 超时（秒）(默认: {cfg_reviewer_timeout}，来自 config.yaml)",
    )

    # ============================================================
    # stream-json 流式日志配置参数
    # ============================================================
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream-log",
        dest="stream_log_enabled",
        action="store_true",
        help="启用 stream-json 流式日志",
    )
    stream_group.add_argument(
        "--no-stream-log",
        dest="stream_log_enabled",
        action="store_false",
        help="禁用 stream-json 流式日志",
    )
    parser.set_defaults(stream_log_enabled=None)

    parser.add_argument(
        "--stream-log-console",
        dest="stream_log_console",
        action="store_true",
        help="流式日志输出到控制台",
    )
    parser.add_argument(
        "--no-stream-log-console",
        dest="stream_log_console",
        action="store_false",
        help="关闭流式日志控制台输出",
    )
    parser.set_defaults(stream_log_console=None)

    parser.add_argument(
        "--stream-log-detail-dir",
        type=str,
        default=None,
        help="stream-json 详细日志目录",
    )
    parser.add_argument(
        "--stream-log-raw-dir",
        type=str,
        default=None,
        help="stream-json 原始日志目录",
    )

    # ============================================================
    # 执行模式和 Cloud 配置参数 - tri-state (None=未指定，使用 config.yaml)
    # ============================================================
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default=None,
        help=f"执行模式: cli=本地CLI, auto=自动选择(Cloud优先), cloud=强制Cloud (默认: {cfg_execution_mode}，来自 config.yaml)",
    )

    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="Cloud API Key（可选，默认从 CURSOR_API_KEY 环境变量读取）",
    )

    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=None,
        help=f"Cloud 执行超时时间（秒，默认 {cfg_cloud_timeout}，来自 config.yaml cloud_agent.timeout）",
    )

    parser.add_argument(
        "--cloud-auth-timeout",
        type=int,
        default=None,
        help=f"Cloud 认证超时时间（秒，默认 {cfg_cloud_auth_timeout}，来自 config.yaml）",
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
            knowledge_context.append(
                {
                    "title": doc.title or result.url,
                    "url": doc.url,
                    "content": doc.content[:2000],  # 限制内容长度
                    "score": result.score,
                }
            )

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
            knowledge_context.append(
                {
                    "title": doc.title or entry.url,
                    "url": doc.url,
                    "summary": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                }
            )

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
            knowledge_context.append(
                {
                    "title": doc.title or result.url,
                    "url": doc.url,
                    "content": doc.content[:2000],
                    "score": result.score,
                }
            )

    return knowledge_context


def _build_cloud_auth_config(
    args: argparse.Namespace | Any,
    resolved: dict,
) -> CloudAuthConfig | None:
    """构建 Cloud 认证配置

    使用统一的 build_cloud_auth_config 函数构建配置，
    确保配置来源优先级一致。

    优先级（从高到低）：
    1. 命令行参数 --cloud-api-key（CLI 显式指定）
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml 中的 cloud_agent.api_key

    其他配置（auth_timeout, base_url 等）也遵循 CLI > config.yaml > DEFAULT_* 优先级。

    Args:
        args: 命令行参数
        resolved: resolve_orchestrator_settings 返回的配置字典

    Returns:
        CloudAuthConfig 或 None（未配置 key 时）
    """
    # 使用统一的 build_cloud_auth_config 函数
    cli_api_key = getattr(args, "cloud_api_key", None)
    cloud_auth_overrides = {
        "api_key": cli_api_key,
        "auth_timeout": resolved.get("cloud_auth_timeout"),
    }
    cloud_auth_dict = build_cloud_auth_config(overrides=cloud_auth_overrides)

    if not cloud_auth_dict:
        return None

    return CloudAuthConfig(**cloud_auth_dict)


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

    # 构建 CLI overrides 字典（仅包含显式指定的参数）
    # tri-state: None 表示未指定，让 resolve_orchestrator_settings 从 config.yaml 读取
    cli_overrides = {}

    # workers: CLI 显式参数（None 表示未指定）
    if args.workers is not None:
        cli_overrides["workers"] = args.workers

    # max_iterations: CLI 显式参数（None 表示未指定）
    if args.max_iterations is not None:
        cli_overrides["max_iterations"] = parse_max_iterations(args.max_iterations)

    # enable_sub_planners: CLI 显式参数（None 表示未指定）
    if args.enable_sub_planners is not None:
        cli_overrides["enable_sub_planners"] = args.enable_sub_planners

    # strict_review: CLI 显式参数（None 表示未指定）
    if args.strict_review is not None:
        cli_overrides["strict_review"] = args.strict_review

    # 模型配置: CLI 显式参数（None 表示未指定）
    if args.planner_model is not None:
        cli_overrides["planner_model"] = args.planner_model
    if args.worker_model is not None:
        cli_overrides["worker_model"] = args.worker_model
    if args.reviewer_model is not None:
        cli_overrides["reviewer_model"] = args.reviewer_model

    # 超时配置: CLI 显式参数（None 表示未指定）
    if args.planner_timeout is not None:
        cli_overrides["planner_timeout"] = args.planner_timeout
    if args.worker_timeout is not None:
        cli_overrides["worker_timeout"] = args.worker_timeout
    if args.reviewer_timeout is not None:
        cli_overrides["reviewer_timeout"] = args.reviewer_timeout

    # 执行模式和 Cloud 配置: CLI 显式参数（None 表示未指定）
    if args.execution_mode is not None:
        cli_overrides["execution_mode"] = args.execution_mode
    if args.cloud_timeout is not None:
        cli_overrides["cloud_timeout"] = args.cloud_timeout
    if args.cloud_auth_timeout is not None:
        cli_overrides["cloud_auth_timeout"] = args.cloud_auth_timeout

    # 使用 resolve_orchestrator_settings 统一解析
    resolved = resolve_orchestrator_settings(overrides=cli_overrides)

    # 无限迭代提示
    if resolved["max_iterations"] == -1:
        logger.info("无限迭代模式已启用（按 Ctrl+C 中断）")

    # 流式日志配置
    stream_config = resolve_stream_log_config(
        cli_enabled=args.stream_log_enabled,
        cli_console=args.stream_log_console,
        cli_detail_dir=args.stream_log_detail_dir,
        cli_raw_dir=args.stream_log_raw_dir,
    )

    # 解析执行模式
    execution_mode_str = resolved["execution_mode"]
    execution_mode_map = {
        "cli": ExecutionMode.CLI,
        "auto": ExecutionMode.AUTO,
        "cloud": ExecutionMode.CLOUD,
    }
    execution_mode = execution_mode_map.get(execution_mode_str, ExecutionMode.CLI)

    # 构建 Cloud 认证配置（按统一优先级规则）
    # 优先级: CLI --cloud-api-key > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
    cloud_auth_config = _build_cloud_auth_config(args, resolved)

    # 根据执行模式决定超时时间
    # - Cloud/Auto 模式：使用 resolved settings 中的 cloud_timeout
    # - CLI 模式：使用标准 timeout
    if execution_mode in (ExecutionMode.CLOUD, ExecutionMode.AUTO):
        timeout = resolved["cloud_timeout"]
        logger.info(f"Cloud/Auto 执行模式，使用 cloud_timeout={timeout}s")
    else:
        timeout = None  # 使用默认超时

    # 创建配置
    cursor_config = CursorAgentConfig(
        working_directory=args.directory,
        stream_events_enabled=stream_config["enabled"],
        stream_log_console=stream_config["console"],
        stream_log_detail_dir=stream_config["detail_dir"],
        stream_log_raw_dir=stream_config["raw_dir"],
        timeout=timeout if timeout else 300,  # 默认 300 秒
    )

    config = OrchestratorConfig(
        working_directory=args.directory,
        max_iterations=resolved["max_iterations"],
        worker_pool_size=resolved["workers"],
        enable_sub_planners=resolved["enable_sub_planners"],
        strict_review=resolved["strict_review"],
        cursor_config=cursor_config,
        # 流式日志配置
        stream_events_enabled=stream_config["enabled"],
        stream_log_console=stream_config["console"],
        stream_log_detail_dir=stream_config["detail_dir"],
        stream_log_raw_dir=stream_config["raw_dir"],
        # 模型配置
        planner_model=resolved["planner_model"],
        worker_model=resolved["worker_model"],
        reviewer_model=resolved["reviewer_model"],
        # 超时配置
        planner_timeout=resolved["planner_timeout"],
        worker_task_timeout=resolved["worker_timeout"],
        reviewer_timeout=resolved["reviewer_timeout"],
        # 执行模式和 Cloud 认证配置
        execution_mode=execution_mode,
        cloud_auth_config=cloud_auth_config,
    )

    # 构建增强的目标（包含知识库上下文）
    enhanced_goal = args.goal
    if knowledge_context:
        context_text = "\n\n## 参考文档（来自知识库）\n\n"
        for i, doc in enumerate(knowledge_context, 1):
            context_text += f"### {i}. {doc['title']}\n"
            context_text += f"URL: {doc['url']}\n"
            if "content" in doc:
                context_text += f"内容:\n```\n{doc['content'][:1000]}\n```\n\n"
            elif "summary" in doc:
                context_text += f"摘要: {doc['summary']}\n\n"

        enhanced_goal = f"{args.goal}\n{context_text}"

    # 创建并运行编排器（传递知识库管理器）
    orchestrator = Orchestrator(config, knowledge_manager=cursor_docs_manager)

    # 如果启用语义搜索，初始化并注入
    if args.use_semantic_search:
        try:
            from indexing import SemanticSearch

            vector_store = cursor_docs_manager._vector_store
            embedding_model = getattr(vector_store, "_embedding_model", None)
            vector_backend = getattr(vector_store, "_vector_store", None)
            if vector_store and embedding_model is not None and vector_backend is not None:
                semantic_search = SemanticSearch(
                    embedding_model,
                    vector_backend,
                )
                # TODO: 注入到 orchestrator
                logger.info("语义搜索已启用")
            else:
                logger.warning("语义搜索初始化失败: 向量存储未初始化")
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

    if result.get("final_score"):
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
