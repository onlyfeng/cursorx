#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 主入口"""
import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from coordinator import Orchestrator, OrchestratorConfig
from core.config import (
    get_config,
    resolve_stream_log_config,
    resolve_orchestrator_settings,
    parse_max_iterations,
    build_cloud_client_config,
    build_cursor_agent_config,
    build_cloud_auth_config,
)
from cursor.client import CursorAgentConfig
from cursor.cloud_client import CloudAuthConfig, CloudClientFactory
from cursor.executor import ExecutionMode


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
    cfg_worker_pool_size = config.system.worker_pool_size
    cfg_max_iterations = config.system.max_iterations
    cfg_enable_sub_planners = config.system.enable_sub_planners
    cfg_strict_review = config.system.strict_review
    cfg_execution_mode = config.cloud_agent.execution_mode
    cfg_cloud_timeout = config.cloud_agent.timeout
    cfg_cloud_auth_timeout = config.cloud_agent.auth_timeout

    parser = argparse.ArgumentParser(
        description="规划者-执行者 多Agent系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 通过统一入口运行 basic 模式
  python run.py --mode basic "实现一个 REST API 服务"
  python run.py --mode basic "重构 src 目录下的代码" --workers 5
  python run.py --mode basic "添加单元测试" --strict --max-iterations 5

  # 或直接运行本脚本
  python scripts/run_basic.py "实现一个 REST API 服务"
  python scripts/run_basic.py "重构 src 目录下的代码" --workers 5
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

    # workers 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help=f"Worker 池大小 (默认: {cfg_worker_pool_size}，来自 config.yaml)",
    )

    # max_iterations 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-m", "--max-iterations",
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
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟模式（测试用）",
    )

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

    # 执行模式参数（tri-state）- 与 run.py 对齐
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default=None,
        help=f"执行模式: cli=本地CLI, auto=自动选择(Cloud优先), cloud=强制Cloud (默认: {cfg_execution_mode}，来自 config.yaml)",
    )

    # Cloud 认证参数（与 run.py 对齐）
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="Cloud API Key（可选，默认从 CURSOR_API_KEY 环境变量读取）",
    )

    # Cloud 超时参数 - tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--cloud-auth-timeout",
        type=int,
        default=None,
        help=f"Cloud 认证超时时间（秒，默认 {cfg_cloud_auth_timeout}，来自 config.yaml）",
    )

    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=None,
        help=f"Cloud 执行超时时间（秒，默认 {cfg_cloud_timeout}，来自 config.yaml）",
    )

    return parser.parse_args()


async def run_orchestrator(args: argparse.Namespace) -> dict:
    """运行编排器"""
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

    # execution_mode: CLI 显式参数（None 表示未指定）
    cli_execution_mode = getattr(args, "execution_mode", None)
    if cli_execution_mode is not None:
        cli_overrides["execution_mode"] = cli_execution_mode

    # cloud_timeout: CLI 显式参数（None 表示未指定）
    cli_cloud_timeout = getattr(args, "cloud_timeout", None)
    if cli_cloud_timeout is not None:
        cli_overrides["cloud_timeout"] = cli_cloud_timeout

    # cloud_auth_timeout: CLI 显式参数（None 表示未指定）
    cli_cloud_auth_timeout = getattr(args, "cloud_auth_timeout", None)
    if cli_cloud_auth_timeout is not None:
        cli_overrides["cloud_auth_timeout"] = cli_cloud_auth_timeout

    # 使用 resolve_orchestrator_settings 统一解析
    resolved = resolve_orchestrator_settings(overrides=cli_overrides)

    # 无限迭代提示
    if resolved["max_iterations"] == -1:
        logger.info("无限迭代模式已启用（按 Ctrl+C 中断）")

    # 流式日志配置（统一解析）
    stream_config = resolve_stream_log_config(
        cli_enabled=args.stream_log_enabled,
        cli_console=args.stream_log_console,
        cli_detail_dir=args.stream_log_detail_dir,
        cli_raw_dir=args.stream_log_raw_dir,
    )

    # 使用统一的 build_cursor_agent_config 构建配置
    cursor_config_dict = build_cursor_agent_config(
        working_directory=args.directory,
        overrides={
            "stream_events_enabled": stream_config["enabled"],
            "stream_log_console": stream_config["console"],
            "stream_log_detail_dir": stream_config["detail_dir"],
            "stream_log_raw_dir": stream_config["raw_dir"],
        },
    )
    cursor_config = CursorAgentConfig(**cursor_config_dict)

    # 解析执行模式（从 resolved 获取统一解析后的值）
    execution_mode_str = resolved["execution_mode"]
    execution_mode_map = {
        "cli": ExecutionMode.CLI,
        "auto": ExecutionMode.AUTO,
        "cloud": ExecutionMode.CLOUD,
    }
    execution_mode = execution_mode_map.get(execution_mode_str, ExecutionMode.CLI)

    # 使用统一的 build_cloud_auth_config 构建 Cloud 认证配置
    # 优先级：--cloud-api-key > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
    cli_api_key = getattr(args, "cloud_api_key", None)
    cloud_auth_overrides = {
        "api_key": cli_api_key,
        "auth_timeout": resolved["cloud_auth_timeout"],
    }
    cloud_auth_dict = build_cloud_auth_config(overrides=cloud_auth_overrides)

    cloud_auth_config = None
    if cloud_auth_dict:
        cloud_auth_config = CloudAuthConfig(**cloud_auth_dict)
        logger.debug(f"已配置 Cloud 认证: auth_timeout={cloud_auth_dict['auth_timeout']}s")

    config = OrchestratorConfig(
        working_directory=args.directory,
        max_iterations=resolved["max_iterations"],
        worker_pool_size=resolved["workers"],
        enable_sub_planners=resolved["enable_sub_planners"],
        strict_review=resolved["strict_review"],
        cursor_config=cursor_config,
        stream_events_enabled=stream_config["enabled"],
        stream_log_console=stream_config["console"],
        stream_log_detail_dir=stream_config["detail_dir"],
        stream_log_raw_dir=stream_config["raw_dir"],
        # 模型配置（从 resolved 获取）
        planner_model=resolved.get("planner_model"),
        worker_model=resolved.get("worker_model"),
        reviewer_model=resolved.get("reviewer_model"),
        # 执行模式和 Cloud 认证配置
        execution_mode=execution_mode,
        cloud_auth_config=cloud_auth_config,
    )

    # 如果是模拟模式，替换 Cursor 客户端
    if args.mock:
        logger.info("使用模拟模式")
        # 这里可以注入 Mock 客户端

    # 日志记录执行模式
    logger.info(f"执行模式: {execution_mode.value}")
    if cloud_auth_config:
        logger.info("Cloud 认证已配置")

    # 创建并运行编排器
    orchestrator = Orchestrator(config)
    result = await orchestrator.run(args.goal)

    return result


def print_result(result: dict) -> None:
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)

    print(f"\n状态: {'成功' if result.get('success') else '未完成'}")
    print(f"目标: {result.get('goal', 'N/A')}")
    print(f"完成迭代: {result.get('iterations_completed', 0)}")
    print("\n任务统计:")
    print(f"  - 创建: {result.get('total_tasks_created', 0)}")
    print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
    print(f"  - 失败: {result.get('total_tasks_failed', 0)}")

    if result.get('final_score'):
        print(f"\n最终评分: {result['final_score']:.1f}")

    if result.get('iterations'):
        print("\n迭代详情:")
        for it in result['iterations']:
            status_emoji = "✓" if it.get('review_passed') else "→"
            print(f"  {status_emoji} 迭代 {it['id']}: {it['tasks_completed']}/{it['tasks_created']} 任务完成")

    print("=" * 60)


def main() -> None:
    """主函数"""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("规划者-执行者 多Agent系统 启动")
    logger.info(f"目标: {args.goal}")

    try:
        result = asyncio.run(run_orchestrator(args))
        print_result(result)

        # 根据结果设置退出码
        sys.exit(0 if result.get("success") else 1)

    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
