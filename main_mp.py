#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 主入口（多进程版本）"""
import asyncio
import argparse
import sys
from typing import Any
from pathlib import Path
from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from coordinator.orchestrator_mp import MultiProcessOrchestrator, MultiProcessOrchestratorConfig


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )
    
    # 文件日志
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/agent_mp_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="规划者-执行者 多Agent系统（多进程版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_mp.py "实现一个 REST API 服务"
  python main_mp.py "重构 src 目录下的代码" --workers 5
  python main_mp.py "添加单元测试" --strict --max-iterations 5

模型配置示例:
  python main_mp.py "实现功能" --planner-model gpt-5.2-high --worker-model opus-4.5-thinking
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
        help="Worker 进程数量 (默认: 3)",
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
        "--no-sub-planners",
        action="store_true",
        help="禁用子规划者",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    parser.add_argument(
        "--planning-timeout",
        type=float,
        default=120.0,
        help="规划超时时间（秒）",
    )
    
    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=300.0,
        help="任务执行超时时间（秒）",
    )
    
    # 模型配置
    parser.add_argument(
        "--planner-model",
        type=str,
        default="gpt-5.2-high",
        help="规划者使用的模型 (默认: gpt-5.2-high)",
    )
    
    parser.add_argument(
        "--worker-model",
        type=str,
        default="opus-4.5-thinking",
        help="执行者使用的模型 (默认: opus-4.5-thinking)",
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
    
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """加载 YAML 配置"""
    if not config_path.exists():
        return {}
    try:
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        logger.warning(f"加载配置失败: {e}")
        return {}


def resolve_stream_log_config(args: argparse.Namespace, config_data: dict[str, Any]) -> dict[str, Any]:
    """解析流式日志配置（CLI 优先）"""
    logging_config = config_data.get("logging", {}) if isinstance(config_data, dict) else {}
    stream_config = logging_config.get("stream_json", {}) if isinstance(logging_config, dict) else {}

    enabled = stream_config.get("enabled", False)
    console = stream_config.get("console", True)
    detail_dir = stream_config.get("detail_dir", "logs/stream_json/detail/")
    raw_dir = stream_config.get("raw_dir", "logs/stream_json/raw/")

    if args.stream_log_enabled is not None:
        enabled = args.stream_log_enabled
    if args.stream_log_console is not None:
        console = args.stream_log_console
    if args.stream_log_detail_dir:
        detail_dir = args.stream_log_detail_dir
    if args.stream_log_raw_dir:
        raw_dir = args.stream_log_raw_dir

    return {
        "enabled": enabled,
        "console": console,
        "detail_dir": detail_dir,
        "raw_dir": raw_dir,
    }


async def run_orchestrator(args: argparse.Namespace) -> dict:
    """运行编排器"""
    config_data = load_yaml_config(Path(__file__).with_name("config.yaml"))
    stream_config = resolve_stream_log_config(args, config_data)

    config = MultiProcessOrchestratorConfig(
        working_directory=args.directory,
        max_iterations=args.max_iterations,
        worker_count=args.workers,
        enable_sub_planners=not args.no_sub_planners,
        strict_review=args.strict,
        planning_timeout=args.planning_timeout,
        execution_timeout=args.execution_timeout,
        # 模型配置
        planner_model=args.planner_model,
        worker_model=args.worker_model,
        reviewer_model=args.worker_model,  # 评审者与执行者使用相同模型
        stream_events_enabled=stream_config["enabled"],
        stream_log_console=stream_config["console"],
        stream_log_detail_dir=stream_config["detail_dir"],
        stream_log_raw_dir=stream_config["raw_dir"],
    )
    
    logger.info("模型配置:")
    logger.info(f"  - 规划者: {config.planner_model}")
    logger.info(f"  - 执行者: {config.worker_model}")
    logger.info(f"  - 评审者: {config.reviewer_model}")
    
    orchestrator = MultiProcessOrchestrator(config)
    result = await orchestrator.run(args.goal)
    
    return result


def print_result(result: dict) -> None:
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果（多进程模式）")
    print("=" * 60)
    
    print(f"\n状态: {'成功' if result.get('success') else '未完成'}")
    print(f"目标: {result.get('goal', 'N/A')}")
    print(f"完成迭代: {result.get('iterations_completed', 0)}")
    
    print("\n任务统计:")
    print(f"  - 创建: {result.get('total_tasks_created', 0)}")
    print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
    print(f"  - 失败: {result.get('total_tasks_failed', 0)}")
    
    # 进程信息
    process_info = result.get('process_info', {})
    if process_info:
        print("\n进程信息:")
        for agent_id, info in process_info.items():
            status = "存活" if info.get('alive') else "已停止"
            print(f"  - {info.get('type', 'unknown')}: PID {info.get('pid', 'N/A')} ({status})")
    
    # 迭代详情
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
    
    logger.info("规划者-执行者 多Agent系统（多进程版本）启动")
    logger.info(f"目标: {args.goal}")
    logger.info(f"Worker 进程数: {args.workers}")
    
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
