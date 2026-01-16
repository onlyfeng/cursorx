#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 主入口"""
import asyncio
import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from coordinator import Orchestrator, OrchestratorConfig
from cursor.client import CursorAgentConfig


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
        description="规划者-执行者 多Agent系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "实现一个 REST API 服务"
  python main.py "重构 src 目录下的代码" --workers 5
  python main.py "添加单元测试" --strict --max-iterations 5
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
        "--mock",
        action="store_true",
        help="使用模拟模式（测试用）",
    )
    
    return parser.parse_args()


async def run_orchestrator(args: argparse.Namespace) -> dict:
    """运行编排器"""
    # 创建配置
    cursor_config = CursorAgentConfig(
        working_directory=args.directory,
    )
    
    config = OrchestratorConfig(
        working_directory=args.directory,
        max_iterations=args.max_iterations,
        worker_pool_size=args.workers,
        enable_sub_planners=not args.no_sub_planners,
        strict_review=args.strict,
        cursor_config=cursor_config,
    )
    
    # 如果是模拟模式，替换 Cursor 客户端
    if args.mock:
        from cursor.client import MockCursorAgentClient
        logger.info("使用模拟模式")
        # 这里可以注入 Mock 客户端
    
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
    print(f"\n任务统计:")
    print(f"  - 创建: {result.get('total_tasks_created', 0)}")
    print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
    print(f"  - 失败: {result.get('total_tasks_failed', 0)}")
    
    if result.get('final_score'):
        print(f"\n最终评分: {result['final_score']:.1f}")
    
    if result.get('iterations'):
        print(f"\n迭代详情:")
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
