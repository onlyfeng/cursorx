#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 主入口（多进程版本）"""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.committer import CommitterAgent, CommitterConfig
from coordinator.orchestrator_mp import MultiProcessOrchestrator, MultiProcessOrchestratorConfig
from cursor.client import CursorAgentConfig


def parse_max_iterations(value: str) -> int:
    """解析最大迭代次数参数

    Args:
        value: 参数值，可以是数字、MAX、-1 或 0

    Returns:
        迭代次数，-1 表示无限迭代
    """
    value_upper = value.upper().strip()

    # MAX 或 -1 或 0 表示无限迭代
    if value_upper in ("MAX", "UNLIMITED", "INF", "INFINITE"):
        return -1

    try:
        num = int(value)
        # -1 或 0 也表示无限迭代
        if num <= 0:
            return -1
        return num
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"无效的迭代次数: {value}。使用正整数或 MAX/-1 表示无限迭代"
        )


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
        type=str,
        default="10",
        help="最大迭代次数 (默认: 10，使用 MAX 或 -1 表示无限迭代直到完成或用户中断)",
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

    # 自动提交相关参数
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="任务完成后自动提交代码更改",
    )

    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="自动推送到远程仓库（需配合 --auto-commit 使用）",
    )

    parser.add_argument(
        "--commit-message",
        type=str,
        default="",
        help="自定义提交信息前缀（默认使用自动生成的提交信息）",
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


async def run_commit_phase(
    working_directory: str,
    auto_push: bool = False,
    commit_message_prefix: str = "",
    iterations_completed: int = 0,
    tasks_completed: int = 0,
) -> dict[str, Any]:
    """执行提交阶段

    Args:
        working_directory: 工作目录
        auto_push: 是否推送到远程
        commit_message_prefix: 提交信息前缀
        iterations_completed: 完成的迭代次数
        tasks_completed: 完成的任务数量

    Returns:
        提交结果字典，包含:
        - total_commits: 总提交次数
        - commit_hashes: 提交哈希列表
        - commit_messages: 提交信息列表
        - pushed_commits: 已推送次数
        - files_changed: 变更文件列表
    """
    logger.info("=" * 50)
    logger.info("提交阶段")
    logger.info("=" * 50)

    # 创建 CommitterAgent
    cursor_config = CursorAgentConfig(working_directory=working_directory)
    committer_config = CommitterConfig(
        working_directory=working_directory,
        auto_push=auto_push,
        commit_message_style="conventional",
        cursor_config=cursor_config,
    )
    committer = CommitterAgent(committer_config)

    # 检查是否有变更
    status = committer.check_status()
    if not status.get("is_repo"):
        logger.warning("当前目录不是 Git 仓库，跳过提交")
        return {
            "total_commits": 0,
            "commit_hashes": [],
            "commit_messages": [],
            "pushed_commits": 0,
            "files_changed": [],
            "error": "不是 Git 仓库",
        }

    if not status.get("has_changes"):
        logger.info("没有需要提交的变更")
        return {
            "total_commits": 0,
            "commit_hashes": [],
            "commit_messages": [],
            "pushed_commits": 0,
            "files_changed": [],
        }

    # 生成或使用提供的提交信息
    if commit_message_prefix:
        commit_message = commit_message_prefix
    else:
        # 根据执行结果生成提交信息
        commit_message = await committer.generate_commit_message()

    # 添加迭代信息到提交信息
    if iterations_completed > 0 or tasks_completed > 0:
        suffix = f"\n\n迭代次数: {iterations_completed}, 完成任务: {tasks_completed}"
        if not commit_message_prefix:
            # 自动生成的信息可以直接追加
            commit_message = commit_message.rstrip() + suffix
        else:
            # 用户自定义信息保持原样
            commit_message = commit_message_prefix

    # 执行提交
    commit_result = committer.commit(commit_message)

    if not commit_result.success:
        logger.error(f"提交失败: {commit_result.error}")
        return {
            "total_commits": 0,
            "commit_hashes": [],
            "commit_messages": [],
            "pushed_commits": 0,
            "files_changed": commit_result.files_changed,
            "error": commit_result.error,
        }

    logger.info(f"提交成功: {commit_result.commit_hash[:8]} - {commit_message.split(chr(10))[0][:50]}")

    # 可选推送
    pushed_commits = 0
    push_error = ""
    if auto_push:
        push_result = committer.push()
        if push_result.success:
            pushed_commits = 1
            logger.info("推送成功")
        else:
            push_error = push_result.error
            logger.warning(f"推送失败: {push_error}")

    # 获取提交摘要
    summary = committer.get_commit_summary()

    result = {
        "total_commits": summary.get("successful_commits", 1),
        "commit_hashes": summary.get("commit_hashes", [commit_result.commit_hash]),
        "commit_messages": [commit_message],
        "pushed_commits": pushed_commits,
        "files_changed": summary.get("files_changed", commit_result.files_changed),
    }

    if push_error:
        result["push_error"] = push_error

    return result


async def run_orchestrator(args: argparse.Namespace) -> dict:
    """运行编排器"""
    config_data = load_yaml_config(Path(__file__).with_name("config.yaml"))
    stream_config = resolve_stream_log_config(args, config_data)

    # 解析最大迭代次数
    max_iterations = parse_max_iterations(args.max_iterations)
    if max_iterations == -1:
        logger.info("无限迭代模式已启用（按 Ctrl+C 中断）")

    config = MultiProcessOrchestratorConfig(
        working_directory=args.directory,
        max_iterations=max_iterations,
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

    if args.auto_commit:
        logger.info("自动提交: 启用")
        if args.auto_push:
            logger.info("自动推送: 启用")

    orchestrator = MultiProcessOrchestrator(config)
    result = await orchestrator.run(args.goal)

    # 完成后提交阶段
    if args.auto_commit:
        commits_result = await run_commit_phase(
            working_directory=args.directory,
            auto_push=args.auto_push,
            commit_message_prefix=args.commit_message,
            iterations_completed=result.get("iterations_completed", 0),
            tasks_completed=result.get("total_tasks_completed", 0),
        )
        result["commits"] = commits_result

        # 更新 pushed 标志（与协程版 Orchestrator 对齐）
        if commits_result.get("pushed_commits", 0) > 0:
            result["pushed"] = True

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

    # 提交信息
    commits_info = result.get('commits', {})
    if commits_info:
        print("\n提交信息:")
        total_commits = commits_info.get('total_commits', 0)
        if total_commits > 0:
            print(f"  提交数量: {total_commits}")

            # 显示提交哈希
            commit_hashes = commits_info.get('commit_hashes', [])
            if commit_hashes:
                for i, hash_val in enumerate(commit_hashes[-3:], 1):  # 显示最近3个
                    short_hash = hash_val[:8] if len(hash_val) > 8 else hash_val
                    print(f"  提交 {i}: {short_hash}")

            # 显示提交信息摘要
            commit_messages = commits_info.get('commit_messages', [])
            if commit_messages:
                print("  提交信息摘要:")
                for msg in commit_messages[-3:]:  # 显示最近3条
                    # 截取第一行作为摘要
                    summary = msg.split('\n')[0][:60]
                    print(f"    - {summary}")

            # 显示变更文件数量
            files_changed = commits_info.get('files_changed', [])
            if files_changed:
                print(f"  变更文件: {len(files_changed)} 个")

            # 显示推送状态
            pushed_commits = commits_info.get('pushed_commits', 0)
            if pushed_commits > 0:
                print(f"  推送状态: 已推送 {pushed_commits} 个提交到远程仓库")
            elif commits_info.get('push_error'):
                print(f"  推送状态: 推送失败 - {commits_info['push_error']}")
        else:
            if commits_info.get('error'):
                print(f"  错误: {commits_info['error']}")
            else:
                print("  无代码更改，未创建提交")

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
