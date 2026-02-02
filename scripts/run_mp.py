#!/usr/bin/env python3
"""规划者-执行者 多Agent系统 主入口（多进程版本）"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from agents.committer import CommitterAgent, CommitterConfig
from coordinator.orchestrator_mp import MultiProcessOrchestrator, MultiProcessOrchestratorConfig
from core.config import (
    get_config,
    parse_max_iterations,
    resolve_orchestrator_settings,
    resolve_stream_log_config,
)
from core.execution_policy import ExecutionDecision, compute_decision_inputs
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

    # 文件日志
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/agent_mp_{time}.log",
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
    # 在函数内部按需加载配置，仅用于帮助信息显示（不作为 argparse 默认值）
    config = get_config()
    logger.debug(f"加载配置: config_path={getattr(config, '_config_path', 'unknown')}")

    # 从配置中获取帮助信息显示用的默认值
    cfg_workers = config.system.worker_pool_size
    cfg_max_iterations = config.system.max_iterations
    cfg_planning_timeout = config.planner.timeout
    cfg_execution_timeout = config.worker.task_timeout
    cfg_review_timeout = config.reviewer.timeout
    cfg_strict_review = config.system.strict_review
    cfg_enable_sub_planners = config.system.enable_sub_planners
    cfg_planner_model = config.models.planner
    cfg_worker_model = config.models.worker
    cfg_reviewer_model = config.models.reviewer

    parser = argparse.ArgumentParser(
        description="规划者-执行者 多Agent系统（多进程版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 通过统一入口运行 iterate 模式（多进程）
  python run.py --mode iterate "实现一个 REST API 服务"
  python run.py --mode iterate "重构 src 目录下的代码" --workers 5
  python run.py --mode iterate "添加单元测试" --strict --max-iterations 5

  # 或直接运行本脚本
  python scripts/run_mp.py "实现一个 REST API 服务"
  python scripts/run_mp.py "重构 src 目录下的代码" --workers 5

模型配置示例:
  python run.py --mode iterate "实现功能" --planner-model gpt-5.2-high --worker-model gpt-5.2-codex-high
  python scripts/run_mp.py "实现功能" --planner-model gpt-5.2-high --worker-model gpt-5.2-codex-high
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
        help=f"Worker 进程数量 (默认: {cfg_workers}，来自 config.yaml)",
    )

    # max_iterations 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-m",
        "--max-iterations",
        type=str,
        default=None,
        help=f"最大迭代次数 (默认: {cfg_max_iterations}，使用 MAX 或 -1 表示无限迭代直到完成或用户中断，来自 config.yaml)",
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

    # 超时参数使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--planning-timeout",
        type=float,
        default=None,
        help=f"规划超时时间（秒）(默认: {cfg_planning_timeout}，来自 config.yaml)",
    )

    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=None,
        help=f"任务执行超时时间（秒）(默认: {cfg_execution_timeout}，来自 config.yaml)",
    )

    parser.add_argument(
        "--review-timeout",
        type=float,
        default=None,
        help=f"评审超时时间（秒）(默认: {cfg_review_timeout}，来自 config.yaml)",
    )

    # 模型配置 - 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help=f"规划者使用的模型 (默认: {cfg_planner_model}，来自 config.yaml)",
    )

    parser.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help=f"执行者使用的模型 (默认: {cfg_worker_model}，来自 config.yaml)",
    )

    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=None,
        help=f"评审者使用的模型 (默认: {cfg_reviewer_model}，来自 config.yaml)",
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
        commit_message = commit_message_prefix if commit_message_prefix else commit_message.rstrip() + suffix

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
    # 构建 CLI overrides 字典（仅包含显式指定的参数）
    # tri-state: None 表示未指定，让 resolve_orchestrator_settings 从 config.yaml 读取
    cli_overrides = {}

    # workers: CLI 显式参数（None 表示未指定）
    if args.workers is not None:
        cli_overrides["workers"] = args.workers

    # max_iterations: CLI 显式参数（None 表示未指定）
    if args.max_iterations is not None:
        cli_overrides["max_iterations"] = parse_max_iterations(args.max_iterations)

    # 超时参数: CLI 显式参数（None 表示未指定）
    if args.planning_timeout is not None:
        cli_overrides["planner_timeout"] = args.planning_timeout
    if args.execution_timeout is not None:
        cli_overrides["worker_timeout"] = args.execution_timeout
    if args.review_timeout is not None:
        cli_overrides["reviewer_timeout"] = args.review_timeout

    # 模型配置: CLI 显式参数（None 表示未指定）
    if args.planner_model is not None:
        cli_overrides["planner_model"] = args.planner_model
    if args.worker_model is not None:
        cli_overrides["worker_model"] = args.worker_model
    if args.reviewer_model is not None:
        cli_overrides["reviewer_model"] = args.reviewer_model

    # enable_sub_planners: CLI 显式参数（None 表示未指定）
    if args.enable_sub_planners is not None:
        cli_overrides["enable_sub_planners"] = args.enable_sub_planners

    # strict_review: CLI 显式参数（None 表示未指定）
    if args.strict_review is not None:
        cli_overrides["strict_review"] = args.strict_review

    # auto_commit/auto_push
    if args.auto_commit:
        cli_overrides["auto_commit"] = True
    if args.auto_push:
        cli_overrides["auto_push"] = True

    # 统一执行决策：检测 & 前缀、构建回退提示
    decision_inputs = compute_decision_inputs(args, original_prompt=args.goal)
    decision: ExecutionDecision = decision_inputs.build_decision()

    if decision.user_message:
        if decision.message_level == "warning":
            logger.warning(decision.user_message)
        else:
            logger.info(decision.user_message)

    if decision.prefix_routed:
        logger.info("检测到 & 前缀请求 Cloud，但 run_mp 仅支持 CLI/MP，已忽略 Cloud 路由")

    if decision.sanitized_prompt and decision.sanitized_prompt != args.goal:
        logger.info("已移除 & 前缀，按本地 CLI 方式执行")
        args.goal = decision.sanitized_prompt

    # 使用 resolve_orchestrator_settings 统一解析
    resolved = resolve_orchestrator_settings(
        overrides=cli_overrides,
        prefix_routed=decision.prefix_routed,
    )

    # ========== execution_mode 检测与处理 ==========
    # MP 编排器（run_mp.py）仅支持 execution_mode=cli
    # 如果配置为 cloud/auto，打印警告并强制回退到 CLI
    config_execution_mode = resolved.get("execution_mode", "cli")
    final_execution_mode = config_execution_mode  # 默认使用配置值

    if config_execution_mode in ("cloud", "auto"):
        # 日志级别决策：使用 INFO 级别避免"每次都警告"的问题
        # 参见 core/execution_policy.py 中的"警告与日志策略决策"
        # 当前无法区分显式配置（--execution-mode auto）和隐式默认（config.yaml 默认值）
        # 因此统一使用 INFO 级别，用户可通过 --verbose 查看
        logger.info(f"ℹ MP 编排器不支持 execution_mode={config_execution_mode}，回退到 CLI 模式")
        logger.info(
            "提示: 若需使用 Cloud/Auto 模式，请改用:\n"
            "  python run.py --mode iterate --orchestrator basic --execution-mode auto\n"
            '  python scripts/run_iterate.py --execution-mode cloud "任务描述"'
        )
        final_execution_mode = "cli"
        # 更新 resolved 以反映实际使用的模式
        resolved["execution_mode"] = "cli"

    # 记录最终采用的 execution_mode
    logger.info(f"执行模式: {final_execution_mode} (配置值: {config_execution_mode})")

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

    config = MultiProcessOrchestratorConfig(
        working_directory=args.directory,
        max_iterations=resolved["max_iterations"],
        worker_count=resolved["workers"],
        enable_sub_planners=resolved["enable_sub_planners"],
        strict_review=resolved["strict_review"],
        planning_timeout=resolved["planner_timeout"],
        execution_timeout=resolved["worker_timeout"],
        review_timeout=resolved["reviewer_timeout"],
        # 模型配置
        planner_model=resolved["planner_model"],
        worker_model=resolved["worker_model"],
        reviewer_model=resolved["reviewer_model"],
        # 执行模式（MP 编排器强制 CLI，此处记录配置值）
        execution_mode=final_execution_mode,
        stream_events_enabled=stream_config["enabled"],
        stream_log_console=stream_config["console"],
        stream_log_detail_dir=stream_config["detail_dir"],
        stream_log_raw_dir=stream_config["raw_dir"],
    )

    logger.info("模型配置:")
    logger.info(f"  - 规划者: {config.planner_model}")
    logger.info(f"  - 执行者: {config.worker_model}")
    logger.info(f"  - 评审者: {config.reviewer_model}")

    if resolved["auto_commit"]:
        logger.info("自动提交: 启用")
        if resolved["auto_push"]:
            logger.info("自动推送: 启用")

    orchestrator = MultiProcessOrchestrator(config)
    result = await orchestrator.run(args.goal)

    # 将 execution_mode 信息写入结果
    result["execution_mode"] = final_execution_mode
    result["execution_mode_config"] = config_execution_mode  # 原始配置值

    # 完成后提交阶段
    if resolved["auto_commit"]:
        commits_result = await run_commit_phase(
            working_directory=args.directory,
            auto_push=resolved["auto_push"],
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

    # 显示执行模式信息
    execution_mode = result.get("execution_mode", "cli")
    execution_mode_config = result.get("execution_mode_config", execution_mode)
    if execution_mode != execution_mode_config:
        print(f"执行模式: {execution_mode} (配置值: {execution_mode_config}，已回退)")
    else:
        print(f"执行模式: {execution_mode}")

    print("\n任务统计:")
    print(f"  - 创建: {result.get('total_tasks_created', 0)}")
    print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
    print(f"  - 失败: {result.get('total_tasks_failed', 0)}")

    # 进程信息
    process_info = result.get("process_info", {})
    if process_info:
        print("\n进程信息:")
        for _agent_id, info in process_info.items():
            status = "存活" if info.get("alive") else "已停止"
            print(f"  - {info.get('type', 'unknown')}: PID {info.get('pid', 'N/A')} ({status})")

    # 迭代详情
    if result.get("iterations"):
        print("\n迭代详情:")
        for it in result["iterations"]:
            status_emoji = "✓" if it.get("review_passed") else "→"
            print(f"  {status_emoji} 迭代 {it['id']}: {it['tasks_completed']}/{it['tasks_created']} 任务完成")

    # 提交信息
    commits_info = result.get("commits", {})
    if commits_info:
        print("\n提交信息:")
        total_commits = commits_info.get("total_commits", 0)
        if total_commits > 0:
            print(f"  提交数量: {total_commits}")

            # 显示提交哈希
            commit_hashes = commits_info.get("commit_hashes", [])
            if commit_hashes:
                for i, hash_val in enumerate(commit_hashes[-3:], 1):  # 显示最近3个
                    short_hash = hash_val[:8] if len(hash_val) > 8 else hash_val
                    print(f"  提交 {i}: {short_hash}")

            # 显示提交信息摘要
            commit_messages = commits_info.get("commit_messages", [])
            if commit_messages:
                print("  提交信息摘要:")
                for msg in commit_messages[-3:]:  # 显示最近3条
                    # 截取第一行作为摘要
                    summary = msg.split("\n")[0][:60]
                    print(f"    - {summary}")

            # 显示变更文件数量
            files_changed = commits_info.get("files_changed", [])
            if files_changed:
                print(f"  变更文件: {len(files_changed)} 个")

            # 显示推送状态
            pushed_commits = commits_info.get("pushed_commits", 0)
            if pushed_commits > 0:
                print(f"  推送状态: 已推送 {pushed_commits} 个提交到远程仓库")
            elif commits_info.get("push_error"):
                print(f"  推送状态: 推送失败 - {commits_info['push_error']}")
        else:
            if commits_info.get("error"):
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
