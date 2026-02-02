#!/usr/bin/env python3
"""CursorX 统一入口脚本

提供统一的命令行入口，支持多种运行模式和自然语言任务描述。

================================================================================
运行模式
================================================================================

  - basic:     基本协程模式（规划-执行-审核）
  - mp:        多进程模式（并行执行）
  - knowledge: 知识库增强模式（自动搜索相关文档）
  - iterate:   自我迭代模式（检查更新、更新知识库、执行任务）
  - auto:      自动分析模式（使用 Agent 分析任务并选择最佳模式）

================================================================================
副作用控制策略矩阵 (Side Effect Control)
================================================================================

详细策略矩阵参见: core/execution_policy.py

+---------------+----------------+----------------+----------------+-----------------+
| 策略          | 网络请求       | 文件写入       | Git 操作       | 适用场景        |
+===============+================+================+================+=================+
| normal        | 允许           | 允许           | 允许           | 正常执行        |
| (默认)        |                |                |                |                 |
+---------------+----------------+----------------+----------------+-----------------+
| skip-online   | 禁止在线检查   | 允许           | 允许           | 离线环境        |
| (--skip-online)| 本地缓存优先  |                |                | CI/CD 加速      |
+---------------+----------------+----------------+----------------+-----------------+
| dry-run       | 允许           | 禁止           | 禁止           | 预览/调试       |
| (--dry-run)   | (用于分析)     | (仅日志输出)   | (仅日志输出)   | 安全检查        |
+---------------+----------------+----------------+----------------+-----------------+
| minimal       | 禁止           | 禁止           | 禁止           | 最小副作用      |
| (两者组合)    |                |                |                | 纯分析场景      |
+---------------+----------------+----------------+----------------+-----------------+

副作用控制参数:
  --skip-online   跳过在线文档检查，使用本地缓存
  --dry-run       仅分析不执行，不修改任何文件
  --force-update  强制更新知识库（忽略缓存）

================================================================================
用法示例
================================================================================

  # 显式指定模式
  python run.py --mode basic "实现 REST API"
  python run.py --mode mp "重构代码" --workers 5
  python run.py --mode iterate "更新 CLI 支持"

  # 自动模式（Agent 分析任务）
  python run.py "启动自我迭代，跳过在线更新"
  python run.py "使用多进程模式重构 src 目录"

  # 无限迭代
  python run.py "实现功能" --max-iterations MAX

  # 副作用控制示例
  python run.py --skip-online "离线模式任务"        # 跳过在线检查
  python run.py --dry-run "预览任务分解"            # 仅分析不执行
  python run.py --skip-online --dry-run "纯本地分析" # 最小副作用
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from cursor.client import CursorAgentConfig

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _configure_windows_stdio() -> None:
    """Windows 控制台默认编码可能不支持中文，回退到安全输出。"""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="backslashreplace")
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass


_configure_windows_stdio()

from loguru import logger

from core.cloud_utils import (
    CLOUD_PREFIX,
    is_cloud_request,
    strip_cloud_prefix,
)
from core.config import (
    DEFAULT_CLOUD_AUTH_TIMEOUT,
    DEFAULT_CLOUD_TIMEOUT,
    DEFAULT_ENABLE_SUB_PLANNERS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_STRICT_REVIEW,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WORKER_POOL_SIZE,
    MAX_CONSOLE_PREVIEW_CHARS,
    MAX_GOAL_SUMMARY_CHARS,
    MAX_KNOWLEDGE_DOC_PREVIEW_CHARS,
    TRUNCATION_HINT,
    TRUNCATION_HINT_OUTPUT,
    UnifiedOptions,
    build_cloud_client_config,
    build_cursor_agent_config,
    build_unified_overrides,
    get_config,
    parse_max_iterations,
    parse_max_iterations_for_argparse,
    print_debug_config,
    resolve_stream_log_config,
)
from core.execution_policy import (
    CloudFailureInfo,
    CloudFailureKind,
    ExecutionDecision,
    build_cooldown_info,
    build_execution_decision,
    classify_cloud_failure,
    compute_decision_inputs,
    compute_message_dedup_key,
    compute_side_effects,
    validate_requested_mode_invariant,
)
from core.output_contract import (
    CooldownInfoFields,
    ResultFields,
    build_cloud_error_result,
    build_cloud_success_result,
    prepare_cooldown_message,
)
from core.project_workspace import inspect_project_state
from cursor.cloud_client import CloudAuthConfig, CloudClientFactory
from cursor.executor import ExecutionMode

# 显式导出核心解析函数，供测试与外部调用使用
__all__ = [
    "parse_max_iterations",
    "parse_max_iterations_for_argparse",
    "CLOUD_PREFIX",
    "is_cloud_request",
    "strip_cloud_prefix",
]

# ============================================================
# 运行模式定义
# ============================================================


class RunMode(str, Enum):
    """运行模式"""

    BASIC = "basic"  # 基本协程模式
    MP = "mp"  # 多进程模式
    KNOWLEDGE = "knowledge"  # 知识库增强模式
    ITERATE = "iterate"  # 自我迭代模式
    AUTO = "auto"  # 自动分析模式
    PLAN = "plan"  # 仅规划模式（不执行）
    ASK = "ask"  # 问答模式（直接对话）
    CLOUD = "cloud"  # Cloud 模式（云端执行）


# 模式别名映射
MODE_ALIASES = {
    "default": RunMode.BASIC,
    "basic": RunMode.BASIC,
    "agent": RunMode.BASIC,
    "simple": RunMode.BASIC,
    "mp": RunMode.MP,
    "multiprocess": RunMode.MP,
    "parallel": RunMode.MP,
    "knowledge": RunMode.KNOWLEDGE,
    "kb": RunMode.KNOWLEDGE,
    "docs": RunMode.KNOWLEDGE,
    "iterate": RunMode.ITERATE,
    "self-iterate": RunMode.ITERATE,
    "self": RunMode.ITERATE,
    "update": RunMode.ITERATE,
    "auto": RunMode.AUTO,
    "smart": RunMode.AUTO,
    "plan": RunMode.PLAN,
    "planning": RunMode.PLAN,
    "analyze": RunMode.PLAN,
    "ask": RunMode.ASK,
    "chat": RunMode.ASK,
    "question": RunMode.ASK,
    "q": RunMode.ASK,
    "cloud": RunMode.CLOUD,
    "cloud-agent": RunMode.CLOUD,
    "background": RunMode.CLOUD,
    "bg": RunMode.CLOUD,
}


@dataclass
class TaskAnalysis:
    """任务分析结果"""

    mode: RunMode = RunMode.BASIC
    goal: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


# ============================================================
# 颜色输出
# ============================================================


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"
    BOLD = "\033[1m"
    NC = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.NC} {text}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.NC} {text}")


# ============================================================
# 参数解析
# ============================================================

# 使用统一的 parse_max_iterations 函数（从 core.config 导入）
# parse_max_iterations_for_argparse 用于 argparse 类型转换


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    默认值优先级:
    1. 自然语言显式指定（最高）
    2. CLI 显式参数
    3. config.yaml 配置文件
    4. 代码默认值（最低）

    Tri-state 参数设计:
    - 部分参数使用 default=None 实现三态逻辑
    - None 表示"未显式指定"，允许后续按优先级合并
    - 在 Runner._merge_options 中实现最终值解析
    """
    # 加载配置，获取用于帮助信息显示的默认值
    config = get_config()

    # 从配置获取值（仅用于帮助信息显示，不作为 argparse 默认值）
    cfg_worker_pool_size = config.system.worker_pool_size
    cfg_max_iterations = config.system.max_iterations
    cfg_planner_model = config.models.planner
    cfg_worker_model = config.models.worker
    cfg_reviewer_model = config.models.reviewer
    cfg_planner_timeout = config.planner.timeout
    cfg_worker_timeout = config.worker.task_timeout
    cfg_reviewer_timeout = config.reviewer.timeout
    cfg_cloud_timeout = config.cloud_agent.timeout
    cfg_cloud_auth_timeout = config.cloud_agent.auth_timeout
    cfg_agent_cli_timeout = config.agent_cli.timeout
    cfg_execution_mode = config.cloud_agent.execution_mode
    cfg_enable_sub_planners = config.system.enable_sub_planners
    cfg_strict_review = config.system.strict_review

    parser = argparse.ArgumentParser(
        description="CursorX 统一入口 - 多 Agent 协作系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式：
  basic      基本协程模式（默认）
  mp         多进程模式（并行执行）
  knowledge  知识库增强模式
  iterate    自我迭代模式
  auto       自动分析模式（Agent 分析任务选择最佳模式）

示例：
  # 显式指定模式
  python run.py --mode basic "实现 REST API"
  python run.py --mode mp "重构代码" --workers 5
  python run.py --mode iterate --skip-online "优化 CLI"

  # 自动模式（自然语言描述任务）
  python run.py "启动自我迭代，跳过在线更新，优化代码"
  python run.py "使用多进程并行重构 src 目录下的代码"

  # 无限迭代直到完成
  python run.py "实现功能" --max-iterations MAX
        """,
    )

    # 任务描述（可选，支持自然语言）
    parser.add_argument(
        "task",
        nargs="?",
        default="",
        help="任务描述（支持自然语言，会自动分析并路由到合适的模式）",
    )

    # 运行模式
    parser.add_argument(
        "--mode",
        "-M",
        type=str,
        default="auto",
        choices=list(MODE_ALIASES.keys()),
        help="运行模式 (默认: auto，自动分析任务选择模式)",
    )

    # 通用参数
    # --directory 使用 tri-state 设计：
    # - default=None 表示用户未显式指定
    # - 运行时解析为 resolved_directory = args.directory or "."
    # - args._directory_user_set 标记是否为用户显式指定
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=None,
        help="工作目录 (默认: 当前目录)；显式指定时触发工程初始化/参考子工程发现",
    )

    # workers 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=f"Worker 数量 (默认: {cfg_worker_pool_size}，来自 config.yaml)",
    )

    # max_iterations 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-m",
        "--max-iterations",
        type=str,
        default=None,
        help=f"最大迭代次数 (默认: {cfg_max_iterations}，MAX/-1 表示无限迭代)",
    )

    # strict_review 使用互斥组实现 tri-state
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        action="store_true",
        dest="strict_review",
        default=None,
        help=f"启用严格评审模式 (config.yaml 默认: {cfg_strict_review})",
    )
    strict_group.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict_review",
        help="禁用严格评审模式",
    )

    # enable_sub_planners 使用互斥组实现 tri-state
    sub_planners_group = parser.add_mutually_exclusive_group()
    sub_planners_group.add_argument(
        "--enable-sub-planners",
        action="store_true",
        dest="enable_sub_planners",
        default=None,
        help=f"启用子规划者 (config.yaml 默认: {cfg_enable_sub_planners})",
    )
    sub_planners_group.add_argument(
        "--no-sub-planners",
        action="store_false",
        dest="enable_sub_planners",
        help="禁用子规划者",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="静默模式（仅 WARNING 及以上日志）",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别（优先于 --verbose/--quiet）",
    )

    parser.add_argument(
        "--heartbeat-debug",
        action="store_true",
        help="启用心跳调试日志",
    )

    # 卡死诊断参数
    stall_group = parser.add_argument_group("卡死诊断", "控制卡死检测和恢复的诊断输出")
    stall_diag_toggle = stall_group.add_mutually_exclusive_group()
    stall_diag_toggle.add_argument(
        "--stall-diagnostics",
        action="store_true",
        dest="stall_diagnostics_enabled",
        default=None,
        help="启用卡死诊断日志",
    )
    stall_diag_toggle.add_argument(
        "--no-stall-diagnostics",
        action="store_false",
        dest="stall_diagnostics_enabled",
        help="禁用卡死诊断日志",
    )
    stall_group.add_argument(
        "--stall-diagnostics-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="卡死诊断日志级别（默认: warning）",
    )
    stall_group.add_argument(
        "--stall-recovery-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="卡死检测/恢复间隔（秒，默认: 30.0）",
    )
    stall_group.add_argument(
        "--execution-health-check-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="执行阶段健康检查间隔（秒，默认: 30.0）",
    )
    stall_group.add_argument(
        "--health-warning-cooldown",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="健康检查告警冷却时间（秒，默认: 60.0）",
    )

    # 副作用控制参数（详见 core/execution_policy.py 策略矩阵）
    side_effect_group = parser.add_argument_group(
        "副作用控制", "控制网络请求、文件写入、Git 操作的策略（参见 core/execution_policy.py）"
    )

    side_effect_group.add_argument(
        "--skip-online",
        action="store_true",
        help="跳过在线文档检查，使用本地缓存（禁止网络请求，允许文件写入）",
    )

    side_effect_group.add_argument(
        "--dry-run",
        action="store_true",
        help="仅分析不执行，不修改任何文件（允许网络请求用于分析，禁止文件写入和Git操作）",
    )

    side_effect_group.add_argument(
        "--force-update",
        action="store_true",
        help="强制更新知识库，忽略缓存（与 --skip-online 互斥）",
    )

    side_effect_group.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "最小副作用模式（等价于 --skip-online --dry-run 组合）。"
            "副作用控制: network=禁止, disk_write=禁止, knowledge_write=禁止。"
            "适用场景: 纯分析/预览/CI 验证等无副作用需求的场景"
        ),
    )

    # 知识库模式专用参数
    parser.add_argument(
        "--use-knowledge",
        action="store_true",
        help="[knowledge] 使用知识库上下文",
    )

    parser.add_argument(
        "--search-knowledge",
        type=str,
        metavar="QUERY",
        help="[knowledge] 搜索知识库获取相关信息",
    )

    parser.add_argument(
        "--self-update",
        action="store_true",
        help="[knowledge] 自我更新模式",
    )

    # 知识库注入参数（MP 模式）
    knowledge_injection_group = parser.add_mutually_exclusive_group()
    knowledge_injection_group.add_argument(
        "--enable-knowledge-injection",
        action="store_true",
        dest="enable_knowledge_injection",
        default=True,
        help="[mp] 启用知识库注入（默认）",
    )
    knowledge_injection_group.add_argument(
        "--no-knowledge-injection",
        action="store_false",
        dest="enable_knowledge_injection",
        help="[mp] 禁用知识库注入",
    )

    parser.add_argument(
        "--knowledge-top-k",
        type=int,
        default=3,
        metavar="N",
        help="[mp] 知识库注入时使用的最大文档数 (默认: 3)",
    )

    parser.add_argument(
        "--knowledge-max-chars-per-doc",
        type=int,
        default=1200,
        metavar="N",
        help="[mp] 知识库注入时单文档最大字符数 (默认: 1200)",
    )

    parser.add_argument(
        "--knowledge-max-total-chars",
        type=int,
        default=3000,
        metavar="N",
        help="[mp] 知识库注入时总字符上限 (默认: 3000)",
    )

    # 多进程模式专用参数 - 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help=f"[mp] 规划者模型 (默认: {cfg_planner_model}，来自 config.yaml)",
    )

    parser.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help=f"[mp] 执行者模型 (默认: {cfg_worker_model}，来自 config.yaml)",
    )

    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=None,
        help=f"[mp] 评审者模型 (默认: {cfg_reviewer_model}，来自 config.yaml)",
    )

    # 流式日志配置（默认来源: logging.stream_json.*）
    # 使用 tri-state (None=未指定, 使用 config.yaml 默认值)
    stream_log_group = parser.add_mutually_exclusive_group()
    stream_log_group.add_argument(
        "--stream-log",
        action="store_true",
        dest="stream_log_enabled",
        default=None,
        help="启用 stream-json 流式日志",
    )
    stream_log_group.add_argument(
        "--no-stream-log",
        action="store_false",
        dest="stream_log_enabled",
        help="禁用 stream-json 流式日志",
    )

    parser.add_argument(
        "--stream-log-console",
        dest="stream_log_console",
        action="store_true",
        default=None,
        help="流式日志输出到控制台",
    )
    parser.add_argument(
        "--no-stream-log-console",
        dest="stream_log_console",
        action="store_false",
        help="关闭流式日志控制台输出",
    )

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

    # 禁用自动模式分析（直接使用指定模式）
    parser.add_argument(
        "--no-auto-analyze",
        action="store_true",
        help="禁用自动分析，直接使用指定模式",
    )

    # 自动提交相关参数（默认禁用，需显式开启）
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        dest="auto_commit",
        default=False,
        help="启用自动提交（需显式指定）",
    )

    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="启用自动推送（需配合 --auto-commit，自动 git push）",
    )

    parser.add_argument(
        "--commit-per-iteration",
        action="store_true",
        help="每次迭代都提交（默认仅在全部完成时提交）",
    )

    # 编排器选择参数（用于 iterate 模式）- tri-state
    orchestrator_group = parser.add_mutually_exclusive_group()
    orchestrator_group.add_argument(
        "--orchestrator",
        type=str,
        choices=["mp", "basic"],
        default=None,
        help="[iterate] 编排器类型: mp=多进程(默认), basic=协程模式",
    )
    orchestrator_group.add_argument(
        "--no-mp",
        action="store_true",
        dest="no_mp",
        default=None,
        help="[iterate] 禁用多进程编排器，使用基本协程编排器",
    )

    # 执行模式参数（全局）- tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default=None,
        help=(
            f"执行模式: cli=本地CLI, auto=自动选择(Cloud优先), cloud=强制Cloud "
            f"(默认: {cfg_execution_mode}，来自 config.yaml)。"
            "⚠ 重要: auto/cloud 模式强制使用 basic 编排器；"
            "如需使用 MP 编排器，请显式指定 --execution-mode cli --orchestrator mp"
        ),
    )

    # 角色级执行模式参数（可选，默认继承全局 execution-mode）
    parser.add_argument(
        "--planner-execution-mode",
        type=str,
        choices=["cli", "auto", "cloud", "plan"],
        default=None,
        help="规划者执行模式（默认继承 --execution-mode）",
    )

    parser.add_argument(
        "--worker-execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default=None,
        help="执行者执行模式（默认继承 --execution-mode）",
    )

    parser.add_argument(
        "--reviewer-execution-mode",
        type=str,
        choices=["cli", "auto", "cloud", "ask"],
        default=None,
        help="评审者执行模式（默认继承 --execution-mode）",
    )

    # Cloud 认证参数（与 scripts/run_iterate.py 对齐）
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

    # Cloud & 前缀自动检测参数 - tri-state (None=未指定，使用 config.yaml)
    # 使用互斥组实现成对开关：--auto-detect-cloud-prefix / --no-auto-detect-cloud-prefix
    auto_detect_prefix_group = parser.add_mutually_exclusive_group()
    auto_detect_prefix_group.add_argument(
        "--auto-detect-cloud-prefix",
        action="store_true",
        dest="auto_detect_cloud_prefix",
        default=None,
        help="启用 & 前缀自动检测（覆盖 config.yaml 中的 cloud_agent.auto_detect_cloud_prefix）",
    )
    auto_detect_prefix_group.add_argument(
        "--no-auto-detect-cloud-prefix",
        action="store_false",
        dest="auto_detect_cloud_prefix",
        help="禁用 & 前缀自动检测（& 前缀将被忽略，不会触发 Cloud 路由）",
    )

    # Cloud 后台模式参数
    # 三态逻辑：--cloud-background 启用后台，--no-cloud-background 禁用后台，不指定则使用默认行为
    cloud_background_group = parser.add_mutually_exclusive_group()
    cloud_background_group.add_argument(
        "--cloud-background",
        action="store_true",
        dest="cloud_background",
        default=None,
        help="Cloud 模式使用后台执行（提交后立即返回，不等待完成）",
    )
    cloud_background_group.add_argument(
        "--no-cloud-background",
        action="store_false",
        dest="cloud_background",
        help="Cloud 模式使用前台执行（等待任务完成）",
    )

    # 流式控制台渲染参数（默认关闭，避免噪声）
    stream_render_group = parser.add_argument_group("流式控制台渲染")

    stream_render_group.add_argument(
        "--stream-console-renderer",
        action="store_true",
        dest="stream_console_renderer",
        default=False,
        help="启用流式控制台渲染器（默认关闭）",
    )

    stream_render_group.add_argument(
        "--stream-advanced-renderer",
        action="store_true",
        dest="stream_advanced_renderer",
        default=False,
        help="使用高级终端渲染器（支持状态栏、打字效果等，默认关闭）",
    )

    stream_render_group.add_argument(
        "--stream-typing-effect",
        action="store_true",
        dest="stream_typing_effect",
        default=False,
        help="启用打字机效果（默认关闭）",
    )

    stream_render_group.add_argument(
        "--stream-typing-delay",
        type=float,
        default=0.02,
        metavar="SECONDS",
        help="打字延迟（秒，默认 0.02）",
    )

    stream_render_group.add_argument(
        "--stream-word-mode",
        action="store_true",
        dest="stream_word_mode",
        default=True,
        help="逐词输出模式（默认开启）",
    )

    stream_render_group.add_argument(
        "--no-stream-word-mode",
        action="store_false",
        dest="stream_word_mode",
        help="逐字符输出模式（禁用逐词模式）",
    )

    stream_render_group.add_argument(
        "--stream-color-enabled",
        action="store_true",
        dest="stream_color_enabled",
        default=True,
        help="启用颜色输出（默认开启）",
    )

    stream_render_group.add_argument(
        "--no-stream-color",
        action="store_false",
        dest="stream_color_enabled",
        help="禁用颜色输出",
    )

    stream_render_group.add_argument(
        "--stream-show-word-diff",
        action="store_true",
        dest="stream_show_word_diff",
        default=False,
        help="显示逐词差异（默认关闭）",
    )

    # 配置调试参数
    parser.add_argument(
        "--print-config",
        action="store_true",
        dest="print_config",
        default=False,
        help="打印配置调试信息并退出（输出格式稳定，便于脚本化 grep/CI 断言）",
    )

    args = parser.parse_args()

    # 标记用户是否显式设置了编排器选项
    # 检查 sys.argv 中是否包含 --orchestrator 或 --no-mp
    args._orchestrator_user_set = any(arg in sys.argv for arg in ["--orchestrator", "--no-mp"])

    # 标记用户是否显式设置了 --directory 参数
    # tri-state 设计：None 表示未指定，运行时解析为当前目录
    args._directory_user_set = args.directory is not None

    # 解析为实际目录（未指定时使用当前目录）
    if args.directory is None:
        args.directory = "."

    return args


# ============================================================
# 自然语言任务分析
# ============================================================


class TaskAnalyzer:
    """任务分析器

    使用规则匹配和 Agent 分析自然语言任务，确定最佳运行模式。
    支持 '&' 前缀自动识别为 Cloud 模式。
    """

    # 模式关键词映射
    MODE_KEYWORDS = {
        RunMode.CLOUD: [
            "云端",
            "cloud",
            "后台执行",
            "background",
            "推送云端",
            "云端任务",
            "cloud agent",
        ],
        RunMode.ITERATE: [
            "自我迭代",
            "self-iterate",
            "iterate",
            "迭代更新",
            "更新知识库",
            "检查更新",
            "changelog",
            "自我更新",
        ],
        RunMode.MP: [
            "多进程",
            "multiprocess",
            "并行",
            "parallel",
            "多worker",
            "多 worker",
            "并发",
        ],
        RunMode.KNOWLEDGE: [
            "知识库",
            "knowledge",
            "文档搜索",
            "搜索文档",
            "cursor 文档",
            "参考文档",
            "docs",
        ],
        RunMode.PLAN: [
            "规划",
            "plan",
            "planning",
            "分析任务",
            "任务分析",
            "制定计划",
            "仅规划",
            "只规划",
            "分解任务",
        ],
        RunMode.ASK: [
            "问答",
            "ask",
            "chat",
            "对话",
            "提问",
            "询问",
            "直接问",
            "回答",
            "解答",
            "question",
        ],
    }

    # 选项关键词映射（启用）
    OPTION_KEYWORDS = {
        "skip_online": ["跳过在线", "skip-online", "离线", "不检查更新", "跳过更新"],
        "dry_run": ["仅分析", "dry-run", "预览", "不执行"],
        "strict": ["严格", "strict", "严格审核", "严格评审"],
        "force_update": ["强制更新", "force-update", "强制刷新"],
        "self_update": ["自我更新", "self-update", "更新自身"],
        "use_knowledge": ["使用知识库", "use-knowledge", "启用知识库"],
        "auto_commit": ["启用提交", "开启提交", "自动提交", "enable-commit"],
        "auto_push": ["自动推送", "auto-push", "自动 push", "启用推送"],
        "stream_log": ["启用流式日志", "开启流式", "stream-log", "详细日志", "实时日志"],
        "minimal": ["最小副作用", "minimal", "纯分析", "最小模式", "无副作用"],
    }

    # 禁用选项关键词映射
    DISABLE_KEYWORDS = {
        "no_auto_commit": ["禁用提交", "关闭提交", "不提交", "跳过提交", "no-commit", "禁用自动提交"],
        "no_stream_log": ["禁用流式日志", "关闭流式", "no-stream", "简洁模式", "静默模式"],
        "no_auto_push": ["禁用推送", "不推送", "关闭推送", "no-push"],
    }

    # 非并行/协程模式关键词映射（用于 iterate 模式的编排器选择）
    NON_PARALLEL_KEYWORDS = [
        "禁用多进程",
        "禁用 mp",
        "不使用多进程",
        "单进程",
        "非并行",
        "顺序执行",
        "协程模式",
        "基本模式",
        "basic 编排器",
        "basic编排器",
        "no-mp",
        "no_mp",
        "--no-mp",
        "使用协程",
        "使用 basic",
    ]

    # 执行模式关键词映射（用于解析 execution_mode）
    EXECUTION_MODE_KEYWORDS = {
        "auto": [
            "execution_mode=auto",
            "execution-mode=auto",
            "execution-mode auto",
            "执行模式自动",
            "执行模式 auto",
            "自动执行模式",
            "auto 执行模式",
            "云端优先",
            "cloud 优先",
            "优先云端",
        ],
        "cloud": [
            "execution_mode=cloud",
            "execution-mode=cloud",
            "execution-mode cloud",
            "执行模式云端",
            "执行模式 cloud",
            "强制云端",
            "cloud 执行模式",
            "仅云端执行",
            "只用云端",
            "只使用云端",
        ],
        "cli": [
            "execution_mode=cli",
            "execution-mode=cli",
            "execution-mode cli",
            "执行模式本地",
            "执行模式 cli",
            "本地执行",
            "cli 执行模式",
            "仅本地执行",
            "只用本地",
            "只使用本地 cli",
        ],
    }

    # 无限迭代关键词
    UNLIMITED_KEYWORDS = ["无限", "持续", "一直", "不停", "循环", "max", "unlimited"]

    # Agent 允许输出的选项字段
    AGENT_ALLOWED_OPTIONS = {
        "workers",
        "max_iterations",
        "strict",
        "enable_sub_planners",
        "execution_mode",
        "orchestrator",
        "no_mp",
        "auto_commit",
        "auto_push",
        "commit_per_iteration",
        "dry_run",
        "skip_online",
        "force_update",
        "self_update",
        "use_knowledge",
        "stream_log",
        "cloud_background",
        "directory",
    }

    # 类级别标志：跟踪已输出的消息（避免重复打印）
    _shown_messages: set = set()

    def __init__(self, use_agent: bool = True):
        """初始化分析器

        Args:
            use_agent: 是否使用 Agent 进行高级分析
        """
        self.use_agent = use_agent

    @classmethod
    def reset_shown_messages(cls) -> None:
        """重置已显示消息标志（主要用于测试）"""
        cls._shown_messages = set()

    def analyze(self, task: str, args: argparse.Namespace) -> TaskAnalysis:
        """分析任务描述

        Args:
            task: 任务描述
            args: 命令行参数

        Returns:
            分析结果
        """
        if not task.strip():
            # 没有任务描述，使用默认模式
            return TaskAnalysis(
                mode=MODE_ALIASES.get(args.mode, RunMode.BASIC),
                goal="",
                reasoning="无任务描述，使用指定模式",
            )

        # 先用 Agent 进行参数 + 任务解析（规划者模型，默认只读 plan）
        agent_analysis = self._agent_analysis(task, args) if self.use_agent else None

        rule_analysis = self._rule_based_analysis(task, args)

        analysis = self._merge_analysis(
            original_task=task,
            agent_analysis=agent_analysis,
            rule_analysis=rule_analysis,
        )

        # 确保 goal 不为空
        if not analysis.goal:
            analysis.goal = task

        return analysis

    def _rule_based_analysis(self, task: str, args: argparse.Namespace | Any) -> TaskAnalysis:
        """基于规则的分析

        使用 core.execution_policy.build_execution_decision 统一处理：
        - & 前缀检测与 Cloud 路由决策
        - execution_mode 解析
        - orchestrator 选择
        - prompt 清理

        优先级规则:
        1. 显式 args.execution_mode 参数（最高优先级）
           - execution_mode=cli: 忽略 '&' 前缀，当作普通字符
           - execution_mode=cloud/auto: 强制使用对应模式
        2. '&' 前缀触发（当 execution_mode 未显式指定时）
           - cloud_enabled=True + has_api_key: 路由到 Cloud
           - 否则回退到 CLI
        3. config.yaml 的 execution_mode（当无 CLI 显式设置且无 & 前缀时）
           - execution_mode=auto/cloud: 强制 basic 编排器

        与 scripts/run_iterate.py 保持一致的 requested_mode 确定逻辑：
        - CLI 显式设置 > & 前缀触发 > config.yaml 默认值
        """
        task_lower = task.lower()
        analysis = TaskAnalysis(goal=task)
        reasoning_parts = []

        # === 使用 compute_decision_inputs 统一构建决策输入 ===
        # 此 helper 函数封装了以下逻辑：
        # - 获取 CLI 参数 (execution_mode, orchestrator, no_mp)
        # - 获取 Cloud 配置 (cloud_enabled, has_api_key)
        # - 检测 & 前缀（语法层面）
        # - 使用 resolve_requested_mode_for_decision 确定 requested_mode
        # - 使用 resolve_mode_source 确定 mode_source
        decision_inputs = compute_decision_inputs(args, original_prompt=task)

        # 使用集中化的不变式验证（避免入口分叉）
        config = get_config()
        validate_requested_mode_invariant(
            has_ampersand_prefix=decision_inputs.has_ampersand_prefix,
            cli_execution_mode=getattr(args, "execution_mode", None),
            requested_mode_for_decision=decision_inputs.requested_mode,
            config_execution_mode=config.cloud_agent.execution_mode,
            caller_name="run.py._build_execution_analysis",
        )

        # 使用 DecisionInputs.build_decision() 构建 ExecutionDecision
        decision: ExecutionDecision = decision_inputs.build_decision()

        # 根据决策结果设置 analysis
        # 统一使用 ExecutionDecision 中的字段：
        # - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
        # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud）
        # - triggered_by_prefix: prefix_routed 的兼容别名
        #
        # 【重要】保存 ExecutionDecision 对象到 analysis.options
        # 供 _merge_options 复用，避免重复决策导致重复提示
        analysis.options["_execution_decision"] = decision

        if decision.prefix_routed:
            # & 前缀成功触发 Cloud 模式
            analysis.mode = RunMode.CLOUD
            # 注意：不用 effective_mode 覆盖 execution_mode
            # execution_mode 应保持用户请求的值（或 None），让 config.yaml 生效
            # effective_mode 单独保存供 _get_execution_mode 使用
            analysis.options["cloud_background"] = True  # Cloud Relay 语义
            analysis.options["has_ampersand_prefix"] = decision.has_ampersand_prefix
            analysis.options["prefix_routed"] = True
            analysis.options["triggered_by_prefix"] = True  # 兼容别名
            analysis.options["requested_mode"] = decision.requested_mode
            analysis.options["effective_mode"] = decision.effective_mode
            # 不设置 execution_mode，让 _merge_options 使用 config.yaml 默认值
            analysis.options["orchestrator"] = decision.orchestrator
            analysis.goal = decision.sanitized_prompt
            reasoning_parts.append(decision.mode_reason)
            # Cloud 前缀触发时，不执行关键词检测
            # 【统一输出位置】决策消息输出（仅在有用户提示且未输出过时）
            # 根据 message_level 决定使用 print_warning 还是 print_info
            if decision.user_message:
                msg_key = compute_message_dedup_key(decision.user_message)
                if msg_key not in TaskAnalyzer._shown_messages:
                    TaskAnalyzer._shown_messages.add(msg_key)
                    if decision.message_level == "warning":
                        print_warning(decision.user_message)
                    else:
                        print_info(decision.user_message)
        else:
            # 非 Cloud 触发，使用清理后的 prompt
            analysis.goal = decision.sanitized_prompt
            analysis.options["has_ampersand_prefix"] = decision.has_ampersand_prefix
            analysis.options["prefix_routed"] = False
            analysis.options["triggered_by_prefix"] = False  # 兼容别名
            analysis.options["requested_mode"] = decision.requested_mode
            analysis.options["effective_mode"] = decision.effective_mode
            analysis.options["orchestrator"] = decision.orchestrator
            # 不在此处设置 execution_mode，让 _merge_options 处理
            # 这样可以保持 config.yaml 默认值的优先级

            # 【统一输出位置】决策消息在此处输出（仅在有用户提示且未输出过时）
            # 后续流程复用 _execution_decision 对象，不再重复输出
            # 根据 message_level 决定使用 print_warning 还是 print_info
            if decision.user_message:
                # 使用稳定哈希去重，避免多次分析时重复输出
                msg_key = compute_message_dedup_key(decision.user_message)
                if msg_key not in TaskAnalyzer._shown_messages:
                    TaskAnalyzer._shown_messages.add(msg_key)
                    if decision.message_level == "warning":
                        print_warning(decision.user_message)
                    else:
                        print_info(decision.user_message)
                reasoning_parts.append(decision.mode_reason)

            # 检测模式关键词（仅在非 Cloud 触发时执行）
            for mode, keywords in self.MODE_KEYWORDS.items():
                if any(kw.lower() in task_lower for kw in keywords):
                    analysis.mode = mode
                    matched = [kw for kw in keywords if kw.lower() in task_lower]
                    reasoning_parts.append(f"检测到 {mode.value} 模式关键词: {matched}")
                    break

        # 检测启用选项关键词
        for option, keywords in self.OPTION_KEYWORDS.items():
            if any(kw.lower() in task_lower for kw in keywords):
                analysis.options[option] = True
                reasoning_parts.append(f"检测到启用选项 {option}")

        # 检测禁用选项关键词
        for option, keywords in self.DISABLE_KEYWORDS.items():
            if any(kw.lower() in task_lower for kw in keywords):
                # 将 no_xxx 转换为 xxx = False
                real_option = option.replace("no_", "")
                analysis.options[real_option] = False
                reasoning_parts.append(f"检测到禁用选项 {real_option}")

        # 检测无限迭代
        if any(kw in task_lower for kw in self.UNLIMITED_KEYWORDS):
            analysis.options["max_iterations"] = -1
            reasoning_parts.append("检测到无限迭代关键词")

        # 检测非并行/协程模式关键词（影响 iterate 模式的编排器选择）
        if any(kw.lower() in task_lower for kw in self.NON_PARALLEL_KEYWORDS):
            analysis.options["no_mp"] = True
            analysis.options["orchestrator"] = "basic"
            reasoning_parts.append("检测到非并行关键词，使用协程编排器")

        # 检测执行模式关键词（execution_mode）
        for exec_mode, keywords in self.EXECUTION_MODE_KEYWORDS.items():
            if any(kw.lower() in task_lower for kw in keywords):
                analysis.options["execution_mode"] = exec_mode
                matched = [kw for kw in keywords if kw.lower() in task_lower]
                reasoning_parts.append(f"检测到执行模式 {exec_mode}: {matched}")
                break

        # 提取 Worker 数量
        worker_match = re.search(r"(\d+)\s*(个)?\s*(worker|进程|并行)", task_lower)
        if worker_match:
            analysis.options["workers"] = int(worker_match.group(1))
            reasoning_parts.append(f"提取 Worker 数量: {worker_match.group(1)}")

        analysis.reasoning = "; ".join(reasoning_parts) if reasoning_parts else "默认模式"
        return analysis

    def _merge_analysis(
        self,
        original_task: str,
        agent_analysis: Optional[TaskAnalysis],
        rule_analysis: TaskAnalysis,
    ) -> TaskAnalysis:
        """合并 Agent 与规则分析结果"""
        if not agent_analysis:
            return rule_analysis

        merged_goal = agent_analysis.goal or rule_analysis.goal or original_task
        merged_options = {**rule_analysis.options, **agent_analysis.options}
        merged_reasoning = agent_analysis.reasoning or rule_analysis.reasoning
        merged_mode = agent_analysis.mode

        # '&' 前缀触发时，强制 Cloud 模式和目标去前缀
        # 内部使用 prefix_routed 进行分支判断
        if rule_analysis.options.get("prefix_routed"):
            merged_mode = RunMode.CLOUD
            merged_goal = rule_analysis.goal
            merged_options.update(rule_analysis.options)
            merged_reasoning = self._merge_reasoning(
                merged_reasoning,
                "检测到 '&' 前缀，强制使用 Cloud 后台模式",
            )
            return TaskAnalysis(
                mode=merged_mode,
                goal=merged_goal,
                options=merged_options,
                reasoning=merged_reasoning,
            )

        # 如果 Agent 未明确模式，使用规则分析的模式
        if merged_mode == RunMode.BASIC and rule_analysis.mode != RunMode.BASIC:
            merged_mode = rule_analysis.mode
            merged_reasoning = self._merge_reasoning(merged_reasoning, rule_analysis.reasoning)

        return TaskAnalysis(
            mode=merged_mode,
            goal=merged_goal,
            options=merged_options,
            reasoning=merged_reasoning,
        )

    @staticmethod
    def _merge_reasoning(primary: str, extra: str) -> str:
        if not primary:
            return extra
        if not extra:
            return primary
        if extra in primary:
            return primary
        return f"{primary}; {extra}"

    def _agent_analysis(self, task: str, args: Optional[argparse.Namespace] = None) -> Optional[TaskAnalysis]:
        """使用 Agent 分析任务（只读模式）

        使用 --mode plan 确保只读执行，不会修改任何文件。
        """
        try:
            context_payload = self._build_agent_context(task, args)
            prompt = self._build_agent_prompt(task, context_payload)
            cmd = ["agent", "-p", prompt, "--output-format", "text", "--mode", "plan"]

            model = self._resolve_planner_model(args)
            if model:
                cmd.extend(["--model", model])

            # 调用 agent CLI（使用 ask 模式，确保只读执行）
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            if result.returncode != 0:
                return None

            # 解析 JSON 响应
            output = result.stdout.strip()

            # 空输出返回 None
            if not output:
                return None

            # 尝试提取 JSON
            json_match = re.search(r"\{[\s\S]*\}", output)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            mode_str = str(data.get("mode", "basic")).lower()
            mode = MODE_ALIASES.get(mode_str, RunMode.BASIC)

            options = self._normalize_agent_options(data.get("options", {}), args)

            return TaskAnalysis(
                mode=mode,
                goal=data.get("refined_goal", task),
                options=options,
                reasoning=data.get("reasoning", "Agent 分析"),
            )

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Agent 分析失败: {e}")
            return None

    def _normalize_agent_options(
        self,
        options: Any,
        args: Optional[argparse.Namespace],
    ) -> dict[str, Any]:
        """规范化 Agent 返回的 options

        基于 tri-state/argv 的显式检测映射：
        - 对于用户显式设置的字段（args.xxx is not None 或 _user_set 标记），
          直接丢弃 Agent 返回的同名 option
        - 仅允许 Agent 补全未显式设置的字段
        - 对于 store_true 类型（如 auto_commit），采用"只允许 Agent 从 False 推到 True，
          但不允许覆盖显式 True"的策略

        与 scripts/run_iterate.py 保持一致的"用户显式优先"策略。
        """
        if not isinstance(options, dict):
            return {}

        normalized: dict[str, Any] = {}
        for key, value in options.items():
            if key not in self.AGENT_ALLOWED_OPTIONS:
                continue
            normalized[key] = value

        # 类型归一化
        if "workers" in normalized:
            try:
                normalized["workers"] = int(normalized["workers"])
            except (TypeError, ValueError):
                normalized.pop("workers", None)

        if "max_iterations" in normalized:
            try:
                normalized["max_iterations"] = int(normalized["max_iterations"])
            except (TypeError, ValueError):
                normalized.pop("max_iterations", None)

        if args is None:
            return normalized

        # ============================================================
        # 基于 tri-state/argv 的显式检测映射
        # 用户显式设置的字段，直接丢弃 Agent 返回的同名 option
        # ============================================================

        # tri-state 字段：args.xxx is not None 表示用户显式设置
        tri_state_fields = [
            ("workers", "workers"),
            ("max_iterations", "max_iterations"),
            ("execution_mode", "execution_mode"),
            ("strict_review", "strict"),  # Agent 可能返回 "strict"
            ("strict_review", "strict_review"),
            ("enable_sub_planners", "enable_sub_planners"),
            ("cloud_timeout", "cloud_timeout"),
            ("cloud_auth_timeout", "cloud_auth_timeout"),
            ("planner_model", "planner_model"),
            ("worker_model", "worker_model"),
            ("reviewer_model", "reviewer_model"),
        ]

        for args_attr, option_key in tri_state_fields:
            if getattr(args, args_attr, None) is not None:
                normalized.pop(option_key, None)

        # _user_set 标记字段
        if getattr(args, "_directory_user_set", False):
            normalized.pop("directory", None)

        if getattr(args, "_orchestrator_user_set", False):
            normalized.pop("orchestrator", None)
            normalized.pop("no_mp", None)

        # ============================================================
        # store_true 类型字段处理策略
        # 只允许 Agent 从 False 推到 True，但不允许覆盖用户显式设置的 True
        # ============================================================

        # store_true 字段：用户显式设置为 True 时，丢弃 Agent 返回的值
        # （Agent 返回 False 不能覆盖用户的 True）
        store_true_fields = [
            "auto_commit",
            "auto_push",
            "commit_per_iteration",
            "dry_run",
            "skip_online",
            "force_update",
            "use_knowledge",
            "self_update",
        ]

        for field in store_true_fields:
            cli_value = getattr(args, field, False)
            if cli_value is True:
                # 用户显式设置为 True，丢弃 Agent 返回的值（不允许覆盖）
                normalized.pop(field, None)
            elif field in normalized and normalized[field] is True:
                # Agent 返回 True，用户未显式设置：保留 Agent 的值（允许从 False 推到 True）
                pass
            elif field in normalized and normalized[field] is False:
                # Agent 返回 False：保持用户默认值（False），移除 Agent 的显式 False
                # 避免 Agent 用 False 覆盖用户可能在自然语言中表达的意图
                normalized.pop(field, None)

        return normalized

    def _resolve_planner_model(self, args: Optional[argparse.Namespace]) -> Optional[str]:
        """解析规划者模型，用于 Agent 任务解析"""
        env_model = os.getenv("TASK_ANALYSIS_MODEL")
        if env_model:
            return env_model

        if args and getattr(args, "planner_model", None):
            return args.planner_model

        try:
            config = get_config()
            return config.models.planner
        except Exception:
            return DEFAULT_PLANNER_MODEL

    def _build_agent_context(
        self,
        task: str,
        args: Optional[argparse.Namespace],
    ) -> dict[str, Any]:
        """构建 Agent 解析上下文"""
        context: dict[str, Any] = {
            "task": task,
        }

        if args:
            context["cli_args"] = _build_cli_overrides_from_args(args)
            context["directory"] = args.directory
            context["directory_user_set"] = getattr(args, "_directory_user_set", False)
            context["explicit_mode"] = getattr(args, "mode", None)
            context["cloud_background"] = getattr(args, "cloud_background", None)

            if args.directory and context["directory_user_set"]:
                context["project_summary"] = self._collect_project_summary(args.directory)

        return context

    @staticmethod
    def _collect_project_summary(directory: str) -> dict[str, Any]:
        """检查目录并生成简要摘要"""
        path = Path(directory).resolve()
        summary: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
        }

        try:
            info = inspect_project_state(path)
            summary.update(
                {
                    "state": info.state.value,
                    "detected_language": info.detected_language,
                    "marker_files": info.marker_files[:5],
                    "source_files_count": info.source_files_count,
                }
            )
        except Exception as exc:
            summary["error"] = str(exc)

        # 采样顶层文件/目录
        if summary.get("exists") and summary.get("is_dir"):
            try:
                entries = []
                for item in path.iterdir():
                    entries.append(item.name + ("/" if item.is_dir() else ""))
                    if len(entries) >= 20:
                        break
                summary["top_level_entries"] = entries
            except Exception:
                summary["top_level_entries"] = []

        return summary

    def _build_agent_prompt(self, task: str, context_payload: dict[str, Any]) -> str:
        """构建任务解析提示词"""
        context_json = json.dumps(context_payload, ensure_ascii=False)
        return (
            "你是规划者Agent，请先解析参数和任务信息，并给出结构化结果。\n"
            "请根据已提供参数 + 任务描述 + 目录检查结果，推断应执行的模式与参数。\n"
            "如果用户已显式给出参数，必须优先保留，不要擅自覆盖。\n"
            "如果任务描述中包含参数形式（如 --workers 5 / --directory /path），需要解析出来。\n"
            "若目录为空或仅文档，结合任务描述推断可能的开发语言并反映在 refined_goal 中（如“使用 Python ...”）。\n"
            "仅输出 JSON，不要添加额外文字。\n\n"
            f"上下文: {context_json}\n\n"
            "可用模式: basic/mp/knowledge/iterate/cloud/plan/ask\n"
            "可用选项字段:\n"
            "- directory, workers, max_iterations, strict, enable_sub_planners\n"
            "- skip_online, force_update, self_update, use_knowledge\n"
            "- execution_mode, orchestrator, no_mp, cloud_background\n"
            "- auto_commit, auto_push, commit_per_iteration, dry_run\n"
            "- stream_log\n\n"
            "输出格式示例:\n"
            "{\n"
            '  "mode": "iterate",\n'
            '  "options": {"skip_online": true, "workers": 3},\n'
            '  "reasoning": "任务需要迭代并跳过在线检查",\n'
            '  "refined_goal": "在目标目录中执行自我迭代更新"\n'
            "}"
        )


# ============================================================
# 运行器
# ============================================================


class Runner:
    """运行器

    根据分析结果调用对应的入口脚本。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # max_iterations 使用 tri-state，None 表示未显式指定
        # 实际值将在 _merge_options 中按优先级解析

    async def run(self, analysis: TaskAnalysis) -> dict[str, Any]:
        """运行任务

        Args:
            analysis: 任务分析结果

        Returns:
            执行结果
        """
        mode = analysis.mode
        goal = analysis.goal or self.args.task
        options = analysis.options

        # 合并命令行参数和分析结果
        merged_options = self._merge_options(options)

        print_header(f"CursorX - {self._get_mode_name(mode)}")
        print_info(f"任务: {goal}")
        print_info(f"模式: {mode.value}")
        if analysis.reasoning:
            print_info(f"分析: {analysis.reasoning}")
        if merged_options:
            print_info(f"选项: {merged_options}")
        print()

        # 根据模式调用对应的运行函数
        if mode == RunMode.BASIC:
            return await self._run_basic(goal, merged_options)
        elif mode == RunMode.MP:
            return await self._run_mp(goal, merged_options)
        elif mode == RunMode.KNOWLEDGE:
            return await self._run_knowledge(goal, merged_options)
        elif mode == RunMode.ITERATE:
            return await self._run_iterate(goal, merged_options)
        elif mode == RunMode.PLAN:
            return await self._run_plan(goal, merged_options)
        elif mode == RunMode.ASK:
            return await self._run_ask(goal, merged_options)
        elif mode == RunMode.CLOUD:
            return await self._run_cloud(goal, merged_options)
        else:
            return await self._run_basic(goal, merged_options)

    def _merge_options(self, analysis_options: dict) -> dict:
        """合并命令行参数和分析结果

        优先级（从高到低，与 scripts/run_iterate.py 保持一致）：
        1. CLI 显式参数 (self.args.xxx 且非 None) — 用户显式优先
        2. 自然语言/Agent 分析结果 (analysis_options) — 仅补全未显式设置的字段
        3. config.yaml 配置
        4. 代码默认值

        tri-state 参数处理：
        - CLI 参数 default=None 表示"未显式指定"
        - 当 CLI 参数为 None 时，允许 analysis_options 或 config.yaml 补全
        - CLI 显式参数不可被 Agent 分析结果覆盖

        store_true 参数处理：
        - CLI 显式设置为 True 时，不可被 Agent 覆盖
        - Agent 只能将 False 推到 True（补全未设置的字段）

        使用 build_unified_overrides 统一解析核心编排器配置。
        决策快照字段（来自 build_execution_decision）会被保留。

        字段语义说明：
        - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
        - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud）
        - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）
        """
        # 【重要】优先使用 _execution_decision 对象（在 _rule_based_analysis 中构建）
        # 这确保整个 iterate 流程只构建一次决策，避免重复决策和重复提示
        execution_decision: Optional[ExecutionDecision] = analysis_options.get("_execution_decision")

        # 获取 prefix_routed 标志（优先从决策对象，回退到快照字段）
        # 内部统一使用 prefix_routed，triggered_by_prefix 仅作为输出兼容字段
        if execution_decision is not None:
            prefix_routed = execution_decision.prefix_routed
        else:
            prefix_routed = analysis_options.get("prefix_routed", False)

        # 使用统一的 build_unified_overrides 解析核心配置
        # 核心字段（workers/max_iterations/execution_mode/orchestrator/模型/超时/auto_commit 等）
        # 以 UnifiedOptions.resolved 为准
        #
        # 【关键修改】传入 execution_decision 对象以复用决策结果
        # 这避免了 build_unified_overrides 内部再次调用 resolve_orchestrator_settings
        # 时因缺少上下文而做出不一致的决策
        unified: UnifiedOptions = build_unified_overrides(
            args=self.args,
            nl_options=analysis_options,
            execution_decision=execution_decision,  # 复用决策对象
            prefix_routed=prefix_routed,
        )

        # 获取解析后的核心配置
        resolved = unified.resolved

        # 记录编排器用户设置元字段（用于后续 no_mp 计算）
        orchestrator_user_set = getattr(self.args, "_orchestrator_user_set", False)
        cli_no_mp = getattr(self.args, "no_mp", None)

        # minimal 模式判断（用于 skip_online 等非核心选项）
        minimal_mode = analysis_options.get("minimal") or getattr(self.args, "minimal", False)

        # 解析工作目录（允许分析结果在未显式指定时覆盖）
        directory = self.args.directory
        if analysis_options.get("directory") and not getattr(self.args, "_directory_user_set", False):
            directory = analysis_options["directory"]

        # 构建 options 字典
        options = {
            "directory": directory,
            "workers": resolved["workers"],
            "max_iterations": resolved["max_iterations"],
            "strict": resolved["strict_review"],
            "enable_sub_planners": resolved["enable_sub_planners"],
            "verbose": self.args.verbose,
            # 超时配置
            "planner_timeout": resolved["planner_timeout"],
            "worker_timeout": resolved["worker_timeout"],
            "reviewer_timeout": resolved["reviewer_timeout"],
            "cloud_timeout": resolved["cloud_timeout"],
            "cloud_auth_timeout": resolved["cloud_auth_timeout"],
            # 模型配置
            "planner_model": resolved["planner_model"],
            "worker_model": resolved["worker_model"],
            "reviewer_model": resolved["reviewer_model"],
            # 编排器配置
            "orchestrator": resolved["orchestrator"],
            "execution_mode": resolved["execution_mode"],
            # 提交控制
            "auto_commit": resolved["auto_commit"],
            "auto_push": resolved["auto_push"] and resolved["auto_commit"],
            "commit_per_iteration": resolved["commit_per_iteration"],
            # dry_run: minimal 模式强制 dry_run=True
            # 权威定义来源: core/execution_policy.py Side Effect Control Strategy Matrix
            "dry_run": resolved["dry_run"] or minimal_mode,
        }

        # 自我迭代选项
        # minimal 模式强制 skip_online=True + dry_run=True（已在上面处理）
        if analysis_options.get("skip_online") or self.args.skip_online or minimal_mode:
            options["skip_online"] = True
        if analysis_options.get("force_update") or self.args.force_update:
            options["force_update"] = True

        # minimal 模式标记（用于传递给子模块）
        options["minimal"] = minimal_mode

        # 知识库选项
        if analysis_options.get("use_knowledge") or self.args.use_knowledge:
            options["use_knowledge"] = True
        if analysis_options.get("self_update") or self.args.self_update:
            options["self_update"] = True
        if self.args.search_knowledge:
            options["search_knowledge"] = self.args.search_knowledge

        # 流式日志配置（使用 resolve_stream_log_config 统一解析）
        nl_stream_enabled = analysis_options.get("stream_log")
        cli_stream_enabled = getattr(self.args, "stream_log_enabled", None)
        cli_stream_console = getattr(self.args, "stream_log_console", None)
        cli_stream_detail_dir = getattr(self.args, "stream_log_detail_dir", None)
        cli_stream_raw_dir = getattr(self.args, "stream_log_raw_dir", None)

        # 自然语言覆盖 CLI（仅当自然语言显式指定时）
        if nl_stream_enabled is not None:
            cli_stream_enabled = nl_stream_enabled

        stream_config = resolve_stream_log_config(
            cli_enabled=cli_stream_enabled,
            cli_console=cli_stream_console,
            cli_detail_dir=cli_stream_detail_dir,
            cli_raw_dir=cli_stream_raw_dir,
        )

        options["stream_log"] = stream_config["enabled"]
        options["stream_log_console"] = stream_config["console"]
        options["stream_log_detail_dir"] = stream_config["detail_dir"]
        options["stream_log_raw_dir"] = stream_config["raw_dir"]

        # 编排器用户设置元字段（复用前面定义的变量）
        options["_orchestrator_user_set"] = orchestrator_user_set

        # no_mp 跟随 orchestrator 设置
        # 如果用户显式设置了 orchestrator，则使用 CLI 的 no_mp 值
        if orchestrator_user_set and cli_no_mp is not None:
            options["no_mp"] = cli_no_mp
        elif orchestrator_user_set:
            # 用户显式设置 orchestrator=mp 时，no_mp 应为 False
            options["no_mp"] = options["orchestrator"] == "basic"
        elif "no_mp" in analysis_options:
            options["no_mp"] = analysis_options["no_mp"]
        elif cli_no_mp is not None:
            options["no_mp"] = cli_no_mp
        else:
            options["no_mp"] = options["orchestrator"] == "basic"

        # 知识库注入选项（用于 MP 模式）
        options["enable_knowledge_injection"] = getattr(self.args, "enable_knowledge_injection", True)
        options["knowledge_top_k"] = getattr(self.args, "knowledge_top_k", 3)
        options["knowledge_max_chars_per_doc"] = getattr(self.args, "knowledge_max_chars_per_doc", 1200)
        options["knowledge_max_total_chars"] = getattr(self.args, "knowledge_max_total_chars", 3000)

        # Cloud 认证配置
        options["cloud_api_key"] = getattr(self.args, "cloud_api_key", None)

        # Cloud 后台模式配置
        cli_cloud_background = getattr(self.args, "cloud_background", None)
        if cli_cloud_background is not None:
            options["cloud_background"] = cli_cloud_background
        elif "cloud_background" in analysis_options:
            options["cloud_background"] = analysis_options["cloud_background"]
        else:
            options["cloud_background"] = False

        # 决策快照字段（与 ExecutionDecision 一致）
        # 【重要】优先使用 _execution_decision 对象（在 _rule_based_analysis 中构建）
        # 若缺失，使用 compute_decision_inputs 统一重建，避免混用 analysis_options 中可能语义漂移的字段
        #
        # 字段语义说明：
        # - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
        # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
        # - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）
        # - requested_mode: 用户请求的执行模式（来自 CLI 或 config.yaml）
        # - effective_mode: 经过决策后实际使用的执行模式
        #
        # 区分 execution_mode (requested) vs effective_mode (实际):
        # - options["execution_mode"]: 始终为 requested_mode（来自 resolved）
        # - options["effective_mode"]: 决策后的实际模式（可能与 requested 不同）
        if execution_decision is None:
            # 【关键修改】execution_decision 缺失时，使用 compute_decision_inputs 统一重建
            # 此 helper 封装了：
            # - 获取 CLI 参数和 Cloud 配置
            # - 从 analysis_options 提取原始 prompt（支持 "_original_goal" 或 "goal" 字段）
            # - 检测 & 前缀并处理虚拟 prompt 构造
            # - 使用 resolve_requested_mode_for_decision 确定 requested_mode
            # - 使用 resolve_mode_source 确定 mode_source
            decision_inputs = compute_decision_inputs(self.args, nl_options=analysis_options)
            execution_decision = decision_inputs.build_decision()

        # 使用 ExecutionDecision 对象的字段（无论是传入的还是重新构建的）
        # 内部分支统一使用 prefix_routed，triggered_by_prefix 仅作为输出兼容字段
        options["has_ampersand_prefix"] = execution_decision.has_ampersand_prefix
        options["prefix_routed"] = execution_decision.prefix_routed
        options["triggered_by_prefix"] = execution_decision.prefix_routed  # 兼容别名
        options["requested_mode"] = execution_decision.requested_mode
        options["effective_mode"] = execution_decision.effective_mode

        # 角色级执行模式
        options["planner_execution_mode"] = getattr(self.args, "planner_execution_mode", None)
        options["worker_execution_mode"] = getattr(self.args, "worker_execution_mode", None)
        options["reviewer_execution_mode"] = getattr(self.args, "reviewer_execution_mode", None)

        # 流式控制台渲染配置
        options["stream_console_renderer"] = getattr(self.args, "stream_console_renderer", False)
        options["stream_advanced_renderer"] = getattr(self.args, "stream_advanced_renderer", False)
        options["stream_typing_effect"] = getattr(self.args, "stream_typing_effect", False)
        options["stream_typing_delay"] = getattr(self.args, "stream_typing_delay", 0.02)
        options["stream_word_mode"] = getattr(self.args, "stream_word_mode", True)
        options["stream_color_enabled"] = getattr(self.args, "stream_color_enabled", True)
        options["stream_show_word_diff"] = getattr(self.args, "stream_show_word_diff", False)

        # 日志控制参数
        options["quiet"] = getattr(self.args, "quiet", False)
        options["log_level"] = getattr(self.args, "log_level", None)
        options["heartbeat_debug"] = getattr(self.args, "heartbeat_debug", False)

        # 卡死诊断参数
        options["stall_diagnostics_enabled"] = getattr(self.args, "stall_diagnostics_enabled", None)
        options["stall_diagnostics_level"] = getattr(self.args, "stall_diagnostics_level", None)
        options["stall_recovery_interval"] = getattr(self.args, "stall_recovery_interval", 30.0)
        options["execution_health_check_interval"] = getattr(self.args, "execution_health_check_interval", 30.0)
        options["health_warning_cooldown"] = getattr(self.args, "health_warning_cooldown", 60.0)

        return options

    def _build_cursor_config(self, options: dict) -> CursorAgentConfig:
        """从 options 字典构建 CursorAgentConfig

        使用统一的 build_cursor_agent_config 函数构建配置，
        确保配置优先级处理一致。

        Args:
            options: _merge_options 返回的选项字典

        Returns:
            构建好的 CursorAgentConfig 实例
        """
        from cursor.client import CursorAgentConfig

        # 将 options 转换为 build_cursor_agent_config 需要的 overrides
        overrides = {
            "stream_events_enabled": options.get("stream_log", False),
            "stream_log_console": options.get("stream_log_console", True),
            "stream_log_detail_dir": options.get("stream_log_detail_dir"),
            "stream_log_raw_dir": options.get("stream_log_raw_dir"),
        }

        # 调用统一的构建函数
        config_dict = build_cursor_agent_config(
            working_directory=options["directory"],
            overrides=overrides,
        )

        return CursorAgentConfig(**config_dict)

    def _get_mode_name(self, mode: RunMode) -> str:
        """获取模式显示名称"""
        names = {
            RunMode.BASIC: "基本模式",
            RunMode.MP: "多进程模式",
            RunMode.KNOWLEDGE: "知识库增强模式",
            RunMode.ITERATE: "自我迭代模式",
            RunMode.AUTO: "自动模式",
            RunMode.PLAN: "规划模式",
            RunMode.ASK: "问答模式",
            RunMode.CLOUD: "Cloud 模式",
        }
        return names.get(mode, mode.value)

    def _get_execution_mode(self, options: dict) -> ExecutionMode:
        """获取执行模式

        统一读取顺序：
        1. 优先使用决策快照中的 effective_mode（由 build_execution_decision 生成）
        2. 如果没有决策快照，则使用 build_execution_decision 重新计算

        字段语义说明：
        - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
        - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）

        Args:
            options: 合并后的选项字典

        Returns:
            ExecutionMode 枚举值
        """
        # 优先使用决策快照中的 effective_mode
        effective_mode_str = options.get("effective_mode")

        if not effective_mode_str:
            # 没有决策快照，使用 build_execution_decision 重新计算
            # requested_mode: 用户请求的执行模式（来自 CLI 或 config.yaml）
            requested_mode = options.get("execution_mode", "cli")
            # 内部使用 prefix_routed 进行分支（不读取 triggered_by_prefix）
            prefix_routed = options.get("prefix_routed", False)

            # 获取 cloud_enabled 配置
            config = get_config()
            cloud_enabled = config.cloud_agent.enabled

            # 检查 API Key
            api_key = CloudClientFactory.resolve_api_key()
            has_api_key = bool(api_key)

            # 构建执行决策
            # 注意：这里 prompt=None 且 auto_detect_cloud_prefix=False
            # 因为 options 中已包含处理过的决策结果
            # mode_source: 从 _execution_decision 对象中获取，若无则使用 config
            execution_decision_obj = options.get("_execution_decision")
            if execution_decision_obj and hasattr(execution_decision_obj, "mode_source"):
                decision_mode_source = execution_decision_obj.mode_source
            else:
                decision_mode_source = "config"
            decision = build_execution_decision(
                prompt=None,
                requested_mode=requested_mode,
                cloud_enabled=cloud_enabled,
                has_api_key=has_api_key,
                auto_detect_cloud_prefix=False,
                mode_source=decision_mode_source,
            )
            effective_mode_str = decision.effective_mode

        # 映射到 ExecutionMode 枚举
        mode_map = {
            "cli": ExecutionMode.CLI,
            "auto": ExecutionMode.AUTO,
            "cloud": ExecutionMode.CLOUD,
            "plan": ExecutionMode.PLAN,
            "ask": ExecutionMode.ASK,
        }
        return mode_map.get(effective_mode_str, ExecutionMode.CLI)

    def _parse_execution_mode(self, mode_str: Optional[str]) -> Optional[ExecutionMode]:
        """解析执行模式字符串为 ExecutionMode 枚举

        Args:
            mode_str: 执行模式字符串（cli/auto/cloud/plan/ask）或 None

        Returns:
            ExecutionMode 枚举值或 None
        """
        if mode_str is None:
            return None

        mode_map = {
            "cli": ExecutionMode.CLI,
            "auto": ExecutionMode.AUTO,
            "cloud": ExecutionMode.CLOUD,
            "plan": ExecutionMode.PLAN,
            "ask": ExecutionMode.ASK,
        }
        return mode_map.get(mode_str.lower())

    def _get_cloud_auth_config(self, options: dict) -> Optional[CloudAuthConfig]:
        """获取 Cloud 认证配置

        配置 API Key 的三种方式:
          1. export CURSOR_API_KEY=your_key
          2. --cloud-api-key your_key
          3. agent login

        优先级: --cloud-api-key > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml

        Args:
            options: 合并后的选项字典

        Returns:
            CloudAuthConfig 或 None（未配置 key 时）
        """
        # 使用 CloudClientFactory.resolve_api_key 统一解析 API Key
        # 优先级: CLI 显式参数 > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
        cli_api_key = options.get("cloud_api_key")
        api_key = CloudClientFactory.resolve_api_key(explicit_api_key=cli_api_key)

        if not api_key:
            return None

        # 从 config.yaml 获取默认配置
        cloud_config = build_cloud_client_config()

        # 解析 auth_timeout，优先级: CLI > config.yaml > DEFAULT_*
        cli_auth_timeout = options.get("cloud_auth_timeout")
        auth_timeout = (
            cli_auth_timeout
            if cli_auth_timeout is not None
            else cloud_config.get("auth_timeout", DEFAULT_CLOUD_AUTH_TIMEOUT)
        )

        # 解析 base_url，优先级: config.yaml > DEFAULT_*（暂无 CLI 参数）
        base_url = cloud_config.get("base_url", "https://api.cursor.com")

        # 解析 max_retries，优先级: config.yaml > DEFAULT_*（暂无 CLI 参数）
        max_retries = cloud_config.get("max_retries", 3)

        return CloudAuthConfig(
            api_key=api_key,
            auth_timeout=auth_timeout,
            api_base_url=base_url,
            max_retries=max_retries,
        )

    async def _search_knowledge_docs(
        self,
        query: str,
        options: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """搜索知识库文档

        Args:
            query: 搜索查询
            options: 运行选项，包含 minimal/dry_run/skip_online 等控制参数

        Returns:
            知识库文档列表，每个文档包含 title, url, content 字段
            只读模式下若 index 不存在则返回空列表并打印提示
        """
        try:
            from knowledge import KnowledgeStorage

            options = options or {}
            minimal = options.get("minimal", False)
            dry_run = options.get("dry_run", False)

            # minimal/dry_run 模式下使用只读 storage，避免创建目录
            use_read_only = minimal or dry_run
            if use_read_only:
                storage = KnowledgeStorage.create_read_only()
            else:
                storage = KnowledgeStorage()

            await storage.initialize()

            # 只读模式下检查索引是否存在
            if use_read_only and not storage.index_path.exists():
                print_info("知识库索引不存在，跳过搜索（只读模式）")
                return []

            results = await storage.search(query, limit=5)

            knowledge_docs: list[dict[str, Any]] = []
            for result in results:
                doc = await storage.load_document(result.doc_id)
                if doc:
                    knowledge_docs.append(
                        {
                            "title": doc.title,
                            "url": doc.url,
                            "content": doc.content[:MAX_CONSOLE_PREVIEW_CHARS],
                        }
                    )
            return knowledge_docs
        except Exception as e:
            logger.warning(f"知识库搜索失败: {e}")
            return []

    def _build_enhanced_goal(self, goal: str, knowledge_docs: list[dict[str, Any]]) -> str:
        """构建增强后的目标（将知识库文档拼接到 goal）

        Args:
            goal: 原始目标
            knowledge_docs: 知识库文档列表

        Returns:
            增强后的目标字符串
        """
        if not knowledge_docs:
            return goal

        context_text = "\n\n## 参考文档（来自知识库）\n\n"
        for i, doc in enumerate(knowledge_docs, 1):
            context_text += f"### {i}. {doc['title']}\n"
            # 截断过长内容（使用统一的知识库文档预览限制）
            content = doc["content"][:MAX_KNOWLEDGE_DOC_PREVIEW_CHARS]
            if len(doc["content"]) > MAX_KNOWLEDGE_DOC_PREVIEW_CHARS:
                content += TRUNCATION_HINT
            context_text += f"```\n{content}\n```\n\n"

        return f"{goal}\n{context_text}"

    async def _run_basic(self, goal: str, options: dict) -> dict:
        """运行基本模式"""
        from coordinator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig

        # 创建 CursorAgentConfig，注入 resolved stream 配置
        cursor_config = CursorAgentConfig(
            working_directory=options["directory"],
            stream_events_enabled=options.get("stream_log", False),
            stream_log_console=options.get("stream_log_console", True),
            stream_log_detail_dir=options.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=options.get("stream_log_raw_dir", "logs/stream_json/raw/"),
        )

        # 获取执行模式和 Cloud 认证配置
        execution_mode = self._get_execution_mode(options)
        cloud_auth_config = self._get_cloud_auth_config(options)

        # 解析角色级执行模式（可选，默认继承全局 execution_mode）
        planner_exec_mode = self._parse_execution_mode(options.get("planner_execution_mode"))
        worker_exec_mode = self._parse_execution_mode(options.get("worker_execution_mode"))
        reviewer_exec_mode = self._parse_execution_mode(options.get("reviewer_execution_mode"))

        config = OrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_pool_size=options["workers"],
            # 系统配置（tri-state 已在 _merge_options 中解析）
            enable_sub_planners=options.get("enable_sub_planners", DEFAULT_ENABLE_SUB_PLANNERS),
            strict_review=options.get("strict", DEFAULT_STRICT_REVIEW),
            cursor_config=cursor_config,
            # 流式日志配置（使用 resolved stream config）
            stream_events_enabled=options.get("stream_log", False),
            stream_log_console=options.get("stream_log_console", True),
            stream_log_detail_dir=options.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=options.get("stream_log_raw_dir", "logs/stream_json/raw/"),
            enable_auto_commit=options.get("auto_commit", False),
            auto_push=options.get("auto_push", False),
            commit_per_iteration=options.get("commit_per_iteration", False),
            # 模型配置
            planner_model=options.get("planner_model", DEFAULT_PLANNER_MODEL),
            worker_model=options.get("worker_model", DEFAULT_WORKER_MODEL),
            reviewer_model=options.get("reviewer_model", DEFAULT_REVIEWER_MODEL),
            # 执行模式和 Cloud 认证配置
            execution_mode=execution_mode,
            cloud_auth_config=cloud_auth_config,
            # 角色级执行模式（可选）
            planner_execution_mode=planner_exec_mode,
            worker_execution_mode=worker_exec_mode,
            reviewer_execution_mode=reviewer_exec_mode,
            # 流式控制台渲染配置
            stream_console_renderer=options.get("stream_console_renderer", False),
            stream_advanced_renderer=options.get("stream_advanced_renderer", False),
            stream_typing_effect=options.get("stream_typing_effect", False),
            stream_typing_delay=options.get("stream_typing_delay", 0.02),
            stream_word_mode=options.get("stream_word_mode", True),
            stream_color_enabled=options.get("stream_color_enabled", True),
            stream_show_word_diff=options.get("stream_show_word_diff", False),
        )

        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        orchestrator = Orchestrator(config)
        return await orchestrator.run(goal)

    async def _run_mp(self, goal: str, options: dict) -> dict:
        """运行多进程模式

        支持知识库增强：
        - 方案 B: 当指定 --search-knowledge 时，搜索知识库并将文档拼接到 goal
        - 方案 C: 通过 enable_knowledge_search 配置启用 MP 任务级知识库注入

        注意: MP 编排器不支持 Cloud/Auto 执行模式，如果用户指定了非 CLI 执行模式，
        会发出警告但继续使用 CLI 模式执行（MP 编排器内部始终使用本地 CLI）。
        """
        from coordinator import MultiProcessOrchestrator, MultiProcessOrchestratorConfig

        # MP 编排器不支持 Cloud/Auto 执行模式，检查并发出警告
        execution_mode_str = options.get("execution_mode", "cli")
        if execution_mode_str != "cli":
            print_warning(
                f"MP 编排器不支持执行模式 '{execution_mode_str}'，"
                "将使用 CLI 模式执行。如需使用 Cloud/Auto 模式，请使用 --orchestrator basic"
            )

        # 方案 B: 搜索知识库并增强 goal
        enhanced_goal = goal
        search_query = options.get("search_knowledge")
        if search_query:
            knowledge_docs = await self._search_knowledge_docs(search_query, options)
            if knowledge_docs:
                enhanced_goal = self._build_enhanced_goal(goal, knowledge_docs)
                print_success(f"已从知识库加载 {len(knowledge_docs)} 个相关文档")

        # 方案 C: 当指定 --search-knowledge 时，同时启用 MP 任务级知识库注入
        # 否则保持默认行为（enable_knowledge_search=True 但仅对 Cursor 相关任务生效）
        enable_knowledge_search = options.get("use_knowledge", True) if search_query else True

        config = MultiProcessOrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_count=options["workers"],
            # 系统配置（tri-state 已在 _merge_options 中解析）
            enable_sub_planners=options.get("enable_sub_planners", DEFAULT_ENABLE_SUB_PLANNERS),
            strict_review=options.get("strict", DEFAULT_STRICT_REVIEW),
            planner_model=options.get("planner_model", DEFAULT_PLANNER_MODEL),
            worker_model=options.get("worker_model", DEFAULT_WORKER_MODEL),
            reviewer_model=options.get("reviewer_model", DEFAULT_REVIEWER_MODEL),
            # 流式日志配置（使用 resolved stream config）
            stream_events_enabled=options.get("stream_log", False),
            stream_log_console=options.get("stream_log_console", True),
            stream_log_detail_dir=options.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=options.get("stream_log_raw_dir", "logs/stream_json/raw/"),
            # 自动提交配置透传（默认禁用）
            enable_auto_commit=options.get("auto_commit", False),
            auto_push=options.get("auto_push", False),
            commit_per_iteration=options.get("commit_per_iteration", False),
            # commit_on_complete 语义：当 commit_per_iteration=False 时默认仅在完成时提交
            commit_on_complete=not options.get("commit_per_iteration", False),
            # 执行模式配置（MP 主要使用 CLI，但保持配置兼容性）
            execution_mode=execution_mode_str,
            planner_execution_mode=options.get("planner_execution_mode"),
            worker_execution_mode=options.get("worker_execution_mode"),
            reviewer_execution_mode=options.get("reviewer_execution_mode"),
            # 知识库配置（方案 C）
            enable_knowledge_search=enable_knowledge_search,
            # 知识库注入配置
            enable_knowledge_injection=options.get("enable_knowledge_injection", True),
            knowledge_top_k=options.get("knowledge_top_k", 3),
            knowledge_max_chars_per_doc=options.get("knowledge_max_chars_per_doc", 1200),
            knowledge_max_total_chars=options.get("knowledge_max_total_chars", 3000),
            # 日志配置透传到子进程（verbose 模式控制 DEBUG 日志输出）
            verbose=options.get("verbose", False),
            log_level="DEBUG" if options.get("verbose", False) else "INFO",
            heartbeat_debug=False,  # 心跳调试日志默认关闭，避免刷屏
        )

        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        orchestrator = MultiProcessOrchestrator(config)
        return await orchestrator.run(enhanced_goal)

    async def _run_knowledge(self, goal: str, options: dict) -> dict:
        """运行知识库增强模式"""
        from coordinator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig
        from knowledge import KnowledgeManager, KnowledgeStorage

        # ============================================================
        # 副作用控制策略
        # ============================================================
        # 从 options 获取副作用控制参数
        skip_online = options.get("skip_online", False)
        dry_run = options.get("dry_run", False)
        minimal = options.get("minimal", False)

        # 计算副作用控制策略
        policy = compute_side_effects(
            skip_online=skip_online,
            dry_run=dry_run,
            minimal=minimal,
        )

        # 初始化知识库（如果允许）
        # dry_run/minimal 模式下跳过目录创建，但仍可使用只读 storage
        manager = KnowledgeManager(name="cursor-docs")
        if policy.allow_directory_create:
            await manager.initialize()
        else:
            print_info("跳过知识库初始化（dry_run 模式）")

        # 搜索相关文档（读取操作，始终允许）
        knowledge_context: list[dict[str, str]] | None = None
        if options.get("search_knowledge"):
            # 使用只读模式初始化 storage（如果不允许创建目录）
            if policy.allow_directory_create:
                storage = KnowledgeStorage()
            else:
                storage = KnowledgeStorage.create_read_only()
            await storage.initialize()

            # 只读模式下检查索引是否存在
            if not policy.allow_directory_create and not storage.index_path.exists():
                print_info("知识库索引不存在，跳过搜索（只读模式）")
                knowledge_context = []
            else:
                results = await storage.search(options["search_knowledge"], limit=5)
                knowledge_context = []
                for result in results:
                    doc = await storage.load_document(result.doc_id)
                    if doc:
                        knowledge_context.append(
                            {
                                "title": doc.title,
                                "url": doc.url,
                                "content": doc.content[:MAX_CONSOLE_PREVIEW_CHARS],
                            }
                        )

        # 构建增强目标
        enhanced_goal = goal
        if knowledge_context:
            context_text = "\n\n## 参考文档（来自知识库）\n\n"
            for i, doc_info in enumerate(knowledge_context, 1):
                context_text += f"### {i}. {doc_info['title']}\n"
                context_text += f"```\n{doc_info['content'][:MAX_KNOWLEDGE_DOC_PREVIEW_CHARS]}\n```\n\n"
            enhanced_goal = f"{goal}\n{context_text}"

        # 创建 CursorAgentConfig，注入 resolved stream 配置
        cursor_config = CursorAgentConfig(
            working_directory=options["directory"],
            stream_events_enabled=options.get("stream_log", False),
            stream_log_console=options.get("stream_log_console", True),
            stream_log_detail_dir=options.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=options.get("stream_log_raw_dir", "logs/stream_json/raw/"),
        )

        # 获取执行模式和 Cloud 认证配置
        execution_mode = self._get_execution_mode(options)
        cloud_auth_config = self._get_cloud_auth_config(options)

        # 解析角色级执行模式（可选）
        planner_exec_mode = self._parse_execution_mode(options.get("planner_execution_mode"))
        worker_exec_mode = self._parse_execution_mode(options.get("worker_execution_mode"))
        reviewer_exec_mode = self._parse_execution_mode(options.get("reviewer_execution_mode"))

        config = OrchestratorConfig(
            working_directory=options["directory"],
            max_iterations=options["max_iterations"],
            worker_pool_size=options["workers"],
            # 系统配置（tri-state 已在 _merge_options 中解析）
            enable_sub_planners=options.get("enable_sub_planners", DEFAULT_ENABLE_SUB_PLANNERS),
            strict_review=options.get("strict", DEFAULT_STRICT_REVIEW),
            cursor_config=cursor_config,
            # 流式日志配置（使用 resolved stream config）
            stream_events_enabled=options.get("stream_log", False),
            stream_log_console=options.get("stream_log_console", True),
            stream_log_detail_dir=options.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=options.get("stream_log_raw_dir", "logs/stream_json/raw/"),
            # 自动提交配置（需显式开启）
            enable_auto_commit=options.get("auto_commit", False),
            auto_push=options.get("auto_push", False),
            commit_per_iteration=options.get("commit_per_iteration", False),
            # 执行模式和 Cloud 认证配置
            execution_mode=execution_mode,
            cloud_auth_config=cloud_auth_config,
            # 角色级执行模式（可选）
            planner_execution_mode=planner_exec_mode,
            worker_execution_mode=worker_exec_mode,
            reviewer_execution_mode=reviewer_exec_mode,
            # 流式控制台渲染配置
            stream_console_renderer=options.get("stream_console_renderer", False),
            stream_advanced_renderer=options.get("stream_advanced_renderer", False),
            stream_typing_effect=options.get("stream_typing_effect", False),
            stream_typing_delay=options.get("stream_typing_delay", 0.02),
            stream_word_mode=options.get("stream_word_mode", True),
            stream_color_enabled=options.get("stream_color_enabled", True),
            stream_show_word_diff=options.get("stream_show_word_diff", False),
        )

        if options["max_iterations"] == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        orchestrator = Orchestrator(config, knowledge_manager=manager)
        return await orchestrator.run(enhanced_goal)

    async def _run_iterate(self, goal: str, options: dict) -> dict:
        """运行自我迭代模式"""
        # 导入自我迭代模块
        from scripts.run_iterate import (
            SelfIterator,
        )

        # 创建模拟的 args 对象
        # 字段命名与 scripts/run_iterate.py 的 parse_args 保持一致
        class IterateArgs:
            def __init__(self, goal: str, opts: dict):
                # 基础参数（与 scripts/run_iterate.py argparse 参数对齐）
                self.requirement = goal
                self.skip_online = opts.get("skip_online", False)
                self.changelog_url = "https://cursor.com/cn/changelog"
                self.dry_run = opts.get("dry_run", False)
                # minimal: 最小副作用模式（等效于 skip_online + dry_run）
                self.minimal = opts.get("minimal", False)
                # max_iterations: 字符串类型（与 argparse 一致）
                self.max_iterations = str(opts.get("max_iterations", DEFAULT_MAX_ITERATIONS))
                # workers: 整数类型
                self.workers = opts.get("workers", DEFAULT_WORKER_POOL_SIZE)
                self.force_update = opts.get("force_update", False)
                self.verbose = opts.get("verbose", False)
                # 工作目录
                self.directory = opts.get("directory", ".")

                # 系统配置（从 config.yaml 或 _merge_options 传入）
                self.enable_sub_planners = opts.get("enable_sub_planners", DEFAULT_ENABLE_SUB_PLANNERS)
                self.strict_review = opts.get("strict", DEFAULT_STRICT_REVIEW)

                # 自动提交选项
                self.auto_commit = opts.get("auto_commit", False)
                self.auto_push = opts.get("auto_push", False)
                self.commit_per_iteration = opts.get("commit_per_iteration", False)
                self.commit_message = opts.get("commit_message", "")

                # 编排器选项（tri-state 已在 _merge_options 中处理）
                self.orchestrator = opts.get("orchestrator", "mp")
                self.no_mp = opts.get("no_mp", False)
                # 元字段：标记用户是否显式设置了编排器选项
                # 当 True 时，SelfIterator 不会被 requirement 中的非并行关键词覆盖
                self._orchestrator_user_set = opts.get("_orchestrator_user_set", False)

                # 执行模式和 Cloud 认证参数（与 scripts/run_iterate.py 对齐）
                self.execution_mode = opts.get("execution_mode", "cli")
                self.cloud_api_key = opts.get("cloud_api_key")
                # cloud_auth_timeout: 整数类型（与 argparse 一致）
                self.cloud_auth_timeout = opts.get("cloud_auth_timeout", DEFAULT_CLOUD_AUTH_TIMEOUT)
                # cloud_timeout: 整数类型（与 argparse 一致）
                self.cloud_timeout = opts.get("cloud_timeout", DEFAULT_CLOUD_TIMEOUT)

                # 角色级执行模式（从 _merge_options 映射）
                self.planner_execution_mode = opts.get("planner_execution_mode")
                self.worker_execution_mode = opts.get("worker_execution_mode")
                self.reviewer_execution_mode = opts.get("reviewer_execution_mode")

                # 日志控制参数
                self.quiet = opts.get("quiet", False)
                self.log_level = opts.get("log_level")
                self.heartbeat_debug = opts.get("heartbeat_debug", False)

                # 卡死诊断参数
                self.stall_diagnostics_enabled = opts.get("stall_diagnostics_enabled")
                self.stall_diagnostics_level = opts.get("stall_diagnostics_level")
                self.stall_recovery_interval = opts.get("stall_recovery_interval", 30.0)
                self.execution_health_check_interval = opts.get("execution_health_check_interval", 30.0)
                self.health_warning_cooldown = opts.get("health_warning_cooldown", 60.0)

                # 流式控制台渲染配置
                self.stream_console_renderer = opts.get("stream_console_renderer", False)
                self.stream_advanced_renderer = opts.get("stream_advanced_renderer", False)
                self.stream_typing_effect = opts.get("stream_typing_effect", False)
                self.stream_typing_delay = opts.get("stream_typing_delay", 0.02)
                self.stream_word_mode = opts.get("stream_word_mode", True)
                self.stream_color_enabled = opts.get("stream_color_enabled", True)
                self.stream_show_word_diff = opts.get("stream_show_word_diff", False)

                # 文档源配置参数（tri-state：None=使用 config.yaml 默认值）
                self.max_fetch_urls = opts.get("max_fetch_urls")
                self.fallback_core_docs_count = opts.get("fallback_core_docs_count")
                self.llms_txt_url = opts.get("llms_txt_url")
                self.llms_cache_path = opts.get("llms_cache_path")

        # 从原始 args 中获取 _orchestrator_user_set 元字段
        # 这确保用户显式传入 --orchestrator/--no-mp 时，SelfIterator 不会被
        # requirement 中的非并行关键词反向覆盖
        options["_orchestrator_user_set"] = getattr(self.args, "_orchestrator_user_set", False)

        iterate_args = IterateArgs(goal, options)
        iterator = SelfIterator(iterate_args)

        return await iterator.run()

    async def _run_plan(self, goal: str, options: dict) -> dict:
        """运行规划模式（仅规划不执行）

        使用 PlanAgentExecutor 确保:
        - mode=plan（规划模式）
        - force_write=False（只读保证，不修改文件）
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import PlanAgentExecutor

        print_info("仅规划模式：分析任务并生成执行计划...")

        # 构建规划提示
        prompt = f"""你是一个任务规划专家。请分析以下任务并制定详细的执行计划。

任务目标：{goal}

请提供：
1. 任务分解（将任务拆分为具体的子任务）
2. 执行顺序（哪些任务可以并行，哪些需要顺序执行）
3. 预估复杂度（每个子任务的难度评估）
4. 潜在风险（可能遇到的问题和解决方案）
5. 推荐运行模式（basic/mp/knowledge/iterate）

请以结构化的方式输出计划。"""

        try:
            # 创建规划模式执行器（强制 mode=plan, force_write=False）
            config = CursorAgentConfig(
                working_directory=options.get("directory", str(project_root)),
                timeout=300,
            )
            executor = PlanAgentExecutor(config=config)

            # 执行规划
            result = await executor.execute(
                prompt=prompt,
                working_directory=options.get("directory", str(project_root)),
                timeout=300,
            )

            if result.success:
                plan_output = result.output.strip()
                print("\n" + "=" * 60)
                print("执行计划")
                print("=" * 60)
                print(plan_output)
                print("=" * 60 + "\n")

                # 使用输出契约字段常量确保字段名一致
                return {
                    ResultFields.SUCCESS: True,
                    ResultFields.GOAL: goal,
                    ResultFields.MODE: "plan",
                    "plan": plan_output,
                    "dry_run": True,
                }
            else:
                error_msg = result.error or "未知错误"
                print_error(f"规划失败: {error_msg}")
                return {
                    ResultFields.SUCCESS: False,
                    ResultFields.GOAL: goal,
                    ResultFields.MODE: "plan",
                    ResultFields.ERROR: error_msg,
                }

        except asyncio.TimeoutError:
            print_error("规划超时")
            return {
                ResultFields.SUCCESS: False,
                ResultFields.GOAL: goal,
                ResultFields.MODE: "plan",
                ResultFields.ERROR: "timeout",
            }
        except Exception as e:
            print_error(f"规划异常: {e}")
            return {
                ResultFields.SUCCESS: False,
                ResultFields.GOAL: goal,
                ResultFields.MODE: "plan",
                ResultFields.ERROR: str(e),
            }

    async def _run_ask(self, goal: str, options: dict) -> dict:
        """运行问答模式（直接对话）

        使用 AskAgentExecutor 确保:
        - mode=ask（问答模式）
        - force_write=False（只读保证，不修改文件）
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import AskAgentExecutor

        print_info("问答模式：直接与 Agent 对话...")

        try:
            # 创建问答模式执行器（强制 mode=ask, force_write=False）
            config = CursorAgentConfig(
                working_directory=options.get("directory", str(project_root)),
                timeout=120,
            )
            executor = AskAgentExecutor(config=config)

            # 执行问答
            result = await executor.execute(
                prompt=goal,
                working_directory=options.get("directory", str(project_root)),
                timeout=120,
            )

            if result.success:
                answer = result.output.strip()
                print("\n" + "=" * 60)
                print("回答")
                print("=" * 60)
                print(answer)
                print("=" * 60 + "\n")

                # 使用输出契约字段常量确保字段名一致
                return {
                    ResultFields.SUCCESS: True,
                    ResultFields.GOAL: goal,
                    ResultFields.MODE: "ask",
                    "answer": answer,
                }
            else:
                error_msg = result.error or "未知错误"
                print_error(f"问答失败: {error_msg}")
                return {
                    ResultFields.SUCCESS: False,
                    ResultFields.GOAL: goal,
                    ResultFields.MODE: "ask",
                    ResultFields.ERROR: error_msg,
                }

        except asyncio.TimeoutError:
            print_error("问答超时")
            return {
                ResultFields.SUCCESS: False,
                ResultFields.GOAL: goal,
                ResultFields.MODE: "ask",
                ResultFields.ERROR: "timeout",
            }
        except Exception as e:
            print_error(f"问答异常: {e}")
            return {
                ResultFields.SUCCESS: False,
                ResultFields.GOAL: goal,
                ResultFields.MODE: "ask",
                ResultFields.ERROR: str(e),
            }

    async def _run_cloud(self, goal: str, options: dict) -> dict:
        """运行 Cloud 模式（云端执行）

        Cloud/Auto 语义统一说明:
        - cloud_enabled: 控制 '&' 前缀的自动检测
        - execution_mode=auto: Cloud 优先，按错误类型冷却后回退到 CLI
        - force_write: 独立于 auto_commit，由 --force 控制
        - auto_commit: 需显式 --auto-commit 开启

        执行模式:
        - background=False: 前台模式，等待完成（--execution-mode cloud 默认）
        - background=True: 后台模式 (Cloud Relay)，立即返回 session_id（& 前缀默认）

        配置 API Key 的三种方式:
          1. export CURSOR_API_KEY=your_key
          2. --cloud-api-key your_key
          3. agent login

        超时: 使用 --cloud-timeout（默认 300 秒），不从 max_iterations 推导
        """
        from cursor.client import CursorAgentConfig
        from cursor.executor import AgentExecutorFactory, ExecutionMode

        # 获取后台模式配置
        # 默认行为：& 前缀触发默认 True，--execution-mode cloud 默认 False
        cloud_background_value = options.get("cloud_background")
        if cloud_background_value is None:
            cloud_background = bool(options.get("prefix_routed", False))
        else:
            cloud_background = cloud_background_value

        # 字段语义说明：
        # - has_ampersand_prefix: 语法检测（用于消息中是否提及 & 前缀）
        # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
        has_ampersand_prefix = options.get("has_ampersand_prefix", False)
        prefix_routed = options.get("prefix_routed", False)

        if cloud_background:
            print_info("Cloud 后台模式：任务将提交到云端后台执行...")
        else:
            print_info("Cloud 前台模式：任务将在云端执行并等待完成...")

        # 复用已有的 Cloud 认证配置获取逻辑
        # 优先级：--cloud-api-key > CURSOR_API_KEY 环境变量
        cloud_auth_config = self._get_cloud_auth_config(options)

        # 检查 API key 是否已配置，使用 Policy 构建 fallback 消息
        if cloud_auth_config is None:
            # 构造 CloudFailureInfo 并使用 build_cooldown_info 统一构建 cooldown_info
            failure_info = CloudFailureInfo(
                kind=CloudFailureKind.NO_KEY,
                message="未配置 API Key",
                retryable=False,
                retry_after=None,
            )
            no_key_cooldown_info = build_cooldown_info(
                failure_info=failure_info,
                fallback_reason="未配置 API Key",
                requested_mode="cloud",
                has_ampersand_prefix=has_ampersand_prefix,
                mode_source="cli",
            )
            # 输出用户友好消息（使用 prepare_cooldown_message 按 level 输出，与去重一致）
            msg_output = prepare_cooldown_message(no_key_cooldown_info)
            if msg_output and msg_output.dedup_key not in TaskAnalyzer._shown_messages:
                TaskAnalyzer._shown_messages.add(msg_output.dedup_key)
                if msg_output.level == "warning":
                    print_warning(msg_output.user_message)
                else:
                    print_info(msg_output.user_message)
            return build_cloud_error_result(
                goal=goal,
                error="未配置 API Key",
                failure_kind=CloudFailureKind.NO_KEY.value,
                background=cloud_background,
                has_ampersand_prefix=has_ampersand_prefix,
                prefix_routed=prefix_routed,
                retry_after=None,  # 无 key 分支无需重试等待
                retryable=False,
                cooldown_info=no_key_cooldown_info,
            )

        try:
            # force_write 默认 False，与本地 CLI 保持一致
            # 仅当用户显式传入 --force 或 options["force_write"]=True 时才允许写入
            force_write = options.get("force_write", False)

            # 使用独立的 Cloud 超时参数（不从 max_iterations 推导）
            # 默认 300 秒（5 分钟），可通过 --cloud-timeout 覆盖
            cloud_timeout = options.get("cloud_timeout", 300)

            # 创建 Cloud 执行器配置
            config = CursorAgentConfig(
                working_directory=options.get("directory", str(project_root)),
                timeout=cloud_timeout,
                force_write=force_write,
            )
            executor = AgentExecutorFactory.create(
                mode=ExecutionMode.CLOUD,
                cli_config=config,
                cloud_auth_config=cloud_auth_config,
            )

            # 执行任务（后台模式优先使用 submit_background_task）
            submit_background = getattr(executor, "submit_background_task", None)
            if cloud_background and callable(submit_background):
                submit_result = submit_background(prompt=goal)
                if inspect.isawaitable(submit_result):
                    result = await submit_result
                else:
                    result = await executor.execute(
                        prompt=goal,
                        working_directory=options.get("directory", str(project_root)),
                        timeout=cloud_timeout,
                        background=cloud_background,
                    )
            else:
                result = await executor.execute(
                    prompt=goal,
                    working_directory=options.get("directory", str(project_root)),
                    timeout=cloud_timeout,
                    background=cloud_background,
                )

            session_id = result.session_id
            resume_command = f"agent --resume {session_id}" if session_id else None

            # 统一读取结果中的 cooldown_info 并输出回退消息（仅输出一次）
            # 使用 prepare_cooldown_message 纯函数计算 dedup_key 和 level
            msg_output = prepare_cooldown_message(result.cooldown_info)
            if msg_output and msg_output.dedup_key not in TaskAnalyzer._shown_messages:
                TaskAnalyzer._shown_messages.add(msg_output.dedup_key)
                if msg_output.level == "warning":
                    print_warning(msg_output.user_message)
                else:
                    print_info(msg_output.user_message)

            if result.success:
                output = result.output.strip()

                # 后台模式：简化输出，不打印大段内容（任务还未完成）
                if cloud_background:
                    print("\n" + "=" * 60)
                    print("Cloud 后台任务已提交")
                    print("=" * 60)
                    if session_id:
                        print(f"\n{Colors.CYAN}会话 ID: {session_id}{Colors.NC}")
                        print(f"{Colors.CYAN}恢复会话: {resume_command}{Colors.NC}")
                        print(f"{Colors.CYAN}查看历史: agent ls{Colors.NC}\n")
                    print("=" * 60 + "\n")
                else:
                    # 前台模式：打印完整结果
                    print("\n" + "=" * 60)
                    print("Cloud 执行结果")
                    print("=" * 60)
                    print(output[:MAX_CONSOLE_PREVIEW_CHARS] if len(output) > MAX_CONSOLE_PREVIEW_CHARS else output)
                    if len(output) > MAX_CONSOLE_PREVIEW_CHARS:
                        print(TRUNCATION_HINT_OUTPUT)
                    print("=" * 60)

                    # 输出 session_id 使用说明
                    if session_id:
                        print(f"\n{Colors.CYAN}会话 ID: {session_id}{Colors.NC}")
                        print(f"{Colors.CYAN}恢复会话: {resume_command}{Colors.NC}\n")

                retry_after = int(result.retry_after) if result.retry_after is not None else None
                return build_cloud_success_result(
                    goal=goal,
                    output=output if not cloud_background else "",
                    session_id=session_id,
                    files_modified=result.files_modified,
                    background=cloud_background,
                    has_ampersand_prefix=has_ampersand_prefix,
                    prefix_routed=prefix_routed,
                    failure_kind=result.failure_kind,
                    retry_after=retry_after,
                    cooldown_info=result.cooldown_info,
                )
            else:
                error_msg = result.error or "未知错误"
                # 如果已输出 cooldown_info 中的用户消息，则不再重复输出错误
                if not (result.cooldown_info and result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)):
                    print_error(f"Cloud 执行失败: {error_msg}")
                retry_after = int(result.retry_after) if result.retry_after is not None else None
                return build_cloud_error_result(
                    goal=goal,
                    error=error_msg,
                    failure_kind=result.failure_kind or CloudFailureKind.UNKNOWN.value,
                    session_id=session_id,
                    background=cloud_background,
                    has_ampersand_prefix=has_ampersand_prefix,
                    prefix_routed=prefix_routed,
                    retry_after=retry_after,
                    retryable=retry_after is not None,
                    cooldown_info=result.cooldown_info,
                )

        except asyncio.TimeoutError as e:
            # 使用 Policy 分类错误
            failure_info = classify_cloud_failure(e)
            cloud_timeout = options.get("cloud_timeout", 300)
            # 先构建 cooldown_info，再使用 prepare_cooldown_message 按 level 输出
            timeout_cooldown_info = build_cooldown_info(
                failure_info=failure_info,
                fallback_reason=f"执行超时 ({cloud_timeout}s)",
                requested_mode="cloud",
                has_ampersand_prefix=has_ampersand_prefix,
                mode_source="cli",  # cloud 模式来自 CLI 显式指定
            )
            # 按 level 输出回退消息（warning/info），避免使用 print_error
            msg_output = prepare_cooldown_message(timeout_cooldown_info)
            if msg_output and msg_output.dedup_key not in TaskAnalyzer._shown_messages:
                TaskAnalyzer._shown_messages.add(msg_output.dedup_key)
                if msg_output.level == "warning":
                    print_warning(msg_output.user_message)
                else:
                    print_info(msg_output.user_message)
            return build_cloud_error_result(
                goal=goal,
                error=f"执行超时 ({cloud_timeout}s)",
                failure_kind=failure_info.kind.value,
                background=cloud_background,
                has_ampersand_prefix=has_ampersand_prefix,
                prefix_routed=prefix_routed,
                retry_after=failure_info.retry_after,
                retryable=failure_info.retryable,
                cooldown_info=timeout_cooldown_info,
            )
        except Exception as e:
            # 使用 Policy 分类错误
            failure_info = classify_cloud_failure(e)
            # 先构建 cooldown_info，再使用 prepare_cooldown_message 按 level 输出
            exception_cooldown_info = build_cooldown_info(
                failure_info=failure_info,
                fallback_reason=str(e),
                requested_mode="cloud",
                has_ampersand_prefix=has_ampersand_prefix,
                mode_source="cli",  # cloud 模式来自 CLI 显式指定
            )
            # 按 level 输出回退消息（warning/info），避免使用 print_error
            msg_output = prepare_cooldown_message(exception_cooldown_info)
            if msg_output and msg_output.dedup_key not in TaskAnalyzer._shown_messages:
                TaskAnalyzer._shown_messages.add(msg_output.dedup_key)
                if msg_output.level == "warning":
                    print_warning(msg_output.user_message)
                else:
                    print_info(msg_output.user_message)
            return build_cloud_error_result(
                goal=goal,
                error=str(e),
                failure_kind=failure_info.kind.value,
                background=cloud_background,
                has_ampersand_prefix=has_ampersand_prefix,
                prefix_routed=prefix_routed,
                retry_after=failure_info.retry_after,
                retryable=failure_info.retryable,
                cooldown_info=exception_cooldown_info,
            )


# ============================================================
# 结果输出
# ============================================================


def print_result(result: dict) -> None:
    """打印执行结果"""
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)

    success = result.get("success", False)
    print(f"\n状态: {Colors.GREEN if success else Colors.YELLOW}{'成功' if success else '未完成'}{Colors.NC}")

    if result.get("goal"):
        goal_preview = result["goal"][:MAX_GOAL_SUMMARY_CHARS]
        if len(result["goal"]) > MAX_GOAL_SUMMARY_CHARS:
            goal_preview += TRUNCATION_HINT
        print(f"目标: {goal_preview}")

    if "iterations_completed" in result:
        print(f"完成迭代: {result['iterations_completed']}")

    if "total_tasks_created" in result:
        print("\n任务统计:")
        print(f"  - 创建: {result.get('total_tasks_created', 0)}")
        print(f"  - 完成: {result.get('total_tasks_completed', 0)}")
        print(f"  - 失败: {result.get('total_tasks_failed', 0)}")

    if result.get("final_score"):
        print(f"\n最终评分: {result['final_score']:.1f}")

    if result.get("dry_run"):
        print(f"\n{Colors.YELLOW}(Dry-run 模式，未实际执行){Colors.NC}")

    # 显示 minimal 模式和副作用状态（仅在启用时显示）
    if result.get("minimal"):
        print("\n副作用控制:")
        print("  模式: minimal（最小副作用）")
        side_effects = result.get("side_effects", {})
        if side_effects:
            print(f"  - network: {'允许' if side_effects.get('network') else '禁止'}")
            print(f"  - disk_write: {'允许' if side_effects.get('disk_write') else '禁止'}")
            print(f"  - knowledge_write: {'允许' if side_effects.get('knowledge_write') else '禁止'}")

    print("=" * 60)


# ============================================================
# 日志配置
# ============================================================


def setup_logging(
    verbose: bool = False,
    enable_file_logging: bool = True,
) -> None:
    """配置日志

    Args:
        verbose: 是否启用详细输出（DEBUG 级别）
        enable_file_logging: 是否启用文件日志。为 False 时只保留 stderr sink，
                             不创建日志目录和文件 sink（minimal 模式）
    """
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        filter=lambda record: record["level"].name != "DEBUG" or verbose,
    )

    # minimal 模式：只保留 stderr sink，不创建日志目录和文件 sink
    if not enable_file_logging:
        return

    # 确保日志目录存在
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        str(log_dir / "run_{time}.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


# ============================================================
# 主入口
# ============================================================


def _build_cli_overrides_from_args(args: argparse.Namespace) -> dict:
    """从命令行参数构建 CLI overrides 字典

    用于 format_debug_config 和 resolve_orchestrator_settings。
    委托给 core.config.build_cli_overrides_from_args 统一实现。

    Args:
        args: 命令行参数

    Returns:
        CLI overrides 字典
    """
    from core.config import build_cli_overrides_from_args

    return build_cli_overrides_from_args(args)


async def async_main() -> int:
    """异步主函数"""
    args = parse_args()
    setup_logging(args.verbose)

    # 处理 --print-config 参数：打印配置调试信息并退出
    if getattr(args, "print_config", False):
        import os

        cli_overrides = _build_cli_overrides_from_args(args)
        # 检测是否有 API Key（用于判断 effective_mode）
        has_api_key = bool(
            getattr(args, "cloud_api_key", None)
            or os.environ.get("CURSOR_API_KEY")
            or os.environ.get("CURSOR_CLOUD_API_KEY")
        )
        print_debug_config(
            cli_overrides,
            source_label="run.py",
            has_api_key=has_api_key,
        )
        return 0

    # 解析模式
    explicit_mode = MODE_ALIASES.get(args.mode, RunMode.AUTO)

    # 任务分析（优先使用执行者 Agent 进行参数/任务解析）
    analysis: TaskAnalysis
    if args.task and not args.no_auto_analyze:
        print_info("分析参数与任务...")
        analyzer = TaskAnalyzer(use_agent=True)
        analysis = analyzer.analyze(args.task, args)
    else:
        if explicit_mode == RunMode.AUTO:
            explicit_mode = RunMode.BASIC
        analysis = TaskAnalysis(
            mode=explicit_mode,
            goal=args.task,
            reasoning="使用指定模式",
        )

    # 显式指定模式时，优先使用指定模式（但保留解析出的参数/目标）
    if explicit_mode != RunMode.AUTO:
        analysis.mode = explicit_mode
        analysis.reasoning = TaskAnalyzer._merge_reasoning(
            analysis.reasoning,
            "显式指定模式，覆盖自动判断",
        )

    # 检查是否有任务
    if not analysis.goal:
        print_error("请提供任务描述")
        print_info('用法: python run.py [--mode MODE] "任务描述"')
        return 1

    # 运行任务
    runner = Runner(args)

    try:
        result = await runner.run(analysis)
        print_result(result)
        return 0 if result.get("success") else 1

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}用户中断{Colors.NC}")
        return 130


def main() -> None:
    """主函数"""
    import gc
    import warnings

    # 抑制子进程清理时的事件循环关闭警告
    # 这是 Python 3.10+ 的已知问题，不影响程序功能
    warnings.filterwarnings(
        "ignore",
        message=".*Event loop is closed.*",
        category=RuntimeWarning,
    )

    try:
        # 使用自定义事件循环运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exit_code = loop.run_until_complete(async_main())
        finally:
            # 清理待处理的任务
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # 等待所有任务完成
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            # 关闭异步生成器
            loop.run_until_complete(loop.shutdown_asyncgens())
            # 关闭默认执行器
            loop.run_until_complete(loop.shutdown_default_executor())
            # 强制垃圾回收，在循环关闭前清理子进程对象
            gc.collect()
            loop.close()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)


if __name__ == "__main__":
    # 设置异常钩子抑制子进程清理错误
    def _subprocess_exception_handler(loop, context):
        """忽略子进程清理时的事件循环关闭错误"""
        if "Event loop is closed" in str(context.get("message", "")):
            return
        loop.default_exception_handler(context)

    main()
