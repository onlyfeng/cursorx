#!/usr/bin/env python3
"""CursorX 统一入口脚本

提供统一的命令行入口，支持多种运行模式和自然语言任务描述。

运行模式：
  - basic:     基本协程模式（规划-执行-审核）
  - mp:        多进程模式（并行执行）
  - knowledge: 知识库增强模式（自动搜索相关文档）
  - iterate:   自我迭代模式（检查更新、更新知识库、执行任务）
  - auto:      自动分析模式（使用 Agent 分析任务并选择最佳模式）

用法示例：
  # 显式指定模式
  python run.py --mode basic "实现 REST API"
  python run.py --mode mp "重构代码" --workers 5
  python run.py --mode iterate "更新 CLI 支持"

  # 自动模式（Agent 分析任务）
  python run.py "启动自我迭代，跳过在线更新"
  python run.py "使用多进程模式重构 src 目录"

  # 无限迭代
  python run.py "实现功能" --max-iterations MAX
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger

from core.cloud_utils import CLOUD_PREFIX, is_cloud_request, strip_cloud_prefix
from core.config import (
    DEFAULT_PLANNER_MODEL,
    DEFAULT_WORKER_MODEL,
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_CLOUD_TIMEOUT,
    DEFAULT_CLOUD_AUTH_TIMEOUT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_WORKER_POOL_SIZE,
    DEFAULT_ENABLE_SUB_PLANNERS,
    DEFAULT_STRICT_REVIEW,
    get_config,
    resolve_stream_log_config,
    resolve_orchestrator_settings,
    parse_max_iterations,
    parse_max_iterations_for_argparse,
    build_cursor_agent_config,
    format_debug_config,
    print_debug_config,
)
from cursor.cloud_client import CloudAuthConfig, CloudClientFactory
from cursor.executor import ExecutionMode
from core.config import build_cloud_client_config
from core.project_workspace import inspect_project_state

# ============================================================
# 运行模式定义
# ============================================================

class RunMode(str, Enum):
    """运行模式"""
    BASIC = "basic"           # 基本协程模式
    MP = "mp"                 # 多进程模式
    KNOWLEDGE = "knowledge"   # 知识库增强模式
    ITERATE = "iterate"       # 自我迭代模式
    AUTO = "auto"             # 自动分析模式
    PLAN = "plan"             # 仅规划模式（不执行）
    ASK = "ask"               # 问答模式（直接对话）
    CLOUD = "cloud"           # Cloud 模式（云端执行）


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
        "--mode", "-M",
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
        "-d", "--directory",
        type=str,
        default=None,
        help="工作目录 (默认: 当前目录)；显式指定时触发工程初始化/参考子工程发现",
    )

    # workers 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help=f"Worker 数量 (默认: {cfg_worker_pool_size}，来自 config.yaml)",
    )

    # max_iterations 使用 tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "-m", "--max-iterations",
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
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )

    parser.add_argument(
        "-q", "--quiet",
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

    # 自我迭代模式专用参数
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="[iterate] 跳过在线文档检查",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="[iterate] 仅分析不执行",
    )

    parser.add_argument(
        "--force-update",
        action="store_true",
        help="[iterate] 强制更新知识库",
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
        help=f"执行模式: cli=本地CLI, auto=自动选择(Cloud优先), cloud=强制Cloud (默认: {cfg_execution_mode}，来自 config.yaml)",
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
    args._orchestrator_user_set = any(
        arg in sys.argv for arg in ["--orchestrator", "--no-mp"]
    )

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
            "云端", "cloud", "后台执行", "background",
            "推送云端", "云端任务", "cloud agent",
        ],
        RunMode.ITERATE: [
            "自我迭代", "self-iterate", "iterate", "迭代更新",
            "更新知识库", "检查更新", "changelog", "自我更新",
        ],
        RunMode.MP: [
            "多进程", "multiprocess", "并行", "parallel",
            "多worker", "多 worker", "并发",
        ],
        RunMode.KNOWLEDGE: [
            "知识库", "knowledge", "文档搜索", "搜索文档",
            "cursor 文档", "参考文档", "docs",
        ],
        RunMode.PLAN: [
            "规划", "plan", "planning", "分析任务", "任务分析",
            "制定计划", "仅规划", "只规划", "分解任务",
        ],
        RunMode.ASK: [
            "问答", "ask", "chat", "对话", "提问", "询问",
            "直接问", "回答", "解答", "question",
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
    }

    # 禁用选项关键词映射
    DISABLE_KEYWORDS = {
        "no_auto_commit": ["禁用提交", "关闭提交", "不提交", "跳过提交", "no-commit", "禁用自动提交"],
        "no_stream_log": ["禁用流式日志", "关闭流式", "no-stream", "简洁模式", "静默模式"],
        "no_auto_push": ["禁用推送", "不推送", "关闭推送", "no-push"],
    }

    # 非并行/协程模式关键词映射（用于 iterate 模式的编排器选择）
    NON_PARALLEL_KEYWORDS = [
        "禁用多进程", "禁用 mp", "不使用多进程", "单进程",
        "非并行", "顺序执行", "协程模式", "基本模式",
        "basic 编排器", "basic编排器", "no-mp", "no_mp",
        "--no-mp", "使用协程", "使用 basic",
    ]

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

    def __init__(self, use_agent: bool = True):
        """初始化分析器

        Args:
            use_agent: 是否使用 Agent 进行高级分析
        """
        self.use_agent = use_agent

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

        # 先用 Agent 进行参数 + 任务解析（执行者模型，默认只读 ask）
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

    def _rule_based_analysis(self, task: str, args: argparse.Namespace) -> TaskAnalysis:
        """基于规则的分析"""
        task_lower = task.lower()
        analysis = TaskAnalysis(goal=task)
        reasoning_parts = []

        # 优先检测 '&' 前缀（Cloud 模式）
        if is_cloud_request(task):
            analysis.mode = RunMode.CLOUD
            # 设置 Cloud 模式相关选项
            analysis.options["execution_mode"] = "cloud"
            # '&' 前缀触发时，默认使用后台模式（Cloud Relay 语义）
            analysis.options["cloud_background"] = True
            analysis.options["triggered_by_prefix"] = True
            # 去除 '&' 前缀，保留实际任务内容
            analysis.goal = strip_cloud_prefix(task)
            reasoning_parts.append("检测到 '&' 前缀，使用 Cloud 后台模式")
            # Cloud 模式默认不开启 auto_commit（安全策略）
            # allow_write/force_write 由用户显式控制
        else:
            # 检测模式关键词
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

        # 提取 Worker 数量
        worker_match = re.search(r'(\d+)\s*(个)?\s*(worker|进程|并行)', task_lower)
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
        if rule_analysis.options.get("triggered_by_prefix"):
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

        使用 --mode ask 确保只读执行，不会修改任何文件。
        """
        try:
            context_payload = self._build_agent_context(task, args)
            prompt = self._build_agent_prompt(task, context_payload)
            cmd = ["agent", "-p", prompt, "--output-format", "text", "--mode", "ask"]

            model = self._resolve_worker_model(args)
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
        """规范化 Agent 返回的 options"""
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

        # 如果用户显式设置了目录，则不覆盖
        if args and getattr(args, "_directory_user_set", False):
            normalized.pop("directory", None)

        # 如果用户显式设置了编排器，则不允许覆盖
        if args and getattr(args, "_orchestrator_user_set", False):
            normalized.pop("orchestrator", None)
            normalized.pop("no_mp", None)

        return normalized

    def _resolve_worker_model(self, args: Optional[argparse.Namespace]) -> Optional[str]:
        """解析执行者模型，用于 Agent 任务解析"""
        env_model = os.getenv("TASK_ANALYSIS_MODEL")
        if env_model:
            return env_model

        if args and getattr(args, "worker_model", None):
            return args.worker_model

        try:
            config = get_config()
            return config.models.worker
        except Exception:
            return DEFAULT_WORKER_MODEL

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
            summary.update({
                "state": info.state.value,
                "detected_language": info.detected_language,
                "marker_files": info.marker_files[:5],
                "source_files_count": info.source_files_count,
            })
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
            "你是执行者Agent，请先解析参数和任务信息，并给出结构化结果。\n"
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

        优先级（从高到低）：
        1. 自然语言显式指定 (analysis_options)
        2. CLI 显式参数 (self.args.xxx 且非 None)
        3. config.yaml 配置
        4. 代码默认值

        tri-state 参数处理：
        - CLI 参数 default=None 表示"未显式指定"
        - 当 CLI 参数为 None 时，使用 config.yaml 值
        - 自然语言分析结果可覆盖 CLI 参数

        使用 resolve_orchestrator_settings 统一解析核心编排器配置。
        """
        # 构建 CLI overrides 字典，仅包含显式指定的参数
        cli_overrides = {}

        # workers: CLI 显式参数
        if self.args.workers is not None:
            cli_overrides["workers"] = self.args.workers

        # max_iterations: CLI 显式参数
        if self.args.max_iterations is not None:
            cli_overrides["max_iterations"] = parse_max_iterations(self.args.max_iterations)

        # strict_review: CLI 显式参数
        cli_strict = getattr(self.args, "strict_review", None)
        if cli_strict is not None:
            cli_overrides["strict_review"] = cli_strict

        # enable_sub_planners: CLI 显式参数
        cli_sub_planners = getattr(self.args, "enable_sub_planners", None)
        if cli_sub_planners is not None:
            cli_overrides["enable_sub_planners"] = cli_sub_planners

        # execution_mode: CLI 显式参数
        cli_execution_mode = getattr(self.args, "execution_mode", None)
        if cli_execution_mode is not None:
            cli_overrides["execution_mode"] = cli_execution_mode

        # cloud_timeout: CLI 显式参数
        cli_cloud_timeout = getattr(self.args, "cloud_timeout", None)
        if cli_cloud_timeout is not None:
            cli_overrides["cloud_timeout"] = cli_cloud_timeout

        # cloud_auth_timeout: CLI 显式参数
        cli_cloud_auth_timeout = getattr(self.args, "cloud_auth_timeout", None)
        if cli_cloud_auth_timeout is not None:
            cli_overrides["cloud_auth_timeout"] = cli_cloud_auth_timeout

        # dry_run: CLI 显式参数
        if self.args.dry_run:
            cli_overrides["dry_run"] = True

        # auto_commit/auto_push: CLI 显式参数
        cli_auto_commit = getattr(self.args, "auto_commit", False)
        cli_auto_push = getattr(self.args, "auto_push", False)
        cli_commit_per_iteration = getattr(self.args, "commit_per_iteration", False)
        if cli_auto_commit:
            cli_overrides["auto_commit"] = True
        if cli_auto_push:
            cli_overrides["auto_push"] = True
        if cli_commit_per_iteration:
            cli_overrides["commit_per_iteration"] = True

        # 编排器配置
        cli_orchestrator = getattr(self.args, "orchestrator", None)
        cli_no_mp = getattr(self.args, "no_mp", None)
        if cli_orchestrator is not None:
            cli_overrides["orchestrator"] = cli_orchestrator
        elif cli_no_mp is True:
            cli_overrides["orchestrator"] = "basic"

        # 模型配置: CLI 显式参数（None 表示未指定）
        if self.args.planner_model is not None:
            cli_overrides["planner_model"] = self.args.planner_model
        if self.args.worker_model is not None:
            cli_overrides["worker_model"] = self.args.worker_model
        if self.args.reviewer_model is not None:
            cli_overrides["reviewer_model"] = self.args.reviewer_model

        # 应用自然语言覆盖（最高优先级）
        # 但对于 orchestrator，需要检查 _orchestrator_user_set 标志
        orchestrator_user_set = getattr(self.args, "_orchestrator_user_set", False)
        for key in ["workers", "max_iterations", "strict", "enable_sub_planners",
                    "execution_mode", "orchestrator", "auto_commit", "auto_push", "dry_run"]:
            if key in analysis_options and analysis_options[key] is not None:
                # 如果用户显式设置了 orchestrator，则不被自然语言覆盖
                if key == "orchestrator" and orchestrator_user_set:
                    continue
                # strict 映射为 strict_review
                override_key = "strict_review" if key == "strict" else key
                cli_overrides[override_key] = analysis_options[key]

        # 使用统一的 resolve_orchestrator_settings 解析核心配置
        resolved = resolve_orchestrator_settings(overrides=cli_overrides)

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
            "dry_run": resolved["dry_run"],
        }

        # 自我迭代选项
        if analysis_options.get("skip_online") or self.args.skip_online:
            options["skip_online"] = True
        if analysis_options.get("force_update") or self.args.force_update:
            options["force_update"] = True

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
        options["enable_knowledge_injection"] = getattr(
            self.args, "enable_knowledge_injection", True
        )
        options["knowledge_top_k"] = getattr(self.args, "knowledge_top_k", 3)
        options["knowledge_max_chars_per_doc"] = getattr(
            self.args, "knowledge_max_chars_per_doc", 1200
        )
        options["knowledge_max_total_chars"] = getattr(
            self.args, "knowledge_max_total_chars", 3000
        )

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

        # 标记是否由 & 前缀触发
        options["triggered_by_prefix"] = analysis_options.get("triggered_by_prefix", False)

        # 角色级执行模式
        options["planner_execution_mode"] = getattr(
            self.args, "planner_execution_mode", None
        )
        options["worker_execution_mode"] = getattr(
            self.args, "worker_execution_mode", None
        )
        options["reviewer_execution_mode"] = getattr(
            self.args, "reviewer_execution_mode", None
        )

        # 流式控制台渲染配置
        options["stream_console_renderer"] = getattr(
            self.args, "stream_console_renderer", False
        )
        options["stream_advanced_renderer"] = getattr(
            self.args, "stream_advanced_renderer", False
        )
        options["stream_typing_effect"] = getattr(
            self.args, "stream_typing_effect", False
        )
        options["stream_typing_delay"] = getattr(
            self.args, "stream_typing_delay", 0.02
        )
        options["stream_word_mode"] = getattr(
            self.args, "stream_word_mode", True
        )
        options["stream_color_enabled"] = getattr(
            self.args, "stream_color_enabled", True
        )
        options["stream_show_word_diff"] = getattr(
            self.args, "stream_show_word_diff", False
        )

        # 日志控制参数
        options["quiet"] = getattr(self.args, "quiet", False)
        options["log_level"] = getattr(self.args, "log_level", None)
        options["heartbeat_debug"] = getattr(self.args, "heartbeat_debug", False)

        # 卡死诊断参数
        options["stall_diagnostics_enabled"] = getattr(
            self.args, "stall_diagnostics_enabled", None
        )
        options["stall_diagnostics_level"] = getattr(
            self.args, "stall_diagnostics_level", None
        )
        options["stall_recovery_interval"] = getattr(
            self.args, "stall_recovery_interval", 30.0
        )
        options["execution_health_check_interval"] = getattr(
            self.args, "execution_health_check_interval", 30.0
        )
        options["health_warning_cooldown"] = getattr(
            self.args, "health_warning_cooldown", 60.0
        )

        return options

    def _build_cursor_config(self, options: dict) -> "CursorAgentConfig":
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

        Args:
            options: 合并后的选项字典

        Returns:
            ExecutionMode 枚举值
        """
        mode_str = options.get("execution_mode", "cli")
        mode_map = {
            "cli": ExecutionMode.CLI,
            "auto": ExecutionMode.AUTO,
            "cloud": ExecutionMode.CLOUD,
        }
        return mode_map.get(mode_str, ExecutionMode.CLI)

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

        使用 CloudClientFactory.resolve_api_key 统一解析 API Key，
        确保配置来源优先级一致。

        优先级（从高到低）：
        1. --cloud-api-key 参数（CLI 显式指定）
        2. 环境变量 CURSOR_API_KEY
        3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        4. config.yaml 中的 cloud_agent.api_key

        其他配置（auth_timeout, base_url 等）也遵循 CLI > config.yaml > DEFAULT_* 优先级。

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

    async def _search_knowledge_docs(self, query: str) -> list[dict[str, Any]]:
        """搜索知识库文档

        Args:
            query: 搜索查询

        Returns:
            知识库文档列表，每个文档包含 title, url, content 字段
        """
        try:
            from knowledge import KnowledgeStorage

            storage = KnowledgeStorage()
            await storage.initialize()
            results = await storage.search(query, limit=5)

            knowledge_docs: list[dict[str, Any]] = []
            for result in results:
                doc = await storage.load_document(result.doc_id)
                if doc:
                    knowledge_docs.append({
                        "title": doc.title,
                        "url": doc.url,
                        "content": doc.content[:2000],
                    })
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
            # 截断过长内容
            content = doc['content'][:1000]
            if len(doc['content']) > 1000:
                content += "..."
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
        planner_exec_mode = self._parse_execution_mode(
            options.get("planner_execution_mode")
        )
        worker_exec_mode = self._parse_execution_mode(
            options.get("worker_execution_mode")
        )
        reviewer_exec_mode = self._parse_execution_mode(
            options.get("reviewer_execution_mode")
        )

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
            knowledge_docs = await self._search_knowledge_docs(search_query)
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

        # 初始化知识库
        manager = KnowledgeManager(name="cursor-docs")
        await manager.initialize()

        # 搜索相关文档
        knowledge_context = None
        if options.get("search_knowledge"):
            storage = KnowledgeStorage()
            await storage.initialize()
            results = await storage.search(options["search_knowledge"], limit=5)
            knowledge_context = []
            for result in results:
                doc = await storage.load_document(result.doc_id)
                if doc:
                    knowledge_context.append({
                        "title": doc.title,
                        "url": doc.url,
                        "content": doc.content[:2000],
                    })

        # 构建增强目标
        enhanced_goal = goal
        if knowledge_context:
            context_text = "\n\n## 参考文档（来自知识库）\n\n"
            for i, doc in enumerate(knowledge_context, 1):
                context_text += f"### {i}. {doc['title']}\n"
                context_text += f"```\n{doc['content'][:1000]}\n```\n\n"
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
        planner_exec_mode = self._parse_execution_mode(
            options.get("planner_execution_mode")
        )
        worker_exec_mode = self._parse_execution_mode(
            options.get("worker_execution_mode")
        )
        reviewer_exec_mode = self._parse_execution_mode(
            options.get("reviewer_execution_mode")
        )

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
                self.cloud_api_key = opts.get("cloud_api_key", None)
                # cloud_auth_timeout: 整数类型（与 argparse 一致）
                self.cloud_auth_timeout = opts.get("cloud_auth_timeout", DEFAULT_CLOUD_AUTH_TIMEOUT)
                # cloud_timeout: 整数类型（与 argparse 一致）
                self.cloud_timeout = opts.get("cloud_timeout", DEFAULT_CLOUD_TIMEOUT)

                # 角色级执行模式（从 _merge_options 映射）
                self.planner_execution_mode = opts.get("planner_execution_mode", None)
                self.worker_execution_mode = opts.get("worker_execution_mode", None)
                self.reviewer_execution_mode = opts.get("reviewer_execution_mode", None)

                # 日志控制参数
                self.quiet = opts.get("quiet", False)
                self.log_level = opts.get("log_level", None)
                self.heartbeat_debug = opts.get("heartbeat_debug", False)

                # 卡死诊断参数
                self.stall_diagnostics_enabled = opts.get("stall_diagnostics_enabled", None)
                self.stall_diagnostics_level = opts.get("stall_diagnostics_level", None)
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
        from cursor.executor import PlanAgentExecutor
        from cursor.client import CursorAgentConfig

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

                return {
                    "success": True,
                    "goal": goal,
                    "mode": "plan",
                    "plan": plan_output,
                    "dry_run": True,
                }
            else:
                error_msg = result.error or "未知错误"
                print_error(f"规划失败: {error_msg}")
                return {
                    "success": False,
                    "goal": goal,
                    "mode": "plan",
                    "error": error_msg,
                }

        except asyncio.TimeoutError:
            print_error("规划超时")
            return {"success": False, "goal": goal, "mode": "plan", "error": "timeout"}
        except Exception as e:
            print_error(f"规划异常: {e}")
            return {"success": False, "goal": goal, "mode": "plan", "error": str(e)}

    async def _run_ask(self, goal: str, options: dict) -> dict:
        """运行问答模式（直接对话）

        使用 AskAgentExecutor 确保:
        - mode=ask（问答模式）
        - force_write=False（只读保证，不修改文件）
        """
        from cursor.executor import AskAgentExecutor
        from cursor.client import CursorAgentConfig

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

                return {
                    "success": True,
                    "goal": goal,
                    "mode": "ask",
                    "answer": answer,
                }
            else:
                error_msg = result.error or "未知错误"
                print_error(f"问答失败: {error_msg}")
                return {
                    "success": False,
                    "goal": goal,
                    "mode": "ask",
                    "error": error_msg,
                }

        except asyncio.TimeoutError:
            print_error("问答超时")
            return {"success": False, "goal": goal, "mode": "ask", "error": "timeout"}
        except Exception as e:
            print_error(f"问答异常: {e}")
            return {"success": False, "goal": goal, "mode": "ask", "error": str(e)}

    async def _run_cloud(self, goal: str, options: dict) -> dict:
        """运行 Cloud 模式（云端执行）

        使用 CloudAgentExecutor 执行任务，特点:
        - 支持前台/后台两种执行模式
        - 前台模式（background=False）：等待任务完成，返回完整结果
        - 后台模式（background=True）：提交任务后立即返回，不等待完成
        - 权限语义：force_write 由用户显式控制，不受 auto_commit 影响

        后台模式行为（cloud_background=True）：
        - & 前缀触发时默认使用后台模式（Cloud Relay 语义）
        - 提交任务后立即返回 session_id
        - 不等待任务完成，不打印大段输出
        - 用户可通过 agent --resume <session_id> 恢复会话查看结果

        前台模式行为（cloud_background=False）：
        - --execution-mode cloud 显式指定时默认使用前台模式
        - 等待任务完成并返回完整结果
        - 适合脚本/自动化场景，需要立即获取执行结果

        权限策略:
        - Cloud 模式默认 force_write=True（允许修改文件）
        - auto_commit 独立于 force_write，需用户显式启用

        超时策略:
        - 使用独立的 --cloud-timeout 参数（默认 300 秒）
        - 不从 max_iterations 推导，避免 -1 等特殊值导致问题
        """
        from cursor.executor import AgentExecutorFactory, ExecutionMode
        from cursor.client import CursorAgentConfig

        # 获取后台模式配置
        # 默认行为：& 前缀触发默认 True，--execution-mode cloud 默认 False
        cloud_background = options.get("cloud_background", False)
        triggered_by_prefix = options.get("triggered_by_prefix", False)

        if cloud_background:
            print_info("Cloud 后台模式：任务将提交到云端后台执行...")
        else:
            print_info("Cloud 前台模式：任务将在云端执行并等待完成...")

        # 复用已有的 Cloud 认证配置获取逻辑
        # 优先级：--cloud-api-key > CURSOR_API_KEY 环境变量
        cloud_auth_config = self._get_cloud_auth_config(options)

        # 检查 API key 是否已配置
        if cloud_auth_config is None:
            error_msg = (
                "未配置 Cloud API Key。请通过以下方式之一配置:\n"
                "  1. 命令行参数: --cloud-api-key YOUR_KEY\n"
                "  2. 环境变量: export CURSOR_API_KEY=YOUR_KEY\n"
                "  3. 运行 'agent login' 进行登录认证"
            )
            print_error(error_msg)
            return {
                "success": False,
                "goal": goal,
                "mode": "cloud",
                "background": cloud_background,
                "error": "未配置 API Key",
                "session_id": None,
                "resume_command": None,
            }

        try:
            # Cloud 模式下 force_write 默认为 True（允许修改文件）
            # 但与 auto_commit 独立，auto_commit 仍需用户显式启用
            force_write = options.get("force_write", True)

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

            # 执行任务，传递 background 参数
            result = await executor.execute(
                prompt=goal,
                working_directory=options.get("directory", str(project_root)),
                timeout=cloud_timeout,
                background=cloud_background,
            )

            session_id = result.session_id
            resume_command = f"agent --resume {session_id}" if session_id else None

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
                    print(output[:2000] if len(output) > 2000 else output)
                    if len(output) > 2000:
                        print("\n... (输出已截断)")
                    print("=" * 60)

                    # 输出 session_id 使用说明
                    if session_id:
                        print(f"\n{Colors.CYAN}会话 ID: {session_id}{Colors.NC}")
                        print(f"{Colors.CYAN}恢复会话: {resume_command}{Colors.NC}\n")

                return {
                    "success": True,
                    "goal": goal,
                    "mode": "cloud",
                    "background": cloud_background,
                    "output": output if not cloud_background else "",
                    "session_id": session_id,
                    "resume_command": resume_command,
                    "files_modified": result.files_modified,
                    # 额外元数据
                    "triggered_by_prefix": triggered_by_prefix,
                }
            else:
                error_msg = result.error or "未知错误"
                print_error(f"Cloud 执行失败: {error_msg}")
                return {
                    "success": False,
                    "goal": goal,
                    "mode": "cloud",
                    "background": cloud_background,
                    "error": error_msg,
                    "session_id": session_id,
                    "resume_command": resume_command,
                }

        except asyncio.TimeoutError:
            cloud_timeout = options.get("cloud_timeout", 300)
            print_error(f"Cloud 执行超时 ({cloud_timeout}s)")
            return {
                "success": False,
                "goal": goal,
                "mode": "cloud",
                "background": cloud_background,
                "error": f"执行超时 ({cloud_timeout}s)",
                "session_id": None,
                "resume_command": None,
            }
        except Exception as e:
            print_error(f"Cloud 执行异常: {e}")
            return {
                "success": False,
                "goal": goal,
                "mode": "cloud",
                "background": cloud_background,
                "error": str(e),
                "session_id": None,
                "resume_command": None,
            }


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
        goal_preview = result["goal"][:100]
        if len(result["goal"]) > 100:
            goal_preview += "..."
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

    print("=" * 60)


# ============================================================
# 日志配置
# ============================================================

def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        filter=lambda record: record["level"].name != "DEBUG" or verbose,
    )

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

    Args:
        args: 命令行参数

    Returns:
        CLI overrides 字典
    """
    cli_overrides = {}

    # workers: CLI 显式参数
    if args.workers is not None:
        cli_overrides["workers"] = args.workers

    # max_iterations: CLI 显式参数
    if args.max_iterations is not None:
        cli_overrides["max_iterations"] = parse_max_iterations(args.max_iterations)

    # strict_review: CLI 显式参数
    cli_strict = getattr(args, "strict_review", None)
    if cli_strict is not None:
        cli_overrides["strict_review"] = cli_strict

    # enable_sub_planners: CLI 显式参数
    cli_sub_planners = getattr(args, "enable_sub_planners", None)
    if cli_sub_planners is not None:
        cli_overrides["enable_sub_planners"] = cli_sub_planners

    # execution_mode: CLI 显式参数
    cli_execution_mode = getattr(args, "execution_mode", None)
    if cli_execution_mode is not None:
        cli_overrides["execution_mode"] = cli_execution_mode

    # cloud_timeout: CLI 显式参数
    cli_cloud_timeout = getattr(args, "cloud_timeout", None)
    if cli_cloud_timeout is not None:
        cli_overrides["cloud_timeout"] = cli_cloud_timeout

    # cloud_auth_timeout: CLI 显式参数
    cli_cloud_auth_timeout = getattr(args, "cloud_auth_timeout", None)
    if cli_cloud_auth_timeout is not None:
        cli_overrides["cloud_auth_timeout"] = cli_cloud_auth_timeout

    # dry_run: CLI 显式参数
    if args.dry_run:
        cli_overrides["dry_run"] = True

    # auto_commit/auto_push: CLI 显式参数
    cli_auto_commit = getattr(args, "auto_commit", False)
    cli_auto_push = getattr(args, "auto_push", False)
    cli_commit_per_iteration = getattr(args, "commit_per_iteration", False)
    if cli_auto_commit:
        cli_overrides["auto_commit"] = True
    if cli_auto_push:
        cli_overrides["auto_push"] = True
    if cli_commit_per_iteration:
        cli_overrides["commit_per_iteration"] = True

    # 编排器配置
    cli_orchestrator = getattr(args, "orchestrator", None)
    cli_no_mp = getattr(args, "no_mp", None)
    if cli_orchestrator is not None:
        cli_overrides["orchestrator"] = cli_orchestrator
    elif cli_no_mp is True:
        cli_overrides["orchestrator"] = "basic"

    # 模型配置: CLI 显式参数（None 表示未指定）
    if args.planner_model is not None:
        cli_overrides["planner_model"] = args.planner_model
    if args.worker_model is not None:
        cli_overrides["worker_model"] = args.worker_model
    if args.reviewer_model is not None:
        cli_overrides["reviewer_model"] = args.reviewer_model

    return cli_overrides


async def async_main() -> int:
    """异步主函数"""
    args = parse_args()
    setup_logging(args.verbose)

    # 处理 --print-config 参数：打印配置调试信息并退出
    if getattr(args, "print_config", False):
        cli_overrides = _build_cli_overrides_from_args(args)
        print_debug_config(cli_overrides, source_label="run.py")
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
        print_info("用法: python run.py [--mode MODE] \"任务描述\"")
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
