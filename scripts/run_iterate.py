#!/usr/bin/env python3
"""自我迭代脚本

实现完整的自我迭代工作流：
用户输入需求 → 分析在线文档更新 → 更新知识库 → 总结迭代内容 → 启动 Agent 执行

用法:
    # 完整自我迭代（检查在线更新 + 用户需求）
    python scripts/run_iterate.py "增加对新斜杠命令的支持"

    # 仅基于知识库迭代（跳过在线检查）
    python scripts/run_iterate.py --skip-online "优化 CLI 参数处理"

    # 纯自动模式（无额外需求，仅检查更新）
    python scripts/run_iterate.py

    # 指定 changelog URL
    python scripts/run_iterate.py --changelog-url "https://cursor.com/cn/changelog"

    # 仅分析不执行
    python scripts/run_iterate.py --dry-run "分析改进点"
"""
import argparse
import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from agents.committer import CommitterAgent, CommitterConfig
from core.cloud_utils import CLOUD_PREFIX, is_cloud_request, strip_cloud_prefix
from coordinator import (
    MultiProcessOrchestrator,
    MultiProcessOrchestratorConfig,
    Orchestrator,
    OrchestratorConfig,
)
from cursor.client import CursorAgentConfig
from cursor.cloud_client import CloudAuthConfig
from cursor.executor import ExecutionMode
from knowledge import (
    Document,
    FetchConfig,
    KnowledgeManager,
    KnowledgeStorage,
    WebFetcher,
)

# ============================================================
# 配置常量
# ============================================================

# 默认 Changelog URL
DEFAULT_CHANGELOG_URL = "https://cursor.com/cn/changelog"

# 相关文档 URL（用于补充更新）
CURSOR_DOC_URLS = [
    "https://cursor.com/cn/docs/cli/overview",
    "https://cursor.com/cn/docs/cli/using",
    "https://cursor.com/cn/docs/cli/reference/parameters",
    "https://cursor.com/cn/docs/cli/reference/slash-commands",
    "https://cursor.com/cn/docs/cli/mcp",
    "https://cursor.com/cn/docs/cli/hooks",
    "https://cursor.com/cn/docs/cli/subagents",
    "https://cursor.com/cn/docs/cli/skills",
    # Jan 16 2026: plan/ask 模式专页
    "https://cursor.com/cn/docs/cli/modes/plan",
    "https://cursor.com/cn/docs/cli/modes/ask",
    # Jan 20 2026: 审阅与代码评审
    "https://cursor.com/cn/docs/agent/review",
    "https://cursor.com/cn/docs/cli/cookbook/code-review",
    # Jan 20 2026: Cloud Agent（云代理能力）
    "https://cursor.com/cn/docs/cloud-agent",
    "https://cursor.com/cn/docs/cloud-agent/overview",
    "https://cursor.com/cn/docs/cloud-agent/getting-started",
    "https://cursor.com/cn/docs/cloud-agent/api",
    "https://cursor.com/cn/docs/cloud-agent/api/streaming",
    "https://cursor.com/cn/docs/cloud-agent/api/sessions",
]

# 更新关键词模式（用于识别 Changelog 中的更新点）
UPDATE_KEYWORDS = [
    r"新增|新功能|新特性|new feature",
    r"改进|优化|改善|improved",
    r"修复|fix|bugfix",
    r"更新|update|updated",
    r"支持|support",
    r"弃用|deprecated",
    r"移除|removed",
]

# 知识库名称
KB_NAME = "cursor-docs"

# 禁用多进程编排器的关键词（在 requirement 中检测）
DISABLE_MP_KEYWORDS = [
    "非并行", "不并行", "串行", "协程",
    "单进程", "basic", "no-mp", "no_mp",
    "禁用多进程", "禁用mp", "关闭多进程",
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
    MAGENTA = "\033[0;35m"
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


def print_step(step: int, text: str):
    print(f"\n{Colors.MAGENTA}[步骤 {step}]{Colors.NC} {Colors.BOLD}{text}{Colors.NC}")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ChangelogEntry:
    """Changelog 条目"""
    date: str = ""
    version: str = ""
    title: str = ""
    content: str = ""
    category: str = ""  # feature, fix, improvement, etc.
    keywords: list[str] = field(default_factory=list)


@dataclass
class UpdateAnalysis:
    """更新分析结果"""
    has_updates: bool = False
    entries: list[ChangelogEntry] = field(default_factory=list)
    new_features: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    related_doc_urls: list[str] = field(default_factory=list)
    summary: str = ""
    raw_content: str = ""


@dataclass
class IterationContext:
    """迭代上下文"""
    user_requirement: str = ""
    update_analysis: Optional[UpdateAnalysis] = None
    knowledge_context: list[dict] = field(default_factory=list)
    iteration_goal: str = ""
    dry_run: bool = False


# ============================================================
# 参数解析
# ============================================================

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


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="自我迭代脚本 - 分析在线文档更新、更新知识库、启动 Agent 执行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整自我迭代（检查在线更新 + 用户需求）
  python scripts/run_iterate.py "增加对新斜杠命令的支持"

  # 仅基于知识库迭代（跳过在线检查）
  python scripts/run_iterate.py --skip-online "优化 CLI 参数处理"

  # 纯自动模式（无额外需求，仅检查更新）
  python scripts/run_iterate.py

  # 仅分析不执行
  python scripts/run_iterate.py --dry-run "分析改进点"
        """,
    )

    parser.add_argument(
        "requirement",
        nargs="?",
        default="",
        help="额外需求（可选）",
    )

    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="跳过在线文档检查，仅基于现有知识库迭代",
    )

    parser.add_argument(
        "--changelog-url",
        type=str,
        default=DEFAULT_CHANGELOG_URL,
        help=f"Changelog URL (默认: {DEFAULT_CHANGELOG_URL})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅分析不执行，显示将要执行的迭代计划",
    )

    parser.add_argument(
        "--max-iterations",
        type=str,
        default="5",
        help="最大迭代次数 (默认: 5，使用 MAX 或 -1 表示无限迭代直到完成或用户中断)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Worker 池大小 (默认: 3)",
    )

    parser.add_argument(
        "--force-update",
        action="store_true",
        help="强制更新知识库（即使内容未变化）",
    )

    # 日志控制参数组
    log_group = parser.add_argument_group("日志控制")
    log_verbosity = log_group.add_mutually_exclusive_group()
    log_verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出（DEBUG 级别日志）",
    )
    log_verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式（仅 WARNING 及以上日志，默认行为已优化为较少日志）",
    )
    log_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别（默认: INFO，--verbose 时为 DEBUG，--quiet 时为 WARNING）",
    )
    log_group.add_argument(
        "--heartbeat-debug",
        action="store_true",
        help="启用心跳调试日志（仅调试时使用，默认关闭以减少日志输出）",
    )

    # 卡死诊断参数组
    # 设计目的：默认简洁输出避免刷屏，quiet 模式下降级到 DEBUG，verbose 模式下可启用更多诊断
    stall_group = parser.add_argument_group("卡死诊断", "控制卡死检测和恢复的诊断输出")
    stall_diag_toggle = stall_group.add_mutually_exclusive_group()
    stall_diag_toggle.add_argument(
        "--stall-diagnostics",
        action="store_true",
        dest="stall_diagnostics_enabled",
        default=None,
        help="启用卡死诊断日志（默认关闭，疑似卡死时再启用以排查问题）",
    )
    stall_diag_toggle.add_argument(
        "--no-stall-diagnostics",
        action="store_false",
        dest="stall_diagnostics_enabled",
        help="禁用卡死诊断日志（完全关闭摘要诊断输出）",
    )
    stall_group.add_argument(
        "--stall-diagnostics-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="卡死诊断日志级别（默认: warning，quiet 模式下为 debug 以减少输出）",
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
        help="执行阶段健康检查间隔（秒，默认: 30.0，不宜过低以避免频繁检查）",
    )
    stall_group.add_argument(
        "--health-warning-cooldown",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="健康检查告警冷却时间（秒，同一 agent 在此时间内不重复告警，默认: 60.0）",
    )

    # 自动提交相关参数
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="迭代完成后自动提交代码更改",
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

    parser.add_argument(
        "--commit-per-iteration",
        action="store_true",
        help="每次迭代都提交（默认仅在全部完成时提交）",
    )

    # 编排器选择参数
    orchestrator_group = parser.add_mutually_exclusive_group()
    orchestrator_group.add_argument(
        "--orchestrator",
        type=str,
        choices=["mp", "basic"],
        default="mp",
        help="编排器类型: mp=多进程(默认), basic=协程模式",
    )
    orchestrator_group.add_argument(
        "--no-mp",
        action="store_true",
        help="禁用多进程编排器，使用基本协程编排器（等同于 --orchestrator basic）",
    )

    # 执行模式参数
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default="cli",
        help="执行模式: cli=本地CLI(默认), auto=自动选择(Cloud优先), cloud=强制Cloud",
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

    # Cloud 认证配置
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="Cloud API Key（可选，默认从 CURSOR_API_KEY 环境变量读取）",
    )

    parser.add_argument(
        "--cloud-auth-timeout",
        type=int,
        default=30,
        help="Cloud 认证超时时间（秒，默认 30）",
    )

    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=600,
        help="Cloud 执行超时时间（秒，默认 600，即 10 分钟）",
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

    args = parser.parse_args()

    # 检测用户是否显式设置了编排器参数
    # 通过检查 sys.argv 中是否存在相关参数来判断
    args._orchestrator_user_set = any(
        arg in sys.argv for arg in [
            "--orchestrator", "--no-mp", "-no-mp",
        ]
    )

    return args


# ============================================================
# 在线文档分析
# ============================================================

class ChangelogAnalyzer:
    """Changelog 分析器

    从 Cursor Changelog 页面获取更新内容并分析。
    支持多种解析策略：优先日期标题块、备用月份标题块、保底全页单条。
    支持基线比较：通过 fingerprint 检测内容是否真正更新。
    """

    def __init__(
        self,
        changelog_url: str = DEFAULT_CHANGELOG_URL,
        storage: Optional[KnowledgeStorage] = None,
    ):
        self.changelog_url = changelog_url
        self.fetcher = WebFetcher(FetchConfig(timeout=60))
        # 用于基线读取的存储实例（可选）
        self._storage = storage

    async def fetch_changelog(self) -> Optional[str]:
        """获取 Changelog 内容

        Returns:
            Changelog 页面内容，失败返回 None
        """
        print_info(f"获取 Changelog: {self.changelog_url}")

        await self.fetcher.initialize()
        result = await self.fetcher.fetch(self.changelog_url)

        if result.success:
            print_success(f"Changelog 获取成功 ({len(result.content)} 字符)")
            return result.content
        else:
            print_error(f"Changelog 获取失败: {result.error}")
            return None

    def _clean_content(self, content: str) -> str:
        """清理 HTML/Markdown 混合内容

        移除 HTML 标签、多余空白等，保留纯文本和 Markdown 格式。
        支持主内容区域截取：当检测到明确的内容锚点时，截取主内容区域以降低噪声。

        Args:
            content: 原始内容

        Returns:
            清理后的内容
        """
        if not content:
            return ""

        # 1. 尝试截取主内容区域（降低噪声干扰）
        content = self._extract_main_content(content)

        # 2. 移除 HTML 注释
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # 3. 移除 script 和 style 标签及内容
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # 4. 移除 nav、header、footer 等导航元素
        content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # 5. 移除常见 HTML 标签，保留内容
        # 保留 <a> 标签的文本
        content = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', content, flags=re.IGNORECASE)
        # 移除其他标签
        content = re.sub(r'<[^>]+>', '', content)

        # 6. 解码常见 HTML 实体
        html_entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '...',
            '&copy;': '©',
            '&reg;': '®',
            '&trade;': '™',
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)

        # 7. 移除常见的网页噪声模式
        # 移除 "Skip to content"、"Back to top" 等导航文本
        content = re.sub(
            r'\b(?:Skip to (?:content|main)|Back to top|Jump to|'
            r'Table of Contents|Navigation|Menu|Search)\b[^\n]*',
            '', content, flags=re.IGNORECASE
        )

        # 移除版权信息行
        content = re.sub(
            r'^.*(?:©|Copyright|All [Rr]ights [Rr]eserved).*$',
            '', content, flags=re.MULTILINE
        )

        # 8. 移除多余空行（保留最多2个连续空行）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 9. 移除行首行尾空白
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)

        return content.strip()

    def _extract_main_content(self, content: str) -> str:
        """尝试从页面中提取主内容区域

        使用多种策略尝试定位主内容区域：
        1. 检测 <main>、<article> 等语义标签
        2. 检测常见的内容容器 class/id（如 changelog、content、main-content）
        3. 检测 Markdown 标题锚点（如 # Changelog、## Updates）
        4. 如果无法定位，返回原始内容

        Args:
            content: 原始 HTML/Markdown 内容

        Returns:
            提取出的主内容区域，或原始内容（如无法定位）
        """
        # 策略1: 提取 <main> 标签内容
        main_match = re.search(
            r'<main[^>]*>(.*?)</main>',
            content, flags=re.DOTALL | re.IGNORECASE
        )
        if main_match:
            logger.debug("使用 <main> 标签提取主内容")
            return main_match.group(1)

        # 策略2: 提取 <article> 标签内容
        article_match = re.search(
            r'<article[^>]*>(.*?)</article>',
            content, flags=re.DOTALL | re.IGNORECASE
        )
        if article_match:
            logger.debug("使用 <article> 标签提取主内容")
            return article_match.group(1)

        # 策略3: 提取带有 changelog/content 相关 class/id 的 div
        # 匹配 class="changelog" 或 id="main-content" 等
        content_div_match = re.search(
            r'<div[^>]*(?:class|id)=["\'][^"\']*'
            r'(?:changelog|main-content|content-area|page-content|docs-content)'
            r'[^"\']*["\'][^>]*>(.*?)</div>',
            content, flags=re.DOTALL | re.IGNORECASE
        )
        if content_div_match:
            logger.debug("使用内容容器 div 提取主内容")
            return content_div_match.group(1)

        # 策略4: 检测 Markdown 标题锚点
        # 查找第一个 changelog 相关的标题
        anchor_patterns = [
            # "# Changelog" 或 "# 更新日志"
            r'(#{1,2}\s*(?:Changelog|Updates?|更新日志|版本历史|What\'s New).*)',
            # 日期格式标题（如 "## Jan 16, 2026" 或 "## 2026-01-16"）
            r'(#{1,2}\s*(?:\d{4}[-/]\d{2}[-/]\d{2}|'
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s+\d{1,2}[,\s]+\d{4}).*)',
        ]

        for pattern in anchor_patterns:
            anchor_match = re.search(pattern, content, flags=re.IGNORECASE)
            if anchor_match:
                # 从锚点位置开始截取到文档末尾
                start_pos = anchor_match.start()
                extracted = content[start_pos:]
                # 如果提取的内容足够长（至少100字符），使用它
                if len(extracted) >= 100:
                    logger.debug(f"使用 Markdown 锚点提取主内容，起始: {anchor_match.group()[:50]}...")
                    return extracted

        # 策略5: 如果无法定位主内容区域，返回原始内容
        return content

    def _parse_by_date_headers(self, content: str) -> list[ChangelogEntry]:
        """策略1: 按日期标题块解析（优先）

        匹配格式如：## 2024-01-15、### Jan 16, 2026 等

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 匹配多种日期格式：
        # - ## 2024-01-15
        # - ### 2024/01/15
        # - ## Jan 16, 2026
        # - ## January 16 2026
        date_pattern = r'\n(?=#{1,3}\s*(?:' \
            r'\d{4}[-/]\d{2}[-/]\d{2}|' \
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}[,\s]+\d{4}' \
            r'))'

        sections = re.split(date_pattern, content, flags=re.IGNORECASE)

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_section(section)
            if entry and (entry.date or entry.content):
                entries.append(entry)

        return entries

    def _parse_by_category_date_headers(self, content: str) -> list[ChangelogEntry]:
        """策略2: 按类别+日期标题块解析

        匹配格式如：CLI Jan 16, 2026、Agent Dec 10, 2025、
        Feature January 20 2026 等 "<Category> <Month Day, Year>" 格式。

        这种格式常见于按产品或功能分类的 changelog，每个分类下
        有独立的日期标识。

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 匹配 "<Category> <Month Day, Year>" 格式
        # 例如: CLI Jan 16, 2026 或 Agent December 20 2025
        # Category: 1-3个单词（字母/数字/连字符），日期：月份 日, 年
        category_date_pattern = (
            r'\n(?='
            r'(?:#{1,3}\s*)?'  # 可选的 markdown 标题符号
            r'(?:[\w-]+(?:\s+[\w-]+){0,2})\s+'  # 类别：1-3个单词
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s+\d{1,2}[,\s]+\d{4}'  # 日期：月 日, 年
            r')'
        )

        sections = re.split(category_date_pattern, content, flags=re.IGNORECASE)

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_category_date_section(section)
            if entry and (entry.date or entry.content):
                entries.append(entry)

        return entries

    def _parse_category_date_section(self, section: str) -> Optional[ChangelogEntry]:
        """解析类别+日期格式的 section

        Args:
            section: 单个 section 的内容

        Returns:
            解析出的 ChangelogEntry，解析失败返回 None
        """
        if not section.strip():
            return None

        entry = ChangelogEntry()
        lines = section.strip().split('\n')

        if lines:
            title_line = lines[0].strip()

            # 移除可能的 markdown 标题符号
            title_line = re.sub(r'^#+\s*', '', title_line)

            # 尝试提取类别和日期
            # 格式: <Category> <Month Day, Year>
            category_date_match = re.match(
                r'^([\w-]+(?:\s+[\w-]+){0,2})\s+'
                r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'\s+\d{1,2}[,\s]+\d{4})',
                title_line, re.IGNORECASE
            )

            if category_date_match:
                category_part = category_date_match.group(1).strip()
                date_part = category_date_match.group(2).strip()

                entry.category = self._normalize_category(category_part)
                entry.date = date_part
                entry.title = title_line

        # 提取内容
        entry.content = '\n'.join(lines[1:]).strip()

        # 如果没有从标题行获取到 category，尝试从内容分析
        if not entry.category or entry.category == 'other':
            entry.category = self._categorize_content(entry.content)

        # 提取关键词
        for pattern in UPDATE_KEYWORDS:
            matches = re.findall(pattern, entry.content, re.IGNORECASE)
            entry.keywords.extend(matches)

        return entry if entry.content or entry.date else None

    def _normalize_category(self, category_str: str) -> str:
        """标准化类别字符串

        将类别字符串映射到标准分类（feature, fix, improvement, other）。

        Args:
            category_str: 原始类别字符串（如 CLI, Agent, Feature 等）

        Returns:
            标准化的类别: feature, fix, improvement, other
        """
        cat_lower = category_str.lower()

        # 特性相关
        if any(kw in cat_lower for kw in ['feature', 'new', 'add', 'launch']):
            return 'feature'

        # 修复相关
        if any(kw in cat_lower for kw in ['fix', 'bug', 'patch', 'hotfix']):
            return 'fix'

        # 改进相关
        if any(kw in cat_lower for kw in ['improve', 'enhance', 'update', 'optim']):
            return 'improvement'

        # CLI/Agent 等产品类别，默认为 feature
        if any(kw in cat_lower for kw in ['cli', 'agent', 'cloud', 'mcp', 'hook']):
            return 'feature'

        return 'other'

    def _parse_by_version_headers(self, content: str) -> list[ChangelogEntry]:
        """策略3: 按版本标题块解析（备用）

        匹配格式如：## v1.0.0、### Version 2.1

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 匹配版本号格式
        version_pattern = r'\n(?=#{1,3}\s*(?:v(?:ersion)?\s*)?\d+\.\d+)'

        sections = re.split(version_pattern, content, flags=re.IGNORECASE)

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_section(section)
            if entry and (entry.version or entry.content):
                entries.append(entry)

        return entries

    def _parse_fallback(self, content: str) -> list[ChangelogEntry]:
        """策略3: 保底策略，将全页作为单条 entry

        当其他策略都无法解析出有效条目时使用。

        Args:
            content: 清理后的内容

        Returns:
            包含单条 entry 的列表
        """
        if not content.strip():
            return []

        entry = ChangelogEntry()
        entry.title = "Changelog"
        entry.content = content.strip()
        entry.category = self._categorize_content(entry.content)

        # 尝试从内容中提取日期
        date_match = re.search(
            r'(\d{4}[-/]\d{2}[-/]\d{2})|'
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s+\d{1,2}[,\s]+\d{4})',
            content, re.IGNORECASE
        )
        if date_match:
            entry.date = date_match.group(0)

        # 提取关键词
        for pattern in UPDATE_KEYWORDS:
            matches = re.findall(pattern, entry.content, re.IGNORECASE)
            entry.keywords.extend(matches)

        return [entry]

    def _parse_section(self, section: str) -> Optional[ChangelogEntry]:
        """解析单个 section 为 ChangelogEntry

        Args:
            section: 单个 section 的内容

        Returns:
            解析出的 ChangelogEntry，解析失败返回 None
        """
        if not section.strip():
            return None

        entry = ChangelogEntry()
        lines = section.strip().split('\n')

        # 提取标题行
        if lines:
            title_line = lines[0].strip()

            # 尝试提取日期（多种格式）
            # 格式1: 2024-01-15 或 2024/01/15
            date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', title_line)
            if date_match:
                entry.date = date_match.group(1)
            else:
                # 格式2: Jan 16, 2026 或 January 16 2026
                date_match = re.search(
                    r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                    r'\s+\d{1,2}[,\s]+\d{4})',
                    title_line, re.IGNORECASE
                )
                if date_match:
                    entry.date = date_match.group(1)

            # 尝试提取版本
            version_match = re.search(r'v?(\d+\.\d+(?:\.\d+)?)', title_line)
            if version_match:
                entry.version = version_match.group(1)

            entry.title = re.sub(r'^#+\s*', '', title_line)

        # 提取内容
        entry.content = '\n'.join(lines[1:]).strip()

        # 分析类别
        entry.category = self._categorize_content(entry.content)

        # 提取关键词
        for pattern in UPDATE_KEYWORDS:
            matches = re.findall(pattern, entry.content, re.IGNORECASE)
            entry.keywords.extend(matches)

        return entry if entry.content else None

    def _categorize_content(self, content: str) -> str:
        """根据内容分析类别

        Args:
            content: 条目内容

        Returns:
            类别字符串: feature, fix, improvement, other
        """
        content_lower = content.lower()

        # 检测新功能
        if any(kw in content_lower for kw in [
            '新增', '新功能', 'new feature', 'new:', 'added', '添加',
            'introducing', 'launch', '发布', '支持'
        ]):
            return 'feature'

        # 检测修复
        if any(kw in content_lower for kw in [
            '修复', 'fix', 'bug', 'fixed', 'resolved', '解决', 'patch'
        ]):
            return 'fix'

        # 检测改进
        if any(kw in content_lower for kw in [
            '改进', '优化', 'improve', 'enhance', 'update', 'better',
            '更新', '升级', 'upgrade', 'performance'
        ]):
            return 'improvement'

        return 'other'

    def parse_changelog(self, content: str) -> list[ChangelogEntry]:
        """解析 Changelog 内容

        使用分层解析策略：
        1. 优先按日期标题块解析
        2. 按类别+日期标题块解析（如 "CLI Jan 16, 2026"）
        3. 备用按版本标题块解析
        4. 保底将全页作为单条 entry

        Args:
            content: Changelog 页面内容

        Returns:
            Changelog 条目列表
        """
        if not content:
            return []

        # 清理 HTML/Markdown 混合内容
        cleaned_content = self._clean_content(content)

        if not cleaned_content:
            return []

        # 策略1: 按日期标题块解析
        entries = self._parse_by_date_headers(cleaned_content)
        if entries:
            logger.debug(f"使用日期标题策略解析，得到 {len(entries)} 条")
            return entries

        # 策略2: 按类别+日期标题块解析（如 "CLI Jan 16, 2026"）
        entries = self._parse_by_category_date_headers(cleaned_content)
        if entries:
            logger.debug(f"使用类别+日期标题策略解析，得到 {len(entries)} 条")
            return entries

        # 策略3: 按版本标题块解析
        entries = self._parse_by_version_headers(cleaned_content)
        if entries:
            logger.debug(f"使用版本标题策略解析，得到 {len(entries)} 条")
            return entries

        # 策略4: 保底全页单条
        logger.debug("使用保底策略，全页作为单条 entry")
        return self._parse_fallback(cleaned_content)

    def extract_update_points(self, entries: list[ChangelogEntry]) -> UpdateAnalysis:
        """提取更新要点

        Args:
            entries: Changelog 条目列表

        Returns:
            更新分析结果
        """
        analysis = UpdateAnalysis()
        analysis.entries = entries
        analysis.has_updates = len(entries) > 0

        for entry in entries:
            # 分类
            if entry.category == 'feature':
                analysis.new_features.append(f"[{entry.date or entry.version}] {entry.title}")
            elif entry.category == 'fix':
                analysis.fixes.append(f"[{entry.date or entry.version}] {entry.title}")
            elif entry.category == 'improvement':
                analysis.improvements.append(f"[{entry.date or entry.version}] {entry.title}")

            # 检测相关文档 URL
            for doc_url in CURSOR_DOC_URLS:
                # 从 entry 内容中匹配关键词
                doc_keywords = self._extract_doc_keywords(doc_url)
                if any(kw.lower() in entry.content.lower() for kw in doc_keywords):
                    if doc_url not in analysis.related_doc_urls:
                        analysis.related_doc_urls.append(doc_url)

        # 生成摘要
        analysis.summary = self._generate_summary(analysis)

        return analysis

    def _extract_doc_keywords(self, url: str) -> list[str]:
        """从 URL 提取关键词

        关键词映射覆盖 Cursor CLI 主要特性，包括 Jan 16 2026 新增的
        plan/ask 模式、cloud relay、diff 视图等功能。
        """
        # 从 URL 路径提取关键词
        path = url.split('/')[-1]
        keywords = [path.replace('-', ' ')]

        # 提取完整路径用于多级路径匹配（如 modes/plan, modes/ask）
        full_path = '/'.join(url.split('/')[4:])  # 提取 docs/cli/ 之后的部分

        # 添加特定关键词映射
        # 包含 Jan 16 2026 新特性: plan/ask 模式、cloud relay、diff
        # 包含 Jan 20 2026 新特性: agent review、code-review cookbook、cloud-agent
        keyword_map = {
            # 基础参数文档
            'parameters': [
                '参数', 'parameter', 'option', '选项',
                # Jan 16 2026: plan/ask 模式
                'plan', 'ask', '--mode', 'mode', '模式',
                '规划模式', '问答模式', '代理模式',
                # 输出格式
                'output-format', 'stream-json', 'json',
            ],
            # 斜杠命令
            'slash-commands': [
                '斜杠命令', 'slash', '/',
                # Jan 16 2026: /plan 和 /ask 命令
                '/plan', '/ask', '/model', '/models',
                '/rules', '/commands', '/mcp',
            ],
            # MCP 服务器
            'mcp': [
                'mcp', '服务器', 'server',
                # Jan 16 2026: cloud relay
                'cloud relay', 'relay', '云中继',
                'mcp enable', 'mcp disable', 'mcp list',
            ],
            # Hooks
            'hooks': [
                'hook', '钩子',
                'beforeShellExecution', 'afterShellExecution',
                'beforeFileEdit', 'afterFileEdit',
            ],
            # 子代理
            'subagents': [
                'subagent', '子代理', 'agent',
                'foreground', 'background', '前台', '后台',
            ],
            # 技能
            'skills': [
                'skill', '技能', 'SKILL.md',
            ],
            # CLI 概览
            'overview': [
                'cli', '命令行', 'agent',
                # Jan 16 2026: diff 视图
                'diff', '差异', 'changes', '变更',
                'review', '审阅', 'Ctrl+R',
            ],
            # 使用指南
            'using': [
                'using', '使用', '教程',
                # Jan 16 2026: 交互模式增强
                'interactive', '交互', '快捷键',
                'diff view', 'diff 视图',
            ],
            # Jan 16 2026: plan 模式专页
            'modes/plan': [
                'plan', 'plan mode', '规划模式', '--mode plan',
                '规划', 'planner', '任务规划', '只读模式',
                '分析', 'analyze', 'readonly',
            ],
            # Jan 16 2026: ask 模式专页
            'modes/ask': [
                'ask', 'ask mode', '问答模式', '--mode ask',
                '问答', '咨询', 'question', 'query',
                '只读', 'readonly', '解释',
            ],
            # Jan 20 2026: Agent Review 审阅功能
            'agent/review': [
                'review', '审阅', '代码审阅', '变更审阅',
                'diff', 'Ctrl+R', 'accept', 'reject',
                '接受', '拒绝', '变更', 'changes',
                'inline diff', '内联差异',
            ],
            # Jan 20 2026: CLI Cookbook - 代码评审
            'cookbook/code-review': [
                'code review', '代码评审', '代码审查',
                'pr review', 'pull request', 'PR',
                '评审', '审查', 'review workflow',
                'gh pr', 'git diff', 'reviewer',
            ],
            # Jan 20 2026: Cloud Agent 云代理
            'cloud-agent': [
                'cloud agent', '云代理', '云端代理',
                'cloud', '云端', 'remote agent',
                'background task', '后台任务',
                '&', 'cloud relay', '云中继',
            ],
            'cloud-agent/overview': [
                'cloud agent', '云代理', 'overview',
                '概览', '云端执行', 'remote execution',
            ],
            'cloud-agent/getting-started': [
                'getting started', '快速开始', '入门',
                'cloud setup', '云端配置',
            ],
            'cloud-agent/api': [
                'cloud api', '云端 API', 'api',
                'REST', 'endpoint', '接口',
                'programmatic', '程序化调用',
            ],
            'cloud-agent/api/streaming': [
                'streaming', '流式', '流式响应',
                'stream', 'SSE', 'server-sent events',
                'real-time', '实时',
            ],
            'cloud-agent/api/sessions': [
                'session', '会话', 'sessions',
                'resume', '恢复会话', 'session_id',
                '会话管理', 'session management',
            ],
        }

        for key, kws in keyword_map.items():
            # 支持单级路径（如 mcp）和多级路径（如 modes/plan）匹配
            if key in path or key in full_path:
                keywords.extend(kws)

        return keywords

    def _generate_summary(self, analysis: UpdateAnalysis) -> str:
        """生成更新摘要"""
        parts = []

        if analysis.new_features:
            parts.append(f"新功能 ({len(analysis.new_features)}项)")
        if analysis.improvements:
            parts.append(f"改进 ({len(analysis.improvements)}项)")
        if analysis.fixes:
            parts.append(f"修复 ({len(analysis.fixes)}项)")

        if parts:
            return f"检测到更新: {', '.join(parts)}"
        else:
            return "未检测到新的更新"

    def compute_fingerprint(self, content: str) -> str:
        """计算内容的 fingerprint（SHA256 前16位）

        使用清理后的内容计算哈希，确保比较时忽略空白差异。

        Args:
            content: 原始内容

        Returns:
            16字符的 SHA256 哈希前缀
        """
        # 使用清理后的内容计算指纹，确保一致性
        cleaned = self._clean_content(content)
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]

    async def _get_baseline_fingerprint(self) -> Optional[str]:
        """获取基线 fingerprint

        从知识库中读取上次保存的 changelog 内容哈希。

        Returns:
            上次内容的 fingerprint，不存在时返回 None
        """
        if self._storage is None:
            return None

        try:
            # 确保存储已初始化
            if not self._storage._initialized:
                await self._storage.initialize()

            # 使用便捷方法获取 content_hash
            return self._storage.get_content_hash_by_url(self.changelog_url)
        except Exception as e:
            logger.warning(f"获取基线 fingerprint 失败: {e}")
            return None

    async def analyze(self) -> UpdateAnalysis:
        """执行完整分析

        支持基线比较：
        1. 获取当前 changelog 内容
        2. 计算 fingerprint 并与知识库中的基线比较
        3. 如果 fingerprint 相同，返回 has_updates=False

        Returns:
            更新分析结果
        """
        print_section("分析 Changelog")

        content = await self.fetch_changelog()
        if not content:
            return UpdateAnalysis()

        # 计算当前内容的 fingerprint
        current_fingerprint = self.compute_fingerprint(content)
        logger.debug(f"当前 changelog fingerprint: {current_fingerprint}")

        # 获取基线 fingerprint 进行比较
        baseline_fingerprint = await self._get_baseline_fingerprint()
        if baseline_fingerprint:
            logger.debug(f"基线 fingerprint: {baseline_fingerprint}")
            if current_fingerprint == baseline_fingerprint:
                # 内容未变化，返回无更新结果
                print_info(f"Changelog 内容未变化 (fingerprint: {current_fingerprint[:8]}...)")
                analysis = UpdateAnalysis(
                    has_updates=False,
                    entries=[],
                    summary="未检测到新的更新",
                    raw_content=content,
                    related_doc_urls=[],  # 无更新时不设置相关文档
                )
                return analysis
            else:
                print_info(f"检测到内容变化 (fingerprint: {baseline_fingerprint[:8]}... → {current_fingerprint[:8]}...)")
        else:
            print_info("首次分析，无基线比较")

        entries = self.parse_changelog(content)
        print_info(f"解析到 {len(entries)} 个更新条目")

        analysis = self.extract_update_points(entries)
        analysis.raw_content = content

        # 显示分析结果
        if analysis.has_updates:
            print_success(analysis.summary)
            if analysis.new_features:
                print(f"  新功能: {len(analysis.new_features)} 项")
            if analysis.improvements:
                print(f"  改进: {len(analysis.improvements)} 项")
            if analysis.fixes:
                print(f"  修复: {len(analysis.fixes)} 项")
            if analysis.related_doc_urls:
                print(f"  相关文档: {len(analysis.related_doc_urls)} 个")
        else:
            print_warning("未检测到新的更新")

        return analysis


# ============================================================
# 知识库更新
# ============================================================

class KnowledgeUpdater:
    """知识库更新器

    根据更新分析结果更新知识库。
    """

    def __init__(self):
        self.storage = KnowledgeStorage()
        self.fetcher = WebFetcher(FetchConfig(timeout=60))
        self.manager = KnowledgeManager(name=KB_NAME)

    async def initialize(self):
        """初始化"""
        await self.storage.initialize()
        await self.manager.initialize()
        await self.fetcher.initialize()

    async def update_from_analysis(
        self,
        analysis: UpdateAnalysis,
        force: bool = False,
        changelog_url: str = DEFAULT_CHANGELOG_URL,
    ) -> dict[str, Any]:
        """根据更新分析结果更新知识库

        当 analysis.has_updates=False 时（基线 fingerprint 匹配），
        跳过抓取 related docs，减少网络访问。

        Args:
            analysis: 更新分析结果
            force: 是否强制更新
            changelog_url: 实际使用的 changelog URL（用于保存文档和 baseline key）

        Returns:
            更新结果统计
        """
        print_section("更新知识库")

        results = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "urls_processed": [],
            "no_updates_detected": not analysis.has_updates,
            "changelog_url": changelog_url,  # 记录实际使用的 URL
        }

        # 1. 保存 Changelog 内容
        if analysis.raw_content:
            changelog_doc = Document(
                url=changelog_url,  # 使用实际的 changelog_url 而非 DEFAULT_CHANGELOG_URL
                title="Cursor Changelog",
                content=analysis.raw_content,
                metadata={
                    "source": "changelog",
                    "category": "cursor-docs",
                    "updated_at": datetime.now().isoformat(),
                },
            )

            success, msg = await self.storage.save_document(changelog_doc, force=force)
            if success:
                results["updated"] += 1
                print_success("Changelog 已保存")
            else:
                if "未变化" in msg:
                    results["skipped"] += 1
                    print_info("Changelog 内容未变化，跳过")
                else:
                    results["failed"] += 1
                    print_warning(f"Changelog 保存失败: {msg}")

        # 2. 获取并保存相关文档（仅在有更新时）
        # 当 has_updates=False（基线 fingerprint 匹配）时，跳过抓取以减少网络访问
        if not analysis.has_updates:
            print_info("无内容更新，跳过抓取相关文档")
            logger.debug("跳过 related docs 抓取: has_updates=False (fingerprint 匹配)")
        else:
            urls_to_fetch = analysis.related_doc_urls.copy()

            # 如果没有相关 URL 但有更新，获取所有文档
            if not urls_to_fetch and analysis.has_updates:
                urls_to_fetch = CURSOR_DOC_URLS[:5]  # 获取前 5 个核心文档

            if urls_to_fetch:
                print_info(f"获取 {len(urls_to_fetch)} 个相关文档...")

                for url in urls_to_fetch:
                    result = await self._fetch_and_save_doc(url, force)
                    results["urls_processed"].append(url)

                    if result == "updated":
                        results["updated"] += 1
                    elif result == "skipped":
                        results["skipped"] += 1
                    else:
                        results["failed"] += 1

        # 显示统计
        print("\n知识库更新完成:")
        print(f"  更新: {results['updated']} 个")
        print(f"  跳过: {results['skipped']} 个")
        print(f"  失败: {results['failed']} 个")

        return results

    async def _fetch_and_save_doc(
        self,
        url: str,
        force: bool = False,
    ) -> str:
        """获取并保存单个文档

        Returns:
            "updated", "skipped", 或 "failed"
        """
        try:
            # 获取网页
            result = await self.fetcher.fetch(url)
            if not result.success:
                print_warning(f"获取失败: {url}")
                return "failed"

            # 提取标题
            title = self._extract_title(result.content) or url.split('/')[-1]

            # 创建文档
            doc = Document(
                url=url,
                title=title,
                content=result.content,
                metadata={
                    "source": "web",
                    "category": "cursor-docs",
                    "fetch_method": result.method_used.value,
                    "updated_at": datetime.now().isoformat(),
                },
            )

            # 保存
            success, msg = await self.storage.save_document(doc, force=force)

            if success:
                print_success(f"已更新: {title}")
                return "updated"
            else:
                if "未变化" in msg:
                    print_info(f"未变化: {title}")
                    return "skipped"
                else:
                    print_warning(f"保存失败: {msg}")
                    return "failed"

        except Exception as e:
            print_error(f"处理失败 {url}: {e}")
            return "failed"

    def _extract_title(self, content: str) -> str:
        """从内容提取标题"""
        # 尝试从 HTML title 标签提取
        match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 尝试从第一行提取
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            first_line = re.sub(r'^#+\s*', '', first_line)
            if first_line and len(first_line) <= 200:
                return first_line

        return ""

    async def get_knowledge_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """从知识库获取相关上下文

        Args:
            query: 查询文本
            limit: 返回结果数量

        Returns:
            相关文档列表
        """
        results = await self.storage.search(query, limit=limit)

        context = []
        for result in results:
            doc = await self.storage.load_document(result.doc_id)
            if doc:
                context.append({
                    "title": doc.title,
                    "url": doc.url,
                    "content": doc.content[:2000],
                    "score": result.score,
                })

        return context

    async def get_stats(self) -> dict:
        """获取知识库统计"""
        return await self.storage.get_stats()


# ============================================================
# 迭代目标生成
# ============================================================

class IterationGoalBuilder:
    """迭代目标构建器

    根据更新分析和用户需求构建迭代目标。
    """

    def build_goal(self, context: IterationContext) -> str:
        """构建迭代目标

        Args:
            context: 迭代上下文

        Returns:
            完整的迭代目标
        """
        parts = []

        # 1. 基础目标描述
        parts.append("# 自我迭代任务\n")
        parts.append("根据以下信息对系统进行更新和改进:\n")

        # 2. 用户需求
        if context.user_requirement:
            parts.append("\n## 用户需求\n")
            parts.append(context.user_requirement)
            parts.append("\n")

        # 3. 更新分析
        if context.update_analysis and context.update_analysis.has_updates:
            analysis = context.update_analysis

            parts.append("\n## 检测到的更新\n")
            parts.append(f"摘要: {analysis.summary}\n")

            if analysis.new_features:
                parts.append("\n### 新功能\n")
                for feature in analysis.new_features[:5]:
                    parts.append(f"- {feature}\n")

            if analysis.improvements:
                parts.append("\n### 改进\n")
                for improvement in analysis.improvements[:5]:
                    parts.append(f"- {improvement}\n")

            if analysis.fixes:
                parts.append("\n### 修复\n")
                for fix in analysis.fixes[:5]:
                    parts.append(f"- {fix}\n")

        # 4. 知识库上下文
        if context.knowledge_context:
            parts.append("\n## 参考文档（来自知识库）\n")
            for i, doc in enumerate(context.knowledge_context[:3], 1):
                parts.append(f"\n### {i}. {doc['title']}\n")
                parts.append(f"URL: {doc['url']}\n")
                content_preview = doc['content'][:500]
                parts.append(f"```\n{content_preview}\n```\n")

        # 5. 执行指导
        parts.append("\n## 执行指导\n")
        parts.append("1. 分析上述更新和需求，确定需要修改的代码文件\n")
        parts.append("2. 更新代码以支持新功能或修复问题\n")
        parts.append("3. 确保修改与现有代码风格一致\n")
        parts.append("4. 更新相关文档（如 AGENTS.md）如有必要\n")

        return ''.join(parts)

    def get_summary(self, context: IterationContext) -> str:
        """获取迭代摘要（用于显示）

        Args:
            context: 迭代上下文

        Returns:
            简短摘要
        """
        parts = []

        if context.user_requirement:
            parts.append(f"用户需求: {context.user_requirement[:50]}...")

        if context.update_analysis and context.update_analysis.has_updates:
            parts.append(context.update_analysis.summary)

        if context.knowledge_context:
            parts.append(f"知识库上下文: {len(context.knowledge_context)} 个文档")

        return "; ".join(parts) if parts else "无具体迭代目标"


# ============================================================
# 主执行流程
# ============================================================

class SelfIterator:
    """自我迭代器

    协调整个自我迭代流程。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.knowledge_updater = KnowledgeUpdater()
        # 传递 storage 给 ChangelogAnalyzer 用于基线比较
        self.changelog_analyzer = ChangelogAnalyzer(
            args.changelog_url,
            storage=self.knowledge_updater.storage,
        )
        self.goal_builder = IterationGoalBuilder()

        # 处理 '&' 前缀：如果 requirement 以 '&' 开头，去除前缀
        user_requirement = args.requirement
        self._is_cloud_request = is_cloud_request(user_requirement)
        if self._is_cloud_request:
            user_requirement = strip_cloud_prefix(user_requirement)
            logger.debug(f"检测到 '&' 前缀，原始 requirement: {args.requirement}")
            logger.debug(f"处理后 requirement: {user_requirement}")

        self.context = IterationContext(
            user_requirement=user_requirement,
            dry_run=args.dry_run,
        )

    async def run(self) -> dict[str, Any]:
        """执行自我迭代流程

        Returns:
            执行结果
        """
        print_header("自我迭代脚本")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"用户需求: {self.args.requirement or '(无)'}")
        if self._is_cloud_request:
            print("Cloud 模式: 启用（检测到 '&' 前缀）")
            print(f"实际任务: {self.context.user_requirement}")
        print(f"跳过在线检查: {self.args.skip_online}")
        print(f"仅分析模式: {self.args.dry_run}")
        if self.args.auto_commit:
            print("自动提交: 启用")
            print(f"自动推送: {'启用' if self.args.auto_push else '禁用'}")
            if self.args.commit_message:
                print(f"提交信息前缀: {self.args.commit_message}")

        try:
            # 步骤 1: 分析在线更新
            if not self.args.skip_online:
                print_step(1, "分析在线文档更新")
                self.context.update_analysis = await self.changelog_analyzer.analyze()
            else:
                print_step(1, "跳过在线文档检查")
                self.context.update_analysis = UpdateAnalysis()

            # 步骤 2: 更新知识库
            print_step(2, "更新知识库")
            await self.knowledge_updater.initialize()

            if not self.args.skip_online and self.context.update_analysis:
                await self.knowledge_updater.update_from_analysis(
                    self.context.update_analysis,
                    force=self.args.force_update,
                    changelog_url=self.args.changelog_url,  # 传入实际使用的 changelog_url
                )
            else:
                # 显示现有知识库统计
                stats = await self.knowledge_updater.get_stats()
                print_info(f"现有知识库: {stats.get('document_count', 0)} 个文档")

            # 步骤 3: 获取知识库上下文
            print_step(3, "加载知识库上下文")

            # 根据用户需求或更新内容搜索相关文档
            search_query = self.args.requirement
            if not search_query and self.context.update_analysis:
                # 使用更新关键词搜索
                keywords = []
                for entry in self.context.update_analysis.entries[:3]:
                    keywords.extend(entry.keywords[:2])
                search_query = ' '.join(keywords) if keywords else "CLI agent"

            if search_query:
                self.context.knowledge_context = await self.knowledge_updater.get_knowledge_context(
                    search_query, limit=5
                )

                # 如果搜索结果为空，尝试更宽泛的搜索
                if not self.context.knowledge_context:
                    # 尝试按单词搜索
                    for word in search_query.split()[:3]:
                        if len(word) >= 2:
                            results = await self.knowledge_updater.get_knowledge_context(word, limit=3)
                            self.context.knowledge_context.extend(results)
                            if len(self.context.knowledge_context) >= 3:
                                break

                # 如果仍然为空，获取最新的文档作为通用上下文
                if not self.context.knowledge_context:
                    entries = await self.knowledge_updater.storage.list_documents(limit=3)
                    for entry in entries:
                        doc = await self.knowledge_updater.storage.load_document(entry.doc_id)
                        if doc:
                            self.context.knowledge_context.append({
                                "title": doc.title,
                                "url": doc.url,
                                "content": doc.content[:2000],
                                "score": 0.5,
                            })

                print_info(f"找到 {len(self.context.knowledge_context)} 个相关文档")

            # 步骤 4: 总结迭代目标
            print_step(4, "总结迭代目标")
            self.context.iteration_goal = self.goal_builder.build_goal(self.context)

            summary = self.goal_builder.get_summary(self.context)
            print_info(f"迭代摘要: {summary}")

            # 仅分析模式
            if self.args.dry_run:
                print_section("迭代目标预览（dry-run 模式）")
                print(self.context.iteration_goal[:2000])
                if len(self.context.iteration_goal) > 2000:
                    print("\n... (已截断)")

                return {
                    "success": True,
                    "dry_run": True,
                    "summary": summary,
                    "goal_length": len(self.context.iteration_goal),
                }

            # 步骤 5: 启动 Agent 系统
            print_step(5, "启动 Agent 系统执行迭代")
            result = await self._run_agent_system()

            return result

        except Exception as e:
            logger.exception(f"自我迭代失败: {e}")
            print_error(f"执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _run_commit_phase(
        self,
        iterations_completed: int = 0,
        tasks_completed: int = 0,
    ) -> dict[str, Any]:
        """执行提交阶段（MP 执行路径）

        复用 CommitterAgent 执行 Git 提交操作。

        Args:
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
        print_section("提交阶段")

        # 创建 CommitterAgent
        cursor_config = CursorAgentConfig(working_directory=str(project_root))
        committer_config = CommitterConfig(
            working_directory=str(project_root),
            auto_push=self.args.auto_push,
            commit_message_style="conventional",
            cursor_config=cursor_config,
        )
        committer = CommitterAgent(committer_config)

        # 检查是否有变更
        status = committer.check_status()
        if not status.get("is_repo"):
            print_warning("当前目录不是 Git 仓库，跳过提交")
            return {
                "total_commits": 0,
                "commit_hashes": [],
                "commit_messages": [],
                "pushed_commits": 0,
                "files_changed": [],
                "error": "不是 Git 仓库",
            }

        if not status.get("has_changes"):
            print_info("没有需要提交的变更")
            return {
                "total_commits": 0,
                "commit_hashes": [],
                "commit_messages": [],
                "pushed_commits": 0,
                "files_changed": [],
            }

        # 生成或使用提供的提交信息
        if self.args.commit_message:
            commit_message = self.args.commit_message
        else:
            # 根据执行结果生成提交信息
            print_info("正在生成提交信息...")
            commit_message = await committer.generate_commit_message()

        # 添加迭代信息到提交信息
        if iterations_completed > 0 or tasks_completed > 0:
            suffix = f"\n\n迭代次数: {iterations_completed}, 完成任务: {tasks_completed}"
            if not self.args.commit_message:
                # 自动生成的信息可以直接追加
                commit_message = commit_message.rstrip() + suffix

        # 执行提交
        commit_result = committer.commit(commit_message)

        if not commit_result.success:
            print_error(f"提交失败: {commit_result.error}")
            return {
                "total_commits": 0,
                "commit_hashes": [],
                "commit_messages": [],
                "pushed_commits": 0,
                "files_changed": commit_result.files_changed,
                "error": commit_result.error,
            }

        # 提取提交信息第一行用于显示
        commit_summary = commit_message.split('\n')[0][:50]
        print_success(f"提交成功: {commit_result.commit_hash[:8]} - {commit_summary}")

        # 可选推送
        pushed_commits = 0
        push_error = ""
        if self.args.auto_push:
            print_info("正在推送到远程仓库...")
            push_result = committer.push()
            if push_result.success:
                pushed_commits = 1
                print_success("推送成功")
            else:
                push_error = push_result.error
                print_warning(f"推送失败: {push_error}")

        # 获取提交摘要
        summary = committer.get_commit_summary()

        result: dict[str, Any] = {
            "total_commits": summary.get("successful_commits", 1),
            "commit_hashes": summary.get("commit_hashes", [commit_result.commit_hash]),
            "commit_messages": [commit_message],
            "pushed_commits": pushed_commits,
            "files_changed": summary.get("files_changed", commit_result.files_changed),
        }

        if push_error:
            result["push_error"] = push_error

        return result

    def _get_log_level(self) -> str:
        """获取日志级别

        优先级:
        1. 显式指定 --log-level 参数
        2. --verbose 标志 → DEBUG
        3. --quiet 标志 → WARNING
        4. 默认值 → INFO

        Returns:
            日志级别字符串: DEBUG, INFO, WARNING, ERROR
        """
        # 优先使用显式指定的 --log-level
        explicit_level = getattr(self.args, "log_level", None)
        if explicit_level:
            return explicit_level.upper()

        # --verbose 标志
        if getattr(self.args, "verbose", False):
            return "DEBUG"

        # --quiet 标志
        if getattr(self.args, "quiet", False):
            return "WARNING"

        # 默认值
        return "INFO"

    def _get_stall_diagnostics_enabled(self) -> bool:
        """获取卡死诊断是否启用

        优先级:
        1. 显式指定 --stall-diagnostics → 启用
        2. 显式指定 --no-stall-diagnostics → 禁用
        3. 默认关闭（疑似卡死时再启用）

        Returns:
            是否启用卡死诊断
        """
        # 显式指定
        explicit_enabled = getattr(self.args, "stall_diagnostics_enabled", None)
        if explicit_enabled is not None:
            return explicit_enabled

        # 默认关闭（疑似卡死时再启用 --stall-diagnostics）
        return False

    def _get_stall_diagnostics_level(self) -> str:
        """获取卡死诊断日志级别

        优先级:
        1. 显式指定 --stall-diagnostics-level
        2. --quiet 模式下自动降级到 debug（减少输出）
        3. --verbose 模式下使用 info（更多诊断但不占用 warning）
        4. 默认值 warning

        设计目的:
        - 默认 warning 级别在正常运行时提供关键诊断
        - quiet 模式下降级到 debug 避免刷屏（仅真正的 ERROR 可见）
        - verbose 模式下可启用更多诊断

        Returns:
            日志级别字符串: debug, info, warning, error
        """
        # 显式指定优先
        explicit_level = getattr(self.args, "stall_diagnostics_level", None)
        if explicit_level:
            return explicit_level.lower()

        # --quiet 模式下降级到 debug（减少输出，仅保留真正的 ERROR）
        if getattr(self.args, "quiet", False):
            return "debug"

        # --verbose 模式下使用 info（比 warning 更详细但不占用 warning 级别）
        if getattr(self.args, "verbose", False):
            return "info"

        # 默认 warning
        return "warning"

    def _get_execution_mode(self) -> ExecutionMode:
        """获取执行模式

        优先级:
        1. 检测 requirement 是否以 '&' 开头，如果是则自动切换到 CLOUD 模式
        2. 从命令行参数 --execution-mode 读取
        3. 默认使用 CLI 模式

        权限语义:
        - Cloud 模式下 force_write 默认为 True（允许修改文件）
        - auto_commit 独立于 force_write，需用户显式启用
        - 确保 allow_write/force_write 的权限语义与 auto_commit 策略不冲突

        Returns:
            ExecutionMode 枚举值
        """
        # 优先检测 requirement 是否以 '&' 开头
        requirement = getattr(self.args, "requirement", "")
        if is_cloud_request(requirement):
            logger.info(f"检测到 '&' 前缀，自动切换到 Cloud 模式")
            # 去除 & 前缀，保留实际任务内容
            # 注意：这里只是检测，不修改 args.requirement
            # 实际 goal 会在 IterationGoalBuilder 中处理
            return ExecutionMode.CLOUD

        # 从命令行参数读取
        mode_str = getattr(self.args, "execution_mode", "cli")
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

    def _get_cloud_auth_config(self) -> Optional[CloudAuthConfig]:
        """获取 Cloud 认证配置

        优先级：
        1. 命令行参数 --cloud-api-key
        2. 环境变量 CURSOR_API_KEY
        3. 如果都没有，返回 None

        Returns:
            CloudAuthConfig 或 None
        """
        import os

        api_key = getattr(self.args, "cloud_api_key", None)
        if not api_key:
            api_key = os.environ.get("CURSOR_API_KEY")

        if not api_key:
            return None

        auth_timeout = getattr(self.args, "cloud_auth_timeout", 30)
        return CloudAuthConfig(
            api_key=api_key,
            auth_timeout=auth_timeout,
        )

    def _get_orchestrator_type(self) -> str:
        """获取编排器类型

        优先级：
        1. execution_mode != cli 时强制使用 basic（Cloud/Auto 模式不支持 MP）
        2. 用户显式设置的编排器选项（通过命令行参数 --orchestrator 或 --no-mp）
        3. 从 requirement 中检测的非并行关键词
        4. 默认值 "mp"

        Returns:
            "mp" 或 "basic"
        """
        # 0. 检查执行模式：Cloud/Auto 模式强制使用 basic 编排器
        execution_mode = self._get_execution_mode()
        if execution_mode != ExecutionMode.CLI:
            # Cloud/Auto 模式不支持多进程编排器，强制使用 basic
            return "basic"

        # 1. 检查用户是否显式设置了 no_mp
        if getattr(self.args, "no_mp", False):
            return "basic"

        # 2. 检查用户是否显式设置了 orchestrator
        orchestrator = getattr(self.args, "orchestrator", "mp")
        if orchestrator == "basic":
            return "basic"

        # 3. 如果用户显式设置了编排器选项，尊重用户设置
        if getattr(self.args, "_orchestrator_user_set", False):
            return orchestrator

        # 4. 从 requirement 中检测非并行关键词
        requirement = getattr(self.args, "requirement", "")
        if requirement and _detect_disable_mp_from_requirement(requirement):
            logger.debug(f"从 requirement 检测到非并行关键词，选择 basic 编排器")
            return "basic"

        return orchestrator

    def _has_orchestrator_committed(self, result: dict[str, Any]) -> bool:
        """检测编排器是否已完成提交

        提交去重策略：如果编排器返回的结果中已包含有效的 commits 信息，
        则 SelfIterator 不应再次调用 CommitterAgent 避免重复提交。

        检测条件（满足任一即表示已提交）：
        1. result["commits"]["total_commits"] > 0
        2. result["commits"]["commit_hashes"] 非空
        3. result["iterations"] 中任意迭代有 commit_hash

        Args:
            result: 编排器返回的执行结果

        Returns:
            True 表示编排器已完成提交，False 表示未提交
        """
        # 检查 commits 摘要
        commits = result.get("commits", {})
        if commits:
            # 检查 total_commits
            if commits.get("total_commits", 0) > 0:
                logger.debug(f"检测到编排器提交: total_commits={commits.get('total_commits')}")
                return True

            # 检查 commit_hashes
            commit_hashes = commits.get("commit_hashes", [])
            if commit_hashes and any(h for h in commit_hashes if h):
                logger.debug(f"检测到编排器提交: commit_hashes={commit_hashes}")
                return True

            # 检查 successful_commits（兼容不同字段命名）
            if commits.get("successful_commits", 0) > 0:
                logger.debug(f"检测到编排器提交: successful_commits={commits.get('successful_commits')}")
                return True

        # 检查迭代级别的提交信息
        iterations = result.get("iterations", [])
        for it in iterations:
            commit_hash = it.get("commit_hash")
            if commit_hash and commit_hash.strip():
                logger.debug(f"检测到迭代 {it.get('id')} 已提交: {commit_hash}")
                return True

        return False

    async def _run_agent_system(self) -> dict[str, Any]:
        """运行 Agent 系统

        根据配置选择使用多进程编排器（MP）或协程编排器（basic）。
        MP 启动失败时自动回退到 basic 编排器。

        策略：当 execution_mode != cli 时，强制使用 basic 编排器并给出清晰日志提示。

        Returns:
            执行结果
        """
        print_section("Agent 系统执行")

        # 解析最大迭代次数
        max_iterations = parse_max_iterations(self.args.max_iterations)
        if max_iterations == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        # 获取执行模式
        execution_mode = self._get_execution_mode()
        print_info(f"执行模式: {execution_mode.value}")

        # 创建知识库管理器
        manager = KnowledgeManager(name=KB_NAME)
        await manager.initialize()

        # 确定编排器类型
        orchestrator_type = self._get_orchestrator_type()
        use_fallback = False
        result: dict[str, Any] = {}

        # 检查 execution_mode 与 orchestrator 的兼容性
        # Cloud/Auto 模式不支持多进程编排器
        user_requested_mp = getattr(self.args, "orchestrator", "mp") == "mp" and not getattr(self.args, "no_mp", False)
        if execution_mode != ExecutionMode.CLI and user_requested_mp:
            # 用户请求使用 MP，但 execution_mode 是 Cloud/Auto
            print_warning(
                f"执行模式 '{execution_mode.value}' 不支持多进程编排器，"
                "已自动切换到基本协程编排器（basic）"
            )
            logger.warning(
                f"execution_mode={execution_mode.value} 不支持 MP 编排器，"
                "自动切换到 basic 编排器"
            )

        if orchestrator_type == "mp":
            # 尝试使用多进程编排器
            result = await self._run_with_mp_orchestrator(max_iterations, manager)

            # 检查是否需要回退
            if result.get("_fallback_required"):
                use_fallback = True
                fallback_reason = result.get("_fallback_reason", "未知错误")
                print_warning(f"MP 编排器启动失败: {fallback_reason}")
                print_warning("正在回退到基本协程编排器...")
                logger.warning(f"MP 编排器回退: {fallback_reason}")
        else:
            # 直接使用基本编排器
            use_fallback = True
            if execution_mode != ExecutionMode.CLI:
                print_info(f"使用基本协程编排器（执行模式: {execution_mode.value}）")
            else:
                print_info("使用基本协程编排器（--no-mp 或 --orchestrator basic）")

        if use_fallback:
            # 使用基本协程编排器
            result = await self._run_with_basic_orchestrator(max_iterations, manager)

        # 完成后提交阶段（提交去重策略）
        # 如果编排器已经完成了提交，则跳过 SelfIterator 的二次提交
        if self.args.auto_commit:
            orchestrator_already_committed = self._has_orchestrator_committed(result)
            if orchestrator_already_committed:
                logger.info("编排器已完成提交，跳过 SelfIterator 二次提交")
                print_info("检测到编排器已提交，跳过重复提交")
            else:
                commits_result = await self._run_commit_phase(
                    result.get("iterations_completed", 0),
                    result.get("total_tasks_completed", 0),
                )
                result["commits"] = commits_result

                # 更新 pushed 标志
                if commits_result.get("pushed_commits", 0) > 0:
                    result["pushed"] = True

        # 显示结果
        self._print_execution_result(result)

        return result

    async def _build_enhanced_goal(self, goal: str, manager: KnowledgeManager) -> str:
        """构建增强后的目标（方案 B: 在 goal 构建阶段注入知识库上下文）

        当知识库上下文可用时，将文档内容拼接到 goal 中，提供更丰富的上下文。

        Args:
            goal: 原始目标
            manager: 知识库管理器

        Returns:
            增强后的目标字符串
        """
        # 检查是否已经有知识库上下文（从 IterationGoalBuilder 构建的）
        if self.context.knowledge_context:
            # 已经通过 goal_builder 注入了，不需要重复注入
            logger.debug("知识库上下文已在 goal 构建阶段注入，跳过增强")
            return goal

        # 尝试从知识库搜索相关文档
        search_query = self.args.requirement
        if not search_query:
            # 没有用户需求，无法搜索
            return goal

        try:
            # 搜索知识库
            results = await self.knowledge_updater.get_knowledge_context(search_query, limit=5)

            if not results:
                logger.debug("知识库搜索无结果，使用原始目标")
                return goal

            # 构建增强目标
            context_text = "\n\n## 参考文档（来自知识库，方案 B 注入）\n\n"
            for i, doc in enumerate(results, 1):
                context_text += f"### {i}. {doc['title']}\n"
                # 截断过长内容
                content = doc['content'][:1000]
                if len(doc['content']) > 1000:
                    content += "..."
                context_text += f"```\n{content}\n```\n\n"

            enhanced_goal = f"{goal}\n{context_text}"
            print_success(f"知识库增强: 已注入 {len(results)} 个相关文档到目标上下文")
            logger.info(f"知识库增强（方案 B）: 注入 {len(results)} 个文档，增加 {len(context_text)} 字符")

            return enhanced_goal

        except Exception as e:
            logger.warning(f"知识库增强失败，使用原始目标: {e}")
            return goal

    async def _run_with_mp_orchestrator(
        self,
        max_iterations: int,
        manager: KnowledgeManager,
    ) -> dict[str, Any]:
        """使用多进程编排器执行

        支持知识库增强：
        - 方案 B: 在 goal 构建阶段注入知识库上下文（通过 _build_enhanced_goal）
        - 方案 C: 启用 MP 任务级知识库注入（通过 enable_knowledge_injection 配置）

        Args:
            max_iterations: 最大迭代次数
            manager: 知识库管理器

        Returns:
            执行结果（如果失败，包含 _fallback_required=True）
        """
        logger.info("使用 MultiProcessOrchestrator (MP 多进程编排器)")
        print_info(f"MP 编排器配置: worker_count={self.args.workers}, max_iterations={max_iterations}")

        try:
            # 方案 B: 在 goal 构建阶段注入知识库上下文
            enhanced_goal = await self._build_enhanced_goal(self.context.iteration_goal, manager)

            # 计算日志级别（优先级: --log-level > --verbose/--quiet > 默认 INFO）
            log_level = self._get_log_level()

            # 计算卡死诊断配置（quiet 模式下降级）
            stall_diagnostics_enabled = self._get_stall_diagnostics_enabled()
            stall_diagnostics_level = self._get_stall_diagnostics_level()

            config = MultiProcessOrchestratorConfig(
                working_directory=str(project_root),
                max_iterations=max_iterations,
                worker_count=self.args.workers,
                # 方案 C: 启用 MP 任务级知识库注入
                enable_knowledge_search=True,
                enable_knowledge_injection=True,
                # 自动提交配置透传（与 CommitPolicy 语义对齐）
                enable_auto_commit=self.args.auto_commit,
                auto_push=self.args.auto_push,
                commit_per_iteration=getattr(self.args, "commit_per_iteration", False),
                # commit_on_complete 语义：当 commit_per_iteration=False 时仅在完成时提交
                commit_on_complete=not getattr(self.args, "commit_per_iteration", False),
                # 执行模式配置（MP 主要使用 CLI，但保持配置兼容性）
                execution_mode=getattr(self.args, "execution_mode", "cli"),
                planner_execution_mode=getattr(self.args, "planner_execution_mode", None),
                worker_execution_mode=getattr(self.args, "worker_execution_mode", None),
                reviewer_execution_mode=getattr(self.args, "reviewer_execution_mode", None),
                # 日志配置透传到子进程
                verbose=getattr(self.args, "verbose", False),
                log_level=log_level,
                heartbeat_debug=getattr(self.args, "heartbeat_debug", False),
                # 卡死诊断配置透传
                stall_diagnostics_enabled=stall_diagnostics_enabled,
                stall_diagnostics_level=stall_diagnostics_level,
                stall_recovery_interval=getattr(self.args, "stall_recovery_interval", 30.0),
                execution_health_check_interval=getattr(self.args, "execution_health_check_interval", 30.0),
                health_warning_cooldown_seconds=getattr(self.args, "health_warning_cooldown", 60.0),
                # 流式控制台渲染配置透传
                stream_console_renderer=getattr(self.args, "stream_console_renderer", False),
                stream_advanced_renderer=getattr(self.args, "stream_advanced_renderer", False),
                stream_typing_effect=getattr(self.args, "stream_typing_effect", False),
                stream_typing_delay=getattr(self.args, "stream_typing_delay", 0.02),
                stream_word_mode=getattr(self.args, "stream_word_mode", True),
                stream_color_enabled=getattr(self.args, "stream_color_enabled", True),
                stream_show_word_diff=getattr(self.args, "stream_show_word_diff", False),
            )

            # 创建多进程编排器
            orchestrator = MultiProcessOrchestrator(config)
            logger.info(f"MP 编排器已创建: worker_count={config.worker_count}, 知识库注入={config.enable_knowledge_injection}")

            # 执行（使用增强后的目标）
            print_info("开始执行迭代任务...")
            result = await orchestrator.run(enhanced_goal)
            return result

        except asyncio.TimeoutError as e:
            logger.error(f"MP 编排器超时: {e}")
            return {
                "_fallback_required": True,
                "_fallback_reason": f"启动超时: {e}",
            }
        except OSError as e:
            # 进程创建失败（如资源不足）
            logger.error(f"MP 编排器进程创建失败: {e}")
            return {
                "_fallback_required": True,
                "_fallback_reason": f"进程创建失败: {e}",
            }
        except RuntimeError as e:
            # 运行时错误（如事件循环问题）
            logger.error(f"MP 编排器运行时错误: {e}")
            return {
                "_fallback_required": True,
                "_fallback_reason": f"运行时错误: {e}",
            }
        except Exception as e:
            # 其他未预期的异常
            logger.error(f"MP 编排器启动失败: {e}")
            return {
                "_fallback_required": True,
                "_fallback_reason": str(e),
            }

    async def _run_with_basic_orchestrator(
        self,
        max_iterations: int,
        manager: KnowledgeManager,
    ) -> dict[str, Any]:
        """使用基本协程编排器执行

        支持 KnowledgeManager 注入、auto_commit/auto_push、execution_mode 和 cloud_auth_config。

        当 execution_mode 为 CLOUD 或 AUTO 时，使用 --cloud-timeout 参数覆盖默认 timeout，
        确保 Cloud/Auto 走相同的超时策略（默认 600 秒）。

        Args:
            max_iterations: 最大迭代次数
            manager: 知识库管理器

        Returns:
            执行结果
        """
        logger.info("使用 Orchestrator (基本协程编排器)")

        # 获取执行模式和 Cloud 认证配置
        execution_mode = self._get_execution_mode()
        cloud_auth_config = self._get_cloud_auth_config()

        print_info(f"协程编排器配置: worker_pool_size={self.args.workers}, max_iterations={max_iterations}")
        print_info(f"执行模式: {execution_mode.value}")
        if cloud_auth_config:
            print_info("Cloud 认证: 已配置")

        # 根据执行模式决定超时时间
        # - Cloud/Auto 模式：使用 --cloud-timeout 参数（默认 600 秒）
        # - CLI 模式：使用 CursorAgentConfig 默认值（300 秒）
        if execution_mode in (ExecutionMode.CLOUD, ExecutionMode.AUTO):
            cloud_timeout = getattr(self.args, "cloud_timeout", 600)
            cursor_config = CursorAgentConfig(
                working_directory=str(project_root),
                timeout=cloud_timeout,
            )
            print_info(f"Cloud 超时: {cloud_timeout}s")
            logger.info(f"Cloud/Auto 执行模式，使用 cloud_timeout={cloud_timeout}s")
        else:
            cursor_config = CursorAgentConfig(working_directory=str(project_root))

        # 解析角色级执行模式（可选）
        planner_exec_mode = self._parse_execution_mode(
            getattr(self.args, "planner_execution_mode", None)
        )
        worker_exec_mode = self._parse_execution_mode(
            getattr(self.args, "worker_execution_mode", None)
        )
        reviewer_exec_mode = self._parse_execution_mode(
            getattr(self.args, "reviewer_execution_mode", None)
        )

        config = OrchestratorConfig(
            working_directory=str(project_root),
            max_iterations=max_iterations,
            worker_pool_size=self.args.workers,
            cursor_config=cursor_config,
            # 自动提交配置透传（与 CommitPolicy 语义对齐）
            enable_auto_commit=self.args.auto_commit,
            auto_push=self.args.auto_push,
            commit_per_iteration=getattr(self.args, "commit_per_iteration", False),
            # commit_on_complete 语义：当 commit_per_iteration=False 时仅在完成时提交
            commit_on_complete=not getattr(self.args, "commit_per_iteration", False),
            # 执行模式和 Cloud 认证配置
            execution_mode=execution_mode,
            cloud_auth_config=cloud_auth_config,
            # 角色级执行模式（可选）
            planner_execution_mode=planner_exec_mode,
            worker_execution_mode=worker_exec_mode,
            reviewer_execution_mode=reviewer_exec_mode,
            # 流式控制台渲染配置透传
            stream_console_renderer=getattr(self.args, "stream_console_renderer", False),
            stream_advanced_renderer=getattr(self.args, "stream_advanced_renderer", False),
            stream_typing_effect=getattr(self.args, "stream_typing_effect", False),
            stream_typing_delay=getattr(self.args, "stream_typing_delay", 0.02),
            stream_word_mode=getattr(self.args, "stream_word_mode", True),
            stream_color_enabled=getattr(self.args, "stream_color_enabled", True),
            stream_show_word_diff=getattr(self.args, "stream_show_word_diff", False),
        )

        # 创建编排器，注入知识库管理器
        orchestrator = Orchestrator(config, knowledge_manager=manager)
        logger.info(
            f"协程编排器已创建: worker_pool_size={config.worker_pool_size}, "
            f"execution_mode={config.execution_mode.value}"
        )

        # 执行
        print_info("开始执行迭代任务...")
        result = await orchestrator.run(self.context.iteration_goal)
        return result

    def _print_execution_result(self, result: dict[str, Any]) -> None:
        """打印执行结果

        Args:
            result: 执行结果字典
        """
        print_section("执行结果")
        print(f"状态: {'成功' if result.get('success') else '未完成'}")
        print(f"迭代次数: {result.get('iterations_completed', 0)}")
        print(f"任务创建: {result.get('total_tasks_created', 0)}")
        print(f"任务完成: {result.get('total_tasks_completed', 0)}")
        print(f"任务失败: {result.get('total_tasks_failed', 0)}")

        if result.get('final_score'):
            print(f"最终评分: {result['final_score']:.1f}")

        # 显示提交信息
        commits_info = result.get('commits', {})
        if commits_info:
            print_section("提交信息")
            total_commits = commits_info.get('total_commits', 0)
            if total_commits > 0:
                print_success(f"提交数量: {total_commits}")

                # 显示提交哈希
                commit_hashes = commits_info.get('commit_hashes', [])
                if commit_hashes:
                    for i, hash_val in enumerate(commit_hashes[-3:], 1):  # 显示最近3个
                        print(f"  提交 {i}: {hash_val[:8] if len(hash_val) > 8 else hash_val}")

                # 显示提交信息摘要
                commit_messages = commits_info.get('commit_messages', [])
                if commit_messages:
                    print("提交信息摘要:")
                    for msg in commit_messages[-3:]:  # 显示最近3条
                        # 截取第一行作为摘要
                        summary = msg.split('\n')[0][:60]
                        print(f"  - {summary}")

                # 显示推送状态
                pushed_commits = commits_info.get('pushed_commits', 0)
                if self.args.auto_push:
                    if pushed_commits > 0:
                        print_success(f"推送状态: 已推送 {pushed_commits} 个提交到远程仓库")
                    else:
                        # 开启了 auto_push 但推送失败，显示警告
                        print_warning("推送状态: 推送失败，请手动执行 git push")
                elif result.get('pushed'):
                    print_success("推送状态: 已推送到远程仓库")
            else:
                print_info("无代码更改，未创建提交")


# ============================================================
# 日志配置
# ============================================================

def setup_logging(verbose: bool = False, quiet: bool = False, log_level: str = None):
    """配置日志

    Args:
        verbose: 详细输出模式（DEBUG 级别）
        quiet: 静默模式（WARNING 级别）
        log_level: 显式指定的日志级别（优先级最高）
    """
    logger.remove()

    # 计算日志级别（优先级: log_level > verbose > quiet > 默认 INFO）
    if log_level:
        level = log_level.upper()
    elif verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        filter=lambda record: record["level"].name != "DEBUG" or verbose,
    )
    logger.add(
        "logs/self_iterate_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


# ============================================================
# 主入口
# ============================================================

def _detect_disable_mp_from_requirement(requirement: str) -> bool:
    """从 requirement 中检测是否包含禁用多进程的关键词

    Args:
        requirement: 用户需求字符串

    Returns:
        True 表示检测到禁用多进程关键词，False 表示未检测到
    """
    if not requirement:
        return False

    requirement_lower = requirement.lower()
    for keyword in DISABLE_MP_KEYWORDS:
        if keyword.lower() in requirement_lower:
            logger.debug(f"检测到禁用 MP 关键词: '{keyword}' in requirement")
            return True
    return False


def main():
    """主函数"""
    args = parse_args()
    setup_logging(
        verbose=args.verbose,
        quiet=getattr(args, "quiet", False),
        log_level=getattr(args, "log_level", None),
    )

    # 关键词检测：仅当用户未显式设置编排器时，从 requirement 检测是否需要禁用 MP
    # 条件：no_mp=False 且 orchestrator 为默认值 "mp" 且 _orchestrator_user_set=False
    if (
        not getattr(args, "no_mp", False)
        and getattr(args, "orchestrator", "mp") == "mp"
        and not getattr(args, "_orchestrator_user_set", False)
    ):
        if _detect_disable_mp_from_requirement(args.requirement):
            args.no_mp = True
            print_info("检测到需求中的关键词，自动切换到协程编排器（basic）")
            logger.info(f"关键词触发 no_mp=True: requirement='{args.requirement[:50]}...'")

    # 创建并运行迭代器
    iterator = SelfIterator(args)

    try:
        result = asyncio.run(iterator.run())

        # 退出码
        if result.get("success"):
            if result.get("dry_run"):
                print_success("\nDry-run 分析完成")
            else:
                print_success("\n自我迭代完成")
            sys.exit(0)
        else:
            print_error("\n自我迭代未完成")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(130)


if __name__ == "__main__":
    main()
