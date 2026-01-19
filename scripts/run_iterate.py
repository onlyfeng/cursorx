#!/usr/bin/env python3
"""自我迭代脚本

实现完整的自我迭代工作流：
用户输入需求 → 分析在线文档更新 → 更新知识库 → 总结迭代内容 → 启动 Agent 执行

用法:
    # 完整自我迭代（检查在线更新 + 用户需求）
    python scripts/self_iterate.py "增加对新斜杠命令的支持"

    # 仅基于知识库迭代（跳过在线检查）
    python scripts/self_iterate.py --skip-online "优化 CLI 参数处理"

    # 纯自动模式（无额外需求，仅检查更新）
    python scripts/self_iterate.py

    # 指定 changelog URL
    python scripts/self_iterate.py --changelog-url "https://cursor.com/cn/changelog"

    # 仅分析不执行
    python scripts/self_iterate.py --dry-run "分析改进点"
"""
import argparse
import asyncio
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
from coordinator import (
    MultiProcessOrchestrator,
    MultiProcessOrchestratorConfig,
    Orchestrator,
    OrchestratorConfig,
)
from cursor.client import CursorAgentConfig
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
  python scripts/self_iterate.py "增加对新斜杠命令的支持"

  # 仅基于知识库迭代（跳过在线检查）
  python scripts/self_iterate.py --skip-online "优化 CLI 参数处理"

  # 纯自动模式（无额外需求，仅检查更新）
  python scripts/self_iterate.py

  # 仅分析不执行
  python scripts/self_iterate.py --dry-run "分析改进点"
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

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
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

    return parser.parse_args()


# ============================================================
# 在线文档分析
# ============================================================

class ChangelogAnalyzer:
    """Changelog 分析器

    从 Cursor Changelog 页面获取更新内容并分析。
    支持多种解析策略：优先日期标题块、备用月份标题块、保底全页单条。
    """

    def __init__(self, changelog_url: str = DEFAULT_CHANGELOG_URL):
        self.changelog_url = changelog_url
        self.fetcher = WebFetcher(FetchConfig(timeout=60))

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

        Args:
            content: 原始内容

        Returns:
            清理后的内容
        """
        if not content:
            return ""

        # 移除 HTML 注释
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # 移除 script 和 style 标签及内容
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # 移除常见 HTML 标签，保留内容
        # 保留 <a> 标签的文本
        content = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', content, flags=re.IGNORECASE)
        # 移除其他标签
        content = re.sub(r'<[^>]+>', '', content)

        # 解码常见 HTML 实体
        html_entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)

        # 移除多余空行（保留最多2个连续空行）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 移除行首行尾空白
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)

        return content.strip()

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

    def _parse_by_version_headers(self, content: str) -> list[ChangelogEntry]:
        """策略2: 按版本标题块解析（备用）

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
        2. 备用按版本标题块解析
        3. 保底将全页作为单条 entry

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

        # 策略2: 按版本标题块解析
        entries = self._parse_by_version_headers(cleaned_content)
        if entries:
            logger.debug(f"使用版本标题策略解析，得到 {len(entries)} 条")
            return entries

        # 策略3: 保底全页单条
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

        # 添加特定关键词映射
        # 包含 Jan 16 2026 新特性: plan/ask 模式、cloud relay、diff
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
        }

        for key, kws in keyword_map.items():
            if key in path:
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

    async def analyze(self) -> UpdateAnalysis:
        """执行完整分析

        Returns:
            更新分析结果
        """
        print_section("分析 Changelog")

        content = await self.fetch_changelog()
        if not content:
            return UpdateAnalysis()

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
    ) -> dict[str, Any]:
        """根据更新分析结果更新知识库

        Args:
            analysis: 更新分析结果
            force: 是否强制更新

        Returns:
            更新结果统计
        """
        print_section("更新知识库")

        results = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "urls_processed": [],
        }

        # 1. 保存 Changelog 内容
        if analysis.raw_content:
            changelog_doc = Document(
                url=DEFAULT_CHANGELOG_URL,
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

        # 2. 获取并保存相关文档
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
        self.changelog_analyzer = ChangelogAnalyzer(args.changelog_url)
        self.knowledge_updater = KnowledgeUpdater()
        self.goal_builder = IterationGoalBuilder()
        self.context = IterationContext(
            user_requirement=args.requirement,
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

    def _get_orchestrator_type(self) -> str:
        """获取编排器类型

        Returns:
            "mp" 或 "basic"
        """
        if getattr(self.args, "no_mp", False):
            return "basic"
        return getattr(self.args, "orchestrator", "mp")

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

        Returns:
            执行结果
        """
        print_section("Agent 系统执行")

        # 解析最大迭代次数
        max_iterations = parse_max_iterations(self.args.max_iterations)
        if max_iterations == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        # 创建知识库管理器
        manager = KnowledgeManager(name=KB_NAME)
        await manager.initialize()

        # 确定编排器类型
        orchestrator_type = self._get_orchestrator_type()
        use_fallback = False
        result: dict[str, Any] = {}

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

        支持 KnowledgeManager 注入和 auto_commit/auto_push。

        Args:
            max_iterations: 最大迭代次数
            manager: 知识库管理器

        Returns:
            执行结果
        """
        logger.info("使用 Orchestrator (基本协程编排器)")
        print_info(f"协程编排器配置: worker_pool_size={self.args.workers}, max_iterations={max_iterations}")

        cursor_config = CursorAgentConfig(working_directory=str(project_root))

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
        )

        # 创建编排器，注入知识库管理器
        orchestrator = Orchestrator(config, knowledge_manager=manager)
        logger.info(f"协程编排器已创建: worker_pool_size={config.worker_pool_size}")

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

def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
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

def main():
    """主函数"""
    args = parse_args()
    setup_logging(args.verbose)

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
