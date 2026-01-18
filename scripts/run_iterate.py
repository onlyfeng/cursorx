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

from knowledge import (
    WebFetcher,
    FetchConfig,
    KnowledgeStorage,
    KnowledgeManager,
    Document,
)
from coordinator import Orchestrator, OrchestratorConfig
from cursor.client import CursorAgentConfig


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
    
    return parser.parse_args()


# ============================================================
# 在线文档分析
# ============================================================

class ChangelogAnalyzer:
    """Changelog 分析器
    
    从 Cursor Changelog 页面获取更新内容并分析。
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
    
    def parse_changelog(self, content: str) -> list[ChangelogEntry]:
        """解析 Changelog 内容
        
        Args:
            content: Changelog 页面内容
            
        Returns:
            Changelog 条目列表
        """
        entries: list[ChangelogEntry] = []
        
        # 按日期/版本分割（常见格式：## 2024-01-15 或 ### v1.0.0）
        sections = re.split(r'\n(?=#{1,3}\s*(?:\d{4}[-/]\d{2}[-/]\d{2}|v?\d+\.\d+))', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            entry = ChangelogEntry()
            lines = section.strip().split('\n')
            
            # 提取标题行
            if lines:
                title_line = lines[0].strip()
                # 尝试提取日期
                date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', title_line)
                if date_match:
                    entry.date = date_match.group(1)
                
                # 尝试提取版本
                version_match = re.search(r'v?(\d+\.\d+(?:\.\d+)?)', title_line)
                if version_match:
                    entry.version = version_match.group(1)
                
                entry.title = re.sub(r'^#+\s*', '', title_line)
            
            # 提取内容
            entry.content = '\n'.join(lines[1:]).strip()
            
            # 分析类别和关键词
            content_lower = entry.content.lower()
            if any(kw in content_lower for kw in ['新增', '新功能', 'new', 'feature']):
                entry.category = 'feature'
            elif any(kw in content_lower for kw in ['修复', 'fix', 'bug']):
                entry.category = 'fix'
            elif any(kw in content_lower for kw in ['改进', '优化', 'improve', 'enhance']):
                entry.category = 'improvement'
            else:
                entry.category = 'other'
            
            # 提取关键词
            for pattern in UPDATE_KEYWORDS:
                matches = re.findall(pattern, entry.content, re.IGNORECASE)
                entry.keywords.extend(matches)
            
            if entry.content:
                entries.append(entry)
        
        return entries
    
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
        """从 URL 提取关键词"""
        # 从 URL 路径提取关键词
        path = url.split('/')[-1]
        keywords = [path.replace('-', ' ')]
        
        # 添加特定关键词映射
        keyword_map = {
            'parameters': ['参数', 'parameter', 'option', '选项'],
            'slash-commands': ['斜杠命令', 'slash', '/'],
            'mcp': ['mcp', '服务器', 'server'],
            'hooks': ['hook', '钩子'],
            'subagents': ['subagent', '子代理'],
            'skills': ['skill', '技能'],
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
        print(f"\n知识库更新完成:")
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
            print(f"自动提交: 启用")
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
    
    async def _run_agent_system(self) -> dict[str, Any]:
        """运行 Agent 系统
        
        Returns:
            执行结果
        """
        print_section("Agent 系统执行")
        
        # 解析最大迭代次数
        max_iterations = parse_max_iterations(self.args.max_iterations)
        if max_iterations == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")
        
        # 创建配置
        cursor_config = CursorAgentConfig(
            working_directory=str(project_root),
        )
        
        config = OrchestratorConfig(
            working_directory=str(project_root),
            max_iterations=max_iterations,
            worker_pool_size=self.args.workers,
            cursor_config=cursor_config,
            # 自动提交配置
            enable_auto_commit=self.args.auto_commit,
            auto_push=self.args.auto_push,
            commit_on_complete=True,  # 仅在完成时提交
        )
        
        # 创建知识库管理器
        manager = KnowledgeManager(name=KB_NAME)
        await manager.initialize()
        
        # 创建编排器
        orchestrator = Orchestrator(config, knowledge_manager=manager)
        
        # 执行
        print_info("开始执行迭代任务...")
        result = await orchestrator.run(self.context.iteration_goal)
        
        # 显示结果
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
                    print(f"提交信息摘要:")
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
                    print_success(f"推送状态: 已推送到远程仓库")
            else:
                print_info("无代码更改，未创建提交")
        
        return result


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
