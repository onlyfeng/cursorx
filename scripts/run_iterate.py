#!/usr/bin/env python3
"""自我迭代脚本

实现完整的自我迭代工作流：
用户输入需求 → 分析在线文档更新 → 更新知识库 → 总结迭代内容 → 启动 Agent 执行

================================================================================
URL 裁决规则表参考
================================================================================

本脚本在抓取文档时使用的 URL 裁决逻辑定义于:
    knowledge/doc_url_strategy.py 模块的【URL 裁决规则表】

裁决流程摘要：
1. URL 选择 (url_strategy) → 决定候选 URL 是否进入 urls_to_fetch
2. 外链分类 → 判定 URL 是内链还是外链
3. 抓取策略 (fetch_policy) → 决定外链的处理方式
4. 路径前缀检查 → 决定内链是否允许实际抓取

相关配置项：
- config.yaml: knowledge_docs_update.url_strategy (阶段 1)
- config.yaml: knowledge_docs_update.docs_source.fetch_policy (阶段 2-4)
- config.yaml: docs_source.allowed_doc_url_prefixes (仅用于 load_core_docs 过滤)
- CLI: --allowed-path-prefixes, --external-link-mode 等
- [DEPRECATED] CLI: --allowed-url-prefixes（请使用 --allowed-path-prefixes）

fetch_policy 当前作用范围：
- 配置已解析并在 update_from_analysis() 主流程中**实际调用**
- apply_fetch_policy() 返回结果**已用于覆盖** urls_to_fetch
- 外链过滤（external_link_mode）**已生效**：外链按策略从 urls_to_fetch 移除/记录
- 内链路径前缀检查（enforce_path_prefixes）默认禁用，可通过 --enforce-path-prefixes 启用
- Phase B 计划将 enforce_path_prefixes 默认值改为 True

迁移计划：
- Phase A（当前）：仅警告，不改变行为
- Phase B（计划中）：启用 fetch_policy gate，默认最小抓取面
详细计划参见 knowledge/doc_url_strategy.py 的【两阶段迁移计划】和【fetch_policy 作用范围说明】

================================================================================
副作用控制策略矩阵 (Side Effect Control)
================================================================================

详细策略矩阵参见: core/execution_policy.py

+---------------+----------------+----------------+----------------+-----------------+
| 策略          | 网络请求       | 文件写入       | Git 操作       | 适用场景        |
+===============+================+================+================+=================+
| normal        | 允许           | 允许           | 允许           | 正常执行        |
| (默认)        | 在线检查文档   | 更新知识库     | 自动提交(显式) |                 |
+---------------+----------------+----------------+----------------+-----------------+
| skip-online   | 禁止在线检查   | 允许           | 允许           | 离线环境        |
| (--skip-online)| 本地缓存优先  | 更新知识库     | 自动提交(显式) | CI/CD 加速      |
+---------------+----------------+----------------+----------------+-----------------+
| dry-run       | 允许           | 禁止           | 禁止           | 预览/调试       |
| (--dry-run)   | (用于分析)     | 仅日志输出     | 仅日志输出     | 安全检查        |
+---------------+----------------+----------------+----------------+-----------------+
| minimal       | 禁止           | 禁止           | 禁止           | 最小副作用      |
| (两者组合)    |                |                |                | 纯分析场景      |
+---------------+----------------+----------------+----------------+-----------------+

**各阶段副作用说明**:

| 阶段                     | 副作用类型     | normal | skip-online | dry-run | minimal |
|--------------------------|----------------|--------|-------------|---------|---------|
| 1. 检查 changelog        | 网络请求       | ✓      | ✗           | ✓       | ✗       |
| 2. 获取 llms.txt         | 网络请求       | ✓      | ✗(用缓存)   | ✓       | ✗       |
| 3. 抓取文档内容          | 网络请求       | ✓      | ✗           | ✓       | ✗       |
| 4. 更新知识库            | 文件写入       | ✓      | ✓           | ✗       | ✗       |
| 5. 执行 Agent 任务       | 文件修改       | ✓      | ✓           | ✗       | ✗       |
| 6. Git 提交              | Git 操作       | ✓(显式)| ✓(显式)     | ✗       | ✗       |

================================================================================
用法示例
================================================================================

    # 完整自我迭代（检查在线更新 + 用户需求）
    python scripts/run_iterate.py "增加对新斜杠命令的支持"

    # 仅基于知识库迭代（跳过在线检查）
    python scripts/run_iterate.py --skip-online "优化 CLI 参数处理"

    # 纯自动模式（无额外需求，仅检查更新）
    python scripts/run_iterate.py

    # 指定 changelog URL
    python scripts/run_iterate.py --changelog-url "https://cursor.com/cn/changelog"

    # 仅分析不执行（预览任务分解）
    python scripts/run_iterate.py --dry-run "分析改进点"

    # 最小副作用模式（纯本地分析）
    python scripts/run_iterate.py --skip-online --dry-run "分析代码结构"
"""

import argparse
import asyncio
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from agents.committer import CommitterAgent, CommitterConfig
from coordinator import (
    MultiProcessOrchestrator,
    MultiProcessOrchestratorConfig,
    Orchestrator,
    OrchestratorConfig,
)

# core.cloud_utils 的 is_cloud_request/strip_cloud_prefix:
# - 已由 build_execution_decision 内部调用
# - 重新导出供外部测试使用（如 test_cloud_integration.py 验证模块一致性）
from core.cloud_utils import (
    CLOUD_PREFIX,
    is_cloud_request,
    strip_cloud_prefix,
)
from core.config import (
    DEFAULT_ALLOWED_DOMAINS as CONFIG_DEFAULT_ALLOWED_DOMAINS,
)
from core.config import (
    DEFAULT_ALLOWED_PATH_PREFIXES as CONFIG_DEFAULT_ALLOWED_PATH_PREFIXES,
)
from core.config import (
    DEFAULT_ALLOWED_URL_PREFIXES as CONFIG_DEFAULT_ALLOWED_URL_PREFIXES,  # deprecated alias
)
from core.config import (
    DEFAULT_CHANGELOG_URL as CONFIG_DEFAULT_CHANGELOG_URL,
)
from core.config import (
    DEFAULT_CLOUD_AUTH_TIMEOUT,
    DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES,
    DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES_BOTH,
    MAX_CONSOLE_PREVIEW_CHARS,
    MAX_KNOWLEDGE_DOC_PREVIEW_CHARS,
    TRUNCATION_HINT,
    ResolvedSettings,
    UnifiedOptions,
    # Deprecated 警告统一机制
    _warn_deprecated_once,
    build_cloud_client_config,
    build_unified_overrides,
    parse_max_iterations,
    # parse_max_iterations 用于解析 max_iterations 参数（重新导出供测试使用）
    print_debug_config,
    resolve_settings,
)
from core.config import (
    DEFAULT_EXTERNAL_LINK_ALLOWLIST as CONFIG_DEFAULT_EXTERNAL_LINK_ALLOWLIST,
)
from core.config import (
    DEFAULT_EXTERNAL_LINK_MODE as CONFIG_DEFAULT_EXTERNAL_LINK_MODE,
)
from core.config import (
    DEFAULT_FALLBACK_CORE_DOCS_COUNT as CONFIG_DEFAULT_FALLBACK_CORE_DOCS_COUNT,
)
from core.config import (
    DEFAULT_LLMS_CACHE_PATH as CONFIG_DEFAULT_LLMS_CACHE_PATH,
)
from core.config import (
    DEFAULT_LLMS_TXT_URL as CONFIG_DEFAULT_LLMS_TXT_URL,
)
from core.config import (
    DEFAULT_MAX_FETCH_URLS as CONFIG_DEFAULT_MAX_FETCH_URLS,
)
from core.config import (
    DEFAULT_URL_STRATEGY_DEDUPLICATE as CONFIG_DEFAULT_URL_STRATEGY_DEDUPLICATE,
)
from core.config import (
    DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS as CONFIG_DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS,
)
from core.config import (
    DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT as CONFIG_DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
)
from core.config import (
    # URL 策略默认值
    DEFAULT_URL_STRATEGY_MAX_URLS as CONFIG_DEFAULT_URL_STRATEGY_MAX_URLS,
)
from core.config import (
    DEFAULT_URL_STRATEGY_NORMALIZE as CONFIG_DEFAULT_URL_STRATEGY_NORMALIZE,
)
from core.config import (
    DEFAULT_URL_STRATEGY_PREFER_CHANGELOG as CONFIG_DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
)
from core.config import (
    DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS as CONFIG_DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS,
)
from core.execution_policy import (
    ExecutionDecision,
    SideEffectPolicy,
    compute_decision_inputs,
    compute_message_dedup_key,
    compute_side_effects,
    validate_requested_mode_invariant,
)
from core.output_contract import (
    IterateResultFields,
    prepare_cooldown_message,
)
from core.project_workspace import (
    ProjectInfo,
    ProjectState,
    ReferenceProject,
    TaskAnalysis,
    WorkspacePreparationResult,
    prepare_workspace,
)
from cursor.client import CursorAgentConfig
from cursor.mcp import MCPManager
from cursor.cloud_client import CloudAuthConfig, CloudClientFactory
from cursor.executor import ExecutionMode
from knowledge import (
    ContentCleaner,
    FetchConfig,
    KnowledgeManager,
    KnowledgeStorage,
    WebFetcher,
)
from knowledge.doc_sources import (
    DEFAULT_ALLOWED_DOC_URL_PREFIXES,
    load_core_docs,
    load_core_docs_with_fallback,
)
from knowledge.doc_url_strategy import (
    DocURLStrategyConfig,
    deduplicate_urls,
    normalize_url,
)
from knowledge.doc_url_strategy import (
    is_allowed_doc_url as _is_allowed_doc_url_with_config,
)
from knowledge.doc_url_strategy import (
    parse_llms_txt_urls as strategy_parse_llms_txt,
)
from knowledge.doc_url_strategy import (
    select_urls_to_fetch as strategy_select_urls,
)
from knowledge.fetcher import FetchMethod, FetchResult, UrlPolicy, sanitize_url_for_log
from knowledge.parser import ContentCleanMode
from knowledge.storage import ReadOnlyStorageError

# ============================================================
# 配置常量
# ============================================================

# 使用 core.config 中的默认值（确保唯一权威来源）
DEFAULT_CHANGELOG_URL = CONFIG_DEFAULT_CHANGELOG_URL
DEFAULT_MAX_FETCH_URLS = CONFIG_DEFAULT_MAX_FETCH_URLS
DEFAULT_FALLBACK_CORE_DOCS_COUNT = CONFIG_DEFAULT_FALLBACK_CORE_DOCS_COUNT
DEFAULT_LLMS_TXT_URL = CONFIG_DEFAULT_LLMS_TXT_URL
DEFAULT_LLMS_CACHE_PATH = CONFIG_DEFAULT_LLMS_CACHE_PATH

# 在线抓取策略默认值
DEFAULT_ALLOWED_PATH_PREFIXES = CONFIG_DEFAULT_ALLOWED_PATH_PREFIXES
DEFAULT_ALLOWED_URL_PREFIXES = CONFIG_DEFAULT_ALLOWED_URL_PREFIXES  # deprecated alias
DEFAULT_ALLOWED_DOMAINS = CONFIG_DEFAULT_ALLOWED_DOMAINS
DEFAULT_EXTERNAL_LINK_MODE = CONFIG_DEFAULT_EXTERNAL_LINK_MODE
DEFAULT_EXTERNAL_LINK_ALLOWLIST = CONFIG_DEFAULT_EXTERNAL_LINK_ALLOWLIST

# URL 策略默认值（与 config.yaml knowledge_docs_update.url_strategy 同步）
DEFAULT_URL_STRATEGY_MAX_URLS = CONFIG_DEFAULT_URL_STRATEGY_MAX_URLS
DEFAULT_URL_STRATEGY_PREFER_CHANGELOG = CONFIG_DEFAULT_URL_STRATEGY_PREFER_CHANGELOG
DEFAULT_URL_STRATEGY_DEDUPLICATE = CONFIG_DEFAULT_URL_STRATEGY_DEDUPLICATE
DEFAULT_URL_STRATEGY_NORMALIZE = CONFIG_DEFAULT_URL_STRATEGY_NORMALIZE
DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT = CONFIG_DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS = CONFIG_DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS

# [DEPRECATED] 最小保底文档 URL（仅作为最后 fallback 使用）
# 正式来源已迁移到 cursor_cli_docs.txt 和 cursor_agent_docs.txt 文件
# 通过 load_core_docs_with_fallback() 加载，此列表将在后续版本中删除
CURSOR_DOC_URLS = [
    "https://cursor.com/cn/docs/cli/overview",  # CLI 概述
    "https://cursor.com/cn/docs/cli/using",  # CLI 使用
    "https://cursor.com/cn/docs/cli/reference/parameters",  # CLI 参数参考
]

# ============================================================
# Minimal Preset 配置覆盖
# ============================================================
# 当 --minimal 启用时，覆盖以下配置以实现最小副作用模式：
# - skip_online=True: 禁止在线检查（网络请求）
# - dry_run=True: 禁止文件写入和 Git 操作
# - skip_knowledge_init=True: 跳过知识库初始化（不创建目录）
# - skip_knowledge_context=True: 跳过知识库上下文加载
#
# 权威定义来源: core/execution_policy.py Side Effect Control Strategy Matrix
# minimal = skip_online + dry_run（禁止所有网络请求、文件写入、Git 操作）
MINIMAL_PRESET: dict[str, Any] = {
    "skip_online": True,
    "dry_run": True,  # minimal 模式强制 dry_run=True
    "skip_knowledge_init": True,
    "skip_knowledge_context": True,
}


def get_core_docs() -> list[str]:
    """获取核心文档 URL 列表

    优先从配置文件（cursor_cli_docs.txt、cursor_agent_docs.txt）加载，
    如果加载失败则使用 CURSOR_DOC_URLS 作为 fallback。

    Returns:
        核心文档 URL 列表（去重、过滤后）
    """
    return load_core_docs_with_fallback(
        legacy_urls=CURSOR_DOC_URLS,
    )


# llms.txt 来源（在线优先，写入缓存，失败回退缓存 -> 仓库文件）
# 运行时通过 resolve_docs_source_config 获取最终值
LLMS_TXT_URL = DEFAULT_LLMS_TXT_URL
project_root = Path(__file__).parent.parent
LLMS_TXT_CACHE_PATH = project_root / DEFAULT_LLMS_CACHE_PATH
LLMS_TXT_CACHE_PATH_RESOLVED = LLMS_TXT_CACHE_PATH  # 别名保持向后兼容
LLMS_TXT_LOCAL_FALLBACK = project_root / "cursor_docs_full.txt"

# 允许的文档 URL 前缀（完整 URL 前缀，含 scheme/host）
# 用途：is_allowed_doc_url 包装函数进行前缀匹配过滤
# 格式：完整 URL 前缀，如 "https://cursor.com/docs"
#
# 配置项区分说明：
# - docs_source.allowed_doc_url_prefixes: 完整 URL 前缀（含域名），用于 load_core_docs 过滤
# - fetch_policy.allowed_path_prefixes: 路径前缀（不含域名），用于抓取策略过滤
# - url_strategy.allowed_url_prefixes: 完整 URL 前缀，用于 URL 选择与过滤
# - url_strategy.allowed_domains: 域名白名单，用于 URL 选择时的域名过滤
#
# 权威来源：knowledge.doc_sources.DEFAULT_ALLOWED_DOC_URL_PREFIXES
ALLOWED_DOC_URL_PREFIXES = DEFAULT_ALLOWED_DOC_URL_PREFIXES

# [DEPRECATED] 旧名别名（向后兼容，避免外部导入/测试立即破坏）
# 请使用 ALLOWED_DOC_URL_PREFIXES，此别名将在未来版本中移除
ALLOWED_DOC_DOMAINS = ALLOWED_DOC_URL_PREFIXES


def _derive_allowed_domains_from_prefixes(prefixes: list[str]) -> list[str]:
    """从 URL 前缀列表推导域名白名单

    从 ALLOWED_DOC_URL_PREFIXES 这样的 URL 前缀列表中提取 netloc（域名），
    并去重为策略层可用的域名白名单。

    Args:
        prefixes: URL 前缀列表，如 ["https://cursor.com/docs", ...]

    Returns:
        去重后的域名列表，如 ["cursor.com"]

    Examples:
        >>> _derive_allowed_domains_from_prefixes([
        ...     "https://cursor.com/cn/docs",
        ...     "https://cursor.com/docs",
        ... ])
        ['cursor.com']
    """
    from urllib.parse import urlparse

    domains: set[str] = set()
    for prefix in prefixes:
        try:
            parsed = urlparse(prefix)
            if parsed.netloc:
                domains.add(parsed.netloc.lower())
        except Exception:
            continue
    # 返回排序后的列表以确保确定性
    return sorted(domains)


# 从 ALLOWED_DOC_URL_PREFIXES 推导的域名白名单
# 用途：策略层 DocURLStrategyConfig.allowed_domains 的备选值
# 格式：纯域名列表，如 ["cursor.com"]
#
# 与 ALLOWED_DOC_URL_PREFIXES 的区别：
# - ALLOWED_DOC_URL_PREFIXES: 完整 URL 前缀，精确匹配路径（优先级更高）
# - ALLOWED_DOC_URL_PREFIXES_NETLOC: 仅域名，允许该域名下所有路径
ALLOWED_DOC_URL_PREFIXES_NETLOC = _derive_allowed_domains_from_prefixes(ALLOWED_DOC_URL_PREFIXES)

# [DEPRECATED] 旧名别名（向后兼容，避免外部导入/测试立即破坏）
# 请使用 ALLOWED_DOC_URL_PREFIXES_NETLOC，此别名将在未来版本中移除
ALLOWED_DOC_DOMAINS_NETLOC = ALLOWED_DOC_URL_PREFIXES_NETLOC


# normalize_url, deduplicate_urls 已从 knowledge.doc_url_strategy 导入

# 默认文档 URL 策略配置（用于 is_allowed_doc_url 包装函数）
# 使用 allowed_url_prefixes 进行前缀匹配（优先级高于 allowed_domains）
_DEFAULT_DOC_URL_CONFIG = DocURLStrategyConfig(
    allowed_url_prefixes=ALLOWED_DOC_URL_PREFIXES,  # 使用本地定义的 URL 前缀白名单
    allowed_domains=ALLOWED_DOC_URL_PREFIXES_NETLOC,  # 域名白名单作为备选
    exclude_patterns=[
        r".*\.(png|jpg|jpeg|gif|svg|ico|css|js|woff|woff2|ttf|eot)$",
    ],
)


def is_allowed_doc_url(url: str, base_url: str = "https://cursor.com") -> bool:
    """检查 URL 是否属于允许的文档 URL 前缀

    包装 knowledge.doc_url_strategy.is_allowed_doc_url，
    使用 ALLOWED_DOC_URL_PREFIXES 进行前缀匹配检查。

    Args:
        url: 待检查的 URL
        base_url: 基础 URL（用于规范化相对路径）

    Returns:
        True 如果 URL 属于允许的 URL 前缀
    """
    # 委托给 doc_url_strategy 模块的 is_allowed_doc_url 函数
    return _is_allowed_doc_url_with_config(url, _DEFAULT_DOC_URL_CONFIG, base_url)


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
    "非并行",
    "不并行",
    "串行",
    "协程",
    "单进程",
    "basic",
    "no-mp",
    "no_mp",
    "禁用多进程",
    "禁用mp",
    "关闭多进程",
]


# ============================================================
# 颜色输出
# ============================================================


# 显式导出核心解析函数，供测试与外部调用使用
__all__ = [
    "CLOUD_PREFIX",
    "is_cloud_request",
    "strip_cloud_prefix",
]


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
    links: list[dict[str, str]] = field(default_factory=list)  # 从 changelog 中提取的链接


@dataclass
class UrlSelectionLog:
    """URL 选择阶段的结构化日志（供上层汇总和诊断）

    Attributes:
        source: 来源阶段标识 (changelog_analysis/url_selection/knowledge_update)
        input_counts: 各来源 URL 输入数量
        output_count: 最终选择的 URL 数量
        filtered_count: 被过滤/去重的 URL 数量
        rejection_summary: 拒绝原因汇总 {policy_type: count}
        selected_urls: 选中的 URL 列表（已截断敏感部分）
        duration_ms: 处理耗时（毫秒）
        fetch_policy_filtered: apply_fetch_policy 过滤的 URL 及原因
        external_links_recorded: 被记录（但未抓取）的外链
    """

    source: str
    input_counts: dict[str, int] = field(default_factory=dict)
    output_count: int = 0
    filtered_count: int = 0
    rejection_summary: dict[str, int] = field(default_factory=dict)
    selected_urls: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    # apply_fetch_policy 过滤结果
    fetch_policy_filtered: list[dict[str, str]] = field(default_factory=list)
    external_links_recorded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        result = {
            "source": self.source,
            "input_counts": self.input_counts,
            "output_count": self.output_count,
            "filtered_count": self.filtered_count,
            "rejection_summary": self.rejection_summary,
            "selected_urls": self.selected_urls,
            "duration_ms": round(self.duration_ms, 2),
        }
        # 仅在有数据时包含 fetch_policy 相关字段
        if self.fetch_policy_filtered:
            result["fetch_policy_filtered"] = self.fetch_policy_filtered
        if self.external_links_recorded:
            result["external_links_recorded"] = self.external_links_recorded
        return result

    def log_structured(self) -> None:
        """输出结构化日志"""
        logger.info(
            f"[URL_SELECTION] source={self.source} | "
            f"input={sum(self.input_counts.values())} "
            f"({', '.join(f'{k}={v}' for k, v in self.input_counts.items())}) | "
            f"output={self.output_count} | filtered={self.filtered_count} | "
            f"rejections={self.rejection_summary}"
        )
        # 记录 fetch_policy 过滤信息
        if self.fetch_policy_filtered:
            logger.info(
                f"[FETCH_POLICY] filtered={len(self.fetch_policy_filtered)} | "
                f"external_recorded={len(self.external_links_recorded)}"
            )


@dataclass
class ChangelogAnalysisLog:
    """Changelog 分析阶段的结构化日志

    Attributes:
        changelog_url: changelog URL（已截断敏感部分）
        fetch_success: 是否成功获取
        fetch_method: 使用的获取方式
        content_length: 内容长度（字符）
        quality_score: 内容质量评分
        entry_count: 解析的条目数量
        fingerprint: 内容指纹（前 8 位）
        baseline_match: 是否与基线匹配
        links_extracted: 提取的链接数量 {allowed: n, external: n}
        duration_ms: 处理耗时（毫秒）
    """

    changelog_url: str
    fetch_success: bool = False
    fetch_method: str = ""
    content_length: int = 0
    quality_score: float = 0.0
    entry_count: int = 0
    fingerprint: str = ""
    baseline_match: Optional[bool] = None
    links_extracted: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "changelog_url": self.changelog_url,
            "fetch_success": self.fetch_success,
            "fetch_method": self.fetch_method,
            "content_length": self.content_length,
            "quality_score": round(self.quality_score, 3),
            "entry_count": self.entry_count,
            "fingerprint": self.fingerprint,
            "baseline_match": self.baseline_match,
            "links_extracted": self.links_extracted,
            "duration_ms": round(self.duration_ms, 2),
        }

    def log_structured(self) -> None:
        """输出结构化日志"""
        logger.info(
            f"[CHANGELOG_ANALYSIS] url={self.changelog_url} | "
            f"success={self.fetch_success} | method={self.fetch_method} | "
            f"content_len={self.content_length} | quality={self.quality_score:.3f} | "
            f"entries={self.entry_count} | fingerprint={self.fingerprint} | "
            f"baseline_match={self.baseline_match} | links={self.links_extracted}"
        )


@dataclass
class UpdateAnalysis:
    """更新分析结果"""

    has_updates: bool = False
    entries: list[ChangelogEntry] = field(default_factory=list)
    new_features: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    related_doc_urls: list[str] = field(default_factory=list)
    changelog_links: list[str] = field(
        default_factory=list
    )  # 从 changelog 中提取并归一化的文档链接（仅允许抓取范围内的 URL）
    external_links: list[str] = field(default_factory=list)  # 从 changelog 中提取的外部链接（不在允许抓取范围内）
    summary: str = ""
    raw_content: str = ""
    fingerprint: str = ""  # 清洗后内容的 fingerprint（用于保存到 storage 作为 baseline）
    # 结构化日志（可选）
    analysis_log: Optional[ChangelogAnalysisLog] = None


@dataclass
class IterationContext:
    """迭代上下文"""

    user_requirement: str = ""
    update_analysis: Optional[UpdateAnalysis] = None
    knowledge_context: list[dict] = field(default_factory=list)
    iteration_goal: str = ""
    dry_run: bool = False
    # 目录驱动工程创建/扩展相关
    project_info: Optional[ProjectInfo] = None
    reference_projects: list[ReferenceProject] = field(default_factory=list)
    workspace_preparation: Optional[WorkspacePreparationResult] = None
    task_analysis: Optional[TaskAnalysis] = None
    # 迭代助手上下文（.iteration + Engram + 规则摘要）
    iteration_assistant: Optional["IterationAssistantContext"] = None


@dataclass
class EngramStatus:
    """Engram MCP 可用性状态"""

    available: bool = False
    verified: bool = False
    server_name: str = ""
    tools: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "verified": self.verified,
            "server_name": self.server_name,
            "tools": self.tools,
            "reason": self.reason,
        }


@dataclass
class IterationDocs:
    """迭代文档内容"""

    iteration_id: str
    plan_path: Optional[str] = None
    regression_path: Optional[str] = None
    readme_path: Optional[str] = None
    plan_content: str = ""
    regression_content: str = ""
    readme_content: str = ""

    def to_dict(self, max_chars: int = 2000) -> dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "plan_path": self.plan_path,
            "regression_path": self.regression_path,
            "readme_path": self.readme_path,
            "plan_excerpt": _truncate_text(self.plan_content, max_chars),
            "regression_excerpt": _truncate_text(self.regression_content, max_chars),
            "readme_excerpt": _truncate_text(self.readme_content, max_chars),
        }


@dataclass
class IterationAssistantContext:
    """迭代助手上下文"""

    iteration_dir: Optional[str] = None
    iteration_id: Optional[str] = None
    git_policy: str = "absent"  # tracked/ignored/untracked/absent
    docs: Optional[IterationDocs] = None
    rules_summary: str = ""
    engram: EngramStatus = field(default_factory=EngramStatus)
    bootstrap_performed: bool = False
    bootstrap_reason: str = ""

    def to_dict(self, max_doc_chars: int = 2000, max_rules_chars: int = 2000) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "iteration_dir": self.iteration_dir,
            "iteration_id": self.iteration_id,
            "git_policy": self.git_policy,
            "rules_summary": _truncate_text(self.rules_summary, max_rules_chars),
            "engram": self.engram.to_dict(),
            "bootstrap_performed": self.bootstrap_performed,
            "bootstrap_reason": self.bootstrap_reason,
        }
        if self.docs:
            payload["docs"] = self.docs.to_dict(max_doc_chars)
        return payload


def _truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def _read_text_safe(path: Path) -> str:
    try:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"读取文件失败: {path} - {e}")
        return ""


def detect_iteration_dir(working_directory: Path) -> Optional[Path]:
    iteration_dir = working_directory / ".iteration"
    if iteration_dir.exists() and iteration_dir.is_dir():
        return iteration_dir
    return None


def select_iteration_id(iteration_dir: Path, explicit_id: Optional[str] = None) -> str:
    if explicit_id:
        return str(explicit_id)
    # 选择最大纯数字目录名
    numeric_ids: list[int] = []
    for child in iteration_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            try:
                numeric_ids.append(int(child.name))
            except ValueError:
                continue
    if numeric_ids:
        return str(max(numeric_ids))
    # 稳定一致的默认值
    return "1"


def _bootstrap_iteration_docs(
    iteration_dir: Path,
    iteration_id: str,
    allow_bootstrap: bool,
    allow_write: bool,
) -> tuple[Optional[Path], Optional[Path], bool, str]:
    if not allow_bootstrap:
        return None, None, False, "bootstrap_disabled"
    if not allow_write:
        return None, None, False, "side_effect_policy_disallow_write"

    plan_path = iteration_dir / iteration_id / "plan.md"
    regression_path = iteration_dir / iteration_id / "regression.md"
    created = False

    try:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if not plan_path.exists():
            plan_path.write_text("# 计划\n\n", encoding="utf-8")
            created = True
        if not regression_path.exists():
            regression_path.write_text("# 回归与进度\n\n", encoding="utf-8")
            created = True
        return plan_path, regression_path, created, "ok"
    except Exception as e:
        logger.warning(f"初始化 .iteration 文档失败: {e}")
        return None, None, False, f"bootstrap_failed:{e}"


def load_iteration_docs(iteration_dir: Path, iteration_id: str) -> IterationDocs:
    plan_path = iteration_dir / iteration_id / "plan.md"
    regression_path = iteration_dir / iteration_id / "regression.md"
    readme_path = iteration_dir / "README.md"

    return IterationDocs(
        iteration_id=iteration_id,
        plan_path=str(plan_path) if plan_path.exists() else None,
        regression_path=str(regression_path) if regression_path.exists() else None,
        readme_path=str(readme_path) if readme_path.exists() else None,
        plan_content=_read_text_safe(plan_path),
        regression_content=_read_text_safe(regression_path),
        readme_content=_read_text_safe(readme_path),
    )


def detect_iteration_git_policy(working_directory: Path, iteration_dir: Optional[Path]) -> str:
    if iteration_dir is None or not iteration_dir.exists():
        return "absent"

    # 检查是否为 Git 仓库
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return "untracked"
    except Exception:
        return "untracked"

    # 检查是否被忽略
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(iteration_dir)],
            cwd=working_directory,
            check=False,
        )
        if ignored.returncode == 0:
            return "ignored"
    except Exception:
        pass

    # 检查是否有被跟踪的文件
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--", str(iteration_dir)],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.stdout.strip():
            return "tracked"
    except Exception:
        pass

    return "untracked"


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def load_target_project_rules_summary(working_directory: Path, max_chars: int = 2000) -> str:
    sections: list[str] = []

    agents_path = working_directory / "AGENTS.md"
    if agents_path.exists():
        content = _read_text_safe(agents_path)
        # 取前 80 行作为摘要
        agents_excerpt = "\n".join(content.splitlines()[:80])
        sections.append("## AGENTS.md (节选)\n" + agents_excerpt)

    rules_dir = working_directory / ".cursor" / "rules"
    if rules_dir.exists() and rules_dir.is_dir():
        rule_lines = ["## .cursor/rules 摘要"]
        for rule_file in sorted(rules_dir.glob("*.mdc")):
            content = _read_text_safe(rule_file)
            frontmatter = _parse_frontmatter(content)
            desc = frontmatter.get("description", "")
            always_apply = frontmatter.get("alwaysApply", "")
            globs = frontmatter.get("globs", "")
            rule_lines.append(
                f"- {rule_file.name}: {desc}"
                + (f" | alwaysApply={always_apply}" if always_apply else "")
                + (f" | globs={globs}" if globs else "")
            )
        sections.append("\n".join(rule_lines))

    summary = "\n\n".join(sections).strip()
    return _truncate_text(summary, max_chars)


async def detect_engram_mcp(agent_path: str = "agent") -> EngramStatus:
    manager = MCPManager(agent_path=agent_path)
    if not manager.check_available():
        return EngramStatus(available=False, verified=False, reason="agent mcp 不可用")

    servers = await manager.list_servers()
    candidates = [s for s in servers if "engram" in s.name.lower()]
    if not candidates:
        return EngramStatus(available=False, verified=False, reason="未发现 Engram MCP 服务器")

    # 优先使用第一个候选
    server = candidates[0]
    tools = await manager.list_tools(server.identifier)
    tool_names = [t.name for t in tools if t.name]

    if tool_names:
        return EngramStatus(
            available=True,
            verified=True,
            server_name=server.identifier,
            tools=tool_names,
            reason="tools/list 返回非空",
        )

    return EngramStatus(
        available=False,
        verified=False,
        server_name=server.identifier,
        tools=[],
        reason="tools/list 返回为空或解析失败",
    )


async def build_iteration_assistant_context(
    working_directory: Path,
    side_effect_policy: "SideEffectPolicy",
    iteration_id: Optional[str] = None,
    iteration_source: str = "auto",
    iteration_git_policy: str = "auto",
    allow_bootstrap: bool = True,
    agent_path: str = "agent",
) -> IterationAssistantContext:
    context = IterationAssistantContext()

    use_iteration = iteration_source in ("auto", "iteration")
    use_engram = iteration_source in ("auto", "engram")

    if use_iteration:
        iteration_dir = detect_iteration_dir(working_directory)
        context.iteration_dir = str(iteration_dir) if iteration_dir else None
        if iteration_dir:
            context.iteration_id = select_iteration_id(iteration_dir, iteration_id)
            # Git 策略检测（可被覆盖）
            if iteration_git_policy != "auto":
                context.git_policy = iteration_git_policy
            else:
                context.git_policy = detect_iteration_git_policy(working_directory, iteration_dir)

            # 可选 bootstrap（仅在允许写入时）
            plan_path, regression_path, created, reason = _bootstrap_iteration_docs(
                iteration_dir=iteration_dir,
                iteration_id=context.iteration_id,
                allow_bootstrap=allow_bootstrap,
                allow_write=side_effect_policy.allow_directory_create and side_effect_policy.allow_file_write,
            )
            context.bootstrap_performed = created
            context.bootstrap_reason = reason

            context.docs = load_iteration_docs(iteration_dir, context.iteration_id)
        else:
            context.git_policy = "absent"

    if use_engram:
        context.engram = await detect_engram_mcp(agent_path=agent_path)
    else:
        context.engram = EngramStatus(
            available=False,
            verified=False,
            reason=f"iteration_source={iteration_source}",
        )

    context.rules_summary = load_target_project_rules_summary(working_directory)
    return context


# ============================================================
# 参数解析
# ============================================================

# 使用统一的 parse_max_iterations 函数（从 core.config 导入）


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    默认值优先级:
    1. 命令行参数（最高）
    2. config.yaml 中的配置
    3. 代码硬编码的默认值（最低）
    """
    # 从配置管理器获取默认值
    from core.config import get_config

    config = get_config()

    # 系统配置默认值
    default_max_iterations = config.system.max_iterations
    default_workers = config.system.worker_pool_size

    # Cloud Agent 配置默认值
    default_cloud_timeout = config.cloud_agent.timeout
    default_cloud_auth_timeout = config.cloud_agent.auth_timeout
    default_execution_mode = config.cloud_agent.execution_mode

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
        "--minimal",
        action="store_true",
        help=(
            "最小副作用模式（等价于 --skip-online --dry-run 组合）。"
            "副作用控制: network=禁止, disk_write=禁止, git=禁止。"
            "不在线 fetch、不创建目录、不执行 Orchestrator.run。"
            "如需上下文仅读取仓库内置 cursor_docs_full.txt 或已存在 KB（只读）。"
            "适用场景: 纯分析/预览/CI 验证等无副作用需求的场景"
        ),
    )

    # 文档源配置参数（tri-state 设计：None=未指定，使用 config.yaml）
    docs_source_group = parser.add_argument_group("文档源配置", "控制知识库更新时的文档获取策略")
    docs_source_group.add_argument(
        "--changelog-url",
        type=str,
        default=None,
        help=f"Changelog URL (默认: {DEFAULT_CHANGELOG_URL}，来自 config.yaml)",
    )
    docs_source_group.add_argument(
        "--max-fetch-urls",
        type=int,
        default=None,
        help=f"单次更新最多抓取的 URL 数量 (默认: {DEFAULT_MAX_FETCH_URLS}，来自 config.yaml)",
    )
    docs_source_group.add_argument(
        "--fallback-core-docs-count",
        type=int,
        default=None,
        help=f"回退核心文档数量 (默认: {DEFAULT_FALLBACK_CORE_DOCS_COUNT}，来自 config.yaml)",
    )
    docs_source_group.add_argument(
        "--llms-txt-url",
        type=str,
        default=None,
        help=f"llms.txt 在线来源 URL (默认: {DEFAULT_LLMS_TXT_URL}，来自 config.yaml)",
    )
    docs_source_group.add_argument(
        "--llms-cache-path",
        type=str,
        default=None,
        help=f"llms.txt 本地缓存路径 (默认: {DEFAULT_LLMS_CACHE_PATH}，来自 config.yaml)",
    )
    docs_source_group.add_argument(
        "--allowed-doc-url-prefixes",
        type=str,
        default=None,
        help=(
            "允许的文档 URL 前缀（完整 URL 前缀，含 scheme/host），逗号分隔。"
            "用于 load_core_docs 过滤核心文档。"
            "格式：https://cursor.com/docs,https://cursor.com/cn/docs "
            "注意：与 --allowed-path-prefixes（路径前缀，不含域名）格式不同。"
            "(默认: https://cursor.com/cn/docs,https://cursor.com/docs,...，"
            "来自 config.yaml docs_source.allowed_doc_url_prefixes)"
        ),
    )

    # 迭代上下文配置
    iteration_group = parser.add_argument_group("迭代上下文", "控制 .iteration 与 Engram MCP 的集成")
    iteration_group.add_argument(
        "--iteration-id",
        type=str,
        default=None,
        help="显式指定迭代 ID（默认自动选择 .iteration 下最大数字目录）",
    )
    iteration_group.add_argument(
        "--iteration-source",
        type=str,
        choices=["auto", "iteration", "engram", "none"],
        default="auto",
        help="迭代上下文来源（默认 auto：自动探测 .iteration 与 Engram MCP）",
    )
    iteration_group.add_argument(
        "--iteration-git-policy",
        type=str,
        choices=["auto", "tracked", "untracked", "ignored"],
        default="auto",
        help="强制指定 .iteration 的 Git 策略（默认 auto 自动识别）",
    )
    iteration_group.add_argument(
        "--no-iteration-bootstrap",
        action="store_true",
        help="禁用自动初始化 .iteration/<ITER_ID>/plan.md 与 regression.md",
    )

    # 在线抓取策略参数（tri-state 设计）
    # 术语说明：fetch_policy 控制哪些 URL 可以被抓取（网络请求层面）
    # 参见 knowledge/doc_url_strategy.py 模块契约的【术语表】
    fetch_policy_group = parser.add_argument_group(
        "在线抓取策略 (fetch_policy)",
        "控制知识库更新时的在线文档抓取策略（保持最小抓取面）。"
        "与 url_strategy 不同：fetch_policy 决定是否发起抓取请求，url_strategy 决定如何选择/过滤 URL",
    )
    fetch_policy_group.add_argument(
        "--allowed-path-prefixes",
        type=str,
        dest="allowed_path_prefixes",
        default=None,
        help=(
            f"允许抓取的 URL 路径前缀（不含 scheme/host/域名），逗号分隔。"
            f"格式：路径前缀，如 'docs,cn/docs'。"
            f"用于 fetch_policy 层控制哪些路径可被抓取。"
            f"注意：与 --allowed-doc-url-prefixes（完整 URL 前缀，含域名）格式不同。"
            f"注意：与 --url-allowed-prefixes（url_strategy 的完整 URL 前缀）用途不同。"
            f"(默认: {','.join(DEFAULT_ALLOWED_PATH_PREFIXES)}，"
            f"来自 config.yaml docs_source.fetch_policy.allowed_path_prefixes)"
        ),
    )
    # [DEPRECATED] 保留旧参数作为向后兼容别名
    fetch_policy_group.add_argument(
        "--allowed-url-prefixes",
        type=str,
        dest="allowed_url_prefixes_deprecated",
        default=None,
        help=(
            "[DEPRECATED] 已废弃，请使用 --allowed-path-prefixes。"
            "允许抓取的 URL 路径前缀（不含域名），逗号分隔。"
            f"(默认: {','.join(DEFAULT_ALLOWED_PATH_PREFIXES)})"
        ),
    )
    fetch_policy_group.add_argument(
        "--allowed-domains",
        type=str,
        default=None,
        help=(
            "fetch_policy 的域名白名单，逗号分隔。空值表示仅允许主域名。"
            "支持子域名匹配，如 'cursor.com' 可匹配 api.cursor.com "
            "(默认: 空，来自 config.yaml fetch_policy)"
        ),
    )
    fetch_policy_group.add_argument(
        "--external-link-mode",
        type=str,
        choices=["record_only", "skip_all", "fetch_allowlist"],
        default=None,
        help=(
            f"外链处理模式: record_only=仅记录不抓取, skip_all=跳过, "
            f"fetch_allowlist=抓取白名单 (默认: {DEFAULT_EXTERNAL_LINK_MODE}，来自 config.yaml fetch_policy)"
        ),
    )
    fetch_policy_group.add_argument(
        "--external-link-allowlist",
        type=str,
        default=None,
        help=("外链允许白名单，逗号分隔。仅 fetch_allowlist 模式下生效 (默认: 空，来自 config.yaml fetch_policy)"),
    )
    fetch_policy_group.add_argument(
        "--enforce-path-prefixes",
        action="store_true",
        dest="enforce_path_prefixes",
        default=None,
        help=(
            "启用内链路径前缀检查（阶段 4 gate）。"
            "启用后，内链必须匹配 --allowed-path-prefixes 中的任一前缀才允许抓取。"
            "不匹配的内链将被拒绝并记录原因为 'internal_link_path_not_allowed'。"
            "(默认: False，来自 config.yaml fetch_policy.enforce_path_prefixes)"
        ),
    )
    fetch_policy_group.add_argument(
        "--no-enforce-path-prefixes",
        action="store_false",
        dest="enforce_path_prefixes",
        help=("禁用内链路径前缀检查（Phase A 行为，保持向后兼容）。"),
    )

    # 文档 URL 选择策略参数（tri-state 设计：None=未指定，使用 config.yaml）
    # 术语说明：url_strategy 控制如何过滤/排序/选择 URL（数据处理层面）
    # 参见 knowledge/doc_url_strategy.py 模块契约的【术语表】
    url_strategy_group = parser.add_argument_group(
        "URL 选择策略 (url_strategy)",
        "控制文档 URL 的过滤、规范化、去重和优先级排序。"
        "与 fetch_policy 不同：url_strategy 决定如何选择/过滤 URL，fetch_policy 决定是否发起抓取请求",
    )
    url_strategy_group.add_argument(
        "--url-allowed-domains",
        type=str,
        action="append",
        dest="url_allowed_domains",
        default=None,
        metavar="DOMAIN",
        help=(
            "url_strategy 的域名白名单（支持重复指定或逗号分隔）。"
            "用于 URL 选择与过滤时的域名匹配（仅当 --url-allowed-prefixes 为空时生效）。"
            "支持子域名匹配，如 'cursor.com' 可匹配 api.cursor.com。"
            "示例: --url-allowed-domains cursor.com --url-allowed-domains docs.cursor.com "
            "或 --url-allowed-domains cursor.com,docs.cursor.com "
            "(默认: cursor.com，来自 config.yaml url_strategy.allowed_domains)"
        ),
    )
    url_strategy_group.add_argument(
        "--url-exclude-patterns",
        type=str,
        action="append",
        dest="url_exclude_patterns",
        default=None,
        metavar="PATTERN",
        help=(
            "URL 过滤的排除模式（正则表达式，支持重复指定或逗号分隔）。"
            "示例: --url-exclude-patterns '.*\\.pdf$' --url-exclude-patterns '.*#.*' "
            "(默认: 排除图片/PDF/锚点链接，来自 config.yaml url_strategy.exclude_patterns)"
        ),
    )
    url_strategy_group.add_argument(
        "--prefer-changelog",
        action="store_true",
        dest="url_prefer_changelog",
        default=None,
        help=f"URL 选择时优先处理 Changelog 文档 (默认: {DEFAULT_URL_STRATEGY_PREFER_CHANGELOG}，来自 config.yaml url_strategy)",
    )
    url_strategy_group.add_argument(
        "--no-prefer-changelog",
        action="store_false",
        dest="url_prefer_changelog",
        help="URL 选择时不优先处理 Changelog 文档",
    )
    url_strategy_group.add_argument(
        "--url-deduplicate",
        action="store_true",
        dest="url_deduplicate",
        default=None,
        help=f"启用 URL 去重（基于规范化后的 URL 进行去重）(默认: {DEFAULT_URL_STRATEGY_DEDUPLICATE}，来自 config.yaml url_strategy)",
    )
    url_strategy_group.add_argument(
        "--no-url-deduplicate",
        action="store_false",
        dest="url_deduplicate",
        help="禁用 URL 去重（保留所有变体）",
    )
    url_strategy_group.add_argument(
        "--url-normalize",
        action="store_true",
        dest="url_normalize",
        default=None,
        help=f"启用 URL 规范化（移除锚点、统一斜杠、scheme/host 小写化）(默认: {DEFAULT_URL_STRATEGY_NORMALIZE}，来自 config.yaml url_strategy)",
    )
    url_strategy_group.add_argument(
        "--no-url-normalize",
        action="store_false",
        dest="url_normalize",
        help="禁用 URL 规范化",
    )
    url_strategy_group.add_argument(
        "--keyword-boost-weight",
        type=float,
        dest="url_keyword_boost_weight",
        default=None,
        metavar="WEIGHT",
        help=(
            f"URL 选择时关键词匹配的权重增益（0.0-2.0，1.0=无增益，2.0=双倍权重）"
            f"(默认: {DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT}，来自 config.yaml url_strategy)"
        ),
    )
    url_strategy_group.add_argument(
        "--url-allowed-prefixes",
        type=str,
        action="append",
        dest="url_allowed_prefixes",
        default=None,
        metavar="PREFIX",
        help=(
            "url_strategy 的完整 URL 前缀列表（支持重复指定或逗号分隔）。"
            "用于 URL 选择与过滤时的精确前缀匹配（优先级高于 --url-allowed-domains）。"
            "格式：完整 URL 前缀（含 scheme/host），如 'https://cursor.com/docs'。"
            "注意：与 --allowed-path-prefixes（fetch_policy 的路径前缀，不含域名）格式和用途不同。"
            "注意：与 --allowed-doc-url-prefixes（docs_source 的完整 URL 前缀）用途不同。"
            "示例: --url-allowed-prefixes https://cursor.com/docs "
            "--url-allowed-prefixes https://cursor.com/cn/docs "
            "或 --url-allowed-prefixes 'https://cursor.com/docs,https://cursor.com/cn/docs' "
            "(默认: 空列表，回退到 --url-allowed-domains 过滤；"
            "来自 config.yaml url_strategy.allowed_url_prefixes)"
        ),
    )
    url_strategy_group.add_argument(
        "--url-max-urls",
        type=int,
        dest="url_max_urls",
        default=None,
        metavar="COUNT",
        help=(
            f"URL 选择的最大返回 URL 数量 "
            f"(默认: {DEFAULT_URL_STRATEGY_MAX_URLS}，来自 config.yaml url_strategy.max_urls)"
        ),
    )
    url_strategy_group.add_argument(
        "--url-fallback-core-docs-count",
        type=int,
        dest="url_fallback_core_docs_count",
        default=None,
        metavar="COUNT",
        help=(
            f"当其他来源不足时，URL 选择从 core_docs 补充的数量 "
            f"(默认: {DEFAULT_FALLBACK_CORE_DOCS_COUNT}，来自 config.yaml url_strategy.fallback_core_docs_count)"
        ),
    )

    side_effect_group.add_argument(
        "--dry-run",
        action="store_true",
        help="仅分析不执行，不修改任何文件（允许网络请求用于分析，禁止文件写入和Git操作）",
    )

    # tri-state: default=None 表示未指定，运行时从 config.yaml 读取
    parser.add_argument(
        "--max-iterations",
        type=str,
        default=None,
        help=f"最大迭代次数 (默认: {default_max_iterations}，来自 config.yaml system.max_iterations；使用 MAX 或 -1 表示无限迭代)",
    )

    # tri-state: default=None 表示未指定，运行时从 config.yaml 读取
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Worker 池大小 (默认: {default_workers}，来自 config.yaml system.worker_pool_size)",
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
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出（DEBUG 级别日志）",
    )
    log_verbosity.add_argument(
        "-q",
        "--quiet",
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

    # 编排器选择参数 - tri-state (None=未指定，使用 config.yaml/resolve_orchestrator_settings 默认)
    # 注意：当 --execution-mode=auto/cloud 时，系统会强制使用 basic 编排器
    orchestrator_group = parser.add_mutually_exclusive_group()
    orchestrator_group.add_argument(
        "--orchestrator",
        type=str,
        choices=["mp", "basic"],
        default=None,
        help="编排器类型: mp=多进程, basic=协程模式 (默认: mp；execution_mode=auto/cloud 时强制 basic)",
    )
    orchestrator_group.add_argument(
        "--no-mp",
        action="store_true",
        dest="no_mp",
        default=None,
        help="禁用多进程编排器，使用 basic 协程编排器 (execution_mode=auto/cloud 时自动生效)",
    )

    # 执行模式参数 - tri-state (None=未指定，使用 config.yaml)
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["cli", "auto", "cloud"],
        default=None,
        help=(
            f"执行模式: cli=本地CLI, auto=自动选择(Cloud优先), cloud=强制Cloud "
            f"(默认: {default_execution_mode}，来自 config.yaml)。"
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

    # Cloud 认证配置
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
        help=f"Cloud 认证超时时间（秒，默认 {default_cloud_auth_timeout}，来自 config.yaml）",
    )

    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=None,
        help=f"Cloud 执行超时时间（秒，默认 {default_cloud_timeout}，来自 config.yaml cloud_agent.timeout）",
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

    # 检测用户是否显式设置了编排器参数
    # 通过检查 sys.argv 中是否存在相关参数来判断
    # 注意：检测列表与 run.py 保持一致
    args._orchestrator_user_set = any(arg in sys.argv for arg in ["--orchestrator", "--no-mp"])

    # 标记用户是否显式设置了 --directory 参数
    # tri-state 设计：None 表示未指定，运行时解析为当前目录
    args._directory_user_set = args.directory is not None

    # 解析为实际目录（未指定时使用当前目录）
    if args.directory is None:
        args.directory = "."

    return args


# ============================================================
# 文档源配置解析
# ============================================================


@dataclass
class ResolvedFetchPolicyConfig:
    """解析后的在线抓取策略配置

    使用 tri-state 设计：CLI 参数 > config.yaml > 默认值
    """

    allowed_path_prefixes: list[str]
    allowed_domains: list[str]
    external_link_mode: str
    external_link_allowlist: list[str]
    enforce_path_prefixes: bool = False  # 是否启用内链路径前缀检查（阶段 4 gate）


@dataclass
class ResolvedDocsSourceConfig:
    """解析后的文档源配置

    使用 tri-state 设计：CLI 参数 > config.yaml > 默认值
    """

    max_fetch_urls: int
    fallback_core_docs_count: int
    llms_txt_url: str
    llms_cache_path: str
    changelog_url: str
    fetch_policy: ResolvedFetchPolicyConfig
    allowed_doc_url_prefixes: list[str]


def _parse_comma_separated_list(value: Optional[str]) -> Optional[list[str]]:
    """解析逗号分隔的字符串为列表

    三态语义设计:
    - None: 未指定，使用 config.yaml 或默认值
    - []: 显式清空，不使用该过滤规则（允许所有或回退到其他规则）
    - ["a", "b"]: 显式指定，使用该列表进行过滤

    Args:
        value: 逗号分隔的字符串，或 None

    Returns:
        解析后的列表，None 表示未指定，[] 表示显式清空
    """
    if value is None:
        return None
    # 空字符串表示显式清空 -> 返回空列表 []
    if not value.strip():
        return []
    # 分割并去除空白，过滤空字符串
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_docs_source_config(args: argparse.Namespace | Any) -> ResolvedDocsSourceConfig:
    """解析文档源配置

    优先级: CLI 参数 > config.yaml > DEFAULT_* 常量

    Args:
        args: 命令行参数

    Returns:
        解析后的文档源配置
    """
    from core.config import get_config

    config = get_config()
    docs_source = config.knowledge_docs_update.docs_source
    fetch_policy = docs_source.fetch_policy

    # 使用 tri-state 设计：CLI 显式指定 > config.yaml > 默认值
    max_fetch_urls = args.max_fetch_urls if args.max_fetch_urls is not None else docs_source.max_fetch_urls
    fallback_core_docs_count = (
        args.fallback_core_docs_count
        if args.fallback_core_docs_count is not None
        else docs_source.fallback_core_docs_count
    )
    llms_txt_url = args.llms_txt_url if args.llms_txt_url is not None else docs_source.llms_txt_url
    llms_cache_path = args.llms_cache_path if args.llms_cache_path is not None else docs_source.llms_cache_path
    changelog_url = args.changelog_url if args.changelog_url is not None else docs_source.changelog_url

    # 解析在线抓取策略配置
    # CLI 参数为逗号分隔的字符串，需要解析为列表
    # 支持新参数 --allowed-path-prefixes 和旧参数 --allowed-url-prefixes（deprecated）
    cli_allowed_path_prefixes = _parse_comma_separated_list(getattr(args, "allowed_path_prefixes", None))
    cli_allowed_url_prefixes_deprecated = _parse_comma_separated_list(
        getattr(args, "allowed_url_prefixes_deprecated", None)
    )
    # 处理新旧 CLI 参数：新参数优先（使用统一警告机制，每类警告仅一次）
    if cli_allowed_path_prefixes is not None and cli_allowed_url_prefixes_deprecated is not None:
        _warn_deprecated_once(
            DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES_BOTH,
            "同时指定了 --allowed-path-prefixes 和 --allowed-url-prefixes。"
            "--allowed-path-prefixes 优先级更高，已忽略 --allowed-url-prefixes。"
            "请移除已废弃的 --allowed-url-prefixes 参数。",
        )
        cli_allowed_path_prefixes_final = cli_allowed_path_prefixes
    elif cli_allowed_path_prefixes is not None:
        cli_allowed_path_prefixes_final = cli_allowed_path_prefixes
    elif cli_allowed_url_prefixes_deprecated is not None:
        _warn_deprecated_once(
            DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES,
            "--allowed-url-prefixes 已废弃，请改用 --allowed-path-prefixes。旧参数将在后续版本中移除。",
        )
        cli_allowed_path_prefixes_final = cli_allowed_url_prefixes_deprecated
    else:
        cli_allowed_path_prefixes_final = None

    cli_allowed_domains = _parse_comma_separated_list(getattr(args, "allowed_domains", None))
    cli_external_link_allowlist = _parse_comma_separated_list(getattr(args, "external_link_allowlist", None))
    cli_external_link_mode = getattr(args, "external_link_mode", None)

    # 导入校验函数
    from knowledge.doc_url_strategy import (
        validate_external_link_allowlist,
        validate_external_link_mode,
        validate_fetch_policy_path_prefixes,
    )

    # 确定 external_link_mode 值
    # CLI 参数优先（argparse 已通过 choices 校验），否则使用 config.yaml 值（需校验）
    if cli_external_link_mode is not None:
        final_external_link_mode = cli_external_link_mode
    else:
        # config.yaml 值需要校验
        final_external_link_mode = validate_external_link_mode(
            fetch_policy.external_link_mode, context="docs_source.fetch_policy.external_link_mode (from config.yaml)"
        )

    # 确定 allowed_path_prefixes 的最终值
    final_allowed_path_prefixes = (
        cli_allowed_path_prefixes_final
        if cli_allowed_path_prefixes_final is not None
        else fetch_policy.allowed_path_prefixes
    )

    # 校验 allowed_path_prefixes 格式（路径前缀应不含 scheme）
    # 如果检测到完整 URL 格式，记录 warning 但保持原值不变
    # 确保 CLI 与 config.yaml 行为一致
    final_allowed_path_prefixes = validate_fetch_policy_path_prefixes(
        final_allowed_path_prefixes, context="docs_source.fetch_policy.allowed_path_prefixes"
    )

    # 确定 external_link_allowlist 的最终值
    raw_external_link_allowlist = (
        cli_external_link_allowlist if cli_external_link_allowlist is not None else fetch_policy.external_link_allowlist
    )

    # 校验 external_link_allowlist 格式
    # 校验并记录无效项警告（返回结构化对象）
    validated_allowlist = validate_external_link_allowlist(
        raw_external_link_allowlist if raw_external_link_allowlist else [],
        context="docs_source.fetch_policy.external_link_allowlist",
    )
    # 重建有效项列表（domains + prefixes 展平，保持向后兼容）
    final_external_link_allowlist = validated_allowlist.domains + validated_allowlist.prefixes

    # 解析 enforce_path_prefixes
    # CLI 参数优先（如有），否则使用 config.yaml 值
    cli_enforce_path_prefixes = getattr(args, "enforce_path_prefixes", None)
    final_enforce_path_prefixes = (
        cli_enforce_path_prefixes if cli_enforce_path_prefixes is not None else fetch_policy.enforce_path_prefixes
    )

    resolved_fetch_policy = ResolvedFetchPolicyConfig(
        allowed_path_prefixes=final_allowed_path_prefixes,
        allowed_domains=(cli_allowed_domains if cli_allowed_domains is not None else fetch_policy.allowed_domains),
        external_link_mode=final_external_link_mode,
        external_link_allowlist=final_external_link_allowlist,
        enforce_path_prefixes=final_enforce_path_prefixes,
    )

    # 解析 allowed_doc_url_prefixes（完整 URL 前缀，用于 load_core_docs 过滤）
    cli_allowed_doc_url_prefixes = _parse_comma_separated_list(getattr(args, "allowed_doc_url_prefixes", None))
    allowed_doc_url_prefixes = (
        cli_allowed_doc_url_prefixes
        if cli_allowed_doc_url_prefixes is not None
        else docs_source.allowed_doc_url_prefixes
    )

    return ResolvedDocsSourceConfig(
        max_fetch_urls=max_fetch_urls,
        fallback_core_docs_count=fallback_core_docs_count,
        llms_txt_url=llms_txt_url,
        llms_cache_path=llms_cache_path,
        changelog_url=changelog_url,
        fetch_policy=resolved_fetch_policy,
        allowed_doc_url_prefixes=allowed_doc_url_prefixes,
    )


# ============================================================
# URL 策略配置解析
# ============================================================


@dataclass
class ResolvedURLStrategyConfig:
    """解析后的 URL 策略配置

    使用 tri-state 设计：CLI 参数 > config.yaml > 默认值
    用于控制文档 URL 的匹配、归一化和优先级策略。

    Attributes:
        allowed_domains: 允许的域名列表（仅当 allowed_url_prefixes 为空时生效）
        allowed_url_prefixes: 允许的 URL 前缀列表（优先级高于 allowed_domains）
        exclude_patterns: URL 排除模式（正则表达式列表）
        max_urls: 最大返回 URL 数量
        fallback_core_docs_count: 当其他来源不足时，从 core_docs 补充的数量
        prefer_changelog: 是否优先处理 Changelog 文档
        deduplicate: 是否启用 URL 去重
        normalize: 是否启用 URL 归一化
        keyword_boost_weight: 关键词匹配权重增益
        priority_weights: 各来源的优先级权重（从 config.yaml 读取，暂不支持 CLI 覆盖）
    """

    allowed_domains: list[str]
    allowed_url_prefixes: list[str]
    exclude_patterns: list[str]
    max_urls: int
    fallback_core_docs_count: int
    prefer_changelog: bool
    deduplicate: bool
    normalize: bool
    keyword_boost_weight: float
    priority_weights: dict[str, float]


def _parse_append_or_comma_separated(
    values: Optional[list[str]],
) -> Optional[list[str]]:
    """解析 action="append" 参数，支持重复指定或逗号分隔

    处理以下输入形式：
    - --arg value1 --arg value2  => ["value1", "value2"]
    - --arg value1,value2        => ["value1", "value2"]
    - --arg value1 --arg value2,value3 => ["value1", "value2", "value3"]

    Args:
        values: argparse append 模式收集的值列表，或 None

    Returns:
        展开后的值列表，None 表示未指定
    """
    if values is None:
        return None

    result: list[str] = []
    for value in values:
        # 每个 value 可能是逗号分隔的多个值
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(item)

    return result if result else None


def resolve_doc_url_strategy_config(args: argparse.Namespace) -> ResolvedURLStrategyConfig:
    """解析 URL 策略配置

    优先级: CLI 参数 > config.yaml > DEFAULT_* 常量

    Args:
        args: 命令行参数

    Returns:
        解析后的 URL 策略配置
    """
    from core.config import get_config

    config = get_config()
    url_strategy = config.knowledge_docs_update.url_strategy

    # 解析 CLI 参数（支持 append 模式和逗号分隔）
    cli_allowed_domains = _parse_append_or_comma_separated(getattr(args, "url_allowed_domains", None))
    cli_allowed_url_prefixes = _parse_append_or_comma_separated(getattr(args, "url_allowed_prefixes", None))
    cli_exclude_patterns = _parse_append_or_comma_separated(getattr(args, "url_exclude_patterns", None))
    cli_max_urls = getattr(args, "url_max_urls", None)
    cli_fallback_core_docs_count = getattr(args, "url_fallback_core_docs_count", None)
    cli_prefer_changelog = getattr(args, "url_prefer_changelog", None)
    cli_deduplicate = getattr(args, "url_deduplicate", None)
    cli_normalize = getattr(args, "url_normalize", None)
    cli_keyword_boost_weight = getattr(args, "url_keyword_boost_weight", None)

    # priority_weights 暂不支持 CLI 覆盖，仅从 config.yaml 读取
    # 未来如需支持 CLI 覆盖，可添加 --url-priority-weights 参数（JSON 格式）

    # 导入校验函数
    from knowledge.doc_url_strategy import validate_url_strategy_prefixes

    # 校验 allowed_url_prefixes 格式
    # CLI 参数不需要校验（假定用户知道自己在做什么）
    # config.yaml 值需要校验（检查是否使用了旧版路径前缀格式）
    if cli_allowed_url_prefixes is not None:
        final_allowed_url_prefixes = cli_allowed_url_prefixes
    else:
        # config.yaml 值校验（仅输出 warning，不修改值）
        final_allowed_url_prefixes = validate_url_strategy_prefixes(
            url_strategy.allowed_url_prefixes, context="url_strategy.allowed_url_prefixes (from config.yaml)"
        )

    # 使用 tri-state 设计：CLI 显式指定 > config.yaml > 默认值
    return ResolvedURLStrategyConfig(
        allowed_domains=(cli_allowed_domains if cli_allowed_domains is not None else url_strategy.allowed_domains),
        allowed_url_prefixes=final_allowed_url_prefixes,
        exclude_patterns=(cli_exclude_patterns if cli_exclude_patterns is not None else url_strategy.exclude_patterns),
        max_urls=(cli_max_urls if cli_max_urls is not None else url_strategy.max_urls),
        fallback_core_docs_count=(
            cli_fallback_core_docs_count
            if cli_fallback_core_docs_count is not None
            else url_strategy.fallback_core_docs_count
        ),
        prefer_changelog=(cli_prefer_changelog if cli_prefer_changelog is not None else url_strategy.prefer_changelog),
        deduplicate=(cli_deduplicate if cli_deduplicate is not None else url_strategy.deduplicate),
        normalize=(cli_normalize if cli_normalize is not None else url_strategy.normalize),
        keyword_boost_weight=(
            cli_keyword_boost_weight if cli_keyword_boost_weight is not None else url_strategy.keyword_boost_weight
        ),
        priority_weights=url_strategy.priority_weights,
    )


# ============================================================
# 统一策略构建器
# ============================================================


@dataclass
class DocAllowlistResult:
    """文档 URL 允许列表构建结果

    统一 _extract_links_from_html 和 _build_urls_to_fetch 使用的配置，
    确保链接分类和 URL 选择使用相同的过滤规则。

    Attributes:
        config: 统一的 DocURLStrategyConfig 配置
    """

    config: DocURLStrategyConfig

    def is_allowed(self, url: str, base_url: str = "https://cursor.com") -> bool:
        """检查 URL 是否在允许范围内

        委托给 doc_url_strategy 模块的 is_allowed_doc_url 函数，
        使用统一构建的配置。

        Args:
            url: 待检查的 URL
            base_url: 基础 URL（用于规范化相对路径）

        Returns:
            True 如果 URL 在允许范围内
        """
        return _is_allowed_doc_url_with_config(url, self.config, base_url)


def build_doc_allowlist(
    url_strategy_config: Optional[ResolvedURLStrategyConfig] = None,
    fetch_policy: Optional[ResolvedFetchPolicyConfig] = None,
    allowed_doc_url_prefixes: Optional[list[str]] = None,
) -> DocAllowlistResult:
    """构建统一的文档 URL 允许列表配置

    统一 _extract_links_from_html 和 _build_urls_to_fetch 使用的配置，
    确保链接分类和 URL 选择使用相同的过滤规则。

    三态语义设计（allowed_url_prefixes）:
    - None: 使用下一优先级或默认值
    - []: 不使用 prefixes 限制，回退到 allowed_domains 或 allow-all
    - ["..."]: 使用指定的前缀列表进行过滤

    配置优先级（allowed_url_prefixes）:
    1. url_strategy_config.allowed_url_prefixes（如果 is not None，包括 []）
    2. allowed_doc_url_prefixes（如果 is not None，包括 []）
    3. 回退到 ALLOWED_DOC_URL_PREFIXES（模块默认值）

    配置优先级（allowed_domains）:
    1. url_strategy_config.allowed_domains（如果提供）
    2. 回退到 ALLOWED_DOC_URL_PREFIXES_NETLOC（从 URL 前缀推导）

    注意：
    - fetch_policy 参数预留用于未来扩展（如 external_link_mode 影响过滤行为）
    - 当前实现主要使用 url_strategy_config 和 allowed_doc_url_prefixes

    Args:
        url_strategy_config: 解析后的 URL 策略配置（来自 CLI/config.yaml）
        fetch_policy: 解析后的在线抓取策略配置（预留，用于 external_link_mode 等）
        allowed_doc_url_prefixes: 允许的文档 URL 前缀列表（三态语义：None=下一优先级，[]=不限制，[...]=指定过滤）

    Returns:
        DocAllowlistResult 包含统一配置和 is_allowed 检查方法

    Examples:
        >>> # 使用 url_strategy_config 构建
        >>> result = build_doc_allowlist(url_strategy_config=resolved_config)
        >>> result.is_allowed("https://cursor.com/docs/guide")
        True

        >>> # 使用 allowed_doc_url_prefixes 构建
        >>> result = build_doc_allowlist(
        ...     allowed_doc_url_prefixes=["https://cursor.com/docs"]
        ... )
        >>> result.config.allowed_url_prefixes
        ['https://cursor.com/docs']

        >>> # 空列表表示不限制（回退到 allowed_domains 或 allow-all）
        >>> result = build_doc_allowlist(allowed_doc_url_prefixes=[])
        >>> result.config.allowed_url_prefixes
        []
    """
    # 确定 allowed_url_prefixes
    # 三态语义设计:
    # - None: 未指定，使用下一优先级或默认值
    # - []: 显式清空，不使用 prefixes 限制（回退到 allowed_domains 或 allow-all）
    # - ["..."]: 显式指定，使用该列表进行过滤
    #
    # 优先级: url_strategy_config > allowed_doc_url_prefixes > 模块默认值
    # 注意：显式检查 is not None 以区分 [] 和 None
    if url_strategy_config is not None and url_strategy_config.allowed_url_prefixes is not None:
        # 包括空列表 [] 也会被使用（表示不使用 prefixes 限制）
        final_allowed_url_prefixes = url_strategy_config.allowed_url_prefixes
    elif allowed_doc_url_prefixes is not None:
        # 包括空列表 [] 也会被使用
        final_allowed_url_prefixes = allowed_doc_url_prefixes
    else:
        final_allowed_url_prefixes = ALLOWED_DOC_URL_PREFIXES

    # 确定 allowed_domains
    # 优先级: url_strategy_config > 从 URL 前缀推导
    if url_strategy_config is not None:
        final_allowed_domains = url_strategy_config.allowed_domains
    else:
        final_allowed_domains = ALLOWED_DOC_URL_PREFIXES_NETLOC

    # 确定 exclude_patterns
    if url_strategy_config is not None:
        final_exclude_patterns = url_strategy_config.exclude_patterns
    else:
        # 默认排除模式（与 _DEFAULT_DOC_URL_CONFIG 保持一致）
        final_exclude_patterns = [
            r".*\.(png|jpg|jpeg|gif|svg|ico|css|js|woff|woff2|ttf|eot)$",
        ]

    # 构建统一配置
    if url_strategy_config is not None:
        config = DocURLStrategyConfig(
            allowed_url_prefixes=final_allowed_url_prefixes,
            allowed_domains=final_allowed_domains,
            max_urls=url_strategy_config.max_urls,
            fallback_core_docs_count=url_strategy_config.fallback_core_docs_count,
            prefer_changelog=url_strategy_config.prefer_changelog,
            deduplicate=url_strategy_config.deduplicate,
            normalize=url_strategy_config.normalize,
            keyword_boost_weight=url_strategy_config.keyword_boost_weight,
            exclude_patterns=final_exclude_patterns,
            priority_weights=url_strategy_config.priority_weights,
        )
    else:
        # 回退到模块默认值（向后兼容）
        # 显式填充与 core.config.DEFAULT_URL_STRATEGY_* 一致的关键字段
        # 确保与"默认契约"一致，防止配置漂移
        config = DocURLStrategyConfig(
            allowed_url_prefixes=final_allowed_url_prefixes,
            allowed_domains=final_allowed_domains,
            exclude_patterns=final_exclude_patterns,
            # 显式使用 core.config 默认值，确保一致性
            max_urls=CONFIG_DEFAULT_URL_STRATEGY_MAX_URLS,
            fallback_core_docs_count=CONFIG_DEFAULT_FALLBACK_CORE_DOCS_COUNT,
            prefer_changelog=CONFIG_DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
            deduplicate=CONFIG_DEFAULT_URL_STRATEGY_DEDUPLICATE,
            normalize=CONFIG_DEFAULT_URL_STRATEGY_NORMALIZE,
            keyword_boost_weight=CONFIG_DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
            priority_weights=CONFIG_DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS.copy(),
        )

    logger.debug(
        f"build_doc_allowlist: allowed_url_prefixes={len(final_allowed_url_prefixes)}, "
        f"allowed_domains={final_allowed_domains}"
    )

    return DocAllowlistResult(config=config)


# ============================================================
# 在线文档分析
# ============================================================


@dataclass
class ContentQualityConfig:
    """内容质量评估配置

    定义内容质量阈值和关键词匹配规则，用于判断 fetch 结果是否满足质量要求。
    """

    # 最小有效文本长度（字符）
    min_text_length: int = 200

    # 日期模式匹配（Changelog 内容通常包含日期）
    # 支持格式: Jan 16, 2026 / 2026-01-16 / January 16, 2026
    date_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}[,\s]+\d{4}\b",
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        ]
    )

    # Changelog 关键词（匹配任意一个即可）
    changelog_keywords: list[str] = field(
        default_factory=lambda: [
            "changelog",
            "更新日志",
            "release",
            "版本",
            "更新",
            "new feature",
            "新功能",
            "improvement",
            "改进",
            "bug fix",
            "修复",
            "v\\d+\\.\\d+",
        ]
    )

    # 质量阈值（0.0-1.0），低于此值触发重试
    quality_threshold: float = 0.4

    # 最大重试次数（按不同方法重试）
    max_quality_retries: int = 2


@dataclass
class FetchAttemptLog:
    """Fetch 尝试日志

    记录每次 fetch 尝试的详细信息，用于调试和分析。
    """

    method_used: FetchMethod
    duration: float
    success: bool
    content_length: int
    quality_score: float
    error: Optional[str] = None


class ChangelogAnalyzer:
    """Changelog 分析器

    从 Cursor Changelog 页面获取更新内容并分析。
    支持多种解析策略：优先日期标题块、备用月份标题块、保底全页单条。
    支持基线比较：通过 fingerprint 检测内容是否真正更新。
    支持内容质量检测：若首次 fetch 质量不足，按优先级用不同方法重试。
    支持外链策略控制：通过 fetch_policy 控制外链的分类和记录行为。
    """

    def __init__(
        self,
        changelog_url: str = DEFAULT_CHANGELOG_URL,
        storage: Optional[KnowledgeStorage] = None,
        quality_config: Optional[ContentQualityConfig] = None,
        doc_allowlist: Optional[DocAllowlistResult] = None,
        fetch_policy: Optional[ResolvedFetchPolicyConfig] = None,
    ):
        self.changelog_url = changelog_url
        self.fetcher = WebFetcher(FetchConfig(timeout=60))
        # 用于基线读取的存储实例（可选）
        self._storage = storage
        # 内容质量评估配置
        self.quality_config = quality_config or ContentQualityConfig()
        # Fetch 尝试日志（用于调试）
        self._fetch_attempts: list[FetchAttemptLog] = []
        # 文档 URL 允许列表配置（用于 _extract_links_from_html）
        # 如果未提供，使用模块默认配置（向后兼容）
        self._doc_allowlist = doc_allowlist
        # fetch_policy 配置（用于控制外链行为）
        # 包含 external_link_mode 和 external_link_allowlist
        self._fetch_policy = fetch_policy

    def _assess_content_quality(self, content: str | None) -> float:
        """评估内容质量

        综合评估内容的质量得分，基于以下因素：
        1. 文本长度（占比 30%）
        2. 日期模式匹配（占比 40%）
        3. Changelog 关键词匹配（占比 30%）

        Args:
            content: 待评估的内容

        Returns:
            质量得分 (0.0-1.0)
        """
        if not content:
            return 0.0

        config = self.quality_config
        score = 0.0

        # 1. 文本长度评分（30%）
        text_length = len(content.strip())
        if text_length >= config.min_text_length:
            # 超过阈值得满分，否则按比例计算
            length_score = min(1.0, text_length / config.min_text_length)
        else:
            length_score = text_length / config.min_text_length
        score += length_score * 0.3

        # 2. 日期模式匹配评分（40%）
        date_matches = 0
        for pattern in config.date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            date_matches += len(matches)
        # 找到 1 个日期得 50%，找到 3 个以上得满分
        date_score = min(1.0, date_matches / 3.0) if date_matches > 0 else 0.0
        score += date_score * 0.4

        # 3. Changelog 关键词匹配评分（30%）
        keyword_matches = 0
        content_lower = content.lower()
        for keyword in config.changelog_keywords:
            if re.search(keyword, content_lower, re.IGNORECASE):
                keyword_matches += 1
        # 匹配 1 个关键词得 30%，匹配 5 个以上得满分
        keyword_score = min(1.0, keyword_matches / 5.0) if keyword_matches > 0 else 0.0
        score += keyword_score * 0.3

        return round(score, 3)

    def _get_retry_methods(self, used_method: FetchMethod) -> list[FetchMethod]:
        """获取重试方法列表

        根据已使用的方法和可用方法，返回按优先级排序的重试方法列表。
        优先级: Playwright > MCP > CURL > LYNX

        Args:
            used_method: 已使用的方法

        Returns:
            重试方法列表（不包含已使用的方法）
        """
        available = self.fetcher.get_available_methods()

        # 按优先级排序（Playwright 优先，因为 Changelog 页面可能需要 JS 渲染）
        priority_order = [
            FetchMethod.PLAYWRIGHT,
            FetchMethod.MCP,
            FetchMethod.CURL,
            FetchMethod.LYNX,
        ]

        retry_methods = []
        for method in priority_order:
            if method in available and method != used_method:
                retry_methods.append(method)

        return retry_methods

    def _log_fetch_attempt(
        self,
        result: FetchResult,
        quality_score: float,
    ) -> None:
        """记录 Fetch 尝试日志

        Args:
            result: Fetch 结果
            quality_score: 质量评分
        """
        # 安全获取属性值（兼容 mock 对象和缺失属性）
        try:
            method_used = result.method_used if isinstance(result.method_used, FetchMethod) else FetchMethod.AUTO
        except (AttributeError, TypeError):
            method_used = FetchMethod.AUTO

        try:
            duration = float(result.duration) if result.duration is not None else 0.0
        except (AttributeError, TypeError, ValueError):
            duration = 0.0

        try:
            success = bool(result.success)
        except (AttributeError, TypeError):
            success = False

        try:
            content_length = len(result.content) if success and result.content else 0
        except (AttributeError, TypeError):
            content_length = 0

        try:
            error = result.error if isinstance(result.error, str) else str(result.error) if result.error else None
        except (AttributeError, TypeError):
            error = None

        attempt = FetchAttemptLog(
            method_used=method_used,
            duration=duration,
            success=success,
            content_length=content_length,
            quality_score=quality_score,
            error=error,
        )
        self._fetch_attempts.append(attempt)

        # 输出调试日志
        logger.debug(
            f"Fetch 尝试: method={method_used.value}, "
            f"duration={duration:.2f}s, success={success}, "
            f"content_length={content_length}, quality={quality_score:.3f}"
        )

    async def fetch_changelog(self) -> Optional[str]:
        """获取 Changelog 内容

        支持内容质量检测和按方法重试：
        1. 首次使用 AUTO 模式获取
        2. 若成功但质量不足，按优先级尝试其他方法（最多重试 max_quality_retries 次）
        3. 返回质量最高的结果

        Returns:
            Changelog 页面内容，失败返回 None
        """
        print_info(f"获取 Changelog: {self.changelog_url}")
        self._fetch_attempts = []  # 重置尝试日志

        await self.fetcher.initialize()

        # 第一次尝试（AUTO 模式）
        result = await self.fetcher.fetch(self.changelog_url)

        if not result.success:
            print_error(f"Changelog 获取失败: {result.error}")
            self._log_fetch_attempt(result, 0.0)
            return None

        # 评估内容质量
        quality_score = self._assess_content_quality(result.content)
        result.quality_score = quality_score
        self._log_fetch_attempt(result, quality_score)

        best_result = result
        best_quality = quality_score

        # 检查是否需要重试
        if quality_score >= self.quality_config.quality_threshold:
            print_success(
                f"Changelog 获取成功 ({len(result.content)} 字符, "
                f"质量评分: {quality_score:.2f}, 方法: {result.method_used.value})"
            )
            return result.content

        # 质量不足，尝试其他方法
        print_warning(
            f"内容质量评分较低 ({quality_score:.2f} < {self.quality_config.quality_threshold}), 尝试其他获取方法..."
        )

        retry_methods = self._get_retry_methods(result.method_used)
        retries_done = 0

        for method in retry_methods:
            if retries_done >= self.quality_config.max_quality_retries:
                break

            logger.info(f"尝试使用 {method.value} 方法重新获取...")
            retry_result = await self.fetcher.fetch(self.changelog_url, method=method)
            retries_done += 1

            if not retry_result.success:
                self._log_fetch_attempt(retry_result, 0.0)
                continue

            retry_quality = self._assess_content_quality(retry_result.content)
            retry_result.quality_score = retry_quality
            self._log_fetch_attempt(retry_result, retry_quality)

            # 更新最佳结果
            if retry_quality > best_quality:
                best_result = retry_result
                best_quality = retry_quality

            # 如果达到质量阈值，立即返回
            if retry_quality >= self.quality_config.quality_threshold:
                print_success(
                    f"Changelog 获取成功 ({len(retry_result.content)} 字符, "
                    f"质量评分: {retry_quality:.2f}, 方法: {method.value})"
                )
                return retry_result.content

        # 返回质量最高的结果（即使未达到阈值）
        if best_result.success:
            print_warning(
                f"Changelog 获取完成，但质量评分未达阈值 "
                f"(最佳: {best_quality:.2f}, 阈值: {self.quality_config.quality_threshold})"
            )
            logger.info(
                f"Fetch 尝试汇总: {len(self._fetch_attempts)} 次尝试, "
                f"最佳方法: {best_result.method_used.value}, 最佳质量: {best_quality:.3f}"
            )
            return best_result.content

        print_error("所有获取方法均失败")
        return None

    def _is_html_content(self, content: str) -> bool:
        """检测内容是否为 HTML

        通过多种特征判断内容是否为 HTML 格式：
        1. 包含 <html 或 <body 标签
        2. 包含大量 HTML 标签（超过阈值）
        3. 包含 DOCTYPE 声明

        Args:
            content: 待检测的内容

        Returns:
            True 如果内容是 HTML，否则 False
        """
        if not content:
            return False

        # 检测 DOCTYPE 或 <html>/<body> 标签
        if re.search(r"<!DOCTYPE\s+html|<html[\s>]|<body[\s>]", content, re.IGNORECASE):
            return True

        # 检测 HTML 标签密度（每 1000 字符超过 5 个标签视为 HTML）
        tag_count = len(re.findall(r"<[a-zA-Z][^>]*>", content))
        content_len = len(content)
        if content_len > 0 and (tag_count * 1000 / content_len) > 5:
            return True

        # 检测常见 HTML 结构标签
        structure_tags = re.findall(
            r"<(?:div|span|p|h[1-6]|ul|ol|li|table|tr|td|a|img|script|style|nav|header|footer|main|article|section)[\s>]",
            content,
            re.IGNORECASE,
        )
        return len(structure_tags) >= 3

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

        original_len = len(content)
        original_lines = content.count("\n") + 1

        # 0. 检测并处理 minified 内容（确保 \n 充分）
        # 如果内容行数过少但长度很长，可能是 minified 或压缩内容
        if original_lines < 10 and original_len > 1000:
            logger.debug(f"检测到 minified 内容：行数={original_lines}, 长度={original_len}，尝试恢复换行")
            # 定义月份名称模式
            month_pattern = (
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            )

            # 在常见分隔符前插入换行
            # 在 markdown 标题前换行
            content = re.sub(r"(?<!^)(?<!\n)(#{1,6}\s+)", r"\n\1", content)

            # 在日期格式前后换行（确保日期作为独立一行）
            # 格式: Jan 16, 2026 或 2026-01-16
            content = re.sub(
                r"(?<!\n)(" + month_pattern + r"\s+\d{1,2}[,\s]+\d{4})", r"\n\1\n", content, flags=re.IGNORECASE
            )
            content = re.sub(r"(?<!\n)(\d{4}[-/]\d{2}[-/]\d{2})", r"\n\1\n", content)

            # 在版本号前后换行（确保版本号作为独立一行）
            # 格式: v1.0.0 / Cursor 0.47
            content = re.sub(
                r"(?<!\n)((?:Cursor\s+)?v?\d+\.\d+(?:\.\d+)?)(?=\s|$)", r"\n\1\n", content, flags=re.IGNORECASE
            )

            # 在列表项前换行
            content = re.sub(r"(?<!\n)([-*]\s+)", r"\n\1", content)
            # 在句号+大写字母处换行（句子边界）
            content = re.sub(r"([.!?])\s+([A-Z])", r"\1\n\2", content)

            # 清理多余的连续换行
            content = re.sub(r"\n{3,}", "\n\n", content)

            # 更新统计
            original_lines = content.count("\n") + 1
            logger.debug(f"minified 内容恢复后：行数={original_lines}")

        # 1. 尝试截取主内容区域（降低噪声干扰）
        content = self._extract_main_content(content)

        # 2. 根据内容类型选择清理策略
        if self._is_html_content(content):
            # HTML 内容：使用 ContentCleaner 进行专业清理
            content = self._clean_html_content(content)
        else:
            # 非 HTML 内容（纯文本/Markdown）：使用轻量清理
            content = self._clean_text_content(content)

        # 3. 通用后处理：移除多余空行、行首行尾空白
        content = re.sub(r"\n{3,}", "\n\n", content)
        lines = [line.strip() for line in content.split("\n")]
        content = "\n".join(lines)
        content = content.strip()

        # 4. 输出清理统计日志
        cleaned_len = len(content)
        cleaned_lines = content.count("\n") + 1 if content else 0
        logger.debug(
            f"内容清理统计: 原始 {original_len} 字符/{original_lines} 行 → "
            f"清理后 {cleaned_len} 字符/{cleaned_lines} 行 "
            f"(压缩率: {100 - cleaned_len * 100 // max(original_len, 1):.0f}%)"
        )

        return content

    def _extract_links_from_html(self, content: str) -> dict[str, list[dict[str, str]]]:
        """从 HTML 中提取链接

        使用 BeautifulSoup 提取 <a href> 链接，补全为绝对路径，
        然后按 allowlist 分类为 allowed（允许抓取）和 external（外部链接）。

        根据 fetch_policy.external_link_mode 控制外链行为：
        - skip_all：外链不记录，external 列表为空
        - record_only（默认）：外链仅记录到 external 列表
        - fetch_allowlist：匹配 allowlist 的外链也放入 allowed 列表

        Args:
            content: HTML 内容

        Returns:
            包含 'allowed' 和 'external' 两个键的字典，每个值为链接列表，
            每个链接包含 href 和 text 字段
        """
        result: dict[str, list[dict[str, str]]] = {
            "allowed": [],
            "external": [],
        }

        if not content:
            return result

        try:
            from bs4 import BeautifulSoup as BS4
        except ImportError:
            logger.warning("BeautifulSoup 未安装，无法提取链接")
            return result

        # 获取 external_link_mode（默认 record_only）
        external_link_mode = "record_only"
        external_link_allowlist: list[str] = []
        if self._fetch_policy is not None:
            external_link_mode = self._fetch_policy.external_link_mode
            external_link_allowlist = self._fetch_policy.external_link_allowlist or []

        try:
            # 优先使用 lxml 解析器，回退到 html.parser（Python 内置）
            try:
                soup = BS4(content, "lxml")
            except Exception:
                # lxml 未安装或不可用时回退到内置解析器
                soup = BS4(content, "html.parser")

            # 第一步：提取并 normalize 全部链接
            all_links: list[dict[str, str]] = []
            for a_tag in soup.find_all("a", href=True):
                href = str(a_tag.get("href", "")).strip()
                text = a_tag.get_text(strip=True)

                if not href:
                    continue

                # 跳过 JavaScript 链接和 mailto 链接
                if href.startswith(("javascript:", "mailto:", "#")):
                    continue

                # 归一化 URL
                normalized = normalize_url(href, base_url="https://cursor.com")
                all_links.append(
                    {
                        "href": normalized,
                        "text": text,
                    }
                )

            # 第二步：去重（保持顺序）
            seen: set[str] = set()
            unique_links: list[dict[str, str]] = []
            for link in all_links:
                if link["href"] not in seen:
                    seen.add(link["href"])
                    unique_links.append(link)

            # 第三步：按 allowlist 分类，结合 external_link_mode 策略
            # 使用统一配置（如果提供），否则回退到模块默认
            for link in unique_links:
                if self._doc_allowlist is not None:
                    # 使用统一构建的配置（与 _build_urls_to_fetch 保持一致）
                    allowed = self._doc_allowlist.is_allowed(link["href"])
                else:
                    # 回退到模块默认配置（向后兼容）
                    allowed = is_allowed_doc_url(link["href"])

                if allowed:
                    result["allowed"].append(link)
                else:
                    # 根据 external_link_mode 处理外链
                    if external_link_mode == "skip_all":
                        # skip_all：外链不记录，直接跳过
                        pass
                    elif external_link_mode == "fetch_allowlist":
                        # fetch_allowlist：检查是否匹配 allowlist
                        if self._matches_external_allowlist(link["href"], external_link_allowlist):
                            # 匹配 allowlist 的外链也放入 allowed
                            result["allowed"].append(link)
                        else:
                            # 不匹配的外链仍记录到 external
                            result["external"].append(link)
                    else:
                        # record_only（默认）：外链仅记录到 external
                        result["external"].append(link)

            logger.debug(
                f"从 HTML 中提取到 {len(result['allowed'])} 个允许链接, "
                f"{len(result['external'])} 个外部链接 "
                f"(external_link_mode={external_link_mode})"
            )
            return result

        except Exception as e:
            logger.warning(f"提取链接失败: {e}")
            return result

    def _matches_external_allowlist(self, url: str, allowlist: list[str]) -> bool:
        """检查 URL 是否匹配外链 allowlist

        allowlist 项支持两种格式：
        - 域名格式（如 "github.com"）：匹配该域名及其子域名
        - URL 前缀格式（如 "https://github.com/cursor"）：精确前缀匹配

        Args:
            url: 待检查的 URL
            allowlist: 外链 allowlist 列表

        Returns:
            True 如果匹配 allowlist 中的任意项
        """
        if not allowlist:
            return False

        from knowledge.doc_url_strategy import _matches_allowlist

        return _matches_allowlist(url, allowlist)

    def _clean_html_content(self, content: str) -> str:
        """清理 HTML 内容

        使用 ContentCleaner 进行专业的 HTML 清理，保留结构层级信息。

        Args:
            content: HTML 内容

        Returns:
            清理后的纯文本
        """
        try:
            # 使用 ContentCleaner 清理 HTML
            cleaner = ContentCleaner()
            cleaned = cleaner.clean_to_text(content)

            # 如果清理结果过短，可能清理过度，回退到基本清理
            if len(cleaned) < 50 and len(content) > 200:
                logger.debug("ContentCleaner 清理结果过短，回退到基本清理")
                return self._basic_html_clean(content)

            return cleaned
        except Exception as e:
            logger.warning(f"ContentCleaner 清理失败，回退到基本清理: {e}")
            return self._basic_html_clean(content)

    def _basic_html_clean(self, content: str) -> str:
        """基本 HTML 清理（回退方案）

        当 ContentCleaner 失败或清理过度时使用的简单清理逻辑。

        Args:
            content: HTML 内容

        Returns:
            清理后的文本
        """
        # 移除 HTML 注释
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # 移除 script 和 style 标签及内容
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # 移除 nav、header、footer 等导航元素
        content = re.sub(r"<nav[^>]*>.*?</nav>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<header[^>]*>.*?</header>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<footer[^>]*>.*?</footer>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # 块级元素替换为换行（保持结构）
        content = re.sub(r"</(?:div|p|li|tr|h[1-6]|article|section|main)>", "\n", content, flags=re.IGNORECASE)
        content = re.sub(r"<(?:br|hr)[^>]*/?>", "\n", content, flags=re.IGNORECASE)

        # 保留 <a> 标签的文本
        content = re.sub(r"<a[^>]*>([^<]*)</a>", r"\1", content, flags=re.IGNORECASE)

        # 移除其他标签
        content = re.sub(r"<[^>]+>", "", content)

        # 解码常见 HTML 实体
        html_entities = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&#39;": "'",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "...",
            "&copy;": "©",
            "&reg;": "®",
            "&trade;": "™",
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)

        return content

    def _clean_text_content(self, content: str) -> str:
        """清理纯文本/Markdown 内容

        轻量级清理，不依赖 HTML 解析器，减少对 # 标题的硬依赖。

        Args:
            content: 纯文本或 Markdown 内容

        Returns:
            清理后的文本
        """
        # 移除常见的网页噪声模式
        # 移除 "Skip to content"、"Back to top" 等导航文本
        content = re.sub(
            r"\b(?:Skip to (?:content|main)|Back to top|Jump to|"
            r"Table of Contents|Navigation|Menu|Search)\b[^\n]*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        # 移除版权信息行
        content = re.sub(r"^.*(?:©|Copyright|All [Rr]ights [Rr]eserved).*$", "", content, flags=re.MULTILINE)

        # 移除 URL 行（单独成行的链接）
        content = re.sub(r"^\s*https?://[^\s]+\s*$", "", content, flags=re.MULTILINE)

        # 解码常见 HTML 实体（即使是纯文本也可能包含从网页抓取的实体）
        html_entities = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&#39;": "'",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "...",
            "&copy;": "©",
            "&reg;": "®",
            "&trade;": "™",
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)

        return content

    def _extract_main_content(self, content: str) -> str:
        """尝试从页面中提取主内容区域

        使用多种策略尝试定位主内容区域：
        1. 检测 <main>、<article> 等语义标签
        2. 检测常见的内容容器 class/id（如 changelog、content、main-content）
        3. 检测内容锚点（支持多种格式，不仅限于 Markdown 标题）
        4. 如果无法定位，返回原始内容

        Args:
            content: 原始 HTML/Markdown 内容

        Returns:
            提取出的主内容区域，或原始内容（如无法定位）
        """
        # 策略1: 提取 <main> 标签内容
        main_match = re.search(r"<main[^>]*>(.*?)</main>", content, flags=re.DOTALL | re.IGNORECASE)
        if main_match:
            logger.debug("使用 <main> 标签提取主内容")
            return main_match.group(1)

        # 策略2: 提取 <article> 标签内容
        article_match = re.search(r"<article[^>]*>(.*?)</article>", content, flags=re.DOTALL | re.IGNORECASE)
        if article_match:
            logger.debug("使用 <article> 标签提取主内容")
            return article_match.group(1)

        # 策略3: 提取带有 changelog/content 相关 class/id 的 div
        # 匹配 class="changelog" 或 id="main-content" 等
        content_div_match = re.search(
            r'<div[^>]*(?:class|id)=["\'][^"\']*'
            r"(?:changelog|main-content|content-area|page-content|docs-content)"
            r'[^"\']*["\'][^>]*>(.*?)</div>',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if content_div_match:
            logger.debug("使用内容容器 div 提取主内容")
            return content_div_match.group(1)

        # 策略4: 检测内容锚点（支持多种格式）
        anchor_patterns = [
            # Markdown 标题格式: "# Changelog" 或 "# 更新日志"
            r"(#{1,3}\s*(?:Changelog|Updates?|更新日志|版本历史|What\'s New|Release Notes?).*)",
            # HTML 标题格式: <h1>Changelog</h1> 或 <h2>Updates</h2>
            r"(<h[1-3][^>]*>(?:Changelog|Updates?|更新日志|版本历史|What\'s New|Release Notes?)[^<]*</h[1-3]>.*)",
            # 日期格式标题（Markdown）: "## Jan 16, 2026" 或 "## 2026-01-16"
            r"(#{1,3}\s*(?:\d{4}[-/]\d{2}[-/]\d{2}|"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}[,\s]+\d{4}).*)",
            # 日期格式标题（HTML）: <h2>Jan 16, 2026</h2>
            r"(<h[1-3][^>]*>\s*(?:\d{4}[-/]\d{2}[-/]\d{2}|"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}[,\s]+\d{4})[^<]*</h[1-3]>.*)",
            # 版本号格式: "## v1.2.3" 或 "## Version 1.2.3"
            r"(#{1,3}\s*(?:v(?:ersion)?\s*)?\d+\.\d+(?:\.\d+)?.*)",
            r"(<h[1-3][^>]*>\s*(?:v(?:ersion)?\s*)?\d+\.\d+(?:\.\d+)?[^<]*</h[1-3]>.*)",
        ]

        for pattern in anchor_patterns:
            anchor_match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if anchor_match:
                # 从锚点位置开始截取到文档末尾
                start_pos = anchor_match.start()
                extracted = content[start_pos:]
                # 如果提取的内容足够长（至少100字符），使用它
                if len(extracted) >= 100:
                    preview = anchor_match.group()[:50].replace("\n", " ")
                    logger.debug(f"使用内容锚点提取主内容，起始: {preview}...")
                    return extracted

        # 策略5: 如果无法定位主内容区域，返回原始内容
        return content

    def _parse_by_date_headers(self, content: str) -> list[ChangelogEntry]:
        """策略1: 按日期标题块解析（优先）

        匹配格式如：
        - ## 2024-01-15（带 # 的日期标题）
        - ### Jan 16, 2026
        - 2024-01-15（无 # 的整行日期）
        - Jan 16, 2026（无 # 的整行日期）

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 定义月份名称模式
        month_names = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        )

        # 匹配多种日期格式（支持带 # 或无 # 的整行日期）：
        # - ## 2024-01-15（带标题符号）
        # - ### 2024/01/15
        # - ## Jan 16, 2026
        # - 2024-01-15（无标题符号，整行日期）
        # - Jan 16, 2026（无标题符号，整行日期）
        date_pattern = (
            r"\n(?="
            r"\s*"  # 允许行首空白
            r"(?:#{1,3}\s*)?"  # 可选的 markdown 标题符号
            r"(?:"
            r"\d{4}[-/]\d{2}[-/]\d{2}"  # ISO 日期格式
            r"|" + month_names + r"\s+\d{1,2}[,\s]+\d{4}"  # 月份名称格式
            r")"
            r"\s*$"  # 确保是整行（行尾）
            r")"
        )

        sections = re.split(date_pattern, content, flags=re.IGNORECASE | re.MULTILINE)

        # 如果只有1个section且未匹配到日期模式，说明未分割成功
        if len(sections) == 1:
            # 检查原内容是否以日期开头
            first_line = content.strip().split("\n")[0] if content.strip() else ""
            if not re.match(date_pattern.replace(r"\n(?=", r"^(?:"), first_line.strip(), flags=re.IGNORECASE):
                return []

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_section(section)
            # 日期策略：条目必须有日期才算有效
            if entry and entry.date:
                entries.append(entry)

        return entries

    def _parse_by_category_date_headers(self, content: str) -> list[ChangelogEntry]:
        """策略2: 按类别+日期标题块解析

        匹配格式如：CLI Jan 16, 2026、Agent Dec 10, 2025、
        Feature January 20 2026、New Feature Release Jan 1, 2026 等
        "<Category> <Month Day, Year>" 格式。

        这种格式常见于按产品或功能分类的 changelog，每个分类下
        有独立的日期标识。

        支持的标题格式：
        - 带 # 的标题：## CLI Jan 16, 2026
        - 不带 # 的标题：CLI Jan 16, 2026
        - 多词类别（1-5 个词）：New Feature Release Jan 1, 2026

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 定义月份名称模式
        month_names = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        )

        # 匹配 "<Category> <Month Day, Year>" 格式
        # 例如: CLI Jan 16, 2026 或 Agent December 20 2025
        # Category: 1-5个单词（字母/数字/连字符），日期：月份 日, 年
        # 放宽词数限制以支持 "New Feature Release" 等多词类别
        category_date_pattern = (
            r"\n(?="
            r"\s*"  # 允许行首空白
            r"(?:#{1,3}\s*)?"  # 可选的 markdown 标题符号
            r"(?:[\w-]+(?:\s+[\w-]+){0,4})\s+"  # 类别：1-5个单词（放宽限制）
            + month_names
            + r"\s+\d{1,2}[,\s]+\d{4}"  # 日期：月 日, 年
            r")"
        )

        sections = re.split(category_date_pattern, content, flags=re.IGNORECASE)

        # 如果只有1个section，说明未匹配到分隔符
        if len(sections) == 1:
            return []

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_category_date_section(section)
            # 类别+日期策略：条目必须有日期才算有效
            if entry and entry.date:
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
        lines = section.strip().split("\n")

        if lines:
            title_line = lines[0].strip()

            # 移除可能的 markdown 标题符号
            title_line = re.sub(r"^#+\s*", "", title_line)

            # 尝试提取类别和日期
            # 格式: <Category> <Month Day, Year>
            # Category: 1-5 个词（放宽限制以支持多词类别）
            category_date_match = re.match(
                r"^([\w-]+(?:\s+[\w-]+){0,4})\s+"
                r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                r"\s+\d{1,2}[,\s]+\d{4})",
                title_line,
                re.IGNORECASE,
            )

            if category_date_match:
                category_part = category_date_match.group(1).strip()
                date_part = category_date_match.group(2).strip()

                entry.category = self._normalize_category(category_part)
                entry.date = date_part
                entry.title = title_line

        # 提取内容
        entry.content = "\n".join(lines[1:]).strip()

        # 如果没有从标题行获取到 category，尝试从内容分析
        if not entry.category or entry.category == "other":
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
        if any(kw in cat_lower for kw in ["feature", "new", "add", "launch"]):
            return "feature"

        # 修复相关
        if any(kw in cat_lower for kw in ["fix", "bug", "patch", "hotfix"]):
            return "fix"

        # 改进相关
        if any(kw in cat_lower for kw in ["improve", "enhance", "update", "optim"]):
            return "improvement"

        # CLI/Agent 等产品类别，默认为 feature
        if any(kw in cat_lower for kw in ["cli", "agent", "cloud", "mcp", "hook"]):
            return "feature"

        return "other"

    def _parse_by_version_headers(self, content: str) -> list[ChangelogEntry]:
        """策略3: 按版本标题块解析（备用）

        匹配格式如：
        - ## v1.0.0（带 # 的版本标题）
        - ### Version 2.1
        - v1.0.0（无 # 的整行版本号）
        - Cursor 0.47（带产品名称前缀）
        - Cursor 0.47.1

        Args:
            content: 清理后的内容

        Returns:
            解析出的条目列表
        """
        entries: list[ChangelogEntry] = []

        # 匹配版本号格式（支持带 # 或无 #，支持 Cursor 前缀）
        # - ## v1.0.0
        # - ### Version 2.1
        # - v1.0.0（无标题符号）
        # - Cursor 0.47 或 Cursor 0.47.1
        version_pattern = (
            r"\n(?="
            r"\s*"  # 允许行首空白
            r"(?:#{1,3}\s*)?"  # 可选的 markdown 标题符号
            r"(?:Cursor\s+)?"  # 可选的 Cursor 前缀
            r"(?:v(?:ersion)?\s*)?"  # 可选的 v/version 前缀
            r"\d+\.\d+(?:\.\d+)?"  # 版本号：主.次 或 主.次.修订
            r"\s*$"  # 确保是整行（行尾）
            r")"
        )

        sections = re.split(version_pattern, content, flags=re.IGNORECASE | re.MULTILINE)

        # 如果只有1个section，说明未匹配到分隔符
        if len(sections) == 1:
            return []

        for section in sections:
            if not section.strip():
                continue

            entry = self._parse_section(section)
            # 版本策略：条目必须有版本号才算有效
            if entry and entry.version:
                entries.append(entry)

        return entries

    def _parse_fallback(self, content: str, reason: str = "无分段命中") -> list[ChangelogEntry]:
        """策略4: 保底策略，将全页作为单条 entry

        当其他策略都无法解析出有效条目时使用。

        Args:
            content: 清理后的内容
            reason: 保底触发原因（用于日志诊断）

        Returns:
            包含单条 entry 的列表
        """
        if not content.strip():
            logger.debug(f"保底策略触发，原因: {reason}，但内容为空")
            return []

        # 计算内容统计信息，用于诊断
        line_count = content.count("\n") + 1
        char_count = len(content)

        # 输出保底触发原因
        logger.warning(f"保底策略触发 - 原因: {reason}，内容统计: {char_count} 字符/{line_count} 行")

        entry = ChangelogEntry()
        entry.title = "Changelog"
        entry.content = content.strip()
        entry.category = self._categorize_content(entry.content)

        # 尝试从内容中提取日期
        date_match = re.search(
            r"(\d{4}[-/]\d{2}[-/]\d{2})|"
            r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}[,\s]+\d{4})",
            content,
            re.IGNORECASE,
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
        lines = section.strip().split("\n")

        # 提取标题行
        if lines:
            title_line = lines[0].strip()

            # 尝试提取日期（多种格式）
            # 格式1: 2024-01-15 或 2024/01/15
            date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", title_line)
            if date_match:
                entry.date = date_match.group(1)
            else:
                # 格式2: Jan 16, 2026 或 January 16 2026
                date_match = re.search(
                    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                    r"\s+\d{1,2}[,\s]+\d{4})",
                    title_line,
                    re.IGNORECASE,
                )
                if date_match:
                    entry.date = date_match.group(1)

            # 尝试提取版本（支持 Cursor 前缀）
            # 格式: v1.0.0, Version 1.0, Cursor 0.47, Cursor 0.47.1
            version_match = re.search(r"(?:Cursor\s+)?v?(?:ersion\s*)?(\d+\.\d+(?:\.\d+)?)", title_line, re.IGNORECASE)
            if version_match:
                entry.version = version_match.group(1)

            entry.title = re.sub(r"^#+\s*", "", title_line)

        # 提取内容
        entry.content = "\n".join(lines[1:]).strip()

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
        if any(
            kw in content_lower
            for kw in [
                "新增",
                "新功能",
                "new feature",
                "new:",
                "added",
                "添加",
                "introducing",
                "launch",
                "发布",
                "支持",
            ]
        ):
            return "feature"

        # 检测修复
        if any(kw in content_lower for kw in ["修复", "fix", "bug", "fixed", "resolved", "解决", "patch"]):
            return "fix"

        # 检测改进
        if any(
            kw in content_lower
            for kw in [
                "改进",
                "优化",
                "improve",
                "enhance",
                "update",
                "better",
                "更新",
                "升级",
                "upgrade",
                "performance",
            ]
        ):
            return "improvement"

        return "other"

    def parse_changelog(self, content: str) -> list[ChangelogEntry]:
        """解析 Changelog 内容

        使用分层解析策略：
        1. 优先按日期标题块解析
        2. 按类别+日期标题块解析（如 "CLI Jan 16, 2026"）
        3. 备用按版本标题块解析
        4. 保底将全页作为单条 entry（并输出触发原因）

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

        # 收集保底触发原因（用于诊断）
        fallback_reasons: list[str] = []
        cleaned_line_count = cleaned_content.count("\n") + 1

        # 策略1: 按日期标题块解析
        entries = self._parse_by_date_headers(cleaned_content)
        if entries:
            logger.debug(f"使用日期标题策略解析，得到 {len(entries)} 条")
            return entries
        fallback_reasons.append("日期标题策略未命中")

        # 策略2: 按类别+日期标题块解析（如 "CLI Jan 16, 2026"）
        entries = self._parse_by_category_date_headers(cleaned_content)
        if entries:
            logger.debug(f"使用类别+日期标题策略解析，得到 {len(entries)} 条")
            return entries
        fallback_reasons.append("类别+日期策略未命中")

        # 策略3: 按版本标题块解析
        entries = self._parse_by_version_headers(cleaned_content)
        if entries:
            logger.debug(f"使用版本标题策略解析，得到 {len(entries)} 条")
            return entries
        fallback_reasons.append("版本标题策略未命中")

        # 检查是否因行数过少导致解析失败
        if cleaned_line_count < 5:
            fallback_reasons.append(f"清洗后行数过少({cleaned_line_count}行)")

        # 策略4: 保底全页单条，传递触发原因
        reason = "; ".join(fallback_reasons)
        logger.debug(f"使用保底策略，全页作为单条 entry (原因: {reason})")
        return self._parse_fallback(cleaned_content, reason=reason)

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
            if entry.category == "feature":
                analysis.new_features.append(f"[{entry.date or entry.version}] {entry.title}")
            elif entry.category == "fix":
                analysis.fixes.append(f"[{entry.date or entry.version}] {entry.title}")
            elif entry.category == "improvement":
                analysis.improvements.append(f"[{entry.date or entry.version}] {entry.title}")

            # 检测相关文档 URL（使用 get_core_docs 动态加载）
            for doc_url in get_core_docs():
                # 从 entry 内容中匹配关键词
                doc_keywords = self._extract_doc_keywords(doc_url)
                if (
                    any(kw.lower() in entry.content.lower() for kw in doc_keywords)
                    and doc_url not in analysis.related_doc_urls
                ):
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
        path = url.split("/")[-1]
        keywords = [path.replace("-", " ")]

        # 提取完整路径用于多级路径匹配（如 modes/plan, modes/ask）
        full_path = "/".join(url.split("/")[4:])  # 提取 docs/cli/ 之后的部分

        # 添加特定关键词映射
        # 包含 Jan 16 2026 新特性: plan/ask 模式、cloud relay、diff
        # 包含 Jan 20 2026 新特性: agent review、code-review cookbook、cloud-agent
        keyword_map = {
            # 基础参数文档
            "parameters": [
                "参数",
                "parameter",
                "option",
                "选项",
                # Jan 16 2026: plan/ask 模式
                "plan",
                "ask",
                "--mode",
                "mode",
                "模式",
                "规划模式",
                "问答模式",
                "代理模式",
                # 输出格式
                "output-format",
                "stream-json",
                "json",
            ],
            # 斜杠命令
            "slash-commands": [
                "斜杠命令",
                "slash",
                "/",
                # Jan 16 2026: /plan 和 /ask 命令
                "/plan",
                "/ask",
                "/model",
                "/models",
                "/rules",
                "/commands",
                "/mcp",
            ],
            # MCP 服务器
            "mcp": [
                "mcp",
                "服务器",
                "server",
                # Jan 16 2026: cloud relay
                "cloud relay",
                "relay",
                "云中继",
                "mcp enable",
                "mcp disable",
                "mcp list",
            ],
            # Hooks
            "hooks": [
                "hook",
                "钩子",
                "beforeShellExecution",
                "afterShellExecution",
                "beforeFileEdit",
                "afterFileEdit",
            ],
            # 子代理
            "subagents": [
                "subagent",
                "子代理",
                "agent",
                "foreground",
                "background",
                "前台",
                "后台",
            ],
            # 技能
            "skills": [
                "skill",
                "技能",
                "SKILL.md",
            ],
            # CLI 概览
            "overview": [
                "cli",
                "命令行",
                "agent",
                # Jan 16 2026: diff 视图
                "diff",
                "差异",
                "changes",
                "变更",
                "review",
                "审阅",
                "Ctrl+R",
            ],
            # 使用指南
            "using": [
                "using",
                "使用",
                "教程",
                # Jan 16 2026: 交互模式增强
                "interactive",
                "交互",
                "快捷键",
                "diff view",
                "diff 视图",
            ],
            # Jan 16 2026: plan 模式专页
            "modes/plan": [
                "plan",
                "plan mode",
                "规划模式",
                "--mode plan",
                "规划",
                "planner",
                "任务规划",
                "只读模式",
                "分析",
                "analyze",
                "readonly",
            ],
            # Jan 16 2026: ask 模式专页
            "modes/ask": [
                "ask",
                "ask mode",
                "问答模式",
                "--mode ask",
                "问答",
                "咨询",
                "question",
                "query",
                "只读",
                "readonly",
                "解释",
            ],
            # Jan 20 2026: Agent Review 审阅功能
            "agent/review": [
                "review",
                "审阅",
                "代码审阅",
                "变更审阅",
                "diff",
                "Ctrl+R",
                "accept",
                "reject",
                "接受",
                "拒绝",
                "变更",
                "changes",
                "inline diff",
                "内联差异",
            ],
            # Jan 20 2026: CLI Cookbook - 代码评审
            "cookbook/code-review": [
                "code review",
                "代码评审",
                "代码审查",
                "pr review",
                "pull request",
                "PR",
                "评审",
                "审查",
                "review workflow",
                "gh pr",
                "git diff",
                "reviewer",
            ],
            # Jan 20 2026: Cloud Agent 云代理
            "cloud-agent": [
                "cloud agent",
                "云代理",
                "云端代理",
                "cloud",
                "云端",
                "remote agent",
                "background task",
                "后台任务",
                "&",
                "cloud relay",
                "云中继",
            ],
            "cloud-agent/overview": [
                "cloud agent",
                "云代理",
                "overview",
                "概览",
                "云端执行",
                "remote execution",
            ],
            "cloud-agent/getting-started": [
                "getting started",
                "快速开始",
                "入门",
                "cloud setup",
                "云端配置",
            ],
            "cloud-agent/api": [
                "cloud api",
                "云端 API",
                "api",
                "REST",
                "endpoint",
                "接口",
                "programmatic",
                "程序化调用",
            ],
            "cloud-agent/api/streaming": [
                "streaming",
                "流式",
                "流式响应",
                "stream",
                "SSE",
                "server-sent events",
                "real-time",
                "实时",
            ],
            "cloud-agent/api/sessions": [
                "session",
                "会话",
                "sessions",
                "resume",
                "恢复会话",
                "session_id",
                "会话管理",
                "session management",
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

    async def _get_baseline_fingerprint(self) -> tuple[Optional[str], str]:
        """获取基线 fingerprint

        从知识库中读取上次保存的 changelog 清洗后内容哈希。
        采用分层策略保证比较口径一致：
        1. 优先读取索引中的 cleaned_fingerprint（已存储的归一化 fingerprint）
        2. 如果不存在（向后兼容），加载 doc.content 并调用 compute_fingerprint()

        Returns:
            (fingerprint, source) 元组：
            - fingerprint: 上次内容的 fingerprint，不存在时返回 None
            - source: 来源说明（"index"/"computed"/"none"/"error"）
        """
        if self._storage is None:
            return None, "none"

        try:
            # 确保存储已初始化
            if not self._storage._initialized:
                await self._storage.initialize()

            # 策略 1: 优先读取索引中的 cleaned_fingerprint
            cleaned_fp = self._storage.get_cleaned_fingerprint_by_url(self.changelog_url)
            if cleaned_fp:
                logger.debug(f"基线读取来源: index (cleaned_fingerprint: {cleaned_fp[:8]}...)")
                return cleaned_fp, "index"

            # 策略 2: 向后兼容 - 加载文档内容并重新计算
            doc = await self._storage.load_document_by_url(self.changelog_url)
            if doc and doc.content:
                computed_fp = self.compute_fingerprint(doc.content)
                logger.debug(f"基线读取来源: computed (从 doc.content 计算: {computed_fp[:8]}...)")
                return computed_fp, "computed"

            # 无基线
            logger.debug("基线读取来源: none (文档不存在)")
            return None, "none"

        except Exception as e:
            logger.warning(f"获取基线 fingerprint 失败: {e}")
            return None, "error"

    async def analyze(self) -> UpdateAnalysis:
        """执行完整分析

        ================================================================
        阶段表 (Phase Table)
        ================================================================
        | 阶段 | 操作                        | 副作用类型   | 可配置跳过 |
        |------|-----------------------------|--------------|------------|
        | 1    | fetch_changelog()           | 网络读取     | offline    |
        | 2    | compute_fingerprint()       | 无           | -          |
        | 3    | _get_baseline_fingerprint() | 文件读取     | -          |
        | 4    | _extract_links_from_html()  | 无           | -          |
        | 5    | parse_changelog()           | 无           | -          |
        | 6    | extract_update_points()     | 无           | -          |

        ================================================================
        副作用矩阵 (Side Effect Matrix)
        ================================================================
        | 副作用类型   | 目标                    | 策略控制              |
        |--------------|-------------------------|----------------------|
        | 网络读取     | changelog_url           | offline 模式跳过     |
        | 网络读取     | (重试方法)              | quality_config       |
        | 文件读取     | KnowledgeStorage index  | storage 可选注入     |

        ================================================================
        入口参数到下游字段映射 (Parameter Mapping)
        ================================================================
        | 入口参数        | 下游使用位置                    | 说明                   |
        |-----------------|--------------------------------|------------------------|
        | changelog_url   | fetch_changelog()              | HTTP 请求目标          |
        | storage         | _get_baseline_fingerprint()    | 读取基线 fingerprint   |
        | quality_config  | _assess_content_quality()      | 评估内容质量阈值       |
        | doc_allowlist   | _extract_links_from_html()     | 链接分类过滤规则       |
        | fetch_policy    | (记录外链策略，不在此处执行)   | Phase A 仅解析不执行   |

        ================================================================
        doc_url_strategy 调用点
        ================================================================
        - _extract_links_from_html(): 使用 doc_allowlist.config 进行链接分类
        - 实际 URL 抓取策略: 在 KnowledgeUpdater._build_urls_to_fetch() 中执行

        支持基线比较：
        1. 获取当前 changelog 内容
        2. 计算 fingerprint 并与知识库中的基线比较
        3. 如果 fingerprint 相同，返回 has_updates=False

        Returns:
            更新分析结果
        """
        import time

        start_time = time.time()

        print_section("分析 Changelog")

        # 初始化结构化日志
        analysis_log = ChangelogAnalysisLog(
            changelog_url=sanitize_url_for_log(self.changelog_url),
        )

        content = await self.fetch_changelog()
        if not content:
            analysis_log.fetch_success = False
            analysis_log.duration_ms = (time.time() - start_time) * 1000
            analysis_log.log_structured()
            return UpdateAnalysis(analysis_log=analysis_log)

        # 记录获取成功信息
        analysis_log.fetch_success = True
        analysis_log.content_length = len(content)
        # 从 _fetch_attempts 获取最后使用的方法和质量评分
        if self._fetch_attempts:
            last_attempt = self._fetch_attempts[-1]
            analysis_log.fetch_method = last_attempt.method_used.value
            analysis_log.quality_score = last_attempt.quality_score

        # 计算当前内容的 fingerprint（使用清洗后内容）
        current_fingerprint = self.compute_fingerprint(content)
        analysis_log.fingerprint = current_fingerprint[:8] + "..."
        logger.debug(f"当前 changelog fingerprint: {current_fingerprint[:8]}...")

        # 获取基线 fingerprint 进行比较（返回值为 (fingerprint, source) 元组）
        baseline_fingerprint, baseline_source = await self._get_baseline_fingerprint()

        # 日志输出 baseline 与 current 的短前缀及差异原因
        if baseline_fingerprint:
            logger.info(
                f"Fingerprint 比较: baseline={baseline_fingerprint[:8]}... (来源: {baseline_source}) "
                f"vs current={current_fingerprint[:8]}..."
            )
            if current_fingerprint == baseline_fingerprint:
                # 内容未变化，返回无更新结果
                print_info(f"Changelog 内容未变化 (fingerprint: {current_fingerprint[:8]}...)")
                analysis_log.baseline_match = True
                analysis_log.entry_count = 0
                analysis_log.duration_ms = (time.time() - start_time) * 1000
                analysis_log.log_structured()

                analysis = UpdateAnalysis(
                    has_updates=False,
                    entries=[],
                    summary="未检测到新的更新",
                    raw_content=content,
                    related_doc_urls=[],  # 无更新时不设置相关文档
                    fingerprint=current_fingerprint,  # 即使无更新也记录 fingerprint
                    analysis_log=analysis_log,
                )
                return analysis
            else:
                analysis_log.baseline_match = False
                print_info(
                    f"检测到内容变化 (fingerprint: {baseline_fingerprint[:8]}... → {current_fingerprint[:8]}...)"
                )
        else:
            # 记录无基线的原因
            analysis_log.baseline_match = None  # 无基线，无法比较
            reason_map = {
                "none": "首次分析/无基线",
                "error": "基线读取失败",
            }
            reason = reason_map.get(baseline_source, "未知原因")
            logger.info(f"无基线比较: {reason}, current={current_fingerprint[:8]}...")
            print_info(f"首次分析，无基线比较 ({reason})")

        # 在清理前提取 HTML 中的链接（分类为 allowed 和 external）
        extracted_links = self._extract_links_from_html(content)
        allowed_link_urls = [link["href"] for link in extracted_links["allowed"]]
        external_link_urls = [link["href"] for link in extracted_links["external"]]

        # 记录链接提取统计
        analysis_log.links_extracted = {
            "allowed": len(allowed_link_urls),
            "external": len(external_link_urls),
        }

        entries = self.parse_changelog(content)
        analysis_log.entry_count = len(entries)
        print_info(f"解析到 {len(entries)} 个更新条目")

        analysis = self.extract_update_points(entries)
        analysis.raw_content = content
        analysis.fingerprint = current_fingerprint  # 记录 fingerprint 用于保存到 storage
        analysis.changelog_links = allowed_link_urls  # 保存允许抓取范围内的链接
        analysis.external_links = external_link_urls  # 保存外部链接（不在允许抓取范围内）

        # 完成结构化日志
        analysis_log.duration_ms = (time.time() - start_time) * 1000
        analysis_log.log_structured()
        analysis.analysis_log = analysis_log

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
            if analysis.changelog_links:
                print(f"  Changelog 内链接: {len(analysis.changelog_links)} 个")
            if analysis.external_links:
                print(f"  外部链接: {len(analysis.external_links)} 个")
        else:
            print_warning("未检测到新的更新")

        return analysis


# ============================================================
# 知识库更新
# ============================================================


class KnowledgeUpdater:
    """知识库更新器

    根据更新分析结果更新知识库。
    使用 KnowledgeManager 统一处理文档抓取与创建。

    特性：
    - changelog 文档：清洗内容存主体，raw HTML 存 metadata（用于追溯）
    - docs 文档：统一清洗为 Markdown 格式
    - 支持配置最大抓取 URL 数量
    - 支持配置 llms.txt 来源和缓存路径
    - dry_run 模式：禁止所有写入操作（缓存写入、知识库写入）
    - 支持通过 SideEffectPolicy 统一控制副作用

    副作用控制优先级（从高到低）：
    1. 显式参数（offline, disable_cache_write, dry_run）
    2. side_effect_policy 派生值
    3. 默认值（False）

    示例：
        # 使用 SideEffectPolicy 统一控制
        policy = compute_side_effects(skip_online=True, dry_run=True)
        updater = KnowledgeUpdater(side_effect_policy=policy)

        # 或使用传统参数（向后兼容）
        updater = KnowledgeUpdater(offline=True, dry_run=True)
    """

    def __init__(
        self,
        max_fetch_urls: int = DEFAULT_MAX_FETCH_URLS,
        fallback_core_docs_count: int = DEFAULT_FALLBACK_CORE_DOCS_COUNT,
        llms_txt_url: str = DEFAULT_LLMS_TXT_URL,
        llms_cache_path: Optional[str] = None,
        llms_local_fallback: Optional[str] = None,
        url_strategy_config: Optional[ResolvedURLStrategyConfig] = None,
        allowed_doc_url_prefixes: Optional[list[str]] = None,
        offline: bool = False,
        disable_cache_write: bool = False,
        fetch_policy: Optional[ResolvedFetchPolicyConfig] = None,
        changelog_url: str = DEFAULT_CHANGELOG_URL,
        dry_run: bool = False,
        # 新增：SideEffectPolicy 统一控制副作用
        side_effect_policy: Optional["SideEffectPolicy"] = None,
    ):
        # ============================================================
        # 副作用控制：从 side_effect_policy 派生或使用显式参数
        # ============================================================
        # 保存 policy 引用（用于调试和下游传递）
        self._side_effect_policy = side_effect_policy

        # 副作用控制优先级：显式参数 > policy 派生 > 默认值
        # 这确保向后兼容：已有代码使用显式参数不受影响
        if side_effect_policy is not None:
            # 从 policy 派生默认值，但允许显式参数覆盖
            # offline: 由 allow_network_fetch 控制
            if not offline:  # 仅当显式参数为 False 时才使用 policy
                offline = not side_effect_policy.allow_network_fetch
            # disable_cache_write: 由 allow_cache_write 控制
            if not disable_cache_write:
                disable_cache_write = not side_effect_policy.allow_cache_write
            # dry_run: 由 allow_file_write 控制
            if not dry_run:
                dry_run = not side_effect_policy.allow_file_write

        # fetch_policy 配置（用于 apply_fetch_policy 过滤外链）
        self.fetch_policy = fetch_policy
        # llms.txt 来源配置
        self.llms_txt_url = llms_txt_url
        # changelog URL（用于推导主域名）
        self.changelog_url = changelog_url

        # 从 llms_txt_url 和 changelog_url 推导主域名（用于内链/外链判定）
        from knowledge.doc_url_strategy import derive_primary_domains

        self._primary_domains = derive_primary_domains(
            llms_txt_url=self.llms_txt_url,
            changelog_url=self.changelog_url,
        )

        # 允许的文档 URL 前缀（完整 URL，用于 load_core_docs 过滤）
        self.allowed_doc_url_prefixes = (
            allowed_doc_url_prefixes if allowed_doc_url_prefixes is not None else DEFAULT_ALLOWED_DOC_URL_PREFIXES
        )

        # ============================================================
        # 构建 UrlPolicy（基于内部文档域名 + fetch_policy 放行范围）
        # ============================================================
        # 1. 内部文档域名（从 primary_domains 和 allowed_doc_url_prefixes 推导）
        # 2. fetch_policy 的 allowed_domains（外链域名白名单）
        # 合并为 WebFetcher/KnowledgeManager 使用的统一 UrlPolicy
        self._url_policy = self._build_url_policy(
            primary_domains=self._primary_domains,
            allowed_doc_url_prefixes=self.allowed_doc_url_prefixes,
            fetch_policy=fetch_policy,
        )

        # dry_run 模式：禁止所有写入操作
        self.dry_run = dry_run
        # 当 dry_run=True 时，强制禁用缓存写入
        if dry_run:
            disable_cache_write = True

        # 保留 storage 用于基线比较（向后兼容）
        # dry_run 模式下使用只读 storage，允许读取既有 index 但禁止写入
        if dry_run:
            self.storage = KnowledgeStorage.create_read_only()
        else:
            self.storage = KnowledgeStorage()
        # 保留 fetcher 用于 llms.txt 获取（内部使用），使用统一 UrlPolicy
        self.fetcher = WebFetcher(FetchConfig(timeout=60, url_policy=self._url_policy))
        # 主要使用 KnowledgeManager 处理文档，注入统一 UrlPolicy
        self.manager = KnowledgeManager(name=KB_NAME, url_policy=self._url_policy)
        # 可配置的抓取限制
        self.max_fetch_urls = max_fetch_urls
        self.fallback_core_docs_count = fallback_core_docs_count
        # 缓存路径（支持相对路径和绝对路径）
        if llms_cache_path is not None:
            cache_path = Path(llms_cache_path)
            if not cache_path.is_absolute():
                cache_path = project_root / cache_path
            self.llms_cache_path = cache_path
        else:
            self.llms_cache_path = LLMS_TXT_CACHE_PATH_RESOLVED
        # 本地回退路径
        if llms_local_fallback is not None:
            fallback_path = Path(llms_local_fallback)
            if not fallback_path.is_absolute():
                fallback_path = project_root / fallback_path
            self.llms_local_fallback = fallback_path
        else:
            self.llms_local_fallback = LLMS_TXT_LOCAL_FALLBACK
        # URL 策略配置（用于构建 DocURLStrategyConfig）
        self.url_strategy_config = url_strategy_config
        # 离线模式：跳过在线 fetch
        self.offline = offline
        # 禁用缓存写入：跳过缓存更新
        self.disable_cache_write = disable_cache_write

        # 构建统一的文档 URL 允许列表配置
        # 确保 _extract_links_from_html 和 _build_urls_to_fetch 使用相同的过滤规则
        self._doc_allowlist = build_doc_allowlist(
            url_strategy_config=url_strategy_config,
            allowed_doc_url_prefixes=self.allowed_doc_url_prefixes,
        )

    @staticmethod
    def _build_url_policy(
        primary_domains: list[str],
        allowed_doc_url_prefixes: list[str],
        fetch_policy: Optional[ResolvedFetchPolicyConfig],
    ) -> UrlPolicy:
        """构建 UrlPolicy（基于内部文档域名 + fetch_policy 放行范围）

        合并多个来源的域名/URL 前缀配置，生成统一的 UrlPolicy 用于
        WebFetcher 和 KnowledgeManager 的 URL 安全校验。

        Args:
            primary_domains: 主域名列表（从 llms_txt_url 和 changelog_url 推导）
            allowed_doc_url_prefixes: 允许的文档 URL 前缀（完整 URL）
            fetch_policy: 解析后的 fetch_policy 配置（可选）

        Returns:
            构建的 UrlPolicy 实例

        配置来源优先级：
        1. primary_domains: 始终包含（内部文档必需）
        2. allowed_doc_url_prefixes: 从 URL 前缀推导域名
        3. fetch_policy.allowed_domains: 外链域名白名单（如有）
        """
        # 1. 从 allowed_doc_url_prefixes 推导域名
        derived_domains = _derive_allowed_domains_from_prefixes(allowed_doc_url_prefixes)

        # 2. 合并所有域名来源
        all_domains: set[str] = set()
        # 添加主域名（内部文档必需）
        all_domains.update(primary_domains)
        # 添加从 URL 前缀推导的域名
        all_domains.update(derived_domains)
        # 添加 fetch_policy 的外链域名白名单（如有）
        if fetch_policy and fetch_policy.allowed_domains:
            all_domains.update(fetch_policy.allowed_domains)

        # 3. 构建 UrlPolicy
        # 注意：allowed_url_prefixes 保留原有的完整 URL 前缀配置
        # allowed_domains 使用合并后的域名白名单
        return UrlPolicy(
            allowed_schemes=["http", "https"],  # 默认只允许 HTTP/HTTPS
            allowed_domains=sorted(all_domains),  # 合并后的域名白名单
            allowed_url_prefixes=allowed_doc_url_prefixes,  # 原有 URL 前缀配置
            deny_private_networks=True,  # 拒绝私网地址（安全策略）
        )

    @property
    def url_policy(self) -> UrlPolicy:
        """获取当前使用的 URL 安全策略"""
        return self._url_policy

    async def initialize(self):
        """初始化"""
        await self.storage.initialize()
        await self.manager.initialize()
        await self.fetcher.initialize()  # 仍需初始化用于 llms.txt

    async def update_from_analysis(
        self,
        analysis: UpdateAnalysis,
        force: bool = False,
        changelog_url: str = DEFAULT_CHANGELOG_URL,
    ) -> dict[str, Any]:
        """根据更新分析结果更新知识库

        ================================================================
        阶段表 (Phase Table)
        ================================================================
        | 阶段 | 操作                        | 副作用类型   | 可配置跳过         |
        |------|-----------------------------|--------------|-------------------|
        | 1    | _save_changelog()           | 文件写入     | dry_run           |
        | 2    | _build_urls_to_fetch()      | 网络读取     | has_updates=False |
        | 3    | apply_fetch_policy()        | 无           | fetch_policy=None |
        | 4    | _fetch_related_docs()       | 文件写入     | dry_run           |

        ================================================================
        副作用矩阵 (Side Effect Matrix)
        ================================================================
        | 副作用类型   | 目标                          | 策略控制              |
        |--------------|-------------------------------|----------------------|
        | 网络读取     | llms_txt_url                  | offline 模式跳过     |
        | 网络读取     | urls_to_fetch                 | has_updates=False 跳过|
        | 文件写入     | .cursor/knowledge/docs/*.md   | dry_run=True 禁止    |
        | 文件写入     | .cursor/knowledge/metadata/*  | dry_run=True 禁止    |
        | 文件写入     | .cursor/knowledge/index.json  | dry_run=True 禁止    |
        | 缓存写入     | llms.txt 缓存文件             | disable_cache_write  |

        ================================================================
        入口参数到下游字段映射 (Parameter Mapping)
        ================================================================
        | 入口参数        | 下游使用位置                    | 说明                   |
        |-----------------|--------------------------------|------------------------|
        | analysis        | _save_changelog()              | raw_content/fingerprint|
        | analysis        | _build_urls_to_fetch()         | changelog_links 等     |
        | force           | _save_changelog/fetch_docs     | 强制覆盖已有文档       |
        | changelog_url   | _save_changelog()              | 文档 URL (baseline key)|
        | dry_run (实例)  | 所有写入操作                    | True 时跳过写入        |
        | fetch_policy    | apply_fetch_policy()           | 外链过滤策略           |

        ================================================================
        doc_url_strategy 调用点
        ================================================================
        - _build_urls_to_fetch(): 调用 select_urls_to_fetch()（Phase A 已生效）
        - apply_fetch_policy(): 调用 doc_url_strategy.apply_fetch_policy()
          * external_link_mode: record_only（默认）/ skip_all / fetch_allowlist
          * Phase A: 配置已解析，外链分类记录已生效
          * Phase B: 将实际执行抓取门控

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

        results: dict[str, Any] = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "urls_processed": [],
            "no_updates_detected": not analysis.has_updates,
            "changelog_url": changelog_url,  # 记录实际使用的 URL
        }

        # 1. 保存 Changelog 内容（使用 KnowledgeManager.add_content）
        if analysis.raw_content:
            changelog_result = await self._save_changelog(analysis, changelog_url, force)
            if changelog_result == "updated":
                results["updated"] += 1
            elif changelog_result == "skipped":
                results["skipped"] += 1
            else:
                results["failed"] += 1

        # 2. 获取并保存相关文档（仅在有更新时）
        # 当 has_updates=False（基线 fingerprint 匹配）时，跳过抓取以减少网络访问
        if not analysis.has_updates:
            print_info("无内容更新，跳过抓取相关文档")
            logger.debug("跳过 related docs 抓取: has_updates=False (fingerprint 匹配)")
        else:
            # 构建 urls_to_fetch，按优先级：
            # 1. changelog 内链接（优先级最高）
            # 2. keyword_map 命中的 related_doc_urls
            # 3. llms.txt 来源（关键词命中后取 Top-N）
            # 4. 回退到核心 docs
            urls_to_fetch, selection_log = await self._build_urls_to_fetch(
                analysis,
                max_urls=self.max_fetch_urls,
                fallback_count=self.fallback_core_docs_count,
            )
            results["url_selection_log"] = selection_log.to_dict()  # 记录结构化日志到结果

            # 3. apply_fetch_policy 过滤外链和内链路径 gate
            # 根据 fetch_policy.external_link_mode 策略处理外链：
            # - record_only：外链从 urls_to_fetch 移除但记录
            # - skip_all：外链既不抓取也不记录
            # - fetch_allowlist：仅允许白名单命中的外链
            # 当 enforce_path_prefixes=True 时，对内链执行阶段 4 路径前缀检查
            if urls_to_fetch and self.fetch_policy is not None:
                from knowledge.doc_url_strategy import apply_fetch_policy

                policy_result = apply_fetch_policy(
                    urls=urls_to_fetch,
                    fetch_policy_mode=self.fetch_policy.external_link_mode,
                    base_url=self.changelog_url,
                    primary_domains=self._primary_domains,
                    allowed_domains=self.fetch_policy.allowed_domains,
                    external_link_allowlist=self.fetch_policy.external_link_allowlist,
                    allowed_path_prefixes=self.fetch_policy.allowed_path_prefixes,
                    enforce_path_prefixes=self.fetch_policy.enforce_path_prefixes,
                )

                # 更新 urls_to_fetch 为过滤后的列表
                urls_to_fetch = policy_result.urls_to_fetch

                # 记录过滤结果到 selection_log（可观测）
                if policy_result.filtered_urls:
                    selection_log.fetch_policy_filtered = policy_result.filtered_urls
                    selection_log.external_links_recorded = policy_result.external_links_recorded
                    # 统计内链路径过滤的数量
                    internal_filtered = sum(
                        1 for f in policy_result.filtered_urls if f.get("reason") == "internal_link_path_not_allowed"
                    )
                    logger.info(
                        f"apply_fetch_policy 过滤: "
                        f"移除 {len(policy_result.filtered_urls)} 个 URL "
                        f"(内链路径 {internal_filtered}，外链 {len(policy_result.filtered_urls) - internal_filtered})，"
                        f"记录 {len(policy_result.external_links_recorded)} 个外链"
                    )

                # 更新结果中的日志
                results["url_selection_log"] = selection_log.to_dict()

            if urls_to_fetch:
                # 使用批量方式获取文档（KnowledgeManager.add_urls）
                doc_results = await self._fetch_related_docs(urls_to_fetch, force)
                results["urls_processed"] = urls_to_fetch
                results["updated"] += doc_results["updated"]
                results["skipped"] += doc_results["skipped"]
                results["failed"] += doc_results["failed"]
                results["persisted"] = doc_results.get("persisted", 0)

        # 显示统计
        print("\n知识库更新完成:")
        print(f"  更新: {results['updated']} 个")
        print(f"  跳过: {results['skipped']} 个")
        print(f"  失败: {results['failed']} 个")
        if results.get("persisted", 0) > 0:
            print(f"  持久化: {results['persisted']} 个")

        return results

    async def _save_changelog(
        self,
        analysis: UpdateAnalysis,
        changelog_url: str,
        force: bool,
    ) -> str:
        """保存 Changelog 内容

        策略：
        - 主内容存清洗后的稳定文本（使用 CHANGELOG 清洗模式）
        - raw HTML 存入 metadata（preserve_raw=True）
        - 持久化到 KnowledgeStorage（.cursor/knowledge/）

        Returns:
            "updated", "skipped", 或 "failed"
        """
        try:
            # 构建 metadata
            changelog_metadata: dict[str, Any] = {
                "source": "changelog",
                "category": "cursor-docs",
                "updated_at": datetime.now().isoformat(),
            }
            # 如果 analysis 包含 fingerprint，传入 metadata
            if analysis.fingerprint:
                changelog_metadata["original_fingerprint"] = analysis.fingerprint
                logger.debug(f"保存 changelog: fingerprint={analysis.fingerprint[:8]}...")

            # 如果 analysis 包含 external_links，写入 metadata（用于追溯）
            if analysis.external_links:
                changelog_metadata["external_links"] = analysis.external_links
                logger.debug(f"记录 {len(analysis.external_links)} 个外部链接到 metadata")

            # 使用 KnowledgeManager.add_content（CHANGELOG 清洗模式 + 保留原始内容）
            doc = await self.manager.add_content(
                url=changelog_url,
                content=analysis.raw_content,
                title="Cursor Changelog",
                metadata=changelog_metadata,
                force_refresh=force,
                clean_mode=ContentCleanMode.CHANGELOG,
                preserve_raw=True,  # 保留原始 HTML 到 metadata
            )

            if doc:
                # 持久化到 KnowledgeStorage
                try:
                    saved, msg = await self.storage.save_document(doc, force=force)
                    if saved:
                        logger.info(f"Changelog 持久化成功: {doc.id}")
                    else:
                        logger.debug(f"Changelog 持久化跳过: {msg}")
                except ReadOnlyStorageError as e:
                    # dry-run/minimal 模式下只读 storage 会阻止写入
                    logger.info(f"Changelog 持久化跳过（只读模式）: {e.operation}")
                    print_info("Changelog 已创建（临时上下文，未持久化）")

                print_success("Changelog 已保存")
                return "updated"
            else:
                print_info("Changelog 内容未变化或保存失败")
                return "skipped"

        except Exception as e:
            print_warning(f"Changelog 保存失败: {e}")
            logger.exception(f"Changelog 保存异常: {e}")
            return "failed"

    async def _fetch_related_docs(
        self,
        urls: list[str],
        force: bool,
    ) -> dict[str, int]:
        """使用 KnowledgeManager.add_urls 批量获取并保存相关文档

        Args:
            urls: URL 列表
            force: 是否强制更新

        Returns:
            统计结果 {"updated": n, "skipped": n, "failed": n, "persisted": n}
        """
        results = {"updated": 0, "skipped": 0, "failed": 0, "persisted": 0}

        if not urls:
            return results

        print_info(f"获取 {len(urls)} 个相关文档...")

        # 用于记录是否处于只读模式（仅打印一次提示）
        read_only_mode_logged = False

        try:
            # 使用 KnowledgeManager.add_urls 批量处理（MARKDOWN 清洗模式）
            docs = await self.manager.add_urls(
                urls=urls,
                metadata={
                    "source": "web",
                    "category": "cursor-docs",
                    "updated_at": datetime.now().isoformat(),
                },
                force_refresh=force,
                clean_mode=ContentCleanMode.MARKDOWN,  # docs 统一清洗为 Markdown
                preserve_raw=False,  # docs 不需要保留原始内容
            )

            # 统计结果
            added_urls = {doc.url for doc in docs}
            for url in urls:
                if url in added_urls:
                    # 查找对应文档并显示标题
                    for doc in docs:
                        if doc.url == url:
                            print_success(f"已更新: {doc.title}")
                            # 持久化到 KnowledgeStorage
                            try:
                                saved, msg = await self.storage.save_document(doc, force=force)
                                if saved:
                                    results["persisted"] += 1
                                    logger.debug(f"文档持久化成功: {doc.id}")
                                else:
                                    logger.debug(f"文档持久化跳过: {msg}")
                            except ReadOnlyStorageError as e:
                                # dry-run/minimal 模式下只读 storage 会阻止写入
                                if not read_only_mode_logged:
                                    logger.info(f"文档持久化跳过（只读模式）: {e.operation}")
                                    print_info("文档已创建（临时上下文，未持久化）")
                                    read_only_mode_logged = True
                            break
                    results["updated"] += 1
                else:
                    # 不在返回列表中可能是跳过（已存在）或失败
                    print_info(f"已存在/跳过: {url.split('/')[-1]}")
                    results["skipped"] += 1

        except Exception as e:
            print_error(f"批量获取文档失败: {e}")
            logger.exception(f"_fetch_related_docs 异常: {e}")
            results["failed"] = len(urls)

        return results

    async def _build_urls_to_fetch(
        self,
        analysis: UpdateAnalysis,
        max_urls: int = DEFAULT_MAX_FETCH_URLS,
        fallback_count: int = DEFAULT_FALLBACK_CORE_DOCS_COUNT,
    ) -> tuple[list[str], UrlSelectionLog]:
        """构建要抓取的 URL 列表

        ================================================================
        阶段表 (Phase Table)
        ================================================================
        | 阶段 | 操作                          | 副作用类型   | 可配置跳过   |
        |------|-------------------------------|--------------|-------------|
        | 1    | _extract_keywords_from_analysis | 无          | -           |
        | 2    | _fetch_llms_txt()             | 网络/文件读取| offline     |
        | 3    | load_core_docs()              | 无           | -           |
        | 4    | strategy_select_urls()        | 无           | -           |

        ================================================================
        副作用矩阵 (Side Effect Matrix)
        ================================================================
        | 副作用类型   | 目标                    | 策略控制              |
        |--------------|-------------------------|----------------------|
        | 网络读取     | llms_txt_url            | offline 模式跳过     |
        | 文件读取     | llms.txt 缓存           | 缓存命中时使用       |
        | 文件读取     | cursor_docs_full.txt    | 兜底本地文件         |

        ================================================================
        入口参数到下游字段映射 (Parameter Mapping)
        ================================================================
        | 入口参数        | 下游使用位置                    | 说明                   |
        |-----------------|--------------------------------|------------------------|
        | analysis        | changelog_links/related_doc_urls| URL 候选来源          |
        | max_urls        | DocURLStrategyConfig.max_urls  | 最大返回数量           |
        | fallback_count  | config.fallback_core_docs_count| 兜底补充数量           |

        ================================================================
        doc_url_strategy 调用点（Phase A 已生效）
        ================================================================
        - strategy_select_urls(): 调用 doc_url_strategy.select_urls_to_fetch()
          * 输入: changelog_links, llms_txt_content, related_doc_urls, core_docs
          * 配置: _doc_allowlist.config（DocURLStrategyConfig 实例）
          * 功能: 规范化、过滤、关键词匹配、优先级排序、去重、截断
          * Phase A 状态: 已完全生效，URL 选择逻辑已委托给策略模块

        使用 DocURLStrategy 模块进行优先级排序、关键词匹配、去重和截断。
        IO 操作（获取 llms.txt）保留在本类中。

        优先级（由 DocURLStrategy.select_urls_to_fetch 处理）：
        1. changelog 内链接（优先级最高）
        2. llms.txt 来源
        3. keyword_map 命中的 related_doc_urls
        4. 回退到核心 docs

        Args:
            analysis: 更新分析结果
            max_urls: 最大抓取 URL 数量
            fallback_count: 回退核心文档数量

        Returns:
            (去重后的 URL 列表, 结构化日志)
        """
        import time

        start_time = time.time()

        # 提取关键词（用于 DocURLStrategy 的关键词匹配）
        keywords = self._extract_keywords_from_analysis(analysis)

        # 获取 llms.txt 内容（IO 操作保留在此类）
        llms_txt_content = await self._fetch_llms_txt()

        # 使用统一构建的配置（与 _extract_links_from_html 保持一致）
        # 这确保链接分类和 URL 选择使用相同的过滤规则
        config = self._doc_allowlist.config

        # 特殊处理：方法参数可以覆盖 max_urls / fallback_core_docs_count
        # 这保持了向后兼容性（调用方可以自定义这两个值）
        # 但 allowed_url_prefixes / allowed_domains 等核心过滤规则使用统一配置
        if max_urls != DEFAULT_MAX_FETCH_URLS and max_urls != config.max_urls:
            # 方法参数显式指定了不同的值，创建新配置覆盖
            config = DocURLStrategyConfig(
                allowed_url_prefixes=config.allowed_url_prefixes,
                allowed_domains=config.allowed_domains,
                max_urls=max_urls,
                fallback_core_docs_count=fallback_count,
                prefer_changelog=config.prefer_changelog,
                deduplicate=config.deduplicate,
                normalize=config.normalize,
                keyword_boost_weight=config.keyword_boost_weight,
                exclude_patterns=config.exclude_patterns,
                priority_weights=config.priority_weights,
            )
        elif fallback_count != DEFAULT_FALLBACK_CORE_DOCS_COUNT and fallback_count != config.fallback_core_docs_count:
            # 仅 fallback_count 不同
            config = DocURLStrategyConfig(
                allowed_url_prefixes=config.allowed_url_prefixes,
                allowed_domains=config.allowed_domains,
                max_urls=config.max_urls,
                fallback_core_docs_count=fallback_count,
                prefer_changelog=config.prefer_changelog,
                deduplicate=config.deduplicate,
                normalize=config.normalize,
                keyword_boost_weight=config.keyword_boost_weight,
                exclude_patterns=config.exclude_patterns,
                priority_weights=config.priority_weights,
            )

        # 统计输入 URL 总数（用于计算过滤数量）
        changelog_links = analysis.changelog_links or []
        related_doc_urls = analysis.related_doc_urls or []
        llms_urls_raw = strategy_parse_llms_txt(llms_txt_content or "", base_url="https://cursor.com")
        # 使用配置的 allowed_doc_url_prefixes 加载核心文档
        core_docs = load_core_docs(
            allowed_url_prefixes=self.allowed_doc_url_prefixes,
        )

        # 记录输入统计
        input_counts = {
            "changelog_links": len(changelog_links),
            "related_doc_urls": len(related_doc_urls),
            "llms_txt": len(llms_urls_raw),
            "core_docs": len(core_docs),
        }
        total_input_urls = sum(input_counts.values())

        # 调用 DocURLStrategy 进行 URL 选择
        # 将解析、关键词匹配、优先级拼装、去重截断全部委托给 strategy
        # 使用 load_core_docs 动态加载核心文档 URL（根据配置的 allowed_doc_url_prefixes 过滤）
        # 注意：config.allowed_url_prefixes 会在 is_allowed_doc_url 中进行前缀过滤
        final_urls = strategy_select_urls(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=llms_txt_content,
            core_docs=core_docs,
            keywords=list(keywords),
            config=config,
            base_url="https://cursor.com",
        )

        # 计算过滤数量
        filtered_count = total_input_urls - len(final_urls)

        # 构建结构化日志
        selection_log = UrlSelectionLog(
            source="url_selection",
            input_counts=input_counts,
            output_count=len(final_urls),
            filtered_count=filtered_count,
            # 截断 URL 用于日志输出（避免泄露敏感信息）
            selected_urls=[sanitize_url_for_log(u) for u in final_urls[:10]],  # 只记录前 10 个
            duration_ms=(time.time() - start_time) * 1000,
        )

        # 输出结构化日志
        selection_log.log_structured()

        # 输出选择结果日志（包含过滤/去重数量）- 保留原有日志格式
        logger.info(
            f"构建 urls_to_fetch（via DocURLStrategy）: "
            f"输入 {total_input_urls} 个，最终 {len(final_urls)} 个，"
            f"过滤/去重 {filtered_count} 个"
        )
        return final_urls, selection_log

    def _extract_keywords_from_analysis(self, analysis: UpdateAnalysis) -> set[str]:
        """从更新分析结果中提取关键词

        Args:
            analysis: 更新分析结果

        Returns:
            关键词集合（小写）
        """
        keywords: set[str] = set()

        # 从 entries 提取关键词
        for entry in analysis.entries:
            keywords.update(kw.lower() for kw in entry.keywords if kw)

        # 从 new_features 提取关键词（提取方括号外的词）
        for feature in analysis.new_features:
            words = re.findall(r"\b\w{3,}\b", feature)
            keywords.update(w.lower() for w in words)

        return keywords

    async def _fetch_llms_txt(self) -> Optional[str]:
        """获取 llms.txt 内容

        ================================================================
        阶段表 (Phase Table)
        ================================================================
        | 阶段 | 操作                    | 副作用类型   | 可配置跳过         |
        |------|-------------------------|--------------|-------------------|
        | 1    | 在线 fetch llms_txt_url | 网络读取     | offline=True      |
        | 2    | 写入缓存                | 文件写入     | disable_cache_write|
        | 3    | 读取缓存文件            | 文件读取     | -                 |
        | 4    | 读取本地 fallback 文件  | 文件读取     | -                 |

        ================================================================
        副作用矩阵 (Side Effect Matrix)
        ================================================================
        | 副作用类型   | 目标                          | 策略控制              |
        |--------------|-------------------------------|----------------------|
        | 网络读取     | llms_txt_url                  | offline 模式跳过     |
        | 文件写入     | llms_cache_path               | disable_cache_write  |
        | 文件读取     | llms_cache_path               | 缓存命中时使用       |
        | 文件读取     | llms_local_fallback           | 兜底本地文件         |

        ================================================================
        入口参数到下游字段映射 (Parameter Mapping)
        ================================================================
        | 实例字段              | 下游使用位置              | 说明                   |
        |-----------------------|--------------------------|------------------------|
        | llms_txt_url          | fetcher.fetch()          | 在线获取目标           |
        | llms_cache_path       | Path.read_text/write_text| 缓存文件路径           |
        | llms_local_fallback   | Path.read_text()         | 仓库内置 fallback      |
        | offline               | 阶段 1 条件              | True 时跳过在线 fetch  |
        | disable_cache_write   | 阶段 2 条件              | True 时跳过缓存写入    |

        策略：
        1. 在线 fetch 成功 -> 写入缓存并返回（offline=True 时跳过此步骤）
        2. 在线失败 -> 读取缓存文件
        3. 缓存不存在 -> 读取仓库 cursor_docs_full.txt
        4. 都失败 -> 返回 None

        使用实例配置的 llms_txt_url 和 llms_cache_path。
        当 offline=True 时跳过在线 fetch，直接使用缓存/本地文件。
        当 disable_cache_write=True 时跳过缓存写入。

        Returns:
            llms.txt 内容，失败返回 None
        """
        # 1. 尝试在线 fetch（使用实例配置的 URL）
        # offline 模式下跳过在线 fetch
        if not self.offline:
            try:
                result = await self.fetcher.fetch(self.llms_txt_url)
                if result.success and result.content:
                    logger.debug(f"在线获取 llms.txt 成功 ({len(result.content)} 字符)")
                    # 写入缓存（disable_cache_write 时跳过）
                    if not self.disable_cache_write:
                        self._write_llms_txt_cache(result.content)
                    return result.content
            except Exception as e:
                logger.warning(f"在线获取 llms.txt 失败: {e}")
        else:
            logger.debug("offline 模式：跳过在线获取 llms.txt")

        # 2. 回退到缓存文件（使用实例配置的路径）
        try:
            if self.llms_cache_path.exists():
                content = self.llms_cache_path.read_text(encoding="utf-8")
                logger.debug(f"读取缓存 llms.txt 成功 ({len(content)} 字符)")
                return content
        except Exception as e:
            logger.warning(f"读取缓存 llms.txt 失败: {e}")

        # 3. 回退到仓库文件（使用实例配置的路径）
        try:
            if self.llms_local_fallback.exists():
                content = self.llms_local_fallback.read_text(encoding="utf-8")
                logger.debug(f"读取仓库 llms.txt 成功 ({len(content)} 字符)")
                return content
        except Exception as e:
            logger.warning(f"读取仓库 llms.txt 失败: {e}")

        return None

    def _write_llms_txt_cache(self, content: str) -> bool:
        """写入 llms.txt 缓存

        使用实例配置的 llms_cache_path。

        Args:
            content: llms.txt 内容

        Returns:
            是否写入成功
        """
        try:
            # 确保缓存目录存在
            self.llms_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.llms_cache_path.write_text(content, encoding="utf-8")
            logger.debug(f"写入 llms.txt 缓存成功: {self.llms_cache_path}")
            return True
        except Exception as e:
            logger.warning(f"写入 llms.txt 缓存失败: {e}")
            return False

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
                context.append(
                    {
                        "title": doc.title,
                        "url": doc.url,
                        "content": doc.content[:MAX_CONSOLE_PREVIEW_CHARS],
                        "score": result.score,
                    }
                )

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

        # 2. 目标工程目录（当用户显式指定 --directory 时）
        if context.project_info:
            project_info = context.project_info
            parts.append("\n## 目标工程目录\n")
            parts.append(f"- 路径: {project_info.path}\n")
            parts.append(f"- 状态: {project_info.state.value}\n")
            if project_info.detected_language:
                parts.append(f"- 检测语言: {project_info.detected_language}\n")
            if project_info.is_newly_initialized:
                parts.append("- **刚初始化的工程**：请根据用户需求补充完整功能\n")
            if project_info.marker_files:
                parts.append(f"- 标记文件: {', '.join(project_info.marker_files[:5])}\n")

        # 3. 参考子工程（仅参考/不要在其内直接改 unless 明确要求）
        if context.reference_projects:
            parts.append("\n## 参考工程（子目录）\n")
            parts.append("以下是在工作目录下发现的参考工程，可参考其代码结构和实现方式。\n")
            parts.append("**注意**: 仅参考，不要在其内直接修改，除非用户明确要求。\n")
            for i, ref in enumerate(context.reference_projects[:5], 1):
                parts.append(f"\n### {i}. {ref.relative_path}\n")
                parts.append(f"- 语言: {ref.detected_language or '未知'}\n")
                parts.append(f"- 标记文件: {', '.join(ref.marker_files[:3])}\n")
                if ref.description:
                    parts.append(f"- 描述: {ref.description}\n")

        # 4. 任务解析（Agent 结构化）
        if context.task_analysis:
            task_analysis = context.task_analysis
            parts.append("\n## 任务解析\n")
            if task_analysis.language:
                parts.append(f"- 语言: {task_analysis.language}\n")
            if task_analysis.project_name:
                parts.append(f"- 项目名: {task_analysis.project_name}\n")
            if task_analysis.framework:
                parts.append(f"- 框架: {task_analysis.framework}\n")
            if task_analysis.params:
                params_preview = ", ".join(f"{k}={v}" for k, v in list(task_analysis.params.items())[:5])
                parts.append(f"- 参数: {params_preview}\n")

        # 5. 用户需求
        if context.user_requirement:
            parts.append("\n## 用户需求\n")
            parts.append(context.user_requirement)
            parts.append("\n")

        # 5.1 迭代助手上下文（.iteration + Engram + 规则摘要）
        if context.iteration_assistant:
            import json

            parts.append("\n## 迭代上下文（.iteration / Engram / 规则）\n")
            payload = context.iteration_assistant.to_dict()
            parts.append("```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n")

        # 6. 更新分析
        if context.update_analysis and context.update_analysis.has_updates:
            update_analysis = context.update_analysis

            parts.append("\n## 检测到的更新\n")
            parts.append(f"摘要: {update_analysis.summary}\n")

            if update_analysis.new_features:
                parts.append("\n### 新功能\n")
                for feature in update_analysis.new_features[:5]:
                    parts.append(f"- {feature}\n")

            if update_analysis.improvements:
                parts.append("\n### 改进\n")
                for improvement in update_analysis.improvements[:5]:
                    parts.append(f"- {improvement}\n")

            if update_analysis.fixes:
                parts.append("\n### 修复\n")
                for fix in update_analysis.fixes[:5]:
                    parts.append(f"- {fix}\n")

        # 7. 知识库上下文
        if context.knowledge_context:
            parts.append("\n## 参考文档（来自知识库）\n")
            for i, doc in enumerate(context.knowledge_context[:3], 1):
                parts.append(f"\n### {i}. {doc['title']}\n")
                parts.append(f"URL: {doc['url']}\n")
                content_preview = doc["content"][:500]
                parts.append(f"```\n{content_preview}\n```\n")

        # 8. 执行指导
        parts.append("\n## 执行指导\n")
        parts.append("1. 分析上述更新和需求，确定需要修改的代码文件\n")
        parts.append("2. 更新代码以支持新功能或修复问题\n")
        parts.append("3. 确保修改与现有代码风格一致\n")
        parts.append("4. 更新相关文档（如 AGENTS.md）如有必要\n")

        return "".join(parts)

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

        if context.iteration_assistant and context.iteration_assistant.iteration_id:
            parts.append(f"迭代上下文: {context.iteration_assistant.iteration_id}")

        return "; ".join(parts) if parts else "无具体迭代目标"


# ============================================================
# 主执行流程
# ============================================================


class SelfIterator:
    """自我迭代器

    协调整个自我迭代流程。

    Cloud/Auto 语义统一说明:
    - cloud_enabled: 控制 '&' 前缀的自动检测，False 时 & 前缀视为普通字符
    - execution_mode=auto: Cloud 优先，按错误类型冷却后回退到 CLI
    - force_write: 独立于 auto_commit，由 --force 控制
    - auto_commit: 需显式 --auto-commit 开启

    配置 API Key 的三种方式:
      1. export CURSOR_API_KEY=your_key
      2. --cloud-api-key your_key
      3. agent login
    """

    # 类级别标志：跟踪已输出的消息（避免重复打印）
    _shown_messages: set = set()

    @classmethod
    def reset_shown_messages(cls) -> None:
        """重置已显示消息标志（主要用于测试）"""
        cls._shown_messages = set()

    def __init__(self, args: argparse.Namespace | Any):
        self.args = args
        # 解析文档源配置（tri-state: CLI > config.yaml > 默认值）
        self.docs_source_config = resolve_docs_source_config(args)

        # 解析 URL 策略配置（tri-state: CLI > config.yaml > 默认值）
        # 用于后续构建 DocURLStrategyConfig
        self.url_strategy_config = resolve_doc_url_strategy_config(args)

        # ============================================================
        # 副作用控制策略：使用 compute_side_effects 统一计算
        # ============================================================
        # 从 CLI 参数获取副作用控制标志
        skip_online = getattr(args, "skip_online", False)
        is_dry_run = getattr(args, "dry_run", False)
        is_minimal = getattr(args, "minimal", False)

        # 计算副作用控制策略
        self._side_effect_policy = compute_side_effects(
            skip_online=skip_online,
            dry_run=is_dry_run,
            minimal=is_minimal,
        )

        # 使用解析后的配置和 SideEffectPolicy 初始化 KnowledgeUpdater
        # SideEffectPolicy 统一控制：offline, disable_cache_write, dry_run
        self.knowledge_updater = KnowledgeUpdater(
            max_fetch_urls=self.docs_source_config.max_fetch_urls,
            fallback_core_docs_count=self.docs_source_config.fallback_core_docs_count,
            llms_txt_url=self.docs_source_config.llms_txt_url,
            llms_cache_path=self.docs_source_config.llms_cache_path,
            url_strategy_config=self.url_strategy_config,
            allowed_doc_url_prefixes=self.docs_source_config.allowed_doc_url_prefixes,
            fetch_policy=self.docs_source_config.fetch_policy,  # 用于 apply_fetch_policy 过滤外链
            changelog_url=self.docs_source_config.changelog_url,  # 用于推导主域名
            side_effect_policy=self._side_effect_policy,  # 统一副作用控制
        )
        # 传递 storage 和统一的 doc_allowlist 给 ChangelogAnalyzer
        # 确保 _extract_links_from_html 和 _build_urls_to_fetch 使用相同的过滤规则
        self.changelog_analyzer = ChangelogAnalyzer(
            self.docs_source_config.changelog_url,
            storage=self.knowledge_updater.storage,
            doc_allowlist=self.knowledge_updater._doc_allowlist,
        )
        self.goal_builder = IterationGoalBuilder()

        # 解析工作目录（支持相对路径和绝对路径）
        work_dir = Path(args.directory).resolve()
        self.working_directory = work_dir

        # ============================================================
        # 执行决策：统一使用 build_execution_decision 构建决策对象
        # ============================================================
        # 决策对象包含：
        # - effective_mode: 有效执行模式 (cli/cloud/auto/plan/ask)
        # - requested_mode: 请求的执行模式
        # - orchestrator: 编排器类型 (mp/basic)
        # - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
        # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud）
        # - triggered_by_prefix: prefix_routed 的兼容别名
        # - sanitized_prompt: 清理后的 prompt（移除 & 前缀）
        # - user_message: 用户友好消息（仅构建，由入口脚本决定是否打印）

        # === 使用 compute_decision_inputs 统一构建决策输入 ===
        # 此 helper 函数封装了以下逻辑：
        # - 获取 CLI 参数 (execution_mode, orchestrator, no_mp)
        # - 获取 Cloud 配置 (cloud_enabled, has_api_key)
        # - 检测 & 前缀（语法层面）
        # - 使用 resolve_requested_mode_for_decision 确定 requested_mode
        # - 使用 resolve_mode_source 确定 mode_source
        decision_inputs = compute_decision_inputs(args, original_prompt=args.requirement)

        # 使用集中化的不变式验证（避免入口分叉）
        from core.config import get_config

        config = get_config()
        validate_requested_mode_invariant(
            has_ampersand_prefix=decision_inputs.has_ampersand_prefix,
            cli_execution_mode=getattr(args, "execution_mode", None),
            requested_mode_for_decision=decision_inputs.requested_mode,
            config_execution_mode=config.cloud_agent.execution_mode,
            caller_name="SelfIterator.__init__",
        )

        # 使用 DecisionInputs.build_decision() 构建 ExecutionDecision
        self._execution_decision: ExecutionDecision = decision_inputs.build_decision()

        # 使用决策对象的字段
        # 字段语义说明：
        # - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
        # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
        # - _triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出，由 prefix_routed 派生）
        self._has_ampersand_prefix = self._execution_decision.has_ampersand_prefix
        self._prefix_routed = self._execution_decision.prefix_routed
        # 兼容别名：仅用于输出，内部分支统一使用 _prefix_routed
        self._triggered_by_prefix = self._prefix_routed
        user_requirement = self._execution_decision.sanitized_prompt

        # 统一输出用户提示消息（集中在一个位置打印一次，使用稳定哈希去重）
        # 根据 message_level 决定使用 print_warning 还是 print_info
        if self._execution_decision.user_message:
            msg_key = compute_message_dedup_key(self._execution_decision.user_message)
            if msg_key not in SelfIterator._shown_messages:
                SelfIterator._shown_messages.add(msg_key)
                if self._execution_decision.message_level == "warning":
                    print_warning(self._execution_decision.user_message)
                else:
                    print_info(self._execution_decision.user_message)

        # 记录决策日志
        logger.debug(
            f"执行决策: effective_mode={self._execution_decision.effective_mode}, "
            f"orchestrator={self._execution_decision.orchestrator}, "
            f"prefix_routed={self._prefix_routed}"
        )
        logger.info(f"执行模式策略: {self._execution_decision.mode_reason}")

        self.context = IterationContext(
            user_requirement=user_requirement,
            dry_run=args.dry_run,
        )
        self._iteration_context_payload: Optional[dict[str, Any]] = None

        # 保存 _directory_user_set 标记（用于工程准备逻辑）
        self._directory_user_set = getattr(args, "_directory_user_set", False)

        # ============================================================
        # Minimal 模式处理
        # ============================================================
        # 解析 minimal 标志并应用 MINIMAL_PRESET 覆盖
        # 权威定义来源: core/execution_policy.py Side Effect Control Strategy Matrix
        # minimal = skip_online + dry_run（禁止所有网络请求、文件写入、Git 操作）
        self._is_minimal = getattr(args, "minimal", False)
        if self._is_minimal:
            # 应用 MINIMAL_PRESET 覆盖
            args.skip_online = True  # 禁止网络请求
            args.dry_run = True  # 禁止文件写入和 Git 操作
            # 同步到 context（确保 dry_run 一致）
            self.context.dry_run = True
            logger.info(
                "minimal 模式启用: skip_online=True, dry_run=True, "
                "跳过知识库初始化和上下文加载，不执行 Orchestrator.run"
            )

        # 统一解析配置（CLI 参数覆盖 config.yaml）
        self._resolved_settings = self._resolve_config_settings()

    async def _load_iteration_assistant_context(self) -> None:
        """加载迭代助手上下文（.iteration + Engram + 规则摘要）"""
        iteration_id = getattr(self.args, "iteration_id", None)
        iteration_source = getattr(self.args, "iteration_source", "auto") or "auto"
        iteration_git_policy = getattr(self.args, "iteration_git_policy", "auto") or "auto"
        allow_bootstrap = not getattr(self.args, "no_iteration_bootstrap", False)
        agent_path = getattr(self._resolved_settings, "agent_path", "agent")

        self.context.iteration_assistant = await build_iteration_assistant_context(
            working_directory=self.working_directory,
            side_effect_policy=self._side_effect_policy,
            iteration_id=iteration_id,
            iteration_source=iteration_source,
            iteration_git_policy=iteration_git_policy,
            allow_bootstrap=allow_bootstrap,
            agent_path=agent_path,
        )

        self._iteration_context_payload = self.context.iteration_assistant.to_dict()

    def _get_iteration_context_payload(self) -> Optional[dict[str, Any]]:
        if self._iteration_context_payload:
            return self._iteration_context_payload
        if self.context.iteration_assistant:
            self._iteration_context_payload = self.context.iteration_assistant.to_dict()
            return self._iteration_context_payload
        return None

    async def _prepare_workspace(self) -> bool:
        """工程准备（仅当用户显式指定 --directory 时触发）

        逻辑：
        1. 探测目录状态
        2. 如果是空目录/仅文档目录：
           a. 推断语言（从任务文本）
           b. 如果推断成功，生成脚手架
           c. 如果推断失败，返回提示信息并退出
        3. 无论是否生成脚手架，都发现参考子工程
        4. 将结果缓存到 self.context

        只读模式保护：
        - 当 --dry-run 时保持只读语义：不写入文件
        - 仅输出"建议创建的工程结构/需要指定语言"的提示并返回

        Returns:
            True 表示工程准备成功，False 表示需要用户干预
        """
        print_section("工程准备")

        # 调用统一的工程准备函数
        result = prepare_workspace(
            target_dir=self.working_directory,
            task_text=self.context.user_requirement,
            explicit_language=None,  # 目前不支持显式语言参数
            force_scaffold=False,
            dry_run=self.context.dry_run,
        )

        # 缓存结果到 context
        self.context.workspace_preparation = result
        self.context.project_info = result.project_info
        self.context.reference_projects = result.reference_projects
        self.context.task_analysis = result.task_analysis

        # 处理工程准备结果
        if result.error:
            # 工程准备失败（如无法推断语言）
            print_error(f"工程准备失败: {result.error}")
            if result.hint:
                print("\n" + result.hint)
            return False

        # 显示工程信息
        project_info = result.project_info
        if project_info.is_newly_initialized:
            print_success(f"已初始化 {project_info.detected_language} 工程")
            if result.scaffold_result:
                print_info(f"创建文件: {', '.join(result.scaffold_result.created_files)}")
        elif project_info.state == ProjectState.EXISTING_PROJECT:
            print_info(f"检测到已有工程: {project_info.detected_language or '未知语言'}")
            if project_info.marker_files:
                print_info(f"标记文件: {', '.join(project_info.marker_files[:3])}")
        elif project_info.state == ProjectState.HAS_SOURCE_FILES:
            print_info(
                f"检测到源码文件 ({project_info.source_files_count} 个)，语言: {project_info.detected_language or '未知'}"
            )

        # 显示参考子工程
        if result.reference_projects:
            print_info(f"发现 {len(result.reference_projects)} 个参考子工程:")
            for ref in result.reference_projects[:3]:
                print(f"  - {ref.relative_path}: {ref.description}")

        return True

    def _resolve_config_settings(self) -> ResolvedSettings:
        """统一解析配置设置

        使用 build_unified_overrides 统一构建配置，确保与 run.py 入口的逻辑一致。
        从 config.yaml 加载基础配置，CLI 参数覆盖（仅当显式指定时）。

        设计要点:
        - 使用 build_unified_overrides(args, execution_decision) 得到 UnifiedOptions
        - 用 UnifiedOptions.resolved 填充 ResolvedSettings 的核心字段
        - 保持 resolve_settings 仅用于 stream_json 等扩展配置，避免重复覆盖核心字段
        - 内部分支统一使用 prefix_routed（由 execution_decision 提供）

        字段语义说明：
        - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
        - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）

        Returns:
            ResolvedSettings 包含完整的解析后配置
        """
        # 使用统一的 build_unified_overrides 构建配置
        # 传入 execution_decision 确保：
        # 1. prefix_routed 来源于 decision.prefix_routed（内部分支使用）
        # 2. 编排器选择与 _execution_decision 保持一致
        # 3. CLI overrides 构建逻辑与 run.py 一致
        unified_options: UnifiedOptions = build_unified_overrides(
            args=self.args,
            execution_decision=self._execution_decision,
        )

        # 从 UnifiedOptions.resolved 获取核心配置
        resolved = unified_options.resolved

        # 使用 resolve_settings 仅获取扩展配置（stream_json 等）
        # 核心字段（workers/max_iterations/cloud_timeout 等）由 unified_options.resolved 提供
        stream_settings = resolve_settings(
            cli_workers=None,  # 核心配置已由 build_unified_overrides 处理
            cli_max_iterations=None,
            cli_cloud_timeout=None,
            cli_cloud_auth_timeout=None,
            cli_execution_mode=None,
            # 流式日志配置从 CLI 透传
            cli_stream_events_enabled=None,  # 目前无对应 CLI 参数
            cli_stream_log_console=None,
            cli_stream_log_detail_dir=None,
            cli_stream_log_raw_dir=None,
        )

        # 构建 ResolvedSettings，核心字段从 unified_options.resolved，扩展配置从 resolve_settings
        return ResolvedSettings(
            # 模型配置（从 unified_options.resolved）
            planner_model=resolved["planner_model"],
            worker_model=resolved["worker_model"],
            reviewer_model=resolved["reviewer_model"],
            # 超时配置（从 unified_options.resolved）
            planning_timeout=resolved["planner_timeout"],
            execution_timeout=resolved["worker_timeout"],
            review_timeout=resolved["reviewer_timeout"],
            agent_cli_timeout=stream_settings.agent_cli_timeout,  # 从 resolve_settings
            cloud_timeout=resolved["cloud_timeout"],
            cloud_auth_timeout=resolved["cloud_auth_timeout"],
            # 系统配置（从 unified_options.resolved）
            max_iterations=resolved["max_iterations"],
            worker_pool_size=resolved["workers"],
            enable_sub_planners=resolved["enable_sub_planners"],
            strict_review=resolved["strict_review"],
            # Cloud Agent 配置（混合来源）
            cloud_enabled=stream_settings.cloud_enabled,  # 从 resolve_settings
            execution_mode=resolved["execution_mode"],
            cloud_api_base_url=stream_settings.cloud_api_base_url,  # 从 resolve_settings
            # 流式日志配置（从 resolve_settings）
            stream_events_enabled=stream_settings.stream_events_enabled,
            stream_log_console=stream_settings.stream_log_console,
            stream_log_detail_dir=stream_settings.stream_log_detail_dir,
            stream_log_raw_dir=stream_settings.stream_log_raw_dir,
        )

    def _build_side_effects(self) -> dict[str, bool]:
        """构建副作用状态字典

        根据当前模式配置返回各类副作用的启用状态。
        minimal 模式下所有副作用均禁止。

        Returns:
            副作用状态字典，包含:
            - network: 是否允许网络请求
            - disk_write: 是否允许磁盘写入
            - knowledge_write: 是否允许知识库写入
        """
        if self._is_minimal:
            # minimal 模式：全部禁止
            return {
                "network": False,
                "disk_write": False,
                "knowledge_write": False,
            }

        # 非 minimal 模式：根据各参数独立判断
        skip_online = getattr(self.args, "skip_online", False)
        dry_run = getattr(self.args, "dry_run", False)

        return {
            "network": not skip_online,
            "disk_write": not dry_run,
            "knowledge_write": not skip_online and not dry_run,
        }

    def _build_dry_run_stats(self) -> dict[str, int]:
        """构建 dry-run 模式统计信息

        收集本应执行但因 dry-run 而跳过的操作数量。

        Returns:
            dry-run 统计字典，包含:
            - would_fetch_urls: 本应抓取的 URL 数量
            - would_write_docs: 本应写入的文档数量
            - would_write_cache: 本应写入的缓存文件数量
        """
        stats = {
            "would_fetch_urls": 0,
            "would_write_docs": 0,
            "would_write_cache": 0,
        }

        # 从 update_analysis 获取本应抓取的 URL 数量
        if self.context.update_analysis:
            analysis = self.context.update_analysis
            # changelog 中提取的链接
            changelog_links = len(analysis.changelog_links or [])
            # related_doc_urls
            related_docs = len(analysis.related_doc_urls or [])
            stats["would_fetch_urls"] = changelog_links + related_docs

            # 如果有更新，本应写入 changelog 文档
            if analysis.has_updates or analysis.raw_content:
                stats["would_write_docs"] += 1
            # 加上 related docs 数量（假设全部成功抓取会写入）
            stats["would_write_docs"] += min(stats["would_fetch_urls"], self.knowledge_updater.max_fetch_urls)

        # 如果不是 skip_online 模式，可能会写入 llms.txt 缓存
        if not getattr(self.args, "skip_online", False):
            stats["would_write_cache"] = 1  # llms.txt cache

        return stats

    def _build_execution_decision_fields(self) -> dict[str, Any]:
        """构建执行决策相关字段

        从 _execution_decision 对象提取关键字段，用于统一注入到所有返回分支。

        Returns:
            执行决策字段字典，包含:
            - has_ampersand_prefix: 语法检测，原始 prompt 是否有 & 前缀
            - prefix_routed: 策略决策，& 前缀是否成功触发 Cloud 模式
            - triggered_by_prefix: prefix_routed 的兼容别名
            - requested_mode: 原始请求模式
            - effective_mode: 有效执行模式
            - orchestrator: 编排器类型
        """
        decision = self._execution_decision
        return {
            "has_ampersand_prefix": decision.has_ampersand_prefix,
            "prefix_routed": decision.prefix_routed,
            "triggered_by_prefix": decision.triggered_by_prefix,  # 兼容别名
            "requested_mode": decision.requested_mode,
            "effective_mode": decision.effective_mode,
            "orchestrator": decision.orchestrator,
        }

    async def run(self) -> dict[str, Any]:
        """执行自我迭代流程

        ================================================================
        阶段表 (Phase Table)
        ================================================================
        | 步骤 | 阶段                      | 副作用类型     | 可配置跳过           |
        |------|---------------------------|----------------|---------------------|
        | 0    | _prepare_workspace()      | 文件读取       | 非显式 --directory   |
        | 1    | changelog_analyzer.analyze| 网络读取       | skip_online/minimal |
        | 2    | knowledge_updater.init    | 目录创建       | minimal             |
        | 2    | update_from_analysis()    | 网络+文件写入  | skip_online/minimal |
        | 3    | get_knowledge_context()   | 文件读取       | minimal             |
        | 4    | goal_builder.build_goal() | 无             | -                   |
        | 5    | _run_agent_system()       | 文件写入+Git   | dry_run             |

        ================================================================
        副作用矩阵 (Side Effect Matrix)
        ================================================================
        | 副作用类型   | 目标                          | 策略控制               |
        |--------------|-------------------------------|------------------------|
        | 网络读取     | changelog_url                 | skip_online/minimal    |
        | 网络读取     | llms_txt_url                  | skip_online/minimal    |
        | 网络读取     | urls_to_fetch                 | skip_online/minimal    |
        | 目录创建     | .cursor/knowledge/            | minimal 时跳过         |
        | 文件写入     | .cursor/knowledge/**          | dry_run=True 禁止      |
        | 缓存写入     | llms.txt 缓存                 | disable_cache_write    |
        | Git 操作     | 提交/推送                     | auto_commit 控制       |

        ================================================================
        入口参数到下游字段映射 (Parameter Mapping)
        ================================================================
        | CLI 参数              | 下游字段                      | 说明                   |
        |-----------------------|------------------------------|------------------------|
        | --skip-online         | args.skip_online             | 跳过步骤 1 在线分析    |
        | --dry-run             | args.dry_run                 | 步骤 5 仅预览不执行    |
        | --minimal             | _is_minimal                  | 最小化模式             |
        | --directory           | working_directory            | 目标工程目录           |
        | --auto-commit         | args.auto_commit             | 启用自动提交           |
        | --auto-push           | args.auto_push               | 启用自动推送           |
        | --force-update        | args.force_update            | 强制覆盖已有文档       |
        | --execution-mode      | _execution_decision          | 执行模式决策           |
        | --orchestrator        | _execution_decision          | 编排器类型             |
        | --changelog-url       | docs_source_config           | Changelog URL          |
        | --llms-txt-url        | docs_source_config           | llms.txt URL           |

        ================================================================
        doc_url_strategy 调用点总结
        ================================================================
        本方法通过以下路径间接调用 doc_url_strategy 模块：

        1. changelog_analyzer.analyze() → _extract_links_from_html()
           - 使用 doc_allowlist.config 进行链接分类（allowed/external）
           - Phase A 已生效：链接分类逻辑已委托给策略模块

        2. knowledge_updater.update_from_analysis() → _build_urls_to_fetch()
           - 调用 select_urls_to_fetch() 进行 URL 选择
           - Phase A 已生效：URL 优先级、去重、截断已委托给策略模块

        3. knowledge_updater.update_from_analysis() → apply_fetch_policy()
           - 根据 external_link_mode 过滤外链
           - Phase A 状态：配置已解析，外链记录已生效，实际抓取门控 Phase B 启用

        Returns:
            执行结果
        """
        print_header("自我迭代脚本")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"工作目录: {self.working_directory}")
        print(f"用户需求: {self.args.requirement or '(无)'}")
        if self._prefix_routed:
            print("Cloud 模式: 启用（检测到 '&' 前缀成功触发）")
            print(f"实际任务: {self.context.user_requirement}")
        print(f"跳过在线检查: {self.args.skip_online}")
        print(f"仅分析模式: {self.args.dry_run}")
        # Minimal 模式摘要（简洁输出，仅在启用时显示详情）
        if self._is_minimal:
            print("最小化模式: 启用")
            print("  - 跳过在线文档检查")
            print("  - 跳过知识库初始化")
            print("  - 跳过知识库上下文加载")
            print("  - 直接进入迭代目标构建")
        if self.args.auto_commit:
            print("自动提交: 启用")
            print(f"自动推送: {'启用' if self.args.auto_push else '禁用'}")
            if self.args.commit_message:
                print(f"提交信息前缀: {self.args.commit_message}")

        try:
            # 步骤 0: 工程准备（仅当用户显式指定 --directory 时）
            if self._directory_user_set:
                workspace_ok = await self._prepare_workspace()
                if not workspace_ok:
                    # 工程准备失败（如无法推断语言），已输出提示，直接返回
                    # 使用输出契约字段常量确保字段名一致
                    return {
                        IterateResultFields.SUCCESS: False,
                        IterateResultFields.ERROR: "工程准备失败",
                        IterateResultFields.HINT: self.context.workspace_preparation.hint
                        if self.context.workspace_preparation
                        else None,
                        IterateResultFields.MINIMAL: self._is_minimal,
                        IterateResultFields.SIDE_EFFECTS: self._build_side_effects(),
                        IterateResultFields.COOLDOWN_INFO: None,
                        **self._build_execution_decision_fields(),
                    }

            # 加载迭代助手上下文（.iteration + Engram + 规则摘要）
            await self._load_iteration_assistant_context()

            # 步骤 1: 分析在线更新
            # 使用 SideEffectPolicy 控制是否调用 ChangelogAnalyzer.analyze
            # - allow_network_fetch=False: 跳过在线分析（skip_online/minimal 模式）
            policy = self._side_effect_policy
            if not policy.allow_network_fetch:
                if policy.minimal:
                    print_step(1, "跳过在线文档检查（minimal 模式）")
                else:
                    print_step(1, "跳过在线文档检查（skip_online 模式）")
                self.context.update_analysis = UpdateAnalysis()
            else:
                print_step(1, "分析在线文档更新")
                self.context.update_analysis = await self.changelog_analyzer.analyze()

            # 步骤 2: 更新知识库
            # 使用 SideEffectPolicy 控制是否调用 KnowledgeUpdater.initialize/update_from_analysis
            # - allow_directory_create=False: 跳过知识库初始化（dry_run/minimal 模式）
            # - allow_network_fetch=False: 跳过 update_from_analysis（skip_online/minimal 模式）
            if not policy.allow_directory_create:
                if policy.minimal:
                    print_step(2, "跳过知识库初始化（minimal 模式）")
                    print_info("minimal 模式：不初始化知识库，不进行在线 fetch")
                else:
                    print_step(2, "跳过知识库初始化（dry_run 模式）")
                    print_info("dry_run 模式：不初始化知识库")
            else:
                print_step(2, "更新知识库")
                await self.knowledge_updater.initialize()

                if policy.allow_network_fetch and self.context.update_analysis:
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
            # 使用 SideEffectPolicy 控制是否加载知识库上下文
            # - allow_directory_create=False: 跳过（因为目录可能未创建）
            # - minimal 模式下跳过（保持 knowledge_context 为空）
            if not policy.allow_directory_create:
                if policy.minimal:
                    print_step(3, "跳过知识库上下文加载（minimal 模式）")
                    print_info("minimal 模式：不加载知识库上下文")
                else:
                    print_step(3, "跳过知识库上下文加载（dry_run 模式）")
                    print_info("dry_run 模式：不加载知识库上下文")
                # knowledge_context 保持空，iteration_goal 中不包含知识库章节
                self.context.knowledge_context = []
            else:
                print_step(3, "加载知识库上下文")

                # 根据用户需求或更新内容搜索相关文档
                search_query = self.args.requirement
                if not search_query and self.context.update_analysis:
                    # 使用更新关键词搜索
                    keywords = []
                    for entry in self.context.update_analysis.entries[:3]:
                        keywords.extend(entry.keywords[:2])
                    search_query = " ".join(keywords) if keywords else "CLI agent"

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
                        doc_entries = await self.knowledge_updater.storage.list_documents(limit=3)
                        for doc_entry in doc_entries:
                            doc = await self.knowledge_updater.storage.load_document(doc_entry.doc_id)
                            if doc:
                                self.context.knowledge_context.append(
                                    {
                                        "title": doc.title,
                                        "url": doc.url,
                                        "content": doc.content[:MAX_CONSOLE_PREVIEW_CHARS],
                                        "score": 0.5,
                                    }
                                )

                    print_info(f"找到 {len(self.context.knowledge_context)} 个相关文档")

            # 步骤 4: 总结迭代目标
            print_step(4, "总结迭代目标")
            self.context.iteration_goal = self.goal_builder.build_goal(self.context)

            summary = self.goal_builder.get_summary(self.context)
            print_info(f"迭代摘要: {summary}")

            # 仅分析模式
            if self.args.dry_run:
                print_section("迭代目标预览（dry-run 模式）")
                print(self.context.iteration_goal[:MAX_CONSOLE_PREVIEW_CHARS])
                if len(self.context.iteration_goal) > MAX_CONSOLE_PREVIEW_CHARS:
                    print("\n" + TRUNCATION_HINT)

                # dry-run 专属提示：显示本应执行的操作
                dry_run_stats = self._build_dry_run_stats()
                print_section("Dry-run 副作用摘要（未执行）")
                print("  知识库存储模式: 只读 (read_only=True)")
                print("  缓存写入: 已禁用 (disable_cache_write=True)")
                print(f"  本应抓取的 URL 数量: {dry_run_stats.get('would_fetch_urls', 0)}")
                print(f"  本应写入的文档数量: {dry_run_stats.get('would_write_docs', 0)}")
                print(f"  本应写入的缓存文件: {dry_run_stats.get('would_write_cache', 0)}")
                print("  (以上操作在 dry-run 模式下均未执行)")

                # 使用输出契约字段常量确保字段名一致
                return {
                    IterateResultFields.SUCCESS: True,
                    IterateResultFields.DRY_RUN: True,
                    IterateResultFields.SUMMARY: summary,
                    IterateResultFields.GOAL_LENGTH: len(self.context.iteration_goal),
                    IterateResultFields.MINIMAL: self._is_minimal,
                    IterateResultFields.SIDE_EFFECTS: self._build_side_effects(),
                    IterateResultFields.DRY_RUN_STATS: dry_run_stats,
                    IterateResultFields.COOLDOWN_INFO: None,
                    **self._build_execution_decision_fields(),
                }

            # 步骤 5: 启动 Agent 系统
            print_step(5, "启动 Agent 系统执行迭代")
            result = await self._run_agent_system()

            # 注入 minimal 和 side_effects 元信息（使用输出契约字段常量）
            result[IterateResultFields.MINIMAL] = self._is_minimal
            result[IterateResultFields.SIDE_EFFECTS] = self._build_side_effects()

            # 注入执行决策字段
            result.update(self._build_execution_decision_fields())

            # 确保 cooldown_info 字段存在（即使为 None）
            result.setdefault(IterateResultFields.COOLDOWN_INFO, None)

            return result

        except Exception as e:
            logger.exception(f"自我迭代失败: {e}")
            print_error(f"执行失败: {e}")
            # 使用输出契约字段常量确保字段名一致
            return {
                IterateResultFields.SUCCESS: False,
                IterateResultFields.ERROR: str(e),
                IterateResultFields.MINIMAL: self._is_minimal,
                IterateResultFields.SIDE_EFFECTS: self._build_side_effects(),
                IterateResultFields.COOLDOWN_INFO: None,
                **self._build_execution_decision_fields(),
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

        # 创建 CommitterAgent - 使用用户指定的工作目录
        cursor_config = CursorAgentConfig(working_directory=str(self.working_directory))
        committer_config = CommitterConfig(
            working_directory=str(self.working_directory),
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
        commit_summary = commit_message.split("\n")[0][:50]
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

        统一读取顺序：
        优先使用 __init__ 中已构建的 _execution_decision 对象的 effective_mode 快照，
        确保决策一致性。所有用户提示消息已在 __init__ 中统一输出，这里不重复打印。

        字段语义说明：
        - effective_mode: 有效执行模式（由 build_execution_decision 生成的快照）
        - prefix_routed: 策略决策（用于内部分支），不在此方法使用
        - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）

        Returns:
            ExecutionMode 枚举值
        """
        # 使用已保存的决策对象的 effective_mode 快照，避免重复决策
        effective_mode = self._execution_decision.effective_mode

        logger.debug(f"执行模式决策: {effective_mode} ({self._execution_decision.mode_reason})")

        # 映射到 ExecutionMode 枚举
        mode_map = {
            "cli": ExecutionMode.CLI,
            "cloud": ExecutionMode.CLOUD,
            "auto": ExecutionMode.AUTO,
            "plan": ExecutionMode.PLAN,
            "ask": ExecutionMode.ASK,
        }
        return mode_map.get(effective_mode, ExecutionMode.CLI)

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

        配置 API Key 的三种方式:
          1. export CURSOR_API_KEY=your_key
          2. --cloud-api-key your_key
          3. agent login

        优先级: --cloud-api-key > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml

        Returns:
            CloudAuthConfig 或 None（未配置 key 时）
        """
        # 使用 CloudClientFactory.resolve_api_key 统一解析 API Key
        # 优先级: CLI 显式参数 > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
        cli_api_key = getattr(self.args, "cloud_api_key", None)
        api_key = CloudClientFactory.resolve_api_key(explicit_api_key=cli_api_key)

        if not api_key:
            return None

        # 从 config.yaml 获取默认配置
        cloud_config = build_cloud_client_config()

        # 解析 auth_timeout，优先级: CLI > config.yaml > DEFAULT_*
        cli_auth_timeout = getattr(self.args, "cloud_auth_timeout", None)
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

    def _get_orchestrator_type(self) -> str:
        """获取编排器类型

        使用 __init__ 中已构建的 _execution_decision 对象，确保决策一致性。

        编排器选择规则（已在 build_execution_decision 中统一处理）:
        1. 用户显式设置的编排器选项（--orchestrator 或 --no-mp）
        2. prefix_routed=True: 强制 basic（& 前缀成功触发 Cloud，内部分支使用 prefix_routed）
        3. requested_mode=auto/cloud: 强制 basic（与 CLI help 对齐）
        4. 从 requirement 中检测的非并行关键词（本方法额外处理）
        5. 默认值 "mp"

        字段语义说明：
        - prefix_routed: 策略决策（用于内部分支判断）
        - triggered_by_prefix: prefix_routed 的兼容别名（仅用于输出）

        返回值与 _execution_decision.orchestrator 保持一致，
        但额外支持从 requirement 检测非并行关键词。

        Returns:
            "mp" 或 "basic"
        """
        # 使用已保存的决策对象
        decision_orchestrator = self._execution_decision.orchestrator

        # 如果决策是 basic（由于 Cloud/Auto 模式或 prefix_routed）
        # 检查是否需要输出编排器兼容性警告
        if decision_orchestrator == "basic":
            # 判断是否是因为 Cloud/Auto 强制切换的（而非用户显式设置）
            reason = self._execution_decision.orchestrator_reason
            is_cloud_forced = (
                "Cloud 模式" in reason
                or "execution_mode" in reason.lower()
                or "cloud" in reason.lower()
                or "auto" in reason.lower()
            )

            # 检查用户是否期望使用 MP（没有显式设置 --orchestrator basic 或 --no-mp）
            user_explicitly_set_basic = getattr(self.args, "orchestrator", None) == "basic" or getattr(
                self.args, "no_mp", False
            )

            if is_cloud_forced and not user_explicitly_set_basic:
                # 用户没有显式设置 basic，但因 Cloud 模式被强制切换，输出警告
                print_warning(f"⚠ {reason}\n→ 已自动切换到基本协程编排器（basic）")
                logger.warning(reason)

            return "basic"

        # 决策允许 mp，但额外检查 requirement 中的非并行关键词
        # 注意：仅当用户未显式设置编排器时，才从 requirement 检测关键词
        user_explicitly_set_orchestrator = getattr(self.args, "_orchestrator_user_set", False)
        if not user_explicitly_set_orchestrator:
            requirement = getattr(self.args, "requirement", "")
            if requirement and _detect_disable_mp_from_requirement(requirement):
                logger.debug("从 requirement 检测到非并行关键词，选择 basic 编排器")
                return "basic"

        return decision_orchestrator

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

        返回结果包含以下元数据字段（不覆盖编排器原有字段）:
        - orchestrator_type: 最终使用的编排器类型 ("mp" 或 "basic")
        - orchestrator_requested: 请求的编排器类型
        - fallback_occurred: 是否发生了回退
        - fallback_reason: 回退原因（如果有）
        - execution_mode: 执行模式
        - max_iterations_configured: 配置的最大迭代次数

        Returns:
            执行结果
        """
        print_section("Agent 系统执行")

        # 使用 resolved settings 中的 max_iterations（已处理 tri-state 和 config.yaml）
        max_iterations = self._resolved_settings.max_iterations
        if max_iterations == -1:
            print_info("无限迭代模式已启用（按 Ctrl+C 中断）")

        # 获取执行模式
        execution_mode = self._get_execution_mode()
        print_info(f"执行模式: {execution_mode.value}")

        # 创建知识库管理器
        manager = KnowledgeManager(name=KB_NAME)
        await manager.initialize()

        # 确定编排器类型（警告已在 _get_orchestrator_type 中输出）
        orchestrator_type = self._get_orchestrator_type()
        orchestrator_requested = orchestrator_type  # 记录请求的编排器类型
        use_fallback = False
        fallback_reason: Optional[str] = None
        result: dict[str, Any] = {}

        if orchestrator_type == "mp":
            # 尝试使用多进程编排器
            result = await self._run_with_mp_orchestrator(max_iterations, manager)

            # 检查是否需要回退
            if result.get("_fallback_required"):
                use_fallback = True
                fallback_reason = result.get("_fallback_reason", "未知错误")
                orchestrator_type = "basic"  # 更新为实际使用的编排器类型
                print_warning(f"MP 编排器启动失败: {fallback_reason}")
                print_warning("正在回退到基本协程编排器...")
                logger.warning(f"MP 编排器回退: {fallback_reason}")
        else:
            # 直接使用基本编排器（非回退，而是显式选择）
            if execution_mode != ExecutionMode.CLI:
                print_info(f"使用基本协程编排器（执行模式: {execution_mode.value}）")
                fallback_reason = f"execution_mode={execution_mode.value} 不支持 MP 编排器"
            else:
                print_info("使用基本协程编排器（--no-mp 或 --orchestrator basic）")
                fallback_reason = "用户显式指定 basic 编排器"

        if use_fallback or orchestrator_type == "basic":
            # 使用基本协程编排器
            result = await self._run_with_basic_orchestrator(max_iterations, manager)

        # 注入元数据字段（不覆盖编排器原有字段）
        if "orchestrator_type" not in result:
            result["orchestrator_type"] = orchestrator_type
        if "orchestrator_requested" not in result:
            result["orchestrator_requested"] = orchestrator_requested
        if "fallback_occurred" not in result:
            result["fallback_occurred"] = use_fallback
        if "fallback_reason" not in result:
            result["fallback_reason"] = fallback_reason
        if "execution_mode" not in result:
            result["execution_mode"] = execution_mode.value
        if "max_iterations_configured" not in result:
            result["max_iterations_configured"] = max_iterations

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

        # 保护：如果知识库更新器尚未初始化，则跳过增强
        # 单元测试中可能直接调用私有执行方法（未执行 run() 的初始化流程），
        # 此时触发 storage.initialize()/向量索引初始化可能导致耗时或卡住。
        storage_inited = getattr(self.knowledge_updater.storage, "_initialized", False)
        manager_inited = getattr(self.knowledge_updater.manager, "_initialized", False)
        if not storage_inited or not manager_inited:
            logger.debug("知识库更新器未初始化，跳过知识库增强")
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
                # 截断过长内容（使用统一的知识库文档预览限制）
                content = doc["content"][:MAX_KNOWLEDGE_DOC_PREVIEW_CHARS]
                if len(doc["content"]) > MAX_KNOWLEDGE_DOC_PREVIEW_CHARS:
                    content += TRUNCATION_HINT
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

        配置注入策略:
        1. 从 _resolved_settings 获取 config.yaml 中的基础配置
        2. CLI 参数覆盖对应配置（--workers/--max-iterations 等）
        3. 构建 MultiProcessOrchestratorConfig 时注入:
           - models: planner_model/worker_model/reviewer_model
           - timeouts: planning_timeout/execution_timeout/review_timeout
           - stream 日志与渲染选项
        4. 确保 execution_mode!=cli 时自动使用 basic 编排器（在 _get_orchestrator_type 中处理）

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

        # 使用 resolved settings 获取配置（CLI 参数已覆盖）
        settings = self._resolved_settings

        print_info(f"MP 编排器配置: worker_count={settings.worker_pool_size}, max_iterations={max_iterations}")

        try:
            # 方案 B: 在 goal 构建阶段注入知识库上下文
            enhanced_goal = await self._build_enhanced_goal(self.context.iteration_goal, manager)

            # 计算日志级别（优先级: --log-level > --verbose/--quiet > 默认 INFO）
            log_level = self._get_log_level()

            # 计算卡死诊断配置（quiet 模式下降级）
            stall_diagnostics_enabled = self._get_stall_diagnostics_enabled()
            stall_diagnostics_level = self._get_stall_diagnostics_level()

            # 构建 MultiProcessOrchestratorConfig，注入 models/timeouts/stream 配置
            config = MultiProcessOrchestratorConfig(
                working_directory=str(self.working_directory),
                max_iterations=max_iterations,
                worker_count=settings.worker_pool_size,
                iteration_context=self._get_iteration_context_payload(),
                # 从 resolved settings 注入系统配置
                enable_sub_planners=settings.enable_sub_planners,
                strict_review=settings.strict_review,
                # 从 resolved settings 注入模型配置
                planner_model=settings.planner_model,
                worker_model=settings.worker_model,
                reviewer_model=settings.reviewer_model,
                # 从 resolved settings 注入超时配置
                planning_timeout=settings.planning_timeout,
                execution_timeout=settings.execution_timeout,
                review_timeout=settings.review_timeout,
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
                # 注意：execution_mode!=cli 时会在 _get_orchestrator_type 中自动回退到 basic
                execution_mode=getattr(self.args, "execution_mode", "cli"),
                planner_execution_mode=getattr(self.args, "planner_execution_mode", None),
                worker_execution_mode=getattr(self.args, "worker_execution_mode", None),
                reviewer_execution_mode=getattr(self.args, "reviewer_execution_mode", None),
                # 从 resolved settings 注入流式日志配置
                stream_events_enabled=settings.stream_events_enabled,
                stream_log_console=settings.stream_log_console,
                stream_log_detail_dir=settings.stream_log_detail_dir,
                stream_log_raw_dir=settings.stream_log_raw_dir,
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
                # 流式控制台渲染配置透传（CLI 参数）
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
            logger.info(
                f"MP 编排器已创建: worker_count={config.worker_count}, "
                f"知识库注入={config.enable_knowledge_injection}, "
                f"models=(planner={settings.planner_model}, worker={settings.worker_model}, reviewer={settings.reviewer_model})"
            )

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

        配置注入策略:
        1. 从 _resolved_settings 获取 config.yaml 中的基础配置
        2. CLI 参数覆盖对应配置（--workers/--max-iterations/--cloud-timeout 等）
        3. 构建 CursorAgentConfig 时注入 agent_cli/cloud_agent/stream_json 参数
        4. 构建 OrchestratorConfig 时注入 models/timeouts/enable_sub_planners/strict_review

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

        # 使用 resolved settings 获取配置（CLI 参数已覆盖）
        settings = self._resolved_settings

        print_info(f"协程编排器配置: worker_pool_size={settings.worker_pool_size}, max_iterations={max_iterations}")
        print_info(f"执行模式: {execution_mode.value}")
        if cloud_auth_config:
            print_info("Cloud 认证: 已配置")

        # 根据执行模式决定超时时间
        # - Cloud/Auto 模式：使用 resolved settings 中的 cloud_timeout
        # - CLI 模式：使用 resolved settings 中的 agent_cli_timeout
        if execution_mode in (ExecutionMode.CLOUD, ExecutionMode.AUTO):
            timeout = settings.cloud_timeout
            print_info(f"Cloud 超时: {timeout}s")
            logger.info(f"Cloud/Auto 执行模式，使用 cloud_timeout={timeout}s")
        else:
            timeout = settings.agent_cli_timeout

        # 构建 CursorAgentConfig，注入 agent_cli/cloud_agent/stream_json 参数
        cursor_config = CursorAgentConfig(
            working_directory=str(self.working_directory),
            timeout=timeout,
            # 流式日志配置注入（从 config.yaml）
            stream_events_enabled=settings.stream_events_enabled,
            stream_log_console=settings.stream_log_console,
            stream_log_detail_dir=settings.stream_log_detail_dir,
            stream_log_raw_dir=settings.stream_log_raw_dir,
            # Cloud Agent 配置注入
            cloud_enabled=settings.cloud_enabled,
            cloud_api_base=settings.cloud_api_base_url,
            cloud_timeout=settings.cloud_timeout,
        )

        # 解析角色级执行模式（可选）
        planner_exec_mode = self._parse_execution_mode(getattr(self.args, "planner_execution_mode", None))
        worker_exec_mode = self._parse_execution_mode(getattr(self.args, "worker_execution_mode", None))
        reviewer_exec_mode = self._parse_execution_mode(getattr(self.args, "reviewer_execution_mode", None))

        # 构建 OrchestratorConfig，注入 models/timeouts/enable_sub_planners/strict_review
        config = OrchestratorConfig(
            working_directory=str(self.working_directory),
            max_iterations=max_iterations,
            worker_pool_size=settings.worker_pool_size,
            cursor_config=cursor_config,
            iteration_context=self._get_iteration_context_payload(),
            # 从 resolved settings 注入系统配置
            enable_sub_planners=settings.enable_sub_planners,
            strict_review=settings.strict_review,
            # 从 resolved settings 注入模型配置
            planner_model=settings.planner_model,
            worker_model=settings.worker_model,
            reviewer_model=settings.reviewer_model,
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
            # 流式日志配置透传（从 config.yaml）
            stream_events_enabled=settings.stream_events_enabled,
            stream_log_console=settings.stream_log_console,
            stream_log_detail_dir=settings.stream_log_detail_dir,
            stream_log_raw_dir=settings.stream_log_raw_dir,
            # 流式控制台渲染配置透传（CLI 参数）
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
            f"execution_mode={config.execution_mode.value}, "
            f"models=(planner={settings.planner_model}, worker={settings.worker_model}, reviewer={settings.reviewer_model})"
        )

        # 执行
        print_info("开始执行迭代任务...")
        result = await orchestrator.run(self.context.iteration_goal)
        return result

    def _print_execution_result(self, result: dict[str, Any]) -> None:
        """打印执行结果

        统一读取结果中的 failure_kind/retry_after/cooldown_info，
        确保 & 前缀和 --execution-mode cloud 两种触发场景文案一致。

        Args:
            result: 执行结果字典
        """
        # 统一输出回退消息（仅输出一次，使用 prepare_cooldown_message 纯函数计算 dedup_key 和 level）
        msg_output = prepare_cooldown_message(result.get(IterateResultFields.COOLDOWN_INFO))
        if msg_output and msg_output.dedup_key not in SelfIterator._shown_messages:
            SelfIterator._shown_messages.add(msg_output.dedup_key)
            if msg_output.level == "warning":
                print_warning(msg_output.user_message)
            else:
                print_info(msg_output.user_message)

        print_section("执行结果")
        print(f"状态: {'成功' if result.get('success') else '未完成'}")
        print(f"迭代次数: {result.get('iterations_completed', 0)}")
        print(f"任务创建: {result.get('total_tasks_created', 0)}")
        print(f"任务完成: {result.get('total_tasks_completed', 0)}")
        print(f"任务失败: {result.get('total_tasks_failed', 0)}")

        if result.get("final_score"):
            print(f"最终评分: {result['final_score']:.1f}")

        # 显示提交信息
        commits_info = result.get("commits", {})
        if commits_info:
            print_section("提交信息")
            total_commits = commits_info.get("total_commits", 0)
            if total_commits > 0:
                print_success(f"提交数量: {total_commits}")

                # 显示提交哈希
                commit_hashes = commits_info.get("commit_hashes", [])
                if commit_hashes:
                    for i, hash_val in enumerate(commit_hashes[-3:], 1):  # 显示最近3个
                        print(f"  提交 {i}: {hash_val[:8] if len(hash_val) > 8 else hash_val}")

                # 显示提交信息摘要
                commit_messages = commits_info.get("commit_messages", [])
                if commit_messages:
                    print("提交信息摘要:")
                    for msg in commit_messages[-3:]:  # 显示最近3条
                        # 截取第一行作为摘要
                        summary = msg.split("\n")[0][:60]
                        print(f"  - {summary}")

                # 显示推送状态
                pushed_commits = commits_info.get("pushed_commits", 0)
                if self.args.auto_push:
                    if pushed_commits > 0:
                        print_success(f"推送状态: 已推送 {pushed_commits} 个提交到远程仓库")
                    else:
                        # 开启了 auto_push 但推送失败，显示警告
                        print_warning("推送状态: 推送失败，请手动执行 git push")
                elif result.get("pushed"):
                    print_success("推送状态: 已推送到远程仓库")
            else:
                print_info("无代码更改，未创建提交")

        # 显示 minimal 模式和副作用状态（仅在启用时显示）
        if result.get("minimal"):
            print_section("副作用控制")
            print("模式: minimal（最小副作用）")
            side_effects = result.get("side_effects", {})
            if side_effects:
                print(f"  - network: {'允许' if side_effects.get('network') else '禁止'}")
                print(f"  - disk_write: {'允许' if side_effects.get('disk_write') else '禁止'}")
                print(f"  - knowledge_write: {'允许' if side_effects.get('knowledge_write') else '禁止'}")


# ============================================================
# 日志配置
# ============================================================


def setup_logging(verbose: bool = False, quiet: bool = False, log_level: str | None = None):
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


def main():
    """主函数"""
    args = parse_args()
    setup_logging(
        verbose=args.verbose,
        quiet=getattr(args, "quiet", False),
        log_level=getattr(args, "log_level", None),
    )

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
            source_label="scripts/run_iterate.py",
            has_api_key=has_api_key,
        )
        sys.exit(0)

    # 关键词检测：仅当用户未显式设置编排器时，从 requirement 检测是否需要禁用 MP
    # 条件：no_mp=False 且 orchestrator 为默认值 "mp" 且 _orchestrator_user_set=False
    if (
        not getattr(args, "no_mp", False)
        and getattr(args, "orchestrator", "mp") == "mp"
        and not getattr(args, "_orchestrator_user_set", False)
    ) and _detect_disable_mp_from_requirement(args.requirement):
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
