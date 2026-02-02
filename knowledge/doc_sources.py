"""文档源加载模块

提供从配置文件加载核心文档 URL 的功能。
支持从多个列表文件（如 cursor_cli_docs.txt、cursor_agent_docs.txt）读取 URL，
跳过注释和空行，进行 normalize + allowed_url_prefixes 过滤 + 稳定去重。

典型用法:
    from knowledge.doc_sources import load_core_docs

    # 使用默认配置加载
    urls = load_core_docs()

    # 指定自定义文件列表
    urls = load_core_docs(source_files=["my_docs.txt"])
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from knowledge.doc_url_strategy import (
    DocURLStrategyConfig,
    deduplicate_urls,
    is_allowed_doc_url,
    normalize_url,
)

# 默认文档源文件列表（相对于项目根目录）
DEFAULT_DOC_SOURCE_FILES = [
    "cursor_cli_docs.txt",
    "cursor_agent_docs.txt",
]

# ============================================================
# 默认允许的文档 URL 前缀
# ============================================================
#
# 用途：过滤 load_core_docs 加载的核心文档 URL
# 格式：**完整 URL 前缀**（含 scheme/host），如 "https://cursor.com/docs"
#
# 与其他配置的区别：
# | 配置项                              | 格式            | 用途                       |
# |-------------------------------------|-----------------|----------------------------|
# | DEFAULT_ALLOWED_DOC_URL_PREFIXES    | 完整 URL 前缀   | load_core_docs 过滤        |
# | fetch_policy.allowed_path_prefixes  | 路径前缀        | 在线抓取时的 URL 过滤      |
# | url_strategy.allowed_url_prefixes   | 完整 URL 前缀   | URL 选择与优先级排序       |
# ============================================================
DEFAULT_ALLOWED_DOC_URL_PREFIXES = [
    "https://cursor.com/cn/docs",
    "https://cursor.com/docs",
    "https://cursor.com/cn/changelog",
    "https://cursor.com/changelog",
]

# ============================================================
# [DEPRECATED] 旧名别名 - DEFAULT_ALLOWED_DOC_DOMAINS
# ============================================================
#
# 新名称: DEFAULT_ALLOWED_DOC_URL_PREFIXES
# 旧名称: DEFAULT_ALLOWED_DOC_DOMAINS (误导性命名，实际是 URL 前缀而非域名)
#
# 迁移指南:
#   旧代码: from knowledge.doc_sources import DEFAULT_ALLOWED_DOC_DOMAINS
#   新代码: from knowledge.doc_sources import DEFAULT_ALLOWED_DOC_URL_PREFIXES
#
# 移除计划: 将在 v2.0 版本中移除此别名
# ============================================================
DEFAULT_ALLOWED_DOC_DOMAINS = DEFAULT_ALLOWED_DOC_URL_PREFIXES

# ============================================================
# Deprecated 警告机制（统一管理，每类警告仅输出一次）
# ============================================================
#
# 统一的 deprecated 警告 key 命名规则：
# - doc_sources.param.allowed_domains.*: 参数 allowed_domains 的废弃警告
#
# 测试用关键文案片段（用于断言，防止未来修改破坏兼容提示）：
# - "[DEPRECATED]" (统一前缀)
# - "已弃用" (中文提示)
# - "此别名将在未来版本中移除" (移除计划)
# ============================================================

# 记录已警告的旧别名（避免重复警告）
_deprecated_alias_warned: set[str] = set()

# deprecated 警告的统一 key 常量（便于测试断言和维护）
DEPRECATED_KEY_PARAM_ALLOWED_DOMAINS_IS_VALID = "doc_sources.param.allowed_domains.is_valid_doc_url"
DEPRECATED_KEY_PARAM_ALLOWED_DOMAINS_LOAD_CORE = "doc_sources.param.allowed_domains.load_core_docs"
DEPRECATED_KEY_PARAM_ALLOWED_DOMAINS_WITH_FALLBACK = "doc_sources.param.allowed_domains.load_core_docs_with_fallback"

# deprecated 警告的关键文案片段（用于测试断言）
DEPRECATED_MSG_PREFIX = "[DEPRECATED]"
DEPRECATED_MSG_ALIAS_DEPRECATED = "已弃用"
DEPRECATED_MSG_WILL_REMOVE = "此别名将在未来版本中移除"


def _warn_deprecated_alias(old_name: str, new_name: str) -> bool:
    """发出旧别名的弃用警告（每个别名仅警告一次）

    Args:
        old_name: 旧名称（同时作为 key 使用）
        new_name: 新名称

    Returns:
        True 如果发出了警告，False 如果该别名已经警告过
    """
    if old_name in _deprecated_alias_warned:
        return False
    _deprecated_alias_warned.add(old_name)
    logger.warning(
        f"{DEPRECATED_MSG_PREFIX} '{old_name}' {DEPRECATED_MSG_ALIAS_DEPRECATED}，请使用 '{new_name}'。"
        f"{DEPRECATED_MSG_WILL_REMOVE}。"
    )
    return True


def reset_deprecated_alias_warnings() -> None:
    """重置 deprecated 别名警告状态（仅供测试使用）

    清空已警告的别名集合，使警告可以再次触发。
    """
    _deprecated_alias_warned.clear()


def _get_project_root() -> Path:
    """获取项目根目录

    通过查找特征文件（如 config.yaml）来确定项目根目录。

    Returns:
        项目根目录的 Path 对象
    """
    # 从当前文件向上查找
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    # 回退到 knowledge 目录的父目录
    return Path(__file__).resolve().parent.parent


def parse_url_list_file(
    file_path: Path,
    base_url: Optional[str] = None,
) -> list[str]:
    """解析 URL 列表文件

    读取文件内容，跳过注释行（以 # 开头）和空行，
    返回 URL 列表（保持原始顺序）。

    Args:
        file_path: 文件路径
        base_url: 基础 URL（用于规范化相对路径）

    Returns:
        URL 列表（未去重、未过滤）

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if not file_path.exists():
        raise FileNotFoundError(f"URL 列表文件不存在: {file_path}")

    urls: list[str] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"读取文件失败 {file_path}: {e}")
        return urls

    for line in content.splitlines():
        # 去除首尾空白
        line = line.strip()

        # 跳过空行
        if not line:
            continue

        # 跳过注释行（以 # 开头）
        if line.startswith("#"):
            continue

        # 规范化 URL（如果提供了 base_url）
        line = normalize_url(line, base_url) if base_url else normalize_url(line)

        # 跳过规范化后为空的 URL
        if line:
            urls.append(line)

    return urls


def is_valid_doc_url(
    url: str,
    allowed_url_prefixes: Optional[list[str]] = None,
    *,
    allowed_domains: Optional[list[str]] = None,  # [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes
    config: Optional[DocURLStrategyConfig] = None,  # 支持直接传入配置
) -> bool:
    """检查 URL 是否为有效的文档 URL

    使用前缀匹配检查 URL 是否属于允许的文档 URL 前缀。
    内部委托给 doc_url_strategy.is_allowed_doc_url() 实现统一的过滤逻辑。

    Args:
        url: 待检查的 URL
        allowed_url_prefixes: 允许的 URL 前缀列表（None 使用默认值）
        allowed_domains: [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes（将在未来版本移除）
        config: 直接传入的策略配置（优先级最高）

    Returns:
        True 如果 URL 有效，False 否则
    """
    if not url:
        return False

    # 如果传入了 config，直接使用
    if config is not None:
        return is_allowed_doc_url(url, config)

    # [DEPRECATED] 如果使用了旧参数名 allowed_domains，发出警告
    if allowed_domains is not None and allowed_url_prefixes is None:
        _warn_deprecated_alias("allowed_domains (is_valid_doc_url 参数)", "allowed_url_prefixes")

    # 构造配置：优先使用 allowed_url_prefixes，其次 allowed_domains（向后兼容）
    prefixes = allowed_url_prefixes or allowed_domains or DEFAULT_ALLOWED_DOC_URL_PREFIXES

    # 构造 DocURLStrategyConfig 并委托给 is_allowed_doc_url
    strategy_config = DocURLStrategyConfig(
        allowed_url_prefixes=prefixes,
        exclude_patterns=[],  # 兼容模式：不应用排除规则
        normalize=True,  # 启用规范化
    )

    return is_allowed_doc_url(url, strategy_config)


def load_core_docs(
    source_files: Optional[list[str]] = None,
    project_root: Optional[Path] = None,
    allowed_url_prefixes: Optional[list[str]] = None,
    base_url: str = "https://cursor.com",
    fallback_urls: Optional[list[str]] = None,
    *,
    allowed_domains: Optional[list[str]] = None,  # [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes
) -> list[str]:
    """从配置文件加载核心文档 URL

    从指定的一个或多个列表文件读取 URL，按顺序合并，
    进行 normalize + allowed_url_prefixes 过滤 + 稳定去重。

    处理流程:
    1. 按顺序读取每个源文件
    2. 解析 URL（跳过注释和空行）
    3. 规范化 URL
    4. 过滤不符合 allowed_url_prefixes 的 URL
    5. 稳定去重（保持首次出现的顺序）
    6. 如果结果为空，使用 fallback_urls

    三态语义设计（allowed_url_prefixes）:
    - None: 使用默认值 DEFAULT_ALLOWED_DOC_URL_PREFIXES
    - []: 不使用 prefixes 限制，加载并保留文件内所有 URL
    - ["..."]: 使用指定的前缀列表进行过滤

    Args:
        source_files: URL 列表文件路径（相对于项目根目录），None 使用默认值
        project_root: 项目根目录，None 自动检测
        allowed_url_prefixes: 允许的 URL 前缀列表（三态语义：None=默认值，[]=不限制，[...]=指定过滤）
        base_url: 基础 URL（用于规范化相对路径）
        fallback_urls: 当所有文件都无法读取时使用的回退 URL 列表
        allowed_domains: [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes（将在未来版本移除）

    Returns:
        去重后的核心文档 URL 列表（确定性顺序）

    Examples:
        >>> urls = load_core_docs()
        >>> len(urls) > 0
        True

        >>> urls = load_core_docs(source_files=["cursor_cli_docs.txt"])
        >>> "https://cursor.com/cn/docs/cli/overview" in urls
        True

        >>> # 空列表表示不限制，加载所有 URL
        >>> urls = load_core_docs(allowed_url_prefixes=[])
        >>> len(urls) > 0  # 所有 URL 都被保留
        True
    """
    # 获取项目根目录
    root = project_root or _get_project_root()

    # 使用默认文件列表
    files = source_files or DEFAULT_DOC_SOURCE_FILES

    # [DEPRECATED] 如果使用了旧参数名 allowed_domains，发出警告
    if allowed_domains is not None and allowed_url_prefixes is None:
        _warn_deprecated_alias("allowed_domains (load_core_docs 参数)", "allowed_url_prefixes")

    # 允许的 URL 前缀（三态语义）
    # - None: 使用默认值 DEFAULT_ALLOWED_DOC_URL_PREFIXES
    # - []: 不使用 prefixes 限制，加载并保留文件内所有 URL（回退到 is_allowed_doc_url 的 allow-all 逻辑）
    # - ["..."]: 使用指定的前缀列表进行过滤
    if allowed_url_prefixes is not None:
        prefixes = allowed_url_prefixes
    elif allowed_domains is not None:
        prefixes = allowed_domains
    else:
        prefixes = DEFAULT_ALLOWED_DOC_URL_PREFIXES

    # 构造过滤配置（统一使用 doc_url_strategy.is_allowed_doc_url）
    # 当 prefixes 为空列表 [] 时，is_allowed_doc_url 会回退到 allowed_domains 检查
    # 如果 allowed_domains 也为空，则允许所有 URL
    filter_config = DocURLStrategyConfig(
        allowed_url_prefixes=prefixes,
        allowed_domains=[],  # 空列表，不使用域名过滤（prefixes 为空时 allow-all）
        exclude_patterns=[],  # 加载核心文档时不应用排除规则
        normalize=True,  # 启用规范化以确保匹配一致性
    )

    all_urls: list[str] = []
    files_read = 0

    # 按顺序读取每个源文件
    for file_name in files:
        file_path = root / file_name

        if not file_path.exists():
            logger.debug(f"文档源文件不存在，跳过: {file_path}")
            continue

        try:
            urls = parse_url_list_file(file_path, base_url)
            all_urls.extend(urls)
            files_read += 1
            logger.debug(f"从 {file_name} 读取了 {len(urls)} 个 URL")
        except Exception as e:
            logger.warning(f"读取文档源文件失败 {file_path}: {e}")
            continue

    # 过滤不符合 allowed_url_prefixes 的 URL（统一调用 is_allowed_doc_url）
    filtered_urls: list[str] = []
    for url in all_urls:
        if is_allowed_doc_url(url, filter_config, base_url):
            filtered_urls.append(url)
        else:
            logger.debug(f"URL 不在允许的前缀范围内，过滤: {url}")

    # 稳定去重（保持首次出现的顺序）
    result = deduplicate_urls(filtered_urls, normalize_before_dedup=False)

    logger.info(f"从 {files_read} 个文件加载了 {len(result)} 个核心文档 URL")

    # 如果结果为空且提供了 fallback，使用 fallback
    if not result and fallback_urls:
        logger.warning("核心文档 URL 列表为空，使用 fallback URL")
        result = deduplicate_urls(
            [
                normalize_url(u, base_url)
                for u in fallback_urls
                if is_allowed_doc_url(normalize_url(u, base_url), filter_config)
            ],
            normalize_before_dedup=False,
        )

    return result


def load_core_docs_with_fallback(
    source_files: Optional[list[str]] = None,
    legacy_urls: Optional[list[str]] = None,
    project_root: Optional[Path] = None,
    allowed_url_prefixes: Optional[list[str]] = None,
    base_url: str = "https://cursor.com",
    *,
    allowed_domains: Optional[list[str]] = None,  # [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes
) -> list[str]:
    """加载核心文档 URL，支持 legacy fallback

    优先从配置文件加载，如果失败则使用 legacy_urls 作为回退。
    用于逐步废弃 CURSOR_DOC_URLS 的过渡期。

    Args:
        source_files: URL 列表文件路径（相对于项目根目录）
        legacy_urls: 旧版硬编码 URL 列表（用于回退）
        project_root: 项目根目录
        allowed_url_prefixes: 允许的 URL 前缀列表
        base_url: 基础 URL（用于规范化相对路径）
        allowed_domains: [DEPRECATED] 旧参数名，请使用 allowed_url_prefixes（将在未来版本移除）

    Returns:
        核心文档 URL 列表
    """
    # [DEPRECATED] 如果使用了旧参数名 allowed_domains，发出警告
    if allowed_domains is not None and allowed_url_prefixes is None:
        _warn_deprecated_alias("allowed_domains (load_core_docs_with_fallback 参数)", "allowed_url_prefixes")

    result = load_core_docs(
        source_files=source_files,
        project_root=project_root,
        allowed_url_prefixes=allowed_url_prefixes or allowed_domains,
        base_url=base_url,
        fallback_urls=legacy_urls,
    )

    return result
