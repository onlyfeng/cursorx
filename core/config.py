"""统一配置加载模块

本模块提供从 config.yaml 加载配置的统一接口，作为所有配置的权威来源。
其他模块应通过此模块获取配置，而非硬编码默认值。

================================================================================
配置优先级
================================================================================

1. 命令行参数（最高）
2. 环境变量
3. config.yaml
4. 代码默认值（最低）

================================================================================
副作用控制策略 (Side Effect Control)
================================================================================

本模块定义的副作用控制策略由入口脚本解析并传递给下游模块。
详细策略矩阵参见: core/execution_policy.py

**策略摘要**:

| 策略        | CLI 参数              | 网络请求 | 文件写入 | Git 操作 |
|-------------|-----------------------|----------|----------|----------|
| normal      | (默认)                | ✓        | ✓        | ✓        |
| skip-online | --skip-online         | ✗        | ✓        | ✓        |
| dry-run     | --dry-run             | ✓        | ✗        | ✗        |
| minimal     | --skip-online --dry-run| ✗       | ✗        | ✗        |

**使用方式**:

    from core.config import get_config

    config = get_config()

    # 获取模型配置
    planner_model = config.models.planner
    worker_model = config.models.worker
    reviewer_model = config.models.reviewer

    # 获取超时配置
    planning_timeout = config.planner.timeout
    worker_timeout = config.worker.task_timeout
    review_timeout = config.reviewer.timeout

**副作用控制参数映射**:

| CLI 参数      | config.yaml 路径            | 默认值 | 说明                       |
|---------------|----------------------------|--------|----------------------------|
| --skip-online | (无，仅 CLI)               | False  | 跳过在线文档检查           |
| --dry-run     | (无，仅 CLI)               | False  | 仅分析不执行               |
| --force-update| (无，仅 CLI)               | False  | 强制更新知识库(忽略缓存)   |

注意: skip-online 和 dry-run 是运行时参数，不在 config.yaml 中配置。

================================================================================
统一字段 Schema 引用
================================================================================

本模块中涉及执行模式决策的配置字段遵循 core/execution_policy.py 定义的
统一 Schema。关键字段映射如下：

| config.yaml 路径                      | Schema 字段              | 说明                     |
|---------------------------------------|--------------------------|--------------------------|
| cloud_agent.execution_mode            | requested_mode           | 请求的执行模式           |
| cloud_agent.enabled                   | cloud_enabled            | Cloud 功能总开关         |
| cloud_agent.auto_detect_cloud_prefix  | auto_detect_cloud_prefix | & 前缀自动检测开关       |

**别名兼容**:
- `auto_detect_prefix`: 旧字段名（向后兼容，已废弃）
- 优先级: CLI（如未来提供）> config.yaml `auto_detect_cloud_prefix` >
  config.yaml `auto_detect_prefix`（兼容别名）> 默认值 True

**重要契约**:
1. resolve_orchestrator_settings() 推荐使用 **prefix_routed** 参数
   表示策略决策层面（& 是否成功触发 Cloud）；
   triggered_by_prefix 作为兼容别名保留
2. 编排器选择基于 **requested_mode** 而非 effective_mode
3. 详细 Schema 定义参见: core/execution_policy.py "统一字段 Schema" 部分
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

# ============================================================
# 配置校验相关常量
# ============================================================
#
# 注意：VALID_EXTERNAL_LINK_MODES 和 VALID_EXECUTION_MODES 的规范定义
# 位于 knowledge/doc_url_strategy.py 中。此处保留向后兼容的别名，
# 通过延迟属性访问避免循环导入。
#
# 推荐导入方式（公共 API）：
#   from knowledge import VALID_EXTERNAL_LINK_MODES, VALID_EXECUTION_MODES
#
# ============================================================

# 有效的 external_link_mode 值（向后兼容别名）
# 规范定义见 knowledge.doc_url_strategy.VALID_EXTERNAL_LINK_MODES
VALID_EXTERNAL_LINK_MODES = frozenset({"record_only", "skip_all", "fetch_allowlist"})

# 有效的 execution_mode 值（向后兼容别名）
# 规范定义见 knowledge.doc_url_strategy.VALID_EXECUTION_MODES
VALID_EXECUTION_MODES = frozenset({"cli", "cloud", "auto"})


# ============================================================
# Deprecated 警告机制（统一管理，每类警告仅输出一次）
# ============================================================
#
# 统一的 deprecated 警告 key 命名规则：
# - config.fetch_policy.allowed_url_prefixes: config.yaml 中使用旧字段名
# - config.fetch_policy.both_fields: config.yaml 中同时使用新旧字段
# - config.url_strategy.path_prefix_format: url_strategy 使用旧版路径前缀格式
# - cli.allowed_url_prefixes: CLI 使用旧参数名
# - cli.allowed_url_prefixes.both: CLI 同时使用新旧参数
#
# 测试用关键文案片段（用于断言，防止未来修改破坏兼容提示）：
# - "allowed_url_prefixes 已废弃" (fetch_policy 旧字段)
# - "allowed_path_prefixes 优先级更高" (新旧同时存在)
# - "旧版.*格式（路径前缀）" (url_strategy 格式检测)
# ============================================================

# 记录已警告的 deprecated key（避免重复警告）
_deprecated_warned: set[str] = set()

# deprecated 警告的统一 key 常量（便于测试断言和维护）
DEPRECATED_KEY_CONFIG_FETCH_POLICY_OLD_FIELD = "config.fetch_policy.allowed_url_prefixes"
DEPRECATED_KEY_CONFIG_FETCH_POLICY_BOTH_FIELDS = "config.fetch_policy.both_fields"
DEPRECATED_KEY_CONFIG_URL_STRATEGY_PATH_FORMAT = "config.url_strategy.path_prefix_format"
DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES = "cli.allowed_url_prefixes"
DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES_BOTH = "cli.allowed_url_prefixes.both"
DEPRECATED_KEY_CLOUD_AGENT_AUTO_DETECT_PREFIX = "config.cloud_agent.auto_detect_prefix"

# deprecated 警告的关键文案片段（用于测试断言）
DEPRECATED_MSG_FETCH_POLICY_OLD_FIELD = "allowed_url_prefixes 已废弃"
DEPRECATED_MSG_FETCH_POLICY_BOTH_FIELDS = "allowed_path_prefixes 优先级更高"
DEPRECATED_MSG_URL_STRATEGY_PATH_FORMAT = "旧版.*格式（路径前缀）"
DEPRECATED_MSG_CLI_OLD_PARAM = "--allowed-url-prefixes 已废弃"
DEPRECATED_MSG_CLI_BOTH_PARAMS = "--allowed-path-prefixes 优先级更高"
DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX = "auto_detect_prefix 已废弃"


def _warn_deprecated_once(key: str, message: str) -> bool:
    """发出 deprecated 警告（每个 key 仅警告一次）

    Args:
        key: 警告的唯一标识符（如 "config.fetch_policy.allowed_url_prefixes"）
        message: 完整的警告信息

    Returns:
        True 如果发出了警告，False 如果该 key 已经警告过
    """
    if key in _deprecated_warned:
        return False
    _deprecated_warned.add(key)
    logger.warning(message)
    return True


def reset_deprecated_warnings() -> None:
    """重置 deprecated 警告状态（仅供测试使用）

    清空已警告的 key 集合，使警告可以再次触发。
    """
    _deprecated_warned.clear()


# ============================================================
# 默认值常量 - 与 config.yaml 保持同步
# ============================================================

# 模型默认值
DEFAULT_PLANNER_MODEL = "gpt-5.2-high"
DEFAULT_WORKER_MODEL = "gpt-5.2-codex-high"
DEFAULT_REVIEWER_MODEL = "gpt-5.2-codex-xhigh"

# 超时默认值（秒）
DEFAULT_PLANNING_TIMEOUT = 500.0
DEFAULT_WORKER_TIMEOUT = 600.0
DEFAULT_REVIEW_TIMEOUT = 300.0
DEFAULT_AGENT_CLI_TIMEOUT = 300
DEFAULT_CLOUD_TIMEOUT = 300
DEFAULT_CLOUD_AUTH_TIMEOUT = 30

# Cloud Cooldown 默认值（与 cursor/executor.py CooldownConfig 保持同步）
DEFAULT_COOLDOWN_RATE_LIMIT_MIN = 30
DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT = 60
DEFAULT_COOLDOWN_RATE_LIMIT_MAX = 300
DEFAULT_COOLDOWN_AUTH = 600  # 10 分钟
DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE = True
DEFAULT_COOLDOWN_NETWORK = 120  # 2 分钟
DEFAULT_COOLDOWN_TIMEOUT = 60  # 1 分钟
DEFAULT_COOLDOWN_UNKNOWN = 300  # 5 分钟

# 系统默认值
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_WORKER_POOL_SIZE = 3
DEFAULT_ENABLE_SUB_PLANNERS = True
DEFAULT_STRICT_REVIEW = False

# 执行模式默认值
DEFAULT_EXECUTION_MODE = "auto"  # cli/cloud/auto

# 流式日志默认值 - 与 config.yaml logging.stream_json 保持同步
# 唯一权威来源：这些常量与 config.yaml 同步，编排器/客户端应使用这些常量或 tri-state 设计
DEFAULT_STREAM_EVENTS_ENABLED = False  # 默认关闭流式日志
DEFAULT_STREAM_LOG_CONSOLE = True  # 默认输出到控制台
DEFAULT_STREAM_LOG_DETAIL_DIR = "logs/stream_json/detail/"
DEFAULT_STREAM_LOG_RAW_DIR = "logs/stream_json/raw/"

# ============================================================
# 控制台预览截断常量
# ============================================================
# 统一控制台输出的截断限制，确保各位置截断行为一致

# Cloud 输出、iteration_goal 等大段文本的控制台预览截断限制
MAX_CONSOLE_PREVIEW_CHARS = 2000

# 知识库文档内容在控制台显示时的截断限制
MAX_KNOWLEDGE_DOC_PREVIEW_CHARS = 1000

# 目标摘要在结果输出中的截断限制
MAX_GOAL_SUMMARY_CHARS = 100

# 截断提示文本
TRUNCATION_HINT = "... (已截断)"
TRUNCATION_HINT_OUTPUT = "\n... (输出已截断)"


# ============================================================
# 配置数据类
# ============================================================


@dataclass
class SystemConfig:
    """系统配置"""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    worker_pool_size: int = DEFAULT_WORKER_POOL_SIZE
    enable_sub_planners: bool = DEFAULT_ENABLE_SUB_PLANNERS
    strict_review: bool = DEFAULT_STRICT_REVIEW


@dataclass
class ModelsConfig:
    """模型配置"""

    planner: str = DEFAULT_PLANNER_MODEL
    worker: str = DEFAULT_WORKER_MODEL
    reviewer: str = DEFAULT_REVIEWER_MODEL


@dataclass
class AgentCliConfig:
    """Agent CLI 配置"""

    path: str = "agent"
    timeout: int = DEFAULT_AGENT_CLI_TIMEOUT
    max_retries: int = 3
    output_format: str = "text"
    api_key: Optional[str] = None


@dataclass
class CloudCooldownConfig:
    """Cloud Cooldown 配置

    控制 Cloud API 失败后的冷却策略，按错误类型自适应冷却时间。

    Attributes:
        rate_limit_min_seconds: RateLimitError 最小冷却时间（秒）
        rate_limit_default_seconds: RateLimitError 默认冷却时间（秒）
        rate_limit_max_seconds: RateLimitError 最大冷却时间（秒）
        auth_cooldown_seconds: AuthError/NO_KEY 冷却时间（秒）
        auth_require_config_change: 是否需要配置变化才能重试
        network_cooldown_seconds: 网络错误冷却时间（秒）
        timeout_cooldown_seconds: 超时错误冷却时间（秒）
        unknown_cooldown_seconds: 未知错误冷却时间（秒）
    """

    rate_limit_min_seconds: int = DEFAULT_COOLDOWN_RATE_LIMIT_MIN
    rate_limit_default_seconds: int = DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT
    rate_limit_max_seconds: int = DEFAULT_COOLDOWN_RATE_LIMIT_MAX
    auth_cooldown_seconds: int = DEFAULT_COOLDOWN_AUTH
    auth_require_config_change: bool = DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE
    network_cooldown_seconds: int = DEFAULT_COOLDOWN_NETWORK
    timeout_cooldown_seconds: int = DEFAULT_COOLDOWN_TIMEOUT
    unknown_cooldown_seconds: int = DEFAULT_COOLDOWN_UNKNOWN


@dataclass
class CloudAgentConfig:
    """Cloud Agent 配置"""

    enabled: bool = False
    execution_mode: str = "auto"
    api_base_url: str = "https://api.cursor.com"
    api_key: Optional[str] = None
    timeout: int = DEFAULT_CLOUD_TIMEOUT
    auth_timeout: int = DEFAULT_CLOUD_AUTH_TIMEOUT
    max_retries: int = 3
    egress_ip_cache_ttl: int = 3600
    egress_ip_api_url: str = ""
    verify_egress_ip: bool = False
    cooldown: CloudCooldownConfig = field(default_factory=CloudCooldownConfig)
    # 是否启用 & 前缀自动检测 Cloud 路由
    # 别名: auto_detect_prefix（兼容，优先级低于 auto_detect_cloud_prefix）
    auto_detect_cloud_prefix: bool = True


@dataclass
class NetworkConfig:
    """网络配置"""

    proxy_url: str = ""
    ssl_verify: bool = True


@dataclass
class PlannerConfig:
    """规划者配置"""

    max_sub_planners: int = 3
    max_tasks_per_plan: int = 10
    exploration_depth: int = 3
    timeout: float = DEFAULT_PLANNING_TIMEOUT


@dataclass
class KnowledgeIntegrationConfig:
    """Worker 知识库集成配置"""

    enabled: bool = False
    search_mode: str = "keyword"
    max_docs: int = 3
    min_score: float = 0.3
    storage_path: str = ".cursor/knowledge"
    enable_vector_index: bool = False
    log_stats: bool = True


@dataclass
class WorkerConfig:
    """执行者配置"""

    task_timeout: float = DEFAULT_WORKER_TIMEOUT
    max_concurrent_tasks: int = 1
    knowledge_integration: KnowledgeIntegrationConfig = field(default_factory=KnowledgeIntegrationConfig)


@dataclass
class ReviewerConfig:
    """评审者配置"""

    strict_mode: bool = False
    timeout: float = DEFAULT_REVIEW_TIMEOUT


# ============================================================
# 知识库文档源配置默认值
# ============================================================

DEFAULT_MAX_FETCH_URLS = 20
DEFAULT_FALLBACK_CORE_DOCS_COUNT = 5
DEFAULT_LLMS_TXT_URL = "https://cursor.com/llms.txt"
DEFAULT_LLMS_CACHE_PATH = ".cursor/cache/llms.txt"
DEFAULT_CHANGELOG_URL = "https://cursor.com/cn/changelog"

# ============================================================
# fetch_policy 默认值（在线抓取策略）
# 保持最小抓取面：仅抓取明确允许的路径，外链仅记录不抓取
# ============================================================
DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES: list[str] = ["docs", "cn/docs", "changelog", "cn/changelog"]
DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS: list[str] = []  # 空表示仅允许主域名
DEFAULT_FETCH_POLICY_EXTERNAL_LINK_MODE = "record_only"  # record_only/skip_all/fetch_allowlist
DEFAULT_FETCH_POLICY_EXTERNAL_LINK_ALLOWLIST: list[str] = []  # 外链白名单
DEFAULT_FETCH_POLICY_ENFORCE_PATH_PREFIXES = False  # 是否启用内链路径前缀检查（Phase B 启用）

# ============================================================
# [DEPRECATED] 向后兼容别名 - fetch_policy 相关
# ============================================================
#
# 以下常量是旧版 API 的兼容别名，已废弃。
# 新代码应使用 DEFAULT_FETCH_POLICY_* 前缀版本。
#
# 迁移指南:
# | 旧名 (DEPRECATED)           | 新名 (使用此版本)                          | 说明                     |
# |-----------------------------|-------------------------------------------|--------------------------|
# | DEFAULT_ALLOWED_PATH_PREFIXES    | DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES   | 路径前缀，如 "docs"       |
# | DEFAULT_ALLOWED_URL_PREFIXES     | DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES   | 旧名误导，实为路径前缀    |
# | DEFAULT_ALLOWED_DOMAINS          | DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS         | 域名白名单               |
# | DEFAULT_EXTERNAL_LINK_MODE       | DEFAULT_FETCH_POLICY_EXTERNAL_LINK_MODE      | 外链处理模式             |
# | DEFAULT_EXTERNAL_LINK_ALLOWLIST  | DEFAULT_FETCH_POLICY_EXTERNAL_LINK_ALLOWLIST | 外链白名单               |
#
# 移除计划: 将在 v2.0 版本中移除这些别名
# ============================================================
DEFAULT_ALLOWED_PATH_PREFIXES = DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES
DEFAULT_ALLOWED_URL_PREFIXES = DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES  # DEPRECATED: 名称误导，实为路径前缀
DEFAULT_ALLOWED_DOMAINS = DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS
DEFAULT_EXTERNAL_LINK_MODE = DEFAULT_FETCH_POLICY_EXTERNAL_LINK_MODE
DEFAULT_EXTERNAL_LINK_ALLOWLIST = DEFAULT_FETCH_POLICY_EXTERNAL_LINK_ALLOWLIST

# 允许的文档 URL 前缀默认值（完整 URL 前缀，用于 load_core_docs 过滤）
# 与 knowledge/doc_sources.py DEFAULT_ALLOWED_DOC_URL_PREFIXES 保持同步
DEFAULT_ALLOWED_DOC_URL_PREFIXES: list[str] = [
    "https://cursor.com/cn/docs",
    "https://cursor.com/docs",
    "https://cursor.com/cn/changelog",
    "https://cursor.com/changelog",
]

# ============================================================
# url_strategy 默认值（与 config.yaml 保持同步）
# 注意：url_strategy 与 fetch_policy 是独立的配置，有不同的默认值
# ============================================================
DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS: list[str] = ["cursor.com"]  # 与 config.yaml 一致
DEFAULT_URL_STRATEGY_MAX_URLS = 20
DEFAULT_URL_STRATEGY_PREFER_CHANGELOG = True
DEFAULT_URL_STRATEGY_DEDUPLICATE = True
DEFAULT_URL_STRATEGY_NORMALIZE = True
DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT = 1.2  # 与 config.yaml 一致（非 2.0）
DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS: list[str] = [
    r".*\.pdf$",
    r".*\.zip$",
    r".*\.(png|jpg|jpeg|gif|svg|ico)$",
    r".*\.(css|js|woff|woff2|ttf|eot)$",  # 静态资源文件
    r".*/api/.*",  # API 端点路径
]  # 与 config.yaml 一致
DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS: dict[str, float] = {
    "changelog": 3.0,  # changelog 链接最高优先级
    "llms_txt": 2.5,  # llms.txt 中的链接
    "related_doc": 2.0,  # 相关文档链接
    "core_doc": 1.5,  # 核心文档
    "keyword_match": 1.0,  # 关键词匹配加成（与 keyword_boost_weight 相乘）
}


@dataclass
class OnlineFetchPolicyConfig:
    """在线抓取策略配置（fetch_policy）

    控制知识库更新时的在线文档抓取策略，确保最小抓取面。

    ============================================================
    裁决规则表参考
    ============================================================

    URL 裁决的完整逻辑参见:
        knowledge/doc_url_strategy.py 模块的【URL 裁决规则表】

    本配置主要影响裁决规则表中的：
    - 阶段 2：外链分类（allowed_domains 参与内/外链判定）
    - 阶段 3：抓取策略（external_link_mode 控制外链处理）
    - 阶段 4：路径前缀检查（allowed_path_prefixes gate）

    ============================================================
    术语说明（参见 knowledge/doc_url_strategy.py 模块契约的【术语表】）
    ============================================================

    本配置属于 **fetch_policy**（在线抓取策略），控制哪些 URL **可以被抓取**。
    与 **url_strategy**（URL 选择策略）不同：
    - fetch_policy: 网络请求层面，决定是否发起抓取
    - url_strategy: 数据处理层面，决定如何过滤/排序/选择 URL

    关键术语：
    - **路径前缀 (path prefix)**: URL 路径部分的前缀，如 "docs", "cn/docs"
      本配置的 allowed_path_prefixes 使用此格式
    - **域名白名单 (allowed_domains)**: 允许的域名列表，支持子域名匹配
    - **外链处理模式 (external_link_mode)**: 跨域链接的处理策略

    ============================================================
    策略矩阵（来源 → 默认允许范围 → 外链处理 → 可配置项）
    ============================================================

    +------------------+------------------------+------------------+--------------------------------+
    | 来源             | 默认允许范围            | 外链处理         | 可配置项 (CLI)                 |
    +==================+========================+==================+================================+
    | llms.txt         | cursor.com/docs/*      | record_only      | --llms-txt-url                 |
    |                  | cursor.com/cn/docs/*   | (仅记录不抓取)   | --llms-cache-path              |
    +------------------+------------------------+------------------+--------------------------------+
    | changelog        | cursor.com/changelog/* | record_only      | --changelog-url                |
    |                  | cursor.com/cn/changelog| (仅记录不抓取)   | --external-link-mode           |
    +------------------+------------------------+------------------+--------------------------------+
    | core_docs        | 本地文件定义的 URL     | N/A              | --fallback-core-docs-count     |
    | (cursor_*_docs.txt)                       |                  |                                |
    +------------------+------------------------+------------------+--------------------------------+

    策略优先级:
    1. allowed_path_prefixes 优先于 allowed_domains（两者互斥生效）
    2. external_link_mode 控制跨域链接行为
    3. 所有配置可被 CLI 参数覆盖

    示例配置:
    ```yaml
    knowledge_docs_update:
      docs_source:
        fetch_policy:
          # 仅抓取官方文档路径（路径前缀格式）
          allowed_path_prefixes:
            - "docs"
            - "cn/docs"
          # 外链仅记录，不抓取（最小抓取面）
          external_link_mode: "record_only"
    ```

    CLI 使用示例:
    ```bash
    # 使用默认策略
    python scripts/run_iterate.py "任务描述"

    # 自定义允许路径前缀（仅抓取 cn/docs 路径）
    python scripts/run_iterate.py --allowed-path-prefixes "cn/docs" "任务描述"

    # 完全跳过外链（不记录）
    python scripts/run_iterate.py --external-link-mode skip_all "任务描述"

    # 允许特定外链域名
    python scripts/run_iterate.py \
        --external-link-mode fetch_allowlist \
        --external-link-allowlist "github.com/cursor" \
        "任务描述"
    ```

    Attributes:
        allowed_path_prefixes: 允许抓取的 URL **路径前缀**列表
            格式：路径前缀（不含 scheme/host），如 "docs", "cn/docs"
            示例: ["docs", "cn/docs", "changelog"] 表示只抓取这些路径开头的页面
            注意：与 url_strategy.allowed_url_prefixes（完整 URL 前缀）格式不同
        allowed_domains: 允许抓取的**域名白名单**（可选）
            空列表表示仅允许主域名（如 cursor.com）
            示例: ["cursor.com", "docs.cursor.com"]
        external_link_mode: **外链处理模式**
            - record_only: 仅记录外链，不抓取（默认，最小抓取面）
            - skip_all: 完全跳过外链，不记录
            - fetch_allowlist: 仅抓取白名单内的外链
        external_link_allowlist: 外链允许白名单（可选）
            仅在 external_link_mode="fetch_allowlist" 时生效
            示例: ["github.com/cursor", "docs.example.com"]
        enforce_path_prefixes: 是否启用内链路径前缀检查（阶段 4 gate）
            默认 False（Phase A 行为不变），设为 True 启用内链路径 gate
            启用后，内链必须匹配 allowed_path_prefixes 中的任一前缀才允许抓取
    """

    allowed_path_prefixes: list[str] = field(default_factory=lambda: DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES.copy())
    allowed_domains: list[str] = field(default_factory=lambda: DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS.copy())
    external_link_mode: str = DEFAULT_FETCH_POLICY_EXTERNAL_LINK_MODE
    external_link_allowlist: list[str] = field(
        default_factory=lambda: DEFAULT_FETCH_POLICY_EXTERNAL_LINK_ALLOWLIST.copy()
    )
    enforce_path_prefixes: bool = DEFAULT_FETCH_POLICY_ENFORCE_PATH_PREFIXES


@dataclass
class UrlStrategyConfig:
    """URL 选择策略配置（url_strategy）

    与 knowledge/doc_url_strategy.py 中的 DocURLStrategyConfig 保持同步。
    控制 URL 选择、过滤、去重和优先级排序的策略参数。

    ============================================================
    裁决规则表参考
    ============================================================

    URL 裁决的完整逻辑参见:
        knowledge/doc_url_strategy.py 模块的【URL 裁决规则表】

    本配置主要影响裁决规则表中的：
    - 阶段 1：URL 选择（exclude_patterns、allowed_url_prefixes、allowed_domains）

    ============================================================
    术语说明（参见 knowledge/doc_url_strategy.py 模块契约的【术语表】）
    ============================================================

    本配置属于 **url_strategy**（URL 选择策略），控制如何**过滤/排序/选择** URL。
    与 **fetch_policy**（在线抓取策略）不同：
    - url_strategy: 数据处理层面，决定如何过滤/排序/选择 URL
    - fetch_policy: 网络请求层面，决定是否发起抓取

    关键术语：
    - **URL 规范化 (normalize)**: 将 URL 转换为标准格式（移除锚点、统一斜杠等）
    - **URL 过滤 (filter)**: 根据白名单/黑名单规则筛选 URL
    - **URL 去重 (deduplicate)**: 移除重复的 URL（基于规范化后的结果）
    - **URL 选择 (select)**: 按优先级和来源权重对 URL 排序并截取
    - **完整 URL 前缀 (full URL prefix)**: 包含 scheme/host 的完整前缀
      本配置的 allowed_url_prefixes 使用此格式

    ============================================================
    URL 处理策略矩阵（处理流程）
    ============================================================

    处理流程: 发现 URL → 过滤 → 规范化 → 去重 → 优先级排序 → 截取

    +------------------+---------------------------+--------------------------------+
    | 处理阶段         | 策略行为                   | 可配置项 (CLI)                 |
    +==================+===========================+================================+
    | 过滤             | 仅保留 allowed_domains    | --url-allowed-domains          |
    |                  | 内的 URL                  | --url-exclude-patterns         |
    +------------------+---------------------------+--------------------------------+
    | 规范化           | 移除锚点、统一斜杠         | --url-normalize/--no-url-normalize |
    +------------------+---------------------------+--------------------------------+
    | 去重             | 相同路径 URL 合并          | --url-deduplicate/--no-url-deduplicate |
    +------------------+---------------------------+--------------------------------+
    | 优先级排序       | 按 priority_weights       | --prefer-changelog             |
    |                  | 和 keyword_boost 评分     | --keyword-boost-weight         |
    +------------------+---------------------------+--------------------------------+
    | 截取             | 取前 max_urls 个          | --max-fetch-urls               |
    +------------------+---------------------------+--------------------------------+

    优先级权重默认值:
    - changelog: 3.0（最高优先级）
    - llms_txt: 2.5
    - related_doc: 2.0
    - core_doc: 1.5
    - keyword_match: 1.0（与 keyword_boost_weight 相乘）

    示例配置:
    ```yaml
    knowledge_docs_update:
      url_strategy:
        allowed_domains:
          - "cursor.com"
        # allowed_url_prefixes 为完整 URL 前缀格式（如需限制）
        allowed_url_prefixes:
          - "https://cursor.com/docs"
          - "https://cursor.com/cn/docs"
        exclude_patterns:
          - ".*\\.pdf$"
          - ".*\\.zip$"
        prefer_changelog: true
        deduplicate: true
        normalize: true
        keyword_boost_weight: 1.2
    ```

    CLI 使用示例:
    ```bash
    # 禁用 URL 去重（保留所有变体）
    python scripts/run_iterate.py --no-url-deduplicate "任务描述"

    # 提高关键词匹配权重
    python scripts/run_iterate.py --keyword-boost-weight 2.0 "任务描述"

    # 限制最大返回 URL 数量
    python scripts/run_iterate.py --max-fetch-urls 10 "任务描述"
    ```

    Attributes:
        allowed_domains: **域名白名单**（仅在 allowed_url_prefixes 为空时生效）
            支持子域名匹配，如 ["cursor.com"] 可匹配 api.cursor.com
        allowed_url_prefixes: 允许的**完整 URL 前缀**列表（优先级高于 allowed_domains）
            格式：完整 URL 前缀（含 scheme/host），如 "https://cursor.com/docs"
            空列表表示不限制（使用 allowed_domains 进行过滤）
            注意：与 fetch_policy.allowed_url_prefixes（路径前缀）格式不同
            注意：旧版本使用路径前缀格式（如 "docs"），已废弃
        max_urls: 最大返回 URL 数量
        fallback_core_docs_count: 当其他来源不足时，从 core_docs 补充的数量
        prefer_changelog: 是否优先处理 changelog 链接
        deduplicate: 是否启用**URL 去重**
        normalize: 是否启用**URL 规范化**
        keyword_boost_weight: 关键词匹配时的权重提升倍数
        exclude_patterns: URL 排除模式（正则表达式列表）
        priority_weights: 各来源的优先级权重
    """

    allowed_domains: list[str] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS.copy())
    # 完整 URL 前缀列表，默认为空（不限制，使用 allowed_domains 过滤）
    # 旧版本使用路径前缀（如 "docs"），已废弃，解析时会给出 warning
    allowed_url_prefixes: list[str] = field(default_factory=list)
    max_urls: int = DEFAULT_URL_STRATEGY_MAX_URLS
    fallback_core_docs_count: int = DEFAULT_FALLBACK_CORE_DOCS_COUNT
    prefer_changelog: bool = DEFAULT_URL_STRATEGY_PREFER_CHANGELOG
    deduplicate: bool = DEFAULT_URL_STRATEGY_DEDUPLICATE
    normalize: bool = DEFAULT_URL_STRATEGY_NORMALIZE
    keyword_boost_weight: float = DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
    exclude_patterns: list[str] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS.copy())
    priority_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS.copy())


@dataclass
class DocsSourceConfig:
    """文档源配置

    控制知识库更新时的文档获取策略。

    Attributes:
        max_fetch_urls: 单次更新最多抓取的 URL 数量
        fallback_core_docs_count: 回退核心文档数量（当其他来源不足时）
        llms_txt_url: llms.txt 在线来源 URL
        llms_cache_path: llms.txt 本地缓存路径（相对于项目根目录）
        changelog_url: Changelog URL
        fetch_policy: 在线抓取策略配置
        allowed_doc_url_prefixes: 允许的文档 URL 前缀列表（完整 URL 前缀）
            用于 load_core_docs 过滤核心文档 URL
            示例: ["https://cursor.com/cn/docs", "https://cursor.com/docs"]
    """

    max_fetch_urls: int = DEFAULT_MAX_FETCH_URLS
    fallback_core_docs_count: int = DEFAULT_FALLBACK_CORE_DOCS_COUNT
    llms_txt_url: str = DEFAULT_LLMS_TXT_URL
    llms_cache_path: str = DEFAULT_LLMS_CACHE_PATH
    changelog_url: str = DEFAULT_CHANGELOG_URL
    fetch_policy: OnlineFetchPolicyConfig = field(default_factory=OnlineFetchPolicyConfig)
    allowed_doc_url_prefixes: list[str] = field(default_factory=lambda: DEFAULT_ALLOWED_DOC_URL_PREFIXES.copy())


@dataclass
class KnowledgeDocsUpdateConfig:
    """知识库文档更新配置

    统一封装知识库更新相关的配置项，用于 KnowledgeUpdater 初始化。

    ============================================================
    完整策略矩阵
    ============================================================

    文档来源策略:
    +------------------+------------------------+------------------+--------------------------------+
    | 来源             | 默认允许范围            | 外链处理         | 可配置项 (CLI)                 |
    +==================+========================+==================+================================+
    | llms.txt         | cursor.com/docs/*      | record_only      | --llms-txt-url                 |
    |                  | cursor.com/cn/docs/*   | (仅记录不抓取)   | --llms-cache-path              |
    +------------------+------------------------+------------------+--------------------------------+
    | changelog        | cursor.com/changelog/* | record_only      | --changelog-url                |
    |                  | cursor.com/cn/changelog| (仅记录不抓取)   | --external-link-mode           |
    +------------------+------------------------+------------------+--------------------------------+
    | core_docs        | 本地文件定义的 URL     | N/A              | --fallback-core-docs-count     |
    | (cursor_*_docs.txt)                       |                  |                                |
    +------------------+------------------------+------------------+--------------------------------+

    URL 处理策略:
    +------------------+---------------------------+--------------------------------+
    | 处理阶段         | 策略行为                   | 可配置项 (CLI)                 |
    +==================+===========================+================================+
    | 过滤             | 仅保留 allowed_domains    | --url-allowed-domains          |
    |                  | 内的 URL                  | --url-exclude-patterns         |
    +------------------+---------------------------+--------------------------------+
    | 规范化           | 移除锚点、统一斜杠         | --url-normalize/--no-url-normalize |
    +------------------+---------------------------+--------------------------------+
    | 去重             | 相同路径 URL 合并          | --url-deduplicate/--no-url-deduplicate |
    +------------------+---------------------------+--------------------------------+
    | 优先级排序       | changelog(3.0) > llms_txt(2.5) | --prefer-changelog         |
    |                  | > related_doc(2.0) > core_doc(1.5) | --keyword-boost-weight  |
    +------------------+---------------------------+--------------------------------+
    | 截取             | 取前 max_urls 个          | --max-fetch-urls               |
    +------------------+---------------------------+--------------------------------+

    外链处理模式:
    - record_only: 仅记录外链，不抓取（默认，最小抓取面）
    - skip_all: 完全跳过外链，不记录
    - fetch_allowlist: 仅抓取白名单内的外链

    示例配置:
    ```yaml
    knowledge_docs_update:
      docs_source:
        max_fetch_urls: 20
        llms_txt_url: "https://cursor.com/llms.txt"
        llms_cache_path: ".cursor/cache/llms.txt"
        changelog_url: "https://cursor.com/cn/changelog"
        # 核心文档 URL 前缀过滤（仅用于 load_core_docs）
        allowed_doc_url_prefixes:
          - "https://cursor.com/cn/docs"
          - "https://cursor.com/docs"
        fetch_policy:
          # 路径前缀格式（不含 scheme/host）
          allowed_path_prefixes: ["docs", "cn/docs", "changelog", "cn/changelog"]
          # [DEPRECATED] allowed_url_prefixes 为废弃别名，请使用上方字段
          allowed_domains: []  # 空表示仅主域名
          external_link_mode: "record_only"
          external_link_allowlist: []
          # 注：当前版本 fetch_policy 已在 update_from_analysis() 中实际调用
          # 外链过滤已生效；内链路径检查默认禁用，可通过 --enforce-path-prefixes 启用
      url_strategy:
        allowed_domains: ["cursor.com"]
        # 完整 URL 前缀格式（必须含 scheme/host）
        # allowed_url_prefixes:
        #   - "https://cursor.com/docs"
        #   - "https://cursor.com/cn/docs"
        max_urls: 20
        prefer_changelog: true
        deduplicate: true
        normalize: true
        keyword_boost_weight: 1.2
        exclude_patterns:
          - ".*\\.pdf$"
          - ".*\\.(png|jpg|jpeg|gif|svg|ico)$"
    ```

    Attributes:
        docs_source: 文档源配置（URL 来源、缓存路径等）
        url_strategy: URL 策略配置（选择、过滤、去重、优先级）
    """

    docs_source: DocsSourceConfig = field(default_factory=DocsSourceConfig)
    url_strategy: UrlStrategyConfig = field(default_factory=UrlStrategyConfig)


@dataclass
class IndexingSearchConfig:
    """索引搜索配置"""

    top_k: int = 10
    min_score: float = 0.3
    include_context: bool = True
    context_lines: int = 3


@dataclass
class IndexingConfig:
    """索引配置"""

    enabled: bool = True
    model: str = "all-MiniLM-L6-v2"
    persist_path: str = ".cursor/vector_index/"
    chunk_size: int = 500
    chunk_overlap: int = 50
    include_patterns: list[str] = field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
        ]
    )
    search: IndexingSearchConfig = field(default_factory=IndexingSearchConfig)


@dataclass
class StreamJsonLoggingConfig:
    """流式 JSON 日志配置

    默认值使用 DEFAULT_STREAM_* 常量，确保与 config.yaml 保持同步。
    这是 stream 配置的唯一权威默认值来源，避免多处硬编码导致的漂移问题。
    """

    enabled: bool = DEFAULT_STREAM_EVENTS_ENABLED
    console: bool = DEFAULT_STREAM_LOG_CONSOLE
    detail_dir: str = DEFAULT_STREAM_LOG_DETAIL_DIR
    raw_dir: str = DEFAULT_STREAM_LOG_RAW_DIR


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    file: str = "logs/agent.log"
    rotation: str = "10 MB"
    retention: str = "7 days"
    stream_json: StreamJsonLoggingConfig = field(default_factory=StreamJsonLoggingConfig)


@dataclass
class AppConfig:
    """应用总配置"""

    system: SystemConfig = field(default_factory=SystemConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    agent_cli: AgentCliConfig = field(default_factory=AgentCliConfig)
    cloud_agent: CloudAgentConfig = field(default_factory=CloudAgentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    knowledge_docs_update: KnowledgeDocsUpdateConfig = field(default_factory=KnowledgeDocsUpdateConfig)


# ============================================================
# 配置文件查找函数
# ============================================================


def find_config_file() -> Optional[Path]:
    """查找配置文件

    按以下顺序查找:
    1. 当前目录的 config.yaml
    2. 项目根目录的 config.yaml（通过查找 .git 目录识别）
    3. 模块所在目录的 config.yaml

    Returns:
        配置文件路径，未找到时返回 None
    """
    # 当前目录
    current = Path.cwd() / "config.yaml"
    if current.exists():
        return current

    # 项目根目录（通过查找 .git 或其他标记）
    for parent in Path.cwd().parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path
        # 遇到 .git 目录，说明到了项目根
        if (parent / ".git").exists():
            break

    # 模块所在目录
    module_dir = Path(__file__).parent.parent
    module_config = module_dir / "config.yaml"
    if module_config.exists():
        return module_config

    return None


# ============================================================
# 配置管理器
# ============================================================


class ConfigManager:
    """配置管理器 - 单例模式

    从 config.yaml 加载配置，提供统一的配置访问接口。

    使用方式:
        config = ConfigManager.get_instance()
        planner_model = config.models.planner
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[AppConfig] = None
    _config_path: Optional[Path] = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 只在第一次初始化时加载配置
        if self._config is None:
            self._load_config()

    def _find_config_file(self) -> Optional[Path]:
        """查找配置文件

        按以下顺序查找:
        1. 当前目录的 config.yaml
        2. 项目根目录的 config.yaml
        3. 模块所在目录的 config.yaml

        委托给公共函数 find_config_file() 实现。
        """
        return find_config_file()

    def _load_config(self) -> None:
        """加载配置文件"""
        config_path = self._find_config_file()

        if config_path and config_path.exists():
            self._config_path = config_path
            try:
                with open(config_path, encoding="utf-8") as f:
                    raw_config = yaml.safe_load(f) or {}
                self._config = self._parse_config(raw_config)
                logger.debug(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
                self._config = AppConfig()
        else:
            logger.debug("未找到 config.yaml，使用默认配置")
            self._config = AppConfig()

    def _validate_external_link_mode(self, value: str, context: str = "fetch_policy.external_link_mode") -> str:
        """校验 external_link_mode 值

        委托到 knowledge.doc_url_strategy.validate_external_link_mode() 实现。
        如果值无效，记录 warning 并返回默认值。

        Args:
            value: 待校验的值
            context: 上下文描述（用于日志）

        Returns:
            如果值有效返回原值，否则返回默认值 "record_only"
        """
        # 延迟导入避免循环依赖
        from knowledge.doc_url_strategy import validate_external_link_mode

        return validate_external_link_mode(value, context)

    def _validate_execution_mode(self, value: str, context: str = "cloud_agent.execution_mode") -> str:
        """校验 execution_mode 值

        委托到 knowledge.doc_url_strategy.validate_execution_mode() 实现。
        如果值无效，记录 warning 并返回默认值。

        Args:
            value: 待校验的值
            context: 上下文描述（用于日志）

        Returns:
            如果值有效返回原值，否则返回默认值 "auto"
        """
        # 延迟导入避免循环依赖
        from knowledge.doc_url_strategy import validate_execution_mode

        return validate_execution_mode(value, context)

    def _parse_fetch_policy(self, fp_raw: dict[str, Any]) -> OnlineFetchPolicyConfig:
        """解析 fetch_policy 配置，提供向后兼容

        同时接受 allowed_path_prefixes（新）和 allowed_url_prefixes（旧）字段：
        - 新字段 allowed_path_prefixes 优先
        - 如果两者同时存在，使用新字段并输出 warning
        - 仅使用旧字段时，输出 deprecation warning

        Args:
            fp_raw: fetch_policy 原始配置字典

        Returns:
            解析后的 OnlineFetchPolicyConfig
        """
        # 延迟导入校验函数，避免循环导入
        from knowledge.doc_url_strategy import validate_fetch_policy_path_prefixes

        has_new_field = "allowed_path_prefixes" in fp_raw
        has_old_field = "allowed_url_prefixes" in fp_raw

        # 确定 allowed_path_prefixes 的值
        if has_new_field and has_old_field:
            # 两者同时存在：新字段优先，给出 warning（每类警告仅一次）
            _warn_deprecated_once(
                DEPRECATED_KEY_CONFIG_FETCH_POLICY_BOTH_FIELDS,
                "fetch_policy 同时配置了 allowed_path_prefixes 和 allowed_url_prefixes。"
                "allowed_path_prefixes 优先级更高，已忽略 allowed_url_prefixes。"
                "请移除已废弃的 allowed_url_prefixes 配置。",
            )
            allowed_path_prefixes = fp_raw["allowed_path_prefixes"]
        elif has_new_field:
            # 仅使用新字段
            allowed_path_prefixes = fp_raw["allowed_path_prefixes"]
        elif has_old_field:
            # 仅使用旧字段：deprecation warning（每类警告仅一次）
            _warn_deprecated_once(
                DEPRECATED_KEY_CONFIG_FETCH_POLICY_OLD_FIELD,
                "fetch_policy.allowed_url_prefixes 已废弃，请改用 allowed_path_prefixes。旧字段名将在后续版本中移除。",
            )
            allowed_path_prefixes = fp_raw["allowed_url_prefixes"]
        else:
            # 两者都不存在：使用默认值
            allowed_path_prefixes = DEFAULT_ALLOWED_PATH_PREFIXES.copy()

        # 校验 allowed_path_prefixes 格式（路径前缀应不含 scheme）
        # 如果检测到完整 URL 格式，记录 warning 但保持原值不变
        allowed_path_prefixes = validate_fetch_policy_path_prefixes(
            allowed_path_prefixes, context="docs_source.fetch_policy.allowed_path_prefixes"
        )

        # 校验 external_link_mode 值
        raw_external_link_mode = fp_raw.get("external_link_mode", DEFAULT_EXTERNAL_LINK_MODE)
        validated_external_link_mode = self._validate_external_link_mode(
            raw_external_link_mode, "fetch_policy.external_link_mode"
        )

        # 校验 external_link_allowlist 格式
        # 导入校验函数（延迟导入避免循环依赖）
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        raw_external_link_allowlist = fp_raw.get("external_link_allowlist", DEFAULT_EXTERNAL_LINK_ALLOWLIST.copy())
        # 校验并记录无效项警告（返回结构化对象）
        validated_allowlist = validate_external_link_allowlist(
            raw_external_link_allowlist, context="docs_source.fetch_policy.external_link_allowlist (from config.yaml)"
        )
        # 重建有效项列表（domains + prefixes 展平，保持向后兼容）
        validated_external_link_allowlist = validated_allowlist.domains + validated_allowlist.prefixes

        return OnlineFetchPolicyConfig(
            allowed_path_prefixes=allowed_path_prefixes,
            allowed_domains=fp_raw.get("allowed_domains", DEFAULT_ALLOWED_DOMAINS.copy()),
            external_link_mode=validated_external_link_mode,
            external_link_allowlist=validated_external_link_allowlist,
            enforce_path_prefixes=fp_raw.get("enforce_path_prefixes", DEFAULT_FETCH_POLICY_ENFORCE_PATH_PREFIXES),
        )

    def _parse_url_strategy_allowed_prefixes(self, prefixes: list[str]) -> list[str]:
        """解析 url_strategy.allowed_url_prefixes，提供向后兼容

        新语义：完整 URL 前缀（如 "https://cursor.com/docs"）
        旧语义（已废弃）：路径前缀（如 "docs"）

        如果检测到旧格式，会输出 warning 并保持原值（不自动转换）。

        Args:
            prefixes: 配置中的 allowed_url_prefixes 列表

        Returns:
            解析后的 URL 前缀列表
        """
        if not prefixes:
            return []

        # 检测旧格式：不以 http:// 或 https:// 开头的值
        deprecated_prefixes = [p for p in prefixes if not p.startswith("http://") and not p.startswith("https://")]

        if deprecated_prefixes:
            # 使用统一警告机制，每类警告仅一次
            _warn_deprecated_once(
                DEPRECATED_KEY_CONFIG_URL_STRATEGY_PATH_FORMAT,
                f"检测到旧版 url_strategy.allowed_url_prefixes 格式（路径前缀）: "
                f"{deprecated_prefixes}。"
                f"新版本要求使用完整 URL 前缀（如 'https://cursor.com/docs'）。"
                f"请更新配置文件。当前值将保持不变，但可能无法正确过滤 URL。",
            )

        return prefixes

    def _parse_config(self, raw: dict[str, Any]) -> AppConfig:
        """解析原始配置字典为 AppConfig 对象"""
        config = AppConfig()

        # 解析 system
        if "system" in raw:
            sys_raw = raw["system"]
            config.system = SystemConfig(
                max_iterations=sys_raw.get("max_iterations", DEFAULT_MAX_ITERATIONS),
                worker_pool_size=sys_raw.get("worker_pool_size", DEFAULT_WORKER_POOL_SIZE),
                enable_sub_planners=sys_raw.get("enable_sub_planners", DEFAULT_ENABLE_SUB_PLANNERS),
                strict_review=sys_raw.get("strict_review", DEFAULT_STRICT_REVIEW),
            )

        # 解析 models
        if "models" in raw:
            models_raw = raw["models"]
            config.models = ModelsConfig(
                planner=models_raw.get("planner", DEFAULT_PLANNER_MODEL),
                worker=models_raw.get("worker", DEFAULT_WORKER_MODEL),
                reviewer=models_raw.get("reviewer", DEFAULT_REVIEWER_MODEL),
            )

        # 解析 agent_cli
        if "agent_cli" in raw:
            cli_raw = raw["agent_cli"]
            config.agent_cli = AgentCliConfig(
                path=cli_raw.get("path", "agent"),
                timeout=cli_raw.get("timeout", DEFAULT_AGENT_CLI_TIMEOUT),
                max_retries=cli_raw.get("max_retries", 3),
                output_format=cli_raw.get("output_format", "text"),
                api_key=cli_raw.get("api_key"),
            )

        # 解析 cloud_agent
        if "cloud_agent" in raw:
            cloud_raw = raw["cloud_agent"]
            cooldown_raw = cloud_raw.get("cooldown", {})
            # 校验 execution_mode 值
            raw_execution_mode = cloud_raw.get("execution_mode", "auto")
            validated_execution_mode = self._validate_execution_mode(raw_execution_mode, "cloud_agent.execution_mode")
            # 解析 auto_detect_cloud_prefix，支持别名 auto_detect_prefix
            # 优先级: auto_detect_cloud_prefix > auto_detect_prefix > 默认值 True
            if "auto_detect_cloud_prefix" in cloud_raw:
                auto_detect_cloud_prefix = cloud_raw.get("auto_detect_cloud_prefix", True)
            elif "auto_detect_prefix" in cloud_raw:
                # 旧字段 fallback，发出 deprecated 警告
                _warn_deprecated_once(
                    DEPRECATED_KEY_CLOUD_AGENT_AUTO_DETECT_PREFIX,
                    "cloud_agent.auto_detect_prefix 已废弃，请改用 auto_detect_cloud_prefix。"
                    "旧字段名将在后续版本中移除。",
                )
                auto_detect_cloud_prefix = cloud_raw.get("auto_detect_prefix", True)
            else:
                auto_detect_cloud_prefix = True
            config.cloud_agent = CloudAgentConfig(
                enabled=cloud_raw.get("enabled", False),
                execution_mode=validated_execution_mode,
                api_base_url=cloud_raw.get("api_base_url", "https://api.cursor.com"),
                api_key=cloud_raw.get("api_key"),
                timeout=cloud_raw.get("timeout", DEFAULT_CLOUD_TIMEOUT),
                auth_timeout=cloud_raw.get("auth_timeout", DEFAULT_CLOUD_AUTH_TIMEOUT),
                max_retries=cloud_raw.get("max_retries", 3),
                egress_ip_cache_ttl=cloud_raw.get("egress_ip_cache_ttl", 3600),
                egress_ip_api_url=cloud_raw.get("egress_ip_api_url", ""),
                verify_egress_ip=cloud_raw.get("verify_egress_ip", False),
                auto_detect_cloud_prefix=auto_detect_cloud_prefix,
                cooldown=CloudCooldownConfig(
                    rate_limit_min_seconds=cooldown_raw.get("rate_limit_min_seconds", DEFAULT_COOLDOWN_RATE_LIMIT_MIN),
                    rate_limit_default_seconds=cooldown_raw.get(
                        "rate_limit_default_seconds", DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT
                    ),
                    rate_limit_max_seconds=cooldown_raw.get("rate_limit_max_seconds", DEFAULT_COOLDOWN_RATE_LIMIT_MAX),
                    auth_cooldown_seconds=cooldown_raw.get("auth_cooldown_seconds", DEFAULT_COOLDOWN_AUTH),
                    auth_require_config_change=cooldown_raw.get(
                        "auth_require_config_change", DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE
                    ),
                    network_cooldown_seconds=cooldown_raw.get("network_cooldown_seconds", DEFAULT_COOLDOWN_NETWORK),
                    timeout_cooldown_seconds=cooldown_raw.get("timeout_cooldown_seconds", DEFAULT_COOLDOWN_TIMEOUT),
                    unknown_cooldown_seconds=cooldown_raw.get("unknown_cooldown_seconds", DEFAULT_COOLDOWN_UNKNOWN),
                ),
            )

        # 解析 network
        if "network" in raw:
            net_raw = raw["network"]
            config.network = NetworkConfig(
                proxy_url=net_raw.get("proxy_url", ""),
                ssl_verify=net_raw.get("ssl_verify", True),
            )

        # 解析 planner
        if "planner" in raw:
            planner_raw = raw["planner"]
            config.planner = PlannerConfig(
                max_sub_planners=planner_raw.get("max_sub_planners", 3),
                max_tasks_per_plan=planner_raw.get("max_tasks_per_plan", 10),
                exploration_depth=planner_raw.get("exploration_depth", 3),
                timeout=float(planner_raw.get("timeout", DEFAULT_PLANNING_TIMEOUT)),
            )

        # 解析 worker
        if "worker" in raw:
            worker_raw = raw["worker"]
            ki_raw = worker_raw.get("knowledge_integration", {})
            config.worker = WorkerConfig(
                task_timeout=float(worker_raw.get("task_timeout", DEFAULT_WORKER_TIMEOUT)),
                max_concurrent_tasks=worker_raw.get("max_concurrent_tasks", 1),
                knowledge_integration=KnowledgeIntegrationConfig(
                    enabled=ki_raw.get("enabled", False),
                    search_mode=ki_raw.get("search_mode", "keyword"),
                    max_docs=ki_raw.get("max_docs", 3),
                    min_score=ki_raw.get("min_score", 0.3),
                    storage_path=ki_raw.get("storage_path", ".cursor/knowledge"),
                    enable_vector_index=ki_raw.get("enable_vector_index", False),
                    log_stats=ki_raw.get("log_stats", True),
                ),
            )

        # 解析 reviewer
        if "reviewer" in raw:
            reviewer_raw = raw["reviewer"]
            config.reviewer = ReviewerConfig(
                strict_mode=reviewer_raw.get("strict_mode", False),
                timeout=float(reviewer_raw.get("timeout", DEFAULT_REVIEW_TIMEOUT)),
            )

        # 解析 indexing
        if "indexing" in raw:
            idx_raw = raw["indexing"]
            search_raw = idx_raw.get("search", {})
            config.indexing = IndexingConfig(
                enabled=idx_raw.get("enabled", True),
                model=idx_raw.get("model", "all-MiniLM-L6-v2"),
                persist_path=idx_raw.get("persist_path", ".cursor/vector_index/"),
                chunk_size=idx_raw.get("chunk_size", 500),
                chunk_overlap=idx_raw.get("chunk_overlap", 50),
                include_patterns=idx_raw.get(
                    "include_patterns", ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"]
                ),
                exclude_patterns=idx_raw.get(
                    "exclude_patterns",
                    [
                        "**/node_modules/**",
                        "**/.git/**",
                        "**/venv/**",
                        "**/__pycache__/**",
                        "**/dist/**",
                        "**/build/**",
                    ],
                ),
                search=IndexingSearchConfig(
                    top_k=search_raw.get("top_k", 10),
                    min_score=search_raw.get("min_score", 0.3),
                    include_context=search_raw.get("include_context", True),
                    context_lines=search_raw.get("context_lines", 3),
                ),
            )

        # 解析 logging
        if "logging" in raw:
            log_raw = raw["logging"]
            sj_raw = log_raw.get("stream_json", {})
            config.logging = LoggingConfig(
                level=log_raw.get("level", "INFO"),
                file=log_raw.get("file", "logs/agent.log"),
                rotation=log_raw.get("rotation", "10 MB"),
                retention=log_raw.get("retention", "7 days"),
                stream_json=StreamJsonLoggingConfig(
                    # 使用 DEFAULT_STREAM_* 常量作为回退值，确保唯一权威来源
                    enabled=sj_raw.get("enabled", DEFAULT_STREAM_EVENTS_ENABLED),
                    console=sj_raw.get("console", DEFAULT_STREAM_LOG_CONSOLE),
                    detail_dir=sj_raw.get("detail_dir", DEFAULT_STREAM_LOG_DETAIL_DIR),
                    raw_dir=sj_raw.get("raw_dir", DEFAULT_STREAM_LOG_RAW_DIR),
                ),
            )

        # 解析 knowledge_docs_update
        if "knowledge_docs_update" in raw:
            kdu_raw = raw["knowledge_docs_update"]
            ds_raw = kdu_raw.get("docs_source", {})
            fp_raw = ds_raw.get("fetch_policy", {})
            us_raw = kdu_raw.get("url_strategy", {})
            config.knowledge_docs_update = KnowledgeDocsUpdateConfig(
                docs_source=DocsSourceConfig(
                    max_fetch_urls=ds_raw.get("max_fetch_urls", DEFAULT_MAX_FETCH_URLS),
                    fallback_core_docs_count=ds_raw.get("fallback_core_docs_count", DEFAULT_FALLBACK_CORE_DOCS_COUNT),
                    llms_txt_url=ds_raw.get("llms_txt_url", DEFAULT_LLMS_TXT_URL),
                    llms_cache_path=ds_raw.get("llms_cache_path", DEFAULT_LLMS_CACHE_PATH),
                    changelog_url=ds_raw.get("changelog_url", DEFAULT_CHANGELOG_URL),
                    fetch_policy=self._parse_fetch_policy(fp_raw),
                    allowed_doc_url_prefixes=ds_raw.get(
                        "allowed_doc_url_prefixes", DEFAULT_ALLOWED_DOC_URL_PREFIXES.copy()
                    ),
                ),
                url_strategy=UrlStrategyConfig(
                    allowed_domains=us_raw.get("allowed_domains", DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS.copy()),
                    allowed_url_prefixes=self._parse_url_strategy_allowed_prefixes(
                        us_raw.get("allowed_url_prefixes", [])
                    ),
                    max_urls=us_raw.get("max_urls", DEFAULT_URL_STRATEGY_MAX_URLS),
                    fallback_core_docs_count=us_raw.get("fallback_core_docs_count", DEFAULT_FALLBACK_CORE_DOCS_COUNT),
                    prefer_changelog=us_raw.get("prefer_changelog", DEFAULT_URL_STRATEGY_PREFER_CHANGELOG),
                    deduplicate=us_raw.get("deduplicate", DEFAULT_URL_STRATEGY_DEDUPLICATE),
                    normalize=us_raw.get("normalize", DEFAULT_URL_STRATEGY_NORMALIZE),
                    keyword_boost_weight=us_raw.get("keyword_boost_weight", DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT),
                    exclude_patterns=us_raw.get("exclude_patterns", DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS.copy()),
                    priority_weights=us_raw.get("priority_weights", DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS.copy()),
                ),
            )

        return config

    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()

    @property
    def config(self) -> AppConfig:
        """获取配置对象"""
        if self._config is None:
            self._load_config()
        return self._config  # type: ignore

    @property
    def config_path(self) -> Optional[Path]:
        """获取配置文件路径"""
        return self._config_path

    # 便捷属性访问
    @property
    def system(self) -> SystemConfig:
        return self.config.system

    @property
    def models(self) -> ModelsConfig:
        return self.config.models

    @property
    def agent_cli(self) -> AgentCliConfig:
        return self.config.agent_cli

    @property
    def cloud_agent(self) -> CloudAgentConfig:
        return self.config.cloud_agent

    @property
    def network(self) -> NetworkConfig:
        return self.config.network

    @property
    def planner(self) -> PlannerConfig:
        return self.config.planner

    @property
    def worker(self) -> WorkerConfig:
        return self.config.worker

    @property
    def reviewer(self) -> ReviewerConfig:
        return self.config.reviewer

    @property
    def indexing(self) -> IndexingConfig:
        return self.config.indexing

    @property
    def logging(self) -> LoggingConfig:
        return self.config.logging

    @property
    def knowledge_docs_update(self) -> KnowledgeDocsUpdateConfig:
        return self.config.knowledge_docs_update

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（主要用于测试）"""
        cls._instance = None
        cls._config = None
        cls._config_path = None


# ============================================================
# 便捷函数
# ============================================================


def get_config() -> ConfigManager:
    """获取配置管理器单例

    Returns:
        ConfigManager 单例实例
    """
    return ConfigManager.get_instance()


# ============================================================
# max_iterations 统一解析工具
# ============================================================

# 无限迭代的规范值
UNLIMITED_ITERATIONS = -1

# 支持的无限迭代关键词（大写形式，比较时需先将输入转为大写）
UNLIMITED_KEYWORDS = frozenset({"MAX", "UNLIMITED", "INF", "INFINITE"})


class MaxIterationsParseError(ValueError):
    """max_iterations 解析错误"""

    pass


def normalize_max_iterations(value: int) -> int:
    """标准化 max_iterations 值

    将各种表示无限迭代的值（-1, 0, 负数）统一转换为规范值 UNLIMITED_ITERATIONS (-1)。

    Args:
        value: 整数值

    Returns:
        标准化后的迭代次数:
        - 正整数: 返回原值
        - 0 或负数: 返回 UNLIMITED_ITERATIONS (-1)
    """
    if value <= 0:
        return UNLIMITED_ITERATIONS
    return value


def parse_max_iterations(value: str, raise_on_error: bool = True) -> int:
    """解析 max_iterations 参数

    支持的格式：
    - 正整数字符串: "10", "5" → 原值
    - 无限迭代关键词: "MAX", "UNLIMITED", "INF", "INFINITE" → -1
    - 零或负数字符串: "0", "-1", "-5" → -1
    - 带空白的字符串: "  10  ", "  MAX  " → 正常解析

    Args:
        value: 字符串形式的参数值
        raise_on_error: 解析失败时是否抛出异常
            - True: 抛出 MaxIterationsParseError
            - False: 返回 DEFAULT_MAX_ITERATIONS

    Returns:
        解析后的迭代次数:
        - 正整数: 有限迭代
        - -1 (UNLIMITED_ITERATIONS): 无限迭代

    Raises:
        MaxIterationsParseError: 当 raise_on_error=True 且解析失败时

    Examples:
        >>> parse_max_iterations("10")
        10
        >>> parse_max_iterations("MAX")
        -1
        >>> parse_max_iterations("unlimited")
        -1
        >>> parse_max_iterations("-1")
        -1
        >>> parse_max_iterations("0")
        -1
        >>> parse_max_iterations("abc")  # raise_on_error=True
        MaxIterationsParseError: 无效的迭代次数: abc
        >>> parse_max_iterations("abc", raise_on_error=False)
        10  # DEFAULT_MAX_ITERATIONS
    """
    value_stripped = value.strip()
    value_upper = value_stripped.upper()

    # 检查是否为无限迭代关键词
    if value_upper in UNLIMITED_KEYWORDS:
        return UNLIMITED_ITERATIONS

    # 尝试解析为整数
    try:
        num = int(value_stripped)
        return normalize_max_iterations(num)
    except ValueError as exc:
        if raise_on_error:
            raise MaxIterationsParseError(f"无效的迭代次数: {value}。使用正整数或 MAX/-1 表示无限迭代") from exc
        return DEFAULT_MAX_ITERATIONS


def parse_max_iterations_for_argparse(value: str) -> int:
    """argparse 类型转换器专用的 max_iterations 解析函数

    与 parse_max_iterations 相同，但将 MaxIterationsParseError 转换为
    argparse.ArgumentTypeError，以便 argparse 能正确处理错误消息。

    Args:
        value: 字符串形式的参数值

    Returns:
        解析后的迭代次数

    Raises:
        argparse.ArgumentTypeError: 解析失败时
    """
    import argparse

    try:
        return parse_max_iterations(value, raise_on_error=True)
    except MaxIterationsParseError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def get_model_config() -> ModelsConfig:
    """获取模型配置"""
    return get_config().models


def get_timeout_config() -> dict[str, float]:
    """获取超时配置

    Returns:
        包含各阶段超时时间的字典
    """
    config = get_config()
    return {
        "planning_timeout": config.planner.timeout,
        "execution_timeout": config.worker.task_timeout,
        "review_timeout": config.reviewer.timeout,
        "agent_cli_timeout": float(config.agent_cli.timeout),
        "cloud_timeout": float(config.cloud_agent.timeout),
    }


def resolve_stream_log_config(
    cli_enabled: Optional[bool] = None,
    cli_console: Optional[bool] = None,
    cli_detail_dir: Optional[str] = None,
    cli_raw_dir: Optional[str] = None,
    config_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """解析流式日志配置（CLI 参数优先）

    优先级: CLI 显式指定 > config.yaml > 默认值

    Args:
        cli_enabled: CLI 指定的 enabled 值（None 表示未指定）
        cli_console: CLI 指定的 console 值（None 表示未指定）
        cli_detail_dir: CLI 指定的 detail_dir 值（None 或空字符串表示未指定）
        cli_raw_dir: CLI 指定的 raw_dir 值（None 或空字符串表示未指定）
        config_data: 显式传入的配置字典（用于测试或外部注入）

    Returns:
        包含 enabled, console, detail_dir, raw_dir 的配置字典
    """
    # 兼容旧签名：resolve_stream_log_config(args, config_data)
    if hasattr(cli_enabled, "stream_log_enabled") and isinstance(cli_console, dict):
        args = cli_enabled
        config_data = cli_console
        cli_enabled = getattr(args, "stream_log_enabled", None)
        cli_console = getattr(args, "stream_log_console", None)
        cli_detail_dir = getattr(args, "stream_log_detail_dir", None)
        cli_raw_dir = getattr(args, "stream_log_raw_dir", None)

    if config_data is not None:
        stream_config = config_data.get("logging", {}).get("stream_json", {})
        enabled = stream_config.get("enabled", False)
        console = stream_config.get("console", True)
        detail_dir = stream_config.get("detail_dir", "logs/stream_json/detail/")
        raw_dir = stream_config.get("raw_dir", "logs/stream_json/raw/")
    else:
        config = get_config()
        stream_config = config.logging.stream_json
        enabled = stream_config.enabled
        console = stream_config.console
        detail_dir = stream_config.detail_dir
        raw_dir = stream_config.raw_dir

    # CLI 参数覆盖（仅当显式指定时）
    if cli_enabled is not None:
        enabled = cli_enabled
    if cli_console is not None:
        console = cli_console
    if cli_detail_dir:
        detail_dir = cli_detail_dir
    if cli_raw_dir:
        raw_dir = cli_raw_dir

    return {
        "enabled": enabled,
        "console": console,
        "detail_dir": detail_dir,
        "raw_dir": raw_dir,
    }


# ============================================================
# 统一配置解析器 API
# ============================================================


def _resolve_with_priority(
    cli_value: Any,
    env_key: Optional[str],
    yaml_value: Any,
    default_value: Any,
    fallback_env_keys: Optional[list[str]] = None,
) -> Any:
    """按优先级解析配置值

    优先级: CLI 显式值 > 环境变量 > 备选环境变量 > config.yaml > 默认值

    Args:
        cli_value: CLI 命令行传入的值（None 表示未指定）
        env_key: 主环境变量名（None 表示不使用环境变量）
        yaml_value: 从 config.yaml 读取的值
        default_value: 代码默认值
        fallback_env_keys: 备选环境变量名列表（按顺序尝试）

    Returns:
        解析后的配置值
    """
    import os

    # 1. CLI 显式值（最高优先级）
    if cli_value is not None:
        return cli_value

    # 2. 主环境变量
    if env_key:
        env_value = os.environ.get(env_key)
        if env_value:
            return env_value

    # 3. 备选环境变量（按顺序尝试）
    if fallback_env_keys:
        for fallback_key in fallback_env_keys:
            env_value = os.environ.get(fallback_key)
            if env_value:
                return env_value

    # 4. config.yaml 值
    if yaml_value is not None:
        return yaml_value

    # 5. 代码默认值
    return default_value


def build_cursor_agent_config(
    working_directory: str = ".",
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建 CursorAgentConfig 所需的配置字典

    从 config.yaml 加载配置并应用覆盖项，生成可用于
    构建 CursorAgentConfig 的配置字典。

    优先级: overrides (CLI) > 环境变量 > config.yaml > DEFAULT_*

    配置映射:
    - agent_cli.path → agent_path
    - agent_cli.timeout → timeout
    - agent_cli.max_retries → max_retries
    - agent_cli.output_format → output_format
    - agent_cli.api_key → api_key (环境变量 CURSOR_API_KEY 优先)
    - cloud_agent.api_base_url → cloud_api_base
    - cloud_agent.timeout → cloud_timeout
    - cloud_agent.enabled → cloud_enabled
    - cloud_agent.execution_mode → execution_mode
    - cloud_agent.auth_timeout → cloud_auth_timeout
    - cloud_agent.max_retries → cloud_max_retries
    - logging.stream_json.* → stream_* 配置
    - models.* → model (根据角色)

    Args:
        working_directory: 工作目录
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - agent_path: agent CLI 路径
            - timeout: 超时时间（秒）
            - max_retries: 最大重试次数
            - output_format: 输出格式 (text/json/stream-json)
            - api_key: API 密钥
            - model: 模型名称
            - mode: CLI 工作模式 (plan/ask/agent)
            - force_write: 是否强制写入
            - execution_mode: 执行模式 (cli/cloud/auto)
            - cloud_enabled: 是否启用 Cloud
            - cloud_timeout: Cloud 超时时间
            - cloud_auth_timeout: Cloud 认证超时时间
            - cloud_max_retries: Cloud 最大重试次数
            - stream_events_enabled: 是否启用流式事件日志
            - stream_log_console: 是否输出到控制台
            - stream_log_detail_dir: 详细日志目录
            - stream_log_raw_dir: 原始日志目录

    Returns:
        可用于构建 CursorAgentConfig 的配置字典

    示例:
        >>> config_dict = build_cursor_agent_config(
        ...     working_directory="/path/to/project",
        ...     overrides={"timeout": 600, "model": "gpt-5.2-codex-xhigh"}
        ... )
        >>> from cursor.client import CursorAgentConfig
        >>> agent_config = CursorAgentConfig(**config_dict)
    """
    overrides = overrides or {}
    config = get_config()
    agent_cli = config.agent_cli
    cloud_agent = config.cloud_agent
    stream_json = config.logging.stream_json

    # 构建配置字典，遵循优先级
    result = {
        # 基础配置
        "working_directory": working_directory,
        # agent CLI 配置映射
        "agent_path": _resolve_with_priority(
            overrides.get("agent_path"),
            None,
            agent_cli.path,
            "agent",
        ),
        "timeout": _resolve_with_priority(
            overrides.get("timeout"),
            None,
            agent_cli.timeout,
            DEFAULT_AGENT_CLI_TIMEOUT,
        ),
        "max_retries": _resolve_with_priority(
            overrides.get("max_retries"),
            None,
            agent_cli.max_retries,
            3,
        ),
        "output_format": _resolve_with_priority(
            overrides.get("output_format"),
            None,
            agent_cli.output_format,
            "text",
        ),
        # API 密钥 - 环境变量优先，支持 CURSOR_CLOUD_API_KEY 作为备选
        "api_key": _resolve_with_priority(
            overrides.get("api_key"),
            "CURSOR_API_KEY",
            agent_cli.api_key,
            None,
            fallback_env_keys=["CURSOR_CLOUD_API_KEY"],
        ),
        # 模型配置
        "model": _resolve_with_priority(
            overrides.get("model"),
            None,
            config.models.worker,  # 默认使用 worker 模型
            DEFAULT_WORKER_MODEL,
        ),
        # CLI 工作模式
        "mode": _resolve_with_priority(
            overrides.get("mode"),
            None,
            None,
            None,
        ),
        # 写入控制
        "force_write": _resolve_with_priority(
            overrides.get("force_write"),
            None,
            None,
            False,
        ),
        # 执行模式
        "execution_mode": _resolve_with_priority(
            overrides.get("execution_mode"),
            None,
            cloud_agent.execution_mode,
            DEFAULT_EXECUTION_MODE,
        ),
        # Cloud 配置
        "cloud_enabled": _resolve_with_priority(
            overrides.get("cloud_enabled"),
            None,
            cloud_agent.enabled,
            False,
        ),
        "cloud_api_base": _resolve_with_priority(
            overrides.get("cloud_api_base"),
            None,
            cloud_agent.api_base_url,
            "https://api.cursor.com",
        ),
        "cloud_timeout": _resolve_with_priority(
            overrides.get("cloud_timeout"),
            None,
            cloud_agent.timeout,
            DEFAULT_CLOUD_TIMEOUT,
        ),
        "cloud_auth_timeout": _resolve_with_priority(
            overrides.get("cloud_auth_timeout"),
            None,
            cloud_agent.auth_timeout,
            DEFAULT_CLOUD_AUTH_TIMEOUT,
        ),
        "cloud_max_retries": _resolve_with_priority(
            overrides.get("cloud_max_retries"),
            None,
            cloud_agent.max_retries,
            3,
        ),
        # 流式日志配置 (logging.stream_json.*)
        "stream_events_enabled": _resolve_with_priority(
            overrides.get("stream_events_enabled"),
            None,
            stream_json.enabled,
            False,
        ),
        "stream_log_console": _resolve_with_priority(
            overrides.get("stream_log_console"),
            None,
            stream_json.console,
            True,
        ),
        "stream_log_detail_dir": _resolve_with_priority(
            overrides.get("stream_log_detail_dir"),
            None,
            stream_json.detail_dir,
            "logs/stream_json/detail/",
        ),
        "stream_log_raw_dir": _resolve_with_priority(
            overrides.get("stream_log_raw_dir"),
            None,
            stream_json.raw_dir,
            "logs/stream_json/raw/",
        ),
    }

    return result


def build_cursor_agent_config_for_role(
    role: str,
    working_directory: str = ".",
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """为指定角色构建 CursorAgentConfig 配置字典

    根据角色自动选择模型和超时配置。

    Args:
        role: 角色名称 (planner/worker/reviewer)
        working_directory: 工作目录
        overrides: CLI 或调用方传入的覆盖配置

    Returns:
        可用于构建 CursorAgentConfig 的配置字典

    示例:
        >>> config_dict = build_cursor_agent_config_for_role(
        ...     role="planner",
        ...     working_directory="/path/to/project",
        ... )
        >>> # config_dict 包含 planner 专用的模型和超时设置
    """
    overrides = overrides or {}
    config = get_config()

    # 根据角色选择模型和超时
    role_lower = role.lower()
    if role_lower == "planner":
        default_model = config.models.planner
        default_timeout = config.planner.timeout
        default_mode = "plan"  # Planner 使用规划模式
    elif role_lower == "worker":
        default_model = config.models.worker
        default_timeout = config.worker.task_timeout
        default_mode = "agent"  # Worker 使用完整代理模式
    elif role_lower == "reviewer":
        default_model = config.models.reviewer
        default_timeout = config.reviewer.timeout
        default_mode = "plan"  # Reviewer 使用规划模式（只读）
    else:
        # 未知角色，使用默认配置
        default_model = config.models.worker
        default_timeout = DEFAULT_AGENT_CLI_TIMEOUT
        default_mode = None

    # 合并角色默认值到 overrides（overrides 优先级更高）
    role_defaults = {
        "model": default_model,
        "timeout": int(default_timeout),
        "mode": default_mode,
    }

    # 角色默认值只有在 overrides 中未指定时才生效
    merged_overrides = role_defaults.copy()
    for key, value in overrides.items():
        if value is not None:
            merged_overrides[key] = value

    return build_cursor_agent_config(
        working_directory=working_directory,
        overrides=merged_overrides,
    )


def build_cloud_client_config(
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建 Cloud Client 所需的配置字典

    从 config.yaml 加载 cloud_agent 配置并应用覆盖项。

    优先级: overrides (CLI) > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml > DEFAULT_*

    配置映射:
    - cloud_agent.api_base_url → base_url
    - cloud_agent.timeout → timeout
    - cloud_agent.auth_timeout → auth_timeout
    - cloud_agent.max_retries → max_retries
    - cloud_agent.api_key → api_key (环境变量优先)

    API Key 优先级（从高到低）:
    1. overrides["api_key"]（CLI 显式传入）
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml 中的 cloud_agent.api_key
    5. None

    Args:
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - base_url: API 基础 URL
            - timeout: 请求超时时间（秒）
            - auth_timeout: 认证超时时间（秒）
            - max_retries: 最大重试次数
            - api_key: API 密钥

    Returns:
        可用于构建 CloudAuthConfig/CursorCloudClient 的配置字典

    示例:
        >>> config_dict = build_cloud_client_config(
        ...     overrides={"timeout": 600}
        ... )
        >>> from cursor.cloud_client import CloudAuthConfig
        >>> auth_config = CloudAuthConfig(**config_dict)
    """
    overrides = overrides or {}
    config = get_config()
    cloud_agent = config.cloud_agent

    result = {
        "base_url": _resolve_with_priority(
            overrides.get("base_url"),
            None,
            cloud_agent.api_base_url,
            "https://api.cursor.com",
        ),
        "timeout": _resolve_with_priority(
            overrides.get("timeout"),
            None,
            cloud_agent.timeout,
            DEFAULT_CLOUD_TIMEOUT,
        ),
        "auth_timeout": _resolve_with_priority(
            overrides.get("auth_timeout"),
            None,
            cloud_agent.auth_timeout,
            DEFAULT_CLOUD_AUTH_TIMEOUT,
        ),
        "max_retries": _resolve_with_priority(
            overrides.get("max_retries"),
            None,
            cloud_agent.max_retries,
            3,
        ),
        # API 密钥 - 环境变量优先，支持 CURSOR_CLOUD_API_KEY 作为备选
        "api_key": _resolve_with_priority(
            overrides.get("api_key"),
            "CURSOR_API_KEY",
            cloud_agent.api_key,
            None,
            fallback_env_keys=["CURSOR_CLOUD_API_KEY"],
        ),
    }

    return result


def resolve_orchestrator_settings(
    overrides: Optional[dict[str, Any]] = None,
    triggered_by_prefix: bool = False,
    *,
    prefix_routed: Optional[bool] = None,
) -> dict[str, Any]:
    """解析编排器设置

    从 config.yaml 加载系统配置并应用覆盖项，生成编排器所需的配置。

    完整的参数优先级规则参见: core/execution_policy.py 模块文档中的
    "参数级别优先级表 (Parameter Priority Matrix)"

    优先级: overrides (CLI) > prefix_routed > config.yaml > DEFAULT_*

    **本函数职责边界**:
    此函数**仅**基于以下两个条件做编排器强制切换：
    1. `overrides["execution_mode"]` - 当值为 auto/cloud 时强制 basic
    2. `prefix_routed` 参数 - 当为 True 时强制 basic

    **重要：prefix_routed 参数的语义**:
    此函数的 `prefix_routed` 参数**仅**表示"& 前缀成功触发 Cloud"的情况。
    "& 前缀存在但未成功路由仍强制 basic"的规则（R-2）由 `build_execution_decision()`
    负责处理，不在此函数的职责范围内。

    **"& 意图导致 basic" 规则的正确处理方式**:
    正确的调用流程是：
    1. 调用 `build_execution_decision()` 进行策略决策（包括 & 前缀检测和 R-2/R-3 规则判断）
    2. 使用 `build_unified_overrides(..., execution_decision=...)` 构建 overrides
    3. 调用此函数时，& 前缀相关的编排器决策已通过 `overrides["orchestrator"]` 传入

    **推荐调用方式**:
    ```python
    # 推荐：使用 build_unified_overrides 统一构建
    decision = build_execution_decision(prompt, cli_execution_mode, ...)
    overrides = build_unified_overrides(cli_args, execution_decision=decision)
    settings = resolve_orchestrator_settings(overrides)
    ```

    **重要调用规范**:
    入口脚本（run.py、scripts/run_iterate.py）在调用此函数时：
    1. 将 CLI 显式参数放入 overrides 字典
    2. overrides["execution_mode"] 应为 requested_mode（用户请求的模式），
       而非 effective_mode（实际使用的模式）
    3. 若使用 `build_unified_overrides()`，& 前缀相关的编排器决策已自动处理

    配置映射:
    - system.max_iterations → max_iterations
    - system.worker_pool_size → worker_pool_size / workers
    - system.enable_sub_planners → enable_sub_planners
    - system.strict_review → strict_review
    - planner.timeout → planner_timeout
    - worker.task_timeout → worker_timeout
    - reviewer.timeout → reviewer_timeout
    - cloud_agent.timeout → cloud_timeout
    - cloud_agent.auth_timeout → cloud_auth_timeout
    - models.* → planner_model, worker_model, reviewer_model

    编排器强制切换规则（本函数直接处理）:
    - overrides["execution_mode"] = auto/cloud: 强制 basic
    - prefix_routed=True: 强制 basic
    - 其他情况: 默认使用 mp 编排器（或 overrides["orchestrator"] 指定的值）

    **注意**: config.yaml 中 cloud_agent.execution_mode=auto/cloud 时，
    会通过 resolved execution_mode 触发强制 basic。这与 AGENTS.md
    中的设计一致："请求 Cloud 即强制 basic"。

    Args:
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - max_iterations: 最大迭代次数
            - workers: Worker 池大小
            - worker_pool_size: Worker 池大小（别名）
            - enable_sub_planners: 是否启用子规划者
            - strict_review: 严格评审模式
            - planner_timeout: 规划超时
            - worker_timeout: Worker 任务超时
            - reviewer_timeout: 评审超时
            - cloud_timeout: Cloud 执行超时
            - cloud_auth_timeout: Cloud 认证超时
            - planner_model: Planner 模型
            - worker_model: Worker 模型
            - reviewer_model: Reviewer 模型
            - orchestrator: 编排器类型 (mp/basic)，可由 build_unified_overrides 自动设置
            - execution_mode: 执行模式 (cli/cloud/auto)
            - auto_commit: 是否自动提交
            - auto_push: 是否自动推送
            - commit_per_iteration: 每次迭代都提交
            - dry_run: 是否为干运行模式
        triggered_by_prefix: [DEPRECATED] 请使用 prefix_routed 参数。
            此参数保留以兼容旧代码，若 prefix_routed 已指定则忽略此参数。
        prefix_routed: & 前缀是否成功触发 Cloud 模式。
            当 True 时，强制使用 basic 编排器。

            **注意**: 此参数主要用于向后兼容。推荐的新代码应使用
            `build_unified_overrides(..., execution_decision=...)` 来构建 overrides，
            其中 & 前缀的决策结果会自动通过 `overrides["orchestrator"]` 传入。

    Returns:
        编排器配置字典，包含以下键:
        - max_iterations: 最大迭代次数
        - workers: Worker 池大小
        - enable_sub_planners: 是否启用子规划者
        - strict_review: 严格评审模式
        - planner_timeout: 规划超时（秒）
        - worker_timeout: Worker 任务超时（秒）
        - reviewer_timeout: 评审超时（秒）
        - cloud_timeout: Cloud 执行超时（秒）
        - cloud_auth_timeout: Cloud 认证超时（秒）
        - planner_model: Planner 模型
        - worker_model: Worker 模型
        - reviewer_model: Reviewer 模型
        - orchestrator: 编排器类型
        - orchestrator_forced_reason: 编排器强制切换原因（如果有）
        - execution_mode: 执行模式
        - prefix_routed: & 前缀是否成功触发（等同于 triggered_by_prefix 参数）
        - auto_commit: 自动提交
        - auto_push: 自动推送
        - commit_per_iteration: 每次迭代提交
        - dry_run: 干运行模式

    示例:
        >>> settings = resolve_orchestrator_settings(
        ...     overrides={"workers": 5, "max_iterations": 20}
        ... )
        >>> # settings 包含所有编排器需要的配置

        >>> # & 前缀成功触发时强制 basic（推荐写法）
        >>> settings = resolve_orchestrator_settings(
        ...     prefix_routed=True  # 推荐用法
        ... )
        >>> settings["orchestrator"]
        'basic'

        >>> # 兼容别名：triggered_by_prefix（已弃用，保留向后兼容）
        >>> settings = resolve_orchestrator_settings(
        ...     triggered_by_prefix=True  # 等同于 prefix_routed=True
        ... )
        >>> settings["orchestrator"]
        'basic'
    """
    # 参数别名处理：prefix_routed 优先于 triggered_by_prefix
    # 这确保新代码可以使用更明确的参数名，同时兼容旧代码
    effective_prefix_routed: bool
    if prefix_routed is not None:
        effective_prefix_routed = prefix_routed
    else:
        effective_prefix_routed = triggered_by_prefix
        # 当使用旧参数 triggered_by_prefix 且其值非默认时，输出 deprecation 提示
        if triggered_by_prefix:
            logger.debug(
                "[resolve_orchestrator_settings] 参数 'triggered_by_prefix' 已弃用 (deprecated)，"
                "请迁移使用 'prefix_routed' 关键字参数。"
                "示例: resolve_orchestrator_settings(overrides=..., prefix_routed=True)"
            )

    overrides = overrides or {}
    config = get_config()
    system = config.system
    cloud_agent = config.cloud_agent
    models = config.models

    # 处理 workers 的别名
    workers = overrides.get("workers") or overrides.get("worker_pool_size")

    result = {
        # 迭代控制
        "max_iterations": _resolve_with_priority(
            overrides.get("max_iterations"),
            None,
            system.max_iterations,
            DEFAULT_MAX_ITERATIONS,
        ),
        # Worker 池配置
        "workers": _resolve_with_priority(
            workers,
            None,
            system.worker_pool_size,
            DEFAULT_WORKER_POOL_SIZE,
        ),
        # 子规划者和评审模式
        "enable_sub_planners": _resolve_with_priority(
            overrides.get("enable_sub_planners"),
            None,
            system.enable_sub_planners,
            DEFAULT_ENABLE_SUB_PLANNERS,
        ),
        "strict_review": _resolve_with_priority(
            overrides.get("strict_review"),
            None,
            system.strict_review,
            DEFAULT_STRICT_REVIEW,
        ),
        # 角色超时配置 - 从各角色配置映射
        "planner_timeout": _resolve_with_priority(
            overrides.get("planner_timeout"),
            None,
            config.planner.timeout,
            DEFAULT_PLANNING_TIMEOUT,
        ),
        "worker_timeout": _resolve_with_priority(
            overrides.get("worker_timeout"),
            None,
            config.worker.task_timeout,
            DEFAULT_WORKER_TIMEOUT,
        ),
        "reviewer_timeout": _resolve_with_priority(
            overrides.get("reviewer_timeout"),
            None,
            config.reviewer.timeout,
            DEFAULT_REVIEW_TIMEOUT,
        ),
        # Cloud 超时配置
        "cloud_timeout": _resolve_with_priority(
            overrides.get("cloud_timeout"),
            None,
            cloud_agent.timeout,
            DEFAULT_CLOUD_TIMEOUT,
        ),
        "cloud_auth_timeout": _resolve_with_priority(
            overrides.get("cloud_auth_timeout"),
            None,
            cloud_agent.auth_timeout,
            DEFAULT_CLOUD_AUTH_TIMEOUT,
        ),
        # 模型配置
        "planner_model": _resolve_with_priority(
            overrides.get("planner_model"),
            None,
            models.planner,
            DEFAULT_PLANNER_MODEL,
        ),
        "worker_model": _resolve_with_priority(
            overrides.get("worker_model"),
            None,
            models.worker,
            DEFAULT_WORKER_MODEL,
        ),
        "reviewer_model": _resolve_with_priority(
            overrides.get("reviewer_model"),
            None,
            models.reviewer,
            DEFAULT_REVIEWER_MODEL,
        ),
        # 编排器类型
        "orchestrator": _resolve_with_priority(
            overrides.get("orchestrator"),
            None,
            None,
            "mp",  # 默认多进程编排器
        ),
        # 执行模式
        "execution_mode": _resolve_with_priority(
            overrides.get("execution_mode"),
            None,
            cloud_agent.execution_mode,
            DEFAULT_EXECUTION_MODE,
        ),
        # 提交控制
        "auto_commit": _resolve_with_priority(
            overrides.get("auto_commit"),
            None,
            None,
            False,
        ),
        "auto_push": _resolve_with_priority(
            overrides.get("auto_push"),
            None,
            None,
            False,
        ),
        "commit_per_iteration": _resolve_with_priority(
            overrides.get("commit_per_iteration"),
            None,
            None,
            False,
        ),
        # 干运行模式
        "dry_run": _resolve_with_priority(
            overrides.get("dry_run"),
            None,
            None,
            False,
        ),
    }

    # 添加元信息字段
    result["prefix_routed"] = effective_prefix_routed
    result["orchestrator_forced_reason"] = None

    # 处理 execution_mode 对 orchestrator 的影响
    # 与 CLI help 和 AGENTS.md 对齐：execution_mode=auto/cloud 即强制 basic
    #
    # 编排器强制 basic 的条件（任一满足即可）:
    # 1. effective_prefix_routed=True (即 prefix_routed): & 前缀成功触发 Cloud 模式
    # 2. 最终解析的 execution_mode=auto/cloud（无论来自 CLI 还是 config.yaml）
    #
    # 这确保了与 AGENTS.md 中的描述一致：
    # "当 `--execution-mode` 为 `cloud` 或 `auto` 时，系统会强制使用 basic 编排器"
    #
    # 注意：此处 resolved_execution_mode 是 requested_mode（用户请求的模式），
    # 而非 effective_mode（实际使用的模式）。这符合"请求 Cloud 即强制 basic"的设计，
    # 即使因缺少 API Key 导致 effective_mode 回退到 CLI，编排器仍应是 basic。
    # 参见: core/execution_policy.py 中的 "规则 1: requested_mode vs effective_mode"
    resolved_execution_mode = result["execution_mode"]  # 实际是 requested_mode
    mode_forces_basic = resolved_execution_mode in ("cloud", "auto")
    should_force_basic = effective_prefix_routed or mode_forces_basic

    if should_force_basic:
        original_orchestrator = result["orchestrator"]
        if original_orchestrator == "mp":
            # 确定强制切换的原因（优先级：& 前缀 > execution_mode）
            if effective_prefix_routed:
                reason = "& 前缀成功触发 Cloud 模式 (prefix_routed=True)"
            elif mode_forces_basic:
                # 区分 CLI 参数和 config.yaml 来源
                cli_mode = overrides.get("execution_mode")
                if cli_mode and cli_mode.lower() in ("cloud", "auto"):
                    reason = f"CLI 显式指定 execution_mode={cli_mode}"
                else:
                    reason = f"config.yaml 中 cloud_agent.execution_mode={resolved_execution_mode}"
            else:
                reason = "未知原因"  # 不应到达此分支

            logger.debug(f"[resolve_orchestrator_settings] {reason}，不兼容 MP 编排器，自动切换到 basic 编排器")
            result["orchestrator"] = "basic"
            result["orchestrator_forced_reason"] = reason
    else:
        logger.debug(
            f"[resolve_orchestrator_settings] execution_mode={resolved_execution_mode}, "
            f"prefix_routed={effective_prefix_routed} -> 允许使用 {result['orchestrator']} 编排器"
        )

    return result


def resolve_agent_timeouts() -> dict[str, float]:
    """解析各角色的超时配置

    从 config.yaml 读取 planner/worker/reviewer 的超时配置。

    Returns:
        包含各角色超时时间的字典:
        - planner_timeout: 规划超时（秒）
        - worker_timeout: Worker 任务超时（秒）
        - reviewer_timeout: 评审超时（秒）
        - agent_cli_timeout: Agent CLI 超时（秒）
        - cloud_timeout: Cloud API 超时（秒）
    """
    config = get_config()
    return {
        "planner_timeout": float(config.planner.timeout),
        "worker_timeout": float(config.worker.task_timeout),
        "reviewer_timeout": float(config.reviewer.timeout),
        "agent_cli_timeout": float(config.agent_cli.timeout),
        "cloud_timeout": float(config.cloud_agent.timeout),
    }


def build_orchestrator_config(
    working_directory: str = ".",
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建 OrchestratorConfig 所需的完整配置字典

    统一编排器配置的构建逻辑，入口脚本只需解析 CLI 参数生成 overrides，
    然后调用此函数获取最终配置。

    配置优先级: overrides (CLI) > 环境变量 > config.yaml > DEFAULT_*

    配置映射:
    - system.max_iterations → max_iterations
    - system.worker_pool_size → worker_pool_size
    - system.enable_sub_planners → enable_sub_planners
    - system.strict_review → strict_review
    - planner.timeout → planner_timeout
    - worker.task_timeout → worker_timeout
    - reviewer.timeout → reviewer_timeout
    - models.* → planner_model, worker_model, reviewer_model
    - cloud_agent.* → cloud_timeout, cloud_auth_timeout, execution_mode
    - agent_cli.* → agent_path, timeout, max_retries, api_key
    - logging.stream_json.* → stream_* 配置

    tri-state 参数说明:
    - overrides 中的 None 值表示"未显式指定"
    - 未显式指定的参数使用 config.yaml 的值
    - config.yaml 未配置的参数使用 DEFAULT_* 常量

    Args:
        working_directory: 工作目录
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - max_iterations: 最大迭代次数
            - workers / worker_pool_size: Worker 池大小
            - enable_sub_planners: 是否启用子规划者
            - strict_review: 严格评审模式
            - planner_timeout: 规划超时
            - worker_timeout: Worker 任务超时
            - reviewer_timeout: 评审超时
            - cloud_timeout: Cloud 执行超时
            - cloud_auth_timeout: Cloud 认证超时
            - planner_model: Planner 模型
            - worker_model: Worker 模型
            - reviewer_model: Reviewer 模型
            - orchestrator: 编排器类型 (mp/basic)
            - execution_mode: 执行模式 (cli/cloud/auto)
            - auto_commit: 是否自动提交
            - auto_push: 是否自动推送
            - commit_per_iteration: 每次迭代都提交
            - dry_run: 是否为干运行模式
            - stream_events_enabled: 流式事件日志开关
            - stream_log_console: 流式日志控制台输出
            - stream_log_detail_dir: 流式日志详细目录
            - stream_log_raw_dir: 流式日志原始目录
            - stream_console_renderer: 启用流式控制台渲染器
            - stream_advanced_renderer: 高级渲染器
            - stream_typing_effect: 打字机效果
            - stream_typing_delay: 打字延迟
            - stream_word_mode: 逐词输出模式
            - stream_color_enabled: 颜色输出
            - stream_show_word_diff: 显示逐词差异
            - api_key: API 密钥
            - agent_path: agent CLI 路径

    Returns:
        可用于构建 OrchestratorConfig 的完整配置字典

    示例:
        >>> # 入口脚本只需构建 overrides
        >>> overrides = {"workers": args.workers, "max_iterations": args.max_iterations}
        >>> config_dict = build_orchestrator_config(
        ...     working_directory="/path/to/project",
        ...     overrides=overrides,
        ... )
        >>> # 直接使用配置字典构建 OrchestratorConfig
        >>> from coordinator import OrchestratorConfig
        >>> orch_config = OrchestratorConfig(**config_dict)
    """
    overrides = overrides or {}
    config = get_config()
    system = config.system
    cloud_agent = config.cloud_agent
    agent_cli = config.agent_cli
    models = config.models
    stream_json = config.logging.stream_json

    # 处理 workers 的别名
    workers = overrides.get("workers") or overrides.get("worker_pool_size")

    # ========== 系统配置 ==========
    max_iterations = _resolve_with_priority(
        overrides.get("max_iterations"),
        None,
        system.max_iterations,
        DEFAULT_MAX_ITERATIONS,
    )

    worker_pool_size = _resolve_with_priority(
        workers,
        None,
        system.worker_pool_size,
        DEFAULT_WORKER_POOL_SIZE,
    )

    enable_sub_planners = _resolve_with_priority(
        overrides.get("enable_sub_planners"),
        None,
        system.enable_sub_planners,
        DEFAULT_ENABLE_SUB_PLANNERS,
    )

    strict_review = _resolve_with_priority(
        overrides.get("strict_review"),
        None,
        system.strict_review,
        DEFAULT_STRICT_REVIEW,
    )

    # ========== 超时配置 ==========
    planner_timeout = _resolve_with_priority(
        overrides.get("planner_timeout"),
        None,
        config.planner.timeout,
        DEFAULT_PLANNING_TIMEOUT,
    )

    worker_timeout = _resolve_with_priority(
        overrides.get("worker_timeout"),
        None,
        config.worker.task_timeout,
        DEFAULT_WORKER_TIMEOUT,
    )

    reviewer_timeout = _resolve_with_priority(
        overrides.get("reviewer_timeout"),
        None,
        config.reviewer.timeout,
        DEFAULT_REVIEW_TIMEOUT,
    )

    cloud_timeout = _resolve_with_priority(
        overrides.get("cloud_timeout"),
        None,
        cloud_agent.timeout,
        DEFAULT_CLOUD_TIMEOUT,
    )

    cloud_auth_timeout = _resolve_with_priority(
        overrides.get("cloud_auth_timeout"),
        None,
        cloud_agent.auth_timeout,
        DEFAULT_CLOUD_AUTH_TIMEOUT,
    )

    agent_cli_timeout = _resolve_with_priority(
        overrides.get("agent_cli_timeout"),
        None,
        agent_cli.timeout,
        DEFAULT_AGENT_CLI_TIMEOUT,
    )

    # ========== 模型配置 ==========
    planner_model = _resolve_with_priority(
        overrides.get("planner_model"),
        None,
        models.planner,
        DEFAULT_PLANNER_MODEL,
    )

    worker_model = _resolve_with_priority(
        overrides.get("worker_model"),
        None,
        models.worker,
        DEFAULT_WORKER_MODEL,
    )

    reviewer_model = _resolve_with_priority(
        overrides.get("reviewer_model"),
        None,
        models.reviewer,
        DEFAULT_REVIEWER_MODEL,
    )

    # ========== 编排器类型 ==========
    orchestrator = _resolve_with_priority(
        overrides.get("orchestrator"),
        None,
        None,
        "mp",  # 默认多进程编排器
    )

    # ========== 执行模式 ==========
    # 注意：这里解析的是 requested_execution_mode（来自 CLI 参数或 config.yaml）
    execution_mode = _resolve_with_priority(
        overrides.get("execution_mode"),
        None,
        cloud_agent.execution_mode,
        DEFAULT_EXECUTION_MODE,
    )

    # 处理 requested_execution_mode 对 orchestrator 的影响
    # 与 CLI help 对齐：requested=auto/cloud 即强制 basic，不受 key/enable 影响
    if execution_mode in ("cloud", "auto") and orchestrator == "mp":
        logger.debug(f"requested_execution_mode={execution_mode} 不兼容 MP 编排器，自动切换到 basic 编排器")
        orchestrator = "basic"

    # ========== 提交控制 ==========
    auto_commit = _resolve_with_priority(
        overrides.get("auto_commit"),
        None,
        None,
        False,
    )

    auto_push = _resolve_with_priority(
        overrides.get("auto_push"),
        None,
        None,
        False,
    )

    commit_per_iteration = _resolve_with_priority(
        overrides.get("commit_per_iteration"),
        None,
        None,
        False,
    )

    dry_run = _resolve_with_priority(
        overrides.get("dry_run"),
        None,
        None,
        False,
    )

    # ========== Agent CLI 配置 ==========
    agent_path = _resolve_with_priority(
        overrides.get("agent_path"),
        None,
        agent_cli.path,
        "agent",
    )

    max_retries = _resolve_with_priority(
        overrides.get("max_retries"),
        None,
        agent_cli.max_retries,
        3,
    )

    # API Key 优先级: CLI > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml
    api_key = _resolve_with_priority(
        overrides.get("api_key"),
        "CURSOR_API_KEY",
        agent_cli.api_key,
        None,
        fallback_env_keys=["CURSOR_CLOUD_API_KEY"],
    )

    # ========== 流式日志配置 ==========
    stream_events_enabled = _resolve_with_priority(
        overrides.get("stream_events_enabled"),
        None,
        stream_json.enabled,
        False,
    )

    stream_log_console = _resolve_with_priority(
        overrides.get("stream_log_console"),
        None,
        stream_json.console,
        True,
    )

    stream_log_detail_dir = _resolve_with_priority(
        overrides.get("stream_log_detail_dir"),
        None,
        stream_json.detail_dir,
        "logs/stream_json/detail/",
    )

    stream_log_raw_dir = _resolve_with_priority(
        overrides.get("stream_log_raw_dir"),
        None,
        stream_json.raw_dir,
        "logs/stream_json/raw/",
    )

    # ========== 流式控制台渲染配置 ==========
    stream_console_renderer = _resolve_with_priority(
        overrides.get("stream_console_renderer"),
        None,
        None,
        False,
    )

    stream_advanced_renderer = _resolve_with_priority(
        overrides.get("stream_advanced_renderer"),
        None,
        None,
        False,
    )

    stream_typing_effect = _resolve_with_priority(
        overrides.get("stream_typing_effect"),
        None,
        None,
        False,
    )

    stream_typing_delay = _resolve_with_priority(
        overrides.get("stream_typing_delay"),
        None,
        None,
        0.02,
    )

    stream_word_mode = _resolve_with_priority(
        overrides.get("stream_word_mode"),
        None,
        None,
        True,
    )

    stream_color_enabled = _resolve_with_priority(
        overrides.get("stream_color_enabled"),
        None,
        None,
        True,
    )

    stream_show_word_diff = _resolve_with_priority(
        overrides.get("stream_show_word_diff"),
        None,
        None,
        False,
    )

    # ========== Cloud 认证配置 ==========
    cloud_api_base = _resolve_with_priority(
        overrides.get("cloud_api_base"),
        None,
        cloud_agent.api_base_url,
        "https://api.cursor.com",
    )

    cloud_enabled = _resolve_with_priority(
        overrides.get("cloud_enabled"),
        None,
        cloud_agent.enabled,
        False,
    )

    # ========== 角色级执行模式 ==========
    planner_execution_mode = overrides.get("planner_execution_mode")
    worker_execution_mode = overrides.get("worker_execution_mode")
    reviewer_execution_mode = overrides.get("reviewer_execution_mode")

    # ========== 构建结果字典 ==========
    result = {
        # 基础配置
        "working_directory": working_directory,
        # 系统配置
        "max_iterations": max_iterations,
        "worker_pool_size": worker_pool_size,
        "enable_sub_planners": enable_sub_planners,
        "strict_review": strict_review,
        # 超时配置
        "planner_timeout": planner_timeout,
        "worker_timeout": worker_timeout,
        "reviewer_timeout": reviewer_timeout,
        "cloud_timeout": cloud_timeout,
        "cloud_auth_timeout": cloud_auth_timeout,
        "agent_cli_timeout": agent_cli_timeout,
        # 模型配置
        "planner_model": planner_model,
        "worker_model": worker_model,
        "reviewer_model": reviewer_model,
        # 编排器类型
        "orchestrator": orchestrator,
        # 执行模式
        "execution_mode": execution_mode,
        # 提交控制
        "auto_commit": auto_commit,
        "auto_push": auto_push,
        "commit_per_iteration": commit_per_iteration,
        "dry_run": dry_run,
        # Agent CLI 配置
        "agent_path": agent_path,
        "max_retries": max_retries,
        "api_key": api_key,
        # 流式日志配置
        "stream_events_enabled": stream_events_enabled,
        "stream_log_console": stream_log_console,
        "stream_log_detail_dir": stream_log_detail_dir,
        "stream_log_raw_dir": stream_log_raw_dir,
        # 流式控制台渲染配置
        "stream_console_renderer": stream_console_renderer,
        "stream_advanced_renderer": stream_advanced_renderer,
        "stream_typing_effect": stream_typing_effect,
        "stream_typing_delay": stream_typing_delay,
        "stream_word_mode": stream_word_mode,
        "stream_color_enabled": stream_color_enabled,
        "stream_show_word_diff": stream_show_word_diff,
        # Cloud 配置
        "cloud_api_base": cloud_api_base,
        "cloud_enabled": cloud_enabled,
        # 角色级执行模式
        "planner_execution_mode": planner_execution_mode,
        "worker_execution_mode": worker_execution_mode,
        "reviewer_execution_mode": reviewer_execution_mode,
    }

    return result


def build_cooldown_config(
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建 CooldownConfig 所需的配置字典

    从 config.yaml 加载 cloud_agent.cooldown 配置并应用覆盖项。

    优先级: overrides (CLI) > config.yaml > DEFAULT_COOLDOWN_* 常量

    配置映射:
    - cloud_agent.cooldown.rate_limit_min_seconds → rate_limit_min_seconds
    - cloud_agent.cooldown.rate_limit_default_seconds → rate_limit_default_seconds
    - cloud_agent.cooldown.rate_limit_max_seconds → rate_limit_max_seconds
    - cloud_agent.cooldown.auth_cooldown_seconds → auth_cooldown_seconds
    - cloud_agent.cooldown.auth_require_config_change → auth_require_config_change
    - cloud_agent.cooldown.network_cooldown_seconds → network_cooldown_seconds
    - cloud_agent.cooldown.timeout_cooldown_seconds → timeout_cooldown_seconds
    - cloud_agent.cooldown.unknown_cooldown_seconds → unknown_cooldown_seconds

    Args:
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - rate_limit_min_seconds: RateLimitError 最小冷却时间（秒）
            - rate_limit_default_seconds: RateLimitError 默认冷却时间（秒）
            - rate_limit_max_seconds: RateLimitError 最大冷却时间（秒）
            - auth_cooldown_seconds: AuthError/NO_KEY 冷却时间（秒）
            - auth_require_config_change: 是否需要配置变化才能重试
            - network_cooldown_seconds: 网络错误冷却时间（秒）
            - timeout_cooldown_seconds: 超时错误冷却时间（秒）
            - unknown_cooldown_seconds: 未知错误冷却时间（秒）

    Returns:
        可用于构建 cursor.executor.CooldownConfig 的配置字典

    示例:
        >>> config_dict = build_cooldown_config(
        ...     overrides={"auth_cooldown_seconds": 1200}
        ... )
        >>> from cursor.executor import CooldownConfig
        >>> cooldown_config = CooldownConfig(**config_dict)
    """
    overrides = overrides or {}
    config = get_config()
    cooldown = config.cloud_agent.cooldown

    result = {
        "rate_limit_min_seconds": _resolve_with_priority(
            overrides.get("rate_limit_min_seconds"),
            None,
            cooldown.rate_limit_min_seconds,
            DEFAULT_COOLDOWN_RATE_LIMIT_MIN,
        ),
        "rate_limit_default_seconds": _resolve_with_priority(
            overrides.get("rate_limit_default_seconds"),
            None,
            cooldown.rate_limit_default_seconds,
            DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT,
        ),
        "rate_limit_max_seconds": _resolve_with_priority(
            overrides.get("rate_limit_max_seconds"),
            None,
            cooldown.rate_limit_max_seconds,
            DEFAULT_COOLDOWN_RATE_LIMIT_MAX,
        ),
        "auth_cooldown_seconds": _resolve_with_priority(
            overrides.get("auth_cooldown_seconds"),
            None,
            cooldown.auth_cooldown_seconds,
            DEFAULT_COOLDOWN_AUTH,
        ),
        "auth_require_config_change": _resolve_with_priority(
            overrides.get("auth_require_config_change"),
            None,
            cooldown.auth_require_config_change,
            DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE,
        ),
        "network_cooldown_seconds": _resolve_with_priority(
            overrides.get("network_cooldown_seconds"),
            None,
            cooldown.network_cooldown_seconds,
            DEFAULT_COOLDOWN_NETWORK,
        ),
        "timeout_cooldown_seconds": _resolve_with_priority(
            overrides.get("timeout_cooldown_seconds"),
            None,
            cooldown.timeout_cooldown_seconds,
            DEFAULT_COOLDOWN_TIMEOUT,
        ),
        "unknown_cooldown_seconds": _resolve_with_priority(
            overrides.get("unknown_cooldown_seconds"),
            None,
            cooldown.unknown_cooldown_seconds,
            DEFAULT_COOLDOWN_UNKNOWN,
        ),
    }

    return result


def build_cloud_auth_config(
    overrides: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """构建 CloudAuthConfig 所需的配置字典

    统一 Cloud 认证配置的构建逻辑。

    API Key 优先级（从高到低）:
    1. overrides["api_key"]（CLI 显式传入）
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml 中的 cloud_agent.api_key

    Args:
        overrides: CLI 或调用方传入的覆盖配置，支持的键:
            - api_key: API 密钥
            - auth_timeout: 认证超时时间（秒）
            - base_url: API 基础 URL
            - max_retries: 最大重试次数

    Returns:
        可用于构建 CloudAuthConfig 的配置字典，如果未配置 api_key 则返回 None

    示例:
        >>> config_dict = build_cloud_auth_config(
        ...     overrides={"api_key": args.cloud_api_key}
        ... )
        >>> if config_dict:
        ...     from cursor.cloud_client import CloudAuthConfig
        ...     auth_config = CloudAuthConfig(**config_dict)
    """
    overrides = overrides or {}

    # 使用 build_cloud_client_config 获取基础配置
    cloud_config = build_cloud_client_config(overrides=overrides)

    # 如果没有 api_key，返回 None
    if not cloud_config.get("api_key"):
        return None

    return {
        "api_key": cloud_config["api_key"],
        "auth_timeout": cloud_config["auth_timeout"],
        "api_base_url": cloud_config["base_url"],
        "max_retries": cloud_config["max_retries"],
    }


def get_model_for_role(role: str) -> str:
    """获取指定角色的模型配置

    Args:
        role: 角色名称 (planner/worker/reviewer)

    Returns:
        模型名称
    """
    config = get_config()
    role_lower = role.lower()
    role_map = {
        "planner": config.models.planner,
        "worker": config.models.worker,
        "reviewer": config.models.reviewer,
    }
    return role_map.get(role_lower, config.models.worker)  # 默认返回 worker 模型


# ============================================================
# 配置调试输出工具
# ============================================================

# 配置调试输出的前缀，便于脚本化 grep/CI 断言
CONFIG_DEBUG_PREFIX = "[CONFIG]"


def format_debug_config(
    cli_overrides: Optional[dict[str, Any]] = None,
    source_label: str = "unknown",
    has_api_key: bool = False,
    cloud_enabled: Optional[bool] = None,
) -> str:
    """格式化配置调试输出

    生成稳定格式的配置输出，便于脚本化 grep/CI 断言。

    固定输出键集合（按顺序）:
    - config_path: 配置文件路径
    - source: 调用来源标识
    - max_iterations: 最大迭代次数
    - workers: Worker 池大小
    - requested_mode: 用户请求的执行模式（CLI 参数或 config.yaml，requested 语义）
    - effective_mode: 实际生效的执行模式（可能因缺少 API Key 而回退）
    - orchestrator: 编排器类型
    - orchestrator_fallback: 编排器回退信息
    - planner_model: Planner 模型
    - worker_model: Worker 模型
    - reviewer_model: Reviewer 模型
    - cloud_timeout: Cloud 执行超时（秒）
    - cloud_auth_timeout: Cloud 认证超时（秒）
    - auto_commit: 是否自动提交
    - auto_push: 是否自动推送
    - dry_run: 是否仅分析不执行
    - strict_review: 是否严格审查
    - enable_sub_planners: 是否启用子规划器

    输出格式示例:
    [CONFIG] config_path: /path/to/config.yaml
    [CONFIG] source: run.py
    [CONFIG] max_iterations: 10
    [CONFIG] workers: 3
    [CONFIG] requested_mode: auto
    [CONFIG] effective_mode: cli
    [CONFIG] orchestrator: basic
    [CONFIG] orchestrator_fallback: mp->basic (requested_mode=auto forces basic)
    [CONFIG] planner_model: gpt-5.2-high
    [CONFIG] worker_model: gpt-5.2-codex-xhigh
    [CONFIG] reviewer_model: gpt-5.2-codex
    [CONFIG] cloud_timeout: 300
    [CONFIG] cloud_auth_timeout: 30
    [CONFIG] auto_commit: false
    [CONFIG] auto_push: false
    [CONFIG] dry_run: false
    [CONFIG] strict_review: false
    [CONFIG] enable_sub_planners: true

    关键语义说明:
    - requested_mode: 用户通过 CLI 或 config.yaml 请求的执行模式（requested 语义）
      注意：字段名为 requested_mode 而非 execution_mode，明确表示这是用户请求的模式
    - effective_mode: 实际生效的执行模式（auto/cloud 无 API Key 时回退到 cli）
    - orchestrator 选择基于 requested_mode，而非 effective_mode
    - 即 requested_mode=auto ⇒ orchestrator=basic，即使 effective_mode=cli

    Args:
        cli_overrides: CLI 命令行传入的覆盖配置
        source_label: 调用来源标识（如 "run.py", "scripts/run_iterate.py"）
        has_api_key: 是否配置了有效的 API Key
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置（None 时从配置读取）

    Returns:
        格式化后的配置调试字符串（多行）
    """
    cli_overrides = cli_overrides or {}

    # 获取配置管理器
    config_manager = get_config()
    config_path = config_manager.config_path

    # 使用 resolve_orchestrator_settings 获取解析后的配置
    resolved = resolve_orchestrator_settings(overrides=cli_overrides)

    # 检测编排器回退情况
    # 注意：execution_mode 是 requested_mode（CLI 参数或 config.yaml）
    original_orchestrator = cli_overrides.get("orchestrator", "mp")
    final_orchestrator = resolved["orchestrator"]
    requested_mode = resolved["execution_mode"]

    # 解析 effective_mode（实际生效的执行模式）
    # 导入 execution_policy 模块的函数
    from core.execution_policy import resolve_effective_execution_mode

    # 获取 cloud_enabled 配置
    if cloud_enabled is None:
        cloud_enabled = config_manager.cloud_agent.enabled

    effective_mode, effective_reason = resolve_effective_execution_mode(
        requested_mode=requested_mode,
        has_ampersand_prefix=False,  # --print-config 不涉及 & 前缀
        cloud_enabled=cloud_enabled,
        has_api_key=has_api_key,
    )

    # 判断是否发生了回退（基于 requested_mode，与 CLI help 对齐）
    # 关键规则：编排器选择基于 requested_mode，而非 effective_mode
    orchestrator_fallback = "none"
    if original_orchestrator == "mp" and final_orchestrator == "basic" and requested_mode in ("cloud", "auto"):
        # 详细解释：为什么 requested_mode 决定编排器
        if effective_mode != requested_mode:
            # effective_mode 与 requested_mode 不同（发生了回退）
            orchestrator_fallback = (
                f"mp->basic (requested_mode={requested_mode} forces basic, even when effective_mode={effective_mode})"
            )
        else:
            # effective_mode 与 requested_mode 相同
            orchestrator_fallback = f"mp->basic (requested_mode={requested_mode})"

    # 构建输出行
    lines = [
        f"{CONFIG_DEBUG_PREFIX} config_path: {config_path or '(default)'}",
        f"{CONFIG_DEBUG_PREFIX} source: {source_label}",
        f"{CONFIG_DEBUG_PREFIX} max_iterations: {resolved['max_iterations']}",
        f"{CONFIG_DEBUG_PREFIX} workers: {resolved['workers']}",
        f"{CONFIG_DEBUG_PREFIX} requested_mode: {requested_mode}",
        f"{CONFIG_DEBUG_PREFIX} effective_mode: {effective_mode}",
        f"{CONFIG_DEBUG_PREFIX} orchestrator: {final_orchestrator}",
        f"{CONFIG_DEBUG_PREFIX} orchestrator_fallback: {orchestrator_fallback}",
        f"{CONFIG_DEBUG_PREFIX} planner_model: {resolved['planner_model']}",
        f"{CONFIG_DEBUG_PREFIX} worker_model: {resolved['worker_model']}",
        f"{CONFIG_DEBUG_PREFIX} reviewer_model: {resolved['reviewer_model']}",
        f"{CONFIG_DEBUG_PREFIX} cloud_timeout: {resolved['cloud_timeout']}",
        f"{CONFIG_DEBUG_PREFIX} cloud_auth_timeout: {resolved['cloud_auth_timeout']}",
        f"{CONFIG_DEBUG_PREFIX} auto_commit: {str(resolved['auto_commit']).lower()}",
        f"{CONFIG_DEBUG_PREFIX} auto_push: {str(resolved['auto_push']).lower()}",
        f"{CONFIG_DEBUG_PREFIX} dry_run: {str(resolved['dry_run']).lower()}",
        f"{CONFIG_DEBUG_PREFIX} strict_review: {str(resolved['strict_review']).lower()}",
        f"{CONFIG_DEBUG_PREFIX} enable_sub_planners: {str(resolved['enable_sub_planners']).lower()}",
    ]

    return "\n".join(lines)


def print_debug_config(
    cli_overrides: Optional[dict[str, Any]] = None,
    source_label: str = "unknown",
    has_api_key: bool = False,
    cloud_enabled: Optional[bool] = None,
) -> None:
    """打印配置调试信息到标准输出

    Args:
        cli_overrides: CLI 命令行传入的覆盖配置
        source_label: 调用来源标识
        has_api_key: 是否配置了有效的 API Key
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
    """
    output = format_debug_config(
        cli_overrides,
        source_label,
        has_api_key=has_api_key,
        cloud_enabled=cloud_enabled,
    )
    print(output)


@dataclass
class ResolvedSettings:
    """统一的配置解析结果

    包含从 config.yaml 和 CLI 参数解析后的完整配置。
    用于 SelfIterator 初始化和编排器构建时注入参数。

    优先级规则:
    1. CLI 显式参数（最高）
    2. config.yaml 中的配置
    3. 代码默认值（最低）
    """

    # 模型配置
    planner_model: str = DEFAULT_PLANNER_MODEL
    worker_model: str = DEFAULT_WORKER_MODEL
    reviewer_model: str = DEFAULT_REVIEWER_MODEL

    # 超时配置
    planning_timeout: float = DEFAULT_PLANNING_TIMEOUT
    execution_timeout: float = DEFAULT_WORKER_TIMEOUT
    review_timeout: float = DEFAULT_REVIEW_TIMEOUT
    agent_cli_timeout: int = DEFAULT_AGENT_CLI_TIMEOUT
    cloud_timeout: int = DEFAULT_CLOUD_TIMEOUT
    cloud_auth_timeout: int = DEFAULT_CLOUD_AUTH_TIMEOUT

    # 系统配置
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    worker_pool_size: int = DEFAULT_WORKER_POOL_SIZE
    enable_sub_planners: bool = DEFAULT_ENABLE_SUB_PLANNERS
    strict_review: bool = DEFAULT_STRICT_REVIEW

    # Cloud Agent 配置
    cloud_enabled: bool = False
    execution_mode: str = "auto"
    cloud_api_base_url: str = "https://api.cursor.com"

    # 流式日志配置
    stream_events_enabled: bool = False
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"


def resolve_settings(
    cli_workers: Optional[int] = None,
    cli_max_iterations: Optional[int] = None,
    cli_cloud_timeout: Optional[int] = None,
    cli_cloud_auth_timeout: Optional[int] = None,
    cli_execution_mode: Optional[str] = None,
    cli_planner_model: Optional[str] = None,
    cli_worker_model: Optional[str] = None,
    cli_reviewer_model: Optional[str] = None,
    cli_stream_events_enabled: Optional[bool] = None,
    cli_stream_log_console: Optional[bool] = None,
    cli_stream_log_detail_dir: Optional[str] = None,
    cli_stream_log_raw_dir: Optional[str] = None,
) -> ResolvedSettings:
    """统一的配置解析器

    从 config.yaml 加载配置，并应用 CLI 参数覆盖。
    确保配置来源优先级一致: CLI 显式参数 > config.yaml > 默认值

    Args:
        cli_workers: CLI 指定的 worker 池大小
        cli_max_iterations: CLI 指定的最大迭代次数
        cli_cloud_timeout: CLI 指定的 cloud 超时时间
        cli_cloud_auth_timeout: CLI 指定的 cloud 认证超时时间
        cli_execution_mode: CLI 指定的执行模式
        cli_planner_model: CLI 指定的 planner 模型
        cli_worker_model: CLI 指定的 worker 模型
        cli_reviewer_model: CLI 指定的 reviewer 模型
        cli_stream_events_enabled: CLI 指定的流式事件是否启用
        cli_stream_log_console: CLI 指定的流式日志控制台输出
        cli_stream_log_detail_dir: CLI 指定的流式日志详细目录
        cli_stream_log_raw_dir: CLI 指定的流式日志原始目录

    Returns:
        ResolvedSettings 包含完整的解析后配置
    """
    config = get_config()

    # 构建 resolved settings，从 config.yaml 读取基础值
    settings = ResolvedSettings(
        # 模型配置
        planner_model=config.models.planner,
        worker_model=config.models.worker,
        reviewer_model=config.models.reviewer,
        # 超时配置
        planning_timeout=config.planner.timeout,
        execution_timeout=config.worker.task_timeout,
        review_timeout=config.reviewer.timeout,
        agent_cli_timeout=config.agent_cli.timeout,
        cloud_timeout=config.cloud_agent.timeout,
        cloud_auth_timeout=config.cloud_agent.auth_timeout,
        # 系统配置
        max_iterations=config.system.max_iterations,
        worker_pool_size=config.system.worker_pool_size,
        enable_sub_planners=config.system.enable_sub_planners,
        strict_review=config.system.strict_review,
        # Cloud Agent 配置
        cloud_enabled=config.cloud_agent.enabled,
        execution_mode=config.cloud_agent.execution_mode,
        cloud_api_base_url=config.cloud_agent.api_base_url,
        # 流式日志配置
        stream_events_enabled=config.logging.stream_json.enabled,
        stream_log_console=config.logging.stream_json.console,
        stream_log_detail_dir=config.logging.stream_json.detail_dir,
        stream_log_raw_dir=config.logging.stream_json.raw_dir,
    )

    # 应用 CLI 参数覆盖（仅当显式指定时）
    if cli_workers is not None:
        settings.worker_pool_size = cli_workers
    if cli_max_iterations is not None:
        settings.max_iterations = cli_max_iterations
    if cli_cloud_timeout is not None:
        settings.cloud_timeout = cli_cloud_timeout
    if cli_cloud_auth_timeout is not None:
        settings.cloud_auth_timeout = cli_cloud_auth_timeout
    if cli_execution_mode is not None:
        settings.execution_mode = cli_execution_mode
    if cli_planner_model is not None:
        settings.planner_model = cli_planner_model
    if cli_worker_model is not None:
        settings.worker_model = cli_worker_model
    if cli_reviewer_model is not None:
        settings.reviewer_model = cli_reviewer_model
    if cli_stream_events_enabled is not None:
        settings.stream_events_enabled = cli_stream_events_enabled
    if cli_stream_log_console is not None:
        settings.stream_log_console = cli_stream_log_console
    if cli_stream_log_detail_dir:
        settings.stream_log_detail_dir = cli_stream_log_detail_dir
    if cli_stream_log_raw_dir:
        settings.stream_log_raw_dir = cli_stream_log_raw_dir

    return settings


# ============================================================
# 统一 Overrides 构建 API
# ============================================================


@dataclass
class UnifiedOptions:
    """统一的配置解析结果

    封装从 argparse args、nl_options 和 ExecutionDecision 解析后的完整配置。
    包含 requested/effective/prefix 状态信息，供入口脚本使用。

    这是 build_unified_overrides 的输出结构，用于：
    1. run.py 的 Runner._merge_options
    2. scripts/run_iterate.py 的 SelfIterator._resolve_settings

    设计原则：
    - overrides: 用于传递给 resolve_orchestrator_settings 的原始 overrides
    - resolved: resolve_orchestrator_settings 的输出字典
    - 状态字段: requested_mode / effective_mode / has_ampersand_prefix / prefix_routed

    ================================================================================
    & 前缀字段语义说明（重要：内部条件分支规范）
    ================================================================================

    1. has_ampersand_prefix (语法检测层面)
       - 定义：原始 prompt 文本是否以 '&' 开头
       - 用途：消息构建、日志记录、UI 显示

    2. prefix_routed (策略决策层面) - **内部条件分支优先使用此字段**
       - 定义：& 前缀是否成功触发 Cloud 模式
       - 用途：决定执行模式、编排器选择等条件分支
       - 等价于 triggered_by_prefix（该字段为兼容别名）

    3. triggered_by_prefix (兼容别名) - **仅用于兼容输出，避免新代码引用**
       - 定义：prefix_routed 的别名
       - 注意：此字段保留以兼容历史输出，内部条件分支请使用 prefix_routed

    ================================================================================
    使用规范
    ================================================================================

    内部条件分支（决定执行行为）应使用 prefix_routed：
        if options.prefix_routed:
            # & 前缀成功触发 Cloud 模式
            orchestrator = "basic"

    消息构建/日志记录应使用 has_ampersand_prefix：
        if options.has_ampersand_prefix:
            message += "（由 & 前缀触发）"
    """

    # === overrides 字典（用于 resolve_orchestrator_settings） ===
    overrides: dict[str, Any] = field(default_factory=dict)

    # === resolve_orchestrator_settings 的解析结果 ===
    resolved: dict[str, Any] = field(default_factory=dict)

    # === 执行模式状态 ===
    requested_mode: Optional[str] = None  # 原始请求模式（CLI 参数或 config.yaml）
    effective_mode: str = DEFAULT_EXECUTION_MODE  # 有效执行模式（快照，优先使用）

    # === & 前缀状态（策略决策层面） ===
    # & 前缀是否成功触发 Cloud（策略决策）
    # 此字段对应 ExecutionDecision.prefix_routed
    # 内部条件分支应使用 prefix_routed 属性访问此字段（见下方属性定义）
    # triggered_by_prefix 字段名保留以兼容历史输出
    triggered_by_prefix: bool = False

    # === 编排器选择 ===
    orchestrator: str = "mp"  # 最终编排器类型

    # === 原始 prompt 状态（语法检测层面） ===
    has_ampersand_prefix: bool = False  # 语法检测：原始 prompt 是否有 & 前缀
    sanitized_prompt: Optional[str] = None  # 清理后的 prompt（移除 & 前缀）

    # === 用户消息（仅构建，不打印）===
    user_message: Optional[str] = None

    @property
    def prefix_routed(self) -> bool:
        """& 前缀是否成功触发 Cloud 模式（策略决策层面）

        **内部条件分支应优先使用此属性**

        此属性是 triggered_by_prefix 字段的语义明确别名。
        用于决定执行模式、编排器选择等需要进行条件分支的场景。

        示例：
            if options.prefix_routed:
                # & 前缀成功触发 Cloud 模式
                orchestrator = "basic"
        """
        return self.triggered_by_prefix

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "overrides": self.overrides,
            "resolved": self.resolved,
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            # 新字段（语义明确）
            "prefix_routed": self.prefix_routed,
            "has_ampersand_prefix": self.has_ampersand_prefix,
            # 兼容字段
            "triggered_by_prefix": self.triggered_by_prefix,
            "orchestrator": self.orchestrator,
            "sanitized_prompt": self.sanitized_prompt,
            "user_message": self.user_message,
        }


def build_cli_overrides_from_args(
    args: Any,
    nl_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """从 argparse args 构建 CLI overrides 字典

    这是 run.py 和 scripts/run_iterate.py 共享的 overrides 组装逻辑。
    将两入口的重复代码统一到此处，后续仅需改一处即可影响两入口。

    优先级规则：
    1. CLI 显式参数（最高，仅当 args 中对应值不为 None 时）
    2. nl_options（自然语言解析结果，仅当 CLI 未显式指定时）
    3. 不设置（让 resolve_orchestrator_settings 使用 config.yaml）

    Args:
        args: argparse.Namespace 命令行参数
        nl_options: 自然语言解析结果字典（可选），支持的键：
            - workers: Worker 池大小
            - max_iterations: 最大迭代次数
            - execution_mode: 执行模式
            - strict: 严格评审模式（映射为 strict_review）
            - enable_sub_planners: 启用子规划者
            - auto_commit: 自动提交
            - auto_push: 自动推送
            - dry_run: 干运行模式
            - orchestrator: 编排器类型
            - directory: 工作目录

    Returns:
        CLI overrides 字典，仅包含显式指定的参数

    示例:
        >>> args = argparse.Namespace(workers=5, max_iterations=None, ...)
        >>> overrides = build_cli_overrides_from_args(args)
        >>> # overrides = {"workers": 5}  # 仅包含显式指定的参数
    """
    nl_options = nl_options or {}
    cli_overrides: dict[str, Any] = {}

    # ========== 核心配置 ==========

    # workers: CLI > nl_options > (不设置)
    cli_workers = getattr(args, "workers", None)
    if cli_workers is not None:
        cli_overrides["workers"] = cli_workers
    elif "workers" in nl_options and nl_options["workers"] is not None:
        cli_overrides["workers"] = nl_options["workers"]

    # max_iterations: CLI > nl_options > (不设置)
    max_iterations_raw = getattr(args, "max_iterations", None)
    if max_iterations_raw is not None:
        cli_overrides["max_iterations"] = parse_max_iterations(str(max_iterations_raw), raise_on_error=False)
    elif "max_iterations" in nl_options and nl_options["max_iterations"] is not None:
        cli_overrides["max_iterations"] = nl_options["max_iterations"]

    # strict_review: CLI > nl_options (strict 映射) > (不设置)
    cli_strict = getattr(args, "strict_review", None)
    if cli_strict is not None:
        cli_overrides["strict_review"] = cli_strict
    elif "strict" in nl_options and nl_options["strict"] is not None:
        cli_overrides["strict_review"] = nl_options["strict"]

    # enable_sub_planners: CLI > nl_options > (不设置)
    cli_sub_planners = getattr(args, "enable_sub_planners", None)
    if cli_sub_planners is not None:
        cli_overrides["enable_sub_planners"] = cli_sub_planners
    elif "enable_sub_planners" in nl_options and nl_options["enable_sub_planners"] is not None:
        cli_overrides["enable_sub_planners"] = nl_options["enable_sub_planners"]

    # ========== 执行模式 ==========

    # execution_mode: CLI > nl_options > (不设置)
    # 注意：这里传递的是 requested_mode，不是 effective_mode
    cli_execution_mode = getattr(args, "execution_mode", None)
    if cli_execution_mode is not None:
        cli_overrides["execution_mode"] = cli_execution_mode
    elif "execution_mode" in nl_options and nl_options["execution_mode"] is not None:
        cli_overrides["execution_mode"] = nl_options["execution_mode"]

    # ========== Cloud 配置 ==========

    # cloud_timeout: CLI > (不设置)
    cli_cloud_timeout = getattr(args, "cloud_timeout", None)
    if cli_cloud_timeout is not None:
        cli_overrides["cloud_timeout"] = cli_cloud_timeout

    # cloud_auth_timeout: CLI > (不设置)
    cli_cloud_auth_timeout = getattr(args, "cloud_auth_timeout", None)
    if cli_cloud_auth_timeout is not None:
        cli_overrides["cloud_auth_timeout"] = cli_cloud_auth_timeout

    # ========== 副作用控制 ==========

    # minimal: preset 模式（等效于 skip_online + dry_run）
    minimal_mode = nl_options.get("minimal", False) or getattr(args, "minimal", False)
    if minimal_mode:
        # minimal 作为 preset，仅在未显式指定时生效
        if "dry_run" not in cli_overrides and not getattr(args, "dry_run", False):
            cli_overrides["dry_run"] = True
        if "skip_online" not in cli_overrides and not getattr(args, "skip_online", False):
            cli_overrides["skip_online"] = True

    # dry_run: CLI > nl_options > (不设置)
    if getattr(args, "dry_run", False) or "dry_run" in nl_options and nl_options["dry_run"]:
        cli_overrides["dry_run"] = True

    # ========== 提交控制 ==========

    # auto_commit: CLI > nl_options > (不设置)
    if getattr(args, "auto_commit", False) or "auto_commit" in nl_options and nl_options["auto_commit"]:
        cli_overrides["auto_commit"] = True

    # auto_push: CLI > nl_options > (不设置)
    if getattr(args, "auto_push", False) or "auto_push" in nl_options and nl_options["auto_push"]:
        cli_overrides["auto_push"] = True

    # commit_per_iteration: CLI > (不设置)
    if getattr(args, "commit_per_iteration", False):
        cli_overrides["commit_per_iteration"] = True

    # ========== 编排器配置 ==========

    # 编排器配置需要特殊处理：
    # 1. 如果 nl_options 中有 orchestrator（来自 ExecutionDecision），优先使用
    # 2. 否则使用 CLI 参数
    # 3. 检查 _orchestrator_user_set 标志判断用户是否显式指定
    #
    # tri-state 设计：
    # - orchestrator: None=未指定（使用 config.yaml）
    # - no_mp: None=未指定，True=显式设置 --no-mp
    # - _orchestrator_user_set: 元字段，标记用户是否显式传入 --orchestrator 或 --no-mp
    orchestrator_user_set = getattr(args, "_orchestrator_user_set", False)
    cli_orchestrator = getattr(args, "orchestrator", None)
    # 使用 None 作为默认值（与 run.py 保持一致的 tri-state 策略）
    cli_no_mp = getattr(args, "no_mp", None)

    if "orchestrator" in nl_options and not orchestrator_user_set:
        # 使用决策快照中的编排器（build_execution_decision 已处理好兼容性）
        cli_overrides["orchestrator"] = nl_options["orchestrator"]
    elif cli_orchestrator is not None:
        cli_overrides["orchestrator"] = cli_orchestrator
    elif cli_no_mp is True:
        # 显式检查 True（cli_no_mp 可能是 None/True，不会是 False）
        cli_overrides["orchestrator"] = "basic"

    # ========== 模型配置 ==========

    # planner_model: CLI > (不设置)
    if getattr(args, "planner_model", None) is not None:
        cli_overrides["planner_model"] = args.planner_model

    # worker_model: CLI > (不设置)
    if getattr(args, "worker_model", None) is not None:
        cli_overrides["worker_model"] = args.worker_model

    # reviewer_model: CLI > (不设置)
    if getattr(args, "reviewer_model", None) is not None:
        cli_overrides["reviewer_model"] = args.reviewer_model

    return cli_overrides


def build_unified_overrides(
    args: Any,
    nl_options: Optional[dict[str, Any]] = None,
    execution_decision: Optional[Any] = None,
    triggered_by_prefix: bool = False,
    *,
    prefix_routed: Optional[bool] = None,
) -> UnifiedOptions:
    """构建统一的配置解析结果

    这是 run.py 和 scripts/run_iterate.py 共享的配置解析入口。
    统一处理 argparse args、nl_options 和 ExecutionDecision 的输入，
    输出包含 overrides、resolved 配置和执行状态的完整结构。

    设计目标：
    - 将两入口的重复 overrides 组装逻辑统一到此函数
    - 后续仅需修改此函数即可影响两入口
    - 提供完整的状态信息（requested/effective/prefix）

    优先级规则：
    1. CLI 显式参数（最高）
    2. nl_options（自然语言解析结果）
    3. ExecutionDecision（来自 build_execution_decision）
    4. config.yaml（由 resolve_orchestrator_settings 处理）
    5. 代码默认值（最低）

    Args:
        args: argparse.Namespace 命令行参数
        nl_options: 自然语言解析结果字典（可选）
        execution_decision: ExecutionDecision 实例（可选）
            来自 core.execution_policy.build_execution_decision
        triggered_by_prefix: [DEPRECATED] 请使用 prefix_routed 参数。
            此参数保留以兼容旧代码。
        prefix_routed: & 前缀是否成功触发 Cloud 模式（策略决策层面）。
            当 execution_decision 提供时，此参数会被 execution_decision.prefix_routed 覆盖。
            新代码应使用此参数而非 triggered_by_prefix。

    Returns:
        UnifiedOptions 包含完整的配置解析结果

    示例:
        >>> from core.config import build_unified_overrides
        >>> from core.execution_policy import build_execution_decision
        >>>
        >>> # 方式 1：仅使用 args
        >>> options = build_unified_overrides(args)
        >>>
        >>> # 方式 2：结合 nl_options（run.py 场景）
        >>> options = build_unified_overrides(args, nl_options=analysis.options)
        >>>
        >>> # 方式 3：结合 ExecutionDecision（推荐）
        >>> decision = build_execution_decision(prompt, ...)
        >>> options = build_unified_overrides(args, execution_decision=decision)
        >>>
        >>> # 使用解析结果
        >>> print(options.effective_mode)  # "cli" / "cloud" / "auto"
        >>> print(options.orchestrator)    # "mp" / "basic"
        >>> print(options.resolved["workers"])  # 3
    """
    # 参数别名处理：prefix_routed 优先于 triggered_by_prefix

    nl_options = nl_options or {}

    # ========== 从 ExecutionDecision 提取信息 ==========
    #
    # 字段语义说明：
    # - has_ampersand_prefix: 语法检测（是否存在 & 前缀）
    # - prefix_routed: 策略决策（& 前缀是否成功触发 Cloud），用于内部分支
    # - triggered_by_prefix: prefix_routed 的兼容别名（由 prefix_routed 派生）

    if execution_decision is None:
        # 【关键修改】execution_decision 缺失时，使用 compute_decision_inputs 统一重建
        # 此 helper 封装了：
        # - 获取 CLI 参数和 Cloud 配置
        # - 从 nl_options 提取原始 prompt（支持 "_original_goal" 或 "goal" 字段）
        # - 检测 & 前缀并处理虚拟 prompt 构造（封装在 VIRTUAL_PROMPT_FOR_PREFIX_DETECTION）
        # - 使用 resolve_requested_mode_for_decision 确定 requested_mode
        # - 使用 resolve_mode_source 确定 mode_source
        #
        # 【重要】auto_detect_cloud_prefix 配置决定 & 前缀是否参与 Cloud 路由：
        # - 从 config.yaml 的 cloud_agent.auto_detect_cloud_prefix 读取
        # - CLI 参数可覆盖配置值（tri-state: None/True/False）
        # - 当 auto_detect_cloud_prefix=False 时，& 前缀被忽略：
        #   * prefix_routed 始终为 False
        #   * 编排器允许 mp（不因 & 前缀强制 basic）
        # - 详见 core/execution_policy.compute_decision_inputs 和 AGENTS.md R-3 规则
        from core.execution_policy import compute_decision_inputs

        decision_inputs = compute_decision_inputs(args, nl_options=nl_options)
        execution_decision = decision_inputs.build_decision()

    # 使用 ExecutionDecision 的状态信息（无论是传入的还是重新构建的）
    # 内部分支统一使用 decision_prefix_routed，triggered_by_prefix 仅作为输出兼容字段
    decision_prefix_routed = getattr(execution_decision, "prefix_routed", False)
    requested_mode = getattr(execution_decision, "requested_mode", None)
    effective_mode = getattr(execution_decision, "effective_mode", DEFAULT_EXECUTION_MODE)
    has_ampersand_prefix = getattr(execution_decision, "has_ampersand_prefix", False)
    sanitized_prompt = getattr(execution_decision, "sanitized_prompt", None)
    user_message = getattr(execution_decision, "user_message", None)

    # 从 ExecutionDecision 获取 orchestrator（如果有）
    # 【重要】nl_options 中的 orchestrator 优先，因为可能是自然语言检测后覆盖的
    # execution_decision.orchestrator 是构建时的值，可能已被后续处理覆盖
    if hasattr(execution_decision, "orchestrator") and "orchestrator" not in nl_options:
        nl_options = nl_options.copy()
        nl_options["orchestrator"] = execution_decision.orchestrator

    # ========== 构建 CLI overrides ==========

    cli_overrides = build_cli_overrides_from_args(args, nl_options)

    # ========== 调用 resolve_orchestrator_settings ==========

    resolved = resolve_orchestrator_settings(
        overrides=cli_overrides,
        prefix_routed=decision_prefix_routed,  # 使用新参数名
    )

    # ========== 确定最终的 execution_mode ==========

    # 使用 ExecutionDecision 的 effective_mode
    # 注：execution_decision 总是存在（如果缺失会在上面重新构建）
    final_effective_mode = effective_mode

    # ========== 构建 UnifiedOptions ==========

    return UnifiedOptions(
        overrides=cli_overrides,
        resolved=resolved,
        requested_mode=requested_mode or resolved["execution_mode"],
        effective_mode=final_effective_mode,
        triggered_by_prefix=decision_prefix_routed,  # 兼容字段，值来自 prefix_routed
        orchestrator=resolved["orchestrator"],
        has_ampersand_prefix=has_ampersand_prefix,
        sanitized_prompt=sanitized_prompt,
        user_message=user_message,
    )
