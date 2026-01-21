"""统一配置加载模块

本模块提供从 config.yaml 加载配置的统一接口，作为所有配置的权威来源。
其他模块应通过此模块获取配置，而非硬编码默认值。

配置优先级:
1. 命令行参数（最高）
2. 环境变量
3. config.yaml
4. 代码默认值（最低）

使用方式:
    from core.config import get_config, ConfigManager

    # 获取单例配置管理器
    config = get_config()

    # 获取模型配置
    planner_model = config.models.planner
    worker_model = config.models.worker
    reviewer_model = config.models.reviewer

    # 获取超时配置
    planning_timeout = config.planner.timeout
    worker_timeout = config.worker.task_timeout
    review_timeout = config.reviewer.timeout
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


# ============================================================
# 默认值常量 - 与 config.yaml 保持同步
# ============================================================

# 模型默认值
DEFAULT_PLANNER_MODEL = "gpt-5.2-high"
DEFAULT_WORKER_MODEL = "opus-4.5-thinking"
DEFAULT_REVIEWER_MODEL = "gpt-5.2-codex"

# 超时默认值（秒）
DEFAULT_PLANNING_TIMEOUT = 500.0
DEFAULT_WORKER_TIMEOUT = 600.0
DEFAULT_REVIEW_TIMEOUT = 300.0
DEFAULT_AGENT_CLI_TIMEOUT = 300
DEFAULT_CLOUD_TIMEOUT = 300
DEFAULT_CLOUD_AUTH_TIMEOUT = 30

# 系统默认值
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_WORKER_POOL_SIZE = 3
DEFAULT_ENABLE_SUB_PLANNERS = True
DEFAULT_STRICT_REVIEW = False

# 流式日志默认值 - 与 config.yaml logging.stream_json 保持同步
# 唯一权威来源：这些常量与 config.yaml 同步，编排器/客户端应使用这些常量或 tri-state 设计
DEFAULT_STREAM_EVENTS_ENABLED = False        # 默认关闭流式日志
DEFAULT_STREAM_LOG_CONSOLE = True            # 默认输出到控制台
DEFAULT_STREAM_LOG_DETAIL_DIR = "logs/stream_json/detail/"
DEFAULT_STREAM_LOG_RAW_DIR = "logs/stream_json/raw/"


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
class CloudAgentConfig:
    """Cloud Agent 配置"""
    enabled: bool = False
    execution_mode: str = "cli"
    api_base_url: str = "https://api.cursor.com"
    api_key: Optional[str] = None
    timeout: int = DEFAULT_CLOUD_TIMEOUT
    auth_timeout: int = DEFAULT_CLOUD_AUTH_TIMEOUT
    max_retries: int = 3
    egress_ip_cache_ttl: int = 3600
    egress_ip_api_url: str = ""
    verify_egress_ip: bool = False


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
    knowledge_integration: KnowledgeIntegrationConfig = field(
        default_factory=KnowledgeIntegrationConfig
    )


@dataclass
class ReviewerConfig:
    """评审者配置"""
    strict_mode: bool = False
    timeout: float = DEFAULT_REVIEW_TIMEOUT


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
    include_patterns: List[str] = field(default_factory=lambda: [
        "**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**", "**/.git/**", "**/venv/**",
        "**/__pycache__/**", "**/dist/**", "**/build/**"
    ])
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
    stream_json: StreamJsonLoggingConfig = field(
        default_factory=StreamJsonLoggingConfig
    )


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
                with open(config_path, "r", encoding="utf-8") as f:
                    raw_config = yaml.safe_load(f) or {}
                self._config = self._parse_config(raw_config)
                logger.debug(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
                self._config = AppConfig()
        else:
            logger.debug("未找到 config.yaml，使用默认配置")
            self._config = AppConfig()

    def _parse_config(self, raw: Dict[str, Any]) -> AppConfig:
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
            config.cloud_agent = CloudAgentConfig(
                enabled=cloud_raw.get("enabled", False),
                execution_mode=cloud_raw.get("execution_mode", "cli"),
                api_base_url=cloud_raw.get("api_base_url", "https://api.cursor.com"),
                api_key=cloud_raw.get("api_key"),
                timeout=cloud_raw.get("timeout", DEFAULT_CLOUD_TIMEOUT),
                auth_timeout=cloud_raw.get("auth_timeout", DEFAULT_CLOUD_AUTH_TIMEOUT),
                max_retries=cloud_raw.get("max_retries", 3),
                egress_ip_cache_ttl=cloud_raw.get("egress_ip_cache_ttl", 3600),
                egress_ip_api_url=cloud_raw.get("egress_ip_api_url", ""),
                verify_egress_ip=cloud_raw.get("verify_egress_ip", False),
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
                include_patterns=idx_raw.get("include_patterns", [
                    "**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"
                ]),
                exclude_patterns=idx_raw.get("exclude_patterns", [
                    "**/node_modules/**", "**/.git/**", "**/venv/**",
                    "**/__pycache__/**", "**/dist/**", "**/build/**"
                ]),
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
    except ValueError:
        if raise_on_error:
            raise MaxIterationsParseError(
                f"无效的迭代次数: {value}。使用正整数或 MAX/-1 表示无限迭代"
            )
        else:
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
    except MaxIterationsParseError as e:
        raise argparse.ArgumentTypeError(str(e))


def get_model_config() -> ModelsConfig:
    """获取模型配置"""
    return get_config().models


def get_timeout_config() -> Dict[str, float]:
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
    config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        stream_config = (
            config_data.get("logging", {})
            .get("stream_json", {})
        )
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
    fallback_env_keys: Optional[List[str]] = None,
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        ...     overrides={"timeout": 600, "model": "opus-4.5-thinking"}
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
            "cli",
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """解析编排器设置

    从 config.yaml 加载系统配置并应用覆盖项，生成编排器所需的配置。

    优先级: overrides (CLI) > config.yaml > DEFAULT_*

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
            - orchestrator: 编排器类型 (mp/basic)
            - execution_mode: 执行模式 (cli/cloud/auto)
            - auto_commit: 是否自动提交
            - auto_push: 是否自动推送
            - commit_per_iteration: 每次迭代都提交
            - dry_run: 是否为干运行模式

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
        - execution_mode: 执行模式
        - auto_commit: 自动提交
        - auto_push: 自动推送
        - commit_per_iteration: 每次迭代提交
        - dry_run: 干运行模式

    示例:
        >>> settings = resolve_orchestrator_settings(
        ...     overrides={"workers": 5, "max_iterations": 20}
        ... )
        >>> # settings 包含所有编排器需要的配置
    """
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
            "cli",
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

    # 处理 execution_mode 对 orchestrator 的影响
    # Cloud/Auto 模式强制使用 basic 编排器
    if result["execution_mode"] in ("cloud", "auto"):
        if result["orchestrator"] == "mp":
            logger.debug(
                f"execution_mode={result['execution_mode']} 不兼容 MP 编排器，"
                "自动切换到 basic 编排器"
            )
            result["orchestrator"] = "basic"

    return result


def resolve_agent_timeouts() -> Dict[str, float]:
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    execution_mode = _resolve_with_priority(
        overrides.get("execution_mode"),
        None,
        cloud_agent.execution_mode,
        "cli",
    )

    # Cloud/Auto 模式强制使用 basic 编排器
    if execution_mode in ("cloud", "auto"):
        if orchestrator == "mp":
            logger.debug(
                f"execution_mode={execution_mode} 不兼容 MP 编排器，"
                "自动切换到 basic 编排器"
            )
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
    import os

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


def build_cloud_auth_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
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
    if role_lower == "planner":
        return config.models.planner
    elif role_lower == "worker":
        return config.models.worker
    elif role_lower == "reviewer":
        return config.models.reviewer
    else:
        return config.models.worker  # 默认返回 worker 模型


# ============================================================
# 配置调试输出工具
# ============================================================

# 配置调试输出的前缀，便于脚本化 grep/CI 断言
CONFIG_DEBUG_PREFIX = "[CONFIG]"


def format_debug_config(
    cli_overrides: Optional[Dict[str, Any]] = None,
    source_label: str = "unknown",
) -> str:
    """格式化配置调试输出

    生成稳定格式的配置输出，便于脚本化 grep/CI 断言。
    输出包含:
    - config_path: 配置文件路径
    - resolved_settings: 关键配置字段
    - execution_mode 回退信息（如 mp 在 cloud/auto 下回退到 basic）

    输出格式:
    [CONFIG] config_path: /path/to/config.yaml
    [CONFIG] source: run.py
    [CONFIG] max_iterations: 10
    [CONFIG] workers: 3
    [CONFIG] execution_mode: cli
    [CONFIG] orchestrator: mp
    [CONFIG] orchestrator_fallback: none
    [CONFIG] planner_model: gpt-5.2-high
    [CONFIG] worker_model: opus-4.5-thinking
    [CONFIG] reviewer_model: gpt-5.2-codex
    [CONFIG] cloud_timeout: 300
    [CONFIG] auto_commit: false
    [CONFIG] dry_run: false

    Args:
        cli_overrides: CLI 命令行传入的覆盖配置
        source_label: 调用来源标识（如 "run.py", "scripts/run_iterate.py"）

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
    original_orchestrator = cli_overrides.get("orchestrator", "mp")
    final_orchestrator = resolved["orchestrator"]
    execution_mode = resolved["execution_mode"]

    # 判断是否发生了回退
    orchestrator_fallback = "none"
    if original_orchestrator == "mp" and final_orchestrator == "basic":
        if execution_mode in ("cloud", "auto"):
            orchestrator_fallback = f"mp->basic (execution_mode={execution_mode})"

    # 构建输出行
    lines = [
        f"{CONFIG_DEBUG_PREFIX} config_path: {config_path or '(default)'}",
        f"{CONFIG_DEBUG_PREFIX} source: {source_label}",
        f"{CONFIG_DEBUG_PREFIX} max_iterations: {resolved['max_iterations']}",
        f"{CONFIG_DEBUG_PREFIX} workers: {resolved['workers']}",
        f"{CONFIG_DEBUG_PREFIX} execution_mode: {execution_mode}",
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
    cli_overrides: Optional[Dict[str, Any]] = None,
    source_label: str = "unknown",
) -> None:
    """打印配置调试信息到标准输出

    Args:
        cli_overrides: CLI 命令行传入的覆盖配置
        source_label: 调用来源标识
    """
    output = format_debug_config(cli_overrides, source_label)
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
    execution_mode: str = "cli"
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
