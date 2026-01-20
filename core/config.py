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
DEFAULT_CLOUD_TIMEOUT = 600
DEFAULT_CLOUD_AUTH_TIMEOUT = 30

# 系统默认值
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_WORKER_POOL_SIZE = 3
DEFAULT_ENABLE_SUB_PLANNERS = True
DEFAULT_STRICT_REVIEW = False


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
    """流式 JSON 日志配置"""
    enabled: bool = False
    console: bool = True
    detail_dir: str = "logs/stream_json/detail/"
    raw_dir: str = "logs/stream_json/raw/"


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
                    enabled=sj_raw.get("enabled", False),
                    console=sj_raw.get("console", True),
                    detail_dir=sj_raw.get("detail_dir", "logs/stream_json/detail/"),
                    raw_dir=sj_raw.get("raw_dir", "logs/stream_json/raw/"),
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
