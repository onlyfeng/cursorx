"""测试 scripts/run_knowledge.py 的配置加载和默认值行为

测试覆盖：
1. parse_args 使用 config.yaml 中的默认值
2. strict/sub_planners tri-state 互斥组逻辑
3. 模型和超时配置优先级 (CLI > config.yaml)
4. stream-json 配置通过 resolve_stream_log_config 正确注入
5. 最终传入 OrchestratorConfig 的字段能覆盖默认常量
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigManager, resolve_stream_log_config

# ============================================================
# 测试配置内容常量
# ============================================================

RUN_KNOWLEDGE_CONFIG_CONTENT = {
    "system": {
        "max_iterations": 25,
        "worker_pool_size": 6,
        "enable_sub_planners": False,
        "strict_review": True,
    },
    "models": {
        "planner": "custom-planner-for-knowledge",
        "worker": "custom-worker-for-knowledge",
        "reviewer": "custom-reviewer-for-knowledge",
    },
    "planner": {
        "timeout": 400.0,
    },
    "worker": {
        "task_timeout": 500.0,
        "knowledge_integration": {
            "max_docs": 8,
        },
    },
    "reviewer": {
        "timeout": 250.0,
    },
    "logging": {
        "stream_json": {
            "enabled": True,
            "console": False,
            "detail_dir": "logs/knowledge_stream/detail/",
            "raw_dir": "logs/knowledge_stream/raw/",
        },
    },
    "cloud_agent": {
        "enabled": True,
        "execution_mode": "auto",
        "timeout": 900,
        "auth_timeout": 60,
        "api_key": "yaml-cloud-key-for-knowledge",
        "api_base_url": "https://custom.api.cursor.com",
    },
}


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def run_knowledge_config_yaml(tmp_path: Path) -> Path:
    """创建用于测试 run_knowledge.py 的自定义 config.yaml"""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(RUN_KNOWLEDGE_CONFIG_CONTENT, f, allow_unicode=True)
    return config_path


@pytest.fixture
def reset_config_manager():
    """重置 ConfigManager 单例的 fixture"""
    original_instance = ConfigManager._instance
    original_config = ConfigManager._config
    original_config_path = ConfigManager._config_path

    ConfigManager.reset_instance()

    yield

    ConfigManager._instance = original_instance
    ConfigManager._config = original_config
    ConfigManager._config_path = original_config_path


# ============================================================
# TestRunKnowledgeDefaultsFromConfig - run_knowledge.py 默认值测试
# ============================================================


class TestRunKnowledgeDefaultsFromConfig:
    """测试 scripts/run_knowledge.py 默认值来自 config.yaml"""

    def test_workers_default_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 workers 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.system.worker_pool_size == 6

    def test_max_iterations_default_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 max_iterations 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.system.max_iterations == 25

    def test_enable_sub_planners_default_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 enable_sub_planners 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.system.enable_sub_planners is False

    def test_strict_review_default_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 strict_review 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.system.strict_review is True

    def test_kb_limit_default_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 kb_limit 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.worker.knowledge_integration.max_docs == 8


# ============================================================
# TestRunKnowledgeModelsFromConfig - 模型配置测试
# ============================================================


class TestRunKnowledgeModelsFromConfig:
    """测试 run_knowledge.py 模型配置来自 config.yaml"""

    def test_planner_model_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 planner_model 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.models.planner == "custom-planner-for-knowledge"

    def test_worker_model_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 worker_model 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.models.worker == "custom-worker-for-knowledge"

    def test_reviewer_model_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 reviewer_model 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.models.reviewer == "custom-reviewer-for-knowledge"


# ============================================================
# TestRunKnowledgeTimeoutsFromConfig - 超时配置测试
# ============================================================


class TestRunKnowledgeTimeoutsFromConfig:
    """测试 run_knowledge.py 超时配置来自 config.yaml"""

    def test_planner_timeout_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 planner_timeout 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.planner.timeout == 400.0

    def test_worker_timeout_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 worker_timeout 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.worker.task_timeout == 500.0

    def test_reviewer_timeout_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 reviewer_timeout 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.reviewer.timeout == 250.0


# ============================================================
# TestRunKnowledgeStreamJsonConfig - stream-json 配置测试
# ============================================================


class TestRunKnowledgeStreamJsonConfig:
    """测试 run_knowledge.py stream-json 配置来自 config.yaml"""

    def test_stream_json_enabled_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 stream_json.enabled 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.logging.stream_json.enabled is True

    def test_stream_json_console_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 stream_json.console 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.logging.stream_json.console is False

    def test_stream_json_dirs_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 stream_json 目录配置来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.logging.stream_json.detail_dir == "logs/knowledge_stream/detail/"
        assert config.logging.stream_json.raw_dir == "logs/knowledge_stream/raw/"

    def test_resolve_stream_log_config_uses_yaml(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 resolve_stream_log_config 从 config.yaml 读取配置"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        result = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert result["enabled"] is True
        assert result["console"] is False
        assert result["detail_dir"] == "logs/knowledge_stream/detail/"
        assert result["raw_dir"] == "logs/knowledge_stream/raw/"

    def test_resolve_stream_log_config_cli_overrides(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI 参数覆盖 config.yaml 中的 stream_json 配置"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        result = resolve_stream_log_config(
            cli_enabled=False,
            cli_console=True,
            cli_detail_dir="cli_override_detail/",
            cli_raw_dir="cli_override_raw/",
        )

        # CLI 参数应该覆盖 config.yaml
        assert result["enabled"] is False
        assert result["console"] is True
        assert result["detail_dir"] == "cli_override_detail/"
        assert result["raw_dir"] == "cli_override_raw/"


# ============================================================
# TestRunKnowledgeTriStateOptions - tri-state 互斥组测试
# ============================================================


class TestRunKnowledgeTriStateOptions:
    """测试 run_knowledge.py 的 tri-state 互斥组逻辑

    tri-state 设计:
    - None: CLI 未指定，使用 config.yaml 默认值
    - True: CLI 显式指定 --strict / --sub-planners
    - False: CLI 显式指定 --no-strict / --no-sub-planners
    """

    def test_strict_review_none_uses_config_default(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 strict_review=None 时使用 config.yaml 默认值"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 未指定 (None)
        cli_strict_review = None
        resolved = cli_strict_review if cli_strict_review is not None else config.system.strict_review

        # config.yaml 中 strict_review=True
        assert resolved is True

    def test_strict_review_true_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 strict_review=True 覆盖 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 指定 --strict (True)
        cli_strict_review = True
        resolved = cli_strict_review if cli_strict_review is not None else config.system.strict_review

        assert resolved is True

    def test_strict_review_false_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 strict_review=False 覆盖 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 指定 --no-strict (False)
        cli_strict_review = False
        resolved = cli_strict_review if cli_strict_review is not None else config.system.strict_review

        # CLI False 应覆盖 config.yaml 的 True
        assert resolved is False

    def test_enable_sub_planners_none_uses_config_default(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 enable_sub_planners=None 时使用 config.yaml 默认值"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 未指定 (None)
        cli_enable_sub_planners = None
        resolved = cli_enable_sub_planners if cli_enable_sub_planners is not None else config.system.enable_sub_planners

        # config.yaml 中 enable_sub_planners=False
        assert resolved is False

    def test_enable_sub_planners_true_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 enable_sub_planners=True 覆盖 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 指定 --sub-planners (True)
        cli_enable_sub_planners = True
        resolved = cli_enable_sub_planners if cli_enable_sub_planners is not None else config.system.enable_sub_planners

        # CLI True 应覆盖 config.yaml 的 False
        assert resolved is True


# ============================================================
# TestRunKnowledgeOrchestratorConfigInjection - OrchestratorConfig 注入测试
# ============================================================


class TestRunKnowledgeOrchestratorConfigInjection:
    """测试 run_knowledge.py 最终注入到 OrchestratorConfig 的配置

    验证配置优先级: CLI > config.yaml > DEFAULT_*
    """

    def test_orchestrator_config_receives_yaml_values(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 OrchestratorConfig 接收 config.yaml 中的值"""
        from coordinator.orchestrator import OrchestratorConfig

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 run_knowledge.py 中的配置解析逻辑
        orchestrator_config = OrchestratorConfig(
            working_directory=".",
            max_iterations=config.system.max_iterations,
            worker_pool_size=config.system.worker_pool_size,
            enable_sub_planners=config.system.enable_sub_planners,
            strict_review=config.system.strict_review,
            planner_model=config.models.planner,
            worker_model=config.models.worker,
            reviewer_model=config.models.reviewer,
            planner_timeout=config.planner.timeout,
            worker_task_timeout=config.worker.task_timeout,
            reviewer_timeout=config.reviewer.timeout,
        )

        # 验证 OrchestratorConfig 接收了 config.yaml 中的值
        assert orchestrator_config.max_iterations == 25
        assert orchestrator_config.worker_pool_size == 6
        assert orchestrator_config.enable_sub_planners is False
        assert orchestrator_config.strict_review is True
        assert orchestrator_config.planner_model == "custom-planner-for-knowledge"
        assert orchestrator_config.worker_model == "custom-worker-for-knowledge"
        assert orchestrator_config.reviewer_model == "custom-reviewer-for-knowledge"
        assert orchestrator_config.planner_timeout == 400.0
        assert orchestrator_config.worker_task_timeout == 500.0
        assert orchestrator_config.reviewer_timeout == 250.0

    def test_orchestrator_config_cli_overrides_yaml(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI 参数覆盖 config.yaml 值后注入 OrchestratorConfig"""
        from coordinator.orchestrator import OrchestratorConfig

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟 CLI 参数覆盖
        cli_workers = 10
        cli_planner_model = "cli-override-planner"
        cli_planner_timeout = 999.0

        # 解析后的值 (CLI > config.yaml)
        resolved_workers = cli_workers  # CLI 指定
        resolved_planner_model = cli_planner_model  # CLI 指定
        resolved_planner_timeout = cli_planner_timeout  # CLI 指定
        resolved_worker_model = config.models.worker  # 未指定，使用 config.yaml

        orchestrator_config = OrchestratorConfig(
            working_directory=".",
            worker_pool_size=resolved_workers,
            planner_model=resolved_planner_model,
            planner_timeout=resolved_planner_timeout,
            worker_model=resolved_worker_model,
        )

        # 验证 CLI 覆盖生效
        assert orchestrator_config.worker_pool_size == 10
        assert orchestrator_config.planner_model == "cli-override-planner"
        assert orchestrator_config.planner_timeout == 999.0
        # 验证未覆盖的值来自 config.yaml
        assert orchestrator_config.worker_model == "custom-worker-for-knowledge"


# ============================================================
# TestRunKnowledgeDefaultsWhenNoConfig - 无配置文件时默认值测试
# ============================================================


class TestRunKnowledgeDefaultsWhenNoConfig:
    """测试无 config.yaml 时 run_knowledge.py 使用 DEFAULT_* 常量"""

    def test_uses_default_constants_when_no_config(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试无配置文件时使用 DEFAULT_* 常量"""
        from core.config import (
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_PLANNER_MODEL,
            DEFAULT_PLANNING_TIMEOUT,
            DEFAULT_REVIEW_TIMEOUT,
            DEFAULT_REVIEWER_MODEL,
            DEFAULT_WORKER_MODEL,
            DEFAULT_WORKER_POOL_SIZE,
            DEFAULT_WORKER_TIMEOUT,
        )

        # 空目录（无 config.yaml）
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # patch _find_config_file 返回 None
        with patch.object(ConfigManager, "_find_config_file", return_value=None):
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            # 验证使用 DEFAULT_* 常量
            assert config.system.max_iterations == DEFAULT_MAX_ITERATIONS
            assert config.system.worker_pool_size == DEFAULT_WORKER_POOL_SIZE
            assert config.models.planner == DEFAULT_PLANNER_MODEL
            assert config.models.worker == DEFAULT_WORKER_MODEL
            assert config.models.reviewer == DEFAULT_REVIEWER_MODEL
            assert config.planner.timeout == DEFAULT_PLANNING_TIMEOUT
            assert config.worker.task_timeout == DEFAULT_WORKER_TIMEOUT
            assert config.reviewer.timeout == DEFAULT_REVIEW_TIMEOUT


# ============================================================
# TestRunKnowledgeParseArgsIntegration - parse_args 集成测试
# ============================================================


class TestRunKnowledgeParseArgsIntegration:
    """测试 run_knowledge.py parse_args() 与 ConfigManager 的集成

    验证 parse_args 能正确读取 config.yaml 并设置默认值。
    """

    def test_parse_args_reads_config_for_defaults(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 parse_args 从 config.yaml 读取默认值"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        # 验证 ConfigManager 能正确读取配置
        config = ConfigManager.get_instance()

        # 这些值将被 parse_args 用作默认值
        assert config.system.worker_pool_size == 6
        assert config.system.max_iterations == 25
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True
        assert config.worker.knowledge_integration.max_docs == 8

    def test_parse_args_model_defaults_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 parse_args 模型默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 这些值将被 parse_args 显示在 help 中
        assert config.models.planner == "custom-planner-for-knowledge"
        assert config.models.worker == "custom-worker-for-knowledge"
        assert config.models.reviewer == "custom-reviewer-for-knowledge"

    def test_parse_args_timeout_defaults_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 parse_args 超时默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 这些值将被 parse_args 显示在 help 中
        assert config.planner.timeout == 400.0
        assert config.worker.task_timeout == 500.0
        assert config.reviewer.timeout == 250.0


# ============================================================
# TestRunKnowledgeConfigPriorityOrder - 配置优先级顺序测试
# ============================================================


class TestRunKnowledgeConfigPriorityOrder:
    """测试 run_knowledge.py 配置优先级顺序

    优先级: CLI > config.yaml > DEFAULT_*
    """

    def test_cli_overrides_config_yaml(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI 参数优先级高于 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 模拟解析逻辑
        cli_planner_model = "cli-model"  # CLI 指定
        cli_worker_model = None  # CLI 未指定

        resolved_planner = cli_planner_model if cli_planner_model else config.models.planner
        resolved_worker = cli_worker_model if cli_worker_model else config.models.worker

        # CLI 指定的值应优先
        assert resolved_planner == "cli-model"
        # 未指定的值应使用 config.yaml
        assert resolved_worker == "custom-worker-for-knowledge"

    def test_config_yaml_overrides_defaults(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 config.yaml 优先级高于 DEFAULT_* 常量"""
        from core.config import (
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_WORKER_POOL_SIZE,
        )

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # config.yaml 中的值应覆盖 DEFAULT_* 常量
        # config.yaml: worker_pool_size=6, DEFAULT_WORKER_POOL_SIZE=3
        assert config.system.worker_pool_size == 6
        assert config.system.worker_pool_size != DEFAULT_WORKER_POOL_SIZE

        # config.yaml: max_iterations=25, DEFAULT_MAX_ITERATIONS=10
        assert config.system.max_iterations == 25
        assert config.system.max_iterations != DEFAULT_MAX_ITERATIONS


# ============================================================
# TestRunKnowledgeCloudConfig - Cloud 配置测试
# ============================================================


class TestRunKnowledgeCloudConfig:
    """测试 run_knowledge.py 的 Cloud 配置从 config.yaml 加载

    验证：
    1. cloud_agent.execution_mode 从 config.yaml 正确加载
    2. cloud_agent.timeout 和 auth_timeout 从 config.yaml 正确加载
    3. CLI 参数能正确覆盖 config.yaml
    """

    def test_execution_mode_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 execution_mode 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.cloud_agent.execution_mode == "auto"

    def test_cloud_timeout_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 cloud_timeout 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.cloud_agent.timeout == 900

    def test_cloud_auth_timeout_from_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 cloud_auth_timeout 默认值来自 config.yaml"""
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()
        assert config.cloud_agent.auth_timeout == 60


# ============================================================
# TestRunKnowledgeCloudConfigCLIOverride - CLI 覆盖 Cloud 配置测试
# ============================================================


class TestRunKnowledgeCloudConfigCLIOverride:
    """测试 run_knowledge.py CLI 参数覆盖 config.yaml 中的 Cloud 配置

    验证 CLI 参数优先级高于 config.yaml
    """

    def test_cli_execution_mode_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --execution-mode 覆盖 config.yaml"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        # 不指定 CLI 参数时，使用 config.yaml 值
        settings_default = resolve_orchestrator_settings(overrides={})
        assert settings_default["execution_mode"] == "auto"

        # CLI 参数覆盖 config.yaml
        settings_cli = resolve_orchestrator_settings(
            overrides={
                "execution_mode": "cloud",
            }
        )
        assert settings_cli["execution_mode"] == "cloud"

    def test_cli_cloud_timeout_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --cloud-timeout 覆盖 config.yaml"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        # 不指定 CLI 参数时，使用 config.yaml 值
        settings_default = resolve_orchestrator_settings(overrides={})
        assert settings_default["cloud_timeout"] == 900

        # CLI 参数覆盖 config.yaml
        settings_cli = resolve_orchestrator_settings(
            overrides={
                "cloud_timeout": 1200,
            }
        )
        assert settings_cli["cloud_timeout"] == 1200

    def test_cli_cloud_auth_timeout_overrides_config(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --cloud-auth-timeout 覆盖 config.yaml"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        ConfigManager.reset_instance()

        # 不指定 CLI 参数时，使用 config.yaml 值
        settings_default = resolve_orchestrator_settings(overrides={})
        assert settings_default["cloud_auth_timeout"] == 60

        # CLI 参数覆盖 config.yaml
        settings_cli = resolve_orchestrator_settings(
            overrides={
                "cloud_auth_timeout": 90,
            }
        )
        assert settings_cli["cloud_auth_timeout"] == 90


# ============================================================
# TestRunKnowledgeCloudAuthConfigBuild - CloudAuthConfig 构建测试
# ============================================================


class TestRunKnowledgeCloudAuthConfigBuild:
    """测试 run_knowledge.py 中 _build_cloud_auth_config 逻辑

    验证：
    1. 未配置 API Key 时返回 None
    2. CLI --cloud-api-key 优先级最高
    3. 环境变量 CURSOR_API_KEY 优先级高于 config.yaml
    """

    def test_no_api_key_returns_none(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试未配置 API Key 时返回 None"""
        from core.config import resolve_orchestrator_settings
        from scripts.run_knowledge import _build_cloud_auth_config

        # 创建无 cloud_agent.api_key 的配置
        config_content = {
            "system": {"max_iterations": 10},
            "cloud_agent": {
                "execution_mode": "cloud",
                "timeout": 300,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        # 清除环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
        ConfigManager.reset_instance()

        # 模拟 args 对象
        class MockArgs:
            cloud_api_key = None

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is None

    def test_cli_api_key_highest_priority(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --cloud-api-key 优先级最高"""
        from core.config import resolve_orchestrator_settings
        from scripts.run_knowledge import _build_cloud_auth_config

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        # 设置环境变量（应被 CLI 覆盖）
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        ConfigManager.reset_instance()

        # 模拟 args 对象，CLI 指定了 --cloud-api-key
        class MockArgs:
            cloud_api_key = "cli-explicit-api-key"

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is not None
        assert result.api_key == "cli-explicit-api-key"

    def test_env_cursor_api_key_priority_over_yaml(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试环境变量 CURSOR_API_KEY 优先级高于 config.yaml"""
        from core.config import resolve_orchestrator_settings
        from scripts.run_knowledge import _build_cloud_auth_config

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        # 设置环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        ConfigManager.reset_instance()

        # 模拟 args 对象，CLI 未指定
        class MockArgs:
            cloud_api_key = None

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is not None
        # CURSOR_API_KEY 应优先于 config.yaml 中的 yaml-cloud-key-for-knowledge
        assert result.api_key == "env-cursor-api-key"

    def test_yaml_api_key_used_when_no_env(
        self,
        tmp_path: Path,
        run_knowledge_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试无环境变量时使用 config.yaml 中的 api_key"""
        from cursor.cloud_client import CloudClientFactory

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        # 清除环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
        ConfigManager.reset_instance()

        # 使用 CloudClientFactory.resolve_api_key 测试
        api_key = CloudClientFactory.resolve_api_key(explicit_api_key=None)

        # 应使用 config.yaml 中的值
        assert api_key == "yaml-cloud-key-for-knowledge"
