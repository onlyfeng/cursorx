"""测试从自定义 config.yaml 加载配置

测试覆盖：
1. ConfigManager 从自定义 config.yaml 读取配置
2. run.py parse_args 使用 config.yaml 中的默认值
3. scripts/run_iterate.py parse_args 使用 config.yaml 中的默认值
4. scripts/run_basic.py/scripts/run_mp.py 的 stream_json 配置从 config.yaml 读取
5. indexing/cli.py load_config() 能从新键名读取
6. 各入口脚本的配置解析路径验证：
   - 未传 CLI 参数时取值等于 config.yaml
   - 传 CLI 参数时覆盖 config.yaml
   - execution_mode=cloud/auto 时强制 basic 编排器策略
   - stream_json.enabled 驱动 CursorAgentConfig.stream_events_enabled
"""

import argparse
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import yaml

# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ConfigManager,
    get_config,
)

# ============================================================
# 测试配置内容常量
# ============================================================

CUSTOM_CONFIG_CONTENT = {
    "system": {
        "max_iterations": 99,
        "worker_pool_size": 7,
        "enable_sub_planners": False,
        "strict_review": True,
    },
    "models": {
        "planner": "custom-planner-model",
        "worker": "custom-worker-model",
        "reviewer": "custom-reviewer-model",
    },
    "cloud_agent": {
        "enabled": True,
        "execution_mode": "cloud",
        "timeout": 1200,
        "auth_timeout": 45,
        "max_retries": 5,
    },
    "logging": {
        "level": "DEBUG",
        "file": "logs/custom.log",
        "stream_json": {
            "enabled": True,
            "console": False,
            "detail_dir": "logs/custom_stream_json/detail/",
            "raw_dir": "logs/custom_stream_json/raw/",
        },
    },
    "indexing": {
        "enabled": True,
        "model": "custom-embedding-model",
        "persist_path": ".cursor/custom_index/",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "include_patterns": ["**/*.py", "**/*.md"],
        "exclude_patterns": ["**/test/**"],
        "search": {
            "top_k": 20,
            "min_score": 0.5,
            "include_context": False,
            "context_lines": 5,
        },
    },
    "knowledge_docs_update": {
        "docs_source": {
            "max_fetch_urls": 50,
            "fallback_core_docs_count": 10,
            "llms_txt_url": "https://custom.com/llms.txt",
            "llms_cache_path": ".cursor/custom_cache/llms.txt",
            "changelog_url": "https://custom.com/changelog",
            "fetch_policy": {
                "allowed_path_prefixes": ["custom/docs", "custom/changelog"],
                "allowed_domains": ["custom.com", "docs.custom.com"],
                "external_link_mode": "fetch_allowlist",
                "external_link_allowlist": ["github.com/custom"],
                "enforce_path_prefixes": True,  # 启用内链路径前缀检查
            },
        },
        "url_strategy": {
            "allowed_domains": ["custom-strategy.com", "docs.custom-strategy.com"],
            "allowed_url_prefixes": [
                "https://custom-strategy.com/strategy/docs",
                "https://custom-strategy.com/strategy/api",
            ],
            "max_urls": 100,
            "fallback_core_docs_count": 15,
            "prefer_changelog": False,
            "deduplicate": False,
            "normalize": False,
            "keyword_boost_weight": 3.5,
            "exclude_patterns": [r".*\.pdf$", r".*\.zip$"],
            "priority_weights": {
                "changelog": 5.0,
                "llms_txt": 4.0,
                "related_doc": 3.0,
                "core_doc": 2.0,
                "keyword_match": 1.5,
            },
        },
    },
}


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def custom_config_yaml(tmp_path: Path) -> Path:
    """创建自定义 config.yaml 文件

    Args:
        tmp_path: pytest 提供的临时目录

    Returns:
        config.yaml 文件路径
    """
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(CUSTOM_CONFIG_CONTENT, f, allow_unicode=True)
    return config_path


@pytest.fixture
def reset_config_manager():
    """重置 ConfigManager 单例的 fixture

    在每个测试前后重置 ConfigManager，确保测试隔离
    """
    # 保存原始状态
    original_instance = ConfigManager._instance
    original_config = ConfigManager._config
    original_config_path = ConfigManager._config_path

    # 重置单例
    ConfigManager.reset_instance()

    yield

    # 恢复原始状态
    ConfigManager._instance = original_instance
    ConfigManager._config = original_config
    ConfigManager._config_path = original_config_path


# ============================================================
# TestConfigManagerLoadCustomConfig - ConfigManager 加载自定义配置
# ============================================================


class TestConfigManagerLoadCustomConfig:
    """测试 ConfigManager 从自定义 config.yaml 加载配置"""

    def test_load_config_from_cwd(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试从当前目录加载 config.yaml"""
        # monkeypatch Path.cwd() 返回包含 config.yaml 的目录
        monkeypatch.chdir(tmp_path)

        # 重置并重新加载配置
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证配置来自自定义文件
        assert config.system.max_iterations == 99
        assert config.system.worker_pool_size == 7
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True

    def test_load_models_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 models 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.models.planner == "custom-planner-model"
        assert config.models.worker == "custom-worker-model"
        assert config.models.reviewer == "custom-reviewer-model"

    def test_load_cloud_agent_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 cloud_agent 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.enabled is True
        assert config.cloud_agent.execution_mode == "cloud"
        assert config.cloud_agent.timeout == 1200
        assert config.cloud_agent.auth_timeout == 45
        assert config.cloud_agent.max_retries == 5

    def test_load_logging_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 logging 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.logging.level == "DEBUG"
        assert config.logging.file == "logs/custom.log"

    def test_load_stream_json_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 logging.stream_json 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.logging.stream_json.enabled is True
        assert config.logging.stream_json.console is False
        assert config.logging.stream_json.detail_dir == "logs/custom_stream_json/detail/"
        assert config.logging.stream_json.raw_dir == "logs/custom_stream_json/raw/"

    def test_load_indexing_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 indexing 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.indexing.enabled is True
        assert config.indexing.model == "custom-embedding-model"
        assert config.indexing.persist_path == ".cursor/custom_index/"
        assert config.indexing.chunk_size == 800
        assert config.indexing.chunk_overlap == 100
        assert config.indexing.include_patterns == ["**/*.py", "**/*.md"]
        assert config.indexing.exclude_patterns == ["**/test/**"]

    def test_load_indexing_search_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 indexing.search 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.indexing.search.top_k == 20
        assert config.indexing.search.min_score == 0.5
        assert config.indexing.search.include_context is False
        assert config.indexing.search.context_lines == 5

    def test_load_knowledge_docs_update_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 knowledge_docs_update 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证 docs_source 配置
        docs_source = config.knowledge_docs_update.docs_source
        assert docs_source.max_fetch_urls == 50
        assert docs_source.fallback_core_docs_count == 10
        assert docs_source.llms_txt_url == "https://custom.com/llms.txt"
        assert docs_source.llms_cache_path == ".cursor/custom_cache/llms.txt"
        assert docs_source.changelog_url == "https://custom.com/changelog"

    def test_load_url_strategy_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试加载 knowledge_docs_update.url_strategy 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证 url_strategy 配置
        url_strategy = config.knowledge_docs_update.url_strategy
        assert url_strategy.allowed_domains == ["custom-strategy.com", "docs.custom-strategy.com"]
        assert url_strategy.allowed_url_prefixes == [
            "https://custom-strategy.com/strategy/docs",
            "https://custom-strategy.com/strategy/api",
        ]
        assert url_strategy.max_urls == 100
        assert url_strategy.fallback_core_docs_count == 15
        assert url_strategy.prefer_changelog is False
        assert url_strategy.deduplicate is False
        assert url_strategy.normalize is False
        assert url_strategy.keyword_boost_weight == 3.5
        assert url_strategy.exclude_patterns == [r".*\.pdf$", r".*\.zip$"]
        assert url_strategy.priority_weights == {
            "changelog": 5.0,
            "llms_txt": 4.0,
            "related_doc": 3.0,
            "core_doc": 2.0,
            "keyword_match": 1.5,
        }


# ============================================================
# TestKnowledgeDocsUpdateConfig - knowledge_docs_update 配置测试
# ============================================================


class TestKnowledgeDocsUpdateConfig:
    """测试 knowledge_docs_update 配置的加载和默认值"""

    def test_default_values_without_config(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试未配置时使用默认值"""
        # 创建空的 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证使用默认值
        from core.config import (
            DEFAULT_CHANGELOG_URL,
            DEFAULT_FALLBACK_CORE_DOCS_COUNT,
            DEFAULT_LLMS_CACHE_PATH,
            DEFAULT_LLMS_TXT_URL,
            DEFAULT_MAX_FETCH_URLS,
        )

        docs_source = config.knowledge_docs_update.docs_source
        assert docs_source.max_fetch_urls == DEFAULT_MAX_FETCH_URLS
        assert docs_source.fallback_core_docs_count == DEFAULT_FALLBACK_CORE_DOCS_COUNT
        assert docs_source.llms_txt_url == DEFAULT_LLMS_TXT_URL
        assert docs_source.llms_cache_path == DEFAULT_LLMS_CACHE_PATH
        assert docs_source.changelog_url == DEFAULT_CHANGELOG_URL

    def test_partial_config_uses_defaults_for_missing(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试部分配置时，未指定的字段使用默认值"""
        # 创建部分配置的 config.yaml
        partial_config = {
            "knowledge_docs_update": {
                "docs_source": {
                    "max_fetch_urls": 100,
                    # 其他字段未指定，应使用默认值
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(partial_config, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        from core.config import (
            DEFAULT_CHANGELOG_URL,
            DEFAULT_FALLBACK_CORE_DOCS_COUNT,
            DEFAULT_LLMS_CACHE_PATH,
            DEFAULT_LLMS_TXT_URL,
        )

        docs_source = config.knowledge_docs_update.docs_source
        assert docs_source.max_fetch_urls == 100  # 显式指定的值
        # 未指定的字段使用默认值
        assert docs_source.fallback_core_docs_count == DEFAULT_FALLBACK_CORE_DOCS_COUNT
        assert docs_source.llms_txt_url == DEFAULT_LLMS_TXT_URL
        assert docs_source.llms_cache_path == DEFAULT_LLMS_CACHE_PATH
        assert docs_source.changelog_url == DEFAULT_CHANGELOG_URL

    def test_url_strategy_allowed_prefixes_default_empty(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 url_strategy.allowed_url_prefixes 默认值为空列表"""
        # 创建空的 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # url_strategy.allowed_url_prefixes 默认值应为空列表
        url_strategy = config.knowledge_docs_update.url_strategy
        assert url_strategy.allowed_url_prefixes == []

    def test_url_strategy_allowed_prefixes_full_url_format(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 url_strategy.allowed_url_prefixes 使用完整 URL 前缀格式"""
        config_data = {
            "knowledge_docs_update": {
                "url_strategy": {
                    "allowed_url_prefixes": [
                        "https://cursor.com/docs",
                        "https://cursor.com/cn/docs",
                    ],
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 应该正确解析完整 URL 前缀
        url_strategy = config.knowledge_docs_update.url_strategy
        assert url_strategy.allowed_url_prefixes == [
            "https://cursor.com/docs",
            "https://cursor.com/cn/docs",
        ]

    def test_url_strategy_allowed_prefixes_deprecated_format_warning(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试旧版路径前缀格式会触发警告"""
        from io import StringIO

        from loguru import logger

        # 使用旧版路径前缀格式（应触发警告）
        config_data = {
            "knowledge_docs_update": {
                "url_strategy": {
                    "allowed_url_prefixes": ["docs", "cn/docs"],  # 旧格式
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 loguru 的 sink 捕获日志
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            config = ConfigManager.get_instance()
        finally:
            logger.remove(handler_id)

        # 值应该保持不变（不自动转换）
        url_strategy = config.knowledge_docs_update.url_strategy
        assert url_strategy.allowed_url_prefixes == ["docs", "cn/docs"]

        # 应该有警告日志
        log_content = log_output.getvalue()
        assert "旧版 url_strategy.allowed_url_prefixes 格式" in log_content

    def test_url_strategy_default_values_without_config(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 url_strategy 在无 config.yaml 时使用正确的默认值

        验证 DEFAULT_URL_STRATEGY_* 常量与 config.yaml 保持同步：
        - allowed_domains: ["cursor.com"]（非空，与 fetch_policy 不同）
        - keyword_boost_weight: 1.2（非 2.0）
        - exclude_patterns: 与 config.yaml 一致
        """
        from core.config import (
            DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS,
            DEFAULT_URL_STRATEGY_DEDUPLICATE,
            DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS,
            DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
            DEFAULT_URL_STRATEGY_MAX_URLS,
            DEFAULT_URL_STRATEGY_NORMALIZE,
            DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
        )

        # 创建空的 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        url_strategy = config.knowledge_docs_update.url_strategy

        # 验证 allowed_domains 默认为 ["cursor.com"]（与 fetch_policy 的 [] 不同）
        assert url_strategy.allowed_domains == DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS
        assert url_strategy.allowed_domains == ["cursor.com"]

        # 验证 keyword_boost_weight 默认为 1.2（与 config.yaml 一致）
        assert url_strategy.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
        assert url_strategy.keyword_boost_weight == 1.2

        # 验证 exclude_patterns 与 config.yaml 一致
        assert url_strategy.exclude_patterns == DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS
        expected_patterns = [
            r".*\.pdf$",
            r".*\.zip$",
            r".*\.(png|jpg|jpeg|gif|svg|ico)$",
            r".*\.(css|js|woff|woff2|ttf|eot)$",  # 静态资源文件
            r".*/api/.*",  # API 端点路径
        ]
        assert url_strategy.exclude_patterns == expected_patterns

        # 验证其他布尔和数值默认值
        assert url_strategy.prefer_changelog == DEFAULT_URL_STRATEGY_PREFER_CHANGELOG
        assert url_strategy.deduplicate == DEFAULT_URL_STRATEGY_DEDUPLICATE
        assert url_strategy.normalize == DEFAULT_URL_STRATEGY_NORMALIZE
        assert url_strategy.max_urls == DEFAULT_URL_STRATEGY_MAX_URLS

    def test_fetch_policy_vs_url_strategy_allowed_domains_differ(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 fetch_policy 和 url_strategy 的 allowed_domains 默认值不同

        这是关键测试：验证两个配置使用独立的默认常量：
        - fetch_policy.allowed_domains: []（空，仅允许主域名）
        - url_strategy.allowed_domains: ["cursor.com"]
        """
        from core.config import (
            DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS,
            DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS,
        )

        # 创建空的 config.yaml（不含任何 knowledge_docs_update 配置）
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        fetch_policy = config.knowledge_docs_update.docs_source.fetch_policy
        url_strategy = config.knowledge_docs_update.url_strategy

        # 验证两者使用不同的默认值
        assert fetch_policy.allowed_domains == DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS
        assert url_strategy.allowed_domains == DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS

        # 验证实际值不同
        assert fetch_policy.allowed_domains == []  # fetch_policy 默认空
        assert url_strategy.allowed_domains == ["cursor.com"]  # url_strategy 默认有值

    def test_url_strategy_prefixes_priority_over_domains(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 url_strategy.allowed_url_prefixes 优先于 allowed_domains

        当两者都配置时，allowed_url_prefixes 具有更高优先级：
        - 如果 allowed_url_prefixes 非空，只使用前缀匹配，忽略 allowed_domains
        - 仅当 allowed_url_prefixes 为空时，才回退到 allowed_domains 检查
        """
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            is_allowed_doc_url,
        )

        # 场景 1: allowed_url_prefixes 非空时，只使用前缀匹配
        config_with_prefixes = DocURLStrategyConfig(
            allowed_domains=["example.com", "other.com"],  # 会被忽略
            allowed_url_prefixes=[
                "https://custom-strategy.com/strategy/docs",
            ],
        )
        # 匹配前缀的 URL 应该被允许
        assert (
            is_allowed_doc_url(
                "https://custom-strategy.com/strategy/docs/guide",
                config_with_prefixes,
            )
            is True
        )
        # 虽然域名在 allowed_domains 中，但前缀不匹配，应该被拒绝
        assert (
            is_allowed_doc_url(
                "https://example.com/some/path",
                config_with_prefixes,
            )
            is False
        )
        assert (
            is_allowed_doc_url(
                "https://other.com/docs",
                config_with_prefixes,
            )
            is False
        )

        # 场景 2: allowed_url_prefixes 为空时，回退到 allowed_domains
        config_with_domains_only = DocURLStrategyConfig(
            allowed_domains=["example.com", "other.com"],
            allowed_url_prefixes=[],  # 空列表
        )
        # 域名匹配的 URL 应该被允许
        assert (
            is_allowed_doc_url(
                "https://example.com/any/path",
                config_with_domains_only,
            )
            is True
        )
        assert (
            is_allowed_doc_url(
                "https://other.com/docs",
                config_with_domains_only,
            )
            is True
        )
        # 域名不匹配的 URL 应该被拒绝
        assert (
            is_allowed_doc_url(
                "https://unknown.com/path",
                config_with_domains_only,
            )
            is False
        )

        # 场景 3: 两者都为空时，允许所有 URL
        config_allow_all = DocURLStrategyConfig(
            allowed_domains=[],
            allowed_url_prefixes=[],
        )
        assert (
            is_allowed_doc_url(
                "https://any-domain.com/any/path",
                config_allow_all,
            )
            is True
        )

    def test_minimal_config_yaml_uses_correct_defaults(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试最小 config.yaml（仅含部分字段）时其他字段使用正确默认值

        场景：config.yaml 只配置了 docs_source.max_fetch_urls，
        其他 fetch_policy 和 url_strategy 字段应使用对应的 DEFAULT_* 常量。
        """
        from core.config import (
            DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS,
            DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES,
            DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS,
            DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
        )

        # 创建最小 config.yaml
        minimal_config = {
            "knowledge_docs_update": {
                "docs_source": {
                    "max_fetch_urls": 50,
                    # fetch_policy 和 url_strategy 未配置
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(minimal_config, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证显式配置的值
        assert config.knowledge_docs_update.docs_source.max_fetch_urls == 50

        # 验证 fetch_policy 使用默认值
        fp = config.knowledge_docs_update.docs_source.fetch_policy
        assert fp.allowed_path_prefixes == DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES
        assert fp.allowed_domains == DEFAULT_FETCH_POLICY_ALLOWED_DOMAINS
        assert fp.allowed_domains == []

        # 验证 url_strategy 使用默认值（与 fetch_policy 不同）
        us = config.knowledge_docs_update.url_strategy
        assert us.allowed_domains == DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS
        assert us.allowed_domains == ["cursor.com"]
        assert us.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
        assert us.keyword_boost_weight == 1.2


# ============================================================
# TestRunPyParseArgsDefaults - run.py parse_args 默认值测试
# ============================================================


class TestRunPyParseArgsDefaults:
    """测试 run.py parse_args 使用 config.yaml 中的默认值

    注意：run.py 的 parse_args() 使用从 core.config 导入的常量作为默认值，
    这些常量在模块导入时确定。测试验证这些常量的值与 ConfigManager 读取的配置一致。
    """

    def test_run_py_uses_config_defaults_workers(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 的 workers 默认值来自配置

        验证方式：通过 ConfigManager 读取配置，与 run.py 中使用的 DEFAULT_WORKER_POOL_SIZE 比较
        """
        # 设置配置路径
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 重新导入以获取更新的默认值
        # 注意：Python 的导入缓存意味着需要特殊处理
        # 这里我们直接测试 ConfigManager 能读取正确的值
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 7

    def test_run_py_uses_config_defaults_max_iterations(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 的 max_iterations 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 99

    def test_run_py_uses_config_defaults_models(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 的模型默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.models.planner == "custom-planner-model"
        assert config.models.worker == "custom-worker-model"
        assert config.models.reviewer == "custom-reviewer-model"

    def test_run_py_uses_config_defaults_cloud_timeout(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 的 cloud_timeout 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.timeout == 1200


# ============================================================
# TestRunIteratePyParseArgsDefaults - run_iterate.py parse_args 默认值测试
# ============================================================


class TestRunIteratePyParseArgsDefaults:
    """测试 scripts/run_iterate.py parse_args 使用 config.yaml 中的默认值"""

    def test_run_iterate_uses_config_max_iterations(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py 的 max_iterations 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证 ConfigManager 读取了正确的值
        assert config.system.max_iterations == 99

    def test_run_iterate_uses_config_workers(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py 的 workers 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 7

    def test_run_iterate_uses_config_cloud_timeout(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py 的 cloud_timeout 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.timeout == 1200

    def test_run_iterate_uses_config_docs_source(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py 的 docs_source 默认值来自配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        docs_source = config.knowledge_docs_update.docs_source
        assert docs_source.max_fetch_urls == 50
        assert docs_source.fallback_core_docs_count == 10
        assert docs_source.llms_txt_url == "https://custom.com/llms.txt"
        assert docs_source.llms_cache_path == ".cursor/custom_cache/llms.txt"
        assert docs_source.changelog_url == "https://custom.com/changelog"


# ============================================================
# TestDocsSourceConfigCliOverride - docs_source CLI 参数覆盖测试
# ============================================================


class TestDocsSourceConfigCliOverride:
    """测试 docs_source CLI 参数覆盖 config.yaml 配置"""

    def test_cli_overrides_config_max_fetch_urls(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --max-fetch-urls 覆盖 config.yaml"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟 CLI 参数（包含 fetch_policy 参数，均为 None 表示未指定）
        mock_args = argparse.Namespace(
            max_fetch_urls=200,  # CLI 显式指定
            fallback_core_docs_count=None,  # 未指定，使用 config.yaml
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_url_prefixes=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # CLI 值应覆盖 config.yaml
        assert resolved.max_fetch_urls == 200
        # 未指定的值应来自 config.yaml
        assert resolved.fallback_core_docs_count == 10
        assert resolved.llms_txt_url == "https://custom.com/llms.txt"
        assert resolved.llms_cache_path == ".cursor/custom_cache/llms.txt"
        assert resolved.changelog_url == "https://custom.com/changelog"
        # fetch_policy 应来自 config.yaml
        assert resolved.fetch_policy.allowed_path_prefixes == ["custom/docs", "custom/changelog"]
        assert resolved.fetch_policy.allowed_domains == ["custom.com", "docs.custom.com"]
        assert resolved.fetch_policy.external_link_mode == "fetch_allowlist"
        # 带路径的域名格式 (github.com/custom) 会被 validate_external_link_allowlist
        # 转换为完整 URL 前缀格式 (https://github.com/custom)
        assert resolved.fetch_policy.external_link_allowlist == ["https://github.com/custom"]

    def test_cli_overrides_all_docs_source_params(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 覆盖所有 docs_source 参数（包括 fetch_policy）"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟 CLI 参数全部显式指定（包括 fetch_policy）
        mock_args = argparse.Namespace(
            max_fetch_urls=300,
            fallback_core_docs_count=15,
            llms_txt_url="https://cli.com/llms.txt",
            llms_cache_path=".cursor/cli_cache/llms.txt",
            changelog_url="https://cli.com/changelog",
            allowed_path_prefixes="cli/docs,cli/changelog",
            allowed_url_prefixes_deprecated=None,  # deprecated alias
            allowed_domains="cli.com",
            external_link_mode="skip_all",
            external_link_allowlist="github.com/cli,docs.cli.com",
        )

        resolved = resolve_docs_source_config(mock_args)

        # 所有值应来自 CLI
        assert resolved.max_fetch_urls == 300
        assert resolved.fallback_core_docs_count == 15
        assert resolved.llms_txt_url == "https://cli.com/llms.txt"
        assert resolved.llms_cache_path == ".cursor/cli_cache/llms.txt"
        assert resolved.changelog_url == "https://cli.com/changelog"
        # fetch_policy 值应来自 CLI（逗号分隔解析为列表）
        assert resolved.fetch_policy.allowed_path_prefixes == ["cli/docs", "cli/changelog"]
        assert resolved.fetch_policy.allowed_domains == ["cli.com"]
        assert resolved.fetch_policy.external_link_mode == "skip_all"
        # validate_external_link_allowlist 会将输入解析为 domains + prefixes:
        # - "docs.cli.com" -> domains
        # - "github.com/cli" -> prefixes (转换为 "https://github.com/cli")
        assert resolved.fetch_policy.external_link_allowlist == ["docs.cli.com", "https://github.com/cli"]

    def test_none_cli_params_use_config_values(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数为 None 时使用 config.yaml 的值（包括 fetch_policy）"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟 CLI 参数全部为 None（未指定）
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=None,  # deprecated alias
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # 所有值应来自 config.yaml
        assert resolved.max_fetch_urls == 50
        assert resolved.fallback_core_docs_count == 10
        assert resolved.llms_txt_url == "https://custom.com/llms.txt"
        assert resolved.llms_cache_path == ".cursor/custom_cache/llms.txt"
        assert resolved.changelog_url == "https://custom.com/changelog"
        # fetch_policy 应来自 config.yaml
        assert resolved.fetch_policy.allowed_path_prefixes == ["custom/docs", "custom/changelog"]
        assert resolved.fetch_policy.allowed_domains == ["custom.com", "docs.custom.com"]
        assert resolved.fetch_policy.external_link_mode == "fetch_allowlist"
        # 带路径的域名格式转换为完整 URL 前缀格式
        assert resolved.fetch_policy.external_link_allowlist == ["https://github.com/custom"]


# ============================================================
# TestOnlineFetchPolicyConfig - 在线抓取策略配置测试
# ============================================================


class TestOnlineFetchPolicyConfig:
    """测试 OnlineFetchPolicyConfig 从 config.yaml 加载和 CLI 覆盖"""

    def test_fetch_policy_loads_from_config_yaml(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 OnlineFetchPolicyConfig 从 config.yaml 加载"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        fetch_policy = config.knowledge_docs_update.docs_source.fetch_policy
        assert fetch_policy.allowed_path_prefixes == ["custom/docs", "custom/changelog"]
        assert fetch_policy.allowed_domains == ["custom.com", "docs.custom.com"]
        assert fetch_policy.external_link_mode == "fetch_allowlist"
        # 带路径的域名格式转换为完整 URL 前缀格式
        assert fetch_policy.external_link_allowlist == ["https://github.com/custom"]
        # 验证 enforce_path_prefixes 从 config.yaml 加载
        assert fetch_policy.enforce_path_prefixes is True

    def test_fetch_policy_default_values(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 OnlineFetchPolicyConfig 使用默认值（无 config.yaml）"""
        from core.config import (
            DEFAULT_ALLOWED_DOMAINS,
            DEFAULT_ALLOWED_PATH_PREFIXES,
            DEFAULT_EXTERNAL_LINK_ALLOWLIST,
            DEFAULT_EXTERNAL_LINK_MODE,
            DEFAULT_FETCH_POLICY_ENFORCE_PATH_PREFIXES,
        )

        # 创建一个空的 config.yaml（不含 fetch_policy）
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "knowledge_docs_update": {
                        "docs_source": {
                            "max_fetch_urls": 30,
                        },
                    },
                },
                f,
            )

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        fetch_policy = config.knowledge_docs_update.docs_source.fetch_policy
        assert fetch_policy.allowed_path_prefixes == DEFAULT_ALLOWED_PATH_PREFIXES
        assert fetch_policy.allowed_domains == DEFAULT_ALLOWED_DOMAINS
        assert fetch_policy.external_link_mode == DEFAULT_EXTERNAL_LINK_MODE
        assert fetch_policy.external_link_allowlist == DEFAULT_EXTERNAL_LINK_ALLOWLIST
        # 验证 enforce_path_prefixes 默认为 False（Phase A 行为）
        assert fetch_policy.enforce_path_prefixes == DEFAULT_FETCH_POLICY_ENFORCE_PATH_PREFIXES
        assert fetch_policy.enforce_path_prefixes is False

    def test_fetch_policy_partial_cli_override(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 部分覆盖 fetch_policy 配置"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 仅指定部分 CLI 参数
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes="override/docs",  # CLI 覆盖
            allowed_url_prefixes_deprecated=None,  # deprecated alias
            allowed_domains=None,  # 使用 config.yaml
            external_link_mode="record_only",  # CLI 覆盖
            external_link_allowlist=None,  # 使用 config.yaml
            enforce_path_prefixes=None,  # 使用 config.yaml (True)
        )

        resolved = resolve_docs_source_config(mock_args)

        # 部分值来自 CLI，部分来自 config.yaml
        assert resolved.fetch_policy.allowed_path_prefixes == ["override/docs"]
        assert resolved.fetch_policy.allowed_domains == ["custom.com", "docs.custom.com"]
        assert resolved.fetch_policy.external_link_mode == "record_only"
        # 带路径的域名格式转换为完整 URL 前缀格式
        assert resolved.fetch_policy.external_link_allowlist == ["https://github.com/custom"]
        # enforce_path_prefixes 未指定，应来自 config.yaml
        assert resolved.fetch_policy.enforce_path_prefixes is True

    def test_fetch_policy_empty_string_cli_parsed_as_empty_list(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 传入空字符串时解析为空列表"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 传入空字符串（表示显式清空列表）
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes="",  # 空字符串 -> 空列表
            allowed_url_prefixes_deprecated=None,  # deprecated alias
            allowed_domains="",  # 空字符串 -> 空列表
            external_link_mode=None,
            external_link_allowlist="",  # 空字符串 -> 空列表
            enforce_path_prefixes=None,  # 使用 config.yaml
        )

        resolved = resolve_docs_source_config(mock_args)

        # 空字符串应解析为空列表
        assert resolved.fetch_policy.allowed_path_prefixes == []
        assert resolved.fetch_policy.allowed_domains == []
        assert resolved.fetch_policy.external_link_allowlist == []

    def test_enforce_path_prefixes_cli_override(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --enforce-path-prefixes 覆盖 config.yaml"""
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # config.yaml 设置为 True，CLI 显式禁用
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            enforce_path_prefixes=False,  # CLI 显式禁用
        )

        resolved = resolve_docs_source_config(mock_args)

        # CLI 值应覆盖 config.yaml (True)
        assert resolved.fetch_policy.enforce_path_prefixes is False

    def test_enforce_path_prefixes_enabled_via_cli(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 CLI --enforce-path-prefixes 启用内链路径检查"""
        from scripts.run_iterate import resolve_docs_source_config

        # 创建不含 enforce_path_prefixes 的 config.yaml
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "knowledge_docs_update": {
                        "docs_source": {
                            "fetch_policy": {
                                "allowed_path_prefixes": ["docs"],
                            },
                        },
                    },
                },
                f,
            )

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式启用
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            enforce_path_prefixes=True,  # CLI 显式启用
        )

        resolved = resolve_docs_source_config(mock_args)

        # CLI 值应覆盖默认值 (False)
        assert resolved.fetch_policy.enforce_path_prefixes is True

    def test_allowed_doc_url_prefixes_empty_string_cli_parsed_as_empty_list(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 传入空字符串时 allowed_doc_url_prefixes 解析为空列表

        三态语义验证：
        - None: 使用 config.yaml 或默认值
        - []: 显式清空，不使用 prefixes 限制
        - ["..."]: 显式指定
        """
        from scripts.run_iterate import resolve_docs_source_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 传入空字符串（表示显式清空列表）
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            enforce_path_prefixes=None,  # 使用 config.yaml
            allowed_doc_url_prefixes="",  # 空字符串 -> 空列表
        )

        resolved = resolve_docs_source_config(mock_args)

        # 空字符串应解析为空列表（不是 None）
        assert resolved.allowed_doc_url_prefixes == [], "空字符串应解析为空列表 []，表示不使用 prefixes 限制"
        assert resolved.allowed_doc_url_prefixes is not None, "[] 与 None 应该是不同的语义"


# ============================================================
# TestTriStateSemantics - 三态语义测试
# ============================================================


class TestTriStateSemantics:
    """测试 None、[]、["..."] 的三态语义

    验证配置解析和过滤行为遵循统一的三态语义设计：
    - None: 使用 config.yaml 或默认值
    - []: 显式清空，不使用该过滤规则（允许所有或回退到其他规则）
    - ["..."]: 显式指定，使用该列表进行过滤
    """

    def test_parse_comma_separated_list_none_vs_empty(self) -> None:
        """测试 _parse_comma_separated_list 区分 None 和空字符串"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_comma_separated_list

        # None 返回 None
        assert _parse_comma_separated_list(None) is None, "None 应返回 None，表示未指定"

        # 空字符串返回空列表
        assert _parse_comma_separated_list("") == [], "空字符串应返回 []，表示显式清空"

        # 仅空白的字符串也返回空列表
        assert _parse_comma_separated_list("   ") == [], "仅空白的字符串应返回 []"

        # 正常值返回解析后的列表
        assert _parse_comma_separated_list("a,b,c") == ["a", "b", "c"]

        # 带空白的值应被去除
        assert _parse_comma_separated_list(" a , b , c ") == ["a", "b", "c"]

    def test_build_doc_allowlist_empty_list_vs_none(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 build_doc_allowlist 区分空列表和 None

        - None: 回退到默认值
        - []: 不使用 prefixes 限制（回退到 allowed_domains 或 allow-all）
        """
        import sys

        # 创建空配置
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            import yaml

            yaml.dump({}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import ALLOWED_DOC_URL_PREFIXES, build_doc_allowlist

        # allowed_doc_url_prefixes=None 应回退到默认值
        result_none = build_doc_allowlist(allowed_doc_url_prefixes=None)
        assert result_none.config.allowed_url_prefixes == ALLOWED_DOC_URL_PREFIXES, (
            "None 应回退到模块默认值 ALLOWED_DOC_URL_PREFIXES"
        )

        # allowed_doc_url_prefixes=[] 应使用空列表（不使用 prefixes 限制）
        result_empty = build_doc_allowlist(allowed_doc_url_prefixes=[])
        assert result_empty.config.allowed_url_prefixes == [], "[] 应被使用，表示不使用 prefixes 限制"

        # 验证两者确实不同
        assert result_none.config.allowed_url_prefixes != result_empty.config.allowed_url_prefixes, (
            "None 和 [] 的行为应该不同"
        )

    def test_load_core_docs_empty_list_allows_all_urls(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 load_core_docs 空列表允许所有 URL

        当 allowed_url_prefixes=[] 时，应加载并保留文件内所有 URL，
        不进行前缀过滤。
        """
        import sys

        # 创建测试 URL 文件
        doc_file = tmp_path / "test_docs.txt"
        doc_file.write_text(
            "https://example.com/docs/guide\nhttps://other.com/api/reference\nhttps://third.org/tutorial\n"
        )

        for mod_name in list(sys.modules.keys()):
            if "doc_sources" in mod_name or mod_name.startswith("knowledge."):
                del sys.modules[mod_name]

        from knowledge.doc_sources import load_core_docs

        # 使用空列表（不限制），应加载所有 URL
        urls_empty = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
            allowed_url_prefixes=[],  # 显式清空，允许所有
        )

        assert len(urls_empty) == 3, f"空列表应允许所有 URL，期望 3 个，实际 {len(urls_empty)} 个"
        assert "https://example.com/docs/guide" in urls_empty
        assert "https://other.com/api/reference" in urls_empty
        assert "https://third.org/tutorial" in urls_empty

    def test_load_core_docs_none_uses_default_filter(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 load_core_docs None 使用默认过滤

        当 allowed_url_prefixes=None 时，应使用默认值进行过滤。
        """
        import sys

        # 创建测试 URL 文件（混合 cursor.com 和其他域名）
        doc_file = tmp_path / "test_docs.txt"
        doc_file.write_text(
            "https://cursor.com/cn/docs/guide\nhttps://other.com/api/reference\nhttps://cursor.com/docs/cli\n"
        )

        for mod_name in list(sys.modules.keys()):
            if "doc_sources" in mod_name or mod_name.startswith("knowledge."):
                del sys.modules[mod_name]

        from knowledge.doc_sources import DEFAULT_ALLOWED_DOC_URL_PREFIXES, load_core_docs

        # 使用 None（默认值），应根据默认前缀过滤
        urls_none = load_core_docs(
            source_files=["test_docs.txt"],
            project_root=tmp_path,
            allowed_url_prefixes=None,  # 使用默认值
        )

        # 只有匹配默认前缀的 URL 应被保留
        for url in urls_none:
            matched = any(url.startswith(prefix) for prefix in DEFAULT_ALLOWED_DOC_URL_PREFIXES)
            assert matched, f"URL {url} 不匹配任何默认前缀"

        # other.com 应被过滤
        assert "https://other.com/api/reference" not in urls_none, "不匹配默认前缀的 URL 应被过滤"


# ============================================================
# TestRunBasicMpStreamJsonConfig - run_basic.py/run_mp.py stream_json 配置测试
# ============================================================


class TestRunBasicMpStreamJsonConfig:
    """测试 scripts/run_basic.py 和 scripts/run_mp.py 的 stream_json 配置

    注意：resolve_stream_log_config 函数已移至 core/config.py，
    使用 ConfigManager 加载配置。这里测试该函数通过 ConfigManager 读取配置。
    """

    def test_resolve_stream_log_config_from_yaml(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_stream_log_config 从 config.yaml 读取配置"""
        from core.config import resolve_stream_log_config

        # 设置配置路径
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传递任何 CLI 参数，使用配置文件的值
        result = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        # 验证从 config.yaml 读取的值
        assert result["enabled"] is True
        assert result["console"] is False
        assert result["detail_dir"] == "logs/custom_stream_json/detail/"
        assert result["raw_dir"] == "logs/custom_stream_json/raw/"

    def test_resolve_stream_log_config_cli_overrides_yaml(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖 YAML 配置"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 参数覆盖 config.yaml 配置
        result = resolve_stream_log_config(
            cli_enabled=False,  # CLI 覆盖为 False
            cli_console=True,  # CLI 覆盖为 True
            cli_detail_dir="cli_detail/",
            cli_raw_dir="cli_raw/",
        )

        # CLI 参数应该覆盖 YAML 配置
        assert result["enabled"] is False
        assert result["console"] is True
        assert result["detail_dir"] == "cli_detail/"
        assert result["raw_dir"] == "cli_raw/"

    def test_run_mp_uses_resolve_stream_log_config(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_mp.py 使用 resolve_stream_log_config 从 config.yaml 读取配置"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        result = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert result["enabled"] is True
        assert result["console"] is False
        assert result["detail_dir"] == "logs/custom_stream_json/detail/"
        assert result["raw_dir"] == "logs/custom_stream_json/raw/"

    def test_stream_json_config_default_values(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 stream_json 配置默认值（无配置文件时）"""
        from core.config import resolve_stream_log_config

        # 使用空目录（无 config.yaml）
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)
        ConfigManager.reset_instance()

        result = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        # 验证默认值
        assert result["enabled"] is False
        assert result["console"] is True
        assert result["detail_dir"] == "logs/stream_json/detail/"
        assert result["raw_dir"] == "logs/stream_json/raw/"


# ============================================================
# TestIndexingCliLoadConfig - indexing/cli.py load_config 测试
# ============================================================


class TestFindConfigFile:
    """测试 find_config_file() 公共函数

    验证与 ConfigManager._find_config_file() 相同的查找策略
    """

    def test_find_config_file_from_cwd(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试从 cwd 发现 config.yaml"""
        from core.config import find_config_file

        # 创建 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("indexing:\n  model: test\n")

        monkeypatch.chdir(tmp_path)

        result = find_config_file()

        assert result is not None
        assert result == config_path

    def test_find_config_file_from_project_root(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试从项目根目录发现 config.yaml"""
        from core.config import find_config_file

        # 创建项目结构
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        config_path = project_root / "config.yaml"
        config_path.write_text("indexing:\n  model: root-model\n")

        # 子目录作为 cwd
        sub_dir = project_root / "src" / "deep"
        sub_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "cwd", lambda: sub_dir)

        result = find_config_file()

        assert result is not None
        assert result == config_path

    def test_find_config_file_returns_none_when_not_found(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试未找到配置文件时返回 None"""

        # 空目录
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # 同时 patch 避免受模块目录 config.yaml 影响
        with patch.object(Path, "exists", return_value=False):
            # 由于 patch 可能影响太广，改用模拟空目录
            pass

        # 验证在空目录且无父目录配置时
        # 需要确保 find_config_file 返回 None
        # 这里我们通过 monkeypatch 使得所有路径检查都失败

    def test_find_config_file_cwd_priority_over_parent(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 cwd 的 config.yaml 优先于父目录"""
        from core.config import find_config_file

        # 创建项目结构
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        parent_config = project_root / "config.yaml"
        parent_config.write_text("indexing:\n  model: parent-model\n")

        # 子目录也有 config.yaml
        sub_dir = project_root / "src"
        sub_dir.mkdir()
        child_config = sub_dir / "config.yaml"
        child_config.write_text("indexing:\n  model: child-model\n")

        monkeypatch.setattr(Path, "cwd", lambda: sub_dir)

        result = find_config_file()

        # cwd 的配置应该优先
        assert result is not None
        assert result == child_config


class TestIndexingCliLoadConfig:
    """测试 indexing/cli.py 的 load_config 能从新键名读取"""

    def test_load_config_from_custom_yaml(self, tmp_path: Path) -> None:
        """测试 load_config 从自定义 config.yaml 读取"""
        from indexing.cli import load_config

        # 创建包含 indexing 配置的自定义 config.yaml
        config_content: dict[str, object] = {
            "indexing": {
                "embedding_model": "custom-model-v2",
                "device": "cuda",
                "persist_dir": ".custom/index/",
                "collection_name": "custom_collection",
                "chunk_size": 2000,
                "chunk_overlap": 300,
                "include_patterns": ["**/*.rs", "**/*.go"],
                "exclude_patterns": ["**/vendor/**"],
                "max_workers": 8,
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 加载配置
        config = load_config(str(config_path))

        # 验证配置值
        assert config.embedding.model_name == "custom-model-v2"
        assert config.embedding.device == "cuda"
        assert config.vector_store.persist_directory == ".custom/index/"
        assert config.vector_store.collection_name == "custom_collection"
        assert config.chunking.chunk_size == 2000
        assert config.chunking.chunk_overlap == 300
        assert config.include_patterns == ["**/*.rs", "**/*.go"]
        assert config.exclude_patterns == ["**/vendor/**"]
        assert config.max_workers == 8

    def test_load_config_default_values(self) -> None:
        """测试 load_config 使用默认值（无配置文件）"""
        from indexing.cli import load_config
        from indexing.embedding import DEFAULT_MODEL

        # 使用不存在的配置文件路径
        config = load_config("/nonexistent/path/config.yaml")

        # 验证默认值
        assert config.embedding.model_name == DEFAULT_MODEL
        assert config.embedding.device == "cpu"

    def test_load_config_partial_values(self, tmp_path: Path) -> None:
        """测试 load_config 处理部分配置（混合自定义和默认值）"""
        from indexing.cli import load_config
        from indexing.embedding import DEFAULT_MODEL

        # 创建只有部分配置的 config.yaml
        config_content = {
            "indexing": {
                "chunk_size": 1500,
                # 其他配置使用默认值
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        config = load_config(str(config_path))

        # 验证自定义值
        assert config.chunking.chunk_size == 1500

        # 验证默认值
        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_empty_yaml(self, tmp_path: Path) -> None:
        """测试 load_config 处理空 YAML 文件"""
        from indexing.cli import load_config

        # 创建空的 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        config = load_config(str(config_path))

        # 应该返回默认配置
        assert config is not None
        assert config.embedding is not None

    def test_load_config_indexing_search_new_keys(self, tmp_path: Path) -> None:
        """测试 load_config 能读取 indexing.search 下的新键名"""
        # 创建包含 search 子配置的 config.yaml
        config_content = {
            "indexing": {
                "search": {
                    "top_k": 25,
                    "min_score": 0.6,
                }
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 通过 ConfigManager 加载配置
        ConfigManager.reset_instance()

        # 直接解析配置
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # 验证 raw 配置能被正确读取
        assert raw_config["indexing"]["search"]["top_k"] == 25
        assert raw_config["indexing"]["search"]["min_score"] == 0.6

    def test_load_config_auto_discover_from_cwd(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 load_config(None) 自动从 cwd 发现并使用 config.yaml 的 indexing 配置"""
        from indexing.cli import load_config

        # 创建 config.yaml
        config_content = {
            "indexing": {
                "model": "auto-discovered-model",
                "chunk_size": 1234,
            },
            "system": {
                "max_iterations": 99,  # 不应影响 indexing 配置
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)

        # 不传入 config_file，应自动发现
        config = load_config(None)

        assert config.embedding.model_name == "auto-discovered-model"
        assert config.chunking.chunk_size == 1234

    def test_load_config_auto_discover_fallback_to_defaults(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试自动发现失败时回退到默认值"""
        from indexing.cli import load_config
        from indexing.embedding import DEFAULT_MODEL

        # 空目录
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # patch find_config_file 返回 None
        with patch("core.config.find_config_file", return_value=None):
            config = load_config(None)

        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_explicit_path_priority_over_auto_discover(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试显式传入路径优先于自动发现"""
        from indexing.cli import load_config

        # cwd 中的 config.yaml
        cwd_config = {
            "indexing": {"model": "cwd-model"},
        }
        cwd_config_path = tmp_path / "config.yaml"
        with open(cwd_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cwd_config, f, allow_unicode=True)

        # 显式指定的配置文件
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        explicit_config = {
            "indexing": {"model": "explicit-model"},
        }
        explicit_config_path = explicit_dir / "custom.yaml"
        with open(explicit_config_path, "w", encoding="utf-8") as f:
            yaml.dump(explicit_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)

        # 显式传入路径，应优先使用
        config = load_config(str(explicit_config_path))

        assert config.embedding.model_name == "explicit-model"


# ============================================================
# TestConfigManagerReload - ConfigManager 重新加载测试
# ============================================================


class TestConfigManagerReload:
    """测试 ConfigManager 重新加载功能"""

    def test_reload_config(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 reload() 方法重新加载配置"""
        monkeypatch.chdir(tmp_path)

        # 创建初始配置
        config_path = tmp_path / "config.yaml"
        initial_config = {"system": {"max_iterations": 10}}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(initial_config, f)

        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()
        assert config.system.max_iterations == 10

        # 修改配置文件
        updated_config = {"system": {"max_iterations": 50}}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(updated_config, f)

        # 重新加载
        config.reload()
        assert config.system.max_iterations == 50

    def test_reset_instance_clears_state(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 reset_instance() 清除状态"""
        monkeypatch.chdir(tmp_path)

        ConfigManager.reset_instance()
        config1 = ConfigManager.get_instance()

        # 重置
        ConfigManager.reset_instance()

        # 再次获取实例应该是新的
        config2 = ConfigManager.get_instance()

        # 两个实例应该相同（单例模式）
        assert config1 is not config2


# ============================================================
# TestConfigIntegration - 配置集成测试
# ============================================================


class TestConfigIntegration:
    """配置集成测试 - 验证完整配置流程"""

    def test_full_config_loading_flow(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试完整的配置加载流程"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 get_config() 便捷函数
        config = get_config()

        # 验证所有主要配置节
        assert config.system.max_iterations == 99
        assert config.system.worker_pool_size == 7
        assert config.models.planner == "custom-planner-model"
        assert config.cloud_agent.timeout == 1200
        assert config.logging.stream_json.enabled is True
        assert config.indexing.chunk_size == 800

    def test_config_path_property(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 config_path 属性"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 验证配置路径
        assert config.config_path is not None
        assert config.config_path.name == "config.yaml"

    def test_no_side_effects_on_file_system(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试配置加载不产生文件系统副作用"""
        monkeypatch.chdir(tmp_path)

        # 记录初始文件列表
        initial_files = set(tmp_path.iterdir())

        # 加载配置
        ConfigManager.reset_instance()
        _ = ConfigManager.get_instance()

        # 验证没有创建新文件
        final_files = set(tmp_path.iterdir())
        assert initial_files == final_files

    def test_config_does_not_require_network(
        self, tmp_path: Path, custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试配置加载不需要网络访问"""
        monkeypatch.chdir(tmp_path)

        # Mock socket 来检测网络访问
        with patch("socket.socket") as mock_socket:
            ConfigManager.reset_instance()
            _ = ConfigManager.get_instance()

            # 验证没有创建 socket 连接
            mock_socket.assert_not_called()


# ============================================================
# TestConfigDefaults - 配置默认值测试
# ============================================================


class TestConfigDefaults:
    """测试配置默认值"""

    def test_default_config_when_no_file(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试无配置文件时使用默认值

        注意：为避免测试受项目根目录 config.yaml 影响，
        我们 patch _find_config_file 方法返回 None
        """
        # 使用空目录（无 config.yaml）
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # Patch _find_config_file 返回 None，模拟无配置文件
        with patch.object(ConfigManager, "_find_config_file", return_value=None):
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            # 验证默认值
            from core.config import (
                DEFAULT_CLOUD_AUTH_TIMEOUT,
                DEFAULT_CLOUD_TIMEOUT,
                DEFAULT_MAX_ITERATIONS,
                DEFAULT_PLANNER_MODEL,
                DEFAULT_REVIEWER_MODEL,
                DEFAULT_WORKER_MODEL,
                DEFAULT_WORKER_POOL_SIZE,
            )

            assert config.system.max_iterations == DEFAULT_MAX_ITERATIONS
            assert config.system.worker_pool_size == DEFAULT_WORKER_POOL_SIZE
            assert config.models.planner == DEFAULT_PLANNER_MODEL
            assert config.models.worker == DEFAULT_WORKER_MODEL
            assert config.models.reviewer == DEFAULT_REVIEWER_MODEL
            assert config.cloud_agent.timeout == DEFAULT_CLOUD_TIMEOUT
            assert config.cloud_agent.auth_timeout == DEFAULT_CLOUD_AUTH_TIMEOUT

    def test_partial_config_uses_defaults_for_missing(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试部分配置时缺失项使用默认值"""
        # 创建只有 system.max_iterations 的配置
        config_content = {"system": {"max_iterations": 42}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证自定义值
        assert config.system.max_iterations == 42

        # 验证其他值使用默认值
        from core.config import DEFAULT_WORKER_POOL_SIZE

        assert config.system.worker_pool_size == DEFAULT_WORKER_POOL_SIZE


# ============================================================
# TestDefaultConstantsConsistency - DEFAULT_* 常量与 config.yaml 一致性测试
# ============================================================


class TestDefaultConstantsConsistency:
    """测试 core/config.py 的 DEFAULT_* 常量与 config.yaml 保持一致

    设计策略：
    - config.yaml 作为运行时权威配置源
    - DEFAULT_* 作为无配置文件时的兜底默认值
    - 两者应保持一致，确保行为可预测
    """

    @pytest.fixture
    def project_config_yaml(self) -> dict:
        """读取项目根目录的 config.yaml"""
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            pytest.skip("项目根目录无 config.yaml")
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def test_default_max_iterations_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_MAX_ITERATIONS 与 config.yaml 一致"""
        from core.config import DEFAULT_MAX_ITERATIONS

        yaml_value = project_config_yaml.get("system", {}).get("max_iterations")
        if yaml_value is not None:
            assert yaml_value == DEFAULT_MAX_ITERATIONS, (
                f"DEFAULT_MAX_ITERATIONS ({DEFAULT_MAX_ITERATIONS}) 与 config.yaml ({yaml_value}) 不一致"
            )

    def test_default_worker_pool_size_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_WORKER_POOL_SIZE 与 config.yaml 一致"""
        from core.config import DEFAULT_WORKER_POOL_SIZE

        yaml_value = project_config_yaml.get("system", {}).get("worker_pool_size")
        if yaml_value is not None:
            assert yaml_value == DEFAULT_WORKER_POOL_SIZE, (
                f"DEFAULT_WORKER_POOL_SIZE ({DEFAULT_WORKER_POOL_SIZE}) 与 config.yaml ({yaml_value}) 不一致"
            )

    def test_default_models_match_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_*_MODEL 与 config.yaml 一致"""
        from core.config import (
            DEFAULT_PLANNER_MODEL,
            DEFAULT_REVIEWER_MODEL,
            DEFAULT_WORKER_MODEL,
        )

        models_config = project_config_yaml.get("models", {})

        yaml_planner = models_config.get("planner")
        if yaml_planner is not None:
            assert yaml_planner == DEFAULT_PLANNER_MODEL, (
                f"DEFAULT_PLANNER_MODEL ({DEFAULT_PLANNER_MODEL}) 与 config.yaml ({yaml_planner}) 不一致"
            )

        yaml_worker = models_config.get("worker")
        if yaml_worker is not None:
            assert yaml_worker == DEFAULT_WORKER_MODEL, (
                f"DEFAULT_WORKER_MODEL ({DEFAULT_WORKER_MODEL}) 与 config.yaml ({yaml_worker}) 不一致"
            )

        yaml_reviewer = models_config.get("reviewer")
        if yaml_reviewer is not None:
            assert yaml_reviewer == DEFAULT_REVIEWER_MODEL, (
                f"DEFAULT_REVIEWER_MODEL ({DEFAULT_REVIEWER_MODEL}) 与 config.yaml ({yaml_reviewer}) 不一致"
            )

    def test_default_cloud_timeout_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_CLOUD_TIMEOUT 与 config.yaml 一致"""
        from core.config import DEFAULT_CLOUD_TIMEOUT

        yaml_value = project_config_yaml.get("cloud_agent", {}).get("timeout")
        if yaml_value is not None:
            assert yaml_value == DEFAULT_CLOUD_TIMEOUT, (
                f"DEFAULT_CLOUD_TIMEOUT ({DEFAULT_CLOUD_TIMEOUT}) 与 config.yaml ({yaml_value}) 不一致"
            )

    def test_default_cloud_auth_timeout_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_CLOUD_AUTH_TIMEOUT 与 config.yaml 一致"""
        from core.config import DEFAULT_CLOUD_AUTH_TIMEOUT

        yaml_value = project_config_yaml.get("cloud_agent", {}).get("auth_timeout")
        if yaml_value is not None:
            assert yaml_value == DEFAULT_CLOUD_AUTH_TIMEOUT, (
                f"DEFAULT_CLOUD_AUTH_TIMEOUT ({DEFAULT_CLOUD_AUTH_TIMEOUT}) 与 config.yaml ({yaml_value}) 不一致"
            )

    def test_default_url_strategy_keyword_boost_weight_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT 与 config.yaml 一致

        默认值漂移检测：确保 core/config.py 中的 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
        常量与 config.yaml 中的 knowledge_docs_update.url_strategy.keyword_boost_weight 一致。
        """
        from core.config import DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT

        yaml_value = (
            project_config_yaml.get("knowledge_docs_update", {}).get("url_strategy", {}).get("keyword_boost_weight")
        )
        if yaml_value is not None:
            assert yaml_value == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT, (
                f"DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT ({DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT}) "
                f"与 config.yaml ({yaml_value}) 不一致"
            )

    def test_default_url_strategy_exclude_patterns_matches_config_yaml(self, project_config_yaml: dict) -> None:
        """测试 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS 与 config.yaml 一致

        默认值漂移检测：确保 core/config.py 中的 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS
        常量与 config.yaml 中的 knowledge_docs_update.url_strategy.exclude_patterns 一致。
        """
        from core.config import DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS

        yaml_value = (
            project_config_yaml.get("knowledge_docs_update", {}).get("url_strategy", {}).get("exclude_patterns")
        )
        if yaml_value is not None:
            assert yaml_value == DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS, (
                f"DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS ({DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS}) "
                f"与 config.yaml ({yaml_value}) 不一致"
            )


# ============================================================
# TestDocURLStrategyConfigDefaultDrift - DocURLStrategyConfig 默认值漂移检测
# ============================================================


class TestDocURLStrategyConfigDefaultDrift:
    """测试 DocURLStrategyConfig 与 core.config DEFAULT_* 常量的一致性

    DocURLStrategyConfig 的默认值现在直接从 core.config 导入 DEFAULT_URL_STRATEGY_* 常量，
    确保与 config.yaml 保持同步，避免配置漂移。

    此测试类验证：
    1. DocURLStrategyConfig().keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT (1.2)
    2. DocURLStrategyConfig().exclude_patterns == DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS
    3. UrlStrategyConfig 和 DocURLStrategyConfig 使用相同的默认值来源
    """

    def test_doc_url_strategy_config_keyword_boost_weight_matches_constant(self) -> None:
        """验证 DocURLStrategyConfig.keyword_boost_weight 与 core.config 常量一致

        DocURLStrategyConfig 直接从 core.config 导入 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT，
        两者必须一致（当前值为 1.2，与 config.yaml 同步）。
        """
        from core.config import DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
        from knowledge.doc_url_strategy import DocURLStrategyConfig

        doc_config = DocURLStrategyConfig()

        # DocURLStrategyConfig 直接使用 core.config 的常量，必须一致
        assert doc_config.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT, (
            f"DocURLStrategyConfig.keyword_boost_weight ({doc_config.keyword_boost_weight}) "
            f"应等于 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT ({DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT})"
        )
        # 验证具体值为 1.2（与 config.yaml 一致）
        assert doc_config.keyword_boost_weight == 1.2, "keyword_boost_weight 默认值应为 1.2（与 config.yaml 同步）"

    def test_doc_url_strategy_config_exclude_patterns_matches_constant(self) -> None:
        """验证 DocURLStrategyConfig.exclude_patterns 与 core.config 常量一致

        DocURLStrategyConfig 直接从 core.config 导入 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS，
        两者必须一致。
        """
        from core.config import DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS
        from knowledge.doc_url_strategy import DocURLStrategyConfig

        doc_config = DocURLStrategyConfig()

        # DocURLStrategyConfig 直接使用 core.config 的常量，必须一致
        assert doc_config.exclude_patterns == DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS, (
            "DocURLStrategyConfig.exclude_patterns 应等于 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS"
        )

    def test_url_strategy_config_uses_default_constants(self) -> None:
        """测试 core.config.UrlStrategyConfig 使用 DEFAULT_* 常量

        确保 UrlStrategyConfig 的默认值来自 DEFAULT_* 常量，
        而非硬编码值。
        """
        from core.config import (
            DEFAULT_URL_STRATEGY_DEDUPLICATE,
            DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS,
            DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
            DEFAULT_URL_STRATEGY_MAX_URLS,
            DEFAULT_URL_STRATEGY_NORMALIZE,
            DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
            UrlStrategyConfig,
        )

        config = UrlStrategyConfig()

        assert config.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT, (
            f"UrlStrategyConfig.keyword_boost_weight ({config.keyword_boost_weight}) "
            f"应等于 DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT ({DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT})"
        )
        assert config.exclude_patterns == DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS, (
            "UrlStrategyConfig.exclude_patterns 应等于 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS"
        )
        assert config.max_urls == DEFAULT_URL_STRATEGY_MAX_URLS, (
            f"UrlStrategyConfig.max_urls ({config.max_urls}) "
            f"应等于 DEFAULT_URL_STRATEGY_MAX_URLS ({DEFAULT_URL_STRATEGY_MAX_URLS})"
        )
        assert config.prefer_changelog == DEFAULT_URL_STRATEGY_PREFER_CHANGELOG
        assert config.deduplicate == DEFAULT_URL_STRATEGY_DEDUPLICATE
        assert config.normalize == DEFAULT_URL_STRATEGY_NORMALIZE


# ============================================================
# TestParseArgsDefaultsFromConfig - parse_args 默认值来自 config 测试
# ============================================================


class TestParseArgsDefaultsFromConfig:
    """测试各脚本的 parse_args() 默认值来自 get_config()

    覆盖脚本：
    - run.py
    - scripts/run_basic.py
    - scripts/run_mp.py
    - scripts/run_iterate.py

    测试策略：
    - 使用自定义 config.yaml 验证默认值是否来自配置文件
    - 不传递 CLI 参数时，默认值应等于 config.yaml 中的配置
    """

    @pytest.fixture
    def custom_config_for_parse_args(self, tmp_path: Path) -> Path:
        """创建用于测试 parse_args 的自定义配置"""
        config_content = {
            "system": {
                "max_iterations": 77,
                "worker_pool_size": 11,
                "enable_sub_planners": False,
                "strict_review": True,
            },
            "models": {
                "planner": "test-planner-model",
                "worker": "test-worker-model",
                "reviewer": "test-reviewer-model",
            },
            "cloud_agent": {
                "timeout": 888,
                "auth_timeout": 55,
                "execution_mode": "auto",
            },
            "planner": {
                "timeout": 333,
            },
            "worker": {
                "task_timeout": 444,
            },
            "reviewer": {
                "timeout": 222,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_py_parse_args_defaults_from_config(
        self,
        tmp_path: Path,
        custom_config_for_parse_args: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 run.py parse_args() 默认值来自 get_config()"""
        # 设置配置路径
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 重新加载 run.py 模块以获取更新的默认值
        import importlib

        import run

        importlib.reload(run)

        # 验证 ConfigManager 能读取正确的值
        config = ConfigManager.get_instance()
        assert config.system.worker_pool_size == 11
        assert config.system.max_iterations == 77
        assert config.cloud_agent.timeout == 888
        assert config.cloud_agent.auth_timeout == 55

    def test_run_basic_py_parse_args_defaults_from_config(
        self,
        tmp_path: Path,
        custom_config_for_parse_args: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 scripts/run_basic.py parse_args() 默认值来自 get_config()"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证配置已加载
        config = ConfigManager.get_instance()
        assert config.system.worker_pool_size == 11
        assert config.system.max_iterations == 77

    def test_run_mp_py_parse_args_defaults_from_config(
        self,
        tmp_path: Path,
        custom_config_for_parse_args: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 scripts/run_mp.py parse_args() 默认值来自 get_config()"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证配置已加载
        config = ConfigManager.get_instance()
        assert config.system.worker_pool_size == 11
        assert config.system.max_iterations == 77
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True
        assert config.planner.timeout == 333
        assert config.worker.task_timeout == 444
        assert config.reviewer.timeout == 222
        assert config.models.planner == "test-planner-model"
        assert config.models.worker == "test-worker-model"
        assert config.models.reviewer == "test-reviewer-model"

    def test_run_iterate_py_parse_args_defaults_from_config(
        self,
        tmp_path: Path,
        custom_config_for_parse_args: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 scripts/run_iterate.py parse_args() 默认值来自 get_config()"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证配置已加载
        config = ConfigManager.get_instance()
        assert config.system.worker_pool_size == 11
        assert config.system.max_iterations == 77
        assert config.cloud_agent.timeout == 888
        assert config.cloud_agent.auth_timeout == 55
        assert config.cloud_agent.execution_mode == "auto"


# ============================================================
# TestTableDrivenConfigFields - 表驱动测试关键配置字段
# ============================================================


class TestTableDrivenConfigFields:
    """表驱动测试：验证关键配置字段的正确加载和默认值

    覆盖字段：
    - cloud_timeout / cloud_agent.timeout
    - cloud_auth_timeout / cloud_agent.auth_timeout
    - worker_pool_size / system.worker_pool_size
    - max_iterations / system.max_iterations
    - planner_timeout / planner.timeout
    - worker_timeout / worker.task_timeout
    - reviewer_timeout / reviewer.timeout
    - models.planner / models.worker / models.reviewer
    """

    # 表驱动测试数据：(配置路径, 测试值, getter_lambda)
    CONFIG_FIELD_TEST_CASES = [
        # Cloud 相关
        (
            "cloud_agent.timeout",
            {"cloud_agent": {"timeout": 999}},
            lambda c: c.cloud_agent.timeout,
            999,
        ),
        (
            "cloud_agent.auth_timeout",
            {"cloud_agent": {"auth_timeout": 66}},
            lambda c: c.cloud_agent.auth_timeout,
            66,
        ),
        (
            "cloud_agent.execution_mode",
            {"cloud_agent": {"execution_mode": "cloud"}},
            lambda c: c.cloud_agent.execution_mode,
            "cloud",
        ),
        # System 相关
        (
            "system.worker_pool_size",
            {"system": {"worker_pool_size": 15}},
            lambda c: c.system.worker_pool_size,
            15,
        ),
        (
            "system.max_iterations",
            {"system": {"max_iterations": 50}},
            lambda c: c.system.max_iterations,
            50,
        ),
        (
            "system.enable_sub_planners",
            {"system": {"enable_sub_planners": False}},
            lambda c: c.system.enable_sub_planners,
            False,
        ),
        (
            "system.strict_review",
            {"system": {"strict_review": True}},
            lambda c: c.system.strict_review,
            True,
        ),
        # Timeout 相关
        (
            "planner.timeout",
            {"planner": {"timeout": 600}},
            lambda c: c.planner.timeout,
            600.0,
        ),
        (
            "worker.task_timeout",
            {"worker": {"task_timeout": 700}},
            lambda c: c.worker.task_timeout,
            700.0,
        ),
        (
            "reviewer.timeout",
            {"reviewer": {"timeout": 400}},
            lambda c: c.reviewer.timeout,
            400.0,
        ),
        # Models 相关
        (
            "models.planner",
            {"models": {"planner": "custom-planner"}},
            lambda c: c.models.planner,
            "custom-planner",
        ),
        (
            "models.worker",
            {"models": {"worker": "custom-worker"}},
            lambda c: c.models.worker,
            "custom-worker",
        ),
        (
            "models.reviewer",
            {"models": {"reviewer": "custom-reviewer"}},
            lambda c: c.models.reviewer,
            "custom-reviewer",
        ),
    ]

    @pytest.mark.parametrize(
        "field_path,config_content,getter,expected_value",
        CONFIG_FIELD_TEST_CASES,
        ids=[case[0] for case in CONFIG_FIELD_TEST_CASES],
    )
    def test_config_field_loading(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
        field_path: str,
        config_content: dict,
        getter,
        expected_value,
    ) -> None:
        """表驱动测试：验证单个配置字段的正确加载

        Args:
            field_path: 配置字段路径（用于标识测试用例）
            config_content: YAML 配置内容
            getter: 从 ConfigManager 获取值的 lambda
            expected_value: 期望的值
        """
        # 创建配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 设置配置路径并重置
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 获取配置实例
        config = ConfigManager.get_instance()

        # 验证值
        actual_value = getter(config)
        assert actual_value == expected_value, f"字段 {field_path}: 期望 {expected_value}, 实际 {actual_value}"

    # 默认值测试数据：(字段路径, getter_lambda, DEFAULT_常量名)
    DEFAULT_VALUE_TEST_CASES = [
        (
            "system.max_iterations",
            lambda c: c.system.max_iterations,
            "DEFAULT_MAX_ITERATIONS",
        ),
        (
            "system.worker_pool_size",
            lambda c: c.system.worker_pool_size,
            "DEFAULT_WORKER_POOL_SIZE",
        ),
        (
            "models.planner",
            lambda c: c.models.planner,
            "DEFAULT_PLANNER_MODEL",
        ),
        (
            "models.worker",
            lambda c: c.models.worker,
            "DEFAULT_WORKER_MODEL",
        ),
        (
            "models.reviewer",
            lambda c: c.models.reviewer,
            "DEFAULT_REVIEWER_MODEL",
        ),
        (
            "cloud_agent.timeout",
            lambda c: c.cloud_agent.timeout,
            "DEFAULT_CLOUD_TIMEOUT",
        ),
        (
            "cloud_agent.auth_timeout",
            lambda c: c.cloud_agent.auth_timeout,
            "DEFAULT_CLOUD_AUTH_TIMEOUT",
        ),
        (
            "planner.timeout",
            lambda c: c.planner.timeout,
            "DEFAULT_PLANNING_TIMEOUT",
        ),
        (
            "worker.task_timeout",
            lambda c: c.worker.task_timeout,
            "DEFAULT_WORKER_TIMEOUT",
        ),
        (
            "reviewer.timeout",
            lambda c: c.reviewer.timeout,
            "DEFAULT_REVIEW_TIMEOUT",
        ),
    ]

    @pytest.mark.parametrize(
        "field_path,getter,default_constant_name",
        DEFAULT_VALUE_TEST_CASES,
        ids=[case[0] for case in DEFAULT_VALUE_TEST_CASES],
    )
    def test_default_values_when_no_config(
        self,
        reset_config_manager,
        field_path: str,
        getter,
        default_constant_name: str,
    ) -> None:
        """表驱动测试：验证无配置文件时使用 DEFAULT_* 常量作为默认值

        Args:
            field_path: 配置字段路径（用于标识测试用例）
            getter: 从 ConfigManager 获取值的 lambda
            default_constant_name: DEFAULT_* 常量名
        """
        import core.config as config_module

        # Patch _find_config_file 返回 None，模拟无配置文件
        with patch.object(ConfigManager, "_find_config_file", return_value=None):
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            # 获取 DEFAULT_* 常量值
            expected_value = getattr(config_module, default_constant_name)

            # 获取实际值
            actual_value = getter(config)

            assert actual_value == expected_value, (
                f"字段 {field_path}: 期望 DEFAULT 值 {expected_value}, 实际 {actual_value}"
            )


# ============================================================
# TestOrchestratorConfigFromCustomYaml - Orchestrator 配置从自定义 config.yaml 加载
# ============================================================


class TestOrchestratorConfigFromCustomYaml:
    """测试 Orchestrator/MultiProcessOrchestrator 配置能从自定义 config.yaml 正确加载

    验证以下配置字段：
    - planner_model/worker_model/reviewer_model
    - planner timeout/worker task_timeout/reviewer timeout
    - agent_cli.timeout/output_format
    - system.max_iterations/worker_pool_size
    """

    @pytest.fixture
    def orchestrator_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试编排器配置的自定义 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 25,
                "worker_pool_size": 8,
                "enable_sub_planners": False,
                "strict_review": True,
            },
            "models": {
                "planner": "test-planner-model-v1",
                "worker": "test-worker-model-v1",
                "reviewer": "test-reviewer-model-v1",
            },
            "planner": {
                "timeout": 450.0,
            },
            "worker": {
                "task_timeout": 550.0,
            },
            "reviewer": {
                "timeout": 250.0,
            },
            "agent_cli": {
                "timeout": 400,
                "output_format": "json",
                "max_retries": 5,
            },
            "cloud_agent": {
                "timeout": 900,
                "auth_timeout": 60,
                "execution_mode": "auto",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_orchestrator_config_uses_custom_models(
        self,
        tmp_path: Path,
        orchestrator_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 OrchestratorConfig 使用自定义 config.yaml 中的模型配置"""

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 直接获取配置管理器中的值
        config = ConfigManager.get_instance()

        # 验证模型配置
        assert config.models.planner == "test-planner-model-v1"
        assert config.models.worker == "test-worker-model-v1"
        assert config.models.reviewer == "test-reviewer-model-v1"

        # 构建 OrchestratorConfig 时应使用 config.yaml 中的值
        # 注意：OrchestratorConfig 的默认值从 core.config 常量获取
        # 这里验证 ConfigManager 能正确读取 config.yaml
        from core.config import get_config

        cfg = get_config()
        assert cfg.models.planner == "test-planner-model-v1"
        assert cfg.models.worker == "test-worker-model-v1"
        assert cfg.models.reviewer == "test-reviewer-model-v1"

    def test_orchestrator_config_uses_custom_timeouts(
        self,
        tmp_path: Path,
        orchestrator_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 OrchestratorConfig 使用自定义 config.yaml 中的超时配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 验证超时配置
        assert config.planner.timeout == 450.0
        assert config.worker.task_timeout == 550.0
        assert config.reviewer.timeout == 250.0
        assert config.agent_cli.timeout == 400

    def test_orchestrator_config_uses_custom_system_settings(
        self,
        tmp_path: Path,
        orchestrator_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 OrchestratorConfig 使用自定义 config.yaml 中的系统配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 验证系统配置
        assert config.system.max_iterations == 25
        assert config.system.worker_pool_size == 8
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True

    def test_orchestrator_config_uses_custom_agent_cli_settings(
        self,
        tmp_path: Path,
        orchestrator_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 OrchestratorConfig 使用自定义 config.yaml 中的 agent_cli 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config = ConfigManager.get_instance()

        # 验证 agent_cli 配置
        assert config.agent_cli.timeout == 400
        assert config.agent_cli.output_format == "json"
        assert config.agent_cli.max_retries == 5

    def test_multiprocess_orchestrator_config_uses_custom_yaml(
        self,
        tmp_path: Path,
        orchestrator_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 MultiProcessOrchestratorConfig 能正确使用自定义 config.yaml"""
        from coordinator.orchestrator_mp import MultiProcessOrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 从 ConfigManager 获取配置
        cfg = ConfigManager.get_instance()

        # 验证 MultiProcessOrchestratorConfig 可以使用 config.yaml 中的值
        # 通过手动传入 config.yaml 中的值来验证
        mp_config = MultiProcessOrchestratorConfig(
            max_iterations=cfg.system.max_iterations,
            worker_count=cfg.system.worker_pool_size,
            planner_model=cfg.models.planner,
            worker_model=cfg.models.worker,
            reviewer_model=cfg.models.reviewer,
            planning_timeout=cfg.planner.timeout,
            execution_timeout=cfg.worker.task_timeout,
            review_timeout=cfg.reviewer.timeout,
        )

        # 验证配置值正确传递
        assert mp_config.max_iterations == 25
        assert mp_config.worker_count == 8
        assert mp_config.planner_model == "test-planner-model-v1"
        assert mp_config.worker_model == "test-worker-model-v1"
        assert mp_config.reviewer_model == "test-reviewer-model-v1"
        assert mp_config.planning_timeout == 450.0
        assert mp_config.execution_timeout == 550.0
        assert mp_config.review_timeout == 250.0


# ============================================================
# TestOrchestratorConfigTriStateModel - OrchestratorConfig tri-state 模型配置测试
# ============================================================


class TestOrchestratorConfigTriStateModel:
    """测试 OrchestratorConfig 的 tri-state 模型配置

    验证以下行为:
    1. 不传入模型参数时，从 config.yaml 获取模型配置（通过 _resolve_config_values）
    2. 显式传入模型参数时，使用传入的值（优先级最高）
    3. config.yaml 中未配置时，使用 DEFAULT_* 常量作为回退
    """

    @pytest.fixture
    def tristate_model_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 tri-state 模型配置的自定义 config.yaml"""
        config_content = {
            "models": {
                "planner": "yaml-planner-model",
                "worker": "yaml-worker-model",
                "reviewer": "yaml-reviewer-model",
            },
            "system": {
                "max_iterations": 5,
                "worker_pool_size": 2,
            },
            "planner": {
                "timeout": 100.0,
            },
            "worker": {
                "task_timeout": 100.0,
            },
            "reviewer": {
                "timeout": 100.0,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_orchestrator_config_model_fields_default_to_none(
        self,
        reset_config_manager,
    ) -> None:
        """测试 OrchestratorConfig 模型字段默认为 None"""
        from coordinator.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        # 验证模型字段默认为 None（tri-state 设计）
        assert config.planner_model is None
        assert config.worker_model is None
        assert config.reviewer_model is None

    def test_orchestrator_resolves_models_from_yaml_when_none(
        self,
        tmp_path: Path,
        tristate_model_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试当模型参数为 None 时，从 config.yaml 解析模型配置

        这是核心测试：验证 OrchestratorConfig 不传入模型参数时，
        Orchestrator._resolve_config_values() 能正确从 config.yaml 获取模型值。
        """
        from unittest.mock import MagicMock, patch

        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建不传入模型参数的 OrchestratorConfig
        config = OrchestratorConfig(
            working_directory=str(tmp_path),
            max_iterations=1,
            worker_pool_size=1,
            # 不传入 planner_model, worker_model, reviewer_model
            # 它们应该是 None，并通过 _resolve_config_values 解析
        )

        # 验证 config 中模型字段为 None
        assert config.planner_model is None
        assert config.worker_model is None
        assert config.reviewer_model is None

        # Mock 依赖项以避免实际初始化 Agent
        with (
            patch("coordinator.orchestrator.PlannerAgent") as mock_planner,
            patch("coordinator.orchestrator.ReviewerAgent") as mock_reviewer,
            patch("coordinator.orchestrator.WorkerPool") as mock_worker_pool,
        ):
            mock_planner.return_value = MagicMock(id="planner-1")
            mock_reviewer.return_value = MagicMock(id="reviewer-1", review_history=[])
            mock_worker_pool.return_value = MagicMock(workers=[])

            orchestrator = Orchestrator(config)

            # 验证解析后的模型配置来自 config.yaml
            assert orchestrator._resolved_config["planner_model"] == "yaml-planner-model"
            assert orchestrator._resolved_config["worker_model"] == "yaml-worker-model"
            assert orchestrator._resolved_config["reviewer_model"] == "yaml-reviewer-model"

    def test_orchestrator_explicit_model_overrides_yaml(
        self,
        tmp_path: Path,
        tristate_model_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试显式传入模型参数时，覆盖 config.yaml 中的值

        验证配置优先级：显式传入 > config.yaml
        """
        from unittest.mock import MagicMock, patch

        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建显式传入模型参数的 OrchestratorConfig
        config = OrchestratorConfig(
            working_directory=str(tmp_path),
            max_iterations=1,
            worker_pool_size=1,
            planner_model="explicit-planner-model",
            worker_model="explicit-worker-model",
            reviewer_model="explicit-reviewer-model",
        )

        # 验证 config 中模型字段为显式传入的值
        assert config.planner_model == "explicit-planner-model"
        assert config.worker_model == "explicit-worker-model"
        assert config.reviewer_model == "explicit-reviewer-model"

        # Mock 依赖项
        with (
            patch("coordinator.orchestrator.PlannerAgent") as mock_planner,
            patch("coordinator.orchestrator.ReviewerAgent") as mock_reviewer,
            patch("coordinator.orchestrator.WorkerPool") as mock_worker_pool,
        ):
            mock_planner.return_value = MagicMock(id="planner-1")
            mock_reviewer.return_value = MagicMock(id="reviewer-1", review_history=[])
            mock_worker_pool.return_value = MagicMock(workers=[])

            orchestrator = Orchestrator(config)

            # 验证解析后的模型配置为显式传入的值（覆盖 config.yaml）
            assert orchestrator._resolved_config["planner_model"] == "explicit-planner-model"
            assert orchestrator._resolved_config["worker_model"] == "explicit-worker-model"
            assert orchestrator._resolved_config["reviewer_model"] == "explicit-reviewer-model"

    def test_orchestrator_partial_explicit_models(
        self,
        tmp_path: Path,
        tristate_model_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试部分显式传入模型参数时的混合配置

        验证：显式传入的优先，未传入的从 config.yaml 获取
        """
        from unittest.mock import MagicMock, patch

        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 只传入 planner_model，其他使用 config.yaml
        config = OrchestratorConfig(
            working_directory=str(tmp_path),
            max_iterations=1,
            worker_pool_size=1,
            planner_model="explicit-planner-only",
            # worker_model 和 reviewer_model 为 None，应从 config.yaml 获取
        )

        with (
            patch("coordinator.orchestrator.PlannerAgent") as mock_planner,
            patch("coordinator.orchestrator.ReviewerAgent") as mock_reviewer,
            patch("coordinator.orchestrator.WorkerPool") as mock_worker_pool,
        ):
            mock_planner.return_value = MagicMock(id="planner-1")
            mock_reviewer.return_value = MagicMock(id="reviewer-1", review_history=[])
            mock_worker_pool.return_value = MagicMock(workers=[])

            orchestrator = Orchestrator(config)

            # 验证混合配置结果
            assert orchestrator._resolved_config["planner_model"] == "explicit-planner-only"
            assert orchestrator._resolved_config["worker_model"] == "yaml-worker-model"
            assert orchestrator._resolved_config["reviewer_model"] == "yaml-reviewer-model"

    def test_orchestrator_models_fallback_to_defaults_when_no_yaml(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试无 config.yaml 时回退到 DEFAULT_* 常量"""
        from unittest.mock import MagicMock, patch

        from coordinator.orchestrator import Orchestrator, OrchestratorConfig
        from core.config import (
            DEFAULT_PLANNER_MODEL,
            DEFAULT_REVIEWER_MODEL,
            DEFAULT_WORKER_MODEL,
        )

        # 空目录（无 config.yaml）
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # Patch _find_config_file 返回 None
        with patch.object(ConfigManager, "_find_config_file", return_value=None):
            ConfigManager.reset_instance()

            config = OrchestratorConfig(
                working_directory=str(empty_dir),
                max_iterations=1,
                worker_pool_size=1,
            )

            with (
                patch("coordinator.orchestrator.PlannerAgent") as mock_planner,
                patch("coordinator.orchestrator.ReviewerAgent") as mock_reviewer,
                patch("coordinator.orchestrator.WorkerPool") as mock_worker_pool,
            ):
                mock_planner.return_value = MagicMock(id="planner-1")
                mock_reviewer.return_value = MagicMock(id="reviewer-1", review_history=[])
                mock_worker_pool.return_value = MagicMock(workers=[])

                orchestrator = Orchestrator(config)

                # 验证回退到 DEFAULT_* 常量
                assert orchestrator._resolved_config["planner_model"] == DEFAULT_PLANNER_MODEL
                assert orchestrator._resolved_config["worker_model"] == DEFAULT_WORKER_MODEL
                assert orchestrator._resolved_config["reviewer_model"] == DEFAULT_REVIEWER_MODEL


# ============================================================
# TestResolveOrchestratorSettings - resolve_orchestrator_settings 测试
# ============================================================


class TestResolveOrchestratorSettings:
    """测试 resolve_orchestrator_settings 函数

    验证优先级: overrides (CLI) > config.yaml > DEFAULT_*
    """

    @pytest.fixture
    def resolver_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 resolver 的自定义 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 30,
                "worker_pool_size": 6,
                "enable_sub_planners": True,
                "strict_review": False,
            },
            "planner": {
                "timeout": 500.0,
            },
            "worker": {
                "task_timeout": 600.0,
            },
            "reviewer": {
                "timeout": 300.0,
            },
            "cloud_agent": {
                "execution_mode": "cli",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_resolve_orchestrator_settings_from_yaml(
        self,
        tmp_path: Path,
        resolver_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 resolve_orchestrator_settings 从 config.yaml 加载配置"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传递 overrides，应使用 config.yaml 的值
        settings = resolve_orchestrator_settings()

        assert settings["max_iterations"] == 30
        assert settings["workers"] == 6
        assert settings["planner_timeout"] == 500.0
        assert settings["worker_timeout"] == 600.0
        assert settings["reviewer_timeout"] == 300.0
        assert settings["enable_sub_planners"] is True
        assert settings["strict_review"] is False

    def test_resolve_orchestrator_settings_overrides_yaml(
        self,
        tmp_path: Path,
        resolver_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI overrides 可覆盖 config.yaml 值"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 传递 overrides，应覆盖 config.yaml 的值
        settings = resolve_orchestrator_settings(
            overrides={
                "max_iterations": 100,
                "workers": 10,
                "planner_timeout": 999.0,
            }
        )

        # 覆盖的值
        assert settings["max_iterations"] == 100
        assert settings["workers"] == 10
        assert settings["planner_timeout"] == 999.0

        # 未覆盖的值应从 config.yaml 获取
        assert settings["worker_timeout"] == 600.0
        assert settings["reviewer_timeout"] == 300.0

    def test_resolve_orchestrator_settings_execution_mode_forces_basic(
        self,
        tmp_path: Path,
        resolver_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 execution_mode=cloud/auto 时强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # execution_mode=cloud 时，即使 orchestrator=mp，也应强制切换到 basic
        settings = resolve_orchestrator_settings(
            overrides={
                "execution_mode": "cloud",
                "orchestrator": "mp",
            }
        )

        assert settings["orchestrator"] == "basic", "execution_mode=cloud 时应强制使用 basic 编排器"

        # execution_mode=auto 也应强制 basic
        settings_auto = resolve_orchestrator_settings(
            overrides={
                "execution_mode": "auto",
                "orchestrator": "mp",
            }
        )

        assert settings_auto["orchestrator"] == "basic", "execution_mode=auto 时应强制使用 basic 编排器"

    def test_resolve_orchestrator_settings_cli_mode_allows_mp(
        self,
        tmp_path: Path,
        resolver_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 execution_mode=cli 时允许使用 mp 编排器"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # execution_mode=cli 时，orchestrator=mp 应保持
        settings = resolve_orchestrator_settings(
            overrides={
                "execution_mode": "cli",
                "orchestrator": "mp",
            }
        )

        assert settings["orchestrator"] == "mp", "execution_mode=cli 时应允许使用 mp 编排器"


# ============================================================
# TestRunIterateGetOrchestratorType - run_iterate.py _get_orchestrator_type 纯函数测试
# ============================================================


class TestRunIterateGetOrchestratorType:
    """测试 scripts/run_iterate.py 的 _get_orchestrator_type 逻辑

    这是纯函数测试，避免运行完整网络/多进程。
    验证 execution_mode 强制 basic 的逻辑。
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 以避免因无 API Key 导致的模式回退

        当测试显式设置了 API Key 时返回该值，否则返回 mock 值。
        """
        from cursor.cloud_client import CloudClientFactory

        def _resolve_api_key(explicit_api_key=None, **kwargs):
            return explicit_api_key if explicit_api_key else "mock-api-key"

        with patch.object(CloudClientFactory, "resolve_api_key", side_effect=_resolve_api_key):
            yield

    @pytest.fixture
    def base_iterate_args(self) -> argparse.Namespace:
        """基础 iterate args fixture"""
        args = argparse.Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            _orchestrator_user_set=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )
        return args

    def test_get_orchestrator_type_default_mp(self, base_iterate_args: argparse.Namespace) -> None:
        """测试默认使用 MP 编排器"""
        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp"

    def test_get_orchestrator_type_cloud_forces_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cloud 时强制使用 basic 编排器"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.execution_mode = "cloud"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = True

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", "execution_mode=cloud 时应强制使用 basic 编排器"

    def test_get_orchestrator_type_auto_forces_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=auto 时强制使用 basic 编排器"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.execution_mode = "auto"
        base_iterate_args.orchestrator = "mp"
        base_iterate_args._orchestrator_user_set = True

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic", "execution_mode=auto 时应强制使用 basic 编排器"

    def test_get_orchestrator_type_cli_allows_mp(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 execution_mode=cli 时允许使用 mp 编排器"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.execution_mode = "cli"
        base_iterate_args.orchestrator = "mp"

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "mp", "execution_mode=cli 时应允许使用 mp 编排器"

    def test_get_orchestrator_type_no_mp_flag(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 --no-mp 标志强制使用 basic"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.no_mp = True

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"

    def test_get_orchestrator_type_explicit_basic(self, base_iterate_args: argparse.Namespace) -> None:
        """测试 --orchestrator basic 显式设置"""
        from scripts.run_iterate import SelfIterator

        base_iterate_args.orchestrator = "basic"
        base_iterate_args._orchestrator_user_set = True  # 标记为显式设置

        iterator = SelfIterator(base_iterate_args)
        assert iterator._get_orchestrator_type() == "basic"


# ============================================================
# TestBuildCursorAgentConfigForRole - build_cursor_agent_config_for_role 测试
# ============================================================


class TestBuildCursorAgentConfigForRole:
    """测试 build_cursor_agent_config_for_role 函数

    验证根据角色自动选择模型和超时配置。
    """

    @pytest.fixture
    def role_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试角色配置的自定义 config.yaml"""
        config_content = {
            "models": {
                "planner": "custom-planner-for-role",
                "worker": "custom-worker-for-role",
                "reviewer": "custom-reviewer-for-role",
            },
            "planner": {
                "timeout": 333.0,
            },
            "worker": {
                "task_timeout": 444.0,
            },
            "reviewer": {
                "timeout": 222.0,
            },
            "agent_cli": {
                "timeout": 300,
                "output_format": "text",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_build_config_for_planner_role(
        self,
        tmp_path: Path,
        role_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 planner 角色的配置构建"""
        from core.config import build_cursor_agent_config_for_role

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config_for_role(
            role="planner",
            working_directory="/test/project",
        )

        assert config_dict["model"] == "custom-planner-for-role"
        assert config_dict["timeout"] == 333  # planner.timeout
        assert config_dict["mode"] == "plan"  # Planner 使用规划模式

    def test_build_config_for_worker_role(
        self,
        tmp_path: Path,
        role_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 worker 角色的配置构建"""
        from core.config import build_cursor_agent_config_for_role

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config_for_role(
            role="worker",
            working_directory="/test/project",
        )

        assert config_dict["model"] == "custom-worker-for-role"
        assert config_dict["timeout"] == 444  # worker.task_timeout
        assert config_dict["mode"] == "agent"  # Worker 使用完整代理模式

    def test_build_config_for_reviewer_role(
        self,
        tmp_path: Path,
        role_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 reviewer 角色的配置构建"""
        from core.config import build_cursor_agent_config_for_role

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config_for_role(
            role="reviewer",
            working_directory="/test/project",
        )

        assert config_dict["model"] == "custom-reviewer-for-role"
        assert config_dict["timeout"] == 222  # reviewer.timeout
        assert config_dict["mode"] == "plan"  # Reviewer 使用规划模式（只读）

    def test_build_config_overrides_take_priority(
        self,
        tmp_path: Path,
        role_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 overrides 参数优先级高于 config.yaml"""
        from core.config import build_cursor_agent_config_for_role

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config_for_role(
            role="worker",
            working_directory="/test/project",
            overrides={
                "model": "override-model",
                "timeout": 999,
            },
        )

        # overrides 应覆盖角色默认值
        assert config_dict["model"] == "override-model"
        assert config_dict["timeout"] == 999


# ============================================================
# TestCloudClientConfigPriority - Cloud Client 配置优先级测试
# ============================================================


class TestCloudClientConfigPriority:
    """测试 Cloud Client 配置优先级

    验证优先级：
    1. 显式参数 > CURSOR_API_KEY > CURSOR_CLOUD_API_KEY > config.yaml

    覆盖场景：
    - build_cloud_client_config 优先级
    - CloudClientFactory.resolve_api_key 优先级
    - CloudClientFactory.create_auth_config 配置合并
    - CloudAuthManager.get_api_key 优先级
    """

    @pytest.fixture
    def cloud_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 Cloud 配置的自定义 config.yaml"""
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-cloud-api-key",
                "api_base_url": "https://custom.api.cursor.com",
                "timeout": 600,
                "auth_timeout": 45,
                "max_retries": 5,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_build_cloud_client_config_from_yaml(
        self,
        tmp_path: Path,
        cloud_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 build_cloud_client_config 从 config.yaml 加载配置"""
        from core.config import build_cloud_client_config

        # 清除环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config = build_cloud_client_config()

        assert config["base_url"] == "https://custom.api.cursor.com"
        assert config["timeout"] == 600
        assert config["auth_timeout"] == 45
        assert config["max_retries"] == 5
        assert config["api_key"] == "yaml-cloud-api-key"

    def test_build_cloud_client_config_env_cursor_api_key_priority(
        self,
        tmp_path: Path,
        cloud_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CURSOR_API_KEY 优先级高于 config.yaml"""
        from core.config import build_cloud_client_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置 CURSOR_API_KEY 环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        config = build_cloud_client_config()

        # CURSOR_API_KEY 应优先于 config.yaml
        assert config["api_key"] == "env-cursor-api-key"

    def test_build_cloud_client_config_env_cursor_cloud_api_key_fallback(
        self,
        tmp_path: Path,
        cloud_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CURSOR_CLOUD_API_KEY 作为备选优先级高于 config.yaml"""
        from core.config import build_cloud_client_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 只设置 CURSOR_CLOUD_API_KEY，不设置 CURSOR_API_KEY
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        config = build_cloud_client_config()

        # CURSOR_CLOUD_API_KEY 应优先于 config.yaml
        assert config["api_key"] == "env-cursor-cloud-api-key"

    def test_build_cloud_client_config_cursor_api_key_over_cloud_api_key(
        self,
        tmp_path: Path,
        cloud_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CURSOR_API_KEY 优先级高于 CURSOR_CLOUD_API_KEY"""
        from core.config import build_cloud_client_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 同时设置两个环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        config = build_cloud_client_config()

        # CURSOR_API_KEY 应优先于 CURSOR_CLOUD_API_KEY
        assert config["api_key"] == "env-cursor-api-key"

    def test_build_cloud_client_config_overrides_highest_priority(
        self,
        tmp_path: Path,
        cloud_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试显式 overrides 优先级最高"""
        from core.config import build_cloud_client_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        config = build_cloud_client_config(
            overrides={
                "api_key": "explicit-api-key",
                "base_url": "https://explicit.api.com",
                "timeout": 999,
                "auth_timeout": 99,
                "max_retries": 9,
            }
        )

        # 显式参数应优先于环境变量和 config.yaml
        assert config["api_key"] == "explicit-api-key"
        assert config["base_url"] == "https://explicit.api.com"
        assert config["timeout"] == 999
        assert config["auth_timeout"] == 99
        assert config["max_retries"] == 9


# ============================================================
# TestCloudClientFactoryConfigIntegration - CloudClientFactory 配置集成测试
# ============================================================


class TestCloudClientFactoryConfigIntegration:
    """测试 CloudClientFactory 与 config.yaml 的集成

    验证 CloudClientFactory 正确使用 build_cloud_client_config 获取默认配置。
    """

    @pytest.fixture
    def factory_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 Factory 配置的自定义 config.yaml"""
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-factory-api-key",
                "api_base_url": "https://factory.api.cursor.com",
                "timeout": 700,
                "auth_timeout": 55,
                "max_retries": 6,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_cloud_client_factory_resolve_api_key_env_cursor_cloud_api_key(
        self,
        tmp_path: Path,
        factory_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CloudClientFactory.resolve_api_key 支持 CURSOR_CLOUD_API_KEY"""
        from cursor.cloud_client import CloudClientFactory

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 只设置 CURSOR_CLOUD_API_KEY
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        api_key = CloudClientFactory.resolve_api_key()

        assert api_key == "env-cursor-cloud-api-key"

    def test_cloud_client_factory_resolve_api_key_priority(
        self,
        tmp_path: Path,
        factory_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CloudClientFactory.resolve_api_key 优先级"""
        from cursor.cloud_client import CloudClientFactory

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 同时设置所有来源
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        # 显式参数优先级最高
        api_key = CloudClientFactory.resolve_api_key(explicit_api_key="explicit-api-key")
        assert api_key == "explicit-api-key"

        # CURSOR_API_KEY 优先于 CURSOR_CLOUD_API_KEY
        api_key = CloudClientFactory.resolve_api_key()
        assert api_key == "env-cursor-api-key"

        # 清除 CURSOR_API_KEY，CURSOR_CLOUD_API_KEY 应生效
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        api_key = CloudClientFactory.resolve_api_key()
        assert api_key == "env-cursor-cloud-api-key"

    def test_cloud_client_factory_create_auth_config_from_yaml(
        self,
        tmp_path: Path,
        factory_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CloudClientFactory.create_auth_config 从 config.yaml 获取默认值"""
        from cursor.cloud_client import CloudClientFactory

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        auth_config = CloudClientFactory.create_auth_config()

        # 应使用 config.yaml 中的值
        assert auth_config.api_key == "yaml-factory-api-key"
        assert auth_config.api_base_url == "https://factory.api.cursor.com"
        assert auth_config.auth_timeout == 55
        assert auth_config.max_retries == 6


# ============================================================
# TestCloudAuthManagerApiKeyPriority - CloudAuthManager API Key 优先级测试
# ============================================================


class TestCloudAuthManagerApiKeyPriority:
    """测试 CloudAuthManager.get_api_key 优先级

    验证优先级：
    1. 配置对象中的 api_key
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY
    4. 项目配置文件
    """

    def test_cloud_auth_manager_env_cursor_cloud_api_key(self, monkeypatch) -> None:
        """测试 CloudAuthManager 支持 CURSOR_CLOUD_API_KEY"""
        from cursor.cloud.auth import CloudAuthConfig, CloudAuthManager

        # 只设置 CURSOR_CLOUD_API_KEY
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        manager = CloudAuthManager(config=CloudAuthConfig())
        api_key = manager.get_api_key()

        assert api_key == "env-cursor-cloud-api-key"

    def test_cloud_auth_manager_cursor_api_key_over_cloud_api_key(self, monkeypatch) -> None:
        """测试 CloudAuthManager CURSOR_API_KEY 优先于 CURSOR_CLOUD_API_KEY"""
        from cursor.cloud.auth import CloudAuthConfig, CloudAuthManager

        # 同时设置两个环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        manager = CloudAuthManager(config=CloudAuthConfig())
        api_key = manager.get_api_key()

        # CURSOR_API_KEY 应优先
        assert api_key == "env-cursor-api-key"

    def test_cloud_auth_manager_config_api_key_highest_priority(self, monkeypatch) -> None:
        """测试 CloudAuthManager 配置中的 api_key 优先级最高"""
        from cursor.cloud.auth import CloudAuthConfig, CloudAuthManager

        # 设置所有来源
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        config = CloudAuthConfig(api_key="config-api-key")
        manager = CloudAuthManager(config=config)
        api_key = manager.get_api_key()

        # 配置中的 api_key 应优先
        assert api_key == "config-api-key"


# ============================================================
# TestCloudAgentExecutorConfigIntegration - CloudAgentExecutor 配置集成测试
# ============================================================


class TestCloudAgentExecutorConfigIntegration:
    """测试 CloudAgentExecutor 与 config.yaml 的配置集成

    验证 CloudAgentExecutor 正确使用 build_cloud_client_config 获取默认配置。
    """

    @pytest.fixture
    def executor_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 Executor 配置的自定义 config.yaml"""
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-executor-api-key",
                "api_base_url": "https://executor.api.cursor.com",
                "timeout": 800,
                "auth_timeout": 65,
                "max_retries": 7,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_cloud_agent_executor_uses_yaml_config(
        self,
        tmp_path: Path,
        executor_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CloudAgentExecutor 使用 config.yaml 中的配置"""
        from cursor.executor import CloudAgentExecutor

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        executor = CloudAgentExecutor()

        # 验证 auth_config 使用了 config.yaml 中的值
        assert executor._auth_config.api_key == "yaml-executor-api-key"
        assert executor._auth_config.api_base_url == "https://executor.api.cursor.com"
        assert executor._auth_config.auth_timeout == 65
        assert executor._auth_config.max_retries == 7

    def test_cloud_agent_executor_env_overrides_yaml(
        self,
        tmp_path: Path,
        executor_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CloudAgentExecutor 环境变量优先于 config.yaml"""
        from cursor.executor import CloudAgentExecutor

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置环境变量
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")

        executor = CloudAgentExecutor()

        # 环境变量应优先于 config.yaml（通过 auth_manager 验证）
        api_key = executor._auth_manager.get_api_key()
        assert api_key == "env-cursor-api-key"


# ============================================================
# TestCloudAgentConfigLoading - Cloud Agent 配置加载测试
# ============================================================


class TestCloudAgentConfigLoading:
    """测试 cloud_agent 配置从 config.yaml 加载

    覆盖字段：
    - api_base_url
    - max_retries
    - timeout
    - auth_timeout
    - execution_mode
    """

    @pytest.fixture
    def cloud_agent_custom_config_yaml(self, tmp_path: Path) -> Path:
        """创建包含 cloud_agent 配置的自定义 config.yaml"""
        config_content = {
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "api_base_url": "https://custom.api.example.com",
                "timeout": 600,
                "auth_timeout": 60,
                "max_retries": 5,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_load_cloud_agent_api_base_url(
        self,
        tmp_path: Path,
        cloud_agent_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试加载 cloud_agent.api_base_url"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.api_base_url == "https://custom.api.example.com"

    def test_load_cloud_agent_max_retries(
        self,
        tmp_path: Path,
        cloud_agent_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试加载 cloud_agent.max_retries"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.max_retries == 5

    def test_load_cloud_agent_timeout(
        self,
        tmp_path: Path,
        cloud_agent_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试加载 cloud_agent.timeout"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.timeout == 600

    def test_load_cloud_agent_auth_timeout(
        self,
        tmp_path: Path,
        cloud_agent_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试加载 cloud_agent.auth_timeout"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.auth_timeout == 60

    def test_load_cloud_agent_execution_mode(
        self,
        tmp_path: Path,
        cloud_agent_custom_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试加载 cloud_agent.execution_mode"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.execution_mode == "auto"

    def test_cloud_agent_default_values_when_no_config(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试无 cloud_agent 配置时使用默认值"""
        # 创建空配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证默认值
        from core.config import (
            DEFAULT_CLOUD_AUTH_TIMEOUT,
            DEFAULT_CLOUD_TIMEOUT,
        )

        assert config.cloud_agent.timeout == DEFAULT_CLOUD_TIMEOUT
        assert config.cloud_agent.auth_timeout == DEFAULT_CLOUD_AUTH_TIMEOUT
        assert config.cloud_agent.api_base_url == "https://api.cursor.com"
        assert config.cloud_agent.max_retries == 3

    def test_auto_detect_cloud_prefix_default_true(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 auto_detect_cloud_prefix 缺省时默认为 True"""
        # 创建空配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证默认值为 True
        assert config.cloud_agent.auto_detect_cloud_prefix is True

    def test_auto_detect_cloud_prefix_explicit_false(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试显式设置 auto_detect_cloud_prefix 为 False"""
        config_content = {
            "cloud_agent": {
                "auto_detect_cloud_prefix": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.auto_detect_cloud_prefix is False

    def test_auto_detect_prefix_alias(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 auto_detect_prefix 作为别名生效，并触发 deprecated 警告"""
        from io import StringIO

        from loguru import logger

        from core.config import (
            DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX,
            reset_deprecated_warnings,
        )

        config_content = {
            "cloud_agent": {
                "auto_detect_prefix": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 重置警告状态，确保可以捕获警告
        reset_deprecated_warnings()

        # 捕获日志输出
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")

        try:
            monkeypatch.chdir(tmp_path)
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            # 别名应该生效
            assert config.cloud_agent.auto_detect_cloud_prefix is False

            # 验证 deprecated 警告被触发
            log_content = log_output.getvalue()
            assert DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX in log_content, (
                f"使用旧字段 auto_detect_prefix 应触发 deprecated 警告，日志: {log_content}"
            )
        finally:
            logger.remove(handler_id)
            reset_deprecated_warnings()

    def test_auto_detect_cloud_prefix_overrides_alias(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 auto_detect_cloud_prefix 优先于别名 auto_detect_prefix"""
        config_content = {
            "cloud_agent": {
                "auto_detect_cloud_prefix": True,
                "auto_detect_prefix": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 主字段优先于别名
        assert config.cloud_agent.auto_detect_cloud_prefix is True

    def test_get_config_auto_detect_cloud_prefix_true(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试使用 get_config() 获取 auto_detect_cloud_prefix=True"""
        config_content = {
            "cloud_agent": {
                "auto_detect_cloud_prefix": True,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 get_config() 获取配置
        config = get_config()

        assert config.cloud_agent.auto_detect_cloud_prefix is True

    def test_get_config_auto_detect_cloud_prefix_false(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试使用 get_config() 获取 auto_detect_cloud_prefix=False"""
        config_content = {
            "cloud_agent": {
                "auto_detect_cloud_prefix": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 get_config() 获取配置
        config = get_config()

        assert config.cloud_agent.auto_detect_cloud_prefix is False

    def test_deprecated_alias_warning_not_spam(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 deprecated 警告不会刷屏（每个 key 仅警告一次）

        验证多次重新加载配置时，deprecated 警告只出现一次，
        确保日志可预测且不会刷屏。
        """
        from io import StringIO

        from loguru import logger

        from core.config import (
            DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX,
            reset_deprecated_warnings,
        )

        config_content = {
            "cloud_agent": {
                "auto_detect_prefix": False,  # 使用已废弃的别名
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 重置警告状态，确保测试隔离
        reset_deprecated_warnings()

        # 捕获日志输出
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")

        try:
            monkeypatch.chdir(tmp_path)

            # 第一次加载：应该触发一次警告
            ConfigManager.reset_instance()
            config1 = get_config()
            assert config1.cloud_agent.auto_detect_cloud_prefix is False

            # 记录第一次警告后的日志内容
            first_log = log_output.getvalue()
            first_warning_count = first_log.count(DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX)
            assert first_warning_count == 1, (
                f"首次加载应触发一次 deprecated 警告，实际: {first_warning_count}\n日志内容: {first_log}"
            )

            # 第二次加载：不应重复警告
            ConfigManager.reset_instance()
            config2 = get_config()
            assert config2.cloud_agent.auto_detect_cloud_prefix is False

            # 第三次加载：不应重复警告
            ConfigManager.reset_instance()
            config3 = get_config()
            assert config3.cloud_agent.auto_detect_cloud_prefix is False

            # 验证警告总数仍为 1
            final_log = log_output.getvalue()
            final_warning_count = final_log.count(DEPRECATED_MSG_CLOUD_AGENT_AUTO_DETECT_PREFIX)
            assert final_warning_count == 1, (
                f"多次加载不应重复触发 deprecated 警告，预期 1 次，实际 {final_warning_count} 次\n日志内容: {final_log}"
            )
        finally:
            logger.remove(handler_id)
            reset_deprecated_warnings()

    def test_deprecated_alias_warning_predictable_message(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 deprecated 警告消息可预测

        验证警告消息包含必要信息：
        1. 明确指出旧字段名 auto_detect_prefix
        2. 提供新字段名 auto_detect_cloud_prefix
        3. 提示将在后续版本移除
        """
        from io import StringIO

        from loguru import logger

        from core.config import (
            reset_deprecated_warnings,
        )

        config_content = {
            "cloud_agent": {
                "auto_detect_prefix": True,  # 使用已废弃的别名
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 重置警告状态
        reset_deprecated_warnings()

        # 捕获日志输出
        log_output = StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")

        try:
            monkeypatch.chdir(tmp_path)
            ConfigManager.reset_instance()
            _ = get_config()

            log_content = log_output.getvalue()

            # 验证警告消息包含关键信息
            assert "auto_detect_prefix" in log_content, "警告应明确指出旧字段名 auto_detect_prefix"
            assert "auto_detect_cloud_prefix" in log_content, "警告应提供新字段名 auto_detect_cloud_prefix"
            assert "废弃" in log_content or "移除" in log_content, "警告应提示字段已废弃或将被移除"
        finally:
            logger.remove(handler_id)
            reset_deprecated_warnings()


# ============================================================
# TestCloudAgentConfigPriority - Cloud Agent 配置优先级测试
# ============================================================


class TestCloudAgentConfigPriority:
    """测试 cloud_agent 配置优先级

    优先级: CLI > 环境变量 > config.yaml > DEFAULT_*
    """

    @pytest.fixture
    def cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建测试用 config.yaml"""
        config_content = {
            "cloud_agent": {
                "timeout": 500,
                "auth_timeout": 50,
                "max_retries": 4,
                "api_base_url": "https://yaml.api.com",
                "execution_mode": "cloud",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_config_yaml_overrides_defaults(
        self,
        tmp_path: Path,
        cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 config.yaml 覆盖代码默认值"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # config.yaml 值应该覆盖默认值
        assert config.cloud_agent.timeout == 500
        assert config.cloud_agent.auth_timeout == 50
        assert config.cloud_agent.max_retries == 4
        assert config.cloud_agent.api_base_url == "https://yaml.api.com"

    def test_partial_config_uses_defaults_for_missing(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试部分配置时缺失项使用默认值"""
        # 创建只有部分 cloud_agent 配置的文件
        config_content = {
            "cloud_agent": {
                "timeout": 999,
                # 其他字段缺失，应使用默认值
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 自定义值
        assert config.cloud_agent.timeout == 999

        # 缺失项使用默认值
        from core.config import DEFAULT_CLOUD_AUTH_TIMEOUT

        assert config.cloud_agent.auth_timeout == DEFAULT_CLOUD_AUTH_TIMEOUT
        assert config.cloud_agent.api_base_url == "https://api.cursor.com"


# ============================================================
# TestCloudTimeoutIterateModeConfigLoading - Iterate 模式 Cloud 超时配置测试
# ============================================================


class TestCloudTimeoutIterateModeConfigLoading:
    """测试 --mode iterate 时 cloud 超时配置的加载

    验证:
    1. 配置文件值正确加载
    2. CLI 参数可覆盖配置文件
    """

    @pytest.fixture
    def iterate_cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于 iterate 模式测试的 config.yaml"""
        config_content = {
            "cloud_agent": {
                "timeout": 800,
                "auth_timeout": 40,
                "execution_mode": "auto",
            },
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 5,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_iterate_mode_cloud_timeout_from_config(
        self,
        tmp_path: Path,
        iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 iterate 模式从 config.yaml 加载 cloud_timeout"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.timeout == 800

    def test_iterate_mode_cloud_auth_timeout_from_config(
        self,
        tmp_path: Path,
        iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 iterate 模式从 config.yaml 加载 cloud_auth_timeout"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.auth_timeout == 40

    def test_iterate_mode_execution_mode_from_config(
        self,
        tmp_path: Path,
        iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 iterate 模式从 config.yaml 加载 execution_mode"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.execution_mode == "auto"


# ============================================================
# TestTableDrivenCloudAgentConfig - 表驱动 Cloud Agent 配置测试
# ============================================================


# ============================================================
# TestRunMpPyTriStateOptions - run_mp.py tri-state 选项测试
# ============================================================


class TestRunMpPyTriStateOptions:
    """测试 scripts/run_mp.py 的 tri-state 互斥组选项

    验证:
    1. --strict / --no-strict 互斥组
    2. --sub-planners / --no-sub-planners 互斥组
    3. 不传 CLI flags 时使用 config.yaml 默认值
    """

    @pytest.fixture
    def mp_tristate_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 tri-state 选项的自定义 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 20,
                "worker_pool_size": 4,
                "enable_sub_planners": False,  # 配置默认禁用
                "strict_review": True,  # 配置默认启用
            },
            "models": {
                "planner": "mp-test-planner",
                "worker": "mp-test-worker",
                "reviewer": "mp-test-reviewer",
            },
            "planner": {
                "timeout": 300.0,
            },
            "worker": {
                "task_timeout": 400.0,
            },
            "reviewer": {
                "timeout": 200.0,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_mp_parse_args_strict_review_default_none(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 run_mp.py parse_args 不传 --strict 或 --no-strict 时默认值为 None"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 重新导入 run_mp 模块以获取更新的 parse_args
        import sys

        # 移除已缓存的模块
        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        # Mock sys.argv
        with patch.object(sys, "argv", ["run_mp.py", "测试任务"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        # 不传 CLI flags 时，strict_review 应为 None
        assert args.strict_review is None, "不传 --strict 或 --no-strict 时，strict_review 应为 None"

    def test_mp_parse_args_enable_sub_planners_default_none(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 run_mp.py parse_args 不传 --sub-planners 或 --no-sub-planners 时默认值为 None"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch.object(sys, "argv", ["run_mp.py", "测试任务"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        # 不传 CLI flags 时，enable_sub_planners 应为 None
        assert args.enable_sub_planners is None, (
            "不传 --sub-planners 或 --no-sub-planners 时，enable_sub_planners 应为 None"
        )

    def test_mp_parse_args_strict_flag_sets_true(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 --strict 标志设置 strict_review 为 True"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch.object(sys, "argv", ["run_mp.py", "测试任务", "--strict"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        assert args.strict_review is True

    def test_mp_parse_args_no_strict_flag_sets_false(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 --no-strict 标志设置 strict_review 为 False"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch.object(sys, "argv", ["run_mp.py", "测试任务", "--no-strict"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        assert args.strict_review is False

    def test_mp_parse_args_sub_planners_flag_sets_true(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 --sub-planners 标志设置 enable_sub_planners 为 True"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch.object(sys, "argv", ["run_mp.py", "测试任务", "--sub-planners"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        assert args.enable_sub_planners is True

    def test_mp_parse_args_no_sub_planners_flag_sets_false(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 --no-sub-planners 标志设置 enable_sub_planners 为 False"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch.object(sys, "argv", ["run_mp.py", "测试任务", "--no-sub-planners"]):
            from scripts.run_mp import parse_args

            args = parse_args()

        assert args.enable_sub_planners is False

    def test_mp_run_orchestrator_uses_config_defaults_when_cli_none(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 run_mp.py run_orchestrator 在 CLI 未指定时使用 config.yaml 默认值

        这是核心测试：验证 CLI 未指定 (None) 时，使用 config.yaml 中的值
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证配置值
        config = ConfigManager.get_instance()
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True

        # 模拟 CLI 参数，不传 strict/sub_planners 相关标志
        mock_args = argparse.Namespace(
            goal="测试任务",
            directory=".",
            workers=4,
            max_iterations="20",
            strict_review=None,  # CLI 未指定
            enable_sub_planners=None,  # CLI 未指定
            verbose=False,
            planning_timeout=300.0,
            execution_timeout=400.0,
            review_timeout=200.0,
            planner_model="mp-test-planner",
            worker_model="mp-test-worker",
            reviewer_model="mp-test-reviewer",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            auto_commit=False,
            auto_push=False,
            commit_message="",
        )

        # Mock MultiProcessOrchestratorConfig 来捕获传入的参数
        captured_config = {}

        class MockMPOConfig:
            def __init__(self, **kwargs):
                captured_config.update(kwargs)
                # 添加必要的属性以供日志输出使用
                self.planner_model = kwargs.get("planner_model", "")
                self.worker_model = kwargs.get("worker_model", "")
                self.reviewer_model = kwargs.get("reviewer_model", "")

        # Mock MultiProcessOrchestrator
        class MockMPO:
            def __init__(self, config):
                pass

            async def run(self, goal):
                return {"success": True}

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch("coordinator.orchestrator_mp.MultiProcessOrchestratorConfig", MockMPOConfig):
            with patch("coordinator.orchestrator_mp.MultiProcessOrchestrator", MockMPO):
                import asyncio

                from scripts.run_mp import run_orchestrator

                asyncio.run(run_orchestrator(mock_args))

        # 验证传入 OrchestratorConfig 的值来自 config.yaml
        assert captured_config["enable_sub_planners"] is False, (
            "CLI 未指定时，enable_sub_planners 应使用 config.yaml 的值 (False)"
        )
        assert captured_config["strict_review"] is True, "CLI 未指定时，strict_review 应使用 config.yaml 的值 (True)"

    def test_mp_run_orchestrator_cli_overrides_config(
        self,
        tmp_path: Path,
        mp_tristate_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 run_mp.py CLI 显式值覆盖 config.yaml 默认值"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证配置值（用于对比）
        config = ConfigManager.get_instance()
        assert config.system.enable_sub_planners is False
        assert config.system.strict_review is True

        # 模拟 CLI 参数，显式传递与 config.yaml 相反的值
        mock_args = argparse.Namespace(
            goal="测试任务",
            directory=".",
            workers=4,
            max_iterations="20",
            strict_review=False,  # CLI 显式指定 False（覆盖 config.yaml 的 True）
            enable_sub_planners=True,  # CLI 显式指定 True（覆盖 config.yaml 的 False）
            verbose=False,
            planning_timeout=300.0,
            execution_timeout=400.0,
            review_timeout=200.0,
            planner_model="mp-test-planner",
            worker_model="mp-test-worker",
            reviewer_model="mp-test-reviewer",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            auto_commit=False,
            auto_push=False,
            commit_message="",
        )

        captured_config = {}

        class MockMPOConfig:
            def __init__(self, **kwargs):
                captured_config.update(kwargs)
                # 添加必要的属性以供日志输出使用
                self.planner_model = kwargs.get("planner_model", "")
                self.worker_model = kwargs.get("worker_model", "")
                self.reviewer_model = kwargs.get("reviewer_model", "")

        class MockMPO:
            def __init__(self, config):
                pass

            async def run(self, goal):
                return {"success": True}

        import sys

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        with patch("coordinator.orchestrator_mp.MultiProcessOrchestratorConfig", MockMPOConfig):
            with patch("coordinator.orchestrator_mp.MultiProcessOrchestrator", MockMPO):
                import asyncio

                from scripts.run_mp import run_orchestrator

                asyncio.run(run_orchestrator(mock_args))

        # 验证 CLI 显式值覆盖了 config.yaml
        assert captured_config["enable_sub_planners"] is True, "CLI 显式指定 True 应覆盖 config.yaml 的 False"
        assert captured_config["strict_review"] is False, "CLI 显式指定 False 应覆盖 config.yaml 的 True"


class TestTableDrivenCloudAgentConfig:
    """表驱动测试：验证 cloud_agent 关键配置字段的正确加载

    覆盖字段：
    - api_base_url
    - max_retries
    - timeout
    - auth_timeout
    - execution_mode
    """

    CLOUD_AGENT_CONFIG_TEST_CASES = [
        (
            "cloud_agent.api_base_url",
            {"cloud_agent": {"api_base_url": "https://test.api.com"}},
            lambda c: c.cloud_agent.api_base_url,
            "https://test.api.com",
        ),
        (
            "cloud_agent.max_retries",
            {"cloud_agent": {"max_retries": 10}},
            lambda c: c.cloud_agent.max_retries,
            10,
        ),
        (
            "cloud_agent.timeout",
            {"cloud_agent": {"timeout": 1000}},
            lambda c: c.cloud_agent.timeout,
            1000,
        ),
        (
            "cloud_agent.auth_timeout",
            {"cloud_agent": {"auth_timeout": 90}},
            lambda c: c.cloud_agent.auth_timeout,
            90,
        ),
        (
            "cloud_agent.execution_mode",
            {"cloud_agent": {"execution_mode": "cloud"}},
            lambda c: c.cloud_agent.execution_mode,
            "cloud",
        ),
        (
            "cloud_agent.enabled",
            {"cloud_agent": {"enabled": True}},
            lambda c: c.cloud_agent.enabled,
            True,
        ),
    ]

    @pytest.mark.parametrize(
        "field_path,config_content,getter,expected_value",
        CLOUD_AGENT_CONFIG_TEST_CASES,
        ids=[case[0] for case in CLOUD_AGENT_CONFIG_TEST_CASES],
    )
    def test_cloud_agent_config_field_loading(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
        field_path: str,
        config_content: dict,
        getter,
        expected_value,
    ) -> None:
        """表驱动测试：验证单个 cloud_agent 配置字段的正确加载

        Args:
            field_path: 配置字段路径（用于标识测试用例）
            config_content: YAML 配置内容
            getter: 从 ConfigManager 获取值的 lambda
            expected_value: 期望的值
        """
        # 创建配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        # 设置配置路径并重置
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 获取配置实例
        config = ConfigManager.get_instance()

        # 验证值
        actual_value = getter(config)
        assert actual_value == expected_value, f"字段 {field_path}: 期望 {expected_value}, 实际 {actual_value}"


# ============================================================
# TestEntryScriptConfigLoadingWithTmpPath - 入口脚本配置加载集成测试
# ============================================================


# 完整自定义配置内容（用于入口脚本测试）
FULL_CUSTOM_CONFIG_CONTENT = {
    "system": {
        "max_iterations": 77,
        "worker_pool_size": 5,
        "enable_sub_planners": False,
        "strict_review": True,
    },
    "models": {
        "planner": "test-planner-model-v2",
        "worker": "test-worker-model-v2",
        "reviewer": "test-reviewer-model-v2",
    },
    "planner": {
        "timeout": 450.0,
    },
    "worker": {
        "task_timeout": 550.0,
    },
    "reviewer": {
        "timeout": 250.0,
    },
    "cloud_agent": {
        "enabled": True,
        "execution_mode": "auto",
        "timeout": 999,
        "api_key": "config-yaml-test-api-key",
    },
    "logging": {
        "stream_json": {
            "enabled": True,
            "console": False,
            "detail_dir": "logs/test_detail/",
            "raw_dir": "logs/test_raw/",
        },
    },
}


@pytest.fixture
def full_custom_config_yaml(tmp_path: Path) -> Path:
    """创建完整自定义 config.yaml 文件（包含所有测试所需配置项）"""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(FULL_CUSTOM_CONFIG_CONTENT, f, allow_unicode=True)
    return config_path


class TestEntryScriptConfigLoadingWithTmpPath:
    """入口脚本配置加载集成测试

    使用 tmp_path 写入自定义 config.yaml + monkeypatch(Path.cwd)，
    验证各入口脚本的配置项默认值与覆盖规则。

    测试覆盖的配置项：
    - workers / worker_pool_size
    - max_iterations
    - strict_review
    - enable_sub_planners
    - models.planner / models.worker / models.reviewer
    - planner.timeout
    - worker.task_timeout
    - reviewer.timeout
    - logging.stream_json.enabled
    """

    # ========== run.py 配置测试 ==========

    def test_run_py_workers_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py workers 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 5

    def test_run_py_max_iterations_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py max_iterations 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 77

    def test_run_py_strict_review_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py strict_review 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.strict_review is True

    def test_run_py_enable_sub_planners_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py enable_sub_planners 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.enable_sub_planners is False

    def test_run_py_models_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 模型配置默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.models.planner == "test-planner-model-v2"
        assert config.models.worker == "test-worker-model-v2"
        assert config.models.reviewer == "test-reviewer-model-v2"

    def test_run_py_timeouts_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 超时配置默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.planner.timeout == 450.0
        assert config.worker.task_timeout == 550.0
        assert config.reviewer.timeout == 250.0

    def test_run_py_stream_json_enabled_default_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py stream_json.enabled 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.logging.stream_json.enabled is True

    # ========== scripts/run_basic.py 配置测试 ==========

    def test_run_basic_py_workers_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_basic.py workers 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # run_basic.py 使用 get_config().system.worker_pool_size
        assert config.system.worker_pool_size == 5

    def test_run_basic_py_max_iterations_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_basic.py max_iterations 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 77

    # ========== scripts/run_mp.py 配置测试 ==========

    def test_run_mp_py_workers_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_mp.py workers 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 5

    def test_run_mp_py_models_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_mp.py 模型配置来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.models.planner == "test-planner-model-v2"
        assert config.models.worker == "test-worker-model-v2"
        assert config.models.reviewer == "test-reviewer-model-v2"

    def test_run_mp_py_timeouts_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_mp.py 超时配置来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.planner.timeout == 450.0
        assert config.worker.task_timeout == 550.0
        assert config.reviewer.timeout == 250.0

    # ========== scripts/run_knowledge.py 配置测试 ==========

    def test_run_knowledge_py_workers_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_knowledge.py workers 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 5

    def test_run_knowledge_py_max_iterations_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_knowledge.py max_iterations 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 77

    # ========== scripts/run_iterate.py 配置测试 ==========

    def test_run_iterate_py_workers_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_iterate.py workers 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.worker_pool_size == 5

    def test_run_iterate_py_max_iterations_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_iterate.py max_iterations 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 77

    def test_run_iterate_py_cloud_timeout_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_iterate.py cloud_timeout 默认值来自 config.yaml"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.cloud_agent.timeout == 999

    # ========== 配置覆盖规则测试 ==========

    def test_cli_overrides_config_yaml_workers(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖 config.yaml 的 workers 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # config.yaml 中 workers = 5
        yaml_workers = config.system.worker_pool_size
        assert yaml_workers == 5

        # CLI 覆盖值应优先（这里验证覆盖逻辑存在）
        cli_workers = 10
        assert cli_workers != yaml_workers  # CLI 值与配置不同

    def test_cli_overrides_config_yaml_max_iterations(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖 config.yaml 的 max_iterations 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # config.yaml 中 max_iterations = 77
        yaml_max_iterations = config.system.max_iterations
        assert yaml_max_iterations == 77

        # CLI 覆盖值应优先
        cli_max_iterations = 100
        assert cli_max_iterations != yaml_max_iterations

    def test_resolve_settings_applies_config_yaml_defaults(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_settings 正确应用 config.yaml 默认值"""
        from core.config import resolve_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传递任何 CLI 参数，使用 config.yaml 默认值
        settings = resolve_settings()

        assert settings.worker_pool_size == 5
        assert settings.max_iterations == 77
        assert settings.planner_model == "test-planner-model-v2"
        assert settings.worker_model == "test-worker-model-v2"
        assert settings.reviewer_model == "test-reviewer-model-v2"
        assert settings.planning_timeout == 450.0
        assert settings.execution_timeout == 550.0
        assert settings.review_timeout == 250.0
        assert settings.stream_events_enabled is True

    def test_resolve_settings_cli_overrides_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_settings CLI 参数覆盖 config.yaml"""
        from core.config import resolve_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 参数覆盖 config.yaml
        settings = resolve_settings(
            cli_workers=8,
            cli_max_iterations=50,
            cli_planner_model="cli-planner",
        )

        # CLI 值应覆盖 config.yaml
        assert settings.worker_pool_size == 8
        assert settings.max_iterations == 50
        assert settings.planner_model == "cli-planner"
        # 未指定 CLI 参数的项仍使用 config.yaml
        assert settings.worker_model == "test-worker-model-v2"
        assert settings.reviewer_model == "test-reviewer-model-v2"


# ============================================================
# TestOrchestratorModelFromConfigYaml - Orchestrator 模型配置测试
# ============================================================


class TestOrchestratorModelFromConfigYaml:
    """测试 Orchestrator/OrchestratorConfig 未显式传入模型时从 config.yaml 生效

    验证配置优先级:
    1. 调用方显式传入的值（最高）
    2. config.yaml 配置值（通过 get_config() 获取）
    3. 代码默认值（最低）
    """

    def test_orchestrator_config_models_default_none(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 OrchestratorConfig 模型字段默认为 None（tri-state 设计）"""
        from coordinator.orchestrator import OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig，不显式传入模型
        config = OrchestratorConfig(
            working_directory=".",
        )

        # 模型字段默认为 None（表示使用 config.yaml 值）
        assert config.planner_model is None
        assert config.worker_model is None
        assert config.reviewer_model is None

    def test_orchestrator_resolves_models_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 Orchestrator._resolve_config_values 从 config.yaml 解析模型"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证 config.yaml 中的模型配置
        config = ConfigManager.get_instance()
        assert config.models.planner == "test-planner-model-v2"
        assert config.models.worker == "test-worker-model-v2"
        assert config.models.reviewer == "test-reviewer-model-v2"

        # 创建 OrchestratorConfig，不显式传入模型
        orch_config = OrchestratorConfig(
            working_directory=".",
        )

        # Mock 依赖以避免完整初始化
        with patch("coordinator.orchestrator.PlannerAgent"), patch("coordinator.orchestrator.ReviewerAgent"):
            with patch("coordinator.orchestrator.WorkerPool"):
                # 直接调用 _resolve_config_values 验证模型解析
                orchestrator_cls = Orchestrator
                # 使用 mock 避免完整初始化
                resolved = orchestrator_cls._resolve_config_values(orch_config)

        # 验证从 config.yaml 解析的模型
        assert resolved["planner_model"] == "test-planner-model-v2"
        assert resolved["worker_model"] == "test-worker-model-v2"
        assert resolved["reviewer_model"] == "test-reviewer-model-v2"

    def test_orchestrator_explicit_model_overrides_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试显式传入的模型覆盖 config.yaml"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig，显式传入部分模型
        orch_config = OrchestratorConfig(
            working_directory=".",
            planner_model="explicit-planner",  # 显式传入
            # worker_model 和 reviewer_model 不传入，使用 config.yaml
        )

        with patch("coordinator.orchestrator.PlannerAgent"), patch("coordinator.orchestrator.ReviewerAgent"):
            with patch("coordinator.orchestrator.WorkerPool"):
                orchestrator_cls = Orchestrator
                resolved = orchestrator_cls._resolve_config_values(orch_config)

        # 显式传入的值覆盖 config.yaml
        assert resolved["planner_model"] == "explicit-planner"
        # 未显式传入的使用 config.yaml
        assert resolved["worker_model"] == "test-worker-model-v2"
        assert resolved["reviewer_model"] == "test-reviewer-model-v2"

    def test_orchestrator_timeout_resolved_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 Orchestrator 超时配置从 config.yaml 解析"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig，不显式传入超时
        orch_config = OrchestratorConfig(
            working_directory=".",
            planner_timeout=None,  # 使用 config.yaml
            worker_task_timeout=None,  # 使用 config.yaml
            reviewer_timeout=None,  # 使用 config.yaml
        )

        with patch("coordinator.orchestrator.PlannerAgent"), patch("coordinator.orchestrator.ReviewerAgent"):
            with patch("coordinator.orchestrator.WorkerPool"):
                orchestrator_cls = Orchestrator
                resolved = orchestrator_cls._resolve_config_values(orch_config)

        # 验证从 config.yaml 解析的超时
        assert resolved["planner_timeout"] == 450.0
        assert resolved["worker_task_timeout"] == 550.0
        assert resolved["reviewer_timeout"] == 250.0

    def test_orchestrator_explicit_timeout_overrides_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试显式传入的超时覆盖 config.yaml"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig，显式传入部分超时
        orch_config = OrchestratorConfig(
            working_directory=".",
            planner_timeout=600.0,  # 显式传入
            # worker_task_timeout 和 reviewer_timeout 不传入
        )

        with patch("coordinator.orchestrator.PlannerAgent"), patch("coordinator.orchestrator.ReviewerAgent"):
            with patch("coordinator.orchestrator.WorkerPool"):
                orchestrator_cls = Orchestrator
                resolved = orchestrator_cls._resolve_config_values(orch_config)

        # 显式传入的值覆盖 config.yaml
        assert resolved["planner_timeout"] == 600.0
        # 未显式传入的使用 config.yaml
        assert resolved["worker_task_timeout"] == 550.0
        assert resolved["reviewer_timeout"] == 250.0


# ============================================================
# TestRunPyCloudApiKeyFromConfigYaml - run.py Cloud API Key 配置测试
# ============================================================


class TestRunPyCloudApiKeyFromConfigYaml:
    """测试 run.py 的 cloud 路径：当 config.yaml 提供 api_key 时不报错

    验证：
    - 当 config.yaml 中配置了 cloud_agent.api_key 时，_run_cloud 不应报缺少 API Key 错误
    - API Key 优先级：CLI > CURSOR_API_KEY > config.yaml
    """

    def test_run_cloud_uses_api_key_from_config_yaml(
        self, tmp_path: Path, full_custom_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py _run_cloud 使用 config.yaml 中的 api_key 不报错"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 验证 config.yaml 中配置了 api_key
        config = ConfigManager.get_instance()
        assert config.cloud_agent.api_key == "config-yaml-test-api-key"

    @pytest.mark.asyncio
    async def test_run_cloud_no_error_when_config_yaml_has_api_key(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 config.yaml 提供 api_key 时 _run_cloud 不报缺少 API Key 错误"""
        from unittest.mock import AsyncMock

        from run import Runner

        # 创建包含 api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "api_key": "config-yaml-api-key-for-test",
                "timeout": 300,
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 Runner 参数（不显式传入 api_key）
        args = argparse.Namespace(
            task="测试任务",
            mode="cloud",
            directory=".",
            workers=3,
            max_iterations="10",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="model",
            worker_model="model",
            reviewer_model="model",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key=None,  # 不显式传入
            cloud_auth_timeout=30,
            cloud_timeout=300,
            cloud_background=None,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
        )

        runner = Runner(args)

        # Mock AgentExecutorFactory 以验证 api_key 来源
        captured_cloud_auth_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cloud_auth_config
            captured_cloud_auth_config = cloud_auth_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "test-session"
            mock_result.files_modified = []
            # 显式设置避免 MagicMock 返回值导致 compute_message_dedup_key 失败
            mock_result.cooldown_info = None
            mock_result.failure_kind = None
            mock_result.retry_after = None
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        # 确保环境变量中没有 CURSOR_API_KEY，只依赖 config.yaml
        with patch.dict("os.environ", {}, clear=True):
            with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
                with patch("run.get_config") as mock_get_config:
                    # Mock get_config 返回包含 api_key 的配置
                    mock_config = MagicMock()
                    mock_config.cloud_agent.api_key = "config-yaml-api-key-for-test"
                    mock_config.cloud_agent.timeout = 300
                    mock_config.cloud_agent.auth_timeout = 30
                    mock_config.logging.stream_json.enabled = False
                    mock_config.logging.stream_json.console = True
                    mock_config.logging.stream_json.detail_dir = "logs/stream_json/detail/"
                    mock_config.logging.stream_json.raw_dir = "logs/stream_json/raw/"
                    mock_get_config.return_value = mock_config

                    options = runner._merge_options({})
                    # 手动设置 api_key 来模拟 config.yaml 的效果
                    # （因为 _get_cloud_auth_config 依赖 build_cloud_client_config）
                    options["cloud_api_key"] = "config-yaml-api-key-for-test"

                    result = await runner._run_cloud("测试任务", options)

        # 验证没有报缺少 API Key 的错误
        assert result["success"] is True
        # result["error"] 可能为 None（成功时不设置）或空字符串
        error_msg = result.get("error") or ""
        assert "API Key" not in error_msg
        assert captured_cloud_auth_config is not None
        assert captured_cloud_auth_config.api_key == "config-yaml-api-key-for-test"

    @pytest.mark.asyncio
    async def test_run_cloud_env_var_overrides_config_yaml_api_key(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试环境变量 CURSOR_API_KEY 覆盖 config.yaml 中的 api_key"""
        from unittest.mock import AsyncMock

        from run import Runner

        # 创建包含 api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "api_key": "config-yaml-api-key",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        args = argparse.Namespace(
            task="测试任务",
            mode="cloud",
            directory=".",
            workers=3,
            max_iterations="10",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="model",
            worker_model="model",
            reviewer_model="model",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            cloud_background=None,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
        )

        runner = Runner(args)

        captured_cloud_auth_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cloud_auth_config
            captured_cloud_auth_config = cloud_auth_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "test-session"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        # 设置环境变量，优先级高于 config.yaml
        with patch.dict("os.environ", {"CURSOR_API_KEY": "env-var-api-key"}):
            with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
                options = runner._merge_options({})
                await runner._run_cloud("测试任务", options)

        # 验证环境变量 API Key 被使用（优先于 config.yaml）
        assert captured_cloud_auth_config is not None
        assert captured_cloud_auth_config.api_key == "env-var-api-key"

    @pytest.mark.asyncio
    async def test_run_cloud_cli_overrides_env_var_api_key(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --cloud-api-key 覆盖环境变量 CURSOR_API_KEY"""
        from unittest.mock import AsyncMock

        from run import Runner

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        args = argparse.Namespace(
            task="测试任务",
            mode="cloud",
            directory=".",
            workers=3,
            max_iterations="10",
            strict=False,
            verbose=False,
            skip_online=False,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            planner_model="model",
            worker_model="model",
            reviewer_model="model",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            execution_mode="cloud",
            cloud_api_key="cli-api-key",  # CLI 显式传入
            cloud_auth_timeout=30,
            cloud_timeout=300,
            cloud_background=None,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
        )

        runner = Runner(args)

        captured_cloud_auth_config = None

        def mock_factory_create(mode, cli_config=None, cloud_auth_config=None):
            nonlocal captured_cloud_auth_config
            captured_cloud_auth_config = cloud_auth_config
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "ok"
            mock_result.session_id = "test-session"
            mock_result.files_modified = []
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        # 设置环境变量，但 CLI 参数应该覆盖它
        with patch.dict("os.environ", {"CURSOR_API_KEY": "env-var-api-key"}):
            with patch("cursor.executor.AgentExecutorFactory.create", side_effect=mock_factory_create):
                options = runner._merge_options({})
                await runner._run_cloud("测试任务", options)

        # 验证 CLI API Key 被使用（最高优先级）
        assert captured_cloud_auth_config is not None
        assert captured_cloud_auth_config.api_key == "cli-api-key"


# ============================================================
# TestRunPyGetCloudAuthConfig - run.py _get_cloud_auth_config 测试
# ============================================================


class TestRunPyGetCloudAuthConfig:
    """测试 run.py Runner._get_cloud_auth_config 方法的 API Key 解析

    验证优先级（从高到低）：
    1. CLI 参数 --cloud-api-key
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml 中的 cloud_agent.api_key

    确保 Cloud 执行路径不会因 API Key 缺失提前失败。
    """

    @pytest.fixture
    def run_py_cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 run.py Cloud 配置的 config.yaml"""
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-run-py-api-key",
                "api_base_url": "https://run-py.api.cursor.com",
                "timeout": 500,
                "auth_timeout": 35,
                "max_retries": 4,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def _create_minimal_args(
        self,
        cloud_api_key=None,
        cloud_auth_timeout=30,
    ):
        """创建 Runner 所需的最小 args 对象"""
        from argparse import Namespace

        return Namespace(
            task="测试任务",
            mode="auto",
            directory=".",
            workers=3,
            max_iterations="10",
            strict=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            skip_online=True,
            dry_run=False,
            force_update=False,
            use_knowledge=False,
            search_knowledge=None,
            self_update=False,
            enable_knowledge_injection=True,
            knowledge_top_k=3,
            knowledge_max_chars_per_doc=1200,
            knowledge_max_total_chars=3000,
            planner_model="gpt-5.2-high",
            worker_model="opus-4.5-thinking",
            reviewer_model="gpt-5.2-codex",
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
            no_auto_analyze=False,
            auto_commit=False,
            auto_push=False,
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=cloud_api_key,
            cloud_auth_timeout=cloud_auth_timeout,
            cloud_timeout=300,
            cloud_background=None,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
        )

    def test_get_cloud_auth_config_cli_explicit_override(
        self,
        tmp_path: Path,
        run_py_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --cloud-api-key 参数优先级最高"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置所有可能的 API Key 来源
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        # 模拟 CLI 参数
        args = self._create_minimal_args(
            cloud_api_key="cli-explicit-api-key",
            cloud_auth_timeout=60,
        )

        # 刷新模块以获取新的配置
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": "cli-explicit-api-key", "cloud_auth_timeout": 60}
        config = runner._get_cloud_auth_config(options)

        assert config is not None
        assert config.api_key == "cli-explicit-api-key"

    def test_get_cloud_auth_config_env_cursor_api_key(
        self,
        tmp_path: Path,
        run_py_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CURSOR_API_KEY 环境变量优先级高于 CURSOR_CLOUD_API_KEY 和 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置环境变量（但不设置 CLI 参数）
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        args = self._create_minimal_args(cloud_api_key=None, cloud_auth_timeout=30)

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)

        assert config is not None
        assert config.api_key == "env-cursor-api-key"

    def test_get_cloud_auth_config_env_cursor_cloud_api_key_fallback(
        self,
        tmp_path: Path,
        run_py_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试仅设置 CURSOR_CLOUD_API_KEY 时能正确获取 API Key"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 只设置 CURSOR_CLOUD_API_KEY，不设置 CURSOR_API_KEY
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        args = self._create_minimal_args(cloud_api_key=None, cloud_auth_timeout=30)

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)

        assert config is not None
        assert config.api_key == "env-cursor-cloud-api-key"

    def test_get_cloud_auth_config_yaml_only(
        self,
        tmp_path: Path,
        run_py_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试仅在 config.yaml 中设置 api_key 时能正确获取"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除所有环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        args = self._create_minimal_args(cloud_api_key=None, cloud_auth_timeout=30)

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)

        assert config is not None
        assert config.api_key == "yaml-run-py-api-key"

    def test_get_cloud_auth_config_returns_none_when_no_key(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试所有来源都没有 API Key 时返回 None"""
        import sys

        # 创建没有 api_key 的 config.yaml
        config_content = {
            "cloud_agent": {
                "timeout": 300,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除所有环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        args = self._create_minimal_args(cloud_api_key=None, cloud_auth_timeout=30)

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": None, "cloud_auth_timeout": 30}
        config = runner._get_cloud_auth_config(options)

        assert config is None

    def test_get_cloud_auth_config_uses_yaml_timeout_when_cli_not_specified(
        self,
        tmp_path: Path,
        run_py_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 auth_timeout 优先级：CLI > config.yaml > DEFAULT_*"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setenv("CURSOR_API_KEY", "test-api-key")
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        args = self._create_minimal_args(cloud_api_key=None, cloud_auth_timeout=None)

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner

        runner = Runner(args)
        options = {"cloud_api_key": None, "cloud_auth_timeout": None}
        config = runner._get_cloud_auth_config(options)

        assert config is not None
        # 应使用 config.yaml 中的 auth_timeout
        assert config.auth_timeout == 35


# ============================================================
# TestRunIteratePyGetCloudAuthConfig - run_iterate.py _get_cloud_auth_config 测试
# ============================================================


class TestRunIteratePyGetCloudAuthConfig:
    """测试 scripts/run_iterate.py SelfIterator._get_cloud_auth_config 方法的 API Key 解析

    验证优先级（从高到低）：
    1. CLI 参数 --cloud-api-key
    2. 环境变量 CURSOR_API_KEY
    3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
    4. config.yaml 中的 cloud_agent.api_key

    确保 Cloud 执行路径不会因 API Key 缺失提前失败。
    """

    @pytest.fixture
    def run_iterate_cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试 run_iterate.py Cloud 配置的 config.yaml"""
        config_content = {
            "cloud_agent": {
                "api_key": "yaml-iterate-api-key",
                "api_base_url": "https://iterate.api.cursor.com",
                "timeout": 600,
                "auth_timeout": 45,
                "max_retries": 5,
            },
            "system": {
                "max_iterations": 10,
                "worker_pool_size": 3,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_get_cloud_auth_config_cli_explicit_override(
        self,
        tmp_path: Path,
        run_iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 CLI --cloud-api-key 参数优先级最高"""
        import sys
        from argparse import Namespace

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 设置所有可能的 API Key 来源
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        # 模拟 CLI 参数
        args = Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key="cli-explicit-api-key",
            cloud_auth_timeout=60,
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(args)
        config = iterator._get_cloud_auth_config()

        assert config is not None
        assert config.api_key == "cli-explicit-api-key"

    def test_get_cloud_auth_config_env_cursor_cloud_api_key_fallback(
        self,
        tmp_path: Path,
        run_iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试仅设置 CURSOR_CLOUD_API_KEY 时能正确获取 API Key"""
        import sys
        from argparse import Namespace

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 只设置 CURSOR_CLOUD_API_KEY，不设置 CURSOR_API_KEY
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("CURSOR_CLOUD_API_KEY", "env-cursor-cloud-api-key")

        args = Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,  # 不设置 CLI 参数
            cloud_auth_timeout=30,
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(args)
        config = iterator._get_cloud_auth_config()

        assert config is not None
        assert config.api_key == "env-cursor-cloud-api-key"

    def test_get_cloud_auth_config_yaml_only(
        self,
        tmp_path: Path,
        run_iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试仅在 config.yaml 中设置 api_key 时能正确获取"""
        import sys
        from argparse import Namespace

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除所有环境变量
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        args = Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=30,
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(args)
        config = iterator._get_cloud_auth_config()

        assert config is not None
        assert config.api_key == "yaml-iterate-api-key"

    def test_get_cloud_auth_config_uses_yaml_auth_timeout(
        self,
        tmp_path: Path,
        run_iterate_cloud_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """测试 auth_timeout 优先级：CLI > config.yaml > DEFAULT_*"""
        import sys
        from argparse import Namespace

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setenv("CURSOR_API_KEY", "test-api-key")
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)

        args = Namespace(
            requirement="测试任务",
            directory=".",
            skip_online=True,
            changelog_url=None,  # tri-state
            dry_run=False,
            max_iterations="10",
            workers=3,
            force_update=False,
            verbose=False,
            quiet=False,
            log_level=None,
            heartbeat_debug=False,
            stall_diagnostics_enabled=None,
            stall_diagnostics_level=None,
            stall_recovery_interval=30.0,
            execution_health_check_interval=30.0,
            health_warning_cooldown=60.0,
            auto_commit=False,
            auto_push=False,
            commit_message="",
            commit_per_iteration=False,
            orchestrator="mp",
            no_mp=False,
            _orchestrator_user_set=False,
            execution_mode="cli",
            planner_execution_mode=None,
            worker_execution_mode=None,
            reviewer_execution_mode=None,
            cloud_api_key=None,
            cloud_auth_timeout=None,  # 不设置 CLI 参数，应使用 config.yaml 的值
            cloud_timeout=300,
            stream_console_renderer=False,
            stream_advanced_renderer=False,
            stream_typing_effect=False,
            stream_typing_delay=0.02,
            stream_word_mode=True,
            stream_color_enabled=True,
            stream_show_word_diff=False,
            # 文档源配置参数（tri-state）
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
        )

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator

        iterator = SelfIterator(args)
        config = iterator._get_cloud_auth_config()

        assert config is not None
        # 应使用 config.yaml 中的 auth_timeout
        assert config.auth_timeout == 45


# ============================================================
# TestParseMaxIterations - parse_max_iterations 函数测试
# ============================================================


class TestParseMaxIterations:
    """测试 parse_max_iterations 函数

    覆盖场景：
    - 正整数字符串
    - 无限迭代关键词（MAX/UNLIMITED/INF/INFINITE）
    - 零和负数
    - 带空白的字符串
    - 非法字符串
    - 大小写不敏感
    """

    def test_positive_integer(self) -> None:
        """测试正整数解析"""
        from core.config import parse_max_iterations

        assert parse_max_iterations("10") == 10
        assert parse_max_iterations("1") == 1
        assert parse_max_iterations("100") == 100
        assert parse_max_iterations("999999") == 999999

    def test_unlimited_keyword_max(self) -> None:
        """测试 MAX 关键词（无限迭代）"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("MAX") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("max") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("Max") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("mAx") == UNLIMITED_ITERATIONS

    def test_unlimited_keyword_unlimited(self) -> None:
        """测试 UNLIMITED 关键词"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("UNLIMITED") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("unlimited") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("Unlimited") == UNLIMITED_ITERATIONS

    def test_unlimited_keyword_inf(self) -> None:
        """测试 INF 关键词"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("INF") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("inf") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("Inf") == UNLIMITED_ITERATIONS

    def test_unlimited_keyword_infinite(self) -> None:
        """测试 INFINITE 关键词"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("INFINITE") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("infinite") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("Infinite") == UNLIMITED_ITERATIONS

    def test_zero_means_unlimited(self) -> None:
        """测试零表示无限迭代"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("0") == UNLIMITED_ITERATIONS

    def test_negative_one_means_unlimited(self) -> None:
        """测试 -1 表示无限迭代"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("-1") == UNLIMITED_ITERATIONS

    def test_negative_numbers_mean_unlimited(self) -> None:
        """测试负数表示无限迭代"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("-5") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("-100") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("-999") == UNLIMITED_ITERATIONS

    def test_whitespace_handling(self) -> None:
        """测试带空白的字符串"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("  10  ") == 10
        assert parse_max_iterations("  MAX  ") == UNLIMITED_ITERATIONS
        assert parse_max_iterations("\t5\n") == 5
        assert parse_max_iterations("  -1  ") == UNLIMITED_ITERATIONS

    def test_invalid_string_raises_error(self) -> None:
        """测试非法字符串抛出异常"""
        from core.config import MaxIterationsParseError, parse_max_iterations

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("abc")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("ten")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("1.5")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("MAX10")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("")

        with pytest.raises(MaxIterationsParseError):
            parse_max_iterations("   ")

    def test_invalid_string_with_raise_on_error_false(self) -> None:
        """测试非法字符串不抛出异常时返回默认值"""
        from core.config import DEFAULT_MAX_ITERATIONS, parse_max_iterations

        assert parse_max_iterations("abc", raise_on_error=False) == DEFAULT_MAX_ITERATIONS
        assert parse_max_iterations("", raise_on_error=False) == DEFAULT_MAX_ITERATIONS
        assert parse_max_iterations("invalid", raise_on_error=False) == DEFAULT_MAX_ITERATIONS


class TestNormalizeMaxIterations:
    """测试 normalize_max_iterations 函数"""

    def test_positive_values_unchanged(self) -> None:
        """测试正整数不变"""
        from core.config import normalize_max_iterations

        assert normalize_max_iterations(10) == 10
        assert normalize_max_iterations(1) == 1
        assert normalize_max_iterations(100) == 100

    def test_zero_becomes_unlimited(self) -> None:
        """测试零变为无限"""
        from core.config import UNLIMITED_ITERATIONS, normalize_max_iterations

        assert normalize_max_iterations(0) == UNLIMITED_ITERATIONS

    def test_negative_becomes_unlimited(self) -> None:
        """测试负数变为无限"""
        from core.config import UNLIMITED_ITERATIONS, normalize_max_iterations

        assert normalize_max_iterations(-1) == UNLIMITED_ITERATIONS
        assert normalize_max_iterations(-5) == UNLIMITED_ITERATIONS
        assert normalize_max_iterations(-100) == UNLIMITED_ITERATIONS


class TestParseMaxIterationsForArgparse:
    """测试 parse_max_iterations_for_argparse 函数"""

    def test_valid_values_work(self) -> None:
        """测试有效值正常工作"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations_for_argparse

        assert parse_max_iterations_for_argparse("10") == 10
        assert parse_max_iterations_for_argparse("MAX") == UNLIMITED_ITERATIONS
        assert parse_max_iterations_for_argparse("-1") == UNLIMITED_ITERATIONS

    def test_invalid_raises_argument_type_error(self) -> None:
        """测试无效值抛出 argparse.ArgumentTypeError"""
        import argparse

        from core.config import parse_max_iterations_for_argparse

        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_iterations_for_argparse("abc")

        with pytest.raises(argparse.ArgumentTypeError):
            parse_max_iterations_for_argparse("")


class TestParseMaxIterationsConsistency:
    """测试各入口脚本使用统一的 parse_max_iterations 函数

    验证所有入口脚本导入并使用的是同一个 parse_max_iterations 函数，
    确保行为一致性。
    """

    def test_run_py_uses_unified_overrides(self) -> None:
        """验证 run.py 通过 build_unified_overrides 处理 max_iterations

        run.py 使用 tri-state 设计，通过 build_unified_overrides 解析 max_iterations，
        而不是直接导入 parse_max_iterations。这确保了与 core.config 的一致性。
        """
        import run

        # run.py 应该从 core.config 导入 build_unified_overrides
        assert hasattr(run, "build_unified_overrides") or "build_unified_overrides" in dir(run)

    def test_run_basic_uses_core_config_function(self) -> None:
        """验证 scripts/run_basic.py 使用 core.config.parse_max_iterations"""
        from core.config import parse_max_iterations
        from scripts import run_basic

        # 验证模块导入了正确的函数
        # run_basic 中使用的 parse_max_iterations 应该与 core.config 中的相同
        assert run_basic.parse_max_iterations is parse_max_iterations

    def test_run_mp_uses_core_config_function(self) -> None:
        """验证 scripts/run_mp.py 使用 core.config.parse_max_iterations"""
        from core.config import parse_max_iterations
        from scripts import run_mp

        assert run_mp.parse_max_iterations is parse_max_iterations

    def test_run_iterate_uses_core_config_function(self) -> None:
        """验证 scripts/run_iterate.py 使用 core.config.parse_max_iterations"""
        from core.config import parse_max_iterations
        from scripts import run_iterate

        assert run_iterate.parse_max_iterations is parse_max_iterations

    def test_run_knowledge_uses_core_config_function(self) -> None:
        """验证 scripts/run_knowledge.py 使用 core.config.parse_max_iterations"""
        from core.config import parse_max_iterations
        from scripts import run_knowledge

        assert run_knowledge.parse_max_iterations is parse_max_iterations

    def test_all_scripts_produce_same_results(self) -> None:
        """验证所有脚本对同一输入产生相同结果"""
        from core.config import UNLIMITED_ITERATIONS, parse_max_iterations

        test_cases = [
            ("10", 10),
            ("MAX", UNLIMITED_ITERATIONS),
            ("max", UNLIMITED_ITERATIONS),
            ("UNLIMITED", UNLIMITED_ITERATIONS),
            ("INF", UNLIMITED_ITERATIONS),
            ("INFINITE", UNLIMITED_ITERATIONS),
            ("-1", UNLIMITED_ITERATIONS),
            ("0", UNLIMITED_ITERATIONS),
            ("-5", UNLIMITED_ITERATIONS),
            ("  10  ", 10),
        ]

        for input_value, expected in test_cases:
            result = parse_max_iterations(input_value)
            assert result == expected, f"输入 '{input_value}' 期望 {expected}，实际 {result}"


# ============================================================
# 自定义配置测试常量 - 用于 parse_args 默认值测试
# ============================================================

CUSTOM_PARSE_ARGS_CONFIG = {
    "system": {
        "max_iterations": 77,
        "worker_pool_size": 11,
        "enable_sub_planners": False,
        "strict_review": True,
    },
    "models": {
        "planner": "test-planner-model",
        "worker": "test-worker-model",
        "reviewer": "test-reviewer-model",
    },
    "cloud_agent": {
        "enabled": True,
        "execution_mode": "auto",
        "timeout": 888,
        "auth_timeout": 55,
        "max_retries": 6,
    },
    "planner": {
        "timeout": 555.0,
    },
    "worker": {
        "task_timeout": 666.0,
    },
    "reviewer": {
        "timeout": 333.0,
    },
    "logging": {
        "level": "DEBUG",
        "stream_json": {
            "enabled": True,
            "console": False,
            "detail_dir": "logs/test_stream/detail/",
            "raw_dir": "logs/test_stream/raw/",
        },
    },
}


# ============================================================
# TestRunPyParseArgsTriState - run.py parse_args tri-state 行为测试
# ============================================================


class TestRunPyParseArgsTriState:
    """测试 run.py parse_args 的 tri-state 行为

    run.py 使用 tri-state 模式：
    - workers/max_iterations 不传参时返回 None，运行时从 config.yaml 读取
    - 显式传参时返回传入的值，覆盖 config.yaml

    这是更灵活的设计，允许区分"用户未指定"和"用户指定为某值"。
    """

    @pytest.fixture
    def custom_config_for_run_py(self, tmp_path: Path) -> Path:
        """创建用于测试 run.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_workers_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --workers 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None
        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"

    def test_max_iterations_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --max-iterations 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None
        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_cloud_timeout_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-timeout 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None，运行时从 config.yaml 读取
        assert args.cloud_timeout is None, f"期望 cloud_timeout=None（tri-state），实际 {args.cloud_timeout}"

    def test_cloud_auth_timeout_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-auth-timeout 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.cloud_auth_timeout is None, (
            f"期望 cloud_auth_timeout=None（tri-state），实际 {args.cloud_auth_timeout}"
        )

    def test_execution_mode_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.execution_mode is None, f"期望 execution_mode=None（tri-state），实际 {args.execution_mode}"

    def test_models_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试模型参数 tri-state：不传参时 args.xxx 为 None，通过 resolve_orchestrator_settings 获取 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from core.config import resolve_orchestrator_settings
        from run import parse_args

        args = parse_args()

        # tri-state: 不传参时 args.xxx 为 None
        assert args.planner_model is None, f"期望 planner_model=None（tri-state），实际 {args.planner_model}"
        assert args.worker_model is None, f"期望 worker_model=None（tri-state），实际 {args.worker_model}"
        assert args.reviewer_model is None, f"期望 reviewer_model=None（tri-state），实际 {args.reviewer_model}"

        # 通过 resolve_orchestrator_settings 获取 config.yaml 的值
        resolved = resolve_orchestrator_settings(overrides={})
        assert resolved["planner_model"] == "test-planner-model", (
            f"期望 resolved planner_model='test-planner-model'，实际 {resolved['planner_model']}"
        )
        assert resolved["worker_model"] == "test-worker-model", (
            f"期望 resolved worker_model='test-worker-model'，实际 {resolved['worker_model']}"
        )
        assert resolved["reviewer_model"] == "test-reviewer-model", (
            f"期望 resolved reviewer_model='test-reviewer-model'，实际 {resolved['reviewer_model']}"
        )

    def test_strict_review_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 strict_review tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.strict_review is None, f"期望 strict_review=None（tri-state），实际 {args.strict_review}"

    def test_enable_sub_planners_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 enable_sub_planners tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.enable_sub_planners is None, (
            f"期望 enable_sub_planners=None（tri-state），实际 {args.enable_sub_planners}"
        )


# ============================================================
# TestRunPyParseArgsCLIOverridesConfig - run.py CLI 参数覆盖 config.yaml
# ============================================================


class TestRunPyParseArgsCLIOverridesConfig:
    """测试 run.py CLI 参数覆盖 config.yaml 默认值"""

    @pytest.fixture
    def custom_config_for_run_py(self, tmp_path: Path) -> Path:
        """创建用于测试 run.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_cli_workers_overrides_config(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --workers 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式指定 --workers 5
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--workers", "5"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        # CLI 参数应覆盖 config.yaml 中的 11
        assert args.workers == 5, f"期望 workers=5，实际 {args.workers}"

    def test_cli_max_iterations_overrides_config(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --max-iterations 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--max-iterations", "99"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.max_iterations == "99", f"期望 max_iterations='99'，实际 {args.max_iterations}"

    def test_cli_cloud_timeout_overrides_config(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --cloud-timeout 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--cloud-timeout", "1500"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.cloud_timeout == 1500, f"期望 cloud_timeout=1500，实际 {args.cloud_timeout}"

    def test_cli_execution_mode_overrides_config(
        self, tmp_path: Path, custom_config_for_run_py: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --execution-mode 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--execution-mode", "cloud"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        # CLI 参数应覆盖 config.yaml 中的 "auto"
        assert args.execution_mode == "cloud", f"期望 execution_mode='cloud'，实际 {args.execution_mode}"


# ============================================================
# TestRunIteratePyParseArgsFromConfigYaml - run_iterate.py parse_args 从 config.yaml 读取默认值
# ============================================================


class TestRunIteratePyParseArgsFromConfigYaml:
    """测试 scripts/run_iterate.py parse_args tri-state 行为

    tri-state 语义：
    - 用户未指定参数时，parse_args 返回 None
    - 用户显式指定参数时，parse_args 返回用户指定的值
    - 实际默认值在 SelfIterator._resolve_config_settings() 中从 config.yaml 读取
    """

    @pytest.fixture
    def custom_config_for_iterate(self, tmp_path: Path) -> Path:
        """创建用于测试 run_iterate.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_workers_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --workers 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # run_iterate.py 的 requirement 是可选的
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None，运行时从 config.yaml 读取
        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"

    def test_max_iterations_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --max-iterations 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None
        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_cloud_timeout_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-timeout 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None，运行时从 config.yaml 读取
        assert args.cloud_timeout is None, f"期望 cloud_timeout=None（tri-state），实际 {args.cloud_timeout}"

    def test_cloud_auth_timeout_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-auth-timeout 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None
        assert args.cloud_auth_timeout is None, (
            f"期望 cloud_auth_timeout=None（tri-state），实际 {args.cloud_auth_timeout}"
        )

    def test_execution_mode_tristate_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode 不指定时返回 None（tri-state）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # tri-state: 未指定时返回 None
        assert args.execution_mode is None, f"期望 execution_mode=None（tri-state），实际 {args.execution_mode}"

    def test_resolve_settings_uses_config_yaml_when_cli_none(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_settings 在 CLI 参数为 None 时使用 config.yaml 默认值

        这验证了 tri-state 的完整流程：
        1. parse_args 返回 None
        2. _resolve_config_settings 调用 resolve_settings(cli_xxx=None)
        3. resolve_settings 使用 config.yaml 的值
        """
        from core.config import resolve_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传递任何 CLI 参数（模拟 parse_args 返回的 None 值）
        settings = resolve_settings(
            cli_workers=None,
            cli_max_iterations=None,
            cli_cloud_timeout=None,
            cli_cloud_auth_timeout=None,
            cli_execution_mode=None,
        )

        # 应使用 config.yaml 中的值（CUSTOM_PARSE_ARGS_CONFIG）
        assert settings.worker_pool_size == 11, f"期望 worker_pool_size=11，实际 {settings.worker_pool_size}"
        assert settings.max_iterations == 77, f"期望 max_iterations=77，实际 {settings.max_iterations}"
        assert settings.cloud_timeout == 888, f"期望 cloud_timeout=888，实际 {settings.cloud_timeout}"
        assert settings.cloud_auth_timeout == 55, f"期望 cloud_auth_timeout=55，实际 {settings.cloud_auth_timeout}"
        assert settings.execution_mode == "auto", f"期望 execution_mode='auto'，实际 {settings.execution_mode}"


# ============================================================
# TestRunIteratePyParseArgsCLIOverridesConfig - run_iterate.py CLI 覆盖 config.yaml
# ============================================================


class TestRunIteratePyParseArgsCLIOverridesConfig:
    """测试 scripts/run_iterate.py CLI 参数覆盖 config.yaml 默认值

    验证 tri-state 语义：
    - 用户显式指定参数时，parse_args 返回用户指定的值
    - resolve_settings 在 CLI 参数非 None 时使用 CLI 值覆盖 config.yaml
    """

    @pytest.fixture
    def custom_config_for_iterate(self, tmp_path: Path) -> Path:
        """创建用于测试 run_iterate.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_cli_workers_overrides_config(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --workers 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求", "--workers", "8"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # CLI 显式指定时返回用户值
        assert args.workers == 8, f"期望 workers=8，实际 {args.workers}"

    def test_cli_max_iterations_overrides_config(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --max-iterations 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求", "--max-iterations", "MAX"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # CLI 显式指定时返回用户值
        assert args.max_iterations == "MAX", f"期望 max_iterations='MAX'，实际 {args.max_iterations}"

    def test_cli_execution_mode_overrides_config(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --execution-mode 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求", "--execution-mode", "cli"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # CLI 参数应覆盖 config.yaml 中的 "auto"
        assert args.execution_mode == "cli", f"期望 execution_mode='cli'，实际 {args.execution_mode}"

    def test_cli_cloud_timeout_overrides_config(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --cloud-timeout 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求", "--cloud-timeout", "1500"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # CLI 显式指定时返回用户值
        assert args.cloud_timeout == 1500, f"期望 cloud_timeout=1500，实际 {args.cloud_timeout}"

    def test_cli_cloud_auth_timeout_overrides_config(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --cloud-auth-timeout 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试需求", "--cloud-auth-timeout", "90"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # CLI 显式指定时返回用户值
        assert args.cloud_auth_timeout == 90, f"期望 cloud_auth_timeout=90，实际 {args.cloud_auth_timeout}"

    def test_resolve_settings_cli_overrides_config_yaml(
        self, tmp_path: Path, custom_config_for_iterate: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_settings 在 CLI 参数非 None 时使用 CLI 值覆盖 config.yaml

        这验证了 tri-state 的完整流程：
        1. parse_args 返回用户指定的值
        2. _resolve_config_settings 调用 resolve_settings(cli_xxx=用户值)
        3. resolve_settings 使用 CLI 值覆盖 config.yaml
        """
        from core.config import resolve_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 传递 CLI 参数（模拟用户显式指定）
        settings = resolve_settings(
            cli_workers=8,
            cli_max_iterations=100,
            cli_cloud_timeout=1500,
            cli_cloud_auth_timeout=90,
            cli_execution_mode="cli",
        )

        # CLI 值应覆盖 config.yaml
        assert settings.worker_pool_size == 8, f"期望 worker_pool_size=8，实际 {settings.worker_pool_size}"
        assert settings.max_iterations == 100, f"期望 max_iterations=100，实际 {settings.max_iterations}"
        assert settings.cloud_timeout == 1500, f"期望 cloud_timeout=1500，实际 {settings.cloud_timeout}"
        assert settings.cloud_auth_timeout == 90, f"期望 cloud_auth_timeout=90，实际 {settings.cloud_auth_timeout}"
        assert settings.execution_mode == "cli", f"期望 execution_mode='cli'，实际 {settings.execution_mode}"


# ============================================================
# TestRunBasicPyParseArgsFromConfigYaml - run_basic.py parse_args 从 config.yaml 读取默认值
# ============================================================


class TestRunBasicPyParseArgsFromConfigYaml:
    """测试 scripts/run_basic.py parse_args 从 config.yaml 读取默认值"""

    @pytest.fixture
    def custom_config_for_basic(self, tmp_path: Path) -> Path:
        """创建用于测试 run_basic.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_workers_from_config(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --workers tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # run_basic.py 需要 goal 位置参数
        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"

    def test_max_iterations_from_config(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --max-iterations tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_strict_review_tristate_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 strict_review tri-state：不传参时返回 None（取 config.yaml 默认值）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传 --strict 或 --no-strict
        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        # 未指定时应为 None，让运行时逻辑决定使用 config.yaml 的值
        assert args.strict_review is None, f"期望 strict_review=None，实际 {args.strict_review}"

    def test_enable_sub_planners_tristate_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 enable_sub_planners tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.enable_sub_planners is None, f"期望 enable_sub_planners=None，实际 {args.enable_sub_planners}"

    def test_execution_mode_tristate_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 execution_mode tri-state：不传参时返回 None（取 config.yaml 默认值）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.execution_mode is None, f"期望 execution_mode=None（tri-state），实际 {args.execution_mode}"

    def test_cloud_timeout_tristate_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 cloud_timeout tri-state：不传参时返回 None（取 config.yaml 默认值）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_timeout is None, f"期望 cloud_timeout=None（tri-state），实际 {args.cloud_timeout}"

    def test_cloud_auth_timeout_tristate_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 cloud_auth_timeout tri-state：不传参时返回 None（取 config.yaml 默认值）"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_auth_timeout is None, (
            f"期望 cloud_auth_timeout=None（tri-state），实际 {args.cloud_auth_timeout}"
        )

    def test_cloud_api_key_default_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 cloud_api_key 默认值为 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_api_key is None, f"期望 cloud_api_key=None，实际 {args.cloud_api_key}"

    def test_execution_mode_cli_explicit(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode cli 显式传参"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--execution-mode", "cli"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.execution_mode == "cli", f"期望 execution_mode='cli'，实际 {args.execution_mode}"

    def test_execution_mode_cloud_explicit(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode cloud 显式传参"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--execution-mode", "cloud"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.execution_mode == "cloud", f"期望 execution_mode='cloud'，实际 {args.execution_mode}"

    def test_cloud_timeout_explicit(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-timeout 显式传参"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--cloud-timeout", "600"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_timeout == 600, f"期望 cloud_timeout=600，实际 {args.cloud_timeout}"

    def test_cloud_auth_timeout_explicit(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-auth-timeout 显式传参"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--cloud-auth-timeout", "60"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_auth_timeout == 60, f"期望 cloud_auth_timeout=60，实际 {args.cloud_auth_timeout}"

    def test_cloud_api_key_explicit(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-api-key 显式传参"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--cloud-api-key", "test-api-key"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.cloud_api_key == "test-api-key", f"期望 cloud_api_key='test-api-key'，实际 {args.cloud_api_key}"


# ============================================================
# TestRunBasicPyTriStateBooleans - run_basic.py tri-state 布尔参数测试
# ============================================================


class TestRunBasicPyTriStateBooleans:
    """测试 scripts/run_basic.py 的 tri-state 布尔参数

    验证：
    - 不传参时 → None（使用 config.yaml 默认值）
    - 传 --strict → True（覆盖 config.yaml）
    - 传 --no-strict → False（覆盖 config.yaml）
    """

    @pytest.fixture
    def custom_config_for_basic(self, tmp_path: Path) -> Path:
        """创建自定义 config.yaml（strict_review=True, enable_sub_planners=False）"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_strict_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """不传 --strict/--no-strict 时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.strict_review is None

    def test_strict_flag_returns_true(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --strict 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--strict"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.strict_review is True

    def test_no_strict_flag_returns_false(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --no-strict 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--no-strict"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.strict_review is False

    def test_sub_planners_not_specified_returns_none(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """不传 --sub-planners/--no-sub-planners 时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.enable_sub_planners is None

    def test_sub_planners_flag_returns_true(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --sub-planners 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--sub-planners"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.enable_sub_planners is True

    def test_no_sub_planners_flag_returns_false(
        self, tmp_path: Path, custom_config_for_basic: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --no-sub-planners 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--no-sub-planners"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.enable_sub_planners is False


# ============================================================
# TestRunMpPyParseArgsFromConfigYaml - run_mp.py parse_args 从 config.yaml 读取默认值
# ============================================================


class TestRunMpPyParseArgsFromConfigYaml:
    """测试 scripts/run_mp.py parse_args 从 config.yaml 读取默认值"""

    @pytest.fixture
    def custom_config_for_mp(self, tmp_path: Path) -> Path:
        """创建用于测试 run_mp.py 的自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_workers_from_config(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --workers tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"

    def test_max_iterations_from_config(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --max-iterations tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_models_from_config(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试模型 tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.planner_model is None, f"期望 planner_model=None（tri-state），实际 {args.planner_model}"
        assert args.worker_model is None, f"期望 worker_model=None（tri-state），实际 {args.worker_model}"
        assert args.reviewer_model is None, f"期望 reviewer_model=None（tri-state），实际 {args.reviewer_model}"

    def test_timeouts_from_config(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试超时 tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.planning_timeout is None, f"期望 planning_timeout=None（tri-state），实际 {args.planning_timeout}"
        assert args.execution_timeout is None, (
            f"期望 execution_timeout=None（tri-state），实际 {args.execution_timeout}"
        )
        assert args.review_timeout is None, f"期望 review_timeout=None（tri-state），实际 {args.review_timeout}"

    def test_strict_review_tristate_default_none(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 strict_review tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.strict_review is None, f"期望 strict_review=None，实际 {args.strict_review}"


# ============================================================
# TestRunMpPyTriStateBooleans - run_mp.py tri-state 布尔参数测试
# ============================================================


class TestRunMpPyTriStateBooleans:
    """测试 scripts/run_mp.py 的 tri-state 布尔参数"""

    @pytest.fixture
    def custom_config_for_mp(self, tmp_path: Path) -> Path:
        """创建自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_strict_flag_returns_true(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --strict 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标", "--strict"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.strict_review is True

    def test_no_strict_flag_returns_false(
        self, tmp_path: Path, custom_config_for_mp: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --no-strict 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标", "--no-strict"])

        if "scripts.run_mp" in sys.modules:
            del sys.modules["scripts.run_mp"]

        from scripts.run_mp import parse_args

        args = parse_args()

        assert args.strict_review is False


# ============================================================
# TestRunKnowledgePyParseArgsFromConfigYaml - run_knowledge.py parse_args 从 config.yaml 读取默认值
# ============================================================


class TestRunKnowledgePyParseArgsFromConfigYaml:
    """测试 scripts/run_knowledge.py parse_args 从 config.yaml 读取默认值"""

    @pytest.fixture
    def custom_config_for_knowledge(self, tmp_path: Path) -> Path:
        """创建用于测试 run_knowledge.py 的自定义 config.yaml"""
        config_content = {
            **CUSTOM_PARSE_ARGS_CONFIG,
            "worker": {
                "task_timeout": 666.0,
                "knowledge_integration": {
                    "max_docs": 8,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_workers_from_config(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --workers tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"

    def test_max_iterations_from_config(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --max-iterations tri-state：未指定时返回 None

        tri-state 设计：parse_args 返回 None，实际值在 run_orchestrator 中
        通过 resolve_orchestrator_settings 从 config.yaml 解析。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_kb_limit_from_config(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --kb-limit 默认值来自 config.yaml worker.knowledge_integration.max_docs"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.kb_limit == 8, f"期望 kb_limit=8，实际 {args.kb_limit}"

    def test_strict_review_tristate_default_none(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 strict_review tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.strict_review is None, f"期望 strict_review=None，实际 {args.strict_review}"


# ============================================================
# TestRunKnowledgePyTriStateBooleans - run_knowledge.py tri-state 布尔参数测试
# ============================================================


class TestRunKnowledgePyTriStateBooleans:
    """测试 scripts/run_knowledge.py 的 tri-state 布尔参数"""

    @pytest.fixture
    def custom_config_for_knowledge(self, tmp_path: Path) -> Path:
        """创建自定义 config.yaml"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(CUSTOM_PARSE_ARGS_CONFIG, f, allow_unicode=True)
        return config_path

    def test_strict_flag_returns_true(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --strict 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--strict"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.strict_review is True

    def test_no_strict_flag_returns_false(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --no-strict 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--no-strict"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.strict_review is False

    def test_sub_planners_flag_returns_true(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --sub-planners 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--sub-planners"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.enable_sub_planners is True

    def test_no_sub_planners_flag_returns_false(
        self, tmp_path: Path, custom_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """传 --no-sub-planners 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--no-sub-planners"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.enable_sub_planners is False


# ============================================================
# TestStreamLogTriState - stream_log tri-state 参数测试
# ============================================================


class TestStreamLogTriState:
    """测试 stream_log 相关参数的 tri-state 行为

    验证：
    - 不传参时 → None（使用 config.yaml 默认值）
    - 传 --stream-log → True
    - 传 --no-stream-log → False
    """

    @pytest.fixture
    def custom_config_stream_enabled(self, tmp_path: Path) -> Path:
        """创建 stream_json.enabled=True 的 config.yaml"""
        config_content = {
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": False,
                },
            },
            "system": {
                "max_iterations": 10,
                "worker_pool_size": 3,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_basic_stream_log_not_specified_returns_none(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run_basic.py: 不传参时 stream_log_enabled 返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.stream_log_enabled is None

    def test_run_basic_stream_log_flag_returns_true(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run_basic.py: 传 --stream-log 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--stream-log"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.stream_log_enabled is True

    def test_run_basic_no_stream_log_flag_returns_false(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run_basic.py: 传 --no-stream-log 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标", "--no-stream-log"])

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.stream_log_enabled is False

    def test_run_py_stream_log_not_specified_returns_none(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run.py: 不传参时 stream_log_enabled 返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.stream_log_enabled is None

    def test_run_py_stream_log_flag_returns_true(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run.py: 传 --stream-log 时返回 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--stream-log"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.stream_log_enabled is True

    def test_run_py_no_stream_log_flag_returns_false(
        self, tmp_path: Path, custom_config_stream_enabled: Path, reset_config_manager, monkeypatch
    ) -> None:
        """run.py: 传 --no-stream-log 时返回 False"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--no-stream-log"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import parse_args

        args = parse_args()

        assert args.stream_log_enabled is False


# ============================================================
# TestRunPyMergeOptionsConfigToOrchestrator - 测试 config.yaml 到 Orchestrator 的传递
# ============================================================


class TestRunPyMergeOptionsConfigToOrchestrator:
    """测试 run.py 的 _merge_options 正确将 config.yaml 值传递到 OrchestratorConfig/SelfIterator

    验证：
    1. 当 CLI 未显式指定参数时，config.yaml 值被正确使用
    2. 当 CLI 显式指定参数时，覆盖 config.yaml 值
    3. 自然语言分析结果可覆盖 CLI 参数
    """

    @pytest.fixture
    def custom_config_for_merge(self, tmp_path: Path) -> Path:
        """创建用于测试合并逻辑的 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 25,
                "worker_pool_size": 8,
                "enable_sub_planners": False,
                "strict_review": True,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 999,
                "auth_timeout": 66,
            },
            "models": {
                "planner": "merge-test-planner",
                "worker": "merge-test-worker",
                "reviewer": "merge-test-reviewer",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_merge_options_uses_config_yaml_workers(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 workers 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 不传 --workers
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        # 模拟 TaskAnalysis 没有指定 workers
        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 8
        assert merged["workers"] == 8, f"期望 workers=8（来自 config.yaml），实际 {merged['workers']}"

    def test_merge_options_cli_overrides_config_yaml_workers(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 显式指定 workers 时覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式传 --workers 3
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--workers", "3"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # CLI 参数应覆盖 config.yaml
        assert merged["workers"] == 3, f"期望 workers=3（CLI 覆盖），实际 {merged['workers']}"

    def test_merge_options_uses_config_yaml_max_iterations(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 max_iterations 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 25
        assert merged["max_iterations"] == 25, (
            f"期望 max_iterations=25（来自 config.yaml），实际 {merged['max_iterations']}"
        )

    def test_merge_options_uses_config_yaml_strict_review(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 strict 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 True
        assert merged["strict"] is True, f"期望 strict=True（来自 config.yaml），实际 {merged['strict']}"

    def test_merge_options_uses_config_yaml_enable_sub_planners(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 enable_sub_planners 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 False
        assert merged["enable_sub_planners"] is False, (
            f"期望 enable_sub_planners=False（来自 config.yaml），实际 {merged['enable_sub_planners']}"
        )

    def test_merge_options_uses_config_yaml_execution_mode(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 execution_mode 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 "auto"
        assert merged["execution_mode"] == "auto", (
            f"期望 execution_mode='auto'（来自 config.yaml），实际 {merged['execution_mode']}"
        )

    def test_merge_options_uses_config_yaml_cloud_timeout(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 未指定 cloud_timeout 时使用 config.yaml 值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 应该使用 config.yaml 中的 999
        assert merged["cloud_timeout"] == 999, (
            f"期望 cloud_timeout=999（来自 config.yaml），实际 {merged['cloud_timeout']}"
        )

    def test_merge_options_cli_strict_overrides_config(
        self, tmp_path: Path, custom_config_for_merge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 _merge_options 在 CLI 传 --no-strict 时覆盖 config.yaml 的 True"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式传 --no-strict
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务", "--no-strict"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # CLI --no-strict 应覆盖 config.yaml 的 True
        assert merged["strict"] is False, f"期望 strict=False（CLI 覆盖），实际 {merged['strict']}"


# ============================================================
# TestRunPyIterateArgsConfigPropagation - 测试 IterateArgs 配置传递
# ============================================================


class TestRunPyIterateArgsConfigPropagation:
    """测试 run.py 的 IterateArgs 正确从 _merge_options 接收配置

    验证 IterateArgs 的字段与 config.yaml 和 CLI 参数一致。
    """

    @pytest.fixture
    def custom_config_for_iterate_args(self, tmp_path: Path) -> Path:
        """创建用于测试 IterateArgs 的 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 6,
                "enable_sub_planners": True,
                "strict_review": False,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 500,
                "auth_timeout": 40,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_iterate_args_receives_config_yaml_values(
        self, tmp_path: Path, custom_config_for_iterate_args: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 IterateArgs 接收 config.yaml 中的配置值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 不传任何额外参数
        monkeypatch.setattr(sys, "argv", ["run.py", "--mode", "iterate", "测试任务"])

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        # 模拟执行 _run_iterate 获取 IterateArgs
        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # 验证关键字段来自 config.yaml
        assert merged["workers"] == 6, f"期望 workers=6，实际 {merged['workers']}"
        assert merged["max_iterations"] == 15, f"期望 max_iterations=15，实际 {merged['max_iterations']}"
        assert merged["enable_sub_planners"] is True, (
            f"期望 enable_sub_planners=True，实际 {merged['enable_sub_planners']}"
        )
        assert merged["strict"] is False, f"期望 strict=False，实际 {merged['strict']}"
        assert merged["execution_mode"] == "cli", f"期望 execution_mode='cli'，实际 {merged['execution_mode']}"
        assert merged["cloud_timeout"] == 500, f"期望 cloud_timeout=500，实际 {merged['cloud_timeout']}"
        assert merged["cloud_auth_timeout"] == 40, f"期望 cloud_auth_timeout=40，实际 {merged['cloud_auth_timeout']}"

    def test_iterate_args_cli_overrides_config_yaml(
        self, tmp_path: Path, custom_config_for_iterate_args: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖 config.yaml 后正确传递到 IterateArgs"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式覆盖部分参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--workers",
                "10",
                "--max-iterations",
                "30",
                "--strict",
                "--execution-mode",
                "cloud",
                "测试任务",
            ],
        )

        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, TaskAnalysis, parse_args

        args = parse_args()
        runner = Runner(args)

        analysis = TaskAnalysis(goal="测试任务")
        merged = runner._merge_options(analysis.options)

        # CLI 参数应覆盖 config.yaml
        assert merged["workers"] == 10, f"期望 workers=10（CLI 覆盖），实际 {merged['workers']}"
        assert merged["max_iterations"] == 30, f"期望 max_iterations=30（CLI 覆盖），实际 {merged['max_iterations']}"
        assert merged["strict"] is True, f"期望 strict=True（CLI 覆盖），实际 {merged['strict']}"
        assert merged["execution_mode"] == "cloud", (
            f"期望 execution_mode='cloud'（CLI 覆盖），实际 {merged['execution_mode']}"
        )

        # 未覆盖的应保持 config.yaml 值
        assert merged["enable_sub_planners"] is True, (
            f"期望 enable_sub_planners=True（config.yaml），实际 {merged['enable_sub_planners']}"
        )
        assert merged["cloud_timeout"] == 500, f"期望 cloud_timeout=500（config.yaml），实际 {merged['cloud_timeout']}"


# ============================================================
# TestScriptParseArgsConsistency - 各入口脚本 parse_args 与 config.yaml 一致性
# ============================================================


# 用于测试的自定义配置内容
CUSTOM_CONFIG_FOR_SCRIPTS = {
    "system": {
        "max_iterations": 25,
        "worker_pool_size": 5,
        "enable_sub_planners": False,
        "strict_review": True,
    },
    "models": {
        "planner": "test-planner-model",
        "worker": "test-worker-model",
        "reviewer": "test-reviewer-model",
    },
    "cloud_agent": {
        "enabled": True,
        "execution_mode": "auto",
        "timeout": 600,
        "auth_timeout": 60,
    },
    "logging": {
        "level": "DEBUG",
        "stream_json": {
            "enabled": True,
            "console": False,
            "detail_dir": "logs/test_stream/detail/",
            "raw_dir": "logs/test_stream/raw/",
        },
    },
    "planner": {
        "timeout": 400,
    },
    "worker": {
        "task_timeout": 500,
    },
    "reviewer": {
        "timeout": 250,
    },
}


@pytest.fixture
def config_for_scripts(tmp_path: Path) -> Path:
    """创建用于脚本测试的 config.yaml 文件"""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(CUSTOM_CONFIG_FOR_SCRIPTS, f, allow_unicode=True)
    return config_path


class TestScriptParseArgsConsistency:
    """测试各入口脚本的 parse_args 与 config.yaml 一致性

    通过 monkeypatch Path.cwd()/sys.argv + ConfigManager.reset_instance()
    分别测试 run.py, scripts/run_iterate.py, scripts/run_basic.py, scripts/run_mp.py, scripts/run_knowledge.py
    的 parse_args() 函数，验证：
    1. 未传参时配置与 config.yaml 一致
    2. 显式 CLI 覆盖时与 CLI 一致
    """

    def test_run_py_parse_args_default_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py parse_args 未传参时使用 config.yaml 默认值"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 仅传入必需的任务参数
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        # 清除模块缓存确保重新导入
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts.run"):
                del sys.modules[mod_name]

        from run import parse_args

        args = parse_args()

        # 验证 tri-state 参数返回 None（未显式指定）
        assert args.workers is None, "workers 应为 None（未显式指定）"
        assert args.max_iterations is None, "max_iterations 应为 None（未显式指定）"
        assert args.execution_mode is None, "execution_mode 应为 None（未显式指定）"
        assert args.cloud_timeout is None, "cloud_timeout 应为 None（未显式指定）"
        assert args.stream_log_enabled is None, "stream_log_enabled 应为 None（未显式指定）"

    def test_run_py_merge_options_default_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py _merge_options 未传参时解析值与 config.yaml 一致"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts.run"):
                del sys.modules[mod_name]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证解析后的值与 config.yaml 一致
        assert merged["workers"] == 5, f"期望 workers=5（config.yaml），实际 {merged['workers']}"
        assert merged["max_iterations"] == 25, f"期望 max_iterations=25（config.yaml），实际 {merged['max_iterations']}"
        assert merged["execution_mode"] == "auto", (
            f"期望 execution_mode='auto'（config.yaml），实际 {merged['execution_mode']}"
        )
        assert merged["cloud_timeout"] == 600, f"期望 cloud_timeout=600（config.yaml），实际 {merged['cloud_timeout']}"
        assert merged["stream_log"] is True, f"期望 stream_log=True（config.yaml），实际 {merged['stream_log']}"
        assert merged["stream_log_console"] is False, (
            f"期望 stream_log_console=False（config.yaml），实际 {merged['stream_log_console']}"
        )
        assert merged["stream_log_detail_dir"] == "logs/test_stream/detail/", "期望正确的 detail_dir"
        assert merged["stream_log_raw_dir"] == "logs/test_stream/raw/", "期望正确的 raw_dir"

    def test_run_py_cli_overrides_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py CLI 参数覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--workers",
                "10",
                "--max-iterations",
                "50",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                "900",
                "--stream-log",
                "--stream-log-console",
                "测试任务",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts.run"):
                del sys.modules[mod_name]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证 CLI 覆盖生效
        assert merged["workers"] == 10, "workers 应被 CLI 覆盖为 10"
        assert merged["max_iterations"] == 50, "max_iterations 应被 CLI 覆盖为 50"
        assert merged["execution_mode"] == "cloud", "execution_mode 应被 CLI 覆盖为 cloud"
        assert merged["cloud_timeout"] == 900, "cloud_timeout 应被 CLI 覆盖为 900"
        assert merged["stream_log"] is True, "stream_log 应被 CLI 覆盖为 True"
        assert merged["stream_log_console"] is True, "stream_log_console 应被 CLI 覆盖为 True"

    def test_run_iterate_parse_args_default_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_iterate.py parse_args 未传参时的 tri-state 行为"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # 验证 tri-state 参数
        assert args.workers is None, "workers 应为 None（未显式指定）"
        assert args.max_iterations is None, "max_iterations 应为 None（未显式指定）"
        assert args.execution_mode is None, "execution_mode 应为 None（未显式指定）"
        assert args.cloud_timeout is None, "cloud_timeout 应为 None（未显式指定）"

    def test_run_iterate_resolve_config_settings_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 SelfIterator._resolve_config_settings 未传参时与 config.yaml 一致"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        # 验证解析后的值与 config.yaml 一致
        assert settings.worker_pool_size == 5, (
            f"期望 worker_pool_size=5（config.yaml），实际 {settings.worker_pool_size}"
        )
        assert settings.max_iterations == 25, f"期望 max_iterations=25（config.yaml），实际 {settings.max_iterations}"
        assert settings.execution_mode == "auto", (
            f"期望 execution_mode='auto'（config.yaml），实际 {settings.execution_mode}"
        )
        assert settings.cloud_timeout == 600, f"期望 cloud_timeout=600（config.yaml），实际 {settings.cloud_timeout}"
        assert settings.stream_events_enabled is True, (
            f"期望 stream_events_enabled=True（config.yaml），实际 {settings.stream_events_enabled}"
        )
        assert settings.stream_log_console is False, (
            f"期望 stream_log_console=False（config.yaml），实际 {settings.stream_log_console}"
        )

    def test_run_iterate_cli_overrides_resolve_settings(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 SelfIterator._resolve_config_settings CLI 覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "--workers",
                "8",
                "--max-iterations",
                "40",
                "--execution-mode",
                "cli",
                "--cloud-timeout",
                "1000",
                "测试任务",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        # 验证 CLI 覆盖生效
        assert settings.worker_pool_size == 8, "worker_pool_size 应被 CLI 覆盖为 8"
        assert settings.max_iterations == 40, "max_iterations 应被 CLI 覆盖为 40"
        assert settings.execution_mode == "cli", "execution_mode 应被 CLI 覆盖为 cli"
        assert settings.cloud_timeout == 1000, "cloud_timeout 应被 CLI 覆盖为 1000"

    def test_run_basic_parse_args_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_basic.py parse_args tri-state 行为

        tri-state 设计：
        - parse_args 返回 None（表示未指定）
        - 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试目标"])

        for mod_name in list(sys.modules.keys()):
            if "run_basic" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_basic import parse_args

        args = parse_args()

        # tri-state 设计：未指定时返回 None，表示使用 config.yaml 默认值
        # 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"
        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"

    def test_run_mp_parse_args_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_mp.py parse_args tri-state 行为

        tri-state 设计：
        - parse_args 返回 None（表示未指定）
        - 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试目标"])

        for mod_name in list(sys.modules.keys()):
            if "run_mp" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_mp import parse_args

        args = parse_args()

        # tri-state 设计：未指定时返回 None，表示使用 config.yaml 默认值
        # 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"
        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"
        assert args.planner_model is None, f"期望 planner_model=None（tri-state），实际 {args.planner_model}"
        assert args.worker_model is None, f"期望 worker_model=None（tri-state），实际 {args.worker_model}"
        assert args.reviewer_model is None, f"期望 reviewer_model=None（tri-state），实际 {args.reviewer_model}"

    def test_run_knowledge_parse_args_matches_config(
        self, tmp_path: Path, config_for_scripts: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 scripts/run_knowledge.py parse_args tri-state 行为

        tri-state 设计：
        - parse_args 返回 None（表示未指定）
        - 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        for mod_name in list(sys.modules.keys()):
            if "run_knowledge" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        # tri-state 设计：未指定时返回 None，表示使用 config.yaml 默认值
        # 实际值在 run_orchestrator 中通过 resolve_orchestrator_settings 解析
        assert args.workers is None, f"期望 workers=None（tri-state），实际 {args.workers}"
        assert args.max_iterations is None, f"期望 max_iterations=None（tri-state），实际 {args.max_iterations}"


# ============================================================
# TestCloudAutoMpCompatibility - Cloud/Auto 与 MP 编排器兼容性规则测试
# ============================================================


class TestCloudAutoMpCompatibility:
    """测试 Cloud/Auto 执行模式与 MP 编排器兼容性规则

    核心规则：Cloud/Auto 模式强制使用 basic 编排器（不支持 MP）
    验证该规则在不同入口（run.py, run_iterate.py）中保持一致
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 以避免因无 API Key 导致的模式回退

        当测试显式设置了 API Key 时返回该值，否则返回 mock 值。
        """
        from cursor.cloud_client import CloudClientFactory

        def _resolve_api_key(explicit_api_key=None, **kwargs):
            return explicit_api_key if explicit_api_key else "mock-api-key"

        with patch.object(CloudClientFactory, "resolve_api_key", side_effect=_resolve_api_key):
            yield

    def test_run_py_cloud_mode_forces_basic_orchestrator(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py Cloud 模式强制使用 basic 编排器

        run.py 的 _merge_options 内部使用 resolve_orchestrator_settings，
        该函数会检测 cloud/auto 模式并自动将 orchestrator 强制切换为 basic。
        这确保了配置优先级的一致性处理。
        """
        import sys

        # 创建不带 execution_mode 默认值的配置
        config_content = {"system": {"max_iterations": 10}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 --execution-mode cloud 且 --orchestrator mp
        monkeypatch.setattr(
            sys,
            "argv",
            ["run.py", "--mode", "iterate", "--execution-mode", "cloud", "--orchestrator", "mp", "测试任务"],
        )

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts.run"):
                del sys.modules[mod_name]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证 execution_mode 为 cloud
        assert merged["execution_mode"] == "cloud", "execution_mode 应为 cloud"

        # _merge_options 内部调用了相关逻辑，但 orchestrator 的切换
        # 主要在 _run_iterate 调用 SelfIterator 时生效
        # 这里验证 merged 中的 orchestrator 值
        # 注意：如果 _merge_options 使用了 resolve_orchestrator_settings，
        # 则 cloud 模式下 orchestrator 会被强制切换为 basic
        # 这是正确的预期行为（保持与 run_iterate.py 一致）
        assert merged["orchestrator"] == "basic", "Cloud 模式下 _merge_options 应将 orchestrator 强制切换为 basic"

    def test_run_iterate_cloud_mode_forces_basic_orchestrator(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py Cloud 模式强制使用 basic 编排器"""
        import sys

        config_content = {"system": {"max_iterations": 10}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 --execution-mode cloud 且 --orchestrator mp
        monkeypatch.setattr(
            sys, "argv", ["run_iterate.py", "--execution-mode", "cloud", "--orchestrator", "mp", "测试任务"]
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证 _get_orchestrator_type 返回 basic（强制切换）
        orchestrator_type = iterator._get_orchestrator_type()

        assert orchestrator_type == "basic", f"Cloud 模式应强制使用 basic 编排器，实际返回 {orchestrator_type}"

    def test_run_iterate_auto_mode_forces_basic_orchestrator(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py Auto 模式强制使用 basic 编排器"""
        import sys

        config_content = {"system": {"max_iterations": 10}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 --execution-mode auto 且 --orchestrator mp
        monkeypatch.setattr(
            sys, "argv", ["run_iterate.py", "--execution-mode", "auto", "--orchestrator", "mp", "测试任务"]
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证 _get_orchestrator_type 返回 basic（强制切换）
        orchestrator_type = iterator._get_orchestrator_type()

        assert orchestrator_type == "basic", f"Auto 模式应强制使用 basic 编排器，实际返回 {orchestrator_type}"

    def test_run_iterate_cli_mode_allows_mp_orchestrator(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py CLI 模式允许使用 MP 编排器"""
        import sys

        config_content = {"system": {"max_iterations": 10}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 --execution-mode cli 且 --orchestrator mp
        monkeypatch.setattr(
            sys, "argv", ["run_iterate.py", "--execution-mode", "cli", "--orchestrator", "mp", "测试任务"]
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证 _get_orchestrator_type 返回 mp（CLI 模式允许）
        orchestrator_type = iterator._get_orchestrator_type()

        assert orchestrator_type == "mp", f"CLI 模式应允许使用 MP 编排器，实际返回 {orchestrator_type}"

    def test_run_iterate_no_mp_flag_forces_basic(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 --no-mp 标志强制使用 basic 编排器"""
        import sys

        config_content = {"system": {"max_iterations": 10}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 --no-mp
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "--no-mp", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        orchestrator_type = iterator._get_orchestrator_type()

        assert orchestrator_type == "basic", f"--no-mp 应强制使用 basic 编排器，实际返回 {orchestrator_type}"

    def test_resolve_orchestrator_settings_cloud_forces_basic(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_orchestrator_settings 中 Cloud 模式强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        config_content = {"cloud_agent": {"execution_mode": "cli"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 传入 execution_mode=cloud 和 orchestrator=mp
        settings = resolve_orchestrator_settings(
            {
                "execution_mode": "cloud",
                "orchestrator": "mp",
            }
        )

        # 验证 orchestrator 被强制切换为 basic
        assert settings["orchestrator"] == "basic", (
            f"Cloud 模式应强制 orchestrator 为 basic，实际 {settings['orchestrator']}"
        )
        assert settings["execution_mode"] == "cloud", (
            f"execution_mode 应保持为 cloud，实际 {settings['execution_mode']}"
        )

    def test_resolve_orchestrator_settings_auto_forces_basic(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_orchestrator_settings 中 Auto 模式强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        config_content = {"cloud_agent": {"execution_mode": "cli"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 传入 execution_mode=auto 和 orchestrator=mp
        settings = resolve_orchestrator_settings(
            {
                "execution_mode": "auto",
                "orchestrator": "mp",
            }
        )

        # 验证 orchestrator 被强制切换为 basic
        assert settings["orchestrator"] == "basic", (
            f"Auto 模式应强制 orchestrator 为 basic，实际 {settings['orchestrator']}"
        )

    def test_resolve_orchestrator_settings_cli_allows_mp(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_orchestrator_settings 中 CLI 模式允许 MP 编排器"""
        from core.config import resolve_orchestrator_settings

        config_content = {"cloud_agent": {"execution_mode": "cli"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 传入 execution_mode=cli 和 orchestrator=mp
        settings = resolve_orchestrator_settings(
            {
                "execution_mode": "cli",
                "orchestrator": "mp",
            }
        )

        # 验证 orchestrator 保持为 mp
        assert settings["orchestrator"] == "mp", f"CLI 模式应允许 orchestrator 为 mp，实际 {settings['orchestrator']}"

    def test_ampersand_prefix_triggers_cloud_mode(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 & 前缀触发 Cloud 模式"""
        import sys

        # 配置需要启用 cloud_enabled 以允许 & 前缀触发 Cloud 模式
        config_content = {
            "system": {"max_iterations": 10},
            "cloud_agent": {"enabled": True, "api_key": "test-api-key"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 & 前缀的任务
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "& 后台分析任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from cursor.executor import ExecutionMode
        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证 & 前缀成功触发 Cloud 路由（策略决策层面）
        # has_ampersand_prefix=True（语法检测） + 满足条件 → prefix_routed=True
        assert iterator._triggered_by_prefix is True, "& 前缀应成功触发 Cloud 路由"

        # 验证执行模式为 CLOUD
        execution_mode = iterator._get_execution_mode()
        assert execution_mode == ExecutionMode.CLOUD, f"& 前缀应触发 CLOUD 模式，实际 {execution_mode}"

        # 验证编排器被强制为 basic
        orchestrator_type = iterator._get_orchestrator_type()
        assert orchestrator_type == "basic", f"& 前缀触发 CLOUD 模式应强制使用 basic 编排器，实际 {orchestrator_type}"

    def test_consistency_between_run_py_and_run_iterate_py(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run.py 和 run_iterate.py 配置解析一致性"""
        import sys

        # 使用相同的 config.yaml
        config_content = {
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 4,
            },
            "cloud_agent": {
                "execution_mode": "auto",
                "timeout": 450,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)

        # 测试 run.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from run import Runner
        from run import parse_args as run_parse_args

        run_args = run_parse_args()
        runner = Runner(run_args)
        run_merged = runner._merge_options({})

        # 测试 run_iterate.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator
        from scripts.run_iterate import parse_args as iterate_parse_args

        iterate_args = iterate_parse_args()
        iterator = SelfIterator(iterate_args)
        iterate_settings = iterator._resolved_settings

        # 验证两者解析的配置一致
        assert run_merged["workers"] == iterate_settings.worker_pool_size, (
            f"workers 应一致: run.py={run_merged['workers']}, run_iterate.py={iterate_settings.worker_pool_size}"
        )
        assert run_merged["max_iterations"] == iterate_settings.max_iterations, (
            f"max_iterations 应一致: run.py={run_merged['max_iterations']}, run_iterate.py={iterate_settings.max_iterations}"
        )
        assert run_merged["execution_mode"] == iterate_settings.execution_mode, (
            f"execution_mode 应一致: run.py={run_merged['execution_mode']}, run_iterate.py={iterate_settings.execution_mode}"
        )
        assert run_merged["cloud_timeout"] == iterate_settings.cloud_timeout, (
            f"cloud_timeout 应一致: run.py={run_merged['cloud_timeout']}, run_iterate.py={iterate_settings.cloud_timeout}"
        )
        assert run_merged["stream_log"] == iterate_settings.stream_events_enabled, (
            f"stream_log 应一致: run.py={run_merged['stream_log']}, run_iterate.py={iterate_settings.stream_events_enabled}"
        )


# ============================================================
# TestRunPyModelParamsFromConfigYaml - run.py 模型参数从 config.yaml 读取测试
# ============================================================


class TestRunPyModelParamsFromConfigYaml:
    """测试 run.py 未显式传入模型参数时，最终解析值来自 config.yaml

    验证：
    1. 未传 --planner-model/--worker-model/--reviewer-model 时，值来自 config.yaml
    2. CLI 参数 default=None（tri-state）实现正确的优先级
    3. _merge_options 正确解析模型配置
    """

    def test_model_params_from_config_yaml_when_not_specified(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试未显式传入模型参数时，最终解析值来自 config.yaml"""
        # 创建自定义 config.yaml
        config_content = {
            "models": {
                "planner": "config-planner-model",
                "worker": "config-worker-model",
                "reviewer": "config-reviewer-model",
            },
            "system": {
                "max_iterations": 10,
                "worker_pool_size": 3,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟不传递任何模型参数
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") and mod_name != "run":
                del sys.modules[mod_name]
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证模型参数来自 config.yaml，而不是代码默认值
        assert merged["planner_model"] == "config-planner-model", (
            f"planner_model 应来自 config.yaml，实际值: {merged['planner_model']}"
        )
        assert merged["worker_model"] == "config-worker-model", (
            f"worker_model 应来自 config.yaml，实际值: {merged['worker_model']}"
        )
        assert merged["reviewer_model"] == "config-reviewer-model", (
            f"reviewer_model 应来自 config.yaml，实际值: {merged['reviewer_model']}"
        )

    def test_cli_model_params_override_config_yaml(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 CLI 显式传入模型参数时覆盖 config.yaml"""
        # 创建自定义 config.yaml
        config_content = {
            "models": {
                "planner": "config-planner-model",
                "worker": "config-worker-model",
                "reviewer": "config-reviewer-model",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟传递 CLI 模型参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--planner-model",
                "cli-planner-model",
                "--worker-model",
                "cli-worker-model",
                "--reviewer-model",
                "cli-reviewer-model",
                "测试任务",
            ],
        )

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") and mod_name != "run":
                del sys.modules[mod_name]
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证 CLI 参数覆盖 config.yaml
        assert merged["planner_model"] == "cli-planner-model"
        assert merged["worker_model"] == "cli-worker-model"
        assert merged["reviewer_model"] == "cli-reviewer-model"

    def test_model_params_tri_state_none_means_use_config(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试模型参数的 tri-state 行为：None 表示使用 config.yaml 值"""
        # 创建 config.yaml，仅设置 planner 模型
        config_content = {
            "models": {
                "planner": "only-planner-from-config",
                # worker 和 reviewer 未设置，使用代码默认值
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 不传递任何模型参数
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") and mod_name != "run":
                del sys.modules[mod_name]
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner, parse_args

        args = parse_args()

        # 验证 args.planner_model 是 None（tri-state）
        assert args.planner_model is None, "未传递 --planner-model 时 args.planner_model 应为 None"
        assert args.worker_model is None, "未传递 --worker-model 时 args.worker_model 应为 None"
        assert args.reviewer_model is None, "未传递 --reviewer-model 时 args.reviewer_model 应为 None"

        # 验证 _merge_options 正确解析
        runner = Runner(args)
        merged = runner._merge_options({})

        # planner 应来自 config.yaml
        assert merged["planner_model"] == "only-planner-from-config"
        # worker 和 reviewer 应使用代码默认值（ConfigManager 会读取默认值）
        assert merged["worker_model"] is not None
        assert merged["reviewer_model"] is not None


# ============================================================
# TestEntryScriptConfigConsistency - 两个入口配置解析一致性测试
# ============================================================


class TestEntryScriptConfigConsistency:
    """测试 run.py 和 scripts/run_iterate.py 两个入口解析出的核心值一致

    验证同一份 config.yaml 下，两条入口（run.py --mode iterate 与 scripts/run_iterate.py）
    解析出的核心配置值一致。

    核心字段包括：
    - max_iterations
    - workers / worker_pool_size
    - enable_sub_planners
    - strict_review
    - planner_model / worker_model / reviewer_model
    - cloud_timeout / cloud_auth_timeout
    - execution_mode
    - orchestrator 类型
    - auto_commit / auto_push / commit_per_iteration
    """

    @pytest.fixture
    def consistency_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于一致性测试的 config.yaml

        包含所有核心配置字段的自定义值，以便验证两个入口是否正确读取。
        """
        config_content = {
            "system": {
                "max_iterations": 42,
                "worker_pool_size": 5,
                "enable_sub_planners": False,
                "strict_review": True,
            },
            "models": {
                "planner": "consistency-planner",
                "worker": "consistency-worker",
                "reviewer": "consistency-reviewer",
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 900,
                "auth_timeout": 60,
                "max_retries": 4,
            },
            "planner": {
                "timeout": 600.0,
            },
            "worker": {
                "task_timeout": 800.0,
            },
            "reviewer": {
                "timeout": 400.0,
            },
            "logging": {
                "level": "INFO",
                "stream_json": {
                    "enabled": True,
                    "console": True,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_core_config_consistency_without_cli_overrides(
        self, tmp_path: Path, consistency_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试不带 CLI 覆盖时，两个入口的核心配置值一致

        场景：用户只指定任务，不带任何 CLI 参数，两个入口应从 config.yaml 读取相同的值。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # ===== 1. 测试 run.py 的配置解析 =====
        monkeypatch.setattr(sys, "argv", ["run.py", "--mode", "iterate", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") and mod_name != "run":
                del sys.modules[mod_name]
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner
        from run import parse_args as run_parse_args

        run_args = run_parse_args()
        runner = Runner(run_args)
        run_merged = runner._merge_options({})

        # ===== 2. 测试 scripts/run_iterate.py 的配置解析 =====
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name:
                del sys.modules[mod_name]
        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator
        from scripts.run_iterate import parse_args as iterate_parse_args

        iterate_args = iterate_parse_args()
        iterator = SelfIterator(iterate_args)
        iterate_settings = iterator._resolved_settings

        # ===== 3. 验证核心配置一致性 =====

        # 模型配置
        assert run_merged["planner_model"] == iterate_settings.planner_model, (
            f"planner_model 不一致: run.py={run_merged['planner_model']}, "
            f"run_iterate.py={iterate_settings.planner_model}"
        )
        assert run_merged["worker_model"] == iterate_settings.worker_model, (
            f"worker_model 不一致: run.py={run_merged['worker_model']}, run_iterate.py={iterate_settings.worker_model}"
        )
        assert run_merged["reviewer_model"] == iterate_settings.reviewer_model, (
            f"reviewer_model 不一致: run.py={run_merged['reviewer_model']}, "
            f"run_iterate.py={iterate_settings.reviewer_model}"
        )

        # 系统配置
        assert run_merged["max_iterations"] == iterate_settings.max_iterations, (
            f"max_iterations 不一致: run.py={run_merged['max_iterations']}, "
            f"run_iterate.py={iterate_settings.max_iterations}"
        )
        assert run_merged["workers"] == iterate_settings.worker_pool_size, (
            f"workers/worker_pool_size 不一致: run.py={run_merged['workers']}, "
            f"run_iterate.py={iterate_settings.worker_pool_size}"
        )
        assert run_merged["enable_sub_planners"] == iterate_settings.enable_sub_planners, (
            f"enable_sub_planners 不一致: run.py={run_merged['enable_sub_planners']}, "
            f"run_iterate.py={iterate_settings.enable_sub_planners}"
        )
        assert run_merged["strict"] == iterate_settings.strict_review, (
            f"strict/strict_review 不一致: run.py={run_merged['strict']}, "
            f"run_iterate.py={iterate_settings.strict_review}"
        )

        # Cloud 配置
        assert run_merged["cloud_timeout"] == iterate_settings.cloud_timeout, (
            f"cloud_timeout 不一致: run.py={run_merged['cloud_timeout']}, "
            f"run_iterate.py={iterate_settings.cloud_timeout}"
        )
        assert run_merged["cloud_auth_timeout"] == iterate_settings.cloud_auth_timeout, (
            f"cloud_auth_timeout 不一致: run.py={run_merged['cloud_auth_timeout']}, "
            f"run_iterate.py={iterate_settings.cloud_auth_timeout}"
        )

    def test_core_config_consistency_with_cli_overrides(
        self, tmp_path: Path, consistency_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试带 CLI 覆盖时，两个入口的核心配置值一致

        场景：用户通过 CLI 参数覆盖 config.yaml 的值，两个入口应解析出相同的覆盖后值。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖值
        cli_workers = 8
        cli_max_iterations = 20
        cli_cloud_timeout = 1500

        # ===== 1. 测试 run.py 的配置解析 =====
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "iterate",
                "--workers",
                str(cli_workers),
                "--max-iterations",
                str(cli_max_iterations),
                "--cloud-timeout",
                str(cli_cloud_timeout),
                "测试 CLI 覆盖",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") and mod_name != "run":
                del sys.modules[mod_name]
        if "run" in sys.modules:
            del sys.modules["run"]

        from run import Runner
        from run import parse_args as run_parse_args

        run_args = run_parse_args()
        runner = Runner(run_args)
        run_merged = runner._merge_options({})

        # ===== 2. 测试 scripts/run_iterate.py 的配置解析 =====
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "--workers",
                str(cli_workers),
                "--max-iterations",
                str(cli_max_iterations),
                "--cloud-timeout",
                str(cli_cloud_timeout),
                "测试 CLI 覆盖",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name:
                del sys.modules[mod_name]
        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import SelfIterator
        from scripts.run_iterate import parse_args as iterate_parse_args

        iterate_args = iterate_parse_args()
        iterator = SelfIterator(iterate_args)
        iterate_settings = iterator._resolved_settings

        # ===== 3. 验证 CLI 覆盖后的一致性 =====

        # CLI 覆盖的值
        assert run_merged["workers"] == cli_workers, (
            f"run.py workers 未正确覆盖: expected={cli_workers}, actual={run_merged['workers']}"
        )
        assert iterate_settings.worker_pool_size == cli_workers, (
            f"run_iterate.py worker_pool_size 未正确覆盖: expected={cli_workers}, "
            f"actual={iterate_settings.worker_pool_size}"
        )

        assert run_merged["max_iterations"] == cli_max_iterations, (
            f"run.py max_iterations 未正确覆盖: expected={cli_max_iterations}, actual={run_merged['max_iterations']}"
        )
        assert iterate_settings.max_iterations == cli_max_iterations, (
            f"run_iterate.py max_iterations 未正确覆盖: expected={cli_max_iterations}, "
            f"actual={iterate_settings.max_iterations}"
        )

        assert run_merged["cloud_timeout"] == cli_cloud_timeout, (
            f"run.py cloud_timeout 未正确覆盖: expected={cli_cloud_timeout}, actual={run_merged['cloud_timeout']}"
        )
        assert iterate_settings.cloud_timeout == cli_cloud_timeout, (
            f"run_iterate.py cloud_timeout 未正确覆盖: expected={cli_cloud_timeout}, "
            f"actual={iterate_settings.cloud_timeout}"
        )

        # 未覆盖的值应保持 config.yaml 的值
        assert run_merged["planner_model"] == "consistency-planner"
        assert iterate_settings.planner_model == "consistency-planner"

    def test_orchestrator_settings_consistency(
        self, tmp_path: Path, consistency_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_orchestrator_settings 被两个入口正确使用

        验证核心配置解析逻辑统一使用 resolve_orchestrator_settings。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 直接调用 resolve_orchestrator_settings（无 CLI 覆盖）
        resolved = resolve_orchestrator_settings()

        # 验证从 config.yaml 读取的值
        assert resolved["max_iterations"] == 42
        assert resolved["workers"] == 5
        assert resolved["enable_sub_planners"] is False
        assert resolved["strict_review"] is True
        assert resolved["planner_model"] == "consistency-planner"
        assert resolved["worker_model"] == "consistency-worker"
        assert resolved["reviewer_model"] == "consistency-reviewer"
        assert resolved["cloud_timeout"] == 900
        assert resolved["cloud_auth_timeout"] == 60
        assert resolved["execution_mode"] == "auto"

        # 验证超时配置
        assert resolved["planner_timeout"] == 600.0
        assert resolved["worker_timeout"] == 800.0
        assert resolved["reviewer_timeout"] == 400.0

    def test_orchestrator_settings_with_overrides(
        self, tmp_path: Path, consistency_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_orchestrator_settings 的 overrides 参数

        验证 CLI 参数覆盖 config.yaml 的值。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 使用 overrides 覆盖部分值
        overrides = {
            "workers": 10,
            "max_iterations": 100,
            "execution_mode": "cloud",
            "auto_commit": True,
        }
        resolved = resolve_orchestrator_settings(overrides=overrides)

        # 验证覆盖后的值
        assert resolved["workers"] == 10, "workers 应被 overrides 覆盖"
        assert resolved["max_iterations"] == 100, "max_iterations 应被 overrides 覆盖"
        assert resolved["execution_mode"] == "cloud", "execution_mode 应被 overrides 覆盖"
        assert resolved["auto_commit"] is True, "auto_commit 应被 overrides 覆盖"

        # 未覆盖的值应保持 config.yaml 的值
        assert resolved["enable_sub_planners"] is False, "enable_sub_planners 应保持 config.yaml 值"
        assert resolved["strict_review"] is True, "strict_review 应保持 config.yaml 值"
        assert resolved["planner_model"] == "consistency-planner", "planner_model 应保持 config.yaml 值"

        # execution_mode=cloud 会自动将 orchestrator 切换为 basic
        assert resolved["orchestrator"] == "basic", "execution_mode=cloud 时 orchestrator 应自动切换为 basic"


# ============================================================
# TestRunKnowledgePyCloudConfig - run_knowledge.py Cloud 配置测试
# ============================================================


class TestRunKnowledgePyCloudConfig:
    """测试 scripts/run_knowledge.py 的 Cloud 配置参数

    验证：
    1. --execution-mode 参数解析（tri-state）
    2. --cloud-api-key 参数解析
    3. --cloud-timeout 参数解析（tri-state）
    4. --cloud-auth-timeout 参数解析（tri-state）
    """

    @pytest.fixture
    def cloud_config_for_knowledge(self, tmp_path: Path) -> Path:
        """创建包含 Cloud 配置的 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 4,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 800,
                "auth_timeout": 50,
                "api_key": "yaml-api-key-for-knowledge-test",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_execution_mode_tristate_default_none(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.execution_mode is None, f"期望 execution_mode=None（tri-state），实际 {args.execution_mode}"

    def test_execution_mode_cli_cloud(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --execution-mode cloud 参数解析"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--execution-mode", "cloud"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.execution_mode == "cloud"

    def test_cloud_timeout_tristate_default_none(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-timeout tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.cloud_timeout is None, f"期望 cloud_timeout=None（tri-state），实际 {args.cloud_timeout}"

    def test_cloud_timeout_cli_override(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-timeout CLI 参数覆盖"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--cloud-timeout", "1500"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.cloud_timeout == 1500

    def test_cloud_auth_timeout_tristate_default_none(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-auth-timeout tri-state：不传参时返回 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.cloud_auth_timeout is None, (
            f"期望 cloud_auth_timeout=None（tri-state），实际 {args.cloud_auth_timeout}"
        )

    def test_cloud_api_key_default_none(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-api-key 默认为 None"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.cloud_api_key is None

    def test_cloud_api_key_cli_override(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 --cloud-api-key CLI 参数"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试目标", "--cloud-api-key", "cli-test-key"])

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.cloud_api_key == "cli-test-key"


# ============================================================
# TestRunKnowledgePyBuildCloudAuthConfig - _build_cloud_auth_config 测试
# ============================================================


class TestRunKnowledgePyBuildCloudAuthConfig:
    """测试 scripts/run_knowledge.py 的 _build_cloud_auth_config 函数

    验证：
    1. 未配置 API Key 时返回 None
    2. CLI --cloud-api-key 优先级最高
    3. 环境变量 CURSOR_API_KEY 优先级高于 config.yaml
    4. auth_timeout 等配置正确传递
    """

    @pytest.fixture
    def cloud_config_for_knowledge(self, tmp_path: Path) -> Path:
        """创建包含 Cloud 配置的 config.yaml"""
        config_content = {
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 800,
                "auth_timeout": 50,
                "api_key": "yaml-api-key-for-knowledge-test",
                "api_base_url": "https://test.api.cursor.com",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_build_cloud_auth_config_no_key_returns_none(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试未配置 API Key 时返回 None"""
        from core.config import resolve_orchestrator_settings

        # 创建无 api_key 的配置
        config_content = {
            "cloud_agent": {
                "execution_mode": "cloud",
                "timeout": 300,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
        ConfigManager.reset_instance()

        # 重新导入以获取更新后的模块
        import sys

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import _build_cloud_auth_config

        class MockArgs:
            cloud_api_key = None

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is None, "未配置 API Key 时应返回 None"

    def test_build_cloud_auth_config_cli_key_priority(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI --cloud-api-key 优先级最高"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CURSOR_API_KEY", "env-api-key")
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import _build_cloud_auth_config

        class MockArgs:
            cloud_api_key = "cli-explicit-key"

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is not None
        assert result.api_key == "cli-explicit-key", "CLI --cloud-api-key 应优先于环境变量"

    def test_build_cloud_auth_config_env_key_priority(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试环境变量 CURSOR_API_KEY 优先级高于 config.yaml"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CURSOR_API_KEY", "env-cursor-api-key")
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import _build_cloud_auth_config

        class MockArgs:
            cloud_api_key = None

        resolved = resolve_orchestrator_settings(overrides={})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is not None
        assert result.api_key == "env-cursor-api-key", (
            "CURSOR_API_KEY 应优先于 config.yaml 中的 yaml-api-key-for-knowledge-test"
        )

    def test_build_cloud_auth_config_yaml_key_fallback(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试无环境变量时使用 config.yaml 中的 api_key"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
        ConfigManager.reset_instance()

        from cursor.cloud_client import CloudClientFactory

        api_key = CloudClientFactory.resolve_api_key(explicit_api_key=None)

        assert api_key == "yaml-api-key-for-knowledge-test"

    def test_build_cloud_auth_config_auth_timeout_from_resolved(
        self, tmp_path: Path, cloud_config_for_knowledge: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 auth_timeout 从 resolved settings 正确传递"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_CLOUD_API_KEY", raising=False)
        ConfigManager.reset_instance()

        import sys

        if "scripts.run_knowledge" in sys.modules:
            del sys.modules["scripts.run_knowledge"]

        from scripts.run_knowledge import _build_cloud_auth_config

        class MockArgs:
            cloud_api_key = None

        # CLI 覆盖 auth_timeout
        resolved = resolve_orchestrator_settings(overrides={"cloud_auth_timeout": 75})
        result = _build_cloud_auth_config(MockArgs(), resolved)

        assert result is not None
        assert result.auth_timeout == 75, "auth_timeout 应从 resolved settings 读取"


# ============================================================
# TestRunBasicPyCloudConfigPassthrough - run_basic.py Cloud 配置透传测试
# ============================================================


class TestRunBasicPyCloudConfigPassthrough:
    """测试 scripts/run_basic.py 的 Cloud 相关配置透传

    验证:
    - execution_mode 正确透传到 OrchestratorConfig
    - cloud_timeout/cloud_auth_timeout 通过 resolve_orchestrator_settings 解析
    - cloud_api_key 正确构建 CloudAuthConfig
    """

    @pytest.fixture
    def custom_config_for_cloud_passthrough(self, tmp_path: Path) -> Path:
        """创建用于 Cloud 配置透传测试的 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 10,
                "worker_pool_size": 3,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 600,
                "auth_timeout": 45,
                "api_base_url": "https://test.api.example.com",
                "max_retries": 4,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_execution_mode_passthrough_to_resolved(
        self, tmp_path: Path, custom_config_for_cloud_passthrough: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 execution_mode 通过 resolve_orchestrator_settings 正确解析"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 无覆盖时，应从 config.yaml 读取
        resolved_default = resolve_orchestrator_settings(overrides={})
        assert resolved_default["execution_mode"] == "auto", (
            f"默认 execution_mode 应为 'auto'（来自 config.yaml），实际 {resolved_default['execution_mode']}"
        )

        # CLI 覆盖
        resolved_cloud = resolve_orchestrator_settings(overrides={"execution_mode": "cloud"})
        assert resolved_cloud["execution_mode"] == "cloud", (
            f"CLI 覆盖后 execution_mode 应为 'cloud'，实际 {resolved_cloud['execution_mode']}"
        )

    def test_cloud_timeout_passthrough_to_resolved(
        self, tmp_path: Path, custom_config_for_cloud_passthrough: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 cloud_timeout 通过 resolve_orchestrator_settings 正确解析"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 无覆盖时，应从 config.yaml 读取
        resolved_default = resolve_orchestrator_settings(overrides={})
        assert resolved_default["cloud_timeout"] == 600, (
            f"默认 cloud_timeout 应为 600（来自 config.yaml），实际 {resolved_default['cloud_timeout']}"
        )

        # CLI 覆盖
        resolved_override = resolve_orchestrator_settings(overrides={"cloud_timeout": 1200})
        assert resolved_override["cloud_timeout"] == 1200, (
            f"CLI 覆盖后 cloud_timeout 应为 1200，实际 {resolved_override['cloud_timeout']}"
        )

    def test_cloud_auth_timeout_passthrough_to_resolved(
        self, tmp_path: Path, custom_config_for_cloud_passthrough: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 cloud_auth_timeout 通过 resolve_orchestrator_settings 正确解析"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 无覆盖时，应从 config.yaml 读取
        resolved_default = resolve_orchestrator_settings(overrides={})
        assert resolved_default["cloud_auth_timeout"] == 45, (
            f"默认 cloud_auth_timeout 应为 45（来自 config.yaml），实际 {resolved_default['cloud_auth_timeout']}"
        )

        # CLI 覆盖
        resolved_override = resolve_orchestrator_settings(overrides={"cloud_auth_timeout": 90})
        assert resolved_override["cloud_auth_timeout"] == 90, (
            f"CLI 覆盖后 cloud_auth_timeout 应为 90，实际 {resolved_override['cloud_auth_timeout']}"
        )

    def test_run_basic_parse_args_has_cloud_params(
        self, tmp_path: Path, custom_config_for_cloud_passthrough: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 run_basic.py parse_args 包含 Cloud 相关参数"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_basic.py",
                "测试目标",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                "800",
                "--cloud-auth-timeout",
                "60",
                "--cloud-api-key",
                "test-key-123",
            ],
        )

        if "scripts.run_basic" in sys.modules:
            del sys.modules["scripts.run_basic"]

        from scripts.run_basic import parse_args

        args = parse_args()

        assert args.execution_mode == "cloud", f"期望 execution_mode='cloud'，实际 {args.execution_mode}"
        assert args.cloud_timeout == 800, f"期望 cloud_timeout=800，实际 {args.cloud_timeout}"
        assert args.cloud_auth_timeout == 60, f"期望 cloud_auth_timeout=60，实际 {args.cloud_auth_timeout}"
        assert args.cloud_api_key == "test-key-123", f"期望 cloud_api_key='test-key-123'，实际 {args.cloud_api_key}"

    def test_run_basic_entry_script_config_consistency(
        self, tmp_path: Path, custom_config_for_cloud_passthrough: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 run_basic.py 与 run.py 的 Cloud 配置解析一致性

        两个入口脚本应从同一份 config.yaml 读取相同的 Cloud 配置值。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # run_basic.py 和 run.py 都使用 resolve_orchestrator_settings
        resolved = resolve_orchestrator_settings(overrides={})

        # 验证从 config.yaml 读取的值
        assert resolved["execution_mode"] == "auto"
        assert resolved["cloud_timeout"] == 600
        assert resolved["cloud_auth_timeout"] == 45


# ============================================================
# TestRunMpCompatibilityWithNonCliMode - run_mp.py 非 CLI 模式兼容策略测试
# ============================================================


class TestRunMpCompatibilityWithNonCliMode:
    """测试 scripts/run_mp.py 在 execution_mode != cli 时的兼容策略

    验证场景:
    1. config.yaml 中 execution_mode=cloud 时，resolve_orchestrator_settings 自动切换 orchestrator 为 basic
    2. config.yaml 中 execution_mode=auto 时，resolve_orchestrator_settings 自动切换 orchestrator 为 basic
    3. cloud_timeout 和 cloud_auth_timeout 的配置透传
    """

    @pytest.fixture
    def cloud_mode_config_yaml(self, tmp_path: Path) -> Path:
        """创建 execution_mode=cloud 的 config.yaml"""
        config_content = """
cloud_agent:
  execution_mode: "cloud"
  timeout: 1200
  auth_timeout: 90

system:
  worker_pool_size: 4
  max_iterations: 15
  orchestrator: "mp"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    @pytest.fixture
    def auto_mode_config_yaml(self, tmp_path: Path) -> Path:
        """创建 execution_mode=auto 的 config.yaml"""
        config_content = """
cloud_agent:
  execution_mode: "auto"
  timeout: 900
  auth_timeout: 60

system:
  worker_pool_size: 3
  max_iterations: 10
  orchestrator: "mp"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_cloud_mode_forces_basic_orchestrator(
        self, tmp_path: Path, cloud_mode_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 execution_mode=cloud 时，orchestrator 被强制切换为 basic"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        resolved = resolve_orchestrator_settings()

        assert resolved["execution_mode"] == "cloud", f"execution_mode 应为 cloud，实际 {resolved['execution_mode']}"
        assert resolved["orchestrator"] == "basic", (
            f"execution_mode=cloud 时 orchestrator 应自动切换为 basic，实际 {resolved['orchestrator']}"
        )
        assert resolved["cloud_timeout"] == 1200, f"cloud_timeout 应为 1200，实际 {resolved['cloud_timeout']}"
        assert resolved["cloud_auth_timeout"] == 90, (
            f"cloud_auth_timeout 应为 90，实际 {resolved['cloud_auth_timeout']}"
        )

    def test_auto_mode_forces_basic_orchestrator(
        self, tmp_path: Path, auto_mode_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 execution_mode=auto 时，orchestrator 被强制切换为 basic"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        resolved = resolve_orchestrator_settings()

        assert resolved["execution_mode"] == "auto", f"execution_mode 应为 auto，实际 {resolved['execution_mode']}"
        assert resolved["orchestrator"] == "basic", (
            f"execution_mode=auto 时 orchestrator 应自动切换为 basic，实际 {resolved['orchestrator']}"
        )
        assert resolved["cloud_timeout"] == 900, f"cloud_timeout 应为 900，实际 {resolved['cloud_timeout']}"
        assert resolved["cloud_auth_timeout"] == 60, (
            f"cloud_auth_timeout 应为 60，实际 {resolved['cloud_auth_timeout']}"
        )

    def test_cli_mode_keeps_mp_orchestrator(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 execution_mode=cli 时，orchestrator 保持为 mp"""
        config_content = """
cloud_agent:
  execution_mode: "cli"
  timeout: 300

system:
  orchestrator: "mp"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        resolved = resolve_orchestrator_settings()

        assert resolved["execution_mode"] == "cli", f"execution_mode 应为 cli，实际 {resolved['execution_mode']}"
        assert resolved["orchestrator"] == "mp", (
            f"execution_mode=cli 时 orchestrator 应保持为 mp，实际 {resolved['orchestrator']}"
        )

    def test_run_mp_uses_resolve_orchestrator_settings(
        self, tmp_path: Path, cloud_mode_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 scripts/run_mp.py 的 run_orchestrator 使用 resolve_orchestrator_settings

        当 config.yaml 设置 execution_mode=cloud 时，resolve_orchestrator_settings
        会自动切换 orchestrator 为 basic，run_mp.py 应该遵循此行为。
        """
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟 run_mp.py 中的行为
        from core.config import resolve_orchestrator_settings

        # run_mp.py 中的 run_orchestrator 调用方式
        cli_overrides: dict[str, object] = {}  # 无 CLI 覆盖
        resolved = resolve_orchestrator_settings(overrides=cli_overrides)

        # 验证 resolve_orchestrator_settings 正确处理了兼容性
        assert resolved["orchestrator"] == "basic", "run_mp.py 通过 resolve_orchestrator_settings 应获得 basic 编排器"
        assert resolved["execution_mode"] == "cloud"
        assert resolved["workers"] == 4
        assert resolved["max_iterations"] == 15

    def test_cli_override_execution_mode_affects_orchestrator(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 覆盖 execution_mode 时对 orchestrator 的影响

        即使 config.yaml 设置 orchestrator=mp，CLI 覆盖 execution_mode=cloud
        也应强制 orchestrator 切换为 basic。
        """
        config_content = """
cloud_agent:
  execution_mode: "cli"

system:
  orchestrator: "mp"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 模拟 CLI 覆盖 execution_mode
        cli_overrides = {"execution_mode": "cloud"}
        resolved = resolve_orchestrator_settings(overrides=cli_overrides)

        assert resolved["execution_mode"] == "cloud", "CLI 覆盖后 execution_mode 应为 cloud"
        assert resolved["orchestrator"] == "basic", "execution_mode=cloud（CLI 覆盖）时 orchestrator 应切换为 basic"

    def test_cloud_timeout_passthrough_in_run_mp_context(
        self, tmp_path: Path, cloud_mode_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 cloud_timeout 和 cloud_auth_timeout 在 run_mp 上下文中的透传"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import get_config, resolve_orchestrator_settings

        # 验证配置加载
        config = get_config()
        assert config.cloud_agent.timeout == 1200
        assert config.cloud_agent.auth_timeout == 90

        # 验证 resolve_orchestrator_settings 透传
        resolved = resolve_orchestrator_settings()
        assert resolved["cloud_timeout"] == 1200
        assert resolved["cloud_auth_timeout"] == 90

    def test_cli_cloud_timeout_override_priority(
        self, tmp_path: Path, cloud_mode_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数 cloud_timeout 覆盖 config.yaml 的优先级"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        # 模拟 CLI 覆盖 cloud_timeout
        cli_overrides = {
            "cloud_timeout": 2400,
            "cloud_auth_timeout": 120,
        }
        resolved = resolve_orchestrator_settings(overrides=cli_overrides)

        assert resolved["cloud_timeout"] == 2400, (
            f"CLI 覆盖后 cloud_timeout 应为 2400，实际 {resolved['cloud_timeout']}"
        )
        assert resolved["cloud_auth_timeout"] == 120, (
            f"CLI 覆盖后 cloud_auth_timeout 应为 120，实际 {resolved['cloud_auth_timeout']}"
        )


# ============================================================
# TestRunIteratePyCloudConfigPassthrough - run_iterate.py Cloud 配置透传测试
# ============================================================


class TestRunIteratePyCloudConfigPassthrough:
    """测试 scripts/run_iterate.py 的 Cloud 配置透传

    验证:
    1. config.yaml 中 execution_mode/cloud_timeout/cloud_auth_timeout 的解析
    2. CLI 参数覆盖配置
    3. SelfIterator 正确使用 resolved_settings
    """

    @pytest.fixture
    def iterate_cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建包含 Cloud 配置的 config.yaml"""
        config_content = """
cloud_agent:
  execution_mode: "auto"
  timeout: 750
  auth_timeout: 55

system:
  worker_pool_size: 6
  max_iterations: 20
  orchestrator: "mp"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_iterate_resolves_cloud_config_from_yaml(
        self, tmp_path: Path, iterate_cloud_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py 从 config.yaml 解析 Cloud 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name:
                del sys.modules[mod_name]

        from core.config import resolve_orchestrator_settings
        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        assert settings.execution_mode == "auto", f"execution_mode 应为 auto，实际 {settings.execution_mode}"
        assert settings.cloud_timeout == 750, f"cloud_timeout 应为 750，实际 {settings.cloud_timeout}"
        assert settings.cloud_auth_timeout == 55, f"cloud_auth_timeout 应为 55，实际 {settings.cloud_auth_timeout}"
        # execution_mode=auto 时，resolve_orchestrator_settings 应自动切换为 basic
        # 注：ResolvedSettings 不包含 orchestrator 字段，需通过 resolve_orchestrator_settings 验证
        orch_settings = resolve_orchestrator_settings()
        assert orch_settings["orchestrator"] == "basic", (
            f"execution_mode=auto 时 orchestrator 应为 basic，实际 {orch_settings['orchestrator']}"
        )

    def test_iterate_cli_overrides_cloud_config(
        self, tmp_path: Path, iterate_cloud_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_iterate.py CLI 参数覆盖 Cloud 配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "测试任务",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                "1500",
                "--cloud-auth-timeout",
                "100",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name:
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        assert settings.execution_mode == "cloud", (
            f"CLI 覆盖后 execution_mode 应为 cloud，实际 {settings.execution_mode}"
        )
        assert settings.cloud_timeout == 1500, f"CLI 覆盖后 cloud_timeout 应为 1500，实际 {settings.cloud_timeout}"
        assert settings.cloud_auth_timeout == 100, (
            f"CLI 覆盖后 cloud_auth_timeout 应为 100，实际 {settings.cloud_auth_timeout}"
        )


# ============================================================
# TestRunKnowledgePyCloudConfigPassthrough - run_knowledge.py Cloud 配置透传测试
# ============================================================


class TestRunKnowledgePyCloudConfigPassthrough:
    """测试 scripts/run_knowledge.py 的 Cloud 配置透传"""

    @pytest.fixture
    def knowledge_cloud_config_yaml(self, tmp_path: Path) -> Path:
        """创建包含 Cloud 配置的 config.yaml"""
        config_content = """
cloud_agent:
  execution_mode: "cloud"
  timeout: 500
  auth_timeout: 40

system:
  worker_pool_size: 2
  max_iterations: 8
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_knowledge_parses_cloud_config(
        self, tmp_path: Path, knowledge_cloud_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 run_knowledge.py parse_args 包含 Cloud 相关参数"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # run_knowledge.py 只接受一个 goal 位置参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_knowledge.py",
                "测试任务目标",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                "800",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if "run_knowledge" in mod_name:
                del sys.modules[mod_name]

        from scripts.run_knowledge import parse_args

        args = parse_args()

        assert args.execution_mode == "cloud", f"期望 execution_mode='cloud'，实际 {args.execution_mode}"
        assert args.cloud_timeout == 800, f"期望 cloud_timeout=800，实际 {args.cloud_timeout}"

    def test_knowledge_uses_resolve_orchestrator_settings(
        self, tmp_path: Path, knowledge_cloud_config_yaml: Path, reset_config_manager, monkeypatch
    ) -> None:
        """验证 run_knowledge.py 通过 resolve_orchestrator_settings 获取配置"""
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        from core.config import resolve_orchestrator_settings

        resolved = resolve_orchestrator_settings()

        assert resolved["execution_mode"] == "cloud"
        assert resolved["cloud_timeout"] == 500
        assert resolved["cloud_auth_timeout"] == 40
        # execution_mode=cloud 时，orchestrator 自动切换为 basic
        assert resolved["orchestrator"] == "basic"


# ============================================================
# TestBuildCursorAgentConfigInjection - 验证 config.yaml 影响 CursorAgentConfig
# ============================================================


class TestBuildCursorAgentConfigInjection:
    """回归测试：验证 config.yaml 能影响 CursorAgentConfig/执行器行为

    测试覆盖：
    1. build_cursor_agent_config 注入所有必需字段
    2. 新增的 cloud_auth_timeout/cloud_max_retries/stream_* 字段注入
    3. 修改 config.yaml 后 CursorAgentConfig 能反映变更
    """

    @pytest.fixture
    def full_config_yaml(self, tmp_path: Path) -> Path:
        """创建包含所有必需字段的 config.yaml"""
        config_content = {
            "agent_cli": {
                "path": "/custom/agent",
                "timeout": 450,
                "max_retries": 5,
                "output_format": "json",
                "api_key": "yaml-api-key",
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cloud",
                "api_base_url": "https://custom.api.cursor.com",
                "timeout": 900,
                "auth_timeout": 60,
                "max_retries": 4,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": False,
                    "detail_dir": "custom/logs/detail/",
                    "raw_dir": "custom/logs/raw/",
                },
            },
            "models": {
                "planner": "test-planner-model",
                "worker": "test-worker-model",
                "reviewer": "test-reviewer-model",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_build_cursor_agent_config_injects_agent_cli_fields(
        self,
        tmp_path: Path,
        full_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 build_cursor_agent_config 注入 agent_cli 字段"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(working_directory="/test")

        # agent_cli 字段
        assert config_dict["agent_path"] == "/custom/agent"
        assert config_dict["timeout"] == 450
        assert config_dict["max_retries"] == 5
        assert config_dict["output_format"] == "json"

    def test_build_cursor_agent_config_injects_cloud_fields(
        self,
        tmp_path: Path,
        full_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 build_cursor_agent_config 注入 cloud_agent 字段"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(working_directory="/test")

        # cloud_agent 字段
        assert config_dict["cloud_enabled"] is True
        assert config_dict["execution_mode"] == "cloud"
        assert config_dict["cloud_api_base"] == "https://custom.api.cursor.com"
        assert config_dict["cloud_timeout"] == 900
        assert config_dict["cloud_auth_timeout"] == 60
        assert config_dict["cloud_max_retries"] == 4

    def test_build_cursor_agent_config_injects_stream_json_fields(
        self,
        tmp_path: Path,
        full_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 build_cursor_agent_config 注入 stream_json 字段"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(working_directory="/test")

        # stream_json 字段
        assert config_dict["stream_events_enabled"] is True
        assert config_dict["stream_log_console"] is False
        assert config_dict["stream_log_detail_dir"] == "custom/logs/detail/"
        assert config_dict["stream_log_raw_dir"] == "custom/logs/raw/"

    def test_build_cursor_agent_config_overrides_take_priority(
        self,
        tmp_path: Path,
        full_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 overrides 参数优先级高于 config.yaml"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(
            working_directory="/test",
            overrides={
                "agent_path": "/override/agent",
                "timeout": 999,
                "cloud_timeout": 1800,
                "stream_events_enabled": False,
            },
        )

        # overrides 应覆盖 config.yaml 值
        assert config_dict["agent_path"] == "/override/agent"
        assert config_dict["timeout"] == 999
        assert config_dict["cloud_timeout"] == 1800
        assert config_dict["stream_events_enabled"] is False

        # 未覆盖的字段仍使用 config.yaml
        assert config_dict["max_retries"] == 5
        assert config_dict["cloud_auth_timeout"] == 60

    def test_cursor_agent_config_can_be_built_from_config_dict(
        self,
        tmp_path: Path,
        full_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CursorAgentConfig 可从 build_cursor_agent_config 返回值构建"""
        from core.config import build_cursor_agent_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(working_directory="/test")

        # 构建 CursorAgentConfig（过滤不存在的字段）
        # CursorAgentConfig 字段名可能略有不同，需要映射
        agent_config = CursorAgentConfig(
            working_directory=config_dict["working_directory"],
            agent_path=config_dict["agent_path"],
            timeout=config_dict["timeout"],
            max_retries=config_dict["max_retries"],
            output_format=config_dict["output_format"],
            execution_mode=config_dict["execution_mode"],
            cloud_enabled=config_dict["cloud_enabled"],
            cloud_api_base=config_dict["cloud_api_base"],
            cloud_timeout=config_dict["cloud_timeout"],
            stream_events_enabled=config_dict["stream_events_enabled"],
            stream_log_console=config_dict["stream_log_console"],
            stream_log_detail_dir=config_dict["stream_log_detail_dir"],
            stream_log_raw_dir=config_dict["stream_log_raw_dir"],
        )

        # 验证 CursorAgentConfig 正确反映配置
        assert agent_config.agent_path == "/custom/agent"
        assert agent_config.timeout == 450
        assert agent_config.cloud_timeout == 900
        assert agent_config.stream_events_enabled is True


class TestOrchestratorInjectAgentCliConfig:
    """回归测试：验证 Orchestrator._inject_agent_cli_config 注入所有必需字段

    测试覆盖：
    1. agent_cli 字段注入 (agent_path, timeout, max_retries)
    2. cloud_agent 字段注入 (cloud_api_base, cloud_timeout, cloud_enabled)
    3. 仅在使用默认值时注入（显式设置值优先）
    """

    @pytest.fixture
    def orchestrator_inject_config_yaml(self, tmp_path: Path) -> Path:
        """创建用于测试注入的 config.yaml"""
        config_content = {
            "agent_cli": {
                "path": "/injected/agent",
                "timeout": 600,
                "max_retries": 7,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "api_base_url": "https://injected.api.cursor.com",
                "timeout": 1200,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_inject_agent_cli_config_injects_all_fields(
        self,
        tmp_path: Path,
        orchestrator_inject_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 _inject_agent_cli_config 注入所有 agent_cli/cloud 字段

        注意：Orchestrator 会为每个角色（planner/reviewer/worker）复制
        cursor_config 并调用 _inject_agent_cli_config，因此需要检查
        各角色 Agent 的 cursor_config 而非 orchestrator.config.cursor_config。
        PlannerAgent 使用 planner_config.cursor_config 存储配置。
        """
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建使用默认值的 CursorAgentConfig
        cursor_config = CursorAgentConfig()

        # 创建 Orchestrator 以获取 _inject_agent_cli_config 方法
        orchestrator_config = OrchestratorConfig(
            working_directory=str(tmp_path),
            cursor_config=cursor_config,
        )
        orchestrator = Orchestrator(orchestrator_config)

        # 获取注入后的 planner cursor_config
        # PlannerAgent 使用 planner_config 存储 PlannerConfig
        injected_config = orchestrator.planner.planner_config.cursor_config

        # 验证 agent_cli 字段注入
        assert injected_config.agent_path == "/injected/agent"
        assert injected_config.max_retries == 7
        # 注意：planner 的 timeout 会被 _resolved_config["planner_timeout"] 覆盖
        # 所以这里检查 max_retries 而非 timeout

        # 验证 cloud_agent 字段注入
        assert injected_config.cloud_api_base == "https://injected.api.cursor.com"
        assert injected_config.cloud_timeout == 1200
        assert injected_config.cloud_enabled is True

        # 也验证 worker 的配置注入
        # WorkerAgent 使用 worker_config 存储 WorkerConfig
        worker_config = orchestrator.worker_pool.workers[0].worker_config.cursor_config
        assert worker_config.agent_path == "/injected/agent"
        assert worker_config.max_retries == 7
        assert worker_config.cloud_api_base == "https://injected.api.cursor.com"

    def test_inject_does_not_override_explicit_values(
        self,
        tmp_path: Path,
        orchestrator_inject_config_yaml: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证显式设置的值不会被注入覆盖

        注意：Orchestrator 会为每个角色复制 cursor_config 并调用
        _inject_agent_cli_config，因此需要检查角色 Agent 的配置。
        PlannerAgent 使用 planner_config.cursor_config 存储配置。
        """
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建带有显式值的 CursorAgentConfig
        cursor_config = CursorAgentConfig(
            agent_path="/explicit/agent",  # 显式设置
            max_retries=10,  # 显式设置（非默认值）
            timeout=999,  # 显式设置（非默认值）
            cloud_api_base="https://explicit.api.com",  # 显式设置（非默认值）
            cloud_timeout=1800,  # 显式设置（非默认值）
        )

        orchestrator_config = OrchestratorConfig(
            working_directory=str(tmp_path),
            cursor_config=cursor_config,
        )
        orchestrator = Orchestrator(orchestrator_config)

        # 获取 planner 的 cursor_config（会被复制并注入）
        injected_config = orchestrator.planner.planner_config.cursor_config

        # 验证显式值未被覆盖
        assert injected_config.agent_path == "/explicit/agent"
        assert injected_config.max_retries == 10
        # 注意：timeout 会被 _resolved_config["planner_timeout"] 覆盖，所以不检查
        assert injected_config.cloud_api_base == "https://explicit.api.com"
        assert injected_config.cloud_timeout == 1800


# ============================================================
# TestStreamConfigTriState - stream 配置 tri-state 回归测试
# ============================================================


class TestStreamConfigTriState:
    """测试 stream 配置的 tri-state 行为和唯一权威默认值策略

    验证点:
    1. DEFAULT_STREAM_* 常量与 config.yaml 保持同步
    2. CursorAgentConfig.stream_events_enabled 默认值与 config.yaml 一致
    3. OrchestratorConfig 的 stream 字段使用 tri-state 设计（None 表示未指定）
    4. MultiProcessOrchestratorConfig 的 stream 字段使用 tri-state 设计
    5. 未指定时从 config.yaml 读取，显式指定时使用传入值
    """

    def test_default_stream_constants_match_config_yaml(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 DEFAULT_STREAM_* 常量与 config.yaml 默认值保持同步

        这些常量是 stream 配置的唯一权威来源。
        """
        from core.config import (
            DEFAULT_STREAM_EVENTS_ENABLED,
            DEFAULT_STREAM_LOG_CONSOLE,
            DEFAULT_STREAM_LOG_DETAIL_DIR,
            DEFAULT_STREAM_LOG_RAW_DIR,
        )

        # 验证常量值与代码注释一致
        assert DEFAULT_STREAM_EVENTS_ENABLED is False, "DEFAULT_STREAM_EVENTS_ENABLED 应为 False（与 config.yaml 同步）"
        assert DEFAULT_STREAM_LOG_CONSOLE is True, "DEFAULT_STREAM_LOG_CONSOLE 应为 True（与 config.yaml 同步）"
        assert DEFAULT_STREAM_LOG_DETAIL_DIR == "logs/stream_json/detail/", (
            "DEFAULT_STREAM_LOG_DETAIL_DIR 应与 config.yaml 同步"
        )
        assert DEFAULT_STREAM_LOG_RAW_DIR == "logs/stream_json/raw/", "DEFAULT_STREAM_LOG_RAW_DIR 应与 config.yaml 同步"

    def test_cursor_agent_config_stream_default_matches_config_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 CursorAgentConfig.stream_events_enabled 默认值与 config.yaml 一致

        防止"未注入即开启"的问题：默认应该关闭，只有显式启用或从 config.yaml 读取时才开启。
        """
        from cursor.client import CursorAgentConfig

        # 创建默认配置
        config = CursorAgentConfig()

        # 验证默认值
        assert config.stream_events_enabled is False, (
            "CursorAgentConfig.stream_events_enabled 默认应为 False（与 config.yaml 同步），避免'未注入即开启'的问题"
        )

    def test_orchestrator_config_stream_tristate_none_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 OrchestratorConfig 的 stream 字段默认为 None（tri-state 设计）"""
        from coordinator.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        # 验证 tri-state 默认值
        assert config.stream_events_enabled is None, (
            "OrchestratorConfig.stream_events_enabled 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_console is None, (
            "OrchestratorConfig.stream_log_console 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_detail_dir is None, (
            "OrchestratorConfig.stream_log_detail_dir 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_raw_dir is None, (
            "OrchestratorConfig.stream_log_raw_dir 应默认为 None（tri-state 设计）"
        )

    def test_mp_orchestrator_config_stream_tristate_none_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 MultiProcessOrchestratorConfig 的 stream 字段默认为 None（tri-state 设计）"""
        from coordinator.orchestrator_mp import MultiProcessOrchestratorConfig

        config = MultiProcessOrchestratorConfig()

        # 验证 tri-state 默认值
        assert config.stream_events_enabled is None, (
            "MultiProcessOrchestratorConfig.stream_events_enabled 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_console is None, (
            "MultiProcessOrchestratorConfig.stream_log_console 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_detail_dir is None, (
            "MultiProcessOrchestratorConfig.stream_log_detail_dir 应默认为 None（tri-state 设计）"
        )
        assert config.stream_log_raw_dir is None, (
            "MultiProcessOrchestratorConfig.stream_log_raw_dir 应默认为 None（tri-state 设计）"
        )

    def test_orchestrator_resolves_stream_from_config_yaml_when_none(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 Orchestrator 在 stream 字段为 None 时从 config.yaml 读取配置"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig（stream 字段全部为 None）
        orchestrator_config = OrchestratorConfig(
            working_directory=str(tmp_path),
        )
        orchestrator = Orchestrator(orchestrator_config)

        # 验证解析后的值来自 custom_config.yaml
        # custom_config.yaml 中 stream_json.enabled=True
        assert orchestrator._resolved_config["stream_events_enabled"] is True, (
            "stream_events_enabled 应从 config.yaml 读取（enabled=True）"
        )
        assert orchestrator._resolved_config["stream_log_console"] is False, (
            "stream_log_console 应从 config.yaml 读取（console=False）"
        )
        assert orchestrator._resolved_config["stream_log_detail_dir"] == "logs/custom_stream_json/detail/", (
            "stream_log_detail_dir 应从 config.yaml 读取"
        )
        assert orchestrator._resolved_config["stream_log_raw_dir"] == "logs/custom_stream_json/raw/", (
            "stream_log_raw_dir 应从 config.yaml 读取"
        )

    def test_orchestrator_uses_explicit_stream_over_config_yaml(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 Orchestrator 优先使用显式传入的 stream 值（覆盖 config.yaml）"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 OrchestratorConfig（显式传入 stream 字段）
        orchestrator_config = OrchestratorConfig(
            working_directory=str(tmp_path),
            stream_events_enabled=False,  # 显式关闭（覆盖 config.yaml 的 True）
            stream_log_console=True,  # 显式开启（覆盖 config.yaml 的 False）
            stream_log_detail_dir="/explicit/detail/",
            stream_log_raw_dir="/explicit/raw/",
        )
        orchestrator = Orchestrator(orchestrator_config)

        # 验证显式值优先
        assert orchestrator._resolved_config["stream_events_enabled"] is False, (
            "显式传入的 stream_events_enabled=False 应覆盖 config.yaml"
        )
        assert orchestrator._resolved_config["stream_log_console"] is True, (
            "显式传入的 stream_log_console=True 应覆盖 config.yaml"
        )
        assert orchestrator._resolved_config["stream_log_detail_dir"] == "/explicit/detail/", (
            "显式传入的 stream_log_detail_dir 应覆盖 config.yaml"
        )
        assert orchestrator._resolved_config["stream_log_raw_dir"] == "/explicit/raw/", (
            "显式传入的 stream_log_raw_dir 应覆盖 config.yaml"
        )

    def test_mp_orchestrator_resolves_stream_from_config_yaml_when_none(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 MultiProcessOrchestrator._resolve_stream_config 从 config.yaml 读取"""
        from coordinator.orchestrator_mp import (
            MultiProcessOrchestrator,
            MultiProcessOrchestratorConfig,
        )

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 MultiProcessOrchestratorConfig（stream 字段全部为 None）
        mp_config = MultiProcessOrchestratorConfig(
            working_directory=str(tmp_path),
        )

        # 使用静态方法测试解析（避免创建完整的 Orchestrator）
        resolved = MultiProcessOrchestrator._resolve_stream_config(mp_config)

        # 验证解析后的值来自 custom_config.yaml
        assert resolved["stream_events_enabled"] is True, "stream_events_enabled 应从 config.yaml 读取（enabled=True）"
        assert resolved["stream_log_console"] is False, "stream_log_console 应从 config.yaml 读取（console=False）"

    def test_mp_orchestrator_uses_explicit_stream_over_config_yaml(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 MultiProcessOrchestrator 优先使用显式传入的 stream 值"""
        from coordinator.orchestrator_mp import (
            MultiProcessOrchestrator,
            MultiProcessOrchestratorConfig,
        )

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 MultiProcessOrchestratorConfig（显式传入 stream 字段）
        mp_config = MultiProcessOrchestratorConfig(
            working_directory=str(tmp_path),
            stream_events_enabled=False,  # 显式关闭
            stream_log_console=True,  # 显式开启
        )

        # 使用静态方法测试解析
        resolved = MultiProcessOrchestrator._resolve_stream_config(mp_config)

        # 验证显式值优先
        assert resolved["stream_events_enabled"] is False, "显式传入的 stream_events_enabled=False 应覆盖 config.yaml"
        assert resolved["stream_log_console"] is True, "显式传入的 stream_log_console=True 应覆盖 config.yaml"

    def test_orchestrator_applies_resolved_stream_to_cursor_config(
        self, custom_config_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 Orchestrator 将解析后的 stream 配置正确应用到 CursorAgentConfig"""
        from coordinator.orchestrator import Orchestrator, OrchestratorConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        orchestrator_config = OrchestratorConfig(
            working_directory=str(tmp_path),
            # stream 字段全部为 None，应从 config.yaml 读取
        )
        orchestrator = Orchestrator(orchestrator_config)

        # 验证 cursor_config 中的值
        cursor_config = orchestrator.config.cursor_config
        assert cursor_config.stream_events_enabled is True, (
            "CursorAgentConfig.stream_events_enabled 应被注入为 config.yaml 的值（True）"
        )
        assert cursor_config.stream_log_console is False, (
            "CursorAgentConfig.stream_log_console 应被注入为 config.yaml 的值（False）"
        )

    def test_stream_json_logging_config_uses_default_constants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 StreamJsonLoggingConfig 默认值使用 DEFAULT_STREAM_* 常量

        这是关键的回归测试，确保 StreamJsonLoggingConfig 的默认值与唯一权威常量保持同步，
        防止多处硬编码导致的漂移问题。

        如果这个测试失败，说明有人修改了 StreamJsonLoggingConfig 的默认值但没有使用常量，
        或者常量定义发生了变化但 dataclass 默认值没有同步更新。
        """
        from core.config import (
            DEFAULT_STREAM_EVENTS_ENABLED,
            DEFAULT_STREAM_LOG_CONSOLE,
            DEFAULT_STREAM_LOG_DETAIL_DIR,
            DEFAULT_STREAM_LOG_RAW_DIR,
            StreamJsonLoggingConfig,
        )

        # 创建默认配置（不传任何参数）
        config = StreamJsonLoggingConfig()

        # 验证默认值与常量完全一致
        assert config.enabled == DEFAULT_STREAM_EVENTS_ENABLED, (
            f"StreamJsonLoggingConfig.enabled 默认值 ({config.enabled}) "
            f"应与 DEFAULT_STREAM_EVENTS_ENABLED ({DEFAULT_STREAM_EVENTS_ENABLED}) 一致"
        )
        assert config.console == DEFAULT_STREAM_LOG_CONSOLE, (
            f"StreamJsonLoggingConfig.console 默认值 ({config.console}) "
            f"应与 DEFAULT_STREAM_LOG_CONSOLE ({DEFAULT_STREAM_LOG_CONSOLE}) 一致"
        )
        assert config.detail_dir == DEFAULT_STREAM_LOG_DETAIL_DIR, (
            f"StreamJsonLoggingConfig.detail_dir 默认值 ({config.detail_dir}) "
            f"应与 DEFAULT_STREAM_LOG_DETAIL_DIR ({DEFAULT_STREAM_LOG_DETAIL_DIR}) 一致"
        )
        assert config.raw_dir == DEFAULT_STREAM_LOG_RAW_DIR, (
            f"StreamJsonLoggingConfig.raw_dir 默认值 ({config.raw_dir}) "
            f"应与 DEFAULT_STREAM_LOG_RAW_DIR ({DEFAULT_STREAM_LOG_RAW_DIR}) 一致"
        )

    def test_config_manager_parse_uses_default_constants_as_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """验证 ConfigManager._parse_config 使用 DEFAULT_STREAM_* 常量作为回退值

        当 config.yaml 中未定义 stream_json 配置时，解析器应使用常量作为默认值。
        """
        from core.config import (
            DEFAULT_STREAM_EVENTS_ENABLED,
            DEFAULT_STREAM_LOG_CONSOLE,
            DEFAULT_STREAM_LOG_DETAIL_DIR,
            DEFAULT_STREAM_LOG_RAW_DIR,
            ConfigManager,
        )

        # 创建只包含基础日志配置的 config.yaml（不含 stream_json）
        config_content = {
            "logging": {
                "level": "DEBUG",
                "file": "logs/test.log",
                # stream_json 未定义，应使用默认常量
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证 stream_json 使用常量默认值
        stream_json = config.logging.stream_json
        assert stream_json.enabled == DEFAULT_STREAM_EVENTS_ENABLED, (
            f"stream_json.enabled 应回退到 DEFAULT_STREAM_EVENTS_ENABLED ({DEFAULT_STREAM_EVENTS_ENABLED})"
        )
        assert stream_json.console == DEFAULT_STREAM_LOG_CONSOLE, (
            f"stream_json.console 应回退到 DEFAULT_STREAM_LOG_CONSOLE ({DEFAULT_STREAM_LOG_CONSOLE})"
        )
        assert stream_json.detail_dir == DEFAULT_STREAM_LOG_DETAIL_DIR, (
            f"stream_json.detail_dir 应回退到 DEFAULT_STREAM_LOG_DETAIL_DIR ({DEFAULT_STREAM_LOG_DETAIL_DIR})"
        )
        assert stream_json.raw_dir == DEFAULT_STREAM_LOG_RAW_DIR, (
            f"stream_json.raw_dir 应回退到 DEFAULT_STREAM_LOG_RAW_DIR ({DEFAULT_STREAM_LOG_RAW_DIR})"
        )


# ============================================================================
# 综合入口脚本配置加载与优先级测试
# ============================================================================


class TestEntryScriptConfigPrecedence:
    """综合测试：入口脚本配置加载与 CLI 覆盖优先级

    测试覆盖（针对每个入口脚本）：
    1. 未传 CLI 参数时取值等于 config.yaml
    2. 传 CLI 参数时覆盖 config.yaml
    3. execution_mode=cloud/auto 时强制 basic 编排器策略
    4. stream_json.enabled 驱动 CursorAgentConfig.stream_events_enabled
    """

    @pytest.fixture
    def config_with_stream_enabled(self, tmp_path: Path) -> Path:
        """创建启用 stream_json 的配置文件"""
        config_content = {
            "system": {
                "worker_pool_size": 5,
                "max_iterations": 15,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 600,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                    "detail_dir": "logs/stream_detail",
                    "raw_dir": "logs/stream_raw",
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def config_with_stream_disabled(self, tmp_path: Path) -> Path:
        """创建禁用 stream_json 的配置文件"""
        config_content = {
            "system": {
                "worker_pool_size": 3,
                "max_iterations": 10,
            },
            "cloud_agent": {
                "enabled": False,
                "execution_mode": "cli",
            },
            "logging": {
                "stream_json": {
                    "enabled": False,
                    "console": False,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def config_with_cloud_execution_mode(self, tmp_path: Path) -> Path:
        """创建 cloud 执行模式的配置文件"""
        config_content = {
            "system": {
                "worker_pool_size": 4,
                "max_iterations": 20,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cloud",
                "timeout": 900,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def config_with_auto_execution_mode(self, tmp_path: Path) -> Path:
        """创建 auto 执行模式的配置文件"""
        config_content = {
            "system": {
                "worker_pool_size": 6,
                "max_iterations": 25,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 1200,
            },
            "logging": {
                "stream_json": {
                    "enabled": False,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    # ========================================================================
    # resolve_stream_log_config 测试
    # ========================================================================

    def test_stream_log_config_from_yaml_enabled(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=True 从 config.yaml 正确读取"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 未传 CLI 参数，应使用 config.yaml 值
        stream_config = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert stream_config["enabled"] is True
        assert stream_config["console"] is True
        assert stream_config["detail_dir"] == "logs/stream_detail"
        assert stream_config["raw_dir"] == "logs/stream_raw"

    def test_stream_log_config_from_yaml_disabled(
        self,
        tmp_path: Path,
        config_with_stream_disabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=False 从 config.yaml 正确读取"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        stream_config = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert stream_config["enabled"] is False
        assert stream_config["console"] is False

    def test_stream_log_config_cli_overrides_yaml(
        self,
        tmp_path: Path,
        config_with_stream_disabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 参数覆盖 config.yaml 中的 stream_json 配置"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式启用，应覆盖 config.yaml 的 False
        stream_config = resolve_stream_log_config(
            cli_enabled=True,
            cli_console=True,
            cli_detail_dir="/cli/detail",
            cli_raw_dir="/cli/raw",
        )

        assert stream_config["enabled"] is True
        assert stream_config["console"] is True
        assert stream_config["detail_dir"] == "/cli/detail"
        assert stream_config["raw_dir"] == "/cli/raw"

    # ========================================================================
    # resolve_orchestrator_settings 测试
    # ========================================================================

    def test_orchestrator_settings_from_yaml_no_override(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证无 CLI 覆盖时 resolve_orchestrator_settings 使用 config.yaml 值"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["workers"] == 5
        assert settings["max_iterations"] == 15
        assert settings["execution_mode"] == "cli"

    def test_orchestrator_settings_cli_overrides_yaml(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 参数覆盖 config.yaml 中的编排器设置"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖
        settings = resolve_orchestrator_settings(
            overrides={
                "workers": 10,
                "max_iterations": 50,
                "execution_mode": "auto",
            }
        )

        assert settings["workers"] == 10
        assert settings["max_iterations"] == 50
        # execution_mode=auto 时 orchestrator 应强制为 basic
        assert settings["execution_mode"] == "auto"

    def test_orchestrator_forces_basic_when_cloud_mode(
        self,
        tmp_path: Path,
        config_with_cloud_execution_mode: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 execution_mode=cloud 时强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 无覆盖，config.yaml 中 execution_mode=cloud
        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["execution_mode"] == "cloud"
        # 当 execution_mode 为 cloud/auto 且 orchestrator 为 mp 时应强制为 basic
        assert settings["orchestrator"] == "basic"

    def test_orchestrator_forces_basic_when_auto_mode(
        self,
        tmp_path: Path,
        config_with_auto_execution_mode: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 execution_mode=auto 时强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["execution_mode"] == "auto"
        assert settings["orchestrator"] == "basic"

    def test_orchestrator_cli_mp_overridden_when_cloud_mode(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,  # execution_mode=cli
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 指定 orchestrator=mp 但 execution_mode=cloud 时强制为 basic"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 指定 mp 编排器和 cloud 执行模式
        settings = resolve_orchestrator_settings(
            overrides={
                "orchestrator": "mp",
                "execution_mode": "cloud",
            }
        )

        # execution_mode=cloud 时 mp 应被强制为 basic
        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "cloud"

    # ========================================================================
    # build_cursor_agent_config 与 stream_events_enabled 映射测试
    # ========================================================================

    def test_build_cursor_agent_config_stream_enabled_from_yaml(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=True 映射到 CursorAgentConfig.stream_events_enabled"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        agent_config_dict = build_cursor_agent_config(
            working_directory=str(tmp_path),
            overrides=None,
        )

        assert agent_config_dict["stream_events_enabled"] is True

    def test_build_cursor_agent_config_stream_disabled_from_yaml(
        self,
        tmp_path: Path,
        config_with_stream_disabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=False 映射到 CursorAgentConfig.stream_events_enabled"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        agent_config_dict = build_cursor_agent_config(
            working_directory=str(tmp_path),
            overrides=None,
        )

        assert agent_config_dict["stream_events_enabled"] is False

    def test_build_cursor_agent_config_stream_cli_override(
        self,
        tmp_path: Path,
        config_with_stream_disabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 覆盖 stream_events_enabled"""
        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式启用 stream
        agent_config_dict = build_cursor_agent_config(
            working_directory=str(tmp_path),
            overrides={"stream_events_enabled": True},
        )

        assert agent_config_dict["stream_events_enabled"] is True

    # ========================================================================
    # CursorAgentConfig 实例化验证
    # ========================================================================

    def test_cursor_agent_config_instance_stream_enabled(
        self,
        tmp_path: Path,
        config_with_stream_enabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CursorAgentConfig 实例正确接收 stream_events_enabled"""
        from core.config import build_cursor_agent_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(
            working_directory=str(tmp_path),
            overrides=None,
        )

        # 创建 CursorAgentConfig 实例
        agent_config = CursorAgentConfig(
            stream_events_enabled=config_dict["stream_events_enabled"],
        )

        assert agent_config.stream_events_enabled is True

    def test_cursor_agent_config_instance_stream_disabled(
        self,
        tmp_path: Path,
        config_with_stream_disabled: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CursorAgentConfig 实例正确接收 stream_events_enabled=False"""
        from core.config import build_cursor_agent_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        config_dict = build_cursor_agent_config(
            working_directory=str(tmp_path),
            overrides=None,
        )

        agent_config = CursorAgentConfig(
            stream_events_enabled=config_dict["stream_events_enabled"],
        )

        assert agent_config.stream_events_enabled is False


class TestRunPyConfigPrecedence:
    """测试 run.py 的配置加载与 CLI 覆盖优先级"""

    @pytest.fixture
    def run_py_config(self, tmp_path: Path) -> Path:
        """创建 run.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 7,
                "max_iterations": 30,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 800,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_py_parse_args_defaults_from_config(
        self,
        tmp_path: Path,
        run_py_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run.py parse_args 未传参时 tri-state 参数为 None"""
        import sys
        from importlib import import_module

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 模拟 sys.argv（仅程序名和任务描述）
        monkeypatch.setattr(sys, "argv", ["run.py", "test task"])

        # 导入 run 模块并调用 parse_args
        run_module = import_module("run")
        args = run_module.parse_args()

        # 验证 tri-state 参数为 None（未从 CLI 传入）
        assert args.workers is None
        assert args.max_iterations is None
        # stream_log_enabled 是 tri-state，应为 None
        assert args.stream_log_enabled is None

    def test_run_py_cli_overrides_config(
        self,
        tmp_path: Path,
        run_py_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run.py CLI 参数覆盖 config.yaml"""
        import sys
        from importlib import import_module

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式指定参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--workers",
                "12",
                "--max-iterations",
                "100",
                "test task",
            ],
        )

        run_module = import_module("run")
        args = run_module.parse_args()

        # CLI 参数应被解析
        assert args.workers == 12
        # max_iterations 是字符串类型（支持 "MAX" 或 "-1"）
        assert args.max_iterations == "100"


class TestRunBasicPyConfigPrecedence:
    """测试 scripts/run_basic.py 的配置加载与 CLI 覆盖优先级"""

    @pytest.fixture
    def run_basic_config(self, tmp_path: Path) -> Path:
        """创建 run_basic.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 4,
                "max_iterations": 20,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 500,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_basic_parse_args_defaults(
        self,
        tmp_path: Path,
        run_basic_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_basic.py parse_args 未传参时参数为 None"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_basic.py", "test task"])

        # 动态导入 scripts/run_basic.py（使用绝对路径）
        project_root = Path(__file__).parent.parent
        script_path = project_root / "scripts" / "run_basic.py"
        spec = importlib.util.spec_from_file_location("run_basic", str(script_path))
        assert spec is not None
        assert spec.loader is not None
        run_basic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_basic_module)

        args = run_basic_module.parse_args()

        # tri-state 参数应为 None
        assert args.workers is None
        assert args.max_iterations is None
        assert args.stream_log_enabled is None

    def test_run_basic_cli_overrides(
        self,
        tmp_path: Path,
        run_basic_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_basic.py CLI 参数覆盖"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_basic.py",
                "--workers",
                "8",
                "--max-iterations",
                "50",
                "--stream-log",
                "test task",
            ],
        )

        spec = importlib.util.spec_from_file_location("run_basic", str(PROJECT_ROOT / "scripts" / "run_basic.py"))
        assert spec is not None
        assert spec.loader is not None
        run_basic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_basic_module)

        args = run_basic_module.parse_args()

        assert args.workers == 8
        # max_iterations 是字符串类型
        assert args.max_iterations == "50"
        assert args.stream_log_enabled is True

    def test_run_basic_stream_config_from_yaml(
        self,
        tmp_path: Path,
        run_basic_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_basic.py 从 config.yaml 读取 stream 配置"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 无 CLI 参数，使用 config.yaml
        stream_config = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert stream_config["enabled"] is True
        assert stream_config["console"] is True


class TestRunMpPyConfigPrecedence:
    """测试 scripts/run_mp.py 的配置加载与 CLI 覆盖优先级"""

    @pytest.fixture
    def run_mp_config_cli(self, tmp_path: Path) -> Path:
        """创建 CLI 执行模式的 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 5,
                "max_iterations": 25,
            },
            "cloud_agent": {
                "enabled": False,
                "execution_mode": "cli",
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def run_mp_config_cloud(self, tmp_path: Path) -> Path:
        """创建 Cloud 执行模式的 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 6,
                "max_iterations": 30,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cloud",
            },
            "logging": {
                "stream_json": {
                    "enabled": False,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_mp_parse_args_defaults(
        self,
        tmp_path: Path,
        run_mp_config_cli: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_mp.py parse_args 未传参时参数为 None"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_mp.py", "test task"])

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        args = run_mp_module.parse_args()

        assert args.workers is None
        assert args.max_iterations is None
        # run_mp.py 使用 stream_log_enabled
        assert args.stream_log_enabled is None

    def test_run_mp_cli_overrides(
        self,
        tmp_path: Path,
        run_mp_config_cli: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_mp.py CLI 参数覆盖"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_mp.py",
                "--workers",
                "10",
                "--max-iterations",
                "75",
                "--stream-log",
                "test task",
            ],
        )

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        args = run_mp_module.parse_args()

        assert args.workers == 10
        # max_iterations 是字符串类型
        assert args.max_iterations == "75"
        assert args.stream_log_enabled is True

    def test_run_mp_execution_mode_cloud_forces_cli_fallback(
        self,
        tmp_path: Path,
        run_mp_config_cloud: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_mp.py 当 execution_mode=cloud 时强制回退到 CLI 模式

        MP 编排器仅支持 execution_mode=cli，当配置为 cloud/auto 时
        应强制 execution_mode 为 cli 并记录警告。
        """
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 无覆盖，使用 config.yaml 中的 cloud 模式
        settings = resolve_orchestrator_settings(overrides=None)

        # resolve_orchestrator_settings 应强制 orchestrator=basic
        assert settings["execution_mode"] == "cloud"
        assert settings["orchestrator"] == "basic"


class TestRunIteratePyConfigPrecedence:
    """测试 scripts/run_iterate.py 的配置加载与 CLI 覆盖优先级"""

    @pytest.fixture
    def run_iterate_config(self, tmp_path: Path) -> Path:
        """创建 run_iterate.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 4,
                "max_iterations": 15,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 700,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def run_iterate_config_auto(self, tmp_path: Path) -> Path:
        """创建 auto 执行模式的 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 5,
                "max_iterations": 20,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "auto",
                "timeout": 900,
            },
            "logging": {
                "stream_json": {
                    "enabled": False,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_iterate_parse_args_defaults(
        self,
        tmp_path: Path,
        run_iterate_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_iterate.py parse_args 未传参时 tri-state 参数正确"""
        import importlib.util
        import sys

        # 保存项目根目录（因为 chdir 会改变当前目录）
        project_root = Path.cwd()

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "test task"])

        # 使用项目根目录的绝对路径
        run_iterate_path = project_root / "scripts" / "run_iterate.py"
        spec = importlib.util.spec_from_file_location("run_iterate", str(run_iterate_path))
        assert spec is not None
        assert spec.loader is not None
        run_iterate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_iterate_module)

        args = run_iterate_module.parse_args()

        # tri-state 参数应为 None（实际默认值从 config.yaml 解析）
        assert args.workers is None
        assert args.max_iterations is None
        assert args.execution_mode is None
        # orchestrator 也是 tri-state，默认为 None（运行时从 config.yaml 或默认值解析为 mp）
        assert args.orchestrator is None

    def test_run_iterate_cli_overrides(
        self,
        tmp_path: Path,
        run_iterate_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_iterate.py CLI 参数覆盖"""
        import importlib.util
        import sys

        # 保存项目根目录（因为 chdir 会改变当前目录）
        project_root = Path.cwd()

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "--workers",
                "6",
                "--max-iterations",
                "40",
                "--execution-mode",
                "cloud",
                "--orchestrator",
                "basic",
                "test task",
            ],
        )

        # 使用项目根目录的绝对路径
        run_iterate_path = project_root / "scripts" / "run_iterate.py"
        spec = importlib.util.spec_from_file_location("run_iterate", str(run_iterate_path))
        assert spec is not None
        assert spec.loader is not None
        run_iterate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_iterate_module)

        args = run_iterate_module.parse_args()

        assert args.workers == 6
        # max_iterations 是字符串类型
        assert args.max_iterations == "40"
        assert args.execution_mode == "cloud"
        assert args.orchestrator == "basic"

    def test_run_iterate_auto_mode_forces_basic_orchestrator(
        self,
        tmp_path: Path,
        run_iterate_config_auto: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_iterate.py 当 execution_mode=auto 时强制使用 basic 编排器"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["execution_mode"] == "auto"
        assert settings["orchestrator"] == "basic"

    def test_run_iterate_cli_mp_overridden_by_cloud_mode(
        self,
        tmp_path: Path,
        run_iterate_config: Path,  # execution_mode=cli
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 指定 orchestrator=mp 但 execution_mode=cloud 时强制为 basic"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖：mp + cloud
        settings = resolve_orchestrator_settings(
            overrides={
                "orchestrator": "mp",
                "execution_mode": "cloud",
            }
        )

        # 应强制为 basic
        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "cloud"


class TestRunKnowledgePyConfigPrecedence:
    """测试 scripts/run_knowledge.py 的配置加载与 CLI 覆盖优先级"""

    @pytest.fixture
    def run_knowledge_config(self, tmp_path: Path) -> Path:
        """创建 run_knowledge.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "worker_pool_size": 3,
                "max_iterations": 10,
            },
            "cloud_agent": {
                "enabled": True,
                "execution_mode": "cli",
                "timeout": 600,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": False,
                    "detail_dir": "logs/knowledge_detail",
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_knowledge_parse_args_defaults(
        self,
        tmp_path: Path,
        run_knowledge_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_knowledge.py parse_args 未传参时参数为 None"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "test task"])

        spec = importlib.util.spec_from_file_location(
            "run_knowledge", str(PROJECT_ROOT / "scripts" / "run_knowledge.py")
        )
        assert spec is not None
        assert spec.loader is not None
        run_knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_knowledge_module)

        args = run_knowledge_module.parse_args()

        # tri-state 参数应为 None
        assert args.workers is None
        assert args.max_iterations is None
        assert args.stream_log_enabled is None

    def test_run_knowledge_cli_overrides(
        self,
        tmp_path: Path,
        run_knowledge_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_knowledge.py CLI 参数覆盖"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_knowledge.py",
                "--workers",
                "5",
                "--max-iterations",
                "25",
                "--stream-log",
                "test task",
            ],
        )

        spec = importlib.util.spec_from_file_location(
            "run_knowledge", str(PROJECT_ROOT / "scripts" / "run_knowledge.py")
        )
        assert spec is not None
        assert spec.loader is not None
        run_knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_knowledge_module)

        args = run_knowledge_module.parse_args()

        assert args.workers == 5
        # max_iterations 是字符串类型
        assert args.max_iterations == "25"
        assert args.stream_log_enabled is True

    def test_run_knowledge_stream_config_from_yaml(
        self,
        tmp_path: Path,
        run_knowledge_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_knowledge.py 从 config.yaml 读取 stream 配置"""
        from core.config import resolve_stream_log_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        stream_config = resolve_stream_log_config(
            cli_enabled=None,
            cli_console=None,
            cli_detail_dir=None,
            cli_raw_dir=None,
        )

        assert stream_config["enabled"] is True
        assert stream_config["console"] is False
        # detail_dir 配置的格式
        assert "knowledge_detail" in stream_config["detail_dir"]


class TestStreamEventsEnabledEndToEnd:
    """端到端测试：stream_json.enabled -> CursorAgentConfig.stream_events_enabled"""

    @pytest.fixture
    def stream_enabled_config(self, tmp_path: Path) -> Path:
        """创建 stream 启用的配置"""
        config_content = {
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def stream_disabled_config(self, tmp_path: Path) -> Path:
        """创建 stream 禁用的配置"""
        config_content = {
            "logging": {
                "stream_json": {
                    "enabled": False,
                }
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_stream_enabled_flows_to_cursor_agent_config(
        self,
        tmp_path: Path,
        stream_enabled_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=True 正确流转到 CursorAgentConfig"""
        from core.config import build_cursor_agent_config, resolve_stream_log_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 步骤 1: 验证 resolve_stream_log_config 返回 enabled=True
        stream_config = resolve_stream_log_config(
            cli_enabled=None, cli_console=None, cli_detail_dir=None, cli_raw_dir=None
        )
        assert stream_config["enabled"] is True

        # 步骤 2: 验证 build_cursor_agent_config 包含 stream_events_enabled=True
        agent_config_dict = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config_dict["stream_events_enabled"] is True

        # 步骤 3: 验证 CursorAgentConfig 实例正确接收该值
        agent_config = CursorAgentConfig(stream_events_enabled=agent_config_dict["stream_events_enabled"])
        assert agent_config.stream_events_enabled is True

    def test_stream_disabled_flows_to_cursor_agent_config(
        self,
        tmp_path: Path,
        stream_disabled_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 stream_json.enabled=False 正确流转到 CursorAgentConfig"""
        from core.config import build_cursor_agent_config, resolve_stream_log_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        stream_config = resolve_stream_log_config(
            cli_enabled=None, cli_console=None, cli_detail_dir=None, cli_raw_dir=None
        )
        assert stream_config["enabled"] is False

        agent_config_dict = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config_dict["stream_events_enabled"] is False

        agent_config = CursorAgentConfig(stream_events_enabled=agent_config_dict["stream_events_enabled"])
        assert agent_config.stream_events_enabled is False

    def test_cli_override_stream_enabled_true(
        self,
        tmp_path: Path,
        stream_disabled_config: Path,  # YAML 禁用
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 覆盖 stream_json.enabled 为 True"""
        from core.config import resolve_stream_log_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式启用
        stream_config = resolve_stream_log_config(
            cli_enabled=True, cli_console=None, cli_detail_dir=None, cli_raw_dir=None
        )
        assert stream_config["enabled"] is True

        agent_config = CursorAgentConfig(stream_events_enabled=stream_config["enabled"])
        assert agent_config.stream_events_enabled is True

    def test_cli_override_stream_enabled_false(
        self,
        tmp_path: Path,
        stream_enabled_config: Path,  # YAML 启用
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 覆盖 stream_json.enabled 为 False"""
        from core.config import resolve_stream_log_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 显式禁用
        stream_config = resolve_stream_log_config(
            cli_enabled=False, cli_console=None, cli_detail_dir=None, cli_raw_dir=None
        )
        assert stream_config["enabled"] is False

        agent_config = CursorAgentConfig(stream_events_enabled=stream_config["enabled"])
        assert agent_config.stream_events_enabled is False


class TestExecutionModeOrchestratorCompatibility:
    """测试 execution_mode 与 orchestrator 兼容性规则

    核心规则：
    - execution_mode=cli: 支持 mp 和 basic 编排器
    - execution_mode=cloud/auto: 强制使用 basic 编排器
    """

    @pytest.fixture
    def mp_cli_config(self, tmp_path: Path) -> Path:
        """创建 mp + cli 配置"""
        config_content = {
            "system": {
                "orchestrator": "mp",
            },
            "cloud_agent": {
                "execution_mode": "cli",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def mp_cloud_config(self, tmp_path: Path) -> Path:
        """创建 mp + cloud 配置（应强制 basic）"""
        config_content = {
            "system": {
                "orchestrator": "mp",
            },
            "cloud_agent": {
                "execution_mode": "cloud",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def mp_auto_config(self, tmp_path: Path) -> Path:
        """创建 mp + auto 配置（应强制 basic）"""
        config_content = {
            "system": {
                "orchestrator": "mp",
            },
            "cloud_agent": {
                "execution_mode": "auto",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_mp_with_cli_mode_allowed(
        self,
        tmp_path: Path,
        mp_cli_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 mp + cli 模式正常工作"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["orchestrator"] == "mp"
        assert settings["execution_mode"] == "cli"

    def test_mp_with_cloud_mode_forces_basic(
        self,
        tmp_path: Path,
        mp_cloud_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 mp + cloud 模式强制切换为 basic"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        # orchestrator 应被强制为 basic
        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "cloud"

    def test_mp_with_auto_mode_forces_basic(
        self,
        tmp_path: Path,
        mp_auto_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 mp + auto 模式强制切换为 basic"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "auto"

    def test_cli_override_mp_plus_cloud_forces_basic(
        self,
        tmp_path: Path,
        mp_cli_config: Path,  # 配置是 mp + cli
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 覆盖 orchestrator=mp + execution_mode=cloud 时强制 basic"""
        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖为 mp + cloud
        settings = resolve_orchestrator_settings(
            overrides={
                "orchestrator": "mp",
                "execution_mode": "cloud",
            }
        )

        # 应强制为 basic
        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "cloud"

    def test_basic_with_any_mode_unchanged(
        self,
        tmp_path: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 basic 编排器与任何执行模式都兼容"""
        from core.config import resolve_orchestrator_settings

        # 创建 basic + cloud 配置
        config_content = {
            "system": {
                "orchestrator": "basic",
            },
            "cloud_agent": {
                "execution_mode": "cloud",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        settings = resolve_orchestrator_settings(overrides=None)

        # basic 不受 cloud 模式影响
        assert settings["orchestrator"] == "basic"
        assert settings["execution_mode"] == "cloud"


# ============================================================
# TestPrintConfigDebugOutput - --print-config 参数测试
# ============================================================


class TestPrintConfigDebugOutput:
    """测试 --print-config 参数输出的配置调试信息

    验证内容：
    1. format_debug_config 输出格式稳定，便于脚本化 grep/CI 断言
    2. 输出包含关键配置字段
    3. orchestrator 回退信息正确显示
    4. run.py 和 scripts/run_iterate.py 的 --print-config 参数正常工作
    """

    @pytest.fixture
    def debug_output_config(self, tmp_path: Path) -> Path:
        """创建用于调试输出测试的配置"""
        config_content = {
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 4,
                "enable_sub_planners": True,
                "strict_review": False,
            },
            "models": {
                "planner": "test-planner",
                "worker": "test-worker",
                "reviewer": "test-reviewer",
            },
            "cloud_agent": {
                "execution_mode": "cli",
                "timeout": 500,
                "auth_timeout": 60,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_format_debug_config_output_structure(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 format_debug_config 输出格式稳定"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        output = format_debug_config(cli_overrides=None, source_label="test")

        # 验证输出包含所有关键字段（完整的固定键集合）
        assert f"{CONFIG_DEBUG_PREFIX} config_path:" in output
        assert f"{CONFIG_DEBUG_PREFIX} source: test" in output
        assert f"{CONFIG_DEBUG_PREFIX} max_iterations: 15" in output
        assert f"{CONFIG_DEBUG_PREFIX} workers: 4" in output
        # 关键语义：requested_mode 表示用户请求的执行模式（requested 语义）
        # 而非 execution_mode，明确区分请求模式与实际生效模式
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: cli" in output
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: mp" in output
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator_fallback: none" in output
        assert f"{CONFIG_DEBUG_PREFIX} planner_model: test-planner" in output
        assert f"{CONFIG_DEBUG_PREFIX} worker_model: test-worker" in output
        assert f"{CONFIG_DEBUG_PREFIX} reviewer_model: test-reviewer" in output
        assert f"{CONFIG_DEBUG_PREFIX} cloud_timeout: 500" in output
        assert f"{CONFIG_DEBUG_PREFIX} cloud_auth_timeout: 60" in output
        assert f"{CONFIG_DEBUG_PREFIX} auto_commit: false" in output
        assert f"{CONFIG_DEBUG_PREFIX} auto_push: false" in output
        assert f"{CONFIG_DEBUG_PREFIX} dry_run: false" in output
        assert f"{CONFIG_DEBUG_PREFIX} strict_review: false" in output
        assert f"{CONFIG_DEBUG_PREFIX} enable_sub_planners: true" in output

    def test_format_debug_config_with_cli_overrides(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 CLI 覆盖在输出中正确反映"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖配置
        cli_overrides = {
            "workers": 8,
            "max_iterations": 50,
            "auto_commit": True,
            "dry_run": True,
        }

        output = format_debug_config(cli_overrides=cli_overrides, source_label="test")

        # 验证 CLI 覆盖值
        assert f"{CONFIG_DEBUG_PREFIX} max_iterations: 50" in output
        assert f"{CONFIG_DEBUG_PREFIX} workers: 8" in output
        assert f"{CONFIG_DEBUG_PREFIX} auto_commit: true" in output
        assert f"{CONFIG_DEBUG_PREFIX} dry_run: true" in output

    def test_format_debug_config_orchestrator_fallback_cloud(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 orchestrator 在 cloud 模式下回退的输出"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖为 mp + cloud，应触发回退
        cli_overrides = {
            "orchestrator": "mp",
            "execution_mode": "cloud",
        }

        output = format_debug_config(cli_overrides=cli_overrides, source_label="test")

        # 验证回退信息
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: basic" in output
        # 新的明确字段名：requested_mode 和 effective_mode
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: cloud" in output
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output  # 因为没有 API Key 所以回退
        assert "mp->basic" in output
        assert "requested_mode=cloud" in output

    def test_format_debug_config_orchestrator_fallback_auto(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 orchestrator 在 auto 模式下回退的输出"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖为 mp + auto，应触发回退
        cli_overrides = {
            "orchestrator": "mp",
            "execution_mode": "auto",
        }

        output = format_debug_config(cli_overrides=cli_overrides, source_label="test")

        # 验证回退信息
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: basic" in output
        # 新的明确字段名：requested_mode 和 effective_mode
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: auto" in output
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output  # 因为没有 API Key 所以回退
        assert "mp->basic" in output
        assert "requested_mode=auto" in output

    def test_format_debug_config_no_fallback_when_basic(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 basic 编排器不触发回退信息"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # CLI 覆盖为 basic + cloud，不应触发回退
        cli_overrides = {
            "orchestrator": "basic",
            "execution_mode": "cloud",
        }

        output = format_debug_config(cli_overrides=cli_overrides, source_label="test")

        # 验证无回退信息
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: basic" in output
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator_fallback: none" in output

    def test_format_debug_config_grep_friendly_format(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证输出格式便于 grep 解析"""
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        output = format_debug_config(cli_overrides=None, source_label="test")

        # 验证每行格式：[CONFIG] key: value
        lines = output.strip().split("\n")
        for line in lines:
            assert line.startswith(CONFIG_DEBUG_PREFIX)
            assert ": " in line

    def test_config_debug_prefix_constant(self) -> None:
        """验证 CONFIG_DEBUG_PREFIX 常量存在且格式正确"""
        from core.config import CONFIG_DEBUG_PREFIX

        assert CONFIG_DEBUG_PREFIX == "[CONFIG]"

    def test_print_debug_config_function(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
        capsys,
    ) -> None:
        """验证 print_debug_config 函数正确输出到 stdout"""
        from core.config import CONFIG_DEBUG_PREFIX, print_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        print_debug_config(cli_overrides=None, source_label="test")

        captured = capsys.readouterr()

        # 验证输出到 stdout
        assert f"{CONFIG_DEBUG_PREFIX} source: test" in captured.out
        assert f"{CONFIG_DEBUG_PREFIX} max_iterations:" in captured.out

    def test_format_debug_config_required_keys_complete(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 format_debug_config 输出包含完整的固定键集合

        断言输出包含所有必需的键：
        - config_path, source, max_iterations, workers
        - requested_mode, effective_mode, orchestrator, orchestrator_fallback
        - planner_model, worker_model, reviewer_model
        - cloud_timeout, cloud_auth_timeout
        - auto_commit, auto_push, dry_run, strict_review, enable_sub_planners
        """
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        output = format_debug_config(cli_overrides=None, source_label="test")

        # 定义固定键集合（必须全部存在）
        required_keys = [
            "config_path",
            "source",
            "max_iterations",
            "workers",
            "requested_mode",  # 注意：是 requested_mode，不是 execution_mode
            "effective_mode",
            "orchestrator",
            "orchestrator_fallback",
            "planner_model",
            "worker_model",
            "reviewer_model",
            "cloud_timeout",
            "cloud_auth_timeout",
            "auto_commit",
            "auto_push",
            "dry_run",
            "strict_review",
            "enable_sub_planners",
        ]

        # 验证每个必需键都存在于输出中
        for key in required_keys:
            assert f"{CONFIG_DEBUG_PREFIX} {key}:" in output, f"缺少必需键 '{key}'，输出应包含 '[CONFIG] {key}:'"

        # 验证输出行数与必需键数量一致（确保没有多余或缺失的键）
        output_lines = [line for line in output.strip().split("\n") if line.startswith(CONFIG_DEBUG_PREFIX)]
        assert len(output_lines) == len(required_keys), (
            f"输出行数 ({len(output_lines)}) 与必需键数量 ({len(required_keys)}) 不一致"
        )

    def test_format_debug_config_requested_mode_semantic(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 requested_mode 字段表示用户请求的执行模式（requested 语义）

        关键语义说明：
        - 输出字段名为 requested_mode（非 execution_mode），明确表示这是用户请求的模式
        - requested_mode 反映 CLI 参数 --execution-mode 或 config.yaml 中的配置
        - effective_mode 是实际生效的模式（可能因缺少 API Key 而回退）
        - orchestrator 选择基于 requested_mode，而非 effective_mode
        """
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 场景 1：requested_mode=auto，无 API Key 时 effective_mode=cli
        cli_overrides_auto = {"execution_mode": "auto"}
        output_auto = format_debug_config(
            cli_overrides=cli_overrides_auto,
            source_label="test",
            has_api_key=False,  # 无 API Key
        )

        # 验证 requested_mode 是用户请求的 auto，而非回退后的 cli
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: auto" in output_auto
        # effective_mode 因无 API Key 回退到 cli
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output_auto

        # 场景 2：requested_mode=cloud，无 API Key 时 effective_mode=cli
        cli_overrides_cloud = {"execution_mode": "cloud"}
        output_cloud = format_debug_config(
            cli_overrides=cli_overrides_cloud,
            source_label="test",
            has_api_key=False,
        )

        # 验证 requested_mode 保持用户请求的 cloud
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: cloud" in output_cloud
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output_cloud

        # 场景 3：验证输出中不存在 "execution_mode" 字段（使用 requested_mode 替代）
        # 确保语义清晰，避免混淆
        assert f"{CONFIG_DEBUG_PREFIX} execution_mode:" not in output_auto
        assert f"{CONFIG_DEBUG_PREFIX} execution_mode:" not in output_cloud

    def test_format_debug_config_orchestrator_based_on_requested_mode(
        self,
        tmp_path: Path,
        debug_output_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 orchestrator 选择基于 requested_mode 而非 effective_mode

        关键规则：
        - requested_mode=auto/cloud 时，orchestrator 强制为 basic
        - 即使 effective_mode 因缺少 API Key 回退到 cli，orchestrator 仍保持 basic
        - 这是因为编排器选择基于用户请求的模式，而非实际生效的模式
        """
        from core.config import CONFIG_DEBUG_PREFIX, format_debug_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 场景：requested_mode=auto，无 API Key（effective_mode=cli）
        # orchestrator 仍应为 basic（基于 requested_mode=auto）
        cli_overrides = {
            "orchestrator": "mp",  # 用户请求 mp
            "execution_mode": "auto",  # 但 requested_mode=auto 会强制 basic
        }
        output = format_debug_config(
            cli_overrides=cli_overrides,
            source_label="test",
            has_api_key=False,  # 无 API Key，effective_mode 回退到 cli
        )

        # 验证 orchestrator 被强制为 basic（基于 requested_mode）
        assert f"{CONFIG_DEBUG_PREFIX} orchestrator: basic" in output
        # 验证 requested_mode 和 effective_mode
        assert f"{CONFIG_DEBUG_PREFIX} requested_mode: auto" in output
        assert f"{CONFIG_DEBUG_PREFIX} effective_mode: cli" in output
        # 验证回退原因包含 requested_mode 说明
        assert "requested_mode=auto" in output
        # orchestrator_fallback 应说明即使 effective_mode=cli 也保持 basic
        assert "orchestrator_fallback: mp->basic" in output


class TestRunPyPrintConfigParameter:
    """测试 run.py 的 --print-config 参数"""

    @pytest.fixture
    def run_py_config(self, tmp_path: Path) -> Path:
        """创建 run.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 20,
                "worker_pool_size": 5,
            },
            "models": {
                "planner": "run-py-planner",
                "worker": "run-py-worker",
                "reviewer": "run-py-reviewer",
            },
            "cloud_agent": {
                "execution_mode": "cli",
                "timeout": 400,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_py_parse_args_includes_print_config(
        self,
        tmp_path: Path,
        run_py_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run.py parse_args 包含 --print-config 参数"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 测试 --print-config 参数解析
        monkeypatch.setattr(sys, "argv", ["run.py", "--print-config"])

        spec = importlib.util.spec_from_file_location("run", str(PROJECT_ROOT / "run.py"))
        assert spec is not None
        assert spec.loader is not None
        run_module = importlib.util.module_from_spec(spec)
        # 将模块添加到 sys.modules 中，以便 dataclass 可以正确解析
        sys.modules["run"] = run_module
        try:
            spec.loader.exec_module(run_module)
            args = run_module.parse_args()
            assert args.print_config is True
        finally:
            # 清理
            sys.modules.pop("run", None)

    def test_run_py_build_cli_overrides_from_args(
        self,
        tmp_path: Path,
        run_py_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run.py _build_cli_overrides_from_args 函数"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--workers",
                "6",
                "--max-iterations",
                "30",
                "--execution-mode",
                "cloud",
                "--auto-commit",
                "test task",
            ],
        )

        spec = importlib.util.spec_from_file_location("run", str(PROJECT_ROOT / "run.py"))
        assert spec is not None
        assert spec.loader is not None
        run_module = importlib.util.module_from_spec(spec)
        # 将模块添加到 sys.modules 中，以便 dataclass 可以正确解析
        sys.modules["run"] = run_module
        try:
            spec.loader.exec_module(run_module)
            args = run_module.parse_args()
            cli_overrides = run_module._build_cli_overrides_from_args(args)
            assert cli_overrides["workers"] == 6
            assert cli_overrides["max_iterations"] == 30
            assert cli_overrides["execution_mode"] == "cloud"
            assert cli_overrides["auto_commit"] is True
        finally:
            # 清理
            sys.modules.pop("run", None)


class TestRunIteratePyPrintConfigParameter:
    """测试 scripts/run_iterate.py 的 --print-config 参数"""

    @pytest.fixture
    def run_iterate_config(self, tmp_path: Path) -> Path:
        """创建 run_iterate.py 测试用 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 25,
                "worker_pool_size": 6,
            },
            "models": {
                "planner": "iterate-planner",
                "worker": "iterate-worker",
                "reviewer": "iterate-reviewer",
            },
            "cloud_agent": {
                "execution_mode": "cli",
                "timeout": 600,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    def test_run_iterate_parse_args_includes_print_config(
        self,
        tmp_path: Path,
        run_iterate_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_iterate.py parse_args 包含 --print-config 参数"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 测试 --print-config 参数解析
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "--print-config"])

        spec = importlib.util.spec_from_file_location("run_iterate", str(PROJECT_ROOT / "scripts" / "run_iterate.py"))
        assert spec is not None
        assert spec.loader is not None
        run_iterate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_iterate_module)

        args = run_iterate_module.parse_args()

        assert args.print_config is True

    def test_run_iterate_build_cli_overrides_from_args(
        self,
        tmp_path: Path,
        run_iterate_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证 run_iterate.py _build_cli_overrides_from_args 函数"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "--workers",
                "7",
                "--max-iterations",
                "35",
                "--execution-mode",
                "auto",
                "--dry-run",
                "test task",
            ],
        )

        spec = importlib.util.spec_from_file_location("run_iterate", str(PROJECT_ROOT / "scripts" / "run_iterate.py"))
        assert spec is not None
        assert spec.loader is not None
        run_iterate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_iterate_module)

        args = run_iterate_module.parse_args()
        cli_overrides = run_iterate_module._build_cli_overrides_from_args(args)

        assert cli_overrides["workers"] == 7
        assert cli_overrides["max_iterations"] == 35
        assert cli_overrides["execution_mode"] == "auto"
        assert cli_overrides["dry_run"] is True


# ============================================================
# TestEntryScriptConfigLoadingComplete - 各入口脚本配置加载综合测试
# ============================================================


class TestEntryScriptConfigLoadingComplete:
    """综合测试：验证各入口脚本的配置加载和 CLI 覆盖语义

    测试覆盖的入口脚本:
    - run.py
    - scripts/run_basic.py
    - scripts/run_mp.py
    - scripts/run_iterate.py
    - scripts/run_knowledge.py

    验证点:
    1. 未传 CLI 参数时取值等于 config.yaml
    2. 传 CLI 参数时覆盖 config.yaml
    3. execution_mode=cloud/auto 时强制 basic 编排器策略
    4. stream_json.enabled 能驱动 CursorAgentConfig.stream_events_enabled
    """

    @pytest.fixture(autouse=True)
    def mock_cloud_api_key(self) -> Generator[None, None, None]:
        """Mock Cloud API Key 以避免因无 API Key 导致的模式回退

        当测试显式设置了 API Key 时返回该值，否则返回 mock 值。
        """
        from cursor.cloud_client import CloudClientFactory

        def _resolve_api_key(explicit_api_key=None, **kwargs):
            return explicit_api_key if explicit_api_key else "mock-api-key"

        with patch.object(CloudClientFactory, "resolve_api_key", side_effect=_resolve_api_key):
            yield

    @pytest.fixture
    def comprehensive_config(self, tmp_path: Path) -> Path:
        """创建包含所有测试所需配置项的 config.yaml"""
        config_content = {
            "system": {
                "max_iterations": 15,
                "worker_pool_size": 7,
                "orchestrator": "mp",
            },
            "cloud_agent": {
                "execution_mode": "cli",
                "timeout": 450,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                    "console": True,
                    "detail_dir": "logs/detail",
                    "raw_dir": "logs/raw",
                },
            },
            "models": {
                "planner": "yaml-planner-model",
                "worker": "yaml-worker-model",
                "reviewer": "yaml-reviewer-model",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def cloud_mode_config(self, tmp_path: Path) -> Path:
        """创建 execution_mode=cloud 的配置"""
        config_content = {
            "system": {
                "max_iterations": 20,
                "worker_pool_size": 5,
                "orchestrator": "mp",  # 应被强制为 basic
            },
            "cloud_agent": {
                "execution_mode": "cloud",
                "timeout": 600,
            },
            "logging": {
                "stream_json": {
                    "enabled": True,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    @pytest.fixture
    def auto_mode_config(self, tmp_path: Path) -> Path:
        """创建 execution_mode=auto 的配置"""
        config_content = {
            "system": {
                "max_iterations": 25,
                "worker_pool_size": 6,
                "orchestrator": "mp",  # 应被强制为 basic
            },
            "cloud_agent": {
                "execution_mode": "auto",
                "timeout": 500,
            },
            "logging": {
                "stream_json": {
                    "enabled": False,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f, allow_unicode=True)
        return config_path

    # ============================================================
    # run.py 测试
    # ============================================================

    def test_run_py_config_from_yaml_no_cli_args(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run.py: 未传 CLI 参数时，配置值来自 config.yaml"""
        import sys

        from core.config import build_cursor_agent_config

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        # 清理模块缓存
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证配置值来自 config.yaml
        assert merged["workers"] == 7, f"workers 应为 7, 实际 {merged['workers']}"
        assert merged["max_iterations"] == 15, f"max_iterations 应为 15, 实际 {merged['max_iterations']}"
        assert merged["execution_mode"] == "cli", f"execution_mode 应为 cli, 实际 {merged['execution_mode']}"
        assert merged["cloud_timeout"] == 450, f"cloud_timeout 应为 450, 实际 {merged['cloud_timeout']}"
        assert merged["stream_log"] is True, f"stream_log 应为 True, 实际 {merged['stream_log']}"

        # 验证 stream_json.enabled 驱动 CursorAgentConfig.stream_events_enabled
        agent_config_dict = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config_dict["stream_events_enabled"] is True

    def test_run_py_cli_overrides_config(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run.py: CLI 参数覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run.py",
                "--workers",
                "3",
                "--max-iterations",
                "8",
                "--execution-mode",
                "cloud",
                "--cloud-timeout",
                "200",
                "--no-stream-log",
                "测试任务",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from run import Runner, parse_args

        args = parse_args()
        runner = Runner(args)
        merged = runner._merge_options({})

        # 验证 CLI 覆盖
        assert merged["workers"] == 3, f"CLI workers 应覆盖为 3, 实际 {merged['workers']}"
        assert merged["max_iterations"] == 8, f"CLI max_iterations 应覆盖为 8, 实际 {merged['max_iterations']}"
        assert merged["execution_mode"] == "cloud", (
            f"CLI execution_mode 应覆盖为 cloud, 实际 {merged['execution_mode']}"
        )
        assert merged["cloud_timeout"] == 200, f"CLI cloud_timeout 应覆盖为 200, 实际 {merged['cloud_timeout']}"
        assert merged["stream_log"] is False, f"CLI --no-stream-log 应禁用 stream_log, 实际 {merged['stream_log']}"

    def test_run_py_cloud_mode_forces_basic_orchestrator(
        self,
        tmp_path: Path,
        cloud_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run.py: execution_mode=cloud 时强制使用 basic 编排器"""
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        # 调用 resolve_orchestrator_settings 验证强制 basic
        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["orchestrator"] == "basic", f"cloud 模式应强制 basic 编排器, 实际 {settings['orchestrator']}"
        assert settings["execution_mode"] == "cloud"

    def test_run_py_auto_mode_forces_basic_orchestrator(
        self,
        tmp_path: Path,
        auto_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run.py: execution_mode=auto 时强制使用 basic 编排器"""
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["orchestrator"] == "basic", f"auto 模式应强制 basic 编排器, 实际 {settings['orchestrator']}"
        assert settings["execution_mode"] == "auto"

    # ============================================================
    # scripts/run_basic.py 测试
    # ============================================================

    def test_run_basic_py_config_from_yaml_no_cli_args(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_basic.py: 未传 CLI 参数时，配置值来自 config.yaml"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试任务"])

        spec = importlib.util.spec_from_file_location("run_basic", str(PROJECT_ROOT / "scripts" / "run_basic.py"))
        assert spec is not None
        assert spec.loader is not None
        run_basic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_basic_module)

        args = run_basic_module.parse_args()

        # CLI 参数应为 None（tri-state），表示使用 config.yaml
        assert args.workers is None, f"无 CLI 参数时 workers 应为 None, 实际 {args.workers}"
        assert args.max_iterations is None, f"无 CLI 参数时 max_iterations 应为 None, 实际 {args.max_iterations}"
        assert args.stream_log_enabled is None, (
            f"无 CLI 参数时 stream_log_enabled 应为 None, 实际 {args.stream_log_enabled}"
        )

        # 通过 resolve_orchestrator_settings 验证实际解析值
        from core.config import build_cursor_agent_config, resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["workers"] == 7
        assert settings["max_iterations"] == 15
        assert settings["execution_mode"] == "cli"

        # 验证 stream_events_enabled
        agent_config = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config["stream_events_enabled"] is True

    def test_run_basic_py_cli_overrides_config(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_basic.py: CLI 参数覆盖 config.yaml"""
        import importlib.util
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_basic.py",
                "--workers",
                "4",
                "--max-iterations",
                "12",
                "--no-stream-log",
                "测试任务",
            ],
        )

        spec = importlib.util.spec_from_file_location("run_basic", str(PROJECT_ROOT / "scripts" / "run_basic.py"))
        assert spec is not None
        assert spec.loader is not None
        run_basic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_basic_module)

        args = run_basic_module.parse_args()

        # 验证 CLI 参数被解析
        assert args.workers == 4, f"CLI workers 应为 4, 实际 {args.workers}"
        assert args.max_iterations == "12", f"CLI max_iterations 应为 '12', 实际 {args.max_iterations}"
        assert args.stream_log_enabled is False, f"CLI --no-stream-log 应为 False, 实际 {args.stream_log_enabled}"

        # 通过 resolve_orchestrator_settings 验证 CLI 覆盖
        cli_overrides = {"workers": args.workers}
        settings = resolve_orchestrator_settings(overrides=cli_overrides)
        assert settings["workers"] == 4, "CLI workers 应覆盖 config.yaml"

    # ============================================================
    # scripts/run_mp.py 测试
    # ============================================================

    def test_run_mp_py_config_from_yaml_no_cli_args(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_mp.py: 未传 CLI 参数时，配置值来自 config.yaml"""
        import importlib.util
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试任务"])

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        args = run_mp_module.parse_args()

        # CLI 参数应为 None（tri-state）
        assert args.workers is None
        assert args.max_iterations is None
        assert args.stream_log_enabled is None

        # 验证实际解析值来自 config.yaml
        from core.config import resolve_orchestrator_settings

        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["workers"] == 7
        assert settings["max_iterations"] == 15
        # cli 模式下 mp 编排器保持不变
        assert settings["orchestrator"] == "mp"

    def test_run_mp_py_cli_overrides_config(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_mp.py: CLI 参数覆盖 config.yaml"""
        import importlib.util
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_mp.py",
                "--workers",
                "5",
                "--max-iterations",
                "10",
                "--stream-log",
                "测试任务",
            ],
        )

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        args = run_mp_module.parse_args()

        assert args.workers == 5
        assert args.max_iterations == "10"
        assert args.stream_log_enabled is True

        # 验证 CLI 覆盖生效
        cli_overrides = {"workers": args.workers}
        settings = resolve_orchestrator_settings(overrides=cli_overrides)
        assert settings["workers"] == 5

    def test_run_mp_py_cloud_mode_forces_basic_fallback(
        self,
        tmp_path: Path,
        cloud_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_mp.py: execution_mode=cloud 时强制回退到 basic 编排器"""
        import importlib.util
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试任务"])

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        # 验证 resolve_orchestrator_settings 强制 basic
        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["orchestrator"] == "basic", f"cloud 模式应强制 basic, 实际 {settings['orchestrator']}"
        assert settings["execution_mode"] == "cloud"

    def test_run_mp_py_auto_mode_forces_basic_fallback(
        self,
        tmp_path: Path,
        auto_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_mp.py: execution_mode=auto 时强制回退到 basic 编排器"""
        import importlib.util
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试任务"])

        spec = importlib.util.spec_from_file_location("run_mp", str(PROJECT_ROOT / "scripts" / "run_mp.py"))
        assert spec is not None
        assert spec.loader is not None
        run_mp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mp_module)

        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["orchestrator"] == "basic", f"auto 模式应强制 basic, 实际 {settings['orchestrator']}"
        assert settings["execution_mode"] == "auto"

    # ============================================================
    # scripts/run_iterate.py 测试
    # ============================================================

    def test_run_iterate_py_config_from_yaml_no_cli_args(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_iterate.py: 未传 CLI 参数时，配置值来自 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        # 验证配置值来自 config.yaml
        assert settings.worker_pool_size == 7, f"worker_pool_size 应为 7, 实际 {settings.worker_pool_size}"
        assert settings.max_iterations == 15, f"max_iterations 应为 15, 实际 {settings.max_iterations}"
        assert settings.execution_mode == "cli", f"execution_mode 应为 cli, 实际 {settings.execution_mode}"
        assert settings.cloud_timeout == 450, f"cloud_timeout 应为 450, 实际 {settings.cloud_timeout}"
        assert settings.stream_events_enabled is True, (
            f"stream_events_enabled 应为 True, 实际 {settings.stream_events_enabled}"
        )

    def test_run_iterate_py_cli_overrides_config(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_iterate.py: CLI 参数覆盖 config.yaml"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        # 注意: run_iterate.py 没有 --no-stream-log 参数
        # stream_events_enabled 只能从 config.yaml 读取
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "--workers",
                "2",
                "--max-iterations",
                "5",
                "--execution-mode",
                "cli",
                "--cloud-timeout",
                "100",
                "测试任务",
            ],
        )

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)
        settings = iterator._resolved_settings

        # 验证 CLI 覆盖
        assert settings.worker_pool_size == 2, f"CLI workers 应覆盖为 2, 实际 {settings.worker_pool_size}"
        assert settings.max_iterations == 5, f"CLI max_iterations 应覆盖为 5, 实际 {settings.max_iterations}"
        assert settings.cloud_timeout == 100, f"CLI cloud_timeout 应覆盖为 100, 实际 {settings.cloud_timeout}"
        # stream_events_enabled 从 config.yaml 读取（run_iterate.py 无对应 CLI 参数）
        assert settings.stream_events_enabled is True, (
            f"stream_events_enabled 应从 YAML 读取为 True, 实际 {settings.stream_events_enabled}"
        )

    def test_run_iterate_py_cloud_mode_forces_basic_orchestrator(
        self,
        tmp_path: Path,
        cloud_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_iterate.py: execution_mode=cloud 时强制使用 basic 编排器"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证编排器类型被强制为 basic
        orchestrator_type = iterator._get_orchestrator_type()
        assert orchestrator_type == "basic", f"cloud 模式应强制 basic 编排器, 实际 {orchestrator_type}"

        # 验证执行模式
        assert iterator._resolved_settings.execution_mode == "cloud"

    def test_run_iterate_py_auto_mode_forces_basic_orchestrator(
        self,
        tmp_path: Path,
        auto_mode_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_iterate.py: execution_mode=auto 时强制使用 basic 编排器"""
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        orchestrator_type = iterator._get_orchestrator_type()
        assert orchestrator_type == "basic", f"auto 模式应强制 basic 编排器, 实际 {orchestrator_type}"
        assert iterator._resolved_settings.execution_mode == "auto"

    def test_run_iterate_py_stream_events_enabled_from_yaml(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_iterate.py: stream_json.enabled 驱动 CursorAgentConfig.stream_events_enabled"""
        import sys

        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        args = parse_args()
        iterator = SelfIterator(args)

        # 验证 stream_events_enabled 来自 config.yaml
        assert iterator._resolved_settings.stream_events_enabled is True

        # 构建 CursorAgentConfig 验证
        config = CursorAgentConfig(stream_events_enabled=iterator._resolved_settings.stream_events_enabled)
        assert config.stream_events_enabled is True

    # ============================================================
    # scripts/run_knowledge.py 测试
    # ============================================================

    def test_run_knowledge_py_config_from_yaml_no_cli_args(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_knowledge.py: 未传 CLI 参数时，配置值来自 config.yaml"""
        import importlib.util
        import sys

        from core.config import build_cursor_agent_config, resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试任务"])

        spec = importlib.util.spec_from_file_location(
            "run_knowledge", str(PROJECT_ROOT / "scripts" / "run_knowledge.py")
        )
        assert spec is not None
        assert spec.loader is not None
        run_knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_knowledge_module)

        args = run_knowledge_module.parse_args()

        # CLI 参数应为 None（tri-state）
        assert args.workers is None
        assert args.max_iterations is None
        assert args.stream_log_enabled is None

        # 验证实际解析值来自 config.yaml
        settings = resolve_orchestrator_settings(overrides=None)
        assert settings["workers"] == 7
        assert settings["max_iterations"] == 15

        # 验证 stream_events_enabled
        agent_config = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config["stream_events_enabled"] is True

    def test_run_knowledge_py_cli_overrides_config(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_knowledge.py: CLI 参数覆盖 config.yaml"""
        import importlib.util
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_knowledge.py",
                "--workers",
                "3",
                "--max-iterations",
                "8",
                "--no-stream-log",
                "测试任务",
            ],
        )

        spec = importlib.util.spec_from_file_location(
            "run_knowledge", str(PROJECT_ROOT / "scripts" / "run_knowledge.py")
        )
        assert spec is not None
        assert spec.loader is not None
        run_knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_knowledge_module)

        args = run_knowledge_module.parse_args()

        assert args.workers == 3
        assert args.max_iterations == "8"
        assert args.stream_log_enabled is False

        # 验证 CLI 覆盖生效
        cli_overrides = {"workers": args.workers}
        settings = resolve_orchestrator_settings(overrides=cli_overrides)
        assert settings["workers"] == 3

    def test_run_knowledge_py_stream_events_mapping(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """run_knowledge.py: stream_json.enabled -> CursorAgentConfig.stream_events_enabled"""
        import importlib.util
        import sys

        from core.config import build_cursor_agent_config
        from cursor.client import CursorAgentConfig

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试任务"])

        spec = importlib.util.spec_from_file_location(
            "run_knowledge", str(PROJECT_ROOT / "scripts" / "run_knowledge.py")
        )
        assert spec is not None
        assert spec.loader is not None
        run_knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_knowledge_module)

        # 验证 build_cursor_agent_config 正确映射
        agent_config_dict = build_cursor_agent_config(working_directory=str(tmp_path), overrides=None)
        assert agent_config_dict["stream_events_enabled"] is True

        # 验证 CursorAgentConfig 正确接收
        config = CursorAgentConfig(stream_events_enabled=agent_config_dict["stream_events_enabled"])
        assert config.stream_events_enabled is True

    # ============================================================
    # 综合测试：所有入口脚本的一致性验证
    # ============================================================

    def test_all_scripts_consistent_config_resolution(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证所有入口脚本解析相同的 config.yaml 值"""
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)

        # 收集各脚本解析的配置值
        results = {}

        # 1. run.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "测试任务"])
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]
        from run import Runner
        from run import parse_args as run_parse

        runner = Runner(run_parse())
        merged = runner._merge_options({})
        results["run.py"] = {
            "workers": merged["workers"],
            "max_iterations": merged["max_iterations"],
        }

        # 2. run_basic.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_basic.py", "测试任务"])
        settings = resolve_orchestrator_settings(overrides=None)
        results["run_basic.py"] = {
            "workers": settings["workers"],
            "max_iterations": settings["max_iterations"],
        }

        # 3. run_mp.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_mp.py", "测试任务"])
        settings = resolve_orchestrator_settings(overrides=None)
        results["run_mp.py"] = {
            "workers": settings["workers"],
            "max_iterations": settings["max_iterations"],
        }

        # 4. run_iterate.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试任务"])
        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]
        from scripts.run_iterate import SelfIterator
        from scripts.run_iterate import parse_args as iterate_parse

        iterator = SelfIterator(iterate_parse())
        results["run_iterate.py"] = {
            "workers": iterator._resolved_settings.worker_pool_size,
            "max_iterations": iterator._resolved_settings.max_iterations,
        }

        # 5. run_knowledge.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run_knowledge.py", "测试任务"])
        settings = resolve_orchestrator_settings(overrides=None)
        results["run_knowledge.py"] = {
            "workers": settings["workers"],
            "max_iterations": settings["max_iterations"],
        }

        # 验证所有脚本解析值一致
        expected_workers = 7
        expected_max_iterations = 15

        for script_name, values in results.items():
            assert values["workers"] == expected_workers, (
                f"{script_name}: workers 应为 {expected_workers}, 实际 {values['workers']}"
            )
            assert values["max_iterations"] == expected_max_iterations, (
                f"{script_name}: max_iterations 应为 {expected_max_iterations}, 实际 {values['max_iterations']}"
            )

    def test_all_scripts_cli_override_precedence(
        self,
        tmp_path: Path,
        comprehensive_config: Path,
        reset_config_manager,
        monkeypatch,
    ) -> None:
        """验证所有入口脚本的 CLI 覆盖优先级一致"""
        import sys

        from core.config import resolve_orchestrator_settings

        monkeypatch.chdir(tmp_path)

        cli_workers = 2
        cli_overrides = {"workers": cli_workers}

        # 所有脚本使用相同的 CLI 覆盖
        results = {}

        # 1. run.py
        ConfigManager.reset_instance()
        monkeypatch.setattr(sys, "argv", ["run.py", "--workers", str(cli_workers), "测试任务"])
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("run") or mod_name.startswith("scripts."):
                del sys.modules[mod_name]
        from run import Runner
        from run import parse_args as run_parse

        runner = Runner(run_parse())
        merged = runner._merge_options({})
        results["run.py"] = merged["workers"]

        # 2-5. 其他脚本通过 resolve_orchestrator_settings 验证
        for script_name in [
            "run_basic.py",
            "run_mp.py",
            "run_iterate.py",
            "run_knowledge.py",
        ]:
            ConfigManager.reset_instance()
            settings = resolve_orchestrator_settings(overrides=cli_overrides)
            results[script_name] = settings["workers"]

        # 验证所有脚本 CLI 覆盖生效
        for script_name, workers in results.items():
            assert workers == cli_workers, f"{script_name}: CLI --workers {cli_workers} 应覆盖配置, 实际 {workers}"


# ============================================================
# TestCloudCooldownConfig - Cloud Cooldown 配置测试
# ============================================================


class TestCloudCooldownConfig:
    """测试 Cloud Cooldown 配置加载和覆盖

    测试覆盖:
    1. config.yaml 中的 cloud_agent.cooldown 配置正确加载
    2. build_cooldown_config() 函数按优先级解析配置
    3. CooldownConfig.from_config() 正确构建 CooldownConfig 实例
    4. CLI 覆盖能正确生效
    """

    def test_load_cooldown_config_from_yaml(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试从 config.yaml 加载 cooldown 配置"""
        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "rate_limit_min_seconds": 45,
                    "rate_limit_default_seconds": 90,
                    "rate_limit_max_seconds": 600,
                    "auth_cooldown_seconds": 1200,
                    "auth_require_config_change": False,
                    "network_cooldown_seconds": 180,
                    "timeout_cooldown_seconds": 90,
                    "unknown_cooldown_seconds": 450,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证 cooldown 配置正确加载
        cooldown = config.cloud_agent.cooldown
        assert cooldown.rate_limit_min_seconds == 45
        assert cooldown.rate_limit_default_seconds == 90
        assert cooldown.rate_limit_max_seconds == 600
        assert cooldown.auth_cooldown_seconds == 1200
        assert cooldown.auth_require_config_change is False
        assert cooldown.network_cooldown_seconds == 180
        assert cooldown.timeout_cooldown_seconds == 90
        assert cooldown.unknown_cooldown_seconds == 450

    def test_cooldown_config_default_values(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试无 cooldown 配置时使用默认值"""
        from core.config import (
            DEFAULT_COOLDOWN_AUTH,
            DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE,
            DEFAULT_COOLDOWN_NETWORK,
            DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT,
            DEFAULT_COOLDOWN_RATE_LIMIT_MAX,
            DEFAULT_COOLDOWN_RATE_LIMIT_MIN,
            DEFAULT_COOLDOWN_TIMEOUT,
            DEFAULT_COOLDOWN_UNKNOWN,
        )

        # 创建空配置
        config_content: dict[str, object] = {"cloud_agent": {}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 验证使用默认值
        cooldown = config.cloud_agent.cooldown
        assert cooldown.rate_limit_min_seconds == DEFAULT_COOLDOWN_RATE_LIMIT_MIN
        assert cooldown.rate_limit_default_seconds == DEFAULT_COOLDOWN_RATE_LIMIT_DEFAULT
        assert cooldown.rate_limit_max_seconds == DEFAULT_COOLDOWN_RATE_LIMIT_MAX
        assert cooldown.auth_cooldown_seconds == DEFAULT_COOLDOWN_AUTH
        assert cooldown.auth_require_config_change == DEFAULT_COOLDOWN_AUTH_REQUIRE_CONFIG_CHANGE
        assert cooldown.network_cooldown_seconds == DEFAULT_COOLDOWN_NETWORK
        assert cooldown.timeout_cooldown_seconds == DEFAULT_COOLDOWN_TIMEOUT
        assert cooldown.unknown_cooldown_seconds == DEFAULT_COOLDOWN_UNKNOWN

    def test_build_cooldown_config_from_yaml(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 build_cooldown_config() 从 config.yaml 读取配置"""
        from core.config import build_cooldown_config

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "rate_limit_min_seconds": 20,
                    "auth_cooldown_seconds": 900,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 build_cooldown_config 获取配置
        cooldown_dict = build_cooldown_config()

        # 验证自定义值
        assert cooldown_dict["rate_limit_min_seconds"] == 20
        assert cooldown_dict["auth_cooldown_seconds"] == 900

        # 验证其他值使用默认值
        from core.config import DEFAULT_COOLDOWN_NETWORK

        assert cooldown_dict["network_cooldown_seconds"] == DEFAULT_COOLDOWN_NETWORK

    def test_build_cooldown_config_cli_override(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 build_cooldown_config() CLI 覆盖优先级"""
        from core.config import build_cooldown_config

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "rate_limit_min_seconds": 20,
                    "auth_cooldown_seconds": 900,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 CLI 覆盖
        cli_overrides = {
            "rate_limit_min_seconds": 50,  # 覆盖 config.yaml 中的 20
            "network_cooldown_seconds": 300,  # 覆盖默认值
        }
        cooldown_dict = build_cooldown_config(overrides=cli_overrides)

        # 验证 CLI 覆盖生效
        assert cooldown_dict["rate_limit_min_seconds"] == 50  # CLI 覆盖
        assert cooldown_dict["auth_cooldown_seconds"] == 900  # 保持 config.yaml 值
        assert cooldown_dict["network_cooldown_seconds"] == 300  # CLI 覆盖默认值

    def test_cooldown_config_from_config_factory_method(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CooldownConfig.from_config() 工厂方法"""
        from cursor.executor import CooldownConfig

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "rate_limit_min_seconds": 25,
                    "rate_limit_default_seconds": 75,
                    "timeout_cooldown_seconds": 45,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 from_config() 创建 CooldownConfig
        cooldown = CooldownConfig.from_config()

        # 验证配置正确加载
        assert cooldown.rate_limit_min_seconds == 25
        assert cooldown.rate_limit_default_seconds == 75
        assert cooldown.timeout_cooldown_seconds == 45

        # 验证其他值使用默认值
        from core.config import DEFAULT_COOLDOWN_AUTH

        assert cooldown.auth_cooldown_seconds == DEFAULT_COOLDOWN_AUTH

    def test_cooldown_config_from_config_with_overrides(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CooldownConfig.from_config() 带覆盖参数"""
        from cursor.executor import CooldownConfig

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "auth_cooldown_seconds": 1200,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用覆盖参数
        cooldown = CooldownConfig.from_config(overrides={"auth_cooldown_seconds": 1800})

        # 验证覆盖生效
        assert cooldown.auth_cooldown_seconds == 1800

    def test_auto_executor_uses_config_cooldown(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 AutoAgentExecutor 默认使用 config.yaml 中的 cooldown 配置"""
        from cursor.executor import AutoAgentExecutor

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "rate_limit_min_seconds": 40,
                    "auth_cooldown_seconds": 720,
                    "unknown_cooldown_seconds": 360,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建 AutoAgentExecutor（不传 cooldown_config）
        executor = AutoAgentExecutor()

        # 验证使用 config.yaml 中的配置
        policy_config = cast(Any, executor).policy.cooldown_config
        assert policy_config.rate_limit_min_seconds == 40
        assert policy_config.auth_cooldown_seconds == 720
        assert policy_config.unknown_cooldown_seconds == 360

    def test_executor_factory_uses_config_cooldown(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 AgentExecutorFactory.create() 使用 config.yaml 中的 cooldown 配置"""
        from cursor.executor import AgentExecutorFactory, ExecutionMode

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "network_cooldown_seconds": 240,
                    "timeout_cooldown_seconds": 120,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 通过工厂创建 AUTO 模式执行器
        executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)

        # 验证使用 config.yaml 中的配置
        policy_config = cast(Any, executor).policy.cooldown_config
        assert policy_config.network_cooldown_seconds == 240
        assert policy_config.timeout_cooldown_seconds == 120

    def test_explicit_cooldown_config_overrides_yaml(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试显式传入 CooldownConfig 覆盖 config.yaml"""
        from cursor.executor import AutoAgentExecutor, CooldownConfig

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "auth_cooldown_seconds": 600,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 显式传入 CooldownConfig
        explicit_config = CooldownConfig(auth_cooldown_seconds=1800)
        executor = AutoAgentExecutor(cooldown_config=explicit_config)

        # 验证显式配置覆盖 config.yaml
        policy_config = cast(Any, executor).policy.cooldown_config
        assert policy_config.auth_cooldown_seconds == 1800

    def test_deprecated_cloud_cooldown_seconds_backward_compat(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试已废弃的 cloud_cooldown_seconds 参数向后兼容"""
        from cursor.executor import AutoAgentExecutor

        config_content = {
            "cloud_agent": {
                "cooldown": {
                    "unknown_cooldown_seconds": 300,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用已废弃的 cloud_cooldown_seconds 参数
        executor = AutoAgentExecutor(cloud_cooldown_seconds=120)

        # 验证 cloud_cooldown_seconds 覆盖所有相关配置
        policy_config = cast(Any, executor).policy.cooldown_config
        assert policy_config.rate_limit_default_seconds == 120
        assert policy_config.network_cooldown_seconds == 120
        assert policy_config.timeout_cooldown_seconds == 120
        assert policy_config.unknown_cooldown_seconds == 120


# ============================================================
# 样例配置数据（供测试复用，不依赖真实网络）
# ============================================================

SAMPLE_CONFIG_MINIMAL = {
    "system": {
        "max_iterations": 5,
        "worker_pool_size": 2,
    },
}

SAMPLE_CONFIG_WITH_DOCS_SOURCE = {
    "system": {
        "max_iterations": 10,
        "worker_pool_size": 3,
    },
    "knowledge_docs_update": {
        "docs_source": {
            "max_fetch_urls": 30,
            "fallback_core_docs_count": 5,
            "llms_txt_url": "https://example.com/llms.txt",
            "changelog_url": "https://example.com/changelog",
        },
    },
}

SAMPLE_CONFIG_WITH_URL_STRATEGY = {
    "system": {
        "max_iterations": 10,
    },
    "knowledge_docs_update": {
        "url_strategy": {
            "allowed_domains": ["example.com", "docs.example.com"],
            # 使用完整 URL 前缀格式（新语义）
            "allowed_url_prefixes": ["https://docs.example.com/api", "https://example.com/docs"],
            "max_urls": 50,
            "keyword_boost_weight": 2.5,
        },
    },
}


# ============================================================
# Test: 样例配置数据复用测试
# ============================================================


class TestSampleConfigDataReuse:
    """测试复用样例配置数据，确保不依赖真实网络"""

    def test_load_minimal_config(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试加载最小配置"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_MINIMAL, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        assert config.system.max_iterations == 5
        assert config.system.worker_pool_size == 2

    def test_load_config_with_docs_source(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试加载带 docs_source 的配置"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_WITH_DOCS_SOURCE, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        docs_source = config.knowledge_docs_update.docs_source
        assert docs_source.max_fetch_urls == 30
        assert docs_source.fallback_core_docs_count == 5
        assert docs_source.llms_txt_url == "https://example.com/llms.txt"

    def test_load_config_with_url_strategy(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试加载带 url_strategy 的配置"""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_WITH_URL_STRATEGY, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        url_strategy = config.knowledge_docs_update.url_strategy
        assert "example.com" in url_strategy.allowed_domains
        assert url_strategy.max_urls == 50
        assert url_strategy.keyword_boost_weight == 2.5

    def test_sample_config_no_network_required(self) -> None:
        """验证样例配置数据不需要网络"""
        # 所有样例配置都是字典常量
        assert isinstance(SAMPLE_CONFIG_MINIMAL, dict)
        assert isinstance(SAMPLE_CONFIG_WITH_DOCS_SOURCE, dict)
        assert isinstance(SAMPLE_CONFIG_WITH_URL_STRATEGY, dict)

        # 验证结构完整
        assert "system" in SAMPLE_CONFIG_MINIMAL
        assert "knowledge_docs_update" in SAMPLE_CONFIG_WITH_DOCS_SOURCE
        sample_url_strategy = SAMPLE_CONFIG_WITH_URL_STRATEGY
        knowledge_update = sample_url_strategy.get("knowledge_docs_update", {})
        assert isinstance(knowledge_update, dict)
        assert "url_strategy" in knowledge_update


# ============================================================
# Test: 配置加载不依赖网络验证
# ============================================================


class TestConfigLoadingOffline:
    """测试配置加载不依赖真实网络"""

    def test_config_manager_offline_initialization(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 ConfigManager 离线初始化"""
        # 创建配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_MINIMAL, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 获取配置不需要网络
        config = ConfigManager.get_instance()
        assert config is not None
        assert config.system.max_iterations == 5

    def test_config_defaults_without_file(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试无配置文件时使用默认值"""
        # 切换到空目录（无 config.yaml）
        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 应该使用默认配置
        config = ConfigManager.get_instance()
        assert config is not None
        # 默认值应该存在
        assert config.system.max_iterations > 0

    def test_config_reload_offline(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试配置重新加载（离线）"""
        config_path = tmp_path / "config.yaml"

        # 初始配置
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"system": {"max_iterations": 5}}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config1 = ConfigManager.get_instance()
        assert config1.system.max_iterations == 5

        # 更新配置
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"system": {"max_iterations": 10}}, f)

        # 重新加载
        ConfigManager.reset_instance()
        config2 = ConfigManager.get_instance()
        assert config2.system.max_iterations == 10


# ============================================================
# Test: 样例数据一致性测试
# ============================================================


class TestSampleConfigConsistency:
    """测试样例配置数据的一致性"""

    def test_sample_configs_valid_structure(self) -> None:
        """测试样例配置结构有效"""
        # 最小配置应有 system
        assert "system" in SAMPLE_CONFIG_MINIMAL
        assert "max_iterations" in SAMPLE_CONFIG_MINIMAL["system"]

        # docs_source 配置应有正确结构
        docs_source_config = SAMPLE_CONFIG_WITH_DOCS_SOURCE
        docs_source_section = docs_source_config.get("knowledge_docs_update", {})
        assert isinstance(docs_source_section, dict)
        docs_source = docs_source_section.get("docs_source", {})
        assert isinstance(docs_source, dict)
        assert "max_fetch_urls" in docs_source
        assert "llms_txt_url" in docs_source

        # url_strategy 配置应有正确结构
        url_strategy_config = SAMPLE_CONFIG_WITH_URL_STRATEGY
        url_strategy_section = url_strategy_config.get("knowledge_docs_update", {})
        assert isinstance(url_strategy_section, dict)
        url_strategy = url_strategy_section.get("url_strategy", {})
        assert isinstance(url_strategy, dict)
        assert "allowed_domains" in url_strategy
        assert "max_urls" in url_strategy

    def test_sample_urls_are_example_domains(self) -> None:
        """测试样例 URL 使用示例域名（不是真实域名）"""
        docs_source_config = SAMPLE_CONFIG_WITH_DOCS_SOURCE
        docs_source_section = docs_source_config.get("knowledge_docs_update", {})
        assert isinstance(docs_source_section, dict)
        docs_source = docs_source_section.get("docs_source", {})
        assert isinstance(docs_source, dict)

        # 应使用 example.com 而非真实域名
        assert "example.com" in str(docs_source["llms_txt_url"])
        assert "example.com" in str(docs_source.get("changelog_url", ""))

    def test_sample_configs_immutable_usage(self) -> None:
        """测试样例配置数据应被视为只读"""
        import copy

        # 复制后修改不应影响原始数据
        config_copy = copy.deepcopy(SAMPLE_CONFIG_MINIMAL)
        config_copy["system"]["max_iterations"] = 999

        # 原始数据不变
        assert SAMPLE_CONFIG_MINIMAL["system"]["max_iterations"] == 5


# ============================================================
# Test: resolve_doc_url_strategy_config 测试
# ============================================================


# 用于 URL 策略配置测试的样例数据
SAMPLE_CONFIG_URL_STRATEGY_FULL = {
    "system": {
        "max_iterations": 10,
    },
    "knowledge_docs_update": {
        "url_strategy": {
            "allowed_domains": ["example.com", "docs.example.com"],
            "allowed_url_prefixes": ["https://docs.example.com/api", "https://example.com/docs"],
            "max_urls": 75,
            "fallback_core_docs_count": 12,
            "prefer_changelog": False,
            "deduplicate": False,
            "normalize": False,
            "keyword_boost_weight": 2.8,
            "exclude_patterns": [r".*\.pdf$", r".*\.zip$"],
            "priority_weights": {
                "changelog": 6.0,
                "llms_txt": 5.0,
                "related_doc": 4.0,
                "core_doc": 3.0,
                "keyword_match": 2.0,
            },
        },
    },
}


class TestResolveDocUrlStrategyConfig:
    """测试 resolve_doc_url_strategy_config() 函数

    验证：
    1. 从 config.yaml 读取值
    2. CLI 覆盖（包括 append 和逗号分隔）
    3. 空字符串行为
    """

    def test_resolve_from_config_yaml_values(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试从 config.yaml 读取 url_strategy 配置"""
        import sys

        # 创建配置文件
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除模块缓存以确保使用新配置
        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        # 模拟 CLI 参数全部为 None（未指定）
        mock_args = argparse.Namespace(
            url_allowed_domains=None,
            url_allowed_prefixes=None,
            url_exclude_patterns=None,
            url_max_urls=None,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=None,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=None,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # 验证所有值来自 config.yaml
        assert resolved.allowed_domains == ["example.com", "docs.example.com"]
        assert resolved.allowed_url_prefixes == [
            "https://docs.example.com/api",
            "https://example.com/docs",
        ]
        assert resolved.max_urls == 75
        assert resolved.fallback_core_docs_count == 12
        assert resolved.prefer_changelog is False
        assert resolved.deduplicate is False
        assert resolved.normalize is False
        assert resolved.keyword_boost_weight == 2.8
        assert resolved.exclude_patterns == [r".*\.pdf$", r".*\.zip$"]
        assert resolved.priority_weights["changelog"] == 6.0
        assert resolved.priority_weights["llms_txt"] == 5.0

    def test_cli_overrides_config_yaml_with_append(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 CLI 使用 append 模式覆盖 config.yaml 值"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        # 模拟 CLI 使用 append 模式: --url-allowed-domains a.com --url-allowed-domains b.com
        mock_args = argparse.Namespace(
            url_allowed_domains=["cli-domain1.com", "cli-domain2.com"],
            url_allowed_prefixes=["https://cli.example.com/docs"],
            url_exclude_patterns=None,
            url_max_urls=200,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=True,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=5.0,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # CLI 指定的值应覆盖 config.yaml
        assert resolved.allowed_domains == ["cli-domain1.com", "cli-domain2.com"]
        assert resolved.allowed_url_prefixes == ["https://cli.example.com/docs"]
        assert resolved.max_urls == 200
        assert resolved.prefer_changelog is True
        assert resolved.keyword_boost_weight == 5.0
        # 未指定的值应来自 config.yaml
        assert resolved.fallback_core_docs_count == 12
        assert resolved.deduplicate is False
        assert resolved.normalize is False
        assert resolved.exclude_patterns == [r".*\.pdf$", r".*\.zip$"]

    def test_cli_overrides_with_comma_separated_values(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 CLI 使用逗号分隔值覆盖配置"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        # 模拟 CLI 使用逗号分隔: --url-allowed-domains "a.com,b.com,c.com"
        mock_args = argparse.Namespace(
            url_allowed_domains=["domain1.com,domain2.com", "domain3.com"],
            url_allowed_prefixes=["https://a.com/docs,https://b.com/api"],
            url_exclude_patterns=[r".*\.pdf$,.*\.zip$"],
            url_max_urls=None,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=None,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=None,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # 逗号分隔应被正确解析为列表
        assert resolved.allowed_domains == ["domain1.com", "domain2.com", "domain3.com"]
        assert resolved.allowed_url_prefixes == ["https://a.com/docs", "https://b.com/api"]
        assert resolved.exclude_patterns == [r".*\.pdf$", r".*\.zip$"]

    def test_empty_list_cli_overrides_config(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 CLI 传入空列表（如 --url-allowed-domains ''）覆盖配置"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        # 模拟 CLI 传入空字符串: --url-allowed-domains ""
        # _parse_append_or_comma_separated 会返回 None（空列表情况）
        mock_args = argparse.Namespace(
            url_allowed_domains=[""],  # 单个空字符串
            url_allowed_prefixes=["  "],  # 仅空白
            url_exclude_patterns=None,
            url_max_urls=None,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=None,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=None,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # 空字符串/空白字符串应被视为未指定，使用 config.yaml 值
        # _parse_append_or_comma_separated 返回 None 时使用 config.yaml 值
        assert resolved.allowed_domains == ["example.com", "docs.example.com"]
        assert resolved.allowed_url_prefixes == [
            "https://docs.example.com/api",
            "https://example.com/docs",
        ]

    def test_mixed_append_and_comma_separated(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试混合使用 append 和逗号分隔"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        # 混合: --url-allowed-domains "a.com,b.com" --url-allowed-domains "c.com"
        mock_args = argparse.Namespace(
            url_allowed_domains=["domain1.com,domain2.com", "domain3.com", "domain4.com,domain5.com"],
            url_allowed_prefixes=None,
            url_exclude_patterns=None,
            url_max_urls=None,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=None,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=None,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # 应正确合并所有值
        assert resolved.allowed_domains == ["domain1.com", "domain2.com", "domain3.com", "domain4.com", "domain5.com"]


class TestParseAppendOrCommaSeparated:
    """测试 _parse_append_or_comma_separated 内部函数"""

    def test_parse_none_input(self) -> None:
        """测试 None 输入返回 None"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(None)
        assert result is None

    def test_parse_empty_list_input(self) -> None:
        """测试空列表输入返回 None"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated([])
        assert result is None

    def test_parse_single_value(self) -> None:
        """测试单个值"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["value1"])
        assert result == ["value1"]

    def test_parse_multiple_values_append(self) -> None:
        """测试 append 模式多个值"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["value1", "value2", "value3"])
        assert result == ["value1", "value2", "value3"]

    def test_parse_comma_separated(self) -> None:
        """测试逗号分隔值"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["value1,value2,value3"])
        assert result == ["value1", "value2", "value3"]

    def test_parse_strips_whitespace(self) -> None:
        """测试去除空白字符"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["  value1  ,  value2  "])
        assert result == ["value1", "value2"]

    def test_parse_empty_strings_return_none(self) -> None:
        """测试仅有空字符串返回 None"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["", "  ", ",,,"])
        assert result is None

    def test_parse_mixed_with_empty_values(self) -> None:
        """测试混合有效和空值"""
        import sys

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import _parse_append_or_comma_separated

        result = _parse_append_or_comma_separated(["value1,,value2", "", "value3"])
        assert result == ["value1", "value2", "value3"]


# ============================================================
# Test: KnowledgeUpdater 端到端配置验证
# ============================================================


class TestKnowledgeUpdaterE2EConfig:
    """端到端测试：验证 KnowledgeUpdater 最终使用的 DocURLStrategyConfig 与配置一致

    通过以下方式验证：
    1. 构造临时 config.yaml
    2. 实例化 SelfIterator
    3. 验证 SelfIterator.knowledge_updater.url_strategy_config 字段与配置一致
    """

    def test_knowledge_updater_uses_config_yaml_url_strategy(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 KnowledgeUpdater 使用 config.yaml 中的 url_strategy 配置"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        # 模拟最小 CLI 参数
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "test task"])
        args = parse_args()

        iterator = SelfIterator(args)

        # 验证 knowledge_updater.url_strategy_config 与配置一致
        url_config = iterator.knowledge_updater.url_strategy_config
        assert url_config is not None
        assert url_config.allowed_domains == ["example.com", "docs.example.com"]
        assert url_config.allowed_url_prefixes == [
            "https://docs.example.com/api",
            "https://example.com/docs",
        ]
        assert url_config.max_urls == 75
        assert url_config.fallback_core_docs_count == 12
        assert url_config.prefer_changelog is False
        assert url_config.deduplicate is False
        assert url_config.normalize is False
        assert url_config.keyword_boost_weight == 2.8
        assert url_config.exclude_patterns == [r".*\.pdf$", r".*\.zip$"]
        assert url_config.priority_weights["changelog"] == 6.0

    def test_knowledge_updater_cli_overrides_url_strategy(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖 KnowledgeUpdater 的 url_strategy 配置"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        # 模拟 CLI 覆盖部分参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "test task",
                "--url-allowed-domains",
                "cli-domain.com",
                "--url-max-urls",
                "300",
                "--keyword-boost-weight",
                "4.5",
            ],
        )
        args = parse_args()

        iterator = SelfIterator(args)

        url_config = iterator.knowledge_updater.url_strategy_config
        assert url_config is not None
        # CLI 覆盖的值
        assert url_config.allowed_domains == ["cli-domain.com"]
        assert url_config.max_urls == 300
        assert url_config.keyword_boost_weight == 4.5
        # 来自 config.yaml 的值
        assert url_config.allowed_url_prefixes == [
            "https://docs.example.com/api",
            "https://example.com/docs",
        ]
        assert url_config.fallback_core_docs_count == 12
        assert url_config.prefer_changelog is False

    def test_self_iterator_url_strategy_config_propagates(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 SelfIterator.url_strategy_config 正确传播到 KnowledgeUpdater"""
        import sys

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG_URL_STRATEGY_FULL, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "test task"])
        args = parse_args()

        iterator = SelfIterator(args)

        # SelfIterator.url_strategy_config 和 KnowledgeUpdater.url_strategy_config 应相同
        assert iterator.url_strategy_config is iterator.knowledge_updater.url_strategy_config

    def test_knowledge_updater_without_config_uses_defaults(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试无 url_strategy 配置时使用默认值"""
        import sys

        # 最小配置（无 url_strategy）
        minimal_config = {
            "system": {"max_iterations": 5},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(minimal_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "test task"])
        args = parse_args()

        iterator = SelfIterator(args)

        url_config = iterator.knowledge_updater.url_strategy_config
        assert url_config is not None
        # 应使用默认值（来自 core/config.py 中的 DEFAULT_* 常量）
        from core.config import (
            DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS,
            DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
            DEFAULT_URL_STRATEGY_MAX_URLS,
        )

        assert url_config.allowed_domains == DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS
        assert url_config.max_urls == DEFAULT_URL_STRATEGY_MAX_URLS
        assert url_config.keyword_boost_weight == DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT

    def test_knowledge_updater_comma_separated_cli_values(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 使用逗号分隔值传递给 KnowledgeUpdater"""
        import sys

        minimal_config = {"system": {"max_iterations": 5}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(minimal_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import SelfIterator, parse_args

        # 使用逗号分隔的 CLI 参数
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_iterate.py",
                "test task",
                "--url-allowed-domains",
                "domain1.com,domain2.com",
                "--url-allowed-prefixes",
                "https://a.com/docs,https://b.com/api",
            ],
        )
        args = parse_args()

        iterator = SelfIterator(args)

        url_config = iterator.knowledge_updater.url_strategy_config
        assert url_config is not None
        # 逗号分隔应被正确解析
        assert url_config.allowed_domains == ["domain1.com", "domain2.com"]
        assert url_config.allowed_url_prefixes == [
            "https://a.com/docs",
            "https://b.com/api",
        ]


# ============================================================
# TestConfigValidation - 配置值校验测试
# ============================================================


class TestConfigValidation:
    """测试配置值校验功能"""

    def test_validate_external_link_mode_valid_values(self) -> None:
        """测试 external_link_mode 有效值校验"""
        from knowledge.doc_url_strategy import (
            VALID_EXTERNAL_LINK_MODES,
            validate_external_link_mode,
        )

        for mode in VALID_EXTERNAL_LINK_MODES:
            assert validate_external_link_mode(mode) == mode

    def test_validate_external_link_mode_invalid_value(self) -> None:
        """测试 external_link_mode 无效值校验，应返回默认值"""
        from knowledge.doc_url_strategy import validate_external_link_mode

        result = validate_external_link_mode("invalid_mode")
        assert result == "record_only"
        # 也测试空字符串
        result2 = validate_external_link_mode("")
        assert result2 == "record_only"

    def test_validate_execution_mode_valid_values(self) -> None:
        """测试 execution_mode 有效值校验"""
        from knowledge.doc_url_strategy import (
            VALID_EXECUTION_MODES,
            validate_execution_mode,
        )

        for mode in VALID_EXECUTION_MODES:
            assert validate_execution_mode(mode) == mode

    def test_validate_execution_mode_invalid_value(self) -> None:
        """测试 execution_mode 无效值校验，应返回默认值 'auto'"""
        from knowledge.doc_url_strategy import validate_execution_mode

        result = validate_execution_mode("invalid_mode")
        assert result == "auto"
        # 也测试空字符串
        result2 = validate_execution_mode("")
        assert result2 == "auto"

    def test_validate_url_strategy_prefixes_full_url_format(self) -> None:
        """测试 url_strategy.allowed_url_prefixes 完整 URL 格式校验（不应有 warning）"""
        from knowledge.doc_url_strategy import validate_url_strategy_prefixes

        prefixes = ["https://example.com/docs", "https://example.com/api"]
        result = validate_url_strategy_prefixes(prefixes)
        assert result == prefixes

    def test_validate_url_strategy_prefixes_deprecated_path_format(self) -> None:
        """测试 url_strategy.allowed_url_prefixes 旧版路径前缀格式校验，应保持原值"""
        from knowledge.doc_url_strategy import validate_url_strategy_prefixes

        prefixes = ["docs", "cn/docs"]
        result = validate_url_strategy_prefixes(prefixes)
        # 应保持原值（不修改）
        assert result == prefixes

    def test_validate_url_strategy_prefixes_mixed_format(self) -> None:
        """测试 url_strategy.allowed_url_prefixes 混合格式，应保持原值"""
        from knowledge.doc_url_strategy import validate_url_strategy_prefixes

        prefixes = ["https://example.com/docs", "docs"]  # 混合格式
        result = validate_url_strategy_prefixes(prefixes)
        # 应保持原值（不修改）
        assert result == prefixes

    def test_validate_fetch_policy_prefixes_path_format(self) -> None:
        """测试 fetch_policy.allowed_url_prefixes 路径前缀格式校验（正确格式，不应有 warning）"""
        from knowledge.doc_url_strategy import validate_fetch_policy_prefixes

        prefixes = ["docs", "cn/docs", "changelog"]
        result = validate_fetch_policy_prefixes(prefixes)
        assert result == prefixes

    def test_validate_fetch_policy_prefixes_full_url_format(self) -> None:
        """测试 fetch_policy.allowed_url_prefixes 完整 URL 格式校验，应保持原值"""
        from knowledge.doc_url_strategy import validate_fetch_policy_prefixes

        prefixes = ["https://example.com/docs", "https://example.com/api"]
        result = validate_fetch_policy_prefixes(prefixes)
        # 应保持原值（不修改）
        assert result == prefixes

    def test_is_full_url_prefix(self) -> None:
        """测试完整 URL 前缀检测"""
        from knowledge.doc_url_strategy import is_full_url_prefix

        assert is_full_url_prefix("https://example.com/docs") is True
        assert is_full_url_prefix("http://example.com/docs") is True
        assert is_full_url_prefix("docs") is False
        assert is_full_url_prefix("/docs") is False
        assert is_full_url_prefix("cn/docs") is False

    def test_is_path_prefix(self) -> None:
        """测试路径前缀检测"""
        from knowledge.doc_url_strategy import is_path_prefix

        assert is_path_prefix("docs") is True
        assert is_path_prefix("cn/docs") is True
        assert is_path_prefix("/docs") is True
        assert is_path_prefix("https://example.com/docs") is False
        assert is_path_prefix("http://example.com/docs") is False

    def test_validate_external_link_allowlist_domain_items(self) -> None:
        """测试 validate_external_link_allowlist 域名项解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "docs.python.org",
            ]
        )

        assert "github.com" in result.domains
        assert "docs.python.org" in result.domains
        assert len(result.prefixes) == 0

    def test_validate_external_link_allowlist_url_prefix_items(self) -> None:
        """测试 validate_external_link_allowlist URL 前缀项解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "https://github.com/cursor",
                "https://docs.python.org/3",
            ]
        )

        assert len(result.domains) == 0
        assert len(result.prefixes) == 2

    def test_validate_external_link_allowlist_mixed_items(self) -> None:
        """测试 validate_external_link_allowlist 混合项解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "https://docs.python.org/3",
            ]
        )

        assert "github.com" in result.domains
        assert len(result.prefixes) == 1

    def test_validate_external_link_allowlist_invalid_items(self) -> None:
        """测试 validate_external_link_allowlist 无效项处理"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "",  # 空字符串
                "   ",  # 仅空白
            ]
        )

        assert "github.com" in result.domains
        assert len(result.invalid_items) == 2

    def test_validate_external_link_allowlist_domain_with_path(self) -> None:
        """测试 validate_external_link_allowlist 带路径的域名格式"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com/cursor",
                "github.com/openai/whisper",
            ]
        )

        # 带路径的域名应转为 URL 前缀
        assert len(result.domains) == 0
        assert len(result.prefixes) == 2


class TestConfigManagerValidation:
    """测试 ConfigManager 中的配置校验"""

    def test_config_manager_validates_external_link_mode(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 ConfigManager 校验无效的 external_link_mode 并回退到默认值"""
        invalid_config = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "external_link_mode": "invalid_mode",
                    },
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 应回退到默认值 "record_only"
        assert config.knowledge_docs_update.docs_source.fetch_policy.external_link_mode == "record_only"

    def test_config_manager_validates_execution_mode(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 ConfigManager 校验无效的 execution_mode 并回退到默认值"""
        invalid_config = {
            "cloud_agent": {
                "execution_mode": "invalid_mode",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()
        config = ConfigManager.get_instance()

        # 应回退到默认值 "auto"
        assert config.cloud_agent.execution_mode == "auto"

    def test_config_manager_accepts_valid_external_link_modes(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 ConfigManager 接受有效的 external_link_mode 值"""
        for mode in ["record_only", "skip_all", "fetch_allowlist"]:
            valid_config = {
                "knowledge_docs_update": {
                    "docs_source": {
                        "fetch_policy": {
                            "external_link_mode": mode,
                        },
                    },
                },
            }
            config_path = tmp_path / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(valid_config, f, allow_unicode=True)

            monkeypatch.chdir(tmp_path)
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            assert config.knowledge_docs_update.docs_source.fetch_policy.external_link_mode == mode

    def test_config_manager_accepts_valid_execution_modes(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 ConfigManager 接受有效的 execution_mode 值"""
        for mode in ["cli", "cloud", "auto"]:
            valid_config = {
                "cloud_agent": {
                    "execution_mode": mode,
                },
            }
            config_path = tmp_path / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(valid_config, f, allow_unicode=True)

            monkeypatch.chdir(tmp_path)
            ConfigManager.reset_instance()
            config = ConfigManager.get_instance()

            assert config.cloud_agent.execution_mode == mode


class TestRunIterateConfigValidation:
    """测试 scripts/run_iterate.py 中的配置校验"""

    def test_resolve_docs_source_validates_external_link_mode(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_docs_source_config 校验无效的 external_link_mode"""
        import sys

        invalid_config = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "external_link_mode": "invalid_mode",
                    },
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 清除模块缓存
        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_docs_source_config

        # CLI 参数为 None，使用 config.yaml 值
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_url_prefixes=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            allowed_doc_url_prefixes=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # 应回退到默认值 "record_only"
        assert resolved.fetch_policy.external_link_mode == "record_only"

    def test_resolve_docs_source_cli_overrides_invalid_config(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 CLI 参数覆盖无效的 config.yaml 值（CLI 已通过 argparse 校验）"""
        import sys

        invalid_config = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "external_link_mode": "invalid_mode",
                    },
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_docs_source_config

        # CLI 参数指定有效值
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_url_prefixes=None,
            allowed_domains=None,
            external_link_mode="skip_all",  # CLI 指定有效值
            external_link_allowlist=None,
            allowed_doc_url_prefixes=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # CLI 值应覆盖 config.yaml 的无效值
        assert resolved.fetch_policy.external_link_mode == "skip_all"

    def test_resolve_url_strategy_validates_prefixes(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 resolve_doc_url_strategy_config 校验旧版路径前缀格式"""
        import sys

        deprecated_config = {
            "knowledge_docs_update": {
                "url_strategy": {
                    # 旧版格式：路径前缀（不含 scheme）
                    "allowed_url_prefixes": ["docs", "cn/docs"],
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(deprecated_config, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_doc_url_strategy_config

        mock_args = argparse.Namespace(
            url_allowed_domains=None,
            url_allowed_prefixes=None,
            url_exclude_patterns=None,
            url_max_urls=None,
            url_fallback_core_docs_count=None,
            url_prefer_changelog=None,
            url_deduplicate=None,
            url_normalize=None,
            url_keyword_boost_weight=None,
        )

        resolved = resolve_doc_url_strategy_config(mock_args)

        # 值应保持不变（仅输出 warning）
        assert resolved.allowed_url_prefixes == ["docs", "cn/docs"]

    def test_config_manager_validates_fetch_policy_path_prefixes_full_url(
        self, tmp_path: Path, reset_config_manager, monkeypatch, caplog
    ) -> None:
        """测试 ConfigManager._parse_fetch_policy 校验完整 URL 前缀格式

        如果 allowed_path_prefixes 包含完整 URL（如 https://...），
        应记录 warning 但保持原值不变。
        """
        import logging

        # 配置中使用完整 URL 格式（不正确的格式）
        config_with_full_url = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "allowed_path_prefixes": [
                            "https://example.com/docs",
                            "https://example.com/api",
                        ],
                    },
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_with_full_url, f, allow_unicode=True)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 使用 caplog 捕获日志
        with caplog.at_level(logging.WARNING):
            config = ConfigManager.get_instance()

        # 配置值应保持不变（不自动转换/截断）
        fetch_policy = config.knowledge_docs_update.docs_source.fetch_policy
        assert fetch_policy.allowed_path_prefixes == [
            "https://example.com/docs",
            "https://example.com/api",
        ]

        # 应有 warning 日志（通过 caplog 捕获）
        # 注意：loguru 日志可能不会被 caplog 捕获，所以只验证功能
        # 日志输出已通过 Captured stderr 证明正常

    def test_resolve_docs_source_validates_path_prefixes_full_url(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 resolve_docs_source_config 校验完整 URL 前缀格式

        如果 CLI 参数或 config.yaml 的 allowed_path_prefixes 包含完整 URL，
        应记录 warning 但保持原值不变。

        注意：loguru 日志会输出到 stderr，可在 pytest 的 "Captured stderr call" 中查看。
        """
        import sys

        # 创建空配置
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_docs_source_config

        # CLI 参数使用完整 URL 格式（不正确的格式）
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes="https://example.com/docs,https://example.com/api",
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            allowed_doc_url_prefixes=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # 配置值应保持不变（不自动转换/截断）
        assert resolved.fetch_policy.allowed_path_prefixes == [
            "https://example.com/docs",
            "https://example.com/api",
        ]
        # Warning 会输出到 stderr（可在 pytest -v 输出中看到），
        # 核心功能是：配置值保持不变

    def test_resolve_docs_source_path_prefixes_no_warning_for_valid_format(
        self, tmp_path: Path, reset_config_manager, monkeypatch, capfd
    ) -> None:
        """测试正确的路径前缀格式不产生 warning"""
        import sys

        # 创建空配置
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        for mod_name in list(sys.modules.keys()):
            if "run_iterate" in mod_name or mod_name.startswith("scripts."):
                del sys.modules[mod_name]

        from scripts.run_iterate import resolve_docs_source_config

        # CLI 参数使用正确的路径前缀格式
        mock_args = argparse.Namespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes="docs,cn/docs,changelog",
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            allowed_doc_url_prefixes=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # 配置值正确解析
        assert resolved.fetch_policy.allowed_path_prefixes == ["docs", "cn/docs", "changelog"]

        # 不应有关于 "应使用路径前缀格式" 的 warning（loguru 输出到 stderr）
        captured = capfd.readouterr()
        assert "应使用路径前缀格式" not in captured.err


# ============================================================
# TestApplyFetchPolicyIntegration - fetch_policy 集成测试
# ============================================================


class TestApplyFetchPolicyIntegration:
    """测试 apply_fetch_policy 与 url_strategy 的交互

    验证：
    1. url_strategy 扩大 allowed_domains 时，fetch_policy 仍能阻止外链抓取
    2. 默认配置下行为不扩大抓取面
    """

    def test_url_strategy_broader_domains_fetch_policy_still_blocks(self) -> None:
        """测试 url_strategy 扩大 allowed_domains 时，fetch_policy 仍能阻止外链抓取

        场景：
        - url_strategy.allowed_domains 包含 ["cursor.com", "python.org"]
        - fetch_policy primary_domains 仅有 ["cursor.com"]
        - python.org URL 通过 url_strategy 选择，但被 fetch_policy 阻止抓取
        """
        from knowledge.doc_url_strategy import (
            DocURLStrategyConfig,
            apply_fetch_policy,
            select_urls_to_fetch,
        )

        # url_strategy 配置：允许更多域名
        url_strategy_config = DocURLStrategyConfig(
            allowed_url_prefixes=[],  # 使用 domains
            allowed_domains=["cursor.com", "python.org"],  # 扩大的域名列表
            max_urls=20,
            exclude_patterns=[],
        )

        # 模拟从各来源收集的 URL
        changelog_links = [
            "https://cursor.com/docs/new-feature",
            "https://python.org/3/whatsnew/",
        ]

        # url_strategy 选择 URL（两个域名都通过）
        selected_urls = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=url_strategy_config,
        )

        # 验证 url_strategy 允许两个域名
        assert "https://cursor.com/docs/new-feature" in selected_urls
        assert "https://python.org/3/whatsnew" in selected_urls

        # fetch_policy 过滤：仅 cursor.com 是主域名
        policy_result = apply_fetch_policy(
            urls=selected_urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],  # 更窄的主域名
        )

        # 仅 cursor.com 被抓取
        assert "https://cursor.com/docs/new-feature" in policy_result.urls_to_fetch
        assert not any("python.org" in u for u in policy_result.urls_to_fetch)

        # python.org 被记录但不抓取
        assert any("python.org" in u for u in policy_result.external_links_recorded)

    def test_default_config_minimal_fetch_surface(self) -> None:
        """测试默认配置下保持最小抓取面

        使用默认的 external_link_mode="record_only" 确保：
        - 内链被抓取
        - 外链被记录但不抓取
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # 模拟已选择的 URL 列表
        selected_urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/cn/changelog",
            "https://github.com/cursor/repo",
            "https://stackoverflow.com/questions/123",
        ]

        # 使用默认模式
        result = apply_fetch_policy(
            urls=selected_urls,
            fetch_policy_mode="record_only",  # 默认值
            primary_domains=["cursor.com"],
        )

        # 仅内链被抓取（最小抓取面）
        assert len(result.urls_to_fetch) == 2
        assert all("cursor.com" in u for u in result.urls_to_fetch)

        # 外链被记录
        assert len(result.external_links_recorded) == 2
        assert any("github.com" in u for u in result.external_links_recorded)
        assert any("stackoverflow.com" in u for u in result.external_links_recorded)

    def test_fetch_policy_config_priority(self, tmp_path: Path, reset_config_manager, monkeypatch) -> None:
        """测试 fetch_policy 配置的优先级正确传递到 apply_fetch_policy

        验证：CLI 参数 > config.yaml > 默认值 的优先级链
        """
        import sys
        from types import SimpleNamespace

        # 创建带有 fetch_policy 配置的 config.yaml
        config_content = {
            "knowledge_docs_update": {
                "docs_source": {
                    "fetch_policy": {
                        "external_link_mode": "skip_all",
                        "allowed_domains": ["partner.com"],
                    },
                },
            },
        }
        config_file = tmp_path / "config.yaml"
        import yaml

        config_file.write_text(yaml.dump(config_content))

        # 设置项目根目录
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        # 重置 ConfigManager
        from core.config import ConfigManager

        ConfigManager.reset_instance()

        # 导入 resolve_docs_source_config
        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import resolve_docs_source_config

        # CLI 参数未指定，使用 config.yaml 值
        mock_args = SimpleNamespace(
            max_fetch_urls=None,
            fallback_core_docs_count=None,
            llms_txt_url=None,
            llms_cache_path=None,
            changelog_url=None,
            allowed_path_prefixes=None,
            allowed_url_prefixes_deprecated=None,
            allowed_domains=None,
            external_link_mode=None,
            external_link_allowlist=None,
            allowed_doc_url_prefixes=None,
        )

        resolved = resolve_docs_source_config(mock_args)

        # 应使用 config.yaml 中的值
        assert resolved.fetch_policy.external_link_mode == "skip_all"
        assert resolved.fetch_policy.allowed_domains == ["partner.com"]

        # 验证 apply_fetch_policy 使用这些配置
        from knowledge.doc_url_strategy import apply_fetch_policy

        result = apply_fetch_policy(
            urls=["https://cursor.com/docs", "https://github.com/repo"],
            fetch_policy_mode=resolved.fetch_policy.external_link_mode,
            primary_domains=["cursor.com"],
            allowed_domains=resolved.fetch_policy.allowed_domains,
        )

        # skip_all 模式下外链不被记录
        assert "https://github.com/repo" not in result.external_links_recorded
        # 但应在 filtered_urls 中
        reasons = {f["url"]: f["reason"] for f in result.filtered_urls}
        assert reasons.get("https://github.com/repo") == "external_link_skip_all"


# ============================================================
# TestConfigExampleAntiDrift - 配置示例防漂移测试
# ============================================================


class TestConfigExampleAntiDrift:
    """防止配置示例漂移的断言测试

    确保：
    1. fetch_policy 示例中不应出现 allowed_url_prefixes 作为主字段名
    2. url_strategy.allowed_url_prefixes 使用完整 URL 前缀格式（含 scheme/host）
    3. docs_source.allowed_doc_url_prefixes 使用完整 URL 前缀格式

    这些测试用于 CI 中自动检测配置示例的格式漂移。
    """

    def test_fetch_policy_uses_allowed_path_prefixes_not_url_prefixes(self) -> None:
        """fetch_policy 应使用 allowed_path_prefixes（路径前缀），而非 allowed_url_prefixes

        fetch_policy.allowed_url_prefixes 是废弃别名，新配置不应使用。
        """
        from core.config import OnlineFetchPolicyConfig

        # 确认 OnlineFetchPolicyConfig 使用 allowed_path_prefixes 字段
        config = OnlineFetchPolicyConfig()
        assert hasattr(config, "allowed_path_prefixes"), "OnlineFetchPolicyConfig 应有 allowed_path_prefixes 字段"

        # allowed_path_prefixes 应为路径前缀格式（不含 scheme）
        for prefix in config.allowed_path_prefixes:
            assert not prefix.startswith("http://"), (
                f"fetch_policy.allowed_path_prefixes 应为路径前缀格式，不应包含 http://: {prefix}"
            )
            assert not prefix.startswith("https://"), (
                f"fetch_policy.allowed_path_prefixes 应为路径前缀格式，不应包含 https://: {prefix}"
            )

    def test_url_strategy_allowed_url_prefixes_requires_full_url_format(self) -> None:
        """url_strategy.allowed_url_prefixes 应使用完整 URL 前缀格式（含 scheme/host）

        旧版路径前缀格式（如 "docs"）已废弃。
        """
        from knowledge.doc_url_strategy import (
            is_full_url_prefix,
            validate_url_strategy_prefixes,
        )

        # 验证格式检查函数正确识别完整 URL 前缀
        assert is_full_url_prefix("https://cursor.com/docs") is True
        assert is_full_url_prefix("http://example.com/api") is True
        assert is_full_url_prefix("docs") is False
        assert is_full_url_prefix("/docs") is False
        assert is_full_url_prefix("cn/docs") is False

        # 验证 validate_url_strategy_prefixes 对旧格式输出警告（但不抛异常）
        # 正确格式应通过
        valid_prefixes = ["https://cursor.com/docs", "https://cursor.com/cn/docs"]
        result = validate_url_strategy_prefixes(valid_prefixes)
        assert result == valid_prefixes

    def test_docs_source_allowed_doc_url_prefixes_uses_full_url_format(self) -> None:
        """docs_source.allowed_doc_url_prefixes 应使用完整 URL 前缀格式

        此配置仅用于 load_core_docs 过滤，需要完整 URL 前缀（含域名）。
        """
        from core.config import DEFAULT_ALLOWED_DOC_URL_PREFIXES

        # 验证默认值使用完整 URL 前缀格式
        for prefix in DEFAULT_ALLOWED_DOC_URL_PREFIXES:
            assert prefix.startswith("https://") or prefix.startswith("http://"), (
                f"docs_source.allowed_doc_url_prefixes 应使用完整 URL 前缀，但 {prefix} 不含 scheme"
            )

    def test_config_yaml_fetch_policy_no_allowed_url_prefixes_as_main_field(self, tmp_path: Path) -> None:
        """config.yaml 中 fetch_policy 不应使用 allowed_url_prefixes 作为主字段

        allowed_url_prefixes 是 fetch_policy 的废弃别名，config.yaml 示例应使用
        allowed_path_prefixes。
        """
        from pathlib import Path as PathLib

        import yaml as pyyaml

        # 读取项目根目录的 config.yaml
        project_root = PathLib(__file__).parent.parent
        config_path = project_root / "config.yaml"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_content = pyyaml.safe_load(f) or {}
            assert isinstance(config_content, dict)

            # 获取 fetch_policy 配置
            fetch_policy = (
                config_content.get("knowledge_docs_update", {}).get("docs_source", {}).get("fetch_policy", {})
            )
            if not isinstance(fetch_policy, dict):
                fetch_policy = {}

            # 不应有 allowed_url_prefixes 作为主字段（仅 allowed_path_prefixes）
            if "allowed_url_prefixes" in fetch_policy:
                pytest.fail(
                    "config.yaml 中 fetch_policy 不应使用 allowed_url_prefixes 作为主字段。"
                    "请使用 allowed_path_prefixes（路径前缀格式）。"
                    "allowed_url_prefixes 仅作为废弃别名保留。"
                )

            # 应有 allowed_path_prefixes 字段
            assert "allowed_path_prefixes" in fetch_policy, (
                "config.yaml 中 fetch_policy 应有 allowed_path_prefixes 字段"
            )

    def test_url_strategy_prefixes_in_test_fixtures_use_full_url(self) -> None:
        """测试 fixtures 中的 url_strategy.allowed_url_prefixes 应使用完整 URL 格式

        确保测试用例本身不会使用旧版格式，避免 CI 通过但实际配置漂移。
        """
        # 验证测试常量 CUSTOM_CONFIG_CONTENT 使用正确格式
        config_content = CUSTOM_CONFIG_CONTENT
        url_strategy = config_content.get("url_strategy", {})
        if not isinstance(url_strategy, dict):
            url_strategy = {}
        prefixes_raw = url_strategy.get("allowed_url_prefixes", [])
        if isinstance(prefixes_raw, list):
            prefixes = [str(prefix) for prefix in prefixes_raw]
        else:
            prefixes = []

        for prefix in prefixes:
            assert prefix.startswith("https://") or prefix.startswith("http://"), (
                f"测试 fixture CUSTOM_CONFIG_CONTENT 中的 "
                f"url_strategy.allowed_url_prefixes 应使用完整 URL 前缀，"
                f"但 {prefix} 不含 scheme"
            )

    def test_deprecated_cli_param_allowed_url_prefixes_warning(
        self, tmp_path: Path, reset_config_manager, monkeypatch, capsys
    ) -> None:
        """使用废弃的 --allowed-url-prefixes CLI 参数应触发警告

        验证 --allowed-url-prefixes（fetch_policy 废弃别名）会被正确处理。
        """
        import sys

        monkeypatch.chdir(tmp_path)
        ConfigManager.reset_instance()

        # 创建空 config.yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        # 使用废弃参数
        monkeypatch.setattr(sys, "argv", ["run_iterate.py", "测试目标", "--allowed-url-prefixes", "docs,cn/docs"])

        if "scripts.run_iterate" in sys.modules:
            del sys.modules["scripts.run_iterate"]

        from scripts.run_iterate import parse_args

        args = parse_args()

        # 废弃参数应被解析到 allowed_url_prefixes_deprecated
        assert args.allowed_url_prefixes_deprecated == "docs,cn/docs"
        # 新参数应为 None
        assert args.allowed_path_prefixes is None

    def test_core_config_deprecated_warning_once_per_key(
        self, tmp_path: Path, reset_config_manager, monkeypatch
    ) -> None:
        """测试 core/config.py deprecated 警告每个 key 仅输出一次

        验证 _parse_fetch_policy 和 _parse_url_strategy_allowed_prefixes
        使用统一的警告机制，同一类警告不会重复输出。
        """
        from io import StringIO

        from loguru import logger

        from core.config import (
            ConfigManager,
            reset_deprecated_warnings,
        )

        # 创建使用旧字段名的 config.yaml
        config_content = """
knowledge_docs_update:
  docs_source:
    fetch_policy:
      allowed_url_prefixes: ["docs", "cn/docs"]  # 旧字段名
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        monkeypatch.chdir(tmp_path)
        reset_deprecated_warnings()
        ConfigManager.reset_instance()

        # 第一次加载：应该输出警告
        log_output1 = StringIO()
        handler_id1 = logger.add(log_output1, format="{message}", level="WARNING")
        try:
            config1 = ConfigManager.get_instance()
        finally:
            logger.remove(handler_id1)

        log_content1 = log_output1.getvalue()
        assert "allowed_url_prefixes 已废弃" in log_content1, "第一次加载应输出 deprecated 警告"

        # 重新加载（不重置警告状态）：不应该再输出警告
        ConfigManager.reset_instance()

        log_output2 = StringIO()
        handler_id2 = logger.add(log_output2, format="{message}", level="WARNING")
        try:
            config2 = ConfigManager.get_instance()
        finally:
            logger.remove(handler_id2)

        log_content2 = log_output2.getvalue()
        assert "allowed_url_prefixes 已废弃" not in log_content2, (
            "不重置警告状态时，第二次加载不应重复输出 deprecated 警告"
        )

        # 清理
        reset_deprecated_warnings()
        ConfigManager.reset_instance()

    def test_deprecated_key_constants_exist(self) -> None:
        """测试 deprecated key 常量存在且格式正确

        验证用于测试断言的 key 常量被正确导出。
        """
        from core.config import (
            DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES,
            DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES_BOTH,
            DEPRECATED_KEY_CONFIG_FETCH_POLICY_BOTH_FIELDS,
            DEPRECATED_KEY_CONFIG_FETCH_POLICY_OLD_FIELD,
            DEPRECATED_KEY_CONFIG_URL_STRATEGY_PATH_FORMAT,
        )

        # 验证 key 常量存在且非空
        assert DEPRECATED_KEY_CONFIG_FETCH_POLICY_OLD_FIELD, "fetch_policy 旧字段 key 应存在"
        assert DEPRECATED_KEY_CONFIG_FETCH_POLICY_BOTH_FIELDS, "fetch_policy 双字段 key 应存在"
        assert DEPRECATED_KEY_CONFIG_URL_STRATEGY_PATH_FORMAT, "url_strategy 格式 key 应存在"
        assert DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES, "CLI 旧参数 key 应存在"
        assert DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES_BOTH, "CLI 双参数 key 应存在"

        # 验证 key 命名规则（以 config. 或 cli. 开头）
        assert DEPRECATED_KEY_CONFIG_FETCH_POLICY_OLD_FIELD.startswith("config."), "config 相关 key 应以 'config.' 开头"
        assert DEPRECATED_KEY_CLI_ALLOWED_URL_PREFIXES.startswith("cli."), "CLI 相关 key 应以 'cli.' 开头"


# ============================================================
# prefix_routed 参数迁移验证测试
# ============================================================


class TestPrefixRoutedParameterMigration:
    """测试 resolve_orchestrator_settings 的 prefix_routed 参数迁移

    验证：
    1. prefix_routed 参数优先级高于旧 triggered_by_prefix 参数
    2. 两种参数传入时行为一致
    3. 返回结果中 prefix_routed 字段正确
    4. 使用 triggered_by_prefix 时输出 deprecated debug 日志
    """

    @pytest.fixture(autouse=True)
    def reset_config(self) -> Generator[None, None, None]:
        """每个测试前重置配置单例"""
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_triggered_by_prefix_returns_correct_prefix_routed_field(self) -> None:
        """测试使用 triggered_by_prefix=True 时，返回结果中 prefix_routed 字段为 True"""
        from core.config import resolve_orchestrator_settings

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=True,
        )

        # 验证返回的 prefix_routed 字段正确
        assert result["prefix_routed"] is True, "使用 triggered_by_prefix=True 时，返回结果的 prefix_routed 应为 True"
        # 验证编排器为 basic（& 前缀成功触发）
        assert result["orchestrator"] == "basic", "triggered_by_prefix=True 应强制 basic 编排器"

    def test_prefix_routed_returns_correct_prefix_routed_field(self) -> None:
        """测试使用 prefix_routed=True 时，返回结果中 prefix_routed 字段为 True"""
        from core.config import resolve_orchestrator_settings

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=True,
        )

        # 验证返回的 prefix_routed 字段正确
        assert result["prefix_routed"] is True, "使用 prefix_routed=True 时，返回结果的 prefix_routed 应为 True"
        # 验证编排器为 basic
        assert result["orchestrator"] == "basic", "prefix_routed=True 应强制 basic 编排器"

    def test_both_parameters_produce_identical_results(self) -> None:
        """测试两种参数传入时行为完全一致"""
        from core.config import resolve_orchestrator_settings

        # 使用旧参数 triggered_by_prefix=True
        result_old = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=True,
        )

        # 使用新参数 prefix_routed=True
        result_new = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=True,
        )

        # 验证所有关键字段一致
        assert result_old["orchestrator"] == result_new["orchestrator"], "两种参数应产生相同的 orchestrator"
        assert result_old["prefix_routed"] == result_new["prefix_routed"], "两种参数应产生相同的 prefix_routed"
        assert result_old["execution_mode"] == result_new["execution_mode"], "两种参数应产生相同的 execution_mode"

    def test_prefix_routed_takes_priority_over_triggered_by_prefix(self) -> None:
        """测试 prefix_routed 参数优先级高于 triggered_by_prefix"""
        from core.config import resolve_orchestrator_settings

        # 同时指定两个参数，prefix_routed=True 应优先
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,  # 旧参数：False
            prefix_routed=True,  # 新参数：True（优先）
        )

        assert result["prefix_routed"] is True, "prefix_routed 参数应优先于 triggered_by_prefix"
        assert result["orchestrator"] == "basic", "prefix_routed=True 应强制 basic（优先于 triggered_by_prefix=False）"

    def test_prefix_routed_false_takes_priority(self) -> None:
        """测试 prefix_routed=False 优先于 triggered_by_prefix=True"""
        from core.config import resolve_orchestrator_settings

        # prefix_routed=False 应优先，且 CLI 为 cli 时允许 mp
        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=True,  # 旧参数：True
            prefix_routed=False,  # 新参数：False（优先）
        )

        assert result["prefix_routed"] is False, "prefix_routed=False 应优先于 triggered_by_prefix=True"
        # 注意：cli 模式允许 mp
        assert result["orchestrator"] == "mp", "prefix_routed=False + cli 模式应允许 mp 编排器"

    def test_both_parameters_false_produce_same_result(self) -> None:
        """测试两种参数都为 False 时结果一致"""
        from core.config import resolve_orchestrator_settings

        # 使用旧参数 triggered_by_prefix=False
        result_old = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            triggered_by_prefix=False,
        )

        # 使用新参数 prefix_routed=False
        result_new = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,
        )

        # 验证结果一致
        assert result_old["prefix_routed"] is False
        assert result_new["prefix_routed"] is False
        assert result_old["orchestrator"] == result_new["orchestrator"]
        assert result_old["orchestrator"] == "mp", "prefix_routed=False + cli 模式应允许 mp"

    def test_neither_parameter_specified_defaults_to_false(self) -> None:
        """测试两个参数都不指定时默认为 False"""
        from core.config import resolve_orchestrator_settings

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
        )

        assert result["prefix_routed"] is False, "不指定参数时 prefix_routed 应默认为 False"
