"""测试索引配置模块

测试配置键名映射与兼容性
"""
import tempfile
from pathlib import Path

import pytest
import yaml

from indexing.cli import load_config
from indexing.config import (
    extract_search_options,
    normalize_indexing_config,
)
from indexing.embedding import DEFAULT_MODEL
from indexing.vector_store import DEFAULT_PERSIST_DIR


class TestNormalizeIndexingConfig:
    """测试配置标准化函数"""

    def test_empty_config(self):
        """测试空配置"""
        result = normalize_indexing_config({})
        assert result == {}

    def test_none_config(self):
        """测试 None 配置"""
        result = normalize_indexing_config(None)
        assert result == {}

    def test_model_to_embedding_model(self):
        """测试 model → embedding_model 映射"""
        raw = {"model": "all-MiniLM-L6-v2"}
        result = normalize_indexing_config(raw)

        assert "embedding_model" in result
        assert result["embedding_model"] == "all-MiniLM-L6-v2"
        assert "model" not in result

    def test_persist_path_to_persist_dir(self):
        """测试 persist_path → persist_dir 映射"""
        raw = {"persist_path": ".cursor/vector_index/"}
        result = normalize_indexing_config(raw)

        assert "persist_dir" in result
        assert result["persist_dir"] == ".cursor/vector_index/"
        assert "persist_path" not in result

    def test_fallback_embedding_model(self):
        """测试使用旧键名 embedding_model 时不会被覆盖"""
        raw = {"embedding_model": "paraphrase-MiniLM-L6-v2"}
        result = normalize_indexing_config(raw)

        assert result["embedding_model"] == "paraphrase-MiniLM-L6-v2"

    def test_fallback_persist_dir(self):
        """测试使用旧键名 persist_dir 时不会被覆盖"""
        raw = {"persist_dir": ".cursor/old_index/"}
        result = normalize_indexing_config(raw)

        assert result["persist_dir"] == ".cursor/old_index/"

    def test_model_priority_over_embedding_model(self):
        """测试新键名 model 优先于旧键名 embedding_model"""
        raw = {
            "model": "new-model",
            "embedding_model": "old-model",
        }
        result = normalize_indexing_config(raw)

        # model 应该覆盖 embedding_model
        assert result["embedding_model"] == "new-model"

    def test_persist_path_priority_over_persist_dir(self):
        """测试新键名 persist_path 优先于旧键名 persist_dir"""
        raw = {
            "persist_path": "/new/path/",
            "persist_dir": "/old/path/",
        }
        result = normalize_indexing_config(raw)

        # persist_path 应该覆盖 persist_dir
        assert result["persist_dir"] == "/new/path/"

    def test_other_fields_preserved(self):
        """测试其他字段保持不变"""
        raw = {
            "model": "test-model",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "enabled": True,
        }
        result = normalize_indexing_config(raw)

        assert result["chunk_size"] == 500
        assert result["chunk_overlap"] == 50
        assert result["enabled"] is True


class TestExtractSearchOptions:
    """测试搜索选项提取函数"""

    def test_empty_config(self):
        """测试空配置"""
        result = extract_search_options({})
        assert result == {}

    def test_no_search_section(self):
        """测试无 search 字段"""
        raw = {"model": "test-model"}
        result = extract_search_options(raw)
        assert result == {}

    def test_extract_all_search_options(self):
        """测试提取所有搜索选项"""
        raw = {
            "search": {
                "top_k": 20,
                "min_score": 0.5,
                "include_context": False,
                "context_lines": 5,
            }
        }
        result = extract_search_options(raw)

        assert result["top_k"] == 20
        assert result["min_score"] == 0.5
        assert result["include_context"] is False
        assert result["context_lines"] == 5

    def test_default_search_options(self):
        """测试搜索选项默认值"""
        raw = {"search": {}}
        result = extract_search_options(raw)

        assert result["top_k"] == 10
        assert result["min_score"] == 0.3
        assert result["include_context"] is True
        assert result["context_lines"] == 3

    def test_partial_search_options(self):
        """测试部分搜索选项"""
        raw = {
            "search": {
                "top_k": 15,
            }
        }
        result = extract_search_options(raw)

        assert result["top_k"] == 15
        # 其他字段使用默认值
        assert result["min_score"] == 0.3


class TestLoadConfigWithMapping:
    """测试 load_config 函数的键名映射"""

    def test_load_config_with_new_keys(self, tmp_path):
        """测试使用新键名加载配置"""
        config_content = """
indexing:
  model: all-MiniLM-L6-v2
  persist_path: .cursor/vector_index/
  chunk_size: 500
  chunk_overlap: 50
  include_patterns:
    - "**/*.py"
    - "**/*.js"
  exclude_patterns:
    - "**/node_modules/**"
    - "**/.git/**"
  search:
    top_k: 10
    min_score: 0.3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        # 验证模型映射
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        # 验证路径映射
        assert config.vector_store.persist_directory == ".cursor/vector_index/"
        # 验证其他字段
        assert config.chunking.chunk_size == 500
        assert config.chunking.chunk_overlap == 50
        assert "**/*.py" in config.include_patterns
        assert "**/node_modules/**" in config.exclude_patterns

    def test_load_config_with_old_keys(self, tmp_path):
        """测试使用旧键名加载配置（向后兼容）"""
        config_content = """
indexing:
  embedding_model: paraphrase-MiniLM-L6-v2
  persist_dir: .cursor/old_index/
  chunk_size: 1000
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        # 旧键名应该也能正常工作
        assert config.embedding.model_name == "paraphrase-MiniLM-L6-v2"
        assert config.vector_store.persist_directory == ".cursor/old_index/"
        assert config.chunking.chunk_size == 1000

    def test_load_config_mixed_keys_new_priority(self, tmp_path):
        """测试新旧键名混用时新键名优先"""
        config_content = """
indexing:
  model: new-model
  embedding_model: old-model
  persist_path: /new/path/
  persist_dir: /old/path/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        # 新键名优先
        assert config.embedding.model_name == "new-model"
        assert config.vector_store.persist_directory == "/new/path/"

    def test_load_config_nonexistent_file(self):
        """测试配置文件不存在时使用 core.config 默认值"""
        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config("/nonexistent/path/config.yaml")

        # 应该使用 core.config 的默认值（不再是 indexing 模块的硬编码默认值）
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path

    def test_load_config_none(self):
        """测试 config_file 为 None 时的行为

        当 config_file 为 None 时，load_config 会自动发现项目 config.yaml。
        如果项目 config.yaml 存在，则使用其中的值；否则使用默认值。
        因此这里只验证返回的配置是有效的。
        """
        config = load_config(None)

        # 验证配置是有效的
        assert config is not None
        assert config.embedding is not None
        assert config.vector_store is not None
        # embedding.model_name 应该有值
        assert config.embedding.model_name is not None
        assert len(config.embedding.model_name) > 0

    def test_load_config_empty_indexing_section(self, tmp_path):
        """测试 indexing 配置段为空时使用默认值"""
        config_content = """
indexing:
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        # 应该使用默认值
        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_with_search_options(self, tmp_path):
        """测试加载包含搜索选项的配置"""
        config_content = """
indexing:
  model: all-MiniLM-L6-v2
  persist_path: .cursor/vector_index/
  search:
    top_k: 20
    min_score: 0.5
    include_context: true
    context_lines: 5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # load_config 应该正常工作，不抛出异常
        config = load_config(str(config_file))

        # 主要验证配置加载成功
        assert config.embedding.model_name == "all-MiniLM-L6-v2"

    def test_load_config_invalid_yaml(self, tmp_path):
        """测试无效 YAML 文件时回退到默认配置"""
        config_content = """
indexing:
  model: [invalid yaml
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # 应该回退到默认配置，不抛出异常
        config = load_config(str(config_file))

        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_full_example(self, tmp_path):
        """测试完整的配置文件示例（与项目 config.yaml 结构一致）"""
        config_content = """
indexing:
  enabled: true
  model: all-MiniLM-L6-v2
  persist_path: .cursor/vector_index/
  chunk_size: 500
  chunk_overlap: 50
  include_patterns:
    - "**/*.py"
    - "**/*.js"
    - "**/*.ts"
    - "**/*.go"
    - "**/*.rs"
  exclude_patterns:
    - "**/node_modules/**"
    - "**/.git/**"
    - "**/venv/**"
    - "**/__pycache__/**"
    - "**/dist/**"
    - "**/build/**"
  search:
    top_k: 10
    min_score: 0.3
    include_context: true
    context_lines: 3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        # 验证所有字段
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.vector_store.persist_directory == ".cursor/vector_index/"
        assert config.chunking.chunk_size == 500
        assert config.chunking.chunk_overlap == 50
        assert len(config.include_patterns) == 5
        assert len(config.exclude_patterns) == 6
        assert "**/*.py" in config.include_patterns
        assert "**/node_modules/**" in config.exclude_patterns


class TestIndexingCliDefaultConfig:
    """测试 indexing/cli.py 在未传 --config 时的行为

    验证：
    1. 未传 --config 时尝试读取项目 config.yaml
    2. 项目 config.yaml 不存在时使用默认配置
    3. 命令行参数的默认值正确设置
    """

    def test_load_config_with_none_path(self):
        """测试 load_config(None) 返回有效配置

        注意：当项目根目录存在 config.yaml 时，load_config 会自动发现并使用它。
        因此这里只验证返回的配置是有效的，而不是严格匹配 DEFAULT_* 常量。
        """
        config = load_config(None)

        # 应该返回有效配置
        assert config is not None
        # embedding.model_name 应该有值
        assert config.embedding.model_name is not None
        assert len(config.embedding.model_name) > 0
        # vector_store.persist_directory 应该有值
        assert config.vector_store.persist_directory is not None
        assert len(config.vector_store.persist_directory) > 0

    def test_load_config_with_empty_string(self):
        """测试 load_config('') 使用默认配置"""
        config = load_config("")

        assert config is not None
        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_nonexistent_path(self):
        """测试不存在的配置文件路径使用 core.config 默认配置"""
        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config("/nonexistent/path/config.yaml")

        assert config is not None
        # 使用 core.config 的默认值（不再是 indexing 模块的硬编码默认值）
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path

    def test_create_parser_default_config_is_none(self):
        """测试 create_parser() 的 --config 默认值为 None"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([])

        # --config 参数默认为 None
        assert args.config is None

    def test_create_parser_accepts_config_argument(self):
        """测试 create_parser() 接受 --config 参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml"])

        assert args.config == "/path/to/config.yaml"

    def test_load_config_from_cwd_config_yaml(self, tmp_path):
        """测试 load_config 能从工作目录的 config.yaml 读取

        注意：load_config 仅在显式传入路径时加载，不会自动搜索 cwd
        """
        config_content = """
indexing:
  model: cwd-test-model
  chunk_size: 999
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # 显式传入路径时应该加载
        config = load_config(str(config_file))

        assert config.embedding.model_name == "cwd-test-model"
        assert config.chunking.chunk_size == 999


class TestIndexingCliCommandParsing:
    """测试 indexing/cli.py 命令行解析

    验证各子命令的参数默认值
    """

    def test_build_command_default_args(self):
        """测试 build 命令的默认参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["build"])

        assert args.command == "build"
        assert args.full is False  # 默认增量构建
        assert args.config is None

    def test_build_command_with_config(self):
        """测试 build 命令使用 --config 参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--config", "custom.yaml", "build"])

        assert args.command == "build"
        assert args.config == "custom.yaml"

    def test_search_command_default_args(self):
        """测试 search 命令的默认参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["search", "测试查询"])

        assert args.command == "search"
        assert args.query == "测试查询"
        assert args.top_k == 10  # 默认值
        assert args.context is False

    def test_status_command_default_args(self):
        """测试 status 命令的默认参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["status"])

        assert args.command == "status"
        assert args.config is None

    def test_info_command_default_args(self):
        """测试 info 命令的默认参数"""
        from indexing.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["info"])

        assert args.command == "info"
        assert args.config is None


class TestIndexingCliConfigIntegration:
    """测试 indexing/cli.py 配置集成

    验证配置从 config.yaml 正确加载并应用到各命令
    """

    def test_load_config_applies_to_embedding_config(self, tmp_path):
        """测试配置正确应用到 EmbeddingConfig"""
        config_content = """
indexing:
  model: custom-embedding-model
  device: cuda
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config.embedding.model_name == "custom-embedding-model"
        assert config.embedding.device == "cuda"

    def test_load_config_applies_to_vector_store_config(self, tmp_path):
        """测试配置正确应用到 VectorStoreConfig"""
        config_content = """
indexing:
  persist_path: .custom/index/
  collection_name: custom_collection
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config.vector_store.persist_directory == ".custom/index/"
        assert config.vector_store.collection_name == "custom_collection"

    def test_load_config_applies_to_chunk_config(self, tmp_path):
        """测试配置正确应用到 ChunkConfig"""
        config_content = """
indexing:
  chunk_size: 2000
  chunk_overlap: 400
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config.chunking.chunk_size == 2000
        assert config.chunking.chunk_overlap == 400

    def test_load_config_applies_patterns(self, tmp_path):
        """测试配置正确应用 include/exclude patterns"""
        config_content = """
indexing:
  include_patterns:
    - "**/*.rs"
    - "**/*.go"
  exclude_patterns:
    - "**/target/**"
    - "**/vendor/**"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert "**/*.rs" in config.include_patterns
        assert "**/*.go" in config.include_patterns
        assert "**/target/**" in config.exclude_patterns
        assert "**/vendor/**" in config.exclude_patterns

    def test_load_config_max_workers(self, tmp_path):
        """测试配置正确应用 max_workers"""
        config_content = """
indexing:
  max_workers: 16
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config.max_workers == 16


class TestIndexingCliProjectConfigFallback:
    """测试 indexing CLI 读取项目 config.yaml 的回退行为

    验证：
    1. 未指定 --config 时的默认行为
    2. 配置文件不存在时的兜底策略
    """

    def test_default_config_file_constant(self):
        """测试默认配置文件常量"""
        from indexing.cli import DEFAULT_CONFIG_FILE

        assert DEFAULT_CONFIG_FILE == "config.yaml"

    def test_load_config_handles_yaml_errors_gracefully(self, tmp_path):
        """测试配置文件解析错误时优雅降级"""
        config_content = """
indexing:
  model: [invalid yaml syntax
  chunk_size: not-closed
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # 应该不抛出异常，返回默认配置
        config = load_config(str(config_file))

        assert config is not None
        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_handles_empty_indexing_section(self, tmp_path):
        """测试 indexing 配置段为空时使用默认值"""
        config_content = """
indexing:
# 空配置段
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config is not None
        assert config.embedding.model_name == DEFAULT_MODEL

    def test_load_config_handles_no_indexing_section(self, tmp_path):
        """测试无 indexing 配置段时使用默认值"""
        config_content = """
system:
  max_iterations: 10
models:
  planner: test-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))

        assert config is not None
        assert config.embedding.model_name == DEFAULT_MODEL


class TestLoadConfigAutoDiscovery:
    """测试 load_config() 自动发现配置文件功能

    验证：
    1. 不传 --config 且 cwd 存在 config.yaml 时，使用其中的 indexing.*
    2. 不传 --config 且项目根存在 config.yaml 时，使用其中的 indexing.*
    3. 不存在配置文件时回退默认值
    4. 显式传入 --config 时优先使用指定路径
    """

    def test_load_config_auto_discover_from_cwd(self, tmp_path, monkeypatch):
        """测试 load_config(None) 自动从 cwd 发现 config.yaml"""
        # 创建 cwd 中的 config.yaml
        config_content = """
indexing:
  model: auto-discover-model
  persist_path: .auto/index/
  chunk_size: 888
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # monkeypatch Path.cwd() 返回 tmp_path
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        # 不传入 config_file，应自动发现
        config = load_config(None)

        assert config.embedding.model_name == "auto-discover-model"
        assert config.vector_store.persist_directory == ".auto/index/"
        assert config.chunking.chunk_size == 888

    def test_load_config_auto_discover_from_project_root(self, tmp_path, monkeypatch):
        """测试 load_config(None) 自动从项目根目录发现 config.yaml"""
        # 创建项目根目录结构
        project_root = tmp_path / "project"
        project_root.mkdir()

        # 创建 .git 目录标记项目根
        (project_root / ".git").mkdir()

        # 在项目根创建 config.yaml
        config_content = """
indexing:
  model: project-root-model
  chunk_size: 777
"""
        (project_root / "config.yaml").write_text(config_content, encoding="utf-8")

        # 创建子目录作为 cwd（不包含 config.yaml）
        sub_dir = project_root / "src" / "submodule"
        sub_dir.mkdir(parents=True)

        # monkeypatch Path.cwd() 返回子目录
        monkeypatch.setattr(Path, "cwd", lambda: sub_dir)

        # 不传入 config_file，应从项目根发现
        config = load_config(None)

        assert config.embedding.model_name == "project-root-model"
        assert config.chunking.chunk_size == 777

    def test_load_config_no_config_file_uses_defaults(self, tmp_path, monkeypatch):
        """测试不存在配置文件时使用 core.config 默认值"""
        from core.config import IndexingConfig

        # 创建空目录（无 config.yaml）
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # monkeypatch Path.cwd() 返回空目录
        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # 同时 patch find_config_file 返回 None，避免受项目根 config.yaml 影响
        from unittest.mock import patch

        core_defaults = IndexingConfig()

        with patch("core.config.find_config_file", return_value=None):
            config = load_config(None)

        # 应使用 core.config 的默认值（不再是 indexing 模块的硬编码默认值）
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path

    def test_load_config_explicit_path_takes_priority(self, tmp_path, monkeypatch):
        """测试显式传入 --config 路径优先于自动发现"""
        # 在 cwd 创建一个 config.yaml
        cwd_config_content = """
indexing:
  model: cwd-model
"""
        cwd_config = tmp_path / "config.yaml"
        cwd_config.write_text(cwd_config_content, encoding="utf-8")

        # 创建另一个目录的配置文件
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        explicit_config_content = """
indexing:
  model: explicit-model
"""
        explicit_config = explicit_dir / "custom.yaml"
        explicit_config.write_text(explicit_config_content, encoding="utf-8")

        # monkeypatch Path.cwd() 返回 tmp_path
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        # 显式传入路径，应优先使用
        config = load_config(str(explicit_config))

        assert config.embedding.model_name == "explicit-model"

    def test_load_config_auto_discover_reads_only_indexing_section(self, tmp_path, monkeypatch):
        """测试自动发现的配置文件仅解析 indexing 段"""
        config_content = """
system:
  max_iterations: 100
  worker_pool_size: 10
models:
  planner: some-planner
  worker: some-worker
indexing:
  model: indexing-only-model
  chunk_size: 666
cloud_agent:
  timeout: 999
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(None)

        # 只解析 indexing 段
        assert config.embedding.model_name == "indexing-only-model"
        assert config.chunking.chunk_size == 666
        # 不应该受其他配置段影响（IndexConfig 没有这些属性）

    def test_load_config_auto_discover_with_old_key_names(self, tmp_path, monkeypatch):
        """测试自动发现配置时旧键名兼容性"""
        config_content = """
indexing:
  embedding_model: old-style-model
  persist_dir: .old/index/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(None)

        # 旧键名应该正常工作
        assert config.embedding.model_name == "old-style-model"
        assert config.vector_store.persist_directory == ".old/index/"

    def test_load_config_auto_discover_new_key_priority(self, tmp_path, monkeypatch):
        """测试自动发现配置时新键名优先于旧键名"""
        config_content = """
indexing:
  model: new-key-model
  embedding_model: old-key-model
  persist_path: /new/path/
  persist_dir: /old/path/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(None)

        # 新键名优先
        assert config.embedding.model_name == "new-key-model"
        assert config.vector_store.persist_directory == "/new/path/"


class TestLoadConfigFallbackToCoreConfig:
    """测试 load_config 未配置字段回退到 core.config 的行为

    验证：
    1. 当 indexing 配置中某些字段未配置时，回退值来自 core.config.IndexingConfig
    2. 这确保了默认值与 config.yaml 中定义的语义一致
    3. indexing 模块不再使用独立的硬编码默认值（除非 core.config 也未提供）
    """

    def test_partial_config_falls_back_to_core_config(self, tmp_path, monkeypatch):
        """测试部分配置时未配置字段回退到 core.config 默认值"""
        # 仅配置 model，其他字段未配置
        config_content = """
indexing:
  model: custom-partial-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        # 获取 core.config 的默认值用于验证
        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # model 使用配置文件中的值
        assert config.embedding.model_name == "custom-partial-model"

        # 未配置字段应回退到 core.config 的默认值
        assert config.vector_store.persist_directory == core_defaults.persist_path
        assert config.chunking.chunk_size == core_defaults.chunk_size
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap
        assert config.include_patterns == core_defaults.include_patterns
        assert config.exclude_patterns == core_defaults.exclude_patterns

    def test_empty_indexing_section_uses_core_config_defaults(self, tmp_path, monkeypatch):
        """测试 indexing 配置段为空时使用 core.config 默认值"""
        config_content = """
indexing:
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # 所有字段应使用 core.config 的默认值
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path
        assert config.chunking.chunk_size == core_defaults.chunk_size
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap

    def test_no_config_file_uses_core_config_defaults(self, tmp_path, monkeypatch):
        """测试配置文件不存在时使用 core.config 默认值"""
        # 创建空目录
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        from core.config import IndexingConfig
        from unittest.mock import patch

        core_defaults = IndexingConfig()

        # patch find_config_file 返回 None，避免受项目根 config.yaml 影响
        with patch("core.config.find_config_file", return_value=None):
            config = load_config(None)

        # 所有字段应使用 core.config 的默认值
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path
        assert config.chunking.chunk_size == core_defaults.chunk_size
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap

    def test_chunk_size_fallback_matches_core_config_not_hardcoded(self, tmp_path, monkeypatch):
        """测试 chunk_size 回退值来自 core.config 而非硬编码值

        历史问题：indexing/cli.py 曾硬编码 chunk_size=1500，chunk_overlap=200
        而 core/config.py IndexingConfig 定义的默认值是 chunk_size=500，chunk_overlap=50
        此测试验证回退值现在来自 core.config，确保一致性。
        """
        # 仅配置 model，chunk_size 和 chunk_overlap 未配置
        config_content = """
indexing:
  model: test-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # 验证回退值来自 core.config 而非 indexing 模块硬编码值
        # core.config.IndexingConfig 定义: chunk_size=500, chunk_overlap=50
        # 旧的硬编码值: chunk_size=1500, chunk_overlap=200
        assert config.chunking.chunk_size == core_defaults.chunk_size
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap

        # 显式验证不是旧的硬编码值（如果 core_defaults 恰好等于旧值则跳过）
        if core_defaults.chunk_size != 1500:
            assert config.chunking.chunk_size != 1500, \
                "chunk_size 不应该是旧的硬编码值 1500"
        if core_defaults.chunk_overlap != 200:
            assert config.chunking.chunk_overlap != 200, \
                "chunk_overlap 不应该是旧的硬编码值 200"

    def test_include_exclude_patterns_fallback_to_core_config(self, tmp_path, monkeypatch):
        """测试 include/exclude patterns 回退到 core.config 默认值"""
        # 仅配置 model，patterns 未配置
        config_content = """
indexing:
  model: patterns-test-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # include_patterns 和 exclude_patterns 应使用 core.config 的默认值
        assert config.include_patterns == core_defaults.include_patterns
        assert config.exclude_patterns == core_defaults.exclude_patterns

    def test_persist_path_fallback_to_core_config(self, tmp_path, monkeypatch):
        """测试 persist_path 回退到 core.config 默认值"""
        # 仅配置 model 和 chunk_size，persist_path 未配置
        config_content = """
indexing:
  model: persist-test-model
  chunk_size: 800
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # persist_directory 应回退到 core.config 的 persist_path
        assert config.vector_store.persist_directory == core_defaults.persist_path

    def test_explicit_config_overrides_core_config_defaults(self, tmp_path, monkeypatch):
        """测试显式配置覆盖 core.config 默认值"""
        config_content = """
indexing:
  model: explicit-model
  persist_path: /explicit/path/
  chunk_size: 999
  chunk_overlap: 111
  include_patterns:
    - "**/*.rs"
  exclude_patterns:
    - "**/target/**"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # 显式配置的值应覆盖 core.config 默认值
        assert config.embedding.model_name == "explicit-model"
        assert config.vector_store.persist_directory == "/explicit/path/"
        assert config.chunking.chunk_size == 999
        assert config.chunking.chunk_overlap == 111
        assert config.include_patterns == ["**/*.rs"]
        assert config.exclude_patterns == ["**/target/**"]

        # 验证这些值与 core.config 默认值不同（除非配置文件值恰好相同）
        if core_defaults.chunk_size != 999:
            assert config.chunking.chunk_size != core_defaults.chunk_size


class TestLoadConfigNoIndexingSectionFallback:
    """测试 load_config(None) 无 indexing 子键时回退到 core.config.indexing 的值

    验证：
    1. config.yaml 存在但无 indexing 段时，使用 core.config.IndexingConfig 默认值
    2. config.yaml 的 indexing 段为空时，使用 core.config.IndexingConfig 默认值
    3. 回退值与 core.config.IndexingConfig 一致，而非 indexing 模块硬编码值
    """

    def test_load_config_no_indexing_section_uses_core_config_defaults(
        self, tmp_path, monkeypatch
    ):
        """测试 config.yaml 无 indexing 段时使用 core.config 默认值"""
        # 创建 config.yaml，但不包含 indexing 段
        config_content = """
system:
  max_iterations: 20
  worker_pool_size: 5
models:
  planner: custom-planner
  worker: custom-worker
cloud_agent:
  timeout: 600
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # 所有 indexing 配置应回退到 core.config 的默认值
        assert config.embedding.model_name == core_defaults.model, (
            f"model 应回退到 core.config 默认值 {core_defaults.model}，"
            f"实际值: {config.embedding.model_name}"
        )
        assert config.vector_store.persist_directory == core_defaults.persist_path, (
            f"persist_path 应回退到 core.config 默认值 {core_defaults.persist_path}，"
            f"实际值: {config.vector_store.persist_directory}"
        )
        assert config.chunking.chunk_size == core_defaults.chunk_size, (
            f"chunk_size 应回退到 core.config 默认值 {core_defaults.chunk_size}，"
            f"实际值: {config.chunking.chunk_size}"
        )
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap, (
            f"chunk_overlap 应回退到 core.config 默认值 {core_defaults.chunk_overlap}，"
            f"实际值: {config.chunking.chunk_overlap}"
        )
        assert config.include_patterns == core_defaults.include_patterns, (
            f"include_patterns 应回退到 core.config 默认值"
        )
        assert config.exclude_patterns == core_defaults.exclude_patterns, (
            f"exclude_patterns 应回退到 core.config 默认值"
        )

    def test_load_config_null_indexing_uses_core_config_defaults(
        self, tmp_path, monkeypatch
    ):
        """测试 config.yaml 的 indexing 段为 null 时使用 core.config 默认值"""
        config_content = """
system:
  max_iterations: 10
indexing:
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # indexing 段为空/null 时应使用 core.config 默认值
        assert config.embedding.model_name == core_defaults.model
        assert config.vector_store.persist_directory == core_defaults.persist_path
        assert config.chunking.chunk_size == core_defaults.chunk_size

    def test_load_config_none_auto_discover_no_indexing_uses_defaults(
        self, tmp_path, monkeypatch
    ):
        """测试 load_config(None) 自动发现时，无 indexing 段使用 core.config 默认值"""
        # 创建 config.yaml，不包含 indexing 段
        config_content = """
models:
  planner: test-planner
  worker: test-worker
  reviewer: test-reviewer
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        # 使用 load_config(None) 触发自动发现
        config = load_config(None)

        # 验证回退到 core.config 默认值
        assert config.embedding.model_name == core_defaults.model, (
            f"自动发现时 model 应回退到 core.config 默认值"
        )
        assert config.vector_store.persist_directory == core_defaults.persist_path, (
            f"自动发现时 persist_path 应回退到 core.config 默认值"
        )
        assert config.chunking.chunk_size == core_defaults.chunk_size, (
            f"自动发现时 chunk_size 应回退到 core.config 默认值"
        )

    def test_load_config_fallback_values_match_core_config_not_hardcoded(
        self, tmp_path, monkeypatch
    ):
        """测试回退值来自 core.config.IndexingConfig 而非 indexing 模块硬编码值

        历史问题：indexing 模块曾有独立的硬编码默认值，可能与 core.config 不一致。
        此测试验证回退值确实来自 core.config。
        """
        # 创建 config.yaml，不包含 indexing 段
        config_content = """
system:
  max_iterations: 10
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # 验证回退值与 core.config.IndexingConfig 精确匹配
        assert config.embedding.model_name == core_defaults.model, (
            f"model 回退值应与 core.config.IndexingConfig.model 一致: "
            f"expected={core_defaults.model}, actual={config.embedding.model_name}"
        )
        assert config.vector_store.persist_directory == core_defaults.persist_path, (
            f"persist_path 回退值应与 core.config.IndexingConfig.persist_path 一致: "
            f"expected={core_defaults.persist_path}, actual={config.vector_store.persist_directory}"
        )
        assert config.chunking.chunk_size == core_defaults.chunk_size, (
            f"chunk_size 回退值应与 core.config.IndexingConfig.chunk_size 一致: "
            f"expected={core_defaults.chunk_size}, actual={config.chunking.chunk_size}"
        )
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap, (
            f"chunk_overlap 回退值应与 core.config.IndexingConfig.chunk_overlap 一致: "
            f"expected={core_defaults.chunk_overlap}, actual={config.chunking.chunk_overlap}"
        )

    def test_load_config_partial_indexing_section_fallback(
        self, tmp_path, monkeypatch
    ):
        """测试 indexing 段部分配置时，未配置字段回退到 core.config"""
        # 创建 config.yaml，只设置部分 indexing 字段
        config_content = """
indexing:
  model: custom-model-only
  # 其他字段未配置，应回退到 core.config 默认值
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        from core.config import IndexingConfig
        core_defaults = IndexingConfig()

        config = load_config(str(config_file))

        # model 使用配置文件中的值
        assert config.embedding.model_name == "custom-model-only"

        # 其他字段回退到 core.config 默认值
        assert config.vector_store.persist_directory == core_defaults.persist_path
        assert config.chunking.chunk_size == core_defaults.chunk_size
        assert config.chunking.chunk_overlap == core_defaults.chunk_overlap
        assert config.include_patterns == core_defaults.include_patterns
        assert config.exclude_patterns == core_defaults.exclude_patterns


class TestLoadConfigExplicitPathPriority:
    """测试 load_config 显式路径优先策略

    验证：
    1. 显式传入 config_file 路径时，优先使用该路径的配置
    2. 即使 cwd 存在 config.yaml，显式路径也优先
    3. 即使项目根目录存在 config.yaml，显式路径也优先
    """

    def test_explicit_path_priority_over_cwd(self, tmp_path, monkeypatch):
        """测试显式路径优先于 cwd 的 config.yaml"""
        # 在 cwd 创建 config.yaml
        cwd_config_content = """
indexing:
  model: cwd-model
  chunk_size: 111
"""
        (tmp_path / "config.yaml").write_text(cwd_config_content, encoding="utf-8")

        # 在另一个目录创建显式指定的配置文件
        explicit_dir = tmp_path / "explicit_config_dir"
        explicit_dir.mkdir()
        explicit_config_content = """
indexing:
  model: explicit-path-model
  chunk_size: 999
"""
        explicit_config = explicit_dir / "custom_config.yaml"
        explicit_config.write_text(explicit_config_content, encoding="utf-8")

        # monkeypatch cwd 为 tmp_path（包含 config.yaml）
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        # 显式传入路径，应该使用 explicit 配置，而非 cwd 配置
        config = load_config(str(explicit_config))

        assert config.embedding.model_name == "explicit-path-model"
        assert config.chunking.chunk_size == 999

    def test_explicit_path_priority_over_project_root(self, tmp_path, monkeypatch):
        """测试显式路径优先于项目根目录的 config.yaml"""
        # 创建项目根目录结构
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        # 在项目根创建 config.yaml
        root_config_content = """
indexing:
  model: project-root-model
  chunk_size: 222
"""
        (project_root / "config.yaml").write_text(root_config_content, encoding="utf-8")

        # 创建子目录作为 cwd
        sub_dir = project_root / "src" / "module"
        sub_dir.mkdir(parents=True)

        # 在完全不同的位置创建显式配置
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        explicit_config_content = """
indexing:
  model: explicit-override-model
  chunk_size: 888
"""
        explicit_config = explicit_dir / "override.yaml"
        explicit_config.write_text(explicit_config_content, encoding="utf-8")

        # monkeypatch cwd 为子目录
        monkeypatch.setattr(Path, "cwd", lambda: sub_dir)

        # 显式路径应该优先于项目根配置
        config = load_config(str(explicit_config))

        assert config.embedding.model_name == "explicit-override-model"
        assert config.chunking.chunk_size == 888

    def test_explicit_path_with_nonexistent_auto_discover(self, tmp_path, monkeypatch):
        """测试显式路径在自动发现失败时仍然工作"""
        # 创建空目录（无 config.yaml）
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # 创建显式配置
        explicit_config_content = """
indexing:
  model: explicit-only-model
  persist_path: /explicit/path/
"""
        explicit_config = tmp_path / "explicit.yaml"
        explicit_config.write_text(explicit_config_content, encoding="utf-8")

        # monkeypatch cwd 为空目录
        monkeypatch.setattr(Path, "cwd", lambda: empty_dir)

        # 显式路径应该正常工作
        config = load_config(str(explicit_config))

        assert config.embedding.model_name == "explicit-only-model"
        assert config.vector_store.persist_directory == "/explicit/path/"


class TestLoadConfigCwdPriorityOverParent:
    """测试 cwd 自动发现优先于父目录

    验证：
    1. 当 cwd 存在 config.yaml 时，使用 cwd 的配置
    2. 当 cwd 不存在但父目录存在时，使用父目录的配置
    3. cwd 的配置优先于项目根（通过 .git 识别）的配置
    """

    def test_cwd_config_priority_over_parent(self, tmp_path, monkeypatch):
        """测试 cwd 的 config.yaml 优先于父目录"""
        # 创建父目录配置
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        parent_config_content = """
indexing:
  model: parent-model
  chunk_size: 100
"""
        (parent_dir / "config.yaml").write_text(parent_config_content, encoding="utf-8")

        # 创建子目录并在其中放置 config.yaml
        child_dir = parent_dir / "child"
        child_dir.mkdir()
        child_config_content = """
indexing:
  model: child-cwd-model
  chunk_size: 200
"""
        (child_dir / "config.yaml").write_text(child_config_content, encoding="utf-8")

        # monkeypatch cwd 为子目录
        monkeypatch.setattr(Path, "cwd", lambda: child_dir)

        # 应该使用子目录（cwd）的配置
        config = load_config(None)

        assert config.embedding.model_name == "child-cwd-model"
        assert config.chunking.chunk_size == 200

    def test_cwd_priority_over_project_root(self, tmp_path, monkeypatch):
        """测试 cwd 配置优先于项目根（.git）的配置"""
        # 创建项目根目录
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        root_config_content = """
indexing:
  model: project-root-model
  chunk_size: 300
"""
        (project_root / "config.yaml").write_text(root_config_content, encoding="utf-8")

        # 创建子目录并在其中放置 config.yaml
        sub_dir = project_root / "src"
        sub_dir.mkdir()
        sub_config_content = """
indexing:
  model: cwd-priority-model
  chunk_size: 400
"""
        (sub_dir / "config.yaml").write_text(sub_config_content, encoding="utf-8")

        # monkeypatch cwd 为子目录
        monkeypatch.setattr(Path, "cwd", lambda: sub_dir)

        # 应该使用子目录（cwd）的配置，而非项目根
        config = load_config(None)

        assert config.embedding.model_name == "cwd-priority-model"
        assert config.chunking.chunk_size == 400

    def test_fallback_to_parent_when_cwd_no_config(self, tmp_path, monkeypatch):
        """测试 cwd 无 config.yaml 时回退到父目录"""
        # 创建父目录配置
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        (parent_dir / ".git").mkdir()  # 标记为项目根
        parent_config_content = """
indexing:
  model: parent-fallback-model
  chunk_size: 500
"""
        (parent_dir / "config.yaml").write_text(parent_config_content, encoding="utf-8")

        # 创建子目录（不包含 config.yaml）
        child_dir = parent_dir / "child_no_config"
        child_dir.mkdir()

        # monkeypatch cwd 为子目录
        monkeypatch.setattr(Path, "cwd", lambda: child_dir)

        # 应该回退到父目录（项目根）的配置
        config = load_config(None)

        assert config.embedding.model_name == "parent-fallback-model"
        assert config.chunking.chunk_size == 500


class TestLoadConfigKeyNameCompatibility:
    """测试新旧键名兼容读取

    验证：
    1. model (新) 与 embedding_model (旧) 兼容
    2. persist_path (新) 与 persist_dir (旧) 兼容
    3. 新键名优先于旧键名
    4. 通过 load_config 读取时键名映射正确
    """

    def test_new_key_model_reads_correctly(self, tmp_path):
        """测试新键名 model 正确读取"""
        config_content = """
indexing:
  model: new-key-test-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        assert config.embedding.model_name == "new-key-test-model"

    def test_old_key_embedding_model_reads_correctly(self, tmp_path):
        """测试旧键名 embedding_model 正确读取（向后兼容）"""
        config_content = """
indexing:
  embedding_model: old-key-compat-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        assert config.embedding.model_name == "old-key-compat-model"

    def test_new_key_persist_path_reads_correctly(self, tmp_path):
        """测试新键名 persist_path 正确读取"""
        config_content = """
indexing:
  persist_path: /new/persist/path/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        assert config.vector_store.persist_directory == "/new/persist/path/"

    def test_old_key_persist_dir_reads_correctly(self, tmp_path):
        """测试旧键名 persist_dir 正确读取（向后兼容）"""
        config_content = """
indexing:
  persist_dir: /old/persist/dir/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        assert config.vector_store.persist_directory == "/old/persist/dir/"

    def test_new_key_priority_model_over_embedding_model(self, tmp_path):
        """测试新键名 model 优先于旧键名 embedding_model"""
        config_content = """
indexing:
  model: priority-new-model
  embedding_model: should-be-ignored-model
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        # 新键名 model 应该优先
        assert config.embedding.model_name == "priority-new-model"

    def test_new_key_priority_persist_path_over_persist_dir(self, tmp_path):
        """测试新键名 persist_path 优先于旧键名 persist_dir"""
        config_content = """
indexing:
  persist_path: /priority/new/path/
  persist_dir: /should/be/ignored/
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        # 新键名 persist_path 应该优先
        assert config.vector_store.persist_directory == "/priority/new/path/"

    def test_mixed_old_new_keys_partial(self, tmp_path):
        """测试部分使用新键名、部分使用旧键名"""
        config_content = """
indexing:
  model: new-style-model
  persist_dir: /old-style/path/
  chunk_size: 600
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        # model 使用新键名
        assert config.embedding.model_name == "new-style-model"
        # persist_dir 使用旧键名（兼容）
        assert config.vector_store.persist_directory == "/old-style/path/"
        # 其他字段正常
        assert config.chunking.chunk_size == 600


class TestSearchOptionsConsistency:
    """测试 indexing.search.* 配置与 load_config 返回值的一致性

    验证：
    1. config.yaml 中的 indexing.search.top_k/min_score/context_lines 正确解析
    2. extract_search_options 与 core.config.IndexingSearchConfig 解析一致
    3. 修改 config.yaml 后，CLI 默认参数/行为反映该值
    """

    def test_search_options_extracted_matches_yaml(self, tmp_path):
        """测试 extract_search_options 正确提取 config.yaml 中的搜索选项"""
        config_content = """
indexing:
  model: test-model
  search:
    top_k: 25
    min_score: 0.6
    include_context: false
    context_lines: 8
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # 直接加载 YAML 并测试 extract_search_options
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
        raw_indexing = data.get("indexing", {})

        search_opts = extract_search_options(raw_indexing)

        assert search_opts["top_k"] == 25
        assert search_opts["min_score"] == 0.6
        assert search_opts["include_context"] is False
        assert search_opts["context_lines"] == 8

    def test_search_options_consistency_with_core_config(self, tmp_path, monkeypatch):
        """测试 search options 与 core.config.IndexingSearchConfig 解析一致"""
        from core.config import ConfigManager, IndexingSearchConfig

        # 保存原始状态
        original_instance = ConfigManager._instance
        original_config = ConfigManager._config
        original_config_path = ConfigManager._config_path

        try:
            config_content = """
indexing:
  model: consistency-test-model
  search:
    top_k: 30
    min_score: 0.45
    include_context: true
    context_lines: 10
"""
            config_file = tmp_path / "config.yaml"
            config_file.write_text(config_content, encoding="utf-8")

            monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

            # 重置 ConfigManager 单例
            ConfigManager.reset_instance()

            # 通过 ConfigManager 读取
            core_config = ConfigManager.get_instance()
            core_search = core_config.indexing.search

            # 通过 extract_search_options 读取
            import yaml
            with open(config_file) as f:
                data = yaml.safe_load(f)
            raw_indexing = data.get("indexing", {})
            extracted_search = extract_search_options(raw_indexing)

            # 两者应该一致
            assert core_search.top_k == extracted_search["top_k"]
            assert core_search.min_score == extracted_search["min_score"]
            assert core_search.include_context == extracted_search["include_context"]
            assert core_search.context_lines == extracted_search["context_lines"]
        finally:
            # 恢复原始状态
            ConfigManager._instance = original_instance
            ConfigManager._config = original_config
            ConfigManager._config_path = original_config_path

    def test_search_options_default_fallback(self, tmp_path):
        """测试 search options 使用默认值"""
        config_content = """
indexing:
  model: default-search-test
  search:
    top_k: 15
    # 其他字段使用默认值
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
        raw_indexing = data.get("indexing", {})

        search_opts = extract_search_options(raw_indexing)

        # top_k 使用配置值
        assert search_opts["top_k"] == 15
        # 其他字段使用 extract_search_options 的默认值
        assert search_opts["min_score"] == 0.3
        assert search_opts["include_context"] is True
        assert search_opts["context_lines"] == 3

    def test_cli_search_command_default_top_k(self, tmp_path, monkeypatch):
        """测试 CLI search 命令的默认 top_k 参数

        验证 CLI 的 --top-k 默认值（硬编码为 10）与 config.yaml 中的设置关系
        """
        from indexing.cli import create_parser

        # 测试 CLI parser 的默认值
        parser = create_parser()
        args = parser.parse_args(["search", "test query"])

        # CLI 默认值是 10（这是 argparse 硬编码的默认值）
        assert args.top_k == 10

        # 测试通过 config.yaml 配置的搜索选项
        config_content = """
indexing:
  model: cli-test-model
  search:
    top_k: 50
    min_score: 0.7
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
        raw_indexing = data.get("indexing", {})

        search_opts = extract_search_options(raw_indexing)

        # config.yaml 中的值
        assert search_opts["top_k"] == 50
        assert search_opts["min_score"] == 0.7

    def test_core_config_indexing_search_defaults(self):
        """测试 core.config.IndexingSearchConfig 的默认值"""
        from core.config import IndexingSearchConfig

        defaults = IndexingSearchConfig()

        # 验证默认值与 extract_search_options 的默认值一致
        assert defaults.top_k == 10
        assert defaults.min_score == 0.3
        assert defaults.include_context is True
        assert defaults.context_lines == 3


class TestIndexingConfigDefaultsAlignment:

    def test_embedding_config_model_name_matches_core_config(self):
        """测试 EmbeddingConfig.model_name 与 core.config.IndexingConfig.model 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import EmbeddingConfig

        core_defaults = CoreIndexingConfig()
        embedding_defaults = EmbeddingConfig()

        assert embedding_defaults.model_name == core_defaults.model, (
            f"EmbeddingConfig.model_name ({embedding_defaults.model_name}) "
            f"应与 IndexingConfig.model ({core_defaults.model}) 一致"
        )

    def test_chunk_config_chunk_size_matches_core_config(self):
        """测试 ChunkConfig.chunk_size 与 core.config.IndexingConfig.chunk_size 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import ChunkConfig

        core_defaults = CoreIndexingConfig()
        chunk_defaults = ChunkConfig()

        assert chunk_defaults.chunk_size == core_defaults.chunk_size, (
            f"ChunkConfig.chunk_size ({chunk_defaults.chunk_size}) "
            f"应与 IndexingConfig.chunk_size ({core_defaults.chunk_size}) 一致"
        )

    def test_chunk_config_chunk_overlap_matches_core_config(self):
        """测试 ChunkConfig.chunk_overlap 与 core.config.IndexingConfig.chunk_overlap 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import ChunkConfig

        core_defaults = CoreIndexingConfig()
        chunk_defaults = ChunkConfig()

        assert chunk_defaults.chunk_overlap == core_defaults.chunk_overlap, (
            f"ChunkConfig.chunk_overlap ({chunk_defaults.chunk_overlap}) "
            f"应与 IndexingConfig.chunk_overlap ({core_defaults.chunk_overlap}) 一致"
        )

    def test_vector_store_config_persist_directory_matches_core_config(self):
        """测试 VectorStoreConfig.persist_directory 与 core.config.IndexingConfig.persist_path 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import VectorStoreConfig

        core_defaults = CoreIndexingConfig()
        vector_defaults = VectorStoreConfig()

        assert vector_defaults.persist_directory == core_defaults.persist_path, (
            f"VectorStoreConfig.persist_directory ({vector_defaults.persist_directory}) "
            f"应与 IndexingConfig.persist_path ({core_defaults.persist_path}) 一致"
        )

    def test_index_config_include_patterns_matches_core_config(self):
        """测试 IndexConfig.include_patterns 与 core.config.IndexingConfig.include_patterns 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import IndexConfig

        core_defaults = CoreIndexingConfig()
        index_defaults = IndexConfig()

        assert index_defaults.include_patterns == core_defaults.include_patterns, (
            f"IndexConfig.include_patterns ({index_defaults.include_patterns}) "
            f"应与 IndexingConfig.include_patterns ({core_defaults.include_patterns}) 一致"
        )

    def test_index_config_exclude_patterns_matches_core_config(self):
        """测试 IndexConfig.exclude_patterns 与 core.config.IndexingConfig.exclude_patterns 一致"""
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import IndexConfig

        core_defaults = CoreIndexingConfig()
        index_defaults = IndexConfig()

        assert index_defaults.exclude_patterns == core_defaults.exclude_patterns, (
            f"IndexConfig.exclude_patterns ({index_defaults.exclude_patterns}) "
            f"应与 IndexingConfig.exclude_patterns ({core_defaults.exclude_patterns}) 一致"
        )

    def test_all_critical_defaults_aligned(self):
        """综合测试：验证所有关键默认值对齐

        这个测试作为安全网，确保如果有人修改了任一配置的默认值，
        测试会失败并提醒需要同时更新另一处。
        """
        from core.config import IndexingConfig as CoreIndexingConfig
        from indexing.config import ChunkConfig, EmbeddingConfig, IndexConfig, VectorStoreConfig

        core_defaults = CoreIndexingConfig()
        embedding_defaults = EmbeddingConfig()
        chunk_defaults = ChunkConfig()
        vector_defaults = VectorStoreConfig()
        index_defaults = IndexConfig()

        # 收集所有不一致的项
        mismatches = []

        if embedding_defaults.model_name != core_defaults.model:
            mismatches.append(
                f"model_name: {embedding_defaults.model_name} != {core_defaults.model}"
            )

        if chunk_defaults.chunk_size != core_defaults.chunk_size:
            mismatches.append(
                f"chunk_size: {chunk_defaults.chunk_size} != {core_defaults.chunk_size}"
            )

        if chunk_defaults.chunk_overlap != core_defaults.chunk_overlap:
            mismatches.append(
                f"chunk_overlap: {chunk_defaults.chunk_overlap} != {core_defaults.chunk_overlap}"
            )

        if vector_defaults.persist_directory != core_defaults.persist_path:
            mismatches.append(
                f"persist_directory: {vector_defaults.persist_directory} != {core_defaults.persist_path}"
            )

        if index_defaults.include_patterns != core_defaults.include_patterns:
            mismatches.append(
                f"include_patterns: {index_defaults.include_patterns} != {core_defaults.include_patterns}"
            )

        if index_defaults.exclude_patterns != core_defaults.exclude_patterns:
            mismatches.append(
                f"exclude_patterns: {index_defaults.exclude_patterns} != {core_defaults.exclude_patterns}"
            )

        assert not mismatches, (
            f"indexing/config.py 与 core/config.py 默认值不一致:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\n请同时更新两处配置以保持一致"
        )
