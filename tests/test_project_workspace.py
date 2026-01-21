"""测试 core/project_workspace.py 目录驱动工程创建/扩展模块"""
import tempfile
from pathlib import Path

import pytest

from core.project_workspace import (
    ProjectState,
    ProjectInfo,
    ReferenceProject,
    ScaffoldResult,
    WorkspacePreparationResult,
    inspect_project_state,
    infer_language,
    scaffold,
    detect_reference_projects,
    prepare_workspace,
    get_language_hint,
    get_supported_languages,
    PROJECT_MARKERS,
    SOURCE_EXTENSIONS,
    LANGUAGE_KEYWORDS,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def force_heuristic_infer(monkeypatch):
    """测试中禁用大模型推断，避免外部依赖"""
    monkeypatch.setenv("LANGUAGE_INFER_MODE", "heuristic")
    monkeypatch.setenv("TASK_ANALYSIS_MODE", "heuristic")
    yield


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def python_project(temp_dir):
    """创建一个 Python 工程目录"""
    (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    return temp_dir


@pytest.fixture
def node_project(temp_dir):
    """创建一个 Node.js 工程目录"""
    (temp_dir / "package.json").write_text('{"name": "test"}\n', encoding="utf-8")
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "index.js").write_text("console.log('hello');\n", encoding="utf-8")
    return temp_dir


@pytest.fixture
def empty_dir(temp_dir):
    """创建一个空目录"""
    return temp_dir


@pytest.fixture
def docs_only_dir(temp_dir):
    """创建一个仅包含文档的目录"""
    (temp_dir / "README.md").write_text("# Test\n", encoding="utf-8")
    (temp_dir / "docs").mkdir()
    (temp_dir / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    return temp_dir


@pytest.fixture
def source_only_dir(temp_dir):
    """创建一个有源码但无工程标记的目录"""
    (temp_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (temp_dir / "utils.py").write_text("def foo(): pass\n", encoding="utf-8")
    return temp_dir


@pytest.fixture
def nested_projects_dir(temp_dir):
    """创建一个包含多个子工程的目录"""
    # 根目录无工程标记
    (temp_dir / "README.md").write_text("# Root\n", encoding="utf-8")

    # 子工程 1: Python
    subdir1 = temp_dir / "backend"
    subdir1.mkdir()
    (subdir1 / "pyproject.toml").write_text('[project]\nname = "backend"\n', encoding="utf-8")
    (subdir1 / "main.py").write_text("print('backend')\n", encoding="utf-8")

    # 子工程 2: Node.js
    subdir2 = temp_dir / "frontend"
    subdir2.mkdir()
    (subdir2 / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")
    (subdir2 / "index.js").write_text("console.log('frontend');\n", encoding="utf-8")

    # 子工程 3: Go
    subdir3 = temp_dir / "service"
    subdir3.mkdir()
    (subdir3 / "go.mod").write_text("module example.com/service\n\ngo 1.21\n", encoding="utf-8")
    (subdir3 / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

    return temp_dir


# ============================================================
# TestInspectProjectState
# ============================================================


class TestInspectProjectState:
    """测试 inspect_project_state 函数"""

    def test_existing_python_project(self, python_project):
        """测试已有 Python 工程"""
        result = inspect_project_state(python_project)

        assert result.state == ProjectState.EXISTING_PROJECT
        assert result.detected_language == "python"
        assert "pyproject.toml" in result.marker_files
        assert result.path == python_project

    def test_existing_node_project(self, node_project):
        """测试已有 Node.js 工程"""
        result = inspect_project_state(node_project)

        assert result.state == ProjectState.EXISTING_PROJECT
        assert result.detected_language == "node"
        assert "package.json" in result.marker_files

    def test_empty_directory(self, empty_dir):
        """测试空目录"""
        result = inspect_project_state(empty_dir)

        assert result.state == ProjectState.EMPTY_OR_DOCS_ONLY
        assert result.detected_language is None
        assert len(result.marker_files) == 0

    def test_docs_only_directory(self, docs_only_dir):
        """测试仅文档目录"""
        result = inspect_project_state(docs_only_dir)

        assert result.state == ProjectState.EMPTY_OR_DOCS_ONLY
        assert result.detected_language is None

    def test_source_only_directory(self, source_only_dir):
        """测试有源码但无工程标记的目录"""
        result = inspect_project_state(source_only_dir)

        assert result.state == ProjectState.HAS_SOURCE_FILES
        assert result.detected_language == "python"
        assert result.source_files_count >= 2

    def test_nonexistent_directory(self, temp_dir):
        """测试不存在的目录"""
        nonexistent = temp_dir / "nonexistent"
        result = inspect_project_state(nonexistent)

        assert result.state == ProjectState.EMPTY_OR_DOCS_ONLY

    def test_file_instead_of_directory(self, temp_dir):
        """测试传入文件而非目录"""
        file_path = temp_dir / "file.txt"
        file_path.write_text("content", encoding="utf-8")

        result = inspect_project_state(file_path)

        assert result.state == ProjectState.EMPTY_OR_DOCS_ONLY


# ============================================================
# TestInferLanguage
# ============================================================


class TestInferLanguage:
    """测试 infer_language 函数"""

    def test_infer_python_english(self):
        """测试推断 Python（英文）"""
        assert infer_language("Create a Python CLI tool") == "python"
        assert infer_language("Build a Flask app") == "python"
        assert infer_language("Implement with Django") == "python"

    def test_infer_python_chinese(self):
        """测试推断 Python（中文）"""
        assert infer_language("用 Python 写一个工具") == "python"
        assert infer_language("Python 项目开发") == "python"

    def test_infer_python_chinese_no_space(self):
        """测试推断 Python（中文连续文本）"""
        assert infer_language("用python写一个工具") == "python"

    def test_infer_node(self):
        """测试推断 Node.js"""
        assert infer_language("Create a Node.js API") == "node"
        assert infer_language("Build with npm") == "node"
        assert infer_language("用 nodejs 写服务器") == "node"

    def test_infer_typescript(self):
        """测试推断 TypeScript"""
        assert infer_language("Write TypeScript code") == "typescript"
        assert infer_language("用 TS 实现") == "typescript"

    def test_infer_go(self):
        """测试推断 Go"""
        assert infer_language("Build a Go service") == "go"
        assert infer_language("用 Golang 写") == "go"

    def test_infer_java(self):
        """测试推断 Java"""
        assert infer_language("Create a Java application") == "java"
        assert infer_language("Spring Boot 项目") == "java"

    def test_infer_rust(self):
        """测试推断 Rust"""
        assert infer_language("Build with Rust") == "rust"
        assert infer_language("Cargo 项目") == "rust"

    def test_infer_cpp_symbol(self):
        """测试推断 C++（符号关键词）"""
        assert infer_language("用C++写一个工具") == "cpp"

    def test_infer_failure(self):
        """测试推断失败"""
        assert infer_language("Create a project") is None
        assert infer_language("Build something") is None
        assert infer_language("") is None

    def test_typescript_priority_over_javascript(self):
        """测试 TypeScript 优先于 JavaScript"""
        # TypeScript 应该优先匹配
        assert infer_language("TypeScript and JavaScript") == "typescript"


# ============================================================
# TestScaffold
# ============================================================


class TestScaffold:
    """测试 scaffold 函数"""

    def test_scaffold_python(self, temp_dir):
        """测试生成 Python 工程骨架"""
        result = scaffold("python", temp_dir, "my_project")

        assert result.success
        assert result.language == "python"
        assert "pyproject.toml" in result.created_files
        assert any("__init__.py" in f for f in result.created_files)

        # 验证文件存在
        assert (temp_dir / "pyproject.toml").exists()
        assert (temp_dir / "src" / "my_project" / "__init__.py").exists()
        assert (temp_dir / "tests" / "__init__.py").exists()

    def test_scaffold_node(self, temp_dir):
        """测试生成 Node.js 工程骨架"""
        result = scaffold("node", temp_dir, "my_app")

        assert result.success
        assert result.language == "node"
        assert "package.json" in result.created_files
        assert "src/index.js" in result.created_files

        # 验证文件存在
        assert (temp_dir / "package.json").exists()
        assert (temp_dir / "src" / "index.js").exists()

    def test_scaffold_typescript(self, temp_dir):
        """测试生成 TypeScript 工程骨架"""
        result = scaffold("typescript", temp_dir, "my_ts_app")

        assert result.success
        assert result.language == "typescript"
        assert "package.json" in result.created_files
        assert "tsconfig.json" in result.created_files
        assert "src/index.ts" in result.created_files

        # 验证文件存在
        assert (temp_dir / "package.json").exists()
        assert (temp_dir / "tsconfig.json").exists()
        assert (temp_dir / "src" / "index.ts").exists()

    def test_scaffold_go(self, temp_dir):
        """测试生成 Go 工程骨架"""
        result = scaffold("go", temp_dir, "my_go_app")

        assert result.success
        assert result.language == "go"
        assert "go.mod" in result.created_files
        assert "cmd/app/main.go" in result.created_files

        # 验证文件存在
        assert (temp_dir / "go.mod").exists()
        assert (temp_dir / "cmd" / "app" / "main.go").exists()

    def test_scaffold_java(self, temp_dir):
        """测试生成 Java 工程骨架"""
        result = scaffold("java", temp_dir, "my_java_app")

        assert result.success
        assert result.language == "java"
        assert "pom.xml" in result.created_files

        # 验证文件存在
        assert (temp_dir / "pom.xml").exists()
        assert (temp_dir / "src" / "main" / "java" / "com" / "example" / "App.java").exists()

    def test_scaffold_rust(self, temp_dir):
        """测试生成 Rust 工程骨架"""
        result = scaffold("rust", temp_dir, "my_rust_app")

        assert result.success
        assert result.language == "rust"
        assert "Cargo.toml" in result.created_files
        assert "src/main.rs" in result.created_files

        # 验证文件存在
        assert (temp_dir / "Cargo.toml").exists()
        assert (temp_dir / "src" / "main.rs").exists()

    def test_scaffold_unsupported_language(self, temp_dir):
        """测试不支持的语言"""
        result = scaffold("cobol", temp_dir, "my_cobol_app")

        assert not result.success
        assert result.error is not None
        assert "不支持" in result.error

    def test_scaffold_creates_directory(self, temp_dir):
        """测试自动创建不存在的目录"""
        new_dir = temp_dir / "new_project"
        result = scaffold("python", new_dir, "my_project")

        assert result.success
        assert new_dir.exists()

    def test_scaffold_default_project_name(self, temp_dir):
        """测试默认项目名（从目录名推断）"""
        result = scaffold("python", temp_dir)

        assert result.success
        # 项目名应该从目录名推断


# ============================================================
# TestDetectReferenceProjects
# ============================================================


class TestDetectReferenceProjects:
    """测试 detect_reference_projects 函数"""

    def test_detect_multiple_subprojects(self, nested_projects_dir):
        """测试发现多个子工程"""
        results = detect_reference_projects(nested_projects_dir)

        assert len(results) >= 3

        # 检查发现的语言
        languages = {r.detected_language for r in results}
        assert "python" in languages
        assert "node" in languages
        assert "go" in languages

        # 检查相对路径
        paths = {r.relative_path for r in results}
        assert "backend" in paths
        assert "frontend" in paths
        assert "service" in paths

    def test_detect_no_subprojects(self, python_project):
        """测试无子工程（根目录本身是工程）"""
        results = detect_reference_projects(python_project)

        # 根目录有工程标记，但不应该返回自身
        # 只有子目录中的工程会被返回
        assert len(results) == 0 or all(r.relative_path != "." for r in results)

    def test_detect_respects_max_depth(self, nested_projects_dir):
        """测试尊重最大深度限制"""
        # 创建深层嵌套的工程
        deep_dir = nested_projects_dir / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        (deep_dir / "pyproject.toml").write_text('[project]\nname = "deep"\n', encoding="utf-8")

        # max_depth=1 只扫描直接子目录
        results = detect_reference_projects(nested_projects_dir, max_depth=1)

        # 深层工程（depth > 1）不应被发现
        deep_paths = [r for r in results if "level" in r.relative_path]
        assert len(deep_paths) == 0

        # max_depth=2 应该发现顶层子工程（backend, frontend, service）
        results2 = detect_reference_projects(nested_projects_dir, max_depth=2)
        shallow_paths = [r for r in results2 if r.relative_path in ("backend", "frontend", "service")]
        assert len(shallow_paths) >= 3

    def test_detect_respects_max_results(self, nested_projects_dir):
        """测试尊重最大结果数限制"""
        results = detect_reference_projects(nested_projects_dir, max_results=2)

        assert len(results) <= 2

    def test_detect_empty_directory(self, empty_dir):
        """测试空目录"""
        results = detect_reference_projects(empty_dir)

        assert len(results) == 0


# ============================================================
# TestPrepareWorkspace
# ============================================================


class TestPrepareWorkspace:
    """测试 prepare_workspace 函数"""

    def test_prepare_existing_project(self, python_project):
        """测试已有工程目录"""
        result = prepare_workspace(
            target_dir=python_project,
            task_text="添加新功能",
        )

        assert result.project_info.state == ProjectState.EXISTING_PROJECT
        assert result.scaffold_result is None  # 不应生成脚手架
        assert result.error is None

    def test_prepare_empty_dir_with_language(self, empty_dir):
        """测试空目录 + 可推断语言"""
        result = prepare_workspace(
            target_dir=empty_dir,
            task_text="用 Python 写一个 CLI 工具",
        )

        assert result.project_info.state == ProjectState.EXISTING_PROJECT  # 已初始化
        assert result.project_info.is_newly_initialized
        assert result.project_info.detected_language == "python"
        assert result.scaffold_result is not None
        assert result.scaffold_result.success
        assert result.error is None

    def test_prepare_empty_dir_dry_run(self, empty_dir):
        """测试空目录 + dry-run 模式不创建脚手架"""
        result = prepare_workspace(
            target_dir=empty_dir,
            task_text="用 Python 写一个 CLI 工具",
            dry_run=True,
        )

        assert result.error is not None
        assert "dry-run" in result.error
        assert result.project_info.detected_language == "python"
        assert not (empty_dir / "pyproject.toml").exists()

    def test_prepare_empty_dir_without_language(self, empty_dir):
        """测试空目录 + 无法推断语言"""
        result = prepare_workspace(
            target_dir=empty_dir,
            task_text="创建一个项目",
        )

        assert result.error is not None
        assert result.hint is not None
        assert "支持的语言" in result.hint

    def test_prepare_with_explicit_language(self, empty_dir):
        """测试显式指定语言"""
        result = prepare_workspace(
            target_dir=empty_dir,
            task_text="创建一个项目",
            explicit_language="rust",
        )

        assert result.project_info.is_newly_initialized
        assert result.project_info.detected_language == "rust"
        assert result.scaffold_result.success
        assert result.error is None

    def test_prepare_discovers_reference_projects(self, nested_projects_dir):
        """测试发现参考子工程"""
        result = prepare_workspace(
            target_dir=nested_projects_dir,
            task_text="添加新功能",
        )

        assert len(result.reference_projects) >= 3


# ============================================================
# TestUtilityFunctions
# ============================================================


class TestUtilityFunctions:
    """测试工具函数"""

    def test_get_supported_languages(self):
        """测试获取支持的语言列表"""
        languages = get_supported_languages()

        assert "python" in languages
        assert "node" in languages
        assert "typescript" in languages
        assert "go" in languages
        assert "java" in languages
        assert "rust" in languages

    def test_get_language_hint(self):
        """测试获取语言提示信息"""
        hint = get_language_hint()

        assert "python" in hint.lower()
        assert "node" in hint.lower() or "javascript" in hint.lower()
        assert "go" in hint.lower()


# ============================================================
# TestConstants
# ============================================================


class TestConstants:
    """测试常量定义"""

    def test_project_markers_coverage(self):
        """测试工程标记文件覆盖主流语言"""
        markers = PROJECT_MARKERS

        # Python
        assert "pyproject.toml" in markers
        assert "requirements.txt" in markers

        # Node.js
        assert "package.json" in markers

        # Go
        assert "go.mod" in markers

        # Java
        assert "pom.xml" in markers

        # Rust
        assert "Cargo.toml" in markers

    def test_source_extensions_coverage(self):
        """测试源码扩展名覆盖主流语言"""
        extensions = SOURCE_EXTENSIONS

        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        assert ".go" in extensions
        assert ".java" in extensions
        assert ".rs" in extensions

    def test_language_keywords_coverage(self):
        """测试语言关键词覆盖主流语言"""
        keywords = LANGUAGE_KEYWORDS

        assert "python" in keywords
        assert "node" in keywords
        assert "typescript" in keywords
        assert "go" in keywords
        assert "java" in keywords
        assert "rust" in keywords
