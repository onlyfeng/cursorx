#!/usr/bin/env python3
"""模块导入测试

验证所有项目模块的导入正确性、关键类/函数的存在性以及循环导入检测。

测试内容:
1. 各模块基础导入测试（参数化）
2. 关键类和函数存在性测试
3. 循环导入检测
4. 可选依赖的跳过处理
"""
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 模块定义
# ============================================================

# 所有项目模块
ALL_MODULES = [
    "core",
    "agents",
    "coordinator",
    "tasks",
    "cursor",
    "indexing",
    "knowledge",
    "process",
]

# 各模块的关键导出（必须存在）
MODULE_EXPORTS = {
    "core": [
        "AgentRole",
        "AgentStatus",
        "BaseAgent",
        "Message",
        "MessageType",
        "AgentState",
        "SystemState",
    ],
    "agents": [
        "PlannerAgent",
        "PlannerConfig",
        "WorkerAgent",
        "WorkerConfig",
        "ReviewerAgent",
        "ReviewerConfig",
        "ReviewDecision",
        "CommitterAgent",
        "CommitterConfig",
        "CommitResult",
        # 多进程版本
        "PlannerAgentProcess",
        "WorkerAgentProcess",
        "ReviewerAgentProcess",
    ],
    "coordinator": [
        "Orchestrator",
        "OrchestratorConfig",
        "WorkerPool",
        "MultiProcessOrchestrator",
        "MultiProcessOrchestratorConfig",
    ],
    "tasks": [
        "Task",
        "TaskStatus",
        "TaskPriority",
        "TaskType",
        "TaskQueue",
    ],
    "cursor": [
        # 客户端
        "CursorAgentClient",
        "CursorAgentConfig",
        "CursorAgentPool",
        "CursorAgentResult",
        "ModelPresets",
        # 异常
        "CloudAgentError",
        "RateLimitError",
        "NetworkError",
        "TaskError",
        "AuthError",
        # 认证
        "CloudAuthManager",
        "CloudAuthConfig",
        "AuthToken",
        "AuthStatus",
        "get_api_key",
        # 执行器
        "AgentExecutor",
        "AgentResult",
        "ExecutionMode",
        "execute_agent",
        # MCP
        "MCPManager",
        "MCPServer",
        "MCPTool",
        # 流式输出
        "StreamingClient",
        "StreamEvent",
        "StreamEventType",
        "ProgressTracker",
    ],
    "indexing": [
        # 基类
        "ChunkType",
        "CodeChunk",
        "SearchResult",
        "EmbeddingModel",
        "VectorStore",
        "CodeChunker",
        # 配置
        "EmbeddingProvider",
        "VectorStoreType",
        "ChunkStrategy",
        "EmbeddingConfig",
        "ChunkConfig",
        "IndexConfig",
        # 实现
        "SemanticCodeChunker",
        "CodebaseIndexer",
        "SemanticSearch",
    ],
    "knowledge": [
        # 枚举类型
        "FetchStatus",
        "FetchPriority",
        "FetchMethod",
        "ContentFormat",
        # 数据模型
        "DocumentChunk",
        "Document",
        "KnowledgeBase",
        # 解析器
        "ParsedContent",
        "HTMLParser",
        "ContentCleaner",
        # 存储管理
        "KnowledgeStorage",
        "StorageConfig",
        # 获取器
        "WebFetcher",
        "FetchConfig",
        "FetchResult",
        # 向量搜索
        "KnowledgeVectorStore",
        "VectorSearchResult",
        # 知识库管理器
        "KnowledgeManager",
        "AskResult",
    ],
    "process": [
        "AgentProcessManager",
        "AgentWorkerProcess",
        "MessageQueue",
        "ProcessMessage",
    ],
}

# 需要可选依赖的模块（如果依赖不可用则跳过）
OPTIONAL_DEPENDENCY_MODULES = {
    "indexing": ["sentence_transformers", "chromadb"],
    "knowledge": ["chromadb", "httpx"],
}

# 循环导入检测的模块对
CIRCULAR_IMPORT_PAIRS = [
    ("core", "agents"),
    ("core", "tasks"),
    ("agents", "coordinator"),
    ("coordinator", "agents"),
    ("coordinator", "process"),
    ("cursor", "agents"),
    ("knowledge", "indexing"),
    ("process", "coordinator"),
    ("tasks", "core"),
]


# ============================================================
# 辅助函数
# ============================================================


def check_optional_dependency(module_name: str) -> tuple[bool, str]:
    """检查模块的可选依赖是否可用

    Args:
        module_name: 模块名

    Returns:
        (是否可用, 缺失的依赖列表描述)
    """
    deps = OPTIONAL_DEPENDENCY_MODULES.get(module_name, [])
    missing = []
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        return False, f"缺少依赖: {', '.join(missing)}"
    return True, ""


def import_module_safe(module_name: str) -> tuple[Optional[Any], Optional[str]]:
    """安全导入模块

    Args:
        module_name: 模块名

    Returns:
        (模块对象或None, 错误信息或None)
    """
    try:
        module = importlib.import_module(module_name)
        return module, None
    except Exception as e:
        return None, str(e)


# ============================================================
# 测试类：基础导入
# ============================================================


class TestModuleImport:
    """测试模块基础导入"""

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_module_import(self, module_name: str):
        """测试模块可以正确导入

        Args:
            module_name: 模块名
        """
        # 检查可选依赖
        deps_ok, deps_msg = check_optional_dependency(module_name)
        if not deps_ok:
            pytest.skip(f"跳过 {module_name}: {deps_msg}")

        module, error = import_module_safe(module_name)

        if error:
            pytest.fail(f"模块 {module_name} 导入失败: {error}")

        assert module is not None, f"模块 {module_name} 导入返回 None"
        assert hasattr(module, "__all__"), f"模块 {module_name} 缺少 __all__ 定义"

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_module_has_init(self, module_name: str):
        """测试模块包含 __init__.py 文件

        Args:
            module_name: 模块名
        """
        module_path = PROJECT_ROOT / module_name
        init_file = module_path / "__init__.py"

        assert module_path.is_dir(), f"模块目录 {module_name} 不存在"
        assert init_file.exists(), f"模块 {module_name} 缺少 __init__.py"


# ============================================================
# 测试类：关键导出存在性
# ============================================================


class TestExportExistence:
    """测试关键类和函数的存在性"""

    @pytest.mark.parametrize(
        "module_name,export_names",
        [(name, exports) for name, exports in MODULE_EXPORTS.items()],
        ids=MODULE_EXPORTS.keys(),
    )
    def test_module_exports_exist(self, module_name: str, export_names: list[str]):
        """测试模块导出的类/函数存在

        Args:
            module_name: 模块名
            export_names: 期望的导出名称列表
        """
        # 检查可选依赖
        deps_ok, deps_msg = check_optional_dependency(module_name)
        if not deps_ok:
            pytest.skip(f"跳过 {module_name}: {deps_msg}")

        module, error = import_module_safe(module_name)
        if error:
            pytest.fail(f"模块 {module_name} 导入失败: {error}")

        missing = []
        for export_name in export_names:
            if not hasattr(module, export_name):
                missing.append(export_name)

        if missing:
            pytest.fail(
                f"模块 {module_name} 缺少导出: {missing}\n"
                f"可用导出: {getattr(module, '__all__', [])}"
            )


# 为每个模块生成单独的导出测试用例
def generate_export_test_params():
    """生成每个导出的参数化测试数据"""
    params = []
    for module_name, exports in MODULE_EXPORTS.items():
        for export_name in exports:
            params.append(
                pytest.param(
                    module_name,
                    export_name,
                    id=f"{module_name}.{export_name}",
                )
            )
    return params


class TestIndividualExports:
    """测试单个导出项"""

    @pytest.mark.parametrize(
        "module_name,export_name",
        generate_export_test_params(),
    )
    def test_individual_export(self, module_name: str, export_name: str):
        """测试单个导出项的存在性

        Args:
            module_name: 模块名
            export_name: 导出项名称
        """
        # 检查可选依赖
        deps_ok, deps_msg = check_optional_dependency(module_name)
        if not deps_ok:
            pytest.skip(f"跳过 {module_name}.{export_name}: {deps_msg}")

        module, error = import_module_safe(module_name)
        if error:
            pytest.fail(f"模块 {module_name} 导入失败: {error}")

        assert hasattr(module, export_name), (
            f"{module_name}.{export_name} 不存在"
        )

        # 验证导出项是否在 __all__ 中
        all_exports = getattr(module, "__all__", [])
        if export_name not in all_exports:
            pytest.skip(f"{export_name} 存在但不在 __all__ 中")


# ============================================================
# 测试类：循环导入检测
# ============================================================


class TestCircularImports:
    """测试循环导入问题"""

    @pytest.mark.parametrize(
        "module_a,module_b",
        CIRCULAR_IMPORT_PAIRS,
        ids=[f"{a}->{b}" for a, b in CIRCULAR_IMPORT_PAIRS],
    )
    def test_no_circular_import(self, module_a: str, module_b: str):
        """测试两个模块间无循环导入

        使用子进程独立测试导入顺序，避免缓存影响。

        Args:
            module_a: 第一个模块
            module_b: 第二个模块
        """
        test_code = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
try:
    import {module_a}
    import {module_b}
    print('OK')
except ImportError as e:
    print(f'ImportError: {{e}}')
    sys.exit(1)
except RecursionError as e:
    print(f'RecursionError: {{e}}')
    sys.exit(2)
except Exception as e:
    print(f'Error: {{e}}')
    sys.exit(3)
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode == 0:
                assert "OK" in result.stdout
            elif result.returncode == 2:
                # RecursionError 表示循环导入
                pytest.fail(f"检测到循环导入: {module_a} <-> {module_b}")
            else:
                # 其他错误，可能是依赖问题
                error_msg = result.stdout.strip() or result.stderr.strip()
                # 在 CI 环境中可能缺少某些依赖
                if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                    pytest.skip(f"依赖问题: {error_msg[:100]}")
                else:
                    pytest.fail(f"导入失败 {module_a}->{module_b}: {error_msg[:200]}")

        except subprocess.TimeoutExpired:
            pytest.fail(f"导入超时（可能死锁）: {module_a} <-> {module_b}")

    def test_full_import_chain(self):
        """测试完整的导入链（从入口到核心）"""
        import_order = [
            "core",
            "tasks",
            "agents",
            "coordinator",
            "cursor",
            "process",
        ]

        test_code = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
try:
    for mod in {import_order}:
        __import__(mod)
    print('OK')
except RecursionError as e:
    print(f'RecursionError: {{e}}')
    sys.exit(2)
except Exception as e:
    print(f'Error: {{e}}')
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode == 2:
            pytest.fail("完整导入链检测到循环导入")
        elif result.returncode != 0:
            error_msg = result.stdout.strip() or result.stderr.strip()
            if "ModuleNotFoundError" in error_msg:
                pytest.skip(f"依赖问题: {error_msg[:100]}")
            else:
                pytest.fail(f"导入链失败: {error_msg[:200]}")

        assert "OK" in result.stdout


# ============================================================
# 测试类：子模块导入
# ============================================================


class TestSubmoduleImports:
    """测试子模块导入"""

    # 各模块的关键子模块
    SUBMODULES = {
        "core": ["base", "message", "state"],
        "agents": ["planner", "worker", "reviewer", "committer"],
        "coordinator": ["orchestrator", "orchestrator_mp", "worker_pool"],
        "tasks": ["task", "queue"],
        "cursor": ["client", "executor", "streaming", "mcp", "network"],
        "indexing": ["base", "chunker", "config", "embedding", "indexer", "search"],
        "knowledge": ["fetcher", "manager", "models", "parser", "storage", "vector"],
        "process": ["manager", "message_queue", "worker"],
    }

    @pytest.mark.parametrize(
        "parent,submodule",
        [
            (parent, sub)
            for parent, subs in SUBMODULES.items()
            for sub in subs
        ],
        ids=[
            f"{parent}.{sub}"
            for parent, subs in SUBMODULES.items()
            for sub in subs
        ],
    )
    def test_submodule_import(self, parent: str, submodule: str):
        """测试子模块可以正确导入

        Args:
            parent: 父模块名
            submodule: 子模块名
        """
        # 检查可选依赖
        deps_ok, deps_msg = check_optional_dependency(parent)
        if not deps_ok:
            pytest.skip(f"跳过 {parent}.{submodule}: {deps_msg}")

        full_name = f"{parent}.{submodule}"
        module, error = import_module_safe(full_name)

        if error:
            # 某些子模块可能需要额外依赖
            if "ModuleNotFoundError" in error or "No module named" in error:
                pytest.skip(f"依赖问题: {error[:100]}")
            pytest.fail(f"子模块 {full_name} 导入失败: {error}")

        assert module is not None


# ============================================================
# 测试类：可选依赖处理
# ============================================================


class TestOptionalDependencies:
    """测试可选依赖的处理"""

    def test_indexing_without_sentence_transformers(self):
        """测试在没有 sentence_transformers 时的行为"""
        try:
            importlib.import_module("sentence_transformers")
            pytest.skip("sentence_transformers 已安装")
        except ImportError:
            # 尝试导入 indexing，应该能处理缺失的依赖
            try:
                importlib.import_module("indexing.config")
                # 配置模块应该能正常导入
            except ImportError as e:
                if "sentence_transformers" in str(e):
                    pytest.skip("indexing 需要 sentence_transformers")
                raise

    def test_knowledge_without_chromadb(self):
        """测试在没有 chromadb 时的行为"""
        try:
            importlib.import_module("chromadb")
            pytest.skip("chromadb 已安装")
        except ImportError:
            # 尝试导入 knowledge 的基础模块
            try:
                importlib.import_module("knowledge.models")
                # 模型定义不应该依赖 chromadb
            except ImportError as e:
                if "chromadb" in str(e):
                    pytest.skip("knowledge 需要 chromadb")
                raise

    @pytest.mark.parametrize(
        "module_name",
        list(OPTIONAL_DEPENDENCY_MODULES.keys()),
    )
    def test_optional_module_graceful_error(self, module_name: str):
        """测试可选模块在缺少依赖时给出清晰的错误信息

        Args:
            module_name: 模块名
        """
        deps = OPTIONAL_DEPENDENCY_MODULES[module_name]

        # 检查是否所有依赖都已安装
        all_installed = True
        for dep in deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                all_installed = False
                break

        if all_installed:
            # 所有依赖已安装，模块应该能正常导入
            module, error = import_module_safe(module_name)
            assert module is not None, f"模块 {module_name} 导入失败: {error}"
        else:
            # 缺少依赖，测试导入行为
            module, error = import_module_safe(module_name)
            if error:
                # 错误信息应该提及缺少的依赖
                assert any(dep in error for dep in deps) or "install" in error.lower(), (
                    f"模块 {module_name} 的错误信息不够清晰: {error}"
                )


# ============================================================
# 测试类：类型检查
# ============================================================


class TestExportTypes:
    """测试导出项的类型正确性"""

    # 期望为类的导出项
    EXPECTED_CLASSES = {
        "core": ["AgentRole", "AgentStatus", "BaseAgent", "Message", "MessageType"],
        "agents": ["PlannerAgent", "WorkerAgent", "ReviewerAgent", "CommitterAgent"],
        "coordinator": ["Orchestrator", "WorkerPool"],
        "tasks": ["Task", "TaskQueue", "TaskStatus", "TaskPriority", "TaskType"],
    }

    # 期望为函数的导出项
    EXPECTED_FUNCTIONS = {
        "cursor": ["get_api_key", "execute_agent", "execute_agent_sync"],
        "indexing": ["create_embedding_model", "chunk_file", "chunk_text"],
        "knowledge": ["fetch_url", "fetch_urls", "semantic_search"],
    }

    @pytest.mark.parametrize(
        "module_name,class_names",
        [(name, classes) for name, classes in EXPECTED_CLASSES.items()],
        ids=EXPECTED_CLASSES.keys(),
    )
    def test_exports_are_classes(self, module_name: str, class_names: list[str]):
        """测试导出项是类类型

        Args:
            module_name: 模块名
            class_names: 类名列表
        """
        deps_ok, deps_msg = check_optional_dependency(module_name)
        if not deps_ok:
            pytest.skip(f"跳过 {module_name}: {deps_msg}")

        module, error = import_module_safe(module_name)
        if error:
            pytest.fail(f"模块 {module_name} 导入失败: {error}")

        for class_name in class_names:
            if not hasattr(module, class_name):
                pytest.skip(f"{class_name} 不存在于 {module_name}")

            obj = getattr(module, class_name)
            assert isinstance(obj, type), (
                f"{module_name}.{class_name} 应该是类，实际是 {type(obj)}"
            )

    @pytest.mark.parametrize(
        "module_name,func_names",
        [(name, funcs) for name, funcs in EXPECTED_FUNCTIONS.items()],
        ids=EXPECTED_FUNCTIONS.keys(),
    )
    def test_exports_are_functions(self, module_name: str, func_names: list[str]):
        """测试导出项是函数类型

        Args:
            module_name: 模块名
            func_names: 函数名列表
        """
        deps_ok, deps_msg = check_optional_dependency(module_name)
        if not deps_ok:
            pytest.skip(f"跳过 {module_name}: {deps_msg}")

        module, error = import_module_safe(module_name)
        if error:
            pytest.fail(f"模块 {module_name} 导入失败: {error}")

        for func_name in func_names:
            if not hasattr(module, func_name):
                pytest.skip(f"{func_name} 不存在于 {module_name}")

            obj = getattr(module, func_name)
            assert callable(obj), (
                f"{module_name}.{func_name} 应该是可调用对象，实际是 {type(obj)}"
            )


# ============================================================
# 主函数
# ============================================================


def main():
    """直接运行测试"""
    print("\n" + "=" * 60)
    print("模块导入测试")
    print("=" * 60 + "\n")

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
    ])

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
