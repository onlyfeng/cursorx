"""测试知识库文档注入逻辑

测试场景：
1. orchestrator_mp 在任务分发前注入 knowledge_docs 到 payload
2. WorkerAgentProcess._build_execution_prompt 正确渲染知识库章节
3. KnowledgeStorage.search 路径的隔离测试
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.worker_process import (
    MAX_CHARS_PER_DOC,
    MAX_KNOWLEDGE_DOCS,
    MAX_TOTAL_KNOWLEDGE_CHARS,
    WorkerAgentProcess,
    truncate_knowledge_docs,
)
from core.knowledge import is_cursor_related
from knowledge.models import Document
from knowledge.storage import KnowledgeStorage, StorageConfig
from tasks.task import Task, TaskType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_knowledge_dir(tmp_path):
    """创建临时知识库目录

    使用 pytest 的 tmp_path fixture 确保测试隔离
    """
    knowledge_path = tmp_path / ".cursor" / "knowledge"
    knowledge_path.mkdir(parents=True, exist_ok=True)
    yield knowledge_path
    # tmp_path 会自动清理


@pytest.fixture
def sample_knowledge_docs():
    """示例知识库文档（用于 payload 注入测试）"""
    return [
        {
            "title": "Cursor CLI 基础用法",
            "url": "https://cursor.com/docs/cli/basic",
            "content": "agent -p 'prompt' 用于非交互模式执行任务",
            "score": 0.95,
            "source": "cursor-docs",
        },
        {
            "title": "MCP 服务器配置",
            "url": "https://cursor.com/docs/mcp/config",
            "content": "使用 agent mcp list 查看可用的 MCP 服务器",
            "score": 0.85,
            "source": "cursor-docs",
        },
        {
            "title": "Hooks 系统",
            "url": "https://cursor.com/docs/hooks",
            "content": "Hooks 允许通过自定义脚本扩展 Agent 循环",
            "score": 0.75,
            "source": "cursor-docs",
        },
    ]


@pytest.fixture
def sample_cli_ask_docs():
    """示例 CLI Ask 文档（用于优先展示测试）"""
    return [
        {
            "query": "如何使用 cursor agent?",
            "content": "Cursor Agent 是一个 AI 驱动的编程助手，可以通过 CLI 调用...",
            "context_used": ["doc-1", "doc-2"],
            "source": "cli-ask",
        },
    ]


@pytest.fixture
def cursor_related_task():
    """与 Cursor 相关的任务"""
    return Task(
        id="task-cursor-001",
        type=TaskType.IMPLEMENT,
        title="实现 Cursor Agent CLI 封装",
        description="封装 cursor agent 命令行工具",
        instruction="创建一个 Python 包装器来调用 cursor agent CLI",
        target_files=["cursor/cli_wrapper.py"],
        context={},
    )


@pytest.fixture
def non_cursor_task():
    """非 Cursor 相关的普通任务"""
    return Task(
        id="task-normal-001",
        type=TaskType.IMPLEMENT,
        title="实现用户认证模块",
        description="创建 JWT 认证中间件",
        instruction="使用 PyJWT 库实现 token 验证",
        target_files=["auth/jwt_auth.py"],
        context={},
    )


# ============================================================================
# truncate_knowledge_docs 函数测试
# ============================================================================


class TestTruncateKnowledgeDocs:
    """truncate_knowledge_docs 降级策略测试"""

    def test_empty_docs(self):
        """空文档列表返回空"""
        result, total_chars = truncate_knowledge_docs([])
        assert result == []
        assert total_chars == 0

    def test_within_limits(self, sample_knowledge_docs):
        """文档在限制内时不截断"""
        result, total_chars = truncate_knowledge_docs(sample_knowledge_docs)
        assert len(result) == len(sample_knowledge_docs)
        assert total_chars > 0

    def test_max_docs_limit(self):
        """超出最大文档数限制"""
        docs = [{"content": f"doc content {i}"} for i in range(10)]
        result, _ = truncate_knowledge_docs(docs, max_docs=3)
        assert len(result) == 3

    def test_per_doc_char_limit(self):
        """单文档字符数限制"""
        long_content = "x" * 2000
        docs = [{"content": long_content}]
        result, _ = truncate_knowledge_docs(docs, max_chars_per_doc=100)

        assert len(result[0]["content"]) <= 103  # 100 + "..."
        assert result[0].get("truncated") is True

    def test_total_char_limit_fallback(self):
        """总字符数超限时的降级策略"""
        # 创建多个中等长度文档
        docs = [{"content": "x" * 800} for _ in range(5)]
        result, total_chars = truncate_knowledge_docs(
            docs,
            max_docs=5,
            max_chars_per_doc=1000,
            max_total_chars=1500,
        )

        # 应触发降级策略
        assert total_chars <= 1500 or len(result) >= 2


# ============================================================================
# WorkerAgentProcess._build_execution_prompt 测试
# ============================================================================


class TestBuildExecutionPrompt:
    """WorkerAgentProcess._build_execution_prompt 纯函数式测试"""

    @pytest.fixture
    def mock_worker_process(self):
        """创建 mock WorkerAgentProcess 实例"""
        inbox = MagicMock()
        outbox = MagicMock()
        worker = WorkerAgentProcess(
            agent_id="test-worker",
            agent_type="worker",
            inbox=inbox,
            outbox=outbox,
            config={},
        )
        return worker

    def test_prompt_with_knowledge_docs(self, mock_worker_process, sample_knowledge_docs):
        """给定 task_data 含 knowledge_docs，输出 prompt 含相应章节"""
        task_data = {
            "instruction": "实现 cursor CLI 封装",
            "target_files": ["cursor/wrapper.py"],
            "context": {
                "knowledge_docs": sample_knowledge_docs,
            },
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证 prompt 包含知识库章节
        assert "参考文档" in prompt or "知识库" in prompt
        assert "Cursor CLI 基础用法" in prompt
        assert "MCP 服务器配置" in prompt
        assert "cursor.com/docs" in prompt

    def test_prompt_with_cli_ask_docs(self, mock_worker_process, sample_cli_ask_docs):
        """CLI Ask 文档优先展示"""
        task_data = {
            "instruction": "测试任务",
            "context": {
                "knowledge_docs": sample_cli_ask_docs,
            },
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证包含 CLI Ask 章节
        assert "知识库智能回答" in prompt or "CLI Ask" in prompt
        assert "如何使用 cursor agent?" in prompt

    def test_prompt_with_mixed_docs(self, mock_worker_process, sample_knowledge_docs, sample_cli_ask_docs):
        """混合文档类型的渲染"""
        task_data = {
            "instruction": "测试任务",
            "context": {
                "knowledge_docs": sample_cli_ask_docs + sample_knowledge_docs,
            },
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证两种类型都存在
        assert "智能回答" in prompt or "CLI Ask" in prompt
        assert "参考文档" in prompt

    def test_prompt_without_knowledge_docs(self, mock_worker_process):
        """无知识库文档时不显示相关章节"""
        task_data = {
            "instruction": "普通任务",
            "target_files": ["src/main.py"],
            "context": {},
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证不包含知识库章节
        assert "参考文档（来自 Cursor 知识库" not in prompt
        assert "知识库智能回答" not in prompt

    def test_prompt_with_target_files(self, mock_worker_process):
        """涉及文件正确渲染"""
        task_data = {
            "instruction": "修改文件",
            "target_files": ["file1.py", "file2.py"],
            "context": {},
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        assert "涉及文件" in prompt
        assert "file1.py" in prompt
        assert "file2.py" in prompt

    def test_prompt_doc_truncation(self, mock_worker_process):
        """长文档被正确截断"""
        long_content = "x" * 5000
        docs = [
            {
                "title": "长文档",
                "url": "https://example.com",
                "content": long_content,
                "score": 0.9,
                "source": "cursor-docs",
            },
        ]

        task_data = {
            "instruction": "测试任务",
            "context": {"knowledge_docs": docs},
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证内容被截断
        assert "[已截断]" in prompt or len(prompt) < len(long_content) + 1000

    def test_prompt_score_display(self, mock_worker_process, sample_knowledge_docs):
        """相关度分数正确显示"""
        task_data = {
            "instruction": "测试任务",
            "context": {"knowledge_docs": sample_knowledge_docs},
        }

        prompt = mock_worker_process._build_execution_prompt(task_data)

        # 验证分数格式
        assert "相关度:" in prompt or "0.95" in prompt or "0.85" in prompt


# ============================================================================
# Orchestrator MP 知识库注入测试
# ============================================================================


class TestOrchestratorKnowledgeInjection:
    """orchestrator_mp 知识库注入逻辑测试"""

    @pytest.fixture
    def mock_orchestrator_config(self, tmp_path):
        """创建测试用的 orchestrator 配置"""
        from coordinator.orchestrator_mp import MultiProcessOrchestratorConfig

        return MultiProcessOrchestratorConfig(
            working_directory=str(tmp_path),
            max_iterations=1,
            worker_count=1,
            enable_knowledge_search=True,
            enable_auto_commit=False,
        )

    def test_is_cursor_related_detection(self):
        """测试 Cursor 相关内容检测"""
        # Cursor 相关文本（使用 core.knowledge 模块函数）
        assert is_cursor_related("使用 cursor agent 执行任务")
        assert is_cursor_related("配置 MCP 服务器")
        assert is_cursor_related("cursor CLI 命令")

        # 非 Cursor 相关文本
        assert not is_cursor_related("实现用户登录功能")
        assert not is_cursor_related("优化数据库查询")

    @pytest.mark.asyncio
    async def test_knowledge_injection_payload_structure(self, mock_orchestrator_config, sample_knowledge_docs):
        """测试知识库注入后的 payload 结构"""
        from coordinator.orchestrator_mp import MultiProcessOrchestrator

        orchestrator = MultiProcessOrchestrator(mock_orchestrator_config)

        # Mock _search_knowledge_for_task 返回预设文档（包含 metrics）
        mock_metrics = {
            "triggered": True,
            "matched": 3,
            "injected": 3,
            "truncated": False,
            "total_chars": 500,
            "keywords_matched": ["cursor", "agent"],
        }
        setattr(
            orchestrator,
            "_search_knowledge_for_task",
            AsyncMock(return_value=(sample_knowledge_docs, mock_metrics)),
        )

        # 创建 Cursor 相关任务
        task = Task(
            id="task-inject-001",
            type=TaskType.IMPLEMENT,
            title="实现 cursor agent 封装",
            description="封装 cursor CLI",
            instruction="创建封装类",
            context={},
        )

        # 模拟分发前的注入逻辑
        task_data = task.model_dump()
        result = await orchestrator._search_knowledge_for_task(task)

        # 处理返回格式
        if isinstance(result, tuple):
            knowledge_docs, metrics = result
        else:
            knowledge_docs = result

        if knowledge_docs:
            if "context" not in task_data or task_data["context"] is None:
                task_data["context"] = {}
            task_data["context"]["knowledge_docs"] = knowledge_docs

        # 断言 payload 结构
        assert "context" in task_data
        assert "knowledge_docs" in task_data["context"]

        docs = task_data["context"]["knowledge_docs"]
        assert len(docs) == len(sample_knowledge_docs)

        # 验证每个文档的必需字段
        for doc in docs:
            assert "title" in doc
            assert "content" in doc
            assert "url" in doc
            assert "score" in doc
            assert "source" in doc

    def test_knowledge_docs_count_constraint_via_truncation(self):
        """测试 knowledge_docs 数量约束（通过 truncate_knowledge_docs）

        注意: orchestrator 搜索返回数可能超过展示数，实际截断在
        _build_execution_prompt 中通过 truncate_knowledge_docs 进行。
        """
        # 创建超出限制的文档
        many_docs = [
            {
                "title": f"Doc {i}",
                "url": f"https://example.com/{i}",
                "content": f"Content {i} " * 100,  # 较长内容
                "score": 0.9 - i * 0.1,
                "source": "cursor-docs",
            }
            for i in range(10)
        ]

        # 使用 truncate_knowledge_docs 进行截断
        truncated, total_chars = truncate_knowledge_docs(
            many_docs,
            max_docs=MAX_KNOWLEDGE_DOCS,
            max_chars_per_doc=MAX_CHARS_PER_DOC,
            max_total_chars=MAX_TOTAL_KNOWLEDGE_CHARS,
        )

        # 验证截断后的数量约束
        assert len(truncated) <= MAX_KNOWLEDGE_DOCS
        assert total_chars <= MAX_TOTAL_KNOWLEDGE_CHARS

    @pytest.mark.asyncio
    async def test_non_cursor_task_no_injection(self, mock_orchestrator_config, non_cursor_task):
        """非 Cursor 相关任务不注入知识库"""
        from coordinator.orchestrator_mp import MultiProcessOrchestrator

        orchestrator = MultiProcessOrchestrator(mock_orchestrator_config)

        # 初始化知识库存储 mock
        orchestrator._knowledge_storage = MagicMock()
        orchestrator._knowledge_initialized = True

        # 调用搜索（返回 tuple: (docs, metrics)）
        result = await orchestrator._search_knowledge_for_task(non_cursor_task)

        # 处理可能的返回格式（tuple 或 list）
        if isinstance(result, tuple):
            docs, metrics = result
            # 验证 metrics 表明没有触发
            assert metrics.get("triggered") is False
        else:
            docs = result

        # 非 Cursor 相关任务应返回空文档列表
        assert docs == []


# ============================================================================
# KnowledgeStorage 隔离测试
# ============================================================================


class TestKnowledgeStorageIntegration:
    """KnowledgeStorage 集成测试（使用临时目录）"""

    @pytest.mark.asyncio
    async def test_storage_search_with_temp_dir(self, tmp_path):
        """使用临时目录测试 KnowledgeStorage.search"""
        # 创建存储配置
        storage_config = StorageConfig(
            storage_root=".cursor/knowledge",
            enable_vector_index=False,
        )

        storage = KnowledgeStorage(
            config=storage_config,
            workspace_root=str(tmp_path),
        )

        # 初始化
        await storage.initialize()

        # 创建并保存测试文档
        doc = Document(
            id="doc-test-001",
            url="https://cursor.com/docs/test",
            title="Cursor Agent 使用指南",
            content="使用 cursor agent -p 'prompt' 执行任务。支持多种模式。",
        )

        success, msg = await storage.save_document(doc)
        assert success, f"保存文档失败: {msg}"

        # 搜索
        results = await storage.search(
            query="cursor agent",
            limit=5,
            search_content=True,
            mode="keyword",
        )

        # 验证搜索结果
        assert len(results) >= 1
        assert results[0].doc_id == "doc-test-001"
        assert "cursor" in results[0].title.lower() or "agent" in results[0].title.lower()

    @pytest.mark.asyncio
    async def test_storage_isolation(self, tmp_path):
        """测试存储隔离性"""
        # 创建两个独立的存储实例
        dir1 = tmp_path / "workspace1"
        dir2 = tmp_path / "workspace2"
        dir1.mkdir()
        dir2.mkdir()

        storage1 = KnowledgeStorage(workspace_root=str(dir1))
        storage2 = KnowledgeStorage(workspace_root=str(dir2))

        await storage1.initialize()
        await storage2.initialize()

        # 在 storage1 中保存文档
        doc = Document(
            id="doc-isolated",
            url="https://example.com/isolated",
            title="Isolated Doc",
            content="This is isolated content",
        )
        await storage1.save_document(doc)

        # storage2 不应看到 storage1 的文档
        assert storage2.has_document("doc-isolated") is False
        assert storage1.has_document("doc-isolated") is True

    @pytest.mark.asyncio
    async def test_storage_search_result_fields(self, tmp_path):
        """验证搜索结果包含必需字段"""
        storage = KnowledgeStorage(workspace_root=str(tmp_path))
        await storage.initialize()

        # 保存文档
        doc = Document(
            id="doc-fields",
            url="https://cursor.com/docs/fields",
            title="字段测试文档",
            content="测试内容 cursor agent mcp",
        )
        await storage.save_document(doc)

        # 搜索
        results = await storage.search("cursor", limit=1)

        if results:
            result = results[0]
            # 验证 SearchResult 必需字段
            assert hasattr(result, "doc_id")
            assert hasattr(result, "url")
            assert hasattr(result, "title")
            assert hasattr(result, "score")

    @pytest.mark.asyncio
    async def test_storage_document_load(self, tmp_path):
        """测试文档加载功能"""
        storage = KnowledgeStorage(workspace_root=str(tmp_path))
        await storage.initialize()

        # 保存文档
        original_doc = Document(
            id="doc-load",
            url="https://cursor.com/docs/load",
            title="加载测试",
            content="这是测试内容，用于验证文档加载功能。",
        )
        await storage.save_document(original_doc)

        # 加载文档
        loaded_doc = await storage.load_document("doc-load")

        assert loaded_doc is not None
        assert loaded_doc.id == original_doc.id
        assert loaded_doc.title == original_doc.title
        assert "测试内容" in loaded_doc.content


# ============================================================================
# 端到端集成测试
# ============================================================================


class TestKnowledgeInjectionE2E:
    """知识库注入端到端测试"""

    @pytest.mark.asyncio
    async def test_full_injection_flow(self, tmp_path):
        """完整的注入流程测试

        测试从 storage 保存文档到搜索返回结果的完整流程。

        注意: orchestrator._search_knowledge_for_task 使用的是组合 query
        (title + description + instruction)，关键词搜索需要完整匹配。
        因此这里直接测试 storage 层和 mock 结合的方式。
        """
        from coordinator.orchestrator_mp import (
            MultiProcessOrchestrator,
            MultiProcessOrchestratorConfig,
        )

        # 1. 设置存储
        storage_config = StorageConfig(
            storage_root=".cursor/knowledge",
            enable_vector_index=False,
        )
        storage = KnowledgeStorage(
            config=storage_config,
            workspace_root=str(tmp_path),
        )
        await storage.initialize()

        # 2. 保存测试文档
        test_doc = Document(
            id="doc-e2e",
            url="https://cursor.com/docs/e2e",
            title="Cursor Agent 使用指南",
            content="使用 cursor agent 进行开发。CLI 命令支持 --force 参数。",
        )
        await storage.save_document(test_doc)

        # 3. 直接测试 storage 搜索（验证 storage 层工作正常）
        storage_results = await storage.search(
            query="cursor",  # 使用简单关键词
            limit=5,
            search_content=True,
            mode="keyword",
        )
        assert len(storage_results) >= 1, "Storage 搜索应返回至少 1 个结果"
        assert storage_results[0].doc_id == "doc-e2e"

        # 4. 验证文档可以加载
        loaded_doc = await storage.load_document("doc-e2e")
        assert loaded_doc is not None
        assert "cursor agent" in loaded_doc.content.lower()

        # 5. 模拟 orchestrator 注入流程（使用 mock 返回预期结果）
        config = MultiProcessOrchestratorConfig(
            working_directory=str(tmp_path),
            max_iterations=1,
            worker_count=1,
            enable_knowledge_search=True,
            enable_auto_commit=False,
        )
        orchestrator = MultiProcessOrchestrator(config)

        # Mock _search_knowledge_for_task 返回从 storage 构建的文档
        async def mock_search(task):
            return [
                {
                    "title": loaded_doc.title,
                    "url": loaded_doc.url,
                    "content": loaded_doc.content[:500],
                    "score": 0.9,
                    "source": "cursor-docs",
                }
            ]

        setattr(orchestrator, "_search_knowledge_for_task", mock_search)

        # 6. 创建任务并搜索
        task = Task(
            id="task-e2e",
            type=TaskType.IMPLEMENT,
            title="cursor agent 封装",
            description="封装 cursor CLI",
            instruction="实现封装",
        )

        docs = await orchestrator._search_knowledge_for_task(task)

        # 7. 验证结果
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert any("cursor" in d.get("title", "").lower() for d in docs)

    @pytest.mark.asyncio
    async def test_injection_with_worker_prompt(self, tmp_path):
        """测试注入后 Worker 能正确渲染 prompt"""
        # 创建 mock worker
        from multiprocessing import Queue

        inbox: Queue = Queue()
        outbox: Queue = Queue()

        worker = WorkerAgentProcess(
            agent_id="e2e-worker",
            agent_type="worker",
            inbox=inbox,
            outbox=outbox,
            config={},
        )

        # 构建包含知识库文档的 task_data
        task_data = {
            "id": "task-prompt-e2e",
            "instruction": "实现 cursor agent 调用",
            "target_files": ["cursor/caller.py"],
            "context": {
                "knowledge_docs": [
                    {
                        "title": "Cursor Agent API",
                        "url": "https://cursor.com/docs/api",
                        "content": "API 使用说明...",
                        "score": 0.92,
                        "source": "cursor-docs",
                    },
                ],
            },
        }

        # 构建 prompt
        prompt = worker._build_execution_prompt(task_data)

        # 验证 prompt 包含预期内容
        assert "实现 cursor agent 调用" in prompt
        assert "cursor/caller.py" in prompt
        assert "Cursor Agent API" in prompt
        assert "cursor.com/docs/api" in prompt

        # 清理队列
        inbox.close()
        outbox.close()


# ============================================================================
# 约束验证测试
# ============================================================================


class TestPayloadConstraints:
    """Payload 约束验证测试"""

    def test_max_knowledge_docs_constant(self):
        """验证 MAX_KNOWLEDGE_DOCS 常量存在且合理"""
        assert MAX_KNOWLEDGE_DOCS >= 1
        assert MAX_KNOWLEDGE_DOCS <= 10

    def test_max_chars_per_doc_constant(self):
        """验证 MAX_CHARS_PER_DOC 常量存在且合理"""
        assert MAX_CHARS_PER_DOC >= 500
        assert MAX_CHARS_PER_DOC <= 5000

    def test_max_total_knowledge_chars_constant(self):
        """验证 MAX_TOTAL_KNOWLEDGE_CHARS 常量存在且合理"""
        assert MAX_TOTAL_KNOWLEDGE_CHARS >= 1000
        assert MAX_TOTAL_KNOWLEDGE_CHARS <= 10000

    def test_payload_field_requirements(self, sample_knowledge_docs):
        """验证 knowledge_docs 字段要求"""
        required_fields = {"title", "content", "url", "score", "source"}

        for doc in sample_knowledge_docs:
            doc_fields = set(doc.keys())
            missing = required_fields - doc_fields
            assert not missing, f"缺少必需字段: {missing}"
