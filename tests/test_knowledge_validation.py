"""知识库验证测试模块

验证知识库是否已正确建立并能正常使用。

测试内容:
1. 知识库初始化验证
2. 文档添加与存储验证
3. 搜索功能验证（关键词、语义、混合）
4. 向量索引验证
5. 完整工作流验证

运行方式:
    pytest tests/test_knowledge_validation.py -v
    python tests/test_knowledge_validation.py  # 直接运行验证
"""
import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

from knowledge import (
    ChunkSplitter,
    Document,
    DocumentChunk,
    KnowledgeManager,
    KnowledgeStorage,
)
from knowledge.vector import KnowledgeVectorConfig

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_workspace():
    """创建临时工作目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    """创建测试用的示例文档"""
    return [
        Document(
            id="doc-python-001",
            url="https://example.com/python-guide",
            title="Python 编程指南",
            content="""Python 是一种高级编程语言，以简洁的语法和强大的功能著称。

Python 广泛应用于 Web 开发、数据科学、人工智能、自动化脚本等领域。

Python 的主要特点包括：
- 易于学习和阅读
- 丰富的标准库
- 跨平台支持
- 动态类型系统
- 支持多种编程范式
""",
        ),
        Document(
            id="doc-ml-001",
            url="https://example.com/machine-learning",
            title="机器学习入门",
            content="""机器学习是人工智能的一个重要分支，让计算机能够从数据中学习模式。

常见的机器学习类型包括：
- 监督学习：使用带标签的数据进行训练
- 无监督学习：在无标签数据中发现模式
- 强化学习：通过与环境交互学习

Python 是机器学习最流行的语言，拥有 scikit-learn、TensorFlow、PyTorch 等强大工具。
""",
        ),
        Document(
            id="doc-web-001",
            url="https://example.com/web-dev",
            title="Web 开发技术",
            content="""现代 Web 开发涉及前端和后端两个方面。

前端技术栈：
- HTML/CSS/JavaScript
- React、Vue、Angular 等框架
- TypeScript 类型安全

后端技术：
- Python (Django, Flask, FastAPI)
- Node.js (Express, Nest.js)
- 数据库 (PostgreSQL, MongoDB)
""",
        ),
    ]


@pytest.fixture
def vector_config(temp_workspace) -> KnowledgeVectorConfig:
    """创建测试用的向量配置"""
    return KnowledgeVectorConfig(
        enabled=True,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_storage_path=temp_workspace,
        chunk_size=200,
        chunk_overlap=20,
        default_top_k=10,
        min_similarity_score=0.1,
    )


# ============================================================
# 第一部分: 知识库初始化验证
# ============================================================

class TestKnowledgeBaseInitialization:
    """测试知识库初始化"""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """验证 KnowledgeManager 能正确初始化"""
        manager = KnowledgeManager(name="test-init")

        assert manager.name == "test-init"
        assert len(manager) == 0
        assert manager._initialized is False

        await manager.initialize()

        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_storage_initialization(self, temp_workspace):
        """验证 KnowledgeStorage 能正确初始化"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)

        await storage.initialize()

        assert storage._initialized is True
        assert storage.storage_path.exists()
        assert storage.docs_path.exists()

    @pytest.mark.asyncio
    async def test_repeated_initialization(self):
        """验证重复初始化的幂等性"""
        manager = KnowledgeManager()

        await manager.initialize()
        first_state = manager._initialized

        await manager.initialize()
        second_state = manager._initialized

        assert first_state == second_state == True

    def test_manager_with_vector_config(self, vector_config):
        """验证带向量配置的管理器初始化"""
        manager = KnowledgeManager(vector_config=vector_config)

        assert manager.vector_search_enabled is True


# ============================================================
# 第二部分: 文档添加与存储验证
# ============================================================

class TestDocumentStorage:
    """测试文档添加与存储"""

    @pytest.mark.asyncio
    async def test_add_document_manually(self):
        """验证手动添加文档"""
        manager = KnowledgeManager()
        manager._initialized = True

        doc = Document(
            url="https://example.com/test",
            title="测试文档",
            content="这是测试内容。",
        )

        manager._knowledge_base.add_document(doc)
        manager._url_to_doc_id[doc.url] = doc.id

        assert len(manager) == 1
        assert manager.get_document(doc.id) is not None

    @pytest.mark.asyncio
    async def test_storage_persistence(self, temp_workspace):
        """验证文档持久化存储"""
        # 保存文档
        storage1 = KnowledgeStorage(workspace_root=temp_workspace)
        await storage1.initialize()

        doc = Document(
            id="doc-persist-test",
            url="https://example.com/persist",
            title="持久化测试",
            content="这个文档应该被持久化保存。",
        )

        success, _ = await storage1.save_document(doc)
        assert success is True

        # 重新加载
        storage2 = KnowledgeStorage(workspace_root=temp_workspace)
        await storage2.initialize()

        loaded = await storage2.load_document("doc-persist-test")
        assert loaded is not None
        assert loaded.title == "持久化测试"

    @pytest.mark.asyncio
    async def test_document_deduplication(self, temp_workspace):
        """验证文档去重"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            url="https://example.com/dedup",
            title="去重测试",
            content="相同内容",
        )

        # 第一次保存
        success1, _ = await storage.save_document(doc)
        assert success1 is True

        # 相同内容第二次保存
        success2, message2 = await storage.save_document(doc)
        assert success2 is False
        assert "未变化" in message2

    @pytest.mark.asyncio
    async def test_document_update(self, temp_workspace):
        """验证文档更新"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            id="doc-update-test",
            url="https://example.com/update",
            title="原标题",
            content="原内容",
        )

        await storage.save_document(doc)

        # 修改内容并强制更新
        doc.title = "新标题"
        doc.content = "新内容"
        success, _ = await storage.save_document(doc, force=True)

        assert success is True

        loaded = await storage.load_document("doc-update-test")
        assert loaded.title == "新标题"

    @pytest.mark.asyncio
    async def test_document_deletion(self, temp_workspace):
        """验证文档删除"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            id="doc-delete-test",
            url="https://example.com/delete",
            title="删除测试",
            content="即将被删除",
        )

        await storage.save_document(doc)
        assert storage.has_document("doc-delete-test") is True

        result = await storage.delete_document("doc-delete-test")
        assert result is True
        assert storage.has_document("doc-delete-test") is False


# ============================================================
# 第三部分: 搜索功能验证
# ============================================================

class TestSearchFunctionality:
    """测试搜索功能"""

    def test_keyword_search_basic(self, sample_documents):
        """验证基本关键词搜索"""
        manager = KnowledgeManager()
        manager._initialized = True

        # 添加测试文档
        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        # 搜索
        results = manager.search("Python")

        assert len(results) >= 2  # Python 出现在多个文档中
        assert any("Python" in r.title for r in results)

    def test_keyword_search_no_match(self, sample_documents):
        """验证无匹配的关键词搜索"""
        manager = KnowledgeManager()
        manager._initialized = True

        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)

        # 使用 min_score > 0 过滤掉无匹配的结果
        results = manager.search("XYZNOTEXIST12345", min_score=0.1)

        assert len(results) == 0

    def test_keyword_search_chinese(self, sample_documents):
        """验证中文关键词搜索"""
        manager = KnowledgeManager()
        manager._initialized = True

        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        results = manager.search("机器学习")

        assert len(results) >= 1
        assert any("机器学习" in r.title for r in results)

    def test_keyword_search_result_ranking(self, sample_documents):
        """验证搜索结果排序"""
        manager = KnowledgeManager()
        manager._initialized = True

        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        results = manager.search("Python", max_results=10)

        # 验证按分数降序排列
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_empty_query(self):
        """验证空查询"""
        manager = KnowledgeManager()
        manager._initialized = True

        results = manager.search("")
        assert results == []

        results = manager.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_async_search_modes(self, sample_documents):
        """验证异步搜索模式"""
        manager = KnowledgeManager()
        manager._initialized = True

        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        # 测试关键词模式
        results = await manager.search_async("Python", search_mode="keyword")
        assert isinstance(results, list)

        # 向量搜索未启用时应回退到关键词
        results = await manager.search_async("Python", search_mode="hybrid")
        assert isinstance(results, list)


# ============================================================
# 第四部分: 向量索引验证
# ============================================================

class TestVectorIndexing:
    """测试向量索引功能"""

    def test_chunk_splitter(self):
        """验证文档分块器"""
        splitter = ChunkSplitter(chunk_size=100, overlap=10)

        text = "这是第一段。\n\n这是第二段。\n\n这是第三段。"
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_with_source_doc(self):
        """验证分块保留来源信息"""
        splitter = ChunkSplitter(chunk_size=50)

        text = "测试内容 " * 20
        chunks = splitter.split(text, source_doc="doc-source-test")

        for chunk in chunks:
            assert chunk.source_doc == "doc-source-test"

    def test_chunk_metadata(self):
        """验证分块元数据"""
        splitter = ChunkSplitter(chunk_size=100)

        text = "测试内容 " * 30
        chunks = splitter.split(text)

        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i

    def test_vector_config_creation(self):
        """验证向量配置创建"""
        config = KnowledgeVectorConfig(
            enabled=True,
            chunk_size=256,
            chunk_overlap=25,
            semantic_weight=0.8,
        )

        assert config.enabled is True
        assert config.chunk_size == 256
        assert config.semantic_weight == 0.8

    def test_vector_config_to_dict(self):
        """验证向量配置序列化"""
        config = KnowledgeVectorConfig(chunk_size=512)
        data = config.to_dict()

        assert "chunk_size" in data
        assert "embedding_model" in data
        assert data["chunk_size"] == 512

    def test_vector_config_from_dict(self):
        """验证向量配置反序列化"""
        data = {
            "enabled": True,
            "chunk_size": 1024,
            "semantic_weight": 0.6,
        }

        config = KnowledgeVectorConfig.from_dict(data)

        assert config.chunk_size == 1024
        assert config.semantic_weight == 0.6


# ============================================================
# 第五部分: 完整工作流验证
# ============================================================

class TestCompleteWorkflow:
    """测试完整工作流"""

    @pytest.mark.asyncio
    async def test_full_workflow_without_network(self, sample_documents, temp_workspace):
        """验证完整工作流（无网络）"""
        # 1. 创建管理器
        manager = KnowledgeManager(name="workflow-test")
        manager._initialized = True

        # 2. 添加文档
        for doc in sample_documents:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        assert len(manager) == 3

        # 3. 搜索
        results = manager.search("Python")
        assert len(results) >= 2

        # 4. 获取文档
        doc = manager.get_document("doc-python-001")
        assert doc is not None
        assert doc.title == "Python 编程指南"

        # 5. 按 URL 获取
        doc = manager.get_document_by_url("https://example.com/python-guide")
        assert doc is not None

        # 6. 列出文档
        docs = manager.list()
        assert len(docs) == 3

        # 7. 分页列出
        docs = manager.list(limit=2)
        assert len(docs) == 2

        # 8. 统计
        stats = manager.stats
        assert stats.document_count == 3

        # 9. 删除文档
        result = manager.remove("doc-web-001")
        assert result is True
        assert len(manager) == 2

        # 10. 清空
        count = manager.clear()
        assert count == 2
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_storage_workflow(self, sample_documents, temp_workspace):
        """验证存储工作流"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        # 保存所有文档
        for doc in sample_documents:
            success, _ = await storage.save_document(doc)
            assert success is True

        # 列出文档
        entries = await storage.list_documents()
        assert len(entries) == 3

        # 搜索
        results = await storage.search("Python")
        assert len(results) >= 2

        # 统计
        stats = await storage.get_stats()
        assert stats["document_count"] == 3

        # 清空
        result = await storage.clear_all()
        assert result is True

        entries = await storage.list_documents()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_add_local_file(self, temp_workspace):
        """验证添加本地文件"""
        manager = KnowledgeManager()
        await manager.initialize()

        # 创建测试文件
        test_file = Path(temp_workspace) / "test_doc.md"
        test_file.write_text("""# 测试文档

这是一个测试 Markdown 文件。

## 内容

包含一些测试内容。
""", encoding="utf-8")

        # 添加文件
        doc = await manager.add_file(test_file)

        assert doc is not None
        assert doc.title == "测试文档"
        assert "测试 Markdown" in doc.content
        assert doc.metadata.get("source_type") == "file"


# ============================================================
# 验证报告生成
# ============================================================

class KnowledgeValidationReport:
    """知识库验证报告生成器"""

    def __init__(self):
        self.results: list[dict] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def add_result(self, name: str, passed: bool, message: str = ""):
        """添加验证结果"""
        self.results.append({
            "name": name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def generate_report(self) -> str:
        """生成验证报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        lines = [
            "=" * 60,
            "知识库验证报告",
            "=" * 60,
            f"测试时间: {self.start_time.isoformat() if self.start_time else 'N/A'}",
            f"总测试数: {total}",
            f"通过: {passed}",
            f"失败: {failed}",
            "-" * 60,
        ]

        for r in self.results:
            status = "✓" if r["passed"] else "✗"
            lines.append(f"{status} {r['name']}")
            if r["message"]:
                lines.append(f"  └── {r['message']}")

        lines.append("=" * 60)

        if failed == 0:
            lines.append("所有验证通过！知识库功能正常。")
        else:
            lines.append(f"警告: {failed} 个验证失败，请检查相关功能。")

        return "\n".join(lines)


async def run_validation():
    """运行完整的知识库验证"""
    report = KnowledgeValidationReport()
    report.start_time = datetime.now()

    print("开始知识库验证...")
    print("-" * 40)

    # 1. 验证模块导入
    try:
        from knowledge import (
            ChunkSplitter,
            Document,
            DocumentChunk,
            KnowledgeManager,
            KnowledgeStorage,
        )
        report.add_result("模块导入", True, "所有核心模块导入成功")
        print("✓ 模块导入正常")
    except ImportError as e:
        report.add_result("模块导入", False, str(e))
        print(f"✗ 模块导入失败: {e}")

    # 2. 验证 KnowledgeManager 初始化
    manager = None
    try:
        manager = KnowledgeManager(name="validation-test")
        await manager.initialize()
        report.add_result("Manager 初始化", True)
        print("✓ KnowledgeManager 初始化成功")
    except Exception as e:
        report.add_result("Manager 初始化", False, str(e))
        print(f"✗ KnowledgeManager 初始化失败: {e}")

    # 3. 验证文档添加
    if manager is not None:
        try:
            doc = Document(
                url="https://example.com/validation",
                title="验证测试文档",
                content="这是用于验证的测试内容。包含 Python 和机器学习相关内容。",
            )
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

            assert len(manager) == 1
            report.add_result("文档添加", True)
            print("✓ 文档添加成功")
        except Exception as e:
            report.add_result("文档添加", False, str(e))
            print(f"✗ 文档添加失败: {e}")

    # 4. 验证搜索功能
    if manager is not None and len(manager) > 0:
        try:
            results = manager.search("Python")
            assert isinstance(results, list)
            report.add_result("关键词搜索", True, f"返回 {len(results)} 个结果")
            print(f"✓ 关键词搜索成功 (返回 {len(results)} 个结果)")
        except Exception as e:
            report.add_result("关键词搜索", False, str(e))
            print(f"✗ 关键词搜索失败: {e}")

    # 5. 验证文档分块
    try:
        splitter = ChunkSplitter(chunk_size=100)
        chunks = splitter.split("测试内容 " * 20)
        assert len(chunks) >= 1
        report.add_result("文档分块", True, f"生成 {len(chunks)} 个分块")
        print(f"✓ 文档分块成功 (生成 {len(chunks)} 个分块)")
    except Exception as e:
        report.add_result("文档分块", False, str(e))
        print(f"✗ 文档分块失败: {e}")

    # 6. 验证向量配置
    try:
        from knowledge.vector import KnowledgeVectorConfig
        config = KnowledgeVectorConfig(enabled=True, chunk_size=256)
        data = config.to_dict()
        restored = KnowledgeVectorConfig.from_dict(data)
        assert restored.chunk_size == 256
        report.add_result("向量配置", True)
        print("✓ 向量配置正常")
    except Exception as e:
        report.add_result("向量配置", False, str(e))
        print(f"✗ 向量配置失败: {e}")

    # 7. 验证存储功能
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = KnowledgeStorage(workspace_root=temp_dir)
            await storage.initialize()

            doc = Document(
                url="https://example.com/storage-test",
                title="存储测试",
                content="存储验证内容",
            )
            success, _ = await storage.save_document(doc)
            assert success is True

            loaded = await storage.load_document(doc.id)
            assert loaded is not None

            report.add_result("存储功能", True)
            print("✓ 存储功能正常")
    except Exception as e:
        report.add_result("存储功能", False, str(e))
        print(f"✗ 存储功能失败: {e}")

    # 8. 验证清理功能
    if manager is not None:
        try:
            count = manager.clear()
            assert len(manager) == 0
            report.add_result("清理功能", True, f"清理 {count} 个文档")
            print(f"✓ 清理功能正常 (清理 {count} 个文档)")
        except Exception as e:
            report.add_result("清理功能", False, str(e))
            print(f"✗ 清理功能失败: {e}")

    report.end_time = datetime.now()

    print()
    print(report.generate_report())

    return report


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # 使用 pytest 运行
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # 直接运行验证
        asyncio.run(run_validation())
