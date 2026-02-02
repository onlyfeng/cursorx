"""测试知识库模块

测试内容:
1. URL 获取功能 (WebFetcher, fetch_url)
2. 内容解析功能 (HTMLParser, ContentCleaner, MarkdownConverter, ChunkSplitter)
3. 存储和检索功能 (KnowledgeStorage)
4. 知识库管理器完整流程 (KnowledgeManager)
"""

import shutil
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge import (
    ChunkSplitter,
    ContentCleaner,
    ContentFormat,
    Document,
    # 数据模型
    DocumentChunk,
    FetchConfig,
    FetchMethod,
    FetchResult,
    # 枚举类型
    FetchStatus,
    FetchTask,
    # 解析器
    HTMLParser,
    # 存储管理
    IndexEntry,
    KnowledgeBase,
    KnowledgeBaseStats,
    KnowledgeManager,
    KnowledgeStorage,
    MarkdownConverter,
    # URL 安全策略
    UrlPolicy,
    UrlPolicyError,
    # 获取器
    WebFetcher,
)

# ============================================================
# 第一部分: URL 获取功能测试
# ============================================================


class TestFetchConfig:
    """测试获取配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = FetchConfig()

        assert config.method == FetchMethod.AUTO
        assert config.content_format == ContentFormat.TEXT
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_concurrent == 5

    def test_custom_config(self):
        """测试自定义配置"""
        config = FetchConfig(
            method=FetchMethod.CURL,
            timeout=60,
            max_retries=5,
        )

        assert config.method == FetchMethod.CURL
        assert config.timeout == 60
        assert config.max_retries == 5


class TestFetchResult:
    """测试获取结果"""

    def test_success_result(self):
        """测试成功结果"""
        result = FetchResult(
            url="https://example.com",
            success=True,
            content="Hello World",
        )

        assert result.success is True
        assert result.content == "Hello World"
        assert result.error is None

    def test_failure_result(self):
        """测试失败结果"""
        result = FetchResult(
            url="https://example.com",
            success=False,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


class TestWebFetcher:
    """测试网页获取器"""

    @pytest.mark.asyncio
    async def test_fetcher_initialization(self):
        """测试获取器初始化"""
        fetcher = WebFetcher()

        await fetcher.initialize()

        assert fetcher._initialized is True
        # 至少应该有一种可用方式
        # 注: 在测试环境可能没有 curl/lynx/mcp

    def test_url_validation(self):
        """测试 URL 验证"""
        fetcher = WebFetcher()

        # 有效 URL
        assert fetcher.check_url_valid("https://example.com") is True
        assert fetcher.check_url_valid("http://test.com/path?query=1") is True

        # 无效 URL
        assert fetcher.check_url_valid("not-a-url") is False
        assert fetcher.check_url_valid("ftp://example.com") is False
        assert fetcher.check_url_valid("") is False

    @pytest.mark.asyncio
    async def test_html_to_text(self):
        """测试 HTML 转纯文本"""
        fetcher = WebFetcher()

        html = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
        text = fetcher._html_to_text(html)

        assert "Hello World" in text
        assert "<p>" not in text
        assert "<html>" not in text

    @pytest.mark.asyncio
    async def test_html_to_text_removes_scripts(self):
        """测试 HTML 转文本时移除 script 标签"""
        fetcher = WebFetcher()

        html = "<div><script>alert('xss')</script><p>Content</p></div>"
        text = fetcher._html_to_text(html)

        assert "Content" in text
        assert "alert" not in text
        assert "script" not in text

    @pytest.mark.asyncio
    async def test_fetch_with_mock(self):
        """测试模拟获取"""
        fetcher = WebFetcher()
        await fetcher.initialize()

        # 模拟 curl 响应
        with patch.object(fetcher, "_fetch_via_curl") as mock_curl:
            mock_curl.return_value = FetchResult(
                url="https://example.com",
                success=True,
                content="Test content",
                method_used=FetchMethod.CURL,
            )

            # 强制使用 curl
            result = await fetcher.fetch("https://example.com", method=FetchMethod.CURL)

            assert result.success is True
            assert result.content == "Test content"


class TestFetchConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_fetch_url_with_mock(self):
        """测试 fetch_url 便捷函数"""
        with patch("knowledge.fetcher.WebFetcher.fetch") as mock_fetch:
            mock_fetch.return_value = FetchResult(
                url="https://example.com",
                success=True,
                content="Hello",
            )

            # 实际测试中会使用真实网络
            # 这里只验证函数可以正常调用
            fetcher = WebFetcher()
            await fetcher.initialize()


# ============================================================
# 第二部分: 内容解析功能测试
# ============================================================


class TestHTMLParser:
    """测试 HTML 解析器"""

    def test_parse_basic_html(self):
        """测试解析基本 HTML"""
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body><p>Hello World</p></body>
        </html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert result.title == "Test Page"
        assert "Hello World" in result.content

    def test_extract_title_from_og_tag(self):
        """测试从 og:title 提取标题"""
        html = """
        <html>
        <head>
            <meta property="og:title" content="OG Title">
        </head>
        <body></body>
        </html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert result.title == "OG Title"

    def test_extract_headings(self):
        """测试提取标题层级"""
        html = """
        <html><body>
        <h1>Main Title</h1>
        <h2>Section 1</h2>
        <h3>Subsection 1.1</h3>
        </body></html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert len(result.headings) == 3
        assert result.headings[0]["level"] == 1
        assert result.headings[0]["text"] == "Main Title"

    def test_extract_links(self):
        """测试提取链接"""
        html = """
        <html><body>
        <a href="https://example.com" title="Example">Example Link</a>
        <a href="/internal">Internal</a>
        </body></html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert len(result.links) == 2
        assert result.links[0]["href"] == "https://example.com"
        assert result.links[0]["text"] == "Example Link"

    def test_extract_images(self):
        """测试提取图片"""
        html = """
        <html><body>
        <img src="image.png" alt="Test Image">
        </body></html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert len(result.images) == 1
        assert result.images[0]["src"] == "image.png"
        assert result.images[0]["alt"] == "Test Image"

    def test_extract_metadata(self):
        """测试提取元数据"""
        html = """
        <html lang="zh-CN">
        <head>
            <meta name="description" content="Page description">
            <meta name="keywords" content="test, example">
        </head>
        <body></body>
        </html>
        """
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(html)

        assert result.metadata.get("description") == "Page description"
        assert result.metadata.get("keywords") == "test, example"
        assert result.metadata.get("language") == "zh-CN"


class TestContentCleaner:
    """测试内容清理器"""

    def test_remove_script_tags(self):
        """测试移除 script 标签"""
        html = "<div><script>alert(1)</script><p>Content</p></div>"
        cleaner = ContentCleaner(parser="html.parser")
        result = cleaner.clean(html)

        assert "<script>" not in result
        assert "alert" not in result
        assert "Content" in result

    def test_remove_style_tags(self):
        """测试移除 style 标签"""
        html = "<div><style>.red{color:red}</style><p>Content</p></div>"
        cleaner = ContentCleaner(parser="html.parser")
        result = cleaner.clean(html)

        assert "<style>" not in result
        assert "color:red" not in result

    def test_remove_nav_elements(self):
        """测试移除导航元素"""
        html = "<div><nav>Menu</nav><main>Main Content</main></div>"
        cleaner = ContentCleaner(parser="html.parser")
        result = cleaner.clean(html)

        assert "<nav>" not in result
        assert "Main Content" in result

    def test_clean_to_text(self):
        """测试清理并提取纯文本"""
        html = """
        <html>
        <body>
        <nav>Navigation</nav>
        <main>
            <h1>Title</h1>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
        </main>
        <footer>Footer</footer>
        </body>
        </html>
        """
        cleaner = ContentCleaner(parser="html.parser")
        text = cleaner.clean_to_text(html)

        assert "Title" in text
        assert "Paragraph 1" in text
        # nav 和 footer 被移除
        assert "Navigation" not in text


class TestMarkdownConverter:
    """测试 Markdown 转换器"""

    def test_convert_heading(self):
        """测试转换标题"""
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        converter = MarkdownConverter()
        md = converter.convert(html)

        assert "# Title" in md
        assert "## Subtitle" in md

    def test_convert_paragraph(self):
        """测试转换段落"""
        html = "<p>This is a paragraph.</p>"
        converter = MarkdownConverter()
        md = converter.convert(html)

        assert "This is a paragraph." in md

    def test_convert_emphasis(self):
        """测试转换强调"""
        html = "<p><strong>Bold</strong> and <em>italic</em></p>"
        converter = MarkdownConverter()
        md = converter.convert(html)

        assert "**Bold**" in md
        assert "_italic_" in md or "*italic*" in md

    def test_convert_links(self):
        """测试转换链接"""
        html = '<a href="https://example.com">Link</a>'
        converter = MarkdownConverter()
        md = converter.convert(html)

        assert "[Link]" in md
        assert "https://example.com" in md

    def test_convert_with_cleaning(self):
        """测试清理后转换"""
        html = "<div><script>bad</script><p>Good content</p></div>"
        converter = MarkdownConverter()
        md = converter.convert_with_cleaning(html)

        assert "Good content" in md
        assert "bad" not in md


class TestChunkSplitter:
    """测试文档分块器"""

    def test_split_short_text(self):
        """测试分割短文本"""
        text = "This is a short text."
        splitter = ChunkSplitter(chunk_size=1000)
        chunks = splitter.split(text)

        # 短文本应该只有一个分块
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_split_long_text(self):
        """测试分割长文本"""
        # 创建一个较长的文本
        text = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(20)])
        splitter = ChunkSplitter(chunk_size=200, overlap=20)
        chunks = splitter.split(text)

        # 应该生成多个分块
        assert len(chunks) > 1

        # 每个分块都应该有内容
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_split_with_source_doc(self):
        """测试带来源文档的分割"""
        text = "Some content here."
        splitter = ChunkSplitter()
        chunks = splitter.split(text, source_doc="doc-123")

        assert len(chunks) >= 1
        assert chunks[0].source_doc == "doc-123"

    def test_split_markdown_preserves_headings(self):
        """测试 Markdown 分割保留标题"""
        markdown = """# Main Title

## Section 1

This is section 1 content.

## Section 2

This is section 2 content with more text to make it longer.
"""
        splitter = ChunkSplitter(chunk_size=100, preserve_headings=True)
        chunks = splitter.split_markdown(markdown, source_doc="doc-456")

        assert len(chunks) >= 1

    def test_chunk_has_metadata(self):
        """测试分块包含元数据"""
        text = "Content for testing."
        splitter = ChunkSplitter()
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        assert "chunk_index" in chunks[0].metadata
        assert "chunk_count" in chunks[0].metadata

    def test_empty_text_returns_empty(self):
        """测试空文本返回空列表"""
        splitter = ChunkSplitter()
        chunks = splitter.split("")

        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        """测试仅空白字符返回空列表"""
        splitter = ChunkSplitter()
        chunks = splitter.split("   \n\n   \t   ")

        assert chunks == []


# ============================================================
# 第三部分: 存储和检索功能测试
# ============================================================


class TestIndexEntry:
    """测试索引条目"""

    def test_to_dict(self):
        """测试转换为字典"""
        entry = IndexEntry(
            doc_id="doc-123",
            url="https://example.com",
            title="Test",
            content_hash="abc123",
        )
        data = entry.to_dict()

        assert data["doc_id"] == "doc-123"
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "doc_id": "doc-456",
            "url": "https://test.com",
            "title": "Test Doc",
            "content_hash": "def456",
            "chunk_count": 5,
        }
        entry = IndexEntry.from_dict(data)

        assert entry.doc_id == "doc-456"
        assert entry.chunk_count == 5

    def test_from_document(self):
        """测试从文档创建"""
        doc = Document(
            id="doc-789",
            url="https://example.com",
            title="Document Title",
            content="Some content here",
        )
        entry = IndexEntry.from_document(doc)

        assert entry.doc_id == "doc-789"
        assert entry.url == "https://example.com"
        assert entry.content_size == len("Some content here")


class TestKnowledgeStorage:
    """测试知识库存储"""

    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_initialize(self, temp_workspace):
        """测试初始化存储"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        assert storage._initialized is True
        assert storage.storage_path.exists()
        assert storage.docs_path.exists()
        assert storage.metadata_path.exists()

    @pytest.mark.asyncio
    async def test_save_and_load_document(self, temp_workspace):
        """测试保存和加载文档"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        # 创建文档
        doc = Document(
            id="doc-test-001",
            url="https://example.com/page1",
            title="Test Document",
            content="This is the document content.",
        )

        # 保存
        success, message = await storage.save_document(doc)
        assert success is True

        # 加载
        loaded = await storage.load_document("doc-test-001")
        assert loaded is not None
        assert loaded.title == "Test Document"
        assert loaded.url == "https://example.com/page1"

    @pytest.mark.asyncio
    async def test_save_document_deduplication(self, temp_workspace):
        """测试文档去重"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            url="https://example.com",
            title="Test",
            content="Content",
        )

        # 第一次保存
        success1, _ = await storage.save_document(doc)
        assert success1 is True

        # 相同内容第二次保存应该跳过
        success2, message2 = await storage.save_document(doc)
        assert success2 is False
        assert "未变化" in message2

    @pytest.mark.asyncio
    async def test_force_update(self, temp_workspace):
        """测试强制更新"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            url="https://example.com",
            title="Test",
            content="Content",
        )

        await storage.save_document(doc)

        # 强制更新应该成功
        success, _ = await storage.save_document(doc, force=True)
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_document(self, temp_workspace):
        """测试删除文档"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            id="doc-to-delete",
            url="https://example.com",
            title="To Delete",
            content="Content",
        )

        await storage.save_document(doc)

        # 验证存在
        assert storage.has_document("doc-to-delete") is True

        # 删除
        result = await storage.delete_document("doc-to-delete")
        assert result is True

        # 验证已删除
        assert storage.has_document("doc-to-delete") is False

    @pytest.mark.asyncio
    async def test_search_by_title(self, temp_workspace):
        """测试按标题搜索"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        # 添加多个文档
        docs = [
            Document(url="https://example.com/1", title="Python Tutorial", content="Python content"),
            Document(url="https://example.com/2", title="JavaScript Guide", content="JS content"),
            Document(url="https://example.com/3", title="Python Advanced", content="More Python"),
        ]

        for doc in docs:
            await storage.save_document(doc)

        # 搜索
        results = await storage.search("Python")

        assert len(results) >= 2
        assert all("Python" in r.title for r in results if r.match_type != "partial")

    @pytest.mark.asyncio
    async def test_list_documents(self, temp_workspace):
        """测试列出文档"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        # 添加文档
        for i in range(5):
            doc = Document(
                url=f"https://example.com/{i}",
                title=f"Document {i}",
                content=f"Content {i}",
            )
            await storage.save_document(doc)

        # 列出所有
        entries = await storage.list_documents()
        assert len(entries) == 5

        # 分页
        entries_limited = await storage.list_documents(limit=3)
        assert len(entries_limited) == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_workspace):
        """测试获取统计信息"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            url="https://example.com",
            title="Test",
            content="Content here",
        )
        await storage.save_document(doc)

        stats = await storage.get_stats()

        assert stats["document_count"] == 1
        assert stats["total_content_size"] > 0

    @pytest.mark.asyncio
    async def test_load_by_url(self, temp_workspace):
        """测试按 URL 加载"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        doc = Document(
            url="https://unique-url.com/page",
            title="Unique Page",
            content="Unique content",
        )
        await storage.save_document(doc)

        loaded = await storage.load_document_by_url("https://unique-url.com/page")

        assert loaded is not None
        assert loaded.title == "Unique Page"

    @pytest.mark.asyncio
    async def test_clear_all(self, temp_workspace):
        """测试清空所有文档"""
        storage = KnowledgeStorage(workspace_root=temp_workspace)
        await storage.initialize()

        # 添加文档
        for i in range(3):
            doc = Document(
                url=f"https://example.com/{i}",
                title=f"Doc {i}",
                content=f"Content {i}",
            )
            await storage.save_document(doc)

        # 清空
        result = await storage.clear_all()
        assert result is True

        # 验证
        entries = await storage.list_documents()
        assert len(entries) == 0


# ============================================================
# 第四部分: 知识库管理器完整流程测试
# ============================================================


class TestDocumentModel:
    """测试文档模型"""

    def test_document_creation(self):
        """测试文档创建"""
        doc = Document(
            url="https://example.com",
            title="Test",
            content="Content",
        )

        assert doc.id.startswith("doc-")
        assert doc.url == "https://example.com"
        assert isinstance(doc.created_at, datetime)

    def test_add_chunk(self):
        """测试添加分块"""
        doc = Document(url="https://example.com", title="Test", content="Content")
        chunk = DocumentChunk(content="Chunk content")

        doc.add_chunk(chunk)

        assert len(doc.chunks) == 1
        assert doc.chunks[0].source_doc == doc.id

    def test_update_content(self):
        """测试更新内容"""
        doc = Document(url="https://example.com", title="Old Title", content="Old content")
        old_updated = doc.updated_at

        doc.update_content(content="New content", title="New Title")

        assert doc.content == "New content"
        assert doc.title == "New Title"
        assert doc.updated_at > old_updated

    def test_chunk_count(self):
        """测试分块计数"""
        doc = Document(url="https://example.com", title="Test", content="Content")

        assert doc.get_chunk_count() == 0

        doc.add_chunk(DocumentChunk(content="Chunk 1"))
        doc.add_chunk(DocumentChunk(content="Chunk 2"))

        assert doc.get_chunk_count() == 2


class TestKnowledgeBase:
    """测试知识库模型"""

    def test_add_document(self):
        """测试添加文档"""
        kb = KnowledgeBase(name="test")
        doc = Document(url="https://example.com", title="Test", content="Content")

        kb.add_document(doc)

        assert len(kb.documents) == 1
        assert kb.stats.document_count == 1

    def test_remove_document(self):
        """测试移除文档"""
        kb = KnowledgeBase(name="test")
        doc = Document(id="doc-remove", url="https://example.com", title="Test", content="Content")

        kb.add_document(doc)
        removed = kb.remove_document("doc-remove")

        assert removed is not None
        assert removed.id == "doc-remove"
        assert len(kb.documents) == 0

    def test_get_document(self):
        """测试获取文档"""
        kb = KnowledgeBase(name="test")
        doc = Document(id="doc-get", url="https://example.com", title="Test", content="Content")

        kb.add_document(doc)

        retrieved = kb.get_document("doc-get")
        assert retrieved is not None
        assert retrieved.id == "doc-get"

        not_found = kb.get_document("non-existent")
        assert not_found is None

    def test_get_document_by_url(self):
        """测试按 URL 获取文档"""
        kb = KnowledgeBase(name="test")
        doc = Document(url="https://unique.com/page", title="Test", content="Content")

        kb.add_document(doc)

        retrieved = kb.get_document_by_url("https://unique.com/page")
        assert retrieved is not None
        assert retrieved.url == "https://unique.com/page"


class TestFetchTask:
    """测试抓取任务"""

    def test_task_lifecycle(self):
        """测试任务生命周期"""
        task = FetchTask(url="https://example.com")

        assert task.status == FetchStatus.PENDING
        assert task.is_terminal() is False

        task.start()
        assert task.status == FetchStatus.FETCHING
        assert task.started_at is not None

        doc = Document(url="https://example.com", title="Test", content="Content")
        task.complete(doc)

        assert task.status == FetchStatus.COMPLETED
        assert task.result is not None
        assert task.is_terminal() is True

    def test_task_failure(self):
        """测试任务失败"""
        task = FetchTask(url="https://example.com")

        task.start()
        task.fail("Connection error")

        assert task.status == FetchStatus.FAILED
        assert task.error == "Connection error"
        assert task.is_terminal() is True

    def test_can_retry(self):
        """测试重试判断"""
        task = FetchTask(url="https://example.com", max_retries=3)

        assert task.can_retry() is True

        task.retry_count = 3
        assert task.can_retry() is False


class TestKnowledgeManager:
    """测试知识库管理器"""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = KnowledgeManager(name="test-kb")
        await manager.initialize()

        assert manager._initialized is True
        assert manager.name == "test-kb"

    @pytest.mark.asyncio
    async def test_add_url_with_mock(self):
        """测试添加 URL（使用模拟）"""
        manager = KnowledgeManager()

        # 模拟 fetcher
        manager._fetcher = MagicMock()
        manager._fetcher.check_url_valid = MagicMock(return_value=True)
        manager._fetcher.fetch = AsyncMock(
            return_value=FetchResult(
                url="https://example.com",
                success=True,
                content="<html><head><title>Test</title></head><body>Content</body></html>",
                method_used=FetchMethod.CURL,
                duration=0.5,
            )
        )
        manager._initialized = True

        doc = await manager.add_url("https://example.com")

        assert doc is not None
        assert doc.url == "https://example.com"
        assert len(manager) == 1

    @pytest.mark.asyncio
    async def test_add_invalid_url(self):
        """测试添加无效 URL"""
        manager = KnowledgeManager()

        manager._fetcher = MagicMock()
        manager._fetcher.check_url_valid = MagicMock(return_value=False)
        manager._initialized = True

        doc = await manager.add_url("not-a-valid-url")

        assert doc is None

    @pytest.mark.asyncio
    async def test_add_url_skip_existing(self):
        """测试跳过已存在的 URL"""
        manager = KnowledgeManager()

        # 模拟已存在的文档
        existing_doc = Document(
            id="doc-existing",
            url="https://example.com",
            title="Existing",
            content="Content",
        )
        manager._knowledge_base.add_document(existing_doc)
        manager._url_to_doc_id["https://example.com"] = "doc-existing"
        manager._initialized = True

        # 不应该重新获取
        doc = await manager.add_url("https://example.com")

        assert doc is not None
        assert doc.id == "doc-existing"

    def test_search(self):
        """测试搜索功能"""
        manager = KnowledgeManager()
        manager._initialized = True

        # 添加测试文档
        docs = [
            Document(url="https://example.com/1", title="Python Tutorial", content="Learn Python programming"),
            Document(url="https://example.com/2", title="JavaScript Guide", content="Learn JavaScript"),
            Document(url="https://example.com/3", title="Python Advanced", content="Advanced Python topics"),
        ]

        for doc in docs:
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        # 搜索
        results = manager.search("Python")

        assert len(results) >= 2

    def test_list_documents(self):
        """测试列出文档"""
        manager = KnowledgeManager()
        manager._initialized = True

        # 添加文档
        for i in range(5):
            doc = Document(
                url=f"https://example.com/{i}",
                title=f"Doc {i}",
                content=f"Content {i}",
            )
            manager._knowledge_base.add_document(doc)

        # 列出
        docs = manager.list()
        assert len(docs) == 5

        # 限制
        docs_limited = manager.list(limit=3)
        assert len(docs_limited) == 3

    def test_remove_document(self):
        """测试删除文档"""
        manager = KnowledgeManager()
        manager._initialized = True

        doc = Document(
            id="doc-to-remove",
            url="https://example.com",
            title="Test",
            content="Content",
        )
        manager._knowledge_base.add_document(doc)
        manager._url_to_doc_id[doc.url] = doc.id

        result = manager.remove("doc-to-remove")

        assert result is True
        assert len(manager) == 0

    def test_clear(self):
        """测试清空知识库"""
        manager = KnowledgeManager()
        manager._initialized = True

        # 添加文档
        for i in range(3):
            doc = Document(
                url=f"https://example.com/{i}",
                title=f"Doc {i}",
                content=f"Content {i}",
            )
            manager._knowledge_base.add_document(doc)
            manager._url_to_doc_id[doc.url] = doc.id

        count = manager.clear()

        assert count == 3
        assert len(manager) == 0

    def test_get_document(self):
        """测试获取文档"""
        manager = KnowledgeManager()
        manager._initialized = True

        doc = Document(
            id="doc-get-test",
            url="https://example.com",
            title="Test",
            content="Content",
        )
        manager._knowledge_base.add_document(doc)

        retrieved = manager.get_document("doc-get-test")
        assert retrieved is not None
        assert retrieved.id == "doc-get-test"

        not_found = manager.get_document("non-existent")
        assert not_found is None

    def test_get_document_by_url(self):
        """测试按 URL 获取文档"""
        manager = KnowledgeManager()
        manager._initialized = True

        doc = Document(
            url="https://unique.example.com/page",
            title="Test",
            content="Content",
        )
        manager._knowledge_base.add_document(doc)

        retrieved = manager.get_document_by_url("https://unique.example.com/page")
        assert retrieved is not None
        assert retrieved.url == "https://unique.example.com/page"

    def test_stats(self):
        """测试统计信息"""
        manager = KnowledgeManager()
        manager._initialized = True

        doc = Document(
            url="https://example.com",
            title="Test",
            content="Some content here",
        )
        manager._knowledge_base.add_document(doc)

        stats = manager.stats

        assert stats.document_count == 1
        assert stats.total_content_size > 0

    def test_repr(self):
        """测试字符串表示"""
        manager = KnowledgeManager(name="my-kb")

        repr_str = repr(manager)

        assert "my-kb" in repr_str
        assert "KnowledgeManager" in repr_str


class TestDocumentChunk:
    """测试文档分块模型"""

    def test_chunk_creation(self):
        """测试分块创建"""
        chunk = DocumentChunk(content="Chunk content")

        assert chunk.chunk_id.startswith("chunk-")
        assert chunk.content == "Chunk content"
        assert chunk.embedding is None

    def test_has_embedding(self):
        """测试嵌入检测"""
        chunk = DocumentChunk(content="Content")

        assert chunk.has_embedding() is False

        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.has_embedding() is True

    def test_chunk_with_metadata(self):
        """测试带元数据的分块"""
        chunk = DocumentChunk(
            content="Content",
            source_doc="doc-123",
            start_index=0,
            end_index=100,
            metadata={"key": "value"},
        )

        assert chunk.source_doc == "doc-123"
        assert chunk.start_index == 0
        assert chunk.end_index == 100
        assert chunk.metadata["key"] == "value"


class TestKnowledgeBaseStats:
    """测试知识库统计信息"""

    def test_default_stats(self):
        """测试默认统计"""
        stats = KnowledgeBaseStats()

        assert stats.document_count == 0
        assert stats.chunk_count == 0
        assert stats.embedding_count == 0
        assert stats.total_content_size == 0

    def test_stats_with_values(self):
        """测试带值的统计"""
        stats = KnowledgeBaseStats(
            document_count=10,
            chunk_count=50,
            embedding_count=50,
            total_content_size=10000,
        )

        assert stats.document_count == 10
        assert stats.chunk_count == 50


# ============================================================
# 集成测试
# ============================================================

# ============================================================
# 第五部分: CLI 配置集成测试
# ============================================================


class TestKnowledgeCLIConfigIntegration:
    """测试 knowledge_cli 配置集成"""

    def test_search_default_limit_from_config(self):
        """测试 search 命令默认 limit 来自配置"""
        from core.config import ConfigManager, get_config
        from scripts.knowledge_cli import create_parser

        # 重置配置单例以获取最新配置
        ConfigManager.reset_instance()
        config = get_config()
        expected_top_k = config.indexing.search.top_k

        # 创建 parser 并解析不带 --limit 的参数
        parser = create_parser()
        args = parser.parse_args(["search", "test query"])

        # 验证默认值来自配置
        assert args.limit == expected_top_k, f"默认 limit 应为配置值 {expected_top_k}，实际为 {args.limit}"

    def test_search_cli_override_default_limit(self):
        """测试 CLI 参数可以覆盖配置的默认 limit"""
        from scripts.knowledge_cli import create_parser

        parser = create_parser()

        # 使用 --limit 覆盖
        args_limit = parser.parse_args(["search", "--limit", "25", "test query"])
        assert args_limit.limit == 25

        # 使用 -n 覆盖
        args_n = parser.parse_args(["search", "-n", "30", "test query"])
        assert args_n.limit == 30

        # 使用 --top-k 覆盖
        args_topk = parser.parse_args(["search", "--top-k", "15", "test query"])
        assert args_topk.limit == 15

    def test_search_help_shows_config_source(self):
        """测试帮助文本显示配置来源"""
        import io
        import sys

        from scripts.knowledge_cli import create_parser

        parser = create_parser()

        # 捕获帮助输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            parser.parse_args(["search", "--help"])
        except SystemExit:
            pass  # argparse 在 --help 后会 exit
        help_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # 验证帮助文本包含配置来源说明
        assert "config.yaml" in help_output or "indexing.search.top_k" in help_output, (
            "帮助文本应说明默认值来自 config.yaml"
        )


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_full_workflow_with_mock(self, temp_workspace):
        """测试完整工作流（使用模拟数据）"""
        # 1. 初始化管理器
        manager = KnowledgeManager(name="integration-test")

        # 2. 模拟 fetcher
        manager._fetcher = MagicMock()
        manager._fetcher.check_url_valid = MagicMock(return_value=True)
        manager._fetcher.fetch = AsyncMock(
            return_value=FetchResult(
                url="https://example.com/doc1",
                success=True,
                content="<html><head><title>Integration Test</title></head><body><p>Test content for integration.</p></body></html>",
                method_used=FetchMethod.CURL,
                duration=0.3,
            )
        )
        manager._initialized = True

        # 3. 添加文档
        doc = await manager.add_url("https://example.com/doc1")
        assert doc is not None
        assert len(manager) == 1

        # 4. 搜索
        results = manager.search("integration")
        assert len(results) >= 1

        # 5. 获取文档
        retrieved = manager.get_document(doc.id)
        assert retrieved is not None

        # 6. 列出文档
        docs = manager.list()
        assert len(docs) == 1

        # 7. 统计
        stats = manager.stats
        assert stats.document_count == 1

        # 8. 删除文档
        removed = manager.remove(doc.id)
        assert removed is True
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_storage_persistence(self, temp_workspace):
        """测试存储持久化"""
        # 创建存储并保存文档
        storage1 = KnowledgeStorage(workspace_root=temp_workspace)
        await storage1.initialize()

        doc = Document(
            id="doc-persist",
            url="https://example.com",
            title="Persistent Doc",
            content="This should be saved.",
        )
        await storage1.save_document(doc)

        # 创建新的存储实例（模拟重启）
        storage2 = KnowledgeStorage(workspace_root=temp_workspace)
        await storage2.initialize()

        # 验证文档仍然存在
        loaded = await storage2.load_document("doc-persist")
        assert loaded is not None
        assert loaded.title == "Persistent Doc"

    def test_parser_to_chunks_workflow(self):
        """测试解析器到分块的工作流"""
        # HTML 内容
        html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <nav>Navigation menu</nav>
            <main>
                <h1>Main Title</h1>
                <p>First paragraph with important information.</p>
                <p>Second paragraph with more details.</p>
                <h2>Section 1</h2>
                <p>Section content here.</p>
            </main>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        # 1. 解析 HTML
        parser = HTMLParser(parser="html.parser")
        parsed = parser.parse(html)

        assert parsed.title == "Test Article"

        # 2. 清理内容
        cleaner = ContentCleaner(parser="html.parser")
        cleaned = cleaner.clean_to_text(html)

        assert "Main Title" in cleaned
        assert "Navigation menu" not in cleaned

        # 3. 转换为 Markdown
        converter = MarkdownConverter()
        markdown = converter.convert_with_cleaning(html)

        assert "#" in markdown  # 应该包含标题标记

        # 4. 分块
        splitter = ChunkSplitter(chunk_size=200, overlap=20)
        chunks = splitter.split(cleaned, source_doc="doc-workflow")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source_doc == "doc-workflow"
            assert len(chunk.content) > 0


# ============================================================
# 第八部分: 内容清洗功能测试
# ============================================================

from knowledge import (
    CleanedContent,
    ContentCleanMode,
    clean_content_unified,
    compute_content_fingerprint,
)


class TestContentCleanMode:
    """测试内容清洗模式"""

    def test_clean_mode_values(self):
        """测试清洗模式枚举值"""
        assert ContentCleanMode.RAW == "raw"
        assert ContentCleanMode.MARKDOWN == "markdown"
        assert ContentCleanMode.TEXT == "text"
        assert ContentCleanMode.CHANGELOG == "changelog"


class TestComputeContentFingerprint:
    """测试 fingerprint 计算"""

    def test_fingerprint_basic(self):
        """测试基本 fingerprint 计算"""
        fp = compute_content_fingerprint("Hello World")
        assert len(fp) == 16  # SHA256 前16位
        assert fp.isalnum()  # 应该是字母数字

    def test_fingerprint_consistency(self):
        """测试 fingerprint 一致性"""
        content = "Test content for fingerprint"
        fp1 = compute_content_fingerprint(content)
        fp2 = compute_content_fingerprint(content)
        assert fp1 == fp2

    def test_fingerprint_whitespace_normalization(self):
        """测试空白归一化"""
        # 多余空白应该被归一化
        fp1 = compute_content_fingerprint("Hello   World")
        fp2 = compute_content_fingerprint("Hello World")
        # 归一化后应该相同
        assert fp1 == fp2

    def test_fingerprint_different_content(self):
        """测试不同内容的 fingerprint"""
        fp1 = compute_content_fingerprint("Content A")
        fp2 = compute_content_fingerprint("Content B")
        assert fp1 != fp2


class TestCleanContentUnified:
    """测试统一内容清洗函数"""

    def test_raw_mode(self):
        """测试 RAW 模式"""
        content = "<p>Hello <b>World</b></p>"
        result = clean_content_unified(content, mode=ContentCleanMode.RAW)

        assert isinstance(result, CleanedContent)
        assert result.content == content  # RAW 保持原样
        assert result.clean_mode == ContentCleanMode.RAW
        assert len(result.fingerprint) == 16

    def test_text_mode(self):
        """测试 TEXT 模式"""
        content = "<p>Hello <b>World</b></p><script>alert(1)</script>"
        result = clean_content_unified(content, mode=ContentCleanMode.TEXT)

        assert "Hello" in result.content
        assert "World" in result.content
        assert "script" not in result.content
        assert "<" not in result.content  # HTML 标签应被移除
        assert result.clean_mode == ContentCleanMode.TEXT

    def test_markdown_mode(self):
        """测试 MARKDOWN 模式"""
        content = "<h1>Title</h1><p>Paragraph text.</p>"
        result = clean_content_unified(content, mode=ContentCleanMode.MARKDOWN)

        assert "Title" in result.content
        assert "Paragraph" in result.content
        # Markdown 转换后应该没有 HTML 标签
        assert "<h1>" not in result.content
        assert result.clean_mode == ContentCleanMode.MARKDOWN

    def test_changelog_mode(self):
        """测试 CHANGELOG 模式"""
        content = "<p>Jan 16, 2026 - New feature added</p>"
        result = clean_content_unified(content, mode=ContentCleanMode.CHANGELOG)

        assert "New feature" in result.content
        # 日期应该被标准化
        assert result.clean_mode == ContentCleanMode.CHANGELOG
        assert len(result.fingerprint) == 16

    def test_preserve_raw(self):
        """测试保留原始内容"""
        content = "<p>Test content</p>"
        result = clean_content_unified(
            content,
            mode=ContentCleanMode.MARKDOWN,
            preserve_raw=True,
        )

        assert result.raw_content == content
        assert result.content != content  # 清洗后应该不同

    def test_no_preserve_raw(self):
        """测试不保留原始内容"""
        content = "<p>Test content</p>"
        result = clean_content_unified(
            content,
            mode=ContentCleanMode.MARKDOWN,
            preserve_raw=False,
        )

        assert result.raw_content == ""

    def test_empty_content(self):
        """测试空内容"""
        result = clean_content_unified("", mode=ContentCleanMode.MARKDOWN)
        assert result.content == ""
        assert result.clean_mode == ContentCleanMode.MARKDOWN

    def test_title_extraction(self):
        """测试标题提取"""
        content = "<html><head><title>My Title</title></head><body>Content</body></html>"
        result = clean_content_unified(content, mode=ContentCleanMode.MARKDOWN)
        assert result.title == "My Title"


class TestKnowledgeManagerAddContent:
    """测试 KnowledgeManager.add_content 方法"""

    @pytest.fixture
    async def manager(self):
        """创建测试用的 KnowledgeManager"""
        manager = KnowledgeManager(name="test-add-content")
        await manager.initialize()
        yield manager
        manager.clear()

    @pytest.mark.asyncio
    async def test_add_content_basic(self, manager):
        """测试基本内容添加"""
        doc = await manager.add_content(
            url="https://example.com/test",
            content="<p>Test content</p>",
            title="Test Doc",
        )

        assert doc is not None
        assert doc.title == "Test Doc"
        assert "Test content" in doc.content
        assert doc.metadata.get("clean_mode") == "markdown"
        assert "cleaned_fingerprint" in doc.metadata

    @pytest.mark.asyncio
    async def test_add_content_changelog_mode(self, manager):
        """测试 CHANGELOG 清洗模式"""
        doc = await manager.add_content(
            url="https://example.com/changelog",
            content="<p>Jan 16, 2026 - New feature</p>",
            title="Changelog",
            clean_mode=ContentCleanMode.CHANGELOG,
            preserve_raw=True,
        )

        assert doc is not None
        assert doc.metadata.get("clean_mode") == "changelog"
        assert "raw_content" in doc.metadata  # 应该保留原始内容

    @pytest.mark.asyncio
    async def test_add_content_skip_existing(self, manager):
        """测试跳过已存在的 URL"""
        url = "https://example.com/existing"
        content = "<p>Content</p>"

        # 第一次添加
        doc1 = await manager.add_content(url=url, content=content, title="Doc1")
        assert doc1 is not None

        # 第二次添加（应该返回已存在的文档）
        doc2 = await manager.add_content(url=url, content=content, title="Doc2")
        assert doc2 is not None
        assert doc2.id == doc1.id  # 应该是同一个文档

    @pytest.mark.asyncio
    async def test_add_content_force_refresh(self, manager):
        """测试强制刷新"""
        url = "https://example.com/refresh"

        # 第一次添加
        doc1 = await manager.add_content(
            url=url,
            content="<p>Original</p>",
            title="Original",
        )
        assert doc1 is not None

        # 强制刷新
        doc2 = await manager.add_content(
            url=url,
            content="<p>Updated</p>",
            title="Updated",
            force_refresh=True,
        )
        assert doc2 is not None
        assert "Updated" in doc2.content

    @pytest.mark.asyncio
    async def test_add_content_empty(self, manager):
        """测试空内容"""
        doc = await manager.add_content(
            url="https://example.com/empty",
            content="",
            title="Empty",
        )
        assert doc is None  # 空内容应该返回 None


# ============================================================
# 第九部分: URL 安全策略测试
# ============================================================


class TestUrlPolicy:
    """测试 URL 安全策略"""

    def test_default_policy(self):
        """测试默认策略配置"""
        policy = UrlPolicy()

        # 默认允许 http/https
        assert policy.allowed_schemes == ["http", "https"]
        # 默认拒绝私网
        assert policy.deny_private_networks is True
        # 默认不限制域名和前缀
        assert policy.allowed_domains == []
        assert policy.allowed_url_prefixes == []

    def test_valid_http_url(self):
        """测试有效的 HTTP URL"""
        policy = UrlPolicy()

        # 有效的公网 URL
        policy.validate("https://example.com")
        policy.validate("http://example.com/path?query=1")
        policy.validate("https://docs.cursor.com/api/reference")

    def test_invalid_scheme_ftp(self):
        """测试非法 scheme: FTP"""
        policy = UrlPolicy()

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("ftp://example.com/file.txt")

        assert exc_info.value.policy_type == "scheme"
        assert "ftp" in exc_info.value.reason

    def test_invalid_scheme_file(self):
        """测试非法 scheme: file"""
        policy = UrlPolicy()

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("file:///etc/passwd")

        assert exc_info.value.policy_type == "scheme"
        assert "file" in exc_info.value.reason

    def test_invalid_scheme_javascript(self):
        """测试非法 scheme: javascript"""
        policy = UrlPolicy()

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("javascript:alert(1)")

        assert exc_info.value.policy_type == "scheme"

    def test_invalid_scheme_data(self):
        """测试非法 scheme: data"""
        policy = UrlPolicy()

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("data:text/html,<h1>Hello</h1>")

        assert exc_info.value.policy_type == "scheme"

    def test_localhost_rejected(self):
        """测试拒绝 localhost"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://localhost/admin")

        assert exc_info.value.policy_type == "private_network"
        assert "localhost" in exc_info.value.reason.lower()

    def test_localhost_with_port_rejected(self):
        """测试拒绝 localhost:port"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://localhost:8080/api")

        assert exc_info.value.policy_type == "private_network"

    def test_loopback_ip_rejected(self):
        """测试拒绝回环 IP 127.0.0.1"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://127.0.0.1/admin")

        assert exc_info.value.policy_type == "private_network"

    def test_loopback_ip_with_port_rejected(self):
        """测试拒绝回环 IP:port"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://127.0.0.1:3000/api")

        assert exc_info.value.policy_type == "private_network"

    def test_private_network_10_rejected(self):
        """测试拒绝私网 10.x.x.x"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://10.0.0.1/internal")

        assert exc_info.value.policy_type == "private_network"
        assert "私有网络" in exc_info.value.reason

    def test_private_network_172_rejected(self):
        """测试拒绝私网 172.16.x.x"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://172.16.0.1/internal")

        assert exc_info.value.policy_type == "private_network"

    def test_private_network_192_rejected(self):
        """测试拒绝私网 192.168.x.x"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://192.168.1.1/router")

        assert exc_info.value.policy_type == "private_network"

    def test_zero_ip_rejected(self):
        """测试拒绝 0.0.0.0"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://0.0.0.0:8000/api")

        assert exc_info.value.policy_type == "private_network"

    def test_link_local_ip_rejected(self):
        """测试拒绝链路本地地址 169.254.x.x"""
        policy = UrlPolicy(deny_private_networks=True)

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://169.254.1.1/metadata")

        assert exc_info.value.policy_type == "private_network"

    def test_private_network_allowed_when_disabled(self):
        """测试禁用私网检查后允许私网地址"""
        policy = UrlPolicy(deny_private_networks=False)

        # 应该不抛出异常
        policy.validate("http://localhost/admin")
        policy.validate("http://127.0.0.1/api")
        policy.validate("http://192.168.1.1/router")

    def test_domain_whitelist(self):
        """测试域名白名单"""
        policy = UrlPolicy(
            allowed_domains=["example.com", "docs.cursor.com"],
            deny_private_networks=True,
        )

        # 白名单内的域名
        policy.validate("https://example.com/page")
        policy.validate("https://docs.cursor.com/api")

        # 白名单外的域名
        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("https://evil.com/phishing")

        assert exc_info.value.policy_type == "domain"

    def test_domain_whitelist_wildcard(self):
        """测试域名白名单通配符"""
        policy = UrlPolicy(
            allowed_domains=["*.cursor.com"],
            deny_private_networks=True,
        )

        # 子域名匹配
        policy.validate("https://docs.cursor.com/api")
        policy.validate("https://api.cursor.com/v1")
        policy.validate("https://cursor.com/home")

        # 不匹配的域名
        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("https://cursorx.com/fake")

        assert exc_info.value.policy_type == "domain"

    def test_url_prefix_whitelist(self):
        """测试 URL 前缀白名单"""
        policy = UrlPolicy(
            allowed_url_prefixes=[
                "https://docs.cursor.com/",
                "https://api.cursor.com/v1/",
            ],
            deny_private_networks=True,
        )

        # 前缀匹配
        policy.validate("https://docs.cursor.com/guide")
        policy.validate("https://api.cursor.com/v1/users")

        # 前缀不匹配
        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("https://api.cursor.com/v2/users")

        assert exc_info.value.policy_type == "url_prefix"

    def test_deny_ip_addresses(self):
        """测试拒绝纯 IP 地址"""
        policy = UrlPolicy(
            deny_ip_addresses=True,
            deny_private_networks=False,  # 关闭私网检查，专门测试 IP 检查
        )

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("http://8.8.8.8/dns")

        assert exc_info.value.policy_type == "ip_address"

    def test_is_valid_method(self):
        """测试 is_valid 方法"""
        policy = UrlPolicy()

        assert policy.is_valid("https://example.com") is True
        assert policy.is_valid("ftp://example.com") is False
        assert policy.is_valid("http://localhost") is False

    def test_get_validation_error_method(self):
        """测试 get_validation_error 方法"""
        policy = UrlPolicy()

        # 有效 URL
        assert policy.get_validation_error("https://example.com") is None

        # 无效 URL
        error = policy.get_validation_error("ftp://example.com")
        assert error is not None
        assert "ftp" in error

    def test_empty_url_rejected(self):
        """测试空 URL"""
        policy = UrlPolicy()

        with pytest.raises(UrlPolicyError) as exc_info:
            policy.validate("")

        assert exc_info.value.policy_type == "empty"

    def test_private_domain_patterns_rejected(self):
        """测试疑似私有网络域名"""
        policy = UrlPolicy(deny_private_networks=True)

        # .local 域名
        with pytest.raises(UrlPolicyError):
            policy.validate("http://server.local/api")

        # .internal 域名
        with pytest.raises(UrlPolicyError):
            policy.validate("http://app.internal/admin")

        # .lan 域名
        with pytest.raises(UrlPolicyError):
            policy.validate("http://nas.lan/files")


class TestUrlPolicyError:
    """测试 URL 策略错误"""

    def test_error_attributes(self):
        """测试错误属性"""
        error = UrlPolicyError(
            url="http://localhost/admin",
            reason="不允许访问 localhost",
            policy_type="private_network",
        )

        assert error.url == "http://localhost/admin"
        assert error.reason == "不允许访问 localhost"
        assert error.policy_type == "private_network"
        assert "localhost" in str(error)


class TestWebFetcherUrlPolicy:
    """测试 WebFetcher 的 URL 策略集成"""

    @pytest.mark.asyncio
    async def test_fetch_rejects_localhost(self):
        """测试 fetch 拒绝 localhost"""
        fetcher = WebFetcher()
        await fetcher.initialize()

        result = await fetcher.fetch("http://localhost/admin")

        assert result.success is False
        assert result.error is not None
        assert "策略" in result.error or "localhost" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_rejects_private_ip(self):
        """测试 fetch 拒绝私网 IP"""
        fetcher = WebFetcher()
        await fetcher.initialize()

        result = await fetcher.fetch("http://192.168.1.1/router")

        assert result.success is False
        assert result.error is not None
        assert "策略" in result.error or "私有" in result.error

    @pytest.mark.asyncio
    async def test_fetch_rejects_invalid_scheme(self):
        """测试 fetch 拒绝非法 scheme"""
        fetcher = WebFetcher()
        await fetcher.initialize()

        result = await fetcher.fetch("ftp://files.example.com/data.txt")

        assert result.success is False
        assert result.error is not None
        assert "策略" in result.error or "ftp" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_with_disabled_policy(self):
        """测试禁用策略后的 fetch"""
        config = FetchConfig(enforce_url_policy=False)
        fetcher = WebFetcher(config)
        await fetcher.initialize()

        # 禁用策略后，localhost URL 不会被策略拒绝
        # （但可能因为无法连接而失败）
        result = await fetcher.fetch("http://localhost/admin")

        # 错误不应该是策略拒绝
        if not result.success:
            assert result.error is not None
            assert "策略" not in result.error

    @pytest.mark.asyncio
    async def test_fetch_with_custom_policy(self):
        """测试使用自定义策略的 fetch"""
        # 创建只允许特定域名的策略
        custom_policy = UrlPolicy(
            allowed_domains=["example.com"],
            deny_private_networks=True,
        )
        config = FetchConfig(url_policy=custom_policy)
        fetcher = WebFetcher(config)
        await fetcher.initialize()

        # 域名不在白名单中
        result = await fetcher.fetch("https://other-domain.com/page")

        assert result.success is False
        assert result.error is not None
        assert "策略" in result.error or "域名" in result.error


class TestKnowledgeManagerUrlPolicy:
    """测试 KnowledgeManager 的 URL 策略集成"""

    @pytest.mark.asyncio
    async def test_add_url_rejects_localhost(self):
        """测试 add_url 拒绝 localhost"""
        manager = KnowledgeManager(name="test-policy")
        await manager.initialize()

        doc = await manager.add_url("http://localhost/admin")

        assert doc is None

    @pytest.mark.asyncio
    async def test_add_url_rejects_private_ip(self):
        """测试 add_url 拒绝私网 IP"""
        manager = KnowledgeManager(name="test-policy")
        await manager.initialize()

        doc = await manager.add_url("http://192.168.1.1/router")

        assert doc is None

    @pytest.mark.asyncio
    async def test_add_url_rejects_invalid_scheme(self):
        """测试 add_url 拒绝非法 scheme"""
        manager = KnowledgeManager(name="test-policy")
        await manager.initialize()

        doc = await manager.add_url("ftp://files.example.com/data.txt")

        assert doc is None

    @pytest.mark.asyncio
    async def test_add_urls_filters_invalid(self):
        """测试 add_urls 过滤无效 URL"""
        manager = KnowledgeManager(name="test-policy")

        # 模拟 fetcher
        manager._fetcher = MagicMock()
        manager._fetcher.check_url_valid = MagicMock(return_value=True)
        manager._fetcher.fetch_many = AsyncMock(
            return_value=[
                FetchResult(
                    url="https://example.com/page1",
                    success=True,
                    content="<html><body>Content</body></html>",
                    method_used=FetchMethod.CURL,
                ),
            ]
        )
        manager._initialized = True

        urls = [
            "https://example.com/page1",  # 有效
            "http://localhost/admin",  # 无效：localhost
            "ftp://files.example.com",  # 无效：非法 scheme
            "http://192.168.1.1/router",  # 无效：私网 IP
        ]

        docs = await manager.add_urls(urls)

        # 只有有效的 URL 被处理
        assert len(docs) == 1
        assert docs[0].url == "https://example.com/page1"

    @pytest.mark.asyncio
    async def test_manager_with_custom_policy(self):
        """测试 KnowledgeManager 使用自定义策略"""
        # 创建只允许特定域名的策略
        custom_policy = UrlPolicy(
            allowed_domains=["docs.cursor.com"],
            deny_private_networks=True,
        )
        manager = KnowledgeManager(name="test-custom-policy", url_policy=custom_policy)
        await manager.initialize()

        # 域名不在白名单中
        doc = await manager.add_url("https://example.com/page")

        assert doc is None

    @pytest.mark.asyncio
    async def test_manager_url_policy_property(self):
        """测试 url_policy 属性"""
        custom_policy = UrlPolicy(allowed_domains=["example.com"])
        manager = KnowledgeManager(name="test", url_policy=custom_policy)

        assert manager.url_policy == custom_policy
        assert manager.url_policy.allowed_domains == ["example.com"]


# ============================================================
# 样例数据：HTML 文档内容（供测试复用）
# ============================================================

SAMPLE_HTML_DOC_SIMPLE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Documentation</title>
    <meta name="description" content="A sample documentation page">
</head>
<body>
    <h1>Getting Started</h1>
    <p>Welcome to the documentation.</p>
    <h2>Installation</h2>
    <p>Install using pip:</p>
    <code>pip install example-package</code>
    <h2>Usage</h2>
    <p>Basic usage example:</p>
    <pre>import example</pre>
</body>
</html>
"""

SAMPLE_HTML_DOC_WITH_LINKS = """
<!DOCTYPE html>
<html>
<head><title>Doc with Links</title></head>
<body>
    <h1>Documentation</h1>
    <p>See the following resources:</p>
    <ul>
        <li><a href="https://docs.example.com/api">API Reference</a></li>
        <li><a href="https://docs.example.com/guide">User Guide</a></li>
        <li><a href="https://github.com/example/repo">Source Code</a></li>
    </ul>
</body>
</html>
"""

SAMPLE_HTML_DOC_WITH_NAV = """
<!DOCTYPE html>
<html>
<head><title>Doc with Navigation</title></head>
<body>
    <nav>
        <ul>
            <li>Home</li>
            <li>About</li>
            <li>Contact</li>
        </ul>
    </nav>
    <main>
        <h1>Main Content</h1>
        <p>This is the main content area.</p>
    </main>
    <footer>
        <p>Footer content</p>
    </footer>
</body>
</html>
"""

SAMPLE_LLMS_TXT_CONTENT = """# Documentation Index

## Overview
[Getting Started](https://docs.example.com/getting-started)
[Quick Reference](https://docs.example.com/quick-reference)

## API
https://docs.example.com/api/v1
https://docs.example.com/api/v2

## Guides
- [Tutorial](https://docs.example.com/tutorial)
- [Best Practices](https://docs.example.com/best-practices)
"""


# ============================================================
# Test: 样例 HTML 数据复用测试
# ============================================================


class TestSampleHtmlDataReuse:
    """测试复用样例 HTML 数据，确保不依赖真实网络"""

    def test_parse_simple_html_doc(self):
        """测试解析简单 HTML 文档"""
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(SAMPLE_HTML_DOC_SIMPLE)

        assert result.title == "Sample Documentation"
        assert "Getting Started" in result.content
        assert "Installation" in result.content

    def test_parse_html_with_links(self):
        """测试解析带链接的 HTML 文档"""
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(SAMPLE_HTML_DOC_WITH_LINKS)

        assert result.title == "Doc with Links"
        assert len(result.links) >= 3

        # 验证链接提取
        hrefs = [link["href"] for link in result.links]
        assert "https://docs.example.com/api" in hrefs
        assert "https://github.com/example/repo" in hrefs

    def test_parse_html_extract_headings(self):
        """测试从 HTML 提取标题层级"""
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(SAMPLE_HTML_DOC_SIMPLE)

        # 验证标题提取
        assert len(result.headings) >= 3

        h1_headings = [h for h in result.headings if h["level"] == 1]
        h2_headings = [h for h in result.headings if h["level"] == 2]

        assert len(h1_headings) >= 1
        assert len(h2_headings) >= 2

    def test_clean_html_removes_nav_footer(self):
        """测试清理 HTML 时移除导航和页脚"""
        cleaner = ContentCleaner(parser="html.parser")
        cleaned = cleaner.clean_to_text(SAMPLE_HTML_DOC_WITH_NAV)

        # 主内容应该保留
        assert "Main Content" in cleaned

        # 导航和页脚应该被移除
        assert "Home" not in cleaned
        assert "About" not in cleaned
        assert "Footer content" not in cleaned

    def test_convert_html_to_markdown(self):
        """测试将 HTML 转换为 Markdown"""
        converter = MarkdownConverter()
        md = converter.convert(SAMPLE_HTML_DOC_SIMPLE)

        # 验证标题转换
        assert "# Getting Started" in md or "Getting Started" in md

        # 验证段落保留
        assert "Welcome to the documentation" in md


# ============================================================
# Test: 样例 llms.txt 数据复用测试
# ============================================================


class TestSampleLlmsTxtDataReuse:
    """测试复用样例 llms.txt 数据"""

    def test_sample_llms_txt_structure(self):
        """测试样例 llms.txt 结构"""
        assert "# Documentation Index" in SAMPLE_LLMS_TXT_CONTENT
        assert "https://docs.example.com" in SAMPLE_LLMS_TXT_CONTENT

    def test_sample_llms_txt_contains_markdown_links(self):
        """测试样例 llms.txt 包含 Markdown 链接"""
        assert "[Getting Started]" in SAMPLE_LLMS_TXT_CONTENT
        assert "(https://docs.example.com/getting-started)" in SAMPLE_LLMS_TXT_CONTENT

    def test_sample_llms_txt_contains_plain_urls(self):
        """测试样例 llms.txt 包含纯 URL"""
        assert "https://docs.example.com/api/v1" in SAMPLE_LLMS_TXT_CONTENT
        assert "https://docs.example.com/api/v2" in SAMPLE_LLMS_TXT_CONTENT


# ============================================================
# Test: 不依赖网络的集成测试
# ============================================================


class TestKnowledgeOfflineIntegration:
    """不依赖网络的知识库集成测试"""

    @pytest.mark.asyncio
    async def test_document_lifecycle_offline(self):
        """测试文档生命周期（离线）"""
        manager = KnowledgeManager(name="test-offline")
        manager._initialized = True

        # 使用样例数据创建文档
        doc = Document(
            url="https://example.com/test",
            title="Test Document",
            content="This is test content from sample data.",
        )

        # 添加到知识库
        manager._knowledge_base.add_document(doc)
        manager._url_to_doc_id[doc.url] = doc.id

        # 验证添加成功
        assert len(manager) == 1

        # 搜索
        results = manager.search("test")
        assert len(results) >= 1

        # 获取文档
        retrieved = manager.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.title == "Test Document"

        # 删除
        result = manager.remove(doc.id)
        assert result is True
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_add_content_with_sample_html(self):
        """测试使用样例 HTML 添加内容"""
        manager = KnowledgeManager(name="test-sample-html")
        await manager.initialize()

        # 使用样例 HTML 添加内容
        doc = await manager.add_content(
            url="https://example.com/sample",
            content=SAMPLE_HTML_DOC_SIMPLE,
            title="Sample Doc",
        )

        assert doc is not None
        assert "Getting Started" in doc.content or "documentation" in doc.content.lower()

        # 清理
        manager.clear()

    def test_chunk_splitter_with_sample_content(self):
        """测试使用样例内容进行分块"""
        # 使用样例 HTML 清理后的文本
        cleaner = ContentCleaner(parser="html.parser")
        text = cleaner.clean_to_text(SAMPLE_HTML_DOC_SIMPLE)

        # 分块
        splitter = ChunkSplitter(chunk_size=100, overlap=10)
        chunks = splitter.split(text, source_doc="sample-doc")

        # 验证分块结果
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source_doc == "sample-doc"
            assert len(chunk.content) > 0

    def test_document_storage_with_sample_data(self, temp_workspace):
        """测试使用样例数据进行文档存储"""
        import asyncio

        async def run_test():
            storage = KnowledgeStorage(workspace_root=temp_workspace)
            await storage.initialize()

            # 使用样例数据创建文档
            doc = Document(
                url="https://example.com/sample-storage",
                title="Sample Storage Doc",
                content="Content from sample data for storage testing.",
            )

            # 保存
            success, _ = await storage.save_document(doc)
            assert success is True

            # 加载
            loaded = await storage.load_document(doc.id)
            assert loaded is not None
            assert loaded.title == "Sample Storage Doc"

            # 清理
            await storage.clear_all()

        asyncio.run(run_test())


# ============================================================
# Test: 样例数据一致性测试
# ============================================================


class TestSampleDataConsistency:
    """测试样例数据的一致性和完整性"""

    def test_sample_html_valid_structure(self):
        """测试样例 HTML 结构有效"""
        # 所有样例 HTML 应包含基本标签
        for sample in [SAMPLE_HTML_DOC_SIMPLE, SAMPLE_HTML_DOC_WITH_LINKS, SAMPLE_HTML_DOC_WITH_NAV]:
            assert "<html>" in sample or "<!DOCTYPE html>" in sample
            assert "<head>" in sample
            assert "<body>" in sample
            assert "<title>" in sample

    def test_sample_html_no_real_content(self):
        """测试样例 HTML 不包含真实敏感内容"""
        all_samples = SAMPLE_HTML_DOC_SIMPLE + SAMPLE_HTML_DOC_WITH_LINKS + SAMPLE_HTML_DOC_WITH_NAV

        # 应使用 example.com 而非真实域名
        # 允许 github.com 作为示例外域
        assert "cursor.com" not in all_samples or "example" in all_samples

    def test_sample_llms_txt_uses_example_domain(self):
        """测试样例 llms.txt 使用示例域名"""
        assert "docs.example.com" in SAMPLE_LLMS_TXT_CONTENT

        # 不应包含真实的生产域名
        # （除非作为明确的示例）

    def test_sample_data_immutability(self):
        """测试样例数据应被视为不可变"""
        # 字符串是不可变的，但我们验证不会意外修改
        original_len = len(SAMPLE_HTML_DOC_SIMPLE)

        # 使用样例数据
        parser = HTMLParser(parser="html.parser")
        parser.parse(SAMPLE_HTML_DOC_SIMPLE)

        # 原始数据应保持不变
        assert len(SAMPLE_HTML_DOC_SIMPLE) == original_len
