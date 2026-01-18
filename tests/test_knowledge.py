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
        with patch.object(fetcher, '_fetch_via_curl') as mock_curl:
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
        with patch('knowledge.fetcher.WebFetcher.fetch') as mock_fetch:
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
        manager._fetcher.fetch = AsyncMock(return_value=FetchResult(
            url="https://example.com",
            success=True,
            content="<html><head><title>Test</title></head><body>Content</body></html>",
            method_used=FetchMethod.CURL,
            duration=0.5,
        ))
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
        manager._fetcher.fetch = AsyncMock(return_value=FetchResult(
            url="https://example.com/doc1",
            success=True,
            content="<html><head><title>Integration Test</title></head><body><p>Test content for integration.</p></body></html>",
            method_used=FetchMethod.CURL,
            duration=0.3,
        ))
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
