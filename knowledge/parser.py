"""HTML 解析与内容处理模块

提供 HTML 解析、内容清理、Markdown 转换和文档分块功能
"""
import re
from typing import Any, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag, NavigableString
import html2text

from .models import DocumentChunk


@dataclass
class ParsedContent:
    """解析后的内容结构"""
    title: str = ""
    content: str = ""
    headings: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HTMLParser:
    """HTML 解析器
    
    解析 HTML 文档，提取结构化内容
    
    示例:
        parser = HTMLParser()
        result = parser.parse("<html><head><title>Test</title></head><body><p>Hello</p></body></html>")
        print(result.title)  # "Test"
        print(result.content)  # "<p>Hello</p>"
    """
    
    def __init__(self, parser: str = "lxml"):
        """初始化解析器
        
        Args:
            parser: BeautifulSoup 解析器类型 (lxml, html.parser, html5lib)
        """
        self.parser = parser
    
    def parse(self, html: str) -> ParsedContent:
        """解析 HTML 内容
        
        Args:
            html: 原始 HTML 字符串
            
        Returns:
            ParsedContent: 解析后的结构化内容
        """
        soup = BeautifulSoup(html, self.parser)
        
        result = ParsedContent()
        
        # 提取标题
        result.title = self._extract_title(soup)
        
        # 提取主体内容
        result.content = self._extract_body(soup)
        
        # 提取标题层级
        result.headings = self._extract_headings(soup)
        
        # 提取链接
        result.links = self._extract_links(soup)
        
        # 提取图片
        result.images = self._extract_images(soup)
        
        # 提取元数据
        result.metadata = self._extract_metadata(soup)
        
        return result
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取文档标题"""
        # 优先使用 <title> 标签
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # 其次使用 og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()
        
        # 最后使用 h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return ""
    
    def _extract_body(self, soup: BeautifulSoup) -> str:
        """提取主体内容 HTML"""
        body = soup.find("body")
        if body:
            return str(body)
        return str(soup)
    
    def _extract_headings(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """提取标题层级结构"""
        headings = []
        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                headings.append({
                    "level": level,
                    "text": tag.get_text(strip=True),
                    "id": tag.get("id", "")
                })
        return headings
    
    def _extract_links(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """提取所有链接"""
        links = []
        for a_tag in soup.find_all("a", href=True):
            links.append({
                "href": str(a_tag["href"]),
                "text": a_tag.get_text(strip=True),
                "title": a_tag.get("title", "")
            })
        return links
    
    def _extract_images(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """提取所有图片"""
        images = []
        for img_tag in soup.find_all("img"):
            images.append({
                "src": img_tag.get("src", ""),
                "alt": img_tag.get("alt", ""),
                "title": img_tag.get("title", "")
            })
        return images
    
    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, Any]:
        """提取页面元数据"""
        metadata: dict[str, Any] = {}
        
        # 提取 meta 标签
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[str(name)] = content
        
        # 提取 lang 属性
        html_tag = soup.find("html")
        if html_tag and isinstance(html_tag, Tag):
            lang = html_tag.get("lang")
            if lang:
                metadata["language"] = lang
        
        return metadata


class ContentCleaner:
    """内容清理器
    
    清理 HTML 中的无关内容，如 script、style、导航等
    
    示例:
        cleaner = ContentCleaner()
        clean_html = cleaner.clean("<div><script>alert(1)</script><p>Hello</p></div>")
        # 结果: "<div><p>Hello</p></div>"
    """
    
    # 默认要移除的标签
    DEFAULT_REMOVE_TAGS = [
        "script", "style", "noscript", "iframe", "frame",
        "nav", "header", "footer", "aside",
        "form", "button", "input", "select", "textarea",
        "svg", "canvas", "video", "audio", "embed", "object"
    ]
    
    # 默认要移除的 class 模式
    DEFAULT_REMOVE_CLASSES = [
        r"nav(igation)?",
        r"menu",
        r"sidebar",
        r"footer",
        r"header",
        r"ad(vert(isement)?)?",
        r"banner",
        r"popup",
        r"modal",
        r"cookie",
        r"social",
        r"share",
        r"comment",
        r"related"
    ]
    
    # 默认要移除的 id 模式
    DEFAULT_REMOVE_IDS = [
        r"nav(igation)?",
        r"menu",
        r"sidebar",
        r"footer",
        r"header",
        r"ad(vert)?",
        r"banner"
    ]
    
    def __init__(
        self,
        remove_tags: Optional[list[str]] = None,
        remove_classes: Optional[list[str]] = None,
        remove_ids: Optional[list[str]] = None,
        parser: str = "lxml"
    ):
        """初始化清理器
        
        Args:
            remove_tags: 要移除的标签列表
            remove_classes: 要移除的 class 模式（正则表达式）
            remove_ids: 要移除的 id 模式（正则表达式）
            parser: BeautifulSoup 解析器类型
        """
        self.remove_tags = remove_tags or self.DEFAULT_REMOVE_TAGS
        self.remove_classes = remove_classes or self.DEFAULT_REMOVE_CLASSES
        self.remove_ids = remove_ids or self.DEFAULT_REMOVE_IDS
        self.parser = parser
        
        # 编译正则表达式
        self._class_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.remove_classes
        ]
        self._id_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.remove_ids
        ]
    
    def clean(self, html: str) -> str:
        """清理 HTML 内容
        
        Args:
            html: 原始 HTML 字符串
            
        Returns:
            清理后的 HTML 字符串
        """
        soup = BeautifulSoup(html, self.parser)
        
        # 移除指定标签
        self._remove_tags(soup)
        
        # 移除匹配 class 模式的元素
        self._remove_by_class(soup)
        
        # 移除匹配 id 模式的元素
        self._remove_by_id(soup)
        
        # 移除空标签
        self._remove_empty_tags(soup)
        
        # 移除注释
        self._remove_comments(soup)
        
        return str(soup)
    
    def clean_to_text(self, html: str) -> str:
        """清理 HTML 并提取纯文本
        
        Args:
            html: 原始 HTML 字符串
            
        Returns:
            清理后的纯文本
        """
        cleaned = self.clean(html)
        soup = BeautifulSoup(cleaned, self.parser)
        
        # 获取文本，使用换行分隔块级元素
        text = soup.get_text(separator="\n", strip=True)
        
        # 合并多个空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    def _remove_tags(self, soup: BeautifulSoup) -> None:
        """移除指定标签"""
        for tag_name in self.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
    
    def _remove_by_class(self, soup: BeautifulSoup) -> None:
        """移除匹配 class 模式的元素"""
        for tag in soup.find_all(class_=True):
            classes = tag.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            
            for cls in classes:
                if any(pattern.search(cls) for pattern in self._class_patterns):
                    tag.decompose()
                    break
    
    def _remove_by_id(self, soup: BeautifulSoup) -> None:
        """移除匹配 id 模式的元素"""
        for tag in soup.find_all(id=True):
            tag_id = tag.get("id", "")
            if any(pattern.search(tag_id) for pattern in self._id_patterns):
                tag.decompose()
    
    def _remove_empty_tags(self, soup: BeautifulSoup) -> None:
        """移除空标签（递归处理）"""
        # 保留的空标签（如 br, hr, img 等）
        preserve_empty = {"br", "hr", "img", "input", "meta", "link"}
        
        changed = True
        while changed:
            changed = False
            for tag in soup.find_all():
                if tag.name in preserve_empty:
                    continue
                # 检查是否为空（无文本内容且无子元素）
                if not tag.get_text(strip=True) and not tag.find_all():
                    tag.decompose()
                    changed = True
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """移除 HTML 注释"""
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()


class MarkdownConverter:
    """Markdown 转换器
    
    将 HTML 内容转换为 Markdown 格式
    
    示例:
        converter = MarkdownConverter()
        md = converter.convert("<h1>Title</h1><p>Hello <strong>world</strong></p>")
        # 结果: "# Title\n\nHello **world**\n"
    """
    
    def __init__(
        self,
        body_width: int = 0,
        ignore_links: bool = False,
        ignore_images: bool = False,
        ignore_emphasis: bool = False,
        wrap_links: bool = True,
        default_image_alt: str = "image",
        mark_code: bool = True,
        bypass_tables: bool = False
    ):
        """初始化转换器
        
        Args:
            body_width: 文本换行宽度，0 表示不换行
            ignore_links: 是否忽略链接
            ignore_images: 是否忽略图片
            ignore_emphasis: 是否忽略强调（加粗/斜体）
            wrap_links: 是否换行显示链接
            default_image_alt: 图片默认 alt 文本
            mark_code: 是否标记代码块
            bypass_tables: 是否跳过表格转换
        """
        self.h2t = html2text.HTML2Text()
        
        # 配置转换选项
        self.h2t.body_width = body_width
        self.h2t.ignore_links = ignore_links
        self.h2t.ignore_images = ignore_images
        self.h2t.ignore_emphasis = ignore_emphasis
        self.h2t.wrap_links = wrap_links
        self.h2t.default_image_alt = default_image_alt
        self.h2t.mark_code = mark_code
        self.h2t.bypass_tables = bypass_tables
        
        # 其他默认设置
        self.h2t.unicode_snob = True          # 使用 Unicode 字符
        self.h2t.escape_snob = True           # 转义特殊字符
        self.h2t.skip_internal_links = False  # 不跳过内部链接
        self.h2t.inline_links = True          # 使用内联链接格式
        self.h2t.protect_links = True         # 保护链接中的特殊字符
        self.h2t.ignore_tables = False        # 不忽略表格
    
    def convert(self, html: str) -> str:
        """将 HTML 转换为 Markdown
        
        Args:
            html: HTML 字符串
            
        Returns:
            Markdown 格式的字符串
        """
        markdown = self.h2t.handle(html)
        
        # 后处理：清理多余空行
        markdown = self._clean_markdown(markdown)
        
        return markdown
    
    def convert_with_cleaning(
        self,
        html: str,
        cleaner: Optional[ContentCleaner] = None
    ) -> str:
        """先清理 HTML 再转换为 Markdown
        
        Args:
            html: HTML 字符串
            cleaner: 内容清理器，为 None 时使用默认清理器
            
        Returns:
            Markdown 格式的字符串
        """
        if cleaner is None:
            cleaner = ContentCleaner()
        
        cleaned_html = cleaner.clean(html)
        return self.convert(cleaned_html)
    
    def _clean_markdown(self, markdown: str) -> str:
        """清理 Markdown 文本"""
        # 移除多余的空行（保留最多两个换行）
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        
        # 移除行尾空格
        markdown = re.sub(r"[ \t]+$", "", markdown, flags=re.MULTILINE)
        
        # 确保文件以单个换行结尾
        markdown = markdown.strip() + "\n"
        
        return markdown


class ChunkSplitter:
    """文档分块器
    
    将长文档分割为适当大小的块，便于向量化和检索
    
    示例:
        splitter = ChunkSplitter(chunk_size=500, overlap=50)
        chunks = splitter.split("很长的文档内容...")
        for chunk in chunks:
            print(f"Chunk: {chunk.content[:50]}...")
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        min_chunk_size: int = 100,
        separators: Optional[list[str]] = None,
        preserve_headings: bool = True
    ):
        """初始化分块器
        
        Args:
            chunk_size: 目标分块大小（字符数）
            overlap: 分块之间的重叠大小
            min_chunk_size: 最小分块大小，小于此值会合并到前一块
            separators: 分隔符列表，按优先级排序
            preserve_headings: 是否在分块时保留标题上下文
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 防止 min_chunk_size 大于 chunk_size 导致所有分块被合并为 1 个
        self.min_chunk_size = min(min_chunk_size, chunk_size)
        self.preserve_headings = preserve_headings
        
        # 默认分隔符（按优先级排序）
        self.separators = separators or [
            "\n## ",      # Markdown 二级标题
            "\n### ",     # Markdown 三级标题
            "\n#### ",    # Markdown 四级标题
            "\n\n",       # 段落分隔
            "\n",         # 换行
            "。",         # 中文句号
            ".",          # 英文句号
            "；",         # 中文分号
            ";",          # 英文分号
            "，",         # 中文逗号
            ",",          # 英文逗号
            " ",          # 空格
        ]
    
    def split(self, text: str, source_doc: Optional[str] = None) -> list[DocumentChunk]:
        """分割文本为多个块
        
        Args:
            text: 要分割的文本
            source_doc: 来源文档 ID
            
        Returns:
            DocumentChunk 列表
        """
        if not text or not text.strip():
            return []
        
        # 递归分割
        chunks_text = self._split_recursive(text, self.separators)
        
        # 合并过小的块
        chunks_text = self._merge_small_chunks(chunks_text)
        
        # 添加重叠
        chunks_text = self._add_overlap(chunks_text, text)
        
        # 创建 DocumentChunk 对象
        chunks = []
        current_index = 0
        
        for i, chunk_text in enumerate(chunks_text):
            # 查找在原文中的位置
            start_index = text.find(chunk_text.split()[0] if chunk_text.split() else "", current_index)
            if start_index == -1:
                start_index = current_index
            
            chunk = DocumentChunk(
                content=chunk_text,
                source_doc=source_doc,
                start_index=start_index,
                end_index=start_index + len(chunk_text),
                metadata={
                    "chunk_index": i,
                    "chunk_count": len(chunks_text)
                }
            )
            chunks.append(chunk)
            current_index = start_index + len(chunk_text) - self.overlap
        
        return chunks
    
    def split_markdown(
        self,
        markdown: str,
        source_doc: Optional[str] = None
    ) -> list[DocumentChunk]:
        """分割 Markdown 文档，保留标题层级信息
        
        Args:
            markdown: Markdown 格式的文本
            source_doc: 来源文档 ID
            
        Returns:
            DocumentChunk 列表，包含标题上下文
        """
        chunks = self.split(markdown, source_doc)
        
        if not self.preserve_headings:
            return chunks
        
        # 为每个块添加标题上下文
        headings = self._extract_markdown_headings(markdown)
        
        for chunk in chunks:
            # 找到该块之前最近的各级标题
            chunk_headings = self._get_context_headings(
                chunk.start_index, headings
            )
            if chunk_headings:
                chunk.metadata["headings"] = chunk_headings
        
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: list[str]
    ) -> list[str]:
        """递归分割文本"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        if not separators:
            # 没有分隔符了，强制按长度分割
            return self._split_by_length(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 使用当前分隔符分割
        parts = text.split(separator)
        
        if len(parts) == 1:
            # 未找到分隔符，尝试下一个
            return self._split_recursive(text, remaining_separators)
        
        # 重新组装并递归处理
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            # 保留分隔符（除了第一个部分）
            if i > 0:
                part = separator + part
            
            if len(current_chunk) + len(part) <= self.chunk_size:
                current_chunk += part
            else:
                if current_chunk.strip():
                    # 如果当前块仍然太大，递归处理
                    if len(current_chunk) > self.chunk_size:
                        chunks.extend(
                            self._split_recursive(current_chunk, remaining_separators)
                        )
                    else:
                        chunks.append(current_chunk)
                current_chunk = part
        
        # 处理最后一块
        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size:
                chunks.extend(
                    self._split_recursive(current_chunk, remaining_separators)
                )
            else:
                chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_length(self, text: str) -> list[str]:
        """按固定长度分割文本"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """合并过小的块"""
        if not chunks:
            return chunks
        
        merged = []
        current = ""
        
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size and current:
                # 尽量合并到当前块，但不要超过 chunk_size
                if len(current) + len(chunk) <= self.chunk_size:
                    current += chunk
                else:
                    if current.strip():
                        merged.append(current)
                    current = chunk
            elif len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                if current.strip():
                    merged.append(current)
                current = chunk
        
        if current.strip():
            merged.append(current)
        
        return merged
    
    def _add_overlap(self, chunks: list[str], original_text: str) -> list[str]:
        """为分块添加重叠部分"""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 获取前一块的结尾作为重叠
            overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) >= self.overlap else prev_chunk
            
            # 尝试在合适的位置截断重叠文本
            overlap_text = self._clean_overlap_boundary(overlap_text)
            
            if overlap_text and not current_chunk.startswith(overlap_text):
                overlapped.append(overlap_text + current_chunk)
            else:
                overlapped.append(current_chunk)
        
        return overlapped
    
    def _clean_overlap_boundary(self, text: str) -> str:
        """清理重叠边界，尽量在句子或词的边界处截断"""
        # 寻找最后一个句子结束符
        for sep in ["。", ".", "！", "!", "？", "?", "；", ";", "\n"]:
            pos = text.rfind(sep)
            if pos > len(text) // 2:  # 至少保留一半
                return text[pos + 1:]
        
        # 寻找最后一个空格
        pos = text.rfind(" ")
        if pos > len(text) // 2:
            return text[pos + 1:]
        
        return text
    
    def _extract_markdown_headings(
        self,
        markdown: str
    ) -> list[dict[str, Any]]:
        """提取 Markdown 标题及其位置"""
        headings = []
        pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        
        for match in pattern.finditer(markdown):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                "level": level,
                "text": text,
                "position": match.start()
            })
        
        return headings
    
    def _get_context_headings(
        self,
        position: int,
        headings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """获取指定位置之前的标题上下文"""
        context = {}  # level -> heading
        
        for heading in headings:
            if heading["position"] >= position:
                break
            
            level = heading["level"]
            # 更新该级别的标题，并清除更低级别的标题
            context[level] = heading
            # 清除更低级别（数字更大）的标题
            for l in list(context.keys()):
                if l > level:
                    del context[l]
        
        # 按级别排序返回
        return [context[l] for l in sorted(context.keys())]
