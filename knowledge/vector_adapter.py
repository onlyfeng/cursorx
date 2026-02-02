"""知识库向量适配器

提供 DocumentChunk 和 CodeChunk 之间的转换，
以及文档内容的智能分块功能
"""

import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from indexing.base import ChunkType, CodeChunk

from .models import Document, DocumentChunk


class DocumentChunkConfig(BaseModel):
    """文档分块配置"""

    chunk_size: int = 1000  # 目标分块大小（字符数）
    chunk_overlap: int = 100  # 分块重叠大小
    min_chunk_size: int = 50  # 最小分块大小
    max_chunk_size: int = 2000  # 最大分块大小

    # 分块策略
    respect_paragraph_boundary: bool = True  # 尊重段落边界
    respect_sentence_boundary: bool = True  # 尊重句子边界

    model_config = ConfigDict(use_enum_values=True)


class DocumentChunkAdapter:
    """文档分块适配器

    在 DocumentChunk 和 CodeChunk 之间进行转换，
    以便利用现有的向量索引基础设施
    """

    def document_chunk_to_code_chunk(self, doc_chunk: DocumentChunk, doc: Document) -> CodeChunk:
        """将 DocumentChunk 转换为 CodeChunk 格式

        保留必要的元数据，使其可以被现有的向量存储处理

        Args:
            doc_chunk: 文档分块
            doc: 来源文档

        Returns:
            转换后的 CodeChunk
        """
        # 构建元数据
        metadata: dict[str, Any] = {
            "source_type": "document",  # 标识来源类型
            "doc_id": doc.id,  # 文档 ID
            "doc_url": doc.url,  # 文档 URL
            "doc_title": doc.title,  # 文档标题
            "original_chunk_id": doc_chunk.chunk_id,  # 原始分块 ID
            "start_index": doc_chunk.start_index,  # 在原文中的起始位置
            "end_index": doc_chunk.end_index,  # 在原文中的结束位置
        }

        # 合并原有元数据
        metadata.update(doc_chunk.metadata)
        metadata.update(doc.metadata)

        # 创建 CodeChunk
        return CodeChunk(
            chunk_id=doc_chunk.chunk_id,
            content=doc_chunk.content,
            file_path=doc.url,  # 使用 URL 作为文件路径
            start_line=doc_chunk.start_index,  # 使用字符位置作为行号
            end_line=doc_chunk.end_index,
            chunk_type=ChunkType.UNKNOWN,  # 文档不是代码
            language="text",  # 标识为文本
            name=doc.title if doc.title else None,  # 使用文档标题作为名称
            embedding=doc_chunk.embedding,
            metadata=metadata,
        )

    def code_chunk_to_document_chunk(self, code_chunk: CodeChunk) -> DocumentChunk:
        """将 CodeChunk 转换回 DocumentChunk 格式

        Args:
            code_chunk: 代码分块

        Returns:
            转换后的 DocumentChunk
        """
        # 从元数据中提取原始信息
        metadata = dict(code_chunk.metadata)

        # 获取原始分块 ID
        original_chunk_id = metadata.pop("original_chunk_id", code_chunk.chunk_id)

        # 获取来源文档 ID
        source_doc = metadata.pop("doc_id", None)

        # 提取位置信息
        start_index = metadata.pop("start_index", code_chunk.start_line)
        end_index = metadata.pop("end_index", code_chunk.end_line)

        # 移除转换过程中添加的元数据
        metadata.pop("source_type", None)
        metadata.pop("doc_url", None)
        metadata.pop("doc_title", None)

        return DocumentChunk(
            chunk_id=original_chunk_id,
            content=code_chunk.content,
            embedding=code_chunk.embedding,
            source_doc=source_doc,
            start_index=start_index,
            end_index=end_index,
            metadata=metadata,
        )

    def prepare_for_embedding(
        self,
        doc: Document,
        chunker: Optional["DocumentTextChunker"] = None,
        config: Optional[DocumentChunkConfig] = None,
    ) -> list[CodeChunk]:
        """将文档分块并准备用于嵌入

        Args:
            doc: 要处理的文档
            chunker: 文档分块器（可选，为 None 时使用默认配置创建）
            config: 分块配置（可选）

        Returns:
            准备好用于嵌入的 CodeChunk 列表
        """
        # 创建分块器
        if chunker is None:
            chunker = DocumentTextChunker(config)

        # 如果文档已有分块，直接使用
        if doc.chunks:
            chunks = doc.chunks
        else:
            # 否则进行分块
            chunks = chunker.chunk_document(
                doc, chunk_size=chunker.config.chunk_size, overlap=chunker.config.chunk_overlap
            )

        # 转换为 CodeChunk 格式
        code_chunks = []
        for chunk in chunks:
            code_chunk = self.document_chunk_to_code_chunk(chunk, doc)
            code_chunks.append(code_chunk)

        return code_chunks


class DocumentTextChunker:
    """文档文本分块器

    将文档内容智能分割成小块，支持：
    - 按段落边界分块
    - 按句子边界分块
    - 滑动窗口分块（带重叠）
    """

    # 段落分隔符模式
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    # 句子结束符模式
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")

    def __init__(self, config: Optional[DocumentChunkConfig] = None):
        """初始化分块器

        Args:
            config: 分块配置，为 None 时使用默认配置
        """
        self.config = config or DocumentChunkConfig()

    def chunk_document(
        self, doc: Document, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> list[DocumentChunk]:
        """将文档内容分块

        Args:
            doc: 要分块的文档
            chunk_size: 分块大小（覆盖配置）
            overlap: 重叠大小（覆盖配置）

        Returns:
            文档分块列表
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        content = doc.content
        if not content or not content.strip():
            return []

        # 根据配置选择分块策略
        if self.config.respect_paragraph_boundary:
            chunks = self._chunk_by_paragraph(content, chunk_size, overlap)
        elif self.config.respect_sentence_boundary:
            chunks = self._chunk_by_sentence(content, chunk_size, overlap)
        else:
            chunks = self._chunk_by_sliding_window(content, chunk_size, overlap)

        # 创建 DocumentChunk 对象
        doc_chunks = []
        for chunk_content, start_index, end_index in chunks:
            if len(chunk_content.strip()) < self.config.min_chunk_size:
                continue

            doc_chunk = DocumentChunk(
                content=chunk_content,
                source_doc=doc.id,
                start_index=start_index,
                end_index=end_index,
                metadata={
                    "title": doc.title,
                    "url": doc.url,
                },
            )
            doc_chunks.append(doc_chunk)

        return doc_chunks

    def _chunk_by_paragraph(self, content: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
        """按段落边界分块

        优先在段落边界分割，如果单个段落过大则进一步分割

        Args:
            content: 文档内容
            chunk_size: 目标分块大小
            overlap: 重叠大小

        Returns:
            (内容, 起始位置, 结束位置) 元组列表
        """
        # 分割段落
        paragraphs = self.PARAGRAPH_PATTERN.split(content)

        # 计算每个段落的位置
        paragraph_positions: list[tuple[str, int, int]] = []
        current_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 查找段落在原文中的位置
            start_pos = content.find(para, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(para)

            paragraph_positions.append((para, start_pos, end_pos))
            current_pos = end_pos

        # 合并段落直到达到目标大小
        chunks: list[tuple[str, int, int]] = []
        current_chunk_parts: list[str] = []
        current_start: Optional[int] = None
        current_end: int = 0
        current_length = 0

        for para, start_pos, end_pos in paragraph_positions:
            para_length = len(para)

            # 如果单个段落超过最大大小，需要进一步分割
            if para_length > self.config.max_chunk_size:
                # 保存当前累积的块
                if current_chunk_parts:
                    chunk_content = "\n\n".join(current_chunk_parts)
                    chunks.append((chunk_content, current_start or 0, current_end))
                    current_chunk_parts = []
                    current_start = None
                    current_length = 0

                # 对大段落进行句子级分割
                if self.config.respect_sentence_boundary:
                    sub_chunks = self._chunk_by_sentence(para, chunk_size, overlap)
                else:
                    sub_chunks = self._chunk_by_sliding_window(para, chunk_size, overlap)

                # 调整位置偏移
                for sub_content, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_content, start_pos + sub_start, start_pos + sub_end))
                continue

            # 检查添加当前段落后是否超过目标大小
            potential_length = current_length + para_length + (2 if current_chunk_parts else 0)

            if potential_length > chunk_size and current_chunk_parts:
                # 保存当前块
                chunk_content = "\n\n".join(current_chunk_parts)
                chunks.append((chunk_content, current_start or 0, current_end))

                # 处理重叠：保留最后一个段落
                if overlap > 0 and current_chunk_parts:
                    last_para = current_chunk_parts[-1]
                    if len(last_para) <= overlap:
                        current_chunk_parts = [last_para]
                        current_length = len(last_para)
                        # 需要找到这个段落的起始位置
                        for p, s, e in paragraph_positions:
                            if p == last_para:
                                current_start = s
                                current_end = e
                                break
                    else:
                        current_chunk_parts = []
                        current_start = None
                        current_length = 0
                else:
                    current_chunk_parts = []
                    current_start = None
                    current_length = 0

            # 添加当前段落
            current_chunk_parts.append(para)
            if current_start is None:
                current_start = start_pos
            current_end = end_pos
            current_length = len("\n\n".join(current_chunk_parts))

        # 保存最后一个块
        if current_chunk_parts:
            chunk_content = "\n\n".join(current_chunk_parts)
            chunks.append((chunk_content, current_start or 0, current_end))

        return chunks

    def _chunk_by_sentence(self, content: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
        """按句子边界分块

        在句子边界处分割，确保每个块包含完整的句子

        Args:
            content: 文档内容
            chunk_size: 目标分块大小
            overlap: 重叠大小

        Returns:
            (内容, 起始位置, 结束位置) 元组列表
        """
        # 分割句子
        sentences = self.SENTENCE_PATTERN.split(content)

        # 如果分割后只有一个元素，说明没有句子分隔符，使用滑动窗口
        if len(sentences) <= 1:
            return self._chunk_by_sliding_window(content, chunk_size, overlap)

        # 计算每个句子的位置
        sentence_positions: list[tuple[str, int, int]] = []
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            start_pos = content.find(sentence, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(sentence)

            sentence_positions.append((sentence, start_pos, end_pos))
            current_pos = end_pos

        # 合并句子直到达到目标大小
        chunks: list[tuple[str, int, int]] = []
        current_chunk_parts: list[str] = []
        current_start: Optional[int] = None
        current_end: int = 0
        current_length = 0

        for sentence, start_pos, end_pos in sentence_positions:
            sentence_length = len(sentence)

            # 如果单个句子超过最大大小
            if sentence_length > self.config.max_chunk_size:
                # 保存当前累积的块
                if current_chunk_parts:
                    chunk_content = " ".join(current_chunk_parts)
                    chunks.append((chunk_content, current_start or 0, current_end))
                    current_chunk_parts = []
                    current_start = None
                    current_length = 0

                # 对长句子使用滑动窗口
                sub_chunks = self._chunk_by_sliding_window(sentence, chunk_size, overlap)
                for sub_content, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_content, start_pos + sub_start, start_pos + sub_end))
                continue

            # 检查添加当前句子后是否超过目标大小
            potential_length = current_length + sentence_length + (1 if current_chunk_parts else 0)

            if potential_length > chunk_size and current_chunk_parts:
                # 保存当前块
                chunk_content = " ".join(current_chunk_parts)
                chunks.append((chunk_content, current_start or 0, current_end))

                # 处理重叠
                overlap_parts: list[str] = []
                overlap_length = 0

                for s in reversed(current_chunk_parts):
                    if overlap_length + len(s) <= overlap:
                        overlap_parts.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                if overlap_parts:
                    current_chunk_parts = overlap_parts
                    current_length = overlap_length - 1
                    # 找到重叠部分的起始位置
                    first_overlap = overlap_parts[0]
                    for s, sp, _ep in sentence_positions:
                        if s == first_overlap:
                            current_start = sp
                            break
                else:
                    current_chunk_parts = []
                    current_start = None
                    current_length = 0

            # 添加当前句子
            current_chunk_parts.append(sentence)
            if current_start is None:
                current_start = start_pos
            current_end = end_pos
            current_length = len(" ".join(current_chunk_parts))

        # 保存最后一个块
        if current_chunk_parts:
            chunk_content = " ".join(current_chunk_parts)
            chunks.append((chunk_content, current_start or 0, current_end))

        return chunks

    def _chunk_by_sliding_window(self, content: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
        """滑动窗口分块

        按固定大小分割，支持重叠

        Args:
            content: 文档内容
            chunk_size: 分块大小
            overlap: 重叠大小

        Returns:
            (内容, 起始位置, 结束位置) 元组列表
        """
        chunks: list[tuple[str, int, int]] = []
        content_length = len(content)

        if content_length == 0:
            return []

        step = max(1, chunk_size - overlap)
        start = 0

        while start < content_length:
            end = min(start + chunk_size, content_length)
            chunk_content = content[start:end]

            # 尝试在单词边界结束（如果不是文档末尾）
            if end < content_length and chunk_content:
                # 查找最后一个空格
                last_space = chunk_content.rfind(" ")
                if last_space > chunk_size // 2:  # 确保至少保留一半内容
                    end = start + last_space
                    chunk_content = content[start:end]

            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunks.append((chunk_content.strip(), start, end))

            start += step

        return chunks

    def estimate_chunks_count(
        self, content: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> int:
        """估算分块数量

        快速估算文档会被分成多少块

        Args:
            content: 文档内容
            chunk_size: 分块大小
            overlap: 重叠大小

        Returns:
            估算的分块数量
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        content_length = len(content)
        if content_length == 0:
            return 0

        if content_length <= chunk_size:
            return 1

        step = max(1, chunk_size - overlap)
        return (content_length - chunk_size) // step + 1
