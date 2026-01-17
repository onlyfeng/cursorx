"""知识库数据模型定义"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class FetchStatus(str, Enum):
    """抓取任务状态"""
    PENDING = "pending"          # 待抓取
    FETCHING = "fetching"        # 抓取中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


class FetchPriority(int, Enum):
    """抓取优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class DocumentChunk(BaseModel):
    """文档分块
    
    将文档内容分割成小块，便于向量化和检索
    """
    chunk_id: str = Field(default_factory=lambda: f"chunk-{uuid.uuid4().hex[:8]}")
    content: str                                    # 分块内容
    embedding: Optional[list[float]] = None         # 向量嵌入（可选）
    source_doc: Optional[str] = None                # 来源文档 ID
    
    # 位置信息
    start_index: int = 0                            # 在原文档中的起始位置
    end_index: int = 0                              # 在原文档中的结束位置
    
    # 元数据
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def has_embedding(self) -> bool:
        """是否已生成向量嵌入"""
        return self.embedding is not None and len(self.embedding) > 0


class Document(BaseModel):
    """文档模型
    
    表示从网页或其他来源获取的文档
    """
    id: str = Field(default_factory=lambda: f"doc-{uuid.uuid4().hex[:8]}")
    url: str                                        # 来源 URL
    title: str = ""                                 # 文档标题
    content: str = ""                               # 原始内容
    chunks: list[DocumentChunk] = Field(default_factory=list)  # 文档分块
    
    # 元数据
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """添加分块"""
        chunk.source_doc = self.id
        self.chunks.append(chunk)
    
    def update_content(self, content: str, title: Optional[str] = None) -> None:
        """更新文档内容"""
        self.content = content
        if title:
            self.title = title
        self.updated_at = datetime.now()
    
    def get_chunk_count(self) -> int:
        """获取分块数量"""
        return len(self.chunks)
    
    def has_embeddings(self) -> bool:
        """是否所有分块都已生成向量嵌入"""
        if not self.chunks:
            return False
        return all(chunk.has_embedding() for chunk in self.chunks)


class KnowledgeBaseStats(BaseModel):
    """知识库统计信息"""
    document_count: int = 0
    chunk_count: int = 0
    embedding_count: int = 0
    total_content_size: int = 0                     # 总内容字符数
    last_updated: Optional[datetime] = None


class KnowledgeBase(BaseModel):
    """知识库
    
    管理多个文档，提供索引和检索功能
    """
    id: str = Field(default_factory=lambda: f"kb-{uuid.uuid4().hex[:8]}")
    name: str                                       # 知识库名称
    description: str = ""                           # 描述
    documents: dict[str, Document] = Field(default_factory=dict)  # 文档集合
    index: dict[str, Any] = Field(default_factory=dict)  # 索引数据
    stats: KnowledgeBaseStats = Field(default_factory=KnowledgeBaseStats)
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_document(self, document: Document) -> None:
        """添加文档"""
        self.documents[document.id] = document
        self._update_stats()
        self.updated_at = datetime.now()
    
    def remove_document(self, doc_id: str) -> Optional[Document]:
        """移除文档"""
        doc = self.documents.pop(doc_id, None)
        if doc:
            self._update_stats()
            self.updated_at = datetime.now()
        return doc
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def get_document_by_url(self, url: str) -> Optional[Document]:
        """根据 URL 获取文档"""
        for doc in self.documents.values():
            if doc.url == url:
                return doc
        return None
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        doc_count = len(self.documents)
        chunk_count = sum(len(doc.chunks) for doc in self.documents.values())
        embedding_count = sum(
            1 for doc in self.documents.values() 
            for chunk in doc.chunks 
            if chunk.has_embedding()
        )
        total_size = sum(len(doc.content) for doc in self.documents.values())
        
        self.stats = KnowledgeBaseStats(
            document_count=doc_count,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            total_content_size=total_size,
            last_updated=datetime.now()
        )
    
    def get_all_chunks(self) -> list[DocumentChunk]:
        """获取所有文档分块"""
        chunks = []
        for doc in self.documents.values():
            chunks.extend(doc.chunks)
        return chunks


class FetchTask(BaseModel):
    """网页抓取任务
    
    用于管理 URL 抓取队列
    """
    id: str = Field(default_factory=lambda: f"fetch-{uuid.uuid4().hex[:8]}")
    url: str                                        # 目标 URL
    status: FetchStatus = FetchStatus.PENDING       # 任务状态
    priority: FetchPriority = FetchPriority.NORMAL  # 优先级
    result: Optional[Document] = None               # 抓取结果（文档）
    
    # 错误信息
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # 配置
    headers: dict[str, str] = Field(default_factory=dict)  # 自定义请求头
    timeout: int = 30                               # 超时时间（秒）
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start(self) -> None:
        """开始抓取"""
        self.status = FetchStatus.FETCHING
        self.started_at = datetime.now()
    
    def complete(self, document: Document) -> None:
        """完成抓取"""
        self.status = FetchStatus.COMPLETED
        self.result = document
        self.completed_at = datetime.now()
    
    def fail(self, error: str) -> None:
        """抓取失败"""
        self.status = FetchStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.retry_count < self.max_retries
    
    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.status in (FetchStatus.COMPLETED, FetchStatus.FAILED, FetchStatus.CANCELLED)
