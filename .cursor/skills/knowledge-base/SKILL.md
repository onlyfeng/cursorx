---
name: knowledge-base
description: 知识库管理与检索。用于添加、管理和搜索外部文档内容，支持从 URL 获取网页信息。
---

# 知识库技能 (Knowledge Base Skill)

管理外部文档资源，提供内容获取、存储和智能检索功能。

## 使用时机

- 需要获取和存储网页内容以供后续参考
- 需要在已收集的文档中搜索相关信息
- 需要管理和组织多个外部资料来源
- 需要引用之前获取的文档内容

## 核心功能

### 1. 添加文档

从 URL 获取内容并添加到知识库：

```python
from knowledge import KnowledgeManager

manager = KnowledgeManager()
await manager.initialize()

# 添加单个 URL
doc = await manager.add_url("https://example.com/docs")

# 批量添加多个 URL
docs = await manager.add_urls([
    "https://example.com/page1",
    "https://example.com/page2",
])
```

### 2. 搜索文档

在知识库中搜索相关内容：

```python
# 关键词搜索
results = manager.search("关键词", max_results=10)

for result in results:
    print(f"文档: {result.title}")
    print(f"URL: {result.url}")
    print(f"匹配片段: {result.snippet}")
    print(f"相关度: {result.score}")
```

### 3. 文档管理

```python
# 列出所有文档
all_docs = manager.list()

# 获取指定文档
doc = manager.get_document("doc-xxx")
doc = manager.get_document_by_url("https://example.com")

# 刷新文档内容（重新获取）
await manager.refresh("doc-xxx")

# 删除文档
manager.remove("doc-xxx")

# 清空知识库
manager.clear()
```

### 4. 查看统计信息

```python
stats = manager.stats
print(f"文档数量: {stats.document_count}")
print(f"分块数量: {stats.chunk_count}")
print(f"总内容大小: {stats.total_content_size}")
```

## 数据模型

### Document（文档）

```python
class Document:
    id: str           # 文档唯一标识
    url: str          # 来源 URL
    title: str        # 文档标题
    content: str      # 文档内容
    chunks: list      # 内容分块
    metadata: dict    # 元数据
    created_at: datetime
    updated_at: datetime
```

### SearchResult（搜索结果）

```python
class SearchResult:
    doc_id: str       # 文档 ID
    url: str          # 文档 URL
    title: str        # 文档标题
    score: float      # 匹配分数
    snippet: str      # 匹配片段
    match_type: str   # 匹配类型（exact/partial）
```

## 使用场景示例

### 场景 1: 收集技术文档

```python
# 收集 API 文档
await manager.add_urls([
    "https://docs.python.org/3/library/asyncio.html",
    "https://pydantic-docs.helpmanual.io/usage/models/",
])

# 后续查询 asyncio 相关内容
results = manager.search("asyncio event loop")
```

### 场景 2: 引用之前获取的内容

```
用户: 帮我查找之前收集的关于 Pydantic 验证的文档

Agent:
1. 搜索知识库: manager.search("Pydantic validation")
2. 获取相关文档内容
3. 引用文档中的具体说明
```

### 场景 3: 更新过期文档

```python
# 刷新特定文档
await manager.refresh("https://example.com/changelog")

# 或通过文档 ID 刷新
await manager.refresh("doc-abc123")
```

## 最佳实践

1. **合理组织文档**: 使用 metadata 添加标签和分类
2. **定期刷新**: 对于动态内容，定期调用 refresh 更新
3. **精确搜索**: 使用具体关键词提高搜索准确度
4. **清理无用文档**: 及时删除不再需要的文档节省空间

## 扩展能力

- **向量搜索**: 可扩展为基于语义的向量相似度搜索
- **文档分块**: 支持将长文档分割成小块便于精确检索
- **嵌入向量**: 支持为文档块生成向量嵌入
