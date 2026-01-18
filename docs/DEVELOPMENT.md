# 开发指南

本文档提供项目开发的完整指南，包括环境设置、代码规范、检查流程和常见问题解决方案。

## 目录

- [开发环境设置](#开发环境设置)
- [代码修改检查清单](#代码修改检查清单)
- [快速验证要点](#快速验证要点)
- [预提交检查流程](#预提交检查流程)
- [常见问题和解决方案](#常见问题和解决方案)
- [模块依赖关系图](#模块依赖关系图)

---

## 开发环境设置

### 系统要求

- Python 3.9+
- pip 最新版本
- Git

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd cursorx

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from core import BaseAgent; print('核心模块加载成功')"
python -c "from agents import PlannerAgent; print('Agent 模块加载成功')"
```

### Cursor CLI 安装

```bash
# 安装 Cursor CLI
curl https://cursor.com/install -fsS | bash

# 设置 API 密钥
export CURSOR_API_KEY=your_api_key_here

# 验证安装
agent --version
agent status
```

### 开发工具安装（可选但推荐）

```bash
# 代码检查工具
pip install flake8 mypy ruff

# 测试工具
pip install pytest pytest-asyncio

# 类型检查
pip install types-PyYAML types-aiofiles
```

### IDE 配置

推荐使用 VSCode 或 Cursor IDE，配置以下设置：

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

---

## 代码修改检查清单

修改代码前后，请逐项核对以下检查清单。

### 接口一致性检查（重点）

| 检查项 | 说明 | 命令 |
|--------|------|------|
| `__init__.py` 导出 | 新增的类/函数必须在 `__init__.py` 的 `__all__` 中声明 | `grep -n "__all__" module/__init__.py` |
| 导入路径 | 使用相对导入 (`from .xxx`) 而非绝对导入 | `grep -rn "^from cursorx" .` |
| 类型注解 | 所有公开函数必须有类型注解 | `mypy module/ --ignore-missing-imports` |
| 命名规范 | 类名 PascalCase，函数名 snake_case | 代码审查 |

#### 接口一致性示例

```python
# ✅ 正确：在 __init__.py 中声明
# module/__init__.py
from .new_class import NewClass

__all__ = [
    "NewClass",  # 新增的类必须添加到这里
]

# ✅ 正确：使用相对导入
from .base import BaseClass
from ..core import Message

# ❌ 错误：使用绝对导入
from cursorx.core import Message  # 不推荐
```

### 依赖声明检查（重点）

| 检查项 | 说明 | 命令 |
|--------|------|------|
| 新依赖声明 | 新增的第三方库必须添加到 `requirements.txt` | `pip freeze \| grep package_name` |
| 版本固定 | 必须指定最低版本号 | 查看 `requirements.txt` 格式 |
| 可选依赖标注 | 可选依赖需要在代码中做 try/except 处理 | 代码审查 |

### 依赖库一致性检查（重点）

| 检查项 | 说明 | 命令 |
|--------|------|------|
| 复用已有依赖 | 新增 import 优先使用 requirements.txt 中已有的库 | `grep "package" requirements.txt` |
| 避免功能重叠 | 不引入与现有依赖功能重叠的新库 | 代码审查 |
| 同步更新依赖 | 如必须引入新库，需同时更新 requirements.txt | `pip-compile requirements.in` |

#### 依赖库一致性示例

```python
# ✅ 正确：使用项目已有的 httpx（而非 aiohttp 或 requests）
import httpx

async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        return await client.get(url)

# ❌ 错误：引入功能重叠的新库
import aiohttp  # 项目已使用 httpx，不应再引入 aiohttp
import requests  # 同步 HTTP 也应使用 httpx

# ✅ 正确：使用项目已有的 pydantic
from pydantic import BaseModel

# ❌ 错误：引入功能重叠的库
from attrs import define  # 项目已使用 pydantic
from dataclasses import dataclass  # 应使用 pydantic
```

#### 依赖处理示例

```python
# ✅ 正确：可选依赖的处理方式
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None

def create_vector_store():
    if not HAS_CHROMADB:
        raise ImportError("chromadb 未安装，请运行: pip install chromadb")
    return chromadb.Client()
```

### 通用检查项

- [ ] 语法正确性：`python -m py_compile file.py`
- [ ] 模块可导入：`python -c "from module import NewClass"`
- [ ] 类型注解完整：`mypy file.py --ignore-missing-imports`
- [ ] 代码风格：`flake8 file.py --max-line-length=120`
- [ ] 测试通过：`pytest tests/test_module.py -v`
- [ ] 文档注释：所有公开 API 必须有 docstring

---

## 快速验证要点

代码修改后，请按以下步骤快速验证，避免常见问题：

### 1. 模块导入强制验证（必须）

**任何代码修改后必须执行以下验证：**

```bash
# 替换 <module> 为实际修改的模块名
python -c "import <module>"

# 示例
python -c "import agents"
python -c "import core"
python -c "import cursor"
python -c "import coordinator"

# 完整模块验证
python -c "from agents import *; from coordinator import *; from core import *; from cursor import *"
```

> **强制要求**：每次修改代码后，必须运行 `python -c "import <module>"` 验证修改的模块可正常导入。

### 2. 入口脚本验证（必须）

**提交前必须验证入口脚本可正常执行：**

```bash
# 提交前必须运行此命令
python run.py --help
```

> **强制要求**：提交前必须运行 `python run.py --help` 验证入口脚本正常工作。

### 3. 完整验证

提交前运行完整检查：

```bash
bash scripts/check_all.sh
```

### 4. 详细说明

遇到问题时，请参阅 [LESSONS_LEARNED.md](LESSONS_LEARNED.md) 获取详细的经验教训和解决方案。

---

## 预提交检查流程

### 快速检查（每次提交前必做）

```bash
# 运行项目健康检查脚本
bash scripts/check_all.sh

# 或手动执行关键检查
python -m py_compile $(find . -name "*.py" -not -path "./venv/*")
```

### 完整检查（重要修改前）

```bash
# 完整检查（包括类型、风格和测试）
bash scripts/check_all.sh --full
```

### 分步检查命令

```bash
# 1. 语法检查
python -m py_compile agents/new_agent.py

# 2. 模块导入检查
PYTHONPATH=. python -c "from agents import NewAgent"

# 3. 类型检查
mypy agents/new_agent.py --ignore-missing-imports

# 4. 代码风格检查
flake8 agents/new_agent.py --max-line-length=120

# 5. 运行相关测试
pytest tests/test_new_agent.py -v

# 6. 运行全部测试
pytest tests/ -v --tb=short
```

### 检查脚本输出解读

```
✓ 通过      - 检查项正常
✗ 失败      - 必须修复后才能提交
⚠ 警告      - 建议修复但不阻止提交
○ 跳过      - 相关工具未安装或不适用
```

---

## 常见问题和解决方案

### 问题 1：模块导入失败 - ImportError

**现象**
```
ImportError: cannot import name 'NewClass' from 'module'
```

**原因**
- 类/函数未在 `__init__.py` 的 `__all__` 中声明
- 循环导入

**解决方案**
```python
# 1. 确保在 __init__.py 中正确导出
# module/__init__.py
from .new_file import NewClass

__all__ = [
    "NewClass",  # 添加这一行
]

# 2. 检查循环导入，使用延迟导入
def get_new_class():
    from .new_file import NewClass  # 延迟导入
    return NewClass
```

### 问题 2：类型检查报错 - Type Errors

**现象**
```
error: Incompatible types in assignment
error: Missing return statement
```

**解决方案**
```python
# ✅ 正确：完整的类型注解
from typing import Optional, List

def process_items(items: List[str]) -> Optional[str]:
    if not items:
        return None
    return items[0]

# ✅ 正确：使用 TypeVar 处理泛型
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
```

### 问题 3：异步函数调用错误 - RuntimeWarning

**现象**
```
RuntimeWarning: coroutine 'xxx' was never awaited
```

**解决方案**
```python
# ❌ 错误
result = async_function()  # 忘记 await

# ✅ 正确
result = await async_function()

# 或在同步上下文中
import asyncio
result = asyncio.run(async_function())
```

### 问题 4：配置文件加载失败 - YAML Error

**现象**
```
yaml.scanner.ScannerError: while scanning a simple key
```

**解决方案**
```bash
# 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# 常见错误：缩进不一致、特殊字符未转义
# 使用 2 空格缩进，字符串值用引号包裹
```

### 问题 5：测试失败 - pytest Errors

**现象**
```
FAILED tests/test_xxx.py::test_function - AssertionError
```

**解决方案**
```bash
# 1. 查看详细错误
pytest tests/test_xxx.py::test_function -v --tb=long

# 2. 只运行失败的测试
pytest tests/ --lf -v

# 3. 调试模式运行
pytest tests/test_xxx.py -v --pdb
```

### 问题 6：依赖版本冲突

**现象**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**解决方案**
```bash
# 1. 创建新的虚拟环境
python -m venv venv_new
source venv_new/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 如果仍有冲突，检查版本约束
pip install package --dry-run
```

### 问题 7：向量存储初始化失败

**现象**
```
chromadb.errors.InvalidCollectionException
```

**解决方案**
```python
# 清理现有索引后重新初始化
import shutil
shutil.rmtree('.cursor/vector_index/', ignore_errors=True)

# 或使用不同的集合名称
config = IndexConfig(collection_name="new_collection")
```

---

## 模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           应用层                                      │
├─────────────────────────────────────────────────────────────────────┤
│  run.py                                                              │
│  scripts/run_*.py                                                    │
└─────────────┬────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         协调层 (coordinator/)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Orchestrator          - 任务编排器（协程版）                           │
│  MultiProcessOrchestrator - 任务编排器（多进程版）                       │
│  WorkerPool            - Worker 池管理                                │
└────────┬──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent 层 (agents/)                           │
├─────────────────────────────────────────────────────────────────────┤
│  PlannerAgent          - 规划者（任务分解）                             │
│  WorkerAgent           - 执行者（代码修改）                             │
│  ReviewerAgent         - 评审者（质量检查）                             │
│  CommitterAgent        - 提交者（Git 操作）                            │
│  *AgentProcess         - 多进程版本                                    │
└────────┬──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        任务层 (tasks/)                               │
├─────────────────────────────────────────────────────────────────────┤
│  Task                  - 任务数据模型                                  │
│  TaskQueue             - 任务队列                                      │
│  TaskStatus/Priority   - 枚举类型                                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         核心层 (core/)                                │
├─────────────────────────────────────────────────────────────────────┤
│  BaseAgent             - Agent 基类                                   │
│  Message               - 消息模型                                      │
│  AgentState/SystemState - 状态管理                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       集成层 (cursor/)                                │
├─────────────────────────────────────────────────────────────────────┤
│  CursorAgentClient     - CLI 客户端                                   │
│  CursorCloudClient     - Cloud API 客户端                             │
│  AgentExecutor         - 执行器抽象                                    │
│  StreamingClient       - 流式输出处理                                  │
│  MCPManager            - MCP 服务器管理                                │
│  EgressIPManager       - 网络/IP 管理                                  │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      扩展层                                          │
├──────────────────────────┬──────────────────────────────────────────┤
│   知识库 (knowledge/)     │   代码索引 (indexing/)                     │
├──────────────────────────┼──────────────────────────────────────────┤
│  KnowledgeManager        │  CodebaseIndexer                         │
│  WebFetcher              │  SemanticSearch                          │
│  KnowledgeVectorStore    │  ChromaVectorStore                       │
│  KnowledgeSemanticSearch │  SentenceTransformerEmbedding            │
└──────────────────────────┴──────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      进程管理 (process/)                              │
├─────────────────────────────────────────────────────────────────────┤
│  AgentProcessManager   - 进程生命周期管理                              │
│  AgentWorkerProcess    - Worker 进程封装                              │
│  MessageQueue          - 进程间消息队列                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 模块间依赖规则

1. **层次依赖**：上层模块可以依赖下层，但下层不能依赖上层
2. **同层依赖**：同层模块可以相互依赖，但需避免循环导入
3. **核心依赖**：`core/` 模块只依赖标准库和 `pydantic`
4. **可选依赖**：`knowledge/` 和 `indexing/` 的向量功能依赖 `sentence-transformers` 和 `chromadb`

### 依赖检查命令

```bash
# 检查模块是否可正确导入
PYTHONPATH=. python -c "
from core import BaseAgent, Message
from agents import PlannerAgent, WorkerAgent, ReviewerAgent
from coordinator import Orchestrator, MultiProcessOrchestrator
from tasks import Task, TaskQueue
from cursor import CursorAgentClient, AgentExecutor
print('所有核心模块加载成功')
"

# 检查可选模块
PYTHONPATH=. python -c "
try:
    from knowledge import KnowledgeManager
    from indexing import SemanticSearch
    print('扩展模块加载成功')
except ImportError as e:
    print(f'扩展模块加载失败: {e}')
"
```

---

## 附录

### 相关文档

- [AGENTS.md](../AGENTS.md) - 系统架构和 CLI 使用说明
- [KNOWLEDGE_BASE_GUIDE.txt](KNOWLEDGE_BASE_GUIDE.txt) - 知识库使用指南

### 常用命令速查

| 操作 | 命令 |
|------|------|
| 快速检查 | `bash scripts/check_all.sh` |
| 完整检查 | `bash scripts/check_all.sh --full` |
| 运行测试 | `pytest tests/ -v` |
| 类型检查 | `mypy core/ agents/ --ignore-missing-imports` |
| 代码风格 | `flake8 . --max-line-length=120` |
| 验证配置 | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` |
| 列出模型 | `agent models` |
| 检查认证 | `agent status` |
