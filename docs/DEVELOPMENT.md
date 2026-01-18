# 开发指南

本文档提供项目开发的完整指南，包括环境设置、代码规范、检查流程和常见问题解决方案。

## 目录

- [开发环境设置](#开发环境设置)
- [代码修改检查清单](#代码修改检查清单)
- [快速验证要点](#快速验证要点)
  - [验证层次对比](#验证层次对比)
- [运行时验证 vs 静态分析](#运行时验证-vs-静态分析)
- [预提交检查流程](#预提交检查流程)
- [常见问题和解决方案](#常见问题和解决方案)
- [异步代码最佳实践](#异步代码最佳实践)
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

### 依赖管理

项目依赖分为**核心依赖**和**可选依赖**两类：

#### 依赖文件说明

| 文件 | 说明 | 用途 |
|------|------|------|
| `requirements.txt` | 完整依赖列表 | 包含所有功能的依赖 |
| `requirements.in` | 核心依赖源文件 | pip-compile 输入 |
| `requirements-optional.txt` | 可选依赖 | ML/向量搜索、浏览器自动化 |
| `requirements-dev.in` | 开发依赖源文件 | 代码检查、类型验证 |
| `requirements-test.in` | 测试依赖源文件 | 测试框架、Mock |

#### 按需安装

```bash
# 方式 1: 使用 requirements 文件
pip install -r requirements.txt                  # 全部依赖
pip install -r requirements-optional.txt         # 仅可选依赖（ML/浏览器）

# 方式 2: 使用 pyproject.toml 分组安装（推荐）
pip install -e .              # 仅核心依赖（最小安装）
pip install -e ".[web]"       # 核心 + 网页处理
pip install -e ".[ml]"        # 核心 + ML/向量搜索
pip install -e ".[dev]"       # 核心 + 开发工具
pip install -e ".[test]"      # 核心 + 测试框架
pip install -e ".[all]"       # 所有依赖
```

#### 可选依赖列表

| 依赖 | 分组 | 用途 | 安装命令 |
|------|------|------|----------|
| `torch` | ml | GPU 加速、深度学习 | `pip install torch` |
| `sentence-transformers` | ml | 本地嵌入模型 | `pip install sentence-transformers` |
| `chromadb` | ml | 向量数据库 | `pip install chromadb` |
| `numpy` | ml | 数值计算 | `pip install numpy` |
| `playwright` | browser | 网页渲染、JS 执行 | `pip install playwright && playwright install chromium` |

#### 核心依赖与可选依赖的区别

- **核心依赖**: 运行 Agent 系统必需的最小依赖集，不安装会导致程序无法启动
- **可选依赖**: 特定功能所需，不安装不影响基础功能，代码中使用 `try-except` 处理

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

### 可选依赖处理规范

项目中存在多个可选依赖（如 `torch`、`numpy`、`playwright`、`chromadb` 等），这些依赖可能不在所有环境中安装。为确保代码的健壮性，请遵循以下规范：

#### 1. 模块级可选依赖导入

```python
# ✅ 正确：使用 try-except 包装可选依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

# ✅ 正确：延迟导入（在函数/方法内部导入）
def use_torch_feature():
    try:
        import torch
    except ImportError as e:
        raise ImportError("请安装 torch: pip install torch") from e
    return torch.tensor([1, 2, 3])
```

#### 2. 测试文件中的可选依赖

```python
# ✅ 正确：测试文件中的可选依赖处理
import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy 未安装")
class TestWithNumpy:
    def test_numpy_feature(self):
        result = np.array([1, 2, 3])
        assert len(result) == 3
```

#### 3. 运行时可选依赖检查

```python
# ✅ 正确：在使用前检查依赖是否可用
try:
    from playwright.async_api import async_playwright
except ImportError:
    # 返回明确的错误信息，而非让程序崩溃
    return FetchResult(
        success=False,
        error="Playwright 未安装，请运行: pip install playwright"
    )
```

#### 4. 项目中的可选依赖列表

| 依赖 | 用途 | 安装命令 |
|------|------|----------|
| `torch` | 嵌入模型 GPU 支持 | `pip install torch` |
| `sentence-transformers` | 本地嵌入模型 | `pip install sentence-transformers` |
| `chromadb` | 向量存储 | `pip install chromadb` |
| `playwright` | 网页获取（JS 渲染） | `pip install playwright && playwright install chromium` |
| `numpy` | 测试 mock（数组生成） | `pip install numpy` |

#### 5. 规范要点

1. **必须使用 try-except**：所有可选依赖导入必须使用 try-except 包装
2. **提供标志变量**：使用 `HAS_XXX` 布尔变量标识依赖是否可用
3. **明确错误信息**：当缺少依赖时，提供清晰的安装指令
4. **测试跳过**：测试中使用 `@pytest.mark.skipif` 优雅跳过依赖缺失的测试
5. **类型注解**：使用 `# type: ignore` 处理 None 赋值的类型警告

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

#### `python -c` 导入测试的局限性

⚠️ **注意**：`python -c "import module"` 仅验证模块的**顶层导入**，存在以下局限性：

| 局限性 | 说明 | 示例 |
|--------|------|------|
| 条件导入 | 仅执行符合条件的分支 | `if TYPE_CHECKING: import xxx` 不会执行 |
| 延迟导入 | 函数内的导入不会执行 | `def foo(): import bar` 不会触发 |
| 可选依赖 | try-except 包装的导入失败会被静默处理 | `try: import torch except: pass` |
| 运行时路径 | 无法检测动态导入错误 | `importlib.import_module(name)` |

**推荐做法**：

```bash
# 1. 基础导入验证
python -c "import agents"

# 2. 实际运行验证（更可靠）
python run.py --help

# 3. 完整测试覆盖
pytest tests/test_imports.py -v
```

### 2. 入口脚本验证（必须）

**提交前必须验证入口脚本可正常执行：**

```bash
# 提交前必须运行此命令
python run.py --help
```

> **强制要求**：提交前必须运行 `python run.py --help` 验证入口脚本正常工作。

#### 多模式入口脚本验证最佳实践

对于支持多种运行模式的入口脚本（如 `run.py`），仅验证 `--help` 不足以覆盖所有代码路径。推荐以下验证策略：

```bash
# 1. 帮助信息验证（基础）
python run.py --help

# 2. 各模式帮助验证
python run.py basic --help 2>/dev/null || true
python run.py iterate --help 2>/dev/null || true
python run.py plan --help 2>/dev/null || true
python run.py ask --help 2>/dev/null || true
python run.py auto --help 2>/dev/null || true

# 3. dry-run 模式验证（如果支持）
python run.py --dry-run "test task" 2>/dev/null || true

# 4. 配置验证模式
python run.py --validate-config 2>/dev/null || true

# 5. 版本信息验证
python run.py --version 2>/dev/null || true
```

**入口脚本验证检查清单**：

| 验证项 | 命令 | 说明 |
|--------|------|------|
| 帮助信息 | `python run.py --help` | 验证参数解析器初始化 |
| 模块导入 | `python -c "import run"` | 验证脚本本身可导入 |
| 配置加载 | `python run.py --validate-config` | 验证配置文件解析（如支持） |
| 依赖检查 | `python run.py --check-deps` | 验证依赖完整性（如支持） |

> **最佳实践**：入口脚本应提供 `--dry-run` 或 `--validate` 选项，用于在不实际执行任务的情况下验证配置和依赖。

### 3. 完整验证

提交前运行完整检查：

```bash
bash scripts/check_all.sh
```

### 4. 详细说明

遇到问题时，请参阅 [LESSONS_LEARNED.md](LESSONS_LEARNED.md) 获取详细的经验教训和解决方案。

### 5. 审核提交流程参考

完整的审核提交流程请参阅 [LESSONS_LEARNED.md 的审核提交流程章节](LESSONS_LEARNED.md#审核提交流程)，包含：

- **事件循环管理检查点**：涉及异步代码时的必需检查
- **多模式验证必须步骤**：入口脚本多模式初始化验证
- **自动化检查工具整合**：工具链执行顺序和配置
- **常见失败场景排查**：问题诊断和解决方案

### 验证层次对比

不同验证方法覆盖的代码路径和检测能力差异显著，选择正确的验证层次可以有效发现隐藏问题。

#### 验证层次对比表

| 验证层次 | 覆盖范围 | 检测能力 | 执行速度 | 使用场景 |
|----------|----------|----------|----------|----------|
| **静态分析** | 语法、类型、导入声明 | 语法错误、类型不匹配、未声明变量 | 快（<5秒） | 每次保存时 |
| **入口验证 (--help)** | 参数解析器初始化路径 | 顶层导入错误、参数定义错误 | 快（<2秒） | 每次修改后 |
| **多模式验证** | 各模式初始化路径 | 条件导入错误、模式特定依赖 | 中（5-15秒） | 提交前必做 |
| **E2E 测试** | 完整业务流程 | 运行时错误、集成问题、逻辑错误 | 慢（30秒-数分钟） | 重要修改/发布前 |

#### --help 验证的局限性

⚠️ **关键提醒**：`python run.py --help` 仅触发参数解析器初始化，**不会执行模式特定的条件导入**。

```python
# 示例：以下代码在 --help 时不会被执行
def run_plan_mode(args):
    from agents import PlannerAgent  # 条件导入，--help 不触发
    agent = PlannerAgent()
    return agent.plan(args.task)

def run_execute_mode(args):
    from coordinator import Orchestrator  # 条件导入，--help 不触发
    orchestrator = Orchestrator()
    return orchestrator.run(args.task)
```

**必须逐模式验证的原因**：
- `--help` 只验证 argparse 定义正确
- 模式分支内的导入错误（如拼写错误、缺失依赖）不会被发现
- CI 环境可能缺少某些可选依赖，导致特定模式失败

#### 一键多模式验证命令

```bash
# ===== 多模式验证（推荐在提交前执行）=====

# 方式 1：完整多模式验证（串行）
python run.py --help && \
python run.py basic --help 2>/dev/null && \
python run.py iterate --help 2>/dev/null && \
python run.py plan --help 2>/dev/null && \
python run.py ask --help 2>/dev/null && \
python run.py auto --help 2>/dev/null && \
echo "✓ 所有模式验证通过"

# 方式 2：快速验证脚本（项目内置）
bash scripts/check_all.sh --mode-check

# 方式 3：结合导入验证的完整验证
python -c "import agents; import core; import cursor; import coordinator" && \
python run.py --help && \
python run.py basic --help 2>/dev/null && \
python run.py iterate --help 2>/dev/null && \
python run.py plan --help 2>/dev/null && \
python run.py ask --help 2>/dev/null && \
python run.py auto --help 2>/dev/null && \
echo "✓ 导入 + 多模式验证通过"
```

#### 验证层次选择指南

| 修改内容 | 最低验证层次 | 推荐验证层次 |
|----------|--------------|--------------|
| 修复 typo、注释 | 静态分析 | 入口验证 |
| 修改函数实现 | 入口验证 | 多模式验证 + 单元测试 |
| 添加/修改导入 | 多模式验证 | 多模式验证 + 导入测试 |
| 修改入口逻辑 | 多模式验证 | E2E 测试 |
| 重构模块结构 | E2E 测试 | 完整测试套件 |
| 添加新模式/子命令 | 多模式验证 | E2E 测试 + 手动验证 |

---

## 运行时验证 vs 静态分析

理解运行时验证和静态分析的区别，有助于选择正确的验证策略。

### 验证方法对比

| 方面 | 运行时验证 | 静态分析 |
|------|-----------|----------|
| **执行时机** | 代码实际运行时 | 代码编写/提交时 |
| **覆盖范围** | 仅执行的代码路径 | 所有代码（包括未执行路径） |
| **依赖要求** | 需要完整运行环境 | 无需运行环境 |
| **错误类型** | 运行时错误、逻辑错误 | 语法错误、类型错误、导入错误 |
| **速度** | 较慢（需要启动解释器） | 较快 |
| **可靠性** | 高（实际执行） | 中（推断分析） |

### 常用工具分类

#### 静态分析工具

```bash
# 语法检查（不执行代码）
python -m py_compile file.py

# 类型检查
mypy file.py --ignore-missing-imports

# 代码风格
flake8 file.py --max-line-length=120
ruff check file.py

# 导入分析（静态）
python -c "import ast; ast.parse(open('file.py').read())"
```

#### 运行时验证工具

```bash
# 模块导入（实际执行顶层代码）
python -c "import module"

# 入口脚本执行
python run.py --help

# 单元测试
pytest tests/test_module.py -v

# 集成测试
pytest tests/test_e2e_*.py -v
```

### 验证策略推荐

根据修改范围选择验证策略：

| 修改类型 | 推荐验证方法 | 命令 |
|----------|-------------|------|
| 修改函数实现 | 单元测试 | `pytest tests/test_xxx.py -v` |
| 修改模块导出 | 导入验证 + 静态分析 | `python -c "import xxx" && mypy xxx/` |
| 修改入口脚本 | 多模式运行验证 | `python run.py --help && python run.py --dry-run` |
| 添加新依赖 | 导入验证 + 集成测试 | `python -c "import xxx" && pytest tests/test_e2e_*.py` |
| 重构代码结构 | 完整测试套件 | `bash scripts/check_all.sh --full` |

### 验证不足导致的常见问题

| 问题 | 原因 | 预防措施 |
|------|------|----------|
| CI 导入失败但本地通过 | 仅做了静态检查，未做运行时验证 | 使用 `python -c "import xxx"` |
| `--help` 通过但实际运行失败 | 延迟导入错误未被发现 | 增加 `--dry-run` 验证 |
| 类型检查通过但运行时崩溃 | 动态类型、反射调用 | 增加单元测试覆盖 |
| 本地通过但 CI 失败 | 缺少可选依赖 | 使用 CI 相同环境测试 |

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

## 异步代码最佳实践

本项目大量使用异步编程模式。本章节提供事件循环管理、异步函数调用的最佳实践指南。

### 事件循环管理指南

#### 核心原则

1. **一个线程只有一个运行中的事件循环**
2. **避免在已有事件循环的上下文中调用 `asyncio.run()`**
3. **优先使用 `await` 而非创建新的事件循环**

#### 事件循环生命周期

```python
import asyncio

# ✅ 正确：了解事件循环的状态
def check_event_loop():
    try:
        loop = asyncio.get_running_loop()
        print(f"当前有运行中的事件循环: {loop}")
        return True
    except RuntimeError:
        print("当前没有运行中的事件循环")
        return False
```

### 何时使用 asyncio.run() vs 自定义事件循环

#### 使用 `asyncio.run()` 的场景

适用于 **同步入口点** 调用异步代码：

```python
import asyncio
from agents import WorkerAgent

# ✅ 正确：在同步的 main 函数或脚本入口使用
def main():
    agent = WorkerAgent()
    result = asyncio.run(agent.execute_task("task-001"))
    print(result)

if __name__ == "__main__":
    main()
```

```python
# ✅ 正确：CLI 工具的入口点
import click
import asyncio

@click.command()
def cli_command():
    """同步 CLI 命令调用异步函数"""
    asyncio.run(async_main())
```

#### 使用自定义事件循环的场景

适用于需要 **精细控制** 或 **长期运行** 的服务：

```python
import asyncio

# ✅ 正确：需要自定义事件循环配置时
def run_with_custom_loop():
    # 创建自定义事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 配置事件循环（如设置调试模式）
        loop.set_debug(True)
        
        # 运行主协程
        loop.run_until_complete(main_coroutine())
    finally:
        # 确保清理
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
```

```python
# ✅ 正确：在已有事件循环中运行多个任务
async def orchestrator_main():
    """协调器主函数 - 管理多个并发任务"""
    tasks = [
        asyncio.create_task(worker_1()),
        asyncio.create_task(worker_2()),
        asyncio.create_task(worker_3()),
    ]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

#### 避免的错误模式

```python
# ❌ 错误：在异步函数中调用 asyncio.run()
async def bad_async_function():
    # 这会导致 "This event loop is already running" 错误
    result = asyncio.run(another_async_function())
    return result

# ✅ 正确：直接 await
async def good_async_function():
    result = await another_async_function()
    return result
```

```python
# ❌ 错误：在异步上下文中创建新的事件循环
async def bad_nested_loop():
    loop = asyncio.new_event_loop()  # 不要这样做！
    loop.run_until_complete(some_coroutine())

# ✅ 正确：使用 await 或 create_task
async def good_nested_async():
    await some_coroutine()
    # 或并发执行
    task = asyncio.create_task(some_coroutine())
    result = await task
```

### 同步/异步混合代码处理

#### 从同步代码调用异步函数

```python
import asyncio

class SyncWrapper:
    """为异步类提供同步接口"""
    
    def __init__(self):
        self._async_client = AsyncClient()
    
    def sync_method(self, param: str) -> str:
        """同步包装方法"""
        # 检查是否已有事件循环
        try:
            loop = asyncio.get_running_loop()
            # 在已有循环中 - 不能使用 asyncio.run()
            # 使用 run_coroutine_threadsafe 或重构代码
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self._async_client.async_method(param),
                loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # 没有运行中的循环 - 可以安全使用 asyncio.run()
            return asyncio.run(self._async_client.async_method(param))
```

#### 从异步代码调用同步函数

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def call_sync_from_async():
    """在异步上下文中调用阻塞的同步函数"""
    loop = asyncio.get_running_loop()
    
    # 使用线程池执行阻塞操作
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            blocking_sync_function,
            arg1, arg2
        )
    return result

def blocking_sync_function(arg1, arg2):
    """模拟阻塞的同步操作（如 I/O、CPU 密集计算）"""
    import time
    time.sleep(1)  # 阻塞操作
    return f"Result: {arg1}, {arg2}"
```

### 并发控制

#### 使用 Semaphore 限制并发数

```python
import asyncio

async def limited_concurrency_example():
    """限制并发任务数量"""
    semaphore = asyncio.Semaphore(5)  # 最多 5 个并发
    
    async def limited_task(task_id: int):
        async with semaphore:
            print(f"Task {task_id} 开始")
            await asyncio.sleep(1)
            print(f"Task {task_id} 完成")
            return task_id
    
    # 创建 20 个任务，但同时只有 5 个在运行
    tasks = [limited_task(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    return results
```

#### 使用超时控制

```python
import asyncio

async def with_timeout_example():
    """为异步操作添加超时"""
    try:
        # 方式 1: asyncio.timeout (Python 3.11+)
        async with asyncio.timeout(5.0):
            result = await long_running_task()
            return result
    except asyncio.TimeoutError:
        print("操作超时")
        return None

async def with_wait_for_example():
    """使用 wait_for（兼容旧版本）"""
    try:
        # 方式 2: asyncio.wait_for (Python 3.7+)
        result = await asyncio.wait_for(
            long_running_task(),
            timeout=5.0
        )
        return result
    except asyncio.TimeoutError:
        print("操作超时")
        return None
```

### 常见异步问题和解决方案

#### 问题 1：RuntimeError: This event loop is already running

**现象**
```
RuntimeError: This event loop is already running
```

**原因**：在已运行的事件循环中调用 `asyncio.run()`

**解决方案**
```python
# ❌ 错误代码
async def handler():
    result = asyncio.run(other_async())  # 错误！

# ✅ 解决方案 1：直接 await
async def handler():
    result = await other_async()

# ✅ 解决方案 2：使用 nest_asyncio（仅用于特殊场景如 Jupyter）
import nest_asyncio
nest_asyncio.apply()  # 允许嵌套事件循环（不推荐生产使用）
```

#### 问题 2：RuntimeWarning: coroutine was never awaited

**现象**
```
RuntimeWarning: coroutine 'xxx' was never awaited
```

**原因**：调用异步函数但未使用 `await`

**解决方案**
```python
# ❌ 错误
async def main():
    result = fetch_data()  # 忘记 await，result 是协程对象

# ✅ 正确
async def main():
    result = await fetch_data()  # 正确等待结果
```

#### 问题 3：Task was destroyed but it is pending

**现象**
```
Task was destroyed but it is pending!
```

**原因**：任务未完成就被取消或程序退出

**解决方案**
```python
import asyncio

async def graceful_shutdown():
    """优雅关闭所有任务"""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
    
    # 等待所有任务完成取消
    await asyncio.gather(*tasks, return_exceptions=True)

# 在程序退出时调用
async def main():
    try:
        await run_application()
    finally:
        await graceful_shutdown()
```

#### 问题 4：异步上下文管理器未正确关闭

**现象**
```
ResourceWarning: unclosed resource
```

**解决方案**
```python
import asyncio
from contextlib import asynccontextmanager

# ✅ 正确：使用 async with 确保资源关闭
async def proper_resource_handling():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# ✅ 正确：手动管理时确保 finally 中关闭
async def manual_handling():
    session = aiohttp.ClientSession()
    try:
        response = await session.get(url)
        return await response.text()
    finally:
        await session.close()  # 确保关闭
```

#### 问题 5：测试中的异步代码

**解决方案**
```python
import pytest
import asyncio

# ✅ 正确：使用 pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

# ✅ 正确：自定义 fixture
@pytest.fixture
def event_loop():
    """为测试创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ✅ 正确：在同步测试中使用 asyncio.run
def test_sync_wrapper():
    result = asyncio.run(async_function())
    assert result == expected
```

### 项目中的异步模式参考

本项目中的异步代码示例位置：

| 模块 | 文件 | 说明 |
|------|------|------|
| `agents/` | `worker.py`, `planner.py` | Agent 异步执行方法 |
| `cursor/` | `client.py`, `streaming.py` | 异步 HTTP 客户端 |
| `coordinator/` | `orchestrator.py` | 任务协调和并发控制 |
| `knowledge/` | `fetcher.py` | 异步网页获取 |

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

### 快速验证命令清单

提交代码前，按以下顺序执行验证命令：

```bash
# ===== 必选验证（每次提交必做）=====

# 1. 语法检查（静态分析）
python -m py_compile run.py agents/*.py core/*.py

# 2. 核心模块导入验证（运行时）
python -c "import agents; import core; import cursor; import coordinator"

# 3. 入口脚本验证（运行时）
python run.py --help

# ===== 推荐验证（重要修改时执行）=====

# 4. 完整导入测试
pytest tests/test_imports.py tests/test_e2e_imports.py -v

# 5. 类型检查
mypy agents/ core/ --ignore-missing-imports

# 6. 单元测试
pytest tests/ -v --tb=short

# ===== 完整验证（发布前执行）=====

# 7. 全量检查脚本
bash scripts/check_all.sh --full

# 8. 安全审计
pip-audit
```

**一键验证命令**：

```bash
# 快速验证（约 10 秒）
python -c "import agents; import core" && python run.py --help && echo "✓ 验证通过"

# 完整验证（约 1-2 分钟）
bash scripts/check_all.sh --full
```
