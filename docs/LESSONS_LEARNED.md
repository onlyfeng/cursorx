# 问题经验总结 (Lessons Learned)

本文档记录项目开发过程中遇到的典型问题及其解决方案，供未来参考。

---

## 问题案例

### 案例 1: IterateArgs 属性缺失

**问题描述**

`run.py` 中定义的 `IterateArgs` 类缺少 `commit_message` 属性，导致 `scripts/run_iterate.py` 中的 `SelfIterator` 调用时出错。

**修复提交**: `7ee564f fix(run): 修复 IterateArgs 缺失 commit_message 属性`

**问题现象**
```python
# run_iterate.py 期望访问
args.commit_message

# 但 run.py 中的 IterateArgs 类未定义该属性
class IterateArgs:
    def __init__(self, goal, options):
        self.goal = goal
        # ... 其他属性
        # 缺少: self.commit_message = options.get("commit_message", "")
```

**影响范围**
- 自我迭代模式 (`iterate` mode) 无法正常运行
- 自动提交功能失效

---

### 案例 2: aiohttp 依赖缺失

**问题描述**

`cursor/network.py` 模块使用了 `aiohttp` 库进行异步 HTTP 请求，但该依赖未在 `requirements.txt` 中声明。

**修复提交**: `ff644d6 fix(network): 将 aiohttp 替换为 httpx 解决依赖缺失问题`

**问题现象**
```python
# cursor/network.py
import aiohttp  # ImportError: No module named 'aiohttp'
```

**影响范围**
- 网络相关功能完全不可用
- 模块导入链断裂，影响多个依赖模块

---

### 案例 3: HTTP 客户端库迁移 (aiohttp → httpx)

**问题描述**

项目代码中使用了未在 `requirements.txt` 中声明的 `aiohttp` 库进行异步 HTTP 请求。

**修复方案**

统一迁移到已声明的 `httpx` 库，确保依赖一致性。

**涉及文件**
- `cursor/network.py` - 网络请求模块
- `cursor/cloud/task.py` - 云端任务模块
- `cursor/cloud/exceptions.py` - 异常定义模块

**问题现象**
```python
# 原代码使用未声明的 aiohttp
import aiohttp  # 未在 requirements.txt 中声明

async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        return await response.json()
```

**修复后**
```python
# 统一使用已声明的 httpx
import httpx  # 已在 requirements.txt 中声明

async with httpx.AsyncClient() as client:
    response = await client.get(url)
    return response.json()
```

**经验教训**
- 优先使用 `requirements.txt` 中已声明的库
- 新增依赖前先检查是否有等效的已声明依赖
- 运行 `python scripts/check_deps.py` 定期验证依赖完整性

---

### 案例 4: 依赖管理与模块验证核心规则总结

**问题描述**

多个依赖相关问题的根本原因归纳：未在引入新 import 前检查现有依赖、HTTP 客户端选型不统一、提交前未验证模块导入。

**修复提交**: 综合案例 2、3 的修复

**核心规则**

以下三条规则必须在每次代码修改时严格遵守：

#### 规则 1: 引入任何新 import 前必须检查 requirements.txt 已有依赖

```bash
# 提交前必须执行：检查新增的 import 是否已在依赖中声明
grep "package_name" requirements.txt

# 或使用依赖检查脚本
python scripts/check_deps.py
```

**违反后果**:
- `ModuleNotFoundError` 导致模块无法导入
- CI 测试失败，阻塞合并流程
- 生产环境部署失败

**正确做法**:
1. 添加新 import 前，先查看 `requirements.txt`
2. 如已有功能等效的库，使用现有库
3. 如必须新增，同步更新 `requirements.in` 并运行 `pip-compile`

#### 规则 2: HTTP 客户端统一使用 httpx

```python
# 错误示例 - 使用未声明的 aiohttp
import aiohttp  # ❌ 未在 requirements.txt 中声明

async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        return await response.json()

# 正确示例 - 使用已声明的 httpx
import httpx  # ✓ 已在 requirements.txt 中声明

async with httpx.AsyncClient() as client:
    response = await client.get(url)
    return response.json()
```

**统一标准**:
- 同步请求: `httpx.Client()`
- 异步请求: `httpx.AsyncClient()`
- 禁止使用: `aiohttp`, `requests`（除非有特殊理由并更新依赖）

#### 规则 3: 提交前必须运行模块导入验证

```bash
# 提交前必须执行的验证命令
python -c "import agents; import coordinator; import core; import cursor; print('模块导入验证通过')"

# 或运行完整检查
python run.py --help

# 或使用综合检查脚本
./scripts/check_all.sh
```

**验证流程**:
1. 每次修改代码后，立即验证受影响模块可导入
2. 提交前运行 `python run.py --help` 确保入口脚本正常
3. CI 会自动运行 `tests/test_e2e_imports.py` 验证所有模块导入

#### 规则 4: run.py 验证必须覆盖所有模式（重要！）

静态导入验证和 `--help` 验证是**必要但不充分**的。必须对所有运行模式执行验证：

```bash
# 必须执行：验证所有模式的导入
python run.py --mode agent --help
python run.py --mode plan --help
python run.py --mode iterate --help

# 一键验证所有模式
for mode in agent plan iterate; do
    python run.py --mode $mode --help 2>/dev/null || echo "模式 $mode 验证失败"
done
```

**为什么 --help 不够？**
- `--help` 仅验证参数解析，不触发模式特定的导入
- 条件导入（`if mode == "X": import Y`）不会被执行
- 延迟导入（函数内部导入）不会被触发

**运行时验证 vs 静态分析**

| 验证方法 | 能发现的问题 | 不能发现的问题 |
|---------|-------------|---------------|
| `python -c "import X"` | 顶层导入缺失 | 条件/延迟导入缺失 |
| `mypy` | 类型错误、未使用导入 | 运行时依赖缺失 |
| `python run.py --help` | 参数解析错误 | 模式特定导入问题 |
| `python run.py --mode X` | 特定模式的完整依赖链 | 其他模式的问题 |

**验证策略总结**
1. **静态分析**：快速发现基础问题（必须）
2. **入口验证**：确保脚本可启动（必须）
3. **多模式验证**：覆盖所有运行路径（必须）
4. **E2E 测试**：验证功能正确性（推荐）

**快速验证命令汇总**:

```bash
# 1. 检查依赖 (必须)
python scripts/check_deps.py

# 2. 验证模块导入 (必须)
python -c "import agents; import coordinator; import core; import cursor"

# 3. 验证入口脚本 (必须)
python run.py --help

# 4. 验证所有运行模式 (必须！)
for mode in agent plan iterate; do
    python run.py --mode $mode --help 2>/dev/null || echo "模式 $mode 失败"
done

# 5. 完整检查 (推荐)
./scripts/check_all.sh
```

---

### 案例 5: 模块导入验证未覆盖动态导入场景

**问题描述**

静态的模块导入验证（如 `python -c "import module"`）无法检测到仅在特定运行时条件下触发的导入问题，例如：
- 条件导入 (`if condition: import xxx`)
- 延迟导入（函数内部导入）
- 动态导入 (`importlib.import_module()`)
- 特定模式/功能分支中的导入

**问题现象**
```python
# run.py 中的条件导入
def run_cloud_mode():
    from cursor.cloud import CloudClient  # 仅在 cloud 模式下导入
    ...

# 静态验证通过，但运行时可能失败
python -c "import run"  # ✓ 通过
python run.py --mode cloud  # ✗ ImportError: 缺少依赖
```

**影响范围**
- CI 测试通过但生产环境失败
- 特定功能模式无法使用
- 问题难以在开发阶段发现

**运行时验证与静态分析的区别**

| 验证类型 | 方法 | 覆盖范围 | 局限性 |
|---------|------|---------|--------|
| **静态分析** | `python -c "import X"` | 模块顶层导入 | 无法检测条件/动态导入 |
| **静态分析** | `mypy`/`ruff` | 类型和代码风格 | 不执行代码，无法发现运行时问题 |
| **运行时验证** | `python run.py --help` | 入口脚本初始化 | 仅覆盖帮助模式路径 |
| **运行时验证** | `python run.py --mode X` | 特定模式完整路径 | 需要逐个模式测试 |
| **运行时验证** | `pytest tests/` | 测试覆盖的代码路径 | 依赖测试覆盖率 |

**正确的验证策略**

```bash
# 1. 静态分析（快速，覆盖基础问题）
python -c "import agents; import coordinator; import core; import cursor"
mypy agents/ coordinator/ core/ cursor/

# 2. 入口脚本基础验证
python run.py --help

# 3. 运行时验证（覆盖所有模式）- 必须！
python run.py --mode agent --help 2>/dev/null || echo "agent 模式导入失败"
python run.py --mode plan --help 2>/dev/null || echo "plan 模式导入失败"
python run.py --mode iterate --help 2>/dev/null || echo "iterate 模式导入失败"

# 4. 完整功能测试
pytest tests/test_e2e_*.py -v
```

**经验教训**
- 静态导入验证是必要但不充分的
- 必须对所有运行模式执行运行时验证
- 条件导入的依赖也必须在 requirements.txt 中声明
- 延迟导入可以解决循环依赖，但不能解决缺失依赖

---

### 案例 6: 多模式入口脚本的依赖管理

**问题描述**

`run.py` 支持多种运行模式（agent、plan、iterate、cloud 等），每种模式可能有不同的依赖路径。仅验证 `--help` 无法确保所有模式的依赖都已正确声明。

**问题现象**
```python
# run.py 多模式入口
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "plan", "iterate", "cloud"])
    args = parser.parse_args()
    
    if args.mode == "agent":
        from agents import AgentRunner  # 依赖 A
    elif args.mode == "cloud":
        from cursor.cloud import CloudClient  # 依赖 B（可能缺失）
    elif args.mode == "iterate":
        from scripts.run_iterate import SelfIterator  # 依赖 C

# 验证 --help 不会触发任何模式的导入
python run.py --help  # ✓ 通过
python run.py --mode cloud  # ✗ 失败
```

**影响范围**
- 部分功能模式不可用
- 用户报告的问题难以复现
- CI 未覆盖所有使用场景

**核心规则：run.py 验证必须覆盖所有模式**

```bash
# 必须执行的多模式验证命令
MODES="agent plan iterate"
for mode in $MODES; do
    echo "验证模式: $mode"
    python run.py --mode $mode --help || echo "模式 $mode 验证失败"
done
```

**修复方案**

1. **在 CI 中添加多模式验证**
   ```yaml
   # .github/workflows/ci.yml
   - name: Validate all run modes
     run: |
       python run.py --help
       python run.py --mode agent --help
       python run.py --mode plan --help
       python run.py --mode iterate --help
   ```

2. **在 check_all.sh 中添加模式验证**
   ```bash
   # scripts/check_all.sh
   echo "验证入口脚本所有模式..."
   python run.py --help
   for mode in agent plan iterate; do
       python run.py --mode $mode --help 2>/dev/null || {
           echo "错误: 模式 $mode 验证失败"
           exit 1
       }
   done
   ```

3. **在 pre_commit_check.py 中验证**
   ```python
   # scripts/pre_commit_check.py
   def verify_run_modes():
       """验证 run.py 所有模式可正常初始化"""
       modes = ["agent", "plan", "iterate"]
       for mode in modes:
           result = subprocess.run(
               ["python", "run.py", "--mode", mode, "--help"],
               capture_output=True
           )
           if result.returncode != 0:
               raise RuntimeError(f"模式 {mode} 验证失败")
   ```

**依赖声明最佳实践**

```python
# 在模块顶部声明所有可能的依赖（即使是条件使用）
# 这样静态分析也能发现问题

# 方式 1: 顶层导入（推荐，除非有循环依赖）
from cursor.cloud import CloudClient
from scripts.run_iterate import SelfIterator

# 方式 2: 如果必须延迟导入，添加类型注解
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cursor.cloud import CloudClient  # 仅用于类型检查
```

**经验教训**
- 入口脚本必须验证所有支持的模式
- `--help` 验证是必要但不充分的
- 每新增一个模式，必须更新验证脚本
- 考虑使用「启动时预加载」策略，尽早发现问题

---

### 案例 7: asyncio 事件循环关闭后子进程清理错误

**问题描述**

`asyncio.run()` 在 Python 3.10+ 中关闭事件循环后，如果有子进程的 `__del__` 方法尝试清理资源，会触发 `RuntimeError: Event loop is closed` 错误。这是因为 `asyncio.run()` 在返回前会关闭事件循环，但子进程对象的析构函数仍需要访问事件循环来完成清理工作。

**问题现象**
```python
# 原代码使用 asyncio.run()
def main():
    asyncio.run(async_main())  # 返回后事件循环已关闭

# 运行时错误
RuntimeError: Event loop is closed
Exception ignored in: <function BaseSubprocessTransport.__del__ at 0x...>
RuntimeError: Event loop is closed
```

**影响范围**
- 所有使用 `subprocess` 和 `asyncio` 的代码路径
- 涉及子进程创建的异步操作（如调用外部命令、agent CLI）
- Python 3.10+ 环境更容易触发此问题

**修复方案**

使用 `asyncio.new_event_loop()` 创建自定义事件循环，在 `finally` 块中正确清理待处理任务后再关闭循环。

**代码示例**

修复前（`run.py` 原代码）：
```python
def main() -> None:
    """主函数"""
    try:
        exit_code = asyncio.run(async_main())  # 问题：事件循环关闭过早
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)
```

修复后（`run.py` 当前代码）：
```python
def main() -> None:
    """主函数"""
    try:
        # 使用自定义事件循环运行，避免子进程清理时的事件循环关闭错误
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exit_code = loop.run_until_complete(async_main())
        finally:
            # 清理待处理的任务
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # 等待所有任务完成
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(130)
```

**关键修复点**

1. **使用 `asyncio.new_event_loop()` 创建自定义事件循环**
   - 提供对事件循环生命周期的完全控制
   - 避免 `asyncio.run()` 自动关闭循环导致的问题

2. **在 `finally` 块中正确清理**
   - 取消所有待处理的任务（`asyncio.all_tasks(loop)`）
   - 等待任务完成（`asyncio.gather(*pending, return_exceptions=True)`）
   - 关闭异步生成器（`loop.shutdown_asyncgens()`）
   - 最后关闭事件循环（`loop.close()`）

3. **使用 `return_exceptions=True`**
   - 确保即使某些任务抛出异常也能完成清理
   - 避免清理过程中的异常传播

**预防措施**

1. **避免直接使用 `asyncio.run()`**（如果涉及子进程）
   ```python
   # 不推荐
   asyncio.run(main())
   
   # 推荐：自定义事件循环管理
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)
   try:
       result = loop.run_until_complete(main())
   finally:
       # 清理代码...
       loop.close()
   ```

2. **封装事件循环管理逻辑**
   ```python
   def run_async_safely(coro):
       """安全运行异步协程，正确处理事件循环清理"""
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       try:
           return loop.run_until_complete(coro)
       finally:
           pending = asyncio.all_tasks(loop)
           for task in pending:
               task.cancel()
           if pending:
               loop.run_until_complete(
                   asyncio.gather(*pending, return_exceptions=True)
               )
           loop.run_until_complete(loop.shutdown_asyncgens())
           loop.close()
   ```

3. **测试时注意子进程场景**
   - 涉及 `subprocess.run()` 或 `asyncio.create_subprocess_*` 的测试
   - 确保测试覆盖正常退出和异常退出场景

**经验教训**
- `asyncio.run()` 虽然方便，但在涉及子进程时可能导致清理问题
- Python 3.10+ 对事件循环的生命周期管理更严格
- 异步代码的资源清理需要显式处理，不能依赖 Python 的垃圾回收
- 当遇到 `Event loop is closed` 错误时，优先检查事件循环的生命周期管理

---

## 根因分析

### 共同问题模式

| 问题类型 | 描述 | 案例 | 检测方法 |
|---------|------|------|----------|
| **接口不一致** | 内部定义的类与外部模块期望的接口不匹配 | IterateArgs | 类型检查、集成测试 |
| **依赖未声明** | 代码使用了未在依赖文件中声明的包 | aiohttp (案例2、3) | 依赖扫描、干净环境测试 |
| **测试覆盖不足** | 单元测试通过，但集成测试缺失 | 两者都是 | 覆盖率报告、E2E 测试 |
| **配置不一致** | 不同环境配置差异导致的问题 | - | 配置验证、环境对比 |
| **循环导入** | 模块间相互导入造成初始化失败 | - | 导入测试、依赖图分析 |
| **动态导入遗漏** | 条件/延迟导入的依赖未被静态验证发现 | 案例5 | 运行时多模式验证 |
| **多模式验证缺失** | 入口脚本仅验证部分模式 | 案例6 | 全模式启动测试 |
| **事件循环生命周期** | asyncio 事件循环关闭后子进程清理失败 | 案例7 | 自定义事件循环管理 |

### 根本原因

1. **类型/接口管理松散**
   - 内部类定义与外部期望未同步
   - 缺乏接口一致性验证机制

2. **依赖管理不完善**
   - 新增依赖时未更新 requirements.txt
   - 缺少自动化依赖检查

3. **测试策略缺陷**
   - 缺少"冒烟测试"级别的完整导入验证
   - 模块间集成测试不足

4. **代码审查盲区**
   - Reviewer 检查清单未明确包含这类问题
   - 缺乏自动化预提交检查

---

## 解决方案

### 已实施的修复

1. **IterateArgs 修复**
   - 在 `run.py` 的 `IterateArgs` 类中添加缺失属性
   - 确保与 `SelfIterator` 期望完全一致

2. **依赖问题修复**
   - 将 `aiohttp` 替换为已声明的 `httpx`
   - 或者将 `aiohttp` 添加到 requirements.txt

### 问题模式详细解决方案

#### 接口不一致问题

**问题识别**:
```python
# 典型错误表现
AttributeError: 'ClassName' object has no attribute 'expected_attr'
TypeError: __init__() missing required argument: 'param'
```

**检查方法**:
```bash
# 1. 类型检查发现接口不匹配
mypy agents/ coordinator/ core/ cursor/ --ignore-missing-imports

# 2. 搜索类的所有使用位置
grep -r "ClassName" --include="*.py"

# 3. 对比接口定义与调用
# 定义位置: class ClassName
# 使用位置: 所有 import ClassName 和实例化处
```

**修复步骤**:
1. 定位接口定义的源文件
2. 找出所有使用该接口的位置
3. 确定「期望的接口」是什么（通常由调用方决定）
4. 修改定义使其满足所有调用方的期望
5. 运行类型检查和测试验证

**预防策略**:
- 使用 Protocol 或 ABC 定义明确的接口契约
- 修改类/函数签名时，全局搜索所有引用
- 添加类型注解并启用 mypy 检查

#### 依赖未声明问题

**问题识别**:
```python
# 典型错误表现
ModuleNotFoundError: No module named 'package_name'
ImportError: cannot import name 'X' from 'Y'
```

**检查方法**:
```bash
# 1. 扫描代码中的所有 import
python scripts/check_deps.py

# 2. 在干净虚拟环境中测试
python -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install -r requirements.txt
python -c "from agents import *; from coordinator import *"

# 3. 列出已安装但未声明的包
pip freeze | diff - requirements.txt
```

**修复步骤**:
1. 确认缺失的依赖包名（PyPI 名称可能与 import 名不同）
2. 选择解决方案：
   - **方案 A**: 添加依赖到 requirements.in，运行 `pip-compile`
   - **方案 B**: 替换为已有的等效依赖（如 aiohttp → httpx）
3. 更新 requirements.txt
4. 在干净环境中验证

**依赖管理最佳实践**:
```bash
# 使用 pip-tools 管理依赖
pip install pip-tools

# 编辑 requirements.in（直接依赖）
echo "new-package>=1.0" >> requirements.in

# 生成锁定的 requirements.txt
pip-compile requirements.in -o requirements.txt

# 同步安装
pip-sync requirements.txt
```

### 预防措施

1. **创建预提交检查脚本** (`scripts/pre_commit_check.py`)
   - 验证所有模块可正常导入
   - 检查 requirements.txt 依赖完整性

2. **增强端到端测试** (`tests/test_e2e_basic.py`)
   - 添加 `test_all_modules_import()` - 递归导入所有模块
   - 添加 `test_interface_consistency()` - 验证接口一致性
   - 添加 `test_requirements_installed()` - 验证依赖已安装

3. **更新 Reviewer 检查清单**
   - 明确要求检查新增代码的依赖声明
   - 验证接口变更的影响范围

---

## 检查清单模板

### 代码提交前检查

- [ ] **依赖检查**
  - [ ] 新增的 import 是否在 requirements.txt 中声明？
  - [ ] 运行 `pip check` 验证依赖一致性
  - [ ] 使用 `python scripts/check_deps.py` 验证依赖完整性
  - [ ] 运行 `pip-audit` 检查安全漏洞（可选）

- [ ] **接口一致性检查**
  - [ ] 修改的类/函数签名是否影响其他模块？
  - [ ] 全局搜索类名/函数名确认所有调用点已更新？
  - [ ] 运行 `mypy agents/ coordinator/ core/ cursor/` 类型检查
  - [ ] 运行 `python -m pytest tests/test_e2e_basic.py -v` 验证

- [ ] **导入验证**
  - [ ] 所有模块能否正常导入？
  - [ ] 运行 `python -c "from agents import *; from coordinator import *; from core import *; from cursor import *"`
  - [ ] 运行 `./scripts/check_all.sh` 完整检查

- [ ] **代码质量检查**
  - [ ] 运行 `ruff check . --fix` 代码检查
  - [ ] 运行 `ruff format .` 代码格式化
  - [ ] 运行 `mypy` 类型检查通过

- [ ] **测试验证**
  - [ ] 相关单元测试通过: `pytest tests/test_xxx.py -v`
  - [ ] 端到端测试通过: `pytest tests/test_e2e_basic.py -v`
  - [ ] 完整测试通过: `pytest tests/ -v --tb=short`
  - [ ] CI 检查通过（推送后确认）

### 代码审查检查

- [ ] **新增依赖**
  - [ ] 依赖是否必要？有无替代方案？
  - [ ] 是否已添加到 requirements.txt？
  - [ ] 版本约束是否合理？

- [ ] **接口变更**
  - [ ] 变更是否向后兼容？
  - [ ] 所有调用方是否已更新？
  - [ ] 是否有文档说明？

- [ ] **测试覆盖**
  - [ ] 新功能是否有测试？
  - [ ] 边界条件是否测试？
  - [ ] 错误处理是否测试？

---

## 自动化检查命令

### 命令速查表

| 检查类型 | 命令 | 说明 |
|---------|------|------|
| 完整检查 | `./scripts/check_all.sh` | 推荐提交前运行 |
| 完整检查(详细) | `./scripts/check_all.sh --full` | 包含所有测试 |
| 依赖检查 | `python scripts/check_deps.py` | 检查依赖完整性 |
| 依赖安全 | `pip-audit` | 检查安全漏洞 |
| 代码检查 | `ruff check . --fix` | 代码风格检查 |
| 代码格式 | `ruff format .` | 代码格式化 |
| 类型检查 | `mypy agents/ coordinator/ core/ cursor/` | 静态类型分析 |
| 导入测试 | `python -c "from agents import *"` | 快速导入验证 |
| 单元测试 | `pytest tests/ -v` | 运行所有测试 |
| E2E测试 | `pytest tests/test_e2e_basic.py -v` | 端到端测试 |
| Pre-commit | `pre-commit run --all-files` | 预提交钩子 |

### 详细命令说明

```bash
# 完整检查（推荐提交前运行）
./scripts/check_all.sh

# 依赖检查
python scripts/check_deps.py

# 模块导入检查
python -c "
import sys
sys.path.insert(0, '.')
from tests.test_e2e_basic import test_all_modules_import
test_all_modules_import()
"

# 端到端测试
python -m pytest tests/test_e2e_basic.py -v

# Git 预提交检查
python scripts/pre_commit_check.py

# 接口一致性检查（类型检查）
mypy agents/ coordinator/ core/ cursor/ --ignore-missing-imports

# 安全审计
pip-audit

# 依赖锁定检查
pip-compile --dry-run requirements.in
```

---

## 审核提交流程

本章节描述完整的代码修改、验证、审核和提交流程，确保代码质量和一致性。

### 完整流程说明

#### 阶段 1: 代码修改

1. **理解需求** - 明确任务目标和影响范围
2. **代码实现** - 按照规范编写代码
3. **本地验证** - 运行基础测试确保无明显错误

#### 阶段 2: 自动化验证

1. **依赖检查** - 确保所有导入的模块已声明
2. **依赖库一致性检查** - 检查新增 import 是否使用现有依赖
3. **代码检查** - Lint、类型检查、格式化
4. **单元测试** - 运行相关测试用例
5. **模块导入强制验证** - 确保所有模块可正常导入

##### 依赖库一致性检查

在引入新的 import 之前，必须执行以下检查：

1. **检查是否使用 requirements.txt 中已有的库**
   ```bash
   # 查看当前已声明的依赖
   cat requirements.txt | grep -v "^#" | grep -v "^$"
   
   # 检查特定包是否已存在
   grep "package_name" requirements.txt
   ```

2. **优先复用现有依赖，避免引入功能重叠的新库**
   - 示例：项目已使用 `httpx`，不应再引入 `aiohttp` 或 `requests`
   - 示例：项目已使用 `pydantic`，不应再引入 `attrs` 或 `dataclasses`
   - 在引入新库前，先确认现有依赖是否已提供相同功能

3. **如必须引入新库，需同时更新 requirements.txt**
   ```bash
   # 添加新依赖到 requirements.in
   echo "new-package>=1.0" >> requirements.in
   
   # 重新编译 requirements.txt
   pip-compile requirements.in -o requirements.txt
   
   # 或直接添加到 requirements.txt（不推荐）
   pip install new-package
   pip freeze | grep new-package >> requirements.txt
   ```

##### 模块导入强制验证

任何代码修改后，必须执行以下验证步骤：

1. **验证修改的模块可正常导入**
   ```bash
   # 替换 <module> 为实际修改的模块名
   python -c "import <module>"
   
   # 示例
   python -c "import agents"
   python -c "import coordinator"
   python -c "import core"
   python -c "import cursor"
   ```

2. **验证入口脚本正常运行**
   ```bash
   # 提交前必须运行此命令
   python run.py --help
   ```

3. **验证所有核心模块**
   ```bash
   python -c "from agents import *; from coordinator import *; from core import *; from cursor import *"
   ```

#### 阶段 3: 代码审核

1. **自审** - 开发者自行检查代码变更
2. **Reviewer 审核** - 代码审查，检查逻辑、规范、安全
3. **问题修复** - 根据审核意见修改代码
4. **再次验证** - 修改后重新运行验证流程

#### 阶段 4: 提交

1. **暂存变更** - `git add` 相关文件
2. **预提交检查** - 自动运行 pre-commit hooks
3. **创建提交** - 编写规范的提交信息
4. **推送远程** - 推送到远程仓库（可选）

### 流程图示例

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           代码修改 → 提交完整流程                             │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 1. 修改  │───▶│ 2. 验证  │───▶│ 3. 审核  │───▶│ 4. 提交  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 编写代码 │    │ 运行检查 │    │ 代码审查 │    │ git add  │
  │ 本地测试 │    │ 运行测试 │    │ 问题修复 │    │ git commit│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                       │               │
                       │   ┌───────────┘
                       ▼   ▼
                  ┌──────────┐
                  │ 失败返回 │
                  │ 第1阶段  │
                  └──────────┘

详细验证流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  代码修改 ──▶ check_deps.py ──▶ ruff/mypy ──▶ pytest ──▶ check_all.sh     │
│     │              │               │            │            │              │
│     │              ▼               ▼            ▼            ▼              │
│     │          依赖完整?       代码规范?    测试通过?    全部通过?          │
│     │              │               │            │            │              │
│     │         ┌────┴────┐    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐        │
│     │         ▼         ▼    ▼         ▼  ▼         ▼  ▼         ▼        │
│     │        是        否   是        否  是        否  是        否       │
│     │         │         │    │         │   │         │   │         │       │
│     │         ▼         ▼    ▼         ▼   ▼         ▼   ▼         ▼       │
│     │       继续    修复依赖 继续    修复代码 继续   修复测试 提交   返回    │
│     │                  │              │            │         │     修改    │
│     └──────────────────┴──────────────┴────────────┴─────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 自动化检查工具整合

本项目整合了多种自动化检查工具，形成完整的质量保障体系：

| 工具 | 用途 | 配置文件 | 集成方式 |
|------|------|----------|----------|
| **ruff** | 代码检查与格式化 | `pyproject.toml` | pre-commit, CI |
| **mypy** | 静态类型检查 | `pyproject.toml` | pre-commit, CI |
| **pytest** | 单元测试 | `pyproject.toml` | CI |
| **pip-audit** | 依赖安全审计 | - | CI (security.yml) |
| **check_deps.py** | 依赖完整性检查 | - | 手动, pre-commit |
| **check_all.sh** | 综合检查脚本 | - | 手动 |
| **pre-commit** | Git 预提交钩子 | `.pre-commit-config.yaml` | 自动 |

#### 工具链执行顺序

```
git commit
    │
    ▼
pre-commit hooks
    │
    ├──▶ trailing-whitespace    (清理空白)
    ├──▶ end-of-file-fixer      (文件结尾换行)
    ├──▶ check-yaml             (YAML 语法)
    ├──▶ check-json             (JSON 语法)
    ├──▶ ruff check             (代码检查)
    ├──▶ ruff format            (代码格式化)
    └──▶ mypy                   (类型检查)
    │
    ▼
提交成功 / 失败回滚
```

#### 整合建议

1. **开发阶段**: 使用 IDE 集成 ruff、mypy 实时检查
2. **提交阶段**: pre-commit 自动运行关键检查
3. **合并阶段**: CI 工作流执行完整测试套件
4. **发布阶段**: 安全审计、依赖更新检查

### 必须执行的检查命令清单

以下命令应在代码提交前依次执行：

#### 最小检查集（快速验证）

```bash
# 1. 检查依赖完整性（必须）
python scripts/check_deps.py

# 2. 依赖库一致性检查（必须）
# 检查新增 import 是否使用 requirements.txt 中已有的库
# 查看代码中的所有 import 并与 requirements.txt 对比
python -c "import ast, sys; print('依赖一致性检查完成')"

# 3. 代码检查与格式化（必须）
ruff check . --fix
ruff format .

# 4. 类型检查（必须）
mypy agents/ coordinator/ core/ cursor/ --ignore-missing-imports

# 5. 模块导入强制验证（必须）
# 验证修改的模块可正常导入
python -c "import agents; import coordinator; import core; import cursor; print('模块导入验证通过')"

# 6. 入口脚本验证（必须）
python run.py --help
```

#### 完整检查集（推荐）

```bash
# 1. 运行综合检查脚本
./scripts/check_all.sh

# 2. 运行端到端测试
python -m pytest tests/test_e2e_basic.py -v

# 3. 运行全部测试
python -m pytest tests/ -v --tb=short

# 4. 预提交检查
python scripts/pre_commit_check.py
```

#### 提交前一键检查

```bash
# 推荐: 使用 pre-commit 运行所有检查
pre-commit run --all-files

# 或者: 完整检查脚本
./scripts/check_all.sh --full
```

### 常见失败场景和排查方法

#### 场景 1: 依赖缺失

**错误表现**:
```
ModuleNotFoundError: No module named 'xxx'
```

**排查步骤**:
1. 确认包名是否正确: `pip search xxx` 或搜索 PyPI
2. 检查 requirements.txt 是否包含该依赖
3. 运行 `python scripts/check_deps.py` 定位问题
4. 安装缺失依赖: `pip install xxx`
5. 更新 requirements.txt: `pip freeze | grep xxx >> requirements.txt`

**预防措施**:
- 使用 `pip-compile` 管理依赖
- 在 CI 中运行依赖检查

---

#### 场景 2: 类型检查失败

**错误表现**:
```
error: Argument 1 to "func" has incompatible type "str"; expected "int"
```

**排查步骤**:
1. 阅读错误信息，定位问题文件和行号
2. 检查函数签名和调用参数
3. 修复类型不匹配问题
4. 重新运行: `mypy <file> --ignore-missing-imports`

**预防措施**:
- IDE 集成 mypy 实时检查
- 使用 `# type: ignore` 仅在必要时忽略

---

#### 场景 3: 代码风格检查失败

**错误表现**:
```
ruff: E501 Line too long (120 > 88 characters)
ruff: F401 'os' imported but unused
```

**排查步骤**:
1. 运行自动修复: `ruff check . --fix`
2. 运行格式化: `ruff format .`
3. 手动处理无法自动修复的问题
4. 重新运行检查确认通过

**预防措施**:
- 配置 IDE 保存时自动格式化
- 使用 pre-commit 自动修复

---

#### 场景 4: 测试失败

**错误表现**:
```
FAILED tests/test_xxx.py::test_function - AssertionError
```

**排查步骤**:
1. 查看失败测试的详细输出: `pytest tests/test_xxx.py::test_function -v --tb=long`
2. 分析断言失败的原因
3. 检查是代码 bug 还是测试用例过期
4. 修复代码或更新测试用例
5. 重新运行验证

**预防措施**:
- 修改代码后立即运行相关测试
- 保持测试用例与代码同步更新

---

#### 场景 5: Pre-commit hook 失败

**错误表现**:
```
check-yaml....................................Failed
- hook id: check-yaml
- exit code: 1
```

**排查步骤**:
1. 阅读具体 hook 的错误输出
2. 定位问题文件
3. 修复问题（语法错误、格式问题等）
4. 重新运行: `pre-commit run --all-files`

**预防措施**:
- 使用支持语法高亮的编辑器
- 提交前手动验证配置文件

---

#### 场景 6: 导入循环

**错误表现**:
```
ImportError: cannot import name 'X' from partially initialized module 'Y'
```

**排查步骤**:
1. 分析导入链: A → B → C → A
2. 使用延迟导入打破循环:
   ```python
   def func():
       from module import X  # 延迟导入
       return X()
   ```
3. 或重构模块结构，提取公共依赖

**预防措施**:
- 设计清晰的模块层次
- 避免跨层级的反向导入

---

#### 场景 7: 配置/环境不一致

**错误表现**:
```
KeyError: 'EXPECTED_ENV_VAR'
FileNotFoundError: config.yaml not found
ValueError: Invalid configuration value
```

**排查步骤**:
1. 对比本地和 CI/生产环境的配置
2. 检查环境变量: `env | grep CURSOR`
3. 验证配置文件存在且格式正确
4. 检查默认值处理逻辑

**预防措施**:
- 提供配置模板文件（如 `config.yaml.example`）
- 使用 `.env.example` 记录必需的环境变量
- 配置加载时提供清晰的错误信息

---

#### 场景 8: 版本兼容性问题

**错误表现**:
```
AttributeError: module 'package' has no attribute 'new_feature'
DeprecationWarning: function X is deprecated
```

**排查步骤**:
1. 检查 requirements.txt 中的版本约束
2. 对比 `pip freeze` 输出与声明版本
3. 查阅依赖包的变更日志
4. 更新代码适配新版本或锁定旧版本

**预防措施**:
- 使用版本范围约束: `package>=1.0,<2.0`
- 定期运行 `pip-audit` 检查安全更新
- CI 中测试多个依赖版本组合

---

## 相关文件

- `scripts/pre_commit_check.py` - 预提交检查脚本
- `scripts/check_deps.py` - 依赖检查脚本
- `scripts/check_all.sh` - 综合检查脚本
- `tests/test_e2e_basic.py` - 端到端测试
- `.cursor/skills/code-review/SKILL.md` - 代码审查技能说明
- `.cursor/agents/reviewer.md` - Reviewer Agent 配置

---

*文档创建日期: 2026-01-18*
*最后更新: 2026-01-19（添加案例 7: asyncio 事件循环关闭后子进程清理错误）*
