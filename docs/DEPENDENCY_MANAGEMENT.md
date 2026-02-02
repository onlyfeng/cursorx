# 依赖管理指南

本文档描述项目的依赖管理策略、工具使用和标准流程。

## 目录

- [依赖管理架构](#依赖管理架构)
- [工具安装与配置](#工具安装与配置)
- [依赖操作流程](#依赖操作流程)
- [解决依赖冲突](#解决依赖冲突)
- [本地开发环境设置](#本地开发环境设置)
- [CI/CD 集成](#cicd-集成)

---

## 依赖管理架构

### 核心原则

- **`requirements.txt` = 核心运行时锁定文件**：仅包含运行 Agent 系统必需的最小依赖
- **全量能力通过 extras 或 optional requirements 安装**：ML、浏览器、开发工具等按需安装

### 文件结构

```
cursorx/
├── pyproject.toml             # 项目元数据和依赖定义（PEP 621）
├── requirements.in            # 核心依赖源文件
├── requirements-dev.in        # 开发依赖源文件
├── requirements-test.in       # 测试依赖源文件
├── requirements-test-ml.in    # ML 测试依赖源文件（不进入默认 CI）
├── requirements-optional.txt  # 可选依赖（ML/浏览器等，手动维护）
├── requirements.txt           # 锁定的核心依赖（sync-deps.sh 自动生成）
├── requirements-dev.txt       # 锁定的开发依赖（sync-deps.sh 自动生成）
├── requirements-test.txt      # 锁定的测试依赖（sync-deps.sh 自动生成）
├── requirements-test-ml.txt   # 锁定的 ML 测试依赖（sync-deps.sh 自动生成，需 Python <= 3.12）
└── scripts/
    └── sync-deps.sh           # 依赖同步脚本（编译/升级/同步）
```

### 依赖分组

| 分组 | 源文件 | 用途 |
|------|--------|------|
| 核心依赖 | `requirements.in` | 运行时必需 |
| 开发依赖 | `requirements-dev.in` | 代码质量、linting |
| 测试依赖 | `requirements-test.in` | pytest、mock 工具 |
| ML 测试依赖 | `requirements-test-ml.in` | 知识库/向量测试 |
| 可选依赖 | `requirements-optional.txt` | ML/向量/浏览器功能 |

### 依赖分层对照表

| 依赖包 | requirements.in | pyproject.toml | 分组 | 进入 CI |
|--------|-----------------|----------------|------|---------|
| **核心运行时** |||||
| asyncio-pool | ✓ | dependencies | core | ✓ |
| aiofiles | ✓ | dependencies | core | ✓ |
| pydantic | ✓ | dependencies | core | ✓ |
| psutil | ✓ | dependencies | core | ✓ |
| loguru | ✓ | dependencies | core | ✓ |
| python-dotenv | ✓ | dependencies | core | ✓ |
| pyyaml | ✓ | dependencies | core | ✓ |
| typing-extensions | ✓ | dependencies | core | ✓ |
| httpx | ✓ | dependencies | core | ✓ |
| websockets | ✓ | dependencies | core | ✓ |
| **网页处理 [web]** |||||
| beautifulsoup4 | - | optional-deps.web | web | ✗ |
| html2text | - | optional-deps.web | web | ✗ |
| lxml | - | optional-deps.web | web | ✗ |
| **ML/向量 [ml]** |||||
| sentence-transformers | - | optional-deps.ml | ml | ✗ |
| chromadb | - | optional-deps.ml | ml | ✗ |
| tiktoken | - | optional-deps.ml | ml | ✗ |
| hnswlib | - | optional-deps.ml | ml | ✗ |
| torch | - | requirements-optional.txt | ml | ✗ |
| numpy | - | requirements-optional.txt | ml | ✗ |
| **浏览器 [browser]** |||||
| playwright | - | requirements-optional.txt | browser | ✗ |
| **开发工具 [dev]** |||||
| mypy | requirements-dev.in | optional-deps.dev | dev | ✓ |
| ruff | requirements-dev.in | optional-deps.dev | dev | ✓ |
| flake8 | requirements-dev.in | optional-deps.dev | dev | ✓ |
| black | requirements-dev.in | optional-deps.dev | dev | ✓ |
| isort | requirements-dev.in | optional-deps.dev | dev | ✓ |
| pre-commit | requirements-dev.in | optional-deps.dev | dev | ✓ |
| pip-audit | requirements-dev.in | optional-deps.dev | dev | ✓ |
| pip-tools | requirements-dev.in | optional-deps.dev | dev | ✓ |
| **测试工具 [test]** |||||
| pytest | requirements-test.in | optional-deps.test | test | ✓ |
| pytest-asyncio | requirements-test.in | optional-deps.test | test | ✓ |
| pytest-cov | requirements-test.in | optional-deps.test | test | ✓ |
| pytest-timeout | requirements-test.in | optional-deps.test | test | ✓ |
| respx | requirements-test.in | optional-deps.test | test | ✓ |

### 依赖类型说明

```python
# pyproject.toml 中的分组
[project.optional-dependencies]
web = [...]      # 网页处理：beautifulsoup4, html2text, lxml
ml = [...]       # ML/向量：sentence-transformers, chromadb, tiktoken, hnswlib
vector = [...]   # 向量搜索（ml 的别名）
dev = [...]      # 开发工具：mypy, ruff, flake8, black, isort, pip-tools
test = [...]     # 测试工具：pytest, pytest-asyncio, respx
all = [...]      # 完整安装
```

### 推荐安装命令

根据使用场景选择对应的安装命令：

#### 场景对照表

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| **CI 构建** | `pip install -r requirements.txt` | 仅核心依赖，最小化安装 |
| **CI 测试** | `pip-sync requirements.txt requirements-test.txt` | 核心 + 测试，精确同步 |
| **本地开发** | `pip-sync requirements.txt requirements-dev.txt requirements-test.txt` | 全量开发环境 |
| **ML 测试** | `pip-sync requirements.txt requirements-test.txt requirements-test-ml.txt` | 包含 ML 依赖（需 Python ≤ 3.12） |

#### 安装命令详解

```bash
# ===== CI 场景 =====

# CI 构建阶段：仅核心依赖
pip install -r requirements.txt

# CI 测试阶段：核心 + 测试（推荐使用 pip-sync 精确同步）
pip-sync requirements.txt requirements-test.txt

# CI lint 阶段：核心 + 开发工具
pip-sync requirements.txt requirements-dev.txt

# ===== 本地开发场景 =====

# 完整开发环境（推荐）
pip-sync requirements.txt requirements-dev.txt requirements-test.txt

# 包含 ML 功能的测试环境（需 Python <= 3.12）
pip-sync requirements.txt requirements-test.txt requirements-test-ml.txt

# ===== pyproject.toml extras 安装（备选方式）=====

pip install -e ".[dev,test]"      # 开发 + 测试
pip install -e ".[ml]"            # ML/向量功能
pip install -e ".[web]"           # 网页处理功能
pip install -e ".[all]"           # 完整安装（所有 extras）
```

> **注意**：`pip-sync` 会精确同步环境（移除不在列表中的包），适合需要干净环境的场景。
> 如果只需添加依赖而不移除，使用 `pip install -r ...` 即可。

---

## 工具安装与配置

### pip-tools（当前使用）

pip-tools 提供 `pip-compile` 和 `pip-sync` 命令，用于依赖锁定和同步。

#### 安装

```bash
pip install pip-tools>=7.3.0
```

#### 验证安装

```bash
pip-compile --version
pip-sync --version
```

### uv（推荐升级）

uv 是 Rust 实现的高性能包管理器，速度比 pip 快 10-100 倍。

#### 安装

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip 安装（备选）
pip install uv
```

#### 验证安装

```bash
uv --version
```

#### 配置（可选）

```bash
# 设置全局缓存目录（多项目共享）
export UV_CACHE_DIR=~/.cache/uv

# 启用更详细的日志
export UV_LOG_LEVEL=debug
```

---

## 依赖操作流程

### 添加新依赖

#### 步骤 1：更新源文件

根据依赖类型，编辑对应的 `.in` 文件：

```bash
# 核心依赖 → requirements.in
echo "new-package>=1.0.0" >> requirements.in

# 开发依赖 → requirements-dev.in
echo "new-dev-tool>=2.0.0" >> requirements-dev.in

# 测试依赖 → requirements-test.in
echo "new-test-lib>=3.0.0" >> requirements-test.in
```

#### 步骤 2：同步 pyproject.toml（重要）

保持 `pyproject.toml` 与 `.in` 文件一致：

```toml
# pyproject.toml
[project]
dependencies = [
    # ... 现有依赖
    "new-package>=1.0.0",  # 添加
]
```

#### 步骤 3：编译锁定文件

```bash
# 使用脚本（推荐）
bash scripts/sync-deps.sh compile

# 或手动编译
pip-compile requirements.in -o requirements.txt --strip-extras
pip-compile requirements-dev.in -o requirements-dev.txt --strip-extras
pip-compile requirements-test.in -o requirements-test.txt --strip-extras
```

#### 步骤 4：安装依赖

```bash
# 使用 pip-sync（精确同步，推荐本地开发使用）
pip-sync requirements.txt requirements-dev.txt requirements-test.txt

# 或使用脚本（同步核心 + 开发 + 测试依赖）
bash scripts/sync-deps.sh sync

# CI 环境推荐分阶段安装（见"推荐安装命令"章节）
```

### 更新依赖

#### 更新单个依赖

```bash
# 编辑 .in 文件中的版本约束
# 然后重新编译
pip-compile requirements.in -o requirements.txt --strip-extras
```

#### 更新所有依赖到最新版本

```bash
# 使用脚本
bash scripts/sync-deps.sh upgrade

# 或手动
pip-compile requirements.in -o requirements.txt --upgrade --strip-extras
pip-compile requirements-dev.in -o requirements-dev.txt --upgrade --strip-extras
pip-compile requirements-test.in -o requirements-test.txt --upgrade --strip-extras
```

### 删除依赖

#### 步骤 1：从源文件移除

```bash
# 编辑 requirements.in，删除对应行
# 同时更新 pyproject.toml
```

#### 步骤 2：重新编译

```bash
bash scripts/sync-deps.sh compile
```

#### 步骤 3：同步环境

```bash
# pip-sync 会自动卸载不再需要的包
pip-sync requirements.txt requirements-dev.txt requirements-test.txt
```

### 使用 uv 的等效操作

```bash
# 添加依赖
uv add new-package

# 添加开发依赖
uv add --dev new-dev-tool

# 添加可选依赖
uv add --optional ml sentence-transformers

# 更新依赖
uv lock --upgrade

# 同步环境
uv sync

# 移除依赖
uv remove package-name
```

---

## 解决依赖冲突

### 诊断冲突

#### 步骤 1：识别冲突

```bash
# 检查依赖一致性
bash scripts/sync-deps.sh check

# 查看详细依赖树
pip-compile requirements.in --verbose

# 使用 pipdeptree（需安装）
pip install pipdeptree
pipdeptree --warn conflict
```

#### 步骤 2：分析冲突原因

常见冲突类型：

```
# 版本不兼容
package-a requires numpy>=1.20
package-b requires numpy<1.20

# 循环依赖
package-a requires package-b
package-b requires package-a

# 平台限制
package requires linux-only-lib  # Windows 不可用
```

### 解决策略

#### 策略 1：调整版本约束

```bash
# 修改 .in 文件，放宽或收紧版本
# 原来
numpy>=1.24.0

# 调整为
numpy>=1.20.0,<2.0.0
```

#### 策略 2：使用替代包

```bash
# 查找功能相似的替代包
# 例如：将 chromadb 替换为 hnswlib
```

#### 策略 3：分离环境

将冲突依赖放入可选依赖组：

```toml
# pyproject.toml
[project.optional-dependencies]
ml = ["sentence-transformers>=2.2.0"]
legacy = ["old-package<2.0.0"]  # 与 ml 冲突，分开安装
```

#### 策略 4：使用 override（高级）

对于 uv：

```toml
# pyproject.toml
[tool.uv]
override-dependencies = [
    "numpy==1.24.0",  # 强制使用特定版本
]
```

### 验证解决方案

```bash
# 清理环境
pip freeze | xargs pip uninstall -y

# 重新安装
pip install -r requirements.txt

# 运行测试
pytest tests/ -v --tb=short

# 检查导入
python -c "from core import BaseAgent; from agents import PlannerAgent; print('OK')"
```

---

## 本地开发环境设置

### 完整设置流程

```bash
# 1. 克隆项目
git clone <repository-url>
cd cursorx

# 2. 创建虚拟环境（推荐 Python 3.11，与锁文件生成基准一致）
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 升级 pip 并安装 pip-tools
pip install --upgrade pip pip-tools>=7.3.0

# 4. 同步完整开发环境
pip-sync requirements.txt requirements-dev.txt requirements-test.txt

# 5. 验证安装
python -c "from core import BaseAgent; print('核心模块加载成功')"
```

> **锁文件生成基准**：所有锁定文件（`requirements*.txt`）基于 Python 3.11 生成。
> 在其他 Python 版本下安装通常可以工作，但如需重新生成锁文件，请使用 Python 3.11。

### 使用 pyproject.toml 安装

```bash
# 仅核心依赖
pip install -e .

# 核心 + 开发
pip install -e ".[dev]"

# 核心 + 测试
pip install -e ".[test]"

# 核心 + ML/向量
pip install -e ".[ml]"

# 完整安装
pip install -e ".[all]"
```

### 使用 uv 设置

```bash
# 安装 Python（如需要）
uv python install 3.11

# 创建虚拟环境并同步
uv sync

# 包含开发依赖
uv sync --dev

# 运行脚本
uv run python scripts/run_basic.py
```

### 环境变量配置

```bash
# 创建 .env 文件
cat > .env << 'EOF'
CURSOR_API_KEY=your_api_key_here
PYTHONPATH=.
EOF

# 加载环境变量
source .env
# 或使用 python-dotenv 自动加载
```

### IDE 配置

#### VSCode / Cursor

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  }
}
```

---

## CI/CD 集成

### GitHub Actions 配置

#### 当前配置（pip）

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: 安装依赖
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: 运行测试
        run: pytest tests/ -v
```

#### 使用 uv（推荐）

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      
      - name: 安装依赖
        run: uv sync --frozen
      
      - name: 运行测试
        run: uv run pytest tests/ -v
```

### 依赖安全审计

```yaml
# .github/workflows/security.yml
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: 安装 pip-audit
        run: pip install pip-audit
      
      - name: 运行安全审计
        run: pip-audit -r requirements.txt
```

### 本地运行 CI 检查

```bash
# 使用脚本运行完整检查
bash scripts/check_all.sh

# 依赖安全审计
bash scripts/sync-deps.sh audit

# 依赖一致性检查
bash scripts/sync-deps.sh check

# 依赖版本检查与冲突检测
python scripts/check_deps.py --verbose
```

### 分层依赖检查

项目使用分层依赖检查策略，确保在不同环境下正确验证依赖：

#### 检查脚本

| 脚本 | 说明 | 检查范围 |
|------|------|----------|
| `scripts/pre_commit_check.py` | 预提交检查 | 默认检查三个文件，可用 `--req-files` 指定 |
| `scripts/check_all.sh` | 完整健康检查 | 检查三个文件存在性及核心/开发/测试依赖 |
| `scripts/check_deps.py` | 依赖冲突检测 | 解析三个文件，检测版本冲突和未声明导入 |

#### 使用示例

```bash
# 预提交检查（默认检查所有三个 requirements 文件）
python scripts/pre_commit_check.py

# 仅检查核心依赖（CI 环境）
python scripts/pre_commit_check.py --req-files requirements.txt

# 检查核心 + 测试依赖
python scripts/pre_commit_check.py --req-files requirements.txt requirements-test.txt

# 依赖冲突检测（自动解析三个文件）
python scripts/check_deps.py --verbose

# 完整健康检查
bash scripts/check_all.sh --full
```

#### 避免误判

分层检查避免了以下常见误判：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `pytest` 报告为未安装 | 仅检查 `requirements.txt` | 同时检查 `requirements-test.txt` |
| `mypy` 报告为未安装 | 仅检查 `requirements.txt` | 同时检查 `requirements-dev.txt` |
| 未声明的导入误报 | 未解析 dev/test 依赖 | `check_deps.py` 解析所有三个文件 |

### Docker 构建

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

CMD ["python", "run.py"]
```

#### 使用 uv 构建

```dockerfile
FROM python:3.11-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 复制项目文件
COPY pyproject.toml uv.lock ./

# 安装依赖（使用锁定文件）
RUN uv sync --frozen --no-dev

COPY . .

CMD ["uv", "run", "python", "run.py"]
```

---

## 常用命令速查

### pip-tools 命令

| 命令 | 说明 |
|------|------|
| `pip-compile requirements.in` | 编译锁定文件 |
| `pip-compile --upgrade` | 升级所有依赖 |
| `pip-compile --upgrade-package pkg` | 升级指定依赖 |
| `pip-sync requirements.txt` | 同步环境 |

### sync-deps.sh 脚本

| 命令 | 说明 |
|------|------|
| `bash scripts/sync-deps.sh compile` | 编译锁定文件 |
| `bash scripts/sync-deps.sh check` | 检查依赖一致性 |
| `bash scripts/sync-deps.sh upgrade` | 升级所有依赖 |
| `bash scripts/sync-deps.sh sync` | 同步安装依赖 |
| `bash scripts/sync-deps.sh audit` | 安全审计 |

### uv 命令

| 命令 | 说明 |
|------|------|
| `uv add package` | 添加依赖 |
| `uv add --dev package` | 添加开发依赖 |
| `uv remove package` | 移除依赖 |
| `uv sync` | 同步环境 |
| `uv lock` | 更新锁定文件 |
| `uv lock --upgrade` | 升级所有依赖 |
| `uv run command` | 在虚拟环境中运行命令 |

---

## 相关文档

- [DEVELOPMENT.md](DEVELOPMENT.md) - 开发指南
- [PACKAGE_MANAGER_COMPARISON.md](PACKAGE_MANAGER_COMPARISON.md) - 包管理工具对比分析
- [pyproject.toml](../pyproject.toml) - 项目配置
