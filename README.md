# CursorX - 多 Agent 协作系统

基于 **Planner-Worker-Reviewer** 模式的多 Agent 系统框架，使用 Cursor CLI (agent) 作为底层执行引擎。

## 特性

- **多进程架构**: 使用 `multiprocessing.Process` 实现真正的并行执行
- **角色分工**: Planner 规划、Worker 执行、Reviewer 评审
- **Cursor CLI 集成**: 完整支持 agent CLI 参数 (`-p`, `--model`, `--force`, `--output-format`)
- **语义搜索**: 基于向量索引的代码库语义搜索，增强上下文理解
- **知识库**: 网页抓取、内容解析、向量化存储
- **MCP 支持**: 自动发现和管理 MCP 服务器
- **流式输出**: 支持 `stream-json` 格式实时进度跟踪
- **Hooks 系统**: 支持文件编辑、Shell 命令等事件钩子
- **GitHub Actions**: 自动代码审查和 CI 修复工作流

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                            │
│                     (协调器/编排器)                           │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────┐          ┌──────────────────────┐
│   Planner Agent      │          │   Reviewer Agent     │
│   (规划者)           │          │   (评审者)           │
│   Model: gpt-5.2-high│          │   Model: opus-4.5    │
│   Mode: readonly     │          │   Mode: readonly     │
└──────────┬───────────┘          └──────────────────────┘
           │                                  ▲
           │ Tasks                            │ Review
           ▼                                  │
┌──────────────────────────────────────────────────────────────┐
│                      Task Queue                              │
│                     (任务队列)                                │
└──────────────────────────┬───────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Worker Agent   │ │  Worker Agent   │ │  Worker Agent   │
│  (执行者 1)     │ │  (执行者 2)     │ │  (执行者 N)     │
│  Model: opus-4.5│ │  Model: opus-4.5│ │  Model: opus-4.5│
│  Mode: --force  │ │  Mode: --force  │ │  Mode: --force  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 快速开始

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/onlyfeng/cursorx.git
cd cursorx

# 创建 Python 环境
conda create -n cursorx python=3.11
conda activate cursorx

# 安装依赖
pip install -r requirements.txt

# 安装 Cursor CLI
curl https://cursor.com/install -fsS | bash

# 设置 API 密钥
export CURSOR_API_KEY=your_api_key_here
```

### 运行系统

使用统一入口脚本 `run.py`，支持自然语言任务描述：

```bash
# 自动模式（Agent 分析任务选择最佳运行模式）
python run.py "实现一个 REST API 服务"
python run.py "启动自我迭代，跳过在线更新，优化代码"
python run.py "使用多进程并行重构 src 目录下的代码"

# 显式指定模式
python run.py --mode basic "实现功能"
python run.py --mode mp "重构代码" --workers 5
python run.py --mode knowledge "查询 CLI 参数用法"
python run.py --mode iterate --skip-online "更新支持"

# 无限迭代直到完成
python run.py "持续优化" --max-iterations MAX
```

### 运行模式

| 模式 | 说明 | 别名 |
|------|------|------|
| `basic` | 基本协程模式（规划-执行-审核） | `default`, `simple` |
| `mp` | 多进程模式（并行执行） | `multiprocess`, `parallel` |
| `knowledge` | 知识库增强模式（自动搜索相关文档） | `kb`, `docs` |
| `iterate` | 自我迭代模式（检查更新、更新知识库） | `self-iterate`, `self`, `update` |
| `auto` | 自动分析模式（Agent 分析任务选择模式） | `smart` |

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task` | 任务描述（支持自然语言） | (必填) |
| `--mode, -M` | 运行模式 | `auto` |
| `-d, --directory` | 工作目录 | `.` |
| `-w, --workers` | Worker 数量 | 3 |
| `-m, --max-iterations` | 最大迭代次数（MAX/-1 表示无限迭代） | 10 |
| `--strict` | 严格评审模式 | False |
| `-v, --verbose` | 详细输出 | False |
| `--skip-online` | [iterate] 跳过在线文档检查 | False |
| `--dry-run` | [iterate] 仅分析不执行 | False |
| `--use-knowledge` | [knowledge] 使用知识库上下文 | False |
| `--stream-log` | 启用流式日志 | False |

### 无限迭代模式

使用 `MAX`、`-1` 或 `0` 作为 `--max-iterations` 参数值，系统将持续迭代直到：
- 审核通过（目标完成）
- 用户按 `Ctrl+C` 中断

```bash
# 无限迭代直到完成
python run.py "实现完整功能" --max-iterations MAX

# 等价写法
python run.py "实现完整功能" -m -1
```

### 自然语言任务描述

统一入口支持自然语言描述任务，Agent 会自动分析并选择最佳模式：

```bash
# 自动识别为 iterate 模式
python run.py "启动自我迭代，跳过在线更新"

# 自动识别为 mp 模式
python run.py "使用 5 个 worker 并行重构代码"

# 自动识别为 knowledge 模式
python run.py "搜索知识库查找 CLI 参数用法"
```

## 项目结构

```
cursorx/
├── run.py                  # 统一入口脚本（支持自然语言任务）
├── config.yaml             # 系统配置
├── requirements.txt        # Python 依赖
├── AGENTS.md               # Agent 规则文档
├── mcp.json                # MCP 服务器配置
│
├── agents/                 # Agent 实现
│   ├── planner.py          # 规划者 Agent
│   ├── worker.py           # 执行者 Agent
│   ├── reviewer.py         # 评审者 Agent
│   └── *_process.py        # 多进程版本
│
├── coordinator/            # 协调器
│   ├── orchestrator.py     # 协程编排器
│   ├── orchestrator_mp.py  # 多进程编排器
│   └── worker_pool.py      # Worker 池
│
├── cursor/                 # Cursor CLI 集成
│   ├── client.py           # Agent 客户端
│   ├── mcp.py              # MCP 管理
│   └── streaming.py        # 流式输出处理
│
├── indexing/               # 语义索引
│   ├── chunker.py          # 代码分块
│   ├── embedding.py        # 向量嵌入
│   ├── vector_store.py     # 向量存储
│   └── search.py           # 语义搜索
│
├── knowledge/              # 知识库
│   ├── fetcher.py          # 网页抓取
│   ├── parser.py           # 内容解析
│   ├── storage.py          # 存储管理
│   └── manager.py          # 知识库管理
│
├── .cursor/                # Cursor 配置
│   ├── agents/             # 子代理配置
│   ├── skills/             # 技能配置
│   ├── rules/              # 规则文件
│   ├── hooks/              # 钩子脚本
│   ├── cli.json            # 权限配置
│   └── hooks.json          # 钩子配置
│
├── scripts/                # 运行脚本
│   ├── run_basic.py        # 基本模式入口
│   ├── run_mp.py           # 多进程模式入口
│   ├── run_knowledge.py    # 知识库增强模式入口
│   ├── run_iterate.py      # 自我迭代模式入口
│   ├── code_review.sh      # 代码审查
│   ├── stream_progress.sh  # 流式进度
│   ├── manage_index.sh     # 索引管理
│   └── knowledge_base.sh   # 知识库操作
│
└── .github/workflows/      # GitHub Actions
    ├── cursor-code-review.yml
    └── cursor-fix-ci.yml
```

## Agent 角色

### Planner (规划者)
- **模型**: `gpt-5.2-high`
- **职责**: 分析目标、探索代码库、分解任务
- **模式**: 只读（不修改文件）
- **支持**: 语义搜索增强

### Worker (执行者)
- **模型**: `opus-4.5-thinking`
- **职责**: 执行具体编码任务
- **模式**: `--force`（可修改文件）
- **支持**: 上下文搜索增强

### Reviewer (评审者)
- **模型**: `opus-4.5-thinking`
- **职责**: 评估完成度、决定是否继续迭代
- **模式**: 只读（不修改文件）

## 语义搜索

系统集成了基于向量索引的语义搜索功能：

```bash
# 构建索引
python -m indexing.cli build --path /path/to/project

# 搜索代码
python -m indexing.cli search "用户认证流程"

# 查看索引状态
python -m indexing.cli status
```

### 配置

在 `config.yaml` 中配置索引参数：

```yaml
indexing:
  enabled: true
  model: all-MiniLM-L6-v2
  persist_path: .cursor/vector_index/
  chunk_size: 500
  search:
    top_k: 10
    min_score: 0.3
```

## 知识库

支持从网页抓取内容构建知识库：

```bash
# 添加网页
python scripts/knowledge_cli.py add https://example.com

# 搜索知识库
python scripts/knowledge_cli.py search "关键词"

# 列出所有文档
python scripts/knowledge_cli.py list
```

## 自我迭代

通过统一入口或直接脚本，支持自动检查在线文档更新、更新知识库并自我迭代：

```bash
# 通过统一入口（推荐）
python run.py --mode iterate "增加对新斜杠命令的支持"
python run.py --mode iterate --skip-online "优化 CLI 参数处理"
python run.py "启动自我迭代，跳过在线更新"  # 自动识别模式

# 直接使用脚本
python scripts/run_iterate.py "增加对新斜杠命令的支持"
python scripts/run_iterate.py --skip-online "优化 CLI 参数处理"
python scripts/run_iterate.py --dry-run "分析改进点"
```

### 自我迭代参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `requirement` | 额外需求（可选） | (空) |
| `--skip-online` | 跳过在线文档检查 | False |
| `--changelog-url` | Changelog URL | cursor.com/cn/changelog |
| `--dry-run` | 仅分析不执行 | False |
| `--max-iterations` | 最大迭代次数（MAX/-1 表示无限迭代） | 5 |
| `--workers` | Worker 池大小 | 3 |
| `--force-update` | 强制更新知识库 | False |
| `--orchestrator` | 编排器类型: `mp`=多进程(默认), `basic`=协程模式 | `mp` |
| `--no-mp` | 禁用多进程编排器，使用基本协程编排器 | False |
| `--execution-mode` | 执行模式: `cli`/`auto`/`cloud`（`cloud`/`auto` 强制使用 basic 编排器） | `cli` |
| `--auto-commit` | 迭代完成后自动提交代码更改 | False |
| `--auto-push` | 自动推送到远程仓库（需配合 `--auto-commit`） | False |
| `--commit-per-iteration` | 每次迭代都提交（默认仅在全部完成时提交） | False |
| `-v, --verbose` | 详细输出 | False |

### 多进程并行执行

自我迭代模式 **默认启用多进程并行执行**（`MultiProcessOrchestrator`），可显著提升任务执行效率：

```bash
# 默认使用多进程编排器（推荐，execution-mode=cli 时）
python run.py --mode iterate "优化代码"
python scripts/run_iterate.py "增加新功能支持"

# 指定使用多进程编排器（显式）
python run.py --mode iterate --orchestrator mp "任务描述"
python scripts/run_iterate.py --orchestrator mp "任务描述"

# 禁用多进程，使用协程编排器
python run.py --mode iterate --no-mp "任务描述"
python scripts/run_iterate.py --orchestrator basic "任务描述"

# 使用 Cloud/Auto 执行模式（自动使用 basic 编排器）
python scripts/run_iterate.py --execution-mode auto "任务描述"
python scripts/run_iterate.py --execution-mode cloud "长时间分析任务"
```

**注意**: 当 `--execution-mode` 为 `cloud` 或 `auto` 时，系统会 **强制使用 basic 编排器**，因为 Cloud/Auto 执行模式不支持多进程编排器。

### 回退策略

当多进程编排器（MP）启动失败时，系统会 **自动回退** 到基本协程编排器，确保任务继续执行：

| 回退场景 | 说明 |
|----------|------|
| 启动超时 | MP 进程创建超时，回退到协程模式 |
| 进程创建失败 | 系统资源不足等 OSError，回退到协程模式 |
| 运行时错误 | 事件循环问题等 RuntimeError，回退到协程模式 |
| 其他异常 | 未预期的启动错误，回退到协程模式 |

回退时会输出警告信息：

```
⚠ MP 编排器启动失败: <错误原因>
⚠ 正在回退到基本协程编排器...
```

### 工作流程

```
用户输入需求 → 分析在线文档更新 → 更新知识库 → 总结迭代内容 → 启动 Agent 执行
```

1. **分析 Changelog**: 从 Cursor 官方 Changelog 获取最新更新
2. **更新知识库**: 将新文档保存到本地知识库
3. **加载上下文**: 从知识库搜索相关文档作为上下文
4. **构建目标**: 整合用户需求和更新内容，生成迭代目标
5. **执行迭代**: 调用 Agent 系统执行代码更新

## Cursor CLI 集成

### 非交互模式

```bash
# 基本用法
agent -p "分析代码结构" --model gpt-5

# 修改文件
agent -p "重构代码" --force

# JSON 输出
agent -p "生成报告" --output-format json

# 流式输出
agent -p "分析项目" --output-format stream-json --stream-partial-output
```

### MCP 服务器

```bash
# 列出服务器
agent mcp list

# 查看工具
agent mcp list-tools fetch

# 启用/禁用
agent mcp enable playwright
agent mcp disable playwright
```

## 配置文件

### `.cursor/cli.json` - 权限配置

```json
{
  "permissions": {
    "allow": ["Shell(git)", "Read(**/*)", "Write(src/**)"],
    "deny": ["Shell(rm -rf)", "Read(.env*)"]
  }
}
```

### `.cursor/hooks.json` - 钩子配置

```json
{
  "version": 1,
  "hooks": {
    "afterFileEdit": [{ "command": "./hooks/format.sh" }],
    "beforeShellExecution": [{ "command": "./hooks/audit.sh" }]
  }
}
```

### `mcp.json` - MCP 服务器配置

```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"]
    }
  }
}
```

## GitHub Actions

项目包含两个工作流：

1. **cursor-code-review.yml**: PR 自动代码审查
2. **cursor-fix-ci.yml**: CI 失败自动修复

配置步骤：
1. 在仓库 Settings > Secrets 中添加 `CURSOR_API_KEY`
2. 工作流会在 PR 提交时自动运行

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码检查

```bash
# 类型检查
mypy .

# 代码格式化
black .
isort .
```

## 许可证

MIT License

## 链接

- [Cursor CLI 文档](https://cursor.com/cn/docs/cli/overview)
- [GitHub 仓库](https://github.com/onlyfeng/cursorx)
