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

```bash
# 基本用法（协程模式）
python main.py "实现一个 REST API 服务"

# 指定工作目录
python main.py "重构代码" -d /path/to/project

# 多进程模式
python main_mp.py "添加单元测试" --workers 5

# 严格评审模式
python main.py "优化性能" --strict --max-iterations 5
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `goal` | 要完成的目标 | (必填) |
| `-d, --directory` | 工作目录 | `.` |
| `-w, --workers` | Worker 数量 | 3 |
| `-m, --max-iterations` | 最大迭代次数 | 10 |
| `--strict` | 严格评审模式 | False |
| `--no-sub-planners` | 禁用子规划者 | False |
| `-v, --verbose` | 详细输出 | False |

## 项目结构

```
cursorx/
├── main.py                 # 主入口（协程模式）
├── main_mp.py              # 多进程模式入口
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
├── scripts/                # 自动化脚本
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
