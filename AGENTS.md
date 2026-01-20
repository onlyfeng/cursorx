# 多 Agent 系统规则

[![CI](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/ci.yml)
[![Lint](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/lint.yml)
[![Security](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/security.yml/badge.svg)](https://github.com/YOUR_USERNAME/cursorx/actions/workflows/security.yml)

本项目是一个基于规划者-执行者模式的多 Agent 系统。

## 安装与配置

```bash
# 安装 Cursor CLI
curl https://cursor.com/install -fsS | bash

# 设置 API 密钥
export CURSOR_API_KEY=your_api_key_here

# 验证安装
agent -p "Hello"

# 列出可用模型
agent models
agent --list-models

# 交互模式
/models
```

> **注意**: `agent` 是主要 CLI 入口，`cursor-agent` 保留为向后兼容别名

## CLI 参数完整列表

参考: https://cursor.com/cn/docs/cli/reference/parameters

### 全局选项

| 选项 | 描述 |
|------|------|
| `-v, --version` | 输出版本号 |
| `-a, --api-key <key>` | API 密钥（也可使用 CURSOR_API_KEY 环境变量） |
| `-p, --print` | 非交互模式，输出到控制台 |
| `--output-format <format>` | 输出格式: text / json / stream-json |
| `--stream-partial-output` | 增量流式输出（配合 stream-json） |
| `-b, --background` | 后台模式 |
| `--fullscreen` | 全屏模式 |
| `--resume [chatId]` | 恢复聊天会话 |
| `-m, --model <model>` | 使用的模型 |
| `--mode <mode>` | 运行模式: plan（规划模式）/ ask（问答模式）/ agent（默认，完整代理模式） |
| `--list-models` | 列出所有可用模型 |
| `-f, --force` | 强制允许修改文件 |
| `-h, --help` | 显示帮助 |

### 运行模式详解（--mode）

| 模式 | 描述 | 只读保证 | 适用场景 |
|------|------|----------|----------|
| `agent` | 完整代理模式（默认） | 否，可修改文件 | 代码编写、重构、功能实现 |
| `plan` | 规划模式 | **是，强制只读** | 任务分析、代码审查、架构规划 |
| `ask` | 问答模式 | **是，强制只读** | 代码解释、问题咨询、知识查询 |

**只读保证机制**:
- `plan` 和 `ask` 模式内部使用 `--mode plan` / `--mode ask` 调用 Cursor CLI
- 内部强制设置 `force_write=False`，即使显式指定 `--force` 也不会修改文件
- 通过 `PlanAgentExecutor` 和 `AskAgentExecutor` 类实现，确保只读语义
- 适合用于 Planner Agent 和咨询场景，确保不会意外修改代码

```bash
# 规划模式：只分析不执行
agent -p "分析这个项目的架构" --mode plan

# 问答模式：仅回答问题
agent -p "解释这段代码的作用" --mode ask

# 完整代理模式：可修改文件（需配合 --force）
agent -p --force "重构这个函数" --mode agent
```

### 命令

| 命令 | 描述 |
|------|------|
| `agent login` | 登录 Cursor |
| `agent logout` | 登出 |
| `agent status` | 查看认证状态 |
| `agent models` | 列出可用模型 |
| `agent mcp` | 管理 MCP 服务器 |
| `agent update` | 更新到最新版本 |
| `agent ls` | 列出历史会话 |
| `agent resume` | 恢复最新会话 |

### 斜杠命令（交互模式）

| 命令 | 描述 |
|------|------|
| `/model <model>` | 设置或列出模型 |
| `/models` | 列出所有可用模型 |
| `/auto-run [on\|off]` | 切换自动运行 |
| `/new-chat` | 开启新聊天 |
| `/vim` | 切换 Vim 按键 |
| `/feedback <msg>` | 提交反馈 |
| `/resume <chat>` | 恢复先前聊天 |
| `/copy-req-id` | 复制请求 ID |
| `/logout` | 退出账号 |
| `/quit` | 退出 |
| `/mcp enable <name>` | 启用 MCP 服务器 |
| `/mcp disable <name>` | 禁用 MCP 服务器 |
| `/plan` | 切换到规划模式（只分析不执行） |
| `/ask` | 切换到问答模式（仅回答问题，不修改文件） |
| `/rules` | 创建/编辑规则 |
| `/commands` | 创建/编辑命令 |

## MCP 服务器管理

Agent 会**自动发现并使用** MCP 工具，无需显式调用。

```bash
# 查看可用的 MCP 服务器
agent mcp list

# 查看服务器提供的工具
agent mcp list-tools playwright

# 身份验证
agent mcp login <identifier>

# 启用/禁用服务器（支持带空格的名称）
agent mcp enable <identifier>
agent mcp disable <identifier>

# 交互模式斜杠命令
/mcp enable <name>
/mcp disable <name>
```

### 配置优先级
项目配置 → 全局配置 → 嵌套级配置

CLI 会自动从父目录发现配置。

### 网页浏览支持

#### 方式1: Fetch（推荐，无需 GUI）

```bash
# 安装
npm install -g @anthropic/mcp-server-fetch

# 使用 - 获取网页内容
agent -p "获取 https://example.com 的内容"
```

#### 方式2: Playwright Headless（无需 GUI）

```bash
# 安装 Playwright + 无头浏览器
npm install -g @anthropic/mcp-server-playwright
npx playwright install chromium  # 安装无头 Chromium

# 使用
agent -p "用 headless 模式打开 google.com"
```

#### 方式3: curl/wget/lynx（Shell 命令）

```bash
# 使用 curl 获取原始内容
agent -p "使用 curl 获取 https://api.github.com 的内容"

# 使用 lynx 获取纯文本（推荐，自动处理 HTML）
agent -p "使用 lynx -dump 获取 https://example.com 的文本内容"

# lynx 还可以获取链接列表
agent -p "使用 lynx -dump -listonly 获取页面所有链接"
```

## 规则与命令管理

```bash
# 交互模式
/rules      # 创建/编辑规则
/commands   # 创建/编辑命令
/models     # 列出/切换模型
```

## Hooks

Hooks 允许通过自定义脚本观察、控制和扩展 Agent 循环。配置文件: `.cursor/hooks.json`

```
.cursor/hooks/
├── audit.sh           # 审计日志
├── afterFileEdit.sh   # 文件编辑后
├── beforeCommand.sh   # 命令执行前
└── ...
```

### Hook 事件类型

| 事件 | 描述 |
|------|------|
| `beforeShellExecution` | Shell 命令执行前 |
| `afterShellExecution` | Shell 命令执行后 |
| `beforeMCPExecution` | MCP 工具调用前 |
| `afterMCPExecution` | MCP 工具调用后 |
| `beforeReadFile` | 文件读取前 |
| `afterFileEdit` | 文件编辑后 |
| `beforeSubmitPrompt` | 提交 prompt 前 |
| `stop` | Agent 结束时 |
| `beforeTabFileRead` | Tab 补全文件读取前 |
| `afterTabFileEdit` | Tab 编辑后 |

### afterFileEdit Hook

```bash
# 可用变量
$file_path   # 被编辑的文件路径
$old_string  # 修改前内容（用于 diff）
$new_string  # 修改后内容
```

**特性**:
- 并行执行，合并响应
- 执行延迟降低 10 倍

### hooks.json 示例

```json
{
  "version": 1,
  "hooks": {
    "afterFileEdit": [{ "command": "./hooks/format.sh" }],
    "beforeShellExecution": [{ "command": "./hooks/audit.sh" }]
  }
}
```

## 权限配置

配置文件: `~/.cursor/cli-config.json`（全局）或 `.cursor/cli.json`（项目级）

### 权限类型

| 格式 | 描述 |
|------|------|
| `Shell(commandBase)` | 控制 Shell 命令访问 |
| `Read(pathOrGlob)` | 控制文件读取权限 |
| `Write(pathOrGlob)` | 控制文件写入权限 |

### 示例配置

```json
{
  "permissions": {
    "allow": [
      "Shell(ls)", "Shell(git)", "Shell(npm)",
      "Read(src/**/*.ts)", "Write(src/**)"
    ],
    "deny": [
      "Shell(rm)", "Read(.env*)", "Write(**/*.key)"
    ]
  }
}
```

**规则**: 拒绝规则优先于允许规则

## 子代理 (Subagents)

子代理是可以将任务委派给的专业化 AI 助手。每个子代理在自己的上下文窗口中运行。

> 注意: Subagents 目前仅在 Nightly 更新通道可用

### 配置位置

| 位置 | 范围 |
|------|------|
| `.cursor/agents/` | 项目级 |
| `~/.cursor/agents/` | 用户级（全局） |

### 文件格式

```markdown
---
name: security-auditor
description: 安全专家。用于审查安全敏感代码时调用。
model: inherit
readonly: true
---

你是一位代码安全审计专家...
```

### 配置字段

| 字段 | 描述 |
|------|------|
| `name` | 唯一标识符 |
| `description` | 何时使用此子代理 |
| `model` | fast / inherit / 具体模型 ID |
| `readonly` | 是否限制写入权限 |

### 运行模式

| 模式 | 行为 |
|------|------|
| **前台** | 阻塞直到完成，立即返回结果 |
| **后台** | 立即返回，独立运行 |

## Skills (技能)

Skills 是为 AI Agent 扩展专门能力的可移植包。

> 注意: Skills 目前仅在 Nightly 更新通道可用

### 配置位置

| 位置 | 范围 |
|------|------|
| `.cursor/skills/` | 项目级 |
| `~/.cursor/skills/` | 用户级（全局） |

### SKILL.md 格式

```markdown
---
name: my-skill
description: 技能描述，Agent 用它来判断何时使用。
---

# My Skill

详细指令...
```

## 处理图像与文件

Agent 支持读取图像、视频等二进制文件。只需在提示中包含文件路径：

```bash
# 分析图像
agent -p "分析这张图片: ./screenshot.png"

# 比较多个图像
agent -p "比较这两张图片的差异: ./before.png ./after.png"

# 结合代码和设计稿
agent -p "查看 src/app.ts 的代码和 designs/homepage.png 的设计稿，提出改进建议"
```

### 工作原理

1. 提示中包含文件路径
2. Agent 通过工具调用自动读取文件
3. 图像被自动处理
4. 支持相对路径和绝对路径

### 批量处理示例

```bash
# 批量处理图像
for image in images/*.png; do
  agent -p "描述图像: $image" > "${image%.png}.txt"
done
```

## CLI 使用参考

### 非交互模式（自动化使用）
```bash
# 基本用法
agent -p "prompt" --model "model-name" --output-format text

# JSON 输出（便于脚本解析）
agent -p "prompt" --model "model-name" --output-format json

# 恢复会话
agent --resume "thread-id" -p "继续之前的任务"

# 查看历史会话
agent ls
```

**重要**:
- 不带 `--force`: 仅提议更改而不应用
- 带 `--force`: 允许直接修改文件，无需确认

```bash
# 只分析，不修改文件
agent -p "分析代码结构"

# 允许修改文件
agent -p --force "重构为 ES6+ 语法"
```

### 实时进度跟踪
```bash
# 消息级别进度跟踪
agent -p "prompt" --output-format stream-json

# 增量流式传输变更内容
agent -p "prompt" --output-format stream-json --stream-partial-output
```

### 交互模式快捷键
| 快捷键 | 功能 |
|--------|------|
| `↑` / `↓` | 翻阅历史消息 |
| `Shift+Enter` | 插入换行 |
| `Ctrl+D` (×2) | 退出 CLI |
| `Ctrl+R` | 审阅更改 |
| `i` | 添加后续说明 |
| `←` / `→` | 切换文件 |
| `@` | 选择上下文文件/文件夹 |
| `/compress` | 压缩上下文 |
| `Tab` | 自动补全命令/文件路径 |
| `Ctrl+C` | 中断当前操作 |
| `Ctrl+L` | 清屏 |
| `Esc` | 取消当前输入 |

## Cloud Agent 使用方法

Cloud Agent 提供云端 API 访问能力，支持程序化调用 Cursor Agent。

### Cloud 执行模式对比

本系统提供三种触发 Cloud 执行的方式，语义各有不同：

| 方式 | 语义 | 使用场景 | 恢复方式 |
|------|------|----------|----------|
| `&` 前缀 | **Cloud Relay**：把这条消息/会话推到云端继续跑 | 交互式提交单条任务到云端 | `agent --resume <session_id>` |
| `--execution-mode cloud` | **强制云端**：本系统强制使用云端执行器（无需依赖 `&`） | 脚本/自动化场景，确保使用云端 | `agent --resume <session_id>` |
| `--execution-mode auto` | **自动选择**：云端优先，失败回退本地 CLI | 推荐默认选择，兼顾可用性和云端优势 | `agent --resume <session_id>` |

#### 最小示例

```bash
# ===== 方式 1: & 前缀（Cloud Relay）=====
# 语义：把这条消息推到云端继续跑
agent -p "& 分析整个代码库的架构"
# 返回 session_id 后可恢复：
agent --resume abc123-session-id

# ===== 方式 2: --execution-mode cloud =====
# 语义：强制使用云端执行器，不依赖 & 前缀
python scripts/run_iterate.py --execution-mode cloud "长时间分析任务"
# 恢复方式同上（脚本会输出 session_id）
agent --resume abc123-session-id

# ===== 方式 3: --execution-mode auto =====
# 语义：云端优先，云端不可用时自动回退到本地 CLI
python scripts/run_iterate.py --execution-mode auto "任务描述"
# 如果使用了云端，可用 session_id 恢复
agent --resume abc123-session-id
# 如果回退到本地 CLI，则无 session_id

# ===== run.py Cloud 模式后台提交 =====
# 使用 run.py 的 Cloud 模式提交后台任务
python run.py --mode iterate --execution-mode cloud "长时间代码分析任务"

# 指定 Cloud 执行超时时间（默认 300 秒）
python run.py --mode iterate --execution-mode cloud --cloud-timeout 1200 "复杂重构任务"

# scripts/run_iterate.py 也支持 --cloud-timeout
python scripts/run_iterate.py --execution-mode cloud --cloud-timeout 900 "分析任务"

# ===== 查看历史会话 =====
agent ls

# ===== 恢复指定会话 =====
agent --resume <session_id>

# ===== 恢复最新会话 =====
agent resume
```

### Cloud Relay（& 前缀）

使用 `&` 前缀可以将任务提交到云端后台执行，无需等待完成即可继续其他工作。

```bash
# 使用 & 前缀提交云端任务
agent -p "& 分析整个代码库的架构"

# 等效于使用 -b (background) 模式
agent -b -p "分析整个代码库的架构"

# 查看后台任务状态
agent ls

# 恢复/查看后台任务结果
agent --resume <session_id>
```

**Cloud Relay 特点**:
- 任务在云端独立运行，本地可断开连接
- 自动轮询任务状态并获取结果
- 支持会话恢复（`--resume`）继续之前的任务
- 适合长时间运行的分析或重构任务

**`&` 前缀路由语义**:

| 输入 | 是否触发 Cloud 模式 | 说明 |
|------|---------------------|------|
| `& 分析代码` | ✓ 是 | 正常触发 |
| `&分析代码` | ✓ 是 | 无空格也有效 |
| `&` | ✗ 否 | 只有 `&` 无实际内容 |
| `&   ` | ✗ 否 | `&` 后仅有空白 |
| `分析 & 代码` | ✗ 否 | `&` 不在开头 |

**Cloud 模式的自动提交**:
- Cloud 模式下 **默认不开启** `auto_commit`（安全策略）
- 如需提交，必须显式指定 `--auto-commit`
- `allow_write/force_write` 由用户显式控制

### 基本使用

```python
from cursor import CloudClient

# 初始化客户端
client = CloudClient(api_key="your_api_key")

# 发送请求
response = client.chat(
    prompt="分析项目结构",
    model="opus-4.5-thinking",
    output_format="json"
)

# 流式响应
for chunk in client.stream_chat(prompt="重构代码"):
    print(chunk)
```

### 配置选项

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `api_key` | API 密钥 | 环境变量 `CURSOR_API_KEY` |
| `base_url` | API 端点 | `https://api.cursor.com` |
| `timeout` | 请求超时（秒） | 300 |
| `max_retries` | 最大重试次数 | 3 |

### 高级功能

```python
# 带上下文的请求
response = client.chat(
    prompt="修复这个 bug",
    context_files=["src/main.py", "tests/test_main.py"],
    mode="agent"  # plan / ask / agent
)

# 恢复会话
response = client.resume_chat(
    session_id="previous_session_id",
    prompt="继续之前的任务"
)

# 批量处理
results = client.batch_process(
    prompts=["任务1", "任务2", "任务3"],
    parallel=True
)
```

### Executor 抽象层

`cursor.executor` 模块提供统一的执行器接口，支持多种执行模式自动切换。

```python
from cursor.executor import AgentExecutorFactory, ExecutionMode

# 创建执行器（自动选择 Cloud 或 CLI）
executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)

# 使用规划模式执行器（只读保证）
plan_executor = AgentExecutorFactory.create(mode=ExecutionMode.PLAN)
result = await plan_executor.execute(prompt="分析代码架构")

# 使用问答模式执行器（只读保证）
ask_executor = AgentExecutorFactory.create(mode=ExecutionMode.ASK)
result = await ask_executor.execute(prompt="解释这段代码")

# Cloud 执行器（后台任务）
cloud_executor = AgentExecutorFactory.create(mode=ExecutionMode.CLOUD)
result = await cloud_executor.execute(prompt="& 长时间分析任务")

# 提交后台任务（不等待完成）
result = await cloud_executor.submit_background_task(prompt="分析任务")
task_id = result.session_id

# 等待后台任务完成
result = await cloud_executor.wait_for_task(task_id, timeout=600)
```

**执行模式对比**:

| 模式 | 描述 | 只读 | 支持 MP 编排器 | 使用场景 |
|------|------|------|----------------|----------|
| `CLI` | 本地 CLI 执行 | 否 | ✓ 是 | 本地开发、完整代理功能 |
| `CLOUD` | Cloud API 执行 | 否 | ✗ 否（强制 basic） | 后台任务、长时间运行 |
| `AUTO` | 自动选择（Cloud 优先，回退 CLI） | 否 | ✗ 否（强制 basic） | 推荐默认选择 |
| `PLAN` | 规划模式 | **是** | ✓ 是 | 任务分析、代码审查 |
| `ASK` | 问答模式 | **是** | ✓ 是 | 代码解释、咨询 |

**编排器兼容性说明**:
- **MP 编排器** (`MultiProcessOrchestrator`): 仅在 `execution_mode=cli` 时可用
- **Cloud/Auto 模式**: 强制使用 basic 编排器，因为 Cloud API 不支持多进程编排
- 系统会自动检测并切换，无需手动处理

### 错误处理

```python
from cursor import CloudClient, CursorError, RateLimitError

try:
    response = client.chat(prompt="任务")
except RateLimitError as e:
    print(f"速率限制，请在 {e.retry_after} 秒后重试")
except CursorError as e:
    print(f"API 错误: {e.message}")
```

## 系统架构

```
Planner (规划者) → TaskQueue → Workers (执行者) → Reviewer (评审者)
     ↑                                                    │
     └────────────── 迭代循环 ←───────────────────────────┘
```

## 自我迭代模式

自我迭代模式（iterate）**默认启用多进程并行执行**（`MultiProcessOrchestrator`），支持高效的任务并行处理。

### 编排器选择

| 编排器 | 参数 | 说明 |
|--------|------|------|
| `MultiProcessOrchestrator` | `--orchestrator mp`（默认） | 多进程并行，适合复杂任务 |
| `Orchestrator` | `--orchestrator basic` 或 `--no-mp` | 协程模式，适合简单任务或资源受限环境 |

**注意**: 当 `--execution-mode` 为 `cloud` 或 `auto` 时，系统会 **强制使用 basic 编排器**，因为 Cloud/Auto 执行模式不支持多进程编排器。此时即使指定 `--orchestrator mp` 也会自动切换到 basic 编排器。

### 运行示例

```bash
# 默认使用多进程编排器（推荐，execution-mode=cli 时）
python run.py --mode iterate "优化代码"
python scripts/run_iterate.py "增加新功能支持"

# 显式指定多进程编排器
python run.py --mode iterate --orchestrator mp "任务描述" --workers 5

# 禁用多进程，使用协程编排器
python run.py --mode iterate --no-mp "任务描述"
python scripts/run_iterate.py --orchestrator basic "任务描述"

# 使用 Cloud/Auto 执行模式（自动使用 basic 编排器）
python scripts/run_iterate.py --execution-mode auto "任务描述"
python scripts/run_iterate.py --execution-mode cloud "长时间分析任务"

# 使用 & 前缀触发 Cloud 模式（等效于 --execution-mode cloud）
python scripts/run_iterate.py "& 后台分析代码架构"

# 配合自动提交
python run.py --mode iterate --auto-commit --auto-push "完成功能"

# 启用差异渲染（Diff 视图增强）
python scripts/run_iterate.py --stream-console-renderer --stream-show-word-diff "重构代码"
```

**注意**: 当使用 `&` 前缀或 `--execution-mode cloud/auto` 时，即使指定 `--orchestrator mp` 也会自动切换到 basic 编排器。

### 回退策略

当 MP 编排器启动失败时，系统 **自动回退** 到协程编排器：

- **启动超时**: 进程创建超时
- **资源不足**: OSError（如进程数限制）
- **运行时错误**: 事件循环问题等
- **其他异常**: 未预期的启动错误

回退提示：
```
⚠ MP 编排器启动失败: <错误原因>
⚠ 正在回退到基本协程编排器...
```

### 参数参考

#### 配置优先级

配置值按以下优先级确定（高到低）：

| 优先级 | 来源 | 示例 |
|--------|------|------|
| 1 (最高) | CLI 参数 | `--workers 5`, `--execution-mode cloud`, `--cloud-api-key` |
| 2 | 环境变量 `CURSOR_API_KEY` | 主要 API Key 环境变量 |
| 3 | 环境变量 `CURSOR_CLOUD_API_KEY` | 备选 API Key 环境变量（仅当 `CURSOR_API_KEY` 未设置时使用） |
| 4 | config.yaml | `worker_pool_size: 3`, `cloud_agent.api_key` |
| 5 (最低) | DEFAULT_* 常量 | 代码中的 `DEFAULT_CLOUD_TIMEOUT = 300` |

**注意**: 优先级 2-3（环境变量细分）仅适用于 API Key 配置。其他配置项使用 4 级优先级: CLI 参数 > 环境变量 > config.yaml > DEFAULT_* 常量。

**示例**: `--workers 5` 会覆盖 config.yaml 中的 `worker_pool_size: 3`

**API Key 优先级说明**: `CURSOR_API_KEY` 优先于 `CURSOR_CLOUD_API_KEY`。后者作为备选，仅在前者未设置时生效。

#### config.yaml 权威配置来源

`config.yaml` 是项目的**权威配置来源**，集中定义所有默认配置。CLI 参数可覆盖其中的值，但 `config.yaml` 始终作为基础配置被加载。

**关键配置项映射**（CLI 参数 → config.yaml 字段）:

| CLI 参数 | config.yaml 路径 | 说明 |
|----------|------------------|------|
| `--workers` | `system.worker_pool_size` | Worker 池大小 |
| `--max-iterations` | `system.max_iterations` | 最大迭代次数 |
| `--execution-mode` | `cloud_agent.execution_mode` | 执行模式: cli/cloud/auto |
| `--cloud-timeout` | `cloud_agent.timeout` | Cloud 执行超时（秒） |
| `--output-format stream-json` | `agent_cli.output_format` + `logging.stream_json.enabled` | 流式 JSON 输出 |

#### 验证配置是否被正确读取（--print-config）

使用 `--print-config` 参数可快速验证入口脚本是否正确加载了 `config.yaml` 中的配置：

```bash
# 验证 run.py 读取的配置
python run.py --print-config

# 验证 scripts/run_iterate.py 读取的配置
python scripts/run_iterate.py --print-config

# 结合 CLI 参数验证覆盖是否生效
python run.py --workers 5 --execution-mode cloud --print-config
```

**输出示例**:

```
[CONFIG] config_path: /path/to/config.yaml
[CONFIG] source: run.py
[CONFIG] max_iterations: 10
[CONFIG] workers: 3
[CONFIG] execution_mode: cli
[CONFIG] orchestrator: mp
[CONFIG] orchestrator_fallback: none
[CONFIG] planner_model: gpt-5.2-high
[CONFIG] worker_model: opus-4.5-thinking
[CONFIG] reviewer_model: gpt-5.2-codex
[CONFIG] cloud_timeout: 300
[CONFIG] cloud_auth_timeout: 30
[CONFIG] auto_commit: false
[CONFIG] auto_push: false
[CONFIG] dry_run: false
[CONFIG] strict_review: false
[CONFIG] enable_sub_planners: true
```

**典型用途**:
- CI/CD 中断言配置正确: `python run.py --print-config | grep "workers: 5"`
- 排查配置不生效问题: 确认 `config_path` 指向正确的配置文件
- 验证编排器回退: 当 `execution_mode` 为 `cloud`/`auto` 时，`orchestrator_fallback` 会显示回退信息

#### 核心执行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--orchestrator` | 编排器类型: `mp`/`basic` | `mp` |
| `--no-mp` | 禁用多进程编排器 | False |
| `--execution-mode` | 执行模式: `cli`/`auto`/`cloud`（`cloud`/`auto` 强制使用 basic 编排器） | `cli` |
| `--workers` | Worker 池大小 | 3 |
| `--max-iterations` | 最大迭代次数（MAX/-1 表示无限迭代） | 10 |
| `--skip-online` | 跳过在线文档检查 | False |
| `--dry-run` | 仅分析不执行 | False |

**重要**: MP 编排器与 Cloud/Auto 执行模式**不兼容**。当 `--execution-mode` 为 `cloud` 或 `auto` 时，系统会**强制使用 basic 编排器**，即使显式指定 `--orchestrator mp` 也会自动切换。

#### 自动提交参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--auto-commit` | 启用自动提交（**必须显式指定才会提交**） | **False** |
| `--auto-push` | 自动推送到远程仓库（需配合 `--auto-commit`） | **False** |
| `--commit-per-iteration` | 每次迭代都提交（默认仅在全部完成时提交，`run.py` 和 `scripts/run_iterate.py` 均支持） | False |

#### 日志与诊断参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-v, --verbose` | 详细输出（DEBUG 级别日志） | False |
| `-q, --quiet` | 静默模式（仅 WARNING 及以上日志） | False |
| `--log-level` | 日志级别: `DEBUG`/`INFO`/`WARNING`/`ERROR`（优先级高于 --verbose/--quiet） | `INFO` |
| `--heartbeat-debug` | 启用心跳调试日志（仅调试时使用，默认关闭以减少日志输出） | False |
| `--stall-diagnostics` | 启用卡死诊断日志（**默认关闭**，疑似卡死时再启用以排查问题） | **False** |
| `--stall-diagnostics-level` | 诊断日志级别: `debug`/`info`/`warning`/`error`（启用诊断时默认 warning） | `warning` |

#### 流式控制台渲染参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stream-console-renderer` | 启用流式控制台渲染器 | False |
| `--stream-advanced-renderer` | 使用高级终端渲染器（支持状态栏、打字效果等） | False |
| `--stream-show-word-diff` | 显示逐词差异（Diff 视图增强） | False |
| `--stream-typing-effect` | 启用打字机效果 | False |
| `--stream-typing-delay` | 打字延迟（秒） | 0.02 |
| `--stream-word-mode` / `--no-stream-word-mode` | 逐词/逐字符输出模式 | True |
| `--stream-color-enabled` / `--no-stream-color` | 颜色输出开关 | True |

#### Cloud 认证参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cloud-api-key` | Cloud API Key（优先级: CLI > `CURSOR_API_KEY` > `CURSOR_CLOUD_API_KEY` > config.yaml） | 环境变量 `CURSOR_API_KEY` |
| `--cloud-auth-timeout` | Cloud 认证超时时间（秒） | 30 |
| `--cloud-timeout` | Cloud 执行超时时间（秒），优先级: CLI > config.yaml > DEFAULT_CLOUD_TIMEOUT | 300 |

**注意**: `CURSOR_CLOUD_API_KEY` 仅作为备选环境变量，当 `CURSOR_API_KEY` 未设置时才会使用。

### 自动提交配置

**默认行为**: `--auto-commit` 和 `--auto-push` **默认禁用（False）**，必须显式指定 `--auto-commit` 才会启用提交功能。

```bash
# 默认：不自动提交（安全模式）
python run.py --mode iterate "任务描述"
python scripts/run_iterate.py "任务描述"

# 显式启用自动提交（必须指定 --auto-commit）
python run.py --mode iterate --auto-commit "完成功能"
python scripts/run_iterate.py --auto-commit "完成功能"

# 启用自动提交并推送到远程（需同时指定 --auto-commit 和 --auto-push）
python run.py --mode iterate --auto-commit --auto-push "完成功能"
python scripts/run_iterate.py --auto-commit --auto-push "完成功能"

# 每次迭代都提交（而非仅在全部完成时提交）
# run.py 和 scripts/run_iterate.py 均支持 --commit-per-iteration
python run.py --mode iterate --auto-commit --commit-per-iteration "分步完成"
python scripts/run_iterate.py --auto-commit --commit-per-iteration "分步完成"
```

**重要**: 不指定 `--auto-commit` 时，即使任务成功完成也不会创建 Git 提交。

**通过自然语言控制提交**:

```bash
# 在任务描述中可以使用关键词控制
python run.py "启用提交，完成功能优化"     # 检测到 "启用提交" 自动开启
python run.py "禁用提交，仅分析代码"       # 检测到 "禁用提交" 显式关闭

# 支持的开启关键词: 启用提交、开启提交、自动提交、enable-commit
# 支持的关闭关键词: 禁用提交、关闭提交、不提交、跳过提交、no-commit
```

**提交去重策略**: 如果编排器（MP 或 basic）已完成提交，SelfIterator 不会重复提交

## Agent 角色定义

### Planner (规划者)
- **模型**: gpt-5.2-high
- **职责**: 分析目标、探索代码库、分解任务
- **限制**: 不要编写任何代码，不要编辑任何文件
- **工具**: 文件搜索、读取、Shell（只读命令如 ls, find, grep）

### Worker (执行者)
- **模型**: opus-4.5-thinking
- **职责**: 执行具体编码任务
- **权限**: 完整的文件操作、Shell 命令
- **要求**: 专注当前任务，不考虑其他任务

### Reviewer (评审者)
- **模型**: gpt-5.2-codex
- **职责**: 评估完成度、决定是否继续迭代
- **限制**: 不要编写任何代码，不要编辑任何文件
- **工具**: 文件读取、git diff/status

## 通用规则

1. 使用中文进行沟通
2. 代码注释保持原有风格
3. 不自动生成 README 或文档文件
4. 遵循项目现有的代码规范
5. 修改代码后进行验证测试

## 输出格式

参考: https://cursor.com/cn/docs/cli/reference/output-format

### text 格式（默认）

仅返回最终答案的简洁输出。

### json 格式

结构化输出，便于脚本解析：

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "duration_ms": 1234,
  "duration_api_ms": 1234,
  "result": "<完整助手文本>",
  "session_id": "<uuid>"
}
```

### stream-json 格式

NDJSON 格式，每行一个 JSON 对象，用于实时进度跟踪：

#### 事件类型

| 类型 | 描述 |
|------|------|
| `system/init` | 会话初始化，包含模型、权限模式等 |
| `user` | 用户消息 |
| `assistant` | 助手消息（完整消息文本） |
| `tool_call` | 工具调用 (started/completed) |
| `result` | 最终结果 |

#### system/init 事件

```json
{
  "type": "system",
  "subtype": "init",
  "apiKeySource": "env|flag|login",
  "cwd": "/absolute/path",
  "session_id": "<uuid>",
  "model": "<模型名称>",
  "permissionMode": "default"
}
```

#### assistant 事件

```json
{
  "type": "assistant",
  "message": {
    "role": "assistant",
    "content": [{ "type": "text", "text": "<消息文本>" }]
  },
  "session_id": "<uuid>"
}
```

#### tool_call 事件

```json
{
  "type": "tool_call",
  "subtype": "started|completed",
  "call_id": "<id>",
  "tool_call": {
    "readToolCall": { "args": { "path": "file.txt" } }
  },
  "session_id": "<uuid>"
}
```

### Agent 角色输出

- **Planner**: JSON 格式任务计划
- **Worker**: text 格式执行摘要
- **Reviewer**: JSON 格式评审结果

## CI/CD

本项目配置了完整的 CI/CD 流水线，确保代码质量和安全性。

### 工作流列表

| 工作流 | 文件 | 说明 |
|--------|------|------|
| CI | `ci.yml` | 多版本 Python 测试、覆盖率统计 |
| Lint | `lint.yml` | flake8、ruff、mypy 代码检查 |
| PR Check | `pr-check.yml` | PR 综合检查（复用 CI 和 Lint） |
| Security | `security.yml` | 依赖安全审计 (pip-audit) |
| Notify Failure | `notify-failure.yml` | 工作流失败通知（Slack/Issue） |

### 触发条件

| 事件 | CI | Lint | PR Check | Security |
|------|----|----- |----------|----------|
| Push 到 main/master | ✓ | ✓ | - | - |
| Push 到 feature/**、fix/** | ✓ | ✓ | - | - |
| Pull Request | ✓ | ✓ | ✓ | ✓ (仅当修改依赖文件) |
| 定时任务 (每周一) | - | - | - | ✓ |
| 手动触发 | - | - | - | ✓ |

### 本地预提交检查

推荐使用 pre-commit 在提交前自动执行代码检查：

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 git hooks（仅需执行一次）
pre-commit install

# 手动运行所有检查
pre-commit run --all-files

# 跳过 hooks（不推荐）
git commit --no-verify -m "message"
```

Pre-commit 会自动执行：
- 移除行尾空白、确保文件换行结尾
- YAML/JSON 语法检查
- Ruff 代码检查与格式化
- MyPy 类型检查

### 手动触发工作流

通过 GitHub Actions 界面手动触发：

1. 进入 GitHub 仓库 → Actions 标签页
2. 选择要运行的工作流（如 Security Audit）
3. 点击 "Run workflow" 按钮
4. 选择分支后点击 "Run workflow" 确认

或使用 GitHub CLI：

```bash
# 手动触发 security 工作流
gh workflow run security.yml

# 指定分支
gh workflow run security.yml --ref main

# 查看工作流运行状态
gh run list --workflow=security.yml
```

### 本地完整检查

```bash
# 运行与 CI 相同的完整检查
bash scripts/check_all.sh --full

# 快速检查（仅关键项）
bash scripts/check_all.sh
```

### 失败通知配置

`notify-failure.yml` 工作流会在其他工作流失败时发送通知。

#### 支持的通知方式

| 方式 | 配置 | 说明 |
|------|------|------|
| GitHub Step Summary | 默认启用 | 在 Actions 界面显示失败摘要 |
| Slack Webhook | 需配置 Secret | 发送到 Slack 频道 |
| GitHub Issue | 需配置 Variable | 持续失败时创建 Issue |

#### Slack 通知配置

1. 在 Slack 中创建 Incoming Webhook
2. 在仓库 Settings → Secrets and variables → Actions → Secrets 中添加：
   - Name: `SLACK_WEBHOOK_URL`
   - Value: `https://hooks.slack.com/services/xxx/xxx/xxx`

#### GitHub Issue 自动创建配置

当工作流连续失败 3 次以上时，自动创建 Issue 追踪问题：

1. 在仓库 Settings → Secrets and variables → Actions → Variables 中添加：
   - Name: `CREATE_ISSUE_ON_FAILURE`
   - Value: `true`

2. Issue 自动管理：
   - 连续失败 3 次：创建新 Issue（标签：`ci-failure`, `automated`, `bug`）
   - 再次失败：在现有 Issue 中添加评论
   - 恢复成功：自动关闭相关 Issue
