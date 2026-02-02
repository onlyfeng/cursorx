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

| 方式 | 语义 | `prefix_routed` | 使用场景 | 恢复方式 |
|------|------|-----------------|----------|----------|
| `&` 前缀 | **Cloud Relay**：把这条消息路由到云端继续跑 | `True`（满足条件时） | 交互式提交单条任务到云端 | `agent --resume <session_id>` |
| `--execution-mode cloud` | **强制云端**：显式使用云端执行器 | `False` | 脚本/自动化场景（**推荐**） | `agent --resume <session_id>` |
| `--execution-mode auto` | **自动选择**：云端优先，失败回退本地 CLI | `False` | 推荐默认选择，兼顾可用性和云端优势 | `agent --resume <session_id>` |

#### 最小示例

```bash
# ===== 方式 1: & 前缀路由（prefix_routed=True）=====
# 语义：把这条消息路由到云端继续跑（需满足 prefix_routed 条件）
agent -p "& 分析整个代码库的架构"
# 返回 session_id 后可恢复：
agent --resume abc123-session-id

# ===== 方式 2（推荐）: --execution-mode cloud =====
# 语义：显式使用云端执行器，无需 & 前缀路由，prefix_routed=False
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

### Cloud Relay（`&` 前缀路由，`prefix_routed`）

使用 `&` 前缀可以将任务路由到云端后台执行（`prefix_routed=True`），无需等待完成即可继续其他工作。

```bash
# 使用 & 前缀路由到云端（需满足 prefix_routed 条件）
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
- 需要 `prefix_routed=True`（满足全部路由条件）才能生效

**`&` 前缀路由语义（`prefix_routed`）**:

| 输入 | `prefix_routed` | 说明 |
|------|-----------------|------|
| `& 分析代码` | `True` | 正常触发 Cloud |
| `&分析代码` | `True` | 无空格也有效 |
| `&` | `False` | 只有 `&` 无实际内容 |
| `&   ` | `False` | `&` 后仅有空白 |
| `分析 & 代码` | `False` | `&` 不在开头 |

> **术语说明**：
> - `prefix_routed`：**策略决策层面**，表示 `&` 前缀是否成功触发 Cloud（**推荐使用**，新代码应统一使用此字段）
> - `has_ampersand_prefix`：**语法检测层面**，仅表示原始文本是否有 `&` 前缀
> - `triggered_by_prefix`：**已废弃的兼容别名**，语义等同于 `prefix_routed`（仅保留用于兼容旧版输出格式，新代码禁止使用）

**`cloud_agent.enabled` 对 `prefix_routed` 的影响**:

`prefix_routed=True` 需要满足以下**全部条件**：

| 条件 | 说明 |
|------|------|
| `has_ampersand_prefix=True` | 语法检测：输入以 `&` 开头 |
| `cloud_agent.enabled=True` | 配置启用：config.yaml 中 `cloud_agent.enabled: true` |
| `has_api_key=True` | 认证可用：设置了 `CURSOR_API_KEY` 或等效配置 |
| `auto_detect_cloud_prefix=True` | 未禁用自动检测（默认启用） |

**注意**：`cloud_agent.enabled` 默认为 `true`（参见 `config.yaml`）。当此配置为 `false` 时，即使 `has_ampersand_prefix=True`，`prefix_routed` 仍为 `False`，任务将使用本地 CLI 执行。

```bash
# 启用 cloud_agent.enabled 后 & 前缀才能使 prefix_routed=True
# 方式 1：修改 config.yaml
cloud_agent:
  enabled: true

# 方式 2（推荐）：显式 Cloud 模式，不依赖 & 前缀路由
python scripts/run_iterate.py --execution-mode cloud "任务描述"
```

**`auto_detect_cloud_prefix` 配置详解**:

| 配置键 | config.yaml 路径 | 默认值 | 说明 |
|--------|------------------|--------|------|
| `auto_detect_cloud_prefix` | `cloud_agent.auto_detect_cloud_prefix` | `true` | 控制 `&` 前缀是否被视为 Cloud 意图 |

**取值效果**:

| 值 | 效果 | 编排器行为 |
|----|------|-----------|
| `true`（默认） | `&` 前缀触发 Cloud 意图检测，影响 `prefix_routed` | R-2 规则生效，编排器强制 basic |
| `false` | `&` 前缀被完全忽略，不触发 Cloud 意图检测 | R-3 规则生效，编排器可使用 mp |

**CLI 参数（tri-state）**:

| CLI 参数 | 效果 | 优先级 |
|----------|------|--------|
| `--auto-detect-cloud-prefix` | 显式启用，覆盖 config.yaml | CLI > config.yaml |
| `--no-auto-detect-cloud-prefix` | 显式禁用，覆盖 config.yaml | CLI > config.yaml |
| 未指定 | 使用 config.yaml 中的值（默认 `true`） | config.yaml |

**与 R-2/R-3 规则的关系**:

| 规则 | 条件 | 效果 |
|------|------|------|
| **R-2** | `auto_detect_cloud_prefix=true` + `has_ampersand_prefix=true` | `&` 前缀表达 Cloud 意图；即使未成功路由（`prefix_routed=False`），编排器仍强制 basic |
| **R-3** | `auto_detect_cloud_prefix=false` 或显式 `--execution-mode cli` | `&` 前缀被忽略，编排器可使用 mp |

**使用示例**:

```bash
# 场景 1：config.yaml auto_detect_cloud_prefix=true（默认）
# & 前缀触发 R-2 规则，编排器强制 basic
python scripts/run_iterate.py "& 分析代码"  # orchestrator=basic

# 场景 2：CLI 显式禁用 auto_detect_cloud_prefix
# & 前缀被忽略，可使用 mp 编排器
python scripts/run_iterate.py --no-auto-detect-cloud-prefix "& 分析代码"  # orchestrator=mp

# 场景 3：CLI 显式启用，覆盖 config.yaml 中的禁用设置
# 假设 config.yaml 中 auto_detect_cloud_prefix=false
python scripts/run_iterate.py --auto-detect-cloud-prefix "& 分析代码"  # orchestrator=basic

# 场景 4：显式 --execution-mode cli 忽略 & 前缀（R-3 规则）
python scripts/run_iterate.py --execution-mode cli "& 分析代码"  # orchestrator=mp
```

**配置优先级**: CLI 参数 > config.yaml（`cloud_agent.auto_detect_cloud_prefix`）

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
| `AUTO` | 自动选择（Cloud 优先，回退 CLI） | 否 | ✗ 否（强制 basic） | **系统默认**（来自 config.yaml） |
| `PLAN` | 规划模式 | **是** | ✓ 是 | 任务分析、代码审查 |
| `ASK` | 问答模式 | **是** | ✓ 是 | 代码解释、咨询 |

> **默认执行模式**: 系统默认使用 `auto` 模式（配置于 `config.yaml` 的 `cloud_agent.execution_mode`）。这意味着：
> - 直接运行 `python scripts/run_iterate.py "任务"` 等效于 `--execution-mode auto`
> - 系统会优先尝试 Cloud 执行，无 API Key 或 Cloud 不可用时自动回退到本地 CLI
> - 无论是否回退，编排器始终保持 basic（基于 requested_mode 语义）

**编排器兼容性说明**:
- **MP 编排器** (`MultiProcessOrchestrator`): 仅在 `requested_mode=cli/plan/ask/None` 时可用
- **Cloud/Auto 模式**: 强制使用 basic 编排器，因为 Cloud API 不支持多进程编排
- 系统会自动检测并切换，无需手动处理

**requested_mode vs effective_mode 关键区别**:

| 概念 | 含义 | 示例 |
|------|------|------|
| `requested_mode` | 用户请求的执行模式（CLI 参数指定） | `--execution-mode auto` → requested_mode=auto |
| `effective_mode` | 实际生效的执行模式（可能因回退而变化） | 无 API Key 时 auto → effective_mode=cli |

**核心规则**：编排器选择基于 **requested_mode**，而非 effective_mode。这意味着：

| 场景 | requested_mode | effective_mode | 编排器 |
|------|---------------|----------------|--------|
| `--execution-mode auto` 有 API Key | auto | cloud | basic（强制） |
| `--execution-mode auto` 无 API Key | auto | cli（回退） | **basic**（仍保持） |
| `--execution-mode cloud` 无 API Key | cloud | cli（回退） | **basic**（仍保持） |
| `--execution-mode cli` | cli | cli | mp（支持） |
| 无指定（默认，来自 config.yaml） | auto | cloud 或 cli（回退） | basic（强制） |

**关键**：
- **默认行为**：未指定 `--execution-mode` 时，系统从 config.yaml 读取默认值 `auto`
- **回退不影响编排器**：即使因缺少 API Key 导致 effective_mode 回退到 CLI，编排器**仍保持 basic**（不会恢复到 mp）
- **如需 MP 编排器**：请显式指定 `--execution-mode cli`

**警告与日志策略**：

| 情况 | 日志级别 | 说明 |
|------|----------|------|
| CLI 显式 `--execution-mode auto/cloud` | WARNING | 用户显式请求，明确提示回退 |
| config.yaml 默认 auto 且未显式指定 | INFO | 避免"每次都警告"的问题 |
| `scripts/run_mp.py` 中的模式不兼容提示 | INFO | 信息提示而非警告 |

**设计决策**（记录于 `core/execution_policy.py`）：
- **编排器选择严格基于 requested_mode**：不因回退而改变，保持语义一致性
- **日志级别区分显式/隐式配置**：避免默认 auto 导致的"每次都警告"问题
- **用户指引**：如需 MP 编排器，请显式指定 `--execution-mode cli`

> **重要**: `execution_mode=auto` 或 `execution_mode=cloud` 与 MP 编排器**不兼容**，系统会**强制切换到 basic 编排器**。推荐在使用 Cloud/Auto 模式时显式指定 `--orchestrator basic` 以避免警告：
>
> ```bash
> # 推荐写法：显式指定 basic 编排器（避免警告）
> python scripts/run_iterate.py --execution-mode auto --orchestrator basic "任务描述"
> python scripts/run_iterate.py --execution-mode cloud --orchestrator basic "后台分析任务"
>
> # 也可省略 --orchestrator basic，系统会自动强制切换（但会输出警告）
> python scripts/run_iterate.py --execution-mode auto "任务描述"
>
> # 如需使用 MP 编排器，必须显式指定 cli 模式
> python scripts/run_iterate.py --execution-mode cli --orchestrator mp "任务描述"
> ```

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

#### Cloud 常见错误代码与处理

| 错误代码 | 含义 | 用户提示 | 下一步操作 |
|----------|------|----------|------------|
| `NO_KEY` | 未配置 API Key | `未设置 CURSOR_API_KEY，Cloud 模式不可用` | 设置环境变量: `export CURSOR_API_KEY=your_key` 或使用 `--cloud-api-key` 参数 |
| `AUTH` | 认证失败（Key 无效或已过期） | `Cloud 认证失败: API Key 无效` | 检查 API Key 是否正确，必要时重新获取: `agent login` |
| `RATE_LIMIT` | 请求频率超限 | `速率限制，请在 {retry_after} 秒后重试` | 等待提示的 `retry_after` 秒数后重试，或降低请求频率 |
| `TIMEOUT` | 请求超时 | `Cloud 请求超时 ({timeout}s)` | 增大 `--cloud-timeout` 参数值，或检查网络连接 |
| `NETWORK` | 网络连接错误 | `无法连接到 Cloud API` | 检查网络连接和代理设置 |

**处理示例**:

```bash
# 错误: 未设置 API Key
# 提示: ⚠ 未设置 CURSOR_API_KEY，Cloud 模式不可用，回退到本地 CLI
# 解决: 设置环境变量
export CURSOR_API_KEY=your_api_key_here

# 错误: 速率限制
# 提示: ⚠ 速率限制，请在 60 秒后重试
# 解决: 等待后重试，或切换到本地 CLI 模式
python scripts/run_iterate.py --execution-mode cli "任务描述"
```

**回退后的编排器状态**:
- 当 `--execution-mode auto/cloud` 因缺少 API Key 或其他错误回退到 CLI 时
- **编排器仍保持 basic**（不会恢复到 mp）
- 原因：编排器选择基于 `requested_mode`（用户请求的模式），而非 `effective_mode`
- 如需使用 MP 编排器，请显式指定 `--execution-mode cli`

**回退场景示例**:

```bash
# 场景 1：auto 模式无 API Key → 回退到 CLI，编排器仍为 basic
$ python scripts/run_iterate.py --execution-mode auto "任务"
# 输出: ⚠ 未设置 CURSOR_API_KEY，Cloud 模式不可用，回退到本地 CLI
# requested_mode=auto, effective_mode=cli, orchestrator=basic

# 场景 2：显式 cli 模式 → 支持 MP 编排器
$ python scripts/run_iterate.py --execution-mode cli "任务"
# requested_mode=cli, effective_mode=cli, orchestrator=mp（默认）
```

**`&` 前缀未成功路由时的编排器（`prefix_routed=False`）**:

| 场景 | auto_detect | prefix_routed | orchestrator | 说明 |
|------|-------------|---------------|--------------|------|
| & + 无 API Key / cloud_disabled | `true`（默认） | `False` | **basic** | R-2: & 前缀表达 Cloud 意图，未成功路由仍强制 basic |
| & + 显式 `--execution-mode cli` | * | `False` | mp | R-3: 显式 cli 忽略 & 前缀 |
| & + `auto_detect_cloud_prefix=false` | `false` | `False` | mp | R-3: 禁用检测忽略 & 前缀 |

- **R-2 规则**: 当 `auto_detect_cloud_prefix=true` 时，& 前缀即表达 Cloud 意图；即使因缺少 API Key 或 `cloud_agent.enabled=false` 未成功路由，编排器**仍强制 basic**
- **R-3 规则**: 仅当显式 `--execution-mode cli` 或 `auto_detect_cloud_prefix=false` 时，& 前缀被忽略，编排器可使用 mp
- 如需 MP 编排器，请显式指定 `--execution-mode cli`

### `execution_mode=plan/ask` 时 `&` 前缀的处理规则

`plan` 和 `ask` 是**只读模式**（readonly mode），设计上用于代码分析、审查和问答场景，不参与 Cloud 路由。

**核心规则**:

| 规则编号 | 规则描述 | 说明 |
|----------|----------|------|
| R-4 | **plan/ask 模式不参与 Cloud 路由** | 只读模式仅用于分析，不需要 Cloud 后台执行 |
| R-5 | **& 前缀在 plan/ask 模式下被忽略** | 与显式 `--execution-mode cli` 的行为一致 |
| R-6 | **plan/ask 模式允许使用 MP 编排器** | 因为它们不是 cloud/auto 模式 |

**决策矩阵**:

| 场景 | requested_mode | has_ampersand_prefix | prefix_routed | orchestrator | 说明 |
|------|----------------|---------------------|---------------|--------------|------|
| `--execution-mode plan` | plan | False | `False` | **mp** | 只读模式，正常使用 |
| `--execution-mode ask` | ask | False | `False` | **mp** | 只读模式，正常使用 |
| `--execution-mode plan` + `& 任务` | plan | True | **False** | **mp** | **& 前缀被忽略** |
| `--execution-mode ask` + `& 任务` | ask | True | **False** | **mp** | **& 前缀被忽略** |

**术语统一**:

| 字段名 | 含义 | 类型 |
|--------|------|------|
| `requested_mode` | 用户请求的执行模式（CLI 参数或 config.yaml） | `str`: cli/cloud/auto/plan/ask |
| `effective_mode` | 实际生效的执行模式（可能因回退而变化） | `str`: cli/cloud/plan/ask |
| `has_ampersand_prefix` | 语法检测层面，原始文本是否以 `&` 开头 | `bool` |
| `prefix_routed` | 策略决策层面，`&` 前缀是否成功触发 Cloud 路由 | `bool` |

**推荐写法**:

```bash
# ✓ 推荐：plan/ask 模式不使用 & 前缀
agent -p "分析项目架构" --mode plan
agent -p "解释这段代码" --mode ask

# ✗ 不推荐：plan/ask 模式使用 & 前缀（会被忽略，造成困惑）
agent -p "& 分析项目架构" --mode plan  # & 前缀被忽略，prefix_routed=False

# ✓ 推荐：需要 Cloud 后台执行时，使用 --execution-mode cloud/auto（而非 plan/ask）
python scripts/run_iterate.py --execution-mode cloud "长时间分析任务"
```

**设计理由**:

1. **语义清晰**: plan/ask 是只读模式，Cloud 后台执行通常用于可能修改文件的任务
2. **避免冲突**: Cloud 模式（`--execution-mode cloud`）和只读模式（`--mode plan/ask`）是不同维度的配置
3. **一致性**: 与显式 `--execution-mode cli` 忽略 & 前缀的行为保持一致

**注意**: 如需在只读模式下使用 Cloud 执行，请改用 `--execution-mode cloud` 或 `--execution-mode auto`，并通过 `--mode plan` 或 `--mode ask` 指定 Cursor CLI 的运行模式（两者不冲突，前者控制执行位置，后者控制 CLI 行为）。

## 系统架构

```
Planner (规划者) → TaskQueue → Workers (执行者) → Reviewer (评审者)
     ↑                                                    │
     └────────────── 迭代循环 ←───────────────────────────┘
```

## 自我迭代模式

自我迭代模式（iterate）的编排器选择取决于执行模式：**在 `execution_mode=cli` 时默认使用 MP 编排器**（`MultiProcessOrchestrator`）；**`auto`/`cloud` 模式（默认 `auto`）强制使用 basic 编排器**。

### 编排器选择

| 编排器 | 参数 | 说明 |
|--------|------|------|
| `MultiProcessOrchestrator` | `--orchestrator mp`（`execution_mode=cli` 时的默认） | 多进程并行，适合复杂任务 |
| `Orchestrator` | `--orchestrator basic` 或 `--no-mp`（`auto`/`cloud` 模式强制） | 协程模式，适合简单任务或资源受限环境 |

**注意**: 系统默认 `execution_mode=auto`（来自 config.yaml），因此**默认使用 basic 编排器**。当 `--execution-mode` 为 `cloud` 或 `auto` 时，系统会 **强制使用 basic 编排器**，因为 Cloud/Auto 执行模式不支持多进程编排器。此时即使指定 `--orchestrator mp` 也会自动切换到 basic 编排器。**如需 MP 编排器，必须显式指定 `--execution-mode cli`**。

> **核心规则**: 编排器选择严格基于 `requested_mode`（用户请求的执行模式），而非 `effective_mode`（实际生效的模式）。即使 `requested_mode=auto/cloud` 因缺少 API Key 而回退到 CLI 执行，编排器**仍强制 basic**，不会恢复到 mp。

### 最小自我迭代运行

快速启动自我迭代的最精简命令，适合快速验证功能或离线环境下运行。

```bash
# 最小运行：跳过在线文档检查 + 仅分析不执行
python scripts/run_iterate.py --skip-online --dry-run "分析代码结构"

# 最小运行 + 禁用多进程（资源受限环境）
python scripts/run_iterate.py --skip-online --dry-run --no-mp "分析任务"

# 最小运行 + 强制本地 CLI 执行（确保不触发 Cloud）
python scripts/run_iterate.py --skip-online --dry-run --execution-mode cli "本地分析"
```

**参数组合说明**:

| 参数 | 作用 | 适用场景 |
|------|------|----------|
| `--skip-online` | 跳过在线文档检查，无网络依赖 | 离线环境、加速启动 |
| `--dry-run` | 仅分析不执行实际修改 | 测试验证、安全预览 |
| `--no-mp` | 禁用多进程编排器，使用协程模式 | 资源受限、调试场景 |
| `--execution-mode cli` | 强制本地 CLI 执行 | 确保不触发 Cloud、本地调试 |

**参数组合关系**:
- `--skip-online` + `--dry-run`: 最安全的测试组合，无网络请求、无文件修改
- `--no-mp` 与 `--execution-mode cli` 兼容，均使用本地资源
- `--execution-mode cloud/auto` 会自动禁用 MP 编排器（等效于隐式 `--no-mp`）

### 运行示例

```bash
# ===== 推荐用法（默认 auto 模式，来自 config.yaml）=====

# 直接运行，默认使用 auto 模式 + basic 编排器
# - 有 API Key: 使用 Cloud 执行
# - 无 API Key: 自动回退到本地 CLI，编排器仍为 basic
python scripts/run_iterate.py "任务描述"

# 显式指定（效果与上面相同，但更清晰）
python scripts/run_iterate.py --execution-mode auto --orchestrator basic "任务描述"

# ===== 强制本地 CLI + 多进程编排器 =====

# 如需使用 MP 编排器，必须显式指定 --execution-mode cli
python run.py --mode iterate --execution-mode cli "优化代码"
python scripts/run_iterate.py --execution-mode cli --orchestrator mp "任务描述" --workers 5

# ===== 禁用多进程，使用协程编排器 =====
python run.py --mode iterate --no-mp "任务描述"
python scripts/run_iterate.py --orchestrator basic "任务描述"

# ===== Cloud 执行模式 =====

# 强制使用 Cloud（有 API Key 时）
python scripts/run_iterate.py --execution-mode cloud "长时间分析任务"
python scripts/run_iterate.py --execution-mode cloud --orchestrator basic "后台任务"

# 使用 & 前缀路由到 Cloud（prefix_routed=True，需 cloud_agent.enabled=true）
python scripts/run_iterate.py "& 后台分析代码架构"

# 配合自动提交
python run.py --mode iterate --auto-commit --auto-push "完成功能"

# 启用差异渲染（Diff 视图增强）
python scripts/run_iterate.py --stream-console-renderer --stream-show-word-diff "重构代码"
```

**注意**: 当 `prefix_routed=True`（`&` 前缀成功路由）或使用 `--execution-mode cloud/auto` 时，即使指定 `--orchestrator mp` 也会自动切换到 basic 编排器。

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

**输出示例**（实际输出会随 config.yaml 配置而变化）:

```
[CONFIG] config_path: /path/to/config.yaml
[CONFIG] source: run.py
[CONFIG] max_iterations: 10
[CONFIG] workers: 3
[CONFIG] execution_mode: auto
[CONFIG] orchestrator: basic
[CONFIG] orchestrator_fallback: none (auto/cloud 强制 basic)
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
| `--orchestrator` | 编排器类型: `mp`/`basic` | `cli` 模式默认 `mp`；`auto`/`cloud` 模式强制 `basic` |
| `--no-mp` | 禁用多进程编排器 | False |
| `--execution-mode` | 执行模式: `cli`/`auto`/`cloud`（`cloud`/`auto` 强制使用 basic 编排器） | `auto`（来自 config.yaml，因此**默认 basic 编排器**） |
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
