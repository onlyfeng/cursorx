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
- **模型**: opus-4.5-thinking
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
