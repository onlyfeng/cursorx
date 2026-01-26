"""Cursor Agent 客户端

通过 Cursor CLI 的 agent 命令执行任务
参考文档: https://cursor.com/cn/docs/cli/overview

CLI 用法:
- 交互模式: agent "prompt"
- 非交互模式: agent -p "prompt" --model "model-name" --output-format text

模型支持：
- 规划任务: gpt-5.2-high (擅长高层规划)
- 编码任务: opus-4.5-thinking (擅长代码生成)
- 评审任务: opus-4.5-thinking (擅长代码审查)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from core.cloud_utils import (
    is_cloud_request as _is_cloud_request_util,
    strip_cloud_prefix as _strip_cloud_prefix_util,
)
from cursor.streaming import (
    AdvancedTerminalRenderer,
    StreamEvent,
    StreamEventLogger,
    StreamEventType,
    StreamRenderer,
    TerminalStreamRenderer,
    parse_stream_event,
)


class CursorAgentConfig(BaseModel):
    """Cursor Agent 配置

    安装:
        curl https://cursor.com/install -fsS | bash

    身份验证:
        export CURSOR_API_KEY=your_api_key_here

    CLI 参数说明 (参考: https://cursor.com/cn/docs/cli/reference/parameters):

    全局选项:
    - -p / --print: 非交互模式，输出到控制台
    - -m / --model <model>: 指定模型
    - --mode <mode>: 工作模式 (plan | ask)
      - plan: 规划模式，仅分析和规划，不修改文件
      - ask: 询问模式，回答问题和提供建议
      - 完整代理模式（agent）为默认行为：**不需要也不应传** `--mode agent`
    - --output-format <format>: text | json | stream-json
    - --stream-partial-output: 增量流式输出（配合 stream-json）
    - -f / --force: 允许直接修改文件，无需确认
    - --resume [chatId]: 恢复之前的会话
    - -a / --api-key <key>: API 密钥
    - -b / --background: 后台模式
    - --fullscreen: 全屏模式
    - --list-models: 列出所有可用模型

    命令:
    - agent login: 登录 Cursor
    - agent logout: 登出
    - agent status: 查看认证状态
    - agent models: 列出可用模型
    - agent mcp: 管理 MCP 服务器
    - agent ls: 列出并恢复会话
    - agent resume: 恢复最新会话

    注意:
    - 非交互模式下 Agent 具有完全写入权限
    - 不带 --force: 仅提议更改
    - 带 --force: 直接修改文件
    - Shell 命令超时 30 秒，不支持交互式命令
    """
    # agent CLI 路径（通过 curl https://cursor.com/install -fsS | bash 安装）
    agent_path: str = "agent"

    # API 密钥（可选，也可通过环境变量 CURSOR_API_KEY 设置）
    api_key: Optional[str] = None

    # 工作目录
    working_directory: str = "."

    # 超时设置
    timeout: int = 300  # 秒

    # 重试设置
    max_retries: int = 3
    retry_delay: float = 2.0

    # 模型设置
    # 规划类任务推荐: gpt-5.2-high
    # 编码类任务推荐: opus-4.5-thinking
    model: str = "opus-4.5-thinking"

    # 输出格式 (--output-format)
    # - "text": 纯文本输出，仅返回最终答案（推荐用于 Worker）
    # - "json": 结构化输出，便于脚本解析（推荐用于 Planner/Reviewer）
    #   成功时返回: {"type": "result", "subtype": "success", "result": "...", "session_id": "..."}
    # - "stream-json": NDJSON 格式，消息级别进度跟踪
    #   事件类型: system/init, user, assistant, tool_call, result
    output_format: str = "text"

    # 是否启用增量流式传输（--stream-partial-output）
    # 配合 stream-json 使用，可以增量流式传输变更内容
    # 每条消息会有多个 assistant 事件，需要拼接 message.content[].text
    stream_partial_output: bool = False

    # 流式事件日志配置（仅在 output_format=stream-json 时生效）
    # 默认值与 config.yaml logging.stream_json 保持同步（使用 core/config.py 中的 DEFAULT_STREAM_* 常量）
    # 注意：调用方应通过 build_cursor_agent_config() 构建配置以正确应用 config.yaml 值
    #
    # 默认关闭策略：stream_events_enabled 默认为 False，与 core.config.DEFAULT_STREAM_EVENTS_ENABLED 同步
    # 这确保了"未注入即关闭"的安全行为，避免在未显式配置时意外开启流式日志
    stream_events_enabled: bool = False  # 与 DEFAULT_STREAM_EVENTS_ENABLED 同步
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"
    stream_agent_id: Optional[str] = None
    stream_agent_role: Optional[str] = None
    stream_agent_name: Optional[str] = None

    # 流式日志聚合配置
    # - True: ASSISTANT 消息聚合后写入 detail 日志（减少日志碎片）
    # - False: 每条 ASSISTANT 消息立即写入（保持向后兼容）
    stream_log_aggregate: bool = True

    # 流式控制台渲染器配置
    # - True: 使用 TerminalStreamRenderer 进行结构化控制台输出
    # - False: 使用 StreamEventLogger 的默认控制台输出（保持向后兼容）
    stream_console_renderer: bool = False

    # TerminalStreamRenderer 详细输出模式（仅 stream_console_renderer=True 时生效）
    stream_console_verbose: bool = False

    # 流式控制台渲染模式（仅 stream_console_renderer=True 时生效）
    # - "streaming": 流式输出模式，实时显示内容增量
    # - "simple": 简单模式，仅显示关键事件
    # - "silent": 静默模式，不输出控制台内容
    stream_console_mode: str = "streaming"

    # 是否启用打字机效果（仅 stream_console_renderer=True 时生效）
    # - True: 逐字符输出，模拟打字效果
    # - False: 直接输出完整内容
    stream_typing_effect: bool = False

    # 打字延迟（秒）- 控制打字效果的速度
    stream_typing_delay: float = 0.02

    # 逐词还是逐字符模式
    # - True: 逐词输出
    # - False: 逐字符输出
    stream_word_mode: bool = True

    # 是否启用颜色输出（仅 stream_console_renderer=True 时生效）
    # - True: 使用 ANSI 颜色代码进行语法高亮和状态区分
    # - False: 纯文本输出，适合无颜色支持的终端
    stream_color_enabled: bool = True

    # 是否使用高级终端渲染器（仅 stream_console_renderer=True 时生效）
    # - True: 使用 AdvancedTerminalRenderer（支持状态栏、逐词显示等高级功能）
    # - False: 使用 TerminalStreamRenderer（基础渲染器）
    stream_advanced_renderer: bool = False

    # 是否显示状态栏（仅 stream_advanced_renderer=True 时生效）
    stream_show_status_bar: bool = True

    # 是否显示逐词差异（仅 stream_console_renderer=True 时生效）
    stream_show_word_diff: bool = False

    # 是否使用非交互模式（-p 参数）
    # 非交互模式下，Agent 具有完全写入权限
    non_interactive: bool = True

    # 是否强制修改文件（--force）
    # - True: 允许直接修改文件，无需确认（Worker 使用）
    # - False: 仅提议更改而不应用（Planner/Reviewer 使用）
    force_write: bool = False

    # 会话 ID（用于恢复之前的对话）
    resume_thread_id: Optional[str] = None

    # 后台模式（--background）
    background: bool = False

    # 全屏模式（--fullscreen）
    fullscreen: bool = False

    # Agent 工作模式（逻辑模式）
    #
    # 说明：当前 Cursor CLI 的 `--mode` 仅支持 plan/ask；
    # 完整代理模式（agent）是默认行为，因此不会向 CLI 传 `--mode agent`。
    #
    # - "plan": 规划模式，仅分析和规划，不修改文件（适合 Planner）
    # - "ask": 询问模式，回答问题和提供建议（适合咨询场景）
    # - "agent": 完整代理模式（默认行为，不传 --mode）
    # - "code": 已废弃，兼容旧配置，等价于 "agent"
    # - None: 不指定模式，使用默认行为
    mode: Optional[str] = None

    # 执行模式
    # - "cli": 通过本地 Cursor CLI 执行
    # - "cloud": 通过 Cloud API 执行
    # - "auto": 自动选择，Cloud 优先，不可用时回退到 CLI
    execution_mode: str = "auto"

    # ========== Cloud Agent 配置 ==========
    # 参考: https://cursor.com/cn/docs/cloud-agent

    # 是否启用 Cloud Agent（使用 & 前缀推送任务到云端）
    cloud_enabled: bool = False

    # Cloud Agent API 端点
    cloud_api_base: str = "https://api.cursor.com"
    cloud_agents_endpoint: str = "/v1/agents"

    # Cloud Agent 超时（云端任务通常需要更长时间）
    cloud_timeout: int = 300  # 5 分钟

    # 轮询间隔（秒）- 用于检查云端任务状态
    cloud_poll_interval: float = 2.0

    # 云端任务重试
    cloud_max_retries: int = 3

    # 是否自动检测 & 前缀并推送到云端
    auto_detect_cloud_prefix: bool = True


# 预定义模型配置
class ModelPresets:
    """模型预设配置"""
    # 规划者模型 - GPT 5.2-high 擅长高层规划和分析
    PLANNER = CursorAgentConfig(
        model="gpt-5.2-high",
        timeout=180,
    )

    # 执行者模型 - Opus 4.5 Thinking 擅长编码
    WORKER = CursorAgentConfig(
        model="opus-4.5-thinking",
        timeout=300,
    )

    # 评审者模型 - Opus 4.5 Thinking 擅长代码审查
    REVIEWER = CursorAgentConfig(
        model="opus-4.5-thinking",
        timeout=120,
    )


class CursorAgentResult(BaseModel):
    """Cursor Agent 执行结果"""
    success: bool
    output: str = ""
    error: Optional[str] = None
    exit_code: int = 0
    duration: float = 0.0  # 秒
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # 额外信息
    files_modified: list[str] = Field(default_factory=list)
    files_edited: list[str] = Field(default_factory=list)
    command_used: str = ""
    session_id: Optional[str] = None  # 从 system/init 事件提取的会话 ID


class CursorAgentClient:
    """Cursor Agent 客户端

    封装对 Cursor agent CLI 的调用
    参考: https://cursor.com/cn/docs/cli/overview

    用法:
    - 非交互模式: agent -p "prompt" --model "model" --output-format text
    - 交互模式: agent "prompt"
    """

    def __init__(self, config: Optional[CursorAgentConfig] = None):
        self.config = config or CursorAgentConfig()
        self._agent_path = self._find_agent_executable()
        self._session_id = os.urandom(8).hex()

    def _find_agent_executable(self) -> str:
        """查找 agent 可执行文件"""
        # 优先使用配置的路径
        if self.config.agent_path != "agent":
            if os.path.isfile(self.config.agent_path):
                return self.config.agent_path

        # 测试/调试场景：允许通过环境变量覆盖 agent CLI 路径
        # 仅当 agent_path 为默认值 "agent" 时生效，避免覆盖用户显式指定的路径。
        # 说明：这里仅接收“单个可执行文件路径”，不支持带参数的复合命令。
        env_agent_path = os.environ.get("AGENT_CLI_PATH")
        if env_agent_path:
            if os.path.isfile(env_agent_path):
                logger.debug(f"使用环境变量 AGENT_CLI_PATH 指定 agent CLI: {env_agent_path}")
                return env_agent_path
            logger.warning(
                f"环境变量 AGENT_CLI_PATH 指向的文件不存在: {env_agent_path}，将回退到默认查找逻辑"
            )

        # 尝试常见路径
        possible_paths = [
            # 通过 which 查找
            shutil.which("agent"),
            # Linux 常见路径
            "/usr/local/bin/agent",
            os.path.expanduser("~/.local/bin/agent"),
            os.path.expanduser("~/.cursor/bin/agent"),
            # macOS
            "/usr/local/bin/agent",
        ]

        for path in possible_paths:
            if path and os.path.isfile(path):
                logger.debug(f"找到 agent CLI: {path}")
                return path

        # 回退到默认（依赖 PATH）
        return self.config.agent_path

    async def execute(
        self,
        instruction: str,
        working_directory: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> CursorAgentResult:
        """执行 Cursor Agent 任务

        统一路由策略:
        - 若 cloud_enabled=True 且 auto_detect_cloud_prefix=True 且 instruction 为 cloud request（以 & 开头）,
          则使用 CursorCloudClient.execute()
        - 否则保持现有本地 CLI 执行

        Args:
            instruction: 给 Agent 的指令
            working_directory: 工作目录
            context: 上下文信息
            timeout: 超时时间
            session_id: 可选的会话 ID，用于恢复之前的会话（映射到 --resume 参数）

        Returns:
            执行结果
        """
        work_dir = working_directory or self.config.working_directory
        timeout_sec = timeout or self.config.timeout

        # 构建完整的 prompt
        full_prompt = self._build_prompt(instruction, context)

        started_at = datetime.now()

        # 统一路由策略: 检查是否应该使用 Cloud 执行
        if self._should_route_to_cloud(instruction):
            return await self._execute_via_cloud(
                instruction=instruction,
                working_directory=work_dir,
                context=context,
                timeout=timeout_sec,
                session_id=session_id,
                started_at=started_at,
            )

        try:
            # 尝试不同的调用方式
            result = await self._execute_cursor_agent(
                prompt=full_prompt,
                working_directory=work_dir,
                timeout=timeout_sec,
                session_id=session_id,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            return CursorAgentResult(
                success=result["success"],
                output=result.get("output", ""),
                error=result.get("error"),
                exit_code=result.get("exit_code", 0),
                duration=duration,
                started_at=started_at,
                completed_at=completed_at,
                files_modified=result.get("files_modified", []),
                files_edited=result.get("files_edited", []),
                command_used=result.get("command", ""),
                session_id=result.get("session_id"),
            )

        except asyncio.TimeoutError:
            logger.error(f"Cursor Agent 执行超时 ({timeout_sec}s)")
            return CursorAgentResult(
                success=False,
                error=f"执行超时 ({timeout_sec}s)",
                exit_code=-1,
                duration=timeout_sec,
                started_at=started_at,
            )
        except Exception as e:
            logger.error(f"Cursor Agent 执行异常: {e}")
            return CursorAgentResult(
                success=False,
                error=str(e),
                exit_code=-1,
                started_at=started_at,
            )

    def _should_route_to_cloud(self, instruction: str) -> bool:
        """检查是否应该路由到 Cloud 执行

        统一路由策略:
        - 若 cloud_enabled=True 且 auto_detect_cloud_prefix=True 且 instruction 为 cloud request,
          则返回 True

        边界情况处理:
        - None 或空字符串返回 False
        - 只有 & 的字符串返回 False（无实际内容）
        - 只有空白字符返回 False
        - & 后面需要有实际内容才认为是 cloud request

        Args:
            instruction: 给 Agent 的指令

        Returns:
            是否应该路由到 Cloud
        """
        # 检查 cloud 是否启用
        if not self.config.cloud_enabled:
            return False

        # 检查是否自动检测 cloud 前缀
        if not self.config.auto_detect_cloud_prefix:
            return False

        # 检查 instruction 是否为 cloud request
        return self._is_cloud_request(instruction)

    @staticmethod
    def _is_cloud_request(prompt: str) -> bool:
        """检测是否是云端请求（以 & 开头）

        委托给 core.cloud_utils.is_cloud_request 实现。
        这是代理方法，核心逻辑在 core.cloud_utils 模块中。

        边界情况处理:
        - None 或空字符串返回 False
        - 只有 & 的字符串返回 False（无实际内容）
        - 只有空白字符返回 False
        - 多个 & 开头只识别第一个

        Args:
            prompt: 任务 prompt

        Returns:
            是否为云端请求
        """
        return _is_cloud_request_util(prompt)

    @staticmethod
    def _strip_cloud_prefix(prompt: str) -> str:
        """去除 Cloud 前缀 &

        委托给 core.cloud_utils.strip_cloud_prefix 实现。
        这是代理方法，核心逻辑在 core.cloud_utils 模块中。

        Args:
            prompt: 可能带 & 前缀的 prompt

        Returns:
            去除前缀后的 prompt
        """
        return _strip_cloud_prefix_util(prompt)

    async def _execute_via_cloud(
        self,
        instruction: str,
        working_directory: str,
        context: Optional[dict[str, Any]],
        timeout: int,
        session_id: Optional[str],
        started_at: datetime,
    ) -> CursorAgentResult:
        """通过 CursorCloudClient 执行任务

        使用 CloudClientFactory.execute_task() 统一执行入口，确保配置来源优先级一致：
        显式参数 > config.api_key > 环境变量 CURSOR_API_KEY

        此方法与 CloudAgentExecutor.execute() 使用相同的 CloudClientFactory 方法，
        确保两条 Cloud 执行路径在 allow_write/timeout/session_id 恢复能力上行为一致。

        Args:
            instruction: 给 Agent 的指令（可能带 & 前缀）
            working_directory: 工作目录
            context: 上下文信息
            timeout: 超时时间
            session_id: 可选的会话 ID
            started_at: 开始时间

        Returns:
            执行结果
        """
        try:
            # 延迟导入避免循环依赖
            from cursor.cloud_client import CloudClientFactory

            # 构建完整的 prompt（包含上下文）
            full_prompt = self._build_prompt(instruction, context)

            logger.info(f"使用 Cloud 路由执行任务: {instruction[:50]}...")

            # 使用 CloudClientFactory.execute_task() 统一执行入口
            # 配置来源优先级: 显式参数 > config.api_key > 环境变量
            cloud_result = await CloudClientFactory.execute_task(
                prompt=full_prompt,
                agent_config=self.config,
                working_directory=working_directory,
                timeout=timeout,
                allow_write=self.config.force_write,
                session_id=session_id,
                wait=True,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            return CursorAgentResult(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                exit_code=0 if cloud_result.success else -1,
                duration=duration,
                started_at=started_at,
                completed_at=completed_at,
                files_modified=cloud_result.files_modified,
                command_used="cloud-agent",
                session_id=cloud_result.task.task_id if cloud_result.task else None,
            )

        except asyncio.TimeoutError:
            logger.error(f"Cloud Agent 执行超时 ({timeout}s)")
            return CursorAgentResult(
                success=False,
                error=f"Cloud 执行超时 ({timeout}s)",
                exit_code=-1,
                duration=timeout,
                started_at=started_at,
            )
        except Exception as e:
            logger.error(f"Cloud Agent 执行异常: {e}")
            return CursorAgentResult(
                success=False,
                error=str(e),
                exit_code=-1,
                started_at=started_at,
            )

    def _build_prompt(self, instruction: str, context: Optional[dict[str, Any]] = None) -> str:
        """构建完整的 prompt"""
        parts = [instruction]

        if context:
            parts.append("\n\n## 上下文信息")
            for key, value in context.items():
                if isinstance(value, (dict, list)):
                    parts.append(f"\n### {key}\n```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```")
                else:
                    parts.append(f"\n### {key}\n{value}")

        return "\n".join(parts)

    async def _execute_cursor_agent(
        self,
        prompt: str,
        working_directory: str,
        timeout: int,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行 agent CLI

        使用 agent -p "prompt" --model "model" --output-format text

        Args:
            prompt: 任务 prompt
            working_directory: 工作目录
            timeout: 超时时间
            session_id: 可选的会话 ID，用于恢复之前的会话
        """
        # 尝试调用 agent CLI
        result = await self._try_agent_cli(prompt, working_directory, timeout, session_id)
        if result:
            return result

        # 回退：模拟模式（开发/测试用）
        logger.warning("无法调用 agent CLI，使用模拟模式")
        return await self._mock_execution(prompt, working_directory)

    async def _try_agent_cli(
        self,
        prompt: str,
        working_directory: str,
        timeout: int,
        session_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """调用 agent CLI

        命令格式:
        - 非交互: agent -p "prompt" --model "model" --output-format json
        - 恢复会话: agent --resume "thread-id" -p "prompt"

        参考: https://cursor.com/cn/docs/cli/overview

        注意: 非交互模式下，Agent 具有完全写入权限

        Args:
            prompt: 任务 prompt
            working_directory: 工作目录
            timeout: 超时时间
            session_id: 可选的会话 ID，优先于 config.resume_thread_id
        """
        # 构建命令参数
        # 兼容测试场景：当 _agent_path 为 .py 脚本时，用当前 Python 解释器启动，
        # 避免依赖脚本的可执行权限位。
        if self._agent_path.endswith(".py"):
            cmd = [sys.executable, self._agent_path]
        else:
            cmd = [self._agent_path]

        # 恢复会话（session_id 参数优先于配置项）
        resume_id = session_id or self.config.resume_thread_id
        if resume_id:
            cmd.extend(["--resume", resume_id])

        # 非交互模式使用 -p 参数（有完全写入权限）
        if self.config.non_interactive:
            cmd.extend(["-p", prompt])
        else:
            cmd.append(prompt)

        # 添加模型参数
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        # 添加工作模式（plan/ask）
        #
        # 重要：当前 Cursor CLI 的 --mode 仅接受 plan/ask。
        # - agent 模式是默认行为，不需要也不应传 --mode agent
        # - 兼容旧的 "code" 模式：视为 "agent"（同样不传 --mode）
        mode_arg: Optional[str] = None
        if self.config.mode:
            effective_mode = "agent" if self.config.mode == "code" else self.config.mode
            if effective_mode in ("plan", "ask"):
                mode_arg = effective_mode
        if mode_arg:
            cmd.extend(["--mode", mode_arg])

        # 添加输出格式
        # text: 纯文本输出
        # json: 结构化输出，便于解析
        # stream-json: 消息级别进度跟踪
        if self.config.output_format:
            cmd.extend(["--output-format", self.config.output_format])

        # 增量流式传输（配合 stream-json 使用）
        if self.config.stream_partial_output:
            cmd.append("--stream-partial-output")

        # 强制修改文件（允许直接修改，无需确认）
        if self.config.force_write:
            cmd.append("--force")

        try:
            logger.debug(f"执行命令: agent -p '...' --model {self.config.model}")

            # 构建环境变量（支持 API 密钥）
            env = os.environ.copy()
            if self.config.api_key:
                env["CURSOR_API_KEY"] = self.config.api_key

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # 创建子进程后立即设置更大的缓冲区限制（32MB），避免超长行导致异常
            # asyncio.StreamReader 默认 _limit 为 64KB，对于大型 JSON 输出可能不够
            if process.stdout and hasattr(process.stdout, '_limit'):
                process.stdout._limit = 32 * 1024 * 1024  # 32MB
                logger.debug("已设置 stdout 缓冲区限制为 32MB")
            if process.stderr and hasattr(process.stderr, '_limit'):
                process.stderr._limit = 32 * 1024 * 1024  # 32MB
                logger.debug("已设置 stderr 缓冲区限制为 32MB")

            if self.config.output_format == "stream-json":
                return await self._handle_stream_json_process(process, timeout)

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # 构建错误信息（包含更多上下文）
            error_msg = None
            if process.returncode != 0:
                error_parts = []
                if error_output.strip():
                    error_parts.append(f"stderr: {error_output.strip()}")
                if output.strip() and not error_output.strip():
                    # 如果 stderr 为空但 stdout 有内容，可能错误在 stdout 中
                    error_parts.append(f"stdout: {output.strip()[:500]}")
                if not error_parts:
                    error_parts.append(f"exit_code: {process.returncode} (无错误输出)")
                error_msg = "; ".join(error_parts)
                logger.warning(f"agent CLI 返回非零退出码 {process.returncode}: {error_msg[:200]}")

            # 构建命令描述（用于日志）
            cmd_desc = f"agent -p '...' --model {self.config.model}"
            if mode_arg:
                cmd_desc += f" --mode {mode_arg}"
            if self.config.force_write:
                cmd_desc += " --force"

            return {
                "success": process.returncode == 0,
                "output": output,
                "error": error_msg,
                "exit_code": process.returncode,
                "command": cmd_desc,
            }

        except FileNotFoundError:
            logger.error(f"找不到 agent CLI: {self._agent_path}")
            logger.info("请先安装: curl https://cursor.com/install -fsS | bash")
            return None
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"agent CLI 执行失败: {e}")
            return None

    def _build_stream_logger(self) -> Optional[StreamEventLogger]:
        """构建流式事件日志器（可选）

        根据配置项控制日志聚合和控制台渲染方式：
        - stream_log_aggregate: 控制 ASSISTANT 消息是否聚合后写入
        - stream_console_renderer: 是否使用 TerminalStreamRenderer
        - stream_console_verbose: TerminalStreamRenderer 详细模式
        """
        if not self.config.stream_events_enabled:
            return None

        # 控制台输出：如果使用 TerminalStreamRenderer，则 StreamEventLogger 不输出到控制台
        # TerminalStreamRenderer 会单独处理控制台输出
        console_output = self.config.stream_log_console
        if self.config.stream_console_renderer:
            # 使用 TerminalStreamRenderer 时，StreamEventLogger 不输出到控制台
            console_output = False

        return StreamEventLogger(
            agent_id=self.config.stream_agent_id,
            agent_role=self.config.stream_agent_role,
            agent_name=self.config.stream_agent_name,
            console=console_output,
            detail_dir=self.config.stream_log_detail_dir,
            raw_dir=self.config.stream_log_raw_dir,
            aggregate_assistant_messages=self.config.stream_log_aggregate,
        )

    def _build_terminal_renderer(self) -> Optional[StreamRenderer]:
        """构建终端流式渲染器（可选）

        仅当 stream_console_renderer=True 时返回渲染器实例。
        根据配置选择使用基础渲染器或高级渲染器：
        - stream_advanced_renderer=True: 使用 AdvancedTerminalRenderer
        - stream_advanced_renderer=False: 使用 TerminalStreamRenderer

        Returns:
            StreamRenderer 实例，如果未启用则返回 None
        """
        if not self.config.stream_console_renderer:
            return None

        if self.config.stream_advanced_renderer:
            # 使用高级渲染器，支持状态栏、打字效果等
            return AdvancedTerminalRenderer(
                use_color=self.config.stream_color_enabled,
                typing_delay=self.config.stream_typing_delay if self.config.stream_typing_effect else 0.0,
                word_mode=self.config.stream_word_mode,
                show_status_bar=self.config.stream_show_status_bar,
                show_word_diff=self.config.stream_show_word_diff,
            )
        else:
            # 使用基础渲染器
            return TerminalStreamRenderer(
                verbose=self.config.stream_console_verbose,
                show_word_diff=self.config.stream_show_word_diff,
            )

    async def _handle_stream_json_process(
        self,
        process: asyncio.subprocess.Process,
        timeout: int,
    ) -> dict[str, Any]:
        """处理 stream-json 输出

        使用 StreamRenderer 接口进行控制台渲染，日志记录与渲染解耦。
        对于 AdvancedTerminalRenderer 特有的 start/finish 方法，
        通过鸭子类型（hasattr）检查进行处理。

        增强功能:
        - 累计 DiffInfo.path / ToolCallInfo.path 形成去重后的 files_modified/files_edited
        - 从 system/init 事件中提取 session_id
        """
        stream_logger = self._build_stream_logger()
        terminal_renderer = self._build_terminal_renderer()
        deadline = asyncio.get_event_loop().time() + timeout
        assistant_chunks: list[str] = []
        fallback_chunks: list[str] = []
        stderr_chunks: list[str] = []

        # 渲染器状态跟踪
        tool_count = 0
        diff_count = 0
        accumulated_text_len = 0

        # 文件跟踪 (去重使用 set)
        files_modified_set: set[str] = set()  # 写入/创建的文件
        files_edited_set: set[str] = set()    # 编辑/修改的文件

        # 会话 ID 跟踪
        session_id: Optional[str] = None

        stderr_task = asyncio.create_task(
            self._collect_stream(process.stderr, deadline, stderr_chunks)
        )

        # 调用高级渲染器的 start 方法（如果存在）
        if terminal_renderer and hasattr(terminal_renderer, 'start'):
            terminal_renderer.start()

        try:
            async for line in self._read_stream_lines(process.stdout, deadline, strip_line=True):
                # 日志记录（与渲染解耦）
                if stream_logger:
                    stream_logger.handle_raw_line(line)

                event = parse_stream_event(line)
                if event:
                    # 日志处理
                    if stream_logger:
                        stream_logger.handle_event(event)

                    # 使用 StreamRenderer 接口渲染控制台输出
                    if terminal_renderer:
                        self._render_event_to_terminal(
                            terminal_renderer,
                            event,
                            tool_count,
                            diff_count,
                            accumulated_text_len,
                        )

                        # 更新计数器（渲染方法会使用这些值）
                        if event.type == StreamEventType.TOOL_STARTED:
                            tool_count += 1
                        elif event.type in (StreamEventType.DIFF_STARTED, StreamEventType.DIFF):
                            diff_count += 1
                        if event.type == StreamEventType.ASSISTANT and event.content:
                            accumulated_text_len += len(event.content)

                    # 从 system/init 事件中提取 session_id
                    if event.type == StreamEventType.SYSTEM_INIT:
                        session_id = event.data.get("session_id")

                    # 收集文件修改信息
                    if event.type in (StreamEventType.TOOL_STARTED, StreamEventType.TOOL_COMPLETED):
                        if event.tool_call and event.tool_call.path:
                            # 写入操作 -> files_modified
                            if event.tool_call.tool_type == "write":
                                files_modified_set.add(event.tool_call.path)
                            # 编辑/替换操作 -> files_edited
                            elif event.tool_call.is_diff or event.tool_call.tool_type in ("edit", "str_replace"):
                                files_edited_set.add(event.tool_call.path)

                    # 差异事件中收集编辑的文件
                    if event.type in (StreamEventType.DIFF, StreamEventType.DIFF_STARTED, StreamEventType.DIFF_COMPLETED):
                        # 优先从 diff_info 提取路径
                        if event.diff_info and event.diff_info.path:
                            files_edited_set.add(event.diff_info.path)
                        # 回退到 tool_call 提取路径
                        elif event.tool_call and event.tool_call.path:
                            files_edited_set.add(event.tool_call.path)

                    # 收集输出内容
                    if event.type == StreamEventType.ASSISTANT:
                        if event.content:
                            assistant_chunks.append(event.content)
                    elif event.type == StreamEventType.MESSAGE:
                        if event.content:
                            fallback_chunks.append(event.content)

            await self._wait_process_with_deadline(process, deadline)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        finally:
            try:
                await stderr_task
            except asyncio.TimeoutError:
                stderr_task.cancel()
            except Exception as e:
                logger.warning(f"读取 stderr 失败: {e}")

            # 调用高级渲染器的 finish 方法（如果存在）
            if terminal_renderer and hasattr(terminal_renderer, 'finish'):
                terminal_renderer.finish()

            # 关闭日志记录器
            if stream_logger:
                stream_logger.close()

        output = "".join(assistant_chunks)
        if not output and fallback_chunks:
            output = "\n".join(fallback_chunks)

        error_output = "".join(stderr_chunks)

        error_msg = None
        if process.returncode != 0:
            error_parts = []
            if error_output.strip():
                error_parts.append(f"stderr: {error_output.strip()}")
            if output.strip() and not error_output.strip():
                error_parts.append(f"stdout: {output.strip()[:500]}")
            if not error_parts:
                error_parts.append(f"exit_code: {process.returncode} (无错误输出)")
            error_msg = "; ".join(error_parts)
            logger.warning(f"agent CLI 返回非零退出码 {process.returncode}: {error_msg[:200]}")

        # 构建命令描述（用于日志）
        cmd_desc = f"agent -p '...' --model {self.config.model}"
        # 与 _call_agent_cli 的行为保持一致：仅在 plan/ask 时展示 --mode
        mode_arg: Optional[str] = None
        if self.config.mode:
            effective_mode = "agent" if self.config.mode == "code" else self.config.mode
            if effective_mode in ("plan", "ask"):
                mode_arg = effective_mode
        if mode_arg:
            cmd_desc += f" --mode {mode_arg}"
        if self.config.force_write:
            cmd_desc += " --force"

        return {
            "success": process.returncode == 0,
            "output": output,
            "error": error_msg,
            "exit_code": process.returncode,
            "command": cmd_desc,
            "files_modified": list(files_modified_set),
            "files_edited": list(files_edited_set),
            "session_id": session_id,
        }

    def _render_event_to_terminal(
        self,
        renderer: StreamRenderer,
        event: StreamEvent,
        tool_count: int,
        diff_count: int,
        accumulated_text_len: int,
    ) -> None:
        """将事件渲染到终端

        使用 StreamRenderer 接口方法进行渲染。
        对于 AdvancedTerminalRenderer 的 render_event 方法，
        通过鸭子类型检查直接调用。

        Args:
            renderer: StreamRenderer 实例
            event: 流式事件
            tool_count: 工具调用计数
            diff_count: 差异操作计数
            accumulated_text_len: 累积文本长度
        """
        # 如果渲染器支持 render_event 方法（如 AdvancedTerminalRenderer），
        # 直接使用该方法处理所有事件
        if hasattr(renderer, 'render_event'):
            renderer.render_event(event)
            return

        # 否则使用 StreamRenderer 接口方法
        if event.type == StreamEventType.SYSTEM_INIT:
            renderer.render_init(event.model)
        elif event.type == StreamEventType.ASSISTANT:
            if event.content:
                new_len = accumulated_text_len + len(event.content)
                renderer.render_assistant(event.content, new_len)
        elif event.type == StreamEventType.TOOL_STARTED:
            renderer.render_tool_started(tool_count + 1, event.tool_call)
        elif event.type == StreamEventType.TOOL_COMPLETED:
            renderer.render_tool_completed(event.tool_call)
        elif event.type == StreamEventType.DIFF_STARTED:
            renderer.render_diff_started(diff_count + 1, event.tool_call)
        elif event.type == StreamEventType.DIFF_COMPLETED:
            renderer.render_diff_completed(
                event.tool_call, event.diff_info, show_diff=True
            )
        elif event.type == StreamEventType.DIFF:
            renderer.render_diff(diff_count + 1, event.diff_info, show_diff=True)
        elif event.type == StreamEventType.RESULT:
            renderer.render_result(
                event.duration_ms, tool_count, accumulated_text_len
            )
        elif event.type == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            renderer.render_error(error)

    async def _wait_process_with_deadline(
        self,
        process: asyncio.subprocess.Process,
        deadline: float,
    ) -> None:
        """等待进程结束（带总超时）"""
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        await asyncio.wait_for(process.wait(), timeout=remaining)

    async def _read_stream_lines(
        self,
        stream: asyncio.StreamReader,
        deadline: float,
        strip_line: bool = True,
    ) -> AsyncIterator[str]:
        """逐行读取流（带总超时）

        支持处理超长行：
        - 设置更大的 limit（32MB）
        - 捕获 ValueError/LimitOverrunError 异常（行超过 limit 时抛出）
        - 使用分块读取作为后备方案，确保超长行被正确读取而非跳过
        - 特别处理 "Separator is found, but chunk is longer than limit" 错误
        - 添加日志记录超长行的处理结果，便于调试
        """
        # 设置更大的缓冲区限制（32MB），避免超长行导致异常
        # asyncio.StreamReader 默认 limit 为 64KB
        if hasattr(stream, '_limit'):
            stream._limit = 32 * 1024 * 1024  # 32MB

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            line: bytes = b''
            long_line_handled = False  # 标记是否通过超长行处理获取了内容

            try:
                line = await asyncio.wait_for(
                    stream.readline(),
                    timeout=min(remaining, 1.0),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.LimitOverrunError as e:
                # "Separator is found, but chunk is longer than limit" 错误
                # 需要读取缓冲区中的数据
                logger.warning(f"检测到超长行 (LimitOverrunError): consumed={e.consumed} bytes")
                try:
                    # 读取超出的数据直到换行符
                    line = await stream.readuntil(b'\n')
                    long_line_handled = True
                    logger.debug(f"超长行读取成功 (readuntil): {len(line)} bytes")
                except asyncio.IncompleteReadError as ire:
                    # 如果流结束了，使用已读取的部分
                    line = ire.partial
                    long_line_handled = True
                    logger.debug(f"超长行读取完成 (流结束): {len(line)} bytes")
                except asyncio.LimitOverrunError as inner_e:
                    # 如果还是太长，使用分块读取
                    logger.warning(f"再次触发 LimitOverrunError: consumed={inner_e.consumed} bytes，使用分块读取")
                    try:
                        line = await self._read_long_line(stream, deadline)
                        long_line_handled = True
                        logger.info(f"超长行分块读取成功: {len(line)} bytes")
                    except Exception as read_err:
                        logger.error(f"分块读取超长行失败: {read_err}")
                        # 尝试跳过这一行继续处理
                        continue
                except Exception as inner_e:
                    logger.warning(f"处理超长行时发生异常: {inner_e}")
                    # 尝试使用分块读取作为最后手段
                    try:
                        line = await self._read_long_line(stream, deadline)
                        long_line_handled = True
                        logger.info(f"超长行异常恢复读取成功: {len(line)} bytes")
                    except Exception as read_err:
                        logger.error(f"异常恢复分块读取失败: {read_err}")
                        continue
            except ValueError as e:
                # 行超过 limit 时抛出 ValueError
                # 使用分块读取作为后备方案
                logger.warning(f"检测到超长行 (ValueError): {e}")
                try:
                    line = await self._read_long_line(stream, deadline)
                    long_line_handled = True
                    logger.info(f"超长行分块读取成功 (ValueError): {len(line)} bytes")
                except Exception as read_err:
                    logger.error(f"分块读取超长行失败 (ValueError): {read_err}")
                    continue
            except Exception as e:
                # 捕获其他读取异常，记录日志但不崩溃
                error_msg = str(e)
                if "Separator is found" in error_msg or "chunk is longer than limit" in error_msg:
                    # 特定的超长行错误
                    logger.warning(f"检测到超长行: {error_msg}")
                    try:
                        line = await self._read_long_line(stream, deadline)
                        long_line_handled = True
                        logger.info(f"超长行分块读取成功 (通用异常): {len(line)} bytes")
                    except Exception as read_err:
                        logger.error(f"分块读取超长行失败 (通用异常): {read_err}")
                        continue
                else:
                    logger.warning(f"读取流时发生异常: {e}")
                    continue

            if not line:
                break

            # 记录超长行处理结果
            if long_line_handled:
                logger.debug(f"超长行已成功处理并返回: {len(line)} bytes")

            text = line.decode("utf-8", errors="replace")
            if strip_line:
                text = text.rstrip("\r\n")
            yield text

    async def _read_long_line(
        self,
        stream: asyncio.StreamReader,
        deadline: float,
        chunk_size: int = 1024 * 1024,  # 1MB 分块
    ) -> bytes:
        """分块读取超长行

        当 readline() 因行过长失败时，使用此方法分块读取直到换行符。

        Args:
            stream: 异步流读取器
            deadline: 超时截止时间
            chunk_size: 每次读取的块大小

        Returns:
            读取到的完整行（包含换行符）
        """
        chunks: list[bytes] = []
        max_line_size = 32 * 1024 * 1024  # 最大行大小 32MB

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                chunk = await asyncio.wait_for(
                    stream.read(chunk_size),
                    timeout=min(remaining, 5.0),
                )
            except asyncio.TimeoutError:
                continue

            if not chunk:
                break

            chunks.append(chunk)

            # 检查是否包含换行符
            if b'\n' in chunk:
                break

            # 防止内存溢出
            total_size = sum(len(c) for c in chunks)
            if total_size > max_line_size:
                logger.warning(f"超长行超过最大限制 ({max_line_size} bytes)，截断处理")
                break

        return b''.join(chunks)

    async def _collect_stream(
        self,
        stream: asyncio.StreamReader,
        deadline: float,
        buffer: list[str],
    ) -> None:
        """收集流输出到缓冲区"""
        async for line in self._read_stream_lines(stream, deadline, strip_line=False):
            buffer.append(line)

    async def _mock_execution(
        self,
        prompt: str,
        working_directory: str,
    ) -> dict[str, Any]:
        """模拟执行（开发/测试用）"""
        logger.info("[Mock] 模拟执行 Cursor Agent")
        logger.debug(f"[Mock] 工作目录: {working_directory}")
        logger.debug(f"[Mock] 指令: {prompt[:200]}...")

        # 模拟一些处理时间
        await asyncio.sleep(0.5)

        return {
            "success": True,
            "output": f"[Mock] 已处理任务:\n{prompt[:100]}...\n\n模拟执行成功。",
            "error": None,
            "exit_code": 0,
            "command": "mock",
        }

    async def execute_with_retry(
        self,
        instruction: str,
        working_directory: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> CursorAgentResult:
        """带重试的执行"""
        retries = max_retries or self.config.max_retries
        last_result = None

        for attempt in range(retries):
            result = await self.execute(instruction, working_directory, context)

            if result.success:
                return result

            last_result = result
            logger.warning(f"Cursor Agent 执行失败 (尝试 {attempt + 1}/{retries}): {result.error}")

            if attempt < retries - 1:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # 指数退避

        return last_result or CursorAgentResult(success=False, error="所有重试均失败")

    def check_agent_available(self) -> bool:
        """检查 agent CLI 是否可用"""
        try:
            # 兼容测试场景：当 _agent_path 为 .py 脚本时，使用 Python 解释器启动
            if self._agent_path.endswith(".py"):
                cmd = [sys.executable, self._agent_path, "--help"]
            else:
                cmd = [self._agent_path, "--help"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def install_instructions() -> str:
        """返回安装说明"""
        return "curl https://cursor.com/install -fsS | bash"

    async def list_models(self) -> list[str]:
        """列出所有可用模型

        使用 agent models 或 agent --list-models
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._agent_path, "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=30,
            )

            output = stdout.decode("utf-8", errors="replace")
            # 解析模型列表（每行一个模型）
            models = [line.strip() for line in output.split("\n") if line.strip()]
            return models

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def list_models_sync(self) -> list[str]:
        """同步版本：列出所有可用模型"""
        try:
            result = subprocess.run(
                [self._agent_path, "models"],
                capture_output=True,
                timeout=30,
            )

            if result.returncode == 0:
                output = result.stdout.decode("utf-8", errors="replace")
                models = [line.strip() for line in output.split("\n") if line.strip()]
                return models
            return []
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    async def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话

        使用 agent ls 命令
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._agent_path, "ls",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=30,
            )

            output = stdout.decode("utf-8", errors="replace")
            # 解析会话列表
            sessions = []
            for line in output.split("\n"):
                if line.strip():
                    sessions.append({"raw": line.strip()})
            return sessions

        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return []

    async def resume_session(self, session_id: Optional[str] = None) -> CursorAgentResult:
        """恢复会话

        Args:
            session_id: 会话 ID，如果为 None 则恢复最新会话
        """
        try:
            cmd = [self._agent_path]
            if session_id:
                cmd.extend(["--resume", session_id])
            else:
                cmd.append("resume")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout,
            )

            return CursorAgentResult(
                success=process.returncode == 0,
                output=stdout.decode("utf-8", errors="replace"),
                error=stderr.decode("utf-8", errors="replace") if process.returncode != 0 else None,
                exit_code=process.returncode,
            )

        except Exception as e:
            logger.error(f"恢复会话失败: {e}")
            return CursorAgentResult(success=False, error=str(e), exit_code=-1)

    async def get_status(self) -> dict[str, Any]:
        """获取认证状态

        使用 agent status 命令
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._agent_path, "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30,
            )

            return {
                "authenticated": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if process.returncode != 0 else None,
            }

        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return {"authenticated": False, "error": str(e)}

    @staticmethod
    def parse_json_output(output: str) -> dict[str, Any]:
        """解析 JSON 格式的输出

        JSON 输出格式:
        {
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": 1234,
            "result": "<完整助手文本>",
            "session_id": "<uuid>"
        }
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"type": "error", "result": output}

    @staticmethod
    def parse_stream_json_output(output: str) -> list[dict[str, Any]]:
        """解析 stream-json 格式的输出 (NDJSON)

        每行是一个 JSON 对象，事件类型包括:
        - system/init: 会话初始化
        - user: 用户消息
        - assistant: 助手消息
        - tool_call: 工具调用 (started/completed)
        - result: 最终结果
        """
        events = []
        for line in output.split("\n"):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events


class CursorAgentPool:
    """Cursor Agent 连接池

    管理多个 Cursor Agent 客户端实例
    用于并行执行多个任务
    """

    def __init__(self, size: int = 3, config: Optional[CursorAgentConfig] = None):
        self.size = size
        self.config = config or CursorAgentConfig()
        self._clients: list[CursorAgentClient] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._initialized = False

    async def initialize(self) -> None:
        """初始化连接池"""
        if self._initialized:
            return

        for i in range(self.size):
            client = CursorAgentClient(self.config)
            self._clients.append(client)
            await self._available.put(client)

        self._initialized = True
        logger.info(f"Cursor Agent 连接池已初始化: {self.size} 个客户端")

    async def acquire(self, timeout: Optional[float] = None) -> CursorAgentClient:
        """获取一个可用的客户端"""
        if not self._initialized:
            await self.initialize()

        try:
            if timeout:
                return await asyncio.wait_for(self._available.get(), timeout=timeout)
            return await self._available.get()
        except asyncio.TimeoutError:
            raise RuntimeError("获取 Cursor Agent 客户端超时")

    async def release(self, client: CursorAgentClient) -> None:
        """释放客户端回池"""
        await self._available.put(client)

    async def execute(
        self,
        instruction: str,
        working_directory: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> CursorAgentResult:
        """使用池中的客户端执行任务"""
        client = await self.acquire()
        try:
            return await client.execute(instruction, working_directory, context)
        finally:
            await self.release(client)
