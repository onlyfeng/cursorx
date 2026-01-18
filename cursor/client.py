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
import asyncio
import subprocess
import json
import os
import shutil
from typing import Any, Optional, AsyncIterator
from pydantic import BaseModel, Field
from loguru import logger
from datetime import datetime

from cursor.streaming import StreamEventLogger, StreamEventType, parse_stream_event


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
    stream_events_enabled: bool = False
    stream_log_console: bool = True
    stream_log_detail_dir: str = "logs/stream_json/detail/"
    stream_log_raw_dir: str = "logs/stream_json/raw/"
    stream_agent_id: Optional[str] = None
    stream_agent_role: Optional[str] = None
    stream_agent_name: Optional[str] = None
    
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
    command_used: str = ""


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
    ) -> CursorAgentResult:
        """执行 Cursor Agent 任务
        
        Args:
            instruction: 给 Agent 的指令
            working_directory: 工作目录
            context: 上下文信息
            timeout: 超时时间
            
        Returns:
            执行结果
        """
        work_dir = working_directory or self.config.working_directory
        timeout_sec = timeout or self.config.timeout
        
        # 构建完整的 prompt
        full_prompt = self._build_prompt(instruction, context)
        
        started_at = datetime.now()
        
        try:
            # 尝试不同的调用方式
            result = await self._execute_cursor_agent(
                prompt=full_prompt,
                working_directory=work_dir,
                timeout=timeout_sec,
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
                command_used=result.get("command", ""),
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
    ) -> dict[str, Any]:
        """执行 agent CLI
        
        使用 agent -p "prompt" --model "model" --output-format text
        """
        # 尝试调用 agent CLI
        result = await self._try_agent_cli(prompt, working_directory, timeout)
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
    ) -> Optional[dict[str, Any]]:
        """调用 agent CLI
        
        命令格式: 
        - 非交互: agent -p "prompt" --model "model" --output-format json
        - 恢复会话: agent --resume "thread-id" -p "prompt"
        
        参考: https://cursor.com/cn/docs/cli/overview
        
        注意: 非交互模式下，Agent 具有完全写入权限
        """
        # 构建命令参数
        cmd = [self._agent_path]
        
        # 恢复会话（如果指定）
        if self.config.resume_thread_id:
            cmd.extend(["--resume", self.config.resume_thread_id])
        
        # 非交互模式使用 -p 参数（有完全写入权限）
        if self.config.non_interactive:
            cmd.extend(["-p", prompt])
        else:
            cmd.append(prompt)
        
        # 添加模型参数
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        
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

            return {
                "success": process.returncode == 0,
                "output": output,
                "error": error_msg,
                "exit_code": process.returncode,
                "command": f"agent -p '...' --model {self.config.model}" + (" --force" if self.config.force_write else ""),
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
        """构建流式事件日志器（可选）"""
        if not self.config.stream_events_enabled:
            return None
        return StreamEventLogger(
            agent_id=self.config.stream_agent_id,
            agent_role=self.config.stream_agent_role,
            agent_name=self.config.stream_agent_name,
            console=self.config.stream_log_console,
            detail_dir=self.config.stream_log_detail_dir,
            raw_dir=self.config.stream_log_raw_dir,
        )

    async def _handle_stream_json_process(
        self,
        process: asyncio.subprocess.Process,
        timeout: int,
    ) -> dict[str, Any]:
        """处理 stream-json 输出"""
        stream_logger = self._build_stream_logger()
        deadline = asyncio.get_event_loop().time() + timeout
        assistant_chunks: list[str] = []
        fallback_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stderr_task = asyncio.create_task(
            self._collect_stream(process.stderr, deadline, stderr_chunks)
        )

        try:
            async for line in self._read_stream_lines(process.stdout, deadline, strip_line=True):
                if stream_logger:
                    stream_logger.handle_raw_line(line)

                event = parse_stream_event(line)
                if event:
                    if stream_logger:
                        stream_logger.handle_event(event)

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

        return {
            "success": process.returncode == 0,
            "output": output,
            "error": error_msg,
            "exit_code": process.returncode,
            "command": f"agent -p '...' --model {self.config.model}" + (" --force" if self.config.force_write else ""),
        }

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
        """逐行读取流（带总超时）"""
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                line = await asyncio.wait_for(
                    stream.readline(),
                    timeout=min(remaining, 1.0),
                )
            except asyncio.TimeoutError:
                continue

            if not line:
                break

            text = line.decode("utf-8", errors="replace")
            if strip_line:
                text = text.rstrip("\r\n")
            yield text

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
            result = subprocess.run(
                [self._agent_path, "--help"],
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
