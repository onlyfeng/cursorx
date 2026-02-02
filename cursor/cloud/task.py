"""任务管理模块

提供任务状态、任务结果、任务选项和任务管理客户端:
- TaskStatus: 任务状态枚举
- TaskResult: 任务执行结果
- CloudTaskOptions: 云端任务选项
- CloudTask: 云端任务
- CloudTaskClient: 任务管理客户端（轮询、WebSocket、SSE）
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:
    from ..streaming import StreamEvent
    from .auth import CloudAuthManager

# 导入异常类和错误处理工具
from .exceptions import (
    AuthError,
    AuthErrorCode,
    CloudAgentError,
    NetworkError,
    RateLimitError,
    TaskError,
    handle_http_error,
)
from .retry import RetryConfig

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class TaskStatus(str, Enum):
    """任务状态

    参考: https://cursor.com/cn/docs/cloud-agent
    """

    PENDING = "pending"  # 等待执行
    QUEUED = "queued"  # 已加入队列
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    TIMEOUT = "timeout"  # 超时


@dataclass
class TaskResult:
    """任务执行结果"""

    task_id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    duration_ms: int = 0
    events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """任务是否成功完成"""
        return self.status == TaskStatus.COMPLETED and self.error is None

    @property
    def is_terminal(self) -> bool:
        """任务是否已结束（无论成功与否）"""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "events": self.events,
            "metadata": self.metadata,
        }


@dataclass
class CloudTaskOptions:
    """云端任务选项"""

    # 模型选择
    model: str | None = None

    # 工作目录
    working_directory: str | None = None

    # 是否允许写入文件
    allow_write: bool = True

    # 超时时间（秒）
    timeout: int = 600

    # 任务优先级 (1-10, 10 最高)
    priority: int = 5

    # 任务标签（用于分类和查询）
    tags: list[str] = field(default_factory=list)

    # 上下文文件（会被上传到云端）
    context_files: list[str] = field(default_factory=list)

    # 环境变量
    env: dict[str, str] = field(default_factory=dict)

    # 回调 URL（任务完成时通知）
    callback_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model,
            "working_directory": self.working_directory,
            "allow_write": self.allow_write,
            "timeout": self.timeout,
            "priority": self.priority,
            "tags": self.tags,
            "context_files": self.context_files,
            "env": self.env,
            "callback_url": self.callback_url,
        }


@dataclass
class CloudTask:
    """云端任务"""

    # 任务 ID（由云端分配）
    task_id: str

    # 任务状态
    status: TaskStatus = TaskStatus.PENDING

    # 原始 prompt（不含 & 前缀）
    prompt: str = ""

    # 任务选项
    options: CloudTaskOptions | None = None

    # 创建时间
    created_at: datetime = field(default_factory=datetime.now)

    # 开始执行时间
    started_at: datetime | None = None

    # 完成时间
    completed_at: datetime | None = None

    # 任务输出
    output: str | None = None

    # 错误信息
    error: str | None = None

    # 修改的文件列表
    files_modified: list[str] = field(default_factory=list)

    # 进度（0-100）
    progress: int = 0

    # 当前执行的步骤描述
    current_step: str | None = None

    @property
    def is_running(self) -> bool:
        """是否正在执行"""
        return self.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING)

    @property
    def is_completed(self) -> bool:
        """是否已完成（成功或失败）"""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT)

    @property
    def is_success(self) -> bool:
        """是否成功完成"""
        return self.status == TaskStatus.COMPLETED

    @property
    def duration(self) -> float | None:
        """任务执行时长（秒）"""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return (end_time - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "prompt": self.prompt,
            "options": self.options.to_dict() if self.options else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output": self.output,
            "error": self.error,
            "files_modified": self.files_modified,
            "progress": self.progress,
            "current_step": self.current_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudTask:
        """从字典创建"""
        options = None
        if data.get("options"):
            opts = data["options"]
            options = CloudTaskOptions(
                model=opts.get("model"),
                working_directory=opts.get("working_directory"),
                allow_write=opts.get("allow_write", True),
                timeout=opts.get("timeout", 600),
                priority=opts.get("priority", 5),
                tags=opts.get("tags", []),
                context_files=opts.get("context_files", []),
                env=opts.get("env", {}),
                callback_url=opts.get("callback_url"),
            )

        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data.get("status", "pending")),
            prompt=data.get("prompt", ""),
            options=options,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            output=data.get("output"),
            error=data.get("error"),
            files_modified=data.get("files_modified", []),
            progress=data.get("progress", 0),
            current_step=data.get("current_step"),
        )


class CloudTaskClient:
    """Cloud 任务管理客户端

    提供任务状态轮询、WebSocket 实时连接、流式事件处理功能。
    支持增强错误处理、指数退避重试、限流响应处理。

    用法:
        client = CloudTaskClient()

        # 长轮询等待任务完成
        result = await client.poll_task_status("task-123", timeout=300)

        # WebSocket 实时监听（如果支持）
        async for event in client.watch_task_ws("task-123"):
            print(event)

        # 流式执行任务
        async for event in client.execute_streaming("实现功能", "gpt-5"):
            print(event)

    Attributes:
        retry_config: 重试配置
    """

    def __init__(
        self,
        auth_manager: CloudAuthManager | None = None,
        agent_path: str = "agent",
        api_base_url: str = "https://api.cursor.com",
        retry_config: RetryConfig | None = None,
    ):
        # 延迟导入避免循环依赖
        if auth_manager is None:
            from .auth import CloudAuthManager

            auth_manager = CloudAuthManager()
        self.auth_manager = auth_manager
        self.agent_path = agent_path
        self.api_base_url = api_base_url
        self.retry_config = retry_config or RetryConfig()
        self._ws_connections: dict[str, Any] = {}

    # ========== 任务状态轮询 ==========

    async def poll_task_status(
        self,
        task_id: str,
        timeout: float = 300.0,
        interval: float = 2.0,
        on_status_change: Callable[[TaskStatus], None] | None = None,
    ) -> TaskResult:
        """长轮询等待任务完成

        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）
            interval: 轮询间隔（秒）
            on_status_change: 状态变化回调

        Returns:
            TaskResult: 任务执行结果

        Raises:
            AuthError: 认证失败
            asyncio.TimeoutError: 轮询超时
        """
        # 延迟导入流式事件类型

        start_time = time.time()
        last_status: TaskStatus | None = None
        collected_events: list[StreamEvent] = []

        logger.info(f"开始轮询任务状态: {task_id}, 超时={timeout}s, 间隔={interval}s")

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"任务轮询超时: {task_id}")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.TIMEOUT,
                    error=f"轮询超时 ({timeout}s)",
                    duration_ms=int(elapsed * 1000),
                    events=collected_events,
                )

            try:
                # 查询任务状态
                current_result = await self._query_task_status(task_id)
                current_status = current_result.status

                # 收集事件
                if current_result.events:
                    collected_events.extend(current_result.events)

                # 状态变化通知
                if current_status != last_status:
                    logger.debug(f"任务 {task_id} 状态变化: {last_status} -> {current_status}")
                    if on_status_change:
                        on_status_change(current_status)
                    last_status = current_status

                # 检查是否终态
                if current_result.is_terminal:
                    current_result.duration_ms = int((time.time() - start_time) * 1000)
                    current_result.events = collected_events
                    logger.info(f"任务 {task_id} 完成: {current_status.value}")
                    return current_result

            except Exception as e:
                logger.warning(f"查询任务状态失败: {e}")
                # 继续轮询，不中断

            # 等待下一次轮询
            await asyncio.sleep(interval)

    async def _query_task_status(self, task_id: str) -> TaskResult:
        """查询单次任务状态

        使用 agent CLI 或 HTTP API 查询任务状态
        """
        # 方式1: 使用 agent CLI 查询（如果有相关命令）
        try:
            result = await self._query_via_cli(task_id)
            if result:
                return result
        except Exception as e:
            logger.debug(f"CLI 查询失败: {e}")

        # 方式2: 使用 HTTP API 查询
        if HTTPX_AVAILABLE:
            try:
                result = await self._query_via_http(task_id)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"HTTP 查询失败: {e}")

        # 默认返回运行中状态
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
        )

    async def _query_via_cli(self, task_id: str) -> TaskResult | None:
        """通过 CLI 查询任务状态"""
        # 尝试使用 agent 命令查询（如果支持）
        # 注意: 这是预留接口，实际命令可能需要调整
        try:
            env = os.environ.copy()
            api_key = self.auth_manager.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            process = await asyncio.create_subprocess_exec(
                self.agent_path,
                "status",
                "--task",
                task_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10,
            )

            if process.returncode == 0:
                output = stdout.decode("utf-8", errors="replace")
                return self._parse_task_status_output(task_id, output)

        except (FileNotFoundError, asyncio.TimeoutError):
            pass

        return None

    async def _query_via_http(self, task_id: str) -> TaskResult | None:
        """通过 HTTP API 查询任务状态

        使用增强的错误处理和重试机制。
        """
        if not HTTPX_AVAILABLE:
            return None

        api_key = self.auth_manager.get_api_key()
        if not api_key:
            return None

        url = f"{self.api_base_url}/v1/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)

                # 成功响应
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_task_api_response(task_id, data)

                # 任务不存在
                elif response.status_code == 404:
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error="任务不存在",
                    )

                # 处理 HTTP 错误
                else:
                    body = response.text
                    response_headers = dict(response.headers)
                    error = handle_http_error(
                        response.status_code,
                        response_headers,
                        body,
                        context=f"查询任务 {task_id}",
                    )

                    # 认证错误
                    if isinstance(error, AuthError):
                        # 降级为 debug 日志，用户提示通过 TaskResult.error 透传
                        logger.debug(f"查询任务认证失败: {error.user_friendly_message}")
                        return TaskResult(
                            task_id=task_id,
                            status=TaskStatus.FAILED,
                            error=error.user_friendly_message,
                        )

                    # 限流错误 - 降级为 debug 日志
                    if isinstance(error, RateLimitError):
                        logger.debug(f"查询限流: {error.user_friendly_message}")

                    # 其他错误
                    logger.debug(f"HTTP API 错误: {error}")

        except asyncio.TimeoutError:
            logger.debug(f"HTTP API 查询超时: task_id={task_id}")
        except httpx.RequestError as e:
            error = NetworkError.from_exception(e, context=f"查询任务 {task_id}")
            logger.debug(f"HTTP 客户端错误: {error}")
        except Exception as e:
            logger.debug(f"HTTP API 请求失败: {e}")

        return None

    def _parse_task_status_output(self, task_id: str, output: str) -> TaskResult:
        """解析 CLI 输出的任务状态"""
        output_lower = output.lower()

        status = TaskStatus.RUNNING
        if "completed" in output_lower or "success" in output_lower:
            status = TaskStatus.COMPLETED
        elif "failed" in output_lower or "error" in output_lower:
            status = TaskStatus.FAILED
        elif "cancelled" in output_lower:
            status = TaskStatus.CANCELLED
        elif "pending" in output_lower:
            status = TaskStatus.PENDING

        return TaskResult(
            task_id=task_id,
            status=status,
            result=output if status == TaskStatus.COMPLETED else None,
            error=output if status == TaskStatus.FAILED else None,
        )

    def _parse_task_api_response(self, task_id: str, data: dict) -> TaskResult:
        """解析 API 响应的任务状态"""
        status_str = data.get("status", "running")
        try:
            status = TaskStatus(status_str)
        except ValueError:
            status = TaskStatus.RUNNING

        return TaskResult(
            task_id=task_id,
            status=status,
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0),
            metadata=data.get("metadata", {}),
        )

    # ========== WebSocket 实时连接 ==========

    async def watch_task_ws(
        self,
        task_id: str,
        timeout: float = 300.0,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """通过 WebSocket 实时监听任务事件

        如果 API 提供 /ws/tasks/{task_id} 端点，则使用 WebSocket 连接

        Args:
            task_id: 任务 ID
            timeout: 连接超时时间
            on_event: 事件回调

        Yields:
            StreamEvent: 流式事件

        Raises:
            ImportError: websockets 未安装
            ConnectionError: WebSocket 连接失败
        """
        # 延迟导入流式事件类型
        from ..streaming import (
            StreamEvent,
            StreamEventType,
            parse_stream_event,
        )

        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets 未安装，无法使用 WebSocket 连接")
            raise ImportError("需要安装 websockets: pip install websockets")

        api_key = self.auth_manager.get_api_key()
        if not api_key:
            raise AuthError(
                "需要 API Key 才能使用 WebSocket",
                AuthErrorCode.INVALID_API_KEY,
            )

        # 构建 WebSocket URL
        ws_base = self.api_base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/ws/tasks/{task_id}"

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        logger.info(f"连接 WebSocket: {ws_url}")

        try:
            async with websockets.connect(
                ws_url,
                extra_headers=headers,
                close_timeout=10,
            ) as websocket:
                self._ws_connections[task_id] = websocket

                try:
                    async with asyncio.timeout(timeout):
                        async for message in self._ws_receive_messages(websocket):
                            event = parse_stream_event(message)
                            if event:
                                if on_event:
                                    on_event(event)
                                yield event

                                # 检查是否结束
                                if event.type in (
                                    StreamEventType.RESULT,
                                    StreamEventType.COMPLETE,
                                    StreamEventType.ERROR,
                                ):
                                    break

                except asyncio.TimeoutError:
                    error_event = StreamEvent(
                        type=StreamEventType.ERROR,
                        data={"error": f"WebSocket 连接超时 ({timeout}s)"},
                    )
                    yield error_event

                finally:
                    self._ws_connections.pop(task_id, None)

        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket 连接失败: {e}")
            raise ConnectionError(f"WebSocket 连接失败: {e}") from e

    async def _ws_receive_messages(self, websocket) -> AsyncIterator[str]:
        """从 WebSocket 接收消息"""
        async for message in websocket:
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
            yield message

    async def close_ws_connection(self, task_id: str) -> None:
        """关闭指定任务的 WebSocket 连接"""
        ws = self._ws_connections.get(task_id)
        if ws:
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"关闭 WebSocket 失败: {e}")
            finally:
                self._ws_connections.pop(task_id, None)

    async def close_all_ws_connections(self) -> None:
        """关闭所有 WebSocket 连接"""
        for task_id in list(self._ws_connections.keys()):
            await self.close_ws_connection(task_id)

    # ========== 流式事件处理 (SSE/NDJSON) ==========

    async def watch_task_sse(
        self,
        task_id: str,
        timeout: float = 300.0,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """通过 Server-Sent Events (SSE) 监听任务事件

        使用增强的错误处理，包括认证错误和限流处理。

        Args:
            task_id: 任务 ID
            timeout: 超时时间
            on_event: 事件回调

        Yields:
            StreamEvent: 流式事件
        """
        # 延迟导入流式事件类型
        from ..streaming import (
            StreamEvent,
            StreamEventType,
            parse_stream_event,
        )

        if not HTTPX_AVAILABLE:
            logger.warning("httpx 未安装，回退到轮询模式")
            async for event in self._fallback_to_polling(task_id, timeout, on_event):
                yield event
            return

        api_key = self.auth_manager.get_api_key()
        if not api_key:
            raise AuthError(
                "需要 API Key",
                AuthErrorCode.INVALID_API_KEY,
            )

        url = f"{self.api_base_url}/v1/tasks/{task_id}/stream"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        }

        logger.info(f"连接 SSE: {url}")

        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    "GET",
                    url,
                    headers=headers,
                ) as response,
            ):
                # 处理非成功响应
                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_text = error_bytes.decode("utf-8", errors="replace")
                    response_headers = dict(response.headers)
                    error = handle_http_error(
                        response.status_code,
                        response_headers,
                        error_text,
                        context=f"SSE 连接 {task_id}",
                    )

                    # 认证错误
                    if isinstance(error, AuthError):
                        # 降级为 debug 日志，用户提示通过 StreamEvent 透传
                        logger.debug(f"SSE 连接认证失败: {error.user_friendly_message}")
                        error_event = StreamEvent(
                            type=StreamEventType.ERROR,
                            data={
                                "error": error.user_friendly_message,
                                "error_type": "auth",
                                "hint": "请检查 CURSOR_API_KEY",
                            },
                        )
                        yield error_event
                        return

                    # 限流错误
                    if isinstance(error, RateLimitError):
                        # 降级为 debug 日志，用户提示通过 StreamEvent 透传
                        logger.debug(f"SSE 连接限流: {error.user_friendly_message}")
                        error_event = StreamEvent(
                            type=StreamEventType.ERROR,
                            data={
                                "error": error.user_friendly_message,
                                "error_type": "rate_limit",
                                "retry_after": error.retry_after,
                            },
                        )
                        yield error_event
                        return

                    # 其他错误
                    error_event = StreamEvent(
                        type=StreamEventType.ERROR,
                        data={"error": str(error), "status_code": response.status_code},
                    )
                    yield error_event
                    return

                    # 读取 SSE 事件流
                    async for event in self._parse_sse_stream(response, parse_stream_event):
                        if on_event:
                            on_event(event)
                        yield event

                        if event.type in (
                            StreamEventType.RESULT,
                            StreamEventType.COMPLETE,
                            StreamEventType.ERROR,
                        ):
                            break

        except asyncio.TimeoutError:
            from ..streaming import StreamEvent, StreamEventType

            logger.warning(f"SSE 连接超时: {timeout}s")
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": f"SSE 连接超时 ({timeout}s)", "error_type": "timeout"},
            )
            yield error_event

        except httpx.RequestError as e:
            from ..streaming import StreamEvent, StreamEventType

            error = NetworkError.from_exception(e, context=f"SSE 连接 {task_id}")
            # 降级为 debug 日志，用户提示通过 StreamEvent 透传
            logger.debug(f"SSE 连接错误: {error.user_friendly_message}")
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={
                    "error": error.user_friendly_message,
                    "error_type": error.error_type,
                },
            )
            yield error_event

    async def _parse_sse_stream(
        self,
        response: httpx.Response,
        parse_func: Callable[[str], StreamEvent | None],
    ) -> AsyncIterator[StreamEvent]:
        """解析 SSE 事件流"""
        buffer = ""
        event_data = ""

        async for chunk in response.aiter_bytes():
            buffer += chunk.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    # 空行表示事件结束
                    if event_data:
                        event = parse_func(event_data)
                        if event:
                            yield event
                        event_data = ""
                    continue

                if line.startswith("event:"):
                    line[6:].strip()
                elif line.startswith("data:"):
                    event_data = line[5:].strip()
                elif line.startswith(":"):
                    # 注释行，忽略
                    continue

    async def _fallback_to_polling(
        self,
        task_id: str,
        timeout: float,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """回退到轮询模式"""
        from ..streaming import StreamEvent, StreamEventType

        logger.info("使用轮询模式监听任务")

        def status_to_event(status: TaskStatus) -> StreamEvent:
            return StreamEvent(
                type=StreamEventType.PROGRESS,
                data={"status": status.value},
            )

        last_status: TaskStatus | None = None

        async def on_status_change(status: TaskStatus) -> None:
            nonlocal last_status
            if status != last_status:
                event = status_to_event(status)
                if on_event:
                    on_event(event)
                last_status = status

        def handle_status_change(status: TaskStatus) -> None:
            asyncio.create_task(on_status_change(status))

        result = await self.poll_task_status(
            task_id=task_id,
            timeout=timeout,
            interval=2.0,
            on_status_change=handle_status_change if on_event else None,
        )

        # 生成最终事件
        if result.is_success:
            final_event = StreamEvent(
                type=StreamEventType.RESULT,
                data=result.to_dict(),
                duration_ms=result.duration_ms,
            )
        else:
            final_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": result.error or "任务失败"},
            )

        yield final_event

    # ========== 流式执行任务 ==========

    async def execute_streaming(
        self,
        prompt: str,
        model: str,
        working_directory: str = ".",
        timeout: float = 300.0,
        on_event: Callable[[StreamEvent], None] | None = None,
        force: bool = False,
        extra_args: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式执行 Agent 任务

        使用 agent CLI 的 stream-json 输出格式执行任务，
        实时返回流式事件。支持增强的错误处理。

        Args:
            prompt: 任务提示
            model: 模型名称
            working_directory: 工作目录
            timeout: 超时时间（秒）
            on_event: 事件回调函数
            force: 是否允许修改文件
            extra_args: 额外的命令行参数

        Yields:
            StreamEvent: 流式事件

        Raises:
            AuthError: 认证失败（未配置 API Key）

        Example:
            client = CloudTaskClient()
            async for event in client.execute_streaming(
                prompt="实现用户登录功能",
                model="gpt-5.2-high",
            ):
                if event.type == StreamEventType.ASSISTANT:
                    print(event.content)
                elif event.type == StreamEventType.TOOL_COMPLETED:
                    print(f"工具完成: {event.tool_call.tool_type}")
        """
        from ..streaming import (
            StreamEvent,
            StreamEventType,
            parse_stream_event,
        )

        # 构建命令
        cmd = [
            self.agent_path,
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--stream-partial-output",
        ]

        if force:
            cmd.append("--force")

        if extra_args:
            cmd.extend(extra_args)

        # 设置环境变量
        env = os.environ.copy()
        api_key = self.auth_manager.get_api_key()
        if api_key:
            env["CURSOR_API_KEY"] = api_key

        logger.info(f"启动流式执行: model={model}, cwd={working_directory}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_stream = process.stdout
            stderr_stream = process.stderr
            if stdout_stream is None or stderr_stream is None:
                raise RuntimeError("流式输出不可用: stdout/stderr 为空")
        except FileNotFoundError:
            init_error = NetworkError(
                message=f"找不到 agent CLI: {self.agent_path}",
                error_type="not_found",
                details={"path": self.agent_path},
            )
            logger.error(init_error.message)
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={
                    "error": init_error.message,
                    "error_type": "not_found",
                    "hint": "请安装 Cursor CLI: curl https://cursor.com/install -fsS | bash",
                },
            )
            yield error_event
            return
        except OSError as e:
            os_error = NetworkError.from_exception(e, context="启动 agent 进程")
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": os_error.message, "error_type": os_error.error_type},
            )
            yield error_event
            return

        start_time = time.time()

        try:
            # 逐行读取流式输出
            async for line in self._read_stream_lines(stdout_stream, timeout):
                event = parse_stream_event(line)
                if event:
                    # 检测错误事件中的认证/限流问题
                    if event.type == StreamEventType.ERROR:
                        error_data = event.data or {}
                        error_msg = str(error_data.get("error", "")).lower()

                        # 检测认证错误
                        if "unauthorized" in error_msg or "401" in error_msg:
                            auth_error = AuthError(
                                message="认证失败",
                                code=AuthErrorCode.INVALID_API_KEY,
                            )
                            # 降级为 debug 日志，用户提示通过 event.data 透传
                            logger.debug(f"SSE 流认证失败: {auth_error.user_friendly_message}")
                            event.data["hint"] = "请检查 CURSOR_API_KEY 环境变量"
                            event.data["error_type"] = "auth"

                        # 检测限流错误
                        elif "rate limit" in error_msg or "429" in error_msg:
                            # 降级为 debug 日志，用户提示通过 event.data 透传
                            logger.debug("SSE 流检测到限流错误")
                            event.data["error_type"] = "rate_limit"
                            event.data["hint"] = "请求过于频繁，请稍后重试"

                    if on_event:
                        try:
                            on_event(event)
                        except Exception as e:
                            logger.warning(f"事件回调执行失败: {e}")
                    yield event

            # 等待进程结束
            await process.wait()

            # 读取 stderr（如果有）
            stderr_data = await stderr_stream.read()
            if stderr_data and process.returncode != 0:
                error_text = stderr_data.decode("utf-8", errors="replace")
                error_lower = error_text.lower()

                # 分析错误类型
                error_type = "unknown"
                hint = None

                if "unauthorized" in error_lower or "401" in error_lower:
                    error_type = "auth"
                    hint = "请检查 CURSOR_API_KEY 环境变量是否正确设置"
                    logger.error(f"认证失败: {error_text[:200]}")
                elif "forbidden" in error_lower or "403" in error_lower:
                    error_type = "auth"
                    hint = "权限不足，请确认账户权限"
                    logger.error(f"权限错误: {error_text[:200]}")
                elif "rate limit" in error_lower or "429" in error_lower:
                    error_type = "rate_limit"
                    hint = "请求过于频繁，请稍后重试"
                    logger.warning(f"限流错误: {error_text[:200]}")
                elif "timeout" in error_lower:
                    error_type = "timeout"
                    logger.warning(f"超时错误: {error_text[:200]}")
                elif "connection" in error_lower or "network" in error_lower:
                    error_type = "network"
                    hint = "请检查网络连接"
                    logger.error(f"网络错误: {error_text[:200]}")
                else:
                    logger.error(f"执行失败 (exit_code={process.returncode}): {error_text[:200]}")

                error_event = StreamEvent(
                    type=StreamEventType.ERROR,
                    data={
                        "error": error_text,
                        "exit_code": process.returncode,
                        "error_type": error_type,
                        **({"hint": hint} if hint else {}),
                    },
                )
                yield error_event

            # 发送完成事件
            elapsed_ms = int((time.time() - start_time) * 1000)
            complete_event = StreamEvent(
                type=StreamEventType.COMPLETE,
                data={"exit_code": process.returncode},
                duration_ms=elapsed_ms,
            )
            if on_event:
                on_event(complete_event)
            yield complete_event

        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass

            elapsed_ms = int((time.time() - start_time) * 1000)
            timeout_error = TaskError(
                message=f"执行超时 ({timeout}s)",
                task_status=TaskStatus.TIMEOUT,
            )
            logger.warning(f"流式执行超时: {timeout}s")

            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={
                    "error": timeout_error.message,
                    "error_type": "timeout",
                    "elapsed_ms": elapsed_ms,
                },
            )
            yield error_event

        except Exception as e:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass

            # 根据异常类型创建对应的错误
            stream_error: CloudAgentError
            if isinstance(e, CloudAgentError):
                stream_error = e
            else:
                stream_error = NetworkError.from_exception(e, context="流式执行")

            logger.error(f"流式执行失败: {stream_error}")

            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={
                    "error": str(stream_error),
                    "error_type": getattr(stream_error, "error_type", "unknown"),
                },
            )
            yield error_event

    async def _read_stream_lines(
        self,
        stream: asyncio.StreamReader,
        timeout: float,
    ) -> AsyncIterator[str]:
        """逐行读取流，带超时控制"""
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                line = await asyncio.wait_for(
                    stream.readline(),
                    timeout=min(remaining, 5.0),  # 每 5 秒检查一次超时
                )
                if not line:
                    break
                yield line.decode("utf-8", errors="replace").strip()
            except asyncio.TimeoutError:
                # 单次读取超时，检查总超时
                if asyncio.get_event_loop().time() >= deadline:
                    raise
                continue

    # ========== 便捷方法 ==========

    async def execute_and_collect(
        self,
        prompt: str,
        model: str,
        **kwargs,
    ) -> TaskResult:
        """执行任务并收集所有事件

        Args:
            prompt: 任务提示
            model: 模型名称
            **kwargs: 传递给 execute_streaming 的其他参数

        Returns:
            TaskResult: 包含所有事件的任务结果
        """
        from ..streaming import StreamEventType

        events: list[StreamEvent] = []
        result_content = ""
        error_content = ""
        duration_ms = 0

        async for event in self.execute_streaming(prompt, model, **kwargs):
            events.append(event)

            if event.type == StreamEventType.ASSISTANT:
                result_content += event.content
            elif event.type == StreamEventType.ERROR:
                error_content = event.data.get("error", "")
            elif event.type in (StreamEventType.RESULT, StreamEventType.COMPLETE):
                duration_ms = event.duration_ms

        # 确定最终状态
        has_error = any(e.type == StreamEventType.ERROR for e in events)
        status = TaskStatus.FAILED if has_error else TaskStatus.COMPLETED

        return TaskResult(
            task_id=f"exec-{int(time.time() * 1000)}",
            status=status,
            result=result_content if not has_error else None,
            error=error_content if has_error else None,
            duration_ms=duration_ms,
            events=[e.data for e in events],
        )

    # ========== 上下文管理器 ==========

    async def __aenter__(self) -> CloudTaskClient:
        """异步上下文管理器入口"""
        await self.auth_manager.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出"""
        await self.close_all_ws_connections()
