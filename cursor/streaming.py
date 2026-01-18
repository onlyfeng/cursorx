"""流式输出处理

支持 --output-format stream-json 和 --stream-partial-output
用于实时跟踪 Agent 执行进度

stream-json 输出格式:
- type: "system", subtype: "init" - 初始化，包含模型信息
- type: "assistant" - 助手消息，包含增量文本
- type: "tool_call", subtype: "started/completed" - 工具调用
- type: "diff" - 差异事件，包含文件编辑的差异信息
- type: "result" - 最终结果，包含耗时
"""
import asyncio
import difflib
import json
import shutil
import sys
import time
from collections.abc import AsyncIterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

# ============== 差异格式化工具函数 ==============

def format_diff(
    old_string: str,
    new_string: str,
    file_path: str = "",
    context_lines: int = 3,
) -> str:
    """生成统一差异格式 (unified diff)

    Args:
        old_string: 原内容
        new_string: 新内容
        file_path: 文件路径
        context_lines: 上下文行数

    Returns:
        统一差异格式的字符串
    """
    old_lines = old_string.splitlines(keepends=True)
    new_lines = new_string.splitlines(keepends=True)

    # 确保最后一行有换行符
    if old_lines and not old_lines[-1].endswith('\n'):
        old_lines[-1] += '\n'
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    from_file = f"a/{file_path}" if file_path else "a/file"
    to_file = f"b/{file_path}" if file_path else "b/file"

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=from_file,
        tofile=to_file,
        n=context_lines,
    )

    return "".join(diff)


def format_inline_diff(old_string: str, new_string: str) -> str:
    """生成行内差异格式，使用 +/- 标记

    Args:
        old_string: 原内容
        new_string: 新内容

    Returns:
        带有 +/- 标记的差异字符串
    """
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    result: List[str] = []

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_lines[i1:i2]:
                result.append(f"  {line}")
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                result.append(f"- {line}")
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                result.append(f"+ {line}")
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                result.append(f"- {line}")
            for line in new_lines[j1:j2]:
                result.append(f"+ {line}")

    return "\n".join(result)


def format_colored_diff(old_string: str, new_string: str, use_ansi: bool = True) -> str:
    """生成带颜色的差异格式（终端显示用）

    Args:
        old_string: 原内容
        new_string: 新内容
        use_ansi: 是否使用 ANSI 颜色码

    Returns:
        带颜色标记的差异字符串
    """
    # ANSI 颜色码
    RED = "\033[31m" if use_ansi else ""
    GREEN = "\033[32m" if use_ansi else ""
    RESET = "\033[0m" if use_ansi else ""

    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    result: List[str] = []

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in old_lines[i1:i2]:
                result.append(f"  {line}")
        elif tag == "delete":
            for line in old_lines[i1:i2]:
                result.append(f"{RED}- {line}{RESET}")
        elif tag == "insert":
            for line in new_lines[j1:j2]:
                result.append(f"{GREEN}+ {line}{RESET}")
        elif tag == "replace":
            for line in old_lines[i1:i2]:
                result.append(f"{RED}- {line}{RESET}")
            for line in new_lines[j1:j2]:
                result.append(f"{GREEN}+ {line}{RESET}")

    return "\n".join(result)


def get_diff_stats(old_string: str, new_string: str) -> dict:
    """获取差异统计信息

    Args:
        old_string: 原内容
        new_string: 新内容

    Returns:
        包含统计信息的字典
    """
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    insertions = 0
    deletions = 0
    modifications = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            deletions += (i2 - i1)
        elif tag == "insert":
            insertions += (j2 - j1)
        elif tag == "replace":
            deletions += (i2 - i1)
            insertions += (j2 - j1)
            modifications += 1

    return {
        "old_lines": len(old_lines),
        "new_lines": len(new_lines),
        "insertions": insertions,
        "deletions": deletions,
        "modifications": modifications,
        "similarity": matcher.ratio(),
    }


class StreamEventType(str, Enum):
    """流式事件类型"""
    # 系统事件
    SYSTEM_INIT = "system_init"       # 系统初始化

    # 助手消息
    ASSISTANT = "assistant"           # 助手消息（增量文本）

    # 工具调用
    TOOL_STARTED = "tool_started"     # 工具调用开始
    TOOL_COMPLETED = "tool_completed" # 工具调用完成

    # 差异/编辑事件
    DIFF = "diff"                     # 差异事件（通用）
    DIFF_STARTED = "diff_started"     # 差异操作开始
    DIFF_COMPLETED = "diff_completed" # 差异操作完成
    EDIT = "edit"                     # 编辑事件

    # 结果
    RESULT = "result"                 # 最终结果
    ERROR = "error"                   # 错误

    # 兼容旧类型
    MESSAGE = "message"
    PROGRESS = "progress"
    COMPLETE = "complete"


@dataclass
class ToolCallInfo:
    """工具调用信息"""
    tool_type: str = ""       # write, read, shell, edit, str_replace 等
    path: str = ""            # 文件路径
    args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    success: bool = False

    # 差异相关字段
    old_string: str = ""      # 替换前的内容
    new_string: str = ""      # 替换后的内容
    is_diff: bool = False     # 是否为差异/编辑操作


@dataclass
class DiffInfo:
    """差异信息"""
    path: str = ""            # 文件路径
    old_string: str = ""      # 原内容
    new_string: str = ""      # 新内容
    line_start: int = 0       # 起始行号
    line_end: int = 0         # 结束行号
    operation: str = "replace" # 操作类型: replace, insert, delete

    def get_unified_diff(self) -> str:
        """生成统一差异格式"""
        return format_diff(self.old_string, self.new_string, self.path)

    def get_inline_diff(self) -> str:
        """生成行内差异格式"""
        return format_inline_diff(self.old_string, self.new_string)


@dataclass
class StreamEvent:
    """流式事件"""
    type: StreamEventType
    subtype: str = ""
    data: dict = field(default_factory=dict)
    timestamp: Optional[float] = None

    # 具体信息
    model: str = ""                           # 模型名称 (system_init)
    content: str = ""                         # 文本内容 (assistant)
    tool_call: Optional[ToolCallInfo] = None  # 工具调用 (tool_*)
    diff_info: Optional[DiffInfo] = None      # 差异信息 (diff_*)
    duration_ms: int = 0                      # 耗时毫秒 (result)

    def get_formatted_diff(self, colored: bool = False) -> str:
        """获取格式化的差异输出

        Args:
            colored: 是否使用颜色

        Returns:
            格式化的差异字符串
        """
        if self.diff_info:
            if colored:
                return format_colored_diff(
                    self.diff_info.old_string,
                    self.diff_info.new_string,
                )
            return self.diff_info.get_unified_diff()

        if self.tool_call and self.tool_call.is_diff:
            if colored:
                return format_colored_diff(
                    self.tool_call.old_string,
                    self.tool_call.new_string,
                )
            return format_diff(
                self.tool_call.old_string,
                self.tool_call.new_string,
                self.tool_call.path,
            )

        return ""


def parse_stream_event(line: str) -> Optional[StreamEvent]:
    """解析 stream-json 输出行"""
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return StreamEvent(
            type=StreamEventType.MESSAGE,
            data={"content": line},
            content=line,
        )

    event_type = data.get("type", "")
    subtype = data.get("subtype", "")

    if event_type == "system" and subtype == "init":
        return StreamEvent(
            type=StreamEventType.SYSTEM_INIT,
            subtype=subtype,
            data=data,
            model=data.get("model", ""),
        )

    if event_type == "assistant":
        content = ""
        message = data.get("message", {})
        contents = message.get("content", [])
        if isinstance(contents, list):
            parts: list[str] = []
            for item in contents:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(text)
            content = "".join(parts)
        elif isinstance(contents, str):
            content = contents

        return StreamEvent(
            type=StreamEventType.ASSISTANT,
            data=data,
            content=content,
        )

    if event_type == "tool_call":
        tool_call = parse_tool_call(data.get("tool_call", {}))

        if subtype == "started":
            # 判断是否为差异操作
            if tool_call.is_diff:
                return StreamEvent(
                    type=StreamEventType.DIFF_STARTED,
                    subtype=subtype,
                    data=data,
                    tool_call=tool_call,
                    diff_info=_extract_diff_info(tool_call),
                )
            return StreamEvent(
                type=StreamEventType.TOOL_STARTED,
                subtype=subtype,
                data=data,
                tool_call=tool_call,
            )
        if subtype == "completed":
            # 判断是否为差异操作
            if tool_call.is_diff:
                return StreamEvent(
                    type=StreamEventType.DIFF_COMPLETED,
                    subtype=subtype,
                    data=data,
                    tool_call=tool_call,
                    diff_info=_extract_diff_info(tool_call),
                )
            return StreamEvent(
                type=StreamEventType.TOOL_COMPLETED,
                subtype=subtype,
                data=data,
                tool_call=tool_call,
            )

    # 处理专门的 diff 事件类型
    if event_type == "diff":
        diff_info = parse_diff_event(data)
        return StreamEvent(
            type=StreamEventType.DIFF,
            subtype=subtype,
            data=data,
            diff_info=diff_info,
        )

    if event_type == "result":
        return StreamEvent(
            type=StreamEventType.RESULT,
            data=data,
            duration_ms=data.get("duration_ms", 0),
        )

    return StreamEvent(
        type=StreamEventType.MESSAGE,
        data=data,
    )


def parse_tool_call(tool_call_data: dict) -> ToolCallInfo:
    """解析工具调用信息"""
    info = ToolCallInfo()

    if "writeToolCall" in tool_call_data:
        write_call = tool_call_data["writeToolCall"]
        info.tool_type = "write"
        info.args = write_call.get("args", {})
        info.path = info.args.get("path", "")

        result = write_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    elif "readToolCall" in tool_call_data:
        read_call = tool_call_data["readToolCall"]
        info.tool_type = "read"
        info.args = read_call.get("args", {})
        info.path = info.args.get("path", "")

        result = read_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    elif "shellToolCall" in tool_call_data:
        shell_call = tool_call_data["shellToolCall"]
        info.tool_type = "shell"
        info.args = shell_call.get("args", {})

        result = shell_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    elif "editToolCall" in tool_call_data:
        # 编辑工具调用（通用编辑）
        edit_call = tool_call_data["editToolCall"]
        info.tool_type = "edit"
        info.args = edit_call.get("args", {})
        info.path = info.args.get("path", "")
        info.old_string = info.args.get("old_string", "")
        info.new_string = info.args.get("new_string", "")
        info.is_diff = True

        result = edit_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    elif "strReplaceToolCall" in tool_call_data:
        # 字符串替换工具调用 (StrReplace)
        str_replace_call = tool_call_data["strReplaceToolCall"]
        info.tool_type = "str_replace"
        info.args = str_replace_call.get("args", {})
        info.path = info.args.get("path", "")
        info.old_string = info.args.get("old_string", "")
        info.new_string = info.args.get("new_string", "")
        info.is_diff = True

        result = str_replace_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    elif "StrReplace" in tool_call_data:
        # 另一种可能的格式
        str_replace_call = tool_call_data["StrReplace"]
        info.tool_type = "str_replace"
        info.args = str_replace_call.get("args", {})
        info.path = info.args.get("path", "")
        info.old_string = info.args.get("old_string", "")
        info.new_string = info.args.get("new_string", "")
        info.is_diff = True

        result = str_replace_call.get("result", {})
        if "success" in result:
            info.success = True
            info.result = result["success"]

    return info


def _extract_diff_info(tool_call: ToolCallInfo) -> Optional[DiffInfo]:
    """从工具调用中提取差异信息"""
    if not tool_call.is_diff:
        return None

    return DiffInfo(
        path=tool_call.path,
        old_string=tool_call.old_string,
        new_string=tool_call.new_string,
        operation="replace" if tool_call.old_string else "insert",
    )


def parse_diff_event(data: dict) -> DiffInfo:
    """解析 diff 类型的事件数据

    Args:
        data: 事件数据字典

    Returns:
        DiffInfo 对象
    """
    diff_info = DiffInfo()

    # 直接从 data 中提取差异信息
    diff_info.path = data.get("path", "")
    diff_info.old_string = data.get("old_string", data.get("oldString", ""))
    diff_info.new_string = data.get("new_string", data.get("newString", ""))
    diff_info.line_start = data.get("line_start", data.get("lineStart", 0))
    diff_info.line_end = data.get("line_end", data.get("lineEnd", 0))
    diff_info.operation = data.get("operation", "replace")

    # 尝试从 diff 子对象中提取
    if "diff" in data:
        diff_data = data["diff"]
        diff_info.path = diff_data.get("path", diff_info.path)
        diff_info.old_string = diff_data.get("old_string", diff_data.get("oldString", diff_info.old_string))
        diff_info.new_string = diff_data.get("new_string", diff_data.get("newString", diff_info.new_string))
        diff_info.line_start = diff_data.get("line_start", diff_data.get("lineStart", diff_info.line_start))
        diff_info.line_end = diff_data.get("line_end", diff_data.get("lineEnd", diff_info.line_end))

    # 尝试从 changes 数组中提取（某些格式）
    if "changes" in data:
        changes = data["changes"]
        if isinstance(changes, list) and len(changes) > 0:
            first_change = changes[0]
            if isinstance(first_change, dict):
                diff_info.old_string = first_change.get("removed", diff_info.old_string)
                diff_info.new_string = first_change.get("added", diff_info.new_string)

    return diff_info


class StreamingClient:
    """流式输出客户端

    使用 --output-format stream-json --stream-partial-output
    实时跟踪 Agent 执行进度
    """

    def __init__(self, agent_path: str = "agent"):
        self.agent_path = agent_path

    async def execute_streaming(
        self,
        prompt: str,
        model: str,
        working_directory: str = ".",
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        timeout: int = 300,
    ) -> AsyncIterator[StreamEvent]:
        """流式执行 Agent 任务

        Args:
            prompt: 任务提示
            model: 模型名称
            working_directory: 工作目录
            on_event: 事件回调函数
            timeout: 超时时间

        Yields:
            StreamEvent: 流式事件
        """
        cmd = [
            self.agent_path,
            "-p", prompt,
            "--model", model,
            "--output-format", "stream-json",
            "--stream-partial-output",
        ]

        logger.debug(f"启动流式执行: {model}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=working_directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # 逐行读取流式输出
            async for line in self._read_lines(process.stdout, timeout):
                event = self._parse_stream_line(line)
                if event:
                    if on_event:
                        on_event(event)
                    yield event

            # 等待进程结束
            await process.wait()

            # 发送完成事件
            complete_event = StreamEvent(
                type=StreamEventType.COMPLETE,
                data={"exit_code": process.returncode},
            )
            if on_event:
                on_event(complete_event)
            yield complete_event

        except asyncio.TimeoutError:
            process.kill()
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": f"执行超时 ({timeout}s)"},
            )
            yield error_event
        except Exception as e:
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
            )
            yield error_event

    async def _read_lines(
        self,
        stream: asyncio.StreamReader,
        timeout: int,
    ) -> AsyncIterator[str]:
        """逐行读取流"""
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                line = await asyncio.wait_for(
                    stream.readline(),
                    timeout=min(remaining, 1.0),
                )
                if not line:
                    break
                yield line.decode("utf-8", errors="replace").strip()
            except asyncio.TimeoutError:
                continue

    def _parse_stream_line(self, line: str) -> Optional[StreamEvent]:
        """解析流式输出行"""
        return parse_stream_event(line)


class StreamRenderer(ABC):
    """流式输出渲染器基类

    定义渲染事件的接口，允许不同的输出方式
    """

    @abstractmethod
    def render_init(self, model: str) -> None:
        """渲染初始化事件"""
        pass

    @abstractmethod
    def render_assistant(self, content: str, accumulated_length: int) -> None:
        """渲染助手消息"""
        pass

    @abstractmethod
    def render_tool_started(self, tool_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具开始事件"""
        pass

    @abstractmethod
    def render_tool_completed(self, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具完成事件"""
        pass

    @abstractmethod
    def render_diff_started(self, diff_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染差异开始事件"""
        pass

    @abstractmethod
    def render_diff_completed(
        self,
        tool: Optional[ToolCallInfo],
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异完成事件"""
        pass

    @abstractmethod
    def render_diff(
        self,
        diff_count: int,
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异事件"""
        pass

    @abstractmethod
    def render_result(self, duration_ms: int, tool_count: int, text_length: int) -> None:
        """渲染结果事件"""
        pass

    @abstractmethod
    def render_error(self, error: str) -> None:
        """渲染错误事件"""
        pass


class TerminalStreamRenderer(StreamRenderer):
    """终端流式输出渲染器

    支持详细模式和精简模式输出
    """

    def __init__(self, verbose: bool = False):
        """初始化渲染器

        Args:
            verbose: 是否使用详细输出模式
        """
        self.verbose = verbose

    def render_init(self, model: str) -> None:
        """渲染初始化事件"""
        if self.verbose:
            logger.info(f"🤖 使用模型: {model}")
        else:
            print(f"[模型] {model}", flush=True)

    def render_assistant(self, content: str, accumulated_length: int) -> None:
        """渲染助手消息"""
        if self.verbose:
            print(f"\r📝 生成中: {accumulated_length} 字符", end="", flush=True)
        # 精简模式不显示增量文本

    def render_tool_started(self, tool_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具开始事件"""
        if not tool:
            return

        if self.verbose:
            if tool.tool_type == "write":
                print(f"\n🔧 工具 #{tool_count}: 创建 {tool.path}")
            elif tool.tool_type == "read":
                print(f"\n📖 工具 #{tool_count}: 读取 {tool.path}")
            elif tool.tool_type == "shell":
                print(f"\n💻 工具 #{tool_count}: 执行命令")
        else:
            # 精简模式
            if tool.tool_type == "write":
                print(f"[创建] {tool.path}", flush=True)
            elif tool.tool_type == "read":
                print(f"[读取] {tool.path}", flush=True)
            elif tool.tool_type == "shell":
                print("[执行] shell 命令", flush=True)

    def render_tool_completed(self, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具完成事件"""
        if not tool or not tool.success:
            return

        if self.verbose:
            if tool.tool_type == "write":
                lines = tool.result.get("linesCreated", 0)
                size = tool.result.get("fileSize", 0)
                print(f"   ✅ 已创建 {lines} 行 ({size} 字节)")
            elif tool.tool_type == "read":
                lines = tool.result.get("totalLines", 0)
                print(f"   ✅ 已读取 {lines} 行")
        # 精简模式不显示完成详情

    def render_diff_started(self, diff_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染差异开始事件"""
        if not tool:
            return

        if self.verbose:
            print(f"\n✏️ 编辑 #{diff_count}: {tool.path}")
        else:
            print(f"[编辑] {tool.path}", flush=True)

    def render_diff_completed(
        self,
        tool: Optional[ToolCallInfo],
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异完成事件"""
        if not tool or not tool.success or not tool.path:
            return

        if self.verbose:
            print(f"   ✅ 已编辑 {tool.path}")
            if show_diff and diff_info:
                stats = get_diff_stats(
                    diff_info.old_string,
                    diff_info.new_string,
                )
                print(f"   📊 +{stats['insertions']} -{stats['deletions']} 行")
        # 精简模式不显示完成详情

    def render_diff(
        self,
        diff_count: int,
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异事件"""
        if not diff_info:
            return

        if self.verbose:
            print(f"\n✏️ 差异 #{diff_count}: {diff_info.path}")
            if show_diff:
                stats = get_diff_stats(diff_info.old_string, diff_info.new_string)
                print(f"   📊 +{stats['insertions']} -{stats['deletions']} 行")
        else:
            if diff_info.path:
                print(f"[差异] {diff_info.path}", flush=True)

    def render_result(self, duration_ms: int, tool_count: int, text_length: int) -> None:
        """渲染结果事件"""
        if self.verbose:
            print(f"\n\n🎯 完成, 耗时 {duration_ms}ms")
            print(f"📊 统计: {tool_count} 个工具, 生成 {text_length} 字符")
        else:
            print(f"[完成] 耗时 {duration_ms}ms", flush=True)

    def render_error(self, error: str) -> None:
        """渲染错误事件"""
        logger.error(f"❌ 错误: {error}")


class ProgressTracker:
    """进度跟踪器

    用于跟踪和显示 Agent 执行进度
    """

    def __init__(
        self,
        verbose: bool = False,
        show_diff: bool = True,
        renderer: Optional[StreamRenderer] = None,
    ):
        """初始化进度跟踪器

        Args:
            verbose: 是否启用详细输出模式
            show_diff: 是否显示差异详情
            renderer: 流式输出渲染器，默认使用 TerminalStreamRenderer
        """
        self.verbose = verbose
        self.show_diff = show_diff
        self.renderer = renderer or TerminalStreamRenderer(verbose=verbose)
        self.events: list[StreamEvent] = []

        # 统计信息
        self.model: str = ""
        self.accumulated_text: str = ""
        self.tool_count: int = 0
        self.diff_count: int = 0
        self.files_written: list[str] = []
        self.files_read: list[str] = []
        self.files_edited: list[str] = []
        self.errors: list[str] = []
        self.duration_ms: int = 0
        self.is_complete: bool = False

    def on_event(self, event: StreamEvent) -> None:
        """处理事件

        更新统计信息并通过 renderer 进行输出
        """
        self.events.append(event)

        if event.type == StreamEventType.SYSTEM_INIT:
            self.model = event.model
            self.renderer.render_init(self.model)

        elif event.type == StreamEventType.ASSISTANT:
            self.accumulated_text += event.content
            self.renderer.render_assistant(event.content, len(self.accumulated_text))

        elif event.type == StreamEventType.TOOL_STARTED:
            self.tool_count += 1
            self.renderer.render_tool_started(self.tool_count, event.tool_call)

        elif event.type == StreamEventType.TOOL_COMPLETED:
            if event.tool_call:
                tool = event.tool_call
                if tool.success:
                    if tool.tool_type == "write":
                        self.files_written.append(tool.path)
                    elif tool.tool_type == "read":
                        self.files_read.append(tool.path)
            self.renderer.render_tool_completed(event.tool_call)

        elif event.type == StreamEventType.DIFF_STARTED:
            self.diff_count += 1
            self.renderer.render_diff_started(self.diff_count, event.tool_call)

        elif event.type == StreamEventType.DIFF_COMPLETED:
            if event.tool_call:
                tool = event.tool_call
                if tool.success and tool.path:
                    self.files_edited.append(tool.path)
            self.renderer.render_diff_completed(event.tool_call, event.diff_info, self.show_diff)

        elif event.type == StreamEventType.DIFF:
            self.diff_count += 1
            if event.diff_info and event.diff_info.path:
                self.files_edited.append(event.diff_info.path)
            self.renderer.render_diff(self.diff_count, event.diff_info, self.show_diff)

        elif event.type == StreamEventType.RESULT:
            self.duration_ms = event.duration_ms
            self.is_complete = True
            self.renderer.render_result(
                self.duration_ms,
                self.tool_count,
                len(self.accumulated_text),
            )

        elif event.type == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            self.errors.append(error)
            self.renderer.render_error(error)

    def get_summary(self) -> dict:
        """获取执行摘要"""
        return {
            "model": self.model,
            "total_events": len(self.events),
            "tool_count": self.tool_count,
            "diff_count": self.diff_count,
            "files_written": self.files_written,
            "files_read": self.files_read,
            "files_edited": self.files_edited,
            "text_length": len(self.accumulated_text),
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "is_complete": self.is_complete,
        }


class StreamEventLogger:
    """流式事件日志器

    支持 ASSISTANT 消息聚合功能：
    - aggregate_assistant_messages=True 时，ASSISTANT 事件会累积到缓冲区
    - 收到非 ASSISTANT 事件或调用 close() 时，缓冲区内容作为完整消息写入 detail 日志
    - raw 日志始终保持每行记录的行为
    """

    def __init__(
        self,
        agent_id: Optional[str],
        agent_role: Optional[str],
        agent_name: Optional[str],
        console: bool = True,
        detail_dir: str = "logs/stream_json/detail/",
        raw_dir: str = "logs/stream_json/raw/",
        aggregate_assistant_messages: bool = True,
    ) -> None:
        self.agent_id = agent_id or "unknown"
        self.agent_role = agent_role or "agent"
        self.agent_name = agent_name or ""
        self.console = console
        self.detail_dir = detail_dir
        self.raw_dir = raw_dir
        self.aggregate_assistant_messages = aggregate_assistant_messages

        self._raw_file = None
        self._detail_file = None
        self._prefix = self._build_prefix()
        self._pending_assistant_text: str = ""  # ASSISTANT 消息聚合缓冲区
        self._prepare_files()

    def _build_prefix(self) -> str:
        """构建日志前缀"""
        suffix = f"({self.agent_name})" if self.agent_name else ""
        return f"{self.agent_role}:{self.agent_id}{suffix}"

    def _prepare_files(self) -> None:
        """准备日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{self.agent_role}_{self.agent_id}_{timestamp}"

        if self.raw_dir:
            try:
                raw_path = Path(self.raw_dir)
                raw_path.mkdir(parents=True, exist_ok=True)
                raw_file = raw_path / f"{base_name}.jsonl"
                self._raw_file = raw_file.open("a", encoding="utf-8")
            except Exception as e:
                logger.warning(f"创建 raw 日志失败: {e}")
                self._raw_file = None

        if self.detail_dir:
            try:
                detail_path = Path(self.detail_dir)
                detail_path.mkdir(parents=True, exist_ok=True)
                detail_file = detail_path / f"{base_name}.log"
                self._detail_file = detail_file.open("a", encoding="utf-8")
            except Exception as e:
                logger.warning(f"创建 detail 日志失败: {e}")
                self._detail_file = None

    def handle_raw_line(self, line: str) -> None:
        """写入 raw NDJSON"""
        if not self._raw_file:
            return
        try:
            self._raw_file.write(f"{line}\n")
            self._raw_file.flush()
        except Exception as e:
            logger.warning(f"写入 raw 日志失败: {e}")

    def handle_event(self, event: StreamEvent) -> None:
        """处理并输出流式事件

        当 aggregate_assistant_messages=True 时:
        - ASSISTANT 事件累积到缓冲区，不立即写入 detail 日志
        - 收到非 ASSISTANT 事件时，先刷新缓冲区，再处理当前事件
        - raw 日志始终保持每行记录的行为
        """
        # ASSISTANT 消息聚合处理
        if self.aggregate_assistant_messages:
            if event.type == StreamEventType.ASSISTANT:
                # 累积 ASSISTANT 内容到缓冲区
                self._pending_assistant_text += event.content
                # 控制台仍然实时输出（增量显示）
                if self.console and event.content:
                    print(event.content, end="", flush=True)
                return
            else:
                # 非 ASSISTANT 事件，先刷新缓冲区
                self._flush_pending_assistant()

        message = self._format_event(event)
        if not message:
            return

        if self.console:
            print(message, flush=True)

        if self._detail_file:
            try:
                self._detail_file.write(f"{message}\n")
                self._detail_file.flush()
            except Exception as e:
                logger.warning(f"写入 detail 日志失败: {e}")

    def _flush_pending_assistant(self) -> None:
        """刷新 ASSISTANT 消息缓冲区到 detail 日志"""
        if not self._pending_assistant_text:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[{timestamp}] [{self._prefix}] {self._pending_assistant_text}"

        # 控制台换行（因为之前是 end="" 输出的）
        if self.console:
            print()  # 换行

        if self._detail_file:
            try:
                self._detail_file.write(f"{message}\n")
                self._detail_file.flush()
            except Exception as e:
                logger.warning(f"写入 detail 日志失败: {e}")

        self._pending_assistant_text = ""

    def _format_event(self, event: StreamEvent) -> str:
        """格式化事件输出"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if event.type == StreamEventType.SYSTEM_INIT:
            return f"[{timestamp}] [{self._prefix}] 初始化模型: {event.model}"

        if event.type == StreamEventType.ASSISTANT:
            return f"[{timestamp}] [{self._prefix}] {event.content}"

        if event.type in (StreamEventType.TOOL_STARTED, StreamEventType.TOOL_COMPLETED):
            tool = event.tool_call
            status = "开始" if event.type == StreamEventType.TOOL_STARTED else "完成"
            tool_type = tool.tool_type if tool else "tool"
            path = tool.path if tool and tool.path else ""
            extra = f" {path}" if path else ""
            return f"[{timestamp}] [{self._prefix}] 工具{status}: {tool_type}{extra}"

        # 差异事件处理
        if event.type in (StreamEventType.DIFF_STARTED, StreamEventType.DIFF_COMPLETED):
            tool = event.tool_call
            status = "开始" if event.type == StreamEventType.DIFF_STARTED else "完成"
            tool_type = tool.tool_type if tool else "edit"
            path = tool.path if tool and tool.path else ""
            extra = f" {path}" if path else ""

            if event.type == StreamEventType.DIFF_COMPLETED and event.diff_info:
                stats = get_diff_stats(
                    event.diff_info.old_string,
                    event.diff_info.new_string,
                )
                extra += f" (+{stats['insertions']} -{stats['deletions']})"

            return f"[{timestamp}] [{self._prefix}] 编辑{status}: {tool_type}{extra}"

        if event.type == StreamEventType.DIFF:
            diff_info = event.diff_info
            if diff_info:
                stats = get_diff_stats(diff_info.old_string, diff_info.new_string)
                path = diff_info.path or "file"
                return f"[{timestamp}] [{self._prefix}] 差异: {path} (+{stats['insertions']} -{stats['deletions']})"
            return f"[{timestamp}] [{self._prefix}] 差异事件"

        if event.type == StreamEventType.RESULT:
            return f"[{timestamp}] [{self._prefix}] 完成 ({event.duration_ms}ms)"

        if event.type == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            return f"[{timestamp}] [{self._prefix}] 错误: {error}"

        if event.type == StreamEventType.MESSAGE and event.content:
            return f"[{timestamp}] [{self._prefix}] {event.content}"

        return ""

    def close(self) -> None:
        """关闭文件句柄

        关闭前会刷新 ASSISTANT 消息缓冲区，确保所有内容都被写入。
        """
        # 先刷新待处理的 ASSISTANT 消息
        self._flush_pending_assistant()

        for handle in (self._raw_file, self._detail_file):
            if handle:
                try:
                    handle.close()
                except Exception as e:
                    logger.warning(f"关闭日志文件失败: {e}")


class AdvancedTerminalRenderer(StreamRenderer):
    """高级终端流式渲染器

    继承 StreamRenderer 基类，实现逐词显示效果，支持状态栏、ANSI 颜色和终端宽度自适应。

    Features:
        - 逐词/逐字符显示，模拟打字效果
        - 状态栏显示模型信息、工具调用计数等
        - ANSI 颜色和样式（可配置关闭）
        - 终端宽度自适应和智能换行
        - 可配置的打字延迟
        - 兼容 StreamRenderer 接口，可与 ProgressTracker 配合使用

    Example:
        renderer = AdvancedTerminalRenderer(use_color=True, typing_delay=0.02)
        renderer.render_event(event)
        renderer.finish()

        # 或通过 ProgressTracker 使用
        tracker = ProgressTracker(renderer=renderer)
        tracker.on_event(event)
    """

    # ANSI 颜色码
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        # 前景色
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        # 亮色
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        # 背景色
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m",
    }

    # 控制序列
    CTRL = {
        "clear_line": "\033[2K",      # 清除整行
        "cursor_up": "\033[1A",       # 光标上移一行
        "cursor_down": "\033[1B",     # 光标下移一行
        "cursor_start": "\033[0G",    # 光标移到行首
        "save_cursor": "\033[s",      # 保存光标位置
        "restore_cursor": "\033[u",   # 恢复光标位置
        "hide_cursor": "\033[?25l",   # 隐藏光标
        "show_cursor": "\033[?25h",   # 显示光标
    }

    def __init__(
        self,
        use_color: bool = True,
        typing_delay: float = 0.0,
        word_mode: bool = True,
        show_status_bar: bool = True,
        status_bar_position: str = "bottom",
        min_width: int = 40,
        max_width: Optional[int] = None,
        output: Optional["sys.stdout"] = None,
    ) -> None:
        """初始化终端流式渲染器

        Args:
            use_color: 是否使用 ANSI 颜色，设为 False 可禁用颜色输出
            typing_delay: 打字延迟（秒），0 表示无延迟，0.02-0.05 有打字机效果
            word_mode: True 为逐词显示，False 为逐字符显示
            show_status_bar: 是否显示状态栏
            status_bar_position: 状态栏位置，"top" 或 "bottom"
            min_width: 最小终端宽度
            max_width: 最大终端宽度，None 表示使用实际终端宽度
            output: 输出流，默认为 sys.stdout
        """
        self.use_color = use_color
        self.typing_delay = typing_delay
        self.word_mode = word_mode
        self.show_status_bar = show_status_bar
        self.status_bar_position = status_bar_position
        self.min_width = min_width
        self.max_width = max_width
        self.output = output or sys.stdout

        # 状态追踪
        self.model: str = ""
        self.tool_count: int = 0
        self.diff_count: int = 0
        self.char_count: int = 0
        self.current_line_len: int = 0
        self.start_time: Optional[float] = None
        self.is_active: bool = False

        # 状态栏内容缓存
        self._last_status: str = ""
        self._status_visible: bool = False

    # ============== StreamRenderer 抽象方法实现 ==============

    def render_init(self, model: str) -> None:
        """渲染初始化事件

        Args:
            model: 模型名称
        """
        if not self.is_active:
            self.start()

        self.model = model
        init_msg = self._color(f"🚀 模型: {self.model}\n", "cyan", "bold")
        self._write(init_msg)
        self._update_status_bar()

    def render_assistant(self, content: str, accumulated_length: int) -> None:
        """渲染助手消息

        Args:
            content: 消息内容（增量文本）
            accumulated_length: 累计文本长度（用于状态显示）
        """
        if not self.is_active:
            self.start()

        self.render_text(content)

    def render_tool_started(self, tool_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具开始事件

        Args:
            tool_count: 工具调用计数
            tool: 工具调用信息
        """
        if not self.is_active:
            self.start()

        self.tool_count = tool_count
        if tool:
            tool_icon = self._get_tool_icon(tool.tool_type)
            path_info = f" {tool.path}" if tool.path else ""
            msg = self._color(
                f"\n{tool_icon} {tool.tool_type}{path_info}...",
                "yellow"
            )
            self._write(msg)
        self._update_status_bar()

    def render_tool_completed(self, tool: Optional[ToolCallInfo]) -> None:
        """渲染工具完成事件

        Args:
            tool: 工具调用信息
        """
        if tool and tool.success:
            self._write(self._color(" ✓", "green"))
        else:
            self._write(self._color(" ✗", "red"))
        self._write("\n")
        self.current_line_len = 0

    def render_diff_started(self, diff_count: int, tool: Optional[ToolCallInfo]) -> None:
        """渲染差异开始事件

        Args:
            diff_count: 差异操作计数
            tool: 工具调用信息
        """
        if not self.is_active:
            self.start()

        self.diff_count = diff_count
        if tool:
            path = tool.path or "file"
            msg = self._color(f"\n✏️ 编辑 {path}...", "green")
            self._write(msg)
        self._update_status_bar()

    def render_diff_completed(
        self,
        tool: Optional[ToolCallInfo],
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异完成事件

        Args:
            tool: 工具调用信息
            diff_info: 差异信息
            show_diff: 是否显示差异详情
        """
        if diff_info and show_diff:
            stats = get_diff_stats(
                diff_info.old_string,
                diff_info.new_string
            )
            stats_msg = self._color(
                f" (+{stats['insertions']} -{stats['deletions']})",
                "dim"
            )
            self._write(stats_msg)
        self._write(self._color(" ✓\n", "green"))
        self.current_line_len = 0

    def render_diff(
        self,
        diff_count: int,
        diff_info: Optional[DiffInfo],
        show_diff: bool,
    ) -> None:
        """渲染差异事件

        Args:
            diff_count: 差异操作计数
            diff_info: 差异信息
            show_diff: 是否显示差异详情
        """
        if not self.is_active:
            self.start()

        self.diff_count = diff_count
        if diff_info:
            path = diff_info.path or "file"
            msg = self._color(f"\n✏️ 编辑 {path}...", "green")
            self._write(msg)

            if show_diff:
                stats = get_diff_stats(diff_info.old_string, diff_info.new_string)
                stats_msg = self._color(
                    f" (+{stats['insertions']} -{stats['deletions']})",
                    "dim"
                )
                self._write(stats_msg)
            self._write(self._color(" ✓\n", "green"))
            self.current_line_len = 0
        self._update_status_bar()

    def render_result(self, duration_ms: int, tool_count: int, text_length: int) -> None:
        """渲染结果事件

        Args:
            duration_ms: 执行耗时（毫秒）
            tool_count: 工具调用总数
            text_length: 生成文本长度
        """
        self._write(self._color(
            f"\n\n✨ 完成 ({duration_ms}ms)\n",
            "green", "bold"
        ))
        self.finish()

    def render_error(self, error: str) -> None:
        """渲染错误事件

        Args:
            error: 错误信息
        """
        self._write(self._color(f"\n❌ 错误: {error}\n", "red", "bold"))

    # ============== 原有方法 ==============

    def _get_terminal_width(self) -> int:
        """获取终端宽度，自适应处理"""
        try:
            width = shutil.get_terminal_size().columns
        except Exception:
            width = 80  # 默认宽度

        # 应用宽度限制
        width = max(width, self.min_width)
        if self.max_width:
            width = min(width, self.max_width)

        return width

    def _color(self, text: str, *styles: str) -> str:
        """应用颜色和样式

        Args:
            text: 要着色的文本
            *styles: 样式名称，如 "red", "bold", "underline"

        Returns:
            带有 ANSI 颜色码的文本（如果 use_color=False 则返回原文本）
        """
        if not self.use_color:
            return text

        prefix = ""
        for style in styles:
            if style in self.COLORS:
                prefix += self.COLORS[style]

        if prefix:
            return f"{prefix}{text}{self.COLORS['reset']}"
        return text

    def _ctrl(self, name: str) -> str:
        """获取控制序列

        Args:
            name: 控制序列名称

        Returns:
            控制序列字符串（如果 use_color=False 则返回空字符串）
        """
        if not self.use_color:
            return ""
        return self.CTRL.get(name, "")

    def _write(self, text: str, flush: bool = True) -> None:
        """写入输出流

        Args:
            text: 要写入的文本
            flush: 是否立即刷新
        """
        try:
            self.output.write(text)
            if flush:
                self.output.flush()
        except Exception:
            pass  # 忽略输出错误

    def _write_with_delay(self, text: str, is_word: bool = False) -> None:
        """带延迟写入（打字效果）

        Args:
            text: 要写入的文本
            is_word: 是否为整词（影响延迟策略）
        """
        if self.typing_delay <= 0:
            self._write(text)
            return

        if is_word and self.word_mode:
            # 整词写入，延迟一次
            self._write(text)
            time.sleep(self.typing_delay)
        else:
            # 逐字符写入
            for char in text:
                self._write(char)
                # 标点符号后延迟更长
                if char in "。，！？.!?,;:":
                    time.sleep(self.typing_delay * 2)
                elif char in " \t":
                    time.sleep(self.typing_delay * 0.5)
                else:
                    time.sleep(self.typing_delay)

    def _wrap_text(self, text: str) -> str:
        """处理文本换行，适应终端宽度

        Args:
            text: 要处理的文本

        Returns:
            处理后的文本
        """
        width = self._get_terminal_width()
        result: List[str] = []

        for line in text.split('\n'):
            if len(line) <= width:
                result.append(line)
                self.current_line_len = len(line)
            else:
                # 需要换行
                while len(line) > width:
                    # 尝试在空格处断行
                    break_point = line.rfind(' ', 0, width)
                    if break_point == -1:
                        break_point = width

                    result.append(line[:break_point])
                    line = line[break_point:].lstrip()

                if line:
                    result.append(line)
                    self.current_line_len = len(line)

        return '\n'.join(result)

    def _build_status_bar(self) -> str:
        """构建状态栏内容

        Returns:
            格式化的状态栏字符串
        """
        width = self._get_terminal_width()

        # 构建各部分
        parts: List[str] = []

        # 模型信息
        if self.model:
            model_display = self.model[:20] + "..." if len(self.model) > 23 else self.model
            parts.append(self._color(f"🤖 {model_display}", "cyan"))

        # 工具计数
        if self.tool_count > 0:
            parts.append(self._color(f"🔧 {self.tool_count}", "yellow"))

        # 差异计数
        if self.diff_count > 0:
            parts.append(self._color(f"✏️ {self.diff_count}", "green"))

        # 字符计数
        parts.append(self._color(f"📝 {self.char_count}", "dim"))

        # 耗时
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                time_str = f"{minutes}m{seconds}s"
            parts.append(self._color(f"⏱️ {time_str}", "dim"))

        # 组装状态栏
        separator = self._color(" │ ", "dim")
        content = separator.join(parts)

        # 计算实际显示宽度（去除 ANSI 码）
        visible_len = len(self._strip_ansi(content))

        # 填充到终端宽度
        padding = max(0, width - visible_len - 2)
        bar = f" {content}{' ' * padding}"

        # 添加背景色
        if self.use_color:
            bar = f"\033[48;5;236m{bar}\033[0m"

        return bar

    def _strip_ansi(self, text: str) -> str:
        """移除 ANSI 转义序列

        Args:
            text: 包含 ANSI 码的文本

        Returns:
            纯文本
        """
        import re
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return ansi_pattern.sub('', text)

    def _update_status_bar(self) -> None:
        """更新状态栏显示"""
        if not self.show_status_bar:
            return

        status = self._build_status_bar()

        if self.status_bar_position == "bottom":
            # 保存光标，移到最后一行，清除并写入状态栏，恢复光标
            self._write(
                f"{self._ctrl('save_cursor')}"
                f"\033[999;1H"  # 移到底部
                f"{self._ctrl('clear_line')}"
                f"{status}"
                f"{self._ctrl('restore_cursor')}"
            )
        else:
            # 顶部状态栏：保存位置，移到第一行，写入，恢复
            self._write(
                f"{self._ctrl('save_cursor')}"
                f"\033[1;1H"  # 移到顶部
                f"{self._ctrl('clear_line')}"
                f"{status}"
                f"{self._ctrl('restore_cursor')}"
            )

        self._last_status = status
        self._status_visible = True

    def _clear_status_bar(self) -> None:
        """清除状态栏"""
        if not self._status_visible:
            return

        if self.status_bar_position == "bottom":
            self._write(
                f"{self._ctrl('save_cursor')}"
                f"\033[999;1H"
                f"{self._ctrl('clear_line')}"
                f"{self._ctrl('restore_cursor')}"
            )
        else:
            self._write(
                f"{self._ctrl('save_cursor')}"
                f"\033[1;1H"
                f"{self._ctrl('clear_line')}"
                f"{self._ctrl('restore_cursor')}"
            )

        self._status_visible = False

    def start(self) -> None:
        """开始渲染会话"""
        self.is_active = True
        self.start_time = time.time()
        self.current_line_len = 0

        # 隐藏光标（可选，减少闪烁）
        if self.use_color:
            self._write(self._ctrl('hide_cursor'))

        # 初始状态栏
        if self.show_status_bar:
            self._update_status_bar()

    def finish(self) -> None:
        """结束渲染会话"""
        self.is_active = False

        # 清除状态栏
        self._clear_status_bar()

        # 显示光标
        if self.use_color:
            self._write(self._ctrl('show_cursor'))

        # 确保换行
        self._write("\n")

    def render_text(self, text: str, style: Optional[str] = None) -> None:
        """渲染文本内容（逐词/逐字符显示）

        Args:
            text: 要渲染的文本
            style: 可选的样式名称
        """
        if not text:
            return

        self.char_count += len(text)

        # 应用样式
        if style:
            text = self._color(text, style)

        if self.word_mode and self.typing_delay > 0:
            # 逐词模式
            import re
            # 分割成词和非词部分
            tokens = re.findall(r'\S+|\s+', text)
            for token in tokens:
                if token.strip():
                    # 处理换行
                    self._handle_line_wrap(token)
                    self._write_with_delay(token, is_word=True)
                else:
                    self._write(token)
                    if '\n' in token:
                        self.current_line_len = 0
        else:
            # 直接输出或逐字符
            self._write_with_delay(text)

        # 更新状态栏
        if self.show_status_bar:
            self._update_status_bar()

    def _handle_line_wrap(self, word: str) -> None:
        """处理词的换行

        Args:
            word: 当前要输出的词
        """
        width = self._get_terminal_width()
        word_len = len(self._strip_ansi(word))

        if self.current_line_len + word_len + 1 > width:
            self._write("\n")
            self.current_line_len = 0

        self.current_line_len += word_len + 1

    def render_event(self, event: StreamEvent) -> None:
        """渲染流式事件

        Args:
            event: 流式事件对象
        """
        if not self.is_active:
            self.start()

        if event.type == StreamEventType.SYSTEM_INIT:
            self.model = event.model
            # 显示初始化信息
            init_msg = self._color(f"🚀 模型: {self.model}\n", "cyan", "bold")
            self._write(init_msg)
            self._update_status_bar()

        elif event.type == StreamEventType.ASSISTANT:
            # 逐词/逐字符显示助手消息
            self.render_text(event.content)

        elif event.type == StreamEventType.TOOL_STARTED:
            self.tool_count += 1
            if event.tool_call:
                tool = event.tool_call
                tool_icon = self._get_tool_icon(tool.tool_type)
                path_info = f" {tool.path}" if tool.path else ""
                msg = self._color(
                    f"\n{tool_icon} {tool.tool_type}{path_info}...",
                    "yellow"
                )
                self._write(msg)
            self._update_status_bar()

        elif event.type == StreamEventType.TOOL_COMPLETED:
            if event.tool_call and event.tool_call.success:
                self._write(self._color(" ✓", "green"))
            else:
                self._write(self._color(" ✗", "red"))
            self._write("\n")
            self.current_line_len = 0

        elif event.type in (StreamEventType.DIFF_STARTED, StreamEventType.DIFF):
            self.diff_count += 1
            if event.tool_call:
                path = event.tool_call.path or "file"
                msg = self._color(f"\n✏️ 编辑 {path}...", "green")
                self._write(msg)
            elif event.diff_info:
                path = event.diff_info.path or "file"
                msg = self._color(f"\n✏️ 编辑 {path}...", "green")
                self._write(msg)
            self._update_status_bar()

        elif event.type == StreamEventType.DIFF_COMPLETED:
            if event.diff_info:
                stats = get_diff_stats(
                    event.diff_info.old_string,
                    event.diff_info.new_string
                )
                stats_msg = self._color(
                    f" (+{stats['insertions']} -{stats['deletions']})",
                    "dim"
                )
                self._write(stats_msg)
            self._write(self._color(" ✓\n", "green"))
            self.current_line_len = 0

        elif event.type == StreamEventType.RESULT:
            duration = event.duration_ms
            self._write(self._color(
                f"\n\n✨ 完成 ({duration}ms)\n",
                "green", "bold"
            ))
            self.finish()

        elif event.type == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            self._write(self._color(f"\n❌ 错误: {error}\n", "red", "bold"))

    def _get_tool_icon(self, tool_type: str) -> str:
        """获取工具图标

        Args:
            tool_type: 工具类型

        Returns:
            对应的 emoji 图标
        """
        icons = {
            "read": "📖",
            "write": "📝",
            "shell": "💻",
            "edit": "✏️",
            "str_replace": "🔄",
            "search": "🔍",
            "grep": "🔎",
            "glob": "📂",
        }
        return icons.get(tool_type, "🔧")

    def render_diff_content(
        self,
        old_string: str,
        new_string: str,
        file_path: str = "",
        colored: bool = True,
    ) -> None:
        """渲染差异内容（详细显示）

        这是一个便捷方法，用于显示具体的差异内容。
        与 StreamRenderer.render_diff 抽象方法不同。

        Args:
            old_string: 原内容
            new_string: 新内容
            file_path: 文件路径
            colored: 是否使用颜色
        """
        if colored and self.use_color:
            diff_text = format_colored_diff(old_string, new_string, use_ansi=True)
        else:
            diff_text = format_diff(old_string, new_string, file_path)

        # 显示差异标题
        if file_path:
            self._write(self._color(f"\n📄 {file_path}\n", "cyan", "bold"))

        # 显示差异内容（带缩进）
        for line in diff_text.split('\n'):
            self._write(f"  {line}\n")

        self.current_line_len = 0

    def print_summary(self) -> None:
        """打印执行摘要"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        summary_parts = [
            self._color("\n" + "─" * 40 + "\n", "dim"),
            self._color("📊 执行摘要\n", "bold"),
        ]

        if self.model:
            summary_parts.append(f"   模型: {self._color(self.model, 'cyan')}\n")

        summary_parts.extend([
            f"   工具调用: {self._color(str(self.tool_count), 'yellow')}\n",
            f"   编辑操作: {self._color(str(self.diff_count), 'green')}\n",
            f"   输出字符: {self._color(str(self.char_count), 'blue')}\n",
            f"   耗时: {self._color(f'{elapsed:.2f}s', 'magenta')}\n",
            self._color("─" * 40 + "\n", "dim"),
        ])

        self._write("".join(summary_parts))
