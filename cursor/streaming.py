"""流式输出处理

支持 --output-format stream-json 和 --stream-partial-output
用于实时跟踪 Agent 执行进度

stream-json 输出格式:
- type: "system", subtype: "init" - 初始化，包含模型信息
- type: "assistant" - 助手消息，包含增量文本
- type: "tool_call", subtype: "started/completed" - 工具调用
- type: "result" - 最终结果，包含耗时
"""
import asyncio
import json
from typing import AsyncIterator, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class StreamEventType(str, Enum):
    """流式事件类型"""
    # 系统事件
    SYSTEM_INIT = "system_init"       # 系统初始化
    
    # 助手消息
    ASSISTANT = "assistant"           # 助手消息（增量文本）
    
    # 工具调用
    TOOL_STARTED = "tool_started"     # 工具调用开始
    TOOL_COMPLETED = "tool_completed" # 工具调用完成
    
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
    tool_type: str = ""       # write, read, shell 等
    path: str = ""            # 文件路径
    args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    success: bool = False


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
    duration_ms: int = 0                      # 耗时毫秒 (result)


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
        """解析流式输出行
        
        格式参考:
        - {"type": "system", "subtype": "init", "model": "gpt-5.2-high"}
        - {"type": "assistant", "message": {"content": [{"text": "..."}]}}
        - {"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {...}}}
        - {"type": "tool_call", "subtype": "completed", "tool_call": {...}}
        - {"type": "result", "duration_ms": 1234}
        """
        if not line:
            return None
        
        try:
            data = json.loads(line)
            event_type = data.get("type", "")
            subtype = data.get("subtype", "")
            
            # 系统初始化
            if event_type == "system" and subtype == "init":
                return StreamEvent(
                    type=StreamEventType.SYSTEM_INIT,
                    subtype=subtype,
                    data=data,
                    model=data.get("model", ""),
                )
            
            # 助手消息
            if event_type == "assistant":
                content = ""
                message = data.get("message", {})
                contents = message.get("content", [])
                if contents and isinstance(contents, list):
                    content = contents[0].get("text", "")
                
                return StreamEvent(
                    type=StreamEventType.ASSISTANT,
                    data=data,
                    content=content,
                )
            
            # 工具调用
            if event_type == "tool_call":
                tool_call = self._parse_tool_call(data.get("tool_call", {}))
                
                if subtype == "started":
                    return StreamEvent(
                        type=StreamEventType.TOOL_STARTED,
                        subtype=subtype,
                        data=data,
                        tool_call=tool_call,
                    )
                elif subtype == "completed":
                    return StreamEvent(
                        type=StreamEventType.TOOL_COMPLETED,
                        subtype=subtype,
                        data=data,
                        tool_call=tool_call,
                    )
            
            # 结果
            if event_type == "result":
                return StreamEvent(
                    type=StreamEventType.RESULT,
                    data=data,
                    duration_ms=data.get("duration_ms", 0),
                )
            
            # 未知类型，返回通用消息
            return StreamEvent(
                type=StreamEventType.MESSAGE,
                data=data,
            )
            
        except json.JSONDecodeError:
            # 非 JSON 行，作为消息处理
            return StreamEvent(
                type=StreamEventType.MESSAGE,
                data={"content": line},
                content=line,
            )
    
    def _parse_tool_call(self, tool_call_data: dict) -> ToolCallInfo:
        """解析工具调用信息"""
        info = ToolCallInfo()
        
        # 写入工具
        if "writeToolCall" in tool_call_data:
            write_call = tool_call_data["writeToolCall"]
            info.tool_type = "write"
            info.args = write_call.get("args", {})
            info.path = info.args.get("path", "")
            
            result = write_call.get("result", {})
            if "success" in result:
                info.success = True
                info.result = result["success"]
        
        # 读取工具
        elif "readToolCall" in tool_call_data:
            read_call = tool_call_data["readToolCall"]
            info.tool_type = "read"
            info.args = read_call.get("args", {})
            info.path = info.args.get("path", "")
            
            result = read_call.get("result", {})
            if "success" in result:
                info.success = True
                info.result = result["success"]
        
        # Shell 工具
        elif "shellToolCall" in tool_call_data:
            shell_call = tool_call_data["shellToolCall"]
            info.tool_type = "shell"
            info.args = shell_call.get("args", {})
            
            result = shell_call.get("result", {})
            if "success" in result:
                info.success = True
                info.result = result["success"]
        
        return info


class ProgressTracker:
    """进度跟踪器
    
    用于跟踪和显示 Agent 执行进度
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.events: list[StreamEvent] = []
        
        # 统计信息
        self.model: str = ""
        self.accumulated_text: str = ""
        self.tool_count: int = 0
        self.files_written: list[str] = []
        self.files_read: list[str] = []
        self.errors: list[str] = []
        self.duration_ms: int = 0
        self.is_complete: bool = False
    
    def on_event(self, event: StreamEvent) -> None:
        """处理事件"""
        self.events.append(event)
        
        if event.type == StreamEventType.SYSTEM_INIT:
            self.model = event.model
            if self.verbose:
                logger.info(f"🤖 使用模型: {self.model}")
        
        elif event.type == StreamEventType.ASSISTANT:
            self.accumulated_text += event.content
            if self.verbose:
                print(f"\r📝 生成中: {len(self.accumulated_text)} 字符", end="", flush=True)
        
        elif event.type == StreamEventType.TOOL_STARTED:
            self.tool_count += 1
            if event.tool_call:
                tool = event.tool_call
                if tool.tool_type == "write":
                    if self.verbose:
                        print(f"\n🔧 工具 #{self.tool_count}: 创建 {tool.path}")
                elif tool.tool_type == "read":
                    if self.verbose:
                        print(f"\n📖 工具 #{self.tool_count}: 读取 {tool.path}")
                elif tool.tool_type == "shell":
                    if self.verbose:
                        print(f"\n💻 工具 #{self.tool_count}: 执行命令")
        
        elif event.type == StreamEventType.TOOL_COMPLETED:
            if event.tool_call:
                tool = event.tool_call
                if tool.success:
                    if tool.tool_type == "write":
                        self.files_written.append(tool.path)
                        lines = tool.result.get("linesCreated", 0)
                        size = tool.result.get("fileSize", 0)
                        if self.verbose:
                            print(f"   ✅ 已创建 {lines} 行 ({size} 字节)")
                    elif tool.tool_type == "read":
                        self.files_read.append(tool.path)
                        lines = tool.result.get("totalLines", 0)
                        if self.verbose:
                            print(f"   ✅ 已读取 {lines} 行")
        
        elif event.type == StreamEventType.RESULT:
            self.duration_ms = event.duration_ms
            self.is_complete = True
            if self.verbose:
                print(f"\n\n🎯 完成, 耗时 {self.duration_ms}ms")
                print(f"📊 统计: {self.tool_count} 个工具, 生成 {len(self.accumulated_text)} 字符")
        
        elif event.type == StreamEventType.ERROR:
            error = event.data.get("error", "未知错误")
            self.errors.append(error)
            logger.error(f"❌ 错误: {error}")
    
    def get_summary(self) -> dict:
        """获取执行摘要"""
        return {
            "model": self.model,
            "total_events": len(self.events),
            "tool_count": self.tool_count,
            "files_written": self.files_written,
            "files_read": self.files_read,
            "text_length": len(self.accumulated_text),
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "is_complete": self.is_complete,
        }
