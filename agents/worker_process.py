"""执行者 Agent 进程

作为独立进程运行的执行者
"""
import asyncio
from typing import Any, Optional
from multiprocessing import Queue

from loguru import logger

from process.worker import AgentWorkerProcess
from process.message_queue import ProcessMessage, ProcessMessageType
from cursor.client import CursorAgentClient, CursorAgentConfig, ModelPresets


class WorkerAgentProcess(AgentWorkerProcess):
    """执行者 Agent 进程
    
    独立进程中运行的执行者，负责：
    1. 接收任务
    2. 调用 Cursor Agent 执行任务
    3. 返回执行结果
    """
    
    SYSTEM_PROMPT = """你是一个代码执行者。你的职责是:

1. 专注完成分配给你的具体任务
2. 按照指令进行代码修改
3. 确保修改的正确性和完整性
4. 完成后报告执行结果

你可以使用的工具:
- 文件操作（读取、创建、编辑文件）
- 搜索工具（查找代码、文件）
- Shell 命令（运行测试、构建等）

Shell 命令限制（重要）:
- 命令会在 30 秒后超时
- 不要运行长时间驻留的服务器或进程
- 不要运行交互式命令（需要用户输入的）
- 每条命令独立执行，目录变更不持久
- 在其他目录运行时使用: cd <dir> && <command>
- 适合: 状态检查、快速构建、文件操作、测试

执行要求:
- 只关注当前任务，不考虑其他任务
- 严格按照指令执行
- 如遇到问题，记录并报告
- 完成后简要总结所做的修改

请执行以下任务:"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        inbox: Queue,
        outbox: Queue,
        config: dict,
    ):
        super().__init__(agent_id, agent_type, inbox, outbox, config)
        self.cursor_client: Optional[CursorAgentClient] = None
        self.current_task_id: Optional[str] = None
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[str] = []
    
    def on_start(self) -> None:
        """进程启动初始化"""
        # 创建 agent CLI 客户端 - 使用 opus-4.5-thinking 进行编码
        # 使用 --force 允许直接修改文件
        stream_enabled = self.config.get("stream_events_enabled", False)
        cursor_config = CursorAgentConfig(
            working_directory=self.config.get("working_directory", "."),
            timeout=self.config.get("task_timeout", 300),
            model=self.config.get("model", "opus-4.5-thinking"),
            output_format="stream-json" if stream_enabled else "text",
            non_interactive=True,  # 非交互模式
            force_write=True,      # --force 允许直接修改文件
            stream_partial_output=stream_enabled,
            stream_events_enabled=stream_enabled,
            stream_log_console=self.config.get("stream_log_console", True),
            stream_log_detail_dir=self.config.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=self.config.get("stream_log_raw_dir", "logs/stream_json/raw/"),
            stream_agent_id=self.agent_id,
            stream_agent_role=self.agent_type,
            stream_agent_name=self.config.get("agent_name"),
        )
        self.cursor_client = CursorAgentClient(cursor_config)
        logger.info(f"[{self.agent_id}] 执行者初始化完成 (模型: {cursor_config.model}, force: True)")
    
    def on_stop(self) -> None:
        """进程停止清理"""
        logger.info(f"[{self.agent_id}] 执行者停止，完成 {len(self.completed_tasks)} 个任务")
    
    def handle_message(self, message: ProcessMessage) -> None:
        """处理业务消息"""
        if message.type == ProcessMessageType.TASK_ASSIGN:
            asyncio.run(self._handle_task(message))
    
    async def _handle_task(self, message: ProcessMessage) -> None:
        """处理任务"""
        task_data = message.payload.get("task", {})
        task_id = task_data.get("id", "unknown")
        
        self.current_task_id = task_id
        
        logger.info(f"[{self.agent_id}] 开始执行任务: {task_id}")
        
        try:
            # 构建执行 prompt
            prompt = self._build_execution_prompt(task_data)
            
            # 构建上下文
            context = {
                "task_id": task_id,
                "task_type": task_data.get("type", "custom"),
                "target_files": task_data.get("target_files", []),
            }
            
            # 发送进度消息
            self._send_message(ProcessMessageType.TASK_PROGRESS, {
                "task_id": task_id,
                "status": "executing",
                "progress": 0,
            })
            
            # 调用 Cursor Agent 执行
            result = await self.cursor_client.execute(
                instruction=prompt,
                context=context,
            )
            
            if result.success:
                self.completed_tasks.append(task_id)
                self._send_message(ProcessMessageType.TASK_RESULT, {
                    "task_id": task_id,
                    "success": True,
                    "output": result.output,
                    "duration": result.duration,
                    "files_modified": result.files_modified,
                }, correlation_id=message.id)
                logger.info(f"[{self.agent_id}] 任务完成: {task_id}")
            else:
                self.failed_tasks.append(task_id)
                self._send_message(ProcessMessageType.TASK_RESULT, {
                    "task_id": task_id,
                    "success": False,
                    "error": result.error,
                    "output": result.output,
                }, correlation_id=message.id)
                logger.error(f"[{self.agent_id}] 任务失败: {task_id} - {result.error}")
                
        except Exception as e:
            logger.exception(f"[{self.agent_id}] 任务执行异常: {e}")
            self.failed_tasks.append(task_id)
            self._send_message(ProcessMessageType.TASK_RESULT, {
                "task_id": task_id,
                "success": False,
                "error": str(e),
            }, correlation_id=message.id)
        finally:
            self.current_task_id = None
    
    def _build_execution_prompt(self, task_data: dict) -> str:
        """构建执行 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 任务: {task_data.get('title', '未命名')}",
            f"\n### 描述\n{task_data.get('description', '')}",
            f"\n### 具体指令\n{task_data.get('instruction', '')}",
        ]
        
        target_files = task_data.get("target_files", [])
        if target_files:
            parts.append(f"\n### 涉及文件\n" + "\n".join(f"- {f}" for f in target_files))
        
        parts.append("\n请开始执行任务:")
        
        return "\n".join(parts)
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "agent_id": self.agent_id,
            "type": "worker",
            "busy": self.current_task_id is not None,
            "current_task": self.current_task_id,
            "completed_count": len(self.completed_tasks),
            "failed_count": len(self.failed_tasks),
        }
