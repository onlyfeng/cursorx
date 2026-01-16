"""执行者 Agent

负责领取任务并专注完成，不关心全局
"""
import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

from core.base import BaseAgent, AgentConfig, AgentRole, AgentStatus
from tasks.task import Task, TaskStatus
from cursor.client import CursorAgentClient, CursorAgentConfig


class WorkerConfig(BaseModel):
    """执行者配置"""
    name: str = "worker"
    working_directory: str = "."
    max_concurrent_tasks: int = 1      # 同时处理的任务数（通常为1）
    task_timeout: int = 300            # 任务超时时间
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)


class WorkerAgent(BaseAgent):
    """执行者 Agent
    
    职责:
    1. 从任务队列领取任务
    2. 专注执行分配的任务
    3. 完成后提交变更
    4. 不与其他执行者协调
    """
    
    # 执行者系统提示
    SYSTEM_PROMPT = """你是一个代码执行者。你的职责是:

1. 专注完成分配给你的具体任务
2. 按照指令进行代码修改
3. 确保修改的正确性和完整性
4. 完成后报告执行结果

执行要求:
- 只关注当前任务，不考虑其他任务
- 严格按照指令执行
- 如遇到问题，记录并报告
- 完成后简要总结所做的修改

请执行以下任务:"""
    
    def __init__(self, config: WorkerConfig):
        agent_config = AgentConfig(
            role=AgentRole.WORKER,
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.worker_config = config
        self.cursor_client = CursorAgentClient(config.cursor_config)
        self.current_task: Optional[Task] = None
        self.completed_tasks: list[str] = []
    
    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行任务指令
        
        Args:
            instruction: 任务指令
            context: 上下文信息
            
        Returns:
            执行结果
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始执行: {instruction[:100]}...")
        
        try:
            # 构建执行 prompt
            prompt = self._build_execution_prompt(instruction, context)
            
            # 调用 Cursor Agent 执行
            result = await self.cursor_client.execute(
                instruction=prompt,
                working_directory=self.worker_config.working_directory,
                context=context,
                timeout=self.worker_config.task_timeout,
            )
            
            if result.success:
                self.update_status(AgentStatus.COMPLETED)
                logger.info(f"[{self.id}] 执行成功")
                return {
                    "success": True,
                    "output": result.output,
                    "duration": result.duration,
                }
            else:
                self.update_status(AgentStatus.FAILED)
                logger.error(f"[{self.id}] 执行失败: {result.error}")
                return {
                    "success": False,
                    "error": result.error,
                    "output": result.output,
                }
                
        except Exception as e:
            logger.exception(f"[{self.id}] 执行异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {"success": False, "error": str(e)}
    
    async def execute_task(self, task: Task) -> Task:
        """执行任务对象
        
        Args:
            task: 任务对象
            
        Returns:
            更新后的任务对象
        """
        self.current_task = task
        task.start()
        task.assigned_to = self.id
        
        logger.info(f"[{self.id}] 执行任务: {task.id} - {task.title}")
        
        # 构建任务上下文
        context = {
            "task_id": task.id,
            "task_type": task.type.value,
            "target_files": task.target_files,
            **task.context,
        }
        
        # 执行任务
        result = await self.execute(task.instruction, context)
        
        if result.get("success"):
            task.complete({
                "output": result.get("output", ""),
                "duration": result.get("duration", 0),
                "worker_id": self.id,
            })
            self.completed_tasks.append(task.id)
            logger.info(f"[{self.id}] 任务完成: {task.id}")
        else:
            task.fail(result.get("error", "未知错误"))
            logger.error(f"[{self.id}] 任务失败: {task.id} - {result.get('error')}")
        
        self.current_task = None
        return task
    
    def _build_execution_prompt(self, instruction: str, context: Optional[dict] = None) -> str:
        """构建执行 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 任务指令\n{instruction}",
        ]
        
        if context:
            if "target_files" in context and context["target_files"]:
                files = "\n".join(f"- {f}" for f in context["target_files"])
                parts.append(f"\n## 涉及文件\n{files}")
            
            # 添加其他上下文
            other_context = {k: v for k, v in context.items() 
                           if k not in ("task_id", "task_type", "target_files")}
            if other_context:
                import json
                parts.append(f"\n## 额外上下文\n```json\n{json.dumps(other_context, ensure_ascii=False, indent=2)}\n```")
        
        parts.append("\n请开始执行任务:")
        
        return "\n".join(parts)
    
    async def reset(self) -> None:
        """重置执行者状态"""
        self.update_status(AgentStatus.IDLE)
        self.current_task = None
        self.clear_context()
        # 不清除 completed_tasks，保留历史记录
        logger.debug(f"[{self.id}] 状态已重置")
    
    def get_statistics(self) -> dict[str, Any]:
        """获取执行统计"""
        return {
            "worker_id": self.id,
            "status": self.status.value,
            "completed_tasks_count": len(self.completed_tasks),
            "current_task": self.current_task.id if self.current_task else None,
        }
