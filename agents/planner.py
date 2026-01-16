"""规划者 Agent

负责探索代码库、分解任务、派生子规划者
"""
import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

from core.base import BaseAgent, AgentConfig, AgentRole, AgentStatus
from tasks.task import Task, TaskType, TaskPriority
from cursor.client import CursorAgentClient, CursorAgentConfig


class PlannerConfig(BaseModel):
    """规划者配置"""
    name: str = "planner"
    working_directory: str = "."
    max_sub_planners: int = 3          # 最大子规划者数量
    max_tasks_per_plan: int = 10       # 单次规划最大任务数
    exploration_depth: int = 3          # 探索深度
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)


class PlannerAgent(BaseAgent):
    """规划者 Agent
    
    职责:
    1. 探索代码库结构
    2. 分析用户目标
    3. 分解为可执行的子任务
    4. 必要时派生子规划者处理特定区域
    """
    
    # 规划者系统提示
    SYSTEM_PROMPT = """你是一个代码项目规划者。你的职责是:

1. 分析用户的目标需求
2. 探索代码库结构，了解项目组织
3. 将大目标分解为具体的、可执行的子任务
4. 为每个任务提供清晰的指令

输出格式要求:
- 每个任务必须是独立可执行的
- 任务描述要具体明确
- 包含涉及的文件路径
- 指定任务类型和优先级

请以 JSON 格式输出任务列表:
```json
{
  "analysis": "对代码库的分析总结",
  "tasks": [
    {
      "title": "任务标题",
      "description": "详细描述",
      "instruction": "具体执行指令",
      "type": "implement|refactor|fix|test|document",
      "priority": "low|normal|high|critical",
      "target_files": ["file1.py", "file2.py"],
      "depends_on": []
    }
  ],
  "sub_planners_needed": [
    {
      "area": "需要深入规划的区域",
      "reason": "为什么需要子规划者"
    }
  ]
}
```"""
    
    def __init__(self, config: PlannerConfig):
        agent_config = AgentConfig(
            role=AgentRole.PLANNER,
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.planner_config = config
        self.cursor_client = CursorAgentClient(config.cursor_config)
        self.sub_planners: list["PlannerAgent"] = []
    
    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行规划任务
        
        Args:
            instruction: 用户目标或规划指令
            context: 上下文（包含已有分析结果等）
            
        Returns:
            规划结果，包含任务列表和子规划者建议
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始规划: {instruction[:100]}...")
        
        try:
            # 构建规划 prompt
            prompt = self._build_planning_prompt(instruction, context)
            
            # 调用 Cursor Agent 执行规划
            result = await self.cursor_client.execute(
                instruction=prompt,
                working_directory=self.planner_config.working_directory,
                context=context,
            )
            
            if not result.success:
                logger.error(f"[{self.id}] 规划失败: {result.error}")
                self.update_status(AgentStatus.FAILED)
                return {"success": False, "error": result.error, "tasks": []}
            
            # 解析规划结果
            plan_result = self._parse_planning_result(result.output)
            
            self.update_status(AgentStatus.COMPLETED)
            logger.info(f"[{self.id}] 规划完成, 生成 {len(plan_result.get('tasks', []))} 个任务")
            
            return {
                "success": True,
                "analysis": plan_result.get("analysis", ""),
                "tasks": plan_result.get("tasks", []),
                "sub_planners_needed": plan_result.get("sub_planners_needed", []),
            }
            
        except Exception as e:
            logger.exception(f"[{self.id}] 规划异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {"success": False, "error": str(e), "tasks": []}
    
    def _build_planning_prompt(self, instruction: str, context: Optional[dict] = None) -> str:
        """构建规划 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 用户目标\n{instruction}",
        ]
        
        if context:
            if "codebase_info" in context:
                parts.append(f"\n## 代码库信息\n{context['codebase_info']}")
            if "previous_analysis" in context:
                parts.append(f"\n## 前序分析\n{context['previous_analysis']}")
            if "constraints" in context:
                parts.append(f"\n## 约束条件\n{context['constraints']}")
        
        parts.append("\n请分析并输出任务规划:")
        
        return "\n".join(parts)
    
    def _parse_planning_result(self, output: str) -> dict[str, Any]:
        """解析规划结果"""
        import json
        import re
        
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        
        # 解析失败，返回原始输出作为分析
        logger.warning(f"[{self.id}] 无法解析规划结果为 JSON")
        return {
            "analysis": output,
            "tasks": [],
            "sub_planners_needed": [],
        }
    
    async def spawn_sub_planner(self, area: str, context: Optional[dict] = None) -> "PlannerAgent":
        """派生子规划者
        
        Args:
            area: 需要深入规划的区域
            context: 上下文信息
            
        Returns:
            子规划者实例
        """
        if len(self.sub_planners) >= self.planner_config.max_sub_planners:
            logger.warning(f"[{self.id}] 已达到最大子规划者数量限制")
            raise ValueError("已达到最大子规划者数量限制")
        
        sub_config = PlannerConfig(
            name=f"{self.config.name}-sub-{len(self.sub_planners)}",
            working_directory=self.planner_config.working_directory,
            max_sub_planners=0,  # 子规划者不能再派生
            max_tasks_per_plan=self.planner_config.max_tasks_per_plan,
            cursor_config=self.planner_config.cursor_config,
        )
        
        sub_planner = PlannerAgent(sub_config)
        sub_planner.set_context("parent_planner", self.id)
        sub_planner.set_context("focus_area", area)
        if context:
            for k, v in context.items():
                sub_planner.set_context(k, v)
        
        self.sub_planners.append(sub_planner)
        logger.info(f"[{self.id}] 派生子规划者: {sub_planner.id} 负责区域: {area}")
        
        return sub_planner
    
    def create_task_from_plan(self, task_data: dict, iteration_id: int) -> Task:
        """从规划结果创建任务"""
        task_type_map = {
            "explore": TaskType.EXPLORE,
            "analyze": TaskType.ANALYZE,
            "implement": TaskType.IMPLEMENT,
            "refactor": TaskType.REFACTOR,
            "fix": TaskType.FIX,
            "test": TaskType.TEST,
            "document": TaskType.DOCUMENT,
        }
        
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }
        
        return Task(
            title=task_data.get("title", "未命名任务"),
            description=task_data.get("description", ""),
            instruction=task_data.get("instruction", task_data.get("description", "")),
            type=task_type_map.get(task_data.get("type", ""), TaskType.CUSTOM),
            priority=priority_map.get(task_data.get("priority", "normal"), TaskPriority.NORMAL),
            target_files=task_data.get("target_files", []),
            context=task_data.get("context", {}),
            created_by=self.id,
            iteration_id=iteration_id,
        )
    
    async def reset(self) -> None:
        """重置规划者状态"""
        self.update_status(AgentStatus.IDLE)
        self.clear_context()
        # 清理子规划者
        for sub in self.sub_planners:
            await sub.reset()
        self.sub_planners.clear()
        logger.debug(f"[{self.id}] 状态已重置")
