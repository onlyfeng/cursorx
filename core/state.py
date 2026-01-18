"""状态管理"""
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .base import AgentStatus, AgentRole


class IterationStatus(str, Enum):
    """迭代状态"""
    PLANNING = "planning"      # 规划中
    EXECUTING = "executing"    # 执行中
    REVIEWING = "reviewing"    # 评审中
    COMMITTING = "committing"  # 提交中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败


class AgentState(BaseModel):
    """单个 Agent 的状态"""
    agent_id: str
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    completed_tasks: list[str] = Field(default_factory=list)
    error_count: int = 0
    last_activity: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class IterationState(BaseModel):
    """单次迭代的状态"""
    iteration_id: int
    status: IterationStatus = IterationStatus.PLANNING
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # 统计信息
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    # 评审结果
    review_passed: bool = False
    review_feedback: Optional[str] = None
    
    # 提交信息（由 Orchestrator 在提交阶段填充）
    commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    pushed: bool = False
    commit_files: list[str] = Field(default_factory=list)


class SystemState(BaseModel):
    """整个系统的状态"""
    # 基本信息
    goal: str = ""                    # 用户目标
    working_directory: str = "."      # 工作目录
    
    # 迭代信息
    current_iteration: int = 0
    max_iterations: int = 10
    iterations: list[IterationState] = Field(default_factory=list)
    
    # Agent 状态
    agents: dict[str, AgentState] = Field(default_factory=dict)
    
    # 全局统计
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    
    # 系统状态
    is_running: bool = False
    is_completed: bool = False
    final_result: Optional[str] = None
    
    def start_new_iteration(self) -> IterationState:
        """开始新的迭代"""
        self.current_iteration += 1
        iteration = IterationState(iteration_id=self.current_iteration)
        self.iterations.append(iteration)
        return iteration
    
    def get_current_iteration(self) -> Optional[IterationState]:
        """获取当前迭代"""
        if self.iterations:
            return self.iterations[-1]
        return None
    
    def register_agent(self, agent_id: str, role: AgentRole) -> AgentState:
        """注册 Agent"""
        state = AgentState(agent_id=agent_id, role=role)
        self.agents[agent_id] = state
        return state
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """更新 Agent 状态"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_activity = datetime.now()
    
    def reset_for_new_iteration(self) -> None:
        """为新迭代重置状态（保留历史记录）"""
        for agent_state in self.agents.values():
            agent_state.status = AgentStatus.IDLE
            agent_state.current_task_id = None
            agent_state.error_count = 0
