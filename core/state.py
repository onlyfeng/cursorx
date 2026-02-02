"""状态管理

本模块定义了系统状态管理相关的类型，包括：
- IterationStatus: 迭代状态枚举
- AgentState: 单个 Agent 的状态
- IterationState: 单次迭代的状态
- SystemState: 整个系统的状态
- CommitPolicy: 提交策略配置
- CommitContext: 提交上下文数据
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .base import AgentRole, AgentStatus

# =============================================================================
# 提交策略类型定义
# =============================================================================


@dataclass
class CommitPolicy:
    """提交策略配置

    定义 Orchestrator 何时触发自动提交。提交触发遵循以下优先级规则：

    触发优先级（从高到低）:
        1. enable_auto_commit=False → 禁用所有自动提交
        2. commit_per_iteration=True → 每次迭代完成后都提交
        3. commit_on_complete=True + decision==COMPLETE → 仅在目标完成时提交

    评审决策对提交的影响:
        - COMPLETE: 允许提交（如果 commit_on_complete=True 或 commit_per_iteration=True）
        - CONTINUE: 仅当 commit_per_iteration=True 时允许提交
        - ADJUST: 仅当 commit_per_iteration=True 时允许提交
        - ABORT: 仅当 commit_per_iteration=True 时允许提交（记录中间进度）

    Attributes:
        enable_auto_commit: 是否启用自动提交（主开关）
        commit_per_iteration: 每次迭代完成后都提交（优先级高于 commit_on_complete）
        commit_on_complete: 仅在评审决策为 COMPLETE 时提交
        auto_push: 提交后是否自动推送到远程仓库

    Example:
        >>> policy = CommitPolicy(
        ...     enable_auto_commit=True,
        ...     commit_per_iteration=False,
        ...     commit_on_complete=True,
        ...     auto_push=False,
        ... )
        >>> policy.should_commit("complete")  # True
        >>> policy.should_commit("continue")  # False
    """

    enable_auto_commit: bool = True
    commit_per_iteration: bool = False
    commit_on_complete: bool = True
    auto_push: bool = False

    def should_commit(self, decision: str) -> bool:
        """根据评审决策判断是否应该提交

        Args:
            decision: 评审决策（complete/continue/adjust/abort）

        Returns:
            True 表示应该提交，False 表示跳过
        """
        if not self.enable_auto_commit:
            return False

        # commit_per_iteration 优先级最高
        if self.commit_per_iteration:
            return True

        # commit_on_complete 仅在 COMPLETE 时触发
        return bool(self.commit_on_complete and decision.lower() == "complete")


@dataclass
class CommitContext:
    """提交上下文数据

    封装从 TaskQueue 汇总的已完成任务信息，作为 CommitterAgent 的输入。

    必需字段:
        - id: 任务唯一标识符
        - title: 任务标题
        - result: 任务执行结果

    可选字段:
        - description: 任务描述（用于生成更详细的 commit message）

    Attributes:
        iteration_id: 当前迭代 ID
        tasks_completed: 已完成任务列表（至少包含 id/title/result）
        review_decision: 评审决策（complete/continue/adjust/abort）
        auto_push: 是否自动推送

    错误处理:
        - commit 失败: result.success=False, error 记录到返回的 commit_result
        - 无变更: result.success=True, commit_hash=None, message="No changes to commit"
        - push 失败: result.success 取决于 commit 是否成功，push_error 记录错误信息

    输出结构:
        commit_result = {
            "success": bool,           # 提交是否成功
            "commit_hash": str | None, # Git 提交哈希（无变更时为 None）
            "message": str,            # 提交信息或错误描述
            "files_changed": list[str],# 变更的文件列表
            "pushed": bool,            # 是否已推送
            "push_error": str | None,  # 推送错误信息（如有）
        }

    IterationState 字段填充:
        - iteration.commit_hash = commit_result.get('commit_hash', '')
        - iteration.commit_message = commit_result.get('message', '')
        - iteration.pushed = commit_result.get('pushed', False)
        - iteration.commit_files = commit_result.get('files_changed', [])

    Example:
        >>> context = CommitContext(
        ...     iteration_id=1,
        ...     tasks_completed=[
        ...         {"id": "task-001", "title": "实现功能A", "result": {"success": True}},
        ...         {"id": "task-002", "title": "修复BugB", "result": {"files": ["a.py"]}},
        ...     ],
        ...     review_decision="complete",
        ...     auto_push=False,
        ... )
    """

    iteration_id: int
    tasks_completed: list[dict[str, Any]] = dc_field(default_factory=list)
    review_decision: str = "continue"
    auto_push: bool = False

    def validate_tasks(self) -> list[str]:
        """验证任务数据完整性

        Returns:
            缺失必需字段的任务 ID 列表（空列表表示验证通过）
        """
        errors = []
        required_fields = {"id", "title", "result"}

        for task in self.tasks_completed:
            task_id = task.get("id", "unknown")
            missing = required_fields - set(task.keys())
            if missing:
                errors.append(f"{task_id}: missing {missing}")

        return errors


class IterationStatus(str, Enum):
    """迭代状态"""

    PLANNING = "planning"  # 规划中
    EXECUTING = "executing"  # 执行中
    REVIEWING = "reviewing"  # 评审中
    COMMITTING = "committing"  # 提交中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class AgentState(BaseModel):
    """单个 Agent 的状态"""

    agent_id: str
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: str | None = None
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
    completed_at: datetime | None = None

    # 统计信息
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # 评审结果
    review_passed: bool = False
    review_feedback: str | None = None

    # 提交信息（由 Orchestrator 在提交阶段填充）
    commit_hash: str | None = None
    commit_message: str | None = None
    pushed: bool = False
    commit_files: list[str] = Field(default_factory=list)
    # 提交/推送错误信息（commit 或 push 失败时记录，不中断主流程）
    commit_error: str | None = None
    push_error: str | None = None


class SystemState(BaseModel):
    """整个系统的状态"""

    # 基本信息
    goal: str = ""  # 用户目标
    working_directory: str = "."  # 工作目录

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
    final_result: str | None = None

    def start_new_iteration(self) -> IterationState:
        """开始新的迭代"""
        self.current_iteration += 1
        iteration = IterationState(iteration_id=self.current_iteration)
        self.iterations.append(iteration)
        return iteration

    def get_current_iteration(self) -> IterationState | None:
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
