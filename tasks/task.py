"""任务定义"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"          # 待处理
    QUEUED = "queued"            # 已入队
    ASSIGNED = "assigned"        # 已分配
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskType(str, Enum):
    """任务类型"""
    EXPLORE = "explore"          # 探索代码库
    ANALYZE = "analyze"          # 分析代码
    IMPLEMENT = "implement"      # 实现功能
    REFACTOR = "refactor"        # 重构代码
    FIX = "fix"                  # 修复问题
    TEST = "test"                # 编写测试
    DOCUMENT = "document"        # 编写文档
    REVIEW = "review"            # 代码评审
    CUSTOM = "custom"            # 自定义任务


class Task(BaseModel):
    """任务模型

    规划者创建任务，执行者领取并完成任务
    """
    # 基本信息
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    type: TaskType = TaskType.CUSTOM
    title: str
    description: str
    instruction: str              # 给 Cursor Agent 的具体指令

    # 状态信息
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # 关联信息
    parent_task_id: Optional[str] = None     # 父任务 ID
    sub_task_ids: list[str] = Field(default_factory=list)  # 子任务 ID 列表
    created_by: Optional[str] = None         # 创建者 Agent ID
    assigned_to: Optional[str] = None        # 分配给的 Agent ID

    # 上下文
    context: dict[str, Any] = Field(default_factory=dict)
    target_files: list[str] = Field(default_factory=list)  # 涉及的文件

    # 结果
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 迭代信息
    iteration_id: int = 0
    retry_count: int = 0
    max_retries: int = 3

    def start(self) -> None:
        """开始执行任务"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self, result: dict[str, Any]) -> None:
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.retry_count < self.max_retries

    def add_subtask(self, subtask_id: str) -> None:
        """添加子任务"""
        if subtask_id not in self.sub_task_ids:
            self.sub_task_ids.append(subtask_id)

    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def to_prompt(self) -> str:
        """生成给 Cursor Agent 的 prompt"""
        prompt_parts = [
            f"## 任务: {self.title}",
            f"\n### 描述\n{self.description}",
            f"\n### 具体指令\n{self.instruction}",
        ]

        if self.target_files:
            prompt_parts.append("\n### 涉及文件\n" + "\n".join(f"- {f}" for f in self.target_files))

        if self.context:
            prompt_parts.append(f"\n### 上下文\n```json\n{self.context}\n```")

        return "\n".join(prompt_parts)

    def to_commit_entry(self) -> dict[str, Any]:
        """生成用于提交的统一条目格式

        用于 CommitterAgent.commit_iteration() 的输入，统一字段命名。

        Returns:
            包含 id, title, description, result 的字典
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description or self.title,  # 回退策略：若 description 为空则使用 title
            "result": self.result,
        }
