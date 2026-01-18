"""Agent 基类定义"""
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Agent 角色类型"""
    PLANNER = "planner"          # 规划者
    SUB_PLANNER = "sub_planner"  # 子规划者
    WORKER = "worker"            # 执行者
    REVIEWER = "reviewer"        # 评审者
    COMMITTER = "committer"      # 提交者


class AgentStatus(str, Enum):
    """Agent 状态"""
    IDLE = "idle"                # 空闲
    RUNNING = "running"          # 执行中
    WAITING = "waiting"          # 等待中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败


class AgentConfig(BaseModel):
    """Agent 配置"""
    role: AgentRole
    name: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    max_retries: int = 3
    timeout: int = 300  # 秒
    working_directory: str = "."

    # Cursor Agent 相关配置
    cursor_agent_mode: bool = True  # 使用 Cursor Agent 模式


class BaseAgent(ABC):
    """Agent 基类

    所有 Agent（Planner/Worker/Reviewer）都继承此类
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = f"{config.role.value}-{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.IDLE
        self._context: dict[str, Any] = {}

    @property
    def role(self) -> AgentRole:
        return self.config.role

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行指令

        Args:
            instruction: 要执行的指令
            context: 上下文信息

        Returns:
            执行结果
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """重置 Agent 状态"""
        pass

    def update_status(self, status: AgentStatus) -> None:
        """更新状态"""
        previous = self.status
        self.status = status
        if previous != status:
            try:
                logger.info(f"[{self.id}] 状态切换: {previous.value} -> {status.value}")
            except Exception as e:
                logger.warning(f"[{self.id}] 记录状态失败: {e}")

    def set_context(self, key: str, value: Any) -> None:
        """设置上下文"""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文"""
        return self._context.get(key, default)

    def clear_context(self) -> None:
        """清除上下文"""
        self._context.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} status={self.status.value}>"
