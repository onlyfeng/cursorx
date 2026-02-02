"""消息协议定义"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class MessageType(str, Enum):
    """消息类型"""

    # 任务相关
    TASK_CREATE = "task_create"  # 创建任务
    TASK_ASSIGN = "task_assign"  # 分配任务
    TASK_COMPLETE = "task_complete"  # 任务完成
    TASK_FAILED = "task_failed"  # 任务失败

    # 规划相关
    PLAN_REQUEST = "plan_request"  # 请求规划
    PLAN_RESULT = "plan_result"  # 规划结果
    SPAWN_PLANNER = "spawn_planner"  # 派生子规划者

    # 执行相关
    EXECUTE_REQUEST = "execute_request"  # 请求执行
    EXECUTE_RESULT = "execute_result"  # 执行结果

    # 评审相关
    REVIEW_REQUEST = "review_request"  # 请求评审
    REVIEW_RESULT = "review_result"  # 评审结果

    # 系统相关
    SYSTEM_RESET = "system_reset"  # 系统重置
    ITERATION_START = "iteration_start"  # 迭代开始
    ITERATION_END = "iteration_end"  # 迭代结束


class Message(BaseModel):
    """消息模型

    Agent 之间通过消息进行通信
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: MessageType
    sender: str  # 发送者 Agent ID
    receiver: Optional[str] = None  # 接收者 Agent ID（None 表示广播）
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 关联消息 ID（用于追踪请求-响应）

    model_config = ConfigDict()

    def create_reply(self, msg_type: MessageType, payload: dict[str, Any]) -> "Message":
        """创建回复消息"""
        return Message(
            type=msg_type,
            sender=self.receiver or "",
            receiver=self.sender,
            payload=payload,
            correlation_id=self.id,
        )

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()
